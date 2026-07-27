/* render.c -- driving the frame decode and converting float planes to the
 * caller's pixel format (8/16-bit gray/gray+alpha/RGB/RGBA), including the
 * EXIF-style orientation fixup.
 */
#include "jxl_internal.h"

#include <math.h>

static int walk_frames(jxl_doc *doc, int frame_no, jxl_fimage *img,
                       int *count_out, jxl_frame_info *info_out);

int jxl_doc_frame_count(jxl_doc *doc) {
    int count = 0;
    if (!doc) return 0;
    if (doc->frame_count > 0) return doc->frame_count;
    if (walk_frames(doc, -1, NULL, &count, NULL) != 0) return 1;
    doc->frame_count = count > 0 ? count : 1;
    return doc->frame_count;
}

int jxl_doc_frame_info(jxl_doc *doc, int frame_no, jxl_frame_info *info) {
    jxl_fimage img;
    int rc;
    if (!doc || !info || frame_no < 0) return -1;
    memset(info, 0, sizeof(*info));
    info->tps_numerator = (int)doc->meta.animation.tps_numerator;
    info->tps_denominator = (int)doc->meta.animation.tps_denominator;
    info->is_last = 1;
    memset(&img, 0, sizeof(img));
    rc = walk_frames(doc, frame_no, &img, NULL, info);
    jxl_fimage_free(doc->ctx, &img);
    return rc;
}

static jxl_format resolve_format(const jxl_image_info *ii, jxl_format fmt) {
    int wide, gray, alpha;
    if (fmt != JXLDEC_FORMAT_NATIVE) return fmt;
    wide = ii->bits_per_sample > 8 || ii->exponent_bits > 0;
    gray = ii->num_color_channels == 1;
    alpha = ii->alpha_bits > 0;
    if (gray) {
        return wide ? (alpha ? JXLDEC_FORMAT_GRAYA16 : JXLDEC_FORMAT_GRAY16)
                    : (alpha ? JXLDEC_FORMAT_GRAYA8 : JXLDEC_FORMAT_GRAY8);
    }
    return wide ? (alpha ? JXLDEC_FORMAT_RGBA64 : JXLDEC_FORMAT_RGB48)
                : (alpha ? JXLDEC_FORMAT_RGBA32 : JXLDEC_FORMAT_RGB24);
}

int jxl_frame_render_info(jxl_doc *doc, int frame_no, jxl_format fmt,
                          jxl_render_info *info) {
    jxl_image_info ii;
    if (!doc || !info) return -1;
    if (jxl_doc_info(doc, &ii) != 0) return -1;
    (void)frame_no;
    info->width = ii.width;
    info->height = ii.height;
    info->format = resolve_format(&ii, fmt);
    return 0;
}

/* ----- float -> integer sample conversion ----- */

static uint32_t quantize(float v, uint32_t maxval) {
    float s;
    if (!(v > 0.0f)) return 0;       /* also catches NaN */
    s = v * (float)maxval + 0.5f;
    if (s >= (float)maxval) return maxval;
    return (uint32_t)s;
}

/* SSE2 is baseline on x64. Quantization is vectorised for all formats, and
   the common RGB24/RGBA32 cases pack four complete pixels at once. Strided
   orientations gather four samples before doing the same SIMD arithmetic.
   -DJXL_RENDER_FORCE_SCALAR builds without it, to diff against. */
#if !defined(JXL_RENDER_FORCE_SCALAR) && \
    (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define JXL_RENDER_SSE2 1
#include <emmintrin.h>
#include <immintrin.h>

/* Four samples through the exact logic of quantize() above. */
static __m128i quantize4m(__m128 v, uint32_t maxval) {
    const __m128 vm = _mm_set1_ps((float)maxval);
    const __m128 zero = _mm_setzero_ps();
    __m128 s = _mm_add_ps(_mm_mul_ps(v, vm), _mm_set1_ps(0.5f));
    __m128i t = _mm_cvttps_epi32(s);                  /* truncates, as (uint32_t) */
    __m128 pos = _mm_cmpgt_ps(v, zero);               /* false for <= 0 and NaN */
    __m128 sat = _mm_cmpge_ps(s, vm);
    __m128i mx = _mm_set1_epi32((int)maxval);
    t = _mm_or_si128(_mm_and_si128(_mm_castps_si128(sat), mx),
                     _mm_andnot_si128(_mm_castps_si128(sat), t));
    return _mm_and_si128(_mm_castps_si128(pos), t);
}

static __m128i quantize4v(const float *src, uint32_t maxval) {
    return quantize4m(_mm_loadu_ps(src), maxval);
}

static __m128i quantize4_stride(const float *src, ptrdiff_t step,
                                uint32_t maxval) {
    return quantize4m(_mm_setr_ps(src[0], src[step], src[2 * step],
                                 src[3 * step]), maxval);
}

static void quantize4(const float *src, uint32_t maxval, uint32_t out[4]) {
    _mm_storeu_si128((__m128i *)out, quantize4v(src, maxval));
}

static void quantize4_stride_store(const float *src, ptrdiff_t step,
                                   uint32_t maxval, uint32_t out[4]) {
    _mm_storeu_si128((__m128i *)out, quantize4_stride(src, step, maxval));
}
#endif

/* Reads a plane sample, replicating edge pixels for subsampled channels. */
static float plane_sample(const jxl_fplane *p, uint32_t x, uint32_t y,
                          uint32_t w, uint32_t h) {
    uint32_t px, py;
    if (p->w == 0 || p->h == 0) return 0.0f;
    px = (p->w == w) ? x : (uint32_t)((uint64_t)x * p->w / (w ? w : 1));
    py = (p->h == h) ? y : (uint32_t)((uint64_t)y * p->h / (h ? h : 1));
    if (px >= p->w) px = p->w - 1;
    if (py >= p->h) py = p->h - 1;
    return p->data[(size_t)py * p->stride + px];
}

/* A plane that plane_sample() would read one-to-one, needing no scaling. */
static int plane_is_full(const jxl_fplane *p, uint32_t w, uint32_t h) {
    return p && p->w == w && p->h == h && w != 0 && h != 0;
}

/* Maps an output coordinate back to the decoded (unoriented) one. */
static void unapply_orientation(uint32_t orientation, uint32_t sw, uint32_t sh,
                                uint32_t ox, uint32_t oy, uint32_t *sx,
                                uint32_t *sy) {
    switch (orientation) {
        case 1: *sx = ox; *sy = oy; break;
        case 2: *sx = sw - 1 - ox; *sy = oy; break;
        case 3: *sx = sw - 1 - ox; *sy = sh - 1 - oy; break;
        case 4: *sx = ox; *sy = sh - 1 - oy; break;
        case 5: *sx = oy; *sy = ox; break;
        case 6: *sx = oy; *sy = sh - 1 - ox; break;
        case 7: *sx = sw - 1 - oy; *sy = sh - 1 - ox; break;
        default: *sx = sw - 1 - oy; *sy = ox; break;
    }
}

typedef struct {
    const jxl_fplane *r, *g, *b, *a;
} jxl_out_planes;

static void pick_planes(const jxl_fimage *img, const jxl_image_metadata *meta,
                        jxl_out_planes *op) {
    memset(op, 0, sizeof(*op));
    if (img->nplane == 0) return;
    op->r = &img->plane[0];
    if (img->ncolor >= 3 && img->nplane >= 3) {
        op->g = &img->plane[1];
        op->b = &img->plane[2];
    } else {
        op->g = op->r;
        op->b = op->r;
    }
    if (meta->alpha_index >= 0) {
        uint32_t idx = img->ncolor + (uint32_t)meta->alpha_index;
        if (idx < img->nplane) op->a = &img->plane[idx];
    }
}

#ifdef JXL_RENDER_SSE2
JXL_TARGET_AVX2
static __m256i quantize8_255(const float *src) {
    const __m256 vm = _mm256_set1_ps(255.0f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 v = _mm256_loadu_ps(src);
    __m256 s = _mm256_add_ps(_mm256_mul_ps(v, vm), _mm256_set1_ps(0.5f));
    __m256i t = _mm256_cvttps_epi32(s);
    __m256 pos = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
    __m256 sat = _mm256_cmp_ps(s, vm, _CMP_GE_OQ);
    __m256i mx = _mm256_set1_epi32(255);
    t = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(sat), mx),
                        _mm256_andnot_si256(_mm256_castps_si256(sat), t));
    return _mm256_and_si256(_mm256_castps_si256(pos), t);
}

/* Eight forward RGB(A) or BGR(A) pixels. Four-component output is already one
   packed dword per lane. For three components, vpshufb removes the unused
   fourth byte independently in each 128-bit half, producing two 12-byte
   groups. Two overlapping 16-byte stores write those groups; the next vector
   (or the existing four-pixel/scalar tail) overwrites the overhang. */
JXL_TARGET_AVX2
static uint32_t write_rgb8_row_avx2(const float *r, const float *g,
                                    const float *b, const float *a,
                                    uint8_t *dst, uint32_t x, uint32_t w,
                                    int ncomp, int bgr) {
    const __m256i opaque = _mm256_set1_epi32(255);
    const __m256i rgb_shuffle = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    const float *c0 = bgr ? b : r;
    const float *c2 = bgr ? r : b;
    uint32_t end = ncomp == 4 ? w : (w >= 8 ? w - 8 : 0);

    for (; x + 8 <= end; x += 8) {
        __m256i qr = quantize8_255(c0 + x);
        __m256i qg = quantize8_255(g + x);
        __m256i qb = quantize8_255(c2 + x);
        __m256i packed = _mm256_or_si256(
            _mm256_or_si256(qr, _mm256_slli_epi32(qg, 8)),
            _mm256_slli_epi32(qb, 16));
        if (ncomp == 4) {
            __m256i qa = a ? quantize8_255(a + x) : opaque;
            packed = _mm256_or_si256(packed, _mm256_slli_epi32(qa, 24));
            _mm256_storeu_si256((__m256i *)(dst + (size_t)x * 4), packed);
        } else {
            __m256i rgb = _mm256_shuffle_epi8(packed, rgb_shuffle);
            _mm_storeu_si128((__m128i *)(dst + (size_t)x * 3),
                             _mm256_castsi256_si128(rgb));
            _mm_storeu_si128((__m128i *)(dst + (size_t)x * 3 + 12),
                             _mm256_extracti128_si256(rgb, 1));
        }
    }
    _mm256_zeroupper();
    return x;
}

/* Transpose eight rows of eight packed pixels. */
JXL_TARGET_AVX2
static void transpose8x8_i32(__m256i p[8]) {
    __m256i t0 = _mm256_unpacklo_epi32(p[0], p[1]);
    __m256i t1 = _mm256_unpackhi_epi32(p[0], p[1]);
    __m256i t2 = _mm256_unpacklo_epi32(p[2], p[3]);
    __m256i t3 = _mm256_unpackhi_epi32(p[2], p[3]);
    __m256i t4 = _mm256_unpacklo_epi32(p[4], p[5]);
    __m256i t5 = _mm256_unpackhi_epi32(p[4], p[5]);
    __m256i t6 = _mm256_unpacklo_epi32(p[6], p[7]);
    __m256i t7 = _mm256_unpackhi_epi32(p[6], p[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi64(t5, t7);
    p[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    p[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    p[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    p[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    p[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    p[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    p[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    p[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

JXL_TARGET_AVX2
static void write_transposed_rgba8_avx2(const jxl_out_planes *op,
                                        uint32_t orientation,
                                        uint32_t sw, uint32_t sh,
                                        uint8_t *dst, int stride) {
    const uint32_t ow = sh, oh = sw;
    const __m256i opaque = _mm256_set1_epi32(255);
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    uint32_t by, bx;

    for (by = 0; by < oh; by += 64) {
        uint32_t ylim = JXL_MIN(by + 64, oh);
        for (bx = 0; bx < ow; bx += 64) {
            uint32_t xlim = JXL_MIN(bx + 64, ow);
            uint32_t oy, ox;
            for (oy = by; oy < ylim; oy += 8) {
                uint32_t sx = (orientation == 7 || orientation == 8)
                                  ? sw - oy - 8 : oy;
                for (ox = bx; ox < xlim; ox += 8) {
                    __m256i p[8];
                    uint32_t i;
                    for (i = 0; i < 8; i++) {
                        uint32_t sy = (orientation == 6 || orientation == 7)
                                          ? sh - 1 - ox - i : ox + i;
                        const float *r = op->r->data +
                                         (size_t)sy * op->r->stride + sx;
                        const float *g = op->g->data +
                                         (size_t)sy * op->g->stride + sx;
                        const float *b = op->b->data +
                                         (size_t)sy * op->b->stride + sx;
                        __m256i qr = quantize8_255(r);
                        __m256i qg = quantize8_255(g);
                        __m256i qb = quantize8_255(b);
                        __m256i qa = opaque;
                        if (op->a) {
                            const float *a = op->a->data +
                                             (size_t)sy * op->a->stride + sx;
                            qa = quantize8_255(a);
                        }
                        p[i] = _mm256_or_si256(
                            _mm256_or_si256(qr, _mm256_slli_epi32(qg, 8)),
                            _mm256_or_si256(_mm256_slli_epi32(qb, 16),
                                            _mm256_slli_epi32(qa, 24)));
                        if (orientation == 7 || orientation == 8)
                            p[i] = _mm256_permutevar8x32_epi32(p[i], reverse);
                    }
                    transpose8x8_i32(p);
                    for (i = 0; i < 8; i++) {
                        _mm256_storeu_si256(
                            (__m256i *)(dst + (size_t)(oy + i) * stride +
                                        (size_t)ox * 4),
                            p[i]);
                    }
                }
            }
        }
    }
    _mm256_zeroupper();
}

/* Orientations 5..8 transpose the image. Walking one output row at a time
   gathers down four widely separated source rows for every vector, which is
   particularly expensive for planar RGBA. Work in 64x64 tiles and transpose
   four already-packed pixels at a time: all four source-plane loads and all
   four output stores are contiguous, while the tile keeps both sides hot.
   The AVX2 twin above applies the same layout eight pixels at a time. */
static void write_transposed_rgba8(const jxl_out_planes *op,
                                   uint32_t orientation,
                                   uint32_t sw, uint32_t sh,
                                   uint8_t *dst, int stride) {
    const uint32_t ow = sh, oh = sw;
    uint32_t by, bx;
    const __m128i opaque = _mm_set1_epi32(255);

    for (by = 0; by < oh; by += 64) {
        uint32_t ylim = JXL_MIN(by + 64, oh);
        for (bx = 0; bx < ow; bx += 64) {
            uint32_t xlim = JXL_MIN(bx + 64, ow);
            uint32_t oy, ox;
            for (oy = by; oy < ylim; oy += 4) {
                uint32_t sx = (orientation == 7 || orientation == 8)
                                  ? sw - oy - 4 : oy;
                for (ox = bx; ox < xlim; ox += 4) {
                    __m128i p[4];
                    uint32_t i;
                    for (i = 0; i < 4; i++) {
                        uint32_t sy = (orientation == 6 || orientation == 7)
                                          ? sh - 1 - ox - i : ox + i;
                        const float *r = op->r->data +
                                         (size_t)sy * op->r->stride + sx;
                        const float *g = op->g->data +
                                         (size_t)sy * op->g->stride + sx;
                        const float *b = op->b->data +
                                         (size_t)sy * op->b->stride + sx;
                        __m128i qr = quantize4v(r, 255);
                        __m128i qg = quantize4v(g, 255);
                        __m128i qb = quantize4v(b, 255);
                        __m128i qa = opaque;
                        if (op->a) {
                            const float *a = op->a->data +
                                             (size_t)sy * op->a->stride + sx;
                            qa = quantize4v(a, 255);
                        }
                        p[i] = _mm_or_si128(
                            _mm_or_si128(qr, _mm_slli_epi32(qg, 8)),
                            _mm_or_si128(_mm_slli_epi32(qb, 16),
                                         _mm_slli_epi32(qa, 24)));
                        if (orientation == 7 || orientation == 8)
                            p[i] = _mm_shuffle_epi32(
                                p[i], _MM_SHUFFLE(0, 1, 2, 3));
                    }
                    {
                        __m128 q0 = _mm_castsi128_ps(p[0]);
                        __m128 q1 = _mm_castsi128_ps(p[1]);
                        __m128 q2 = _mm_castsi128_ps(p[2]);
                        __m128 q3 = _mm_castsi128_ps(p[3]);
                        _MM_TRANSPOSE4_PS(q0, q1, q2, q3);
                        _mm_storeu_si128(
                            (__m128i *)(dst + (size_t)oy * stride +
                                        (size_t)ox * 4),
                            _mm_castps_si128(q0));
                        _mm_storeu_si128(
                            (__m128i *)(dst + (size_t)(oy + 1) * stride +
                                        (size_t)ox * 4),
                            _mm_castps_si128(q1));
                        _mm_storeu_si128(
                            (__m128i *)(dst + (size_t)(oy + 2) * stride +
                                        (size_t)ox * 4),
                            _mm_castps_si128(q2));
                        _mm_storeu_si128(
                            (__m128i *)(dst + (size_t)(oy + 3) * stride +
                                        (size_t)ox * 4),
                            _mm_castps_si128(q3));
                    }
                }
            }
        }
    }
}
#endif

static int write_pixels(jxl_ctx *ctx, jxl_doc *doc, const jxl_fimage *img,
                        jxl_format fmt, uint8_t *dst, int stride) {
    const jxl_image_metadata *meta = &doc->meta;
    jxl_out_planes op;
    uint32_t sw = img->w, sh = img->h;
    uint32_t ow, oh, ox, oy;
    uint32_t orientation = ctx->keep_orientation ? 1 : meta->orientation;
    int ncomp = 0, wide = 0, has_alpha = 0, gray = 0;
    int direct, reverse_x, transposed;
    uint32_t maxval;
    int bgr = ctx->bgr;
#ifdef JXL_RENDER_SSE2
    const int use_avx2 = jxl_has_avx2();
#endif

    pick_planes(img, meta, &op);
    jxl_apply_orientation_dims(orientation, sw, sh, &ow, &oh);

    switch (fmt) {
        case JXLDEC_FORMAT_GRAY8: ncomp = 1; gray = 1; break;
        case JXLDEC_FORMAT_GRAYA8: ncomp = 2; gray = 1; has_alpha = 1; break;
        case JXLDEC_FORMAT_RGB24: ncomp = 3; break;
        case JXLDEC_FORMAT_RGBA32: ncomp = 4; has_alpha = 1; break;
        case JXLDEC_FORMAT_GRAY16: ncomp = 1; gray = 1; wide = 1; break;
        case JXLDEC_FORMAT_GRAYA16: ncomp = 2; gray = 1; has_alpha = 1; wide = 1; break;
        case JXLDEC_FORMAT_RGB48: ncomp = 3; wide = 1; break;
        case JXLDEC_FORMAT_RGBA64: ncomp = 4; has_alpha = 1; wide = 1; break;
        default:
            JXL_ERR(ctx, "render: unsupported output format %d", (int)fmt);
            return -1;
    }
    maxval = wide ? 65535u : 255u;

    /* With every plane at full resolution, orientations 1..4 select a decoded
       row and possibly read it right-to-left; orientations 5..8 select a
       column and advance by +/-stride. The per-pixel coordinate mapping and
       subsampling checks drop out in both cases. */
#ifdef JXL_RENDER_FORCE_GENERAL_ORIENTATION
    direct = (orientation == 1) &&
#else
    direct = (orientation >= 1 && orientation <= 8) &&
#endif
             plane_is_full(op.r, sw, sh) &&
             (gray || (plane_is_full(op.g, sw, sh) &&
                       plane_is_full(op.b, sw, sh))) &&
             (!op.a || plane_is_full(op.a, sw, sh));
    reverse_x = orientation == 2 || orientation == 3;
    transposed = orientation >= 5;

#ifdef JXL_RENDER_SSE2
    if (direct && transposed && !wide && !gray && !bgr && ncomp == 4 &&
        (ow & 3u) == 0 && (oh & 3u) == 0) {
        if (use_avx2 && (ow & 7u) == 0 && (oh & 7u) == 0) {
            write_transposed_rgba8_avx2(&op, orientation, sw, sh, dst, stride);
            return 0;
        }
        write_transposed_rgba8(&op, orientation, sw, sh, dst, stride);
        return 0;
    }
#endif

    for (oy = 0; oy < oh; oy++) {
        uint8_t *row8 = dst + (size_t)oy * stride;
        uint16_t *row16 = (uint16_t *)row8;
        const float *pr = NULL, *pg = NULL, *pb = NULL, *pa = NULL;
        ptrdiff_t sr = 1, sg = 1, sb = 1, sa = 1;
        if (direct) {
            uint32_t sx, sy;
            if (transposed) {
                sx = (orientation == 7 || orientation == 8)
                         ? sw - 1 - oy : oy;
                sy = (orientation == 6 || orientation == 7) ? sh - 1 : 0;
            } else {
                sx = 0;
                sy = (orientation == 3 || orientation == 4)
                         ? sh - 1 - oy : oy;
            }
            pr = op.r->data + (size_t)sy * op.r->stride + sx;
            if (!gray) {
                pg = op.g->data + (size_t)sy * op.g->stride + sx;
                pb = op.b->data + (size_t)sy * op.b->stride + sx;
            }
            if (op.a) pa = op.a->data + (size_t)sy * op.a->stride + sx;
            if (transposed) {
                sr = (ptrdiff_t)op.r->stride;
                sg = gray ? sr : (ptrdiff_t)op.g->stride;
                sb = gray ? sr : (ptrdiff_t)op.b->stride;
                sa = op.a ? (ptrdiff_t)op.a->stride : 1;
                if (orientation == 6 || orientation == 7) {
                    sr = -sr; sg = -sg; sb = -sb; sa = -sa;
                }
            }
        }
        ox = 0;
#ifdef JXL_RENDER_SSE2
        /* Quantise four pixels' worth of each plane up front; the store loop
           below reads those results instead of calling quantize(). Horizontal
           flips reverse the finished lanes, while transposed orientations
           gather four samples at +/-stride. */
        /* A grayscale byte row needs no component interleave. Pack sixteen
           quantized int32 lanes down to bytes and store them together instead
           of bouncing each group through four temporary uint32_t arrays and
           a per-pixel component loop. */
        if (direct && !wide && gray && ncomp == 1 &&
            !transposed && !reverse_x) {
            for (; ox + 16 <= ow; ox += 16) {
                __m128i q0 = quantize4v(pr + ox, maxval);
                __m128i q1 = quantize4v(pr + ox + 4, maxval);
                __m128i q2 = quantize4v(pr + ox + 8, maxval);
                __m128i q3 = quantize4v(pr + ox + 12, maxval);
                __m128i h0 = _mm_packs_epi32(q0, q1);
                __m128i h1 = _mm_packs_epi32(q2, q3);
                _mm_storeu_si128((__m128i *)(row8 + ox),
                                 _mm_packus_epi16(h0, h1));
            }
        }
        if (use_avx2 && direct && !wide && !gray &&
            !transposed && !reverse_x && (ncomp == 3 || ncomp == 4)) {
            ox = write_rgb8_row_avx2(pr, pg, pb, pa, row8, ox, ow, ncomp, bgr);
        }
        /* The two common 8-bit RGB formats pack without any shuffling: each
           lane's three or four components are already 0..255 in separate
           32-bit lanes, so r | g<<8 | b<<16 | a<<24 lays a whole pixel out in
           one lane and four pixels are one store.
           RGB24 has no fourth component to carry the alpha slot, so each
           pixel is written as a 4-byte store whose top byte lands on the next
           pixel and is overwritten by it. The loop stops four pixels short of
           the row so the last such overhang stays inside the row and the
           scalar tail below rewrites it. */
        if (direct && !wide && !gray && (ncomp == 3 || ncomp == 4)) {
            uint32_t lim = ncomp == 4 ? ow : (ow >= 4 ? ow - 4 : 0);
            const __m128i amax = _mm_set1_epi32((int)maxval);
            for (; ox + 4 <= lim; ox += 4) {
                uint32_t qx = reverse_x ? ow - ox - 4 : ox;
                __m128i r, g, b, a;
                if (transposed) {
                    r = quantize4_stride(pr + (ptrdiff_t)ox * sr, sr, maxval);
                    g = quantize4_stride(pg + (ptrdiff_t)ox * sg, sg, maxval);
                    b = quantize4_stride(pb + (ptrdiff_t)ox * sb, sb, maxval);
                    a = (pa && ncomp == 4)
                            ? quantize4_stride(pa + (ptrdiff_t)ox * sa, sa, maxval)
                            : amax;
                } else {
                    r = quantize4v(pr + qx, maxval);
                    g = quantize4v(pg + qx, maxval);
                    b = quantize4v(pb + qx, maxval);
                    a = (pa && ncomp == 4) ? quantize4v(pa + qx, maxval)
                                           : amax;
                }
                if (bgr) {
                    __m128i t = r;
                    r = b;
                    b = t;
                }
                /* RGB24 discards this lane, so do not pay to quantise it. */
                __m128i packed = _mm_or_si128(
                    _mm_or_si128(r, _mm_slli_epi32(g, 8)),
                    _mm_or_si128(_mm_slli_epi32(b, 16), _mm_slli_epi32(a, 24)));
                if (!transposed && reverse_x)
                    packed = _mm_shuffle_epi32(packed, _MM_SHUFFLE(0, 1, 2, 3));
                if (ncomp == 4) {
                    _mm_storeu_si128((__m128i *)(row8 + (size_t)ox * 4), packed);
                } else {
                    uint32_t tmp[4];
                    uint32_t j;
                    _mm_storeu_si128((__m128i *)tmp, packed);
                    for (j = 0; j < 4; j++) {
                        memcpy(row8 + (size_t)(ox + j) * 3, &tmp[j], 4);
                    }
                }
            }
        }
        /* The same overlapping-store trick is particularly useful for
           16-bit RGB: one 64-bit store writes all three components and two
           bytes of the next pixel, which that pixel immediately overwrites.
           RGBA64 fills the whole store without overlap. Keep four scalar
           tail pixels for RGB48 so the final overhang cannot cross the row. */
        if (direct && wide && !gray && !bgr && !transposed && !reverse_x &&
            (ncomp == 3 || ncomp == 4)) {
            uint32_t lim = ncomp == 4 ? ow : (ow >= 4 ? ow - 4 : 0);
            for (; ox + 4 <= lim; ox += 4) {
                uint32_t qr[4], qg[4], qb[4], qa[4], j;
                quantize4(pr + ox, maxval, qr);
                quantize4(pg + ox, maxval, qg);
                quantize4(pb + ox, maxval, qb);
                if (pa && ncomp == 4) quantize4(pa + ox, maxval, qa);
                for (j = 0; j < 4; j++) {
                    uint64_t packed = (uint64_t)qr[j] |
                                      ((uint64_t)qg[j] << 16) |
                                      ((uint64_t)qb[j] << 32);
                    if (ncomp == 4) {
                        uint32_t a = pa ? qa[j] : maxval;
                        packed |= (uint64_t)a << 48;
                    }
                    memcpy(row8 + (size_t)(ox + j) * ncomp * 2,
                           &packed, sizeof(packed));
                }
            }
        }
        if (direct) {
            uint32_t qr[4], qg[4], qb[4], qa[4];
            for (; ox + 4 <= ow; ox += 4) {
                uint32_t j, qx = reverse_x ? ow - ox - 4 : ox;
                if (transposed) {
                    quantize4_stride_store(pr + (ptrdiff_t)ox * sr, sr,
                                           maxval, qr);
                    if (!gray) {
                        quantize4_stride_store(pg + (ptrdiff_t)ox * sg, sg,
                                               maxval, qg);
                        quantize4_stride_store(pb + (ptrdiff_t)ox * sb, sb,
                                               maxval, qb);
                    }
                    if (pa)
                        quantize4_stride_store(pa + (ptrdiff_t)ox * sa, sa,
                                               maxval, qa);
                } else {
                    quantize4(pr + qx, maxval, qr);
                    if (!gray) {
                        quantize4(pg + qx, maxval, qg);
                        quantize4(pb + qx, maxval, qb);
                    }
                    if (pa) quantize4(pa + qx, maxval, qa);
                }
                for (j = 0; j < 4; j++) {
                    uint32_t comps[4];
                    uint32_t px = ox + j;
                    uint32_t qj = (!transposed && reverse_x) ? 3 - j : j;
                    uint32_t r = qr[qj];
                    uint32_t g = gray ? r : qg[qj];
                    uint32_t b = gray ? r : qb[qj];
                    uint32_t a = pa ? qa[qj] : maxval;
                    if (gray) {
                        comps[0] = r;
                        if (has_alpha) comps[1] = a;
                    } else if (bgr) {
                        comps[0] = b; comps[1] = g; comps[2] = r;
                        if (has_alpha) comps[3] = a;
                    } else {
                        comps[0] = r; comps[1] = g; comps[2] = b;
                        if (has_alpha) comps[3] = a;
                    }
                    if (wide) {
                        uint16_t *o = row16 + (size_t)px * ncomp;
                        o[0] = (uint16_t)comps[0];
                        if (ncomp > 1) o[1] = (uint16_t)comps[1];
                        if (ncomp > 2) o[2] = (uint16_t)comps[2];
                        if (ncomp > 3) o[3] = (uint16_t)comps[3];
                    } else {
                        uint8_t *o = row8 + (size_t)px * ncomp;
                        o[0] = (uint8_t)comps[0];
                        if (ncomp > 1) o[1] = (uint8_t)comps[1];
                        if (ncomp > 2) o[2] = (uint8_t)comps[2];
                        if (ncomp > 3) o[3] = (uint8_t)comps[3];
                    }
                }
            }
        }
#endif
        for (; ox < ow; ox++) {
            uint32_t sx, sy;
            float rv, gv, bv, av = 1.0f;
            uint32_t comps[4];

            if (direct) {
                if (transposed) {
                    ptrdiff_t px = (ptrdiff_t)ox;
                    rv = pr[px * sr];
                    if (gray) {
                        gv = bv = rv;
                    } else {
                        gv = pg[px * sg];
                        bv = pb[px * sb];
                    }
                    if (pa) av = pa[px * sa];
                } else {
                    uint32_t px = reverse_x ? ow - 1 - ox : ox;
                    rv = pr[px];
                    if (gray) {
                        gv = bv = rv;
                    } else {
                        gv = pg[px];
                        bv = pb[px];
                    }
                    if (pa) av = pa[px];
                }
            } else {
                unapply_orientation(orientation, sw, sh, ox, oy, &sx, &sy);
                rv = plane_sample(op.r, sx, sy, sw, sh);
                if (gray) {
                    gv = bv = rv;
                } else {
                    gv = plane_sample(op.g, sx, sy, sw, sh);
                    bv = plane_sample(op.b, sx, sy, sw, sh);
                }
                if (op.a) av = plane_sample(op.a, sx, sy, sw, sh);
            }

            if (gray) {
                comps[0] = quantize(rv, maxval);
                if (has_alpha) comps[1] = quantize(av, maxval);
            } else if (bgr) {
                comps[0] = quantize(bv, maxval);
                comps[1] = quantize(gv, maxval);
                comps[2] = quantize(rv, maxval);
                if (has_alpha) comps[3] = quantize(av, maxval);
            } else {
                comps[0] = quantize(rv, maxval);
                comps[1] = quantize(gv, maxval);
                comps[2] = quantize(bv, maxval);
                if (has_alpha) comps[3] = quantize(av, maxval);
            }
            /* ncomp is 1..4; predictable tests beat a loop the compiler has
               to keep general, and the pixel offset is computed once. */
            if (wide) {
                uint16_t *o = row16 + (size_t)ox * ncomp;
                o[0] = (uint16_t)comps[0];
                if (ncomp > 1) o[1] = (uint16_t)comps[1];
                if (ncomp > 2) o[2] = (uint16_t)comps[2];
                if (ncomp > 3) o[3] = (uint16_t)comps[3];
            } else {
                uint8_t *o = row8 + (size_t)ox * ncomp;
                o[0] = (uint8_t)comps[0];
                if (ncomp > 1) o[1] = (uint8_t)comps[1];
                if (ncomp > 2) o[2] = (uint8_t)comps[2];
                if (ncomp > 3) o[3] = (uint8_t)comps[3];
            }
        }
    }
    return 0;
}

/* A frame is displayed when it is a normal frame that either ends the
   animation or has a duration; LF and reference-only frames are decoded for
   their side effects and skipped. */
static int frame_is_keyframe(const jxl_frame_header *fh) {
    if (fh->frame_type != JXL_FRAME_REGULAR &&
        fh->frame_type != JXL_FRAME_SKIP_PROGRESSIVE)
        return 0;
    return fh->is_last || fh->duration != 0;
}

/* Walks the frame chain. frame_no < 0 with count_out set only counts the
   displayed frames. Displayed frames are composited onto the canvas their
   blending info names, so animation frames that carry only a changed
   rectangle land in the right place. */
static int walk_frames(jxl_doc *doc, int frame_no, jxl_fimage *img,
                       int *count_out, jxl_frame_info *info_out) {
    jxl_ctx *ctx = doc->ctx;
    jxl_frame_state st;
    jxl_fimage canvas;
    int canvas_valid = 0;
    jxl_br br;
    int idx = 0;
    int rc = -1;

    memset(&st, 0, sizeof(st));
    memset(&canvas, 0, sizeof(canvas));
    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);

    for (;;) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t end;
        int last, keyframe, want, apply_ct;
        jxl_fimage tmp;

        memset(&toc, 0, sizeof(toc));
        memset(&tmp, 0, sizeof(tmp));
        if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) != 0) {
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }
        end = toc.end_off + toc.total_size;
        last = fh.is_last;
        keyframe = frame_is_keyframe(&fh);
        want = keyframe && idx == frame_no;
        apply_ct = want || !fh.save_before_ct;

        /* These two feed the noise RNG seed, and libjxl advances them in
           InitFrame -- before the frame is decoded, not after. So the first
           visible frame synthesises its noise with visible_frame_index
           already 1. Counting them afterwards seeded every single-frame
           image with 0 and produced a noise field with exactly the right
           distribution and no relation to libjxl's. */
        if (keyframe) {
            st.visible_frames++;
            st.invisible_frames = 0;
        } else {
            st.invisible_frames++;
        }

        if (count_out && !want) {
            /* Counting only: skip the pixel work entirely. */
            if (keyframe) idx++;
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            if (last || end >= doc->container.cs_len) break;
            jxl_br_seek_byte(&br, end);
            continue;
        }

        if (jxl_frame_decode(ctx, doc, &fh, &toc, &st, apply_ct, &tmp) != 0) {
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }

        if (fh.frame_type == JXL_FRAME_LF) {
            jxl_fimage_free(ctx, &st.lf_image);
            st.lf_image = tmp;
            st.lf_valid = 1;
            memset(&tmp, 0, sizeof(tmp));
        } else if (!keyframe) {
            if (fh.save_as_reference < 4) {
                uint32_t slot = fh.save_as_reference;
                jxl_fimage_free(ctx, &st.refs[slot]);
                st.refs[slot] = tmp;
                st.refs_valid[slot] = 1;
                memset(&tmp, 0, sizeof(tmp));
            }
        } else {
            uint32_t src = fh.blending.source;
            int cropped = fh.have_crop || tmp.w != doc->size.width ||
                          tmp.h != doc->size.height;
            int needs_canvas = cropped || fh.blending.mode != JXL_BLEND_REPLACE;
            int failed = 0;

            if (!needs_canvas) {
                jxl_fimage_free(ctx, &canvas);
                canvas = tmp;
                memset(&tmp, 0, sizeof(tmp));
            } else {
                jxl_fimage base;
                memset(&base, 0, sizeof(base));
                if (src < 4 && st.refs_valid[src]) {
                    failed = jxl_fimage_copy(ctx, &base, &st.refs[src]) != 0;
                } else if (canvas_valid) {
                    failed = jxl_fimage_copy(ctx, &base, &canvas) != 0;
                } else {
                    failed = jxl_fimage_blank_like(ctx, &base, &tmp,
                                                   doc->size.width,
                                                   doc->size.height) != 0;
                }
                if (!failed) {
                    jxl_blend_frame(ctx, &base, &tmp, &fh, &doc->meta);
                    jxl_fimage_free(ctx, &canvas);
                    canvas = base;
                } else {
                    jxl_fimage_free(ctx, &base);
                }
                jxl_fimage_free(ctx, &tmp);
            }
            if (failed) {
                jxl_toc_free(ctx, &toc);
                jxl_frame_header_free(ctx, &fh);
                goto done;
            }
            canvas_valid = 1;
            if (fh.save_as_reference < 4 && !fh.is_last) {
                uint32_t slot = fh.save_as_reference;
                jxl_fimage_free(ctx, &st.refs[slot]);
                memset(&st.refs[slot], 0, sizeof(st.refs[slot]));
                if (jxl_fimage_copy(ctx, &st.refs[slot], &canvas) == 0) {
                    st.refs_valid[slot] = 1;
                }
            }
        }

        if (want) {
            *img = canvas;
            memset(&canvas, 0, sizeof(canvas));
            canvas_valid = 0;
            if (info_out) {
                info_out->duration_ticks = (int)fh.duration;
                info_out->is_last = fh.is_last;
            }
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            rc = 0;
            goto done;
        }
        jxl_fimage_free(ctx, &tmp);
        if (keyframe) idx++;
        jxl_toc_free(ctx, &toc);
        jxl_frame_header_free(ctx, &fh);
        if (last || end >= doc->container.cs_len) break;
        jxl_br_seek_byte(&br, end);
    }

    if (count_out) {
        *count_out = idx;
        rc = 0;
    } else {
        JXL_ERR(ctx, "render: no such frame %d", frame_no);
    }

done:
    jxl_fimage_free(ctx, &canvas);
    jxl_frame_state_free(ctx, &st);
    return rc;
}

static int decode_frame_planes(jxl_doc *doc, int frame_no, jxl_fimage *img) {
    return walk_frames(doc, frame_no, img, NULL, NULL);
}

jxl_image *jxl_frame_render(jxl_doc *doc, int frame_no, jxl_format fmt) {
    jxl_ctx *ctx;
    jxl_render_info info;
    jxl_fimage img;
    jxl_image *out = NULL;
    size_t total;

    if (!doc) return NULL;
    ctx = doc->ctx;
    if (jxl_frame_render_info(doc, frame_no, fmt, &info) != 0) return NULL;
    memset(&img, 0, sizeof(img));
    if (decode_frame_planes(doc, frame_no, &img) != 0) return NULL;

    out = (jxl_image *)jxl_calloc(ctx, 1, sizeof(jxl_image));
    if (!out) goto done;
    out->width = info.width;
    out->height = info.height;
    out->format = info.format;
    out->stride = info.width * jxl_format_bpp(info.format);
    if (!jxl_size_mul((size_t)out->stride, (size_t)out->height, &total)) {
        jxl_free(ctx, out);
        out = NULL;
        goto done;
    }
    /* write_pixels covers every byte of every row -- stride is exactly
       width * bpp and it writes all of them -- so zeroing this first is
       pure cost. -DJXL_POISON_UNINIT fills this and the LZ77 window with
       0xCD instead of leaving them to the allocator, so that anything
       depending on the old zeroing shows up as a difference rather than as
       a silent zero that happens to look right. Diffed that way against the
       zero-filled build over the whole corpus: no file changed. */
    out->data = (uint8_t *)jxl_malloc(ctx, total ? total : 1);
#ifdef JXL_POISON_UNINIT
    if (out->data) memset(out->data, 0xCD, total ? total : 1);
#endif
    if (!out->data) {
        jxl_free(ctx, out);
        out = NULL;
        goto done;
    }
    if (write_pixels(ctx, doc, &img, info.format, out->data, out->stride) != 0) {
        jxl_image_destroy(ctx, out);
        out = NULL;
    }

done:
    jxl_fimage_free(ctx, &img);
    return out;
}

int jxl_frame_render_into(jxl_doc *doc, int frame_no, jxl_format fmt,
                          uint8_t *dst, int stride) {
    jxl_ctx *ctx;
    jxl_render_info info;
    jxl_fimage img;
    int rc;

    if (!doc || !dst) return -1;
    ctx = doc->ctx;
    if (jxl_frame_render_info(doc, frame_no, fmt, &info) != 0) return -1;
    memset(&img, 0, sizeof(img));
    if (decode_frame_planes(doc, frame_no, &img) != 0) return -1;
    rc = write_pixels(ctx, doc, &img, info.format, dst, stride);
    jxl_fimage_free(ctx, &img);
    return rc;
}
