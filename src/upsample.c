/* upsample.c -- the non-separable 2x / 4x / 8x upsampling filter.
 *
 * A frame may be coded at 1/2, 1/4 or 1/8 resolution and upsampled by the
 * decoder (cjxl --resampling). Each input sample expands to an NxN block, and
 * every output sample in that block is a 5x5 weighted sum of the input
 * neighbourhood -- a different set of 25 weights per position in the block.
 * The weights live in CustomTransformData (headers.c) and default to
 * jxl_default_up2/up4/up8.
 *
 * The codestream stores only the upper-left quadrant of the NxN weight set
 * and only the upper triangle of that, because the kernel is symmetric under
 * both reflections and under transposition; build_kernel expands it back out.
 *
 * Each output sample is clamped to the min and max of the 5x5 input
 * neighbourhood it came from, which is what stops the filter ringing at
 * edges. libjxl does the same (render_pipeline/stage_upsampling.cc).
 */
#include "jxl_internal.h"

/* Mirror without repeating the edge sample, as everywhere else in the
   decoder. */
static uint32_t up_mirror(int64_t v, uint32_t len) {
    for (;;) {
        if (v < 0) v = -(v + 1);
        else if ((uint64_t)v >= len) v = (int64_t)len * 2 - 1 - v;
        else return (uint32_t)v;
    }
}

/* Expands the stored triangle into the full N*N sets of 25 weights. The
   stored index is the triangular-packed (my, mx) with my <= mx, where i and j
   run over the 5*H taps of one quadrant. */
static void build_kernel(float *kernel, uint32_t shift, const float *w) {
    uint32_t N = 1u << shift, H = N / 2;
    uint32_t ky, kx, py, px;
    for (ky = 0; ky < H; ky++) {
        for (kx = 0; kx < H; kx++) {
            size_t o0 = ((size_t)ky * N + kx) * 25;
            size_t o1 = ((size_t)ky * N + (N - 1 - kx)) * 25;
            size_t o2 = ((size_t)(N - 1 - ky) * N + kx) * 25;
            size_t o3 = ((size_t)(N - 1 - ky) * N + (N - 1 - kx)) * 25;
            for (py = 0; py < 5; py++) {
                for (px = 0; px < 5; px++) {
                    uint32_t j = 5 * ky + py;
                    uint32_t i = 5 * kx + px;
                    uint32_t my = i < j ? i : j;
                    uint32_t mx = i < j ? j : i;
                    size_t idx = (size_t)5 * H * my - (size_t)my * (my - 1) / 2
                                 + mx - my;
                    float v = w[idx];
                    kernel[o0 + (size_t)py * 5 + px] = v;
                    kernel[o1 + (size_t)py * 5 + (4 - px)] = v;
                    kernel[o2 + (size_t)(4 - py) * 5 + px] = v;
                    kernel[o3 + (size_t)(4 - py) * 5 + (4 - px)] = v;
                }
            }
        }
    }
}

/* SSE2 is baseline on x64. This is the innermost operation in the whole
   filter -- N*N of them per input sample, 25 multiply-adds each -- so it is
   the one place worth writing by hand. -DJXL_UPSAMPLE_FORCE_SCALAR builds
   the scalar path alone, so the vector paths can be diffed against it. */
#if !defined(JXL_UPSAMPLE_FORCE_SCALAR) && \
    (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define JXL_UPSAMPLE_SSE2 1
#include <emmintrin.h>
#include <immintrin.h>
#endif

/* These two run only where up_block4 below cannot: the first two and last
   two columns of a row, the last partial block of a row or column, and any
   plane too narrow for the wide path at all. That is a thin border, so both
   are left scalar -- which is also what lets the whole filter be bit-identical
   between the scalar and vector builds. The earlier hand-vectorised dot
   product summed four partial accumulators and reduced them, an association
   the scalar loop does not use, so the two builds did not agree; now they do.
*/
static void up_minmax25(const float *a, float *out_lo, float *out_hi) {
    float lo = a[0], hi = a[0];
    int n;
    for (n = 1; n < 25; n++) {
        if (a[n] < lo) lo = a[n];
        if (a[n] > hi) hi = a[n];
    }
    *out_lo = lo;
    *out_hi = hi;
}

static float up_dot25(const float *a, const float *b) {
    float sum = 0.0f;
    int n;
    for (n = 0; n < 25; n++) sum += a[n] * b[n];
    return sum;
}

#ifdef JXL_UPSAMPLE_SSE2
/* Four consecutive input samples at once, for the interior where none of
   them needs mirroring and all of them write a full N wide block.

   Vectorising across x rather than across the 25 taps changes the shape of
   the work completely. The tap a lane needs is `srow[py][x + px - 2]`, so
   four consecutive x are four consecutive floats: the neighbourhood never
   has to be gathered into nb[] at all, each tap is one unaligned load, and
   the 25 products accumulate in the plain scalar order -- no horizontal
   reduction per output, and the result is bit-identical to the scalar path
   rather than merely close to it, which the per-tap version was not.

   Returns the number of input samples consumed, always 4. */
static void up_block4(const float *const *srow, uint32_t x, uint32_t N,
                      const float *kernel, float *dst_rows[8], uint32_t ny) {
    __m128 vlo, vhi;
    uint32_t oy, ox;
    int py, px;

    {   /* The clamp bounds for each of the four lanes: min and max over that
           lane's own 5x5 window, taken as five sliding 5-wide windows. */
        __m128 mn = _mm_loadu_ps(srow[0] + x - 2), mx = mn;
        for (py = 0; py < 5; py++) {
            const float *r = srow[py] + x - 2;
            for (px = (py == 0) ? 1 : 0; px < 5; px++) {
                __m128 v = _mm_loadu_ps(r + px);
                mn = _mm_min_ps(mn, v);
                mx = _mm_max_ps(mx, v);
            }
        }
        vlo = mn;
        vhi = mx;
    }

    for (oy = 0; oy < ny; oy++) {
        float *drow = dst_rows[oy];
        __m128 out[8];
        for (ox = 0; ox < N; ox++) {
            const float *k = kernel + ((size_t)oy * N + ox) * 25;
            __m128 acc = _mm_setzero_ps();
            __m128 m;
            int t = 0;
            for (py = 0; py < 5; py++) {
                const float *r = srow[py] + x - 2;
                for (px = 0; px < 5; px++, t++) {
                    acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(r + px),
                                                     _mm_set1_ps(k[t])));
                }
            }
            /* Select rather than min/max so an unordered compare keeps acc,
               exactly as the scalar `if (sum < lo)` pair does. */
            m = _mm_cmplt_ps(acc, vlo);
            acc = _mm_or_ps(_mm_and_ps(m, vlo), _mm_andnot_ps(m, acc));
            m = _mm_cmpgt_ps(acc, vhi);
            acc = _mm_or_ps(_mm_and_ps(m, vhi), _mm_andnot_ps(m, acc));
            out[ox] = acc;
        }
        /* Each vector holds one output column for four input samples, but
           the row wants them interleaved. N is 2, 4 or 8; the first two are
           a shuffle, the last falls back to lane stores. */
        if (N == 2) {
            _mm_storeu_ps(drow + x * 2, _mm_unpacklo_ps(out[0], out[1]));
            _mm_storeu_ps(drow + x * 2 + 4, _mm_unpackhi_ps(out[0], out[1]));
        } else if (N == 4) {
            _MM_TRANSPOSE4_PS(out[0], out[1], out[2], out[3]);
            _mm_storeu_ps(drow + x * 4, out[0]);
            _mm_storeu_ps(drow + x * 4 + 4, out[1]);
            _mm_storeu_ps(drow + x * 4 + 8, out[2]);
            _mm_storeu_ps(drow + x * 4 + 12, out[3]);
        } else {
            for (ox = 0; ox < N; ox++) {
                float tmp[4];
                _mm_storeu_ps(tmp, out[ox]);
                drow[(size_t)(x + 0) * N + ox] = tmp[0];
                drow[(size_t)(x + 1) * N + ox] = tmp[1];
                drow[(size_t)(x + 2) * N + ox] = tmp[2];
                drow[(size_t)(x + 3) * N + ox] = tmp[3];
            }
        }
    }
}
/* Eight input samples at a time. Same shape as up_block4 -- the profile put
   26.4% of a resampled decode in that function against libjxl's 6.2% for the
   identical stage, and the difference is lanes: libjxl runs this eight wide
   on AVX2 where we ran four.

   The accumulate widens directly. Only the store needs thought, because each
   vector holds one output column for eight input samples while the row wants
   them interleaved. Splitting each accumulator into its two 128-bit halves
   turns that back into exactly the four-sample interleave up_block4 already
   does, applied twice -- no 8-wide shuffle network, and the N == 2 and N == 4
   cases keep their existing unpack and transpose.

   No FMA, for the usual reason: it would round once where the scalar and SSE2
   paths round twice, and all three are diffed against each other. */
JXL_TARGET_AVX2
static void up_block8(const float *const *srow, uint32_t x, uint32_t N,
                      const float *kernel, float *dst_rows[8], uint32_t ny) {
    __m256 vlo, vhi;
    uint32_t oy, ox;
    int py, px;

    {
        __m256 mn = _mm256_loadu_ps(srow[0] + x - 2), mx = mn;
        for (py = 0; py < 5; py++) {
            const float *r = srow[py] + x - 2;
            for (px = (py == 0) ? 1 : 0; px < 5; px++) {
                __m256 v = _mm256_loadu_ps(r + px);
                mn = _mm256_min_ps(mn, v);
                mx = _mm256_max_ps(mx, v);
            }
        }
        vlo = mn;
        vhi = mx;
    }

    for (oy = 0; oy < ny; oy++) {
        float *drow = dst_rows[oy];
        __m128 lo[8], hi[8];
        for (ox = 0; ox < N; ox++) {
            const float *k = kernel + ((size_t)oy * N + ox) * 25;
            __m256 acc = _mm256_setzero_ps();
            __m256 m;
            int t = 0;
            for (py = 0; py < 5; py++) {
                const float *r = srow[py] + x - 2;
                for (px = 0; px < 5; px++, t++) {
                    acc = _mm256_add_ps(acc,
                        _mm256_mul_ps(_mm256_loadu_ps(r + px),
                                      _mm256_set1_ps(k[t])));
                }
            }
            m = _mm256_cmp_ps(acc, vlo, _CMP_LT_OQ);
            acc = _mm256_or_ps(_mm256_and_ps(m, vlo),
                               _mm256_andnot_ps(m, acc));
            m = _mm256_cmp_ps(acc, vhi, _CMP_GT_OQ);
            acc = _mm256_or_ps(_mm256_and_ps(m, vhi),
                               _mm256_andnot_ps(m, acc));
            lo[ox] = _mm256_castps256_ps128(acc);
            hi[ox] = _mm256_extractf128_ps(acc, 1);
        }
        if (N == 2) {
            _mm_storeu_ps(drow + x * 2, _mm_unpacklo_ps(lo[0], lo[1]));
            _mm_storeu_ps(drow + x * 2 + 4, _mm_unpackhi_ps(lo[0], lo[1]));
            _mm_storeu_ps(drow + (x + 4) * 2, _mm_unpacklo_ps(hi[0], hi[1]));
            _mm_storeu_ps(drow + (x + 4) * 2 + 4,
                          _mm_unpackhi_ps(hi[0], hi[1]));
        } else if (N == 4) {
            _MM_TRANSPOSE4_PS(lo[0], lo[1], lo[2], lo[3]);
            _mm_storeu_ps(drow + x * 4, lo[0]);
            _mm_storeu_ps(drow + x * 4 + 4, lo[1]);
            _mm_storeu_ps(drow + x * 4 + 8, lo[2]);
            _mm_storeu_ps(drow + x * 4 + 12, lo[3]);
            _MM_TRANSPOSE4_PS(hi[0], hi[1], hi[2], hi[3]);
            _mm_storeu_ps(drow + (x + 4) * 4, hi[0]);
            _mm_storeu_ps(drow + (x + 4) * 4 + 4, hi[1]);
            _mm_storeu_ps(drow + (x + 4) * 4 + 8, hi[2]);
            _mm_storeu_ps(drow + (x + 4) * 4 + 12, hi[3]);
        } else {
            for (ox = 0; ox < N; ox++) {
                float t0[4], t1[4];
                _mm_storeu_ps(t0, lo[ox]);
                _mm_storeu_ps(t1, hi[ox]);
                drow[(size_t)(x + 0) * N + ox] = t0[0];
                drow[(size_t)(x + 1) * N + ox] = t0[1];
                drow[(size_t)(x + 2) * N + ox] = t0[2];
                drow[(size_t)(x + 3) * N + ox] = t0[3];
                drow[(size_t)(x + 4) * N + ox] = t1[0];
                drow[(size_t)(x + 5) * N + ox] = t1[1];
                drow[(size_t)(x + 6) * N + ox] = t1[2];
                drow[(size_t)(x + 7) * N + ox] = t1[3];
            }
        }
    }
    _mm256_zeroupper();
}
#endif

/* Picks the weight set for a shift of 1, 2 or 3. */
static const float *weights_for(const jxl_image_metadata *meta, uint32_t shift) {
    if (shift == 1) return meta->up2;
    if (shift == 2) return meta->up4;
    return meta->up8;
}

int jxl_upsample_plane(jxl_ctx *ctx, jxl_fplane *p, uint32_t shift,
                       const jxl_image_metadata *meta, uint32_t out_w,
                       uint32_t out_h) {
    uint32_t N = 1u << shift;
    uint32_t w = p->w, h = p->h, x, y;
    float *kernel = NULL;
    jxl_fplane dst;
    int rc = -1;
#ifdef JXL_UPSAMPLE_SSE2
    const int use_avx2 = jxl_has_avx2();
#endif

    if (shift == 0 || shift > 3) return 0;
    if (w == 0 || h == 0) return 0;

    memset(&dst, 0, sizeof(dst));
    kernel = (float *)jxl_calloc(ctx, (size_t)N * N * 25, sizeof(float));
    if (!kernel) goto done;
    build_kernel(kernel, shift, weights_for(meta, shift));

    /* The upsampled frame is w*N wide but the image may be a few samples
       narrower, so allocate the crop and simply skip the samples past it. */
    if (jxl_fplane_alloc_uninit(ctx, &dst, out_w, out_h) != 0) goto done;

    for (y = 0; y < h; y++) {
        /* Whether this input row's whole N-row output block is inside the
           crop -- the wide paths below all write N full rows. */
#ifdef JXL_UPSAMPLE_SSE2
        const int y_full = (uint64_t)(y + 1) * N <= out_h;
#endif
        /* The five source rows depend only on y, so mirror them once per row
           rather than once per sample -- and inside the row, only the first
           and last two columns need mirroring at all. */
        const float *srow[5];
        int iy;
        for (iy = -2; iy <= 2; iy++) {
            srow[iy + 2] = p->data +
                (size_t)up_mirror((int64_t)y + iy, h) * p->stride;
        }
        for (x = 0; x < w; ) {
            float nb[25];
            float lo, hi;
            uint32_t oy, ox, ny, nx;
            int ix;
            int n = 0;

#ifdef JXL_UPSAMPLE_SSE2
            /* Four (or eight) at a time when they are all interior -- no
               mirroring, so the taps are plain contiguous loads -- and all of
               them write a whole N-wide block inside the crop. The eight-wide
               guard needs four more input samples of margin on the right for
               the last tap, and four more output blocks inside the crop. */
            if (y_full) {
                float *drows[8];
                if (use_avx2 && x >= 2 && x + 10 <= w &&
                    (uint64_t)(x + 8) * N <= out_w) {
                    for (oy = 0; oy < N; oy++) {
                        drows[oy] = dst.data +
                                    (size_t)(y * N + oy) * dst.stride;
                    }
                    up_block8(srow, x, N, kernel, drows, N);
                    x += 8;
                    continue;
                }
                if (x >= 2 && x + 6 <= w && (uint64_t)(x + 4) * N <= out_w) {
                    for (oy = 0; oy < N; oy++) {
                        drows[oy] = dst.data +
                                    (size_t)(y * N + oy) * dst.stride;
                    }
                    up_block4(srow, x, N, kernel, drows, N);
                    x += 4;
                    continue;
                }
            }
#endif

            if (x >= 2 && x + 2 < w) {
                for (iy = 0; iy < 5; iy++) {
                    const float *r = srow[iy] + x - 2;
                    nb[n] = r[0]; nb[n + 1] = r[1]; nb[n + 2] = r[2];
                    nb[n + 3] = r[3]; nb[n + 4] = r[4];
                    n += 5;
                }
            } else {
                for (iy = 0; iy < 5; iy++) {
                    for (ix = -2; ix <= 2; ix++) {
                        nb[n++] = srow[iy][up_mirror((int64_t)x + ix, w)];
                    }
                }
            }
            up_minmax25(nb, &lo, &hi);
            /* The image can end mid-block, so the last input sample of a row
               or column writes fewer than N outputs. Deciding that once per
               input sample keeps the per-output bounds test out of the two
               innermost loops, where it never fires except on that last
               block. */
            ny = y * N < out_h ? out_h - y * N : 0;
            if (ny > N) ny = N;
            nx = x * N < out_w ? out_w - x * N : 0;
            if (nx > N) nx = N;
            for (oy = 0; oy < ny; oy++) {
                float *drow = dst.data + (size_t)(y * N + oy) * dst.stride;
                const float *krow = kernel + ((size_t)oy * N) * 25;
                uint32_t dx0 = x * N;
                for (ox = 0; ox < nx; ox++) {
                    float sum = up_dot25(nb, krow + (size_t)ox * 25);
                    if (sum < lo) sum = lo;
                    if (sum > hi) sum = hi;
                    drow[dx0 + ox] = sum;
                }
            }
            x++;
        }
    }

    jxl_free(ctx, p->data);
    *p = dst;
    memset(&dst, 0, sizeof(dst));
    rc = 0;

done:
    jxl_free(ctx, kernel);
    jxl_free(ctx, dst.data);
    return rc;
}
