/* filter.c -- the two in-loop restoration filters.
 *
 * Gaborish is a fixed 3x3 kernel with per-channel weights, run once after the
 * inverse transform. The edge-preserving filter (EPF) is a self-similarity
 * filter run in up to three passes with progressively smaller kernels; its
 * strength comes from the per-block sigma map carried by the varblock
 * metadata.
 *
 * Gaborish clamps at the borders, EPF mirrors (without repeating the edge
 * sample) -- matching libjxl in both cases. Both resolve that edge handling
 * per row and then run a mirror-free interior, which is where the SSE2 paths
 * live; those are written to associate their adds exactly as the scalar code
 * does, so the two produce bit-identical output (-DJXL_EPF_FORCE_SCALAR
 * builds the scalar side alone to check it).
 */
#include "jxl_internal.h"

#include <math.h>
#include <stddef.h>

/* SSE2 is baseline on x64. epf_pass is about half the decode of a VarDCT
   file, and step 0 alone is 12 taps x 5 SAD offsets x 3 channels per
   sample, so it is where hand-vectorising pays most. */
/* -DJXL_EPF_FORCE_SCALAR builds the scalar path only. The vector path is
   meant to be bit-identical to it, and that is worth being able to prove
   rather than assert: build both and diff the output. */
#if !defined(JXL_EPF_FORCE_SCALAR) &&     (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) ||      (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define JXL_EPF_SSE2 1
#include <emmintrin.h>
#include <immintrin.h>
#endif

static uint32_t jxl_mirror(int64_t offset, uint32_t len) {
    for (;;) {
        if (offset < 0) offset = -(offset + 1);
        else if ((uint64_t)offset >= len) offset = (int64_t)len * 2 - 1 - offset;
        else return (uint32_t)offset;
    }
}

int jxl_apply_gabor(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                    size_t stride, const float weights[3][2]) {
    float *ring;
    int c;
    if (w == 0 || h == 0) return 0;
    /* The filter writes row y from rows y-1, y and y+1 of the input. Row y+1
       has not been written yet when row y is produced, so it can be read
       straight out of the plane; only rows y-1 and y need to survive being
       overwritten. Two saved rows is therefore enough to run the whole thing
       in place, where this used to fill a full w*h scratch plane and copy it
       back -- three full passes over the image per channel, in DRAM, against
       one row-sized buffer that stays in L1. */
    ring = (float *)jxl_malloc(ctx, (size_t)w * 2 * sizeof(float));
    if (!ring) return -1;

    for (c = 0; c < 3; c++) {
        float w0 = weights[c][0], w1 = weights[c][1];
        float gw = 1.0f / (1.0f + w0 * 4.0f + w1 * 4.0f);
        uint32_t x, y;
#ifdef JXL_EPF_SSE2
        const __m128 v0 = _mm_set1_ps(w0), v1 = _mm_set1_ps(w1);
        const __m128 vg = _mm_set1_ps(gw);
#endif
        for (y = 0; y < h; y++) {
            /* Clamping depends only on the row, so resolve the three row
               pointers once instead of calling sample_clamped -- four
               branches -- eight times per sample. */
            float *cur = ring + (size_t)(y & 1u) * w;
            const float *rn, *rc, *rs;
            float *dst = plane[c] + (size_t)y * stride;
            /* Save row y before it is overwritten; row y-1 is still in the
               other half of the ring from the previous iteration. */
            memcpy(cur, dst, (size_t)w * sizeof(float));
            rc = cur;
            rn = y > 0 ? ring + (size_t)((y - 1) & 1u) * w : cur;
            rs = y + 1 < h ? plane[c] + (size_t)(y + 1) * stride : cur;
            uint32_t xlo = w > 1 ? 1 : 0, xhi = w > 1 ? w - 1 : 0;

            for (x = 0; x < xlo; x++) {
                uint32_t xm = 0, xp = (w > 1) ? 1 : 0;
                dst[x] = (rc[x] + (rn[x] + rs[x] + rc[xm] + rc[xp]) * w0 +
                          (rn[xm] + rn[xp] + rs[xm] + rs[xp]) * w1) * gw;
            }
            x = xlo;
#ifdef JXL_EPF_SSE2
            /* Interior only: no clamping, so every tap is a fixed offset.
               Lanes run the same operations in the same order as the scalar
               path, so the result is bit-identical. */
            for (; x + 4 <= xhi; x += 4) {
                __m128 cc = _mm_loadu_ps(rc + x);
                /* Left-to-right, exactly as the scalar expression associates:
                   ((n + s) + w) + e. Float addition is not associative, so
                   pairing the loads up differently changes the last bit --
                   which is how the scalar/vector diff caught it. */
                __m128 side = _mm_add_ps(_mm_loadu_ps(rn + x), _mm_loadu_ps(rs + x));
                __m128 diag;
                __m128 r;
                side = _mm_add_ps(side, _mm_loadu_ps(rc + x - 1));
                side = _mm_add_ps(side, _mm_loadu_ps(rc + x + 1));
                diag = _mm_add_ps(_mm_loadu_ps(rn + x - 1), _mm_loadu_ps(rn + x + 1));
                diag = _mm_add_ps(diag, _mm_loadu_ps(rs + x - 1));
                diag = _mm_add_ps(diag, _mm_loadu_ps(rs + x + 1));
                r = _mm_add_ps(cc, _mm_mul_ps(side, v0));
                r = _mm_add_ps(r, _mm_mul_ps(diag, v1));
                _mm_storeu_ps(dst + x, _mm_mul_ps(r, vg));
            }
#endif
            for (; x < xhi; x++) {
                dst[x] = (rc[x] +
                          (rn[x] + rs[x] + rc[x - 1] + rc[x + 1]) * w0 +
                          (rn[x - 1] + rn[x + 1] + rs[x - 1] + rs[x + 1]) * w1) * gw;
            }
            for (x = xhi; x < w; x++) {
                uint32_t xm = x > 0 ? x - 1 : 0;
                uint32_t xp = x + 1 < w ? x + 1 : w - 1;
                dst[x] = (rc[x] + (rn[x] + rs[x] + rc[xm] + rc[xp]) * w0 +
                          (rn[xm] + rn[xp] + rs[xm] + rs[xp]) * w1) * gw;
            }
        }
    }
    jxl_free(ctx, ring);
    return 0;
}

/* Kernel taps per EPF step, and the offsets summed for the distance term. */
static const int8_t epf_kernel_2[12][2] = {
    {0,-2},{-1,-1},{0,-1},{1,-1},{-2,0},{-1,0},{1,0},{2,0},
    {-1,1},{0,1},{1,1},{0,2}
};
static const int8_t epf_kernel_1[4][2] = {{0,-1},{0,1},{-1,0},{1,0}};
static const int8_t epf_dist_0[5][2] = {{0,-1},{1,0},{0,0},{-1,0},{0,1}};
static const int8_t epf_dist_1[5][2] = {{0,-1},{0,0},{0,1},{-1,0},{1,0}};
static const int8_t epf_dist_2[1][2] = {{0,0}};

/* The weight is 1 + dist * (a negative constant / sigma * step_mul). Sigma
   and step_mul are fixed for a whole sample, so the reciprocal is hoisted out
   of the tap loop; only the multiply-add stays per tap. */
#define EPF_SIGMA_MUL (6.6f * (0.70710678118654752f - 1.0f))

#ifdef JXL_EPF_SSE2
/* epf_pass, eight samples at a time. This is the densest arithmetic in the
   decoder -- step 0 is 12 kernel taps x 5 SAD offsets x 3 channels per
   sample, all out of an L1-resident window -- so it is the one kernel where
   doubling the vector width should actually pay, unlike the element-wise
   colour loops.
 *
 * An octet is tidier than the quad the SSE2 path uses: sigma blocks are 8
 * wide, so an 8-aligned run is exactly one block, and border_sad_mul applies
 * to lanes 0 and 7 only -- a single fixed pattern rather than two. */
JXL_TARGET_AVX2
static void epf_row8(float *in[3], float *out[3], size_t row, uint32_t x,
                     const ptrdiff_t koff[12], const ptrdiff_t doff[5],
                     int nkernel, int ndist, const float cscale[3],
                     float sigma_val, float step_mul, float border_mul,
                     int is_y_border) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 dist8[12], sum8[3], sw, nis, smv;
    int c, k, d;

    if (is_y_border) {
        smv = _mm256_set1_ps(border_mul);
    } else {
        smv = _mm256_setr_ps(border_mul, step_mul, step_mul, step_mul,
                             step_mul, step_mul, step_mul, border_mul);
    }
    nis = _mm256_mul_ps(_mm256_set1_ps(EPF_SIGMA_MUL / sigma_val), smv);

    for (k = 0; k < nkernel; k++) dist8[k] = _mm256_setzero_ps();
    for (c = 0; c < 3; c++) {
        const float *p = in[c] + row + x;
        __m256 cs = _mm256_set1_ps(cscale[c]);
        __m256 cen[5];
        for (d = 0; d < ndist; d++) cen[d] = _mm256_loadu_ps(p + doff[d]);
        for (k = 0; k < nkernel; k++) {
            const float *pk = p + koff[k];
            __m256 acc = _mm256_setzero_ps();
            for (d = 0; d < ndist; d++) {
                acc = _mm256_add_ps(acc, _mm256_and_ps(absmask,
                    _mm256_sub_ps(_mm256_loadu_ps(pk + doff[d]), cen[d])));
            }
            dist8[k] = _mm256_add_ps(dist8[k], _mm256_mul_ps(cs, acc));
        }
    }
    for (c = 0; c < 3; c++) sum8[c] = _mm256_loadu_ps(in[c] + row + x);
    sw = one;
    for (k = 0; k < nkernel; k++) {
        __m256 wgt = _mm256_add_ps(one, _mm256_mul_ps(dist8[k], nis));
        wgt = _mm256_max_ps(wgt, _mm256_setzero_ps());
        sw = _mm256_add_ps(sw, wgt);
        for (c = 0; c < 3; c++) {
            sum8[c] = _mm256_add_ps(sum8[c], _mm256_mul_ps(wgt,
                _mm256_loadu_ps(in[c] + row + x + koff[k])));
        }
    }
    sw = _mm256_div_ps(one, sw);
    for (c = 0; c < 3; c++)
        _mm256_storeu_ps(out[c] + row + x, _mm256_mul_ps(sum8[c], sw));
    _mm256_zeroupper();
}
#endif

/* A sample whose whole footprint is inside the image needs no mirroring, so
   every neighbour it reads is a fixed sample offset from its centre. Those
   offsets depend only on the pass, so they are tabulated once per pass. */
static int epf_pass(float *in[3], float *out[3], uint32_t w, uint32_t h,
                    size_t stride, const float *sigma, uint32_t sigma_stride,
                    const jxl_epf *epf, int step) {
    const int8_t (*kernel)[2];
    const int8_t (*dist_off)[2];
    ptrdiff_t koff[12];      /* kernel tap -> sample offset */
    ptrdiff_t doff[5];       /* SAD footprint tap around the centre */
    float cscale[3];
    int nkernel, ndist, pad;
    float step_mul, border_mul;
    uint32_t x, y;
    int c, k, d;
#ifdef JXL_EPF_SSE2
    __m128 epf_absmask, sm_border, sm_lo, sm_hi;
    const int use_avx2 = jxl_has_avx2();
#endif

    if (step == 0) {
        kernel = epf_kernel_2; nkernel = 12;
        dist_off = epf_dist_0; ndist = 5;
        step_mul = epf->pass0_sigma_scale;
        pad = 3;             /* kernel reaches 2, its footprint one further */
    } else if (step == 1) {
        kernel = epf_kernel_1; nkernel = 4;
        dist_off = epf_dist_1; ndist = 5;
        step_mul = 1.0f;
        pad = 2;
    } else {
        kernel = epf_kernel_1; nkernel = 4;
        dist_off = epf_dist_2; ndist = 1;
        step_mul = epf->pass2_sigma_scale;
        pad = 1;
    }
    border_mul = step_mul * epf->border_sad_mul;
#ifdef JXL_EPF_SSE2
    epf_absmask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    sm_border = _mm_set1_ps(border_mul);
    sm_lo = _mm_setr_ps(border_mul, step_mul, step_mul, step_mul);
    sm_hi = _mm_setr_ps(step_mul, step_mul, step_mul, border_mul);
#endif
    for (c = 0; c < 3; c++) cscale[c] = epf->channel_scale[c];
    for (d = 0; d < ndist; d++)
        doff[d] = (ptrdiff_t)dist_off[d][1] * (ptrdiff_t)stride + dist_off[d][0];
    for (k = 0; k < nkernel; k++)
        koff[k] = (ptrdiff_t)kernel[k][1] * (ptrdiff_t)stride + kernel[k][0];

    for (y = 0; y < h; y++) {
        int is_y_border = ((y + 1) & 6u) == 0;
        /* Rows this close to an edge mirror; they are O(pad) of the image. */
        int y_inside = (y >= (uint32_t)pad && y + (uint32_t)pad < h);
        const float *sigma_row = sigma + (size_t)(y / 8) * sigma_stride;
        size_t row = (size_t)y * stride;
        for (x = 0; x < w; ) {
            float sigma_val = sigma_row[x / 8];

#ifdef JXL_EPF_SSE2
            /* Four samples at a time down the row. Vectorising across x (not
               across taps) means every lane runs the same operations in the
               same order as the scalar path below, so the output is
               bit-identical -- no tolerance risk, and the scalar path stays
               the reference. Requires the whole quad to be in the mirror-free
               interior and inside one sigma block: blocks are 8 wide and the
               quad is 4-aligned, so x/8 is constant across it. */
            if (use_avx2 && y_inside && (x & 7u) == 0 &&
                x >= (uint32_t)pad && x + 7 + (uint32_t)pad < w) {
                if (sigma_val < 0.3f) {
                    for (c = 0; c < 3; c++) {
                        memcpy(out[c] + row + x, in[c] + row + x,
                               8 * sizeof(float));
                    }
                    x += 8;
                    continue;
                }
                epf_row8(in, out, row, x, koff, doff, nkernel, ndist, cscale,
                         sigma_val, step_mul, border_mul, is_y_border);
                x += 8;
                continue;
            }
            if (y_inside && (x & 3u) == 0 &&
                x >= (uint32_t)pad && x + 3 + (uint32_t)pad < w) {
                __m128 dist4[12], sum4[3], sw, nis;
                if (sigma_val < 0.3f) {
                    for (c = 0; c < 3; c++) {
                        _mm_storeu_ps(out[c] + row + x,
                                      _mm_loadu_ps(in[c] + row + x));
                    }
                    x += 4;
                    continue;
                }
                /* sm is per-lane: within an 8-wide block only lanes 0 and 7
                   take border_mul, and a 4-aligned quad covers either 0..3 or
                   4..7, so there are just two patterns. */
                nis = _mm_mul_ps(_mm_set1_ps(EPF_SIGMA_MUL / sigma_val),
                                 is_y_border ? sm_border
                                             : ((x & 7u) == 0 ? sm_lo : sm_hi));
                for (k = 0; k < nkernel; k++) dist4[k] = _mm_setzero_ps();
                for (c = 0; c < 3; c++) {
                    const float *p = in[c] + row + x;
                    __m128 cs = _mm_set1_ps(cscale[c]);
                    __m128 cen[5];
                    for (d = 0; d < ndist; d++) cen[d] = _mm_loadu_ps(p + doff[d]);
                    for (k = 0; k < nkernel; k++) {
                        const float *pk = p + koff[k];
                        __m128 acc = _mm_setzero_ps();
                        for (d = 0; d < ndist; d++) {
                            acc = _mm_add_ps(acc, _mm_and_ps(epf_absmask,
                                _mm_sub_ps(_mm_loadu_ps(pk + doff[d]), cen[d])));
                        }
                        dist4[k] = _mm_add_ps(dist4[k], _mm_mul_ps(cs, acc));
                    }
                }
                for (c = 0; c < 3; c++) sum4[c] = _mm_loadu_ps(in[c] + row + x);
                sw = _mm_set1_ps(1.0f);
                for (k = 0; k < nkernel; k++) {
                    __m128 wgt = _mm_add_ps(_mm_set1_ps(1.0f),
                                            _mm_mul_ps(dist4[k], nis));
                    wgt = _mm_max_ps(wgt, _mm_setzero_ps());
                    sw = _mm_add_ps(sw, wgt);
                    for (c = 0; c < 3; c++) {
                        sum4[c] = _mm_add_ps(sum4[c], _mm_mul_ps(wgt,
                            _mm_loadu_ps(in[c] + row + x + koff[k])));
                    }
                }
                /* Real division, not _mm_rcp_ps: the approximate reciprocal
                   would diverge from the scalar path. */
                sw = _mm_div_ps(_mm_set1_ps(1.0f), sw);
                for (c = 0; c < 3; c++) {
                    _mm_storeu_ps(out[c] + row + x, _mm_mul_ps(sum4[c], sw));
                }
                x += 4;
                continue;
            }
#endif
            float dist[12];   /* SAD to each kernel tap, all channels folded in */
            size_t soff[12];  /* kernel tap -> absolute sample index */
            float sum[3];
            float sum_weights, inv_w, sm, neg_inv_sigma;

            if (sigma_val < 0.3f) {
                for (c = 0; c < 3; c++) out[c][row + x] = in[c][row + x];
                x++;
                continue;
            }
            if (is_y_border || (x & 7u) == 0 || (x & 7u) == 7) sm = border_mul;
            else sm = step_mul;
            neg_inv_sigma = EPF_SIGMA_MUL / sigma_val * sm;

            /* The SADs come first, one channel at a time: a channel's whole
               footprint then comes from one plane, and its centre samples stay
               in registers across the taps. Folding the channels into dist[]
               in channel order keeps the sum bit-identical to doing it per
               tap, which is what libjxl's vector loop also does. */
            for (k = 0; k < nkernel; k++) dist[k] = 0.0f;

            if (y_inside && x >= (uint32_t)pad && x + (uint32_t)pad < w) {
                /* Fast path: nothing mirrors, so every neighbour is a fixed
                   offset off the centre sample. */
                for (k = 0; k < nkernel; k++) soff[k] = row + x + koff[k];
                for (c = 0; c < 3; c++) {
                    const float *p = in[c] + row + x;
                    float cs = cscale[c];
                    float cen[5];
                    for (d = 0; d < ndist; d++) cen[d] = p[doff[d]];
                    for (k = 0; k < nkernel; k++) {
                        const float *pk = p + koff[k];
                        float acc = 0.0f;
                        for (d = 0; d < ndist; d++)
                            acc += fabsf(pk[doff[d]] - cen[d]);
                        dist[k] += cs * acc;
                    }
                }
            } else {
                /* Slow path for the pad-wide frame around the image, where
                   coordinates mirror. The mirrored indices do not depend on
                   the channel, so they are resolved once per sample. */
                size_t bo[5], ao[12][5];
                for (d = 0; d < ndist; d++) {
                    uint32_t bx = jxl_mirror((int64_t)x + dist_off[d][0], w);
                    uint32_t by = jxl_mirror((int64_t)y + dist_off[d][1], h);
                    bo[d] = (size_t)by * stride + bx;
                }
                for (k = 0; k < nkernel; k++) {
                    int64_t kx = (int64_t)x + kernel[k][0];
                    int64_t ky = (int64_t)y + kernel[k][1];
                    for (d = 0; d < ndist; d++) {
                        uint32_t ax = jxl_mirror(kx + dist_off[d][0], w);
                        uint32_t ay = jxl_mirror(ky + dist_off[d][1], h);
                        ao[k][d] = (size_t)ay * stride + ax;
                    }
                    soff[k] = (size_t)jxl_mirror(ky, h) * stride +
                              jxl_mirror(kx, w);
                }
                for (c = 0; c < 3; c++) {
                    const float *p = in[c];
                    float cs = cscale[c];
                    float cen[5];
                    for (d = 0; d < ndist; d++) cen[d] = p[bo[d]];
                    for (k = 0; k < nkernel; k++) {
                        float acc = 0.0f;
                        for (d = 0; d < ndist; d++)
                            acc += fabsf(p[ao[k][d]] - cen[d]);
                        dist[k] += cs * acc;
                    }
                }
            }

            for (c = 0; c < 3; c++) sum[c] = in[c][row + x];
            sum_weights = 1.0f;
            for (k = 0; k < nkernel; k++) {
                float weight = 1.0f + dist[k] * neg_inv_sigma;
                if (weight < 0.0f) weight = 0.0f;
                sum_weights += weight;
                for (c = 0; c < 3; c++) sum[c] += weight * in[c][soff[k]];
            }
            /* One reciprocal and three multiplies, as libjxl does, rather
               than three divisions. */
            inv_w = 1.0f / sum_weights;
            for (c = 0; c < 3; c++) out[c][row + x] = sum[c] * inv_w;
            x++;
        }
    }
    return 0;
}

int jxl_apply_epf(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                  size_t stride, const float *sigma, uint32_t sigma_stride,
                  const jxl_epf *epf) {
    float *scratch[3];
    float *in[3], *out[3], *t;
    int c, rc = -1;

    scratch[0] = scratch[1] = scratch[2] = NULL;
    if (!epf->enabled || w == 0 || h == 0) return 0;
    for (c = 0; c < 3; c++) {
        /* Every pass writes all w columns of every row, so there is nothing
           to zero; only the row padding stays undefined, and the copy back
           below is per row so it never travels. */
        size_t n;
        if (!jxl_size_mul(stride * h, sizeof(float), &n)) goto done;
        scratch[c] = (float *)jxl_malloc(ctx, n);
        if (!scratch[c]) goto done;
    }
    for (c = 0; c < 3; c++) { in[c] = plane[c]; out[c] = scratch[c]; }

    if (epf->iters == 3) {
        if (epf_pass(in, out, w, h, stride, sigma, sigma_stride, epf, 0) != 0) goto done;
        for (c = 0; c < 3; c++) { t = in[c]; in[c] = out[c]; out[c] = t; }
    }
    if (epf_pass(in, out, w, h, stride, sigma, sigma_stride, epf, 1) != 0) goto done;
    for (c = 0; c < 3; c++) { t = in[c]; in[c] = out[c]; out[c] = t; }
    if (epf->iters >= 2) {
        if (epf_pass(in, out, w, h, stride, sigma, sigma_stride, epf, 2) != 0) goto done;
        for (c = 0; c < 3; c++) { t = in[c]; in[c] = out[c]; out[c] = t; }
    }

    /* `in` holds the result; copy it back when it is the scratch buffer. */
    for (c = 0; c < 3; c++) {
        uint32_t y;
        if (in[c] == plane[c]) continue;
        for (y = 0; y < h; y++) {
            memcpy(plane[c] + (size_t)y * stride, in[c] + (size_t)y * stride,
                   (size_t)w * sizeof(float));
        }
    }
    rc = 0;

done:
    for (c = 0; c < 3; c++) jxl_free(ctx, scratch[c]);
    return rc;
}
