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

#ifdef JXL_EPF_SSE2
/* Keep the AVX2 target boundary outside both image loops. Gaborish is only a
   nine-tap stencil, so entering a target helper for every octet would erase a
   useful part of widening the existing SSE2 kernel. */
JXL_TARGET_AVX2
static void gabor_plane_avx2(float *plane, uint32_t w, uint32_t h,
                             size_t stride, float *ring,
                             float w0, float w1, float gw) {
    const __m256 v0 = _mm256_set1_ps(w0);
    const __m256 v1 = _mm256_set1_ps(w1);
    const __m256 vg = _mm256_set1_ps(gw);
    uint32_t x, y;

    for (y = 0; y < h; y++) {
        float *cur = ring + (size_t)(y & 1u) * w;
        const float *rn, *rc, *rs;
        float *dst = plane + (size_t)y * stride;
        uint32_t xlo = w > 1 ? 1 : 0;
        uint32_t xhi = w > 1 ? w - 1 : 0;

        memcpy(cur, dst, (size_t)w * sizeof(float));
        rc = cur;
        rn = y > 0 ? ring + (size_t)((y - 1) & 1u) * w : cur;
        rs = y + 1 < h ? plane + (size_t)(y + 1) * stride : cur;

        for (x = 0; x < xlo; x++) {
            uint32_t xm = 0, xp = (w > 1) ? 1 : 0;
            dst[x] = (rc[x] + (rn[x] + rs[x] + rc[xm] + rc[xp]) * w0 +
                      (rn[xm] + rn[xp] + rs[xm] + rs[xp]) * w1) * gw;
        }
        for (; x + 8 <= xhi; x += 8) {
            __m256 cc = _mm256_loadu_ps(rc + x);
            __m256 side =
                _mm256_add_ps(_mm256_loadu_ps(rn + x),
                              _mm256_loadu_ps(rs + x));
            __m256 diag;
            __m256 r;
            side = _mm256_add_ps(side, _mm256_loadu_ps(rc + x - 1));
            side = _mm256_add_ps(side, _mm256_loadu_ps(rc + x + 1));
            diag = _mm256_add_ps(_mm256_loadu_ps(rn + x - 1),
                                 _mm256_loadu_ps(rn + x + 1));
            diag = _mm256_add_ps(diag, _mm256_loadu_ps(rs + x - 1));
            diag = _mm256_add_ps(diag, _mm256_loadu_ps(rs + x + 1));
            r = _mm256_add_ps(cc, _mm256_mul_ps(side, v0));
            r = _mm256_add_ps(r, _mm256_mul_ps(diag, v1));
            _mm256_storeu_ps(dst + x, _mm256_mul_ps(r, vg));
        }
        for (; x < xhi; x++) {
            dst[x] = (rc[x] +
                      (rn[x] + rs[x] + rc[x - 1] + rc[x + 1]) * w0 +
                      (rn[x - 1] + rn[x + 1] + rs[x - 1] + rs[x + 1]) * w1) *
                     gw;
        }
        for (x = xhi; x < w; x++) {
            uint32_t xm = x > 0 ? x - 1 : 0;
            uint32_t xp = x + 1 < w ? x + 1 : w - 1;
            dst[x] = (rc[x] + (rn[x] + rs[x] + rc[xm] + rc[xp]) * w0 +
                      (rn[xm] + rn[xp] + rs[xm] + rs[xp]) * w1) * gw;
        }
    }
}
#endif

int jxl_apply_gabor(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                    size_t stride, const float weights[3][2]) {
    float *ring;
    int c;
#ifdef JXL_EPF_SSE2
    int use_avx2 = jxl_has_avx2();
#endif
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
        if (use_avx2) {
            gabor_plane_avx2(plane[c], w, h, stride, ring, w0, w1, gw);
            continue;
        }
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
    nis = _mm256_mul_ps(_mm256_set1_ps(sigma_val), smv);

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
    /* This is libjxl's non-high-precision EPF reciprocal as well. Besides
       avoiding the expensive vector divide, matching the reference estimate
       makes the filtered floats converge before the later color transform. */
    sw = _mm256_rcp_ps(sw);
    for (c = 0; c < 3; c++)
        _mm256_storeu_ps(out[c] + row + x, _mm256_mul_ps(sum8[c], sw));
    _mm256_zeroupper();
}

/* Horizontal patch distances are symmetric: the right-neighbour SAD of one
   pixel is the left-neighbour SAD of the next. Consecutive active octets
   carry the last right-edge value in a scalar, and the other seven left
   values are a lane shift of the current right-edge vector. */
#if defined(_MSC_VER)
#define JXL_EPF_NOINLINE __declspec(noinline)
#else
#define JXL_EPF_NOINLINE __attribute__((noinline))
#endif

static JXL_EPF_NOINLINE float epf_hsad_one(
    float *in[3], size_t row, uint32_t x, size_t stride,
    const float cscale[3]) {
    float dist = 0.0f;
    int c;
    for (c = 0; c < 3; c++) {
        const float *p = in[c] + row + x;
        float acc = fabsf(p[-(ptrdiff_t)stride + 1] -
                          p[-(ptrdiff_t)stride]);
        acc += fabsf(p[1] - p[0]);
        acc += fabsf(p[(ptrdiff_t)stride + 1] -
                     p[(ptrdiff_t)stride]);
        acc += fabsf(p[0] - p[-1]);
        acc += fabsf(p[2] - p[1]);
        dist += cscale[c] * acc;
    }
    return dist;
}
#undef JXL_EPF_NOINLINE

JXL_TARGET_AVX2_FMA
static JXL_INLINE_HINT float epf_row8_pass1(
    float *in[3], float *out[3], size_t row, uint32_t x, size_t stride,
    const float cscale[3], float sigma_val, __m256 smv, __m256 absmask,
    __m256 zero, __m256 one, float prev_hsad, float *prev_vsad,
    int reuse_vtop) {
    __m256 dist0 = reuse_vtop ? _mm256_loadu_ps(prev_vsad + x) : zero;
    __m256 dist1 = zero, dist2, dist3 = zero;
    __m256 sum0, sum1, sum2, sw, nis, wgt;
    int c;

    nis = _mm256_mul_ps(_mm256_set1_ps(sigma_val), smv);

    if (!reuse_vtop) {
        for (c = 0; c < 3; c++) {
            const float *p = in[c] + row + x;
            const __m256 p20 =
                _mm256_loadu_ps(p - 2 * (ptrdiff_t)stride);
            const __m256 p21 =
                _mm256_loadu_ps(p - (ptrdiff_t)stride);
            const __m256 p11 =
                _mm256_loadu_ps(p - (ptrdiff_t)stride - 1);
            const __m256 p31 =
                _mm256_loadu_ps(p - (ptrdiff_t)stride + 1);
            const __m256 p12 = _mm256_loadu_ps(p - 1);
            const __m256 p22 = _mm256_loadu_ps(p);
            const __m256 p32 = _mm256_loadu_ps(p + 1);
            const __m256 p23 =
                _mm256_loadu_ps(p + (ptrdiff_t)stride);
            const __m256 cs = _mm256_set1_ps(cscale[c]);
            __m256 acc0;

            acc0 = _mm256_add_ps(zero,
                _mm256_and_ps(absmask, _mm256_sub_ps(p20, p21)));
            acc0 = _mm256_add_ps(acc0,
                _mm256_and_ps(absmask, _mm256_sub_ps(p21, p22)));
            acc0 = _mm256_add_ps(acc0,
                _mm256_and_ps(absmask, _mm256_sub_ps(p22, p23)));
            acc0 = _mm256_add_ps(acc0,
                _mm256_and_ps(absmask, _mm256_sub_ps(p11, p12)));
            acc0 = _mm256_add_ps(acc0,
                _mm256_and_ps(absmask, _mm256_sub_ps(p31, p32)));
            dist0 = _mm256_fmadd_ps(cs, acc0, dist0);
        }
    }

    for (c = 0; c < 3; c++) {
        const float *p = in[c] + row + x;
        const __m256 p21 = _mm256_loadu_ps(p - (ptrdiff_t)stride);
        const __m256 p22 = _mm256_loadu_ps(p);
        const __m256 p23 = _mm256_loadu_ps(p + (ptrdiff_t)stride);
        const __m256 cs = _mm256_set1_ps(cscale[c]);
        __m256 acc1, acc3;

        acc1 = _mm256_add_ps(zero, _mm256_and_ps(
            absmask, _mm256_sub_ps(p21, p22)));
        acc1 = _mm256_add_ps(acc1, _mm256_and_ps(
            absmask, _mm256_sub_ps(p22, p23)));
        acc1 = _mm256_add_ps(acc1,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + 2 * (ptrdiff_t)stride), p23)));
        acc1 = _mm256_add_ps(acc1,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + (ptrdiff_t)stride - 1),
                _mm256_loadu_ps(p - 1))));
        acc1 = _mm256_add_ps(acc1,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + (ptrdiff_t)stride + 1),
                _mm256_loadu_ps(p + 1))));

        acc3 = _mm256_add_ps(zero,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p - (ptrdiff_t)stride + 1), p21)));
        acc3 = _mm256_add_ps(acc3,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + 1), p22)));
        acc3 = _mm256_add_ps(acc3,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + (ptrdiff_t)stride + 1), p23)));
        acc3 = _mm256_add_ps(acc3,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p - 1), p22)));
        acc3 = _mm256_add_ps(acc3,
            _mm256_and_ps(absmask, _mm256_sub_ps(
                _mm256_loadu_ps(p + 2), _mm256_loadu_ps(p + 1))));

        dist1 = _mm256_fmadd_ps(cs, acc1, dist1);
        dist3 = _mm256_fmadd_ps(cs, acc3, dist3);
    }
    if (prev_vsad) _mm256_storeu_ps(prev_vsad + x, dist1);

    {
        /* Shift the preceding right-SADs one lane toward higher x. A
           lane-cross plus lane-local align avoids vpermd's index-vector load
           and longer dependency latency. The first lane is replaced by the
           scalar carried from the preceding octet just below. */
        __m256i carry = _mm256_castps_si256(
            _mm256_permute2f128_ps(dist3, dist3, 0x08));
        dist2 = _mm256_castsi256_ps(_mm256_alignr_epi8(
            _mm256_castps_si256(dist3), carry, 12));
    }
    dist2 = _mm256_blend_ps(dist2, _mm256_set1_ps(prev_hsad), 0x01);

    sum0 = _mm256_loadu_ps(in[0] + row + x);
    sum1 = _mm256_loadu_ps(in[1] + row + x);
    sum2 = _mm256_loadu_ps(in[2] + row + x);
    sw = one;

    wgt = _mm256_fmadd_ps(dist0, nis, one);
    wgt = _mm256_max_ps(wgt, zero);
    sw = _mm256_add_ps(sw, wgt);
    sum0 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[0] + row + x - (ptrdiff_t)stride), sum0);
    sum1 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[1] + row + x - (ptrdiff_t)stride), sum1);
    sum2 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[2] + row + x - (ptrdiff_t)stride), sum2);

    wgt = _mm256_fmadd_ps(dist1, nis, one);
    wgt = _mm256_max_ps(wgt, zero);
    sw = _mm256_add_ps(sw, wgt);
    sum0 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[0] + row + x + stride), sum0);
    sum1 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[1] + row + x + stride), sum1);
    sum2 = _mm256_fmadd_ps(
        wgt, _mm256_loadu_ps(in[2] + row + x + stride), sum2);

    wgt = _mm256_fmadd_ps(dist2, nis, one);
    wgt = _mm256_max_ps(wgt, zero);
    sw = _mm256_add_ps(sw, wgt);
    sum0 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[0] + row + x - 1), sum0);
    sum1 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[1] + row + x - 1), sum1);
    sum2 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[2] + row + x - 1), sum2);

    wgt = _mm256_fmadd_ps(dist3, nis, one);
    wgt = _mm256_max_ps(wgt, zero);
    sw = _mm256_add_ps(sw, wgt);
    sum0 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[0] + row + x + 1), sum0);
    sum1 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[1] + row + x + 1), sum1);
    sum2 = _mm256_fmadd_ps(wgt, _mm256_loadu_ps(in[2] + row + x + 1), sum2);

    sw = _mm256_rcp_ps(sw);
    _mm256_storeu_ps(out[0] + row + x, _mm256_mul_ps(sum0, sw));
    _mm256_storeu_ps(out[1] + row + x, _mm256_mul_ps(sum1, sw));
    _mm256_storeu_ps(out[2] + row + x, _mm256_mul_ps(sum2, sw));
    {
        __m128 hi = _mm256_extractf128_ps(dist3, 1);
        hi = _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(3, 3, 3, 3));
        return _mm_cvtss_f32(hi);
    }
}

/* Keep the AVX2 target boundary outside the hot x loop. Besides avoiding one
   call and one AVX/SSE handoff per octet, this lets the compiler keep the
   channel scales and fixed border pattern live for a whole interior row. */
JXL_TARGET_AVX2_FMA
static uint32_t epf_row_pass1_avx2(
    float *in[3], float *out[3], size_t row, uint32_t x, uint32_t w,
    size_t stride, const float *sigma_row, const float cscale[3],
    float step_mul, float border_mul, int is_y_border, float *prev_vsad,
    const float *prev_sigma_row, int can_reuse_vtop, const uint8_t *copy_row) {
    const __m256 absmask =
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 smv = is_y_border
        ? _mm256_set1_ps(border_mul)
        : _mm256_setr_ps(border_mul, step_mul, step_mul, step_mul,
                         step_mul, step_mul, step_mul, border_mul);
    const int have_vtop = can_reuse_vtop && prev_vsad != NULL;
    float prev_hsad = 0.0f;
    int prev_valid = 0;
    for (; x + 9 < w; x += 8) {
        float sigma_val = sigma_row[x / 8];
        if (sigma_val == 0.0f) {
            int c;
            if (copy_row[x / 8]) {
                for (c = 0; c < 3; c++) {
                    _mm256_storeu_ps(out[c] + row + x,
                                     _mm256_loadu_ps(in[c] + row + x));
                }
            }
            prev_valid = 0;
        } else {
            int reuse_vtop =
                have_vtop && prev_sigma_row[x / 8] != 0.0f;
            if (!prev_valid)
                prev_hsad = epf_hsad_one(in, row, x - 1, stride, cscale);
            prev_hsad = epf_row8_pass1(
                in, out, row, x, stride, cscale, sigma_val, smv, absmask,
                zero, one, prev_hsad, prev_vsad, reuse_vtop);
            prev_valid = 1;
        }
    }
    return x;
}

/* Pass 0 measures twelve patch distances per sample, but they are six
   mirrored pairs: the distance from p to p+k is the distance from p+k to p,
   term for term and in the same order, so the float sums are bit-identical.
   Only the six taps that point right or down are computed, one row at a
   time; the other six are read back from the rows above and from the same
   row shifted. Each buffer is indexed by the x of the patch the tap starts
   from.
 *
 * Distances are computed in octets that start two samples left of a sigma
 * block, [8j-2, 8j+6): block j then reads octets j and j+1 of its own row and
 * of the two rows above, and nothing else. An octet is skipped when no block
 * that could read it has a nonzero sigma, which is what keeps a mostly-flat
 * page as cheap as it was -- a skipped octet leaves stale values behind that
 * no active block looks at. */
enum { EPF_F_H1, EPF_F_H2, EPF_F_VA, EPF_F_VB, EPF_F_VC, EPF_F_V2,
       EPF_F_COUNT };
#define EPF_P0_CHUNK 32u   /* blocks weighted per batch of distances */

/* Octets j0..j1 inclusive of the row at `row`. sig_a and sig_b are the sigma
   rows of this row and of the row two below, which between them cover the
   three rows that read these distances. The octet past the last block is
   pulled left to end at x1, the last sample any tap needs. */
JXL_TARGET_AVX2
static void epf_pass0_fwd_row(float *in[3], size_t row, uint32_t j0,
                              uint32_t j1, uint32_t x1, size_t stride,
                              const float *sig_a, const float *sig_b,
                              const float cscale[3],
                              float *dst[EPF_F_COUNT]) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 cs0 = _mm256_set1_ps(cscale[0]);
    const __m256 cs1 = _mm256_set1_ps(cscale[1]);
    const __m256 cs2 = _mm256_set1_ps(cscale[2]);
    const ptrdiff_t s = (ptrdiff_t)stride;
    uint32_t j;

/* One tap's five-term sum for the channel at p, in epf_dist_0 order (up,
   right, centre, left, down), folded into its running distance. Written out
   rather than looped so the six distances and five centre vectors stay in
   registers: as loops over small arrays the compiler kept them in memory.
   The sums start from their first term instead of from zero, which changes
   no value. */
#define EPF_FWD_ABS(off, cen) \
    _mm256_and_ps(absmask, _mm256_sub_ps(_mm256_loadu_ps(p + (off)), cen))
#define EPF_FWD_TAP(dist, off, FOLD) do {                                  \
        __m256 acc = EPF_FWD_ABS((off) - s, cen0);                         \
        acc = _mm256_add_ps(acc, EPF_FWD_ABS((off) + 1, cen1));            \
        acc = _mm256_add_ps(acc, EPF_FWD_ABS((off), cen2));                \
        acc = _mm256_add_ps(acc, EPF_FWD_ABS((off) - 1, cen3));            \
        acc = _mm256_add_ps(acc, EPF_FWD_ABS((off) + s, cen4));            \
        acc = _mm256_mul_ps(cs, acc);                                      \
        dist = FOLD(dist, acc);                                            \
    } while (0)
#define EPF_FWD_FIRST(dist, acc) (acc)
#define EPF_FWD_NEXT(dist, acc) _mm256_add_ps(dist, acc)
/* The taps are (1,0) (2,0) (-1,1) (0,1) (1,1) (0,2), as EPF_F_*. */
#define EPF_FWD_CHANNEL(c, scale, FOLD) do {                               \
        const float *p = in[c] + row + q;                                  \
        const __m256 cs = (scale);                                         \
        const __m256 cen0 = _mm256_loadu_ps(p - s);                        \
        const __m256 cen1 = _mm256_loadu_ps(p + 1);                        \
        const __m256 cen2 = _mm256_loadu_ps(p);                            \
        const __m256 cen3 = _mm256_loadu_ps(p - 1);                        \
        const __m256 cen4 = _mm256_loadu_ps(p + s);                        \
        EPF_FWD_TAP(d0, 1, FOLD);                                          \
        EPF_FWD_TAP(d1, 2, FOLD);                                          \
        EPF_FWD_TAP(d2, s - 1, FOLD);                                      \
        EPF_FWD_TAP(d3, s, FOLD);                                          \
        EPF_FWD_TAP(d4, s + 1, FOLD);                                      \
        EPF_FWD_TAP(d5, 2 * s, FOLD);                                      \
    } while (0)

    for (j = j0; j <= j1; j++) {
        __m256 d0, d1, d2, d3, d4, d5;
        uint32_t q = j * 8 - 2;
        if (sig_a[j - 1] == 0.0f && sig_b[j - 1] == 0.0f &&
            (j * 8 >= x1 || (sig_a[j] == 0.0f && sig_b[j] == 0.0f)))
            continue;
        if (q + 7 > x1) q = x1 - 7;
        d0 = d1 = d2 = d3 = d4 = d5 = _mm256_setzero_ps();
        EPF_FWD_CHANNEL(0, cs0, EPF_FWD_FIRST);
        EPF_FWD_CHANNEL(1, cs1, EPF_FWD_NEXT);
        EPF_FWD_CHANNEL(2, cs2, EPF_FWD_NEXT);
        _mm256_storeu_ps(dst[EPF_F_H1] + q, d0);
        _mm256_storeu_ps(dst[EPF_F_H2] + q, d1);
        _mm256_storeu_ps(dst[EPF_F_VA] + q, d2);
        _mm256_storeu_ps(dst[EPF_F_VB] + q, d3);
        _mm256_storeu_ps(dst[EPF_F_VC] + q, d4);
        _mm256_storeu_ps(dst[EPF_F_V2] + q, d5);
    }
#undef EPF_FWD_CHANNEL
#undef EPF_FWD_NEXT
#undef EPF_FWD_FIRST
#undef EPF_FWD_TAP
#undef EPF_FWD_ABS
    _mm256_zeroupper();
}

/* The weighting half of epf_row8 for a run of interior octets, with the
   distances taken from the rows epf_pass0_fwd_row filled. dist[k] + dx[k] is
   the row and shift that holds kernel tap k. */
JXL_TARGET_AVX2
static void epf_row_pass0_avx2(float *in[3], float *out[3], size_t row,
                               uint32_t x, uint32_t x1,
                               const float *sigma_row,
                               const ptrdiff_t koff[12],
                               const float *const dist[12],
                               float step_mul, float border_mul,
                               int is_y_border, const uint8_t *copy_row) {
    static const int8_t dx[12] = {0, -1, 0, 1, -2, -1, 0, 0, 0, 0, 0, 0};
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 smv = is_y_border
        ? _mm256_set1_ps(border_mul)
        : _mm256_setr_ps(border_mul, step_mul, step_mul, step_mul,
                         step_mul, step_mul, step_mul, border_mul);
    for (; x < x1; x += 8) {
        const float *p0 = in[0] + row + x;
        const float *p1 = in[1] + row + x;
        const float *p2 = in[2] + row + x;
        float sigma_val = sigma_row[x / 8];
        __m256 sum0, sum1, sum2;
        if (sigma_val == 0.0f && !copy_row[x / 8]) continue;
        sum0 = _mm256_loadu_ps(p0);
        sum1 = _mm256_loadu_ps(p1);
        sum2 = _mm256_loadu_ps(p2);
        if (sigma_val != 0.0f) {
            __m256 nis = _mm256_mul_ps(_mm256_set1_ps(sigma_val), smv);
            __m256 sw = one;
            int k;
            for (k = 0; k < 12; k++) {
                __m256 wgt = _mm256_add_ps(one, _mm256_mul_ps(
                    _mm256_loadu_ps(dist[k] + (ptrdiff_t)x + dx[k]), nis));
                wgt = _mm256_max_ps(wgt, zero);
                sw = _mm256_add_ps(sw, wgt);
                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(wgt,
                    _mm256_loadu_ps(p0 + koff[k])));
                sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(wgt,
                    _mm256_loadu_ps(p1 + koff[k])));
                sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(wgt,
                    _mm256_loadu_ps(p2 + koff[k])));
            }
            sw = _mm256_rcp_ps(sw);
            sum0 = _mm256_mul_ps(sum0, sw);
            sum1 = _mm256_mul_ps(sum1, sw);
            sum2 = _mm256_mul_ps(sum2, sw);
        }
        _mm256_storeu_ps(out[0] + row + x, sum0);
        _mm256_storeu_ps(out[1] + row + x, sum1);
        _mm256_storeu_ps(out[2] + row + x, sum2);
    }
    _mm256_zeroupper();
}
/* Pass 2 for a run of interior octets: four taps, each compared with the
   centre sample alone. epf_row8 does the same arithmetic in the same order;
   what this saves is entering it once per octet, with its tap and distance
   counts as loop bounds, for the cheapest pass of the three. Adding the
   first term to a zero accumulator is left out, which changes no value. */
JXL_TARGET_AVX2
static uint32_t epf_row_pass2_avx2(float *in[3], float *out[3], size_t row,
                                   uint32_t x, uint32_t w, size_t stride,
                                   const float *sigma_row,
                                   const float cscale[3], float step_mul,
                                   float border_mul, int is_y_border,
                                   const uint8_t *copy_row) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 cs0 = _mm256_set1_ps(cscale[0]);
    const __m256 cs1 = _mm256_set1_ps(cscale[1]);
    const __m256 cs2 = _mm256_set1_ps(cscale[2]);
    const __m256 smv = is_y_border
        ? _mm256_set1_ps(border_mul)
        : _mm256_setr_ps(border_mul, step_mul, step_mul, step_mul,
                         step_mul, step_mul, step_mul, border_mul);
    /* epf_kernel_1 order. */
    const ptrdiff_t koff[4] = {-(ptrdiff_t)stride, (ptrdiff_t)stride, -1, 1};
    for (; x + 8 < w; x += 8) {
        const float *p0 = in[0] + row + x;
        const float *p1 = in[1] + row + x;
        const float *p2 = in[2] + row + x;
        float sigma_val = sigma_row[x / 8];
        __m256 c0, c1, c2, sum0, sum1, sum2, sw, nis;
        int k;
        if (sigma_val == 0.0f && !copy_row[x / 8]) continue;
        c0 = _mm256_loadu_ps(p0);
        c1 = _mm256_loadu_ps(p1);
        c2 = _mm256_loadu_ps(p2);
        sum0 = c0; sum1 = c1; sum2 = c2;
        if (sigma_val != 0.0f) {
            nis = _mm256_mul_ps(_mm256_set1_ps(sigma_val), smv);
            sw = one;
            for (k = 0; k < 4; k++) {
                __m256 t0 = _mm256_loadu_ps(p0 + koff[k]);
                __m256 t1 = _mm256_loadu_ps(p1 + koff[k]);
                __m256 t2 = _mm256_loadu_ps(p2 + koff[k]);
                __m256 dist, wgt;
                dist = _mm256_mul_ps(cs0, _mm256_and_ps(absmask,
                                                        _mm256_sub_ps(t0, c0)));
                dist = _mm256_add_ps(dist, _mm256_mul_ps(cs1,
                    _mm256_and_ps(absmask, _mm256_sub_ps(t1, c1))));
                dist = _mm256_add_ps(dist, _mm256_mul_ps(cs2,
                    _mm256_and_ps(absmask, _mm256_sub_ps(t2, c2))));
                wgt = _mm256_add_ps(one, _mm256_mul_ps(dist, nis));
                wgt = _mm256_max_ps(wgt, zero);
                sw = _mm256_add_ps(sw, wgt);
                sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(wgt, t0));
                sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(wgt, t1));
                sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(wgt, t2));
            }
            sw = _mm256_rcp_ps(sw);
            sum0 = _mm256_mul_ps(sum0, sw);
            sum1 = _mm256_mul_ps(sum1, sw);
            sum2 = _mm256_mul_ps(sum2, sw);
        }
        _mm256_storeu_ps(out[0] + row + x, sum0);
        _mm256_storeu_ps(out[1] + row + x, sum1);
        _mm256_storeu_ps(out[2] + row + x, sum2);
    }
    _mm256_zeroupper();
    return x;
}
#endif

/* A sample whose whole footprint is inside the image needs no mirroring, so
   every neighbour it reads is a fixed sample offset from its centre. Those
   offsets depend only on the pass, so they are tabulated once per pass. */
static int epf_pass(float *in[3], float *out[3], uint32_t w, uint32_t h,
                    size_t stride, const float *sigma, uint32_t sigma_stride,
                    const jxl_epf *epf, int step, float *vsad_cache,
                    float *fwd_cache, const uint8_t *copy_map,
                    size_t copy_stride, uint32_t y_begin, uint32_t y_end) {
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
    const int use_avx2_fma = jxl_has_avx2_fma();
    /* Pass 0 octets run from x = 8 while x + 7 + pad < w. */
    const uint32_t p0_x1 = (step == 0 && use_avx2 && fwd_cache && w > 18)
        ? 8 + 8 * ((w - 19) / 8 + 1) : 0;
    const size_t fwd_stride = stride + 16;
#else
    (void)vsad_cache;
    (void)fwd_cache;
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

    for (y = y_begin; y < y_end; y++) {
        int is_y_border = ((y + 1) & 6u) == 0;
        /* Rows this close to an edge mirror; they are O(pad) of the image. */
        int y_inside = (y >= (uint32_t)pad && y + (uint32_t)pad < h);
        const float *sigma_row = sigma + (size_t)(y / 8) * sigma_stride;
        /* Nonzero where a block with sigma 0 still has to reach `out`. */
        const uint8_t *copy_row = copy_map + (size_t)(y / 8) * copy_stride;
#ifdef JXL_EPF_SSE2
        const float *prev_sigma_row = y
            ? sigma + (size_t)((y - 1) / 8) * sigma_stride
            : sigma_row;
#endif
        size_t row = (size_t)y * stride;
#ifdef JXL_EPF_SSE2
        const float *p0_dist[12];
        float *p0_f[EPF_F_COUNT];
        if (p0_x1 && y_inside) {
            /* Buffers: H1, H2, then two generations of VA/VB/VC by row
               parity, then three of V2 by row mod 3. The first interior row
               has no rows above it to inherit from, so those are filled
               whole; its own distances are computed a chunk at a time just
               ahead of the weighting, while the samples are in L1. */
            float **f = p0_f;
            uint32_t r;
            for (r = (y == (uint32_t)pad) ? y - 2 : y; r <= y; r++) {
                f[EPF_F_H1] = fwd_cache;
                f[EPF_F_H2] = fwd_cache + fwd_stride;
                f[EPF_F_VA] = fwd_cache + (2 + (r & 1u)) * fwd_stride;
                f[EPF_F_VB] = fwd_cache + (4 + (r & 1u)) * fwd_stride;
                f[EPF_F_VC] = fwd_cache + (6 + (r & 1u)) * fwd_stride;
                f[EPF_F_V2] = fwd_cache + (8 + r % 3u) * fwd_stride;
                if (r < y) {
                    epf_pass0_fwd_row(
                        in, (size_t)r * stride, 1, p0_x1 / 8, p0_x1, stride,
                        sigma + (size_t)(r / 8) * sigma_stride,
                        sigma + (size_t)((r + 2) / 8) * sigma_stride,
                        cscale, f);
                }
            }
            /* epf_kernel_2 order. */
            p0_dist[0] = fwd_cache + (8 + (y - 2) % 3u) * fwd_stride;
            p0_dist[1] = fwd_cache + (6 + ((y - 1) & 1u)) * fwd_stride;
            p0_dist[2] = fwd_cache + (4 + ((y - 1) & 1u)) * fwd_stride;
            p0_dist[3] = fwd_cache + (2 + ((y - 1) & 1u)) * fwd_stride;
            p0_dist[4] = f[EPF_F_H2];
            p0_dist[5] = f[EPF_F_H1];
            p0_dist[6] = f[EPF_F_H1];
            p0_dist[7] = f[EPF_F_H2];
            p0_dist[8] = f[EPF_F_VA];
            p0_dist[9] = f[EPF_F_VB];
            p0_dist[10] = f[EPF_F_VC];
            p0_dist[11] = f[EPF_F_V2];
        }
#endif
        for (x = 0; x < w; ) {
            float sigma_val = sigma_row[x / 8];

#ifdef JXL_EPF_SSE2
            if (p0_x1 && y_inside && x == 8) {
                while (x < p0_x1) {
                    uint32_t xe = JXL_MIN(x + 8 * EPF_P0_CHUNK, p0_x1);
                    /* A chunk's first octet was the previous chunk's last. */
                    epf_pass0_fwd_row(
                        in, row, x == 8 ? 1 : x / 8 + 1, xe / 8, p0_x1,
                        stride, sigma_row,
                        sigma + (size_t)((y + 2) / 8) * sigma_stride,
                        cscale, p0_f);
                    epf_row_pass0_avx2(in, out, row, x, xe, sigma_row, koff,
                                       p0_dist, step_mul, border_mul,
                                       is_y_border, copy_row);
                    x = xe;
                }
                continue;
            }
            if (use_avx2 && step == 2 && y_inside && x == 8 && w > 16) {
                x = epf_row_pass2_avx2(in, out, row, x, w, stride, sigma_row,
                                       cscale, step_mul, border_mul,
                                       is_y_border, copy_row);
                continue;
            }
            if (use_avx2_fma && step == 1 && y_inside && (x & 7u) == 0 &&
                x >= 2 && x + 9 < w) {
                x = epf_row_pass1_avx2(in, out, row, x, w, stride, sigma_row,
                                       cscale, step_mul, border_mul,
                                       is_y_border, vsad_cache,
                                       prev_sigma_row,
                                       y > (uint32_t)pad, copy_row);
                continue;
            }
            /* Four samples at a time down the row. Vectorising across x (not
               across taps) means every lane runs the same operations in the
               same order as the scalar path below, so the output is
               bit-identical -- no tolerance risk, and the scalar path stays
               the reference. Requires the whole quad to be in the mirror-free
               interior and inside one sigma block: blocks are 8 wide and the
               quad is 4-aligned, so x/8 is constant across it. */
            if (use_avx2 && y_inside && (x & 7u) == 0 &&
                x >= (uint32_t)pad && x + 7 + (uint32_t)pad < w) {
                if (sigma_val == 0.0f) {
                    if (copy_row[x / 8]) {
                        for (c = 0; c < 3; c++) {
                            memcpy(out[c] + row + x, in[c] + row + x,
                                   8 * sizeof(float));
                        }
                    }
                    x += 8;
                    continue;
                }
                epf_row8(in, out, row, x, koff, doff, nkernel, ndist,
                         cscale, sigma_val, step_mul, border_mul,
                         is_y_border);
                x += 8;
                continue;
            }
            if (y_inside && (x & 3u) == 0 &&
                x >= (uint32_t)pad && x + 3 + (uint32_t)pad < w) {
                __m128 dist4[12], sum4[3], sw, nis;
                if (sigma_val == 0.0f) {
                    if (copy_row[x / 8]) {
                        for (c = 0; c < 3; c++) {
                            _mm_storeu_ps(out[c] + row + x,
                                          _mm_loadu_ps(in[c] + row + x));
                        }
                    }
                    x += 4;
                    continue;
                }
                /* sm is per-lane: within an 8-wide block only lanes 0 and 7
                   take border_mul, and a 4-aligned quad covers either 0..3 or
                   4..7, so there are just two patterns. */
                nis = _mm_mul_ps(_mm_set1_ps(sigma_val),
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

            if (sigma_val == 0.0f) {
                if (copy_row[x / 8]) {
                    for (c = 0; c < 3; c++) out[c][row + x] = in[c][row + x];
                }
                x++;
                continue;
            }
            if (is_y_border || (x & 7u) == 0 || (x & 7u) == 7) sm = border_mul;
            else sm = step_mul;
            neg_inv_sigma = sigma_val * sm;

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

/* Copies the filtered blocks of row y of `src` over `dst`. */
static void epf_copy_active_row(float *dst[3], float *src[3], uint32_t w,
                                uint32_t y, size_t stride,
                                const float *sigma, uint32_t sigma_stride) {
    const float *sigma_row = sigma + (size_t)(y / 8) * sigma_stride;
    uint32_t bw = (w + 7) / 8, x;
    size_t row = (size_t)y * stride;
    int c;
    for (x = 0; x < bw; x++) {
        uint32_t n;
        if (sigma_row[x] == 0.0f) continue;
        n = JXL_MIN(8u, w - x * 8);
        for (c = 0; c < 3; c++) {
            memcpy(dst[c] + row + x * 8, src[c] + row + x * 8,
                   n * sizeof(float));
        }
    }
}

/* A few rows of three planes that stand in for a full-size image. The passes
   address a sample as plane + y * stride + x, so `p` is the buffer biased
   back by the row its first slot currently holds: rows base .. base + rows - 1
   are real, and writing the row past them slides the newest `keep` rows to
   the front. The bias is applied as an integer because the biased pointer
   itself lies outside the allocation. */
#define EPF_RING_ROWS 48u
typedef struct {
    float *buf[3];
    float *p[3];
    uint32_t base, keep;
} epf_ring;

static void epf_ring_bias(epf_ring *r, size_t stride) {
    int c;
    for (c = 0; c < 3; c++) {
        r->p[c] = (float *)((uintptr_t)r->buf[c] -
                            (uintptr_t)r->base * stride * sizeof(float));
    }
}

static int epf_ring_init(jxl_ctx *ctx, epf_ring *r, size_t stride,
                         uint32_t keep) {
    size_t n;
    int c;
    /* One spare row: the kernels load a vector that may start in the last
       columns of a row and run on into the next. */
    if (!jxl_size_mul(stride, (EPF_RING_ROWS + 1) * sizeof(float), &n))
        return -1;
    for (c = 0; c < 3; c++) {
        r->buf[c] = (float *)jxl_malloc(ctx, n);
        if (!r->buf[c]) return -1;
    }
    r->base = 0;
    r->keep = keep;
    epf_ring_bias(r, stride);
    return 0;
}

/* Makes row y writable. Rows are written in order. */
static void epf_ring_advance(epf_ring *r, uint32_t y, size_t stride) {
    int c;
    if (y - r->base < EPF_RING_ROWS) return;
    for (c = 0; c < 3; c++) {
        memmove(r->buf[c],
                r->buf[c] + (size_t)(EPF_RING_ROWS - r->keep) * stride,
                (size_t)r->keep * stride * sizeof(float));
    }
    r->base = y - r->keep;
    epf_ring_bias(r, stride);
}

/* The passes used to run one after another over three full-size scratch
   planes. Nothing about the filter needs that: a pass reads at most three
   rows either side of the row it writes, so the passes can follow each other
   down the image a few rows apart. The plane is then read once and written
   once while its rows are still in cache, and the intermediates live in two
   small row rings instead of in 12 bytes per sample of freshly faulted-in
   memory -- on a 12-megapixel page the page faults alone cost more than the
   arithmetic.
 *
 * Where sigma is 0 a sample passes through every pass unchanged, so only
 * filtered blocks are ever written back. With three passes:
 *
 *   pass 0   plane  -> ring a   (plus the unfiltered rim pass 1 reads)
 *   pass 1   ring a -> plane    3 rows behind pass 0
 *   pass 2   plane  -> ring b   2 rows behind pass 1
 *   copy     ring b -> plane    1 row behind pass 2
 *
 * Each lag is the larger of what the pass ahead still reads from the buffer
 * being overwritten (its pad) and how far ahead the pass behind reads. */
int jxl_apply_epf(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                  size_t stride, const float *sigma, uint32_t sigma_stride,
                  const jxl_epf *epf) {
    static const uint32_t step_pad[3] = {3, 2, 1};
    epf_ring ring_a, ring_b;
    float *vsad_cache = NULL;
    float *fwd_cache = NULL;
    uint8_t *copy_map = NULL;
    const uint32_t bw = (w + 7) / 8, bh = (h + 7) / 8;
    int steps[3], nsteps = 0, i;
    uint32_t lag[4], t, t_end;
    int c, rc = -1;

    memset(&ring_a, 0, sizeof(ring_a));
    memset(&ring_b, 0, sizeof(ring_b));
    if (!epf->enabled || w == 0 || h == 0) return 0;

    if (epf->iters == 3) steps[nsteps++] = 0;
    steps[nsteps++] = 1;
    if (epf->iters >= 2) steps[nsteps++] = 2;

    /* Pass i reads the plane when i is even and ring a when it is odd; an
       odd pass count leaves the last output in ring b. `keep` is how far
       behind the newest row the reader of a ring still looks. */
    if (nsteps > 1 && epf_ring_init(ctx, &ring_a, stride, 6) != 0) goto done;
    if ((nsteps & 1) && epf_ring_init(ctx, &ring_b, stride, 3) != 0) goto done;

    /* Row 0 of the map is all zeros, for a pass whose unfiltered blocks are
       already where they belong; the rest marks, per block, an unfiltered
       block that touches a filtered one, which is all pass 1 or 2 can reach
       from a filtered sample. */
    {
        size_t n;
        uint32_t x, y;
        if (!jxl_size_mul(bw, (size_t)bh + 1, &n)) goto done;
        copy_map = (uint8_t *)jxl_calloc(ctx, n, 1);
        if (!copy_map) goto done;
        for (y = 0; y < bh; y++) {
            uint8_t *halo = copy_map + (size_t)(y + 1) * bw;
            uint32_t y0 = y ? y - 1 : 0, y1 = JXL_MIN(y + 1, bh - 1);
            for (x = 0; x < bw; x++) {
                uint32_t x0 = x ? x - 1 : 0, x1 = JXL_MIN(x + 1, bw - 1);
                uint32_t xx, yy;
                for (yy = y0; yy <= y1 && !halo[x]; yy++) {
                    const float *srow = sigma + (size_t)yy * sigma_stride;
                    for (xx = x0; xx <= x1; xx++) {
                        if (srow[xx] != 0.0f) { halo[x] = 1; break; }
                    }
                }
            }
        }
    }
#ifdef JXL_EPF_SSE2
    if (jxl_has_avx2()) {
        size_t n;
        if (jxl_size_mul(stride, sizeof(float), &n))
            vsad_cache = (float *)jxl_malloc(ctx, n);
        /* This row is only a speed cache; allocation failure uses the
           register-only pass-1 path. */
        if (epf->iters == 3 &&
            jxl_size_mul(stride + 16, 11 * sizeof(float), &n))
            fwd_cache = (float *)jxl_malloc(ctx, n);
        /* Likewise: without it pass 0 computes all twelve distances. */
    }
#endif

    lag[0] = 0;
    for (i = 1; i < nsteps; i++) {
        lag[i] = lag[i - 1] +
                 JXL_MAX(step_pad[steps[i - 1]], step_pad[steps[i]]);
    }
    lag[nsteps] = lag[nsteps - 1] + step_pad[steps[nsteps - 1]];
    t_end = h + lag[nsteps];

    for (t = 0; t < t_end; t++) {
        for (i = 0; i < nsteps; i++) {
            uint32_t y = t - lag[i];
            int to_plane = (i & 1) != 0;
            int last = i + 1 == nsteps;
            epf_ring *dst = last ? &ring_b : &ring_a;
            if (t < lag[i] || y >= h) continue;
            if (!to_plane) epf_ring_advance(dst, y, stride);
            /* Only a ring that a later pass reads needs the rim. */
            if (epf_pass(to_plane ? ring_a.p : plane,
                         to_plane ? plane : dst->p, w, h, stride, sigma,
                         sigma_stride, epf, steps[i], vsad_cache, fwd_cache,
                         (!to_plane && !last) ? copy_map + bw : copy_map,
                         (!to_plane && !last) ? bw : 0, y, y + 1) != 0)
                goto done;
        }
        if ((nsteps & 1) && t >= lag[nsteps] && t - lag[nsteps] < h) {
            epf_copy_active_row(plane, ring_b.p, w, t - lag[nsteps], stride,
                                sigma, sigma_stride);
        }
    }
    rc = 0;

done:
    jxl_free(ctx, vsad_cache);
    jxl_free(ctx, fwd_cache);
    jxl_free(ctx, copy_map);
    for (c = 0; c < 3; c++) {
        jxl_free(ctx, ring_a.buf[c]);
        jxl_free(ctx, ring_b.buf[c]);
    }
    return rc;
}
