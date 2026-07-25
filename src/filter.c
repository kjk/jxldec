/* filter.c -- the two in-loop restoration filters.
 *
 * Gaborish is a fixed 3x3 kernel with per-channel weights, run once after the
 * inverse transform. The edge-preserving filter (EPF) is a self-similarity
 * filter run in up to three passes with progressively smaller kernels; its
 * strength comes from the per-block sigma map carried by the varblock
 * metadata.
 *
 * Gaborish clamps at the borders, EPF mirrors (without repeating the edge
 * sample) -- matching libjxl in both cases.
 */
#include "jxl_internal.h"

#include <math.h>
#include <stddef.h>

static uint32_t jxl_mirror(int64_t offset, uint32_t len) {
    for (;;) {
        if (offset < 0) offset = -(offset + 1);
        else if ((uint64_t)offset >= len) offset = (int64_t)len * 2 - 1 - offset;
        else return (uint32_t)offset;
    }
}

static float sample_clamped(const float *p, size_t stride, uint32_t w,
                            uint32_t h, int64_t x, int64_t y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if ((uint64_t)x >= w) x = (int64_t)w - 1;
    if ((uint64_t)y >= h) y = (int64_t)h - 1;
    return p[(size_t)y * stride + (size_t)x];
}

int jxl_apply_gabor(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                    size_t stride, const float weights[3][2]) {
    float *tmp;
    int c;
    if (w == 0 || h == 0) return 0;
    tmp = (float *)jxl_calloc(ctx, (size_t)w * h, sizeof(float));
    if (!tmp) return -1;

    for (c = 0; c < 3; c++) {
        float w0 = weights[c][0], w1 = weights[c][1];
        float gw = 1.0f / (1.0f + w0 * 4.0f + w1 * 4.0f);
        uint32_t x, y;
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                float cc = plane[c][(size_t)y * stride + x];
                float n = sample_clamped(plane[c], stride, w, h, x, (int64_t)y - 1);
                float s = sample_clamped(plane[c], stride, w, h, x, (int64_t)y + 1);
                float we = sample_clamped(plane[c], stride, w, h, (int64_t)x - 1, y);
                float e = sample_clamped(plane[c], stride, w, h, (int64_t)x + 1, y);
                float nw = sample_clamped(plane[c], stride, w, h, (int64_t)x - 1, (int64_t)y - 1);
                float ne = sample_clamped(plane[c], stride, w, h, (int64_t)x + 1, (int64_t)y - 1);
                float sw = sample_clamped(plane[c], stride, w, h, (int64_t)x - 1, (int64_t)y + 1);
                float se = sample_clamped(plane[c], stride, w, h, (int64_t)x + 1, (int64_t)y + 1);
                tmp[(size_t)y * w + x] =
                    (cc + (n + s + we + e) * w0 + (nw + ne + sw + se) * w1) * gw;
            }
        }
        for (y = 0; y < h; y++) {
            memcpy(plane[c] + (size_t)y * stride, tmp + (size_t)y * w,
                   (size_t)w * sizeof(float));
        }
    }
    jxl_free(ctx, tmp);
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
        for (x = 0; x < w; x++) {
            float sigma_val = sigma_row[x / 8];
            float dist[12];   /* SAD to each kernel tap, all channels folded in */
            size_t soff[12];  /* kernel tap -> absolute sample index */
            float sum[3];
            float sum_weights, inv_w, sm, neg_inv_sigma;

            if (sigma_val < 0.3f) {
                for (c = 0; c < 3; c++) out[c][row + x] = in[c][row + x];
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
