/* noise.c -- synthetic photon/film noise.
 *
 * The encoder can strip noise from an image and record only its intensity
 * curve; the decoder regenerates it deterministically. Each group seeds an
 * XorShift128+ generator from its top-left corner (and the frame counters),
 * producing white noise in [1, 2); a 5x5 high-pass convolution turns that into
 * zero-mean blue-ish noise, which is then scaled per pixel by a lookup into
 * the 8-entry intensity curve.
 *
 * We generate the whole frame at once, so neighbouring groups are simply
 * neighbouring pixels; only the frame border needs the mirrored padding
 * libjxl uses.
 */
#include "jxl_internal.h"

#define NOISE_LANES 8

typedef struct {
    uint64_t s0[NOISE_LANES];
    uint64_t s1[NOISE_LANES];
} jxl_xorshift;

static uint64_t split_mix_64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void xorshift_init(jxl_xorshift *r, uint64_t seed0, uint64_t seed1) {
    int i;
    r->s0[0] = split_mix_64(seed0 + 0x9E3779B97F4A7C15ull);
    r->s1[0] = split_mix_64(seed1 + 0x9E3779B97F4A7C15ull);
    for (i = 1; i < NOISE_LANES; i++) {
        r->s0[i] = split_mix_64(r->s0[i - 1]);
        r->s1[i] = split_mix_64(r->s1[i - 1]);
    }
}

/* One batch is NOISE_LANES u64s, read as 2 * NOISE_LANES little-endian u32s. */
static void xorshift_batch(jxl_xorshift *r, uint32_t out[NOISE_LANES * 2]) {
    int i;
    for (i = 0; i < NOISE_LANES; i++) {
        uint64_t s1 = r->s0[i];
        uint64_t s0 = r->s1[i];
        uint64_t ret = s1 + s0;
        r->s0[i] = s0;
        s1 ^= s1 << 23;
        r->s1[i] = s1 ^ (s0 ^ (s1 >> 18) ^ (s0 >> 5));
        out[i * 2] = (uint32_t)ret;
        out[i * 2 + 1] = (uint32_t)(ret >> 32);
    }
}

static float bits_to_unit(uint32_t x) {
    uint32_t b = (x >> 9) | 0x3f800000u;   /* [1, 2) */
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

int jxl_noise_params_read(jxl_br *br, jxl_noise_params *np) {
    int i;
    for (i = 0; i < 8; i++) {
        np->lut[i] = (float)jxl_br_read(br, 10) / 1024.0f;
    }
    return br->err ? -1 : 0;
}

static uint32_t noise_mirror(int64_t v, uint32_t len) {
    for (;;) {
        if (v < 0) v = -(v + 1);
        else if ((uint64_t)v >= len) v = (int64_t)len * 2 - 1 - v;
        else return (uint32_t)v;
    }
}

int jxl_render_noise(jxl_ctx *ctx, jxl_fimage *img, const jxl_noise_params *np,
                     const jxl_frame_header *fh, uint32_t visible_frames,
                     uint32_t invisible_frames, float corr_x, float corr_b) {
    uint32_t width = fh->width, height = fh->height;
    uint32_t group_dim = jxl_frame_group_dim(fh);
    uint32_t groups_per_row, group_rows, gx, gy;
    float *raw[3] = {NULL, NULL, NULL};
    float *conv[3] = {NULL, NULL, NULL};
    uint64_t seed0 = ((uint64_t)visible_frames << 32) + invisible_frames;
    float lut[9];
    uint32_t x, y;
    int c, rc = -1;

    if (img->ncolor < 3 || width == 0 || height == 0) return 0;
    for (c = 0; c < 8; c++) lut[c] = np->lut[c];
    lut[8] = np->lut[7];

    for (c = 0; c < 3; c++) {
        raw[c] = (float *)jxl_calloc(ctx, (size_t)width * height, sizeof(float));
        conv[c] = (float *)jxl_calloc(ctx, (size_t)width * height, sizeof(float));
        if (!raw[c] || !conv[c]) goto done;
    }

    groups_per_row = (width + group_dim - 1) / group_dim;
    group_rows = (height + group_dim - 1) / group_dim;

    for (gy = 0; gy < group_rows; gy++) {
        for (gx = 0; gx < groups_per_row; gx++) {
            uint32_t x0 = gx * group_dim, y0 = gy * group_dim;
            uint32_t gw = JXL_MIN(group_dim, width - x0);
            uint32_t gh = JXL_MIN(group_dim, height - y0);
            uint32_t blocks = (gw + NOISE_LANES * 2 - 1) / (NOISE_LANES * 2);
            jxl_xorshift rng;
            xorshift_init(&rng, seed0, ((uint64_t)x0 << 32) + y0);
            /* The three channels are generated back to back, each covering
               the whole group before the next one starts. */
            for (c = 0; c < 3; c++) {
                uint32_t row, b;
                for (row = 0; row < gh; row++) {
                    for (b = 0; b < blocks; b++) {
                        uint32_t bits[NOISE_LANES * 2];
                        uint32_t k;
                        xorshift_batch(&rng, bits);
                        for (k = 0; k < NOISE_LANES * 2; k++) {
                            uint32_t px = x0 + b * NOISE_LANES * 2 + k;
                            if (px >= x0 + gw) continue;
                            raw[c][(size_t)(y0 + row) * width + px] =
                                bits_to_unit(bits[k]);
                        }
                    }
                }
            }
        }
    }

    /* 5x5 high-pass: 0.16 * sum minus 4 * center. */
    for (c = 0; c < 3; c++) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                float sum = 0.0f;
                int dy, dx;
                for (dy = -2; dy <= 2; dy++) {
                    uint32_t sy = noise_mirror((int64_t)y + dy, height);
                    const float *row = raw[c] + (size_t)sy * width;
                    for (dx = -2; dx <= 2; dx++) {
                        uint32_t sx = noise_mirror((int64_t)x + dx, width);
                        sum += row[sx] * 0.16f;
                    }
                }
                conv[c][(size_t)y * width + x] =
                    sum - raw[c][(size_t)y * width + x] * 4.0f;
            }
        }
    }

    for (y = 0; y < height && y < img->plane[0].h; y++) {
        float *rx = img->plane[0].data + (size_t)y * img->plane[0].stride;
        float *ry = img->plane[1].data + (size_t)y * img->plane[1].stride;
        float *rb = img->plane[2].data + (size_t)y * img->plane[2].stride;
        const float *nxr = conv[0] + (size_t)y * width;
        const float *nyr = conv[1] + (size_t)y * width;
        const float *nbr = conv[2] + (size_t)y * width;
        for (x = 0; x < width && x < img->plane[0].w; x++) {
            float gx_ = rx[x], gy_ = ry[x];
            float in_x = gx_ + gy_;
            float in_y = gy_ - gx_;
            float sx_ = in_x * 3.0f, sy_ = in_y * 3.0f;
            uint32_t ix, iy;
            float fx, fy, sxv, syv, nx, ny;
            if (sx_ < 0.0f) sx_ = 0.0f;
            if (sy_ < 0.0f) sy_ = 0.0f;
            ix = (uint32_t)sx_;
            iy = (uint32_t)sy_;
            if (ix > 7) ix = 7;
            if (iy > 7) iy = 7;
            fx = sx_ - (float)ix;
            fy = sy_ - (float)iy;
            sxv = (lut[ix + 1] - lut[ix]) * fx + lut[ix];
            syv = (lut[iy + 1] - lut[iy]) * fy + lut[iy];
            nx = 0.22f * sxv * (0.0078125f * nxr[x] + 0.9921875f * nbr[x]);
            ny = 0.22f * syv * (0.0078125f * nyr[x] + 0.9921875f * nbr[x]);
            rx[x] += corr_x * (nx + ny) + nx - ny;
            ry[x] += nx + ny;
            rb[x] += corr_b * (nx + ny);
        }
    }
    rc = 0;

done:
    for (c = 0; c < 3; c++) {
        jxl_free(ctx, raw[c]);
        jxl_free(ctx, conv[c]);
    }
    return rc;
}
