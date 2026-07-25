/* vardct.c -- the VarDCT sub-codec: transform-type tables, quantizer and
 * dequantization matrices, HF block contexts, coefficient orders, the HF
 * coefficient decoder, and the per-varblock dequant + inverse transform.
 *
 * A VarDCT frame stores a quantized 1/8-scale LF image (via Modular), per-LF-
 * group varblock metadata (which transform covers which 8x8 block, the HF
 * multiplier, chroma-from-luma factors, EPF sharpness), and per-group HF
 * coefficients. Rendering dequantizes the coefficients, folds the LF image
 * back in as the block's DC, undoes chroma-from-luma, and runs the inverse
 * transform for each varblock.
 */
#include "jxl_internal.h"

#include <math.h>

/* ===================================================================== */
/* transform type tables                                                  */
/* ===================================================================== */

/* {blocks wide, blocks high} per transform type. */
static const uint8_t tr_size[JXL_TR_COUNT][2] = {
    {1,1},{1,1},{1,1},{1,1},{2,2},{4,4},{1,2},{2,1},{1,4},{4,1},{2,4},{4,2},
    {1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{8,8},{4,8},{8,4},{16,16},{8,16},
    {16,8},{32,32},{16,32},{32,16}
};

static const uint8_t tr_matrix_idx[JXL_TR_COUNT] = {
    0,1,2,3,4,5,6,6,7,7,8,8,9,9,10,10,10,10,11,12,12,13,14,14,15,16,16
};

static const uint8_t tr_order[JXL_TR_COUNT] = {
    0,1,1,1,2,3,4,4,5,5,6,6,1,1,1,1,1,1,7,8,8,9,10,10,11,12,12
};

/* {width, height} of the dequantization matrix, in samples. */
static const uint16_t tr_matrix_dims[17][2] = {
    {8,8},{8,8},{8,8},{8,8},{16,16},{32,32},{16,8},{32,8},{32,16},{8,8},
    {8,8},{64,64},{64,32},{128,128},{128,64},{256,256},{256,128}
};

void jxl_tr_select_size(int tr, uint32_t *bw, uint32_t *bh) {
    *bw = tr_size[tr][0];
    *bh = tr_size[tr][1];
}

int jxl_tr_matrix_index(int tr) { return tr_matrix_idx[tr]; }
int jxl_tr_order_id(int tr) { return tr_order[tr]; }

void jxl_tr_matrix_size(int tr, uint32_t *w, uint32_t *h) {
    int idx = tr_matrix_idx[tr];
    *w = tr_matrix_dims[idx][0];
    *h = tr_matrix_dims[idx][1];
}

int jxl_tr_need_transpose(int tr) {
    switch (tr) {
        case JXL_TR_HORNUSS: case JXL_TR_DCT2: case JXL_TR_DCT4:
        case JXL_TR_DCT4X8: case JXL_TR_DCT8X4:
        case JXL_TR_AFV0: case JXL_TR_AFV1: case JXL_TR_AFV2: case JXL_TR_AFV3:
            return 0;
        default: return tr_size[tr][1] >= tr_size[tr][0];
    }
}

/* ===================================================================== */
/* natural coefficient order                                              */
/* ===================================================================== */

static const uint16_t order_block_sizes[13][2] = {
    {8,8},{8,8},{16,16},{32,32},{16,8},{32,8},{32,16},{64,64},{64,32},
    {128,128},{128,64},{256,256},{256,128}
};

/* Coefficients are visited LF-block first, then along anti-diagonals. The
   `x < lbw && y < lbw` test (lbw twice) mirrors libjxl exactly. */
static void fill_natural_order(uint32_t bw, uint32_t bh, uint16_t *out) {
    uint32_t y_scale = bw / bh;
    uint32_t lbw = bw / 8, lbh = bh / 8;
    uint32_t idx = 0, dist;

    while (idx < lbw * lbh) {
        uint32_t x = idx % lbw;
        uint32_t y = idx / lbw;
        out[idx * 2] = (uint16_t)x;
        out[idx * 2 + 1] = (uint16_t)y;
        idx++;
    }
    for (dist = 1; dist < 2 * bw; dist++) {
        uint32_t margin = dist > bw ? dist - bw : 0;
        uint32_t order;
        for (order = margin; order < dist - margin; order++) {
            uint32_t x, y;
            if (dist % 2 == 1) { x = order; y = dist - 1 - order; }
            else { x = dist - 1 - order; y = order; }
            if (x < lbw && y < lbw) continue;
            if (y % y_scale != 0) continue;
            out[idx * 2] = (uint16_t)x;
            out[idx * 2 + 1] = (uint16_t)(y / y_scale);
            idx++;
        }
    }
}

const uint16_t *jxl_natural_order(jxl_ctx *ctx, jxl_natural_orders *no,
                                  int order_id) {
    uint32_t bw, bh;
    if (order_id < 0 || order_id >= 13) return NULL;
    if (no->order[order_id]) return no->order[order_id];
    bw = order_block_sizes[order_id][0];
    bh = order_block_sizes[order_id][1];
    no->order[order_id] =
        (uint16_t *)jxl_calloc(ctx, (size_t)bw * bh * 2, sizeof(uint16_t));
    if (!no->order[order_id]) return NULL;
    fill_natural_order(bw, bh, no->order[order_id]);
    return no->order[order_id];
}

void jxl_natural_orders_free(jxl_ctx *ctx, jxl_natural_orders *no) {
    int i;
    for (i = 0; i < 13; i++) {
        jxl_free(ctx, no->order[i]);
        no->order[i] = NULL;
    }
}

/* ===================================================================== */
/* quantizer / channel correlation / block contexts                       */
/* ===================================================================== */

void jxl_quantizer_read(jxl_br *br, jxl_quantizer *q) {
    q->global_scale = jxl_br_u32(br, 1, 11, 2049, 11, 4097, 12, 8193, 16);
    q->quant_lf = jxl_br_u32(br, 16, 0, 1, 5, 1, 8, 1, 16);
}

void jxl_lf_chan_corr_read(jxl_br *br, jxl_lf_chan_corr *c, int xyb) {
    c->colour_factor = 84;
    c->base_correlation_x = 0.0f;
    /* The default Y-to-B correlation only makes sense for XYB; a YCbCr or
       plain-RGB frame has none (libjxl zeroes it in ColorCorrelationMap). */
    c->base_correlation_b = xyb ? 1.0f : 0.0f;
    c->x_factor_lf = 128;
    c->b_factor_lf = 128;
    if (jxl_br_bool(br)) return;
    c->colour_factor = jxl_br_u32(br, 84, 0, 256, 0, 2, 8, 258, 16);
    c->base_correlation_x = jxl_br_f16(br);
    c->base_correlation_b = jxl_br_f16(br);
    c->x_factor_lf = jxl_br_read(br, 8);
    c->b_factor_lf = jxl_br_read(br, 8);
}

static const uint8_t default_block_ctx_map[39] = {
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14,
    14, 14, 14, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14
};

int jxl_hf_block_ctx_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_block_ctx *bc) {
    uint32_t bsize = 1, i, c;

    memset(bc, 0, sizeof(*bc));
    if (jxl_br_bool(br)) {
        bc->num_block_clusters = 15;
        bc->block_ctx_map_len = 39;
        bc->block_ctx_map = (uint8_t *)jxl_malloc(ctx, 39);
        if (!bc->block_ctx_map) return -1;
        memcpy(bc->block_ctx_map, default_block_ctx_map, 39);
        return 0;
    }

    for (c = 0; c < 3; c++) {
        bc->nlf[c] = jxl_br_read(br, 4);
        if (bc->nlf[c] > 15) return -1;
        bsize *= bc->nlf[c] + 1;
        for (i = 0; i < bc->nlf[c]; i++) {
            uint32_t t = jxl_br_u32(br, 0, 4, 16, 8, 272, 16, 65808, 32);
            bc->lf_thresholds[c][i] = jxl_unpack_signed(t);
        }
    }
    bc->nqf = jxl_br_read(br, 4);
    if (bc->nqf > 15) return -1;
    bsize *= bc->nqf + 1;
    for (i = 0; i < bc->nqf; i++) {
        bc->qf_thresholds[i] = 1 + jxl_br_u32(br, 0, 2, 4, 3, 12, 5, 44, 8);
    }
    if (br->err || bsize > 64) {
        JXL_ERR(ctx, "vardct: bad block context sizes");
        return -1;
    }

    bc->block_ctx_map_len = bsize * 39;
    bc->block_ctx_map = (uint8_t *)jxl_calloc(ctx, bc->block_ctx_map_len, 1);
    if (!bc->block_ctx_map) return -1;
    if (jxl_read_clusters(ctx, br, bc->block_ctx_map_len, bc->block_ctx_map,
                          &bc->num_block_clusters) != 0)
        return -1;
    return 0;
}

void jxl_hf_block_ctx_free(jxl_ctx *ctx, jxl_hf_block_ctx *bc) {
    jxl_free(ctx, bc->block_ctx_map);
    bc->block_ctx_map = NULL;
}

/* ===================================================================== */
/* dequantization matrices                                                */
/* ===================================================================== */

static const float dq_seq_a[7] = {
    -1.025f, -0.78f, -0.65012f, -0.19041574f, -0.20819396f, -0.421064f,
    -0.32733846f
};
static const float dq_seq_b[7] = {
    -0.30419582f, -0.36330363f, -0.3566038f, -0.34430745f, -0.33699593f,
    -0.30180866f, -0.27321684f
};
static const float dq_seq_c[7] = {
    -1.2f, -1.2f, -0.8f, -0.7f, -0.7f, -0.4f, -0.5f
};
static const float dct4x8_params[3][4] = {
    {2198.0505f, -0.96269625f, -0.7619425f, -0.65511405f},
    {764.36554f, -0.926302f, -0.967523f, -0.2784529f},
    {527.10754f, -1.4594386f, -1.4500821f, -1.5843723f}
};
static const float dct4_params[3][4] = {
    {2200.0f, 0.0f, 0.0f, 0.0f},
    {392.0f, 0.0f, 0.0f, 0.0f},
    {112.0f, -0.25f, -0.25f, -0.5f}
};

/* One channel's parameter list for a DCT-family matrix. */
typedef struct {
    float v[32];
    uint32_t n;
} jxl_dq_params;

typedef struct {
    int mode;            /* 0 dct, 1 hornuss, 2 dct2, 3 dct4, 4 dct4x8,
                            5 afv, 7 raw */
    jxl_dq_params dct[3];
    jxl_dq_params dct4x4[3];
    float fixed[3][9];
    float denominator;
    int32_t *raw[3];
    uint32_t raw_w, raw_h;
} jxl_dq_encoding;

static void dq_seq(jxl_dq_encoding *e, float a, float b, float c) {
    int i;
    e->mode = 0;
    e->dct[0].n = e->dct[1].n = e->dct[2].n = 8;
    e->dct[0].v[0] = a;
    e->dct[1].v[0] = b;
    e->dct[2].v[0] = c;
    for (i = 0; i < 7; i++) {
        e->dct[0].v[i + 1] = dq_seq_a[i];
        e->dct[1].v[i + 1] = dq_seq_b[i];
        e->dct[2].v[i + 1] = dq_seq_c[i];
    }
}

static void dq_set_dct(jxl_dq_encoding *e, const float *c0, uint32_t n0,
                       const float *c1, uint32_t n1, const float *c2,
                       uint32_t n2) {
    e->mode = 0;
    memcpy(e->dct[0].v, c0, n0 * sizeof(float)); e->dct[0].n = n0;
    memcpy(e->dct[1].v, c1, n1 * sizeof(float)); e->dct[1].n = n1;
    memcpy(e->dct[2].v, c2, n2 * sizeof(float)); e->dct[2].n = n2;
}

static void dq_default(jxl_dq_encoding *e, int tr) {
    static const float d8_0[6] = {3150.0f, 0.0f, -0.4f, -0.4f, -0.4f, -2.0f};
    static const float d8_1[6] = {560.0f, 0.0f, -0.3f, -0.3f, -0.3f, -0.3f};
    static const float d8_2[6] = {512.0f, -2.0f, -1.0f, 0.0f, -1.0f, -2.0f};
    static const float d16_0[7] = {8996.873f, -1.3000778f, -0.4942453f,
                                   -0.43909377f, -0.6350102f, -0.9017726f,
                                   -1.6162099f};
    static const float d16_1[7] = {3191.4836f, -0.67424583f, -0.80745816f,
                                   -0.4492584f, -0.3586544f, -0.3132239f,
                                   -0.37615025f};
    static const float d16_2[7] = {1157.504f, -2.0531423f, -1.4f, -0.5068713f,
                                   -0.4270873f, -1.4856834f, -4.920914f};
    static const float d32_0[8] = {15718.408f, -1.025f, -0.98f, -0.9012f,
                                   -0.4f, -0.48819396f, -0.421064f, -0.27f};
    static const float d32_1[8] = {7305.7637f, -0.8041958f, -0.76330364f,
                                   -0.5566038f, -0.49785304f, -0.43699592f,
                                   -0.40180868f, -0.27321684f};
    static const float d32_2[8] = {3803.5317f, -3.0607336f, -2.041327f,
                                   -2.023565f, -0.54953897f, -0.4f, -0.4f,
                                   -0.3f};
    static const float d816_0[7] = {7240.7734f, -0.7f, -0.7f, -0.2f, -0.2f,
                                    -0.2f, -0.5f};
    static const float d816_1[7] = {1448.1547f, -0.5f, -0.5f, -0.5f, -0.2f,
                                    -0.2f, -0.2f};
    static const float d816_2[7] = {506.85413f, -1.4f, -0.2f, -0.5f, -0.5f,
                                    -1.5f, -3.6f};
    static const float d832_0[8] = {16283.249f, -1.7812846f, -1.6309059f,
                                    -1.0382179f, -0.85f, -0.7f, -0.9f,
                                    -1.2360638f};
    static const float d832_1[8] = {5089.1577f, -0.3200494f, -0.3536285f,
                                    -0.3034f, -0.61f, -0.5f, -0.5f, -0.6f};
    static const float d832_2[8] = {3397.7761f, -0.32132736f, -0.3450762f,
                                    -0.7034f, -0.9f, -1.0f, -1.0f, -1.1754606f};
    static const float d1632_0[8] = {13844.971f, -0.971138f, -0.658f, -0.42026f,
                                     -0.22712f, -0.2206f, -0.226f, -0.6f};
    static const float d1632_1[8] = {4798.964f, -0.6112531f, -0.8377079f,
                                     -0.7901486f, -0.26927274f, -0.38272768f,
                                     -0.22924222f, -0.20719099f};
    static const float d1632_2[8] = {1807.2369f, -1.2f, -1.2f, -0.7f, -0.7f,
                                     -0.7f, -0.4f, -0.5f};
    int c, i;

    memset(e, 0, sizeof(*e));
    switch (tr) {
        case JXL_TR_DCT8: dq_set_dct(e, d8_0, 6, d8_1, 6, d8_2, 6); break;
        case JXL_TR_HORNUSS:
            e->mode = 1;
            e->fixed[0][0] = 280.0f; e->fixed[0][1] = 3160.0f; e->fixed[0][2] = 3160.0f;
            e->fixed[1][0] = 60.0f;  e->fixed[1][1] = 864.0f;  e->fixed[1][2] = 864.0f;
            e->fixed[2][0] = 18.0f;  e->fixed[2][1] = 200.0f;  e->fixed[2][2] = 200.0f;
            break;
        case JXL_TR_DCT2: {
            static const float p[3][6] = {
                {3840.0f, 2560.0f, 1280.0f, 640.0f, 480.0f, 300.0f},
                {960.0f, 640.0f, 320.0f, 180.0f, 140.0f, 120.0f},
                {640.0f, 320.0f, 128.0f, 64.0f, 32.0f, 16.0f}
            };
            e->mode = 2;
            for (c = 0; c < 3; c++) for (i = 0; i < 6; i++) e->fixed[c][i] = p[c][i];
            break;
        }
        case JXL_TR_DCT4:
            e->mode = 3;
            for (c = 0; c < 3; c++) {
                e->fixed[c][0] = 1.0f;
                e->fixed[c][1] = 1.0f;
                memcpy(e->dct[c].v, dct4_params[c], 4 * sizeof(float));
                e->dct[c].n = 4;
            }
            break;
        case JXL_TR_DCT16: dq_set_dct(e, d16_0, 7, d16_1, 7, d16_2, 7); break;
        case JXL_TR_DCT32: dq_set_dct(e, d32_0, 8, d32_1, 8, d32_2, 8); break;
        case JXL_TR_DCT16X8:
        case JXL_TR_DCT8X16: dq_set_dct(e, d816_0, 7, d816_1, 7, d816_2, 7); break;
        case JXL_TR_DCT32X8:
        case JXL_TR_DCT8X32: dq_set_dct(e, d832_0, 8, d832_1, 8, d832_2, 8); break;
        case JXL_TR_DCT32X16:
        case JXL_TR_DCT16X32: dq_set_dct(e, d1632_0, 8, d1632_1, 8, d1632_2, 8); break;
        case JXL_TR_DCT4X8:
        case JXL_TR_DCT8X4:
            e->mode = 4;
            for (c = 0; c < 3; c++) {
                e->fixed[c][0] = 1.0f;
                memcpy(e->dct[c].v, dct4x8_params[c], 4 * sizeof(float));
                e->dct[c].n = 4;
            }
            break;
        case JXL_TR_AFV0: case JXL_TR_AFV1: case JXL_TR_AFV2: case JXL_TR_AFV3: {
            static const float p[3][9] = {
                {3072.0f, 3072.0f, 256.0f, 256.0f, 256.0f, 414.0f, 0.0f, 0.0f, 0.0f},
                {1024.0f, 1024.0f, 50.0f, 50.0f, 50.0f, 58.0f, 0.0f, 0.0f, 0.0f},
                {384.0f, 384.0f, 12.0f, 12.0f, 12.0f, 22.0f, -0.25f, -0.25f, -0.25f}
            };
            e->mode = 5;
            for (c = 0; c < 3; c++) {
                for (i = 0; i < 9; i++) e->fixed[c][i] = p[c][i];
                memcpy(e->dct[c].v, dct4x8_params[c], 4 * sizeof(float));
                e->dct[c].n = 4;
                memcpy(e->dct4x4[c].v, dct4_params[c], 4 * sizeof(float));
                e->dct4x4[c].n = 4;
            }
            break;
        }
        case JXL_TR_DCT64: dq_seq(e, 23966.166f, 8380.191f, 4493.024f); break;
        case JXL_TR_DCT32X64:
        case JXL_TR_DCT64X32: dq_seq(e, 15358.898f, 5597.3604f, 2919.9617f); break;
        case JXL_TR_DCT128: dq_seq(e, 47932.332f, 16760.383f, 8986.048f); break;
        case JXL_TR_DCT64X128:
        case JXL_TR_DCT128X64: dq_seq(e, 30717.797f, 11194.721f, 5839.9233f); break;
        case JXL_TR_DCT256: dq_seq(e, 95864.664f, 33520.766f, 17972.096f); break;
        default: dq_seq(e, 61435.594f, 24209.441f, 12979.847f); break;
    }
}

static float dq_interpolate(float pos, float max, const float *bands, int len) {
    float scaled_pos, frac;
    int idx;
    float a, b;
    if (len == 1) return bands[0];
    scaled_pos = pos * (float)(len - 1) / max;
    idx = (int)scaled_pos;
    if (idx < 0) idx = 0;
    if (idx >= len - 1) idx = len - 2;
    frac = scaled_pos - (float)idx;
    a = bands[idx];
    b = bands[idx + 1];
    return a * powf(b / a, frac);
}

static float dq_mult(float x) {
    return x > 0.0f ? 1.0f + x : 1.0f / (1.0f - x);
}

static int dct_quant_weights(const float *params, uint32_t n, uint32_t width,
                             uint32_t height, float *out) {
    float bands[32];
    uint32_t i, x, y;
    float last = params[0];
    bands[0] = last;
    for (i = 1; i < n; i++) {
        float band = last * dq_mult(params[i]);
        if (!(band > 0.0f)) return -1;
        bands[i] = band;
        last = band;
    }
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float dx = (float)x / (float)(width - 1);
            float dy = (float)y / (float)(height - 1);
            float distance = sqrtf(dx * dx + dy * dy);
            out[y * width + x] =
                dq_interpolate(distance, 1.4142135623730951f + 1e-6f, bands, (int)n);
        }
    }
    return 0;
}

static const float afv_freqs[16] = {
    0.0f, 0.0f, 0.8517779f, 5.3777843f, 0.0f, 0.0f, 4.734748f, 5.4492455f,
    1.659827f, 4.0f, 7.275749f, 10.423227f, 2.6629324f, 7.6306577f, 8.962389f,
    12.971662f
};

/* Builds the reciprocal dequant weights of one matrix (all 3 channels). */
static int dq_into_matrix(jxl_ctx *ctx, jxl_dq_encoding *e, int tr,
                          float *out[3]) {
    uint32_t width, height, n;
    int c;
    uint32_t i, x, y;

    jxl_tr_matrix_size(tr, &width, &height);
    n = width * height;
    for (c = 0; c < 3; c++) {
        out[c] = (float *)jxl_calloc(ctx, n, sizeof(float));
        if (!out[c]) return -1;
    }

    switch (e->mode) {
        case 0:
            for (c = 0; c < 3; c++) {
                if (dct_quant_weights(e->dct[c].v, e->dct[c].n, width, height,
                                      out[c]) != 0)
                    return -1;
            }
            break;
        case 1:
            for (c = 0; c < 3; c++) {
                for (i = 0; i < 64; i++) out[c][i] = e->fixed[c][0];
                out[c][0] = 1.0f;
                out[c][1] = e->fixed[c][1];
                out[c][8] = e->fixed[c][1];
                out[c][9] = e->fixed[c][2];
            }
            break;
        case 2:
            for (c = 0; c < 3; c++) {
                out[c][0] = 1.0f;
                for (i = 0; i < 6; i++) {
                    uint32_t shift = i / 2;
                    uint32_t dim = 1u << shift;
                    float val = e->fixed[c][i];
                    if (i % 2 == 0) {
                        for (y = 0; y < dim; y++) {
                            for (x = dim; x < dim * 2; x++) {
                                out[c][y * 8 + x] = val;
                                out[c][x * 8 + y] = val;
                            }
                        }
                    } else {
                        for (y = dim; y < dim * 2; y++) {
                            for (x = dim; x < dim * 2; x++) out[c][y * 8 + x] = val;
                        }
                    }
                }
            }
            break;
        case 3:
            for (c = 0; c < 3; c++) {
                float mat[16];
                if (dct_quant_weights(e->dct[c].v, e->dct[c].n, 4, 4, mat) != 0)
                    return -1;
                for (y = 0; y < 4; y++) {
                    for (x = 0; x < 4; x++) {
                        float v = mat[y * 4 + x];
                        out[c][y * 16 + x * 2] = v;
                        out[c][y * 16 + x * 2 + 1] = v;
                        out[c][(y * 2 + 1) * 8 + x * 2] = v;
                        out[c][(y * 2 + 1) * 8 + x * 2 + 1] = v;
                    }
                }
                out[c][1] /= e->fixed[c][0];
                out[c][8] /= e->fixed[c][0];
                out[c][9] /= e->fixed[c][1];
            }
            break;
        case 4:
            for (c = 0; c < 3; c++) {
                float mat[32];
                if (dct_quant_weights(e->dct[c].v, e->dct[c].n, 8, 4, mat) != 0)
                    return -1;
                for (y = 0; y < 4; y++) {
                    memcpy(out[c] + (y * 2) * 8, mat + y * 8, 8 * sizeof(float));
                    memcpy(out[c] + (y * 2 + 1) * 8, mat + y * 8, 8 * sizeof(float));
                }
                out[c][8] /= e->fixed[c][0];
            }
            break;
        case 5:
            for (c = 0; c < 3; c++) {
                float w48[32], w44[16], bands[4];
                float prev;
                if (dct_quant_weights(e->dct[c].v, e->dct[c].n, 8, 4, w48) != 0)
                    return -1;
                if (dct_quant_weights(e->dct4x4[c].v, e->dct4x4[c].n, 4, 4, w44) != 0)
                    return -1;
                bands[0] = e->fixed[c][5];
                prev = bands[0];
                for (i = 1; i < 4; i++) {
                    bands[i] = prev * dq_mult(e->fixed[c][5 + i]);
                    prev = bands[i];
                }
                for (y = 0; y < 4; y++) {
                    for (x = 0; x < 4; x++) {
                        float v;
                        if (x == 0 && y == 0) v = 1.0f;
                        else if (x == 0 && y == 1) v = e->fixed[c][2];
                        else if (x == 1 && y == 0) v = e->fixed[c][3];
                        else if (x == 1 && y == 1) v = e->fixed[c][4];
                        else {
                            v = dq_interpolate(afv_freqs[y * 4 + x] - afv_freqs[2],
                                               afv_freqs[15] - afv_freqs[2] + 1e-6f,
                                               bands, 4);
                        }
                        out[c][16 * y + 2 * x] = v;
                    }
                }
                for (y = 0; y < 4; y++) {
                    float *row0 = out[c] + y * 16;
                    float *row1 = row0 + 8;
                    for (x = 0; x < 8; x++) {
                        row1[x] = (y == 0 && x == 0) ? e->fixed[c][0] : w48[y * 8 + x];
                    }
                    for (x = 0; x < 4; x++) {
                        row0[x * 2 + 1] =
                            (y == 0 && x == 0) ? e->fixed[c][1] : w44[y * 4 + x];
                    }
                }
            }
            break;
        default:   /* raw */
            for (c = 0; c < 3; c++) {
                for (i = 0; i < n; i++) out[c][i] = (float)e->raw[c][i] * e->denominator;
            }
            break;
    }

    if (e->mode != 7) {
        for (c = 0; c < 3; c++) {
            for (i = 0; i < n; i++) out[c][i] = 1.0f / out[c][i];
        }
    }
    for (c = 0; c < 3; c++) {
        for (i = 0; i < n; i++) {
            if (!(out[c][i] < 1e8f) || !(out[c][i] > 0.0f)) {
                JXL_ERR(ctx, "vardct: bad dequant matrix element");
                return -1;
            }
        }
    }
    return 0;
}

static int read_dct_params(jxl_ctx *ctx, jxl_br *br, jxl_dq_params p[3]) {
    uint32_t n = jxl_br_read(br, 4) + 1;
    uint32_t c, i;
    if (n > 32) {
        JXL_ERR(ctx, "vardct: too many dct params");
        return -1;
    }
    for (c = 0; c < 3; c++) p[c].n = n;
    for (c = 0; c < 3; c++) {
        for (i = 0; i < n; i++) p[c].v[i] = jxl_br_f16(br);
    }
    for (c = 0; c < 3; c++) p[c].v[0] *= 64.0f;
    return 0;
}

static void read_fixed(jxl_br *br, float out[3][9], int n) {
    int c, i;
    for (c = 0; c < 3; c++) {
        for (i = 0; i < n; i++) out[c][i] = jxl_br_f16(br);
    }
}

/* The 17 matrix slots, in bitstream order. */
static const uint8_t dq_slot_tr[17] = {
    JXL_TR_DCT8, JXL_TR_HORNUSS, JXL_TR_DCT2, JXL_TR_DCT4, JXL_TR_DCT16,
    JXL_TR_DCT32, JXL_TR_DCT8X16, JXL_TR_DCT8X32, JXL_TR_DCT16X32,
    JXL_TR_DCT4X8, JXL_TR_AFV0, JXL_TR_DCT64, JXL_TR_DCT32X64, JXL_TR_DCT128,
    JXL_TR_DCT64X128, JXL_TR_DCT256, JXL_TR_DCT128X256
};

int jxl_dequant_matrices_read(jxl_ctx *ctx, jxl_br *br,
                              jxl_dequant_matrices *dm, uint32_t bit_depth,
                              uint32_t num_lf_groups,
                              jxl_ma_config *global_ma) {
    int all_default;
    uint32_t slot;
    int rc = -1;
    jxl_dq_encoding e;
    jxl_modular mod;
    jxl_chanlist cl;

    memset(dm, 0, sizeof(*dm));
    all_default = jxl_br_bool(br);

    for (slot = 0; slot < 17; slot++) {
        int tr = dq_slot_tr[slot];
        uint32_t mw, mh;
        int c;

        memset(&mod, 0, sizeof(mod));
        memset(&cl, 0, sizeof(cl));
        jxl_tr_matrix_size(tr, &mw, &mh);

        if (all_default) {
            dq_default(&e, tr);
        } else {
            uint32_t mode = jxl_br_read(br, 3);
            int midx = jxl_tr_matrix_index(tr);
            if (mode >= 1 && mode <= 5 &&
                !(midx == 0 || midx == 1 || midx == 2 || midx == 3 ||
                  midx == 9 || midx == 10)) {
                JXL_ERR(ctx, "vardct: invalid dequant encoding mode");
                goto done;
            }
            memset(&e, 0, sizeof(e));
            switch (mode) {
                case 0: dq_default(&e, tr); break;
                case 1: e.mode = 1; read_fixed(br, e.fixed, 3); break;
                case 2: e.mode = 2; read_fixed(br, e.fixed, 6); break;
                case 3:
                    e.mode = 3;
                    read_fixed(br, e.fixed, 2);
                    if (read_dct_params(ctx, br, e.dct) != 0) goto done;
                    break;
                case 4:
                    e.mode = 4;
                    read_fixed(br, e.fixed, 1);
                    if (read_dct_params(ctx, br, e.dct) != 0) goto done;
                    break;
                case 5: {
                    int ci, i;
                    e.mode = 5;
                    read_fixed(br, e.fixed, 9);
                    for (ci = 0; ci < 3; ci++) {
                        for (i = 0; i < 6; i++) e.fixed[ci][i] *= 64.0f;
                    }
                    if (read_dct_params(ctx, br, e.dct) != 0) goto done;
                    if (read_dct_params(ctx, br, e.dct4x4) != 0) goto done;
                    break;
                }
                case 6:
                    e.mode = 0;
                    if (read_dct_params(ctx, br, e.dct) != 0) goto done;
                    break;
                default: {
                    jxl_mchan_spec specs[3];
                    int k;
                    e.mode = 7;
                    e.denominator = jxl_br_f16(br);
                    for (k = 0; k < 3; k++) {
                        specs[k].w = mw;
                        specs[k].h = mh;
                        specs[k].hshift = 0;
                        specs[k].vshift = 0;
                    }
                    if (jxl_modular_init(ctx, &mod, br, specs, 3, global_ma, 256,
                                         bit_depth) != 0)
                        goto done;
                    if (jxl_modular_transform_channels(ctx, &mod, &cl) != 0) goto done;
                    if (jxl_modular_decode(ctx, &mod, &cl, br,
                                           1 + num_lf_groups * 3 + slot) != 0)
                        goto done;
                    if (jxl_modular_inverse(ctx, &mod, &cl) != 0) goto done;
                    for (k = 0; k < 3; k++) e.raw[k] = mod.base[k].data;
                    e.raw_w = mw;
                    e.raw_h = mh;
                    break;
                }
            }
        }

        if (dq_into_matrix(ctx, &e, tr, dm->matrix[slot]) != 0) goto done;
        /* The transposed copy is what varblocks with need_transpose use. */
        for (c = 0; c < 3; c++) {
            uint32_t i, x, y;
            dm->matrix_tr[slot][c] =
                (float *)jxl_calloc(ctx, (size_t)mw * mh, sizeof(float));
            if (!dm->matrix_tr[slot][c]) goto done;
            for (i = 0; i < mw * mh; i++) {
                x = i % mh;
                y = i / mh;
                dm->matrix_tr[slot][c][i] = dm->matrix[slot][c][x * mw + y];
            }
        }
        jxl_chanlist_free(ctx, &cl);
        jxl_modular_free(ctx, &mod);
    }
    rc = 0;

done:
    jxl_chanlist_free(ctx, &cl);
    jxl_modular_free(ctx, &mod);
    if (rc != 0) jxl_dequant_matrices_free(ctx, dm);
    return rc;
}

void jxl_dequant_matrices_free(jxl_ctx *ctx, jxl_dequant_matrices *dm) {
    int i, c;
    for (i = 0; i < 17; i++) {
        for (c = 0; c < 3; c++) {
            jxl_free(ctx, dm->matrix[i][c]);
            jxl_free(ctx, dm->matrix_tr[i][c]);
            dm->matrix[i][c] = NULL;
            dm->matrix_tr[i][c] = NULL;
        }
    }
}

/* ===================================================================== */
/* HF pass (coefficient orders + distributions)                           */
/* ===================================================================== */

int jxl_hf_pass_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_pass *hp,
                     const jxl_hf_block_ctx *bc, uint32_t num_hf_presets,
                     jxl_natural_orders *no) {
    uint32_t used_orders;
    jxl_dec perm_dec;
    int have_perm = 0;
    uint32_t idx;
    int rc = -1;

    memset(hp, 0, sizeof(*hp));
    memset(&perm_dec, 0, sizeof(perm_dec));

    used_orders = jxl_br_u32(br, 0x5F, 0, 0x13, 0, 0x00, 0, 0, 13);
    if (used_orders != 0) {
        if (jxl_dec_init(ctx, &perm_dec, br, 8) != 0) return -1;
        jxl_dec_begin(&perm_dec, br);
        have_perm = 1;
    }

    for (idx = 0; idx < 13; idx++) {
        if (have_perm && (used_orders & 1)) {
            uint32_t bw = order_block_sizes[idx][0];
            uint32_t bh = order_block_sizes[idx][1];
            uint32_t size = bw * bh;
            uint32_t skip = size / 64;
            const uint16_t *nat = jxl_natural_order(ctx, no, (int)idx);
            uint32_t *perm;
            int c;
            if (!nat) goto done;
            perm = (uint32_t *)jxl_calloc(ctx, size, sizeof(uint32_t));
            if (!perm) goto done;
            for (c = 0; c < 3; c++) {
                uint32_t i;
                if (jxl_read_permutation(ctx, &perm_dec, br, size, skip, perm) != 0) {
                    jxl_free(ctx, perm);
                    goto done;
                }
                hp->order[idx][c] =
                    (uint16_t *)jxl_calloc(ctx, (size_t)size * 2, sizeof(uint16_t));
                if (!hp->order[idx][c]) {
                    jxl_free(ctx, perm);
                    goto done;
                }
                for (i = 0; i < size; i++) {
                    hp->order[idx][c][i * 2] = nat[perm[i] * 2];
                    hp->order[idx][c][i * 2 + 1] = nat[perm[i] * 2 + 1];
                }
            }
            jxl_free(ctx, perm);
        }
        used_orders >>= 1;
    }
    if (have_perm && jxl_dec_finalize(&perm_dec) != 0) {
        JXL_ERR(ctx, "vardct: bad coefficient order stream");
        goto done;
    }

    if (jxl_dec_init(ctx, &hp->dist, br,
                     495 * num_hf_presets * bc->num_block_clusters) != 0)
        goto done;
    hp->have_dist = 1;
    rc = 0;

done:
    jxl_dec_free(&perm_dec);
    if (rc != 0) jxl_hf_pass_free(ctx, hp);
    return rc;
}

void jxl_hf_pass_free(jxl_ctx *ctx, jxl_hf_pass *hp) {
    int i, c;
    for (i = 0; i < 13; i++) {
        for (c = 0; c < 3; c++) {
            jxl_free(ctx, hp->order[i][c]);
            hp->order[i][c] = NULL;
        }
    }
    if (hp->have_dist) jxl_dec_free(&hp->dist);
    hp->have_dist = 0;
}

/* ===================================================================== */
/* HF coefficient decoding                                                */
/* ===================================================================== */

static const uint32_t coeff_freq_context[63] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17,
    18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25,
    25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29,
    30, 30, 30, 30
};
static const uint32_t coeff_num_nonzero_context[63] = {
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152,
    152, 152, 152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206
};

void jxl_jpeg_upsampling_shifts(const uint32_t ju[3], int idx, int *hs,
                                int *vs) {
    int hscale = (ju[0] == 1 || ju[0] == 2) || (ju[1] == 1 || ju[1] == 2) ||
                 (ju[2] == 1 || ju[2] == 2);
    int vscale = (ju[0] == 1 || ju[0] == 3) || (ju[1] == 1 || ju[1] == 3) ||
                 (ju[2] == 1 || ju[2] == 3);
    switch (ju[idx]) {
        case 0: *hs = hscale; *vs = vscale; break;
        case 1: *hs = 0; *vs = 0; break;
        case 2: *hs = 0; *vs = vscale; break;
        default: *hs = hscale; *vs = 0; break;
    }
}

int jxl_write_hf_coeff(jxl_ctx *ctx, jxl_br *br,
                       const jxl_hf_coeff_params *p, float *out[3],
                       size_t stride[3]) {
    const jxl_hf_block_ctx *bc = p->bc;
    jxl_dec *dist = &p->pass->dist;
    uint32_t lf_idx_mul = (bc->nlf[0] + 1) * (bc->nlf[1] + 1) * (bc->nlf[2] + 1);
    uint32_t hf_idx_mul = bc->nqf + 1;
    int hshifts[3], vshifts[3];
    uint32_t ctx_size, cluster_base;
    uint32_t hfp_bits, hfp;
    uint32_t *nz_row[3];
    uint32_t nz_len[3];
    uint32_t x, y;
    int c, rc = -1;

    for (c = 0; c < 3; c++) {
        jxl_jpeg_upsampling_shifts(p->jpeg_upsampling, c, &hshifts[c], &vshifts[c]);
    }

    hfp_bits = jxl_bitlen(p->num_hf_presets > 1 ? p->num_hf_presets - 1 : 0);
    hfp = jxl_br_read(br, (int)hfp_bits);
    if (hfp >= p->num_hf_presets) {
        JXL_ERR(ctx, "vardct: HF preset out of range");
        return -1;
    }
    ctx_size = 495 * bc->num_block_clusters;
    cluster_base = ctx_size * hfp;
    if (cluster_base + ctx_size > dist->num_dist) {
        JXL_ERR(ctx, "vardct: HF cluster map too small");
        return -1;
    }

    jxl_dec_begin(dist, br);

    for (c = 0; c < 3; c++) {
        nz_len[c] = (p->bi_w + ((1u << hshifts[c]) - 1)) >> hshifts[c];
        nz_row[c] = (uint32_t *)jxl_calloc(ctx, nz_len[c] ? nz_len[c] : 1,
                                           sizeof(uint32_t));
        if (!nz_row[c]) goto done;
    }

    for (y = 0; y < p->bi_h; y++) {
        for (x = 0; x < p->bi_w; x++) {
            const jxl_block_info *bi = &p->block_info[(size_t)y * p->bi_stride + x];
            uint32_t bw8, bh8, num_blocks, num_blocks_log, order_id;
            uint32_t lf_idx = 0, hf_idx = 0;
            int32_t qf;
            int ci;

            if (bi->dct_select >= JXL_TR_COUNT) continue;
            qf = bi->hf_mul;
            jxl_tr_select_size(bi->dct_select, &bw8, &bh8);
            num_blocks = bw8 * bh8;
            num_blocks_log = jxl_bitlen(num_blocks) - 1;
            order_id = (uint32_t)jxl_tr_order_id(bi->dct_select);

            if (p->lf_quant[0]) {
                static const int order[3] = {0, 2, 1};
                int k;
                for (k = 0; k < 3; k++) {
                    int cc = order[k];
                    uint32_t i;
                    uint32_t sx = x >> hshifts[cc];
                    uint32_t sy = y >> vshifts[cc];
                    int32_t q;
                    lf_idx *= bc->nlf[cc] + 1;
                    if (sx >= p->lf_quant[cc]->w || sy >= p->lf_quant[cc]->h) continue;
                    q = p->lf_quant[cc]->data[(size_t)sy * p->lf_quant[cc]->stride + sx];
                    for (i = 0; i < bc->nlf[cc]; i++) {
                        if (q > bc->lf_thresholds[cc][i]) lf_idx++;
                    }
                }
            }
            {
                uint32_t i;
                for (i = 0; i < bc->nqf; i++) {
                    if (qf > (int32_t)bc->qf_thresholds[i]) hf_idx++;
                }
            }

            for (ci = 0; ci < 3; ci++) {
                uint32_t ch_idx = (uint32_t)ci * 13 + order_id;
                int cc = (ci == 0) ? 1 : (ci == 1 ? 0 : 2);
                int hshift = hshifts[cc], vshift = vshifts[cc];
                uint32_t sx = x >> hshift, sy = y >> vshift;
                uint32_t idx, block_ctx, nz_ctx, predicted;
                uint32_t non_zeros, non_zeros_val;
                uint32_t dx8;
                uint32_t is_prev_nonzero;
                const uint16_t *ord;
                uint32_t coeff_ctx_base;
                uint32_t k;

                if (hshift != 0 || vshift != 0) {
                    if ((sx << hshift) != x || (sy << vshift) != y) continue;
                    if (p->block_info[(size_t)sy * p->bi_stride + sx].dct_select >=
                        JXL_TR_COUNT)
                        continue;
                }

                idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                if (idx >= bc->block_ctx_map_len) {
                    JXL_ERR(ctx, "vardct: block context out of range");
                    goto done;
                }
                block_ctx = bc->block_ctx_map[idx];

                if (sy == 0) {
                    predicted = (sx == 0) ? 32 : nz_row[cc][sx - 1];
                } else if (sx == 0) {
                    predicted = nz_row[cc][sx];
                } else {
                    predicted = (nz_row[cc][sx] + nz_row[cc][sx - 1] + 1) >> 1;
                }
                nz_ctx = block_ctx +
                         (predicted >= 8 ? 4 + predicted / 2 : predicted) *
                             bc->num_block_clusters;

                non_zeros = jxl_dec_read_clustered(
                    dist, br, dist->clusters[cluster_base + nz_ctx], 0);
                if (non_zeros > (63u << num_blocks_log)) {
                    JXL_ERR(ctx, "vardct: non_zeros too large");
                    goto done;
                }
                non_zeros_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for (dx8 = 0; dx8 < bw8 && sx + dx8 < nz_len[cc]; dx8++) {
                    nz_row[cc][sx + dx8] = non_zeros_val;
                }
                if (non_zeros == 0) continue;

                is_prev_nonzero = (non_zeros <= num_blocks * 4) ? 1 : 0;
                ord = p->pass->order[order_id][cc];
                if (!ord) ord = jxl_natural_order(ctx, p->no, (int)order_id);
                if (!ord) goto done;

                coeff_ctx_base = block_ctx * 458 + 37 * bc->num_block_clusters;
                for (k = num_blocks; k < num_blocks * 64; k++) {
                    uint32_t kk = k - num_blocks;
                    uint32_t nzi = (non_zeros - 1) >> num_blocks_log;
                    uint32_t fi = kk >> num_blocks_log;
                    uint32_t coeff_ctx;
                    uint32_t ucoeff;
                    int32_t coeff;
                    uint32_t dx, dy, px, py;

                    if (nzi > 62 || fi > 62) {
                        JXL_ERR(ctx, "vardct: coefficient context out of range");
                        goto done;
                    }
                    coeff_ctx = (coeff_num_nonzero_context[nzi] +
                                 coeff_freq_context[fi]) * 2 + is_prev_nonzero;
                    if (coeff_ctx >= 458) {
                        JXL_ERR(ctx, "vardct: too many zeros in varblock");
                        goto done;
                    }
                    ucoeff = jxl_dec_read_clustered(
                        dist, br,
                        dist->clusters[cluster_base + coeff_ctx_base + coeff_ctx],
                        0);
                    if (ucoeff == 0) {
                        is_prev_nonzero = 0;
                        continue;
                    }
                    coeff = jxl_unpack_signed(ucoeff) << p->coeff_shift;
                    dx = ord[k * 2];
                    dy = ord[k * 2 + 1];
                    if (jxl_tr_need_transpose(bi->dct_select)) {
                        uint32_t t = dx; dx = dy; dy = t;
                    }
                    px = sx * 8 + dx;
                    py = sy * 8 + dy;
                    {
                        /* Coefficients accumulate as integers, stored in the
                           float plane's bit pattern until dequantization. */
                        int32_t *slot = (int32_t *)&out[cc][(size_t)py * stride[cc] + px];
                        *slot += coeff;
                    }
                    is_prev_nonzero = 1;
                    non_zeros--;
                    if (non_zeros == 0) break;
                }
                if (br->err || dist->err) {
                    JXL_ERR(ctx, "vardct: truncated HF coefficients");
                    goto done;
                }
            }
        }
    }

    if (jxl_dec_finalize(dist) != 0) {
        JXL_ERR(ctx, "vardct: bad HF coefficient ANS state");
        goto done;
    }
    rc = 0;

done:
    for (c = 0; c < 3; c++) jxl_free(ctx, nz_row[c]);
    return rc;
}

/* ===================================================================== */
/* HF metadata (varblock layout, chroma-from-luma, EPF sigma)             */
/* ===================================================================== */

void jxl_hf_meta_free(jxl_ctx *ctx, jxl_hf_meta *m) {
    if (!m) return;
    jxl_free(ctx, m->x_from_y);
    jxl_free(ctx, m->b_from_y);
    jxl_free(ctx, m->block_info);
    jxl_free(ctx, m->epf_sigma);
    memset(m, 0, sizeof(*m));
}

/* Reads the per-LF-group varblock metadata: four Modular channels holding
   the chroma-from-luma factors, the (dct_select, hf_mul) pairs in decode
   order, and the EPF sharpness map. The pairs are then painted over the
   block grid, which is also how the layout gets validated. */
int jxl_hf_meta_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_meta *m,
                     uint32_t num_lf_groups, uint32_t lf_group_idx,
                     uint32_t lf_width, uint32_t lf_height,
                     const uint32_t jpeg_upsampling[3], uint32_t bit_depth,
                     jxl_ma_config *global_ma, const jxl_epf *epf,
                     uint32_t quantizer_global_scale) {
    uint32_t bw = (lf_width + 7) / 8;
    uint32_t bh = (lf_height + 7) / 8;
    uint32_t cfl_w = (lf_width + 63) / 64;
    uint32_t cfl_h = (lf_height + 63) / 64;
    uint32_t nb_blocks;
    jxl_mchan_spec specs[4];
    jxl_modular mod;
    jxl_chanlist cl;
    int h_up = 0, v_up = 0, i;
    uint32_t x, y, data_idx = 0;
    float epf_quant_mul = 0.0f;
    int rc = -1;

    memset(m, 0, sizeof(*m));
    memset(&mod, 0, sizeof(mod));
    memset(&cl, 0, sizeof(cl));

    for (i = 0; i < 3; i++) {
        if (jpeg_upsampling[i] == 1 || jpeg_upsampling[i] == 2) h_up = 1;
        if (jpeg_upsampling[i] == 1 || jpeg_upsampling[i] == 3) v_up = 1;
    }
    if (h_up) bw = ((bw + 1) / 2) * 2;
    if (v_up) bh = ((bh + 1) / 2) * 2;

    nb_blocks = 1 + jxl_br_read(br, (int)jxl_bitlen(bw * bh > 1 ? bw * bh - 1 : 0));
    if (nb_blocks > bw * bh || br->err) {
        JXL_ERR(ctx, "vardct: bad varblock count");
        return -1;
    }

    specs[0].w = cfl_w; specs[0].h = cfl_h;
    specs[1].w = cfl_w; specs[1].h = cfl_h;
    specs[2].w = nb_blocks; specs[2].h = 2;
    specs[3].w = bw; specs[3].h = bh;
    for (i = 0; i < 4; i++) { specs[i].hshift = 0; specs[i].vshift = 0; }

    if (jxl_modular_init(ctx, &mod, br, specs, 4, global_ma, 0, bit_depth) != 0)
        goto done;
    if (jxl_modular_transform_channels(ctx, &mod, &cl) != 0) goto done;
    if (jxl_modular_decode(ctx, &mod, &cl, br,
                           1 + 2 * num_lf_groups + lf_group_idx) != 0)
        goto done;
    if (jxl_modular_inverse(ctx, &mod, &cl) != 0) goto done;

    m->cfl_w = cfl_w;
    m->cfl_h = cfl_h;
    m->bw = bw;
    m->bh = bh;
    m->x_from_y = (int32_t *)jxl_calloc(ctx, (size_t)cfl_w * cfl_h, sizeof(int32_t));
    m->b_from_y = (int32_t *)jxl_calloc(ctx, (size_t)cfl_w * cfl_h, sizeof(int32_t));
    m->block_info =
        (jxl_block_info *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(jxl_block_info));
    m->epf_sigma = (float *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(float));
    if (!m->x_from_y || !m->b_from_y || !m->block_info || !m->epf_sigma) goto done;

    for (y = 0; y < cfl_h; y++) {
        for (x = 0; x < cfl_w; x++) {
            m->x_from_y[(size_t)y * cfl_w + x] =
                mod.base[0].data[(size_t)y * mod.base[0].stride + x];
            m->b_from_y[(size_t)y * cfl_w + x] =
                mod.base[1].data[(size_t)y * mod.base[1].stride + x];
        }
    }
    for (i = 0; i < (int)(bw * bh); i++) m->block_info[i].dct_select = JXL_BLK_UNINIT;

    if (epf && epf->enabled) {
        epf_quant_mul = epf->quant_mul * 65536.0f / (float)quantizer_global_scale;
    }

    for (y = 0; y < bh; y++) {
        x = 0;
        while (x < bw) {
            jxl_block_info *cur = &m->block_info[(size_t)y * bw + x];
            uint32_t dw, dh, dx, dy;
            int32_t dct_select, hf_mul;
            float sigma_base = 0.0f;

            if (cur->dct_select != JXL_BLK_UNINIT) { x++; continue; }
            if (data_idx >= nb_blocks) {
                JXL_ERR(ctx, "vardct: block info does not fill the LF group");
                goto done;
            }
            dct_select = mod.base[2].data[data_idx];
            hf_mul = mod.base[2].data[mod.base[2].stride + data_idx] + 1;
            if (dct_select < 0 || dct_select >= JXL_TR_COUNT || hf_mul <= 0) {
                JXL_ERR(ctx, "vardct: bad varblock descriptor");
                goto done;
            }
            jxl_tr_select_size(dct_select, &dw, &dh);
            if ((x % 32) + dw > 32 || (y % 32) + dh > 32) {
                JXL_ERR(ctx, "vardct: varblock crosses a pass group border");
                goto done;
            }
            if (epf_quant_mul != 0.0f) sigma_base = epf_quant_mul / (float)hf_mul;

            for (dy = 0; dy < dh; dy++) {
                for (dx = 0; dx < dw; dx++) {
                    jxl_block_info *b;
                    if (x + dx >= bw || y + dy >= bh) {
                        JXL_ERR(ctx, "vardct: varblock does not fit the LF group");
                        goto done;
                    }
                    b = &m->block_info[(size_t)(y + dy) * bw + (x + dx)];
                    if (b->dct_select != JXL_BLK_UNINIT) {
                        JXL_ERR(ctx, "vardct: varblocks overlap");
                        goto done;
                    }
                    if (dx == 0 && dy == 0) {
                        b->dct_select = (uint8_t)dct_select;
                        b->hf_mul = hf_mul;
                    } else {
                        b->dct_select = JXL_BLK_OCCUPIED;
                    }
                    if (sigma_base != 0.0f) {
                        int32_t sharp =
                            mod.base[3].data[(size_t)(y + dy) * mod.base[3].stride + (x + dx)];
                        if (sharp < 0 || sharp >= 8) {
                            JXL_ERR(ctx, "vardct: bad EPF sharpness");
                            goto done;
                        }
                        m->epf_sigma[(size_t)(y + dy) * bw + (x + dx)] =
                            sigma_base * epf->sharp_lut[sharp];
                    }
                }
            }
            data_idx++;
            x += dw;
        }
    }
    m->have = 1;
    rc = 0;

done:
    jxl_chanlist_free(ctx, &cl);
    jxl_modular_free(ctx, &mod);
    if (rc != 0) jxl_hf_meta_free(ctx, m);
    return rc;
}

/* ===================================================================== */
/* LF dequantization and smoothing                                        */
/* ===================================================================== */

void jxl_copy_lf_dequant(float *dst, size_t dstride, const jxl_mchan *src,
                         const jxl_quantizer *q, float m_lf,
                         int extra_precision) {
    int precision_scale = 1 << (9 - extra_precision);
    double scale_inv = (double)q->global_scale * (double)q->quant_lf;
    float scale = (float)((double)m_lf * precision_scale / scale_inv);
    uint32_t x, y;
    for (y = 0; y < src->h; y++) {
        const int32_t *row = src->data + (size_t)y * src->stride;
        float *out = dst + (size_t)y * dstride;
        for (x = 0; x < src->w; x++) out[x] = (float)row[x] * scale;
    }
}

/* Smooths the LF image where the local gradient is small -- what the encoder
   assumed when it quantized. */
int jxl_adaptive_lf_smoothing(jxl_ctx *ctx, float *plane[3], uint32_t width,
                              uint32_t height, size_t stride,
                              const float m_lf[3], const jxl_quantizer *q) {
    const float scale_self = 0.052262735f;
    const float scale_side = 0.2034514f;
    const float scale_diag = 0.03348292f;
    double scale_inv = (double)q->global_scale * (double)q->quant_lf;
    float lf[3];
    float *udsum[3];
    uint32_t x, y;
    int c, rc = -1;

    udsum[0] = udsum[1] = udsum[2] = NULL;
    if (width <= 2 || height <= 2) return 0;
    for (c = 0; c < 3; c++) lf[c] = (float)(512.0 * (double)m_lf[c] / scale_inv);

    for (c = 0; c < 3; c++) {
        udsum[c] = (float *)jxl_calloc(ctx, (size_t)width * (height - 2),
                                       sizeof(float));
        if (!udsum[c]) goto done;
        for (y = 0; y + 2 < height; y++) {
            const float *up = plane[c] + (size_t)y * stride;
            const float *down = plane[c] + (size_t)(y + 2) * stride;
            float *out = udsum[c] + (size_t)y * width;
            for (x = 0; x < width; x++) out[x] = up[x] + down[x];
        }
    }

    for (y = 0; y + 2 < height; y++) {
        float *row[3];
        const float *ud[3];
        float prev[3];
        for (c = 0; c < 3; c++) {
            row[c] = plane[c] + (size_t)(y + 1) * stride;
            ud[c] = udsum[c] + (size_t)y * width;
            prev[c] = row[c][0];
        }
        for (x = 1; x + 1 < width; x++) {
            float self[3], wa[3], gap = 0.5f, gap_scale;
            for (c = 0; c < 3; c++) {
                float side = prev[c] + row[c][x + 1] + ud[c][x];
                float diag = ud[c][x - 1] + ud[c][x + 1];
                float gap_t;
                self[c] = row[c][x];
                wa[c] = self[c] * scale_self + side * scale_side + diag * scale_diag;
                gap_t = fabsf(wa[c] - self[c]) / lf[c];
                if (gap_t > gap) gap = gap_t;
            }
            gap_scale = 3.0f - 4.0f * gap;
            if (gap_scale < 0.0f) gap_scale = 0.0f;
            for (c = 0; c < 3; c++) {
                row[c][x] = (wa[c] - self[c]) * gap_scale + self[c];
                prev[c] = self[c];
            }
        }
    }
    rc = 0;

done:
    for (c = 0; c < 3; c++) jxl_free(ctx, udsum[c]);
    return rc;
}

void jxl_cfl_lf(float *x, float *y, float *b, uint32_t w, uint32_t h,
                size_t stride, const jxl_lf_chan_corr *corr) {
    float kx = corr->base_correlation_x +
               ((float)((int32_t)corr->x_factor_lf - 128) / (float)corr->colour_factor);
    float kb = corr->base_correlation_b +
               ((float)((int32_t)corr->b_factor_lf - 128) / (float)corr->colour_factor);
    uint32_t i, j;
    for (j = 0; j < h; j++) {
        float *rx = x + (size_t)j * stride;
        const float *ry = y + (size_t)j * stride;
        float *rb = b + (size_t)j * stride;
        for (i = 0; i < w; i++) {
            rx[i] += kx * ry[i];
            rb[i] += kb * ry[i];
        }
    }
}

/* ===================================================================== */
/* dequantization, chroma-from-luma and the inverse varblock transform     */
/* ===================================================================== */

/* Turns the accumulated integer coefficients of one varblock into floats,
   applying the quantization bias, the dequant matrix and the global scale. */
void jxl_dequant_varblock(float *coeff, size_t stride, int tr, int32_t hf_mul,
                          int channel, const jxl_dequant_matrices *dm,
                          const jxl_quantizer *q, float qm_scale,
                          float quant_bias, float quant_bias_numerator) {
    uint32_t bw, bh, w, h, x, y;
    const float *matrix;
    float mul;
    int slot = jxl_tr_matrix_index(tr);

    jxl_tr_select_size(tr, &bw, &bh);
    w = bw * 8;
    h = bh * 8;
    mul = 65536.0f / ((float)q->global_scale * (float)hf_mul) * qm_scale;
    matrix = jxl_tr_need_transpose(tr) ? dm->matrix_tr[slot][channel]
                                       : dm->matrix[slot][channel];

    for (y = 0; y < h; y++) {
        float *row = coeff + (size_t)y * stride;
        const float *mrow = matrix + (size_t)y * w;
        for (x = 0; x < w; x++) {
            int32_t qn = *(int32_t *)&row[x];
            float v = (float)qn;
            if (fabsf(v) <= 1.0f) v *= quant_bias;
            else v -= quant_bias_numerator / v;
            v *= mrow[x];
            v *= mul;
            row[x] = v;
        }
    }
}

void jxl_cfl_hf(float *cx, float *cy, float *cb, size_t stride, uint32_t gw,
                uint32_t gh, const int32_t *x_from_y, const int32_t *b_from_y,
                uint32_t cfl_stride, const jxl_lf_chan_corr *corr) {
    uint32_t x, y;
    for (y = 0; y < gh; y++) {
        const int32_t *xr = x_from_y + (size_t)(y / 64) * cfl_stride;
        const int32_t *br_ = b_from_y + (size_t)(y / 64) * cfl_stride;
        float *rx = cx + (size_t)y * stride;
        const float *ry = cy + (size_t)y * stride;
        float *rb = cb + (size_t)y * stride;
        for (x = 0; x < gw; x++) {
            float kx = corr->base_correlation_x +
                       ((float)xr[x / 64] / (float)corr->colour_factor);
            float kb = corr->base_correlation_b +
                       ((float)br_[x / 64] / (float)corr->colour_factor);
            rx[x] += kx * ry[x];
            rb[x] += kb * ry[x];
        }
    }
}

/* ----- the small non-DCT varblock transforms ----- */

static void idct2_in_place(float *block, size_t stride, int size) {
    float scratch[64];
    int num = size / 2;
    int x, y;
    for (y = 0; y < num; y++) {
        for (x = 0; x < num; x++) {
            float c00 = block[(size_t)y * stride + x];
            float c01 = block[(size_t)y * stride + x + num];
            float c10 = block[(size_t)(y + num) * stride + x];
            float c11 = block[(size_t)(y + num) * stride + x + num];
            scratch[(2 * y) * size + 2 * x] = c00 + c01 + c10 + c11;
            scratch[(2 * y) * size + 2 * x + 1] = c00 + c01 - c10 - c11;
            scratch[(2 * y + 1) * size + 2 * x] = c00 - c01 + c10 - c11;
            scratch[(2 * y + 1) * size + 2 * x + 1] = c00 - c01 - c10 + c11;
        }
    }
    for (y = 0; y < size; y++) {
        memcpy(block + (size_t)y * stride, scratch + (size_t)y * size,
               (size_t)size * sizeof(float));
    }
}

static void transform_dct2(float *b, size_t s) {
    idct2_in_place(b, s, 2);
    idct2_in_place(b, s, 4);
    idct2_in_place(b, s, 8);
}

static void transform_dct4(float *b, size_t s) {
    float scratch[64];
    int x, y, ix, iy;
    idct2_in_place(b, s, 2);
    for (y = 0; y < 2; y++) {
        for (x = 0; x < 2; x++) {
            float *sc = scratch + (y * 2 + x) * 16;
            for (iy = 0; iy < 4; iy++) {
                for (ix = 0; ix < 4; ix++) {
                    sc[ix * 4 + iy] = b[(size_t)(y + iy * 2) * s + (x + ix * 2)];
                }
            }
            jxl_dct_2d(sc, 4, 4, 4, 1);
        }
    }
    for (y = 0; y < 2; y++) {
        for (x = 0; x < 2; x++) {
            const float *sc = scratch + (y * 2 + x) * 16;
            for (iy = 0; iy < 4; iy++) {
                for (ix = 0; ix < 4; ix++) {
                    b[(size_t)(y * 4 + iy) * s + (x * 4 + ix)] = sc[iy * 4 + ix];
                }
            }
        }
    }
}

static void transform_hornuss(float *b, size_t s) {
    float scratch[64];
    int x, y, ix, iy, i;
    idct2_in_place(b, s, 2);
    for (y = 0; y < 2; y++) {
        for (x = 0; x < 2; x++) {
            float *sc = scratch + (y * 2 + x) * 16;
            float residual_sum = 0.0f, avg;
            for (iy = 0; iy < 4; iy++) {
                for (ix = 0; ix < 4; ix++) {
                    sc[iy * 4 + ix] = b[(size_t)(y + iy * 2) * s + (x + ix * 2)];
                }
            }
            for (i = 1; i < 16; i++) residual_sum += sc[i];
            avg = sc[0] - residual_sum / 16.0f;
            sc[0] = sc[5];
            sc[5] = 0.0f;
            for (i = 0; i < 16; i++) sc[i] += avg;
        }
    }
    for (y = 0; y < 2; y++) {
        for (x = 0; x < 2; x++) {
            const float *sc = scratch + (y * 2 + x) * 16;
            for (iy = 0; iy < 4; iy++) {
                for (ix = 0; ix < 4; ix++) {
                    b[(size_t)(y * 4 + iy) * s + (x * 4 + ix)] = sc[iy * 4 + ix];
                }
            }
        }
    }
}

static void transform_dct4x8(float *b, size_t s, int transposed) {
    float scratch[64];
    int idx, x, y, ix, iy;
    float c0 = b[0], c1 = b[s];
    b[0] = c0 + c1;
    b[s] = c0 - c1;
    for (idx = 0; idx < 2; idx++) {
        float *sc = scratch + idx * 32;
        for (iy = 0; iy < 4; iy++) {
            for (ix = 0; ix < 8; ix++) {
                sc[iy * 8 + ix] = b[(size_t)(iy * 2 + idx) * s + ix];
            }
        }
        jxl_dct_2d(sc, 8, 8, 4, 1);
    }
    if (transposed) {
        for (y = 0; y < 8; y++) {
            for (x = 0; x < 8; x++) b[(size_t)x * s + y] = scratch[y * 8 + x];
        }
    } else {
        for (y = 0; y < 8; y++) {
            memcpy(b + (size_t)y * s, scratch + y * 8, 8 * sizeof(float));
        }
    }
}

extern const float jxl_afv_basis[16][16];

static void transform_afv(float *b, size_t s, int n) {
    int flip_x = n % 2, flip_y = n / 2;
    float coeff_afv[16], samples_afv[16];
    float scratch_4x4[16], scratch_4x8[32];
    int i, j, ix, iy;

    coeff_afv[0] = (b[0] + b[1] + b[s]) * 4.0f;
    for (i = 1; i < 16; i++) {
        iy = i / 4;
        ix = i % 4;
        coeff_afv[i] = b[(size_t)(2 * iy) * s + 2 * ix];
    }
    for (i = 0; i < 16; i++) samples_afv[i] = 0.0f;
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 16; j++) {
            samples_afv[j] += coeff_afv[i] * jxl_afv_basis[i][j];
        }
    }

    for (i = 0; i < 16; i++) scratch_4x4[i] = 0.0f;
    scratch_4x4[0] = b[0] - b[1] + b[s];
    for (iy = 0; iy < 4; iy++) {
        for (ix = 0; ix < 4; ix++) {
            if ((ix | iy) == 0) continue;
            scratch_4x4[ix * 4 + iy] = b[(size_t)(2 * iy) * s + 2 * ix + 1];
        }
    }
    jxl_dct_2d(scratch_4x4, 4, 4, 4, 1);

    for (i = 0; i < 32; i++) scratch_4x8[i] = 0.0f;
    scratch_4x8[0] = b[0] - b[s];
    for (iy = 0; iy < 4; iy++) {
        for (ix = 0; ix < 8; ix++) {
            if ((ix | iy) == 0) continue;
            scratch_4x8[iy * 8 + ix] = b[(size_t)(2 * iy + 1) * s + ix];
        }
    }
    jxl_dct_2d(scratch_4x8, 8, 8, 4, 1);

    for (iy = 0; iy < 4; iy++) {
        int afv_y = flip_y == 0 ? iy : 3 - iy;
        for (ix = 0; ix < 4; ix++) {
            int afv_x = flip_x == 0 ? ix : 3 - ix;
            b[(size_t)(flip_y * 4 + iy) * s + flip_x * 4 + ix] =
                samples_afv[afv_y * 4 + afv_x];
        }
    }
    for (iy = 0; iy < 4; iy++) {
        for (ix = 0; ix < 4; ix++) {
            b[(size_t)(flip_y * 4 + iy) * s + (1 - flip_x) * 4 + ix] =
                scratch_4x4[iy * 4 + ix];
        }
    }
    for (iy = 0; iy < 4; iy++) {
        memcpy(b + (size_t)((1 - flip_y) * 4 + iy) * s, scratch_4x8 + iy * 8,
               8 * sizeof(float));
    }
}

/* Runs the inverse transform of one varblock, after its DC has been filled
   in from the LF image. */
void jxl_transform_varblock(float *coeff, size_t stride, int tr) {
    uint32_t bw, bh;
    jxl_tr_select_size(tr, &bw, &bh);
    switch (tr) {
        case JXL_TR_DCT2: transform_dct2(coeff, stride); break;
        case JXL_TR_DCT4: transform_dct4(coeff, stride); break;
        case JXL_TR_HORNUSS: transform_hornuss(coeff, stride); break;
        case JXL_TR_DCT4X8: transform_dct4x8(coeff, stride, 0); break;
        case JXL_TR_DCT8X4: transform_dct4x8(coeff, stride, 1); break;
        case JXL_TR_AFV0: transform_afv(coeff, stride, 0); break;
        case JXL_TR_AFV1: transform_afv(coeff, stride, 1); break;
        case JXL_TR_AFV2: transform_afv(coeff, stride, 2); break;
        case JXL_TR_AFV3: transform_afv(coeff, stride, 3); break;
        default:
            jxl_dct_2d(coeff, stride, (int)(bw * 8), (int)(bh * 8), 1);
            break;
    }
}

/* Writes the varblock's DC coefficients from the LF image. For multi-block
   transforms the LF samples are themselves DCT'd and rescaled. */
void jxl_fill_varblock_lf(float *coeff, size_t stride, int tr,
                          const float *lf, size_t lf_stride, uint32_t lf_x,
                          uint32_t lf_y) {
    uint32_t bw, bh, x, y;
    int logbw, logbh;

    jxl_tr_select_size(tr, &bw, &bh);
    if (bw == 1 && bh == 1) {
        coeff[0] = lf[(size_t)lf_y * lf_stride + lf_x];
        return;
    }
    for (y = 0; y < bh; y++) {
        for (x = 0; x < bw; x++) {
            coeff[(size_t)y * stride + x] =
                lf[(size_t)(lf_y + y) * lf_stride + (lf_x + x)];
        }
    }
    logbw = (int)jxl_bitlen(bw) - 1;
    logbh = (int)jxl_bitlen(bh) - 1;
    jxl_dct_2d(coeff, stride, (int)bw, (int)bh, 0);
    for (y = 0; y < bh; y++) {
        for (x = 0; x < bw; x++) {
            coeff[(size_t)y * stride + x] /=
                jxl_scale_f((int)y, 5 - logbh) * jxl_scale_f((int)x, 5 - logbw);
        }
    }
}

/* ===================================================================== */
/* chroma upsampling and YCbCr                                            */
/* ===================================================================== */

/* Doubles a subsampled chroma plane in place, one axis at a time, with the
   3/4 + 1/4 kernel libjxl's render pipeline uses. Samples outside the plane
   are clamped, which for a one-sample border is what mirroring gives.

   The plane holds `w` x `h` valid samples in the top-left of a `stride`-wide
   buffer that must already be large enough for the doubled result. */
static void chroma_upsample_h(float *p, uint32_t w, uint32_t h, size_t stride,
                              uint32_t out_w) {
    uint32_t x, y;
    for (y = 0; y < h; y++) {
        float *row = p + (size_t)y * stride;
        /* Right to left, so the source samples are not overwritten first. */
        for (x = w; x-- > 0;) {
            float cur = row[x] * 0.75f;
            float prev = row[x ? x - 1 : 0];
            float next = row[x + 1 < w ? x + 1 : w - 1];
            if (2 * x + 1 < out_w) row[2 * x + 1] = cur + 0.25f * next;
            row[2 * x] = cur + 0.25f * prev;
        }
    }
}

static void chroma_upsample_v(float *p, uint32_t w, uint32_t h, size_t stride,
                              uint32_t out_h) {
    uint32_t x, y;
    for (y = h; y-- > 0;) {
        const float *mid = p + (size_t)y * stride;
        const float *top = p + (size_t)(y ? y - 1 : 0) * stride;
        const float *bot = p + (size_t)(y + 1 < h ? y + 1 : h - 1) * stride;
        float *out0 = p + (size_t)(2 * y) * stride;
        float *out1 = p + (size_t)(2 * y + 1) * stride;
        for (x = 0; x < w; x++) {
            float m = mid[x] * 0.75f;
            float lo = m + 0.25f * top[x];
            float hi = m + 0.25f * bot[x];
            if (2 * y + 1 < out_h) out1[x] = hi;
            out0[x] = lo;
        }
    }
}

void jxl_chroma_upsample(float *p, uint32_t w, uint32_t h, size_t stride,
                         int hs, int vs, uint32_t out_w, uint32_t out_h) {
    if (vs) {
        chroma_upsample_v(p, w, h, stride, out_h);
        h = JXL_MIN(2 * h, out_h);
    }
    if (hs) {
        chroma_upsample_h(p, w, h, stride, out_w);
    }
}

/* Full-range BT.601 as defined by JFIF clause 7. The planes arrive as
   (Cb, Y, Cr) -- the same slots XYB uses for (X, Y, B) -- and leave as
   (R, G, B) already in the image's encoded color space. */
void jxl_ycbcr_to_rgb(float *cb, float *y, float *cr, size_t n) {
    const float crcr = 1.402f;
    const float cgcb = -0.114f * 1.772f / 0.587f;
    const float cgcr = -0.299f * 1.402f / 0.587f;
    const float cbcb = 1.772f;
    size_t i;
    for (i = 0; i < n; i++) {
        float yv = y[i] + 128.0f / 255.0f;
        float b_ = cb[i], r_ = cr[i];
        cb[i] = yv + crcr * r_;
        y[i] = yv + cgcb * b_ + cgcr * r_;
        cr[i] = yv + cbcb * b_;
    }
}
