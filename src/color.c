/* color.c -- XYB to linear RGB, and the transfer functions used to encode
 * linear samples back into the image's declared color space.
 *
 * XYB is an absolute, perceptually-shaped space: undo the cube-root gamma
 * (with the opsin bias), then apply the opsin inverse matrix to land in the
 * image's linear primaries. `intensity_target` scales absolute nits back to
 * the 0..1 range the rest of the pipeline uses.
 */
#include "jxl_internal.h"

#include <math.h>

void jxl_xyb_to_linear(float *x, float *y, float *b, size_t n,
                       const float opsin_inv[9], const float opsin_bias[3],
                       float intensity_target) {
    float itscale = 255.0f / intensity_target;
    float cbrt_ob[3];
    size_t i;
    int k;

    for (k = 0; k < 3; k++) cbrt_ob[k] = cbrtf(opsin_bias[k]);

    for (i = 0; i < n; i++) {
        float gl = y[i] + x[i] - cbrt_ob[0];
        float gm = y[i] - x[i] - cbrt_ob[1];
        float gs = b[i] - cbrt_ob[2];
        float l = gl * gl * gl + opsin_bias[0];
        float m = gm * gm * gm + opsin_bias[1];
        float s = gs * gs * gs + opsin_bias[2];
        l *= itscale;
        m *= itscale;
        s *= itscale;
        x[i] = opsin_inv[0] * l + opsin_inv[1] * m + opsin_inv[2] * s;
        y[i] = opsin_inv[3] * l + opsin_inv[4] * m + opsin_inv[5] * s;
        b[i] = opsin_inv[6] * l + opsin_inv[7] * m + opsin_inv[8] * s;
    }
}

/* ----- transfer functions (linear -> encoded) ----- */

/* libjxl encodes sRGB with a rational-polynomial approximation rather than a
   real powf, so an exact powf here would differ from the reference decoder by
   one 8-bit step on a good fraction of pixels. This is a straight port. */
static const uint8_t srgb_powtable_upper[16] = {
    0x00, 0x0a, 0x19, 0x26, 0x32, 0x41, 0x4d, 0x5c,
    0x68, 0x75, 0x83, 0x8f, 0xa0, 0xaa, 0xb9, 0xc6
};
static const uint8_t srgb_powtable_lower[16] = {
    0x00, 0xb7, 0x04, 0x0d, 0xcb, 0xe7, 0x41, 0x68,
    0x51, 0xd1, 0xeb, 0xf2, 0x00, 0xb7, 0x04, 0x0d
};

static float bits_to_float(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

static uint32_t float_to_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof(b));
    return b;
}

static float tf_srgb(float s) {
    uint32_t sign = float_to_bits(s) & 0x80000000u;
    uint32_t v = float_to_bits(s) & 0x7fffffffu;
    float v_adj = bits_to_float((v | 0x3e800000u) & 0x3effffffu);
    float pow = 0.059914046f;
    uint32_t idx, mul_bits;
    float fv, small, acc, out;

    pow = pow * v_adj - 0.10889456f;
    pow = pow * v_adj + 0.107963754f;
    pow = pow * v_adj + 0.018092343f;

    idx = ((v >> 23) - 118u) & 0xf;
    mul_bits = 0x40000000u | ((uint32_t)srgb_powtable_upper[idx] << 18) |
               ((uint32_t)srgb_powtable_lower[idx] << 10);

    fv = bits_to_float(v);
    small = fv * 12.92f;
    acc = pow * bits_to_float(mul_bits) - 0.055f;
    out = (fv <= 0.0031308f) ? small : acc;
    return bits_to_float(float_to_bits(out) | sign);
}

static float tf_bt709(float v) {
    if (v <= 0.018053968510807f) return 4.5f * v;
    return 1.09929682680944f * powf(v, 0.45f) - 0.09929682680944f;
}

static float tf_pq(float v) {
    /* SMPTE ST 2084, with 1.0 mapping to 10000 nits. */
    const float m1 = 2610.0f / 16384.0f;
    const float m2 = 2523.0f / 4096.0f * 128.0f;
    const float c1 = 3424.0f / 4096.0f;
    const float c2 = 2413.0f / 4096.0f * 32.0f;
    const float c3 = 2392.0f / 4096.0f * 32.0f;
    float p;
    if (v <= 0.0f) return 0.0f;
    p = powf(v, m1);
    return powf((c1 + c2 * p) / (1.0f + c3 * p), m2);
}

static float tf_hlg(float v) {
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    if (v <= 0.0f) return 0.0f;
    if (v <= 1.0f / 12.0f) return sqrtf(3.0f * v);
    return a * logf(12.0f * v - b) + c;
}

void jxl_linear_to_tf(float *v, size_t n, const jxl_colour_encoding *enc,
                      float intensity_target) {
    size_t i;
    (void)intensity_target;

    if (enc->tf_have_gamma) {
        float g = (float)enc->tf_gamma / 1e7f;   /* stored inverted */
        for (i = 0; i < n; i++) {
            float x = v[i];
            v[i] = x <= 0.0f ? 0.0f : powf(x, g);
        }
        return;
    }
    switch (enc->tf) {
        case JXL_TF_LINEAR:
            break;
        case JXL_TF_709:
            for (i = 0; i < n; i++) v[i] = tf_bt709(v[i]);
            break;
        case JXL_TF_PQ:
            for (i = 0; i < n; i++) v[i] = tf_pq(v[i]);
            break;
        case JXL_TF_DCI:
            for (i = 0; i < n; i++) {
                v[i] = v[i] <= 0.0f ? 0.0f : powf(v[i], 1.0f / 2.6f);
            }
            break;
        case JXL_TF_HLG:
            for (i = 0; i < n; i++) v[i] = tf_hlg(v[i]);
            break;
        case JXL_TF_SRGB:
        default:
            for (i = 0; i < n; i++) v[i] = tf_srgb(v[i]);
            break;
    }
}
