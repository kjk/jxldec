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

/* SSE2 is baseline on x64. -DJXL_COLOR_FORCE_SCALAR builds the scalar path
   alone so the bit-identical claim can be diffed rather than asserted.
   Nothing here approximates anything the scalar code does not already
   approximate: tf_srgb is libjxl's polynomial-plus-table, not a powf, and
   the cbrt in the XYB inverse is already hoisted out of the loop. */
#if !defined(JXL_COLOR_FORCE_SCALAR) && \
    (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define JXL_COLOR_SSE2 1
#include <emmintrin.h>
#endif


void jxl_xyb_to_linear(float *x, float *y, float *b, size_t n,
                       const float opsin_inv[9], const float opsin_bias[3],
                       float intensity_target) {
    float itscale = 255.0f / intensity_target;
    float cbrt_ob[3];
    size_t i;
    int k;

    for (k = 0; k < 3; k++) cbrt_ob[k] = cbrtf(opsin_bias[k]);

    i = 0;
#ifdef JXL_COLOR_SSE2
    {
        /* Straight four-at-a-time: the loop body is pure arithmetic, so each
           lane runs the identical sequence and the result is bit-identical.
           The multiply-adds keep the scalar left-to-right association. */
        const __m128 c0 = _mm_set1_ps(cbrt_ob[0]), c1 = _mm_set1_ps(cbrt_ob[1]);
        const __m128 c2 = _mm_set1_ps(cbrt_ob[2]);
        const __m128 b0 = _mm_set1_ps(opsin_bias[0]), b1 = _mm_set1_ps(opsin_bias[1]);
        const __m128 b2 = _mm_set1_ps(opsin_bias[2]);
        const __m128 its = _mm_set1_ps(itscale);
        __m128 oi[9];
        for (k = 0; k < 9; k++) oi[k] = _mm_set1_ps(opsin_inv[k]);
        for (; i + 4 <= n; i += 4) {
            __m128 vx = _mm_loadu_ps(x + i);
            __m128 vy = _mm_loadu_ps(y + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 gl = _mm_sub_ps(_mm_add_ps(vy, vx), c0);
            __m128 gm = _mm_sub_ps(_mm_sub_ps(vy, vx), c1);
            __m128 gs = _mm_sub_ps(vb, c2);
            __m128 l = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gl, gl), gl), b0);
            __m128 m = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gm, gm), gm), b1);
            __m128 s = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gs, gs), gs), b2);
            l = _mm_mul_ps(l, its);
            m = _mm_mul_ps(m, its);
            s = _mm_mul_ps(s, its);
            _mm_storeu_ps(x + i, _mm_add_ps(_mm_add_ps(
                _mm_mul_ps(oi[0], l), _mm_mul_ps(oi[1], m)), _mm_mul_ps(oi[2], s)));
            _mm_storeu_ps(y + i, _mm_add_ps(_mm_add_ps(
                _mm_mul_ps(oi[3], l), _mm_mul_ps(oi[4], m)), _mm_mul_ps(oi[5], s)));
            _mm_storeu_ps(b + i, _mm_add_ps(_mm_add_ps(
                _mm_mul_ps(oi[6], l), _mm_mul_ps(oi[7], m)), _mm_mul_ps(oi[8], s)));
        }
    }
#endif
    for (; i < n; i++) {
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

#ifdef JXL_COLOR_SSE2
/* tf_srgb four at a time. Every step is arithmetic or bit manipulation that
   SSE2 has, bar the 16-entry power table, whose index comes from the
   exponent: those four lookups are done scalar and reassembled, which is
   still far cheaper than the polynomial around them. */
static void tf_srgb_x4(float *v, size_t n, size_t *pos) {
    const __m128i signmask = _mm_set1_epi32((int)0x80000000u);
    const __m128i absmask = _mm_set1_epi32(0x7fffffff);
    const __m128i adj_or = _mm_set1_epi32(0x3e800000);
    const __m128i adj_and = _mm_set1_epi32(0x3effffff);
    const __m128 k3 = _mm_set1_ps(0.059914046f), k2 = _mm_set1_ps(-0.10889456f);
    const __m128 k1 = _mm_set1_ps(0.107963754f), k0 = _mm_set1_ps(0.018092343f);
    const __m128 c1292 = _mm_set1_ps(12.92f), c055 = _mm_set1_ps(0.055f);
    const __m128 cutoff = _mm_set1_ps(0.0031308f);
    size_t i = *pos;
    for (; i + 4 <= n; i += 4) {
        __m128i bits = _mm_castps_si128(_mm_loadu_ps(v + i));
        __m128i sign = _mm_and_si128(bits, signmask);
        __m128i vi = _mm_and_si128(bits, absmask);
        __m128 v_adj = _mm_castsi128_ps(
            _mm_and_si128(_mm_or_si128(vi, adj_or), adj_and));
        __m128 pw = _mm_add_ps(_mm_mul_ps(k3, v_adj), k2);
        __m128 fv = _mm_castsi128_ps(vi);
        __m128i idx = _mm_and_si128(_mm_sub_epi32(_mm_srli_epi32(vi, 23),
                                                  _mm_set1_epi32(118)),
                                    _mm_set1_epi32(0xf));
        int ix[4];
        __m128i mul_bits;
        __m128 small, acc, out;
        pw = _mm_add_ps(_mm_mul_ps(pw, v_adj), k1);
        pw = _mm_add_ps(_mm_mul_ps(pw, v_adj), k0);
        _mm_storeu_si128((__m128i *)ix, idx);
        mul_bits = _mm_setr_epi32(
            (int)(0x40000000u | ((uint32_t)srgb_powtable_upper[ix[0] & 0xf] << 18) |
                  ((uint32_t)srgb_powtable_lower[ix[0] & 0xf] << 10)),
            (int)(0x40000000u | ((uint32_t)srgb_powtable_upper[ix[1] & 0xf] << 18) |
                  ((uint32_t)srgb_powtable_lower[ix[1] & 0xf] << 10)),
            (int)(0x40000000u | ((uint32_t)srgb_powtable_upper[ix[2] & 0xf] << 18) |
                  ((uint32_t)srgb_powtable_lower[ix[2] & 0xf] << 10)),
            (int)(0x40000000u | ((uint32_t)srgb_powtable_upper[ix[3] & 0xf] << 18) |
                  ((uint32_t)srgb_powtable_lower[ix[3] & 0xf] << 10)));
        small = _mm_mul_ps(fv, c1292);
        acc = _mm_sub_ps(_mm_mul_ps(pw, _mm_castsi128_ps(mul_bits)), c055);
        /* fv <= cutoff ? small : acc */
        {
            __m128 m = _mm_cmple_ps(fv, cutoff);
            out = _mm_or_ps(_mm_and_ps(m, small), _mm_andnot_ps(m, acc));
        }
        _mm_storeu_ps(v + i, _mm_castsi128_ps(
            _mm_or_si128(_mm_castps_si128(out), sign)));
    }
    *pos = i;
}
#endif

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
            i = 0;
#ifdef JXL_COLOR_SSE2
            tf_srgb_x4(v, n, &i);
#endif
            for (; i < n; i++) v[i] = tf_srgb(v[i]);
            break;
    }
}

/* ----- primaries and white point -> the opsin inverse matrix ----- */

static void mat3_mul(const float a[9], const float b[9], float out[9]) {
    int i, j, k;
    float t[9];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            float s = 0.0f;
            for (k = 0; k < 3; k++) s += a[i * 3 + k] * b[k * 3 + j];
            t[i * 3 + j] = s;
        }
    }
    memcpy(out, t, sizeof(t));
}

static int mat3_inv(const float m[9], float out[9]) {
    float det;
    float a = m[0], b = m[1], c = m[2];
    float d = m[3], e = m[4], f = m[5];
    float g = m[6], h = m[7], i = m[8];
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det == 0.0f) return -1;
    out[0] = (e * i - f * h) / det;
    out[1] = (c * h - b * i) / det;
    out[2] = (b * f - c * e) / det;
    out[3] = (f * g - d * i) / det;
    out[4] = (a * i - c * g) / det;
    out[5] = (c * d - a * f) / det;
    out[6] = (d * h - e * g) / det;
    out[7] = (b * g - a * h) / det;
    out[8] = (a * e - b * d) / det;
    return 0;
}

/* RGB -> XYZ for the given primaries and white point. */
static int primaries_to_xyz(const float p[6], float wx, float wy, float out[9]) {
    float prim[9], inv[9], w[3], scale[3];
    int i;
    prim[0] = p[0]; prim[1] = p[2]; prim[2] = p[4];
    prim[3] = p[1]; prim[4] = p[3]; prim[5] = p[5];
    prim[6] = 1.0f - p[0] - p[1];
    prim[7] = 1.0f - p[2] - p[3];
    prim[8] = 1.0f - p[4] - p[5];
    if (wy == 0.0f || mat3_inv(prim, inv) != 0) return -1;
    w[0] = wx / wy;
    w[1] = 1.0f;
    w[2] = (1.0f - wx - wy) / wy;
    for (i = 0; i < 3; i++) {
        scale[i] = inv[i * 3] * w[0] + inv[i * 3 + 1] * w[1] + inv[i * 3 + 2] * w[2];
    }
    for (i = 0; i < 3; i++) {
        out[i * 3] = prim[i * 3] * scale[0];
        out[i * 3 + 1] = prim[i * 3 + 1] * scale[1];
        out[i * 3 + 2] = prim[i * 3 + 2] * scale[2];
    }
    return 0;
}

/* Bradford chromatic adaptation from the given white point to D50. */
static int adapt_to_xyz_d50(float wx, float wy, float out[9]) {
    static const float bradford[9] = {
        0.8951f, 0.2664f, -0.1614f,
        -0.7502f, 1.7135f, 0.0367f,
        0.0389f, -0.0685f, 1.0296f
    };
    static const float w50[3] = {0.96422f, 1.0f, 0.82521f};
    float w[3], lms[3], lms50[3], a[9], inv[9];
    int i;
    if (wy == 0.0f) return -1;
    w[0] = wx / wy;
    w[1] = 1.0f;
    w[2] = (1.0f - wx - wy) / wy;
    for (i = 0; i < 3; i++) {
        lms[i] = bradford[i * 3] * w[0] + bradford[i * 3 + 1] * w[1] +
                 bradford[i * 3 + 2] * w[2];
        lms50[i] = bradford[i * 3] * w50[0] + bradford[i * 3 + 1] * w50[1] +
                   bradford[i * 3 + 2] * w50[2];
        if (lms[i] == 0.0f) return -1;
    }
    memset(a, 0, sizeof(a));
    for (i = 0; i < 3; i++) a[i * 3 + i] = lms50[i] / lms[i];
    if (mat3_inv(bradford, inv) != 0) return -1;
    mat3_mul(a, bradford, out);
    mat3_mul(inv, out, out);
    return 0;
}

static void get_primaries(const jxl_colour_encoding *enc, float p[6]) {
    switch (enc->primaries) {
        case JXL_PRIMARIES_2100:
            p[0] = 0.708f; p[1] = 0.292f;
            p[2] = 0.170f; p[3] = 0.797f;
            p[4] = 0.131f; p[5] = 0.046f;
            break;
        case JXL_PRIMARIES_P3:
            p[0] = 0.680f; p[1] = 0.320f;
            p[2] = 0.265f; p[3] = 0.690f;
            p[4] = 0.150f; p[5] = 0.060f;
            break;
        case JXL_PRIMARIES_CUSTOM:
            memcpy(p, enc->prim_xy, 6 * sizeof(float));
            break;
        default:
            p[0] = 0.640f; p[1] = 0.330f;
            p[2] = 0.300f; p[3] = 0.600f;
            p[4] = 0.150f; p[5] = 0.060f;
            break;
    }
}

static void get_white_point(const jxl_colour_encoding *enc, float *wx, float *wy) {
    switch (enc->white_point) {
        case JXL_WP_E: *wx = 1.0f / 3.0f; *wy = 1.0f / 3.0f; break;
        case JXL_WP_DCI: *wx = 0.314f; *wy = 0.351f; break;
        case JXL_WP_CUSTOM: *wx = enc->white_xy[0]; *wy = enc->white_xy[1]; break;
        default: *wx = 0.3127f; *wy = 0.3290f; break;
    }
}

/* The stored opsin inverse matrix lands in linear sRGB. When the image
   declares different primaries or a different white point we fold the
   sRGB -> target conversion into it, exactly like libjxl's
   OutputEncodingInfo::SetFromMetadata. Grayscale output instead collapses the
   three rows to the target's luminance weights. */
void jxl_opsin_matrix_for(const jxl_image_metadata *meta, float out[9]) {
    const jxl_colour_encoding *enc = &meta->colour;
    float luminances[3] = {0.2126f, 0.7152f, 0.0722f};
    int is_gray = (enc->colour_space == JXLDEC_CS_GRAY);

    memcpy(out, meta->opsin_inv, 9 * sizeof(float));

    if (!is_gray &&
        (enc->primaries != JXL_PRIMARIES_SRGB || enc->white_point != JXL_WP_D65)) {
        static const float srgb_p[6] = {0.640f, 0.330f, 0.300f, 0.600f,
                                        0.150f, 0.060f};
        float srgb_to_xyzd50[9], orig_to_xyz[9], adapt[9], tmp[9];
        float p[6], wx, wy;
        if (primaries_to_xyz(srgb_p, 0.3127f, 0.3290f, tmp) != 0) return;
        if (adapt_to_xyz_d50(0.3127f, 0.3290f, adapt) != 0) return;
        mat3_mul(adapt, tmp, srgb_to_xyzd50);

        get_primaries(enc, p);
        get_white_point(enc, &wx, &wy);
        if (primaries_to_xyz(p, wx, wy, orig_to_xyz) != 0) return;
        memcpy(luminances, orig_to_xyz + 3, 3 * sizeof(float));

        if (meta->xyb_encoded) {
            float xyzd50_to_orig[9], srgb_to_orig[9];
            if (adapt_to_xyz_d50(wx, wy, adapt) != 0) return;
            mat3_mul(adapt, orig_to_xyz, tmp);
            if (mat3_inv(tmp, xyzd50_to_orig) != 0) return;
            mat3_mul(xyzd50_to_orig, srgb_to_xyzd50, srgb_to_orig);
            mat3_mul(srgb_to_orig, meta->opsin_inv, out);
        }
    }

    if (is_gray) {
        float luma[9], tmp[9];
        int i;
        for (i = 0; i < 3; i++) memcpy(luma + i * 3, luminances, 3 * sizeof(float));
        memcpy(tmp, out, sizeof(tmp));
        mat3_mul(luma, tmp, out);
    }
}
