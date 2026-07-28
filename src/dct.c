/* dct.c -- the separable DCT-II / DCT-III used by VarDCT.
 *
 * jxl scales the forward transform by 1/n with odd outputs multiplied by
 * sqrt(2); the recursion splits a length-n transform into two length-n/2 ones
 * using the "sec half" table (1 / (2 cos((2k+1)pi/2n))).
 *
 * The 2D transform runs rows then columns, matching jxl-oxide's traversal
 * order so the floating-point rounding is identical -- except for height == 2,
 * which does the vertical butterfly first.
 */
#include "jxl_internal.h"

#include <math.h>

/* SSE2 is baseline on x64. -DJXL_DCT_FORCE_SCALAR builds the scalar path
   alone, so the "bit-identical" claim can be checked by diffing the two
   builds' output rather than taken on trust. */
#if !defined(JXL_DCT_FORCE_SCALAR) &&     (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) ||      (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define JXL_DCT_SSE2 1
#include <emmintrin.h>
#include <immintrin.h>
#endif

#define JXL_SQRT2 1.4142135623730951f

/* 1 / (2 cos((2k+1)pi/(2n))) for k < n/2, tabulated for n = 4..32 and
   computed on demand for the larger sizes. */
static const float sec_half_4[2] = {0.541196100146197f, 1.3065629648763764f};
static const float sec_half_8[4] = {
    0.5097955791041592f, 0.6013448869350453f, 0.8999762231364156f,
    2.5629154477415055f
};
static const float sec_half_16[8] = {
    0.5024192861881557f, 0.5224986149396889f, 0.5669440348163577f,
    0.6468217833599901f, 0.7881546234512502f, 1.060677685990347f,
    1.7224470982383342f, 5.101148618689155f
};
static const float sec_half_32[16] = {
    0.5006029982351963f, 0.5054709598975436f, 0.5154473099226246f,
    0.5310425910897841f, 0.5531038960344445f, 0.5829349682061339f,
    0.6225041230356648f, 0.6748083414550057f, 0.7445362710022984f,
    0.8393496454155268f, 0.9725682378619608f, 1.1694399334328847f,
    1.4841646163141662f, 2.057781009953411f, 3.407608418468719f,
    10.190008123548033f
};

/* Half-tables for n = 64, 128 and 256, built on first use. */
static float sec_half_big[32 + 64 + 128];
static int sec_half_big_init;

static const float *sec_half(int n) {
    switch (n) {
        case 4: return sec_half_4;
        case 8: return sec_half_8;
        case 16: return sec_half_16;
        case 32: return sec_half_32;
        default: break;
    }
    if (!sec_half_big_init) {
        static const int sizes[3] = {64, 128, 256};
        int off = 0, i;
        for (i = 0; i < 3; i++) {
            int m = sizes[i], k;
            for (k = 0; k < m / 2; k++) {
                double theta = (2.0 * k + 1.0) / (2.0 * m) * 3.14159265358979323846;
                sec_half_big[off + k] = (float)(1.0 / cos(theta) / 2.0);
            }
            off += m / 2;
        }
        sec_half_big_init = 1;
    }
    if (n == 64) return sec_half_big;
    if (n == 128) return sec_half_big + 32;
    return sec_half_big + 32 + 64;
}

/* Precomputed for c = 0..31, b = 256. */
static const float jxl_scale_f_tbl[32] = {
    1.0000000000000000f, 0.9996047255830407f, 0.9984194528776054f,
    0.9964458326264695f, 0.9936866130906366f, 0.9901456355893141f,
    0.9858278282666936f, 0.9807391980963174f, 0.9748868211368796f,
    0.9682788310563117f, 0.9609244059440204f, 0.9528337534340876f,
    0.9440180941651672f, 0.9344896436056892f, 0.9242615922757944f,
    0.9133480844001980f, 0.9017641950288744f, 0.8895259056651056f,
    0.8766500784429904f, 0.8631544288990163f, 0.8490574973847023f,
    0.8343786191696513f, 0.8191378932865928f, 0.8033561501721485f,
    0.7870549181591013f, 0.7702563888779096f, 0.7529833816270532f,
    0.7352593067735488f, 0.7171081282466044f, 0.6985543251889097f,
    0.6796228528314652f, 0.6603391026591464f
};

float jxl_scale_f(int c, int logb) {
    return jxl_scale_f_tbl[(c << logb) & 31];
}

static void dct4(const float in[4], float out[4], int inverse) {
    const float sec0 = 0.5411961f, sec1 = 1.306563f;
    if (!inverse) {
        float sum03 = in[0] + in[3];
        float sum12 = in[1] + in[2];
        float tmp0 = (in[0] - in[3]) * sec0;
        float tmp1 = (in[1] - in[2]) * sec1;
        float out0 = (tmp0 + tmp1) / 4.0f;
        float out1 = (tmp0 - tmp1) / 4.0f;
        out[0] = (sum03 + sum12) / 4.0f;
        out[1] = out0 * JXL_SQRT2 + out1;
        out[2] = (sum03 - sum12) / 4.0f;
        out[3] = out1;
    } else {
        float tmp0 = in[1] * JXL_SQRT2;
        float tmp1 = in[1] + in[3];
        float out0 = (tmp0 + tmp1) * sec0;
        float out1 = (tmp0 - tmp1) * sec1;
        float sum02 = in[0] + in[2];
        float sub02 = in[0] - in[2];
        out[0] = sum02 + out0;
        out[1] = sub02 + out1;
        out[2] = sub02 - out1;
        out[3] = sum02 - out0;
    }
}

/* In-place 1D DCT over n samples; scratch must hold n floats. */
static void dct_1d(float *io, int n, float *scratch, int inverse) {
    int i;
    if (n <= 1) return;
    if (n == 2) {
        float t0 = io[0] + io[1];
        float t1 = io[0] - io[1];
        if (!inverse) { io[0] = t0 / 2.0f; io[1] = t1 / 2.0f; }
        else { io[0] = t0; io[1] = t1; }
        return;
    }
    if (n == 4) {
        float in[4], out[4];
        memcpy(in, io, sizeof(in));
        dct4(in, out, inverse);
        memcpy(io, out, sizeof(out));
        return;
    }
    if (n == 8) {
        const float *sec = sec_half_8;
        float in0[4], in1[4], out0[4], out1[4];
        if (!inverse) {
            for (i = 0; i < 4; i++) {
                in0[i] = (io[i] + io[7 - i]) / 2.0f;
                in1[i] = (io[i] - io[7 - i]) * sec[i] / 2.0f;
            }
            dct4(in0, out0, 0);
            dct4(in1, out1, 0);
            for (i = 0; i < 4; i++) io[i * 2] = out0[i];
            out1[0] *= JXL_SQRT2;
            for (i = 0; i < 3; i++) io[i * 2 + 1] = out1[i] + out1[i + 1];
            io[7] = out1[3];
        } else {
            in0[0] = io[0]; in0[1] = io[2]; in0[2] = io[4]; in0[3] = io[6];
            in1[0] = io[1] * JXL_SQRT2;
            in1[1] = io[3] + io[1];
            in1[2] = io[5] + io[3];
            in1[3] = io[7] + io[5];
            dct4(in0, out0, 1);
            dct4(in1, out1, 1);
            for (i = 0; i < 4; i++) {
                float r = out1[i] * sec[i];
                io[i] = out0[i] + r;
                io[7 - i] = out0[i] - r;
            }
        }
        return;
    }

    {
        const float *sec = sec_half(n);
        int half = n / 2;
        float *in0 = scratch;
        float *in1 = scratch + half;
        if (!inverse) {
            for (i = 0; i < half; i++) {
                in0[i] = (io[i] + io[n - i - 1]) / 2.0f;
                in1[i] = (io[i] - io[n - i - 1]) / 2.0f;
            }
            for (i = 0; i < half; i++) in1[i] *= sec[i];
            dct_1d(in0, half, io, 0);
            dct_1d(in1, half, io + half, 0);
            in1[0] *= JXL_SQRT2;
            for (i = 0; i < half - 1; i++) in1[i] += in1[i + 1];
            for (i = 0; i < half; i++) io[i * 2] = in0[i];
            for (i = 0; i < half; i++) io[i * 2 + 1] = in1[i];
        } else {
            for (i = 0; i < half; i++) {
                in0[i] = io[i * 2];
                in1[i] = io[i * 2 + 1];
            }
            for (i = 1; i < half; i++) in1[half - i] += in1[half - i - 1];
            in1[0] *= JXL_SQRT2;
            dct_1d(in0, half, io, 1);
            dct_1d(in1, half, io + half, 1);
            for (i = 0; i < half; i++) in1[i] *= sec[i];
            for (i = 0; i < half; i++) {
                float a = scratch[i];
                float b = scratch[i + half];
                io[i] = a + b;
                io[n - i - 1] = a - b;
            }
        }
    }
}

/* 2D DCT over a w x h block with the given row stride (in floats). */
#ifdef JXL_DCT_SSE2
/* Four columns at a time, one per lane. The column pass already had to gather
   each column out of the row-major block, and four consecutive floats from a
   row are exactly the four columns' values at that row -- so the gather is the
   transpose, and no shuffling is needed.
 *
 * These two mirror dct4 and dct_1d expression for expression, including how
 * the additions associate: every lane then performs the same operations in the
 * same order as the scalar code, so the result is bit-identical. Divisions by
 * 2 and 4 become multiplies by 0.5 and 0.25, which is exact for powers of two.
 */
static void dct4_v4(const __m128 in[4], __m128 out[4], int inverse) {
    const __m128 sec0 = _mm_set1_ps(0.5411961f);
    const __m128 sec1 = _mm_set1_ps(1.306563f);
    const __m128 sqrt2 = _mm_set1_ps(JXL_SQRT2);
    const __m128 quarter = _mm_set1_ps(0.25f);
    if (!inverse) {
        __m128 sum03 = _mm_add_ps(in[0], in[3]);
        __m128 sum12 = _mm_add_ps(in[1], in[2]);
        __m128 tmp0 = _mm_mul_ps(_mm_sub_ps(in[0], in[3]), sec0);
        __m128 tmp1 = _mm_mul_ps(_mm_sub_ps(in[1], in[2]), sec1);
        __m128 out0 = _mm_mul_ps(_mm_add_ps(tmp0, tmp1), quarter);
        __m128 out1 = _mm_mul_ps(_mm_sub_ps(tmp0, tmp1), quarter);
        out[0] = _mm_mul_ps(_mm_add_ps(sum03, sum12), quarter);
        out[1] = _mm_add_ps(_mm_mul_ps(out0, sqrt2), out1);
        out[2] = _mm_mul_ps(_mm_sub_ps(sum03, sum12), quarter);
        out[3] = out1;
    } else {
        __m128 tmp0 = _mm_mul_ps(in[1], sqrt2);
        __m128 tmp1 = _mm_add_ps(in[1], in[3]);
        __m128 out0 = _mm_mul_ps(_mm_add_ps(tmp0, tmp1), sec0);
        __m128 out1 = _mm_mul_ps(_mm_sub_ps(tmp0, tmp1), sec1);
        __m128 sum02 = _mm_add_ps(in[0], in[2]);
        __m128 sub02 = _mm_sub_ps(in[0], in[2]);
        out[0] = _mm_add_ps(sum02, out0);
        out[1] = _mm_add_ps(sub02, out1);
        out[2] = _mm_sub_ps(sub02, out1);
        out[3] = _mm_sub_ps(sum02, out0);
    }
}

static void dct_1d_v4(__m128 *io, int n, __m128 *scratch, int inverse) {
    const __m128 sqrt2 = _mm_set1_ps(JXL_SQRT2);
    const __m128 half_v = _mm_set1_ps(0.5f);
    int i;
    if (n <= 1) return;
    if (n == 2) {
        __m128 t0 = _mm_add_ps(io[0], io[1]);
        __m128 t1 = _mm_sub_ps(io[0], io[1]);
        if (!inverse) { io[0] = _mm_mul_ps(t0, half_v); io[1] = _mm_mul_ps(t1, half_v); }
        else { io[0] = t0; io[1] = t1; }
        return;
    }
    if (n == 4) {
        __m128 in[4], out[4];
        for (i = 0; i < 4; i++) in[i] = io[i];
        dct4_v4(in, out, inverse);
        for (i = 0; i < 4; i++) io[i] = out[i];
        return;
    }
    if (n == 8) {
        __m128 in0[4], in1[4], out0[4], out1[4];
        if (!inverse) {
            for (i = 0; i < 4; i++) {
                in0[i] = _mm_mul_ps(_mm_add_ps(io[i], io[7 - i]), half_v);
                in1[i] = _mm_mul_ps(
                    _mm_mul_ps(_mm_sub_ps(io[i], io[7 - i]),
                               _mm_set1_ps(sec_half_8[i])),
                    half_v);
            }
            dct4_v4(in0, out0, 0);
            dct4_v4(in1, out1, 0);
            for (i = 0; i < 4; i++) io[i * 2] = out0[i];
            out1[0] = _mm_mul_ps(out1[0], sqrt2);
            for (i = 0; i < 3; i++)
                io[i * 2 + 1] = _mm_add_ps(out1[i], out1[i + 1]);
            io[7] = out1[3];
        } else {
            in0[0] = io[0]; in0[1] = io[2];
            in0[2] = io[4]; in0[3] = io[6];
            in1[0] = _mm_mul_ps(io[1], sqrt2);
            in1[1] = _mm_add_ps(io[3], io[1]);
            in1[2] = _mm_add_ps(io[5], io[3]);
            in1[3] = _mm_add_ps(io[7], io[5]);
            dct4_v4(in0, out0, 1);
            dct4_v4(in1, out1, 1);
            for (i = 0; i < 4; i++) {
                __m128 r = _mm_mul_ps(out1[i], _mm_set1_ps(sec_half_8[i]));
                io[i] = _mm_add_ps(out0[i], r);
                io[7 - i] = _mm_sub_ps(out0[i], r);
            }
        }
        return;
    }
    {
        const float *sec = sec_half(n);
        int hn = n / 2;
        __m128 *in0 = scratch;
        __m128 *in1 = scratch + hn;
        if (!inverse) {
            for (i = 0; i < hn; i++) {
                in0[i] = _mm_mul_ps(_mm_add_ps(io[i], io[n - i - 1]), half_v);
                in1[i] = _mm_mul_ps(_mm_sub_ps(io[i], io[n - i - 1]), half_v);
            }
            for (i = 0; i < hn; i++)
                in1[i] = _mm_mul_ps(in1[i], _mm_set1_ps(sec[i]));
            dct_1d_v4(in0, hn, io, 0);
            dct_1d_v4(in1, hn, io + hn, 0);
            in1[0] = _mm_mul_ps(in1[0], sqrt2);
            for (i = 0; i < hn - 1; i++) in1[i] = _mm_add_ps(in1[i], in1[i + 1]);
            for (i = 0; i < hn; i++) io[i * 2] = in0[i];
            for (i = 0; i < hn; i++) io[i * 2 + 1] = in1[i];
        } else {
            for (i = 0; i < hn; i++) {
                in0[i] = io[i * 2];
                in1[i] = io[i * 2 + 1];
            }
            for (i = 1; i < hn; i++)
                in1[hn - i] = _mm_add_ps(in1[hn - i], in1[hn - i - 1]);
            in1[0] = _mm_mul_ps(in1[0], sqrt2);
            dct_1d_v4(in0, hn, io, 1);
            dct_1d_v4(in1, hn, io + hn, 1);
            for (i = 0; i < hn; i++)
                in1[i] = _mm_mul_ps(in1[i], _mm_set1_ps(sec[i]));
            for (i = 0; i < hn; i++) {
                __m128 a = scratch[i];
                __m128 b = scratch[i + hn];
                io[i] = _mm_add_ps(a, b);
                io[n - i - 1] = _mm_sub_ps(a, b);
            }
        }
    }
}
/* Eight columns at a time, the same kernel one lane wider. Worth having only
   for the column pass: that pass gathers whole rows, so widening it is a
   wider load and nothing else, where the row pass would need an 8x8 transpose
   in and out. On the 8x8 blocks that dominate a VarDCT frame this turns the
   two column passes into one.

   No FMA: fusing would round once where the four-lane and scalar versions
   round twice, and these three are meant to agree with them bit for bit. */
JXL_TARGET_AVX2
static void dct4_v8(const __m256 in[4], __m256 out[4], int inverse) {
    const __m256 sec0 = _mm256_set1_ps(0.5411961f);
    const __m256 sec1 = _mm256_set1_ps(1.306563f);
    const __m256 sqrt2 = _mm256_set1_ps(JXL_SQRT2);
    const __m256 quarter = _mm256_set1_ps(0.25f);
    if (!inverse) {
        __m256 sum03 = _mm256_add_ps(in[0], in[3]);
        __m256 sum12 = _mm256_add_ps(in[1], in[2]);
        __m256 tmp0 = _mm256_mul_ps(_mm256_sub_ps(in[0], in[3]), sec0);
        __m256 tmp1 = _mm256_mul_ps(_mm256_sub_ps(in[1], in[2]), sec1);
        __m256 out0 = _mm256_mul_ps(_mm256_add_ps(tmp0, tmp1), quarter);
        __m256 out1 = _mm256_mul_ps(_mm256_sub_ps(tmp0, tmp1), quarter);
        out[0] = _mm256_mul_ps(_mm256_add_ps(sum03, sum12), quarter);
        out[1] = _mm256_add_ps(_mm256_mul_ps(out0, sqrt2), out1);
        out[2] = _mm256_mul_ps(_mm256_sub_ps(sum03, sum12), quarter);
        out[3] = out1;
    } else {
        __m256 tmp0 = _mm256_mul_ps(in[1], sqrt2);
        __m256 tmp1 = _mm256_add_ps(in[1], in[3]);
        __m256 out0 = _mm256_mul_ps(_mm256_add_ps(tmp0, tmp1), sec0);
        __m256 out1 = _mm256_mul_ps(_mm256_sub_ps(tmp0, tmp1), sec1);
        __m256 sum02 = _mm256_add_ps(in[0], in[2]);
        __m256 sub02 = _mm256_sub_ps(in[0], in[2]);
        out[0] = _mm256_add_ps(sum02, out0);
        out[1] = _mm256_add_ps(sub02, out1);
        out[2] = _mm256_sub_ps(sub02, out1);
        out[3] = _mm256_sub_ps(sum02, out0);
    }
}

/* Fixed eight-point leaf shared by direct 8-point transforms and the
   16-point specialization below. Keeping it separate from the runtime-sized
   recursion lets the compiler inline the leaf without a second dispatch
   through dct_1d_v8. */
JXL_TARGET_AVX2
static JXL_INLINE_HINT void dct8_v8(__m256 *io, int inverse) {
    const __m256 sqrt2 = _mm256_set1_ps(JXL_SQRT2);
    const __m256 half_v = _mm256_set1_ps(0.5f);
    __m256 in0[4], in1[4], out0[4], out1[4];
    int i;

    if (!inverse) {
        for (i = 0; i < 4; i++) {
            in0[i] = _mm256_mul_ps(
                _mm256_add_ps(io[i], io[7 - i]), half_v);
            in1[i] = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_sub_ps(io[i], io[7 - i]),
                              _mm256_set1_ps(sec_half_8[i])),
                half_v);
        }
        dct4_v8(in0, out0, 0);
        dct4_v8(in1, out1, 0);
        for (i = 0; i < 4; i++) io[i * 2] = out0[i];
        out1[0] = _mm256_mul_ps(out1[0], sqrt2);
        for (i = 0; i < 3; i++)
            io[i * 2 + 1] = _mm256_add_ps(out1[i], out1[i + 1]);
        io[7] = out1[3];
    } else {
        in0[0] = io[0]; in0[1] = io[2];
        in0[2] = io[4]; in0[3] = io[6];
        in1[0] = _mm256_mul_ps(io[1], sqrt2);
        in1[1] = _mm256_add_ps(io[3], io[1]);
        in1[2] = _mm256_add_ps(io[5], io[3]);
        in1[3] = _mm256_add_ps(io[7], io[5]);
        dct4_v8(in0, out0, 1);
        dct4_v8(in1, out1, 1);
        for (i = 0; i < 4; i++) {
            __m256 r = _mm256_mul_ps(
                out1[i], _mm256_set1_ps(sec_half_8[i]));
            io[i] = _mm256_add_ps(out0[i], r);
            io[7 - i] = _mm256_sub_ps(out0[i], r);
        }
    }
}

/* libjxl's common 8-point inverse kernel. This is deliberately the only DCT
   path that opts into FMA: the fused multiply-adds reduce both the instruction
   count and the dependency depth, while the separate target attribute keeps
   the scalar/SSE2 and all other AVX2 arithmetic unchanged. */
JXL_TARGET_AVX2_FMA
static JXL_INLINE_HINT void idct4_v8_fma(__m256 *io) {
    const __m256 sqrt2 = _mm256_set1_ps(JXL_SQRT2);
    const __m256 m0 = _mm256_set1_ps(sec_half_4[0]);
    const __m256 m1 = _mm256_set1_ps(sec_half_4[1]);
    __m256 e0 = _mm256_add_ps(io[0], io[2]);
    __m256 e1 = _mm256_sub_ps(io[0], io[2]);
    __m256 o0 = _mm256_mul_ps(io[1], sqrt2);
    __m256 o1 = _mm256_add_ps(io[3], io[1]);
    __m256 b0 = _mm256_add_ps(o0, o1);
    __m256 b1 = _mm256_sub_ps(o0, o1);
    io[0] = _mm256_fmadd_ps(m0, b0, e0);
    io[3] = _mm256_fnmadd_ps(m0, b0, e0);
    io[1] = _mm256_fmadd_ps(m1, b1, e1);
    io[2] = _mm256_fnmadd_ps(m1, b1, e1);
}

JXL_TARGET_AVX2_FMA
static JXL_INLINE_HINT void idct8_v8_fma(__m256 *io) {
    const __m256 sqrt2 = _mm256_set1_ps(JXL_SQRT2);
    const __m256 m0 = _mm256_set1_ps(sec_half_8[0]);
    const __m256 m1 = _mm256_set1_ps(sec_half_8[1]);
    const __m256 m2 = _mm256_set1_ps(sec_half_8[2]);
    const __m256 m3 = _mm256_set1_ps(sec_half_8[3]);
    __m256 even[4], odd[4];

    even[0] = io[0]; even[1] = io[2];
    even[2] = io[4]; even[3] = io[6];
    odd[0] = _mm256_mul_ps(io[1], sqrt2);
    odd[1] = _mm256_add_ps(io[3], io[1]);
    odd[2] = _mm256_add_ps(io[5], io[3]);
    odd[3] = _mm256_add_ps(io[7], io[5]);
    idct4_v8_fma(even);
    idct4_v8_fma(odd);
    io[0] = _mm256_fmadd_ps(m0, odd[0], even[0]);
    io[7] = _mm256_fnmadd_ps(m0, odd[0], even[0]);
    io[1] = _mm256_fmadd_ps(m1, odd[1], even[1]);
    io[6] = _mm256_fnmadd_ps(m1, odd[1], even[1]);
    io[2] = _mm256_fmadd_ps(m2, odd[2], even[2]);
    io[5] = _mm256_fnmadd_ps(m2, odd[2], even[2]);
    io[3] = _mm256_fmadd_ps(m3, odd[3], even[3]);
    io[4] = _mm256_fnmadd_ps(m3, odd[3], even[3]);
}

JXL_TARGET_AVX2
static void dct_1d_v8(__m256 *io, int n, __m256 *scratch, int inverse) {
    const __m256 sqrt2 = _mm256_set1_ps(JXL_SQRT2);
    const __m256 half_v = _mm256_set1_ps(0.5f);
    int i;
    if (n <= 1) return;
    if (n == 2) {
        __m256 t0 = _mm256_add_ps(io[0], io[1]);
        __m256 t1 = _mm256_sub_ps(io[0], io[1]);
        if (!inverse) {
            io[0] = _mm256_mul_ps(t0, half_v);
            io[1] = _mm256_mul_ps(t1, half_v);
        } else { io[0] = t0; io[1] = t1; }
        return;
    }
    if (n == 4) {
        __m256 in[4], out[4];
        for (i = 0; i < 4; i++) in[i] = io[i];
        dct4_v8(in, out, inverse);
        for (i = 0; i < 4; i++) io[i] = out[i];
        return;
    }
    if (n == 8) {
        dct8_v8(io, inverse);
        return;
    }
    if (n == 16) {
        __m256 *in0 = scratch;
        __m256 *in1 = scratch + 8;
        if (!inverse) {
            for (i = 0; i < 8; i++) {
                in0[i] = _mm256_mul_ps(
                    _mm256_add_ps(io[i], io[15 - i]), half_v);
                in1[i] = _mm256_mul_ps(
                    _mm256_sub_ps(io[i], io[15 - i]), half_v);
            }
            for (i = 0; i < 8; i++)
                in1[i] = _mm256_mul_ps(
                    in1[i], _mm256_set1_ps(sec_half_16[i]));
            dct8_v8(in0, 0);
            dct8_v8(in1, 0);
            in1[0] = _mm256_mul_ps(in1[0], sqrt2);
            for (i = 0; i < 7; i++)
                in1[i] = _mm256_add_ps(in1[i], in1[i + 1]);
            for (i = 0; i < 8; i++) io[i * 2] = in0[i];
            for (i = 0; i < 8; i++) io[i * 2 + 1] = in1[i];
        } else {
            for (i = 0; i < 8; i++) {
                in0[i] = io[i * 2];
                in1[i] = io[i * 2 + 1];
            }
            for (i = 1; i < 8; i++)
                in1[8 - i] = _mm256_add_ps(in1[8 - i], in1[7 - i]);
            in1[0] = _mm256_mul_ps(in1[0], sqrt2);
            dct8_v8(in0, 1);
            dct8_v8(in1, 1);
            for (i = 0; i < 8; i++)
                in1[i] = _mm256_mul_ps(
                    in1[i], _mm256_set1_ps(sec_half_16[i]));
            for (i = 0; i < 8; i++) {
                __m256 a = scratch[i];
                __m256 b = scratch[i + 8];
                io[i] = _mm256_add_ps(a, b);
                io[15 - i] = _mm256_sub_ps(a, b);
            }
        }
        return;
    }
    {
        const float *sec = sec_half(n);
        int hn = n / 2;
        __m256 *in0 = scratch;
        __m256 *in1 = scratch + hn;
        if (!inverse) {
            for (i = 0; i < hn; i++) {
                in0[i] = _mm256_mul_ps(_mm256_add_ps(io[i], io[n - i - 1]),
                                       half_v);
                in1[i] = _mm256_mul_ps(_mm256_sub_ps(io[i], io[n - i - 1]),
                                       half_v);
            }
            for (i = 0; i < hn; i++)
                in1[i] = _mm256_mul_ps(in1[i], _mm256_set1_ps(sec[i]));
            dct_1d_v8(in0, hn, io, 0);
            dct_1d_v8(in1, hn, io + hn, 0);
            in1[0] = _mm256_mul_ps(in1[0], sqrt2);
            for (i = 0; i < hn - 1; i++)
                in1[i] = _mm256_add_ps(in1[i], in1[i + 1]);
            for (i = 0; i < hn; i++) io[i * 2] = in0[i];
            for (i = 0; i < hn; i++) io[i * 2 + 1] = in1[i];
        } else {
            for (i = 0; i < hn; i++) {
                in0[i] = io[i * 2];
                in1[i] = io[i * 2 + 1];
            }
            for (i = 1; i < hn; i++)
                in1[hn - i] = _mm256_add_ps(in1[hn - i], in1[hn - i - 1]);
            in1[0] = _mm256_mul_ps(in1[0], sqrt2);
            dct_1d_v8(in0, hn, io, 1);
            dct_1d_v8(in1, hn, io + hn, 1);
            for (i = 0; i < hn; i++)
                in1[i] = _mm256_mul_ps(in1[i], _mm256_set1_ps(sec[i]));
            for (i = 0; i < hn; i++) {
                __m256 a = scratch[i];
                __m256 b = scratch[i + hn];
                io[i] = _mm256_add_ps(a, b);
                io[n - i - 1] = _mm256_sub_ps(a, b);
            }
        }
    }
}

/* The column pass, eight wide. Returns the first column not yet done. */
JXL_TARGET_AVX2
static int dct_cols8(float *data, size_t stride, int w, int h, int inverse) {
    __m256 vcol[256], vscratch[256];
    int x = 0, y;
    for (; x + 8 <= w; x += 8) {
        for (y = 0; y < h; y++)
            vcol[y] = _mm256_loadu_ps(data + (size_t)y * stride + x);
        dct_1d_v8(vcol, h, vscratch, inverse);
        for (y = 0; y < h; y++)
            _mm256_storeu_ps(data + (size_t)y * stride + x, vcol[y]);
    }
    _mm256_zeroupper();
    return x;
}

JXL_TARGET_AVX2
static void transpose8_ps(__m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3,
                          __m256 *r4, __m256 *r5, __m256 *r6, __m256 *r7) {
    __m256 t0 = _mm256_unpacklo_ps(*r0, *r1);
    __m256 t1 = _mm256_unpackhi_ps(*r0, *r1);
    __m256 t2 = _mm256_unpacklo_ps(*r2, *r3);
    __m256 t3 = _mm256_unpackhi_ps(*r2, *r3);
    __m256 t4 = _mm256_unpacklo_ps(*r4, *r5);
    __m256 t5 = _mm256_unpackhi_ps(*r4, *r5);
    __m256 t6 = _mm256_unpacklo_ps(*r6, *r7);
    __m256 t7 = _mm256_unpackhi_ps(*r6, *r7);
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xee);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xee);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xee);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xee);
    *r0 = _mm256_permute2f128_ps(s0, s4, 0x20);
    *r1 = _mm256_permute2f128_ps(s1, s5, 0x20);
    *r2 = _mm256_permute2f128_ps(s2, s6, 0x20);
    *r3 = _mm256_permute2f128_ps(s3, s7, 0x20);
    *r4 = _mm256_permute2f128_ps(s0, s4, 0x31);
    *r5 = _mm256_permute2f128_ps(s1, s5, 0x31);
    *r6 = _mm256_permute2f128_ps(s2, s6, 0x31);
    *r7 = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/* The overwhelmingly common 8x8 transform keeps the row-pass result in
   registers for the column pass. This is expression-for-expression the same
   row-then-column sequence as dct_rows8 followed by dct_cols8; it merely
   removes their intermediate 64-float store and reload. */
JXL_TARGET_AVX2
static void dct_8x8(float *data, size_t stride, int inverse) {
    __m256 r[8];
    int y;
    for (y = 0; y < 8; y++) {
        r[y] = _mm256_loadu_ps(data + (size_t)y * stride);
    }
    transpose8_ps(&r[0], &r[1], &r[2], &r[3],
                  &r[4], &r[5], &r[6], &r[7]);
    dct8_v8(r, inverse);
    transpose8_ps(&r[0], &r[1], &r[2], &r[3],
                  &r[4], &r[5], &r[6], &r[7]);
    dct8_v8(r, inverse);
    for (y = 0; y < 8; y++) {
        _mm256_storeu_ps(data + (size_t)y * stride, r[y]);
    }
    _mm256_zeroupper();
}

JXL_TARGET_AVX2_FMA
static JXL_INLINE_HINT void idct_8x8_fma_core(float *data, size_t stride) {
    __m256 r[8];
    int y;
    for (y = 0; y < 8; y++) {
        r[y] = _mm256_loadu_ps(data + (size_t)y * stride);
    }
    transpose8_ps(&r[0], &r[1], &r[2], &r[3],
                  &r[4], &r[5], &r[6], &r[7]);
    idct8_v8_fma(r);
    transpose8_ps(&r[0], &r[1], &r[2], &r[3],
                  &r[4], &r[5], &r[6], &r[7]);
    idct8_v8_fma(r);
    for (y = 0; y < 8; y++) {
        _mm256_storeu_ps(data + (size_t)y * stride, r[y]);
    }
}

JXL_TARGET_AVX2_FMA
static void idct_8x8_fma(float *data, size_t stride) {
    idct_8x8_fma_core(data, stride);
    _mm256_zeroupper();
}

/* Amortize the Windows x64 nonvolatile-SIMD prologue across a whole plane.
   A standalone 8x8 entry saves ten XMM registers and builds an aligned stack
   frame; doing that once per block was a substantial fraction of IDCT time. */
JXL_TARGET_AVX2_FMA
void jxl_idct8x8_plane(float *data, size_t stride,
                       uint32_t blocks_w, uint32_t blocks_h) {
    uint32_t bx, by;
    for (by = 0; by < blocks_h; by++) {
        for (bx = 0; bx < blocks_w; bx++) {
            idct_8x8_fma_core(
                data + (size_t)(by * 8) * stride + bx * 8, stride);
        }
    }
    _mm256_zeroupper();
}

/* Eight rows at once. The transpose keeps each row in one SIMD lane while
   the 1-D kernel runs, then restores the ordinary row-major layout. */
JXL_TARGET_AVX2
static void dct_rows8(float *data, size_t stride, int w, int inverse) {
    __m256 vrow[256], vscratch[256];
    int j;
    for (j = 0; j + 8 <= w; j += 8) {
        __m256 r0 = _mm256_loadu_ps(data + j);
        __m256 r1 = _mm256_loadu_ps(data + stride + j);
        __m256 r2 = _mm256_loadu_ps(data + 2 * stride + j);
        __m256 r3 = _mm256_loadu_ps(data + 3 * stride + j);
        __m256 r4 = _mm256_loadu_ps(data + 4 * stride + j);
        __m256 r5 = _mm256_loadu_ps(data + 5 * stride + j);
        __m256 r6 = _mm256_loadu_ps(data + 6 * stride + j);
        __m256 r7 = _mm256_loadu_ps(data + 7 * stride + j);
        transpose8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        vrow[j] = r0; vrow[j + 1] = r1; vrow[j + 2] = r2; vrow[j + 3] = r3;
        vrow[j + 4] = r4; vrow[j + 5] = r5; vrow[j + 6] = r6; vrow[j + 7] = r7;
    }
    for (; j < w; j++) {
        vrow[j] = _mm256_setr_ps(data[j], data[stride + j],
                                 data[2 * stride + j], data[3 * stride + j],
                                 data[4 * stride + j], data[5 * stride + j],
                                 data[6 * stride + j], data[7 * stride + j]);
    }
    dct_1d_v8(vrow, w, vscratch, inverse);
    for (j = 0; j + 8 <= w; j += 8) {
        __m256 r0 = vrow[j], r1 = vrow[j + 1];
        __m256 r2 = vrow[j + 2], r3 = vrow[j + 3];
        __m256 r4 = vrow[j + 4], r5 = vrow[j + 5];
        __m256 r6 = vrow[j + 6], r7 = vrow[j + 7];
        transpose8_ps(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        _mm256_storeu_ps(data + j, r0);
        _mm256_storeu_ps(data + stride + j, r1);
        _mm256_storeu_ps(data + 2 * stride + j, r2);
        _mm256_storeu_ps(data + 3 * stride + j, r3);
        _mm256_storeu_ps(data + 4 * stride + j, r4);
        _mm256_storeu_ps(data + 5 * stride + j, r5);
        _mm256_storeu_ps(data + 6 * stride + j, r6);
        _mm256_storeu_ps(data + 7 * stride + j, r7);
    }
    for (; j < w; j++) {
        JXL_ALIGN32 float t[8];
        _mm256_store_ps(t, vrow[j]);
        data[j] = t[0];
        data[stride + j] = t[1];
        data[2 * stride + j] = t[2];
        data[3 * stride + j] = t[3];
        data[4 * stride + j] = t[4];
        data[5 * stride + j] = t[5];
        data[6 * stride + j] = t[6];
        data[7 * stride + j] = t[7];
    }
    _mm256_zeroupper();
}

/* Four *rows* at once. Rows are contiguous, so unlike the column pass this
   needs a real transpose in and out -- but the transform in between is the
   same 4-lane kernel, and each lane still sees its own row's values in the
   original order, so the result stays bit-identical. */
static void dct_rows4(float *data, size_t stride, int w, int inverse) {
    __m128 vrow[256], vscratch[256];
    int j;
    for (j = 0; j + 4 <= w; j += 4) {
        __m128 r0 = _mm_loadu_ps(data + j);
        __m128 r1 = _mm_loadu_ps(data + stride + j);
        __m128 r2 = _mm_loadu_ps(data + 2 * stride + j);
        __m128 r3 = _mm_loadu_ps(data + 3 * stride + j);
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
        vrow[j] = r0; vrow[j + 1] = r1; vrow[j + 2] = r2; vrow[j + 3] = r3;
    }
    for (; j < w; j++) {
        vrow[j] = _mm_setr_ps(data[j], data[stride + j],
                              data[2 * stride + j], data[3 * stride + j]);
    }
    dct_1d_v4(vrow, w, vscratch, inverse);
    for (j = 0; j + 4 <= w; j += 4) {
        __m128 r0 = vrow[j], r1 = vrow[j + 1], r2 = vrow[j + 2], r3 = vrow[j + 3];
        _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
        _mm_storeu_ps(data + j, r0);
        _mm_storeu_ps(data + stride + j, r1);
        _mm_storeu_ps(data + 2 * stride + j, r2);
        _mm_storeu_ps(data + 3 * stride + j, r3);
    }
    for (; j < w; j++) {
        float t[4];
        _mm_storeu_ps(t, vrow[j]);
        data[j] = t[0];
        data[stride + j] = t[1];
        data[2 * stride + j] = t[2];
        data[3 * stride + j] = t[3];
    }
}
#else
/* Declared for the batched VarDCT path in decode.c. That path is gated on
   jxl_has_avx2_fma(), which is always 0 without x86, so this is never called. */
void jxl_idct8x8_plane(float *data, size_t stride,
                       uint32_t blocks_w, uint32_t blocks_h) {
    (void)data;
    (void)stride;
    (void)blocks_w;
    (void)blocks_h;
}
#endif /* JXL_DCT_SSE2 */

void jxl_dct_2d(float *data, size_t stride, int w, int h, int inverse) {
    float mul = inverse ? 1.0f : 0.5f;
    float scratch[256];
    float col[256];
    int x, y;
#ifdef JXL_DCT_SSE2
    int use_avx2;
#endif

    if (w * h <= 1) return;

    if (w == 2 && h == 1) {
        float v0 = data[0], v1 = data[1];
        data[0] = (v0 + v1) * mul;
        data[1] = (v0 - v1) * mul;
        return;
    }
    if (w == 1 && h == 2) {
        float v0 = data[0], v1 = data[stride];
        data[0] = (v0 + v1) * mul;
        data[stride] = (v0 - v1) * mul;
        return;
    }
    if (w == 2 && h == 2) {
        float v00 = data[0], v01 = data[1];
        float v10 = data[stride], v11 = data[stride + 1];
        data[0] = (v00 + v01 + v10 + v11) * mul * mul;
        data[1] = (v00 - v01 + v10 - v11) * mul * mul;
        data[stride] = (v00 + v01 - v10 - v11) * mul * mul;
        data[stride + 1] = (v00 - v01 - v10 + v11) * mul * mul;
        return;
    }

    if (h == 1) {
        dct_1d(data, w, scratch, inverse);
        return;
    }
    if (w == 1) {
        for (y = 0; y < h; y++) col[y] = data[(size_t)y * stride];
        dct_1d(col, h, scratch, inverse);
        for (y = 0; y < h; y++) data[(size_t)y * stride] = col[y];
        return;
    }

    if (h == 2) {
        /* Vertical butterfly first, then the two rows. */
        float *row0 = data;
        float *row1 = data + stride;
        for (x = 0; x < w; x++) {
            float v0 = row0[x], v1 = row1[x];
            row0[x] = (v0 + v1) * mul;
            row1[x] = (v0 - v1) * mul;
        }
        dct_1d(row0, w, scratch, inverse);
        dct_1d(row1, w, scratch, inverse);
        return;
    }

#ifdef JXL_DCT_SSE2
    if (w == 8 && h == 8 && jxl_has_avx2()) {
        if (inverse && jxl_has_avx2_fma()) idct_8x8_fma(data, stride);
        else dct_8x8(data, stride, inverse);
        return;
    }
    use_avx2 = jxl_has_avx2();
#endif
    y = 0;
#ifdef JXL_DCT_SSE2
    if (use_avx2) {
        for (; y + 8 <= h; y += 8)
            dct_rows8(data + (size_t)y * stride, stride, w, inverse);
    }
    for (; y + 4 <= h; y += 4)
        dct_rows4(data + (size_t)y * stride, stride, w, inverse);
#endif
    for (; y < h; y++) dct_1d(data + (size_t)y * stride, w, scratch, inverse);
    x = 0;
#ifdef JXL_DCT_SSE2
    if (use_avx2) x = dct_cols8(data, stride, w, h, inverse);
    {
        __m128 vcol[256], vscratch[256];
        for (; x + 4 <= w; x += 4) {
            for (y = 0; y < h; y++)
                vcol[y] = _mm_loadu_ps(data + (size_t)y * stride + x);
            dct_1d_v4(vcol, h, vscratch, inverse);
            for (y = 0; y < h; y++)
                _mm_storeu_ps(data + (size_t)y * stride + x, vcol[y]);
        }
    }
#endif
    for (; x < w; x++) {
        for (y = 0; y < h; y++) col[y] = data[(size_t)y * stride + x];
        dct_1d(col, h, scratch, inverse);
        for (y = 0; y < h; y++) data[(size_t)y * stride + x] = col[y];
    }
}
