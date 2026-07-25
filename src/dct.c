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
void jxl_dct_2d(float *data, size_t stride, int w, int h, int inverse) {
    float mul = inverse ? 1.0f : 0.5f;
    float scratch[256];
    float col[256];
    int x, y;

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

    for (y = 0; y < h; y++) dct_1d(data + (size_t)y * stride, w, scratch, inverse);
    for (x = 0; x < w; x++) {
        for (y = 0; y < h; y++) col[y] = data[(size_t)y * stride + x];
        dct_1d(col, h, scratch, inverse);
        for (y = 0; y < h; y++) data[(size_t)y * stride + x] = col[y];
    }
}
