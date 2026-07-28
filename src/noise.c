/* noise.c -- synthetic photon/film noise.
 *
 * The encoder can strip noise from an image and record only its intensity
 * curve; the decoder regenerates it deterministically. Each group seeds an
 * XorShift128+ generator from its top-left corner (and the frame counters),
 * producing white noise in [1, 2); a 5x5 high-pass convolution turns that into
 * zero-mean blue-ish noise, which is then scaled per pixel by a lookup into
 * the 8-entry intensity curve.
 *
 * The first two random planes are generated whole. The correlated third
 * plane is generated row by row from a saved RNG state per group and filtered
 * through a five-row ring, so it never needs a full-frame allocation.
 * Neighbouring groups are simply neighbouring pixels; only the frame border
 * needs the mirrored padding libjxl uses.
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

/* SSE2 is baseline on x64, so no runtime dispatch and no scalar/vector output
   divergence to worry about: the same lanes are summed in the same order
   either way, only four at a time. */
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define JXL_NOISE_SSE2 1
#include <immintrin.h>
#endif

#ifdef JXL_NOISE_SSE2
/* Advance all eight XorShift lanes together. Storing each 64-bit result as
   two little-endian 32-bit words produces the exact scalar output order;
   turning those words into [1, 2) floats is then just an integer shift and
   exponent-bit OR. */
JXL_TARGET_AVX2
static void xorshift_batch_x8(jxl_xorshift *r, float out[NOISE_LANES * 2]) {
    const __m256i exp = _mm256_set1_epi32(0x3f800000u);
    int i;
    for (i = 0; i < NOISE_LANES; i += 4) {
        __m256i s1 = _mm256_loadu_si256((const __m256i *)(r->s0 + i));
        __m256i s0 = _mm256_loadu_si256((const __m256i *)(r->s1 + i));
        __m256i ret = _mm256_add_epi64(s1, s0);
        __m256i next;
        _mm256_storeu_si256((__m256i *)(r->s0 + i), s0);
        s1 = _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 23));
        next = _mm256_xor_si256(
            s1, _mm256_xor_si256(
                s0, _mm256_xor_si256(_mm256_srli_epi64(s1, 18),
                                     _mm256_srli_epi64(s0, 5))));
        _mm256_storeu_si256((__m256i *)(r->s1 + i), next);
        ret = _mm256_or_si256(_mm256_srli_epi32(ret, 9), exp);
        _mm256_storeu_ps(out + i * 2, _mm256_castsi256_ps(ret));
    }
    _mm256_zeroupper();
}
#endif

/* Generate one row of the third random plane. Calls arrive in increasing y
   order, so each group's saved state advances in exactly the same order as
   generating that group's complete third plane immediately after planes 0
   and 1. Group generators are independent, so interleaving them by row does
   not alter any output. */
static void noise_random_row(float *dst, uint32_t width, uint32_t y,
                             uint32_t group_dim, uint32_t groups_per_row,
                             jxl_xorshift *rngs, int use_avx2) {
    uint32_t gx;
    uint32_t gy = y / group_dim;
    for (gx = 0; gx < groups_per_row; gx++) {
        uint32_t x0 = gx * group_dim;
        uint32_t gw = JXL_MIN(group_dim, width - x0);
        uint32_t blocks = (gw + NOISE_LANES * 2 - 1) /
                          (NOISE_LANES * 2);
        jxl_xorshift *rng = rngs + (size_t)gy * groups_per_row + gx;
        uint32_t b;
        for (b = 0; b < blocks; b++) {
            uint32_t bx = b * NOISE_LANES * 2;
            uint32_t valid = JXL_MIN(NOISE_LANES * 2, gw - bx);
            float *out = dst + x0 + bx;
            uint32_t k;
#ifdef JXL_NOISE_SSE2
            if (use_avx2) {
                if (valid == NOISE_LANES * 2) {
                    xorshift_batch_x8(rng, out);
                } else {
                    float values[NOISE_LANES * 2];
                    xorshift_batch_x8(rng, values);
                    memcpy(out, values, valid * sizeof(float));
                }
                continue;
            }
#else
            (void)use_avx2;
#endif
            {
                uint32_t bits[NOISE_LANES * 2];
                xorshift_batch(rng, bits);
                for (k = 0; k < valid; k++) out[k] = bits_to_unit(bits[k]);
            }
        }
    }
}

/* One row of the 5-wide horizontal box, mirrored at both ends. Splitting the
   row into "needs mirroring" and "does not" is what buys the most here: the
   old code called noise_mirror -- a loop -- for all five taps of every sample,
   including the interior, where the answer is always x+dx. */
static void noise_hbox5(const float *src, float *dst, uint32_t w) {
    uint32_t x, lo, hi;
    lo = w < 2 ? w : 2;
    hi = w < 2 ? w : w - 2;
    if (hi < lo) hi = lo;
    for (x = 0; x < lo; x++) {
        dst[x] = src[noise_mirror((int64_t)x - 2, w)] +
                 src[noise_mirror((int64_t)x - 1, w)] +
                 src[noise_mirror((int64_t)x, w)] +
                 src[noise_mirror((int64_t)x + 1, w)] +
                 src[noise_mirror((int64_t)x + 2, w)];
    }
    x = lo;
#ifdef JXL_NOISE_SSE2
    for (; x + 4 <= hi; x += 4) {
        __m128 s = _mm_loadu_ps(src + x - 2);
        s = _mm_add_ps(s, _mm_loadu_ps(src + x - 1));
        s = _mm_add_ps(s, _mm_loadu_ps(src + x));
        s = _mm_add_ps(s, _mm_loadu_ps(src + x + 1));
        s = _mm_add_ps(s, _mm_loadu_ps(src + x + 2));
        _mm_storeu_ps(dst + x, s);
    }
#endif
    for (; x < hi; x++) {
        dst[x] = src[x - 2] + src[x - 1] + src[x] + src[x + 1] + src[x + 2];
    }
    for (x = hi; x < w; x++) {
        dst[x] = src[noise_mirror((int64_t)x - 2, w)] +
                 src[noise_mirror((int64_t)x - 1, w)] +
                 src[noise_mirror((int64_t)x, w)] +
                 src[noise_mirror((int64_t)x + 1, w)] +
                 src[noise_mirror((int64_t)x + 2, w)];
    }
}

#ifdef JXL_NOISE_SSE2
JXL_TARGET_AVX2
static void noise_hbox5_avx2(const float *src, float *dst, uint32_t w) {
    uint32_t x, lo, hi;
    lo = w < 2 ? w : 2;
    hi = w < 2 ? w : w - 2;
    if (hi < lo) hi = lo;
    for (x = 0; x < lo; x++) {
        dst[x] = src[noise_mirror((int64_t)x - 2, w)] +
                 src[noise_mirror((int64_t)x - 1, w)] +
                 src[noise_mirror((int64_t)x, w)] +
                 src[noise_mirror((int64_t)x + 1, w)] +
                 src[noise_mirror((int64_t)x + 2, w)];
    }
    x = lo;
    for (; x + 8 <= hi; x += 8) {
        __m256 s = _mm256_loadu_ps(src + x - 2);
        s = _mm256_add_ps(s, _mm256_loadu_ps(src + x - 1));
        s = _mm256_add_ps(s, _mm256_loadu_ps(src + x));
        s = _mm256_add_ps(s, _mm256_loadu_ps(src + x + 1));
        s = _mm256_add_ps(s, _mm256_loadu_ps(src + x + 2));
        _mm256_storeu_ps(dst + x, s);
    }
    for (; x < hi; x++) {
        dst[x] = src[x - 2] + src[x - 1] + src[x] + src[x + 1] + src[x + 2];
    }
    for (x = hi; x < w; x++) {
        dst[x] = src[noise_mirror((int64_t)x - 2, w)] +
                 src[noise_mirror((int64_t)x - 1, w)] +
                 src[noise_mirror((int64_t)x, w)] +
                 src[noise_mirror((int64_t)x + 1, w)] +
                 src[noise_mirror((int64_t)x + 2, w)];
    }
    _mm256_zeroupper();
}
#endif

/* The vertical half: five already-horizontally-summed rows, scaled by 0.16,
   minus four times the centre sample. 0.16 * 25 == 4, so the kernel sums to
   zero and the [1, 2) offset of the raw values cancels. */
static void noise_vbox5(const float *const rows[5], const float *centre,
                        float *dst, uint32_t w) {
    uint32_t x = 0;
#ifdef JXL_NOISE_SSE2
    const __m128 k = _mm_set1_ps(0.16f), m4 = _mm_set1_ps(4.0f);
    for (; x + 4 <= w; x += 4) {
        __m128 s = _mm_loadu_ps(rows[0] + x);
        s = _mm_add_ps(s, _mm_loadu_ps(rows[1] + x));
        s = _mm_add_ps(s, _mm_loadu_ps(rows[2] + x));
        s = _mm_add_ps(s, _mm_loadu_ps(rows[3] + x));
        s = _mm_add_ps(s, _mm_loadu_ps(rows[4] + x));
        s = _mm_sub_ps(_mm_mul_ps(s, k),
                       _mm_mul_ps(_mm_loadu_ps(centre + x), m4));
        _mm_storeu_ps(dst + x, s);
    }
#endif
    for (; x < w; x++) {
        float s = rows[0][x] + rows[1][x] + rows[2][x] + rows[3][x] + rows[4][x];
        dst[x] = s * 0.16f - centre[x] * 4.0f;
    }
}

#ifdef JXL_NOISE_SSE2
JXL_TARGET_AVX2
static void noise_vbox5_avx2(const float *const rows[5], const float *centre,
                             float *dst, uint32_t w) {
    const __m256 k = _mm256_set1_ps(0.16f);
    const __m256 m4 = _mm256_set1_ps(4.0f);
    uint32_t x = 0;
    for (; x + 8 <= w; x += 8) {
        __m256 s = _mm256_loadu_ps(rows[0] + x);
        s = _mm256_add_ps(s, _mm256_loadu_ps(rows[1] + x));
        s = _mm256_add_ps(s, _mm256_loadu_ps(rows[2] + x));
        s = _mm256_add_ps(s, _mm256_loadu_ps(rows[3] + x));
        s = _mm256_add_ps(s, _mm256_loadu_ps(rows[4] + x));
        s = _mm256_sub_ps(_mm256_mul_ps(s, k),
                          _mm256_mul_ps(_mm256_loadu_ps(centre + x), m4));
        _mm256_storeu_ps(dst + x, s);
    }
    for (; x < w; x++) {
        float s = rows[0][x] + rows[1][x] + rows[2][x] + rows[3][x] + rows[4][x];
        dst[x] = s * 0.16f - centre[x] * 4.0f;
    }
    _mm256_zeroupper();
}
#endif

#ifdef JXL_NOISE_SSE2
JXL_TARGET_AVX2
static JXL_INLINE_HINT __m256 noise_strength_x8(__m256 v, __m256 lut) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 seven = _mm256_set1_ps(7.0f);
    const __m256i seven_i = _mm256_set1_epi32(7);
    __m256i idx;
    __m256 frac, lo, hi;
    /* Match the scalar path: clamp only negative inputs, then clamp the LUT
       index rather than the value. lut[8] repeats lut[7], so values above the
       table range still interpolate to that final value. */
    v = _mm256_blendv_ps(v, zero,
                         _mm256_cmp_ps(v, zero, _CMP_LT_OQ));
    /* Clamp the conversion operand as well as the resulting integer so a
       corrupt, enormous coefficient cannot turn into a negative gather
       index. Keep the original v for frac: above-range values must retain
       the scalar path's v - 7, although the repeated last LUT entry then
       makes that fraction immaterial. */
    idx = _mm256_cvttps_epi32(_mm256_min_ps(v, seven));
    frac = _mm256_sub_ps(v, _mm256_cvtepi32_ps(idx));
    lo = _mm256_permutevar8x32_ps(lut, idx);
    hi = _mm256_permutevar8x32_ps(
        lut, _mm256_min_epi32(_mm256_add_epi32(idx, _mm256_set1_epi32(1)),
                              seven_i));
    return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(hi, lo), frac), lo);
}

JXL_TARGET_AVX2
static void noise_apply_x8(float *rx, float *ry, float *rb,
                           const float *nxr, const float *nyr,
                           const float *nbr, const float *lut,
                           float corr_x, float corr_b, uint32_t n,
                           uint32_t *idx) {
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 k22 = _mm256_set1_ps(0.22f);
    const __m256 k1 = _mm256_set1_ps(0.0078125f);
    const __m256 k127 = _mm256_set1_ps(0.9921875f);
    const __m256 kx = _mm256_set1_ps(corr_x);
    const __m256 kb = _mm256_set1_ps(corr_b);
    const __m256 lutv = _mm256_loadu_ps(lut);
    uint32_t x = *idx;
    for (; x + 8 <= n; x += 8) {
        __m256 vx = _mm256_loadu_ps(rx + x);
        __m256 vy = _mm256_loadu_ps(ry + x);
        __m256 sx = noise_strength_x8(
            _mm256_mul_ps(_mm256_add_ps(vx, vy), three), lutv);
        __m256 sy = noise_strength_x8(
            _mm256_mul_ps(_mm256_sub_ps(vy, vx), three), lutv);
        __m256 nxmix = _mm256_add_ps(
            _mm256_mul_ps(k1, _mm256_loadu_ps(nxr + x)),
            _mm256_mul_ps(k127, _mm256_loadu_ps(nbr + x)));
        __m256 nymix = _mm256_add_ps(
            _mm256_mul_ps(k1, _mm256_loadu_ps(nyr + x)),
            _mm256_mul_ps(k127, _mm256_loadu_ps(nbr + x)));
        __m256 nx = _mm256_mul_ps(_mm256_mul_ps(k22, sx), nxmix);
        __m256 ny = _mm256_mul_ps(_mm256_mul_ps(k22, sy), nymix);
        __m256 sum = _mm256_add_ps(nx, ny);
        __m256 dx = _mm256_sub_ps(
            _mm256_add_ps(_mm256_mul_ps(kx, sum), nx), ny);
        _mm256_storeu_ps(rx + x, _mm256_add_ps(vx, dx));
        _mm256_storeu_ps(ry + x, _mm256_add_ps(vy, sum));
        _mm256_storeu_ps(rb + x, _mm256_add_ps(
            _mm256_loadu_ps(rb + x), _mm256_mul_ps(kb, sum)));
    }
    _mm256_zeroupper();
    *idx = x;
}
#endif

int jxl_render_noise(jxl_ctx *ctx, jxl_fimage *img, const jxl_noise_params *np,
                     const jxl_frame_header *fh, uint32_t visible_frames,
                     uint32_t invisible_frames, float corr_x, float corr_b) {
    uint32_t width = fh->width, height = fh->height;
    uint32_t group_dim = jxl_frame_group_dim(fh);
    uint32_t groups_per_row, group_rows, gx, gy;
    float *raw[3] = {NULL, NULL, NULL};
    float *hsum = NULL;          /* five sums + five third-plane raw rows */
    jxl_xorshift *corr_rng = NULL;
    uint32_t hsum_row[5];        /* which source row is in each ring slot    */
    uint64_t seed0 = ((uint64_t)visible_frames << 32) + invisible_frames;
    float lut[9];
    uint32_t x, y;
    int c, rc = -1;
    int use_avx2 = 0;
#ifdef JXL_NOISE_SSE2
    use_avx2 = jxl_has_avx2();
#endif

    if (img->ncolor < 3 || width == 0 || height == 0) return 0;
    for (c = 0; c < 8; c++) lut[c] = np->lut[c];
    lut[8] = np->lut[7];

    groups_per_row = (width + group_dim - 1) / group_dim;
    group_rows = (height + group_dim - 1) / group_dim;

    /* Groups tile the frame and each writes its whole extent, and every
       every sample is assigned before it is read, so none of these need
       zeroing first -- calloc was clearing bytes the next loop overwrote. */
    {
        size_t n;
        if (!jxl_size_mul((size_t)width * height, sizeof(float), &n)) goto done;
        for (c = 0; c < 2; c++) {
            raw[c] = (float *)jxl_malloc(ctx, n);
            if (!raw[c]) goto done;
        }
        /* Five horizontal sums plus five raw rows for the streamed correlated
           plane. The first two planes only use the sum half. */
        if (!jxl_size_mul((size_t)width * 10, sizeof(float), &n)) goto done;
        hsum = (float *)jxl_malloc(ctx, n);
        if (!hsum) goto done;
        if (!jxl_size_mul((size_t)groups_per_row * group_rows,
                          sizeof(*corr_rng), &n)) goto done;
        corr_rng = (jxl_xorshift *)jxl_malloc(ctx, n);
        if (!corr_rng) goto done;
    }

    for (gy = 0; gy < group_rows; gy++) {
        for (gx = 0; gx < groups_per_row; gx++) {
            uint32_t x0 = gx * group_dim, y0 = gy * group_dim;
            uint32_t gw = JXL_MIN(group_dim, width - x0);
            uint32_t gh = JXL_MIN(group_dim, height - y0);
            uint32_t blocks = (gw + NOISE_LANES * 2 - 1) / (NOISE_LANES * 2);
            jxl_xorshift rng;
            xorshift_init(&rng, seed0, ((uint64_t)x0 << 32) + y0);
            /* Generate the two independent planes now. Save the state at the
               exact point where the third plane would start. */
            for (c = 0; c < 2; c++) {
                uint32_t row, b;
                for (row = 0; row < gh; row++) {
                    for (b = 0; b < blocks; b++) {
                        uint32_t bx = b * NOISE_LANES * 2;
                        uint32_t valid = JXL_MIN(NOISE_LANES * 2, gw - bx);
                        float *dst = raw[c] +
                            (size_t)(y0 + row) * width + x0 + bx;
                        uint32_t k;
#ifdef JXL_NOISE_SSE2
                        if (use_avx2) {
                            if (valid == NOISE_LANES * 2) {
                                xorshift_batch_x8(&rng, dst);
                            } else {
                                float values[NOISE_LANES * 2];
                                xorshift_batch_x8(&rng, values);
                                memcpy(dst, values, valid * sizeof(float));
                            }
                            continue;
                        }
#endif
                        {
                            uint32_t bits[NOISE_LANES * 2];
                            xorshift_batch(&rng, bits);
                            for (k = 0; k < valid; k++)
                                dst[k] = bits_to_unit(bits[k]);
                        }
                    }
                }
            }
            corr_rng[(size_t)gy * groups_per_row + gx] = rng;
        }
    }

    /* 5x5 high-pass: 0.16 * sum minus 4 * centre. The box is separable, so
       this is a horizontal 5-tap into hsum followed by a vertical one --
       ten adds a sample instead of twenty-five multiply-adds, and the
       mirroring is resolved once per row rather than once per tap. */
    for (c = 0; c < 2; c++) {
        uint32_t slot;
        for (slot = 0; slot < 5; slot++) hsum_row[slot] = (uint32_t)-1;
        for (y = 0; y < height; y++) {
            const float *rows[5];
            int dy;
            /* Fill any of the five needed source rows that is not already in
               the ring. Mirroring at the top and bottom repeats rows, and a
               repeat maps to the same slot holding the same row, so it costs
               nothing; in the interior the five rows are consecutive and land
               in five distinct slots. */
            for (dy = -2; dy <= 2; dy++) {
                uint32_t r = noise_mirror((int64_t)y + dy, height);
                uint32_t sl = r % 5;
                if (hsum_row[sl] != r) {
#ifdef JXL_NOISE_SSE2
                    if (use_avx2)
                        noise_hbox5_avx2(raw[c] + (size_t)r * width,
                                         hsum + (size_t)sl * width, width);
                    else
#endif
                        noise_hbox5(raw[c] + (size_t)r * width,
                                    hsum + (size_t)sl * width, width);
                    hsum_row[sl] = r;
                }
                rows[dy + 2] = hsum + (size_t)sl * width;
            }
            /* In place: vbox reads centre[x] before storing dst[x], and by
               now every horizontal sum that row y feeds has been taken. A
               slot is only recycled for row y+5, which is first needed at
               output row y+3 -- by which point row y is out of the window. */
#ifdef JXL_NOISE_SSE2
            if (use_avx2)
                noise_vbox5_avx2(rows, raw[c] + (size_t)y * width,
                                 raw[c] + (size_t)y * width, width);
            else
#endif
                noise_vbox5(rows, raw[c] + (size_t)y * width,
                            raw[c] + (size_t)y * width, width);
        }
    }

    /* Generate, convolve and consume the correlated plane as a row stream.
       Its raw centre row can be overwritten by the vertical output: no later
       output needs that raw row, and its horizontal sum is already retained. */
    {
        float *corr_raw = hsum + (size_t)width * 5;
        uint32_t next_corr_row = 0;
        uint32_t slot;
        for (slot = 0; slot < 5; slot++) hsum_row[slot] = (uint32_t)-1;
        for (y = 0; y < height && y < img->plane[0].h; y++) {
            const float *rows[5];
            uint32_t last_needed = JXL_MIN(y + 2, height - 1);
            float *rx, *ry, *rb, *nbr;
            const float *nxr, *nyr;
            uint32_t n;
            int dy;
            while (next_corr_row <= last_needed) {
                uint32_t sl = next_corr_row % 5;
                float *src = corr_raw + (size_t)sl * width;
                noise_random_row(src, width, next_corr_row, group_dim,
                                 groups_per_row, corr_rng, use_avx2);
#ifdef JXL_NOISE_SSE2
                if (use_avx2)
                    noise_hbox5_avx2(src, hsum + (size_t)sl * width,
                                     width);
                else
#endif
                    noise_hbox5(src, hsum + (size_t)sl * width, width);
                hsum_row[sl] = next_corr_row++;
            }
            for (dy = -2; dy <= 2; dy++) {
                uint32_t r = noise_mirror((int64_t)y + dy, height);
                rows[dy + 2] = hsum + (size_t)(r % 5) * width;
            }
            nbr = corr_raw + (size_t)(y % 5) * width;
#ifdef JXL_NOISE_SSE2
            if (use_avx2)
                noise_vbox5_avx2(rows, nbr, nbr, width);
            else
#endif
                noise_vbox5(rows, nbr, nbr, width);

            rx = img->plane[0].data + (size_t)y * img->plane[0].stride;
            ry = img->plane[1].data + (size_t)y * img->plane[1].stride;
            rb = img->plane[2].data + (size_t)y * img->plane[2].stride;
            nxr = raw[0] + (size_t)y * width;
            nyr = raw[1] + (size_t)y * width;
            n = JXL_MIN(width, img->plane[0].w);
            x = 0;
#ifdef JXL_NOISE_SSE2
            if (use_avx2)
                noise_apply_x8(rx, ry, rb, nxr, nyr, nbr, lut,
                               corr_x, corr_b, n, &x);
#endif
            for (; x < n; x++) {
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
                nx = 0.22f * sxv *
                     (0.0078125f * nxr[x] + 0.9921875f * nbr[x]);
                ny = 0.22f * syv *
                     (0.0078125f * nyr[x] + 0.9921875f * nbr[x]);
                rx[x] += corr_x * (nx + ny) + nx - ny;
                ry[x] += nx + ny;
                rb[x] += corr_b * (nx + ny);
            }
        }
    }
    rc = 0;

done:
    jxl_free(ctx, corr_rng);
    jxl_free(ctx, hsum);
    for (c = 0; c < 3; c++) {
        jxl_free(ctx, raw[c]);
    }
    return rc;
}
