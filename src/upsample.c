/* upsample.c -- the non-separable 2x / 4x / 8x upsampling filter.
 *
 * A frame may be coded at 1/2, 1/4 or 1/8 resolution and upsampled by the
 * decoder (cjxl --resampling). Each input sample expands to an NxN block, and
 * every output sample in that block is a 5x5 weighted sum of the input
 * neighbourhood -- a different set of 25 weights per position in the block.
 * The weights live in CustomTransformData (headers.c) and default to
 * jxl_default_up2/up4/up8.
 *
 * The codestream stores only the upper-left quadrant of the NxN weight set
 * and only the upper triangle of that, because the kernel is symmetric under
 * both reflections and under transposition; build_kernel expands it back out.
 *
 * Each output sample is clamped to the min and max of the 5x5 input
 * neighbourhood it came from, which is what stops the filter ringing at
 * edges. libjxl does the same (render_pipeline/stage_upsampling.cc).
 */
#include "jxl_internal.h"

/* Mirror without repeating the edge sample, as everywhere else in the
   decoder. */
static uint32_t up_mirror(int64_t v, uint32_t len) {
    for (;;) {
        if (v < 0) v = -(v + 1);
        else if ((uint64_t)v >= len) v = (int64_t)len * 2 - 1 - v;
        else return (uint32_t)v;
    }
}

/* Expands the stored triangle into the full N*N sets of 25 weights. The
   stored index is the triangular-packed (my, mx) with my <= mx, where i and j
   run over the 5*H taps of one quadrant. */
static void build_kernel(float *kernel, uint32_t shift, const float *w) {
    uint32_t N = 1u << shift, H = N / 2;
    uint32_t ky, kx, py, px;
    for (ky = 0; ky < H; ky++) {
        for (kx = 0; kx < H; kx++) {
            size_t o0 = ((size_t)ky * N + kx) * 25;
            size_t o1 = ((size_t)ky * N + (N - 1 - kx)) * 25;
            size_t o2 = ((size_t)(N - 1 - ky) * N + kx) * 25;
            size_t o3 = ((size_t)(N - 1 - ky) * N + (N - 1 - kx)) * 25;
            for (py = 0; py < 5; py++) {
                for (px = 0; px < 5; px++) {
                    uint32_t j = 5 * ky + py;
                    uint32_t i = 5 * kx + px;
                    uint32_t my = i < j ? i : j;
                    uint32_t mx = i < j ? j : i;
                    size_t idx = (size_t)5 * H * my - (size_t)my * (my - 1) / 2
                                 + mx - my;
                    float v = w[idx];
                    kernel[o0 + (size_t)py * 5 + px] = v;
                    kernel[o1 + (size_t)py * 5 + (4 - px)] = v;
                    kernel[o2 + (size_t)(4 - py) * 5 + px] = v;
                    kernel[o3 + (size_t)(4 - py) * 5 + (4 - px)] = v;
                }
            }
        }
    }
}

/* SSE2 is baseline on x64. This is the innermost operation in the whole
   filter -- N*N of them per input sample, 25 multiply-adds each -- so it is
   the one place worth writing by hand. */
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define JXL_UPSAMPLE_SSE2 1
#include <emmintrin.h>
#endif

static float up_dot25(const float *a, const float *b) {
#ifdef JXL_UPSAMPLE_SSE2
    __m128 acc = _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
    __m128 t;
    int i;
    for (i = 4; i + 4 <= 24; i += 4) {
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i),
                                         _mm_loadu_ps(b + i)));
    }
    /* 25 is odd; lanes 0..23 are vectorised and the last tap stays scalar. */
    t = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t) + a[24] * b[24];
#else
    float sum = 0.0f;
    int n;
    for (n = 0; n < 25; n++) sum += a[n] * b[n];
    return sum;
#endif
}

/* Picks the weight set for a shift of 1, 2 or 3. */
static const float *weights_for(const jxl_image_metadata *meta, uint32_t shift) {
    if (shift == 1) return meta->up2;
    if (shift == 2) return meta->up4;
    return meta->up8;
}

int jxl_upsample_plane(jxl_ctx *ctx, jxl_fplane *p, uint32_t shift,
                       const jxl_image_metadata *meta, uint32_t out_w,
                       uint32_t out_h) {
    uint32_t N = 1u << shift;
    uint32_t w = p->w, h = p->h, x, y;
    float *kernel = NULL;
    jxl_fplane dst;
    int rc = -1;

    if (shift == 0 || shift > 3) return 0;
    if (w == 0 || h == 0) return 0;

    memset(&dst, 0, sizeof(dst));
    kernel = (float *)jxl_calloc(ctx, (size_t)N * N * 25, sizeof(float));
    if (!kernel) goto done;
    build_kernel(kernel, shift, weights_for(meta, shift));

    /* The upsampled frame is w*N wide but the image may be a few samples
       narrower, so allocate the crop and simply skip the samples past it. */
    if (jxl_fplane_alloc_uninit(ctx, &dst, out_w, out_h) != 0) goto done;

    for (y = 0; y < h; y++) {
        /* The five source rows depend only on y, so mirror them once per row
           rather than once per sample -- and inside the row, only the first
           and last two columns need mirroring at all. */
        const float *srow[5];
        int iy;
        for (iy = -2; iy <= 2; iy++) {
            srow[iy + 2] = p->data +
                (size_t)up_mirror((int64_t)y + iy, h) * p->stride;
        }
        for (x = 0; x < w; x++) {
            float nb[25];
            float lo, hi;
            uint32_t oy, ox;
            int ix;
            int n = 0;
            if (x >= 2 && x + 2 < w) {
                for (iy = 0; iy < 5; iy++) {
                    const float *r = srow[iy] + x - 2;
                    nb[n] = r[0]; nb[n + 1] = r[1]; nb[n + 2] = r[2];
                    nb[n + 3] = r[3]; nb[n + 4] = r[4];
                    n += 5;
                }
            } else {
                for (iy = 0; iy < 5; iy++) {
                    for (ix = -2; ix <= 2; ix++) {
                        nb[n++] = srow[iy][up_mirror((int64_t)x + ix, w)];
                    }
                }
            }
            lo = hi = nb[0];
            for (n = 1; n < 25; n++) {
                if (nb[n] < lo) lo = nb[n];
                if (nb[n] > hi) hi = nb[n];
            }
            for (oy = 0; oy < N; oy++) {
                uint32_t dy = y * N + oy;
                float *drow;
                if (dy >= out_h) break;
                drow = dst.data + (size_t)dy * dst.stride;
                for (ox = 0; ox < N; ox++) {
                    uint32_t dx = x * N + ox;
                    const float *k = kernel + ((size_t)oy * N + ox) * 25;
                    float sum;
                    if (dx >= out_w) break;
                    sum = up_dot25(nb, k);
                    if (sum < lo) sum = lo;
                    if (sum > hi) sum = hi;
                    drow[dx] = sum;
                }
            }
        }
    }

    jxl_free(ctx, p->data);
    *p = dst;
    memset(&dst, 0, sizeof(dst));
    rc = 0;

done:
    jxl_free(ctx, kernel);
    jxl_free(ctx, dst.data);
    return rc;
}
