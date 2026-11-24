/* fuzz_diff.c -- differential fuzzing: this decoder against libjxl.
 *
 * The other targets find crashes; this one finds wrong pixels on inputs no
 * encoder would write. Each input is decoded by both, and only when *both*
 * succeed are the results compared -- libjxl rejects some inputs this
 * decoder accepts and vice versa, and neither is a bug here. Then:
 *
 *   - image size, channel layout and the number of displayed frames must
 *     match exactly;
 *   - on frames that stay on the integer path (no XYB, Modular only, no
 *     loop filters, upsampling, noise, splines, patches or blending, no
 *     float samples) every sample must agree to within one step, as the
 *     corpus requires (cmd/tests.ts);
 *   - anywhere else both decoders compute in float with different
 *     factorizations, so only a gross error counts: a peak difference over
 *     64 steps or an RMS over 4 (in 8-bit steps).
 *
 * A mismatch aborts, so libFuzzer keeps the input as a crash artifact.
 *
 * Linux and macOS only: it links libjxl's static libraries from
 * deps/libjxl-build (cmd/build.ts), which on Windows are built for MSVC.
 *
 *   bun cmd/fuzz.ts -target diff
 */
#include <jxl/decode.h>

/* Only the public header here: jxl_internal.h and libjxl's headers both
   declare JXL_BLEND_REPLACE and friends (see fuzz_diff_ours.c). */
#include "fuzz_common.h"

#include <math.h>
#include <stdio.h>

#define DIFF_MAX_PIXELS ((uint64_t)1 << 20)
#define DIFF_MAX_FRAMES 4

/* libjxl's allocations go through the same budget as ours. */
static void *ref_alloc(void *opaque, size_t size) {
    (void)opaque;
    return fuzz_alloc(NULL, NULL, size);
}
static void ref_free(void *opaque, void *addr) {
    (void)opaque;
    fuzz_free(NULL, NULL, addr);
}

typedef struct {
    uint32_t w, h, comps;
    int wide;
    int nframes;                      /* every displayed frame */
    uint8_t *px[DIFF_MAX_FRAMES];     /* the first DIFF_MAX_FRAMES of them */
    size_t px_len;
} ref_image;

static void ref_image_free(ref_image *r) {
    int i;
    for (i = 0; i < DIFF_MAX_FRAMES; i++) free(r->px[i]);
}

/* Every displayed frame, coalesced, in the layout jxl_frame_render's
   NATIVE format uses (gray or RGB, plus alpha; 8-bit, or 16-bit native
   endian above 8 bits) -- what test/jxl_ref.c does for cmd/tests.ts. */
static int ref_decode(const uint8_t *data, size_t size, ref_image *r) {
    JxlMemoryManager mm = {NULL, ref_alloc, ref_free};
    JxlDecoder *dec = JxlDecoderCreate(&mm);
    JxlBasicInfo info;
    JxlPixelFormat fmt;
    uint8_t *buf = NULL;
    int rc = -1;

    memset(r, 0, sizeof(*r));
    memset(&fmt, 0, sizeof(fmt));
    if (!dec) return -1;
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
            JXL_DEC_SUCCESS ||
        JxlDecoderSetCoalescing(dec, JXL_TRUE) != JXL_DEC_SUCCESS)
        goto done;
    JxlDecoderSetInput(dec, data, size);
    JxlDecoderCloseInput(dec);
    for (;;) {
        JxlDecoderStatus st = JxlDecoderProcessInput(dec);
        if (st == JXL_DEC_BASIC_INFO) {
            if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) goto done;
            if ((uint64_t)info.xsize * info.ysize > DIFF_MAX_PIXELS) goto done;
            r->w = info.xsize;
            r->h = info.ysize;
            r->wide = info.bits_per_sample > 8 || info.exponent_bits_per_sample > 0;
            r->comps = info.num_color_channels + (info.alpha_bits ? 1 : 0);
            fmt.num_channels = r->comps;
            fmt.data_type = r->wide ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8;
            fmt.endianness = JXL_NATIVE_ENDIAN;
        } else if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            size_t need;
            if (JxlDecoderImageOutBufferSize(dec, &fmt, &need) != JXL_DEC_SUCCESS)
                goto done;
            free(buf);
            buf = (uint8_t *)malloc(need ? need : 1);
            if (!buf) goto done;
            r->px_len = need;
            if (JxlDecoderSetImageOutBuffer(dec, &fmt, buf, need) != JXL_DEC_SUCCESS)
                goto done;
        } else if (st == JXL_DEC_FULL_IMAGE) {
            if (r->nframes < DIFF_MAX_FRAMES) {
                r->px[r->nframes] = buf;
                buf = NULL;
            }
            r->nframes++;
        } else if (st == JXL_DEC_SUCCESS) {
            rc = 0;
            break;
        } else {
            break;
        }
    }
done:
    free(buf);
    JxlDecoderDestroy(dec);
    return rc;
}

/* fuzz_diff_ours.c: whether every frame stays on the integer path. */
int fuzz_diff_frames_exact(jxl_doc *doc);

static void mismatch(const char *what, int frame, double a, double b) {
    fprintf(stderr, "fuzz_diff: %s differs (frame %d): ours %g, libjxl %g\n",
            what, frame, a, b);
    abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    jxl_ctx *ctx;
    jxl_doc *doc;
    jxl_image_info info;
    ref_image ref;
    int n, i, exact;

    if (size < 4) return 0;
    ctx = fuzz_ctx_new();
    if (!ctx) return 0;
    doc = jxl_doc_open(ctx, data, size);
    if (!doc || jxl_doc_info(doc, &info) != 0 ||
        (uint64_t)info.width * (uint64_t)info.height > DIFF_MAX_PIXELS) {
        if (doc) jxl_doc_close(doc);
        jxl_ctx_free(ctx);
        return 0;
    }
    n = jxl_doc_frame_count(doc);
    exact = fuzz_diff_frames_exact(doc);
    if (n <= 0 || ref_decode(data, size, &ref) != 0) {
        ref_image_free(&ref);
        jxl_doc_close(doc);
        jxl_ctx_free(ctx);
        return 0;
    }

    for (i = 0; i < n && i < DIFF_MAX_FRAMES; i++) {
        jxl_image *img = jxl_frame_render(doc, i, JXLDEC_FORMAT_NATIVE);
        size_t k, count;
        double peak = 0, sum = 0, scale, rms;
        if (!img) {
            /* Ours failed where libjxl did not: not comparable. */
            n = -1;
            break;
        }
        if (i == 0) {
            int comps = jxl_format_bpp(img->format) / (ref.wide ? 2 : 1);
            if ((uint32_t)img->width != ref.w || (uint32_t)img->height != ref.h)
                mismatch("width x height", i, img->width * 65536.0 + img->height,
                         ref.w * 65536.0 + ref.h);
            if ((uint32_t)comps != ref.comps || jxl_format_bpp(img->format) !=
                                                   (int)(ref.comps * (ref.wide ? 2 : 1)))
                mismatch("channels / sample size", i, jxl_format_bpp(img->format),
                         ref.comps * (ref.wide ? 2 : 1));
        }
        if (!ref.px[i] || (size_t)img->stride * img->height != ref.px_len) {
            jxl_image_destroy(ctx, img);
            mismatch("buffer size", i, (double)img->stride * img->height,
                     (double)ref.px_len);
        }
        count = ref.px_len / (ref.wide ? 2 : 1);
        scale = ref.wide ? 1.0 / 257.0 : 1.0;   /* in 8-bit steps */
        for (k = 0; k < count; k++) {
            double a, b, d;
            if (ref.wide) {
                uint16_t x, y;
                memcpy(&x, img->data + k * 2, 2);
                memcpy(&y, ref.px[i] + k * 2, 2);
                a = x;
                b = y;
            } else {
                a = img->data[k];
                b = ref.px[i][k];
            }
            d = fabs(a - b);              /* in output steps */
            if (d > peak) peak = d;
            sum += d * scale * d * scale; /* in 8-bit steps */
        }
        jxl_image_destroy(ctx, img);
        rms = sqrt(sum / (double)(count ? count : 1));
        if (exact && peak > 1.0)
            mismatch("integer-path sample, peak in output steps", i, peak, 1.0);
        if (!exact && (peak * scale > 64.0 || rms > 4.0))
            mismatch("pixels, peak / rms in 8-bit steps", i, peak * scale, rms);
    }
    if (n >= 0 && n != ref.nframes)
        mismatch("displayed frame count", -1, n, ref.nframes);

    ref_image_free(&ref);
    jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    return 0;
}
