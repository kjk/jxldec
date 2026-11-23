/* fuzz_target.c -- LLVMFuzzerTestOneInput for the JPEG XL decoder.
 *
 * Each input is treated as a whole .jxl file: open it, read the info and the
 * ICC profile, then render frames, so malformed bytes reach the container,
 * header, entropy-coding, Modular, VarDCT, filter and render paths.
 *
 *   bun cmd/fuzz.ts      clang -fsanitize=address,fuzzer (libFuzzer supplies main)
 *   bun cmd/fuzz-afl.ts  afl-clang-fast -fsanitize=fuzzer (AFL++ libAFLDriver)
 *
 * The corpus this fuzzer seeds from is 1200+ *valid* files, and a valid file
 * never takes an error path: jxl_errorf, the target of every JXL_ERR in the
 * decoder, does not execute once across the whole of cmd/tests.ts. Everything
 * the decoder does when it is lied to is what this is here to cover.
 */
#include "jxl.h"

#include <stdlib.h>
#include <string.h>

/* Budgeted allocator. A crafted size header can declare an enormous canvas,
 * and the decoder has no built-in pixel limit, so bound live bytes instead:
 * such inputs then fail through the library's own allocation-failure paths --
 * which is more of the error handling under test -- rather than tripping
 * libFuzzer's RSS limit and stopping the run. The size lives in a 16-byte
 * header so the returned pointer keeps malloc's alignment. */
#define FUZZ_MEM_BUDGET ((size_t)512 << 20)   /* live bytes per input */

static size_t fuzz_live;

static void *fuzz_alloc(void *user, void *ctx, size_t size) {
    uint8_t *p;
    (void)user;
    (void)ctx;
    if (size > FUZZ_MEM_BUDGET - fuzz_live) return NULL;
    p = (uint8_t *)malloc(size + 16);
    if (!p) return NULL;
    memcpy(p, &size, sizeof(size));
    fuzz_live += size;
    return p + 16;
}

static void fuzz_free(void *user, void *ctx, void *ptr) {
    uint8_t *p;
    size_t size;
    (void)user;
    (void)ctx;
    if (!ptr) return;
    p = (uint8_t *)ptr - 16;
    memcpy(&size, p, sizeof(size));
    fuzz_live -= size;
    free(p);
}

/* Animations are decoded in order and each frame costs a full decode, so a
 * crafted header claiming thousands of frames would stall the fuzzer without
 * covering anything new. */
#define FUZZ_MAX_FRAMES 4

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    jxl_ctx *ctx;
    jxl_doc *doc;
    jxl_image_info info;
    const uint8_t *icc;
    size_t icc_len;
    int nframes, i, sw, sh;
    unsigned flags;

    /* Nothing shorter can carry a signature; skip rather than spend the
     * mutation budget on inputs jxl_signature_check would reject anyway. */
    if (size < 4) return 0;

    fuzz_live = 0;
    ctx = jxl_ctx_new(fuzz_alloc, fuzz_free, NULL, NULL);
    if (!ctx) return 0;

    /* The context flags are keyed off the input, so every combination gets
     * exercised across the corpus and a crash artifact still replays
     * identically. Off the *last* byte and the length, not the first byte:
     * anything that gets past jxl_doc_open starts with 0xFF (bare codestream)
     * or 0x00 (container), so data[0] only ever produced two of the
     * combinations. srgb_output is here because the one real consumer,
     * SumatraPDF, always turns it on. */
    flags = (unsigned)data[size - 1] ^ (unsigned)size;
    jxl_ctx_set_bgr(ctx, flags & 1);
    jxl_ctx_set_keep_orientation(ctx, (flags >> 1) & 1);
    jxl_ctx_set_srgb_output(ctx, (flags >> 2) & 1);

    (void)jxl_signature_check(data, size);
    /* Opens and closes a doc of its own; costs a header parse. */
    (void)jxl_decode_size(ctx, data, size, &sw, &sh);

    doc = jxl_doc_open(ctx, data, size);
    if (!doc) {
        jxl_ctx_free(ctx);
        return 0;
    }

    if (jxl_doc_info(doc, &info) != 0) {
        jxl_doc_close(doc);
        jxl_ctx_free(ctx);
        return 0;
    }

    icc_len = 0;
    icc = jxl_doc_icc_profile(doc, &icc_len);
    if (icc && icc_len) {
        /* Touch it so a bad length is a read the sanitizer sees. */
        volatile uint8_t sink = icc[icc_len - 1];
        (void)sink;
    }

    nframes = jxl_doc_frame_count(doc);
    if (nframes > FUZZ_MAX_FRAMES) nframes = FUZZ_MAX_FRAMES;
    for (i = 0; i < nframes; i++) {
        jxl_frame_info finfo;
        jxl_image *img;
        /* NATIVE resolves from the image; the second format forces the
         * conversion path (gray->rgb, 16->8, alpha attach) instead. */
        jxl_format fmt = (i & 1) ? JXLDEC_FORMAT_RGBA32 : JXLDEC_FORMAT_NATIVE;
        (void)jxl_doc_frame_info(doc, i, &finfo);
        img = jxl_frame_render(doc, i, fmt);
        if (img) jxl_image_destroy(ctx, img);
    }

    /* The caller-buffer path, which is how SumatraPDF renders: the geometry
     * comes from jxl_frame_render_info and the pixels go straight into a
     * buffer of exactly that size, so a row written one sample too long is a
     * heap overflow the sanitizer sees. The stride gets a few bytes of
     * padding on some inputs to cover stride != width * bpp. */
    {
        jxl_render_info ri;
        jxl_format fmt = (flags & 8) ? JXLDEC_FORMAT_RGB24 : JXLDEC_FORMAT_RGBA32;
        if (jxl_frame_render_info(doc, 0, fmt, &ri) == 0 && ri.width > 0 &&
            ri.height > 0) {
            size_t bpp = (size_t)jxl_format_bpp(ri.format);
            size_t stride = (size_t)ri.width * bpp + ((flags >> 4) & 3);
            if (stride <= 0x7fffffffu &&
                stride <= (FUZZ_MEM_BUDGET - fuzz_live) / (size_t)ri.height) {
                uint8_t *buf = (uint8_t *)malloc(stride * (size_t)ri.height);
                if (buf) {
                    (void)jxl_frame_render_into(doc, 0, fmt, buf, (int)stride);
                    free(buf);
                }
            }
        }
    }

    jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    return 0;
}
