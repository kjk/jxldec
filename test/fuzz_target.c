/* fuzz_target.c -- LLVMFuzzerTestOneInput for the JPEG XL decoder.
 *
 * Each input is treated as a whole .jxl file: open it, read the info and the
 * ICC profile, then render frames, so malformed bytes reach the container,
 * header, entropy-coding, Modular, VarDCT, filter and render paths.
 *
 *   bun cmd/fuzz.ts      clang -fsanitize=address,fuzzer (libFuzzer supplies main)
 *   bun cmd/fuzz-afl.ts  afl-clang-fast -fsanitize=fuzzer (AFL++ libAFLDriver)
 *
 * The other targets each take one component on its own (fuzz_headers.c,
 * fuzz_icc.c, fuzz_entropy.c, fuzz_modular.c) and run thousands of times
 * faster; this one is what reaches the interactions between them.
 *
 * The corpus this fuzzer seeds from is 1200+ *valid* files, and a valid file
 * never takes an error path: jxl_errorf, the target of every JXL_ERR in the
 * decoder, does not execute once across the whole of cmd/tests.ts. Everything
 * the decoder does when it is lied to is what this is here to cover.
 */
#include "fuzz_common.h"

/* Animations are decoded in order and each frame costs a full decode, so a
 * crafted header claiming thousands of frames would stall the fuzzer without
 * covering anything new. */
#define FUZZ_MAX_FRAMES 4

/* Pixels a frame may have and still be rendered. Most of a large image's
 * time goes on decoding many groups that all take the same paths, so past
 * this the input buys one execution where a small one buys hundreds; its
 * headers are still walked. 2 Mpx still spans 32 groups and 2 LF groups. */
#define FUZZ_MAX_PIXELS ((uint64_t)2 << 20)

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

    ctx = fuzz_ctx_new();
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
    if ((uint64_t)info.width * (uint64_t)info.height > FUZZ_MAX_PIXELS) {
        for (i = 0; i < nframes; i++) {
            jxl_frame_info finfo;
            (void)jxl_doc_frame_info(doc, i, &finfo);
        }
        jxl_doc_close(doc);
        jxl_ctx_free(ctx);
        return 0;
    }
    /* At most two decodes per input. Rendering a frame walks (decodes)
     * every frame before it, and each render starts over, so rendering
     * frames 0..3 one by one cost ten frame walks -- and a 145-byte input
     * declaring five layered 1-Mpx frames ran past the 25 s timeout under
     * the sanitizers while libjxl, decoding it once, took half a second.
     * Frame 0 goes through one of the two render entry points, chosen per
     * input; an animation adds its fourth (or last) frame, whose walk covers
     * the blending and reference slots of the ones before it. */
    for (i = 0; i < nframes; i++) {
        jxl_frame_info finfo;
        (void)jxl_doc_frame_info(doc, i, &finfo);
    }
    if (flags & 0x40) {
        /* The caller-buffer path, which is how SumatraPDF renders: the
         * geometry comes from jxl_frame_render_info and the pixels go
         * straight into a buffer of exactly that size, so a row written one
         * sample too long is a heap overflow the sanitizer sees. The stride
         * gets a few bytes of padding on some inputs to cover
         * stride != width * bpp. */
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
    } else {
        /* NATIVE resolves from the image; RGBA32 forces the conversion path
         * (gray->rgb, 16->8, alpha attach) instead. */
        jxl_image *img = jxl_frame_render(doc, 0, (flags & 8) ? JXLDEC_FORMAT_RGBA32
                                                              : JXLDEC_FORMAT_NATIVE);
        if (img) jxl_image_destroy(ctx, img);
    }
    if (nframes > 1) {
        jxl_image *img = jxl_frame_render(doc, nframes - 1,
                                          (flags & 0x80) ? JXLDEC_FORMAT_NATIVE
                                                         : JXLDEC_FORMAT_RGBA32);
        if (img) jxl_image_destroy(ctx, img);
    }

    jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    return 0;
}
