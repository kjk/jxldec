/* fuzz_headers.c -- everything up to the pixels, and nothing after.
 *
 * The container (boxes, jxlp concatenation), SizeHeader, ImageMetadata,
 * CustomTransformData, the ICC stream, and every frame's header and TOC
 * (including permuted TOCs, which run an entropy decoder and the Lehmer
 * code) -- what jxl_doc_open, jxl_doc_frame_count and jxl_doc_frame_info
 * read. No group is decoded, so an input costs microseconds where a render
 * costs milliseconds, and the mutations all land on the parsers.
 *
 *   bun cmd/fuzz.ts -target headers
 */
#include "fuzz_common.h"

/* Frame headers are cheap but an animation can claim millions of frames;
   past this the walk is the same loop again. */
#define FUZZ_MAX_FRAMES 64

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    jxl_ctx *ctx;
    jxl_doc *doc;
    jxl_image_info info;
    const uint8_t *icc;
    size_t icc_len = 0;
    int n, i, w, h;

    if (size < 2) return 0;
    ctx = fuzz_ctx_new();
    if (!ctx) return 0;

    (void)jxl_signature_check(data, size);
    (void)jxl_decode_size(ctx, data, size, &w, &h);

    doc = jxl_doc_open(ctx, data, size);
    if (doc) {
        if (jxl_doc_info(doc, &info) == 0) {
            icc = jxl_doc_icc_profile(doc, &icc_len);
            if (icc && icc_len) {
                volatile uint8_t sink = icc[icc_len - 1];
                (void)sink;
            }
            n = jxl_doc_frame_count(doc);
            if (n > FUZZ_MAX_FRAMES) n = FUZZ_MAX_FRAMES;
            for (i = 0; i < n; i++) {
                jxl_frame_info fi;
                jxl_render_info ri;
                (void)jxl_doc_frame_info(doc, i, &fi);
                (void)jxl_frame_render_info(doc, i, JXLDEC_FORMAT_NATIVE, &ri);
            }
        }
        jxl_doc_close(doc);
    }
    jxl_ctx_free(ctx);
    return 0;
}
