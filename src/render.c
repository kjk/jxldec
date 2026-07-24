/* render.c -- frame decoding and pixel output.
 *
 * Owns the decode loop over a document's frames, the conversion of the
 * decoded float image buffers to the caller's output format, and the
 * orientation fixup.
 */
#include "jxl_internal.h"

int jxl_doc_frame_count(jxl_doc *doc) {
    if (!doc) return 0;
    /* Animations are a chain of frames; a still image is a single one. Until
       the frame index is built lazily we report at least one. */
    return 1;
}

int jxl_doc_frame_info(jxl_doc *doc, int frame_no, jxl_frame_info *info) {
    if (!doc || !info || frame_no != 0) return -1;
    memset(info, 0, sizeof(*info));
    info->tps_numerator = (int)doc->meta.animation.tps_numerator;
    info->tps_denominator = (int)doc->meta.animation.tps_denominator;
    info->is_last = 1;
    return 0;
}

int jxl_frame_render_info(jxl_doc *doc, int frame_no, jxl_format fmt,
                          jxl_render_info *info) {
    jxl_image_info ii;
    if (!doc || !info) return -1;
    if (jxl_doc_info(doc, &ii) != 0) return -1;
    (void)frame_no;
    if (fmt == JXL_FORMAT_NATIVE) {
        int wide = ii.bits_per_sample > 8 || ii.exponent_bits > 0;
        int gray = ii.num_color_channels == 1;
        int alpha = ii.alpha_bits > 0;
        if (gray) {
            fmt = wide ? (alpha ? JXL_FORMAT_GRAYA16 : JXL_FORMAT_GRAY16)
                       : (alpha ? JXL_FORMAT_GRAYA8 : JXL_FORMAT_GRAY8);
        } else {
            fmt = wide ? (alpha ? JXL_FORMAT_RGBA64 : JXL_FORMAT_RGB48)
                       : (alpha ? JXL_FORMAT_RGBA32 : JXL_FORMAT_RGB24);
        }
    }
    info->width = ii.width;
    info->height = ii.height;
    info->format = fmt;
    return 0;
}

jxl_image *jxl_frame_render(jxl_doc *doc, int frame_no, jxl_format fmt) {
    (void)frame_no;
    (void)fmt;
    if (doc) JXL_ERR(doc->ctx, "render: not implemented yet");
    return NULL;
}

int jxl_frame_render_into(jxl_doc *doc, int frame_no, jxl_format fmt,
                          uint8_t *dst, int stride) {
    (void)frame_no;
    (void)fmt;
    (void)dst;
    (void)stride;
    if (doc) JXL_ERR(doc->ctx, "render: not implemented yet");
    return -1;
}
