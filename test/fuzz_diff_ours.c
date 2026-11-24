/* fuzz_diff_ours.c -- the part of the differential target (fuzz_diff.c) that
 * needs this decoder's internals. A file of its own because jxl_internal.h
 * and libjxl's headers cannot share a translation unit: both declare
 * JXL_BLEND_REPLACE, JXL_PRIMARIES_SRGB and friends.
 */
#include "jxl_internal.h"

int fuzz_diff_frames_exact(jxl_doc *doc);

/* Whether every frame stays on the integer path, where the two decoders
   must agree sample for sample: no XYB, no float samples, Modular only, no
   loop filters, upsampling, noise, splines, patches, cropping or blending.
   Read with the decoder's own frame-header walk, so it judges the frames
   this decoder actually decoded. */
int fuzz_diff_frames_exact(jxl_doc *doc) {
    jxl_ctx *ctx = doc->ctx;
    const jxl_image_metadata *meta = &doc->meta;
    jxl_br br;
    uint32_t e;
    int i, exact = 1;

    if (meta->xyb_encoded || meta->bit_depth.float_sample) return 0;
    for (e = 0; e < meta->num_extra; e++) {
        if (meta->ec_info[e].bit_depth.float_sample) return 0;
    }
    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    for (i = 0; i < 64 && exact; i++) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t end;
        int last;
        if (jxl_read_frame_header(ctx, &br, &doc->size, meta, &fh) != 0) {
            jxl_frame_header_free(ctx, &fh);
            return 0;
        }
        if (fh.encoding != JXL_ENC_MODULAR || fh.gab.enabled ||
            fh.epf.enabled || fh.upsampling != 1 || fh.have_crop ||
            (fh.flags & (JXL_FF_NOISE | JXL_FF_PATCHES | JXL_FF_SPLINES)) ||
            fh.blending.mode != JXL_BLEND_REPLACE)
            exact = 0;
        for (e = 0; e < meta->num_extra && exact; e++) {
            if (fh.ec_upsampling[e] != 1 ||
                (fh.ec_blending && fh.ec_blending[e].mode != JXL_BLEND_REPLACE))
                exact = 0;
        }
        last = fh.is_last;
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            return 0;
        }
        jxl_frame_header_free(ctx, &fh);
        end = toc.end_off + (size_t)toc.total_size;
        jxl_toc_free(ctx, &toc);
        if (last || end >= doc->container.cs_len) break;
        jxl_br_seek_byte(&br, end);
    }
    return exact;
}
