/* doc.c -- top-level document: open, metadata queries, render entry points. */
#include "jxl_internal.h"

#include <stdlib.h>

jxl_doc *jxl_doc_open(jxl_ctx *ctx, const uint8_t *data, size_t len) {
    jxl_doc *doc;
    jxl_br br;

    if (!ctx || !data) return NULL;
    doc = (jxl_doc *)jxl_calloc(ctx, 1, sizeof(*doc));
    if (!doc) return NULL;
    doc->ctx = ctx;
    doc->data = data;
    doc->len = len;

    if (jxl_container_parse(ctx, data, len, &doc->container) != 0) goto fail;

    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    if (jxl_read_image_header(ctx, &br, &doc->size, &doc->meta) != 0) goto fail;

    if (doc->meta.colour.want_icc) {
        if (jxl_read_icc(ctx, &br, &doc->icc, &doc->icc_len) != 0) goto fail;
    }
    jxl_br_zero_pad_to_byte(&br);
    if (br.err) {
        JXL_ERR(ctx, "codestream: truncated headers");
        goto fail;
    }
    doc->first_frame_bitpos = br.bits_read;
    doc->first_frame_off = br.bits_read / 8;
    return doc;

fail:
    jxl_doc_close(doc);
    return NULL;
}

void jxl_doc_close(jxl_doc *doc) {
    jxl_ctx *ctx;
    if (!doc) return;
    ctx = doc->ctx;
    jxl_image_metadata_free(ctx, &doc->meta);
    jxl_container_free(ctx, &doc->container);
    jxl_free(ctx, doc->icc);
    jxl_free(ctx, doc);
}

int jxl_doc_info(jxl_doc *doc, jxl_image_info *info) {
    const jxl_image_metadata *m;
    uint32_t w, h;

    if (!doc || !info) return -1;
    m = &doc->meta;
    memset(info, 0, sizeof(*info));

    if (doc->ctx->keep_orientation) {
        w = doc->size.width;
        h = doc->size.height;
    } else {
        jxl_apply_orientation_dims(m->orientation, doc->size.width,
                                   doc->size.height, &w, &h);
    }
    info->width = (int)w;
    info->height = (int)h;
    info->bits_per_sample = (int)m->bit_depth.bits_per_sample;
    info->exponent_bits = (int)m->bit_depth.exp_bits;
    info->num_color_channels = (m->colour.colour_space == JXL_CS_GRAY) ? 1 : 3;
    info->num_extra_channels = (int)m->num_extra;
    if (m->alpha_index >= 0) {
        const jxl_ec_info *a = &m->ec_info[m->alpha_index];
        info->alpha_bits = (int)a->bit_depth.bits_per_sample;
        info->alpha_premultiplied = a->alpha_associated;
    }
    info->have_animation = m->have_animation;
    info->num_frames = jxl_doc_frame_count(doc);
    info->orientation = (int)m->orientation;
    info->have_preview = m->have_preview;
    info->uses_original_profile = !m->xyb_encoded;
    info->color_space = m->colour.colour_space;
    if (m->have_intr_size) {
        uint32_t iw, ih;
        if (doc->ctx->keep_orientation) {
            iw = m->intrinsic.width;
            ih = m->intrinsic.height;
        } else {
            jxl_apply_orientation_dims(m->orientation, m->intrinsic.width,
                                       m->intrinsic.height, &iw, &ih);
        }
        info->intrinsic_width = (int)iw;
        info->intrinsic_height = (int)ih;
    } else {
        info->intrinsic_width = info->width;
        info->intrinsic_height = info->height;
    }
    return 0;
}

const uint8_t *jxl_doc_icc_profile(jxl_doc *doc, size_t *len) {
    if (!doc || !doc->icc) {
        if (len) *len = 0;
        return NULL;
    }
    if (len) *len = doc->icc_len;
    return doc->icc;
}

int jxl_decode_size(jxl_ctx *ctx, const uint8_t *data, size_t len,
                    int *width, int *height) {
    jxl_doc *doc = jxl_doc_open(ctx, data, len);
    jxl_image_info info;
    int rc;
    if (!doc) return -1;
    rc = jxl_doc_info(doc, &info);
    if (rc == 0) {
        if (width) *width = info.width;
        if (height) *height = info.height;
    }
    jxl_doc_close(doc);
    return rc;
}

jxl_image *jxl_decode(jxl_ctx *ctx, const uint8_t *data, size_t len,
                      jxl_format fmt) {
    jxl_doc *doc = jxl_doc_open(ctx, data, len);
    jxl_image *img;
    if (!doc) return NULL;
    img = jxl_frame_render(doc, 0, fmt);
    jxl_doc_close(doc);
    return img;
}
