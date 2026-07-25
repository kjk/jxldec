/* frame.c -- frame header, restoration filter parameters and the table of
 * contents that indexes a frame's sections.
 *
 * Frames are the unit of coding: a still image is one frame, an animation is
 * a sequence of them, and reference/LF frames are non-displayed helpers.
 */
#include "jxl_internal.h"

static uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/* ----- geometry helpers ----- */

uint32_t jxl_frame_sample_width(const jxl_frame_header *fh, uint32_t upsampling) {
    uint32_t w = fh->width;
    if (upsampling > 1) w = div_ceil(w, upsampling);
    if (fh->lf_level > 0) {
        uint32_t sh = 3 * fh->lf_level;
        w = (w + (1u << sh) - 1) >> sh;
    }
    return w;
}

uint32_t jxl_frame_sample_height(const jxl_frame_header *fh, uint32_t upsampling) {
    uint32_t h = fh->height;
    if (upsampling > 1) h = div_ceil(h, upsampling);
    if (fh->lf_level > 0) {
        uint32_t sh = 3 * fh->lf_level;
        h = (h + (1u << sh) - 1) >> sh;
    }
    return h;
}

uint32_t jxl_frame_color_width(const jxl_frame_header *fh) {
    return jxl_frame_sample_width(fh, fh->upsampling);
}

uint32_t jxl_frame_color_height(const jxl_frame_header *fh) {
    return jxl_frame_sample_height(fh, fh->upsampling);
}

uint32_t jxl_frame_group_dim(const jxl_frame_header *fh) {
    return 128u << fh->group_size_shift;
}

uint32_t jxl_frame_groups_per_row(const jxl_frame_header *fh) {
    return div_ceil(jxl_frame_color_width(fh), jxl_frame_group_dim(fh));
}

/* The largest subsampling shift over the three channels. The block grid is
   rounded up to a multiple of it so that every subsampled channel still has a
   whole number of blocks. */
static uint32_t max_shift(const jxl_frame_header *fh, int vertical) {
    uint32_t m = 0;
    int i;
    for (i = 0; i < 3; i++) {
        uint32_t mode = fh->jpeg_upsampling[i];
        uint32_t s = vertical ? (mode == 1 || mode == 3) : (mode == 1 || mode == 2);
        if (s > m) m = s;
    }
    return m;
}

uint32_t jxl_frame_blocks_w(const jxl_frame_header *fh) {
    uint32_t sh = max_shift(fh, 0);
    return div_ceil(jxl_frame_color_width(fh), 8u << sh) << sh;
}

uint32_t jxl_frame_blocks_h(const jxl_frame_header *fh) {
    uint32_t sh = max_shift(fh, 1);
    return div_ceil(jxl_frame_color_height(fh), 8u << sh) << sh;
}

uint32_t jxl_frame_lf_groups_per_row(const jxl_frame_header *fh) {
    return div_ceil(jxl_frame_blocks_w(fh), jxl_frame_group_dim(fh));
}

uint32_t jxl_frame_num_groups(const jxl_frame_header *fh) {
    uint32_t dim = jxl_frame_group_dim(fh);
    return div_ceil(jxl_frame_color_width(fh), dim) *
           div_ceil(jxl_frame_color_height(fh), dim);
}

uint32_t jxl_frame_num_lf_groups(const jxl_frame_header *fh) {
    uint32_t dim = jxl_frame_group_dim(fh);
    return div_ceil(jxl_frame_blocks_w(fh), dim) *
           div_ceil(jxl_frame_blocks_h(fh), dim);
}

/* ----- frame header ----- */

static int frame_type_is_normal(jxl_frame_type t) {
    return t == JXL_FRAME_REGULAR || t == JXL_FRAME_SKIP_PROGRESSIVE;
}

/* A frame "resets the canvas" when it replaces everything under it, which is
   what decides whether blending info carries a source index. */
static int test_full_image(const jxl_frame_header *fh, const jxl_size_header *sz) {
    int64_t right, bottom;
    if (fh->x0 > 0 || fh->y0 > 0) return 0;
    right = (int64_t)fh->x0 + fh->width;
    bottom = (int64_t)fh->y0 + fh->height;
    return right >= (int64_t)sz->width && bottom >= (int64_t)sz->height;
}

static int computes_resets_canvas(jxl_blend_mode mode,
                                  const jxl_frame_header *fh,
                                  const jxl_size_header *sz) {
    return mode == JXL_BLEND_REPLACE && (!fh->have_crop || test_full_image(fh, sz));
}

static void read_blending_info(jxl_br *br, jxl_blending_info *bi,
                               int have_extra, const jxl_blend_mode *first_mode,
                               const jxl_frame_header *fh,
                               const jxl_size_header *sz) {
    uint32_t m = jxl_br_u32(br, 0, 0, 1, 0, 2, 0, 3, 2);
    bi->mode = (jxl_blend_mode)m;
    bi->alpha_channel = 0;
    bi->clamp = 0;
    bi->source = 0;
    if (m > JXL_BLEND_MUL) {
        br->err = 1;
        return;
    }
    if (have_extra && (m == JXL_BLEND_BLEND || m == JXL_BLEND_MULADD)) {
        bi->alpha_channel = jxl_br_u32(br, 0, 0, 1, 0, 2, 0, 3, 3);
    }
    if ((have_extra && (m == JXL_BLEND_BLEND || m == JXL_BLEND_MULADD)) ||
        m == JXL_BLEND_MUL) {
        bi->clamp = jxl_br_bool(br);
    }
    if (!computes_resets_canvas(first_mode ? *first_mode : bi->mode, fh, sz)) {
        bi->source = jxl_br_read(br, 2);
    }
}

static void read_gabor(jxl_br *br, jxl_gabor *g) {
    int i, j;
    g->enabled = jxl_br_bool(br);
    for (i = 0; i < 3; i++) {
        g->weights[i][0] = 0.115169525f;
        g->weights[i][1] = 0.061248592f;
    }
    if (!g->enabled) return;
    if (!jxl_br_bool(br)) return;   /* default weights */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 2; j++) g->weights[i][j] = jxl_br_f16(br);
    }
}

static const float epf_sharp_lut_default[8] = {
    0.0f, 1.0f / 7.0f, 2.0f / 7.0f, 3.0f / 7.0f,
    4.0f / 7.0f, 5.0f / 7.0f, 6.0f / 7.0f, 1.0f
};

static void read_epf(jxl_br *br, jxl_epf *e, jxl_encoding encoding) {
    int i;
    memcpy(e->sharp_lut, epf_sharp_lut_default, sizeof(e->sharp_lut));
    e->channel_scale[0] = 40.0f;
    e->channel_scale[1] = 5.0f;
    e->channel_scale[2] = 3.5f;
    e->quant_mul = 0.46f;
    e->pass0_sigma_scale = 0.9f;
    e->pass2_sigma_scale = 6.5f;
    e->border_sad_mul = 2.0f / 3.0f;
    e->sigma_for_modular = 1.0f;

    e->iters = jxl_br_read(br, 2);
    e->enabled = e->iters != 0;
    if (!e->enabled) return;

    if (encoding == JXL_ENC_VARDCT && jxl_br_bool(br)) {
        for (i = 0; i < 8; i++) e->sharp_lut[i] = jxl_br_f16(br);
    }
    if (jxl_br_bool(br)) {
        for (i = 0; i < 3; i++) e->channel_scale[i] = jxl_br_f16(br);
        jxl_br_read(br, 32);   /* reserved, ignored */
    }
    if (jxl_br_bool(br)) {
        if (encoding == JXL_ENC_VARDCT) e->quant_mul = jxl_br_f16(br);
        e->pass0_sigma_scale = jxl_br_f16(br);
        e->pass2_sigma_scale = jxl_br_f16(br);
        e->border_sad_mul = jxl_br_f16(br);
    }
    if (encoding == JXL_ENC_MODULAR) e->sigma_for_modular = jxl_br_f16(br);
}

static void read_extensions_frame(jxl_br *br) {
    uint64_t bits = jxl_br_u64(br);
    uint64_t lens[64];
    int n = 0, i;
    for (i = 0; i < 64; i++) {
        if ((bits >> i) & 1) lens[n++] = jxl_br_u64(br);
    }
    for (i = 0; i < n; i++) jxl_br_skip(br, (size_t)lens[i]);
}


int jxl_read_frame_header(jxl_ctx *ctx, jxl_br *br, const jxl_size_header *size,
                          const jxl_image_metadata *meta, jxl_frame_header *fh) {
    int all_default;
    uint32_t i;
    uint32_t nec = meta->num_extra;

    memset(fh, 0, sizeof(*fh));
    fh->frame_type = JXL_FRAME_REGULAR;
    fh->encoding = JXL_ENC_VARDCT;
    fh->upsampling = 1;
    fh->group_size_shift = 1;
    fh->passes.num_passes = 1;
    fh->width = size->width;
    fh->height = size->height;
    fh->is_last = 1;
    fh->x_qm_scale = 3;
    fh->b_qm_scale = 2;
    fh->gab.enabled = 1;
    for (i = 0; i < 3; i++) {
        fh->gab.weights[i][0] = 0.115169525f;
        fh->gab.weights[i][1] = 0.061248592f;
    }
    fh->epf.enabled = 1;
    fh->epf.iters = 2;
    memcpy(fh->epf.sharp_lut, epf_sharp_lut_default, sizeof(fh->epf.sharp_lut));
    fh->epf.channel_scale[0] = 40.0f;
    fh->epf.channel_scale[1] = 5.0f;
    fh->epf.channel_scale[2] = 3.5f;
    fh->epf.quant_mul = 0.46f;
    fh->epf.pass0_sigma_scale = 0.9f;
    fh->epf.pass2_sigma_scale = 6.5f;
    fh->epf.border_sad_mul = 2.0f / 3.0f;
    fh->epf.sigma_for_modular = 1.0f;

    if (nec) {
        fh->ec_upsampling = (uint32_t *)jxl_calloc(ctx, nec, sizeof(uint32_t));
        fh->ec_blending = (jxl_blending_info *)jxl_calloc(ctx, nec,
                                                          sizeof(jxl_blending_info));
        if (!fh->ec_upsampling || !fh->ec_blending) return -1;
        for (i = 0; i < nec; i++) fh->ec_upsampling[i] = 1;
    }

    all_default = jxl_br_bool(br);
    if (!all_default) {
        fh->frame_type = (jxl_frame_type)jxl_br_read(br, 2);
        fh->encoding = (jxl_encoding)jxl_br_read(br, 1);
        fh->flags = jxl_br_u64(br);
        if (!meta->xyb_encoded) fh->do_ycbcr = jxl_br_bool(br);
    }

    fh->encoded_color_channels =
        (fh->encoding == JXL_ENC_MODULAR && !fh->do_ycbcr && !meta->xyb_encoded &&
         meta->colour.colour_space == JXL_CS_GRAY) ? 1 : 3;

    if (fh->do_ycbcr && !(fh->flags & JXL_FF_USE_LF_FRAME)) {
        for (i = 0; i < 3; i++) fh->jpeg_upsampling[i] = jxl_br_read(br, 2);
    }
    if (!all_default && !(fh->flags & JXL_FF_USE_LF_FRAME)) {
        fh->upsampling = jxl_br_u32(br, 1, 0, 2, 0, 4, 0, 8, 0);
        for (i = 0; i < nec; i++) {
            fh->ec_upsampling[i] = jxl_br_u32(br, 1, 0, 2, 0, 4, 0, 8, 0);
        }
    }
    if (fh->encoding == JXL_ENC_MODULAR) {
        fh->group_size_shift = jxl_br_read(br, 2);
    }
    if (!all_default && meta->xyb_encoded && fh->encoding == JXL_ENC_VARDCT) {
        fh->x_qm_scale = jxl_br_read(br, 3);
        fh->b_qm_scale = jxl_br_read(br, 3);
    } else if (!(meta->xyb_encoded && fh->encoding == JXL_ENC_VARDCT)) {
        fh->x_qm_scale = 2;
        fh->b_qm_scale = 2;
    }

    if (!all_default && fh->frame_type != JXL_FRAME_REFERENCE_ONLY) {
        fh->passes.num_passes = jxl_br_u32(br, 1, 0, 2, 0, 3, 0, 4, 3);
        if (fh->passes.num_passes > 11) {
            JXL_ERR(ctx, "frame: too many passes (%u)",
                    (unsigned)fh->passes.num_passes);
            return -1;
        }
        if (fh->passes.num_passes != 1) {
            fh->passes.num_ds = jxl_br_u32(br, 0, 0, 1, 0, 2, 0, 3, 1);
            if (fh->passes.num_ds > 4) {
                JXL_ERR(ctx, "frame: too many downsampling levels");
                return -1;
            }
            for (i = 0; i + 1 < fh->passes.num_passes; i++) {
                fh->passes.shift[i] = jxl_br_read(br, 2);
            }
            for (i = 0; i < fh->passes.num_ds; i++) {
                fh->passes.downsample[i] = jxl_br_u32(br, 1, 0, 2, 0, 4, 0, 8, 0);
            }
            for (i = 0; i < fh->passes.num_ds; i++) {
                fh->passes.last_pass[i] = jxl_br_u32(br, 0, 0, 1, 0, 2, 0, 0, 3);
            }
        }
    }

    if (fh->frame_type == JXL_FRAME_LF) {
        fh->lf_level = jxl_br_read(br, 2) + 1;
    } else if (!all_default) {
        fh->have_crop = jxl_br_bool(br);
    }
    if (fh->have_crop) {
        if (fh->frame_type != JXL_FRAME_REFERENCE_ONLY) {
            fh->x0 = jxl_unpack_signed(
                jxl_br_u32(br, 0, 8, 256, 11, 2304, 14, 18688, 30));
            fh->y0 = jxl_unpack_signed(
                jxl_br_u32(br, 0, 8, 256, 11, 2304, 14, 18688, 30));
        }
        fh->width = jxl_br_u32(br, 0, 8, 256, 11, 2304, 14, 18688, 30);
        fh->height = jxl_br_u32(br, 0, 8, 256, 11, 2304, 14, 18688, 30);
    }

    fh->blending.mode = JXL_BLEND_REPLACE;
    if (!all_default && frame_type_is_normal(fh->frame_type)) {
        jxl_blend_mode first;
        read_blending_info(br, &fh->blending, nec != 0, NULL, fh, size);
        first = fh->blending.mode;
        for (i = 0; i < nec; i++) {
            read_blending_info(br, &fh->ec_blending[i], nec != 0, &first, fh, size);
        }
        if (meta->have_animation) {
            fh->duration = jxl_br_u32(br, 0, 0, 1, 0, 0, 8, 0, 32);
            if (meta->animation.have_timecodes) fh->timecode = jxl_br_read(br, 32);
        }
        fh->is_last = jxl_br_bool(br);
    } else {
        fh->is_last = (fh->frame_type == JXL_FRAME_REGULAR);
    }

    if (!all_default && fh->frame_type != JXL_FRAME_LF && !fh->is_last) {
        fh->save_as_reference = jxl_br_read(br, 2);
    }

    fh->resets_canvas = computes_resets_canvas(fh->blending.mode, fh, size);
    fh->save_before_ct = !frame_type_is_normal(fh->frame_type);
    if (!all_default) {
        int cond = (fh->frame_type == JXL_FRAME_REFERENCE_ONLY) ||
                   (fh->resets_canvas && !fh->is_last &&
                    (fh->duration == 0 || fh->save_as_reference != 0) &&
                    fh->frame_type != JXL_FRAME_LF);
        if (cond) fh->save_before_ct = jxl_br_bool(br);
        fh->name = jxl_read_name(ctx, br);
        if (!jxl_br_bool(br)) {   /* RestorationFilter all_default */
            read_gabor(br, &fh->gab);
            read_epf(br, &fh->epf, fh->encoding);
            read_extensions_frame(br);
        }
        read_extensions_frame(br);
    }

    if (br->err) {
        JXL_ERR(ctx, "frame: truncated header");
        return -1;
    }
    if (fh->width == 0 || fh->height == 0) {
        JXL_ERR(ctx, "frame: zero-sized frame");
        return -1;
    }
    return 0;
}

void jxl_frame_header_free(jxl_ctx *ctx, jxl_frame_header *fh) {
    if (!fh) return;
    jxl_free(ctx, fh->ec_upsampling);
    jxl_free(ctx, fh->ec_blending);
    jxl_free(ctx, fh->name);
    fh->ec_upsampling = NULL;
    fh->ec_blending = NULL;
    fh->name = NULL;
}

/* ----- table of contents ----- */

uint32_t jxl_toc_index(const jxl_toc *toc, jxl_toc_kind kind, uint32_t pass_idx,
                       uint32_t group_idx) {
    if (toc->count <= 1) return 0;
    switch (kind) {
        case JXL_TOC_LF_GLOBAL: return 0;
        case JXL_TOC_LF_GROUP: return 1 + group_idx;
        case JXL_TOC_HF_GLOBAL: return 1 + toc->num_lf_groups;
        case JXL_TOC_GROUP_PASS:
            return 1 + toc->num_lf_groups + 1 + pass_idx * toc->num_groups +
                   group_idx;
        default: return 0;
    }
}

int jxl_read_toc(jxl_ctx *ctx, jxl_br *br, const jxl_frame_header *fh,
                 jxl_toc *toc) {
    uint32_t num_groups = jxl_frame_num_groups(fh);
    uint32_t num_lf_groups = jxl_frame_num_lf_groups(fh);
    uint32_t num_passes = fh->passes.num_passes;
    uint32_t entry_count;
    uint32_t *perm = NULL;
    uint32_t *sizes = NULL;
    uint32_t i;
    size_t acc;
    int permutated;
    int rc = -1;

    memset(toc, 0, sizeof(*toc));
    entry_count = (num_groups == 1 && num_passes == 1)
                      ? 1
                      : 1 + num_lf_groups + 1 + num_groups * num_passes;
    if (entry_count > 65536) {
        JXL_ERR(ctx, "toc: too many entries (%u)", (unsigned)entry_count);
        return -1;
    }
    toc->count = entry_count;
    toc->num_groups = num_groups;
    toc->num_lf_groups = num_lf_groups;
    toc->num_passes = num_passes;

    permutated = jxl_br_bool(br);
    if (permutated) {
        jxl_dec dec;
        perm = (uint32_t *)jxl_calloc(ctx, entry_count, sizeof(uint32_t));
        if (!perm) return -1;
        if (jxl_dec_init(ctx, &dec, br, 8) != 0) {
            jxl_free(ctx, perm);
            return -1;
        }
        jxl_dec_begin(&dec, br);
        if (jxl_read_permutation(ctx, &dec, br, entry_count, 0, perm) != 0 ||
            jxl_dec_finalize(&dec) != 0) {
            JXL_ERR(ctx, "toc: bad permutation");
            jxl_dec_free(&dec);
            jxl_free(ctx, perm);
            return -1;
        }
        jxl_dec_free(&dec);
    }

    jxl_br_zero_pad_to_byte(br);
    sizes = (uint32_t *)jxl_calloc(ctx, entry_count, sizeof(uint32_t));
    if (!sizes) goto done;
    for (i = 0; i < entry_count; i++) {
        sizes[i] = jxl_br_u32(br, 0, 10, 1024, 14, 17408, 22, 4211712, 30);
    }
    jxl_br_zero_pad_to_byte(br);
    if (br->err) {
        JXL_ERR(ctx, "toc: truncated");
        goto done;
    }

    toc->entries = (jxl_toc_entry *)jxl_calloc(ctx, entry_count,
                                               sizeof(jxl_toc_entry));
    if (!toc->entries) goto done;

    /* sizes[] is in bitstream order; perm maps original index -> bitstream. */
    acc = br->bits_read / 8;
    toc->end_off = acc;
    if (permutated) {
        size_t *offs = (size_t *)jxl_calloc(ctx, entry_count, sizeof(size_t));
        if (!offs) goto done;
        for (i = 0; i < entry_count; i++) {
            offs[i] = acc;
            acc += sizes[i];
            toc->total_size += sizes[i];
        }
        for (i = 0; i < entry_count; i++) {
            uint32_t b = perm[i];
            if (b >= entry_count) {
                jxl_free(ctx, offs);
                JXL_ERR(ctx, "toc: permutation out of range");
                goto done;
            }
            toc->entries[i].offset = offs[b];
            toc->entries[i].size = sizes[b];
        }
        jxl_free(ctx, offs);
    } else {
        for (i = 0; i < entry_count; i++) {
            toc->entries[i].offset = acc;
            toc->entries[i].size = sizes[i];
            acc += sizes[i];
            toc->total_size += sizes[i];
        }
    }
    rc = 0;

done:
    jxl_free(ctx, sizes);
    jxl_free(ctx, perm);
    return rc;
}

void jxl_toc_free(jxl_ctx *ctx, jxl_toc *toc) {
    if (!toc) return;
    jxl_free(ctx, toc->entries);
    toc->entries = NULL;
    toc->count = 0;
}
