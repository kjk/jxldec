/* decode.c -- frame decoding: LfGlobal, the global Modular image, and the
 * per-group Modular streams, assembled into float planes.
 *
 * A frame's data is split into TOC sections. When the frame has a single
 * group and a single pass the TOC collapses to one section holding every
 * sub-stream back to back, so the section reader is simply "keep reading";
 * otherwise each sub-stream starts at its own byte offset.
 *
 * Modular streams are numbered so the MA tree can branch on which one it is
 * decoding: 0 is the global stream, 1 + lf_group is the LF coefficient
 * stream, 1 + num_lf_groups + lf_group is the LF group's modular stream,
 * and pass groups start at 1 + 3 * num_lf_groups + 17.
 */
#include "jxl_internal.h"

static uint32_t div_ceil32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/* ===================================================================== */
/* float planes                                                           */
/* ===================================================================== */

int jxl_fplane_alloc(jxl_ctx *ctx, jxl_fplane *p, uint32_t w, uint32_t h) {
    size_t total;
    if (!jxl_size_mul(w, h, &total)) return -1;
    p->data = (float *)jxl_calloc(ctx, total ? total : 1, sizeof(float));
    if (!p->data) return -1;
    p->w = w;
    p->h = h;
    p->stride = w;
    return 0;
}

int jxl_fimage_alloc(jxl_ctx *ctx, jxl_fimage *img, uint32_t nplane) {
    img->ctx = ctx;
    img->plane = (jxl_fplane *)jxl_calloc(ctx, nplane ? nplane : 1,
                                          sizeof(jxl_fplane));
    if (!img->plane) return -1;
    img->nplane = nplane;
    return 0;
}

void jxl_fimage_free(jxl_ctx *ctx, jxl_fimage *img) {
    uint32_t i;
    if (!img || !img->plane) return;
    for (i = 0; i < img->nplane; i++) jxl_free(ctx, img->plane[i].data);
    jxl_free(ctx, img->plane);
    img->plane = NULL;
    img->nplane = 0;
}

/* ===================================================================== */
/* section readers                                                        */
/* ===================================================================== */

typedef struct {
    const uint8_t *cs;
    size_t cs_len;
    const jxl_toc *toc;
    jxl_br br;        /* used directly when the TOC has a single section */
    int single;
} jxl_sections;

static void sections_init(jxl_sections *s, const uint8_t *cs, size_t cs_len,
                          const jxl_toc *toc) {
    s->cs = cs;
    s->cs_len = cs_len;
    s->toc = toc;
    s->single = toc->count <= 1;
    jxl_br_init(&s->br, cs, cs_len);
    if (s->single) jxl_br_seek_byte(&s->br, toc->entries[0].offset);
}

/* Returns the reader positioned at the start of the given section. With a
   single-section TOC every sub-stream continues in the same reader. */
static jxl_br *section_reader(jxl_sections *s, jxl_toc_kind kind,
                              uint32_t pass_idx, uint32_t group_idx) {
    uint32_t idx;
    if (s->single) return &s->br;
    idx = jxl_toc_index(s->toc, kind, pass_idx, group_idx);
    if (idx >= s->toc->count) return NULL;
    jxl_br_init(&s->br, s->cs, s->cs_len);
    jxl_br_seek_byte(&s->br, s->toc->entries[idx].offset);
    return &s->br;
}

static uint32_t section_size(jxl_sections *s, jxl_toc_kind kind,
                             uint32_t pass_idx, uint32_t group_idx) {
    uint32_t idx;
    if (s->single) return s->toc->entries[0].size;
    idx = jxl_toc_index(s->toc, kind, pass_idx, group_idx);
    if (idx >= s->toc->count) return 0;
    return s->toc->entries[idx].size;
}

/* ===================================================================== */
/* frame decoding                                                         */
/* ===================================================================== */

typedef struct {
    float m_x_lf, m_y_lf, m_b_lf;
} jxl_lf_dequant;

static void lf_dequant_read(jxl_br *br, jxl_lf_dequant *d) {
    d->m_x_lf = 1.0f / 32.0f;
    d->m_y_lf = 1.0f / 4.0f;
    d->m_b_lf = 1.0f / 2.0f;
    if (jxl_br_bool(br)) return;
    d->m_x_lf = jxl_br_f16(br);
    d->m_y_lf = jxl_br_f16(br);
    d->m_b_lf = jxl_br_f16(br);
}

/* Per-pass shift ranges: a channel with min(hshift, vshift) in
   [minshift, maxshift) belongs to that pass. */
typedef struct {
    int32_t minshift[16];
    int32_t maxshift[16];
} jxl_pass_shifts;

static void compute_pass_shifts(const jxl_frame_header *fh,
                                jxl_pass_shifts *out) {
    uint32_t i;
    int32_t maxshift = 3;
    for (i = 0; i < 16; i++) {
        out->minshift[i] = 0;
        out->maxshift[i] = 0;
    }
    for (i = 0; i < fh->passes.num_ds && i < 8; i++) {
        uint32_t lp = fh->passes.last_pass[i];
        int32_t minshift = 0;
        uint32_t d = fh->passes.downsample[i];
        while (d > 1) { minshift++; d >>= 1; }
        if (lp < 16) {
            out->minshift[lp] = minshift;
            out->maxshift[lp] = maxshift;
        }
        maxshift = minshift;
    }
    if (fh->passes.num_passes >= 1 && fh->passes.num_passes <= 16) {
        out->minshift[fh->passes.num_passes - 1] = 0;
        out->maxshift[fh->passes.num_passes - 1] = maxshift;
    }
}

static uint32_t pass_for_shift(const jxl_frame_header *fh,
                               const jxl_pass_shifts *ps, int32_t shift) {
    uint32_t i;
    for (i = 0; i < fh->passes.num_passes && i < 16; i++) {
        if (shift >= ps->minshift[i] && shift < ps->maxshift[i]) return i;
    }
    return fh->passes.num_passes - 1;
}

/* Group channel lists: one list per LF group and one per (pass, group). */
typedef struct {
    jxl_chanlist *lf;         /* num_lf_groups entries */
    jxl_chanlist *pass;       /* num_passes * num_groups entries */
    uint32_t num_lf, num_groups, num_passes;
} jxl_group_lists;

static void group_lists_free(jxl_ctx *ctx, jxl_group_lists *gl) {
    uint32_t i;
    if (gl->lf) {
        for (i = 0; i < gl->num_lf; i++) jxl_chanlist_free(ctx, &gl->lf[i]);
        jxl_free(ctx, gl->lf);
    }
    if (gl->pass) {
        for (i = 0; i < gl->num_passes * gl->num_groups; i++) {
            jxl_chanlist_free(ctx, &gl->pass[i]);
        }
        jxl_free(ctx, gl->pass);
    }
    memset(gl, 0, sizeof(*gl));
}

/* Tiles a channel into groups and appends each tile to the matching list. */
static int split_channel_into_groups(jxl_ctx *ctx, const jxl_mchan *ch,
                                     uint32_t gw, uint32_t gh, uint32_t nx,
                                     uint32_t ny, jxl_chanlist *lists,
                                     uint32_t nlists) {
    uint32_t gx, gy;
    if ((uint64_t)nx * ny > nlists) return -1;
    for (gy = 0; gy < ny; gy++) {
        for (gx = 0; gx < nx; gx++) {
            jxl_mchan tile;
            uint32_t x0 = gx * gw, y0 = gy * gh;
            uint32_t w = 0, h = 0;
            if (x0 < ch->w) w = JXL_MIN(gw, ch->w - x0);
            if (y0 < ch->h) h = JXL_MIN(gh, ch->h - y0);
            if (w == 0 || h == 0) continue;
            tile = *ch;
            tile.w = w;
            tile.h = h;
            tile.ow = w << (ch->hshift > 0 ? ch->hshift : 0);
            tile.oh = h << (ch->vshift > 0 ? ch->vshift : 0);
            tile.data = ch->data + (size_t)y0 * ch->stride + x0;
            if (jxl_chanlist_push(ctx, &lists[gy * nx + gx], &tile) != 0)
                return -1;
        }
    }
    return 0;
}

/* Decodes one group's modular sub-stream: it carries its own modular header
   (transforms + optionally a local MA tree) over the group's channels. */
static int decode_group_modular(jxl_ctx *ctx, jxl_br *br, jxl_chanlist *cl,
                                jxl_ma_config *global_ma, uint32_t group_dim,
                                uint32_t bit_depth, uint32_t stream_idx) {
    jxl_modular gm;
    jxl_chanlist tcl;
    int rc = -1;

    if (cl->n == 0) return 0;
    memset(&tcl, 0, sizeof(tcl));
    if (jxl_modular_init_over(ctx, &gm, br, cl->chans, cl->n, global_ma,
                              group_dim, bit_depth) != 0)
        goto done;
    if (jxl_modular_transform_channels(ctx, &gm, &tcl) != 0) goto done;
    if (jxl_modular_decode(ctx, &gm, &tcl, br, stream_idx) != 0) goto done;
    if (jxl_modular_inverse(ctx, &gm, &tcl) != 0) goto done;
    rc = 0;

done:
    jxl_chanlist_free(ctx, &tcl);
    jxl_modular_free(ctx, &gm);
    return rc;
}

int jxl_frame_decode(jxl_ctx *ctx, jxl_doc *doc, const jxl_frame_header *fh,
                     const jxl_toc *toc, jxl_fimage *out) {
    const jxl_image_metadata *meta = &doc->meta;
    jxl_sections sec;
    jxl_br *br;
    jxl_lf_dequant lf_dequant;
    jxl_ma_config global_ma;
    int has_global_ma = 0;
    jxl_modular gmod;
    jxl_chanlist gcl;
    jxl_chanlist prefix;
    jxl_mchan_spec *specs = NULL;
    jxl_group_lists gl;
    jxl_pass_shifts pshifts;
    uint32_t nspecs = 0, i, split;
    uint32_t color_w, color_h, group_dim, group_dim_shift;
    uint32_t num_lf_groups, num_groups, num_passes;
    uint32_t color_upsampling_shift;
    uint32_t bits;
    int rc = -1;

    memset(&global_ma, 0, sizeof(global_ma));
    memset(&gmod, 0, sizeof(gmod));
    memset(&gcl, 0, sizeof(gcl));
    memset(&prefix, 0, sizeof(prefix));
    memset(&gl, 0, sizeof(gl));
    memset(out, 0, sizeof(*out));

    if (fh->encoding != JXL_ENC_MODULAR) {
        JXL_ERR(ctx, "frame: VarDCT is not implemented yet");
        return -1;
    }
    if (fh->flags & (JXL_FF_PATCHES | JXL_FF_SPLINES | JXL_FF_NOISE)) {
        JXL_ERR(ctx, "frame: patches/splines/noise are not implemented yet");
        return -1;
    }

    color_w = jxl_frame_color_width(fh);
    color_h = jxl_frame_color_height(fh);
    group_dim = jxl_frame_group_dim(fh);
    group_dim_shift = 7 + fh->group_size_shift;
    num_lf_groups = jxl_frame_num_lf_groups(fh);
    num_groups = jxl_frame_num_groups(fh);
    num_passes = fh->passes.num_passes;
    bits = meta->bit_depth.bits_per_sample;

    sections_init(&sec, doc->container.cs, doc->container.cs_len, toc);
    br = section_reader(&sec, JXL_TOC_LF_GLOBAL, 0, 0);
    if (!br) return -1;

    lf_dequant_read(br, &lf_dequant);

    /* ----- global modular ----- */
    if (jxl_br_bool(br)) {
        uint64_t num_channels = (uint64_t)fh->encoded_color_channels + meta->num_extra;
        uint64_t limit = 1024 + (uint64_t)fh->width * fh->height * num_channels / 16;
        if (limit > (1u << 22)) limit = 1u << 22;
        if (jxl_ma_config_read(ctx, br, &global_ma, (size_t)limit) != 0) goto done;
        has_global_ma = 1;
    }

    nspecs = (uint32_t)fh->encoded_color_channels + meta->num_extra;
    specs = (jxl_mchan_spec *)jxl_calloc(ctx, nspecs ? nspecs : 1,
                                         sizeof(jxl_mchan_spec));
    if (!specs) goto done;
    for (i = 0; i < (uint32_t)fh->encoded_color_channels; i++) {
        specs[i].w = color_w;
        specs[i].h = color_h;
        specs[i].hshift = 0;
        specs[i].vshift = 0;
    }
    color_upsampling_shift = 0;
    {
        uint32_t u = fh->upsampling;
        while (u > 1) { color_upsampling_shift++; u >>= 1; }
    }
    for (i = 0; i < meta->num_extra; i++) {
        uint32_t ec_shift = 0;
        uint32_t u = fh->ec_upsampling[i];
        int32_t actual;
        while (u > 1) { ec_shift++; u >>= 1; }
        actual = (int32_t)(ec_shift + meta->ec_info[i].dim_shift) -
                 (int32_t)color_upsampling_shift;
        if (actual < 0) actual = 0;
        specs[fh->encoded_color_channels + i].w = color_w;
        specs[fh->encoded_color_channels + i].h = color_h;
        specs[fh->encoded_color_channels + i].hshift = actual;
        specs[fh->encoded_color_channels + i].vshift = actual;
    }

    if (jxl_modular_init(ctx, &gmod, br, specs, nspecs,
                         has_global_ma ? &global_ma : NULL, group_dim,
                         bits) != 0)
        goto done;
    if (jxl_modular_transform_channels(ctx, &gmod, &gcl) != 0) goto done;

    /* Channels small enough to live in the global stream come first. */
    split = 0;
    while (split < gcl.n) {
        const jxl_mchan *ch = &gcl.chans[split];
        if (split < gcl.nb_meta || (ch->w <= group_dim && ch->h <= group_dim)) {
            split++;
        } else {
            break;
        }
    }
    prefix.chans = gcl.chans;
    prefix.n = split;
    prefix.cap = 0;    /* borrowed: never freed */
    if (jxl_modular_decode(ctx, &gmod, &prefix, br, 0) != 0) goto done;

    /* ----- group channel lists ----- */
    compute_pass_shifts(fh, &pshifts);
    gl.num_lf = num_lf_groups;
    gl.num_groups = num_groups;
    gl.num_passes = num_passes;
    gl.lf = (jxl_chanlist *)jxl_calloc(ctx, num_lf_groups ? num_lf_groups : 1,
                                       sizeof(jxl_chanlist));
    gl.pass = (jxl_chanlist *)jxl_calloc(
        ctx, (size_t)(num_passes ? num_passes : 1) * (num_groups ? num_groups : 1),
        sizeof(jxl_chanlist));
    if (!gl.lf || !gl.pass) goto done;

    for (i = split; i < gcl.n; i++) {
        const jxl_mchan *ch = &gcl.chans[i];
        int hshift = ch->hshift, vshift = ch->vshift;
        if (hshift < 0 || vshift < 0) {
            JXL_ERR(ctx, "frame: unshiftable channel outside the global stream");
            goto done;
        }
        if (hshift < 3 || vshift < 3) {
            int32_t shift = hshift < vshift ? hshift : vshift;
            uint32_t pass_idx = pass_for_shift(fh, &pshifts, shift);
            uint32_t gw = group_dim >> hshift;
            uint32_t gh = group_dim >> vshift;
            uint32_t nx = (ch->ow + group_dim - 1) >> group_dim_shift;
            uint32_t ny = (ch->oh + group_dim - 1) >> group_dim_shift;
            if (gw == 0 || gh == 0) {
                JXL_ERR(ctx, "frame: channel shift too large for the group size");
                goto done;
            }
            if (pass_idx >= num_passes) goto done;
            if (split_channel_into_groups(ctx, ch, gw, gh, nx, ny,
                                          gl.pass + (size_t)pass_idx * num_groups,
                                          num_groups) != 0)
                goto done;
        } else {
            uint32_t gw = group_dim >> (hshift - 3);
            uint32_t gh = group_dim >> (vshift - 3);
            uint32_t nx = (ch->ow + (group_dim << 3) - 1) >> (group_dim_shift + 3);
            uint32_t ny = (ch->oh + (group_dim << 3) - 1) >> (group_dim_shift + 3);
            if (gw == 0 || gh == 0) {
                JXL_ERR(ctx, "frame: channel shift too large for the LF group size");
                goto done;
            }
            if (split_channel_into_groups(ctx, ch, gw, gh, nx, ny, gl.lf,
                                          num_lf_groups) != 0)
                goto done;
        }
    }

    /* ----- LF groups ----- */
    for (i = 0; i < num_lf_groups; i++) {
        uint32_t sz = section_size(&sec, JXL_TOC_LF_GROUP, 0, i);
        if (sz == 0 && !sec.single) continue;
        br = section_reader(&sec, JXL_TOC_LF_GROUP, 0, i);
        if (!br) goto done;
        if (decode_group_modular(ctx, br, &gl.lf[i],
                                 has_global_ma ? &global_ma : NULL, group_dim,
                                 bits, 1 + num_lf_groups + i) != 0)
            goto done;
    }

    /* ----- pass groups ----- */
    {
        uint32_t p, g;
        for (p = 0; p < num_passes; p++) {
            for (g = 0; g < num_groups; g++) {
                jxl_chanlist *cl = &gl.pass[(size_t)p * num_groups + g];
                uint32_t sz = section_size(&sec, JXL_TOC_GROUP_PASS, p, g);
                if (sz == 0 && !sec.single) continue;
                br = section_reader(&sec, JXL_TOC_GROUP_PASS, p, g);
                if (!br) goto done;
                if (decode_group_modular(ctx, br, cl,
                                         has_global_ma ? &global_ma : NULL,
                                         group_dim, bits,
                                         1 + 3 * num_lf_groups + 17 +
                                             p * num_groups + g) != 0)
                    goto done;
            }
        }
    }

    /* ----- undo the global transforms ----- */
    if (jxl_modular_inverse(ctx, &gmod, &gcl) != 0) goto done;

    /* ----- convert to float planes ----- */
    {
        uint32_t ncolor = (uint32_t)fh->encoded_color_channels;
        uint32_t nplane = gmod.nbase;
        float scale = 1.0f;
        if (jxl_fimage_alloc(ctx, out, nplane) != 0) goto done;
        out->ncolor = ncolor;
        out->w = color_w;
        out->h = color_h;
        if (!meta->bit_depth.float_sample) {
            scale = 1.0f / (float)((1u << bits) - 1);
        }
        for (i = 0; i < nplane; i++) {
            const jxl_mchan *ch = &gmod.base[i];
            uint32_t x, y;
            if (jxl_fplane_alloc(ctx, &out->plane[i], ch->w, ch->h) != 0) goto done;
            for (y = 0; y < ch->h; y++) {
                const int32_t *src = ch->data + (size_t)y * ch->stride;
                float *dst = out->plane[i].data + (size_t)y * out->plane[i].stride;
                for (x = 0; x < ch->w; x++) dst[x] = (float)src[x] * scale;
            }
        }
    }
    rc = 0;

done:
    jxl_free(ctx, specs);
    group_lists_free(ctx, &gl);
    jxl_chanlist_free(ctx, &gcl);
    jxl_modular_free(ctx, &gmod);
    if (has_global_ma) jxl_ma_config_free(ctx, &global_ma);
    if (rc != 0) jxl_fimage_free(ctx, out);
    (void)div_ceil32;
    return rc;
}
