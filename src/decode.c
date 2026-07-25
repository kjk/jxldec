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

#include <math.h>

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

/* Same, without the zeroing, for callers that assign every sample before
   anything reads one. Zeroing a frame's worth of planes and immediately
   overwriting it was ~8% of a VarDCT decode.
 *
 * This is deliberately a separate entry point rather than a change to
 * jxl_fplane_alloc: jxl_fimage_blank_like *relies* on the zeros -- it
 * allocates a canvas and writes nothing -- so making the common allocator
 * uninitialised would turn a blank canvas into whatever the heap last held.
 * That is a disclosure bug, not a rendering one, so each caller was checked
 * individually and only the four that fill their plane completely use this.
 *
 * -DJXL_FPLANE_ALWAYS_ZERO turns it back into the zeroing version, so the
 * "every sample is written" claim can be diffed rather than trusted. */
int jxl_fplane_alloc_uninit(jxl_ctx *ctx, jxl_fplane *p, uint32_t w, uint32_t h) {
#ifdef JXL_FPLANE_ALWAYS_ZERO
    return jxl_fplane_alloc(ctx, p, w, h);
#else
    size_t total, bytes;
    if (!jxl_size_mul(w, h, &total)) return -1;
    if (!total) total = 1;
    if (!jxl_size_mul(total, sizeof(float), &bytes)) return -1;
    p->data = (float *)jxl_malloc(ctx, bytes);
    if (!p->data) return -1;
    p->w = w;
    p->h = h;
    p->stride = w;
    return 0;
#endif
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


/* ===================================================================== */
/* VarDCT frame state                                                     */
/* ===================================================================== */

typedef struct {
    jxl_quantizer quantizer;
    jxl_hf_block_ctx block_ctx;
    jxl_lf_chan_corr chan_corr;

    uint32_t bw, bh;              /* frame size in 8x8 blocks (rounded up) */
    uint32_t pw, ph;              /* pixel size == bw*8, bh*8              */
    float *lf[3];                 /* bw x bh, the dequantized LF image      */
    /* Chroma subsampling shifts; a subsampled channel uses only the top-left
       (bw >> hs) x (bh >> vs) corner of every bw-strided buffer. */
    int hs[3], vs[3];
    int32_t *lfq[3];              /* bw x bh, the raw quantized LF values    */
    float *coeff[3];              /* pw x ph, coefficients then samples    */
    jxl_block_info *block_info;   /* bw x bh                               */
    float *epf_sigma;             /* bw x bh                               */
    int32_t *x_from_y, *b_from_y; /* cfl_w x cfl_h                         */
    uint32_t cfl_w, cfl_h;

    jxl_dequant_matrices dm;
    int have_dm;
    jxl_hf_pass *passes;
    uint32_t num_passes;
    uint32_t num_hf_presets;
    jxl_natural_orders no;
} jxl_vardct_state;

static void vardct_state_free(jxl_ctx *ctx, jxl_vardct_state *v) {
    uint32_t i;
    int c;
    for (c = 0; c < 3; c++) {
        jxl_free(ctx, v->lf[c]);
        jxl_free(ctx, v->lfq[c]);
        jxl_free(ctx, v->coeff[c]);
    }
    jxl_free(ctx, v->block_info);
    jxl_free(ctx, v->epf_sigma);
    jxl_free(ctx, v->x_from_y);
    jxl_free(ctx, v->b_from_y);
    if (v->have_dm) jxl_dequant_matrices_free(ctx, &v->dm);
    if (v->passes) {
        for (i = 0; i < v->num_passes; i++) jxl_hf_pass_free(ctx, &v->passes[i]);
        jxl_free(ctx, v->passes);
    }
    jxl_hf_block_ctx_free(ctx, &v->block_ctx);
    jxl_natural_orders_free(ctx, &v->no);
    memset(v, 0, sizeof(*v));
}

static int vardct_state_alloc(jxl_ctx *ctx, jxl_vardct_state *v, uint32_t bw,
                              uint32_t bh) {
    int c;
    v->bw = bw;
    v->bh = bh;
    v->pw = bw * 8;
    v->ph = bh * 8;
    v->cfl_w = (v->pw + 63) / 64;
    v->cfl_h = (v->ph + 63) / 64;
    for (c = 0; c < 3; c++) {
        v->lf[c] = (float *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(float));
        v->lfq[c] = (int32_t *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(int32_t));
        v->coeff[c] = (float *)jxl_calloc(ctx, (size_t)v->pw * v->ph, sizeof(float));
        if (!v->lf[c] || !v->lfq[c] || !v->coeff[c]) return -1;
    }
    v->block_info =
        (jxl_block_info *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(jxl_block_info));
    v->epf_sigma = (float *)jxl_calloc(ctx, (size_t)bw * bh, sizeof(float));
    v->x_from_y =
        (int32_t *)jxl_calloc(ctx, (size_t)v->cfl_w * v->cfl_h, sizeof(int32_t));
    v->b_from_y =
        (int32_t *)jxl_calloc(ctx, (size_t)v->cfl_w * v->cfl_h, sizeof(int32_t));
    if (!v->block_info || !v->epf_sigma || !v->x_from_y || !v->b_from_y) return -1;
    {
        size_t i;
        for (i = 0; i < (size_t)bw * bh; i++) {
            v->block_info[i].dct_select = JXL_BLK_UNINIT;
        }
    }
    return 0;
}

/* Reads the quantized LF image of one LF group and dequantizes it into the
   frame-wide LF planes. */
static int decode_lf_coeff(jxl_ctx *ctx, jxl_br *br, jxl_vardct_state *v,
                           const jxl_lf_dequant *lfd, uint32_t lf_group_idx,
                           uint32_t lf_width, uint32_t lf_height,
                           uint32_t base_bx, uint32_t base_by,
                           uint32_t bit_depth, jxl_ma_config *global_ma) {
    uint32_t extra_precision = jxl_br_read(br, 2);
    uint32_t w = (lf_width + 7) / 8;
    uint32_t h = (lf_height + 7) / 8;
    jxl_mchan_spec specs[3];
    jxl_modular mod;
    jxl_chanlist cl;
    /* Channel order in the bitstream is X, Y, B but the planes are Y, X, B. */
    static const int plane_of[3] = {1, 0, 2};
    float m_lf[3];
    int i, rc = -1;

    memset(&mod, 0, sizeof(mod));
    memset(&cl, 0, sizeof(cl));
    m_lf[0] = lfd->m_x_lf;
    m_lf[1] = lfd->m_y_lf;
    m_lf[2] = lfd->m_b_lf;

    for (i = 0; i < 3; i++) {
        specs[plane_of[i]].w = w >> v->hs[i];
        specs[plane_of[i]].h = h >> v->vs[i];
        specs[plane_of[i]].hshift = 0;
        specs[plane_of[i]].vshift = 0;
    }
    if (jxl_modular_init(ctx, &mod, br, specs, 3, global_ma, 0, bit_depth) != 0)
        goto done;
    if (jxl_modular_transform_channels(ctx, &mod, &cl) != 0) goto done;
    if (jxl_modular_decode(ctx, &mod, &cl, br, 1 + lf_group_idx) != 0) goto done;
    if (jxl_modular_inverse(ctx, &mod, &cl) != 0) goto done;

    /* Plane c (X, Y, B) comes from Modular channel [1, 0, 2][c]. */
    for (i = 0; i < 3; i++) {
        const jxl_mchan *src = &mod.base[plane_of[i]];
        uint32_t bx0 = base_bx >> v->hs[i], by0 = base_by >> v->vs[i];
        uint32_t lim_w = v->bw >> v->hs[i], lim_h = v->bh >> v->vs[i];
        float *dst = v->lf[i] + (size_t)by0 * v->bw + bx0;
        uint32_t xx, yy;
        jxl_copy_lf_dequant(dst, v->bw, src, &v->quantizer, m_lf[i],
                            (int)extra_precision);
        for (yy = 0; yy < src->h && by0 + yy < lim_h; yy++) {
            for (xx = 0; xx < src->w && bx0 + xx < lim_w; xx++) {
                v->lfq[i][(size_t)(by0 + yy) * v->bw + bx0 + xx] =
                    src->data[(size_t)yy * src->stride + xx];
            }
        }
    }
    rc = 0;

done:
    jxl_chanlist_free(ctx, &cl);
    jxl_modular_free(ctx, &mod);
    return rc;
}

/* Copies one LF group's varblock metadata into the frame-wide arrays. */
static void merge_hf_meta(jxl_vardct_state *v, const jxl_hf_meta *m,
                          uint32_t base_bx, uint32_t base_by) {
    uint32_t x, y;
    for (y = 0; y < m->bh && base_by + y < v->bh; y++) {
        for (x = 0; x < m->bw && base_bx + x < v->bw; x++) {
            v->block_info[(size_t)(base_by + y) * v->bw + base_bx + x] =
                m->block_info[(size_t)y * m->bw + x];
            v->epf_sigma[(size_t)(base_by + y) * v->bw + base_bx + x] =
                m->epf_sigma[(size_t)y * m->bw + x];
        }
    }
    for (y = 0; y < m->cfl_h; y++) {
        uint32_t gy = base_by / 8 + y;
        if (gy >= v->cfl_h) break;
        for (x = 0; x < m->cfl_w; x++) {
            uint32_t gx = base_bx / 8 + x;
            if (gx >= v->cfl_w) break;
            v->x_from_y[(size_t)gy * v->cfl_w + gx] = m->x_from_y[(size_t)y * m->cfl_w + x];
            v->b_from_y[(size_t)gy * v->cfl_w + gx] = m->b_from_y[(size_t)y * m->cfl_w + x];
        }
    }
}

/* Dequantizes, undoes chroma-from-luma and inverse-transforms every varblock
   of the frame, leaving XYB samples in v->coeff. */
static void vardct_finish_blocks(jxl_vardct_state *v,
                                 const jxl_image_metadata *meta,
                                 const jxl_frame_header *fh) {
    float qm_scale[3];
    uint32_t bx, by;
    int c;

    qm_scale[0] = powf(0.8f, (float)fh->x_qm_scale - 2.0f);
    qm_scale[1] = 1.0f;
    qm_scale[2] = powf(0.8f, (float)fh->b_qm_scale - 2.0f);

    for (c = 0; c < 3; c++) {
        for (by = 0; by < v->bh; by++) {
            uint32_t sby = by >> v->vs[c];
            if ((sby << v->vs[c]) != by) continue;
            for (bx = 0; bx < v->bw; bx++) {
                const jxl_block_info *bi = &v->block_info[(size_t)by * v->bw + bx];
                uint32_t sbx = bx >> v->hs[c];
                if ((sbx << v->hs[c]) != bx) continue;
                if (bi->dct_select >= JXL_TR_COUNT) continue;
                jxl_dequant_varblock(
                    v->coeff[c] + (size_t)(sby * 8) * v->pw + sbx * 8, v->pw,
                    bi->dct_select, bi->hf_mul, c, &v->dm, &v->quantizer,
                    qm_scale[c], meta->quant_bias[c], meta->quant_bias_numerator);
            }
        }
    }

    /* Chroma-from-luma needs the three channels sample-aligned, so it only
       applies when nothing is subsampled -- as in libjxl's DequantDC. */
    if (!v->hs[0] && !v->vs[0] && !v->hs[2] && !v->vs[2]) {
        jxl_cfl_hf(v->coeff[0], v->coeff[1], v->coeff[2], v->pw, v->pw, v->ph,
                   v->x_from_y, v->b_from_y, v->cfl_w, &v->chan_corr);
    }

    for (c = 0; c < 3; c++) {
        for (by = 0; by < v->bh; by++) {
            uint32_t sby = by >> v->vs[c];
            if ((sby << v->vs[c]) != by) continue;
            for (bx = 0; bx < v->bw; bx++) {
                const jxl_block_info *bi = &v->block_info[(size_t)by * v->bw + bx];
                uint32_t sbx = bx >> v->hs[c];
                float *blk;
                if ((sbx << v->hs[c]) != bx) continue;
                if (bi->dct_select >= JXL_TR_COUNT) continue;
                blk = v->coeff[c] + (size_t)(sby * 8) * v->pw + sbx * 8;
                jxl_fill_varblock_lf(blk, v->pw, bi->dct_select, v->lf[c], v->bw,
                                     sbx, sby);
                jxl_transform_varblock(blk, v->pw, bi->dct_select);
            }
        }
    }
}

void jxl_frame_state_free(jxl_ctx *ctx, jxl_frame_state *st) {
    int i;
    for (i = 0; i < 4; i++) {
        jxl_fimage_free(ctx, &st->refs[i]);
        st->refs_valid[i] = 0;
    }
    jxl_fimage_free(ctx, &st->lf_image);
    st->lf_valid = 0;
}

int jxl_frame_decode(jxl_ctx *ctx, jxl_doc *doc, const jxl_frame_header *fh,
                     const jxl_toc *toc, jxl_frame_state *st, int apply_ct,
                     jxl_fimage *out) {
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
    jxl_vardct_state vd;
    jxl_patches patches;
    jxl_splines splines;
    jxl_noise_params noise;
    int have_patches = 0, have_splines = 0, have_noise = 0;
    int is_vardct;
    uint32_t nspecs = 0, i, split;
    uint32_t color_w, color_h, group_dim, group_dim_shift;
    uint32_t num_lf_groups, num_groups, num_passes;
    uint32_t color_upsampling_shift;
    uint32_t bits;
    uint32_t ncolor;
    int rc = -1;

    memset(&global_ma, 0, sizeof(global_ma));
    memset(&gmod, 0, sizeof(gmod));
    memset(&gcl, 0, sizeof(gcl));
    memset(&prefix, 0, sizeof(prefix));
    memset(&gl, 0, sizeof(gl));
    memset(&vd, 0, sizeof(vd));
    memset(&patches, 0, sizeof(patches));
    memset(&splines, 0, sizeof(splines));
    memset(&noise, 0, sizeof(noise));
    memset(out, 0, sizeof(*out));

    is_vardct = (fh->encoding == JXL_ENC_VARDCT);
    if ((fh->flags & JXL_FF_USE_LF_FRAME) && !st->lf_valid) {
        JXL_ERR(ctx, "frame: LF frame referenced but not decoded");
        return -1;
    }

    /* Two things about extra channels we decline rather than get wrong.
       Every other type -- depth, selection mask, CFA, thermal, black -- is
       carried through as a plane, which is what libjxl does with it too. */
    for (i = 0; i < meta->num_extra; i++) {
        /* A different upsampling factor from the colour channels needs a
           separate, earlier pass (libjxl's !late_ec_upsample path). Decoding
           it as if the factors matched would silently misplace the channel. */
        if (fh->ec_upsampling[i] != fh->upsampling) {
            JXL_ERR(ctx, "frame: extra channel %u upsamples by %u but the "
                         "colour channels by %u; not supported",
                    (unsigned)i, (unsigned)fh->ec_upsampling[i],
                    (unsigned)fh->upsampling);
            return -1;
        }
        /* Spot colour has to be mixed into the colour channels after the
           colour transform -- libjxl's stage_spot.cc, which is simply
              mix = spot[3] * s[x];  p[c][x] = mix * spot[c] + (1-mix) * p[c][x]
           for c in 0..2. Ten lines, deliberately not written: nothing in
           libjxl's testdata carries a spot channel and cjxl cannot produce
           one, so it would be untested code that merely looks handled.
           Carrying the channel through instead would leave the colour
           silently unmixed, so refuse until there is something to verify
           against. */
        if (meta->ec_info[i].type == JXL_EC_SPOT) {
            JXL_ERR(ctx, "frame: extra channel %u is a spot colour; "
                         "not supported", (unsigned)i);
            return -1;
        }
    }

    color_w = jxl_frame_color_width(fh);
    color_h = jxl_frame_color_height(fh);
    group_dim = jxl_frame_group_dim(fh);
    group_dim_shift = 7 + fh->group_size_shift;
    num_lf_groups = jxl_frame_num_lf_groups(fh);
    num_groups = jxl_frame_num_groups(fh);
    num_passes = fh->passes.num_passes;
    bits = meta->bit_depth.bits_per_sample;
    ncolor = (uint32_t)fh->encoded_color_channels;

    sections_init(&sec, doc->container.cs, doc->container.cs_len, toc);
    br = section_reader(&sec, JXL_TOC_LF_GLOBAL, 0, 0);
    if (!br) return -1;

    if (fh->flags & JXL_FF_PATCHES) {
        if (jxl_patches_read(ctx, br, meta, fh, &patches) != 0) goto done;
        have_patches = 1;
    }
    if (fh->flags & JXL_FF_SPLINES) {
        if (jxl_splines_read(ctx, br, fh, &splines) != 0) goto done;
        have_splines = 1;
    }
    if (fh->flags & JXL_FF_NOISE) {
        if (jxl_noise_params_read(br, &noise) != 0) goto done;
        have_noise = 1;
    }
    lf_dequant_read(br, &lf_dequant);

    if (is_vardct) {
        for (i = 0; i < 3; i++) {
            jxl_jpeg_upsampling_shifts(fh->jpeg_upsampling, i, &vd.hs[i],
                                       &vd.vs[i]);
        }
        if (vardct_state_alloc(ctx, &vd, jxl_frame_blocks_w(fh),
                               jxl_frame_blocks_h(fh)) != 0)
            goto done;
        jxl_quantizer_read(br, &vd.quantizer);
        if (jxl_hf_block_ctx_read(ctx, br, &vd.block_ctx) != 0) goto done;
        jxl_lf_chan_corr_read(br, &vd.chan_corr, meta->xyb_encoded);
        if (vd.quantizer.global_scale == 0 || vd.quantizer.quant_lf == 0) {
            JXL_ERR(ctx, "frame: zero quantizer scale");
            goto done;
        }
    }

    /* ----- global modular ----- */
    if (jxl_br_bool(br)) {
        uint64_t num_channels = ncolor + meta->num_extra;
        uint64_t limit = 1024 + (uint64_t)fh->width * fh->height * num_channels / 16;
        if (limit > (1u << 22)) limit = 1u << 22;
        if (jxl_ma_config_read(ctx, br, &global_ma, (size_t)limit) != 0) goto done;
        has_global_ma = 1;
    }

    /* VarDCT frames keep only the extra channels in the global Modular
       image; the color channels are the transform coefficients. */
    nspecs = (is_vardct ? 0 : ncolor) + meta->num_extra;
    specs = (jxl_mchan_spec *)jxl_calloc(ctx, nspecs ? nspecs : 1,
                                         sizeof(jxl_mchan_spec));
    if (!specs) goto done;
    if (!is_vardct) {
        for (i = 0; i < ncolor; i++) {
            specs[i].w = color_w;
            specs[i].h = color_h;
            specs[i].hshift = 0;
            specs[i].vshift = 0;
        }
    }
    color_upsampling_shift = 0;
    {
        uint32_t u = fh->upsampling;
        while (u > 1) { color_upsampling_shift++; u >>= 1; }
    }
    for (i = 0; i < meta->num_extra; i++) {
        uint32_t base = is_vardct ? 0 : ncolor;
        uint32_t ec_shift = 0;
        uint32_t u = fh->ec_upsampling[i];
        int32_t actual;
        while (u > 1) { ec_shift++; u >>= 1; }
        actual = (int32_t)(ec_shift + meta->ec_info[i].dim_shift) -
                 (int32_t)color_upsampling_shift;
        if (actual < 0) actual = 0;
        specs[base + i].w = color_w;
        specs[base + i].h = color_h;
        specs[base + i].hshift = actual;
        specs[base + i].vshift = actual;
    }

    if (jxl_modular_init(ctx, &gmod, br, specs, nspecs,
                         has_global_ma ? &global_ma : NULL, group_dim,
                         bits) != 0)
        goto done;
    if (nspecs) {
        if (jxl_modular_transform_channels(ctx, &gmod, &gcl) != 0) goto done;
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
    } else {
        split = 0;
    }

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
            if (gw == 0 || gh == 0 || pass_idx >= num_passes) {
                JXL_ERR(ctx, "frame: bad channel shift for the group size");
                goto done;
            }
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
                JXL_ERR(ctx, "frame: bad channel shift for the LF group size");
                goto done;
            }
            if (split_channel_into_groups(ctx, ch, gw, gh, nx, ny, gl.lf,
                                          num_lf_groups) != 0)
                goto done;
        }
    }

    /* ----- LF groups ----- */
    {
        uint32_t lf_per_row = jxl_frame_lf_groups_per_row(fh);
        for (i = 0; i < num_lf_groups; i++) {
            uint32_t lg_x = lf_per_row ? i % lf_per_row : 0;
            uint32_t lg_y = lf_per_row ? i / lf_per_row : 0;
            /* LF groups are group_dim *blocks* across, over the padded block
               grid -- not group_dim*8 pixels of the real image. */
            uint32_t blocks_w = jxl_frame_blocks_w(fh);
            uint32_t blocks_h = jxl_frame_blocks_h(fh);
            uint32_t base_bx = lg_x * group_dim;
            uint32_t base_by = lg_y * group_dim;
            uint32_t lf_w = JXL_MIN(group_dim, blocks_w - base_bx) * 8;
            uint32_t lf_h = JXL_MIN(group_dim, blocks_h - base_by) * 8;

            br = section_reader(&sec, JXL_TOC_LF_GROUP, 0, i);
            if (!br) goto done;
            if (is_vardct && !(fh->flags & JXL_FF_USE_LF_FRAME)) {
                if (decode_lf_coeff(ctx, br, &vd, &lf_dequant, i, lf_w, lf_h,
                                    base_bx, base_by, bits,
                                    has_global_ma ? &global_ma : NULL) != 0)
                    goto done;
            }
            if (decode_group_modular(ctx, br, &gl.lf[i],
                                     has_global_ma ? &global_ma : NULL, group_dim,
                                     bits, 1 + num_lf_groups + i) != 0)
                goto done;
            if (is_vardct) {
                jxl_hf_meta hm;
                if (jxl_hf_meta_read(ctx, br, &hm, num_lf_groups, i, lf_w, lf_h,
                                     fh->jpeg_upsampling, bits,
                                     has_global_ma ? &global_ma : NULL, &fh->epf,
                                     vd.quantizer.global_scale) != 0)
                    goto done;
                merge_hf_meta(&vd, &hm, base_bx, base_by);
                jxl_hf_meta_free(ctx, &hm);
            }
        }
    }

    /* ----- HF global ----- */
    if (is_vardct) {
        uint32_t p;
        br = section_reader(&sec, JXL_TOC_HF_GLOBAL, 0, 0);
        if (!br) goto done;
        if (jxl_dequant_matrices_read(ctx, br, &vd.dm, bits, num_lf_groups,
                                      has_global_ma ? &global_ma : NULL) != 0)
            goto done;
        vd.have_dm = 1;
        vd.num_hf_presets =
            jxl_br_read(br, (int)jxl_bitlen(num_groups > 1 ? num_groups - 1 : 0)) + 1;
        vd.passes = (jxl_hf_pass *)jxl_calloc(ctx, num_passes, sizeof(jxl_hf_pass));
        if (!vd.passes) goto done;
        vd.num_passes = num_passes;
        for (p = 0; p < num_passes; p++) {
            if (jxl_hf_pass_read(ctx, br, &vd.passes[p], &vd.block_ctx,
                                 vd.num_hf_presets, &vd.no) != 0)
                goto done;
        }
    }

    /* ----- pass groups ----- */
    {
        uint32_t p, g;
        uint32_t groups_per_row = jxl_frame_groups_per_row(fh);
        uint32_t lf_per_row = jxl_frame_lf_groups_per_row(fh);
        for (p = 0; p < num_passes; p++) {
            for (g = 0; g < num_groups; g++) {
                jxl_chanlist *cl = &gl.pass[(size_t)p * num_groups + g];
                br = section_reader(&sec, JXL_TOC_GROUP_PASS, p, g);
                if (!br) goto done;
                if (is_vardct) {
                    uint32_t gx = groups_per_row ? g % groups_per_row : 0;
                    uint32_t gy = groups_per_row ? g / groups_per_row : 0;
                    uint32_t bx0 = gx * (group_dim / 8);
                    uint32_t by0 = gy * (group_dim / 8);
                    uint32_t bwid = JXL_MIN(group_dim / 8, vd.bw - bx0);
                    uint32_t bhig = JXL_MIN(group_dim / 8, vd.bh - by0);
                    jxl_hf_coeff_params hp;
                    jxl_mchan lfq_view[3];
                    float *outp[3];
                    size_t strides[3];
                    int c;
                    (void)lf_per_row;

                    memset(&hp, 0, sizeof(hp));
                    hp.num_hf_presets = vd.num_hf_presets;
                    hp.bc = &vd.block_ctx;
                    hp.block_info = vd.block_info + (size_t)by0 * vd.bw + bx0;
                    hp.bi_w = bwid;
                    hp.bi_h = bhig;
                    hp.bi_stride = vd.bw;
                    for (c = 0; c < 3; c++) {
                        /* A subsampled channel's samples for this group start
                           at the shifted block origin, not at the group's. */
                        uint32_t sbx0 = bx0 >> vd.hs[c];
                        uint32_t sby0 = by0 >> vd.vs[c];
                        hp.jpeg_upsampling[c] = fh->jpeg_upsampling[c];
                        memset(&lfq_view[c], 0, sizeof(lfq_view[c]));
                        lfq_view[c].data = vd.lfq[c] + (size_t)sby0 * vd.bw + sbx0;
                        lfq_view[c].stride = vd.bw;
                        lfq_view[c].w = (bwid + ((1u << vd.hs[c]) - 1)) >> vd.hs[c];
                        lfq_view[c].h = (bhig + ((1u << vd.vs[c]) - 1)) >> vd.vs[c];
                        hp.lf_quant[c] = &lfq_view[c];
                        outp[c] = vd.coeff[c] + (size_t)(sby0 * 8) * vd.pw +
                                  sbx0 * 8;
                        strides[c] = vd.pw;
                    }
                    hp.pass = &vd.passes[p];
                    hp.coeff_shift = p < 16 ? fh->passes.shift[p] : 0;
                    hp.no = &vd.no;
                    if (jxl_write_hf_coeff(ctx, br, &hp, outp, strides) != 0)
                        goto done;
                }
                if (decode_group_modular(ctx, br, cl,
                                         has_global_ma ? &global_ma : NULL,
                                         group_dim, bits,
                                         1 + 3 * num_lf_groups + 17 +
                                             p * num_groups + g) != 0)
                    goto done;
            }
        }
    }

    if (nspecs && jxl_modular_inverse(ctx, &gmod, &gcl) != 0) goto done;

    /* ----- VarDCT rendering ----- */
    if (is_vardct) {
        float m_lf[3];
        float *planes[3];
        int c;

        m_lf[0] = lf_dequant.m_x_lf;
        m_lf[1] = lf_dequant.m_y_lf;
        m_lf[2] = lf_dequant.m_b_lf;
        if (fh->flags & JXL_FF_USE_LF_FRAME) {
            /* The LF image comes from a previously decoded LF frame, already
               dequantized and in XYB, so neither CfL nor smoothing apply. */
            uint32_t x, y;
            for (c = 0; c < 3; c++) {
                const jxl_fplane *p = &st->lf_image.plane[c];
                for (y = 0; y < vd.bh; y++) {
                    uint32_t sy = y < p->h ? y : (p->h ? p->h - 1 : 0);
                    for (x = 0; x < vd.bw; x++) {
                        uint32_t sx = x < p->w ? x : (p->w ? p->w - 1 : 0);
                        vd.lf[c][(size_t)y * vd.bw + x] =
                            p->data[(size_t)sy * p->stride + sx];
                    }
                }
            }
        } else {
            if (!vd.hs[0] && !vd.vs[0] && !vd.hs[2] && !vd.vs[2]) {
                jxl_cfl_lf(vd.lf[0], vd.lf[1], vd.lf[2], vd.bw, vd.bh, vd.bw,
                           &vd.chan_corr);
            }
            if (!(fh->flags & JXL_FF_SKIP_ADAPTIVE_LF_SMOOTH)) {
                if (jxl_adaptive_lf_smoothing(ctx, vd.lf, vd.bw, vd.bh, vd.bw,
                                              m_lf, &vd.quantizer) != 0)
                    goto done;
            }
        }
        /* Build only the dequant matrices this frame's transforms need. */
        {
            uint8_t used[JXL_TR_COUNT];
            size_t nb = (size_t)vd.bw * vd.bh, k;
            memset(used, 0, sizeof(used));
            for (k = 0; k < nb; k++) {
                uint8_t t = vd.block_info[k].dct_select;
                if (t < JXL_TR_COUNT) used[t] = 1;
            }
            for (k = 0; k < JXL_TR_COUNT; k++) {
                if (!used[k]) continue;
                if (jxl_dequant_matrices_ensure(ctx, &vd.dm, (int)k) != 0) {
                    JXL_ERR(ctx, "vardct: bad dequant matrix");
                    goto done;
                }
            }
        }
        vardct_finish_blocks(&vd, meta, fh);

        /* Bring subsampled chroma back to full resolution before the loop
           filters, which is where libjxl's pipeline puts it. */
        for (c = 0; c < 3; c++) {
            if (!vd.hs[c] && !vd.vs[c]) continue;
            jxl_chroma_upsample(vd.coeff[c],
                                div_ceil32(color_w, 1u << vd.hs[c]),
                                div_ceil32(color_h, 1u << vd.vs[c]), vd.pw,
                                vd.hs[c], vd.vs[c], vd.pw, vd.ph);
        }

        /* Filter the image at its real size, not the block-padded pw x ph.
           libjxl's render pipeline allocates and mirrors at
           frame_dimensions_.xsize_upsampled (simple_render_pipeline.cc), so
           for a width that is not a multiple of 8 its last column mirrors,
           while padding the loop over to pw would instead read the partial
           block's padding columns as if they were image content. That is a
           real difference: it moved the last column and row of every
           500x500 and 510x532 corpus file by up to 10 counts. */
        for (c = 0; c < 3; c++) planes[c] = vd.coeff[c];
        if (fh->gab.enabled) {
            if (jxl_apply_gabor(ctx, planes, color_w, color_h, vd.pw,
                                fh->gab.weights) != 0)
                goto done;
        }
        if (fh->epf.enabled) {
            if (jxl_apply_epf(ctx, planes, color_w, color_h, vd.pw, vd.epf_sigma,
                              vd.bw, &fh->epf) != 0)
                goto done;
        }
    }

    /* ----- assemble the float planes ----- */
    {
        uint32_t nplane = (is_vardct ? 3 : gmod.nbase) +
                          (is_vardct ? meta->num_extra : 0);
        uint32_t base = 0;
        float scale = 1.0f;
        if (jxl_fimage_alloc(ctx, out, nplane) != 0) goto done;
        out->ncolor = is_vardct ? 3 : ncolor;
        out->w = color_w;
        out->h = color_h;
        if (!meta->bit_depth.float_sample) {
            scale = 1.0f / (float)((1u << bits) - 1);
        }
        if (is_vardct) {
            uint32_t x, y;
            for (i = 0; i < 3; i++) {
                if (jxl_fplane_alloc_uninit(ctx, &out->plane[i], color_w, color_h) != 0)
                    goto done;
                for (y = 0; y < color_h; y++) {
                    memcpy(out->plane[i].data + (size_t)y * out->plane[i].stride,
                           vd.coeff[i] + (size_t)y * vd.pw,
                           (size_t)color_w * sizeof(float));
                }
                (void)x;
            }
            base = 3;
        }
        /* Lossy Modular stores XYB as (Y, X, B) integers scaled by the LF
           dequant weights, not as [0, 1] samples. */
        if (!is_vardct && meta->xyb_encoded && gmod.nbase >= 3) {
            float mul[3];
            uint32_t x, y;
            static const int src_of[3] = {1, 0, 2};
            mul[0] = lf_dequant.m_x_lf / 128.0f;
            mul[1] = lf_dequant.m_y_lf / 128.0f;
            mul[2] = lf_dequant.m_b_lf / 128.0f;
            for (y = 0; y < gmod.base[2].h; y++) {
                int32_t *rb = gmod.base[2].data + (size_t)y * gmod.base[2].stride;
                const int32_t *ry = gmod.base[0].data + (size_t)y * gmod.base[0].stride;
                for (x = 0; x < gmod.base[2].w; x++) {
                    rb[x] = (int32_t)((uint32_t)rb[x] + (uint32_t)ry[x]);
                }
            }
            for (i = 0; i < 3; i++) {
                const jxl_mchan *ch = &gmod.base[src_of[i]];
                if (jxl_fplane_alloc_uninit(ctx, &out->plane[i], ch->w, ch->h) != 0) goto done;
                for (y = 0; y < ch->h; y++) {
                    const int32_t *src = ch->data + (size_t)y * ch->stride;
                    float *dst = out->plane[i].data + (size_t)y * out->plane[i].stride;
                    for (x = 0; x < ch->w; x++) dst[x] = (float)src[x] * mul[i];
                }
            }
        }
        for (i = 0; i < gmod.nbase; i++) {
            const jxl_mchan *ch = &gmod.base[i];
            uint32_t x, y;
            uint32_t pi = base + i;
            if (pi >= nplane) break;
            if (!is_vardct && meta->xyb_encoded && i < 3) continue;
            if (jxl_fplane_alloc_uninit(ctx, &out->plane[pi], ch->w, ch->h) != 0) goto done;
            for (y = 0; y < ch->h; y++) {
                const int32_t *src = ch->data + (size_t)y * ch->stride;
                float *dst = out->plane[pi].data + (size_t)y * out->plane[pi].stride;
                for (x = 0; x < ch->w; x++) dst[x] = (float)src[x] * scale;
            }
        }
    }

    /* Patches blend pre-color-transform samples from a reference frame. */
    if (have_patches) {
        if (jxl_apply_patches(ctx, out, &patches, meta, st->refs,
                              st->refs_valid) != 0)
            goto done;
    }

    if (have_splines) {
        float cx = is_vardct ? vd.chan_corr.base_correlation_x : 0.0f;
        float cb = is_vardct ? vd.chan_corr.base_correlation_b : 1.0f;
        if (jxl_render_splines(ctx, out, &splines, fh, cx, cb) != 0) goto done;
    }

    /* Upsampling sits between splines and noise, where libjxl's pipeline puts
       it (dec_cache.cc), so noise is synthesised at full resolution. When the
       colour channels are upsampled and every extra channel shares the same
       factor, libjxl upsamples them all here together ("late_ec_upsample");
       that is the case handled below, and jxl_frame_decode rejects the mixed
       one up front. */
    if (color_upsampling_shift > 0) {
        uint32_t full_w = jxl_frame_sample_width(fh, 1);
        uint32_t full_h = jxl_frame_sample_height(fh, 1);
        for (i = 0; i < out->nplane; i++) {
            /* Scale each plane by the factor rather than stretching it to the
               frame size: a colour plane is ceil(full / N) and lands exactly
               on full once the rounding is cropped off, while a plane that
               was subsampled relative to colour keeps that relationship. */
            uint32_t tw = out->plane[i].w << color_upsampling_shift;
            uint32_t th = out->plane[i].h << color_upsampling_shift;
            if (tw > full_w) tw = full_w;
            if (th > full_h) th = full_h;
            if (jxl_upsample_plane(ctx, &out->plane[i], color_upsampling_shift,
                                   meta, tw, th) != 0)
                goto done;
        }
        out->w = full_w;
        out->h = full_h;
    }

    if (have_noise) {
        float cx = is_vardct ? vd.chan_corr.base_correlation_x : 0.0f;
        float cb = is_vardct ? vd.chan_corr.base_correlation_b : 1.0f;
        if (jxl_render_noise(ctx, out, &noise, fh, st->visible_frames,
                             st->invisible_frames, cx, cb) != 0)
            goto done;
    }

    /* ----- YCbCr / XYB -> the image's color space ----- */
    if (apply_ct && fh->do_ycbcr && out->ncolor >= 3) {
        size_t n = (size_t)out->plane[0].w * out->plane[0].h;
        jxl_ycbcr_to_rgb(out->plane[0].data, out->plane[1].data,
                         out->plane[2].data, n);
    } else if (apply_ct && meta->xyb_encoded && out->ncolor >= 3) {
        size_t n = (size_t)out->plane[0].w * out->plane[0].h;
        float opsin[9];
        jxl_opsin_matrix_for(meta, opsin);
        jxl_xyb_to_linear(out->plane[0].data, out->plane[1].data,
                          out->plane[2].data, n, opsin,
                          meta->opsin_bias, meta->tone_mapping.intensity_target);
        for (i = 0; i < 3; i++) {
            jxl_linear_to_tf(out->plane[i].data, n, &meta->colour,
                             meta->tone_mapping.intensity_target);
        }
        /* Grayscale XYB decodes into three identical planes; drop two so the
           extra channels (alpha!) keep their expected plane indices. */
        if (meta->colour.colour_space == JXLDEC_CS_GRAY && out->ncolor == 3) {
            uint32_t k;
            jxl_free(ctx, out->plane[1].data);
            jxl_free(ctx, out->plane[2].data);
            for (k = 3; k < out->nplane; k++) out->plane[k - 2] = out->plane[k];
            out->nplane -= 2;
            out->ncolor = 1;
        }
    }
    rc = 0;

done:
    jxl_free(ctx, specs);
    group_lists_free(ctx, &gl);
    jxl_chanlist_free(ctx, &gcl);
    jxl_modular_free(ctx, &gmod);
    vardct_state_free(ctx, &vd);
    jxl_patches_free(ctx, &patches);
    jxl_splines_free(ctx, &splines);
    if (has_global_ma) jxl_ma_config_free(ctx, &global_ma);
    if (rc != 0) jxl_fimage_free(ctx, out);
    return rc;
}
