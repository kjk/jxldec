/* patch.c -- patches: rectangles copied out of a previously decoded
 * reference frame and blended into this one.
 *
 * The encoder uses patches for repeated content (text, icons, dithering
 * patterns): it stores the shape once in a ReferenceOnly frame and then lists
 * the positions to stamp it at. Each stamp carries its own blend mode per
 * channel, so the same patch can replace color but alpha-blend transparency.
 *
 * Patches operate on the frame's *pre-color-transform* samples, which is why
 * they run before the XYB conversion and why reference frames are stored
 * "before CT".
 */
#include "jxl_internal.h"

void jxl_patches_free(jxl_ctx *ctx, jxl_patches *p) {
    uint32_t i, j;
    if (!p || !p->refs) return;
    for (i = 0; i < p->n; i++) {
        for (j = 0; j < p->refs[i].ntargets; j++) {
            jxl_free(ctx, p->refs[i].targets[j].blending);
        }
        jxl_free(ctx, p->refs[i].targets);
    }
    jxl_free(ctx, p->refs);
    memset(p, 0, sizeof(*p));
}

int jxl_patches_read(jxl_ctx *ctx, jxl_br *br, const jxl_image_metadata *meta,
                     const jxl_frame_header *fh, jxl_patches *out) {
    jxl_dec dec;
    uint32_t num_refs, i;
    uint32_t max_refs, max_patches, total = 0;
    uint32_t nblend = meta->num_extra + 1;
    uint32_t first_alpha = 0;
    uint32_t nalpha = 0;
    int rc = -1;

    memset(out, 0, sizeof(*out));
    memset(&dec, 0, sizeof(dec));
    out->nblend = nblend;

    for (i = 0; i < meta->num_extra; i++) {
        if (meta->ec_info[i].type == JXL_EC_ALPHA) {
            if (nalpha == 0) first_alpha = i;
            nalpha++;
        }
    }

    if (jxl_dec_init(ctx, &dec, br, 10) != 0) return -1;
    jxl_dec_begin(&dec, br);

    max_refs = (uint32_t)JXL_MIN((uint64_t)1 << 24,
                                 (uint64_t)fh->width * fh->height / 16);
    max_patches = max_refs > (0xffffffffu / 4) ? 0xffffffffu : max_refs * 4;

    num_refs = jxl_dec_read(&dec, br, 0);
    if (num_refs > max_refs || br->err) {
        JXL_ERR(ctx, "patches: too many patch references (%u)", (unsigned)num_refs);
        goto done;
    }
    if (num_refs) {
        out->refs = (jxl_patch_ref *)jxl_calloc(ctx, num_refs, sizeof(jxl_patch_ref));
        if (!out->refs) goto done;
    }
    out->n = num_refs;

    for (i = 0; i < num_refs; i++) {
        jxl_patch_ref *pr = &out->refs[i];
        uint32_t count, t;
        int32_t px = 0, py = 0;
        int have_prev = 0;

        pr->ref_idx = jxl_dec_read(&dec, br, 1);
        if (pr->ref_idx >= 4) {
            JXL_ERR(ctx, "patches: reference index out of range");
            goto done;
        }
        pr->x0 = jxl_dec_read(&dec, br, 3);
        pr->y0 = jxl_dec_read(&dec, br, 3);
        pr->width = jxl_dec_read(&dec, br, 2) + 1;
        pr->height = jxl_dec_read(&dec, br, 2) + 1;
        count = jxl_dec_read(&dec, br, 7) + 1;
        total += count;
        if (total > max_patches || br->err || dec.err) {
            JXL_ERR(ctx, "patches: too many patches");
            goto done;
        }
        pr->targets = (jxl_patch_target *)jxl_calloc(ctx, count,
                                                     sizeof(jxl_patch_target));
        if (!pr->targets) goto done;
        pr->ntargets = count;

        for (t = 0; t < count; t++) {
            jxl_patch_target *tg = &pr->targets[t];
            uint32_t b;
            if (have_prev) {
                int32_t dx = jxl_unpack_signed(jxl_dec_read(&dec, br, 6));
                int32_t dy = jxl_unpack_signed(jxl_dec_read(&dec, br, 6));
                tg->x = (int32_t)((uint32_t)px + (uint32_t)dx);
                tg->y = (int32_t)((uint32_t)py + (uint32_t)dy);
            } else {
                tg->x = (int32_t)jxl_dec_read(&dec, br, 4);
                tg->y = (int32_t)jxl_dec_read(&dec, br, 4);
                have_prev = 1;
            }
            px = tg->x;
            py = tg->y;

            tg->blending = (jxl_patch_blend *)jxl_calloc(ctx, nblend,
                                                         sizeof(jxl_patch_blend));
            if (!tg->blending) goto done;
            for (b = 0; b < nblend; b++) {
                uint32_t raw = jxl_dec_read(&dec, br, 5);
                if (raw > JXL_PATCH_MULADD_BELOW) {
                    JXL_ERR(ctx, "patches: invalid blend mode %u", (unsigned)raw);
                    goto done;
                }
                tg->blending[b].mode = (uint8_t)raw;
                if (raw >= 4 && nalpha >= 2) {
                    tg->blending[b].alpha_channel = jxl_dec_read(&dec, br, 8);
                } else {
                    tg->blending[b].alpha_channel = first_alpha;
                }
                tg->blending[b].clamp =
                    (raw >= 3) ? (jxl_dec_read(&dec, br, 9) != 0) : 0;
            }
            if (br->err || dec.err) {
                JXL_ERR(ctx, "patches: truncated");
                goto done;
            }
        }
    }

    if (jxl_dec_finalize(&dec) != 0) {
        JXL_ERR(ctx, "patches: bad ANS final state");
        goto done;
    }
    rc = 0;

done:
    jxl_dec_free(&dec);
    if (rc != 0) jxl_patches_free(ctx, out);
    return rc;
}

/* ----- blending ----- */

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static const jxl_fplane *plane_or_null(const jxl_fimage *img, uint32_t idx) {
    if (!img || idx >= img->nplane) return NULL;
    return &img->plane[idx];
}

static float plane_at(const jxl_fplane *p, uint32_t x, uint32_t y) {
    if (!p || !p->data || x >= p->w || y >= p->h) return 0.0f;
    return p->data[(size_t)y * p->stride + x];
}

/* Blends one channel of one patch stamp. */
static void blend_channel(jxl_fplane *base, const jxl_fplane *ref,
                          const jxl_fplane *base_alpha,
                          const jxl_fplane *ref_alpha, int is_alpha_channel,
                          const jxl_patch_blend *bi, int premultiplied,
                          int32_t bx, int32_t by, uint32_t rx, uint32_t ry,
                          uint32_t w, uint32_t h) {
    uint32_t dx, dy;
    int swapped = (bi->mode == JXL_PATCH_BLEND_BELOW ||
                   bi->mode == JXL_PATCH_MULADD_BELOW);
    jxl_patch_blend eff = *bi;

    /* Without an alpha channel to weight with, Blend is just Replace and
       MulAdd is just Add. */
    if (!is_alpha_channel && !ref_alpha) {
        if (eff.mode == JXL_PATCH_BLEND_ABOVE || eff.mode == JXL_PATCH_BLEND_BELOW) {
            eff.mode = JXL_PATCH_REPLACE;
        } else if (eff.mode == JXL_PATCH_MULADD_ABOVE ||
                   eff.mode == JXL_PATCH_MULADD_BELOW) {
            eff.mode = JXL_PATCH_ADD;
        }
    }
    bi = &eff;

    for (dy = 0; dy < h; dy++) {
        float *brow = base->data + (size_t)(by + dy) * base->stride + bx;
        const float *rrow = ref->data + (size_t)(ry + dy) * ref->stride + rx;
        for (dx = 0; dx < w; dx++) {
            float b = brow[dx];
            float n = rrow[dx];
            switch (bi->mode) {
                case JXL_PATCH_NONE:
                    break;
                case JXL_PATCH_REPLACE:
                    brow[dx] = n;
                    break;
                case JXL_PATCH_ADD:
                    brow[dx] = b + n;
                    break;
                case JXL_PATCH_MUL:
                    brow[dx] = b * (bi->clamp ? clamp01(n) : n);
                    break;
                case JXL_PATCH_BLEND_ABOVE:
                case JXL_PATCH_BLEND_BELOW: {
                    float ba, na, bs, ns;
                    if (is_alpha_channel) {
                        /* The alpha channel itself mixes rather than blends. */
                        float lo = b, hi = n;
                        if (swapped) { lo = n; hi = b; }
                        if (bi->clamp) hi = clamp01(hi);
                        brow[dx] = lo + hi * (1.0f - lo);
                        break;
                    }
                    if (swapped) {
                        bs = n; ns = b;
                        ba = plane_at(ref_alpha, rx + dx, ry + dy);
                        na = plane_at(base_alpha, (uint32_t)bx + dx, (uint32_t)by + dy);
                    } else {
                        bs = b; ns = n;
                        ba = plane_at(base_alpha, (uint32_t)bx + dx, (uint32_t)by + dy);
                        na = plane_at(ref_alpha, rx + dx, ry + dy);
                    }
                    if (bi->clamp) na = clamp01(na);
                    if (premultiplied) {
                        brow[dx] = ns + bs * (1.0f - na);
                    } else {
                        float bar = 1.0f - ba;
                        float nar = 1.0f - na;
                        float mixed = 1.0f - nar * bar;
                        float recip = mixed > 0.0f ? 1.0f / mixed : 0.0f;
                        brow[dx] = (na * ns + ba * bs * nar) * recip;
                    }
                    break;
                }
                case JXL_PATCH_MULADD_ABOVE:
                case JXL_PATCH_MULADD_BELOW: {
                    float bs, ns, na;
                    if (is_alpha_channel) {
                        if (swapped) brow[dx] = n;   /* replace */
                        break;                       /* else: skip */
                    }
                    if (swapped) {
                        bs = n; ns = b;
                        na = plane_at(base_alpha, (uint32_t)bx + dx, (uint32_t)by + dy);
                    } else {
                        bs = b; ns = n;
                        na = plane_at(ref_alpha, rx + dx, ry + dy);
                    }
                    if (bi->clamp) na = clamp01(na);
                    brow[dx] = bs + na * ns;
                    break;
                }
                default:
                    break;
            }
        }
    }
}

/* Stamps every patch of every reference into the frame. */
int jxl_apply_patches(jxl_ctx *ctx, jxl_fimage *img, const jxl_patches *p,
                      const jxl_image_metadata *meta, jxl_fimage refs[4],
                      const int refs_valid[4]) {
    uint32_t i, t, c;
    uint32_t ncolor = img->ncolor;

    for (i = 0; i < p->n; i++) {
        const jxl_patch_ref *pr = &p->refs[i];
        const jxl_fimage *ref;
        if (pr->ref_idx >= 4 || !refs_valid[pr->ref_idx]) {
            JXL_ERR(ctx, "patches: reference frame %u not available",
                    (unsigned)pr->ref_idx);
            return -1;
        }
        ref = &refs[pr->ref_idx];

        for (t = 0; t < pr->ntargets; t++) {
            const jxl_patch_target *tg = &pr->targets[t];
            for (c = 0; c < img->nplane; c++) {
                const jxl_patch_blend *bi;
                jxl_fplane *base = &img->plane[c];
                const jxl_fplane *rp = plane_or_null(ref, c);
                const jxl_fplane *base_alpha = NULL, *ref_alpha = NULL;
                int is_alpha_channel = 0, premultiplied = 0;
                int64_t bx0, by0, rx0, ry0;
                int64_t w, h;

                if (!rp || !rp->data || !base->data) continue;
                bi = (c < ncolor) ? &tg->blending[0]
                                  : &tg->blending[c - ncolor + 1];
                if (bi->mode == JXL_PATCH_NONE) continue;

                if (bi->mode >= JXL_PATCH_BLEND_ABOVE) {
                    uint32_t ai = ncolor + bi->alpha_channel;
                    is_alpha_channel = (c == ai);
                    base_alpha = plane_or_null(img, ai);
                    ref_alpha = plane_or_null(ref, ai);
                    if (bi->alpha_channel < meta->num_extra) {
                        premultiplied =
                            meta->ec_info[bi->alpha_channel].alpha_associated;
                    }
                }

                /* Clip the stamp to both the frame and the reference. */
                bx0 = tg->x;
                by0 = tg->y;
                rx0 = pr->x0;
                ry0 = pr->y0;
                w = pr->width;
                h = pr->height;
                if (bx0 < 0) { rx0 -= bx0; w += bx0; bx0 = 0; }
                if (by0 < 0) { ry0 -= by0; h += by0; by0 = 0; }
                if (bx0 + w > base->w) w = (int64_t)base->w - bx0;
                if (by0 + h > base->h) h = (int64_t)base->h - by0;
                if (rx0 + w > rp->w) w = (int64_t)rp->w - rx0;
                if (ry0 + h > rp->h) h = (int64_t)rp->h - ry0;
                if (w <= 0 || h <= 0 || rx0 < 0 || ry0 < 0) continue;

                blend_channel(base, rp, base_alpha, ref_alpha, is_alpha_channel,
                              bi, premultiplied, (int32_t)bx0, (int32_t)by0,
                              (uint32_t)rx0, (uint32_t)ry0, (uint32_t)w,
                              (uint32_t)h);
            }
        }
    }
    return 0;
}

/* ----- frame blending (animation) ----- */

/* Frame blend modes map onto the same primitives as patch blend modes. */
static uint8_t frame_mode_to_patch(jxl_blend_mode m) {
    switch (m) {
        case JXL_BLEND_REPLACE: return JXL_PATCH_REPLACE;
        case JXL_BLEND_ADD: return JXL_PATCH_ADD;
        case JXL_BLEND_BLEND: return JXL_PATCH_BLEND_ABOVE;
        case JXL_BLEND_MULADD: return JXL_PATCH_MULADD_ABOVE;
        default: return JXL_PATCH_MUL;
    }
}

/* Composites a decoded frame onto the canvas it declares as its source.
   `frame` covers the rectangle (fh->x0, fh->y0, fh->width, fh->height) of the
   full image; the canvas is always full-size. */
int jxl_blend_frame(jxl_ctx *ctx, jxl_fimage *canvas, const jxl_fimage *frame,
                    const jxl_frame_header *fh,
                    const jxl_image_metadata *meta) {
    uint32_t c;
    uint32_t ncolor = canvas->ncolor;

    for (c = 0; c < canvas->nplane && c < frame->nplane; c++) {
        const jxl_blending_info *fbi =
            (c < ncolor) ? &fh->blending
                         : (fh->ec_blending ? &fh->ec_blending[c - ncolor]
                                            : &fh->blending);
        jxl_patch_blend bi;
        jxl_fplane *base = &canvas->plane[c];
        const jxl_fplane *nf = &frame->plane[c];
        const jxl_fplane *base_alpha = NULL, *new_alpha = NULL;
        int is_alpha_channel = 0, premultiplied = 0;
        int64_t bx0, by0, rx0 = 0, ry0 = 0, w, h;

        if (!base->data || !nf->data) continue;
        bi.mode = frame_mode_to_patch(fbi->mode);
        bi.alpha_channel = fbi->alpha_channel;
        bi.clamp = fbi->clamp;

        if (bi.mode >= JXL_PATCH_BLEND_ABOVE) {
            uint32_t ai = ncolor + bi.alpha_channel;
            is_alpha_channel = (c == ai);
            base_alpha = plane_or_null(canvas, ai);
            new_alpha = plane_or_null(frame, ai);
            if (bi.alpha_channel < meta->num_extra) {
                premultiplied = meta->ec_info[bi.alpha_channel].alpha_associated;
            }
        }

        bx0 = fh->x0;
        by0 = fh->y0;
        w = nf->w;
        h = nf->h;
        if (bx0 < 0) { rx0 -= bx0; w += bx0; bx0 = 0; }
        if (by0 < 0) { ry0 -= by0; h += by0; by0 = 0; }
        if (bx0 + w > base->w) w = (int64_t)base->w - bx0;
        if (by0 + h > base->h) h = (int64_t)base->h - by0;
        if (rx0 + w > nf->w) w = (int64_t)nf->w - rx0;
        if (ry0 + h > nf->h) h = (int64_t)nf->h - ry0;
        if (w <= 0 || h <= 0) continue;

        blend_channel(base, nf, base_alpha, new_alpha, is_alpha_channel, &bi,
                      premultiplied, (int32_t)bx0, (int32_t)by0, (uint32_t)rx0,
                      (uint32_t)ry0, (uint32_t)w, (uint32_t)h);
    }
    (void)ctx;
    return 0;
}

/* Allocates a full-image canvas shaped like `like`. */
int jxl_fimage_blank_like(jxl_ctx *ctx, jxl_fimage *out, const jxl_fimage *like,
                          uint32_t w, uint32_t h) {
    uint32_t i;
    if (jxl_fimage_alloc(ctx, out, like->nplane) != 0) return -1;
    out->ncolor = like->ncolor;
    out->w = w;
    out->h = h;
    for (i = 0; i < like->nplane; i++) {
        uint32_t pw = like->plane[i].w ? w : 0;
        uint32_t ph = like->plane[i].h ? h : 0;
        if (jxl_fplane_alloc(ctx, &out->plane[i], pw, ph) != 0) return -1;
    }
    return 0;
}

int jxl_fimage_copy(jxl_ctx *ctx, jxl_fimage *dst, const jxl_fimage *src) {
    uint32_t i, y;
    if (jxl_fimage_alloc(ctx, dst, src->nplane) != 0) return -1;
    dst->ncolor = src->ncolor;
    dst->w = src->w;
    dst->h = src->h;
    for (i = 0; i < src->nplane; i++) {
        const jxl_fplane *s = &src->plane[i];
        if (jxl_fplane_alloc(ctx, &dst->plane[i], s->w, s->h) != 0) return -1;
        for (y = 0; y < s->h; y++) {
            memcpy(dst->plane[i].data + (size_t)y * dst->plane[i].stride,
                   s->data + (size_t)y * s->stride, (size_t)s->w * sizeof(float));
        }
    }
    return 0;
}
