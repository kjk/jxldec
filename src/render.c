/* render.c -- driving the frame decode and converting float planes to the
 * caller's pixel format (8/16-bit gray/gray+alpha/RGB/RGBA), including the
 * EXIF-style orientation fixup.
 */
#include "jxl_internal.h"

#include <math.h>

static int walk_frames(jxl_doc *doc, int frame_no, jxl_fimage *img,
                       int *count_out, jxl_frame_info *info_out);

int jxl_doc_frame_count(jxl_doc *doc) {
    int count = 0;
    if (!doc) return 0;
    if (doc->frame_count > 0) return doc->frame_count;
    if (walk_frames(doc, -1, NULL, &count, NULL) != 0) return 1;
    doc->frame_count = count > 0 ? count : 1;
    return doc->frame_count;
}

int jxl_doc_frame_info(jxl_doc *doc, int frame_no, jxl_frame_info *info) {
    jxl_fimage img;
    int rc;
    if (!doc || !info || frame_no < 0) return -1;
    memset(info, 0, sizeof(*info));
    info->tps_numerator = (int)doc->meta.animation.tps_numerator;
    info->tps_denominator = (int)doc->meta.animation.tps_denominator;
    info->is_last = 1;
    memset(&img, 0, sizeof(img));
    rc = walk_frames(doc, frame_no, &img, NULL, info);
    jxl_fimage_free(doc->ctx, &img);
    return rc;
}

static jxl_format resolve_format(const jxl_image_info *ii, jxl_format fmt) {
    int wide, gray, alpha;
    if (fmt != JXLDEC_FORMAT_NATIVE) return fmt;
    wide = ii->bits_per_sample > 8 || ii->exponent_bits > 0;
    gray = ii->num_color_channels == 1;
    alpha = ii->alpha_bits > 0;
    if (gray) {
        return wide ? (alpha ? JXLDEC_FORMAT_GRAYA16 : JXLDEC_FORMAT_GRAY16)
                    : (alpha ? JXLDEC_FORMAT_GRAYA8 : JXLDEC_FORMAT_GRAY8);
    }
    return wide ? (alpha ? JXLDEC_FORMAT_RGBA64 : JXLDEC_FORMAT_RGB48)
                : (alpha ? JXLDEC_FORMAT_RGBA32 : JXLDEC_FORMAT_RGB24);
}

int jxl_frame_render_info(jxl_doc *doc, int frame_no, jxl_format fmt,
                          jxl_render_info *info) {
    jxl_image_info ii;
    if (!doc || !info) return -1;
    if (jxl_doc_info(doc, &ii) != 0) return -1;
    (void)frame_no;
    info->width = ii.width;
    info->height = ii.height;
    info->format = resolve_format(&ii, fmt);
    return 0;
}

/* ----- float -> integer sample conversion ----- */

static uint32_t quantize(float v, uint32_t maxval) {
    float s;
    if (!(v > 0.0f)) return 0;       /* also catches NaN */
    s = v * (float)maxval + 0.5f;
    if (s >= (float)maxval) return maxval;
    return (uint32_t)s;
}

/* Reads a plane sample, replicating edge pixels for subsampled channels. */
static float plane_sample(const jxl_fplane *p, uint32_t x, uint32_t y,
                          uint32_t w, uint32_t h) {
    uint32_t px, py;
    if (p->w == 0 || p->h == 0) return 0.0f;
    px = (p->w == w) ? x : (uint32_t)((uint64_t)x * p->w / (w ? w : 1));
    py = (p->h == h) ? y : (uint32_t)((uint64_t)y * p->h / (h ? h : 1));
    if (px >= p->w) px = p->w - 1;
    if (py >= p->h) py = p->h - 1;
    return p->data[(size_t)py * p->stride + px];
}

/* Maps an output coordinate back to the decoded (unoriented) one. */
static void unapply_orientation(uint32_t orientation, uint32_t sw, uint32_t sh,
                                uint32_t ox, uint32_t oy, uint32_t *sx,
                                uint32_t *sy) {
    switch (orientation) {
        case 1: *sx = ox; *sy = oy; break;
        case 2: *sx = sw - 1 - ox; *sy = oy; break;
        case 3: *sx = sw - 1 - ox; *sy = sh - 1 - oy; break;
        case 4: *sx = ox; *sy = sh - 1 - oy; break;
        case 5: *sx = oy; *sy = ox; break;
        case 6: *sx = oy; *sy = sh - 1 - ox; break;
        case 7: *sx = sw - 1 - oy; *sy = sh - 1 - ox; break;
        default: *sx = sw - 1 - oy; *sy = ox; break;
    }
}

typedef struct {
    const jxl_fplane *r, *g, *b, *a;
} jxl_out_planes;

static void pick_planes(const jxl_fimage *img, const jxl_image_metadata *meta,
                        jxl_out_planes *op) {
    memset(op, 0, sizeof(*op));
    if (img->nplane == 0) return;
    op->r = &img->plane[0];
    if (img->ncolor >= 3 && img->nplane >= 3) {
        op->g = &img->plane[1];
        op->b = &img->plane[2];
    } else {
        op->g = op->r;
        op->b = op->r;
    }
    if (meta->alpha_index >= 0) {
        uint32_t idx = img->ncolor + (uint32_t)meta->alpha_index;
        if (idx < img->nplane) op->a = &img->plane[idx];
    }
}

static int write_pixels(jxl_ctx *ctx, jxl_doc *doc, const jxl_fimage *img,
                        jxl_format fmt, uint8_t *dst, int stride) {
    const jxl_image_metadata *meta = &doc->meta;
    jxl_out_planes op;
    uint32_t sw = img->w, sh = img->h;
    uint32_t ow, oh, ox, oy;
    uint32_t orientation = ctx->keep_orientation ? 1 : meta->orientation;
    int ncomp = 0, wide = 0, has_alpha = 0, gray = 0;
    uint32_t maxval;
    int bgr = ctx->bgr;

    pick_planes(img, meta, &op);
    jxl_apply_orientation_dims(orientation, sw, sh, &ow, &oh);

    switch (fmt) {
        case JXLDEC_FORMAT_GRAY8: ncomp = 1; gray = 1; break;
        case JXLDEC_FORMAT_GRAYA8: ncomp = 2; gray = 1; has_alpha = 1; break;
        case JXLDEC_FORMAT_RGB24: ncomp = 3; break;
        case JXLDEC_FORMAT_RGBA32: ncomp = 4; has_alpha = 1; break;
        case JXLDEC_FORMAT_GRAY16: ncomp = 1; gray = 1; wide = 1; break;
        case JXLDEC_FORMAT_GRAYA16: ncomp = 2; gray = 1; has_alpha = 1; wide = 1; break;
        case JXLDEC_FORMAT_RGB48: ncomp = 3; wide = 1; break;
        case JXLDEC_FORMAT_RGBA64: ncomp = 4; has_alpha = 1; wide = 1; break;
        default:
            JXL_ERR(ctx, "render: unsupported output format %d", (int)fmt);
            return -1;
    }
    maxval = wide ? 65535u : 255u;

    for (oy = 0; oy < oh; oy++) {
        uint8_t *row8 = dst + (size_t)oy * stride;
        uint16_t *row16 = (uint16_t *)row8;
        for (ox = 0; ox < ow; ox++) {
            uint32_t sx, sy;
            float rv, gv, bv, av = 1.0f;
            uint32_t comps[4];
            int c;

            unapply_orientation(orientation, sw, sh, ox, oy, &sx, &sy);
            rv = plane_sample(op.r, sx, sy, sw, sh);
            if (gray) {
                gv = bv = rv;
            } else {
                gv = plane_sample(op.g, sx, sy, sw, sh);
                bv = plane_sample(op.b, sx, sy, sw, sh);
            }
            if (op.a) av = plane_sample(op.a, sx, sy, sw, sh);

            if (gray) {
                comps[0] = quantize(rv, maxval);
                if (has_alpha) comps[1] = quantize(av, maxval);
            } else if (bgr) {
                comps[0] = quantize(bv, maxval);
                comps[1] = quantize(gv, maxval);
                comps[2] = quantize(rv, maxval);
                if (has_alpha) comps[3] = quantize(av, maxval);
            } else {
                comps[0] = quantize(rv, maxval);
                comps[1] = quantize(gv, maxval);
                comps[2] = quantize(bv, maxval);
                if (has_alpha) comps[3] = quantize(av, maxval);
            }
            if (wide) {
                for (c = 0; c < ncomp; c++) row16[ox * ncomp + c] = (uint16_t)comps[c];
            } else {
                for (c = 0; c < ncomp; c++) row8[ox * ncomp + c] = (uint8_t)comps[c];
            }
        }
    }
    return 0;
}

/* A frame is displayed when it is a normal frame that either ends the
   animation or has a duration; LF and reference-only frames are decoded for
   their side effects and skipped. */
static int frame_is_keyframe(const jxl_frame_header *fh) {
    if (fh->frame_type != JXL_FRAME_REGULAR &&
        fh->frame_type != JXL_FRAME_SKIP_PROGRESSIVE)
        return 0;
    return fh->is_last || fh->duration != 0;
}

/* Walks the frame chain. frame_no < 0 with count_out set only counts the
   displayed frames. Displayed frames are composited onto the canvas their
   blending info names, so animation frames that carry only a changed
   rectangle land in the right place. */
static int walk_frames(jxl_doc *doc, int frame_no, jxl_fimage *img,
                       int *count_out, jxl_frame_info *info_out) {
    jxl_ctx *ctx = doc->ctx;
    jxl_frame_state st;
    jxl_fimage canvas;
    int canvas_valid = 0;
    jxl_br br;
    int idx = 0;
    int rc = -1;

    memset(&st, 0, sizeof(st));
    memset(&canvas, 0, sizeof(canvas));
    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);

    for (;;) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t end;
        int last, keyframe, want, apply_ct;
        jxl_fimage tmp;

        memset(&toc, 0, sizeof(toc));
        memset(&tmp, 0, sizeof(tmp));
        if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) != 0) {
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }
        end = toc.end_off + toc.total_size;
        last = fh.is_last;
        keyframe = frame_is_keyframe(&fh);
        want = keyframe && idx == frame_no;
        apply_ct = want || !fh.save_before_ct;

        if (count_out && !want) {
            /* Counting only: skip the pixel work entirely. */
            if (keyframe) idx++;
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            if (last || end >= doc->container.cs_len) break;
            jxl_br_seek_byte(&br, end);
            continue;
        }

        if (jxl_frame_decode(ctx, doc, &fh, &toc, &st, apply_ct, &tmp) != 0) {
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            goto done;
        }

        if (fh.frame_type == JXL_FRAME_LF) {
            jxl_fimage_free(ctx, &st.lf_image);
            st.lf_image = tmp;
            st.lf_valid = 1;
            memset(&tmp, 0, sizeof(tmp));
        } else if (!keyframe) {
            if (fh.save_as_reference < 4) {
                uint32_t slot = fh.save_as_reference;
                jxl_fimage_free(ctx, &st.refs[slot]);
                st.refs[slot] = tmp;
                st.refs_valid[slot] = 1;
                memset(&tmp, 0, sizeof(tmp));
            }
        } else {
            uint32_t src = fh.blending.source;
            int cropped = fh.have_crop || tmp.w != doc->size.width ||
                          tmp.h != doc->size.height;
            int needs_canvas = cropped || fh.blending.mode != JXL_BLEND_REPLACE;
            int failed = 0;

            if (!needs_canvas) {
                jxl_fimage_free(ctx, &canvas);
                canvas = tmp;
                memset(&tmp, 0, sizeof(tmp));
            } else {
                jxl_fimage base;
                memset(&base, 0, sizeof(base));
                if (src < 4 && st.refs_valid[src]) {
                    failed = jxl_fimage_copy(ctx, &base, &st.refs[src]) != 0;
                } else if (canvas_valid) {
                    failed = jxl_fimage_copy(ctx, &base, &canvas) != 0;
                } else {
                    failed = jxl_fimage_blank_like(ctx, &base, &tmp,
                                                   doc->size.width,
                                                   doc->size.height) != 0;
                }
                if (!failed) {
                    jxl_blend_frame(ctx, &base, &tmp, &fh, &doc->meta);
                    jxl_fimage_free(ctx, &canvas);
                    canvas = base;
                } else {
                    jxl_fimage_free(ctx, &base);
                }
                jxl_fimage_free(ctx, &tmp);
            }
            if (failed) {
                jxl_toc_free(ctx, &toc);
                jxl_frame_header_free(ctx, &fh);
                goto done;
            }
            canvas_valid = 1;
            if (fh.save_as_reference < 4 && !fh.is_last) {
                uint32_t slot = fh.save_as_reference;
                jxl_fimage_free(ctx, &st.refs[slot]);
                memset(&st.refs[slot], 0, sizeof(st.refs[slot]));
                if (jxl_fimage_copy(ctx, &st.refs[slot], &canvas) == 0) {
                    st.refs_valid[slot] = 1;
                }
            }
        }

        if (want) {
            *img = canvas;
            memset(&canvas, 0, sizeof(canvas));
            canvas_valid = 0;
            if (info_out) {
                info_out->duration_ticks = (int)fh.duration;
                info_out->is_last = fh.is_last;
            }
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            rc = 0;
            goto done;
        }
        jxl_fimage_free(ctx, &tmp);
        if (keyframe) {
            idx++;
            st.visible_frames++;
            st.invisible_frames = 0;
        } else {
            st.invisible_frames++;
        }
        jxl_toc_free(ctx, &toc);
        jxl_frame_header_free(ctx, &fh);
        if (last || end >= doc->container.cs_len) break;
        jxl_br_seek_byte(&br, end);
    }

    if (count_out) {
        *count_out = idx;
        rc = 0;
    } else {
        JXL_ERR(ctx, "render: no such frame %d", frame_no);
    }

done:
    jxl_fimage_free(ctx, &canvas);
    jxl_frame_state_free(ctx, &st);
    return rc;
}

static int decode_frame_planes(jxl_doc *doc, int frame_no, jxl_fimage *img) {
    return walk_frames(doc, frame_no, img, NULL, NULL);
}

jxl_image *jxl_frame_render(jxl_doc *doc, int frame_no, jxl_format fmt) {
    jxl_ctx *ctx;
    jxl_render_info info;
    jxl_fimage img;
    jxl_image *out = NULL;
    size_t total;

    if (!doc) return NULL;
    ctx = doc->ctx;
    if (jxl_frame_render_info(doc, frame_no, fmt, &info) != 0) return NULL;
    memset(&img, 0, sizeof(img));
    if (decode_frame_planes(doc, frame_no, &img) != 0) return NULL;

    out = (jxl_image *)jxl_calloc(ctx, 1, sizeof(jxl_image));
    if (!out) goto done;
    out->width = info.width;
    out->height = info.height;
    out->format = info.format;
    out->stride = info.width * jxl_format_bpp(info.format);
    if (!jxl_size_mul((size_t)out->stride, (size_t)out->height, &total)) {
        jxl_free(ctx, out);
        out = NULL;
        goto done;
    }
    out->data = (uint8_t *)jxl_calloc(ctx, total ? total : 1, 1);
    if (!out->data) {
        jxl_free(ctx, out);
        out = NULL;
        goto done;
    }
    if (write_pixels(ctx, doc, &img, info.format, out->data, out->stride) != 0) {
        jxl_image_destroy(ctx, out);
        out = NULL;
    }

done:
    jxl_fimage_free(ctx, &img);
    return out;
}

int jxl_frame_render_into(jxl_doc *doc, int frame_no, jxl_format fmt,
                          uint8_t *dst, int stride) {
    jxl_ctx *ctx;
    jxl_render_info info;
    jxl_fimage img;
    int rc;

    if (!doc || !dst) return -1;
    ctx = doc->ctx;
    if (jxl_frame_render_info(doc, frame_no, fmt, &info) != 0) return -1;
    memset(&img, 0, sizeof(img));
    if (decode_frame_planes(doc, frame_no, &img) != 0) return -1;
    rc = write_pixels(ctx, doc, &img, info.format, dst, stride);
    jxl_fimage_free(ctx, &img);
    return rc;
}
