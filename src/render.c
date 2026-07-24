/* render.c -- driving the frame decode and converting float planes to the
 * caller's pixel format (8/16-bit gray/gray+alpha/RGB/RGBA), including the
 * EXIF-style orientation fixup.
 */
#include "jxl_internal.h"

#include <math.h>

int jxl_doc_frame_count(jxl_doc *doc) {
    if (!doc) return 0;
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

static jxl_format resolve_format(const jxl_image_info *ii, jxl_format fmt) {
    int wide, gray, alpha;
    if (fmt != JXL_FORMAT_NATIVE) return fmt;
    wide = ii->bits_per_sample > 8 || ii->exponent_bits > 0;
    gray = ii->num_color_channels == 1;
    alpha = ii->alpha_bits > 0;
    if (gray) {
        return wide ? (alpha ? JXL_FORMAT_GRAYA16 : JXL_FORMAT_GRAY16)
                    : (alpha ? JXL_FORMAT_GRAYA8 : JXL_FORMAT_GRAY8);
    }
    return wide ? (alpha ? JXL_FORMAT_RGBA64 : JXL_FORMAT_RGB48)
                : (alpha ? JXL_FORMAT_RGBA32 : JXL_FORMAT_RGB24);
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
        case JXL_FORMAT_GRAY8: ncomp = 1; gray = 1; break;
        case JXL_FORMAT_GRAYA8: ncomp = 2; gray = 1; has_alpha = 1; break;
        case JXL_FORMAT_RGB24: ncomp = 3; break;
        case JXL_FORMAT_RGBA32: ncomp = 4; has_alpha = 1; break;
        case JXL_FORMAT_GRAY16: ncomp = 1; gray = 1; wide = 1; break;
        case JXL_FORMAT_GRAYA16: ncomp = 2; gray = 1; has_alpha = 1; wide = 1; break;
        case JXL_FORMAT_RGB48: ncomp = 3; wide = 1; break;
        case JXL_FORMAT_RGBA64: ncomp = 4; has_alpha = 1; wide = 1; break;
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

/* Decodes frame `frame_no` of the document into float planes. */
static int decode_frame_planes(jxl_doc *doc, int frame_no, jxl_fimage *img) {
    jxl_ctx *ctx = doc->ctx;
    jxl_br br;
    int idx = 0;
    int rc = -1;

    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    for (;;) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t end;
        int last;

        memset(&toc, 0, sizeof(toc));
        if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) != 0) {
            jxl_frame_header_free(ctx, &fh);
            return -1;
        }
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            return -1;
        }
        end = toc.end_off + toc.total_size;
        last = fh.is_last;

        if (idx == frame_no) {
            rc = jxl_frame_decode(ctx, doc, &fh, &toc, img);
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            return rc;
        }
        jxl_toc_free(ctx, &toc);
        jxl_frame_header_free(ctx, &fh);
        idx++;
        if (last || end >= doc->container.cs_len) break;
        jxl_br_seek_byte(&br, end);
    }
    JXL_ERR(ctx, "render: no such frame %d", frame_no);
    return -1;
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
