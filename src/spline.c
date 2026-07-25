/* spline.c -- splines: smooth colored strokes drawn on top of the frame.
 *
 * A spline is a list of control points plus 32 DCT coefficients per XYB
 * channel and 32 more for the stroke width (sigma). The control points are
 * upsampled with a centripetal Catmull-Rom curve, walked at unit arc length,
 * and each sample splats a Gaussian-ish blob whose color and width come from
 * evaluating the DCTs at that position along the curve.
 */
#include "jxl_internal.h"

#include <math.h>

#define SPLINE_MAX_POINTS (1u << 20)

void jxl_splines_free(jxl_ctx *ctx, jxl_splines *sp) {
    uint32_t i;
    if (!sp || !sp->splines) return;
    for (i = 0; i < sp->n; i++) {
        jxl_free(ctx, sp->splines[i].px);
        jxl_free(ctx, sp->splines[i].py);
    }
    jxl_free(ctx, sp->splines);
    memset(sp, 0, sizeof(*sp));
}

int jxl_splines_read(jxl_ctx *ctx, jxl_br *br, const jxl_frame_header *fh,
                     jxl_splines *out) {
    jxl_dec dec;
    uint64_t num_pixels = (uint64_t)fh->width * fh->height;
    uint32_t num_splines, i;
    int64_t *sx = NULL, *sy = NULL;
    int64_t prev_x, prev_y;
    size_t acc_points = 0;
    int rc = -1;

    memset(out, 0, sizeof(*out));
    memset(&dec, 0, sizeof(dec));
    if (jxl_dec_init(ctx, &dec, br, 6) != 0) return -1;
    jxl_dec_begin(&dec, br);

    num_splines = jxl_dec_read(&dec, br, 2);
    if ((uint64_t)num_splines >= JXL_MIN((uint64_t)1 << 24, num_pixels / 4)) {
        JXL_ERR(ctx, "splines: too many splines");
        goto done;
    }
    num_splines += 1;

    sx = (int64_t *)jxl_calloc(ctx, num_splines, sizeof(int64_t));
    sy = (int64_t *)jxl_calloc(ctx, num_splines, sizeof(int64_t));
    if (!sx || !sy) goto done;

    prev_x = (int64_t)jxl_dec_read(&dec, br, 1);
    prev_y = (int64_t)jxl_dec_read(&dec, br, 1);
    sx[0] = prev_x;
    sy[0] = prev_y;
    for (i = 1; i < num_splines; i++) {
        int32_t dx = jxl_unpack_signed(jxl_dec_read(&dec, br, 1));
        int32_t dy = jxl_unpack_signed(jxl_dec_read(&dec, br, 1));
        prev_x += dx;
        prev_y += dy;
        sx[i] = prev_x;
        sy[i] = prev_y;
    }

    out->quant_adjust = jxl_unpack_signed(jxl_dec_read(&dec, br, 0));
    out->splines = (jxl_quant_spline *)jxl_calloc(ctx, num_splines,
                                                  sizeof(jxl_quant_spline));
    if (!out->splines) goto done;
    out->n = num_splines;

    for (i = 0; i < num_splines; i++) {
        jxl_quant_spline *qs = &out->splines[i];
        uint32_t num_points = jxl_dec_read(&dec, br, 3);
        int64_t cx = sx[i], cy = sy[i];
        int64_t dx_acc = 0, dy_acc = 0;
        uint32_t k;
        int c, j;

        acc_points += num_points;
        if (acc_points > JXL_MIN((uint64_t)SPLINE_MAX_POINTS, num_pixels / 2)) {
            JXL_ERR(ctx, "splines: too many control points");
            goto done;
        }
        qs->px = (int64_t *)jxl_calloc(ctx, num_points + 1, sizeof(int64_t));
        qs->py = (int64_t *)jxl_calloc(ctx, num_points + 1, sizeof(int64_t));
        if (!qs->px || !qs->py) goto done;
        qs->npoints = num_points + 1;
        qs->px[0] = cx;
        qs->py[0] = cy;

        for (k = 0; k < num_points; k++) {
            int64_t prev_px = cx, prev_py = cy;
            dx_acc += jxl_unpack_signed(jxl_dec_read(&dec, br, 4));
            dy_acc += jxl_unpack_signed(jxl_dec_read(&dec, br, 4));
            cx += dx_acc;
            cy += dy_acc;
            if (cx == prev_px && cy == prev_py) {
                JXL_ERR(ctx, "splines: duplicate control point");
                goto done;
            }
            qs->px[k + 1] = cx;
            qs->py[k + 1] = cy;
        }

        for (c = 0; c < 3; c++) {
            for (j = 0; j < 32; j++) {
                qs->xyb_dct[c][j] = jxl_unpack_signed(jxl_dec_read(&dec, br, 5));
            }
        }
        for (j = 0; j < 32; j++) {
            qs->sigma_dct[j] = jxl_unpack_signed(jxl_dec_read(&dec, br, 5));
        }
        if (br->err || dec.err) {
            JXL_ERR(ctx, "splines: truncated");
            goto done;
        }
    }

    if (jxl_dec_finalize(&dec) != 0) {
        JXL_ERR(ctx, "splines: bad ANS final state");
        goto done;
    }
    rc = 0;

done:
    jxl_free(ctx, sx);
    jxl_free(ctx, sy);
    jxl_dec_free(&dec);
    if (rc != 0) jxl_splines_free(ctx, out);
    return rc;
}

/* ----- rendering ----- */

typedef struct { float x, y; } jxl_pt;

static jxl_pt pt_add(jxl_pt a, jxl_pt b) { jxl_pt r; r.x = a.x + b.x; r.y = a.y + b.y; return r; }
static jxl_pt pt_sub(jxl_pt a, jxl_pt b) { jxl_pt r; r.x = a.x - b.x; r.y = a.y - b.y; return r; }
static jxl_pt pt_mul(jxl_pt a, float s) { jxl_pt r; r.x = a.x * s; r.y = a.y * s; return r; }
static float pt_norm2(jxl_pt a) { return a.x * a.x + a.y * a.y; }
static float pt_norm(jxl_pt a) { return sqrtf(pt_norm2(a)); }

/* Evaluates a length-32 DCT at a continuous position. */
static float continuous_idct(const float dct[32], float t) {
    float res = dct[0];
    int i;
    for (i = 1; i < 32; i++) {
        float theta = (float)i * (3.14159265358979323846f / 32.0f) * (t + 0.5f);
        res += 1.4142135623730951f * dct[i] * cosf(theta);
    }
    return res;
}

/* libjxl's error-function approximation (L1 error 7e-4). */
static float spline_erf(float x) {
    float ax = fabsf(x);
    float d1 = ax * 7.77394369e-02f + 2.05260015e-04f;
    float d2 = d1 * ax + 2.32120216e-01f;
    float d3 = d2 * ax + 2.77820801e-01f;
    float d4 = d3 * ax + 1.0f;
    float d5 = d4 * d4;
    float inv = 1.0f / d5;
    float result = -inv * inv + 1.0f;
    return x < 0.0f ? -result : result;
}

/* Centripetal Catmull-Rom upsampling: 16 samples per segment. */
static int upsample_points(jxl_ctx *ctx, const jxl_quant_spline *qs,
                           jxl_pt **out, uint32_t *out_n) {
    uint32_t n = qs->npoints, ext_n, i, cnt = 0;
    jxl_pt *ext, *up;
    uint32_t cap;

    if (n == 1) {
        up = (jxl_pt *)jxl_calloc(ctx, 1, sizeof(jxl_pt));
        if (!up) return -1;
        up[0].x = (float)qs->px[0];
        up[0].y = (float)qs->py[0];
        *out = up;
        *out_n = 1;
        return 0;
    }

    ext_n = n + 2;
    ext = (jxl_pt *)jxl_calloc(ctx, ext_n, sizeof(jxl_pt));
    if (!ext) return -1;
    /* Mirror the second point about the first, and vice versa at the end. */
    ext[0].x = 2.0f * (float)qs->px[0] - (float)qs->px[1];
    ext[0].y = 2.0f * (float)qs->py[0] - (float)qs->py[1];
    for (i = 0; i < n; i++) {
        ext[i + 1].x = (float)qs->px[i];
        ext[i + 1].y = (float)qs->py[i];
    }
    ext[ext_n - 1].x = 2.0f * (float)qs->px[n - 1] - (float)qs->px[n - 2];
    ext[ext_n - 1].y = 2.0f * (float)qs->py[n - 1] - (float)qs->py[n - 2];

    cap = 16 * (ext_n - 3) + 1;
    up = (jxl_pt *)jxl_calloc(ctx, cap, sizeof(jxl_pt));
    if (!up) {
        jxl_free(ctx, ext);
        return -1;
    }

    for (i = 0; i + 3 < ext_n; i++) {
        jxl_pt p[4], a[3], b[2];
        float t[4];
        int k, step;
        for (k = 0; k < 4; k++) p[k] = ext[i + k];
        up[cnt++] = p[1];
        t[0] = 0.0f;
        for (k = 1; k < 4; k++) {
            t[k] = t[k - 1] + powf(pt_norm2(pt_sub(p[k], p[k - 1])), 0.25f);
        }
        for (step = 1; step < 16; step++) {
            float knot = t[1] + ((float)step / 16.0f) * (t[2] - t[1]);
            for (k = 0; k < 3; k++) {
                a[k] = pt_add(p[k], pt_mul(pt_sub(p[k + 1], p[k]),
                                           (knot - t[k]) / (t[k + 1] - t[k])));
            }
            for (k = 0; k < 2; k++) {
                b[k] = pt_add(a[k], pt_mul(pt_sub(a[k + 1], a[k]),
                                           (knot - t[k]) / (t[k + 2] - t[k])));
            }
            up[cnt++] = pt_add(b[0], pt_mul(pt_sub(b[1], b[0]),
                                            (knot - t[1]) / (t[2] - t[1])));
        }
    }
    up[cnt].x = (float)qs->px[n - 1];
    up[cnt].y = (float)qs->py[n - 1];
    cnt++;

    jxl_free(ctx, ext);
    *out = up;
    *out_n = cnt;
    return 0;
}

int jxl_render_splines(jxl_ctx *ctx, jxl_fimage *img, const jxl_splines *sp,
                       const jxl_frame_header *fh, float corr_x, float corr_b) {
    uint32_t si, cw, ch;
    int rc = -1;
    jxl_pt *up = NULL;
    jxl_pt *arc_pt = NULL;
    float *arc_len = NULL;

    if (img->ncolor < 3) return 0;

    /* The three colour planes share dimensions; taking the smallest lets each
       sample's bounding box be clamped once instead of bounds-checking every
       pixel of every plane. fh->width/height bound it too. */
    cw = fh->width;
    ch = fh->height;
    for (si = 0; si < 3; si++) {
        if (img->plane[si].w < cw) cw = img->plane[si].w;
        if (img->plane[si].h < ch) ch = img->plane[si].h;
    }

    for (si = 0; si < sp->n; si++) {
        const jxl_quant_spline *qs = &sp->splines[si];
        float xyb_dct[3][32], sigma_dct[32];
        float qa = (float)sp->quant_adjust;
        float inv_qa = qa >= 0.0f ? 1.0f / (1.0f + qa / 8.0f) : 1.0f - qa / 8.0f;
        static const float chan_w[4] = {0.0042f, 0.075f, 0.07f, 0.3333f};
        uint32_t up_n = 0, nsamples = 0, cap, i, next_idx;
        float total_arclength;
        jxl_pt current;
        int c, j;

        for (c = 0; c < 3; c++) {
            for (j = 0; j < 32; j++) {
                xyb_dct[c][j] = (float)qs->xyb_dct[c][j] * chan_w[c] * inv_qa;
            }
        }
        for (j = 0; j < 32; j++) {
            xyb_dct[0][j] += corr_x * xyb_dct[1][j];
            xyb_dct[2][j] += corr_b * xyb_dct[1][j];
            sigma_dct[j] = (float)qs->sigma_dct[j] * chan_w[3] * inv_qa;
        }

        jxl_free(ctx, up);
        up = NULL;
        if (upsample_points(ctx, qs, &up, &up_n) != 0) goto done;

        /* Resample the curve at unit arc length: one sample per unit of
           length, so the count follows the curve's length, not its control
           point count. */
        {
            float total = 0.0f;
            uint32_t k;
            for (k = 1; k < up_n; k++) total += pt_norm(pt_sub(up[k], up[k - 1]));
            if (!(total >= 0.0f) || total > 1e8f) total = 0.0f;
            cap = up_n + (uint32_t)total + 4;
        }
        jxl_free(ctx, arc_pt);
        jxl_free(ctx, arc_len);
        arc_pt = (jxl_pt *)jxl_calloc(ctx, cap, sizeof(jxl_pt));
        arc_len = (float *)jxl_calloc(ctx, cap, sizeof(float));
        if (!arc_pt || !arc_len) goto done;

        current = up[0];
        arc_pt[0] = current;
        arc_len[0] = 1.0f;
        nsamples = 1;
        next_idx = 0;
        while (next_idx < up_n) {
            jxl_pt prev = current;
            float acc = 0.0f;
            for (;;) {
                jxl_pt next;
                float to_next;
                if (next_idx >= up_n) {
                    if (nsamples < cap) {
                        arc_pt[nsamples] = prev;
                        arc_len[nsamples] = acc;
                        nsamples++;
                    }
                    break;
                }
                next = up[next_idx];
                to_next = pt_norm(pt_sub(next, prev));
                if (acc + to_next >= 1.0f) {
                    float f = to_next > 0.0f ? (1.0f - acc) / to_next : 0.0f;
                    current = pt_add(prev, pt_mul(pt_sub(up[next_idx], prev), f));
                    if (nsamples < cap) {
                        arc_pt[nsamples] = current;
                        arc_len[nsamples] = 1.0f;
                        nsamples++;
                    }
                    break;
                }
                acc += to_next;
                prev = next;
                next_idx++;
            }
            if (nsamples >= cap) break;
        }
        if (nsamples == 0) continue;

        total_arclength = (float)nsamples - 2.0f + arc_len[nsamples - 1];
        for (i = 0; i < nsamples; i++) {
            float from_start = (float)i / total_arclength;
            float t, sigma, inv_sigma, values[3], max_color, max_distance;
            float vs[3], px, py;
            int32_t xb, xe, yb, ye, x, y;

            if (from_start > 1.0f) from_start = 1.0f;
            t = 31.0f * from_start;
            sigma = continuous_idct(sigma_dct, t);
            if (sigma == 0.0f) continue;
            inv_sigma = 1.0f / sigma;
            for (c = 0; c < 3; c++) {
                values[c] = continuous_idct(xyb_dct[c], t) * arc_len[i];
            }
            /* The radius past which the Gaussian has dropped below 1e-3 of
               max_color. libjxl's formula, which takes the log of max_color;
               jxl-oxide uses max_color itself, giving a radius ~1.7x larger
               and so ~3x the pixels for the same output. The guard keeps the
               root real for implausibly bright splines. */
            max_color = 0.01f;
            for (c = 0; c < 3; c++) {
                float a = fabsf(values[c]);
                if (a > max_color) max_color = a;
            }
            max_distance = -2.0f * sigma * sigma *
                           (logf(0.1f) * 3.0f - logf(max_color));
            max_distance = max_distance > 0.0f ? sqrtf(max_distance) : 0.0f;

            px = arc_pt[i].x;
            py = arc_pt[i].y;
            xb = (int32_t)floorf(px - max_distance + 0.5f);
            xe = (int32_t)floorf(px + max_distance + 1.5f);
            yb = (int32_t)floorf(py - max_distance + 0.5f);
            ye = (int32_t)floorf(py + max_distance + 1.5f);
            if (xb < 0) xb = 0;
            if (yb < 0) yb = 0;
            if (xe > (int32_t)cw) xe = (int32_t)cw;
            if (ye > (int32_t)ch) ye = (int32_t)ch;

            for (c = 0; c < 3; c++) vs[c] = 0.25f * values[c] * sigma;

            /* The distance and the error-function pair depend only on the
               sample position, so the channel loop is innermost: one sqrt and
               two spline_erf calls per pixel instead of three of each. */
            for (y = yb; y < ye; y++) {
                float *rows[3];
                float dy = (float)y - py;
                float dy2 = dy * dy;
                for (c = 0; c < 3; c++) {
                    rows[c] = img->plane[c].data +
                              (size_t)y * img->plane[c].stride;
                }
                for (x = xb; x < xe; x++) {
                    float dx = (float)x - px;
                    float distance = sqrtf(dx * dx + dy2);
                    float half = 0.5f * distance;
                    float factor = spline_erf((half + 0.35355338f) * inv_sigma) -
                                   spline_erf((half - 0.35355338f) * inv_sigma);
                    float ff = factor * factor;
                    for (c = 0; c < 3; c++) {
                        rows[c][x] += vs[c] * ff;
                    }
                }
            }
        }
    }
    rc = 0;

done:
    jxl_free(ctx, up);
    jxl_free(ctx, arc_pt);
    jxl_free(ctx, arc_len);
    return rc;
}
