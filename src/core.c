/* core.c -- context, allocator wrappers, diagnostics. */
#include "jxl_internal.h"

#include <stdio.h>
#include <stdlib.h>

static void *default_alloc(void *user, void *ctx, size_t size) {
    (void)user;
    (void)ctx;
    return malloc(size);
}

static void default_free(void *user, void *ctx, void *ptr) {
    (void)user;
    (void)ctx;
    free(ptr);
}

jxl_ctx *jxl_ctx_new(jxl_alloc_cb alloc, jxl_free_cb free_cb,
                     jxl_error_cb error, void *user) {
    jxl_ctx *ctx;
    jxl_alloc_cb a = alloc ? alloc : default_alloc;
    jxl_free_cb f = free_cb ? free_cb : default_free;

    /* Bootstrap allocation: no ctx exists yet, so the callback gets NULL. */
    ctx = (jxl_ctx *)a(user, NULL, sizeof(*ctx));
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc = a;
    ctx->free_cb = f;
    ctx->error = error;
    ctx->user = user;
    return ctx;
}

void jxl_ctx_free(jxl_ctx *ctx) {
    jxl_free_cb f;
    void *user;
    if (!ctx) return;
    f = ctx->free_cb;
    user = ctx->user;
    f(user, NULL, ctx);
}

void jxl_ctx_set_bgr(jxl_ctx *ctx, int enable) {
    if (ctx) ctx->bgr = enable ? 1 : 0;
}

void jxl_ctx_set_keep_orientation(jxl_ctx *ctx, int enable) {
    if (ctx) ctx->keep_orientation = enable ? 1 : 0;
}

void jxl_request_abort(jxl_ctx *ctx) {
    if (ctx) ctx->abort_epoch++;
}

void *jxl_malloc(jxl_ctx *ctx, size_t size) {
    if (size == 0) size = 1;
    return ctx->alloc(ctx->user, ctx, size);
}

void *jxl_calloc(jxl_ctx *ctx, size_t count, size_t size) {
    size_t total;
    void *p;
    if (!jxl_size_mul(count, size, &total)) return NULL;
    if (total == 0) total = 1;
    p = ctx->alloc(ctx->user, ctx, total);
    if (p) memset(p, 0, total);
    return p;
}

void *jxl_realloc_array(jxl_ctx *ctx, void *ptr, size_t old_count,
                        size_t new_count, size_t size) {
    size_t old_total, new_total;
    void *p;
    if (!jxl_size_mul(new_count, size, &new_total)) return NULL;
    if (!jxl_size_mul(old_count, size, &old_total)) return NULL;
    p = ctx->alloc(ctx->user, ctx, new_total ? new_total : 1);
    if (!p) return NULL;
    if (ptr && old_total) memcpy(p, ptr, JXL_MIN(old_total, new_total));
    if (new_total > old_total) {
        memset((uint8_t *)p + old_total, 0, new_total - old_total);
    }
    if (ptr) ctx->free_cb(ctx->user, ctx, ptr);
    return p;
}

void jxl_free(jxl_ctx *ctx, void *ptr) {
    if (ptr) ctx->free_cb(ctx->user, ctx, ptr);
}

int jxl_size_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > (size_t)-1 / a) return 0;
    *out = a * b;
    return 1;
}

void jxl_errorf(jxl_ctx *ctx, jxl_severity sev, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    if (!ctx || !ctx->error) return;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    buf[sizeof(buf) - 1] = 0;
    ctx->error(ctx->user, sev, buf);
}

int jxl_format_bpp(jxl_format fmt) {
    switch (fmt) {
        case JXLDEC_FORMAT_GRAY8: return 1;
        case JXLDEC_FORMAT_GRAYA8: return 2;
        case JXLDEC_FORMAT_RGB24: return 3;
        case JXLDEC_FORMAT_RGBA32: return 4;
        case JXLDEC_FORMAT_GRAY16: return 2;
        case JXLDEC_FORMAT_GRAYA16: return 4;
        case JXLDEC_FORMAT_RGB48: return 6;
        case JXLDEC_FORMAT_RGBA64: return 8;
        default: return 0;
    }
}

void jxl_image_destroy(jxl_ctx *ctx, jxl_image *img) {
    if (!img) return;
    jxl_free(ctx, img->data);
    jxl_free(ctx, img);
}
