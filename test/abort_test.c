/* abort_test.c -- regression test for jxl_request_abort.
 *
 *   abort_test file.jxl ...        (bun cmd/test-abort.ts builds and runs it)
 *
 * The promise in jxl.h: an abort requested while a render runs makes it
 * fail (NULL / -1) at the next frame or group boundary; a render started
 * after the request runs normally; the doc stays usable, and rendering again
 * gives the frame. What can go wrong is everything around that: a leak on
 * the abort path, a half-written cache that poisons the next render, or a
 * frame count cached as 1 because the walk that counted it was cut short.
 *
 * Timing-based abort tests are flaky, so the main sweep is deterministic:
 * the allocator callback requests the abort itself when the render reaches
 * its k-th allocation, for k spread over a whole render. Each run must then
 * either fail or produce exactly the reference pixels (an abort after the
 * last check is allowed to lose), must free everything it allocated, and must
 * leave the doc rendering correctly afterwards. Requesting at the very first
 * allocation must fail -- otherwise the abort is not being checked at all.
 * A threaded pass then calls jxl_request_abort from a second thread at a few
 * delays, checking the same invariants, since that is how SumatraPDF uses it.
 */
/* nanosleep and pthreads are POSIX, which -std=c11 alone does not expose. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "jxl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

#define MAX_FRAMES 3     /* frames per file the test renders and checks */
#define SWEEP_POINTS 40  /* abort points per render in the deterministic pass */

static int fails;
#define CHECK(cond, ...)                          \
    do {                                          \
        if (!(cond)) {                            \
            fprintf(stderr, "FAIL %s:%d: ", g_file, __LINE__); \
            fprintf(stderr, __VA_ARGS__);         \
            fputc('\n', stderr);                  \
            fails++;                              \
        }                                         \
    } while (0)

static const char *g_file = "";

/* ----- counting allocator; requests the abort at the k-th allocation ----- */

static size_t g_live;          /* bytes currently allocated through the ctx */
static unsigned long g_count;  /* allocations since the counter was reset */
static unsigned long g_abort_at; /* request the abort on this one; 0 = never */

static void *t_alloc(void *user, void *ctx, size_t size) {
    uint8_t *p;
    (void)user;
    p = (uint8_t *)malloc(size + 16);
    if (!p) return NULL;
    memcpy(p, &size, sizeof(size));
    g_live += size;
    g_count++;
    if (ctx && g_abort_at && g_count == g_abort_at) {
        jxl_request_abort((jxl_ctx *)ctx);
    }
    return p + 16;
}

static void t_free(void *user, void *ctx, void *ptr) {
    uint8_t *p;
    size_t size;
    (void)user;
    (void)ctx;
    if (!ptr) return;
    p = (uint8_t *)ptr - 16;
    memcpy(&size, p, sizeof(size));
    g_live -= size;
    free(p);
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    buf = (uint8_t *)malloc((size_t)n);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static int same_image(const jxl_image *a, const jxl_image *b) {
    int y;
    size_t row;
    if (!a || !b || a->width != b->width || a->height != b->height ||
        a->format != b->format)
        return 0;
    row = (size_t)a->width * (size_t)jxl_format_bpp(a->format);
    for (y = 0; y < a->height; y++) {
        if (memcmp(a->data + (size_t)y * a->stride,
                   b->data + (size_t)y * b->stride, row) != 0)
            return 0;
    }
    return 1;
}

/* ----- the second thread for the threaded pass ----- */

typedef struct {
    jxl_ctx *ctx;
    int delay_ms;
} abort_arg;

static void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

#ifdef _WIN32
static DWORD WINAPI abort_thread(LPVOID p) {
    abort_arg *a = (abort_arg *)p;
    sleep_ms(a->delay_ms);
    jxl_request_abort(a->ctx);
    return 0;
}
#else
static void *abort_thread(void *p) {
    abort_arg *a = (abort_arg *)p;
    sleep_ms(a->delay_ms);
    jxl_request_abort(a->ctx);
    return NULL;
}
#endif

/* ----- per-file test ----- */

static void test_file(const char *path) {
    uint8_t *data;
    size_t len;
    jxl_ctx *ctx;
    jxl_doc *doc;
    jxl_image *ref[MAX_FRAMES];
    int nframes, f, aborted_runs = 0, total_runs = 0;
    size_t base_live;

    g_file = path;
    data = read_file(path, &len);
    if (!data) {
        printf("skip %s (cannot read)\n", path);
        return;
    }
    ctx = jxl_ctx_new(t_alloc, t_free, NULL, NULL);
    CHECK(ctx != NULL, "jxl_ctx_new");
    if (!ctx) { free(data); return; }
    jxl_ctx_set_srgb_output(ctx, 1);

    doc = jxl_doc_open(ctx, data, len);
    CHECK(doc != NULL, "jxl_doc_open");
    if (!doc) { jxl_ctx_free(ctx); free(data); return; }

    nframes = jxl_doc_frame_count(doc);
    if (nframes > MAX_FRAMES) nframes = MAX_FRAMES;
    for (f = 0; f < nframes; f++) {
        ref[f] = jxl_frame_render(doc, f, JXLDEC_FORMAT_RGBA32);
        CHECK(ref[f] != NULL, "reference render of frame %d", f);
        if (!ref[f]) nframes = f;
    }
    if (nframes == 0) goto out;
    base_live = g_live;

    /* 1. An abort requested before a render starts does not affect it. */
    jxl_request_abort(ctx);
    {
        jxl_image *img = jxl_frame_render(doc, nframes - 1, JXLDEC_FORMAT_RGBA32);
        CHECK(same_image(img, ref[nframes - 1]),
              "render after an earlier abort request differs");
        jxl_image_destroy(ctx, img);
    }

    /* 2. Deterministic sweep: abort at the k-th allocation of a render. */
    for (f = 0; f < nframes; f++) {
        unsigned long total, k, step, pts[SWEEP_POINTS + 4];
        jxl_image *img;
        int i, np = 0;

        /* A render allocates the same way every time, so count once. */
        g_count = 0;
        g_abort_at = 0;
        img = jxl_frame_render(doc, f, JXLDEC_FORMAT_RGBA32);
        total = g_count;
        jxl_image_destroy(ctx, img);

        /* The first two allocations, evenly spaced ones across the render,
           and one past the last, which never fires (the no-abort case). */
        step = total / SWEEP_POINTS ? total / SWEEP_POINTS : 1;
        pts[np++] = 1;
        if (total >= 2) pts[np++] = 2;
        for (k = step; k <= total && np < SWEEP_POINTS + 2; k += step) {
            if (k > 2) pts[np++] = k;
        }
        pts[np++] = total + 1;

        for (i = 0; i < np; i++) {
            k = pts[i];
            g_count = 0;
            g_abort_at = k;
            img = jxl_frame_render(doc, f, JXLDEC_FORMAT_RGBA32);
            g_abort_at = 0;
            total_runs++;
            if (!img) {
                aborted_runs++;
            } else {
                CHECK(same_image(img, ref[f]),
                      "frame %d, abort at allocation %lu of %lu: rendered "
                      "but the pixels differ", f, k, total);
            }
            if (k == 1) {
                CHECK(img == NULL, "frame %d: abort at the first allocation "
                      "did not stop the render", f);
            }
            jxl_image_destroy(ctx, img);
            CHECK(g_live == base_live, "frame %d, abort at allocation %lu: "
                  "%ld bytes leaked", f, k, (long)(g_live - base_live));
            /* The doc is still usable and gives the same frame. */
            img = jxl_frame_render(doc, f, JXLDEC_FORMAT_RGBA32);
            CHECK(same_image(img, ref[f]), "frame %d: render after an abort "
                  "at allocation %lu differs", f, k);
            jxl_image_destroy(ctx, img);
        }
    }

    /* 3. jxl_frame_render_into and jxl_doc_frame_info report it as -1. */
    {
        jxl_render_info ri;
        jxl_frame_info fi;
        if (jxl_frame_render_info(doc, 0, JXLDEC_FORMAT_RGBA32, &ri) == 0) {
            size_t stride = (size_t)ri.width * 4;
            uint8_t *buf = (uint8_t *)malloc(stride * (size_t)ri.height);
            if (buf) {
                g_count = 0;
                g_abort_at = 1;
                CHECK(jxl_frame_render_into(doc, 0, JXLDEC_FORMAT_RGBA32, buf,
                                            (int)stride) == -1,
                      "jxl_frame_render_into did not fail on abort");
                g_abort_at = 0;
                CHECK(jxl_frame_render_into(doc, 0, JXLDEC_FORMAT_RGBA32, buf,
                                            (int)stride) == 0,
                      "jxl_frame_render_into after an abort failed");
                CHECK(memcmp(buf, ref[0]->data, stride) == 0,
                      "jxl_frame_render_into after an abort: first row differs");
                free(buf);
            }
        }
        if (jxl_doc_frame_count(doc) > 1) {
            g_count = 0;
            g_abort_at = 1;
            CHECK(jxl_doc_frame_info(doc, 1, &fi) == -1,
                  "jxl_doc_frame_info did not fail on abort");
            g_abort_at = 0;
            CHECK(jxl_doc_frame_info(doc, 1, &fi) == 0,
                  "jxl_doc_frame_info after an abort failed");
        }
        CHECK(g_live == base_live, "render_into / frame_info: %ld bytes leaked",
              (long)(g_live - base_live));
    }

    /* 4. Threaded: the request comes from another thread, as in a viewer. */
    {
        static const int delays[] = {0, 1, 2, 5, 10, 25};
        size_t d;
        for (d = 0; d < sizeof(delays) / sizeof(delays[0]); d++) {
            abort_arg a;
            jxl_image *img;
            int fr = (int)(d % (size_t)nframes);
#ifdef _WIN32
            HANDLE th;
#else
            pthread_t th;
#endif
            a.ctx = ctx;
            a.delay_ms = delays[d];
#ifdef _WIN32
            th = CreateThread(NULL, 0, abort_thread, &a, 0, NULL);
            CHECK(th != NULL, "CreateThread");
            if (!th) continue;
#else
            if (pthread_create(&th, NULL, abort_thread, &a) != 0) {
                CHECK(0, "pthread_create");
                continue;
            }
#endif
            img = jxl_frame_render(doc, fr, JXLDEC_FORMAT_RGBA32);
#ifdef _WIN32
            WaitForSingleObject(th, INFINITE);
            CloseHandle(th);
#else
            pthread_join(th, NULL);
#endif
            if (img) {
                CHECK(same_image(img, ref[fr]), "threaded abort after %d ms: "
                      "rendered but the pixels differ", delays[d]);
            }
            jxl_image_destroy(ctx, img);
            CHECK(g_live == base_live, "threaded abort after %d ms: %ld bytes "
                  "leaked", delays[d], (long)(g_live - base_live));
            img = jxl_frame_render(doc, fr, JXLDEC_FORMAT_RGBA32);
            CHECK(same_image(img, ref[fr]), "render after a threaded abort "
                  "differs");
            jxl_image_destroy(ctx, img);
        }
    }

    CHECK(aborted_runs > 0, "no render was ever aborted");
    printf("ok   %s: %d frame(s), %d of %d sweep renders aborted\n", path,
           nframes, aborted_runs, total_runs);

out:
    for (f = 0; f < nframes; f++) jxl_image_destroy(ctx, ref[f]);
    jxl_doc_close(doc);
    jxl_ctx_free(ctx);
    free(data);
}

/* A frame count whose walk was aborted must not be cached. */
static void test_frame_count(const char *path) {
    uint8_t *data;
    size_t len;
    jxl_ctx *ctx;
    jxl_doc *doc;
    int full, cut;

    g_file = path;
    data = read_file(path, &len);
    if (!data) return;
    ctx = jxl_ctx_new(t_alloc, t_free, NULL, NULL);
    doc = ctx ? jxl_doc_open(ctx, data, len) : NULL;
    if (doc) full = jxl_doc_frame_count(doc);
    else full = 0;
    jxl_doc_close(doc);
    if (full > 1) {
        doc = jxl_doc_open(ctx, data, len);
        g_count = 0;
        g_abort_at = 1;
        cut = jxl_doc_frame_count(doc);
        g_abort_at = 0;
        CHECK(cut == 1, "aborted frame count returned %d, want 1", cut);
        CHECK(jxl_doc_frame_count(doc) == full,
              "frame count after an aborted count is %d, want %d",
              jxl_doc_frame_count(doc), full);
        jxl_doc_close(doc);
        printf("ok   %s: aborted frame count not cached (%d frames)\n", path, full);
    }
    jxl_ctx_free(ctx);
    free(data);
}

int main(int argc, char **argv) {
    int i;
    if (argc < 2) {
        fprintf(stderr, "usage: abort_test file.jxl ...\n");
        return 2;
    }
    for (i = 1; i < argc; i++) {
        test_file(argv[i]);
        test_frame_count(argv[i]);
    }
    if (fails) {
        fprintf(stderr, "abort_test: %d check(s) failed\n", fails);
        return 1;
    }
    printf("abort_test: ok\n");
    return 0;
}
