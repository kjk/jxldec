/* jxl_bench.c -- decode timing, ours against libjxl, in one process.
 *
 *   jxl_bench [-runs N] file.jxl ...
 *
 * Both decoders produce 8-bit interleaved samples of the first frame, which
 * is what an application like SumatraPDF asks for. libjxl runs
 * single-threaded (no JxlThreadParallelRunner) because that is the only
 * configuration we implement.
 *
 * Each file is decoded `runs` times per decoder and the best time is
 * reported: the fastest run is the one least perturbed by the scheduler.
 * Output is one line per file plus a total, in the shape djvudec's bench
 * prints.
 */
#include "jxl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

static void on_error(void *user, jxl_severity sev, const char *msg) {
    (void)user;
    (void)sev;
    (void)msg;
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

/* Returns the decode time in ms, or -1 on failure. */
static double bench_ours(const uint8_t *data, size_t len) {
    jxl_ctx *ctx = jxl_ctx_new(NULL, NULL, on_error, NULL);
    jxl_image *img;
    double t0, dt;
    if (!ctx) return -1;
    t0 = now_ms();
    img = jxl_decode(ctx, data, len, JXL_FORMAT_NATIVE);
    dt = now_ms() - t0;
    if (!img) dt = -1;
    jxl_image_destroy(ctx, img);
    jxl_ctx_free(ctx);
    return dt;
}

/* Defined in bench_libjxl.c, which cannot share a translation unit with
   jxl.h: our public JXL_SIG_* names collide with libjxl's. */
double jxl_bench_libjxl(const uint8_t *data, size_t len);

int main(int argc, char **argv) {
    int runs = 3;
    int i, r, nfiles = 0;
    double tot_ours = 0, tot_ref = 0;
    size_t tot_bytes = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
        }
    }
    printf("%-46s %10s %10s %8s\n", "file", "libjxl", "jxldec", "ratio");

    for (i = 1; i < argc; i++) {
        const char *path = argv[i];
        uint8_t *data;
        size_t len;
        double best_ours = -1, best_ref = -1;
        const char *base;

        if (argv[i][0] == '-') {
            if (strcmp(argv[i], "-runs") == 0) i++;
            continue;
        }
        data = read_file(path, &len);
        if (!data) {
            fprintf(stderr, "cannot read %s\n", path);
            continue;
        }
        for (r = 0; r < runs; r++) {
            double a = bench_ours(data, len);
            double b = jxl_bench_libjxl(data, len);
            if (a >= 0 && (best_ours < 0 || a < best_ours)) best_ours = a;
            if (b >= 0 && (best_ref < 0 || b < best_ref)) best_ref = b;
        }
        free(data);

        base = strrchr(path, '/');
        if (!base) base = strrchr(path, '\\');
        base = base ? base + 1 : path;

        if (best_ours < 0 || best_ref < 0) {
            printf("%-46s %10s %10s %8s\n", base,
                   best_ref < 0 ? "fail" : "-", best_ours < 0 ? "fail" : "-", "-");
            continue;
        }
        printf("%-46s %9.2fms %9.2fms %7.2fx\n", base, best_ref, best_ours,
               best_ref > 0 ? best_ours / best_ref : 0.0);
        tot_ours += best_ours;
        tot_ref += best_ref;
        tot_bytes += len;
        nfiles++;
    }

    if (nfiles) {
        printf("%-46s %9.2fms %9.2fms %7.2fx\n", "TOTAL", tot_ref, tot_ours,
               tot_ref > 0 ? tot_ours / tot_ref : 0.0);
        printf("%d file(s), %.1f MB\n", nfiles, (double)tot_bytes / (1024 * 1024));
    }
    return 0;
}
