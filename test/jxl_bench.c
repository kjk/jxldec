/* jxl_bench.c -- decode timing, ours against libjxl, in one process.
 *
 *   jxl_bench [-runs N] [-bgra] [-list paths.txt] file.jxl ...
 *
 * Both decoders produce 8-bit interleaved samples of the first frame, which
 * is what an application like SumatraPDF asks for. libjxl runs
 * single-threaded (no JxlThreadParallelRunner) because that is the only
 * configuration we implement.
 *
 * Our public names are JXLDEC_-prefixed, so jxl.h and libjxl's jxl/decode.h
 * coexist in this one translation unit.
 *
 * Each file is decoded `runs` times per decoder and the best time is
 * reported: the fastest run is the one least perturbed by the scheduler.
 * Output is one line per file plus a total, in the shape djvudec's bench
 * prints.
 */
#include "jxl.h"

#include <jxl/decode.h>

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

static int output_bgra;

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
    if (output_bgra) jxl_ctx_set_bgr(ctx, 1);
    t0 = now_ms();
    img = jxl_decode(ctx, data, len,
                     output_bgra ? JXLDEC_FORMAT_RGBA32 : JXLDEC_FORMAT_NATIVE);
    dt = now_ms() - t0;
    if (!img) dt = -1;
    jxl_image_destroy(ctx, img);
    jxl_ctx_free(ctx);
    return dt;
}

static double bench_libjxl(const uint8_t *data, size_t len) {
    JxlDecoder *dec = JxlDecoderCreate(NULL);
    JxlBasicInfo info;
    JxlPixelFormat fmt;
    uint8_t *out = NULL;
    size_t out_size = 0;
    double t0, dt = -1;
    int done = 0;

    if (!dec) return -1;
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return -1;
    }
    memset(&fmt, 0, sizeof(fmt));
    fmt.data_type = JXL_TYPE_UINT8;
    fmt.endianness = JXL_NATIVE_ENDIAN;
    fmt.align = 0;

    t0 = now_ms();
    JxlDecoderSetInput(dec, data, len);
    JxlDecoderCloseInput(dec);
    while (!done) {
        JxlDecoderStatus st = JxlDecoderProcessInput(dec);
        switch (st) {
            case JXL_DEC_BASIC_INFO:
                if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) goto fail;
                fmt.num_channels = output_bgra
                                       ? 4
                                       : info.num_color_channels +
                                             (info.alpha_bits ? 1 : 0);
                break;
            case JXL_DEC_NEED_IMAGE_OUT_BUFFER:
                if (JxlDecoderImageOutBufferSize(dec, &fmt, &out_size) !=
                    JXL_DEC_SUCCESS)
                    goto fail;
                free(out);
                out = (uint8_t *)malloc(out_size ? out_size : 1);
                if (!out) goto fail;
                if (JxlDecoderSetImageOutBuffer(dec, &fmt, out, out_size) !=
                    JXL_DEC_SUCCESS)
                    goto fail;
                break;
            case JXL_DEC_FULL_IMAGE:
                done = 1;
                break;
            default:
                goto fail;
        }
    }
    dt = now_ms() - t0;

fail:
    free(out);
    JxlDecoderDestroy(dec);
    return dt;
}

/* One file: decode `runs` times with each decoder, print the best of each. */
static void bench_one(const char *path, int runs, double *tot_ref,
                      double *tot_ours, size_t *tot_bytes, int *nfiles) {
    uint8_t *data;
    size_t len;
    double best_ours = -1, best_ref = -1;
    const char *base, *p;
    int r;

    data = read_file(path, &len);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", path);
        return;
    }
    for (r = 0; r < runs; r++) {
        double a = bench_ours(data, len);
        double b = bench_libjxl(data, len);
        if (a >= 0 && (best_ours < 0 || a < best_ours)) best_ours = a;
        if (b >= 0 && (best_ref < 0 || b < best_ref)) best_ref = b;
    }
    free(data);

    base = path;
    for (p = path; *p; p++) {
        if (*p == '/' || *p == '\\') base = p + 1;
    }

    if (best_ours < 0 || best_ref < 0) {
        printf("%-46s %10s %10s %8s\n", base, best_ref < 0 ? "fail" : "-",
               best_ours < 0 ? "fail" : "-", "-");
        fflush(stdout);
        return;
    }
    printf("%-46s %9.2fms %9.2fms %7.2fx\n", base, best_ref, best_ours,
           best_ref > 0 ? best_ours / best_ref : 0.0);
    fflush(stdout);
    *tot_ours += best_ours;
    *tot_ref += best_ref;
    *tot_bytes += len;
    (*nfiles)++;
}

/* Runs every path in a newline-separated list file. The whole corpus does not
   fit in a Windows command line, so the driver always passes -list. */
static int bench_list(const char *list_path, int runs, double *tot_ref,
                      double *tot_ours, size_t *tot_bytes, int *nfiles) {
    FILE *f = fopen(list_path, "rb");
    char line[4096];
    if (!f) {
        fprintf(stderr, "cannot read list %s\n", list_path);
        return -1;
    }
    while (fgets(line, (int)sizeof(line), f)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;
        bench_one(line, runs, tot_ref, tot_ours, tot_bytes, nfiles);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    const char *list_path = NULL;
    int runs = 3;
    int i, nfiles = 0;
    double tot_ours = 0, tot_ref = 0;
    size_t tot_bytes = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
        } else if (strcmp(argv[i], "-list") == 0 && i + 1 < argc) {
            list_path = argv[++i];
        } else if (strcmp(argv[i], "-bgra") == 0) {
            output_bgra = 1;
        }
    }
    printf("%-46s %10s %10s %8s\n", "file", "libjxl", "jxldec", "ratio");

    if (list_path &&
        bench_list(list_path, runs, &tot_ref, &tot_ours, &tot_bytes, &nfiles) != 0) {
        return 1;
    }
    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (strcmp(argv[i], "-runs") == 0 || strcmp(argv[i], "-list") == 0) i++;
            continue;
        }
        bench_one(argv[i], runs, &tot_ref, &tot_ours, &tot_bytes, &nfiles);
    }

    if (nfiles) {
        printf("%-46s %9.2fms %9.2fms %7.2fx\n", "TOTAL", tot_ref, tot_ours,
               tot_ref > 0 ? tot_ours / tot_ref : 0.0);
        printf("%d file(s), %.1f MB\n", nfiles, (double)tot_bytes / (1024 * 1024));
    }
    return 0;
}
