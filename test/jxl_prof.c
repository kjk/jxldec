/* jxl_prof.c -- decode one file repeatedly, for the profiler.
 *
 *   jxl_prof [-runs N] file.jxl
 *
 * Only our decoder runs, so a sampling profile of this process is a profile
 * of jxldec alone. Built with debug info so winperf can symbolize it; see
 * `bun cmd/prof.ts`.
 *
 * The decode loop is bracketed by winperf section marks, so reading the file,
 * process startup and teardown are dropped from the profile rather than
 * diluting it. The calls are no-ops when the process is not running under
 * `winperf record`, so this harness still runs normally on its own.
 */
#include "jxl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
/* Vendored control client; calls are no-ops when winperf is not recording. */
#include "winperf_control.h"
#else
static void winperf_profile_start(void) {}
static void winperf_profile_stop(void) {}
#endif

static void on_error(void *user, jxl_severity sev, const char *msg) {
    (void)user;
    (void)sev;
    (void)msg;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    int runs = 10, i;
    FILE *f;
    long n;
    uint8_t *data;
    size_t len;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: jxl_prof [-runs N] file.jxl\n");
        return 2;
    }
    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "cannot read %s\n", path);
        return 1;
    }
    fclose(f);
    len = (size_t)n;

    for (i = 0; i < runs; i++) {
        jxl_ctx *ctx;
        jxl_image *img;

        winperf_profile_start();
        ctx = jxl_ctx_new(NULL, NULL, on_error, NULL);
        img = jxl_decode(ctx, data, len, JXLDEC_FORMAT_NATIVE);
        winperf_profile_stop();
        if (!img) {
            fprintf(stderr, "decode failed: %s\n", path);
            return 1;
        }
        jxl_image_destroy(ctx, img);
        jxl_ctx_free(ctx);
    }
    free(data);
    printf("%d run(s) of %s\n", runs, path);
    return 0;
}
