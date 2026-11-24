/* fuzz_seeds.c -- cut seeds for the component fuzz targets out of real files.
 *
 *   jxl_fuzz_seeds OUTDIR file.jxl ...
 *
 * Built with -DJXL_FUZZ_SEEDS (cmd/build.ts), so the decoder calls
 * jxl_fuzz_seed at the start of every ICC stream, every entropy-coded
 * stream and every GlobalModular section, with the bit it starts at and the
 * parameters it was given. Each call writes one seed in the layout
 * fuzz_common.h describes -- bit skip, parameters, then the bytes from that
 * point on (at most 64 KB) -- to OUTDIR/icc, OUTDIR/entropy or
 * OUTDIR/modular, which must exist. Driven by cmd/fuzz.ts when a component
 * target's corpus is empty.
 *
 * A large file starts hundreds of entropy streams that differ only in their
 * data, so per file only the first three of each kind and then every
 * power-of-two-th one are kept.
 */
#include "jxl_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED_MAX_BYTES 65536

static const char *out_dir;
static char label[512];
static int seen[3];
static int written;

static const char *const kind_dir[3] = {"icc", "entropy", "modular"};

void jxl_fuzz_seed(int kind, const jxl_br *br, size_t bitpos,
                   const uint8_t *param, size_t param_len) {
    char path[1024];
    size_t byte = bitpos / 8, avail;
    uint8_t skip = (uint8_t)(bitpos & 7);
    int n;
    FILE *f;

    if (kind < 0 || kind > 2) return;
    n = seen[kind]++;
    if (!(n < 3 || (n & (n - 1)) == 0)) return;
    if (byte >= br->len) return;
    if (kind == JXL_SEED_ENTROPY) {
        /* fuzz_entropy.c takes num_dist as 16 bits, up to 8192. */
        uint32_t nd = (uint32_t)param[0] | ((uint32_t)param[1] << 8) |
                      ((uint32_t)param[2] << 16) | ((uint32_t)param[3] << 24);
        if (nd == 0 || nd > 8192) return;
        param_len = 2;
    } else if (kind == JXL_SEED_MODULAR) {
        /* Only a section that decodes whole: every channel within one
           group, as in a single-group frame (fuzz_modular.c decodes all
           channels as stream 0). */
        size_t i;
        for (i = 0; i < param[0]; i++) {
            const uint8_t *p = param + 2 + i * 5;
            uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
            uint32_t h = (uint32_t)p[2] | ((uint32_t)p[3] << 8);
            if (w > 256 || h > 256) return;
        }
    }
    avail = br->len - byte;
    if (avail > SEED_MAX_BYTES) avail = SEED_MAX_BYTES;
    snprintf(path, sizeof(path), "%s/%s/%s.%d.bin", out_dir, kind_dir[kind],
             label, n);
    f = fopen(path, "wb");
    if (!f) return;
    fwrite(&skip, 1, 1, f);
    if (param_len) fwrite(param, 1, param_len, f);
    fwrite(br->data + byte, 1, avail, f);
    fclose(f);
    written++;
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
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

/* "dir/name.jxl" -> "dir_name": the conformance files are all input.jxl,
   so the parent directory is part of the name. */
static void make_label(const char *path) {
    const char *end = path + strlen(path), *p = end;
    int slashes = 0;
    size_t k = 0;
    while (p > path && slashes < 2) {
        p--;
        if (*p == '/' || *p == '\\') slashes++;
    }
    if (*p == '/' || *p == '\\') p++;
    for (; p < end && k + 1 < sizeof(label); p++) {
        char c = *p;
        label[k++] = (c == '/' || c == '\\' || c == ':') ? '_' : c;
    }
    label[k] = 0;
}

int main(int argc, char **argv) {
    int i;
    if (argc < 3) {
        fprintf(stderr, "usage: jxl_fuzz_seeds OUTDIR file.jxl ...\n");
        return 2;
    }
    out_dir = argv[1];
    for (i = 2; i < argc; i++) {
        size_t len;
        uint8_t *data = read_file(argv[i], &len);
        jxl_ctx *ctx;
        jxl_doc *doc;
        if (!data) continue;
        make_label(argv[i]);
        memset(seen, 0, sizeof(seen));
        ctx = jxl_ctx_new(NULL, NULL, NULL, NULL);
        doc = ctx ? jxl_doc_open(ctx, data, len) : NULL;
        if (doc) {
            int n = jxl_doc_frame_count(doc), f;
            for (f = 0; f < n && f < 4; f++) {
                jxl_image *img = jxl_frame_render(doc, f, JXLDEC_FORMAT_NATIVE);
                if (img) jxl_image_destroy(ctx, img);
            }
            jxl_doc_close(doc);
        }
        if (ctx) jxl_ctx_free(ctx);
        free(data);
    }
    printf("%d\n", written);
    return 0;
}
