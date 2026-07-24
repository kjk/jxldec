/* jxl_test.c -- CLI harness for the C JPEG XL decoder (jbig2dec-flavored).
 *
 *   jxl_test -info in.jxl
 *   jxl_test -out out.ppm in.jxl          (PGM for grayscale, PPM for color)
 *   jxl_test -frame N -out out.ppm in.jxl
 *
 * Writes binary PNM so the output can be compared byte-for-byte against
 * `djxl in.jxl out.ppm`.
 */
#include "jxl.h"
#include "jxl_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void on_error(void *user, jxl_severity sev, const char *msg) {
    (void)user;
    if (sev >= JXL_SEVERITY_WARNING) fprintf(stderr, "jxl: %s\n", msg);
}

static uint8_t *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    long n;
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    buf = (uint8_t *)malloc((size_t)n ? (size_t)n : 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static const char *cs_name(jxl_color_space cs) {
    switch (cs) {
        case JXL_CS_RGB: return "rgb";
        case JXL_CS_GRAY: return "gray";
        case JXL_CS_XYB: return "xyb";
        default: return "unknown";
    }
}

static int cmd_info(jxl_ctx *ctx, const uint8_t *data, size_t len) {
    jxl_doc *doc = jxl_doc_open(ctx, data, len);
    jxl_image_info info;
    if (!doc) return 1;
    if (jxl_doc_info(doc, &info) != 0) {
        jxl_doc_close(doc);
        return 1;
    }
    printf("size %dx%d\n", info.width, info.height);
    printf("bits_per_sample %d\n", info.bits_per_sample);
    if (info.exponent_bits) printf("exponent_bits %d\n", info.exponent_bits);
    printf("color_channels %d\n", info.num_color_channels);
    printf("color_space %s\n", cs_name(info.color_space));
    printf("extra_channels %d\n", info.num_extra_channels);
    if (info.alpha_bits) {
        printf("alpha_bits %d\n", info.alpha_bits);
        printf("alpha_premultiplied %d\n", info.alpha_premultiplied);
    }
    printf("orientation %d\n", info.orientation);
    printf("uses_original_profile %d\n", info.uses_original_profile);
    printf("have_animation %d\n", info.have_animation);
    printf("have_preview %d\n", info.have_preview);
    if (info.intrinsic_width != info.width || info.intrinsic_height != info.height) {
        printf("intrinsic_size %dx%d\n", info.intrinsic_width, info.intrinsic_height);
    }
    {
        size_t icc_len = 0;
        if (jxl_doc_icc_profile(doc, &icc_len)) printf("icc_size %u\n", (unsigned)icc_len);
    }
    jxl_doc_close(doc);
    return 0;
}

/* -frames: parse and dump every frame header in the codestream. */
static int cmd_frames(jxl_ctx *ctx, const uint8_t *data, size_t len) {
    jxl_doc *doc = jxl_doc_open(ctx, data, len);
    jxl_br br;
    int idx = 0;
    if (!doc) return 1;

    jxl_br_init(&br, doc->container.cs, doc->container.cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    for (;;) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t end;
        if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) != 0) break;
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            break;
        }
        printf("frame %d: type %d enc %s %ux%u+%d+%d flags 0x%x passes %u "
               "groups %u lfgroups %u toc %u sections %u bytes last %d\n",
               idx, (int)fh.frame_type,
               fh.encoding == JXL_ENC_MODULAR ? "modular" : "vardct",
               (unsigned)fh.width, (unsigned)fh.height, (int)fh.x0, (int)fh.y0,
               (unsigned)fh.flags, (unsigned)fh.passes.num_passes,
               (unsigned)jxl_frame_num_groups(&fh),
               (unsigned)jxl_frame_num_lf_groups(&fh), (unsigned)toc.count,
               (unsigned)toc.total_size, fh.is_last);
        end = toc.end_off + toc.total_size;
        idx++;
        {
            int last = fh.is_last;
            jxl_toc_free(ctx, &toc);
            jxl_frame_header_free(ctx, &fh);
            if (last) break;
        }
        if (end >= doc->container.cs_len) break;
        jxl_br_seek_byte(&br, end);
    }
    jxl_doc_close(doc);
    return 0;
}

static int write_pnm(const char *path, const jxl_image *img) {
    FILE *f = fopen(path, "wb");
    int comps, maxval, y;
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    switch (img->format) {
        case JXL_FORMAT_GRAY8: comps = 1; maxval = 255; break;
        case JXL_FORMAT_GRAY16: comps = 1; maxval = 65535; break;
        case JXL_FORMAT_RGB24: comps = 3; maxval = 255; break;
        case JXL_FORMAT_RGB48: comps = 3; maxval = 65535; break;
        default:
            fprintf(stderr, "pnm: unsupported format %d\n", (int)img->format);
            fclose(f);
            return 1;
    }
    fprintf(f, "P%d\n%d %d\n%d\n", comps == 1 ? 5 : 6, img->width, img->height,
            maxval);
    if (maxval == 255) {
        for (y = 0; y < img->height; y++) {
            fwrite(img->data + (size_t)y * img->stride, 1,
                   (size_t)img->width * comps, f);
        }
    } else {
        /* PNM is big-endian; our 16-bit output is native-endian. */
        int x;
        for (y = 0; y < img->height; y++) {
            const uint16_t *row = (const uint16_t *)(img->data + (size_t)y * img->stride);
            for (x = 0; x < img->width * comps; x++) {
                uint8_t be[2];
                be[0] = (uint8_t)(row[x] >> 8);
                be[1] = (uint8_t)row[x];
                fwrite(be, 1, 2, f);
            }
        }
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    const char *in_path = NULL;
    const char *out_path = NULL;
    int do_info = 0, do_frames = 0, frame_no = 0;
    int i, rc = 0;
    uint8_t *data;
    size_t len;
    jxl_ctx *ctx;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-info") == 0) do_info = 1;
        else if (strcmp(argv[i], "-frames") == 0) do_frames = 1;
        else if (strcmp(argv[i], "-out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "-frame") == 0 && i + 1 < argc) frame_no = atoi(argv[++i]);
        else if (argv[i][0] != '-') in_path = argv[i];
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (!in_path) {
        fprintf(stderr,
                "usage: jxl_test [-info] [-frames] [-frame N] [-out out.pnm] in.jxl\n");
        return 2;
    }

    data = read_file(in_path, &len);
    if (!data) return 1;

    ctx = jxl_ctx_new(NULL, NULL, on_error, NULL);
    if (!ctx) {
        free(data);
        return 1;
    }

    if (do_info) rc = cmd_info(ctx, data, len);
    if (!rc && do_frames) rc = cmd_frames(ctx, data, len);
    if (!rc && out_path) {
        jxl_doc *doc = jxl_doc_open(ctx, data, len);
        jxl_image *img = NULL;
        if (doc) img = jxl_frame_render(doc, frame_no, JXL_FORMAT_NATIVE);
        if (!img) rc = 1;
        else rc = write_pnm(out_path, img);
        jxl_image_destroy(ctx, img);
        jxl_doc_close(doc);
    }
    if (!do_info && !do_frames && !out_path) rc = cmd_info(ctx, data, len);

    jxl_ctx_free(ctx);
    free(data);
    return rc;
}
