/* jxl_ref.c -- every displayed frame of a file, decoded by libjxl, as PAM.
 *
 *   jxl_ref in.jxl out_prefix      writes out_prefix.0.pam, out_prefix.1.pam ...
 *
 * `djxl in.jxl out.pam` writes only the first frame of an animation, so the
 * oracle in cmd/tests.ts cannot see frames after it -- which is exactly where
 * blending, reference slots and cropped frames live. This asks libjxl for
 * coalesced frames (each one composited onto the canvas, like jxl_frame_render
 * returns them) and writes them in the layout jxl_test uses: gray or RGB, plus
 * alpha when the image has it, 8-bit up to 8 bits per sample and 16-bit above.
 *
 * Prints the frame count on stdout. Built by `bun cmd/build.ts ref` against
 * libjxl's static libraries (MSVC only, like jxl_bench); driven by
 * `bun cmd/tests.ts -all-frames`.
 */
#include <jxl/decode.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static int write_pam(const char *prefix, int idx, const uint8_t *px,
                     uint32_t w, uint32_t h, uint32_t comps, int wide) {
    static const char *tuple[5] = {"", "GRAYSCALE", "GRAYSCALE_ALPHA", "RGB",
                                   "RGB_ALPHA"};
    char path[4096];
    FILE *f;
    snprintf(path, sizeof(path), "%s.%d.pam", prefix, idx);
    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    fprintf(f, "P7\nWIDTH %u\nHEIGHT %u\nDEPTH %u\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n",
            (unsigned)w, (unsigned)h, (unsigned)comps, wide ? 65535 : 255,
            tuple[comps]);
    fwrite(px, 1, (size_t)w * h * comps * (wide ? 2 : 1), f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    uint8_t *data, *out = NULL;
    size_t len, out_size = 0;
    JxlDecoder *dec;
    JxlBasicInfo info;
    JxlPixelFormat fmt;
    int nframes = 0, wide = 0, rc = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: jxl_ref in.jxl out_prefix\n");
        return 2;
    }
    data = read_file(argv[1], &len);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    dec = JxlDecoderCreate(NULL);
    if (!dec) return 1;
    memset(&info, 0, sizeof(info));
    memset(&fmt, 0, sizeof(fmt));
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
            JXL_DEC_SUCCESS ||
        JxlDecoderSetCoalescing(dec, JXL_TRUE) != JXL_DEC_SUCCESS)
        goto done;
    JxlDecoderSetInput(dec, data, len);
    JxlDecoderCloseInput(dec);

    for (;;) {
        JxlDecoderStatus st = JxlDecoderProcessInput(dec);
        if (st == JXL_DEC_BASIC_INFO) {
            if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) goto done;
            wide = info.bits_per_sample > 8 || info.exponent_bits_per_sample > 0;
            fmt.num_channels = info.num_color_channels + (info.alpha_bits ? 1 : 0);
            fmt.data_type = wide ? JXL_TYPE_UINT16 : JXL_TYPE_UINT8;
            /* PAM is big-endian. */
            fmt.endianness = JXL_BIG_ENDIAN;
            fmt.align = 0;
        } else if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            size_t need;
            if (JxlDecoderImageOutBufferSize(dec, &fmt, &need) != JXL_DEC_SUCCESS)
                goto done;
            if (need > out_size) {
                free(out);
                out = (uint8_t *)malloc(need);
                if (!out) goto done;
                out_size = need;
            }
            if (JxlDecoderSetImageOutBuffer(dec, &fmt, out, out_size) !=
                JXL_DEC_SUCCESS)
                goto done;
        } else if (st == JXL_DEC_FULL_IMAGE) {
            if (write_pam(argv[2], nframes, out, info.xsize, info.ysize,
                          fmt.num_channels, wide) != 0)
                goto done;
            nframes++;
        } else if (st == JXL_DEC_SUCCESS) {
            break;
        } else {
            fprintf(stderr, "libjxl: decode failed (status %d)\n", (int)st);
            goto done;
        }
    }
    printf("%d\n", nframes);
    rc = 0;

done:
    free(out);
    JxlDecoderDestroy(dec);
    free(data);
    return rc;
}
