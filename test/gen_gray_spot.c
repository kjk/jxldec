/* gen_gray_spot.c -- writes the two fixtures in test/data/ that no encoder
 * tool will: a grayscale image carrying a spot-colour extra channel.
 *
 *   gen_gray_spot out_lossless.jxl out_lossy.jxl
 *
 * cjxl has no way to declare a spot channel and the conformance suite's one
 * spot file is RGB, so the gray case could only be checked against libjxl by
 * making a file with libjxl's encoder API. The image is a 64x48 horizontal
 * ramp; the spot channel is a soft disc of an orange ink at 70% solidity.
 * One file is lossless Modular (a single gray channel is coded), the other
 * lossy VarDCT (XYB, three planes that collapse to gray after the colour
 * transform) -- the two routes a gray image takes through the decoder.
 *
 * Not part of any build: the fixtures are checked in. To regenerate (MSVC,
 * after `bun cmd/get-deps.ts`), from the repo root:
 *
 *   cl -nologo -MD -DJXL_STATIC_DEFINE -Ideps/libjxl/lib/include
 *      -Ideps/libjxl-build/lib/include test/gen_gray_spot.c
 *      -Fe:out/gen_gray_spot.exe -link deps/libjxl-build/lib/jxl.lib
 *      deps/libjxl-build/lib/jxl_cms.lib
 *      deps/libjxl-build/third_party/brotli/brotlidec.lib
 *      deps/libjxl-build/third_party/brotli/brotlienc.lib
 *      deps/libjxl-build/third_party/brotli/brotlicommon.lib
 *      deps/libjxl-build/third_party/highway/hwy.lib
 */
#include <jxl/encode.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 64
#define H 48

static int encode(const char *path, int lossless) {
    static uint8_t gray[W * H], spot[W * H];
    uint8_t *out = NULL;
    size_t cap = 1 << 16, used = 0;
    JxlEncoder *enc = JxlEncoderCreate(NULL);
    JxlEncoderFrameSettings *fs;
    JxlBasicInfo info;
    JxlExtraChannelInfo ec;
    JxlColorEncoding ce;
    JxlPixelFormat fmt = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    JxlEncoderStatus st;
    FILE *f;
    int x, y, rc = 1;

    for (y = 0; y < H; y++) {
        for (x = 0; x < W; x++) {
            int dx = x - W / 2, dy = y - H / 2;
            int d2 = dx * dx + dy * dy;
            gray[y * W + x] = (uint8_t)(x * 255 / (W - 1));
            spot[y * W + x] = (uint8_t)(d2 >= 400 ? 0 : 255 - d2 * 255 / 400);
        }
    }
    if (!enc) return 1;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = W;
    info.ysize = H;
    info.bits_per_sample = 8;
    info.num_color_channels = 1;
    info.num_extra_channels = 1;
    info.uses_original_profile = lossless ? JXL_TRUE : JXL_FALSE;
    if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS) goto done;

    JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_SPOT_COLOR, &ec);
    ec.bits_per_sample = 8;
    ec.spot_color[0] = 0.9f;
    ec.spot_color[1] = 0.4f;
    ec.spot_color[2] = 0.1f;
    ec.spot_color[3] = 0.7f;
    if (JxlEncoderSetExtraChannelInfo(enc, 0, &ec) != JXL_ENC_SUCCESS) goto done;

    JxlColorEncodingSetToSRGB(&ce, JXL_TRUE);
    if (JxlEncoderSetColorEncoding(enc, &ce) != JXL_ENC_SUCCESS) goto done;

    fs = JxlEncoderFrameSettingsCreate(enc, NULL);
    if (lossless) {
        if (JxlEncoderSetFrameLossless(fs, JXL_TRUE) != JXL_ENC_SUCCESS) goto done;
    } else {
        if (JxlEncoderSetFrameDistance(fs, 1.0f) != JXL_ENC_SUCCESS) goto done;
        if (JxlEncoderSetExtraChannelDistance(fs, 0, 0.0f) != JXL_ENC_SUCCESS)
            goto done;
    }
    if (JxlEncoderAddImageFrame(fs, &fmt, gray, sizeof(gray)) != JXL_ENC_SUCCESS)
        goto done;
    if (JxlEncoderSetExtraChannelBuffer(fs, &fmt, spot, sizeof(spot), 0) !=
        JXL_ENC_SUCCESS)
        goto done;
    JxlEncoderCloseInput(enc);

    out = (uint8_t *)malloc(cap);
    if (!out) goto done;
    for (;;) {
        uint8_t *next = out + used;
        size_t avail = cap - used;
        st = JxlEncoderProcessOutput(enc, &next, &avail);
        used = cap - avail;
        if (st != JXL_ENC_NEED_MORE_OUTPUT) break;
        cap *= 2;
        out = (uint8_t *)realloc(out, cap);
        if (!out) goto done;
    }
    if (st != JXL_ENC_SUCCESS) goto done;
    f = fopen(path, "wb");
    if (!f) goto done;
    fwrite(out, 1, used, f);
    fclose(f);
    printf("%s: %u bytes\n", path, (unsigned)used);
    rc = 0;

done:
    if (rc) fprintf(stderr, "failed to encode %s\n", path);
    free(out);
    JxlEncoderDestroy(enc);
    return rc;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: gen_gray_spot out_lossless.jxl out_lossy.jxl\n");
        return 2;
    }
    return encode(argv[1], 1) || encode(argv[2], 0);
}
