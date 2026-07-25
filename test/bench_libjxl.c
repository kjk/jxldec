/* bench_libjxl.c -- the libjxl half of the benchmark harness.
 *
 * Kept in its own translation unit: our public header and libjxl's both
 * define JXL_SIG_*, so they cannot be included together.
 */
#include <jxl/decode.h>

#include <stdint.h>
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

double jxl_bench_libjxl(const uint8_t *data, size_t len);

double jxl_bench_libjxl(const uint8_t *data, size_t len) {
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
                fmt.num_channels = info.num_color_channels +
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
