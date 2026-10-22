/* oversize_render.c -- regression for the output-stride integer overflow in
 * jxl_frame_render (render.c).
 *
 * The JPEG XL size header leaves width/height uncapped (up to 2^30). The
 * output path computed the row stride as `info.width * jxl_format_bpp(fmt)`
 * in int; for width 2^30 at 4 bytes/pixel that is 2^32, which overflows the
 * signed int to 0. The buffer was then sized stride*height = 0 -> a 1-byte
 * allocation, and write_pixels wrote full width*height*bpp rows into it: a
 * heap overflow.
 *
 * The overflow only bites once the image-sized (multi-gigabyte) canvas decode
 * succeeds, so it is not a cheap end-to-end crash -- the same shape as the
 * IW44 bug in djvudec, and for the same reason a crafted .jxl cannot drive it
 * under a test memory budget (the decode is rejected first). So this checks
 * the guard directly: jxl_output_buffer_size now does the arithmetic in 64
 * bits and rejects any output whose stride does not fit the int field, and
 * jxl_frame_render calls it before decoding.
 */
#include "jxl.h"
#include "jxl_internal.h"

#include <stdio.h>

static int fails;

static void expect_reject(int w, int h, int bpp) {
    int stride = 123;
    size_t total = 123;
    if (jxl_output_buffer_size(w, h, bpp, &stride, &total) == 0) {
        fprintf(stderr,
                "FAIL: %dx%d @ %d bpp accepted (stride=%d total=%zu), want reject\n",
                w, h, bpp, stride, total);
        fails++;
    }
}

static void expect_ok(int w, int h, int bpp, int want_stride, size_t want_total) {
    int stride = 0;
    size_t total = 0;
    if (jxl_output_buffer_size(w, h, bpp, &stride, &total) != 0) {
        fprintf(stderr, "FAIL: %dx%d @ %d bpp rejected, want accept\n", w, h, bpp);
        fails++;
        return;
    }
    if (stride != want_stride || total != want_total) {
        fprintf(stderr,
                "FAIL: %dx%d @ %d bpp -> stride=%d total=%zu, want stride=%d total=%zu\n",
                w, h, bpp, stride, total, want_stride, want_total);
        fails++;
    }
}

int main(void) {
    /* The bug: width 2^30 at 4 bpp -> 2^32 wraps the int stride to 0. */
    expect_reject(1 << 30, 1, 4);
    /* width 2^29 at 8 bpp -> 2^32 as well. */
    expect_reject(1 << 29, 1, 8);
    /* A stride that lands in (INT_MAX, 2^32) is negative as int -- also bad. */
    expect_reject(1 << 30, 1, 3); /* 3 * 2^30 > INT_MAX */
    /* Non-positive arguments. */
    expect_reject(0, 1, 4);
    expect_reject(1, 0, 4);
    expect_reject(16, 16, 0);

    /* Ordinary images are unaffected and keep the exact stride/total. */
    expect_ok(100, 100, 4, 400, (size_t)400 * 100);
    expect_ok(1, 1, 3, 3, 3);
    /* Large but representable: 16384 wide @ 8 bpp = 131072-byte stride. */
    expect_ok(16384, 16384, 8, 131072, (size_t)131072 * 16384);
    /* Widest that still fits an int stride at 1 bpp: 2^30 @ 1 bpp. */
    expect_ok(1 << 30, 1, 1, 1 << 30, (size_t)(1 << 30));

    if (fails) {
        fprintf(stderr, "oversize_render: %d check(s) failed\n", fails);
        return 1;
    }
    printf("oversize_render: ok\n");
    return 0;
}
