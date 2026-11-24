/* fuzz_icc.c -- the ICC profile stream on its own (icc.c).
 *
 * An encoded ICC profile is an entropy-coded byte stream (41 contexts)
 * followed by a predictive decoder that rebuilds the header, the tag table
 * and the tag data -- the part of the format with the most arithmetic on
 * lengths and offsets taken straight from the file. Input: one byte of bit
 * skip, then the bitstream jxl_read_icc reads (fuzz_common.h). Seeds are
 * cut from the corpus files that carry a profile.
 *
 *   bun cmd/fuzz.ts -target icc
 */
#include "jxl_internal.h"
#include "fuzz_common.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const uint8_t *params, *bits;
    size_t nbits;
    uint32_t skip;
    jxl_ctx *ctx;
    jxl_br br;
    uint8_t *icc = NULL;
    size_t icc_len = 0;

    if (fuzz_split(data, size, 0, &params, &bits, &nbits, &skip) != 0)
        return 0;
    ctx = fuzz_ctx_new();
    if (!ctx) return 0;
    jxl_br_init(&br, bits, nbits);
    jxl_br_skip(&br, skip);
    if (jxl_read_icc(ctx, &br, &icc, &icc_len) == 0 && icc && icc_len) {
        volatile uint8_t sink = icc[icc_len - 1];
        (void)sink;
    }
    jxl_free(ctx, icc);
    jxl_ctx_free(ctx);
    return 0;
}
