/* fuzz_entropy.c -- one entropy-coded stream on its own (coding.c).
 *
 * A distribution set -- LZ77 parameters, the context map (simple or itself
 * entropy-coded, with move-to-front), the hybrid-uint configs, and prefix
 * or ANS histograms with their alias tables -- then symbols read from it
 * across every context, LZ77 copies included, and the end-of-stream check.
 * Every stream in a JPEG XL file starts this way (the ICC stream, TOC
 * permutations, MA trees, Modular samples, VarDCT coefficients), so seeds
 * come from all of them.
 *
 * Input: bit skip, num_dist as 2 bytes little-endian, the bitstream
 * (fuzz_common.h).
 *
 *   bun cmd/fuzz.ts -target entropy
 */
#include "jxl_internal.h"
#include "fuzz_common.h"

/* Symbols read before giving up. LZ77 copies can repeat a short stream's
   worth of symbols indefinitely, so this, not the input, bounds the run. */
#define FUZZ_MAX_SYMBOLS 16384

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const uint8_t *params, *bits;
    size_t nbits;
    uint32_t skip, num_dist, i;
    jxl_ctx *ctx;
    jxl_br br;
    jxl_dec dec;

    if (fuzz_split(data, size, 2, &params, &bits, &nbits, &skip) != 0)
        return 0;
    /* Real streams use 1 (ICC sub-streams) to several thousand contexts
       (VarDCT coefficients); keep the context map that size. */
    num_dist = fuzz_range(fuzz_u16(params), 8192);
    ctx = fuzz_ctx_new();
    if (!ctx) return 0;
    jxl_br_init(&br, bits, nbits);
    jxl_br_skip(&br, skip);
    if (jxl_dec_init(ctx, &dec, &br, num_dist) == 0) {
        jxl_dec_begin(&dec, &br);
        for (i = 0; i < FUZZ_MAX_SYMBOLS && !br.err && !dec.err; i++) {
            /* Walk the contexts in a data-dependent order so every
               cluster's histogram is reached, not just the first few. */
            uint32_t c = (i * 2654435761u >> 7) % num_dist;
            (void)jxl_dec_read(&dec, &br, c);
        }
        (void)jxl_dec_finalize(&dec);
        jxl_dec_free(&dec);
    }
    jxl_ctx_free(ctx);
    return 0;
}
