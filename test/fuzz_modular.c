/* fuzz_modular.c -- one Modular stream on its own (modular.c).
 *
 * What decode.c does with a frame's GlobalModular section: the optional
 * global MA tree, the stream header (transforms: RCT, palette with deltas,
 * squeeze; the weighted-predictor header; a local tree), the forward walk of
 * the transforms over the declared channels, the sample decode through
 * every predictor and specialised loop, and the inverse transforms. That is
 * the whole of lossless decoding, at a channel size the input chooses
 * rather than a frame's.
 *
 * Input: bit skip; channel count (1..8); bit depth; for each channel width
 * and height (2 bytes each, little-endian) and a shift; then the bitstream
 * (fuzz_common.h). Seeds are the GlobalModular sections of the corpus's
 * single-group frames, cut out with their real geometry.
 *
 *   bun cmd/fuzz.ts -target modular
 */
#include "jxl_internal.h"
#include "fuzz_common.h"

/* Samples across all channels. Squeeze multiplies channels, palette adds
   some, and the per-sample cost is the point of the target, so keep it to
   one 256x256 group's worth of RGBA. */
#define FUZZ_MAX_SAMPLES ((uint64_t)256 * 256 * 4)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const uint8_t *params, *bits;
    size_t nbits;
    uint32_t skip, nspecs, depth, i;
    uint64_t samples = 0;
    jxl_mchan_spec specs[8];
    jxl_ctx *ctx;
    jxl_br br;
    jxl_ma_config global_ma;
    int has_global = 0;
    jxl_modular mod;
    jxl_chanlist cl;

    if (size < 3) return 0;
    nspecs = fuzz_range(data[1], 8);
    if (fuzz_split(data, size, 2 + nspecs * 5, &params, &bits, &nbits,
                   &skip) != 0)
        return 0;
    depth = fuzz_range(params[1], 32);
    for (i = 0; i < nspecs; i++) {
        const uint8_t *p = params + 2 + i * 5;
        specs[i].w = fuzz_range(fuzz_u16(p), 1024);
        specs[i].h = fuzz_range(fuzz_u16(p + 2), 1024);
        specs[i].hshift = specs[i].vshift = (int)(p[4] % 4);
        samples += (uint64_t)specs[i].w * specs[i].h;
    }
    if (samples > FUZZ_MAX_SAMPLES) return 0;

    ctx = fuzz_ctx_new();
    if (!ctx) return 0;
    memset(&global_ma, 0, sizeof(global_ma));
    memset(&mod, 0, sizeof(mod));
    memset(&cl, 0, sizeof(cl));
    jxl_br_init(&br, bits, nbits);
    jxl_br_skip(&br, skip);

    /* The global tree, with decode.c's node limit for this many samples. */
    if (jxl_br_bool(&br)) {
        uint64_t limit = 1024 + samples / 16;
        if (limit > (1u << 22)) limit = 1u << 22;
        if (jxl_ma_config_read(ctx, &br, &global_ma, (size_t)limit) != 0)
            goto done;
        has_global = 1;
    }
    if (jxl_modular_init(ctx, &mod, &br, specs, nspecs,
                         has_global ? &global_ma : NULL, depth) != 0)
        goto done;
    if (jxl_modular_transform_channels(ctx, &mod, &cl) != 0) goto done;
    if (jxl_modular_decode(ctx, &mod, &cl, &br, 0) != 0) goto done;
    (void)jxl_modular_inverse(ctx, &mod, &cl);

done:
    jxl_chanlist_free(ctx, &cl);
    jxl_modular_free(ctx, &mod);
    if (has_global) jxl_ma_config_free(ctx, &global_ma);
    jxl_ctx_free(ctx);
    return 0;
}
