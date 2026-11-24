/* fuzz_common.h -- shared by the fuzz targets (test/fuzz_*.c).
 *
 * The budgeted allocator every target hands the decoder, and the input
 * layout of the component targets, which feed one decoder component a
 * bitstream that starts mid-byte:
 *
 *   byte 0        bits to skip in the first data byte (low 3 bits)
 *   bytes 1..k    the target's parameters (num_dist, channel geometry, ...)
 *   rest          the bitstream
 *
 * test/fuzz_seeds.c writes seeds in exactly this layout, cut from the corpus
 * at the bit each component started reading.
 */
#ifndef JXL_FUZZ_COMMON_H
#define JXL_FUZZ_COMMON_H

#include "jxl.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* A crafted size header can declare an enormous canvas and the decoder has
 * no built-in pixel limit, so bound live bytes instead: such inputs then
 * fail through the library's own allocation-failure paths -- which is more
 * of the error handling under test -- rather than tripping libFuzzer's RSS
 * limit and stopping the run. The size lives in a 16-byte header so the
 * returned pointer keeps malloc's alignment. */
#define FUZZ_MEM_BUDGET ((size_t)512 << 20)   /* live bytes per input */

static size_t fuzz_live;

static inline void *fuzz_alloc(void *user, void *ctx, size_t size) {
    uint8_t *p;
    (void)user;
    (void)ctx;
    if (size > FUZZ_MEM_BUDGET - fuzz_live) return NULL;
    p = (uint8_t *)malloc(size + 16);
    if (!p) return NULL;
    memcpy(p, &size, sizeof(size));
    fuzz_live += size;
    return p + 16;
}

static inline void fuzz_free(void *user, void *ctx, void *ptr) {
    uint8_t *p;
    size_t size;
    (void)user;
    (void)ctx;
    if (!ptr) return;
    p = (uint8_t *)ptr - 16;
    memcpy(&size, p, sizeof(size));
    fuzz_live -= size;
    free(p);
}

static inline jxl_ctx *fuzz_ctx_new(void) {
    fuzz_live = 0;
    return jxl_ctx_new(fuzz_alloc, fuzz_free, NULL, NULL);
}

/* Splits a component target's input: on success `params` points at
 * nparams parameter bytes, `bits` (of length `nbits` bytes) at the
 * bitstream, and `skip` is the number of leading bits to drop. */
static inline int fuzz_split(const uint8_t *data, size_t size, size_t nparams,
                             const uint8_t **params, const uint8_t **bits,
                             size_t *nbits, uint32_t *skip) {
    if (size < 1 + nparams + 1) return -1;
    *skip = data[0] & 7;
    *params = data + 1;
    *bits = data + 1 + nparams;
    *nbits = size - 1 - nparams;
    return 0;
}

static inline uint32_t fuzz_u16(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}

/* v mapped into 1..n, leaving 1..n itself alone -- so a seed's real
   parameters reach the decoder unchanged and any mutation still lands in
   range. */
static inline uint32_t fuzz_range(uint32_t v, uint32_t n) {
    return v ? (v - 1) % n + 1 : 1;
}

#endif /* JXL_FUZZ_COMMON_H */
