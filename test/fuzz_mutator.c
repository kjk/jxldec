/* fuzz_mutator.c -- a TOC-aware LLVMFuzzerCustomMutator for fuzz_target.c.
 *
 * A frame's TOC records the byte size of every section that follows it
 * (LfGlobal, the LF groups, HfGlobal, each group in each pass). libFuzzer's
 * byte mutations that insert or erase bytes shift every section after the
 * edit, the TOC no longer matches, and the decoder rejects the frame at the
 * first section boundary -- so most mutations of a valid file only ever
 * test the truncation check. This one parses the input with the decoder's
 * own header and TOC readers, lets libFuzzer mutate the bytes of a single
 * section, then writes the frame back with a TOC re-encoded for the new
 * section sizes (and the jxlc box size fixed up), so the mutated section
 * reaches the code that decodes it.
 *
 * Handles a bare codestream or one inside a single jxlc box; a frame with a
 * permuted TOC is written back unpermuted, sections in their semantic order.
 * Anything else, and a third of all calls, gets libFuzzer's ordinary
 * mutation. Linked into the whole-file target only (cmd/build.ts); AFL++ has
 * its own mutators and ignores this hook.
 */
#include "jxl_internal.h"

#include <stdlib.h>
#include <string.h>

size_t LLVMFuzzerMutate(uint8_t *data, size_t size, size_t max_size);

#define MUT_MAX_FRAMES 8

/* A frame's sections in original (semantic) order. A permuted TOC is
   written back unpermuted, with the sections in that order, which is just
   as valid a frame; an unpermuted one comes back byte for byte. */
typedef struct {
    size_t toc_bit;           /* bit (in the codestream) where the TOC starts */
    size_t end;               /* byte offset just past the last section */
    uint32_t count;
    size_t *offs;             /* each section's offset in the input */
    size_t *sizes;
} mut_frame;

typedef struct {
    size_t off;
    uint32_t idx;
} mut_sec;

static int mut_sec_cmp(const void *a, const void *b) {
    const mut_sec *x = (const mut_sec *)a, *y = (const mut_sec *)b;
    if (x->off != y->off) return x->off < y->off ? -1 : 1;
    return x->idx < y->idx ? -1 : x->idx > y->idx;
}

static uint32_t mut_rand(uint32_t *s) {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

typedef struct {
    uint8_t *p;
    size_t cap;   /* bytes */
    size_t pos;   /* bits */
} mut_bw;

static int mut_put(mut_bw *w, uint32_t v, int n) {
    int i;
    for (i = 0; i < n; i++, w->pos++) {
        size_t byte = w->pos >> 3;
        uint8_t bit = (uint8_t)(1u << (w->pos & 7));
        if (byte >= w->cap) return -1;
        if ((v >> i) & 1) w->p[byte] |= bit;
        else w->p[byte] &= (uint8_t)~bit;
    }
    return 0;
}

static int mut_pad(mut_bw *w) {
    while (w->pos & 7) {
        if (mut_put(w, 0, 1) != 0) return -1;
    }
    return 0;
}

/* TOC entry: U32(Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408),
   BitsOffset(30, 4211712)), as jxl_read_toc reads it. */
static int mut_put_size(mut_bw *w, size_t s) {
    if (s < 1024) return mut_put(w, 0, 2) || mut_put(w, (uint32_t)s, 10);
    if (s < 17408) return mut_put(w, 1, 2) || mut_put(w, (uint32_t)(s - 1024), 14);
    if (s < 4211712) return mut_put(w, 2, 2) || mut_put(w, (uint32_t)(s - 17408), 22);
    if (s - 4211712 < ((size_t)1 << 30))
        return mut_put(w, 3, 2) || mut_put(w, (uint32_t)(s - 4211712), 30);
    return -1;
}

static void mut_free_frames(mut_frame *f, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(f[i].offs);
        free(f[i].sizes);
    }
}

/* The frames of the codestream whose TOC can be rewritten. */
static int mut_parse(jxl_ctx *ctx, jxl_doc *doc, mut_frame *out) {
    const uint8_t *cs = doc->container.cs;
    size_t cs_len = doc->container.cs_len;
    jxl_br br;
    int n = 0, i;

    jxl_br_init(&br, cs, cs_len);
    jxl_br_seek_byte(&br, doc->first_frame_off);
    for (i = 0; i < MUT_MAX_FRAMES; i++) {
        jxl_frame_header fh;
        jxl_toc toc;
        size_t toc_bit, end;
        int last;
        uint32_t k;

        if (jxl_read_frame_header(ctx, &br, &doc->size, &doc->meta, &fh) != 0) {
            jxl_frame_header_free(ctx, &fh);
            break;
        }
        toc_bit = jxl_br_bits_read(&br);
        last = fh.is_last;
        if (jxl_read_toc(ctx, &br, &fh, &toc) != 0) {
            jxl_frame_header_free(ctx, &fh);
            break;
        }
        jxl_frame_header_free(ctx, &fh);
        end = toc.end_off + (size_t)toc.total_size;
        if (toc.count > 0 && end <= cs_len) {
            mut_frame *f = &out[n];
            mut_sec *order = (mut_sec *)malloc(toc.count * sizeof(mut_sec));
            f->toc_bit = toc_bit;
            f->end = end;
            f->count = toc.count;
            f->offs = (size_t *)malloc(toc.count * sizeof(size_t));
            f->sizes = (size_t *)malloc(toc.count * sizeof(size_t));
            if (order && f->offs && f->sizes) {
                /* A section ends where the next one in the bitstream
                   starts, whatever the permutation. */
                for (k = 0; k < toc.count; k++) {
                    order[k].off = toc.entries[k].offset;
                    order[k].idx = k;
                    f->offs[k] = toc.entries[k].offset;
                }
                qsort(order, toc.count, sizeof(mut_sec), mut_sec_cmp);
                for (k = 0; k < toc.count; k++) {
                    size_t next = k + 1 < toc.count ? order[k + 1].off : end;
                    f->sizes[order[k].idx] = next - order[k].off;
                }
                n++;
            } else {
                free(f->offs);
                free(f->sizes);
            }
            free(order);
        }
        jxl_toc_free(ctx, &toc);
        if (last || end >= cs_len) break;
        jxl_br_seek_byte(&br, end);
    }
    return n;
}

size_t LLVMFuzzerCustomMutator(uint8_t *data, size_t size, size_t max_size,
                               unsigned int seed) {
    /* Mixed first: xorshift's first output for a small seed is
       seed * 8193 * 33, always a multiple of 3. */
    uint32_t rng = (seed * 2654435761u) ^ 0x9e3779b9u;
    jxl_ctx *ctx = NULL;
    jxl_doc *doc = NULL;
    mut_frame frames[MUT_MAX_FRAMES];
    int nframes = 0;
    uint8_t *sec = NULL, *out = NULL;
    size_t result = 0;

    if (mut_rand(&rng) % 3 == 0 || size < 16) goto plain;
    ctx = jxl_ctx_new(NULL, NULL, NULL, NULL);
    if (!ctx) goto plain;
    doc = jxl_doc_open(ctx, data, size);
    if (!doc || doc->container.cs_owned) goto plain;
    nframes = mut_parse(ctx, doc, frames);
    if (nframes == 0) goto plain;

    {
        const uint8_t *cs = doc->container.cs;
        size_t cs_len = doc->container.cs_len;
        size_t cs_off = (size_t)(cs - data);
        mut_frame *f = &frames[mut_rand(&rng) % (uint32_t)nframes];
        uint32_t s = mut_rand(&rng) % f->count, k;
        size_t sec_len, new_len, cap, other, total;
        size_t box_size_at = 0, box_hdr = 0;
        int box_large = 0;
        mut_bw w;

        if (cs_off > size) goto plain;
        /* A codestream inside a container must be a jxlc box's payload. */
        if (cs_off > 0) {
            if (cs_off >= 8 && memcmp(data + cs_off - 4, "jxlc", 4) == 0) {
                box_size_at = cs_off - 8;
                box_hdr = 8;
            } else if (cs_off >= 16 && memcmp(data + cs_off - 12, "jxlc", 4) == 0) {
                box_size_at = cs_off - 8;   /* the 64-bit largesize */
                box_hdr = 16;
                box_large = 1;
            } else {
                goto plain;
            }
        }
        sec_len = f->sizes[s];

        /* Room for the mutated section: everything else, plus a TOC that
           may grow by up to 4 bytes per entry. */
        other = size - sec_len;
        if (other + (size_t)f->count * 4 + 16 >= max_size) goto plain;
        cap = max_size - other - (size_t)f->count * 4 - 16;
        if (cap < sec_len || cap < 1) goto plain;
        sec = (uint8_t *)malloc(cap);
        out = (uint8_t *)calloc(1, max_size);
        if (!sec || !out) goto plain;
        if (sec_len) memcpy(sec, cs + f->offs[s], sec_len);
        new_len = LLVMFuzzerMutate(sec, sec_len, cap);
        f->sizes[s] = new_len;

        /* Container prefix, then the codestream up to the TOC's byte. */
        memcpy(out, data, cs_off);
        memcpy(out + cs_off, cs, (f->toc_bit >> 3) + 1);
        w.p = out + cs_off;
        w.cap = max_size - cs_off;
        w.pos = f->toc_bit;
        if (mut_put(&w, 0, 1) != 0 || mut_pad(&w) != 0) goto plain;
        for (k = 0; k < f->count; k++) {
            if (mut_put_size(&w, f->sizes[k]) != 0) goto plain;
        }
        if (mut_pad(&w) != 0) goto plain;
        total = cs_off + (w.pos >> 3);
        for (k = 0; k < f->count; k++) {
            const uint8_t *src = k == s ? sec : cs + f->offs[k];
            size_t src_len = f->sizes[k];
            if (total + src_len > max_size) goto plain;
            if (src_len) memcpy(out + total, src, src_len);
            total += src_len;
        }
        /* The rest of the codestream, then anything after it. */
        if (total + (cs_len - f->end) + (size - cs_off - cs_len) > max_size)
            goto plain;
        memcpy(out + total, cs + f->end, cs_len - f->end);
        total += cs_len - f->end;
        if (box_hdr) {
            uint64_t box_len = (uint64_t)(total - cs_off) + box_hdr;
            uint32_t size32 = ((uint32_t)data[cs_off - box_hdr] << 24) |
                     ((uint32_t)data[cs_off - box_hdr + 1] << 16) |
                     ((uint32_t)data[cs_off - box_hdr + 2] << 8) |
                     (uint32_t)data[cs_off - box_hdr + 3];
            if (box_large) {
                int b;
                for (b = 0; b < 8; b++)
                    out[box_size_at + b] = (uint8_t)(box_len >> (56 - 8 * b));
            } else if (size32 != 0) {
                /* 0 means "to the end of the file" and needs no update. */
                if (box_len > 0xffffffffu) goto plain;
                out[box_size_at] = (uint8_t)(box_len >> 24);
                out[box_size_at + 1] = (uint8_t)(box_len >> 16);
                out[box_size_at + 2] = (uint8_t)(box_len >> 8);
                out[box_size_at + 3] = (uint8_t)box_len;
            }
        }
        memcpy(out + total, data + cs_off + cs_len, size - cs_off - cs_len);
        total += size - cs_off - cs_len;
        memcpy(data, out, total);
        result = total;
    }
    goto done;

plain:
    result = LLVMFuzzerMutate(data, size, max_size);
done:
    free(sec);
    free(out);
    mut_free_frames(frames, nframes);
    if (doc) jxl_doc_close(doc);
    if (ctx) jxl_ctx_free(ctx);
    return result;
}
