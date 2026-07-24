/* coding.c -- entropy decoding: prefix (Brotli-style Huffman) codes, ANS with
 * alias tables, hybrid-uint token expansion, LZ77 and the cluster/context map.
 *
 * A jxl_dec bundles everything a "distribution set" needs: the context ->
 * cluster map, one hybrid-uint config per cluster, one histogram per cluster
 * (prefix or ANS, never mixed), and optional LZ77 state. Callers read integers
 * with jxl_dec_read(dec, br, context).
 *
 * ANS alias-table construction must match libjxl's InitAliasTable exactly
 * (LIFO underfull/overfull stacks) or the decoded stream diverges.
 */
#include "jxl_internal.h"

/* ----- bit-length helper: number of bits needed to represent x ----- */

uint32_t jxl_bitlen(uint32_t x) {
    uint32_t n = 0;
    while (x) {
        n++;
        x >>= 1;
    }
    return n;
}

/* ===================================================================== */
/* prefix codes                                                           */
/* ===================================================================== */

#define PFX_MAX_BITS 15
#define PFX_ROOT_BITS 8

static uint32_t reverse_bits(uint32_t v, int n) {
    uint32_t r = 0;
    int i;
    for (i = 0; i < n; i++) {
        r = (r << 1) | ((v >> i) & 1);
    }
    return r;
}

static void pfx_free(jxl_ctx *ctx, jxl_pfx_hist *h) {
    jxl_free(ctx, h->root);
    jxl_free(ctx, h->sub);
    h->root = NULL;
    h->sub = NULL;
}

/* A histogram that always yields the same symbol and consumes no bits. */
static void pfx_single_into(jxl_pfx_hist *h, uint16_t sym) {
    h->root_bits = 0;
    h->root_mask = 0;
    h->single_symbol = (int)sym;
    h->root[0].sym = sym;
    h->root[0].len = 0;
    h->root[0].nested = 0;
    h->nsub = 0;
    h->sub = NULL;
}

static int pfx_single(jxl_ctx *ctx, jxl_pfx_hist *h, uint16_t sym) {
    h->root = (jxl_pfx_entry *)jxl_calloc(ctx, 1, sizeof(jxl_pfx_entry));
    if (!h->root) return -1;
    pfx_single_into(h, sym);
    return 0;
}

/* Build a two-level decode table from canonical code lengths. Codes are
   assigned in (length, symbol) order and read LSB-first, so a code of length
   L occupies every table slot whose low L bits equal the reversed code. */
static int pfx_build(jxl_ctx *ctx, jxl_pfx_hist *h, const uint8_t *lens,
                     uint32_t n) {
    uint32_t count[PFX_MAX_BITS + 1];
    uint32_t next_code[PFX_MAX_BITS + 2];
    uint32_t sub_off[1 << PFX_ROOT_BITS];
    uint8_t sub_maxlen[1 << PFX_ROOT_BITS];
    uint32_t total = 0, sub_total = 0;
    uint32_t code, i, sym;
    int len, max_len = 0, root_bits;
    uint32_t root_size, root_mask;

    memset(count, 0, sizeof(count));
    for (sym = 0; sym < n; sym++) {
        if (lens[sym] > PFX_MAX_BITS) return -1;
        if (lens[sym]) {
            count[lens[sym]]++;
            if (lens[sym] > max_len) max_len = lens[sym];
        }
    }
    if (max_len == 0) return -1;

    /* Kraft equality: the code must be complete. */
    for (len = 1; len <= max_len; len++) total += count[len] << (PFX_MAX_BITS - len);
    if (total != (1u << PFX_MAX_BITS)) return -1;

    code = 0;
    for (len = 1; len <= max_len; len++) {
        next_code[len] = code;
        code = (code + count[len]) << 1;
    }

    root_bits = max_len < PFX_ROOT_BITS ? max_len : PFX_ROOT_BITS;
    root_size = 1u << root_bits;
    root_mask = root_size - 1;

    /* Pass 1: longest code under each root prefix that needs a sub-table. */
    memset(sub_maxlen, 0, sizeof(sub_maxlen));
    {
        uint32_t nc[PFX_MAX_BITS + 2];
        memcpy(nc, next_code, sizeof(nc));
        for (len = 1; len <= max_len; len++) {
            for (sym = 0; sym < n; sym++) {
                uint32_t rev, prefix;
                if (lens[sym] != len) continue;
                rev = reverse_bits(nc[len]++, len);
                if (len <= root_bits) continue;
                prefix = rev & root_mask;
                if ((uint32_t)len > sub_maxlen[prefix]) sub_maxlen[prefix] = (uint8_t)len;
            }
        }
    }
    for (i = 0; i < root_size; i++) {
        sub_off[i] = sub_total;
        if (sub_maxlen[i]) sub_total += 1u << (sub_maxlen[i] - root_bits);
    }

    h->root = (jxl_pfx_entry *)jxl_calloc(ctx, root_size, sizeof(jxl_pfx_entry));
    if (!h->root) return -1;
    if (sub_total) {
        h->sub = (jxl_pfx_entry *)jxl_calloc(ctx, sub_total, sizeof(jxl_pfx_entry));
        if (!h->sub) return -1;
    }
    h->nsub = sub_total;
    h->root_bits = root_bits;
    h->root_mask = root_mask;
    h->single_symbol = -1;

    /* Nested root entries first, so leaf fills can't overwrite them. */
    for (i = 0; i < root_size; i++) {
        if (!sub_maxlen[i]) continue;
        h->root[i].nested = 1;
        h->root[i].sym = (uint16_t)sub_off[i];
        h->root[i].len = (uint8_t)((1u << (sub_maxlen[i] - root_bits)) - 1);
    }

    /* Pass 2: fill leaves. */
    for (len = 1; len <= max_len; len++) {
        for (sym = 0; sym < n; sym++) {
            uint32_t rev, k, step;
            if (lens[sym] != len) continue;
            rev = reverse_bits(next_code[len]++, len);
            if (len <= root_bits) {
                step = 1u << len;
                for (k = rev; k < root_size; k += step) {
                    h->root[k].sym = (uint16_t)sym;
                    h->root[k].len = (uint8_t)len;
                    h->root[k].nested = 0;
                }
            } else {
                uint32_t prefix = rev & root_mask;
                uint32_t hi = rev >> root_bits;
                uint32_t sub_size = 1u << (sub_maxlen[prefix] - root_bits);
                step = 1u << (len - root_bits);
                for (k = hi; k < sub_size; k += step) {
                    jxl_pfx_entry *e = &h->sub[sub_off[prefix] + k];
                    e->sym = (uint16_t)sym;
                    e->len = (uint8_t)len;
                    e->nested = 0;
                }
            }
        }
    }
    return 0;
}

static uint32_t pfx_read(const jxl_pfx_hist *h, jxl_br *br) {
    uint32_t peeked = jxl_br_peek(br, PFX_MAX_BITS);
    const jxl_pfx_entry *e = &h->root[peeked & h->root_mask];
    if (e->nested) {
        const jxl_pfx_entry *e2 =
            &h->sub[e->sym + ((peeked >> h->root_bits) & e->len)];
        jxl_br_consume(br, e2->len);
        return e2->sym;
    }
    jxl_br_consume(br, e->len);
    return e->sym;
}

static int pfx_parse_simple(jxl_ctx *ctx, jxl_br *br, jxl_pfx_hist *h,
                            uint32_t alphabet_size) {
    int alphabet_bits = (int)jxl_bitlen(alphabet_size - 1);
    uint32_t nsym = jxl_br_read(br, 2) + 1;
    uint32_t syms[4];
    uint8_t code_len[4];
    uint8_t *lens;
    uint32_t i;
    int rc;

    if (nsym == 1) {
        uint32_t s = jxl_br_read(br, alphabet_bits);
        if (s >= alphabet_size) return -1;
        return pfx_single(ctx, h, (uint16_t)s);
    }
    for (i = 0; i < nsym; i++) syms[i] = jxl_br_read(br, alphabet_bits);
    if (nsym == 2) {
        code_len[0] = 1; code_len[1] = 1;
    } else if (nsym == 3) {
        code_len[0] = 1; code_len[1] = 2; code_len[2] = 2;
    } else {
        int tree_selector = jxl_br_bool(br);
        if (tree_selector) {
            code_len[0] = 1; code_len[1] = 2; code_len[2] = 3; code_len[3] = 3;
        } else {
            code_len[0] = 2; code_len[1] = 2; code_len[2] = 2; code_len[3] = 2;
        }
    }
    lens = (uint8_t *)jxl_calloc(ctx, alphabet_size, 1);
    if (!lens) return -1;
    for (i = 0; i < nsym; i++) {
        if (syms[i] >= alphabet_size) {
            jxl_free(ctx, lens);
            return -1;
        }
        lens[syms[i]] = code_len[i];
    }
    rc = pfx_build(ctx, h, lens, alphabet_size);
    jxl_free(ctx, lens);
    return rc;
}

static const uint8_t pfx_code_length_order[18] = {
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

static int pfx_parse_complex(jxl_ctx *ctx, jxl_br *br, jxl_pfx_hist *h,
                             uint32_t alphabet_size, uint32_t hskip) {
    uint8_t clcl[18];
    jxl_pfx_entry cl_root[1 << PFX_ROOT_BITS];
    jxl_pfx_hist clh;
    uint8_t *lens = NULL;
    uint32_t bitacc = 0;
    int nonzero_count = 0, nonzero_sym = 0;
    uint32_t i;
    uint32_t repeat_count = 0;
    uint8_t repeat_sym = 0, prev_sym = 8, last_nonzero_sym = 8;
    uint32_t last_repeat_count = 0;
    int rc = -1;

    memset(clcl, 0, sizeof(clcl));
    memset(&clh, 0, sizeof(clh));
    clh.root = cl_root;

    for (i = hskip; i < 18; i++) {
        uint32_t idx = pfx_code_length_order[i];
        uint32_t base = jxl_br_u32(br, 0, 0, 4, 0, 3, 0, 8, 0);
        uint32_t len;
        if (base == 8) {
            if (jxl_br_bool(br)) {
                len = jxl_br_bool(br) ? 5 : 1;
            } else {
                len = 2;
            }
        } else {
            len = base;
        }
        clcl[idx] = (uint8_t)len;
        if (len != 0) {
            nonzero_count++;
            nonzero_sym = (int)idx;
            bitacc += 32u >> len;
            if (bitacc == 32) break;
            if (bitacc > 32) return -1;
        }
    }

    if (nonzero_count == 1) {
        pfx_single_into(&clh, (uint16_t)nonzero_sym);
    } else if (bitacc != 32) {
        return -1;
    } else {
        /* Build into a stack-allocated root table (18 symbols, len <= 5, so
           no sub-table is possible). */
        uint32_t count[PFX_MAX_BITS + 1];
        int len, max_len = 0;
        uint32_t next_code[PFX_MAX_BITS + 2], code = 0, sym, k, step;
        memset(count, 0, sizeof(count));
        for (sym = 0; sym < 18; sym++) {
            if (clcl[sym]) {
                count[clcl[sym]]++;
                if (clcl[sym] > max_len) max_len = clcl[sym];
            }
        }
        for (len = 1; len <= max_len; len++) {
            next_code[len] = code;
            code = (code + count[len]) << 1;
        }
        clh.root_bits = max_len;
        clh.root_mask = (1u << max_len) - 1;
        clh.single_symbol = -1;
        clh.nsub = 0;
        clh.sub = NULL;
        memset(cl_root, 0, sizeof(jxl_pfx_entry) * ((size_t)1 << max_len));
        for (len = 1; len <= max_len; len++) {
            for (sym = 0; sym < 18; sym++) {
                uint32_t rev;
                if (clcl[sym] != len) continue;
                rev = reverse_bits(next_code[len]++, len);
                step = 1u << len;
                for (k = rev; k < (1u << max_len); k += step) {
                    cl_root[k].sym = (uint16_t)sym;
                    cl_root[k].len = (uint8_t)len;
                    cl_root[k].nested = 0;
                }
            }
        }
    }

    lens = (uint8_t *)jxl_calloc(ctx, alphabet_size, 1);
    if (!lens) return -1;

    bitacc = 0;
    for (i = 0; i < alphabet_size; i++) {
        if (repeat_count > 0) {
            lens[i] = repeat_sym;
            repeat_count--;
        } else {
            uint32_t sym = pfx_read(&clh, br);
            if (br->err) goto done;
            if (sym == 0) {
                /* zero-length: symbol not present */
            } else if (sym <= 15) {
                lens[i] = (uint8_t)sym;
                last_nonzero_sym = (uint8_t)sym;
            } else if (sym == 16) {
                repeat_count = jxl_br_read(br, 2) + 3;
                if (prev_sym == 16) {
                    repeat_count += last_repeat_count * 3 - 8;
                    last_repeat_count += repeat_count;
                } else {
                    last_repeat_count = repeat_count;
                }
                repeat_sym = last_nonzero_sym;
                lens[i] = repeat_sym;
                repeat_count--;
            } else {
                repeat_count = jxl_br_read(br, 3) + 3;
                if (prev_sym == 17) {
                    repeat_count += last_repeat_count * 7 - 16;
                    last_repeat_count += repeat_count;
                } else {
                    last_repeat_count = repeat_count;
                }
                repeat_sym = 0;
                lens[i] = repeat_sym;
                repeat_count--;
            }
            prev_sym = (uint8_t)sym;
        }
        if (lens[i] != 0) {
            uint32_t shift = lens[i] >= PFX_MAX_BITS ? 0 : (PFX_MAX_BITS - lens[i]);
            bitacc += 1u << shift;
            if (bitacc > (1u << PFX_MAX_BITS)) goto done;
            if (bitacc == (1u << PFX_MAX_BITS) && repeat_count == 0) break;
        }
    }
    if (bitacc != (1u << PFX_MAX_BITS) || repeat_count > 0) goto done;
    rc = pfx_build(ctx, h, lens, alphabet_size);

done:
    jxl_free(ctx, lens);
    return rc;
}

static int pfx_parse(jxl_ctx *ctx, jxl_br *br, jxl_pfx_hist *h,
                     uint32_t alphabet_size) {
    uint32_t hskip;

    h->root = NULL;
    h->sub = NULL;
    h->nsub = 0;
    h->single_symbol = -1;

    if (alphabet_size == 1) return pfx_single(ctx, h, 0);
    if (alphabet_size > (1u << PFX_MAX_BITS)) return -1;

    hskip = jxl_br_read(br, 2);
    if (hskip == 1) return pfx_parse_simple(ctx, br, h, alphabet_size);
    return pfx_parse_complex(ctx, br, h, alphabet_size, hskip);
}

/* ===================================================================== */
/* ANS                                                                    */
/* ===================================================================== */

#define ANS_LOG_TAB_SIZE 12
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
/* A well-formed ANS stream ends with the state back at this value. */
#define ANS_SIGNATURE 0x130000

static uint32_t ans_read_u8(jxl_br *br) {
    if (jxl_br_bool(br)) {
        int n = (int)jxl_br_read(br, 3);
        return (1u << n) + jxl_br_read(br, n);
    }
    return 0;
}

static uint16_t ans_read_prefix(jxl_br *br) {
    switch (jxl_br_read(br, 3)) {
        case 0: return 10;
        case 1: {
            static const uint16_t vals[4] = {4, 0, 11, 13};
            int i;
            for (i = 0; i < 4; i++) {
                if (jxl_br_bool(br)) return vals[i];
            }
            return 12;
        }
        case 2: return 7;
        case 3: return jxl_br_bool(br) ? 1 : 3;
        case 4: return 6;
        case 5: return 8;
        case 6: return 9;
        default: return jxl_br_bool(br) ? 2 : 5;
    }
}

static void ans_free(jxl_ctx *ctx, jxl_ans_hist *h) {
    jxl_free(ctx, h->buckets);
    h->buckets = NULL;
}

static int ans_parse(jxl_ctx *ctx, jxl_br *br, jxl_ans_hist *h,
                     uint32_t log_alphabet_size) {
    uint32_t table_size = 1u << log_alphabet_size;
    uint32_t log_bucket_size = 12 - log_alphabet_size;
    uint32_t bucket_size = 1u << log_bucket_size;
    uint16_t *dist = NULL;
    uint32_t alphabet_size = 0;
    uint32_t i;
    int rc = -1;
    int32_t single_sym = -1;

    /* Working arrays for alias-table construction. */
    uint16_t *cutoff = NULL, *alias_sym = NULL, *alias_off = NULL;
    uint32_t *underfull = NULL, *overfull = NULL;
    uint32_t nunder = 0, nover = 0;

    h->buckets = NULL;
    dist = (uint16_t *)jxl_calloc(ctx, table_size, sizeof(uint16_t));
    if (!dist) return -1;

    if (jxl_br_bool(br)) {
        if (jxl_br_bool(br)) {
            /* binary: two symbols */
            uint32_t v0 = ans_read_u8(br);
            uint32_t v1 = ans_read_u8(br);
            uint32_t prob;
            if (v0 == v1) goto done;
            alphabet_size = (v0 > v1 ? v0 : v1) + 1;
            if (alphabet_size > table_size) goto done;
            prob = jxl_br_read(br, 12);
            dist[v0] = (uint16_t)prob;
            dist[v1] = (uint16_t)(ANS_TAB_SIZE - prob);
        } else {
            /* unary: one symbol takes the whole range */
            uint32_t val = ans_read_u8(br);
            alphabet_size = val + 1;
            if (alphabet_size > table_size) goto done;
            dist[val] = ANS_TAB_SIZE;
        }
    } else if (jxl_br_bool(br)) {
        /* evenly distributed */
        uint32_t base, leftover;
        alphabet_size = ans_read_u8(br) + 1;
        if (alphabet_size > table_size) goto done;
        base = ANS_TAB_SIZE / alphabet_size;
        leftover = ANS_TAB_SIZE % alphabet_size;
        for (i = 0; i < leftover; i++) dist[i] = (uint16_t)(base + 1);
        for (; i < alphabet_size; i++) dist[i] = (uint16_t)base;
    } else {
        /* compressed distribution */
        int len = 0;
        int16_t shift;
        uint32_t idx = 0, acc = 0;
        uint16_t prev_dist = 0;
        int32_t omit_log = -1, omit_pos = -1;
        uint32_t *rep_start = NULL, *rep_end = NULL;
        uint32_t nrep = 0, rep_cap = 0, rep_idx = 0;

        while (len < 3 && jxl_br_bool(br)) len++;
        shift = (int16_t)(jxl_br_read(br, len) + (1u << len) - 1);
        if (shift > 13) goto done;
        alphabet_size = ans_read_u8(br) + 3;
        if (alphabet_size > table_size) goto done;

        while (idx < alphabet_size) {
            dist[idx] = ans_read_prefix(br);
            if (br->err) { jxl_free(ctx, rep_start); jxl_free(ctx, rep_end); goto done; }
            if (dist[idx] == 13) {
                uint32_t repeat_count = ans_read_u8(br) + 4;
                if (idx + repeat_count > alphabet_size) {
                    jxl_free(ctx, rep_start);
                    jxl_free(ctx, rep_end);
                    goto done;
                }
                if (nrep == rep_cap) {
                    uint32_t ncap = rep_cap ? rep_cap * 2 : 8;
                    uint32_t *a = (uint32_t *)jxl_realloc_array(ctx, rep_start, rep_cap, ncap, sizeof(uint32_t));
                    uint32_t *b = (uint32_t *)jxl_realloc_array(ctx, rep_end, rep_cap, ncap, sizeof(uint32_t));
                    if (!a || !b) { jxl_free(ctx, a); jxl_free(ctx, b); goto done; }
                    rep_start = a;
                    rep_end = b;
                    rep_cap = ncap;
                }
                rep_start[nrep] = idx;
                rep_end[nrep] = idx + repeat_count;
                nrep++;
                idx += repeat_count;
                continue;
            }
            if (omit_pos < 0 || (int32_t)dist[idx] > omit_log) {
                omit_log = dist[idx];
                omit_pos = (int32_t)idx;
            }
            idx++;
        }
        if (omit_pos < 0 ||
            ((uint32_t)omit_pos + 1 < table_size && dist[omit_pos + 1] == 13)) {
            jxl_free(ctx, rep_start);
            jxl_free(ctx, rep_end);
            goto done;
        }

        for (i = 0; i < table_size; i++) {
            uint16_t code = dist[i];
            if (rep_idx < nrep && rep_start[rep_idx] <= i) {
                if (rep_end[rep_idx] == i) {
                    rep_idx++;
                } else {
                    dist[i] = prev_dist;
                    acc += dist[i];
                    if (acc > ANS_TAB_SIZE) break;
                    continue;
                }
                code = dist[i];
            }
            if (code == 0) { prev_dist = 0; continue; }
            if ((int32_t)i == omit_pos) { prev_dist = 0; continue; }
            if (code > 1) {
                int16_t zeros = (int16_t)(code - 1);
                int16_t bitcount = (int16_t)(shift - ((12 - zeros) >> 1));
                if (bitcount < 0) bitcount = 0;
                if (bitcount > zeros) bitcount = zeros;
                code = (uint16_t)((1u << zeros) +
                                  (jxl_br_read(br, bitcount) << (zeros - bitcount)));
                dist[i] = code;
            }
            prev_dist = code;
            acc += code;
            if (acc > ANS_TAB_SIZE) break;
        }
        jxl_free(ctx, rep_start);
        jxl_free(ctx, rep_end);
        if (acc > ANS_TAB_SIZE) goto done;
        dist[omit_pos] = (uint16_t)(ANS_TAB_SIZE - acc);
    }
    if (br->err) goto done;

    h->log_bucket_size = log_bucket_size;
    h->bucket_mask = bucket_size - 1;
    h->buckets = (jxl_ans_bucket *)jxl_calloc(ctx, table_size,
                                              sizeof(jxl_ans_bucket));
    if (!h->buckets) goto done;

    for (i = 0; i < table_size; i++) {
        if (dist[i] == ANS_TAB_SIZE) { single_sym = (int32_t)i; break; }
    }
    if (single_sym >= 0) {
        /* Decoding from this histogram must leave the state unchanged. */
        for (i = 0; i < table_size; i++) {
            h->buckets[i].dist = dist[i];
            h->buckets[i].alias_symbol = (uint8_t)single_sym;
            h->buckets[i].alias_offset = (uint16_t)(bucket_size * i);
            h->buckets[i].alias_cutoff = 0;
            h->buckets[i].alias_dist_xor = (uint16_t)(dist[i] ^ ANS_TAB_SIZE);
        }
        h->single_symbol = single_sym;
        rc = 0;
        goto done;
    }
    h->single_symbol = -1;

    cutoff = (uint16_t *)jxl_calloc(ctx, table_size, sizeof(uint16_t));
    alias_sym = (uint16_t *)jxl_calloc(ctx, table_size, sizeof(uint16_t));
    alias_off = (uint16_t *)jxl_calloc(ctx, table_size, sizeof(uint16_t));
    underfull = (uint32_t *)jxl_calloc(ctx, table_size, sizeof(uint32_t));
    overfull = (uint32_t *)jxl_calloc(ctx, table_size, sizeof(uint32_t));
    if (!cutoff || !alias_sym || !alias_off || !underfull || !overfull) goto done;

    for (i = 0; i < table_size; i++) {
        cutoff[i] = dist[i];
        alias_sym[i] = (uint16_t)(i < alphabet_size ? i : 0);
        alias_off[i] = 0;
        if (dist[i] < bucket_size) underfull[nunder++] = i;
        else if (dist[i] > bucket_size) overfull[nover++] = i;
    }
    while (nover > 0 && nunder > 0) {
        uint32_t o = overfull[--nover];
        uint32_t u = underfull[--nunder];
        uint16_t by = (uint16_t)(bucket_size - cutoff[u]);
        cutoff[o] = (uint16_t)(cutoff[o] - by);
        alias_sym[u] = (uint16_t)o;
        alias_off[u] = cutoff[o];
        if (cutoff[o] < bucket_size) underfull[nunder++] = o;
        else if (cutoff[o] > bucket_size) overfull[nover++] = o;
    }

    for (i = 0; i < table_size; i++) {
        h->buckets[i].dist = dist[i];
        if (cutoff[i] == bucket_size) {
            h->buckets[i].alias_symbol = (uint8_t)i;
            h->buckets[i].alias_offset = 0;
            h->buckets[i].alias_cutoff = 0;
            h->buckets[i].alias_dist_xor = 0;
        } else {
            h->buckets[i].alias_symbol = (uint8_t)alias_sym[i];
            h->buckets[i].alias_offset = (uint16_t)(alias_off[i] - cutoff[i]);
            h->buckets[i].alias_cutoff = (uint8_t)cutoff[i];
            h->buckets[i].alias_dist_xor =
                (uint16_t)(dist[i] ^ dist[alias_sym[i]]);
        }
    }
    rc = 0;

done:
    jxl_free(ctx, dist);
    jxl_free(ctx, cutoff);
    jxl_free(ctx, alias_sym);
    jxl_free(ctx, alias_off);
    jxl_free(ctx, underfull);
    jxl_free(ctx, overfull);
    return rc;
}

static uint32_t ans_read_symbol(const jxl_ans_hist *h, jxl_br *br,
                                uint32_t *state) {
    uint32_t idx = *state & 0xfff;
    uint32_t i = idx >> h->log_bucket_size;
    uint32_t pos = idx & h->bucket_mask;
    const jxl_ans_bucket *b = &h->buckets[i];
    int map_to_alias = pos >= b->alias_cutoff;
    uint32_t symbol = map_to_alias ? b->alias_symbol : i;
    uint32_t offset = (map_to_alias ? b->alias_offset : 0) + pos;
    uint32_t dist = b->dist ^ (map_to_alias ? b->alias_dist_xor : 0);
    uint32_t next_state = (*state >> 12) * dist + offset;

    if (next_state < (1u << 16)) {
        next_state = (next_state << 16) | jxl_br_peek(br, 16);
        jxl_br_consume(br, 16);
    }
    *state = next_state;
    return symbol;
}

/* ===================================================================== */
/* hybrid uint config                                                     */
/* ===================================================================== */

static int int_config_parse(jxl_br *br, jxl_int_config *cfg,
                            uint32_t log_alphabet_size) {
    uint32_t split_exponent_bits = jxl_bitlen(log_alphabet_size);
    uint32_t split_exponent = jxl_br_read(br, (int)split_exponent_bits);
    uint32_t msb = 0, lsb = 0;

    if (split_exponent != log_alphabet_size) {
        uint32_t msb_bits = jxl_bitlen(split_exponent);
        msb = jxl_br_read(br, (int)msb_bits);
        if (msb > split_exponent) return -1;
        lsb = jxl_br_read(br, (int)jxl_bitlen(split_exponent - msb));
    }
    if (lsb + msb > split_exponent) return -1;
    cfg->split_exponent = split_exponent;
    cfg->split = 1u << split_exponent;
    cfg->msb_in_token = msb;
    cfg->lsb_in_token = lsb;
    return 0;
}

static uint32_t read_uint(jxl_br *br, const jxl_int_config *cfg,
                          uint32_t token) {
    uint32_t n, low_bits, msb, lsb;
    uint64_t rest, result;

    if (token < cfg->split) return token;
    msb = cfg->msb_in_token;
    lsb = cfg->lsb_in_token;
    n = (cfg->split_exponent - (msb + lsb) + ((token - cfg->split) >> (msb + lsb))) & 31;
    rest = jxl_br_read(br, (int)n);
    low_bits = token & ((1u << lsb) - 1);
    token >>= lsb;
    token &= (1u << msb) - 1;
    token |= 1u << msb;
    result = (((uint64_t)token << n) | rest) << lsb | low_bits;
    return (uint32_t)result;
}

/* ===================================================================== */
/* the entropy decoder                                                    */
/* ===================================================================== */

static const int8_t lz77_special_distances[120][2] = {
    {0,1},{1,0},{1,1},{-1,1},{0,2},{2,0},{1,2},{-1,2},{2,1},{-2,1},
    {2,2},{-2,2},{0,3},{3,0},{1,3},{-1,3},{3,1},{-3,1},{2,3},{-2,3},
    {3,2},{-3,2},{0,4},{4,0},{1,4},{-1,4},{4,1},{-4,1},{3,3},{-3,3},
    {2,4},{-2,4},{4,2},{-4,2},{0,5},{3,4},{-3,4},{4,3},{-4,3},{5,0},
    {1,5},{-1,5},{5,1},{-5,1},{2,5},{-2,5},{5,2},{-5,2},{4,4},{-4,4},
    {3,5},{-3,5},{5,3},{-5,3},{0,6},{6,0},{1,6},{-1,6},{6,1},{-6,1},
    {2,6},{-2,6},{6,2},{-6,2},{4,5},{-4,5},{5,4},{-5,4},{3,6},{-3,6},
    {6,3},{-6,3},{0,7},{7,0},{1,7},{-1,7},{5,5},{-5,5},{7,1},{-7,1},
    {4,6},{-4,6},{6,4},{-6,4},{2,7},{-2,7},{7,2},{-7,2},{3,7},{-3,7},
    {7,3},{-7,3},{5,6},{-5,6},{6,5},{-6,5},{8,0},{4,7},{-4,7},{7,4},
    {-7,4},{8,1},{8,2},{6,6},{-6,6},{8,3},{5,7},{-5,7},{7,5},{-7,5},
    {8,4},{6,7},{-6,7},{7,6},{-7,6},{8,5},{7,7},{-7,7},{8,6},{8,7}
};

#define LZ77_WINDOW_SIZE (1u << 20)
#define LZ77_WINDOW_MASK (LZ77_WINDOW_SIZE - 1)

void jxl_dec_free(jxl_dec *dec) {
    uint32_t i;
    if (!dec || !dec->ctx) return;
    if (dec->pfx) {
        for (i = 0; i < dec->num_clusters; i++) pfx_free(dec->ctx, &dec->pfx[i]);
        jxl_free(dec->ctx, dec->pfx);
    }
    if (dec->ans) {
        for (i = 0; i < dec->num_clusters; i++) ans_free(dec->ctx, &dec->ans[i]);
        jxl_free(dec->ctx, dec->ans);
    }
    jxl_free(dec->ctx, dec->configs);
    jxl_free(dec->ctx, dec->clusters);
    jxl_free(dec->ctx, dec->window);
    memset(dec, 0, sizeof(*dec));
}

static int dec_parse_inner(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br,
                           uint32_t num_dist);

/* Distribution clustering (the "context map"). */
int jxl_read_clusters(jxl_ctx *ctx, jxl_br *br, uint32_t num_dist,
                      uint8_t *clusters, uint32_t *num_clusters_out) {
    uint32_t i;
    uint32_t num_clusters = 0;
    uint8_t seen[256];

    if (num_dist == 1) {
        clusters[0] = 0;
        *num_clusters_out = 1;
        return 0;
    }
    if (jxl_br_bool(br)) {
        int nbits = (int)jxl_br_read(br, 2);
        for (i = 0; i < num_dist; i++) clusters[i] = (uint8_t)jxl_br_read(br, nbits);
    } else {
        int use_mtf = jxl_br_bool(br);
        jxl_dec sub;
        memset(&sub, 0, sizeof(sub));
        if (num_dist <= 2) {
            if (jxl_br_bool(br)) {   /* lz77_enabled must be false here */
                JXL_ERR(ctx, "context map: LZ77 not allowed");
                return -1;
            }
            if (dec_parse_inner(ctx, &sub, br, 1) != 0) return -1;
        } else {
            if (jxl_dec_init(ctx, &sub, br, 1) != 0) return -1;
        }
        jxl_dec_begin(&sub, br);
        for (i = 0; i < num_dist; i++) {
            uint32_t v = jxl_dec_read(&sub, br, 0);
            if (v > 255 || br->err) {
                JXL_ERR(ctx, "context map: invalid cluster %u", (unsigned)v);
                jxl_dec_free(&sub);
                return -1;
            }
            clusters[i] = (uint8_t)v;
        }
        if (jxl_dec_finalize(&sub) != 0) {
            JXL_ERR(ctx, "context map: bad ANS final state");
            jxl_dec_free(&sub);
            return -1;
        }
        jxl_dec_free(&sub);
        if (use_mtf) {
            uint8_t mtf[256];
            for (i = 0; i < 256; i++) mtf[i] = (uint8_t)i;
            for (i = 0; i < num_dist; i++) {
                uint32_t idx = clusters[i];
                uint8_t v = mtf[idx];
                clusters[i] = v;
                memmove(mtf + 1, mtf, idx);
                mtf[0] = v;
            }
        }
    }

    memset(seen, 0, sizeof(seen));
    for (i = 0; i < num_dist; i++) {
        if (clusters[i] + 1u > num_clusters) num_clusters = clusters[i] + 1u;
        seen[clusters[i]] = 1;
    }
    for (i = 0; i < num_clusters; i++) {
        if (!seen[i]) {
            JXL_ERR(ctx, "context map has a hole at cluster %u", (unsigned)i);
            return -1;
        }
    }
    *num_clusters_out = num_clusters;
    return 0;
}

static int dec_parse_inner(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br,
                           uint32_t num_dist) {
    uint32_t log_alphabet_size;
    uint32_t i;

    dec->ctx = ctx;
    dec->num_dist = num_dist;
    dec->clusters = (uint8_t *)jxl_calloc(ctx, num_dist, 1);
    if (!dec->clusters) return -1;
    if (jxl_read_clusters(ctx, br, num_dist, dec->clusters, &dec->num_clusters) != 0)
        return -1;

    dec->use_prefix = jxl_br_bool(br);
    log_alphabet_size = dec->use_prefix ? 15 : jxl_br_read(br, 2) + 5;

    dec->configs = (jxl_int_config *)jxl_calloc(ctx, dec->num_clusters,
                                                sizeof(jxl_int_config));
    if (!dec->configs) return -1;
    for (i = 0; i < dec->num_clusters; i++) {
        if (int_config_parse(br, &dec->configs[i], log_alphabet_size) != 0) {
            JXL_ERR(ctx, "invalid hybrid uint config");
            return -1;
        }
    }

    if (dec->use_prefix) {
        uint32_t *counts = (uint32_t *)jxl_calloc(ctx, dec->num_clusters,
                                                  sizeof(uint32_t));
        if (!counts) return -1;
        for (i = 0; i < dec->num_clusters; i++) {
            uint32_t count = 1;
            if (jxl_br_bool(br)) {
                int n = (int)jxl_br_read(br, 4);
                count = 1 + (1u << n) + jxl_br_read(br, n);
            }
            if (count > (1u << 15)) {
                jxl_free(ctx, counts);
                JXL_ERR(ctx, "prefix alphabet too large");
                return -1;
            }
            counts[i] = count;
        }
        dec->pfx = (jxl_pfx_hist *)jxl_calloc(ctx, dec->num_clusters,
                                              sizeof(jxl_pfx_hist));
        if (!dec->pfx) { jxl_free(ctx, counts); return -1; }
        for (i = 0; i < dec->num_clusters; i++) {
            if (pfx_parse(ctx, br, &dec->pfx[i], counts[i]) != 0) {
                jxl_free(ctx, counts);
                JXL_ERR(ctx, "invalid prefix histogram");
                return -1;
            }
        }
        jxl_free(ctx, counts);
    } else {
        dec->ans = (jxl_ans_hist *)jxl_calloc(ctx, dec->num_clusters,
                                              sizeof(jxl_ans_hist));
        if (!dec->ans) return -1;
        for (i = 0; i < dec->num_clusters; i++) {
            if (ans_parse(ctx, br, &dec->ans[i], log_alphabet_size) != 0) {
                JXL_ERR(ctx, "invalid ANS histogram");
                return -1;
            }
        }
    }
    return br->err ? -1 : 0;
}

/* Start reading an entropy-coded stream: ANS streams begin with a 32-bit
   initial state, and the LZ77 window starts empty. Prefix-code streams only
   need the state reset. Must be called once per stream, right where libjxl
   constructs its ANSSymbolReader. */
void jxl_dec_begin(jxl_dec *dec, jxl_br *br) {
    if (!dec->use_prefix) dec->state = jxl_br_read(br, 32);
    dec->num_to_copy = 0;
    dec->copy_pos = 0;
    dec->num_decoded = 0;
    dec->err = 0;
}

int jxl_dec_init(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br, uint32_t num_dist) {
    memset(dec, 0, sizeof(*dec));
    dec->ctx = ctx;
    dec->lz77_enabled = jxl_br_bool(br);
    if (dec->lz77_enabled) {
        dec->min_symbol = jxl_br_u32(br, 224, 0, 512, 0, 4096, 0, 8, 15);
        dec->min_length = jxl_br_u32(br, 3, 0, 4, 0, 5, 2, 9, 8);
        if (int_config_parse(br, &dec->lz_len_conf, 8) != 0) {
            JXL_ERR(ctx, "invalid LZ77 length config");
            return -1;
        }
        num_dist += 1;
    }
    return dec_parse_inner(ctx, dec, br, num_dist);
}

/* Lazily allocated: only LZ77 streams need the 4 MB window. */
static int lz77_ensure_window(jxl_dec *dec) {
    if (dec->window) return 0;
    dec->window = (uint32_t *)jxl_calloc(dec->ctx, LZ77_WINDOW_SIZE,
                                         sizeof(uint32_t));
    return dec->window ? 0 : -1;
}

static uint32_t dec_read_symbol(jxl_dec *dec, jxl_br *br, uint32_t cluster) {
    if (dec->use_prefix) return pfx_read(&dec->pfx[cluster], br);
    return ans_read_symbol(&dec->ans[cluster], br, &dec->state);
}

/* Reads one integer using an already-resolved cluster index. */
uint32_t jxl_dec_read_clustered(jxl_dec *dec, jxl_br *br, uint32_t cluster,
                                uint32_t dist_multiplier) {
    uint32_t r;

    if (cluster >= dec->num_clusters) { dec->err = 1; return 0; }

    if (!dec->lz77_enabled) {
        uint32_t token = dec_read_symbol(dec, br, cluster);
        return read_uint(br, &dec->configs[cluster], token);
    }

    if (lz77_ensure_window(dec) != 0) { dec->err = 1; return 0; }

    if (dec->num_to_copy > 0) {
        r = dec->window[dec->copy_pos & LZ77_WINDOW_MASK];
        dec->copy_pos++;
        dec->num_to_copy--;
    } else {
        uint32_t token = dec_read_symbol(dec, br, cluster);
        if (token >= dec->min_symbol) {
            uint32_t lz_cluster = dec->clusters[dec->num_dist - 1];
            uint32_t num_to_copy, distance;
            if (dec->num_decoded == 0) {
                JXL_ERR(dec->ctx, "LZ77 repeat before any symbol");
                dec->err = 1;
                return 0;
            }
            num_to_copy = read_uint(br, &dec->lz_len_conf, token - dec->min_symbol);
            if (num_to_copy > 0xffffffffu - dec->min_length) {
                dec->err = 1;
                return 0;
            }
            dec->num_to_copy = num_to_copy + dec->min_length;

            token = dec_read_symbol(dec, br, lz_cluster);
            distance = read_uint(br, &dec->configs[lz_cluster], token);
            if (dist_multiplier == 0) {
                /* keep distance */
            } else if (distance < 120) {
                int32_t offset = lz77_special_distances[distance][0];
                int32_t d = lz77_special_distances[distance][1];
                int32_t v = offset + (int32_t)dist_multiplier * d - 1;
                distance = v < 0 ? 0 : (uint32_t)v;
            } else {
                distance -= 120;
            }
            if (distance > LZ77_WINDOW_MASK) distance = LZ77_WINDOW_MASK;
            distance += 1;
            if (distance > dec->num_decoded) distance = dec->num_decoded;
            dec->copy_pos = dec->num_decoded - distance;

            r = dec->window[dec->copy_pos & LZ77_WINDOW_MASK];
            dec->copy_pos++;
            dec->num_to_copy--;
        } else {
            r = read_uint(br, &dec->configs[cluster], token);
        }
    }
    dec->window[dec->num_decoded & LZ77_WINDOW_MASK] = r;
    dec->num_decoded++;
    return r;
}

uint32_t jxl_dec_read_mult(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx,
                           uint32_t dist_multiplier) {
    if (ctx_idx >= dec->num_dist) { dec->err = 1; return 0; }
    return jxl_dec_read_clustered(dec, br, dec->clusters[ctx_idx],
                                  dist_multiplier);
}

uint32_t jxl_dec_read(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx) {
    return jxl_dec_read_mult(dec, br, ctx_idx, 0);
}

int jxl_dec_finalize(jxl_dec *dec) {
    if (dec->err) return -1;
    if (dec->use_prefix) return 0;
    return dec->state == ANS_SIGNATURE ? 0 : -1;
}

/* ----- permutation (Lehmer code), used by the frame TOC and coef orders --- */

static uint32_t perm_context(uint32_t x) {
    uint32_t b = jxl_bitlen(x);
    return b < 7 ? b : 7;
}

int jxl_read_permutation(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br, uint32_t size,
                         uint32_t skip, uint32_t *out) {
    uint32_t *lehmer = NULL;
    uint32_t *temp = NULL;
    uint32_t end, i, ntemp;
    uint32_t prev = 0;
    int rc = -1;

    end = jxl_dec_read(dec, br, perm_context(size));
    if (end > size - skip || br->err) {
        JXL_ERR(ctx, "invalid permutation length");
        return -1;
    }
    if (end) {
        lehmer = (uint32_t *)jxl_calloc(ctx, end, sizeof(uint32_t));
        if (!lehmer) return -1;
    }
    for (i = 0; i < end; i++) {
        lehmer[i] = jxl_dec_read(dec, br, perm_context(prev));
        if (lehmer[i] >= size - skip - i || br->err) {
            JXL_ERR(ctx, "invalid permutation entry");
            goto done;
        }
        prev = lehmer[i];
    }

    ntemp = size - skip;
    temp = (uint32_t *)jxl_calloc(ctx, ntemp ? ntemp : 1, sizeof(uint32_t));
    if (!temp) goto done;
    for (i = 0; i < ntemp; i++) temp[i] = skip + i;

    for (i = 0; i < skip; i++) out[i] = i;
    for (i = 0; i < end; i++) {
        uint32_t idx = lehmer[i];
        out[skip + i] = temp[idx];
        memmove(temp + idx, temp + idx + 1, (ntemp - idx - 1) * sizeof(uint32_t));
        ntemp--;
    }
    for (i = 0; i < ntemp; i++) out[skip + end + i] = temp[i];
    rc = 0;

done:
    jxl_free(ctx, lehmer);
    jxl_free(ctx, temp);
    return rc;
}
