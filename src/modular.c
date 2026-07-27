/* modular.c -- the Modular sub-codec: meta-adaptive (MA) decision trees,
 * spatial predictors (including the self-correcting/weighted predictor), the
 * RCT / palette / squeeze transforms, and the per-channel sample decoder.
 *
 * A Modular image is a list of integer channels. The bitstream declares
 * transforms that rewrite that list (squeeze splits channels in half, palette
 * folds several channels into an index channel plus a palette meta-channel,
 * RCT is in-place), the transformed list is what actually gets coded, and the
 * transforms are undone in reverse afterwards.
 *
 * Channels are views (pointer + stride) into buffers owned by the image, so
 * squeeze's split/merge and group tiling are pure pointer arithmetic.
 *
 * Samples are int32 throughout. The bitstream's modular_16bit_buffers flag
 * only promises that 16 bits would suffice, so decoding wider is safe and
 * produces identical values.
 */
#include "jxl_internal.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t div_ceil_u32(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

static int32_t grad_clamped(int32_t n, int32_t w, int32_t nw) {
    int32_t lo = n < w ? n : w;
    int32_t hi = n < w ? w : n;
    int64_t g = (int64_t)n + w - nw;
    if (g < lo) return lo;
    if (g > hi) return hi;
    return (int32_t)g;
}

/* The SELECT predictor, split out of predict_sample so the fixed-leaf loop
   can use it without going through the runtime switch. Same expression,
   including the unsigned differencing that keeps it well-defined at the
   extremes of int32. */
static int32_t sel_pred(int32_t n, int32_t w, int32_t nw) {
    uint64_t dn = (uint64_t)(n > nw ? (int64_t)n - nw : (int64_t)nw - n);
    uint64_t dw = (uint64_t)(w > nw ? (int64_t)w - nw : (int64_t)nw - w);
    return dn < dw ? w : n;
}

/* ===================================================================== */
/* channel lists                                                          */
/* ===================================================================== */

void jxl_chanlist_free(jxl_ctx *ctx, jxl_chanlist *cl) {
    if (!cl) return;
    jxl_free(ctx, cl->chans);
    cl->chans = NULL;
    cl->n = cl->cap = 0;
}

static int chanlist_reserve(jxl_ctx *ctx, jxl_chanlist *cl, uint32_t need) {
    uint32_t cap;
    jxl_mchan *nc;
    if (cl->cap >= need) return 0;
    cap = cl->cap ? cl->cap : 8;
    while (cap < need) cap *= 2;
    nc = (jxl_mchan *)jxl_realloc_array(ctx, cl->chans, cl->cap, cap,
                                        sizeof(jxl_mchan));
    if (!nc) return -1;
    cl->chans = nc;
    cl->cap = cap;
    return 0;
}

int jxl_chanlist_push(jxl_ctx *ctx, jxl_chanlist *cl, const jxl_mchan *ch) {
    if (chanlist_reserve(ctx, cl, cl->n + 1) != 0) return -1;
    cl->chans[cl->n++] = *ch;
    return 0;
}

static int chanlist_insert(jxl_ctx *ctx, jxl_chanlist *cl, uint32_t at,
                           const jxl_mchan *ch) {
    if (chanlist_reserve(ctx, cl, cl->n + 1) != 0) return -1;
    memmove(cl->chans + at + 1, cl->chans + at,
            (cl->n - at) * sizeof(jxl_mchan));
    cl->chans[at] = *ch;
    cl->n++;
    return 0;
}

static void chanlist_remove_range(jxl_chanlist *cl, uint32_t from, uint32_t to) {
    if (to <= from) return;
    memmove(cl->chans + from, cl->chans + to, (cl->n - to) * sizeof(jxl_mchan));
    cl->n -= (to - from);
}

/* ===================================================================== */
/* MA tree                                                                */
/* ===================================================================== */

/* Both arrays grow together: bnode[] is scratch that remembers which binary
   node each flat slot still has to be filled from. */
static int ma_flat_grow(jxl_ctx *ctx, jxl_ma_flat **flat, uint32_t **bnode,
                        uint32_t *cap, uint32_t need) {
    uint32_t ncap = *cap ? *cap : 16;
    jxl_ma_flat *nf;
    uint32_t *nb;

    if (need <= *cap) return 0;
    while (ncap < need) ncap *= 2;
    nf = (jxl_ma_flat *)jxl_realloc_array(ctx, *flat, *cap, ncap,
                                          sizeof(jxl_ma_flat));
    if (!nf) return -1;
    *flat = nf;
    nb = (uint32_t *)jxl_realloc_array(ctx, *bnode, *cap, ncap,
                                       sizeof(uint32_t));
    if (!nb) return -1;
    *bnode = nb;
    *cap = ncap;
    return 0;
}

/* Properties 0 and 1 are the channel and stream index, which do not vary
   within one channel of one stream, so a test on either has an answer that is
   fixed before the first sample is decoded. `fold` says to take that answer
   now and descend past the node. Chains of them collapse in one go. This
   terminates for the same reason the walk does: jxl_ma_config_read has
   already checked that every child index is strictly greater than its
   parent's, so each step moves forward in the array.

   Left is child + 0 and right is child + 1, and ma_flatten below pairs side 0
   with the o0 == 0 slots -- where o0 is `v > split ? 0 : 1`. So `v > value`
   picks the left child, matching the walk exactly. */
static uint32_t ma_fold(const jxl_ma_node *raw, uint32_t i, int fold,
                        int32_t channel, int32_t stream_idx) {
    if (!fold) return i;
    while (raw[i].property == 0 || raw[i].property == 1) {
        int32_t v = raw[i].property == 0 ? channel : stream_idx;
        i = raw[i].child + (v > raw[i].value ? 0u : 1u);
    }
    return i;
}

/* Rewrite the binary tree into the two-levels-per-entry form that
   ma_get_leaf walks. Slots are filled breadth-first: by the time the loop
   reaches slot fi, bnode[fi] names the binary node it stands for, and filling
   it appends the four grandchild slots to the end. A leaf is copied in whole,
   so a finished walk needs no further indirection; a child that is a leaf has
   no test to contribute, so its slot in the parent gets property 0 with a
   split no property value can exceed and both its grandchild slots hold that
   same leaf.

   With `fold` set, every node reached is first pushed past any run of tests
   on properties 0 and 1. Folding has to happen here rather than as a separate
   pass over the binary tree because an entry packs two levels: which node
   lands at which level shifts as constant tests disappear, and only the
   flattening knows that. */
static int ma_flatten(jxl_ctx *ctx, const jxl_ma_node *raw,
                      uint32_t count, uint32_t root,
                      const jxl_ma_leaf *leaves, int fold,
                      int32_t channel, int32_t stream_idx,
                      jxl_ma_flat **out_flat, uint32_t *out_n) {
    jxl_ma_flat *flat = NULL;
    uint32_t *bnode = NULL;
    uint32_t cap = 0, n = 0, fi;
    int rc = -1;

    if (ma_flat_grow(ctx, &flat, &bnode, &cap, 1) != 0) goto done;
    bnode[0] = ma_fold(raw, root, fold, channel, stream_idx);
    n = 1;

    for (fi = 0; fi < n; fi++) {
        uint32_t b = bnode[fi], base, side;

        if (raw[b].property < 0) {
            flat[fi].property = -1;
            flat[fi].u.leaf = leaves[raw[b].child];
            continue;
        }
        /* Each entry that is not a leaf appends four and consumes a distinct
           decision node, so this bound cannot be reached by a well-formed
           tree; it just keeps a malformed one from allocating without end. */
        if (n > 4 * count + 4) {
            JXL_ERR(ctx, "modular: MA tree does not flatten");
            goto done;
        }
        if (ma_flat_grow(ctx, &flat, &bnode, &cap, n + 4) != 0) goto done;
        base = n;
        n += 4;
        flat[fi].property = raw[b].property;
        flat[fi].u.dec.split0 = raw[b].value;
        flat[fi].u.dec.child = base;

        for (side = 0; side < 2; side++) {
            uint32_t c = ma_fold(raw, raw[b].child + side, fold,
                                 channel, stream_idx);   /* left, then right */
            uint32_t slot = base + 2 * side;
            int32_t p, s;

            if (raw[c].property < 0) {
                p = 0;
                s = 0x7fffffff;
                bnode[slot] = c;
                bnode[slot + 1] = c;
            } else {
                p = raw[c].property;
                s = raw[c].value;
                bnode[slot] = ma_fold(raw, raw[c].child, fold,
                                      channel, stream_idx);
                bnode[slot + 1] = ma_fold(raw, raw[c].child + 1, fold,
                                          channel, stream_idx);
            }
            if (side == 0) {
                flat[fi].u.dec.prop1 = p;
                flat[fi].u.dec.split1 = s;
            } else {
                flat[fi].u.dec.prop2 = p;
                flat[fi].u.dec.split2 = s;
            }
        }
    }

    *out_flat = flat;
    *out_n = n;
    flat = NULL;
    rc = 0;

done:
    jxl_free(ctx, bnode);
    jxl_free(ctx, flat);
    return rc;
}

/* Nodes arrive in a "folding" order: a running counter says how many are
   still expected, and the tree is rebuilt afterwards by walking the list
   backwards through a queue (matching libjxl's DecodeTree). */
int jxl_ma_config_read(jxl_ctx *ctx, jxl_br *br, jxl_ma_config *ma,
                       size_t node_limit) {
    jxl_dec tree_dec;
    jxl_ma_node *raw = NULL;
    jxl_ma_leaf *leaves = NULL;
    uint32_t cap = 0, count = 0;
    uint32_t lcap = 0;
    size_t nodes_left = 1;
    uint32_t ctx_count = 0;
    uint32_t *dq = NULL;
    uint32_t dq_head = 0, dq_tail = 0;
    int rc = -1;
    uint32_t i;

    memset(ma, 0, sizeof(*ma));
    memset(&tree_dec, 0, sizeof(tree_dec));

    if (jxl_dec_init(ctx, &tree_dec, br, 6) != 0) return -1;
    jxl_dec_begin(&tree_dec, br);

    while (nodes_left > 0) {
        uint32_t property;
        jxl_ma_node node;

        if (count >= (1u << 26) || count > node_limit) {
            JXL_ERR(ctx, "modular: MA tree too large");
            goto done;
        }
        if (count == cap) {
            uint32_t ncap = cap ? cap * 2 : 16;
            jxl_ma_node *nn = (jxl_ma_node *)jxl_realloc_array(
                ctx, raw, cap, ncap, sizeof(jxl_ma_node));
            if (!nn) goto done;
            raw = nn;
            cap = ncap;
        }
        nodes_left--;

        memset(&node, 0, sizeof(node));
        property = jxl_dec_read(&tree_dec, br, 1);
        if (br->err || tree_dec.err) {
            JXL_ERR(ctx, "modular: truncated MA tree");
            goto done;
        }
        if (property > 0) {
            node.property = (int32_t)(property - 1);
            node.value = jxl_unpack_signed(jxl_dec_read(&tree_dec, br, 0));
            nodes_left += 2;
        } else {
            uint32_t pred = jxl_dec_read(&tree_dec, br, 2);
            uint32_t mul_log, mul_bits;
            jxl_ma_leaf *leaf;
            if (pred > JXL_PRED_AVG_ALL) {
                JXL_ERR(ctx, "modular: bad predictor %u", (unsigned)pred);
                goto done;
            }
            if (ctx_count == lcap) {
                uint32_t ncap = lcap ? lcap * 2 : 16;
                jxl_ma_leaf *nl = (jxl_ma_leaf *)jxl_realloc_array(
                    ctx, leaves, lcap, ncap, sizeof(jxl_ma_leaf));
                if (!nl) goto done;
                leaves = nl;
                lcap = ncap;
            }
            /* Leaves are numbered in read order, which is also the order the
               sample decoder's contexts are numbered, so a leaf's index into
               leaves[] is its context id. */
            leaf = &leaves[ctx_count];
            leaf->predictor = (uint8_t)pred;
            leaf->cluster = 0;      /* filled in once the decoder exists */
            leaf->offset = jxl_unpack_signed(jxl_dec_read(&tree_dec, br, 3));
            mul_log = jxl_dec_read(&tree_dec, br, 4);
            if (mul_log > 30) {
                JXL_ERR(ctx, "modular: bad MA multiplier");
                goto done;
            }
            mul_bits = jxl_dec_read(&tree_dec, br, 5);
            if (mul_bits > (1u << (31 - mul_log)) - 2) {
                JXL_ERR(ctx, "modular: bad MA multiplier bits");
                goto done;
            }
            leaf->multiplier = (mul_bits + 1) << mul_log;
            node.property = -1;
            node.child = ctx_count;
            ctx_count++;
        }
        raw[count++] = node;
    }
    if (getenv("JXL_DEBUG_TREE")) {
        uint32_t k;
        fprintf(stderr, "tree: %u nodes, %u ctx, prefix=%d nclusters=%u\n",
                (unsigned)count, (unsigned)ctx_count, tree_dec.use_prefix,
                (unsigned)tree_dec.num_clusters);
        for (k = 0; k < count && k < 12; k++) {
            if (raw[k].property >= 0) {
                fprintf(stderr, "  [%u] prop=%d value=%d\n",
                        (unsigned)k, (int)raw[k].property, (int)raw[k].value);
            } else {
                const jxl_ma_leaf *lf = &leaves[raw[k].child];
                fprintf(stderr, "  [%u] leaf ctx=%u pred=%u off=%d mul=%u\n",
                        (unsigned)k, (unsigned)raw[k].child,
                        (unsigned)lf->predictor, (int)lf->offset,
                        (unsigned)lf->multiplier);
            }
        }
    }
    if (jxl_dec_finalize(&tree_dec) != 0) {
        JXL_ERR(ctx, "modular: bad MA tree ANS final state (state=0x%x)",
                (unsigned)tree_dec.state);
        goto done;
    }
    jxl_dec_free(&tree_dec);
    memset(&tree_dec, 0, sizeof(tree_dec));

    if (jxl_dec_init(ctx, &ma->dec, br, ctx_count) != 0) goto done;

    /* Map leaf context ids through the sample decoder's cluster map. A leaf's
       index into leaves[] is its context id, so this is a straight walk. */
    if (ctx_count > ma->dec.num_dist) {
        JXL_ERR(ctx, "modular: MA leaf context out of range");
        goto done;
    }
    for (i = 0; i < ctx_count; i++) leaves[i].cluster = ma->dec.clusters[i];

    /* Rebuild the tree: walk the node list backwards, pushing finished
       subtrees onto a queue; a decision node takes the two at the front
       (right first, then left). */
    dq = (uint32_t *)jxl_calloc(ctx, count + 2, sizeof(uint32_t));
    if (!dq) goto done;
    for (i = count; i > 0; i--) {
        uint32_t idx = i - 1;
        if (raw[idx].property >= 0) {
            uint32_t right, left;
            if (dq_tail - dq_head < 2) {
                JXL_ERR(ctx, "modular: malformed MA tree");
                goto done;
            }
            right = dq[dq_head++];
            left = dq[dq_head++];
            /* dq receives indices in strictly decreasing order and is a FIFO,
               so two consecutive pops are always neighbours. Only the left
               index is kept; the walk derives the right one as child + 1. */
            if (right != left + 1) {
                JXL_ERR(ctx, "modular: malformed MA tree (split children)");
                goto done;
            }
            raw[idx].child = left;
        }
        dq[dq_tail++] = idx;
    }
    if (dq_tail - dq_head != 1) {
        JXL_ERR(ctx, "modular: malformed MA tree (%u roots)",
                (unsigned)(dq_tail - dq_head));
        goto done;
    }

    /* The queue only ever holds indices greater than the node being filled
       in, so the rebuild above cannot produce a child that points backwards
       and every walk from the root strictly increases the node index.
       Checking that invariant here, once, lets ma_get_leaf walk without a
       termination guard of its own -- and a guard there could only stop
       mid-walk and hand the caller a decision node in place of a leaf. */
    for (i = 0; i < count; i++) {
        if (raw[i].property < 0) continue;
        if (raw[i].child <= i || raw[i].child + 1 >= count) {
            JXL_ERR(ctx, "modular: MA tree has a cycle at node %u",
                    (unsigned)i);
            goto done;
        }
    }
    if (ma_flatten(ctx, raw, count, dq[dq_head], leaves, 0, 0, 0,
                   &ma->flat, &ma->nflat) != 0)
        goto done;
    /* Handed over, not copied: a channel rebuilds the flat form from these
       with its constant tests folded out. */
    ma->raw = raw;
    ma->nraw = count;
    ma->root = dq[dq_head];
    ma->leaves = leaves;
    raw = NULL;
    leaves = NULL;
    ma->valid = 1;
    rc = 0;

done:
    jxl_free(ctx, dq);
    jxl_free(ctx, raw);
    jxl_free(ctx, leaves);
    jxl_dec_free(&tree_dec);
    if (rc != 0) jxl_ma_config_free(ctx, ma);
    return rc;
}

void jxl_ma_config_free(jxl_ctx *ctx, jxl_ma_config *ma) {
    if (!ma) return;
    jxl_free(ctx, ma->flat);
    ma->flat = NULL;
    ma->nflat = 0;
    jxl_free(ctx, ma->raw);
    ma->raw = NULL;
    jxl_free(ctx, ma->leaves);
    ma->leaves = NULL;
    jxl_free(ctx, (void *)ma->wp_lut);
    ma->wp_lut = NULL;
    ma->nraw = 0;
    ma->root = 0;
    jxl_dec_free(&ma->dec);
    ma->valid = 0;
}

/* ===================================================================== */
/* predictors                                                             */
/* ===================================================================== */

/* (1 << 24) / i for i in 1..64, and 0 at index 0. This used to be built
   lazily, which put an init check and a call in sc_predict -- once per
   sample. As a constant the compiler also knows the values never move. */
static const uint32_t jxl_div_lookup[65] = {
    0u, 16777216u, 8388608u, 5592405u, 4194304u, 3355443u,
    2796202u, 2396745u, 2097152u, 1864135u, 1677721u, 1525201u,
    1398101u, 1290555u, 1198372u, 1118481u, 1048576u, 986895u,
    932067u, 883011u, 838860u, 798915u, 762600u, 729444u,
    699050u, 671088u, 645277u, 621378u, 599186u, 578524u,
    559240u, 541200u, 524288u, 508400u, 493447u, 479349u,
    466033u, 453438u, 441505u, 430185u, 419430u, 409200u,
    399457u, 390167u, 381300u, 372827u, 364722u, 356962u,
    349525u, 342392u, 335544u, 328965u, 322638u, 316551u,
    310689u, 305040u, 299593u, 294337u, 289262u, 284359u,
    279620u, 275036u, 270600u, 266305u, 262144u
};

typedef struct {
    int64_t prediction;
    int32_t max_error;
    int64_t subpred[4];
} jxl_sc_result;

typedef struct {
    uint32_t width, x, y;
    int32_t *true_err_row;
    uint32_t *subpred_err_row;   /* 4 per column */
    jxl_wp_header wp;
    int default_wp;
    int32_t true_err_w, true_err_nw, true_err_n, true_err_ne;
    uint32_t subpred_err_nw_ww[4], subpred_err_n_w[4], subpred_err_ne[4];
} jxl_sc_pred;

typedef struct {
    uint32_t width;
    int32_t *prev_row;
    int32_t *curr_row;
    uint32_t prev_len, curr_len;
    uint32_t x, y;
    int32_t w, n, nw, prev_grad;

    int use_sc;
    jxl_sc_pred sc;

    /* previously decoded channels with identical geometry, newest first */
    const jxl_mchan **prev_chans;
    uint32_t nprev;
} jxl_pred_state;

static void pred_state_free(jxl_ctx *ctx, jxl_pred_state *ps) {
    jxl_free(ctx, ps->prev_row);
    jxl_free(ctx, ps->curr_row);
    jxl_free(ctx, ps->sc.true_err_row);
    jxl_free(ctx, ps->sc.subpred_err_row);
    memset(ps, 0, sizeof(*ps));
}

static int pred_state_reset(jxl_ctx *ctx, jxl_pred_state *ps, uint32_t width,
                            const jxl_wp_header *wp,
                            const jxl_mchan **prev_chans, uint32_t nprev) {
    pred_state_free(ctx, ps);

    ps->width = width;
    ps->prev_row = (int32_t *)jxl_calloc(ctx, width ? width : 1, sizeof(int32_t));
    ps->curr_row = (int32_t *)jxl_calloc(ctx, width ? width : 1, sizeof(int32_t));
    if (!ps->prev_row || !ps->curr_row) return -1;
    ps->prev_chans = prev_chans;
    ps->nprev = nprev;
    if (wp) {
        ps->use_sc = 1;
        ps->sc.width = width;
        ps->sc.wp = *wp;
        ps->sc.default_wp =
            wp->p1 == 16 && wp->p2 == 10 &&
            wp->p3a == 7 && wp->p3b == 7 && wp->p3c == 7 &&
            wp->p3d == 0 && wp->p3e == 0 &&
            wp->w0 == 13 && wp->w1 == 12 &&
            wp->w2 == 12 && wp->w3 == 12;
        ps->sc.true_err_row =
            (int32_t *)jxl_calloc(ctx, width ? width : 1, sizeof(int32_t));
        ps->sc.subpred_err_row =
            (uint32_t *)jxl_calloc(ctx, (size_t)(width ? width : 1) * 4,
                                   sizeof(uint32_t));
        if (!ps->sc.true_err_row || !ps->sc.subpred_err_row) return -1;
    }
    return 0;
}

static int32_t pred_nn(const jxl_pred_state *ps) {
    return ps->x < ps->curr_len ? ps->curr_row[ps->x] : ps->n;
}
static int32_t pred_ne(const jxl_pred_state *ps) {
    if (ps->prev_len == 0 || ps->x + 1 >= ps->width) return ps->n;
    return ps->prev_row[ps->x + 1];
}
static int32_t pred_nee(const jxl_pred_state *ps) {
    if (ps->prev_len == 0 || ps->x + 2 >= ps->width) return pred_ne(ps);
    return ps->prev_row[ps->x + 2];
}
static int32_t pred_ww(const jxl_pred_state *ps) {
    if (ps->x >= 2) return ps->curr_row[ps->x - 2];
    return ps->w;
}

/* One lane of the weight computation. Same arithmetic as before; separate
   only so the four unrolled uses stay readable. */
static JXL_INLINE_HINT uint32_t sc_weight_one(uint32_t err_sum, uint32_t wn,
                                              const uint32_t *dl) {
    uint64_t v = ((uint64_t)err_sum + 1) >> 5;
    uint32_t shift = v > 1 ? jxl_floor_log2_u64(v) : 0;
    uint32_t idx = (err_sum >> shift) + 1;
    if (idx > 64) idx = 64;
    return 4 + ((wn * dl[idx]) >> shift);
}

static void sc_predict(const jxl_sc_pred *sc, int32_t n, int32_t nw, int32_t ne,
                       int32_t w, int32_t nn, jxl_sc_result *out) {
    const uint32_t *dl = jxl_div_lookup;
    int64_t te_w = sc->true_err_w, te_nw = sc->true_err_nw;
    int64_t te_n = sc->true_err_n, te_ne = sc->true_err_ne;
    int64_t n3 = (int64_t)n << 3, nw3 = (int64_t)nw << 3;
    int64_t ne3 = (int64_t)ne << 3, w3 = (int64_t)w << 3;
    int64_t nn3 = (int64_t)nn << 3;
    /* Scalars, not arrays. The four-lane steps below are all constant-trip
       loops, but keeping them as arrays left the values in memory: summing
       four uint32s was 6.7% of the whole decode on its own, which is only
       possible if each one is a reload. Unrolled into named locals they stay
       in registers and the two reductions become three adds each. */
    uint32_t es0, es1, es2, es3;
    uint32_t wt0, wt1, wt2, wt3;
    uint32_t sum_weights;
    int log_weight = 0;
    int64_t s;
    /* The four sub-predictions and the prediction stay in locals until the
       end. They are read back a few lines below, and written through `out`
       the compiler has to assume they might alias `sc` -- so each store came
       straight back as a reload on a path that runs once per sample. Same
       reasoning as the weight lanes below, which were unrolled out of arrays
       for exactly this. */
    int64_t sp0, sp1, sp2, sp3, pred;

    sp0 = w3 + ne3 - n3;
    if (sc->default_wp) {
        sp1 = n3 - (((te_w + te_n + te_ne) * 16) >> 5);
        sp2 = w3 - (((te_w + te_n + te_nw) * 10) >> 5);
        sp3 = n3 - (((te_nw + te_n + te_ne) * 7) >> 5);
    } else {
        sp1 = n3 - (((te_w + te_n + te_ne) * (int64_t)sc->wp.p1) >> 5);
        sp2 = w3 - (((te_w + te_n + te_nw) * (int64_t)sc->wp.p2) >> 5);
        sp3 = n3 - ((te_nw * (int64_t)sc->wp.p3a +
                     te_n * (int64_t)sc->wp.p3b +
                     te_ne * (int64_t)sc->wp.p3c +
                     (nn3 - n3) * (int64_t)sc->wp.p3d +
                     (nw3 - w3) * (int64_t)sc->wp.p3e) >> 5);
    }

    es0 = sc->subpred_err_nw_ww[0] + sc->subpred_err_n_w[0] + sc->subpred_err_ne[0];
    es1 = sc->subpred_err_nw_ww[1] + sc->subpred_err_n_w[1] + sc->subpred_err_ne[1];
    es2 = sc->subpred_err_nw_ww[2] + sc->subpred_err_n_w[2] + sc->subpred_err_ne[2];
    es3 = sc->subpred_err_nw_ww[3] + sc->subpred_err_n_w[3] + sc->subpred_err_ne[3];

#define JXL_SC_WEIGHT(es, wn) (                                                   sc_weight_one((es), (wn), dl))

    if (sc->default_wp) {
        wt0 = JXL_SC_WEIGHT(es0, 13);
        wt1 = JXL_SC_WEIGHT(es1, 12);
        wt2 = JXL_SC_WEIGHT(es2, 12);
        wt3 = JXL_SC_WEIGHT(es3, 12);
    } else {
        wt0 = JXL_SC_WEIGHT(es0, sc->wp.w0);
        wt1 = JXL_SC_WEIGHT(es1, sc->wp.w1);
        wt2 = JXL_SC_WEIGHT(es2, sc->wp.w2);
        wt3 = JXL_SC_WEIGHT(es3, sc->wp.w3);
    }
#undef JXL_SC_WEIGHT

    sum_weights = wt0 + wt1 + wt2 + wt3;
    {
        uint32_t v = sum_weights >> 4;
        if (v > 1) log_weight = (int)jxl_floor_log2_u64(v);
    }
    wt0 >>= log_weight; wt1 >>= log_weight;
    wt2 >>= log_weight; wt3 >>= log_weight;
    sum_weights = wt0 + wt1 + wt2 + wt3;

    s = ((int64_t)sum_weights >> 1) - 1;
    s += sp0 * (int64_t)wt0;
    s += sp1 * (int64_t)wt1;
    s += sp2 * (int64_t)wt2;
    s += sp3 * (int64_t)wt3;
    pred = (s * (int64_t)dl[sum_weights > 64 ? 64 : sum_weights]) >> 24;

    if (((te_n ^ te_w) | (te_n ^ te_nw)) <= 0) {
        int64_t lo = n3 < w3 ? n3 : w3;
        int64_t hi = n3 > w3 ? n3 : w3;
        if (ne3 < lo) lo = ne3;
        if (ne3 > hi) hi = ne3;
        if (pred < lo) pred = lo;
        if (pred > hi) pred = hi;
    }

    out->subpred[0] = sp0;
    out->subpred[1] = sp1;
    out->subpred[2] = sp2;
    out->subpred[3] = sp3;
    out->prediction = pred;

    {
        /* Unrolled for the same reason as the weight lanes above: three
           elements out of a stack array, compared against a running max. */
        int64_t max_error = te_w;
        int64_t i;
        for (i = 0; i < 3; i++) {
            int64_t e = i == 0 ? te_n : (i == 1 ? te_nw : te_ne);
            int64_t a = e < 0 ? -e : e;
            int64_t b = max_error < 0 ? -max_error : max_error;
            if (a > b) max_error = e;
        }
        out->max_error = (int32_t)max_error;
    }
}

static void sc_record(jxl_sc_pred *sc, const jxl_sc_result *pred, int32_t sample) {
    int64_t s = (int64_t)sample << 3;
    int64_t true_err = pred->prediction - s;
    uint32_t subpred_err[4];
    int i;

    for (i = 0; i < 4; i++) {
        int64_t d = pred->subpred[i] - s;
        if (d < 0) d = -d;
        subpred_err[i] = (uint32_t)((d + 3) >> 3);
    }
    sc->true_err_row[sc->x] = (int32_t)true_err;
    for (i = 0; i < 4; i++) sc->subpred_err_row[sc->x * 4 + i] = subpred_err[i];
    sc->x++;

    if (sc->x >= sc->width) {
        sc->y++;
        sc->x = 0;
        sc->true_err_w = 0;
        sc->true_err_n = sc->true_err_row[0];
        sc->true_err_nw = sc->true_err_n;
        for (i = 0; i < 4; i++) {
            sc->subpred_err_n_w[i] = sc->subpred_err_row[i];
            sc->subpred_err_nw_ww[i] = sc->subpred_err_n_w[i];
        }
        if (sc->width <= 1) {
            sc->true_err_ne = sc->true_err_n;
            for (i = 0; i < 4; i++) sc->subpred_err_ne[i] = sc->subpred_err_n_w[i];
        } else {
            sc->true_err_ne = sc->true_err_row[1];
            for (i = 0; i < 4; i++) sc->subpred_err_ne[i] = sc->subpred_err_row[4 + i];
        }
    } else {
        sc->true_err_w = (int32_t)true_err;
        sc->true_err_nw = sc->true_err_n;
        sc->true_err_n = sc->true_err_ne;
        for (i = 0; i < 4; i++) {
            sc->subpred_err_nw_ww[i] = sc->subpred_err_n_w[i];
            sc->subpred_err_n_w[i] = sc->subpred_err_ne[i];
            sc->subpred_err_n_w[i] += subpred_err[i];
        }
        if (sc->x + 1 >= sc->width) {
            sc->true_err_ne = sc->true_err_n;
            for (i = 0; i < 4; i++) sc->subpred_err_ne[i] = sc->subpred_err_n_w[i];
        } else if (sc->y != 0) {
            uint32_t x = sc->x;
            sc->true_err_ne = sc->true_err_row[x + 1];
            for (i = 0; i < 4; i++)
                sc->subpred_err_ne[i] = sc->subpred_err_row[(x + 1) * 4 + i];
        }
    }
}

/* Per-sample decoding context: the 16 cached properties plus the
   self-correcting prediction, if the tree needs it. */
typedef struct {
    int32_t cache[16];
    jxl_sc_result sc;
    int has_sc;
} jxl_props;

static int32_t chan_get(const jxl_mchan *c, uint32_t x, uint32_t y) {
    return c->data[(size_t)y * c->stride + x];
}

static void props_compute(const jxl_pred_state *ps, jxl_props *pr,
                          int32_t channel, int32_t stream_idx) {
    int32_t w_nw = (int32_t)((uint32_t)ps->w - (uint32_t)ps->nw);
    if (ps->use_sc) {
        sc_predict(&ps->sc, ps->n, ps->nw, pred_ne(ps), ps->w, pred_nn(ps),
                   &pr->sc);
        pr->has_sc = 1;
    } else {
        pr->has_sc = 0;
        pr->sc.prediction = 0;
        pr->sc.max_error = 0;
    }
    /* The two static properties go in the same array as the rest so the tree
       walk can index every property uniformly. */
    pr->cache[0] = channel;
    pr->cache[1] = stream_idx;
    pr->cache[2] = (int32_t)ps->y;
    pr->cache[3] = (int32_t)ps->x;
    pr->cache[4] = (int32_t)(ps->n < 0 ? 0u - (uint32_t)ps->n : (uint32_t)ps->n);
    pr->cache[5] = (int32_t)(ps->w < 0 ? 0u - (uint32_t)ps->w : (uint32_t)ps->w);
    pr->cache[6] = ps->n;
    pr->cache[7] = ps->w;
    pr->cache[8] = (int32_t)((uint32_t)ps->w - (uint32_t)ps->prev_grad);
    pr->cache[9] = (int32_t)((uint32_t)w_nw + (uint32_t)ps->n);
    pr->cache[10] = w_nw;
    pr->cache[11] = (int32_t)((uint32_t)ps->nw - (uint32_t)ps->n);
    pr->cache[12] = (int32_t)((uint32_t)ps->n - (uint32_t)pred_ne(ps));
    pr->cache[13] = (int32_t)((uint32_t)ps->n - (uint32_t)pred_nn(ps));
    pr->cache[14] = (int32_t)((uint32_t)ps->w - (uint32_t)pred_ww(ps));
    pr->cache[15] = pr->sc.max_error;
}

static int32_t props_get_extra(const jxl_pred_state *ps, uint32_t prop_extra) {
    uint32_t idx = prop_extra / 4;
    uint32_t sub = prop_extra % 4;
    const jxl_mchan *ch;
    uint32_t x = ps->x, y = ps->y;
    int32_t c, g;

    if (idx >= ps->nprev) return 0;
    ch = ps->prev_chans[idx];
    if (x >= ch->w || y >= ch->h) return 0;
    c = chan_get(ch, x, y);
    if (sub == 0) return (int32_t)(c < 0 ? 0u - (uint32_t)c : (uint32_t)c);
    if (sub == 1) return c;

    if (x == 0 && y == 0) g = 0;
    else if (x == 0) g = chan_get(ch, 0, y - 1);
    else if (y == 0) g = chan_get(ch, x - 1, 0);
    else {
        int32_t nw = chan_get(ch, x - 1, y - 1);
        int32_t n = chan_get(ch, x, y - 1);
        int32_t w = chan_get(ch, x - 1, y);
        g = grad_clamped(n, w, nw);
    }
    if (sub == 2) {
        int64_t d = (int64_t)c - g;
        return (int32_t)(d < 0 ? -d : d);
    }
    return (int32_t)((uint32_t)c - (uint32_t)g);
}

static int32_t predict_sample(const jxl_pred_state *ps, const jxl_props *pr,
                              uint8_t predictor) {
    switch (predictor) {
        case JXL_PRED_ZERO: return 0;
        case JXL_PRED_WEST: return ps->w;
        case JXL_PRED_NORTH: return ps->n;
        case JXL_PRED_AVG_W_N:
            return (int32_t)(((int64_t)ps->w + ps->n) / 2);
        case JXL_PRED_SELECT: {
            int32_t n = ps->n, w = ps->w, nw = ps->nw;
            uint64_t dn = (uint64_t)(n > nw ? (int64_t)n - nw : (int64_t)nw - n);
            uint64_t dw = (uint64_t)(w > nw ? (int64_t)w - nw : (int64_t)nw - w);
            return dn < dw ? w : n;
        }
        case JXL_PRED_GRADIENT: return grad_clamped(ps->n, ps->w, ps->nw);
        case JXL_PRED_SELF_CORRECTING:
            return (int32_t)((pr->sc.prediction + 3) >> 3);
        case JXL_PRED_NORTH_EAST: return pred_ne(ps);
        case JXL_PRED_NORTH_WEST: return ps->nw;
        case JXL_PRED_WEST_WEST: return pred_ww(ps);
        case JXL_PRED_AVG_W_NW:
            return (int32_t)(((int64_t)ps->w + ps->nw) / 2);
        case JXL_PRED_AVG_N_NW:
            return (int32_t)(((int64_t)ps->n + ps->nw) / 2);
        case JXL_PRED_AVG_N_NE:
            return (int32_t)(((int64_t)ps->n + pred_ne(ps)) / 2);
        default: {
            int64_t n = ps->n, w = ps->w;
            int64_t nn = pred_nn(ps), ww = pred_ww(ps);
            int64_t nee = pred_nee(ps), ne = pred_ne(ps);
            return (int32_t)((6 * n - 2 * nn + 7 * w + ww + nee + 3 * ne + 8) / 16);
        }
    }
}

static void pred_record(jxl_pred_state *ps, const jxl_props *pr, int32_t sample) {
    if (ps->use_sc && pr->has_sc) sc_record(&ps->sc, &pr->sc, sample);

    ps->curr_row[ps->x] = sample;
    if (ps->x >= ps->curr_len) ps->curr_len = ps->x + 1;
    ps->x++;

    if (ps->x >= ps->width) {
        int32_t *tmp = ps->prev_row;
        uint32_t tlen = ps->prev_len;
        ps->y++;
        ps->x = 0;
        ps->prev_row = ps->curr_row;
        ps->prev_len = ps->curr_len;
        ps->curr_row = tmp;
        ps->curr_len = tlen;
        ps->prev_grad = 0;
        ps->n = ps->prev_row[0];
        ps->w = ps->n;
        ps->nw = ps->n;
    } else {
        ps->prev_grad = pr->cache[9];
        ps->w = sample;
        if (ps->prev_len == 0) {
            ps->nw = sample;
            ps->n = sample;
        } else {
            ps->nw = ps->n;
            ps->n = ps->prev_row[ps->x];
        }
    }
}

/* ===================================================================== */
/* modular header + transforms                                            */
/* ===================================================================== */

static void wp_header_read(jxl_br *br, jxl_wp_header *wp) {
    wp->p1 = 16; wp->p2 = 10;
    wp->p3a = 7; wp->p3b = 7; wp->p3c = 7; wp->p3d = 0; wp->p3e = 0;
    wp->w0 = 13; wp->w1 = 12; wp->w2 = 12; wp->w3 = 12;
    if (jxl_br_bool(br)) return;   /* default_wp */
    wp->p1 = jxl_br_read(br, 5);
    wp->p2 = jxl_br_read(br, 5);
    wp->p3a = jxl_br_read(br, 5);
    wp->p3b = jxl_br_read(br, 5);
    wp->p3c = jxl_br_read(br, 5);
    wp->p3d = jxl_br_read(br, 5);
    wp->p3e = jxl_br_read(br, 5);
    wp->w0 = jxl_br_read(br, 4);
    wp->w1 = jxl_br_read(br, 4);
    wp->w2 = jxl_br_read(br, 4);
    wp->w3 = jxl_br_read(br, 4);
}

static int transform_read(jxl_ctx *ctx, jxl_br *br, jxl_transform *tr) {
    uint32_t id = jxl_br_read(br, 2);
    memset(tr, 0, sizeof(*tr));
    tr->kind = (int)id;
    if (id == 0) {
        tr->begin_c = jxl_br_u32(br, 0, 3, 8, 6, 72, 10, 1096, 13);
        tr->rct_type = jxl_br_u32(br, 6, 0, 0, 2, 2, 4, 10, 6);
        if (tr->rct_type >= 42) {
            JXL_ERR(ctx, "modular: bad RCT type %u", (unsigned)tr->rct_type);
            return -1;
        }
    } else if (id == 1) {
        tr->begin_c = jxl_br_u32(br, 0, 3, 8, 6, 72, 10, 1096, 13);
        tr->num_c = jxl_br_u32(br, 1, 0, 3, 0, 4, 0, 1, 13);
        tr->nb_colours = jxl_br_u32(br, 0, 8, 256, 10, 1280, 12, 5376, 16);
        tr->nb_deltas = jxl_br_u32(br, 0, 0, 1, 8, 257, 10, 1281, 16);
        tr->d_pred = (uint8_t)jxl_br_read(br, 4);
        if (tr->d_pred > JXL_PRED_AVG_ALL) {
            JXL_ERR(ctx, "modular: bad palette predictor");
            return -1;
        }
    } else if (id == 2) {
        uint32_t i;
        tr->nsp = jxl_br_u32(br, 0, 0, 1, 4, 9, 6, 41, 8);
        if (tr->nsp) {
            tr->sp = (jxl_sq_param *)jxl_calloc(ctx, tr->nsp, sizeof(jxl_sq_param));
            if (!tr->sp) return -1;
        }
        for (i = 0; i < tr->nsp; i++) {
            tr->sp[i].horizontal = jxl_br_bool(br);
            tr->sp[i].in_place = jxl_br_bool(br);
            tr->sp[i].begin_c = jxl_br_u32(br, 0, 3, 8, 6, 72, 10, 1096, 13);
            tr->sp[i].num_c = jxl_br_u32(br, 1, 0, 2, 0, 3, 0, 4, 4);
        }
    } else {
        JXL_ERR(ctx, "modular: invalid transform id %u", (unsigned)id);
        return -1;
    }
    return br->err ? -1 : 0;
}

/* Fills in the implicit squeeze parameters used when none were coded. */
static int squeeze_default_params(jxl_ctx *ctx, jxl_transform *tr,
                                  const jxl_chanlist *cl) {
    uint32_t first = cl->nb_meta;
    uint32_t w, h;
    uint32_t cap = 32, n = 0;
    jxl_sq_param *sp;

    if (tr->nsp != 0) return 0;
    if (first >= cl->n) return -1;
    w = cl->chans[first].w;
    h = cl->chans[first].h;

    sp = (jxl_sq_param *)jxl_calloc(ctx, cap, sizeof(jxl_sq_param));
    if (!sp) return -1;

    if (cl->n - first >= 3) {
        const jxl_mchan *next = &cl->chans[first + 1];
        if (next->w == w && next->h == h) {
            sp[n].horizontal = 1; sp[n].in_place = 0;
            sp[n].begin_c = first + 1; sp[n].num_c = 2;
            n++;
            sp[n].horizontal = 0; sp[n].in_place = 0;
            sp[n].begin_c = first + 1; sp[n].num_c = 2;
            n++;
        }
    }

#define SQ_PUSH(horiz)                                                        \
    do {                                                                      \
        if (n == cap) {                                                       \
            jxl_sq_param *ns = (jxl_sq_param *)jxl_realloc_array(             \
                ctx, sp, cap, cap * 2, sizeof(jxl_sq_param));                 \
            if (!ns) { jxl_free(ctx, sp); return -1; }                        \
            sp = ns; cap *= 2;                                                \
        }                                                                     \
        sp[n].horizontal = (horiz);                                           \
        sp[n].in_place = 1;                                                   \
        sp[n].begin_c = first;                                                \
        sp[n].num_c = cl->n - first;                                          \
        n++;                                                                  \
    } while (0)

    if (h >= w && h > 8) {
        SQ_PUSH(0);
        h = div_ceil_u32(h, 2);
    }
    while (w > 8 || h > 8) {
        if (w > 8) { SQ_PUSH(1); w = div_ceil_u32(w, 2); }
        if (h > 8) { SQ_PUSH(0); h = div_ceil_u32(h, 2); }
    }
#undef SQ_PUSH

    tr->sp = sp;
    tr->nsp = n;
    return 0;
}

/* Applies a transform to the channel list. When `apply_grids` is set the
   channel views are split/merged too; otherwise only the geometry is
   computed (used while reading the header to size the local MA tree). */
static int transform_apply(jxl_ctx *ctx, jxl_transform *tr, jxl_chanlist *cl,
                           int apply_grids) {
    if (tr->kind == 0) {
        uint32_t begin = tr->begin_c, end = tr->begin_c + 3, i;
        uint32_t w, h;
        if (end > cl->n) {
            JXL_ERR(ctx, "modular: RCT channel range out of bounds");
            return -1;
        }
        w = cl->chans[begin].w;
        h = cl->chans[begin].h;
        for (i = begin + 1; i < end; i++) {
            if (cl->chans[i].w != w || cl->chans[i].h != h) {
                JXL_ERR(ctx, "modular: RCT channel size mismatch");
                return -1;
            }
        }
        return 0;
    }

    if (tr->kind == 1) {
        uint32_t begin = tr->begin_c, end = tr->begin_c + tr->num_c, i;
        uint32_t w, h;
        jxl_mchan pal;

        if (tr->num_c == 0 || end > cl->n) {
            JXL_ERR(ctx, "modular: palette channel range out of bounds");
            return -1;
        }
        if (begin < cl->nb_meta) {
            if (end > cl->nb_meta) {
                JXL_ERR(ctx, "modular: palette spans meta channels");
                return -1;
            }
            cl->nb_meta = cl->nb_meta + 2 - tr->num_c;
        } else {
            cl->nb_meta += 1;
        }
        w = cl->chans[begin].w;
        h = cl->chans[begin].h;
        for (i = begin + 1; i < end; i++) {
            if (cl->chans[i].w != w || cl->chans[i].h != h) {
                JXL_ERR(ctx, "modular: palette channel size mismatch");
                return -1;
            }
        }

        if (apply_grids) {
            uint32_t nmem = tr->num_c - 1;
            jxl_free(ctx, tr->saved);
            tr->saved = NULL;
            tr->nsaved = 0;
            if (nmem) {
                tr->saved = (jxl_mchan *)jxl_calloc(ctx, nmem, sizeof(jxl_mchan));
                if (!tr->saved) return -1;
                memcpy(tr->saved, cl->chans + begin + 1, nmem * sizeof(jxl_mchan));
                tr->nsaved = nmem;
            }
        }
        chanlist_remove_range(cl, begin + 1, end);

        memset(&pal, 0, sizeof(pal));
        pal.w = tr->nb_colours;
        pal.h = tr->num_c;
        pal.hshift = -1;
        pal.vshift = -1;
        pal.ow = pal.w;
        pal.oh = pal.h;
        if (apply_grids) {
            if (!tr->pal_buf) {
                size_t total;
                if (!jxl_size_mul(pal.w, pal.h, &total)) return -1;
                tr->pal_buf = (int32_t *)jxl_calloc(ctx, total ? total : 1,
                                                    sizeof(int32_t));
                if (!tr->pal_buf) return -1;
            }
            pal.data = tr->pal_buf;
            pal.stride = pal.w;
            tr->pal = pal;
        }
        return chanlist_insert(ctx, cl, 0, &pal);
    }

    /* squeeze */
    {
        uint32_t s;
        if (squeeze_default_params(ctx, tr, cl) != 0) {
            JXL_ERR(ctx, "modular: bad squeeze parameters");
            return -1;
        }
        for (s = 0; s < tr->nsp; s++) {
            const jxl_sq_param *sp = &tr->sp[s];
            uint32_t begin = sp->begin_c, end = sp->begin_c + sp->num_c;
            uint32_t idx;
            jxl_chanlist residu;

            memset(&residu, 0, sizeof(residu));
            if (end > cl->n) {
                JXL_ERR(ctx, "modular: squeeze range out of bounds");
                return -1;
            }
            if (begin < cl->nb_meta) {
                if (!sp->in_place || end > cl->nb_meta) {
                    JXL_ERR(ctx, "modular: bad squeeze meta range");
                    return -1;
                }
                cl->nb_meta += sp->num_c;
            }

            for (idx = begin; idx < end; idx++) {
                jxl_mchan *ch = &cl->chans[idx];
                jxl_mchan res = *ch;
                uint32_t len;

                if (ch->w == 0 || ch->h == 0) {
                    JXL_ERR(ctx, "modular: cannot squeeze empty channel");
                    jxl_chanlist_free(ctx, &residu);
                    return -1;
                }
                if (ch->hshift > 30 || ch->vshift > 30) {
                    JXL_ERR(ctx, "modular: channel squeezed too far");
                    jxl_chanlist_free(ctx, &residu);
                    return -1;
                }
                if (sp->horizontal) {
                    len = ch->w;
                    ch->w = div_ceil_u32(len, 2);
                    res.w = len / 2;
                    if (ch->hshift >= 0) { ch->hshift++; res.hshift++; }
                    if (apply_grids) res.data = ch->data + ch->w;
                } else {
                    len = ch->h;
                    ch->h = div_ceil_u32(len, 2);
                    res.h = len / 2;
                    if (ch->vshift >= 0) { ch->vshift++; res.vshift++; }
                    if (apply_grids) res.data = ch->data + (size_t)ch->h * ch->stride;
                }
                if (jxl_chanlist_push(ctx, &residu, &res) != 0) {
                    jxl_chanlist_free(ctx, &residu);
                    return -1;
                }
            }

            if (sp->in_place) {
                for (idx = end; idx < cl->n; idx++) {
                    if (jxl_chanlist_push(ctx, &residu, &cl->chans[idx]) != 0) {
                        jxl_chanlist_free(ctx, &residu);
                        return -1;
                    }
                }
                cl->n = end;
            }
            for (idx = 0; idx < residu.n; idx++) {
                if (jxl_chanlist_push(ctx, cl, &residu.chans[idx]) != 0) {
                    jxl_chanlist_free(ctx, &residu);
                    return -1;
                }
            }
            jxl_chanlist_free(ctx, &residu);
        }
    }
    return 0;
}

/* ----- inverse transforms ----- */

static void rct_inverse(jxl_transform *tr, jxl_chanlist *cl) {
    uint32_t permutation = tr->rct_type / 7;
    uint32_t type = tr->rct_type % 7;
    jxl_mchan *a = &cl->chans[tr->begin_c];
    jxl_mchan *b = &cl->chans[tr->begin_c + 1];
    jxl_mchan *c = &cl->chans[tr->begin_c + 2];
    uint32_t x, y;

    for (y = 0; y < a->h; y++) {
        int32_t *ra = a->data + (size_t)y * a->stride;
        int32_t *rb = b->data + (size_t)y * b->stride;
        int32_t *rc = c->data + (size_t)y * c->stride;
        for (x = 0; x < a->w; x++) {
            uint32_t va = (uint32_t)ra[x], vb = (uint32_t)rb[x], vc = (uint32_t)rc[x];
            uint32_t d, e, f;
            if (type == 6) {
                uint32_t tmp = va - (uint32_t)(((int32_t)vc) >> 1);
                e = vc + tmp;
                f = tmp - (uint32_t)(((int32_t)vb) >> 1);
                d = f + vb;
            } else {
                d = va;
                f = (type & 1) ? vc + va : vc;
                if ((type >> 1) == 1) e = vb + va;
                else if ((type >> 1) == 2)
                    e = vb + (uint32_t)(((int32_t)(va + f)) >> 1);
                else e = vb;
            }
            ra[x] = (int32_t)d;
            rb[x] = (int32_t)e;
            rc[x] = (int32_t)f;
        }
        switch (permutation) {
            case 1:
                for (x = 0; x < a->w; x++) {
                    int32_t t = ra[x]; ra[x] = rb[x]; rb[x] = t;
                    t = ra[x]; ra[x] = rc[x]; rc[x] = t;
                }
                break;
            case 2:
                for (x = 0; x < a->w; x++) {
                    int32_t t = ra[x]; ra[x] = rb[x]; rb[x] = t;
                    t = rb[x]; rb[x] = rc[x]; rc[x] = t;
                }
                break;
            case 3:
                for (x = 0; x < a->w; x++) {
                    int32_t t = rb[x]; rb[x] = rc[x]; rc[x] = t;
                }
                break;
            case 4:
                for (x = 0; x < a->w; x++) {
                    int32_t t = ra[x]; ra[x] = rb[x]; rb[x] = t;
                }
                break;
            case 5:
                for (x = 0; x < a->w; x++) {
                    int32_t t = ra[x]; ra[x] = rc[x]; rc[x] = t;
                }
                break;
            default: break;
        }
    }
}

static int32_t squeeze_tendency(int32_t a, int32_t b, int32_t c) {
    if (a >= b && b >= c) {
        int32_t x = (int32_t)((4 * (int64_t)a - 3 * (int64_t)c - b + 6) / 12);
        if (x - (x & 1) > 2 * (a - b)) x = 2 * (a - b) + 1;
        if (x + (x & 1) > 2 * (b - c)) x = 2 * (b - c);
        return x;
    }
    if (a <= b && b <= c) {
        int32_t x = (int32_t)((4 * (int64_t)a - 3 * (int64_t)c - b - 6) / 12);
        if (x + (x & 1) < 2 * (a - b)) x = 2 * (a - b) - 1;
        if (x - (x & 1) < 2 * (b - c)) x = 2 * (b - c);
        return x;
    }
    return 0;
}

/* Eight independent lifting steps. The caller supplies lanes that have no
   dependency on each other: columns for vertical squeeze, rows for
   horizontal squeeze. Returns 0 when an extreme input needs the scalar/int64
   tendency calculation. */
static JXL_TARGET_AVX2 JXL_INLINE_HINT int squeeze_step8(
    __m256i top, __m256i avg, __m256i next, __m256i residu,
    __m256i *first_out, __m256i *second_out) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i two = _mm256_set1_epi32(2);
    const __m256i all = _mm256_set1_epi32(-1);
    const __m256i limit = _mm256_set1_epi32(0x1fffffff);
    const __m256i neg_limit = _mm256_set1_epi32(-0x1fffffff);
    __m256i bad = _mm256_or_si256(
        _mm256_or_si256(_mm256_cmpgt_epi32(avg, limit),
                        _mm256_cmpgt_epi32(neg_limit, avg)),
        _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpgt_epi32(top, limit),
                            _mm256_cmpgt_epi32(neg_limit, top)),
            _mm256_or_si256(_mm256_cmpgt_epi32(next, limit),
                            _mm256_cmpgt_epi32(neg_limit, next))));
    __m256i ba, an, abs_ba, abs_an, abs_bn;
    __m256i pe, po, div3, absdiff, odd, ba2, repl;
    __m256i tendency, diff;

    if (_mm256_movemask_epi8(bad) != 0) return 0;

    ba = _mm256_sub_epi32(top, avg);
    an = _mm256_sub_epi32(avg, next);
    abs_ba = _mm256_abs_epi32(ba);
    abs_an = _mm256_abs_epi32(an);
    abs_bn = _mm256_abs_epi32(_mm256_sub_epi32(top, next));
    pe = _mm256_mul_epu32(abs_ba, _mm256_set1_epi32(0x55555556));
    po = _mm256_mul_epu32(_mm256_srli_epi64(abs_ba, 32),
                          _mm256_set1_epi32(0x55555556));
    div3 = _mm256_or_si256(
        _mm256_srli_epi64(pe, 32),
        _mm256_slli_epi64(_mm256_srli_epi64(po, 32), 32));
    absdiff = _mm256_srli_epi32(
        _mm256_add_epi32(_mm256_add_epi32(div3, abs_bn), two), 2);
    odd = _mm256_and_si256(absdiff, one);
    ba2 = _mm256_add_epi32(_mm256_slli_epi32(abs_ba, 1), odd);
    repl = _mm256_add_epi32(_mm256_slli_epi32(abs_ba, 1), one);
    absdiff = _mm256_blendv_epi8(
        absdiff, repl, _mm256_cmpgt_epi32(absdiff, ba2));
    odd = _mm256_and_si256(absdiff, one);
    {
        __m256i an2 = _mm256_slli_epi32(abs_an, 1);
        __m256i rounded = _mm256_add_epi32(absdiff, odd);
        absdiff = _mm256_blendv_epi8(
            absdiff, an2, _mm256_cmpgt_epi32(rounded, an2));
    }
    tendency = _mm256_blendv_epi8(
        absdiff, _mm256_sub_epi32(zero, absdiff),
        _mm256_cmpgt_epi32(next, top));
    {
        __m256i neq_ba =
            _mm256_xor_si256(_mm256_cmpeq_epi32(ba, zero), all);
        __m256i neq_an =
            _mm256_xor_si256(_mm256_cmpeq_epi32(an, zero), all);
        __m256i nonmono =
            _mm256_cmpgt_epi32(zero, _mm256_xor_si256(ba, an));
        __m256i skip =
            _mm256_and_si256(_mm256_and_si256(neq_ba, neq_an), nonmono);
        tendency = _mm256_blendv_epi8(tendency, zero, skip);
    }
    diff = _mm256_add_epi32(residu, tendency);
    *first_out = _mm256_add_epi32(
        avg, _mm256_srai_epi32(
                 _mm256_add_epi32(diff, _mm256_srli_epi32(diff, 31)), 1));
    *second_out = _mm256_sub_epi32(*first_out, diff);
    return 1;
}

static JXL_TARGET_AVX2 JXL_INLINE_HINT void squeeze_transpose8(__m256i p[8]) {
    __m256i t0 = _mm256_unpacklo_epi32(p[0], p[1]);
    __m256i t1 = _mm256_unpackhi_epi32(p[0], p[1]);
    __m256i t2 = _mm256_unpacklo_epi32(p[2], p[3]);
    __m256i t3 = _mm256_unpackhi_epi32(p[2], p[3]);
    __m256i t4 = _mm256_unpacklo_epi32(p[4], p[5]);
    __m256i t5 = _mm256_unpackhi_epi32(p[4], p[5]);
    __m256i t6 = _mm256_unpacklo_epi32(p[6], p[7]);
    __m256i t7 = _mm256_unpackhi_epi32(p[6], p[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2);
    __m256i u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3);
    __m256i u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6);
    __m256i u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7);
    __m256i u7 = _mm256_unpackhi_epi64(t5, t7);
    p[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
    p[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    p[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
    p[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    p[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    p[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    p[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    p[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

static JXL_TARGET_AVX2 int squeeze_inverse_h_avx2(jxl_ctx *ctx,
                                                  jxl_mchan *merged) {
    uint32_t width = merged->w, height = merged->h;
    uint32_t avg_width = div_ceil_u32(width, 2);
    uint32_t nresidu = width / 2;
    int32_t *scratch;
    uint32_t y, r;

    scratch = (int32_t *)jxl_malloc(
        ctx, (size_t)width * 8 * sizeof(int32_t));
    if (!scratch) return -1;

    for (y = 0; y + 8 <= height; y += 8) {
        uint32_t xi = 0;
        for (r = 0; r < 8; r++) {
            const int32_t *row =
                merged->data + (size_t)(y + r) * merged->stride;
            memcpy(scratch + (size_t)r * width, row,
                   (size_t)width * sizeof(int32_t));
        }

        for (; xi + 8 <= nresidu; xi += 8) {
            __m256i av[8], rv[8], even[8], odd[8];
            __m256i left, ninth;
            uint32_t i;
            uint32_t ni = xi + 8 < avg_width ? xi + 8 : xi + 7;

            for (r = 0; r < 8; r++) {
                const int32_t *srow = scratch + (size_t)r * width;
                av[r] = _mm256_loadu_si256(
                    (const __m256i *)(srow + xi));
                rv[r] = _mm256_loadu_si256(
                    (const __m256i *)(srow + avg_width + xi));
            }
            squeeze_transpose8(av);
            squeeze_transpose8(rv);
            if (xi == 0) {
                left = av[0];
            } else {
                left = _mm256_setr_epi32(
                    merged->data[(size_t)(y + 0) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 1) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 2) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 3) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 4) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 5) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 6) * merged->stride +
                                 2 * xi - 1],
                    merged->data[(size_t)(y + 7) * merged->stride +
                                 2 * xi - 1]);
            }
            ninth = _mm256_setr_epi32(
                scratch[(size_t)0 * width + ni],
                scratch[(size_t)1 * width + ni],
                scratch[(size_t)2 * width + ni],
                scratch[(size_t)3 * width + ni],
                scratch[(size_t)4 * width + ni],
                scratch[(size_t)5 * width + ni],
                scratch[(size_t)6 * width + ni],
                scratch[(size_t)7 * width + ni]);

            for (i = 0; i < 8; i++) {
                __m256i next = i < 7 ? av[i + 1] : ninth;
                if (!squeeze_step8(left, av[i], next, rv[i],
                                   &even[i], &odd[i])) {
                    int32_t lt[8], aa[8], nn[8], rr[8], ee[8], oo[8];
                    uint32_t k;
                    _mm256_storeu_si256((__m256i *)lt, left);
                    _mm256_storeu_si256((__m256i *)aa, av[i]);
                    _mm256_storeu_si256((__m256i *)nn, next);
                    _mm256_storeu_si256((__m256i *)rr, rv[i]);
                    for (k = 0; k < 8; k++) {
                        int32_t diff = (int32_t)(
                            (uint32_t)rr[k] +
                            (uint32_t)squeeze_tendency(
                                lt[k], aa[k], nn[k]));
                        ee[k] = (int32_t)(
                            (uint32_t)aa[k] + (uint32_t)(diff / 2));
                        oo[k] =
                            (int32_t)((uint32_t)ee[k] - (uint32_t)diff);
                    }
                    even[i] = _mm256_loadu_si256((const __m256i *)ee);
                    odd[i] = _mm256_loadu_si256((const __m256i *)oo);
                }
                left = odd[i];
            }

            squeeze_transpose8(even);
            squeeze_transpose8(odd);
            for (r = 0; r < 8; r++) {
                __m256i lo = _mm256_unpacklo_epi32(even[r], odd[r]);
                __m256i hi = _mm256_unpackhi_epi32(even[r], odd[r]);
                __m256i out0 =
                    _mm256_permute2x128_si256(lo, hi, 0x20);
                __m256i out1 =
                    _mm256_permute2x128_si256(lo, hi, 0x31);
                int32_t *row =
                    merged->data + (size_t)(y + r) * merged->stride + 2 * xi;
                _mm256_storeu_si256((__m256i *)row, out0);
                _mm256_storeu_si256((__m256i *)(row + 8), out1);
            }
        }

        for (r = 0; r < 8; r++) {
            const int32_t *avg_row = scratch + (size_t)r * width;
            const int32_t *residu_row = avg_row + avg_width;
            int32_t *row =
                merged->data + (size_t)(y + r) * merged->stride;
            int32_t avg = avg_row[xi];
            int32_t left = xi ? row[2 * xi - 1] : avg;
            uint32_t i;
            for (i = xi; i < nresidu; i++) {
                int32_t next_avg =
                    i + 1 < avg_width ? avg_row[i + 1] : avg;
                int32_t diff = (int32_t)(
                    (uint32_t)residu_row[i] +
                    (uint32_t)squeeze_tendency(left, avg, next_avg));
                int32_t first = (int32_t)(
                    (uint32_t)avg + (uint32_t)(diff / 2));
                int32_t second =
                    (int32_t)((uint32_t)first - (uint32_t)diff);
                row[2 * i] = first;
                row[2 * i + 1] = second;
                avg = next_avg;
                left = second;
            }
            if (width & 1) row[width - 1] = avg_row[avg_width - 1];
        }
    }

    for (; y < height; y++) {
        int32_t *row = merged->data + (size_t)y * merged->stride;
        const int32_t *avg_row, *residu_row;
        int32_t avg, left;
        uint32_t x;
        memcpy(scratch, row, (size_t)width * sizeof(int32_t));
        avg_row = scratch;
        residu_row = scratch + avg_width;
        avg = avg_row[0];
        left = avg;
        for (x = 0; x + 1 < width; x += 2) {
            uint32_t i = x / 2;
            int32_t residu = residu_row[i];
            int32_t next_avg =
                i + 1 < avg_width ? avg_row[i + 1] : avg;
            int32_t diff = (int32_t)(
                (uint32_t)residu +
                (uint32_t)squeeze_tendency(left, avg, next_avg));
            int32_t first = (int32_t)(
                (uint32_t)avg + (uint32_t)(diff / 2));
            int32_t second =
                (int32_t)((uint32_t)first - (uint32_t)diff);
            row[x] = first;
            row[x + 1] = second;
            avg = next_avg;
            left = second;
        }
        if (width & 1) row[width - 1] = avg_row[avg_width - 1];
    }

    jxl_free(ctx, scratch);
    return 0;
}

static int squeeze_inverse_h(jxl_ctx *ctx, jxl_mchan *merged) {
    uint32_t width = merged->w, height = merged->h;
    uint32_t avg_width = div_ceil_u32(width, 2);
    int32_t *scratch;
    uint32_t x, y;

    if (width == 0 || height == 0) return 0;
    if (height >= 8 && width >= 16 && jxl_has_avx2())
        return squeeze_inverse_h_avx2(ctx, merged);
    scratch = (int32_t *)jxl_malloc(ctx, (size_t)width * sizeof(int32_t));
    if (!scratch) return -1;

    for (y = 0; y < height; y++) {
        int32_t *row = merged->data + (size_t)y * merged->stride;
        const int32_t *avg_row, *residu_row;
        int32_t avg, left;
        memcpy(scratch, row, (size_t)width * sizeof(int32_t));
        avg_row = scratch;
        residu_row = scratch + avg_width;
        avg = avg_row[0];
        left = avg;
        for (x = 0; x + 1 < width; x += 2) {
            uint32_t xi = x / 2;
            int32_t residu = residu_row[xi];
            int32_t next_avg = (xi + 1 < avg_width) ? avg_row[xi + 1] : avg;
            int32_t diff = (int32_t)((uint32_t)residu +
                                     (uint32_t)squeeze_tendency(left, avg, next_avg));
            int32_t first = (int32_t)((uint32_t)avg + (uint32_t)(diff / 2));
            int32_t second = (int32_t)((uint32_t)first - (uint32_t)diff);
            row[x] = first;
            row[x + 1] = second;
            avg = next_avg;
            left = second;
        }
        if (width & 1) row[width - 1] = avg_row[avg_width - 1];
    }
    jxl_free(ctx, scratch);
    return 0;
}

#define JXL_SQ_STRIP 16

/* Eight independent columns of the vertical lifting step. This is the
   branchless form used by libjxl's FastUnsqueeze: floor(abs(B-a) / 3) is a
   multiply-high by 0x55555556, and signed diff/2 is an arithmetic shift after
   adding one for negative odd values. Keep extreme samples on the scalar
   int64 path so all int32 inputs retain squeeze_tendency's semantics. */
static JXL_TARGET_AVX2 void squeeze_inverse_v_avx2(
    jxl_mchan *merged, int32_t *scratch, uint32_t width, uint32_t height,
    uint32_t avg_height) {
    uint32_t x, y;

    for (x = 0; x < width; x += JXL_SQ_STRIP) {
        uint32_t nresidu = height / 2;
        uint32_t sw = width - x < JXL_SQ_STRIP ? width - x : JXL_SQ_STRIP;
        int32_t avg[JXL_SQ_STRIP], top[JXL_SQ_STRIP];
        uint32_t dx;

        for (y = 0; y < height; y++) {
            memcpy(scratch + (size_t)y * sw,
                   merged->data + (size_t)y * merged->stride + x,
                   (size_t)sw * sizeof(int32_t));
        }
        for (dx = 0; dx < sw; dx++) {
            avg[dx] = scratch[dx];
            top[dx] = avg[dx];
        }
        for (y = 0; y < nresidu; y++) {
            const int32_t *residu_row =
                scratch + (size_t)(avg_height + y) * sw;
            const int32_t *next_row = (y + 1 < avg_height)
                                          ? scratch + (size_t)(y + 1) * sw
                                          : NULL;
            int32_t *out0 =
                merged->data + (size_t)(2 * y) * merged->stride + x;
            int32_t *out1 = out0 + merged->stride;

            for (dx = 0; dx + 8 <= sw; dx += 8) {
                __m256i av = _mm256_loadu_si256((const __m256i *)(avg + dx));
                __m256i tv = _mm256_loadu_si256((const __m256i *)(top + dx));
                __m256i nv = next_row
                    ? _mm256_loadu_si256((const __m256i *)(next_row + dx))
                    : av;
                __m256i residu = _mm256_loadu_si256(
                    (const __m256i *)(residu_row + dx));
                __m256i first, second;

                if (squeeze_step8(tv, av, nv, residu, &first, &second)) {
                    _mm256_storeu_si256((__m256i *)(out0 + dx), first);
                    _mm256_storeu_si256((__m256i *)(out1 + dx), second);
                    _mm256_storeu_si256((__m256i *)(avg + dx), nv);
                    _mm256_storeu_si256((__m256i *)(top + dx), second);
                } else {
                    uint32_t k;
                    for (k = 0; k < 8; k++) {
                        uint32_t i = dx + k;
                        int32_t next_avg = next_row ? next_row[i] : avg[i];
                        int32_t diff = (int32_t)(
                            (uint32_t)residu_row[i] +
                            (uint32_t)squeeze_tendency(
                                top[i], avg[i], next_avg));
                        int32_t first_s = (int32_t)(
                            (uint32_t)avg[i] + (uint32_t)(diff / 2));
                        int32_t second_s =
                            (int32_t)((uint32_t)first_s - (uint32_t)diff);
                        out0[i] = first_s;
                        out1[i] = second_s;
                        avg[i] = next_avg;
                        top[i] = second_s;
                    }
                }
            }
            for (; dx < sw; dx++) {
                int32_t next_avg = next_row ? next_row[dx] : avg[dx];
                int32_t diff = (int32_t)(
                    (uint32_t)residu_row[dx] +
                    (uint32_t)squeeze_tendency(top[dx], avg[dx], next_avg));
                int32_t first = (int32_t)(
                    (uint32_t)avg[dx] + (uint32_t)(diff / 2));
                int32_t second =
                    (int32_t)((uint32_t)first - (uint32_t)diff);
                out0[dx] = first;
                out1[dx] = second;
                avg[dx] = next_avg;
                top[dx] = second;
            }
        }
        if (height & 1) {
            const int32_t *last = scratch + (size_t)(avg_height - 1) * sw;
            int32_t *out = merged->data +
                           (size_t)(height - 1) * merged->stride + x;
            memcpy(out, last, (size_t)sw * sizeof(int32_t));
        }
    }
}

static int squeeze_inverse_v(jxl_ctx *ctx, jxl_mchan *merged) {
    uint32_t width = merged->w, height = merged->h;
    uint32_t avg_height = div_ceil_u32(height, 2);
    int32_t *scratch;
    uint32_t x, y;

    /* A strip of columns at a time, not one. The lifting step down a column
       is serial -- `top` feeds the next row's tendency -- so this cannot be
       vectorised across y, but neighbouring columns are completely
       independent, and running JXL_SQ_STRIP of them in lockstep is what makes
       the memory access sane. One column at a time walked the image with a
       stride, touching a separate cache line for every row and then coming
       back for the next column once those lines had been evicted; a strip of
       16 is 64 bytes, so every line fetched is used in full, and both the
       copy in and the stores back run sequentially.

       The copy is still needed: the outputs for row 2y overwrite inputs that
       later rows still have to read. The arithmetic is untouched. */
    if (width == 0 || height == 0) return 0;
    scratch = (int32_t *)jxl_malloc(
        ctx, (size_t)height * JXL_SQ_STRIP * sizeof(int32_t));
    if (!scratch) return -1;

    if (jxl_has_avx2()) {
        squeeze_inverse_v_avx2(merged, scratch, width, height, avg_height);
        jxl_free(ctx, scratch);
        return 0;
    }

    for (x = 0; x < width; x += JXL_SQ_STRIP) {
        uint32_t nresidu = height / 2;
        uint32_t sw = width - x < JXL_SQ_STRIP ? width - x : JXL_SQ_STRIP;
        int32_t avg[JXL_SQ_STRIP], top[JXL_SQ_STRIP];
        uint32_t dx;

        for (y = 0; y < height; y++) {
            memcpy(scratch + (size_t)y * sw,
                   merged->data + (size_t)y * merged->stride + x,
                   (size_t)sw * sizeof(int32_t));
        }
        for (dx = 0; dx < sw; dx++) {
            avg[dx] = scratch[dx];
            top[dx] = avg[dx];
        }
        for (y = 0; y < nresidu; y++) {
            const int32_t *residu_row = scratch + (size_t)(avg_height + y) * sw;
            const int32_t *next_row = (y + 1 < avg_height)
                                          ? scratch + (size_t)(y + 1) * sw
                                          : NULL;
            int32_t *out0 = merged->data + (size_t)(2 * y) * merged->stride + x;
            int32_t *out1 = out0 + merged->stride;
            for (dx = 0; dx < sw; dx++) {
                int32_t next_avg = next_row ? next_row[dx] : avg[dx];
                int32_t diff = (int32_t)((uint32_t)residu_row[dx] +
                        (uint32_t)squeeze_tendency(top[dx], avg[dx], next_avg));
                int32_t first = (int32_t)((uint32_t)avg[dx] +
                                          (uint32_t)(diff / 2));
                int32_t second = (int32_t)((uint32_t)first - (uint32_t)diff);
                out0[dx] = first;
                out1[dx] = second;
                avg[dx] = next_avg;
                top[dx] = second;
            }
        }
        if (height & 1) {
            const int32_t *last = scratch + (size_t)(avg_height - 1) * sw;
            int32_t *out = merged->data +
                           (size_t)(height - 1) * merged->stride + x;
            for (dx = 0; dx < sw; dx++) out[dx] = last[dx];
        }
    }
    jxl_free(ctx, scratch);
    return 0;
}

#undef JXL_SQ_STRIP

static const int16_t delta_palette[72][3] = {
    {0,0,0},{4,4,4},{11,0,0},{0,0,-13},{0,-12,0},{-10,-10,-10},
    {-18,-18,-18},{-27,-27,-27},{-18,-18,0},{0,0,-32},{-32,0,0},{-37,-37,-37},
    {0,-32,-32},{24,24,45},{50,50,50},{-45,-24,-24},{-24,-45,-45},{0,-24,-24},
    {-34,-34,0},{-24,0,-24},{-45,-45,-24},{64,64,64},{-32,0,-32},{0,-32,0},
    {-32,0,32},{-24,-45,-24},{45,24,45},{24,-24,-45},{-45,-24,24},{80,80,80},
    {64,0,0},{0,0,-64},{0,-64,-64},{-24,-24,45},{96,96,96},{64,64,0},
    {45,-24,-24},{34,-34,0},{112,112,112},{24,-45,-45},{45,45,-24},{0,-32,32},
    {24,-24,45},{0,96,96},{45,-24,24},{24,-45,-24},{-24,-45,24},{0,-64,0},
    {96,0,0},{128,128,128},{64,0,64},{144,144,144},{96,96,0},{-36,-36,36},
    {45,-24,-45},{45,-45,-24},{0,0,-96},{0,128,128},{0,96,0},{45,24,-45},
    {-128,0,0},{24,-45,24},{-45,24,-45},{64,0,-64},{64,-64,-64},{96,0,96},
    {45,-45,24},{24,45,-45},{64,64,-64},{128,128,0},{0,0,-128},{-24,45,-45}
};

static int palette_inverse(jxl_ctx *ctx, jxl_transform *tr, jxl_chanlist *cl,
                           uint32_t bit_depth, const jxl_wp_header *wp_header) {
    uint32_t num_c = tr->num_c;
    jxl_mchan pal;
    jxl_mchan *targets = NULL;
    uint32_t begin, x, y, c, i;
    uint32_t width, height;
    int32_t nb_colours = (int32_t)tr->nb_colours;
    int32_t nb_deltas = (int32_t)tr->nb_deltas;
    uint8_t *need_delta = NULL;
    int any_delta = 0;
    int rc = -1;

    if (cl->n == 0) return -1;
    pal = cl->chans[0];
    chanlist_remove_range(cl, 0, 1);
    begin = tr->begin_c;
    if (begin >= cl->n) return -1;

    targets = (jxl_mchan *)jxl_calloc(ctx, num_c, sizeof(jxl_mchan));
    if (!targets) return -1;
    targets[0] = cl->chans[begin];
    for (i = 0; i + 1 < num_c; i++) {
        if (i >= tr->nsaved) goto done;
        targets[i + 1] = tr->saved[i];
    }
    width = targets[0].w;
    height = targets[0].h;

    need_delta = (uint8_t *)jxl_calloc(ctx, (size_t)width * height + 1, 1);
    if (!need_delta) goto done;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int32_t index = chan_get(&targets[0], x, y);
            if (index < nb_deltas) {
                need_delta[(size_t)y * width + x] = 1;
                any_delta = 1;
            }
            if (index >= 0 && index < nb_colours) {
                for (c = 0; c < num_c; c++) {
                    targets[c].data[(size_t)y * targets[c].stride + x] =
                        chan_get(&pal, (uint32_t)index, c);
                }
            } else if (index >= nb_colours) {
                int32_t idx = index - nb_colours;
                if (idx < 64) {
                    for (c = 0; c < num_c; c++) {
                        int32_t v = ((idx >> (2 * c)) % 4) *
                                        ((int32_t)(1u << bit_depth) - 1) / 4 +
                                    (int32_t)(1u << (bit_depth >= 3 ? bit_depth - 3 : 0));
                        targets[c].data[(size_t)y * targets[c].stride + x] = v;
                    }
                } else {
                    int32_t k = idx - 64;
                    for (c = 0; c < num_c; c++) {
                        targets[c].data[(size_t)y * targets[c].stride + x] =
                            (k % 5) * ((int32_t)(1u << bit_depth) - 1) / 4;
                        k /= 5;
                    }
                }
            } else {
                for (c = 0; c < num_c; c++) {
                    int32_t v;
                    int32_t di;
                    if (c >= 3) {
                        targets[c].data[(size_t)y * targets[c].stride + x] = 0;
                        continue;
                    }
                    di = -(index + 1);
                    di = di % 143;
                    v = delta_palette[(di + 1) >> 1][c];
                    if ((di & 1) == 0) v = -v;
                    if (bit_depth > 8) {
                        uint32_t sh = bit_depth < 24 ? bit_depth : 24;
                        v = (int32_t)((uint32_t)v << (sh - 8));
                    }
                    targets[c].data[(size_t)y * targets[c].stride + x] = v;
                }
            }
        }
    }

    if (any_delta) {
        jxl_pred_state ps;
        memset(&ps, 0, sizeof(ps));
        for (c = 0; c < num_c; c++) {
            const jxl_wp_header *wp =
                (tr->d_pred == JXL_PRED_SELF_CORRECTING) ? wp_header : NULL;
            if (pred_state_reset(ctx, &ps, width, wp, NULL, 0) != 0) {
                pred_state_free(ctx, &ps);
                goto done;
            }
            for (y = 0; y < height; y++) {
                for (x = 0; x < width; x++) {
                    jxl_props pr;
                    int32_t *sample = &targets[c].data[(size_t)y * targets[c].stride + x];
                    int32_t value = *sample;
                    /* No tree walk here, so the static properties are unused. */
                    props_compute(&ps, &pr, 0, 0);
                    if (need_delta[(size_t)y * width + x]) {
                        int32_t diff = predict_sample(&ps, &pr, tr->d_pred);
                        value = (int32_t)((uint32_t)value + (uint32_t)diff);
                        *sample = value;
                    }
                    pred_record(&ps, &pr, value);
                }
            }
        }
        pred_state_free(ctx, &ps);
    }

    /* Put the member channels back into the list. */
    for (i = 0; i + 1 < num_c; i++) {
        if (chanlist_insert(ctx, cl, begin + 1 + i, &targets[i + 1]) != 0) goto done;
    }
    cl->chans[begin] = targets[0];
    rc = 0;

done:
    jxl_free(ctx, need_delta);
    jxl_free(ctx, targets);
    return rc;
}

static int squeeze_inverse(jxl_ctx *ctx, jxl_transform *tr, jxl_chanlist *cl) {
    uint32_t s = tr->nsp;
    while (s > 0) {
        const jxl_sq_param *sp = &tr->sp[--s];
        uint32_t begin = sp->begin_c;
        uint32_t count = sp->num_c;
        uint32_t end = begin + count;
        jxl_mchan *residual;
        uint32_t from, i;

        if (end > cl->n) return -1;
        if (sp->in_place) {
            if (end + count > cl->n) return -1;
            from = end;
        } else {
            if (cl->n < count) return -1;
            from = cl->n - count;
        }
        residual = (jxl_mchan *)jxl_calloc(ctx, count, sizeof(jxl_mchan));
        if (!residual) return -1;
        memcpy(residual, cl->chans + from, (size_t)count * sizeof(jxl_mchan));
        chanlist_remove_range(cl, from, from + count);

        for (i = 0; i < count; i++) {
            jxl_mchan *ch = &cl->chans[begin + i];
            if (sp->horizontal) {
                ch->w += residual[i].w;
                if (ch->hshift > 0) ch->hshift--;
                if (squeeze_inverse_h(ctx, ch) != 0) {
                    jxl_free(ctx, residual);
                    return -1;
                }
            } else {
                ch->h += residual[i].h;
                if (ch->vshift > 0) ch->vshift--;
                if (squeeze_inverse_v(ctx, ch) != 0) {
                    jxl_free(ctx, residual);
                    return -1;
                }
            }
        }
        jxl_free(ctx, residual);
    }
    return 0;
}

int jxl_modular_inverse(jxl_ctx *ctx, jxl_modular *m, jxl_chanlist *cl) {
    uint32_t i = m->header.ntransforms;
    while (i > 0) {
        jxl_transform *tr = &m->header.transforms[--i];
        if (tr->kind == 0) {
            if (tr->begin_c + 3 > cl->n) return -1;
            rct_inverse(tr, cl);
        } else if (tr->kind == 1) {
            if (palette_inverse(ctx, tr, cl, m->bit_depth, &m->header.wp) != 0)
                return -1;
        } else {
            if (squeeze_inverse(ctx, tr, cl) != 0) return -1;
        }
    }
    return 0;
}

/* ===================================================================== */
/* modular image                                                          */
/* ===================================================================== */

static void shift_size(int hshift, int vshift, uint32_t w, uint32_t h,
                       uint32_t *ow, uint32_t *oh) {
    uint32_t sh = hshift > 0 ? (uint32_t)hshift : 0;
    uint32_t sv = vshift > 0 ? (uint32_t)vshift : 0;
    *ow = (w + ((1u << sh) - 1)) >> sh;
    *oh = (h + ((1u << sv) - 1)) >> sv;
}

/* Shared tail of the two init entry points: reads the modular header,
   applies the transforms to a geometry-only channel list, and reads the
   local MA tree when the stream does not use the global one. */
static int modular_init_common(jxl_ctx *ctx, jxl_modular *m, jxl_br *br,
                               jxl_ma_config *global_ma) {
    jxl_chanlist cl;
    uint32_t i;

    memset(&cl, 0, sizeof(cl));
    m->header.use_global_tree = jxl_br_bool(br);
    wp_header_read(br, &m->header.wp);
    m->header.ntransforms = jxl_br_u32(br, 0, 0, 1, 0, 2, 4, 18, 8);
    if (m->header.ntransforms > 512) {
        JXL_ERR(ctx, "modular: too many transforms");
        return -1;
    }
    if (m->header.ntransforms) {
        m->header.transforms = (jxl_transform *)jxl_calloc(
            ctx, m->header.ntransforms, sizeof(jxl_transform));
        if (!m->header.transforms) return -1;
    }
    for (i = 0; i < m->header.ntransforms; i++) {
        if (transform_read(ctx, br, &m->header.transforms[i]) != 0) return -1;
    }
    if (br->err) {
        JXL_ERR(ctx, "modular: truncated header");
        return -1;
    }

    for (i = 0; i < m->nbase; i++) {
        if (jxl_chanlist_push(ctx, &cl, &m->base[i]) != 0) {
            jxl_chanlist_free(ctx, &cl);
            return -1;
        }
    }
    for (i = 0; i < m->header.ntransforms; i++) {
        if (transform_apply(ctx, &m->header.transforms[i], &cl, 0) != 0) {
            jxl_chanlist_free(ctx, &cl);
            return -1;
        }
    }
    if (cl.n > (1u << 16)) {
        JXL_ERR(ctx, "modular: too many channels after transforms");
        jxl_chanlist_free(ctx, &cl);
        return -1;
    }

    if (m->header.use_global_tree) {
        if (!global_ma || !global_ma->valid) {
            JXL_ERR(ctx, "modular: global MA tree not available");
            jxl_chanlist_free(ctx, &cl);
            return -1;
        }
        m->ma = global_ma;
    } else {
        uint64_t samples = 0;
        size_t limit;
        for (i = 0; i < cl.n; i++) {
            samples += (uint64_t)cl.chans[i].w * cl.chans[i].h;
        }
        limit = (size_t)JXL_MIN(1024 + samples, (uint64_t)1 << 20);
        if (jxl_ma_config_read(ctx, br, &m->local, limit) != 0) {
            jxl_chanlist_free(ctx, &cl);
            return -1;
        }
        m->has_local = 1;
        m->ma = &m->local;
    }
    jxl_chanlist_free(ctx, &cl);
    return 0;
}

int jxl_modular_init(jxl_ctx *ctx, jxl_modular *m, jxl_br *br,
                     const jxl_mchan_spec *specs, uint32_t nspecs,
                     jxl_ma_config *global_ma, uint32_t group_dim,
                     uint32_t bit_depth) {
    uint32_t i;

    memset(m, 0, sizeof(*m));
    m->ctx = ctx;
    m->group_dim = group_dim;
    m->bit_depth = bit_depth;
    if (nspecs == 0) return 0;

    m->base = (jxl_mchan *)jxl_calloc(ctx, nspecs, sizeof(jxl_mchan));
    m->bufs = (int32_t **)jxl_calloc(ctx, nspecs, sizeof(int32_t *));
    if (!m->base || !m->bufs) return -1;
    m->nbase = nspecs;
    m->nbufs = nspecs;
    for (i = 0; i < nspecs; i++) {
        uint32_t w, h;
        size_t total;
        shift_size(specs[i].hshift, specs[i].vshift, specs[i].w, specs[i].h, &w, &h);
        m->base[i].w = w;
        m->base[i].h = h;
        m->base[i].ow = specs[i].w;
        m->base[i].oh = specs[i].h;
        m->base[i].hshift = specs[i].hshift;
        m->base[i].vshift = specs[i].vshift;
        m->base[i].stride = w;
        if (!jxl_size_mul(w, h, &total)) return -1;
        m->bufs[i] = (int32_t *)jxl_calloc(ctx, total ? total : 1, sizeof(int32_t));
        if (!m->bufs[i]) return -1;
        m->base[i].data = m->bufs[i];
    }
    return modular_init_common(ctx, m, br, global_ma);
}

/* Like jxl_modular_init but over channels that already exist (views into
   another image's buffers) -- the "recursive" modular stream of a group. */
int jxl_modular_init_over(jxl_ctx *ctx, jxl_modular *m, jxl_br *br,
                          const jxl_mchan *chans, uint32_t nchans,
                          jxl_ma_config *global_ma, uint32_t group_dim,
                          uint32_t bit_depth) {
    memset(m, 0, sizeof(*m));
    m->ctx = ctx;
    m->group_dim = group_dim;
    m->bit_depth = bit_depth;
    if (nchans == 0) return 0;
    m->base = (jxl_mchan *)jxl_calloc(ctx, nchans, sizeof(jxl_mchan));
    if (!m->base) return -1;
    memcpy(m->base, chans, (size_t)nchans * sizeof(jxl_mchan));
    m->nbase = nchans;
    m->nbufs = 0;
    return modular_init_common(ctx, m, br, global_ma);
}

void jxl_modular_free(jxl_ctx *ctx, jxl_modular *m) {
    uint32_t i;
    if (!m || !m->ctx) return;
    for (i = 0; i < m->header.ntransforms; i++) {
        jxl_free(ctx, m->header.transforms[i].sp);
        jxl_free(ctx, m->header.transforms[i].saved);
        jxl_free(ctx, m->header.transforms[i].pal_buf);
    }
    jxl_free(ctx, m->header.transforms);
    for (i = 0; i < m->nbufs; i++) jxl_free(ctx, m->bufs[i]);
    jxl_free(ctx, m->bufs);
    jxl_free(ctx, m->base);
    if (m->has_local) jxl_ma_config_free(ctx, &m->local);
    memset(m, 0, sizeof(*m));
}

int jxl_modular_transform_channels(jxl_ctx *ctx, jxl_modular *m,
                                   jxl_chanlist *cl) {
    uint32_t i;
    memset(cl, 0, sizeof(*cl));
    for (i = 0; i < m->nbase; i++) {
        if (jxl_chanlist_push(ctx, cl, &m->base[i]) != 0) return -1;
    }
    for (i = 0; i < m->header.ntransforms; i++) {
        if (transform_apply(ctx, &m->header.transforms[i], cl, 1) != 0) return -1;
    }
    return 0;
}

/* ===================================================================== */
/* sample decoding                                                        */
/* ===================================================================== */

/* Walks the decision tree to the leaf governing this sample. Every property
   is read the same way, from cache[] or from the previous-channel accessor:
   the two static properties are no longer special-cased here because
   props_compute stores them in cache[0..1] like the rest. The walk always
   terminates -- jxl_ma_config_read rejects any tree whose child indices are
   not strictly greater than the parent's -- so this returns a real leaf,
   never a decision node abandoned part-way down. */
static const jxl_ma_leaf *ma_get_leaf(const jxl_ma_config *ma,
                                      const jxl_pred_state *ps,
                                      const jxl_props *pr) {
    const jxl_ma_flat *flat = ma->flat;
    const jxl_ma_flat *f = flat;
    while (f->property >= 0) {
        int32_t p0 = f->property, p1 = f->u.dec.prop1, p2 = f->u.dec.prop2;
        int32_t v0, v1, v2;
        uint32_t o0, o1;

        /* All three tests are read up front rather than picking the second
           one after the first has decided. The three loads are independent,
           so they issue together; selecting first would make the second
           address depend on the first result and serialise them. */
        v0 = p0 < 16 ? pr->cache[p0] : props_get_extra(ps, (uint32_t)p0 - 16);
        v1 = p1 < 16 ? pr->cache[p1] : props_get_extra(ps, (uint32_t)p1 - 16);
        v2 = p2 < 16 ? pr->cache[p2] : props_get_extra(ps, (uint32_t)p2 - 16);
        o0 = v0 > f->u.dec.split0 ? 0u : 1u;
        o1 = o0 ? (v2 > f->u.dec.split2 ? 0u : 1u)
                : (v1 > f->u.dec.split1 ? 0u : 1u);
        f = &flat[f->u.dec.child + 2 * o0 + o1];
    }
    return &f->u.leaf;
}

/* True when the tree can ask for the self-correcting predictor or property
   15, the only cases where running it is needed. */
/* Bitmask of the properties any decision node in the tree actually tests.
   libjxl specialises its inner loop on exactly this (encoding.cc's
   is_wp_only / is_gradient_only tracks); the cheapest case to exploit is a
   tree that only tests properties 0 and 1, channel index and stream index,
   which are constant for a whole channel -- so the leaf is too, and both the
   property fill and the tree walk drop out of the per-sample loop.
   Properties >= 32 cannot be represented, and set every bit so the caller
   takes the general path. */
static uint32_t ma_props_mask(const jxl_ma_config *ma) {
    uint32_t i, mask = 0;
    for (i = 0; i < ma->nflat; i++) {
        const jxl_ma_flat *f = &ma->flat[i];
        uint32_t p[3];
        int k;
        if (f->property < 0) continue;
        p[0] = (uint32_t)f->property;
        p[1] = f->u.dec.prop1;
        p[2] = f->u.dec.prop2;
        for (k = 0; k < 3; k++) {
            if (p[k] >= 32) return 0xffffffffu;
            mask |= 1u << p[k];
        }
    }
    return mask;
}

/* Whether any real decision node tests the channel or the stream index --
   the two properties a single channel already knows the answer to. This asks
   the binary tree rather than ma_props_mask because the flat form pads the
   slot of a child that is a leaf with property 0 and an unreachable split, so
   bit 0 of that mask is set on almost every tree whether or not anything
   tests the channel. Folding nothing would still cost a rebuild per channel,
   so the distinction matters. */
static int ma_tests_const_props(const jxl_ma_config *ma) {
    uint32_t i;
    for (i = 0; i < ma->nraw; i++) {
        if (ma->raw[i].property == 0 || ma->raw[i].property == 1) return 1;
    }
    return 0;
}

/* libjxl's HuffRleOnly shortcut shares one distance-one run across channels.
   That is safe only when every channel is guaranteed to take a fixed,
   identity gradient leaf after the channel/stream properties are folded. */
static int ma_all_gradient_noop(const jxl_ma_config *ma) {
    uint32_t i;
    for (i = 0; i < ma->nraw; i++) {
        const jxl_ma_node *n = &ma->raw[i];
        if (n->property >= 0) {
            if (n->property > 1) return 0;
        } else {
            const jxl_ma_leaf *leaf = &ma->leaves[n->child];
            if (leaf->predictor != JXL_PRED_GRADIENT ||
                leaf->offset != 0 || leaf->multiplier != 1)
                return 0;
        }
    }
    return 1;
}

/* The weighted-predictor error (property 15) is unbounded in principle, so a
   lookup table over it is only valid if clamping the index cannot change any
   decision. The walk tests `v > split`, so for a value above the range,
   clamping to HI agrees with the true answer when split < HI; below the
   range, clamping to LO agrees when split >= LO. Checking every split tested
   against property 15 against that is what makes the table safe -- libjxl's
   TreeToLookupTable bails out on the same condition. */
#define JXL_WP_LUT_LO (-8192)
#define JXL_WP_LUT_HI (8191)
#define JXL_WP_LUT_N  (JXL_WP_LUT_HI - JXL_WP_LUT_LO + 1)

static int ma_wp_lut_ok(const jxl_ma_config *ma) {
    uint32_t i;
    for (i = 0; i < ma->nflat; i++) {
        const jxl_ma_flat *f = &ma->flat[i];
        if (f->property < 0) continue;
        if (f->property == 15 &&
            !(f->u.dec.split0 >= JXL_WP_LUT_LO && f->u.dec.split0 < JXL_WP_LUT_HI))
            return 0;
        if (f->u.dec.prop1 == 15 &&
            !(f->u.dec.split1 >= JXL_WP_LUT_LO && f->u.dec.split1 < JXL_WP_LUT_HI))
            return 0;
        if (f->u.dec.prop2 == 15 &&
            !(f->u.dec.split2 >= JXL_WP_LUT_LO && f->u.dec.split2 < JXL_WP_LUT_HI))
            return 0;
    }
    return 1;
}

static int ma_needs_self_correcting(const jxl_ma_config *ma) {
    uint32_t i;
    for (i = 0; i < ma->nflat; i++) {
        const jxl_ma_flat *f = &ma->flat[i];
        if (f->property < 0) {
            if (f->u.leaf.predictor == JXL_PRED_SELF_CORRECTING) return 1;
        } else if (f->property == 15 || f->u.dec.prop1 == 15 ||
                   f->u.dec.prop2 == 15) {
            return 1;
        }
    }
    return 0;
}

/* Deepest previous-channel index any property in the tree can reach. Every
   test lives in some entry's property/prop1/prop2, and the filler property a
   leaf child gets is 0, so it never contributes. */
static uint32_t ma_max_prev_channels(const jxl_ma_config *ma) {
    uint32_t i, max = 0;
    for (i = 0; i < ma->nflat; i++) {
        const jxl_ma_flat *f = &ma->flat[i];
        int32_t props[3];
        int k;

        if (f->property < 0) continue;
        props[0] = f->property;
        props[1] = f->u.dec.prop1;
        props[2] = f->u.dec.prop2;
        for (k = 0; k < 3; k++) {
            if (props[k] >= 16) {
                uint32_t d = ((uint32_t)props[k] - 16) / 4 + 1;
                if (d > max) max = d;
            }
        }
    }
    return max;
}

int jxl_modular_decode(jxl_ctx *ctx, jxl_modular *m, jxl_chanlist *cl,
                       jxl_br *br, uint32_t stream_idx) {
    jxl_ma_config *ma = m->ma;
    jxl_dec *dec;
    uint32_t dist_multiplier = 0;
    uint32_t i, ci;
    jxl_pred_state ps;
    const jxl_mchan **prev = NULL;
    uint32_t max_prev;
    int need_sc;
    uint32_t props_mask;
    int wp_lut_usable = 0;
    int can_fold;
    int wp_lut_ready = 0;   /* table contents already valid                 */
    int wp_lut_local = 0;   /* we allocated it and still own it             */
    const jxl_ma_leaf **wp_lut = NULL;
    jxl_ma_config spec;
    int spec_ok = 0;
    int rle1_fast;
    uint32_t rle1_value = 0, rle1_run = 0;
    int32_t rle1_diff = 0;
    int rle1_have = 0;
    int rc = -1;

    memset(&spec, 0, sizeof(spec));

    if (!ma || !ma->valid) {
        JXL_ERR(ctx, "modular: no MA tree");
        return -1;
    }
    if (cl->n == 0) return 0;

    dec = &ma->dec;
    for (i = 0; i < cl->n; i++) {
        if (cl->chans[i].w > dist_multiplier) dist_multiplier = cl->chans[i].w;
    }
    jxl_dec_begin(dec, br);
    rle1_fast = jxl_dec_is_prefix_rle1(dec) && ma_all_gradient_noop(ma);

    memset(&ps, 0, sizeof(ps));
    need_sc = ma_needs_self_correcting(ma);
    props_mask = ma_props_mask(ma);
    /* Tree tests nothing but channel, stream and the WP error: the leaf is a
       function of one clamped value, so it can be tabulated per channel. */
    wp_lut_usable = need_sc && (props_mask & ~(1u | 2u | 0x8000u)) == 0 &&
                    ma_wp_lut_ok(ma);
    can_fold = ma->raw && ma_tests_const_props(ma);
    if (wp_lut_usable) {
        /* When no real node tests the channel or the stream index -- which is
           exactly what can_fold reports -- the table is a function of the WP
           error alone, so it is the same table for every channel of every
           stream. Building it costs JXL_WP_LUT_N walks, and a VarDCT frame
           with an alpha channel decodes that channel once per group: 56
           streams on flower_alpha.v_e3, which was 56 rebuilds of an identical
           table. Cache it on the tree instead.

           Only safe in that case, and for a second reason as well: with
           can_fold set, each channel walks its own specialised tree whose
           storage is freed at the end of the channel, so the leaf pointers a
           cached table held would dangle. */
        if (!can_fold && ma->wp_lut) {
            wp_lut = ma->wp_lut;
            wp_lut_ready = 1;
        } else {
            wp_lut = (const jxl_ma_leaf **)jxl_calloc(ctx, JXL_WP_LUT_N,
                                                      sizeof(*wp_lut));
            if (!wp_lut) goto done;
            wp_lut_local = 1;
        }
    }
    /* Which specialisation a stream lands on, and how big its tree is. The
       two tracks below only pay when they trigger, and whether they trigger
       is a property of the encoder settings, not of the image -- so when a
       preset turns out slow this is the first thing worth knowing. Sibling
       of JXL_DEBUG_TREE above. */
    if (getenv("JXL_DEBUG_TRACK")) {
        fprintf(stderr, "stream %u: fold=%d mask=0x%x need_sc=%d fixed=%d wplut=%d "
                "nflat=%u nchan=%u w=%u h=%u\n", (unsigned)stream_idx, can_fold,
                (unsigned)props_mask, need_sc,
                (!need_sc && (props_mask & ~3u) == 0), wp_lut_usable,
                (unsigned)ma->nflat, (unsigned)cl->n,
                (unsigned)cl->chans[0].w, (unsigned)cl->chans[0].h);
    }
    max_prev = ma_max_prev_channels(ma);
    if (max_prev) {
        prev = (const jxl_mchan **)jxl_calloc(ctx, cl->n, sizeof(jxl_mchan *));
        if (!prev) return -1;
    }

    for (ci = 0; ci < cl->n; ci++) {
        jxl_mchan *ch = &cl->chans[ci];
        const jxl_ma_config *cma;
        const jxl_ma_leaf *fixed = NULL;
        jxl_props pr0;
        uint32_t nprev = 0;
        uint32_t x, y;
        int channel_need_sc;

        /* Last channel's specialised tree. Freeing here rather than at the
           end of the body means the tracks below can keep `continue`-ing out
           without each having to remember to do it. */
        jxl_free(ctx, spec.flat);
        spec.flat = NULL;
        spec.nflat = 0;
        spec_ok = 0;

        if (ch->w == 0 || ch->h == 0) continue;

        /* Properties 0 and 1 are fixed for the whole of this channel, so
           every test on them has an answer already. Rebuilding the flat tree
           with those folded out shortens every walk this channel will make.
           On lm_d1 that is most of the tree: its main stream averages 12
           entries per walk, 20 of whose ~24 tests are on the channel index.
           The rebuild is O(tree), paid once per channel against a walk per
           sample, so it only wants a channel with more samples than the tree
           has entries. A rebuild that fails to allocate is not an error --
           the general tree is still there to walk. */
        if (can_fold && (uint64_t)ch->w * ch->h >= ma->nflat &&
            ma_flatten(ctx, ma->raw, ma->nraw, ma->root, ma->leaves, 1,
                       (int32_t)ci, (int32_t)stream_idx,
                       &spec.flat, &spec.nflat) == 0) {
            spec_ok = 1;
        }
        cma = spec_ok ? &spec : ma;

        /* Folding can remove every dynamic property test from this channel
           even when another branch of the original tree needed WP. Resolve
           that leaf before allocating predictor state, so a non-WP leaf does
           not pay to zero five error rows it will never read. */
        memset(&pr0, 0, sizeof(pr0));
        if (cma->flat[0].property < 0 ||
            (!need_sc && !spec_ok && (props_mask & ~3u) == 0)) {
            pr0.cache[0] = (int32_t)ci;
            pr0.cache[1] = (int32_t)stream_idx;
            fixed = ma_get_leaf(cma, &ps, &pr0);
            if (fixed->predictor == JXL_PRED_SELF_CORRECTING) fixed = NULL;
        }
        channel_need_sc = need_sc && !fixed;

        if (max_prev) {
            /* Previously decoded channels with identical geometry, newest
               first -- what properties >= 16 address. */
            uint32_t k = ci;
            while (k > 0 && nprev < max_prev) {
                const jxl_mchan *p = &cl->chans[--k];
                if (p->w == ch->w && p->h == ch->h && p->hshift == ch->hshift &&
                    p->vshift == ch->vshift) {
                    prev[nprev++] = p;
                }
            }
        }

        if (pred_state_reset(ctx, &ps, ch->w,
                             channel_need_sc ? &m->header.wp : NULL,
                             prev, nprev) != 0)
            goto done;

        /* Fast track: leaf as a function of the clamped WP error alone.
           Building the table costs JXL_WP_LUT_N walks, so it only pays on a
           channel with many more samples than that. sc_predict still runs --
           the weighted predictor's state is sequential and both decoders have
           to do it -- but the tree walk and fifteen of the sixteen property
           slots drop out of the loop. */
        if (!fixed && wp_lut &&
            (uint64_t)ch->w * ch->h >= 4 * JXL_WP_LUT_N) {
            jxl_props pr;
            uint32_t v;
            memset(&pr, 0, sizeof(pr));
            pr.cache[0] = (int32_t)ci;
            pr.cache[1] = (int32_t)stream_idx;
            if (!wp_lut_ready) {
                for (v = 0; v < JXL_WP_LUT_N; v++) {
                    pr.cache[15] = JXL_WP_LUT_LO + (int32_t)v;
                    wp_lut[v] = ma_get_leaf(cma, &ps, &pr);
                }
                if (!can_fold) {
                    /* Channel-independent, and built from ma->flat, which
                       outlives this call -- hand it to the tree to keep. */
                    ma->wp_lut = wp_lut;
                    wp_lut_local = 0;
                    wp_lut_ready = 1;
                }
            }
            pr.has_sc = 1;
            for (y = 0; y < ch->h; y++) {
                int32_t *row = ch->data + (size_t)y * ch->stride;
                for (x = 0; x < ch->w; x++) {
                    const jxl_ma_leaf *leaf;
                    uint32_t token;
                    int32_t diff, value, me;
                    sc_predict(&ps.sc, ps.n, ps.nw, pred_ne(&ps), ps.w,
                               pred_nn(&ps), &pr.sc);
                    me = pr.sc.max_error;
                    if (me < JXL_WP_LUT_LO) me = JXL_WP_LUT_LO;
                    else if (me > JXL_WP_LUT_HI) me = JXL_WP_LUT_HI;
                    leaf = wp_lut[me - JXL_WP_LUT_LO];
                    token = jxl_dec_read_clustered(dec, br, leaf->cluster,
                                                   dist_multiplier);
                    diff = jxl_unpack_signed(token);
                    diff = (int32_t)((uint32_t)diff * leaf->multiplier +
                                     (uint32_t)leaf->offset);
                    value = (int32_t)((uint32_t)diff +
                            (uint32_t)predict_sample(&ps, &pr, leaf->predictor));
                    row[x] = value;
                    pred_record(&ps, &pr, value);
                }
                if (br->err || dec->err) {
                    JXL_ERR(ctx, "modular: truncated stream %u",
                            (unsigned)stream_idx);
                    goto done;
                }
            }
            continue;
        }

        /* Fast track: the leaf is the same for every sample in the channel,
           so resolve it once here instead of walking the tree a million
           times. A folded tree whose root is already a leaf no longer needs
           the weighted predictor even when some other channel's branch of
           the original tree did: there is no decision left that can read its
           error property. Only a self-correcting leaf itself still needs it.
           libjxl specialises the same way.

           Once the constant tests have been folded out, "the tree only tests
           channel and stream index" is just "what is left is a single leaf",
           and that catches strictly more: a tree that tests real properties
           somewhere can still collapse to a leaf down the branch this
           particular channel takes. Most of lm_d1's channels do exactly that.
           The unspecialised condition stays for the channels too small to
           have been rebuilt. */
        {
            if (fixed) {
                uint32_t cluster = fixed->cluster;
                uint32_t mult = fixed->multiplier;
                int32_t off = fixed->offset;
                uint8_t predictor = fixed->predictor;

                /* The fjxl effort-one stream is prefix-coded LZ77 with a
                   constant distance of one. Its repeats are just runs of the
                   preceding residual, so carrying that residual and the run
                   count directly avoids allocating and touching the generic
                   4 MB LZ window. The all-gradient/noop guard above ensures
                   every channel in this stream reaches this loop, allowing a
                   run to continue across channel boundaries exactly as the
                   entropy stream specifies. */
                if (rle1_fast) {
                    int32_t *row = ch->data;

                    /* The gradient degenerates to west on the first row.
                       Pull it and the first column out of the general loop
                       so interior samples carry no x/y edge branches. */
                    for (x = 0; x < ch->w; x++) {
                        int32_t guess = x ? row[x - 1] : 0;
                        if (rle1_run) {
                            rle1_run--;
                        } else {
                            uint32_t token = jxl_dec_read_prefix_rle1(
                                dec, br, cluster, &rle1_value, &rle1_run,
                                &rle1_have);
                            rle1_diff = jxl_unpack_signed(token);
                        }
                        row[x] = (int32_t)((uint32_t)rle1_diff +
                                           (uint32_t)guess);
                    }
                    if (br->err || dec->err) {
                        JXL_ERR(ctx, "modular: truncated stream %u",
                                (unsigned)stream_idx);
                        goto done;
                    }
                    for (y = 1; y < ch->h; y++) {
                        const int32_t *rtop;
                        row = ch->data + (size_t)y * ch->stride;
                        rtop = row - ch->stride;
                        if (rle1_run) {
                            rle1_run--;
                        } else {
                            uint32_t token = jxl_dec_read_prefix_rle1(
                                dec, br, cluster, &rle1_value, &rle1_run,
                                &rle1_have);
                            rle1_diff = jxl_unpack_signed(token);
                        }
                        row[0] = (int32_t)((uint32_t)rle1_diff +
                                           (uint32_t)rtop[0]);
                        for (x = 1; x < ch->w; x++) {
                            if (rle1_run) {
                                rle1_run--;
                            } else {
                                uint32_t token = jxl_dec_read_prefix_rle1(
                                    dec, br, cluster, &rle1_value, &rle1_run,
                                    &rle1_have);
                                rle1_diff = jxl_unpack_signed(token);
                            }
                            row[x] = (int32_t)((uint32_t)rle1_diff +
                                (uint32_t)grad_clamped(
                                    rtop[x], row[x - 1], rtop[x - 1]));
                        }
                        if (br->err || dec->err) {
                            JXL_ERR(ctx, "modular: truncated stream %u",
                                    (unsigned)stream_idx);
                            goto done;
                        }
                    }
                    continue;
                }

/* With the leaf fixed, what is left per sample is one entropy read and one
   predictor. Running that through predict_sample and pred_record costs a
   switch on a runtime predictor plus a state machine that maintains n/w/nw
   and a pair of scratch rows -- for values that are already sitting in the
   channel. The scratch rows hold exactly what the channel rows hold (both are
   written from the same `value`), and row y+1 is not touched until row y is
   finished, so the neighbours can simply be indexed out of the channel.

   Its edge cases have to be reproduced exactly. pred_record leaves n = w = nw
   equal to the previous sample on row 0, and equal to prev_row[0] at the
   start of every later row; that is the same convention libjxl's fast tracks
   use, so `left` falls back to top, and `top`/`topleft` fall back to `left`.

   Hoisting the predictor out of the loop is what libjxl does too -- its
   "Gradient very fast track" is this loop with the guess spelled out.
   Predictors reaching further than n/w/nw keep the general path below. */
#define JXL_FIXED_LOOP(GUESS)                                                 \
    for (y = 0; y < ch->h; y++) {                                             \
        int32_t *row = ch->data + (size_t)y * ch->stride;                     \
        const int32_t *rtop = y ? row - ch->stride : row;                     \
        for (x = 0; x < ch->w; x++) {                                         \
            int32_t left = x ? row[x - 1] : (y ? rtop[0] : 0);                \
            int32_t top = y ? rtop[x] : left;                                 \
            int32_t topleft = (x && y) ? rtop[x - 1] : left;                  \
            uint32_t token = jxl_dec_read_clustered(dec, br, cluster,         \
                                                    dist_multiplier);         \
            int32_t diff = jxl_unpack_signed(token);                          \
            (void)top; (void)topleft;                                         \
            diff = (int32_t)((uint32_t)diff * mult + (uint32_t)off);          \
            row[x] = (int32_t)((uint32_t)diff + (uint32_t)(GUESS));           \
        }                                                                     \
        if (br->err || dec->err) {                                            \
            JXL_ERR(ctx, "modular: truncated stream %u",                      \
                    (unsigned)stream_idx);                                    \
            goto done;                                                        \
        }                                                                     \
    }

                switch (predictor) {
                    case JXL_PRED_ZERO:
                        JXL_FIXED_LOOP(0) break;
                    case JXL_PRED_WEST:
                        JXL_FIXED_LOOP(left) break;
                    case JXL_PRED_NORTH:
                        JXL_FIXED_LOOP(top) break;
                    case JXL_PRED_NORTH_WEST:
                        JXL_FIXED_LOOP(topleft) break;
                    case JXL_PRED_GRADIENT:
                        JXL_FIXED_LOOP(grad_clamped(top, left, topleft)) break;
                    case JXL_PRED_AVG_W_N:
                        JXL_FIXED_LOOP((int32_t)(((int64_t)left + top) / 2))
                        break;
                    case JXL_PRED_AVG_W_NW:
                        JXL_FIXED_LOOP((int32_t)(((int64_t)left + topleft) / 2))
                        break;
                    case JXL_PRED_AVG_N_NW:
                        JXL_FIXED_LOOP((int32_t)(((int64_t)top + topleft) / 2))
                        break;
                    case JXL_PRED_SELECT:
                        JXL_FIXED_LOOP(sel_pred(top, left, topleft)) break;
                    default:
                        /* Reaches past n/w/nw (north-east, west-west, the
                           averaging predictors over them): the state machine
                           knows how to find those, so keep it. */
                        for (y = 0; y < ch->h; y++) {
                            int32_t *row = ch->data + (size_t)y * ch->stride;
                            for (x = 0; x < ch->w; x++) {
                                uint32_t token = jxl_dec_read_clustered(
                                    dec, br, cluster, dist_multiplier);
                                int32_t diff = jxl_unpack_signed(token);
                                int32_t value;
                                diff = (int32_t)((uint32_t)diff * mult +
                                                 (uint32_t)off);
                                value = (int32_t)((uint32_t)diff +
                                        (uint32_t)predict_sample(&ps, &pr0,
                                                                 predictor));
                                row[x] = value;
                                pred_record(&ps, &pr0, value);
                            }
                            if (br->err || dec->err) {
                                JXL_ERR(ctx, "modular: truncated stream %u",
                                        (unsigned)stream_idx);
                                goto done;
                            }
                        }
                        break;
                }
#undef JXL_FIXED_LOOP
                continue;
            }
        }

        for (y = 0; y < ch->h; y++) {
            int32_t *row = ch->data + (size_t)y * ch->stride;
            for (x = 0; x < ch->w; x++) {
                jxl_props pr;
                const jxl_ma_leaf *leaf;
                uint32_t token;
                int32_t diff, value;

                props_compute(&ps, &pr, (int32_t)ci, (int32_t)stream_idx);
                leaf = ma_get_leaf(cma, &ps, &pr);
                token = jxl_dec_read_clustered(dec, br, leaf->cluster, dist_multiplier);
                diff = jxl_unpack_signed(token);
                diff = (int32_t)((uint32_t)diff * leaf->multiplier +
                                 (uint32_t)leaf->offset);
                value = (int32_t)((uint32_t)diff +
                                  (uint32_t)predict_sample(&ps, &pr, leaf->predictor));
                row[x] = value;
                pred_record(&ps, &pr, value);
            }
            if (br->err || dec->err) {
                JXL_ERR(ctx, "modular: truncated stream %u", (unsigned)stream_idx);
                goto done;
            }
        }
    }

    if (jxl_dec_finalize(dec) != 0) {
        JXL_ERR(ctx, "modular: bad ANS final state (stream %u)",
                (unsigned)stream_idx);
        goto done;
    }
    rc = 0;

done:
    jxl_free(ctx, spec.flat);
    if (wp_lut_local) jxl_free(ctx, wp_lut);
    pred_state_free(ctx, &ps);
    jxl_free(ctx, prev);
    return rc;
}
