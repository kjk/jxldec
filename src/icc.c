/* icc.c -- the embedded ICC profile: entropy-coded byte stream plus the
 * ICC-specific "commands" decompression that rebuilds the profile.
 *
 * Two stages: read_icc_stream pulls enc_size bytes through the generic entropy
 * decoder with a context derived from the previous two bytes; decode_icc then
 * interprets that byte stream as a header-prediction + tag-table + command
 * program that reconstructs the original profile.
 */
#include "jxl_internal.h"

/* ----- growable byte buffer ----- */

typedef struct {
    jxl_ctx *ctx;
    uint8_t *p;
    size_t len;
    size_t cap;
    int err;
} jxl_bytebuf;

static void bb_init(jxl_bytebuf *b, jxl_ctx *ctx) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx;
}

static void bb_free(jxl_bytebuf *b) {
    jxl_free(b->ctx, b->p);
    b->p = NULL;
    b->len = b->cap = 0;
}

static int bb_reserve(jxl_bytebuf *b, size_t need) {
    size_t cap;
    uint8_t *np;
    if (b->err) return -1;
    if (b->len + need <= b->cap) return 0;
    cap = b->cap ? b->cap : 256;
    while (cap < b->len + need) {
        if (cap > ((size_t)-1) / 2) { b->err = 1; return -1; }
        cap *= 2;
    }
    np = (uint8_t *)jxl_realloc_array(b->ctx, b->p, b->cap, cap, 1);
    if (!np) { b->err = 1; return -1; }
    b->p = np;
    b->cap = cap;
    return 0;
}

static void bb_push(jxl_bytebuf *b, uint8_t v) {
    if (bb_reserve(b, 1) != 0) return;
    b->p[b->len++] = v;
}

static void bb_append(jxl_bytebuf *b, const void *src, size_t n) {
    if (bb_reserve(b, n) != 0) return;
    memcpy(b->p + b->len, src, n);
    b->len += n;
}

static void bb_push_be32(jxl_bytebuf *b, uint32_t v) {
    uint8_t t[4];
    t[0] = (uint8_t)(v >> 24);
    t[1] = (uint8_t)(v >> 16);
    t[2] = (uint8_t)(v >> 8);
    t[3] = (uint8_t)v;
    bb_append(b, t, 4);
}

/* ----- varint over a plain byte cursor ----- */

typedef struct {
    const uint8_t *p;
    size_t len;
    size_t pos;
    int err;
} jxl_bytecur;

static int bc_u8(jxl_bytecur *c, uint8_t *out) {
    if (c->pos >= c->len) { c->err = 1; return -1; }
    *out = c->p[c->pos++];
    return 0;
}

static uint64_t bc_varint(jxl_bytecur *c) {
    uint64_t value = 0;
    int shift = 0;
    while (shift < 63) {
        uint8_t b;
        if (bc_u8(c, &b) != 0) return 0;
        value |= (uint64_t)(b & 0x7f) << shift;
        if ((b & 0x80) == 0) break;
        shift += 7;
    }
    return value;
}

/* ----- stage 1: entropy-coded byte stream ----- */

static uint32_t icc_ctx_for(size_t idx, uint8_t b1, uint8_t b2) {
    uint32_t p1, p2;
    if (idx <= 128) return 0;

    if ((b1 >= 'a' && b1 <= 'z') || (b1 >= 'A' && b1 <= 'Z')) p1 = 0;
    else if ((b1 >= '0' && b1 <= '9') || b1 == '.' || b1 == ',') p1 = 1;
    else if (b1 <= 1) p1 = 2 + b1;
    else if (b1 <= 15) p1 = 4;
    else if (b1 >= 241 && b1 <= 254) p1 = 5;
    else if (b1 == 255) p1 = 6;
    else p1 = 7;

    if ((b2 >= 'a' && b2 <= 'z') || (b2 >= 'A' && b2 <= 'Z')) p2 = 0;
    else if ((b2 >= '0' && b2 <= '9') || b2 == '.' || b2 == ',') p2 = 1;
    else if (b2 <= 15) p2 = 2;
    else if (b2 >= 241) p2 = 3;
    else p2 = 4;

    return 1 + p1 + 8 * p2;
}

static int read_icc_stream(jxl_ctx *ctx, jxl_br *br, uint8_t **out,
                           size_t *out_len) {
    uint64_t enc_size = jxl_br_u64(br);
    jxl_dec dec;
    uint8_t *enc = NULL;
    uint8_t b1 = 0, b2 = 0;
    size_t i;
    int rc = -1;

    if (enc_size > (1u << 28)) {
        JXL_ERR(ctx, "icc: encoded profile too large");
        return -1;
    }
    if (jxl_dec_init(ctx, &dec, br, 41) != 0) return -1;

    enc = (uint8_t *)jxl_calloc(ctx, (size_t)enc_size ? (size_t)enc_size : 1, 1);
    if (!enc) goto done;

    for (i = 0; i < (size_t)enc_size; i++) {
        uint32_t sym = jxl_dec_read(&dec, br, icc_ctx_for(i, b1, b2));
        if (sym >= 256 || br->err || dec.err) {
            JXL_ERR(ctx, "icc: bad encoded byte");
            goto done;
        }
        enc[i] = (uint8_t)sym;
        b2 = b1;
        b1 = enc[i];
    }
    if (jxl_dec_finalize(&dec) != 0) {
        JXL_ERR(ctx, "icc: bad ANS final state");
        goto done;
    }
    *out = enc;
    *out_len = (size_t)enc_size;
    enc = NULL;
    rc = 0;

done:
    jxl_free(ctx, enc);
    jxl_dec_free(&dec);
    return rc;
}

/* ----- stage 2: command stream -> ICC profile ----- */

static const char *const icc_common_tags[19] = {
    "rTRC", "rXYZ", "cprt", "wtpt", "bkpt", "rXYZ", "gXYZ", "bXYZ", "kXYZ",
    "rTRC", "gTRC", "bTRC", "kTRC", "chad", "desc", "chrm", "dmnd", "dmdd",
    "lumi"
};

static const char *const icc_common_data[8] = {
    "XYZ ", "desc", "text", "mluc", "para", "curv", "sf32", "gbd "
};

static uint8_t predict_header(size_t idx, uint32_t output_size,
                              const uint8_t *header) {
    static const char mntr[] = "mntrRGB XYZ ";
    if (idx <= 3) return (uint8_t)(output_size >> (8 * (3 - idx)));
    if (idx == 8) return 4;
    if (idx >= 12 && idx <= 23) return (uint8_t)mntr[idx - 12];
    if (idx >= 36 && idx <= 39) return (uint8_t)"acsp"[idx - 36];
    if ((idx == 41 || idx == 42) && header[40] == 'A') return 'P';
    if (idx == 43 && header[40] == 'A') return 'L';
    if (idx == 41 && header[40] == 'M') return 'S';
    if (idx == 42 && header[40] == 'M') return 'F';
    if (idx == 43 && header[40] == 'M') return 'T';
    if (idx == 42 && header[40] == 'S' && header[41] == 'G') return 'I';
    if (idx == 43 && header[40] == 'S' && header[41] == 'G') return ' ';
    if (idx == 42 && header[40] == 'S' && header[41] == 'U') return 'N';
    if (idx == 43 && header[40] == 'S' && header[41] == 'U') return 'W';
    if (idx == 70) return 246;
    if (idx == 71) return 214;
    if (idx == 73) return 1;
    if (idx == 78) return 211;
    if (idx == 79) return 45;
    if (idx >= 80 && idx <= 83) return header[4 + idx - 80];
    return 0;
}

/* De-interleave 2- and 4-byte planes (the encoder groups like-significance
   bytes together to make them more predictable). */
static void shuffle2(const uint8_t *in, size_t len, uint8_t *out) {
    size_t height = len / 2, odd = len % 2, i;
    for (i = 0; i < height; i++) {
        out[2 * i] = in[i];
        out[2 * i + 1] = in[i + height + odd];
    }
    if (odd) out[len - 1] = in[height];
}

static void shuffle4(const uint8_t *in, size_t len, uint8_t *out) {
    size_t step = len / 4, wide = len % 4, i, j, o = 0;
    for (i = 0; i < step; i++) {
        size_t base = i;
        for (j = 0; j < wide; j++) {
            out[o++] = in[base];
            base += step + 1;
        }
        for (j = wide; j < 4; j++) {
            out[o++] = in[base];
            base += step;
        }
    }
    for (i = 1; i <= wide; i++) out[o++] = in[(step + 1) * i - 1];
}

static int decode_icc(jxl_ctx *ctx, const uint8_t *stream, size_t stream_len,
                      uint8_t **out_p, size_t *out_n) {
    jxl_bytecur hdr = {stream, stream_len, 0, 0};
    jxl_bytecur cmds;
    uint64_t output_size, commands_size;
    size_t stream_offset;
    const uint8_t *data;
    size_t data_len;
    size_t header_size;
    jxl_bytebuf out;
    uint8_t *shuf = NULL;
    size_t i;
    int rc = -1;

    output_size = bc_varint(&hdr);
    commands_size = bc_varint(&hdr);
    stream_offset = hdr.pos;
    if (hdr.err || stream_offset + commands_size > stream_len) {
        JXL_ERR(ctx, "icc: invalid commands_size");
        return -1;
    }
    if (output_size > (1u << 28)) {
        JXL_ERR(ctx, "icc: output too large");
        return -1;
    }

    cmds.p = stream + stream_offset;
    cmds.len = (size_t)commands_size;
    cmds.pos = 0;
    cmds.err = 0;

    data = stream + stream_offset + commands_size;
    data_len = stream_len - stream_offset - (size_t)commands_size;

    header_size = output_size < 128 ? (size_t)output_size : 128;
    if (data_len < header_size) {
        JXL_ERR(ctx, "icc: invalid output_size");
        return -1;
    }

    bb_init(&out, ctx);
    for (i = 0; i < header_size; i++) {
        uint8_t p = predict_header(i, (uint32_t)output_size, data);
        bb_push(&out, (uint8_t)(p + data[i]));
    }
    data += header_size;
    data_len -= header_size;
    if (output_size <= 128) goto finish;

    /* Tag table. */
    {
        uint64_t v = bc_varint(&cmds);
        if (v >= 1) {
            uint32_t num_tags = (uint32_t)(v - 1);
            uint32_t prev_tagstart, prev_tagsize = 0;
            if ((output_size - 128) / 12 < num_tags) {
                JXL_ERR(ctx, "icc: num_tags too large");
                goto done;
            }
            bb_push_be32(&out, num_tags);
            prev_tagstart = num_tags * 12 + 128;

            for (;;) {
                uint8_t command;
                uint8_t tagcode;
                const char *tag;
                char tagbuf[4];
                uint32_t tagstart, tagsize;

                if (bc_u8(&cmds, &command) != 0) goto finish;
                tagcode = command & 63;
                if (tagcode == 0) break;
                if (tagcode == 1) {
                    if (data_len < 4) {
                        JXL_ERR(ctx, "icc: short data stream");
                        goto done;
                    }
                    memcpy(tagbuf, data, 4);
                    data += 4;
                    data_len -= 4;
                    tag = tagbuf;
                } else if (tagcode <= 20) {
                    tag = icc_common_tags[tagcode - 2];
                } else {
                    JXL_ERR(ctx, "icc: invalid tagcode");
                    goto done;
                }

                if (command & 64) tagstart = (uint32_t)bc_varint(&cmds);
                else tagstart = prev_tagstart + prev_tagsize;

                if (command & 128) {
                    tagsize = (uint32_t)bc_varint(&cmds);
                } else if (memcmp(tag, "rXYZ", 4) == 0 ||
                           memcmp(tag, "gXYZ", 4) == 0 ||
                           memcmp(tag, "bXYZ", 4) == 0 ||
                           memcmp(tag, "kXYZ", 4) == 0 ||
                           memcmp(tag, "wtpt", 4) == 0 ||
                           memcmp(tag, "bkpt", 4) == 0 ||
                           memcmp(tag, "lumi", 4) == 0) {
                    tagsize = 20;
                } else {
                    tagsize = prev_tagsize;
                }
                if ((uint64_t)tagstart + tagsize > output_size) {
                    JXL_ERR(ctx, "icc: tag out of range");
                    goto done;
                }
                prev_tagstart = tagstart;
                prev_tagsize = tagsize;

                bb_append(&out, tag, 4);
                bb_push_be32(&out, tagstart);
                bb_push_be32(&out, tagsize);
                if (tagcode == 2) {
                    bb_append(&out, "gTRC", 4);
                    bb_push_be32(&out, tagstart);
                    bb_push_be32(&out, tagsize);
                    bb_append(&out, "bTRC", 4);
                    bb_push_be32(&out, tagstart);
                    bb_push_be32(&out, tagsize);
                } else if (tagcode == 3) {
                    bb_append(&out, "gXYZ", 4);
                    bb_push_be32(&out, tagstart + tagsize);
                    bb_push_be32(&out, tagsize);
                    bb_append(&out, "bXYZ", 4);
                    bb_push_be32(&out, tagstart + tagsize * 2);
                    bb_push_be32(&out, tagsize);
                }
                if (cmds.err) goto finish;
            }
        }
    }

    /* Main command loop. */
    for (;;) {
        uint8_t command;
        if (bc_u8(&cmds, &command) != 0) break;
        if (command == 1) {
            size_t num = (size_t)bc_varint(&cmds);
            if (num > data_len) { JXL_ERR(ctx, "icc: short stream"); goto done; }
            bb_append(&out, data, num);
            data += num;
            data_len -= num;
        } else if (command == 2 || command == 3) {
            size_t num = (size_t)bc_varint(&cmds);
            if (num > data_len) { JXL_ERR(ctx, "icc: short stream"); goto done; }
            shuf = (uint8_t *)jxl_malloc(ctx, num ? num : 1);
            if (!shuf) goto done;
            if (command == 2) shuffle2(data, num, shuf);
            else shuffle4(data, num, shuf);
            bb_append(&out, shuf, num);
            jxl_free(ctx, shuf);
            shuf = NULL;
            data += num;
            data_len -= num;
        } else if (command == 4) {
            uint8_t flags;
            size_t width, order, stride, num, k;
            const uint8_t *src;
            if (bc_u8(&cmds, &flags) != 0) { JXL_ERR(ctx, "icc: short stream"); goto done; }
            width = (size_t)((flags & 3) + 1);
            order = (size_t)((flags >> 2) & 3);
            if (width == 3 || order == 3) {
                JXL_ERR(ctx, "icc: bad predictor flags");
                goto done;
            }
            if (flags & 16) {
                stride = (size_t)bc_varint(&cmds);
                if (stride < width) { JXL_ERR(ctx, "icc: stride < width"); goto done; }
            } else {
                stride = width;
            }
            if (stride * 4 >= out.len) {
                JXL_ERR(ctx, "icc: stride too large");
                goto done;
            }
            num = (size_t)bc_varint(&cmds);
            if (num > data_len) { JXL_ERR(ctx, "icc: short stream"); goto done; }
            if (width == 1) {
                src = data;
            } else {
                shuf = (uint8_t *)jxl_malloc(ctx, num ? num : 1);
                if (!shuf) goto done;
                if (width == 2) shuffle2(data, num, shuf);
                else shuffle4(data, num, shuf);
                src = shuf;
            }
            for (k = 0; k < num; k += width) {
                uint32_t prev[3] = {0, 0, 0};
                uint32_t p;
                size_t j;
                for (j = 0; j <= order; j++) {
                    size_t offset = out.len - stride * (j + 1);
                    uint8_t t[4] = {0, 0, 0, 0};
                    memcpy(t + (4 - width), out.p + offset, width);
                    prev[j] = ((uint32_t)t[0] << 24) | ((uint32_t)t[1] << 16) |
                              ((uint32_t)t[2] << 8) | t[3];
                }
                if (order == 0) p = prev[0];
                else if (order == 1) p = 2 * prev[0] - prev[1];
                else p = 3 * (prev[0] - prev[1]) + prev[2];

                for (j = 0; j < width && j < num - k; j++) {
                    uint32_t val = (uint32_t)src[k + j] + (p >> (8 * (width - 1 - j)));
                    bb_push(&out, (uint8_t)val);
                }
            }
            jxl_free(ctx, shuf);
            shuf = NULL;
            data += num;
            data_len -= num;
        } else if (command == 10) {
            static const uint8_t xyz[8] = {'X', 'Y', 'Z', ' ', 0, 0, 0, 0};
            if (data_len < 12) { JXL_ERR(ctx, "icc: short stream"); goto done; }
            bb_append(&out, xyz, 8);
            bb_append(&out, data, 12);
            data += 12;
            data_len -= 12;
        } else if (command >= 16 && command <= 23) {
            static const uint8_t zeros[4] = {0, 0, 0, 0};
            bb_append(&out, icc_common_data[command - 16], 4);
            bb_append(&out, zeros, 4);
        } else {
            JXL_ERR(ctx, "icc: invalid command %u", (unsigned)command);
            goto done;
        }
        if (cmds.err || out.err) goto done;
    }

    if (out.len != (size_t)output_size) {
        JXL_ERR(ctx, "icc: size mismatch (%u vs %u)", (unsigned)out.len,
                (unsigned)output_size);
        goto done;
    }

finish:
    if (out.err) goto done;
    *out_p = out.p;
    *out_n = out.len;
    out.p = NULL;
    rc = 0;

done:
    jxl_free(ctx, shuf);
    bb_free(&out);
    return rc;
}

int jxl_read_icc(jxl_ctx *ctx, jxl_br *br, uint8_t **out, size_t *out_len) {
    uint8_t *enc = NULL;
    size_t enc_len = 0;
    int rc;

    *out = NULL;
    *out_len = 0;
    if (read_icc_stream(ctx, br, &enc, &enc_len) != 0) return -1;
    rc = decode_icc(ctx, enc, enc_len, out, out_len);
    jxl_free(ctx, enc);
    return rc;
}
