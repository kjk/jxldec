/* container.c -- JPEG XL file format (ISOBMFF box) parsing.
 *
 * A .jxl file is either a bare codestream starting with FF 0A, or an ISOBMFF
 * container starting with the 12-byte JXL signature box. In the container the
 * codestream lives in a single `jxlc` box, or is split across `jxlp` boxes
 * (each prefixed with a 4-byte index whose high bit marks the last one).
 *
 * We only need the codestream; metadata boxes (Exif, xml, jbrd) are recorded
 * as raw spans for callers that want them. `brob` (Brotli-compressed) boxes
 * are skipped -- nothing in the decode path needs them.
 */
#include "jxl_internal.h"

static const uint8_t jxl_sig_container[12] = {
    0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a
};

jxl_signature jxl_signature_check(const uint8_t *data, size_t len) {
    if (!data) return JXL_SIG_INVALID;
    if (len == 0) return JXL_SIG_NOT_ENOUGH_BYTES;
    if (data[0] == 0xff) {
        if (len < 2) return JXL_SIG_NOT_ENOUGH_BYTES;
        return data[1] == 0x0a ? JXL_SIG_CODESTREAM : JXL_SIG_INVALID;
    }
    if (data[0] == 0x00) {
        size_t n = JXL_MIN(len, sizeof(jxl_sig_container));
        if (memcmp(data, jxl_sig_container, n) != 0) return JXL_SIG_INVALID;
        if (len < sizeof(jxl_sig_container)) return JXL_SIG_NOT_ENOUGH_BYTES;
        return JXL_SIG_CONTAINER;
    }
    return JXL_SIG_INVALID;
}

static uint32_t rd32be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static uint64_t rd64be(const uint8_t *p) {
    return ((uint64_t)rd32be(p) << 32) | rd32be(p + 4);
}

/* One codestream fragment from a jxlp/jxlc box. */
typedef struct {
    const uint8_t *p;
    size_t len;
} jxl_span;

int jxl_container_parse(jxl_ctx *ctx, const uint8_t *data, size_t len,
                        jxl_container *out) {
    jxl_span *spans = NULL;
    size_t nspans = 0, cap = 0;
    size_t pos = 0;
    size_t total = 0;
    int rc = -1;

    memset(out, 0, sizeof(*out));

    if (jxl_signature_check(data, len) == JXL_SIG_CODESTREAM) {
        out->cs = (uint8_t *)data;   /* not owned; const-cast is deliberate */
        out->cs_len = len;
        out->cs_owned = 0;
        return 0;
    }
    if (len < sizeof(jxl_sig_container) ||
        memcmp(data, jxl_sig_container, sizeof(jxl_sig_container)) != 0) {
        JXL_ERR(ctx, "not a JPEG XL file");
        return -1;
    }
    pos = sizeof(jxl_sig_container);

    while (pos + 8 <= len) {
        uint64_t box_size = rd32be(data + pos);
        const uint8_t *type = data + pos + 4;
        size_t hdr = 8;
        const uint8_t *payload;
        size_t payload_len;

        if (box_size == 1) {
            if (pos + 16 > len) break;
            box_size = rd64be(data + pos + 8);
            hdr = 16;
            if (box_size < 16) {
                JXL_ERR(ctx, "container: bad extended box size");
                goto done;
            }
        } else if (box_size == 0) {
            box_size = len - pos;   /* box extends to end of file */
        } else if (box_size < 8) {
            JXL_ERR(ctx, "container: bad box size %u", (unsigned)box_size);
            goto done;
        }
        if (box_size > len - pos) {
            /* Truncated final box: take what we have. */
            box_size = len - pos;
        }
        payload = data + pos + hdr;
        payload_len = (size_t)box_size - hdr;

        if (memcmp(type, "jxlc", 4) == 0) {
            if (nspans == cap) {
                size_t ncap = cap ? cap * 2 : 8;
                jxl_span *ns = (jxl_span *)jxl_realloc_array(
                    ctx, spans, cap, ncap, sizeof(*spans));
                if (!ns) goto done;
                spans = ns;
                cap = ncap;
            }
            spans[nspans].p = payload;
            spans[nspans].len = payload_len;
            nspans++;
            total += payload_len;
        } else if (memcmp(type, "jxlp", 4) == 0) {
            if (payload_len < 4) {
                JXL_ERR(ctx, "container: short jxlp box");
                goto done;
            }
            if (nspans == cap) {
                size_t ncap = cap ? cap * 2 : 8;
                jxl_span *ns = (jxl_span *)jxl_realloc_array(
                    ctx, spans, cap, ncap, sizeof(*spans));
                if (!ns) goto done;
                spans = ns;
                cap = ncap;
            }
            spans[nspans].p = payload + 4;
            spans[nspans].len = payload_len - 4;
            nspans++;
            total += payload_len - 4;
        } else if (memcmp(type, "Exif", 4) == 0) {
            out->exif = payload;
            out->exif_len = payload_len;
        } else if (memcmp(type, "xml ", 4) == 0) {
            out->xmp = payload;
            out->xmp_len = payload_len;
        } else if (memcmp(type, "jbrd", 4) == 0) {
            out->jbrd = payload;
            out->jbrd_len = payload_len;
        }

        pos += (size_t)box_size;
    }

    if (nspans == 0) {
        JXL_ERR(ctx, "container: no codestream box");
        goto done;
    }
    if (nspans == 1) {
        out->cs = (uint8_t *)spans[0].p;
        out->cs_len = spans[0].len;
        out->cs_owned = 0;
    } else {
        uint8_t *buf = (uint8_t *)jxl_malloc(ctx, total ? total : 1);
        size_t off = 0, i;
        if (!buf) goto done;
        for (i = 0; i < nspans; i++) {
            memcpy(buf + off, spans[i].p, spans[i].len);
            off += spans[i].len;
        }
        out->cs = buf;
        out->cs_len = total;
        out->cs_owned = 1;
    }
    rc = 0;

done:
    jxl_free(ctx, spans);
    return rc;
}

void jxl_container_free(jxl_ctx *ctx, jxl_container *c) {
    if (!c) return;
    if (c->cs_owned) jxl_free(ctx, c->cs);
    c->cs = NULL;
    c->cs_len = 0;
    c->cs_owned = 0;
}
