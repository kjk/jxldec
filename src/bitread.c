/* bitread.c -- LSB-first bit reader for the JPEG XL codestream.
 *
 * Bits are consumed from the least significant end of each byte, and a
 * multi-bit field is assembled LSB-first (spec: "u(n)"). The reader keeps a
 * 64-bit staging buffer holding at least 56 valid bits after a refill, so any
 * read of up to 32 bits is a mask + shift.
 *
 * Reads past the end of the buffer are not fatal: they return zero bits and
 * set the sticky `err` flag, which callers check at record boundaries. This
 * keeps the deeply nested header/entropy decoders free of error plumbing.
 */
#include "jxl_internal.h"

void jxl_br_init(jxl_br *br, const uint8_t *data, size_t len) {
    br->data = data;
    br->len = len;
    br->byte_pos = 0;
    br->buf = 0;
    br->nbits = 0;
    br->bits_read = 0;
    br->err = 0;
}

void jxl_br_refill(jxl_br *br) {
    if (br->byte_pos + 8 <= br->len) {
        uint64_t bits;
        int read_bytes;
        const uint8_t *p = br->data + br->byte_pos;
        bits = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
               ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) |
               ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
               ((uint64_t)p[7] << 56);
        br->buf |= bits << br->nbits;
        read_bytes = (63 - br->nbits) >> 3;
        br->nbits |= 56;
        br->byte_pos += (size_t)read_bytes;
        return;
    }
    while (br->nbits < 56 && br->byte_pos < br->len) {
        br->buf |= (uint64_t)br->data[br->byte_pos] << br->nbits;
        br->nbits += 8;
        br->byte_pos++;
    }
}

uint32_t jxl_br_peek(jxl_br *br, int n) {
    jxl_br_refill(br);
    if (n == 0) return 0;
    return (uint32_t)(br->buf & ((n == 64) ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1)));
}

void jxl_br_consume(jxl_br *br, int n) {
    if (br->nbits < n) {
        /* Ran off the end: report the failure and stop advancing. */
        br->err = 1;
        br->bits_read += (size_t)br->nbits;
        br->buf = 0;
        br->nbits = 0;
        return;
    }
    br->nbits -= n;
    br->bits_read += (size_t)n;
    br->buf >>= n;
}

uint32_t jxl_br_read(jxl_br *br, int n) {
    uint32_t v;
    if (n <= 0) return 0;
    v = jxl_br_peek(br, n);
    jxl_br_consume(br, n);
    return br->err ? 0 : v;
}

void jxl_br_skip(jxl_br *br, size_t n) {
    if ((size_t)br->nbits >= n) {
        br->nbits -= (int)n;
        br->bits_read += n;
        br->buf >>= n;
        return;
    }
    n -= (size_t)br->nbits;
    br->bits_read += (size_t)br->nbits;
    br->buf = 0;
    br->nbits = 0;
    if (n > (br->len - br->byte_pos) * 8) {
        br->bits_read += (br->len - br->byte_pos) * 8;
        br->byte_pos = br->len;
        br->err = 1;
        return;
    }
    br->bits_read += n;
    br->byte_pos += n / 8;
    n %= 8;
    jxl_br_refill(br);
    if ((size_t)br->nbits < n) {
        br->err = 1;
        br->nbits = 0;
        br->buf = 0;
        return;
    }
    br->nbits -= (int)n;
    br->buf >>= n;
}

int jxl_br_bool(jxl_br *br) {
    return (int)jxl_br_read(br, 1);
}

uint32_t jxl_br_u32(jxl_br *br, uint32_t c0, int n0, uint32_t c1, int n1,
                    uint32_t c2, int n2, uint32_t c3, int n3) {
    uint32_t sel = jxl_br_read(br, 2);
    uint32_t c;
    int n;
    switch (sel) {
        case 0: c = c0; n = n0; break;
        case 1: c = c1; n = n1; break;
        case 2: c = c2; n = n2; break;
        default: c = c3; n = n3; break;
    }
    if (n == 0) return c;
    return c + jxl_br_read(br, n);
}

uint64_t jxl_br_u64(jxl_br *br) {
    uint32_t sel = jxl_br_read(br, 2);
    switch (sel) {
        case 0: return 0;
        case 1: return (uint64_t)jxl_br_read(br, 4) + 1;
        case 2: return (uint64_t)jxl_br_read(br, 8) + 17;
        default: {
            uint64_t value = jxl_br_read(br, 12);
            int shift = 12;
            while (jxl_br_read(br, 1) == 1) {
                if (br->err) break;
                if (shift == 60) {
                    value |= (uint64_t)jxl_br_read(br, 4) << shift;
                    break;
                }
                value |= (uint64_t)jxl_br_read(br, 8) << shift;
                shift += 8;
            }
            return value;
        }
    }
}

float jxl_br_f16(jxl_br *br) {
    uint32_t v = jxl_br_read(br, 16);
    uint32_t neg = (v & 0x8000u) << 16;
    uint32_t mantissa = v & 0x3ff;
    uint32_t exponent = (v >> 10) & 0x1f;
    uint32_t bits;
    float out;

    if ((v & 0x7fff) == 0) {
        memcpy(&out, &neg, sizeof(out));
        return out;
    }
    if (exponent == 0x1f) {
        /* NaN / Infinity are not valid in the bitstream. */
        br->err = 1;
        return 0.0f;
    }
    if (exponent == 0) {
        float val = (1.0f / 16384.0f) * ((float)mantissa / 1024.0f);
        return neg ? -val : val;
    }
    bits = (mantissa << 13) | ((exponent + 112) << 23) | neg;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint32_t jxl_br_enum(jxl_br *br) {
    return jxl_br_u32(br, 0, 0, 1, 0, 2, 4, 18, 6);
}

void jxl_br_zero_pad_to_byte(jxl_br *br) {
    size_t rem = br->bits_read & 7;
    if (rem == 0) return;
    if (jxl_br_read(br, (int)(8 - rem)) != 0) {
        br->err = 1;
    }
}

size_t jxl_br_byte_pos(const jxl_br *br) {
    return (br->bits_read + 7) / 8;
}

void jxl_br_seek_byte(jxl_br *br, size_t byte_off) {
    if (byte_off > br->len) {
        br->err = 1;
        byte_off = br->len;
    }
    br->byte_pos = byte_off;
    br->buf = 0;
    br->nbits = 0;
    br->bits_read = byte_off * 8;
}
