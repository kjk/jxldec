/* headers.c -- codestream headers: SizeHeader, ImageMetadata, ColourEncoding,
 * ExtraChannelInfo and CustomTransformData.
 *
 * Layout of the start of a codestream:
 *   FF 0A | SizeHeader | ImageMetadata | CustomTransformData | [ICC] | frames
 * CustomTransformData (opsin inverse matrix + custom upsampling weights) is a
 * separate bundle read unconditionally after ImageMetadata, even when
 * ImageMetadata said "all default" -- a detail that is easy to get wrong.
 */
#include "jxl_internal.h"

/* ----- small helpers ----- */

/* Name: a length-prefixed UTF-8 string. Returns malloc'd NUL-terminated bytes,
   or NULL for the (common) empty name. */
char *jxl_read_name(jxl_ctx *ctx, jxl_br *br) {
    uint32_t len = jxl_br_u32(br, 0, 0, 0, 4, 16, 5, 48, 10);
    char *s;
    uint32_t i;
    if (len == 0 || br->err) return NULL;
    s = (char *)jxl_malloc(ctx, len + 1);
    if (!s) return NULL;
    for (i = 0; i < len; i++) s[i] = (char)jxl_br_read(br, 8);
    s[len] = 0;
    return s;
}

static void read_bit_depth(jxl_br *br, jxl_bit_depth *bd) {
    if (jxl_br_bool(br)) {
        bd->float_sample = 1;
        bd->bits_per_sample = jxl_br_u32(br, 32, 0, 16, 0, 24, 0, 1, 6);
        bd->exp_bits = jxl_br_read(br, 4) + 1;
    } else {
        bd->float_sample = 0;
        bd->bits_per_sample = jxl_br_u32(br, 8, 0, 10, 0, 12, 0, 1, 6);
        bd->exp_bits = 0;
    }
}

static void bit_depth_default(jxl_bit_depth *bd) {
    bd->float_sample = 0;
    bd->bits_per_sample = 8;
    bd->exp_bits = 0;
}

/* Customxy: a chromaticity coordinate scaled by 1e6. */
static float read_customxy(jxl_br *br) {
    uint32_t u = jxl_br_u32(br, 0, 19, 524288, 19, 1048576, 20, 2097152, 21);
    return (float)jxl_unpack_signed(u) / 1e6f;
}

static void size_header_read(jxl_br *br, jxl_size_header *sz) {
    int div8 = jxl_br_bool(br);
    uint32_t h_div8 = 0, ratio, w_div8 = 0;

    if (div8) {
        h_div8 = jxl_br_read(br, 5) + 1;
        sz->height = 8 * h_div8;
    } else {
        sz->height = jxl_br_u32(br, 1, 9, 1, 13, 1, 18, 1, 30);
    }
    ratio = jxl_br_read(br, 3);
    if (ratio == 0) {
        if (div8) {
            w_div8 = jxl_br_read(br, 5) + 1;
            sz->width = 8 * w_div8;
        } else {
            sz->width = jxl_br_u32(br, 1, 9, 1, 13, 1, 18, 1, 30);
        }
    } else {
        uint64_t h = sz->height;
        uint64_t w;
        switch (ratio) {
            case 1: w = h; break;
            case 2: w = h * 12 / 10; break;
            case 3: w = h * 4 / 3; break;
            case 4: w = h * 3 / 2; break;
            case 5: w = h * 16 / 9; break;
            case 6: w = h * 5 / 4; break;
            default: w = h * 2; break;
        }
        sz->width = (uint32_t)w;
    }
}

/* The preview header uses different U32 alternatives than SizeHeader. */
static void preview_header_read(jxl_br *br, jxl_size_header *sz) {
    int div8 = jxl_br_bool(br);
    uint32_t ratio, w_div8 = 1;

    if (div8) {
        uint32_t h_div8 = jxl_br_u32(br, 16, 0, 32, 0, 1, 5, 33, 9);
        sz->height = 8 * h_div8;
    } else {
        sz->height = jxl_br_u32(br, 1, 6, 65, 8, 321, 10, 1345, 12);
    }
    ratio = jxl_br_read(br, 3);
    if (ratio == 0) {
        if (div8) {
            w_div8 = jxl_br_u32(br, 16, 0, 32, 0, 1, 5, 33, 9);
            sz->width = 8 * w_div8;
        } else {
            sz->width = jxl_br_u32(br, 1, 6, 65, 8, 321, 10, 1345, 12);
        }
    } else {
        uint64_t h = sz->height;
        uint64_t w;
        switch (ratio) {
            case 1: w = h; break;
            case 2: w = h * 12 / 10; break;
            case 3: w = h * 4 / 3; break;
            case 4: w = h * 3 / 2; break;
            case 5: w = h * 16 / 9; break;
            case 6: w = h * 5 / 4; break;
            default: w = h * 2; break;
        }
        sz->width = (uint32_t)w;
    }
}

/* ----- extensions: a bitmask plus a bit length per set bit, all skipped ----- */

static void read_extensions(jxl_br *br) {
    uint64_t bits = jxl_br_u64(br);
    uint64_t lens[64];
    int idx[64];
    int n = 0, i;

    for (i = 0; i < 64; i++) {
        if ((bits >> i) & 1) {
            idx[n] = i;
            lens[n] = jxl_br_u64(br);
            n++;
        }
    }
    for (i = 0; i < n; i++) jxl_br_skip(br, (size_t)lens[i]);
}

/* ----- colour encoding ----- */

static void colour_encoding_default(jxl_colour_encoding *c) {
    memset(c, 0, sizeof(*c));
    c->want_icc = 0;
    c->colour_space = JXLDEC_CS_RGB;
    c->white_point = JXL_WP_D65;
    c->primaries = JXL_PRIMARIES_SRGB;
    c->tf = JXL_TF_SRGB;
    c->tf_have_gamma = 0;
    c->rendering_intent = 1;   /* relative colorimetric */
}

static void colour_encoding_read(jxl_ctx *ctx, jxl_br *br,
                                 jxl_colour_encoding *c) {
    colour_encoding_default(c);
    if (jxl_br_bool(br)) return;   /* all_default */

    c->want_icc = jxl_br_bool(br);
    c->colour_space = (jxl_color_space)jxl_br_enum(br);
    if ((uint32_t)c->colour_space > 3) {
        JXL_ERR(ctx, "header: invalid colour space %u", (unsigned)c->colour_space);
        br->err = 1;
        return;
    }
    if (c->want_icc) return;

    if (c->colour_space != JXLDEC_CS_XYB) {
        c->white_point = (jxl_white_point)jxl_br_enum(br);
        if (c->white_point == JXL_WP_CUSTOM) {
            c->white_xy[0] = read_customxy(br);
            c->white_xy[1] = read_customxy(br);
        }
    }
    if (c->colour_space != JXLDEC_CS_XYB && c->colour_space != JXLDEC_CS_GRAY) {
        c->primaries = (jxl_primaries)jxl_br_enum(br);
        if (c->primaries == JXL_PRIMARIES_CUSTOM) {
            int i;
            for (i = 0; i < 6; i++) c->prim_xy[i] = read_customxy(br);
        }
    }
    if (jxl_br_bool(br)) {
        c->tf_have_gamma = 1;
        c->tf_gamma = jxl_br_read(br, 24);
    } else {
        c->tf = (jxl_transfer_function)jxl_br_enum(br);
    }
    c->rendering_intent = jxl_br_enum(br);
}

/* ----- extra channels ----- */

static void ec_info_default(jxl_ec_info *ec) {
    memset(ec, 0, sizeof(*ec));
    ec->type = JXL_EC_ALPHA;
    bit_depth_default(&ec->bit_depth);
    ec->dim_shift = 0;
    ec->alpha_associated = 0;
}

static void ec_info_read(jxl_ctx *ctx, jxl_br *br, jxl_ec_info *ec) {
    uint32_t ty;
    ec_info_default(ec);
    if (jxl_br_bool(br)) return;   /* d_alpha: a plain 8-bit alpha channel */

    ty = jxl_br_enum(br);
    read_bit_depth(br, &ec->bit_depth);
    ec->dim_shift = jxl_br_u32(br, 0, 0, 3, 0, 4, 0, 1, 3);
    ec->name = jxl_read_name(ctx, br);
    ec->type = (jxl_ec_type)ty;

    switch (ty) {
        case JXL_EC_ALPHA:
            ec->alpha_associated = jxl_br_bool(br);
            break;
        case JXL_EC_SPOT:
            ec->spot[0] = jxl_br_f16(br);
            ec->spot[1] = jxl_br_f16(br);
            ec->spot[2] = jxl_br_f16(br);
            ec->spot[3] = jxl_br_f16(br);
            break;
        case JXL_EC_CFA:
            ec->cfa_channel = jxl_br_u32(br, 1, 0, 0, 2, 3, 4, 19, 8);
            break;
        case JXL_EC_DEPTH:
        case JXL_EC_SELECTION_MASK:
        case JXL_EC_BLACK:
        case JXL_EC_THERMAL:
        case JXL_EC_NON_OPTIONAL:
        case JXL_EC_OPTIONAL:
            break;
        default:
            JXL_ERR(ctx, "header: unknown extra channel type %u", (unsigned)ty);
            br->err = 1;
            break;
    }
}

/* ----- CustomTransformData (opsin matrix + upsampling weights) ----- */

static const float default_opsin_inv[9] = {
    11.031566901960783f, -9.866943921568629f, -0.16462299647058826f,
    -3.254147380392157f, 4.418770392156863f,  -0.16462299647058826f,
    -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f
};

static void custom_transform_data_read(jxl_br *br, jxl_image_metadata *meta) {
    uint32_t cw_mask = 0;
    int i;

    memcpy(meta->opsin_inv, default_opsin_inv, sizeof(meta->opsin_inv));
    for (i = 0; i < 3; i++) meta->opsin_bias[i] = -0.0037930732552754493f;
    meta->quant_bias[0] = 1.0f - 0.05465007330715401f;
    meta->quant_bias[1] = 1.0f - 0.07005449891748593f;
    meta->quant_bias[2] = 1.0f - 0.049935103337343655f;
    meta->quant_bias_numerator = 0.145f;
    memcpy(meta->up2, jxl_default_up2, sizeof(meta->up2));
    memcpy(meta->up4, jxl_default_up4, sizeof(meta->up4));
    memcpy(meta->up8, jxl_default_up8, sizeof(meta->up8));

    if (jxl_br_bool(br)) return;   /* all_default */

    if (meta->xyb_encoded) {
        /* OpsinInverseMatrix, itself an all_default bundle. */
        if (!jxl_br_bool(br)) {
            for (i = 0; i < 9; i++) meta->opsin_inv[i] = jxl_br_f16(br);
            for (i = 0; i < 3; i++) meta->opsin_bias[i] = jxl_br_f16(br);
            for (i = 0; i < 3; i++) meta->quant_bias[i] = jxl_br_f16(br);
            meta->quant_bias_numerator = jxl_br_f16(br);
        }
    }
    cw_mask = jxl_br_read(br, 3);
    if (cw_mask & 1) for (i = 0; i < 15; i++) meta->up2[i] = jxl_br_f16(br);
    if (cw_mask & 2) for (i = 0; i < 55; i++) meta->up4[i] = jxl_br_f16(br);
    if (cw_mask & 4) for (i = 0; i < 210; i++) meta->up8[i] = jxl_br_f16(br);
}

/* ----- image metadata ----- */

static void image_metadata_defaults(jxl_image_metadata *meta) {
    memset(meta, 0, sizeof(*meta));
    meta->orientation = 1;
    bit_depth_default(&meta->bit_depth);
    meta->modular_16bit_buffers = 1;
    meta->num_extra = 0;
    meta->alpha_index = -1;
    meta->xyb_encoded = 1;
    colour_encoding_default(&meta->colour);
    meta->tone_mapping.intensity_target = 255.0f;
    meta->tone_mapping.min_nits = 0.0f;
    meta->tone_mapping.relative_to_max_display = 0;
    meta->tone_mapping.linear_below = 0.0f;
}

static int image_metadata_read(jxl_ctx *ctx, jxl_br *br,
                               jxl_image_metadata *meta) {
    int all_default, extra_fields = 0;
    uint32_t i;

    image_metadata_defaults(meta);
    all_default = jxl_br_bool(br);
    if (!all_default) {
        extra_fields = jxl_br_bool(br);
        if (extra_fields) {
            meta->orientation = jxl_br_read(br, 3) + 1;
            meta->have_intr_size = jxl_br_bool(br);
            if (meta->have_intr_size) size_header_read(br, &meta->intrinsic);
            meta->have_preview = jxl_br_bool(br);
            if (meta->have_preview) preview_header_read(br, &meta->preview);
            meta->have_animation = jxl_br_bool(br);
            if (meta->have_animation) {
                meta->animation.tps_numerator =
                    jxl_br_u32(br, 100, 0, 1000, 0, 1, 10, 1, 30);
                meta->animation.tps_denominator =
                    jxl_br_u32(br, 1, 0, 1001, 0, 1, 8, 1, 10);
                meta->animation.num_loops =
                    jxl_br_u32(br, 0, 0, 0, 3, 0, 16, 0, 32);
                meta->animation.have_timecodes = jxl_br_bool(br);
            }
        }
        read_bit_depth(br, &meta->bit_depth);
        meta->modular_16bit_buffers = jxl_br_bool(br);
        /* Count the channels into a local and only publish it once the array
           behind it exists. Assigning meta->num_extra straight from the
           bitstream left it non-zero with ec_info still NULL on both of the
           early returns below, and jxl_image_metadata_free walks num_extra
           entries -- a null dereference on a path only a malformed file takes.
           Every one of the fuzzer's first eight crashes was this. */
        uint32_t n_extra = jxl_br_u32(br, 0, 0, 1, 0, 2, 4, 1, 12);
        if (n_extra > 256) {
            JXL_ERR(ctx, "header: too many extra channels (%u)",
                    (unsigned)n_extra);
            return -1;
        }
        if (n_extra) {
            meta->ec_info = (jxl_ec_info *)jxl_calloc(ctx, n_extra,
                                                      sizeof(jxl_ec_info));
            if (!meta->ec_info) return -1;
            meta->num_extra = n_extra;
            for (i = 0; i < meta->num_extra; i++) {
                ec_info_read(ctx, br, &meta->ec_info[i]);
                if (br->err) return -1;
            }
        }
        meta->xyb_encoded = jxl_br_bool(br);
        colour_encoding_read(ctx, br, &meta->colour);
        if (extra_fields) {
            if (!jxl_br_bool(br)) {   /* ToneMapping all_default */
                meta->tone_mapping.intensity_target = jxl_br_f16(br);
                meta->tone_mapping.min_nits = jxl_br_f16(br);
                meta->tone_mapping.relative_to_max_display = jxl_br_bool(br);
                meta->tone_mapping.linear_below = jxl_br_f16(br);
            }
        }
        read_extensions(br);
    }

    custom_transform_data_read(br, meta);

    for (i = 0; i < meta->num_extra; i++) {
        if (meta->ec_info[i].type == JXL_EC_ALPHA) {
            meta->alpha_index = (int)i;
            break;
        }
    }

    if (br->err) {
        JXL_ERR(ctx, "header: truncated image metadata");
        return -1;
    }
    if (!(meta->tone_mapping.intensity_target > 0.0f)) {
        JXL_ERR(ctx, "header: invalid intensity target");
        return -1;
    }
    return 0;
}

int jxl_read_image_header(jxl_ctx *ctx, jxl_br *br, jxl_size_header *size,
                          jxl_image_metadata *meta) {
    uint32_t sig = jxl_br_read(br, 16);
    if (sig != 0x0aff) {
        JXL_ERR(ctx, "codestream: bad signature 0x%04x", (unsigned)sig);
        return -1;
    }
    size_header_read(br, size);
    if (br->err || size->width == 0 || size->height == 0) {
        JXL_ERR(ctx, "codestream: bad image size");
        return -1;
    }
    return image_metadata_read(ctx, br, meta);
}

void jxl_image_metadata_free(jxl_ctx *ctx, jxl_image_metadata *meta) {
    uint32_t i;
    if (!meta) return;
    /* Belt as well as braces: the caller is jxl_doc_close, which also runs on
       the failure path of jxl_doc_open, so it sees metadata abandoned at
       whatever point parsing gave up. Never trust num_extra alone. */
    if (!meta->ec_info) meta->num_extra = 0;
    for (i = 0; i < meta->num_extra; i++) jxl_free(ctx, meta->ec_info[i].name);
    jxl_free(ctx, meta->ec_info);
    meta->ec_info = NULL;
    meta->num_extra = 0;
}

void jxl_apply_orientation_dims(uint32_t orientation, uint32_t w, uint32_t h,
                                uint32_t *ow, uint32_t *oh) {
    if (orientation >= 5 && orientation <= 8) {
        *ow = h;
        *oh = w;
    } else {
        *ow = w;
        *oh = h;
    }
}
