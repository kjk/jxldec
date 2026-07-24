/* jxl_internal.h -- every cross-module declaration of the C JPEG XL decoder.
 *
 * One header for the whole library (djvudec convention): every source file
 * includes just this one. Sections below mirror the modules:
 *   core / bitread / container / headers / icc / coding / modular / frame /
 *   vardct / dct / filter / color / render / debug
 */
#ifndef JXL_INTERNAL_H
#define JXL_INTERNAL_H

#include "jxl.h"

#include <stdarg.h>
#include <string.h>

/* ===================================================================== */
/* core                                                                   */
/* ===================================================================== */

struct jxl_ctx {
    jxl_alloc_cb alloc;
    jxl_free_cb free_cb;
    jxl_error_cb error;
    void *user;

    int bgr;                 /* emit B,G,R[,A] instead of R,G,B[,A] */
    int keep_orientation;    /* skip the EXIF-style orientation fixup */
    volatile int abort_epoch;
};

void *jxl_malloc(jxl_ctx *ctx, size_t size);
void *jxl_calloc(jxl_ctx *ctx, size_t count, size_t size);
void *jxl_realloc_array(jxl_ctx *ctx, void *ptr, size_t old_count,
                        size_t new_count, size_t size);
void jxl_free(jxl_ctx *ctx, void *ptr);

void jxl_errorf(jxl_ctx *ctx, jxl_severity sev, const char *fmt, ...);

#define JXL_ERR(ctx, ...)  jxl_errorf((ctx), JXL_SEVERITY_ERROR, __VA_ARGS__)
#define JXL_WARN(ctx, ...) jxl_errorf((ctx), JXL_SEVERITY_WARNING, __VA_ARGS__)

/* Overflow-checked multiply for allocation sizes. Returns 0 on overflow. */
int jxl_size_mul(size_t a, size_t b, size_t *out);

#define JXL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define JXL_MAX(a, b) ((a) > (b) ? (a) : (b))

/* ===================================================================== */
/* bitread -- LSB-first bit reader (jxl_br)                                */
/* ===================================================================== */

typedef struct {
    const uint8_t *data;   /* whole buffer                                */
    size_t len;            /* bytes in data                               */
    size_t byte_pos;       /* next byte to pull into buf                  */
    uint64_t buf;          /* bit buffer, LSB = next bit                  */
    int nbits;             /* valid bits in buf                           */
    size_t bits_read;      /* bits consumed since jxl_br_init             */
    int err;               /* sticky: 1 once a read ran past the end      */
} jxl_br;

void jxl_br_init(jxl_br *br, const uint8_t *data, size_t len);
void jxl_br_refill(jxl_br *br);
uint32_t jxl_br_read(jxl_br *br, int n);          /* n <= 32              */
uint32_t jxl_br_peek(jxl_br *br, int n);
void jxl_br_consume(jxl_br *br, int n);
void jxl_br_skip(jxl_br *br, size_t n);
int jxl_br_bool(jxl_br *br);
uint32_t jxl_br_u32(jxl_br *br, uint32_t c0, int n0, uint32_t c1, int n1,
                    uint32_t c2, int n2, uint32_t c3, int n3);
uint64_t jxl_br_u64(jxl_br *br);
float jxl_br_f16(jxl_br *br);
uint32_t jxl_br_enum(jxl_br *br);
void jxl_br_zero_pad_to_byte(jxl_br *br);
/* Bytes consumed so far, rounded up (valid right after zero_pad_to_byte). */
size_t jxl_br_byte_pos(const jxl_br *br);
/* Reposition to an absolute byte offset in the underlying buffer. */
void jxl_br_seek_byte(jxl_br *br, size_t byte_off);

/* jxl_br_u32 takes four (offset, nbits) alternatives; the value is
   offset + u(nbits), and nbits == 0 means the alternative is the constant. */

static inline int32_t jxl_unpack_signed(uint32_t u) {
    return (int32_t)((u >> 1) ^ (~(u & 1) + 1));
}
static inline int64_t jxl_unpack_signed64(uint64_t u) {
    return (int64_t)((u >> 1) ^ (~(u & 1) + 1));
}

/* ===================================================================== */
/* container -- ISOBMFF box parsing                                       */
/* ===================================================================== */

typedef struct {
    uint8_t *cs;        /* codestream bytes                               */
    size_t cs_len;
    int cs_owned;       /* 1 when cs must be freed (concatenated jxlp)    */
    const uint8_t *exif; size_t exif_len;
    const uint8_t *xmp;  size_t xmp_len;
    const uint8_t *jbrd; size_t jbrd_len;   /* jpeg reconstruction data   */
} jxl_container;

int jxl_container_parse(jxl_ctx *ctx, const uint8_t *data, size_t len,
                        jxl_container *out);
void jxl_container_free(jxl_ctx *ctx, jxl_container *c);

/* ===================================================================== */
/* headers -- SizeHeader / ImageMetadata / ColourEncoding                 */
/* ===================================================================== */

typedef enum {
    JXL_EC_ALPHA = 0,
    JXL_EC_DEPTH = 1,
    JXL_EC_SPOT = 2,
    JXL_EC_SELECTION_MASK = 3,
    JXL_EC_BLACK = 4,
    JXL_EC_CFA = 5,
    JXL_EC_THERMAL = 6,
    JXL_EC_NON_OPTIONAL = 15,
    JXL_EC_OPTIONAL = 16
} jxl_ec_type;

typedef struct {
    int float_sample;
    uint32_t bits_per_sample;
    uint32_t exp_bits;
} jxl_bit_depth;

typedef struct {
    jxl_ec_type type;
    jxl_bit_depth bit_depth;
    uint32_t dim_shift;
    char *name;              /* NUL-terminated UTF-8, or NULL             */
    int alpha_associated;    /* JXL_EC_ALPHA                              */
    float spot[4];           /* JXL_EC_SPOT: r,g,b,solidity               */
    uint32_t cfa_channel;    /* JXL_EC_CFA                                */
} jxl_ec_info;

typedef enum {
    JXL_WP_D65 = 1,
    JXL_WP_CUSTOM = 2,
    JXL_WP_E = 10,
    JXL_WP_DCI = 11
} jxl_white_point;

typedef enum {
    JXL_PRIMARIES_SRGB = 1,
    JXL_PRIMARIES_CUSTOM = 2,
    JXL_PRIMARIES_2100 = 9,
    JXL_PRIMARIES_P3 = 11
} jxl_primaries;

typedef enum {
    JXL_TF_709 = 1,
    JXL_TF_UNKNOWN = 2,
    JXL_TF_LINEAR = 8,
    JXL_TF_SRGB = 13,
    JXL_TF_PQ = 16,
    JXL_TF_DCI = 17,
    JXL_TF_HLG = 18
} jxl_transfer_function;

typedef struct {
    int want_icc;
    jxl_color_space colour_space;
    jxl_white_point white_point;
    float white_xy[2];
    jxl_primaries primaries;
    float prim_xy[6];        /* red x,y green x,y blue x,y                */
    int tf_have_gamma;
    uint32_t tf_gamma;       /* gamma * 1e7 (inverse gamma)               */
    jxl_transfer_function tf;
    uint32_t rendering_intent;
} jxl_colour_encoding;

typedef struct {
    float intensity_target;
    float min_nits;
    int relative_to_max_display;
    float linear_below;
} jxl_tone_mapping;

typedef struct {
    uint32_t tps_numerator;
    uint32_t tps_denominator;
    uint32_t num_loops;
    int have_timecodes;
} jxl_animation_header;

typedef struct {
    uint32_t width, height;
} jxl_size_header;

typedef struct {
    uint32_t orientation;         /* 1..8                                 */
    int have_intr_size;
    jxl_size_header intrinsic;
    int have_preview;
    jxl_size_header preview;
    int have_animation;
    jxl_animation_header animation;

    jxl_bit_depth bit_depth;
    int modular_16bit_buffers;
    uint32_t num_extra;
    jxl_ec_info *ec_info;
    int alpha_index;              /* index into ec_info, or -1            */

    int xyb_encoded;
    jxl_colour_encoding colour;
    jxl_tone_mapping tone_mapping;

    float opsin_inv[9];           /* row-major 3x3                        */
    float opsin_bias[3];
    float quant_bias[3];
    float quant_bias_numerator;

    float up2[15], up4[55], up8[210];
} jxl_image_metadata;

/* Parses the 0xFF 0x0A signature, SizeHeader and ImageMetadata. */
int jxl_read_image_header(jxl_ctx *ctx, jxl_br *br, jxl_size_header *size,
                          jxl_image_metadata *meta);
void jxl_image_metadata_free(jxl_ctx *ctx, jxl_image_metadata *meta);

/* Applies the orientation field to a (w,h) pair. */
void jxl_apply_orientation_dims(uint32_t orientation, uint32_t w, uint32_t h,
                                uint32_t *ow, uint32_t *oh);

extern const float jxl_default_up2[15];
extern const float jxl_default_up4[55];
extern const float jxl_default_up8[210];

/* ===================================================================== */
/* icc -- compressed ICC profile                                          */
/* ===================================================================== */

int jxl_read_icc(jxl_ctx *ctx, jxl_br *br, uint8_t **out, size_t *out_len);

/* ===================================================================== */
/* coding -- entropy decoding (prefix codes, ANS, LZ77, context maps)      */
/* ===================================================================== */

/* Number of bits needed to represent x (0 for x == 0). */
uint32_t jxl_bitlen(uint32_t x);

typedef struct {
    uint16_t sym;    /* symbol, or offset into the sub-table when nested */
    uint8_t len;     /* bits to consume, or sub-table index mask when nested */
    uint8_t nested;
} jxl_pfx_entry;

typedef struct {
    jxl_pfx_entry *root;
    jxl_pfx_entry *sub;
    uint32_t nsub;
    int root_bits;
    uint32_t root_mask;
    int single_symbol;   /* >= 0 when the code has exactly one symbol */
} jxl_pfx_hist;

typedef struct {
    uint8_t alias_symbol;
    uint8_t alias_cutoff;
    uint16_t dist;
    uint16_t alias_offset;
    uint16_t alias_dist_xor;
} jxl_ans_bucket;

typedef struct {
    jxl_ans_bucket *buckets;
    uint32_t log_bucket_size;
    uint32_t bucket_mask;
    int single_symbol;
} jxl_ans_hist;

typedef struct {
    uint32_t split_exponent;
    uint32_t split;
    uint32_t msb_in_token;
    uint32_t lsb_in_token;
} jxl_int_config;

typedef struct {
    jxl_ctx *ctx;
    uint8_t *clusters;        /* num_dist entries -> cluster index */
    uint32_t num_dist;
    uint32_t num_clusters;
    jxl_int_config *configs;  /* per cluster */
    int use_prefix;
    jxl_pfx_hist *pfx;        /* per cluster (prefix codes) */
    jxl_ans_hist *ans;        /* per cluster (ANS) */
    uint32_t state;           /* ANS state */

    int lz77_enabled;
    uint32_t min_symbol;
    uint32_t min_length;
    jxl_int_config lz_len_conf;
    uint32_t *window;         /* 1 << 20 entries, allocated on first use */
    uint32_t num_to_copy;
    uint32_t copy_pos;
    uint32_t num_decoded;

    int err;
} jxl_dec;

int jxl_dec_init(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br, uint32_t num_dist);
void jxl_dec_free(jxl_dec *dec);
uint32_t jxl_dec_read(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx);
uint32_t jxl_dec_read_mult(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx,
                           uint32_t dist_multiplier);
int jxl_dec_finalize(jxl_dec *dec);

int jxl_read_clusters(jxl_ctx *ctx, jxl_br *br, uint32_t num_dist,
                      uint8_t *clusters, uint32_t *num_clusters_out);
int jxl_read_permutation(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br, uint32_t size,
                         uint32_t skip, uint32_t *out);

/* ===================================================================== */
/* frame -- frame header, restoration filters, table of contents          */
/* ===================================================================== */

typedef enum {
    JXL_FRAME_REGULAR = 0,
    JXL_FRAME_LF = 1,
    JXL_FRAME_REFERENCE_ONLY = 2,
    JXL_FRAME_SKIP_PROGRESSIVE = 3
} jxl_frame_type;

typedef enum {
    JXL_ENC_VARDCT = 0,
    JXL_ENC_MODULAR = 1
} jxl_encoding;

/* FrameHeader flags */
#define JXL_FF_NOISE                    0x01
#define JXL_FF_PATCHES                  0x02
#define JXL_FF_SPLINES                  0x10
#define JXL_FF_USE_LF_FRAME             0x20
#define JXL_FF_SKIP_ADAPTIVE_LF_SMOOTH  0x80

typedef enum {
    JXL_BLEND_REPLACE = 0,
    JXL_BLEND_ADD = 1,
    JXL_BLEND_BLEND = 2,
    JXL_BLEND_MULADD = 3,
    JXL_BLEND_MUL = 4
} jxl_blend_mode;

typedef struct {
    jxl_blend_mode mode;
    uint32_t alpha_channel;
    int clamp;
    uint32_t source;
} jxl_blending_info;

typedef struct {
    int enabled;
    float weights[3][2];
} jxl_gabor;

typedef struct {
    int enabled;
    uint32_t iters;
    float sharp_lut[8];
    float channel_scale[3];
    float quant_mul;
    float pass0_sigma_scale;
    float pass2_sigma_scale;
    float border_sad_mul;
    float sigma_for_modular;
} jxl_epf;

typedef struct {
    uint32_t num_passes;
    uint32_t num_ds;
    uint32_t shift[16];
    uint32_t downsample[8];
    uint32_t last_pass[8];
} jxl_passes;

typedef struct {
    jxl_frame_type frame_type;
    jxl_encoding encoding;
    uint64_t flags;
    int do_ycbcr;
    int encoded_color_channels;
    uint32_t jpeg_upsampling[3];
    uint32_t upsampling;
    uint32_t *ec_upsampling;      /* num_extra entries */
    uint32_t group_size_shift;
    uint32_t x_qm_scale, b_qm_scale;
    jxl_passes passes;
    uint32_t lf_level;
    int have_crop;
    int32_t x0, y0;
    uint32_t width, height;
    jxl_blending_info blending;
    jxl_blending_info *ec_blending;  /* num_extra entries */
    uint32_t duration;
    uint32_t timecode;
    int is_last;
    uint32_t save_as_reference;
    int resets_canvas;
    int save_before_ct;
    char *name;
    jxl_gabor gab;
    jxl_epf epf;
} jxl_frame_header;

typedef enum {
    JXL_TOC_ALL = 0,
    JXL_TOC_LF_GLOBAL,
    JXL_TOC_LF_GROUP,
    JXL_TOC_HF_GLOBAL,
    JXL_TOC_GROUP_PASS
} jxl_toc_kind;

typedef struct {
    size_t offset;     /* byte offset from the start of the frame header */
    uint32_t size;
} jxl_toc_entry;

typedef struct {
    jxl_toc_entry *entries;   /* in "original" (semantic) order */
    uint32_t count;
    uint32_t num_lf_groups;
    uint32_t num_groups;
    uint32_t num_passes;
    size_t total_size;        /* sum of all section sizes */
    size_t end_off;           /* byte offset just past the TOC */
} jxl_toc;

int jxl_read_frame_header(jxl_ctx *ctx, jxl_br *br, const jxl_size_header *size,
                          const jxl_image_metadata *meta, jxl_frame_header *fh);
void jxl_frame_header_free(jxl_ctx *ctx, jxl_frame_header *fh);
int jxl_read_toc(jxl_ctx *ctx, jxl_br *br, const jxl_frame_header *fh,
                 jxl_toc *toc);
void jxl_toc_free(jxl_ctx *ctx, jxl_toc *toc);
/* Index into toc->entries for a given section. */
uint32_t jxl_toc_index(const jxl_toc *toc, jxl_toc_kind kind, uint32_t pass_idx,
                       uint32_t group_idx);

uint32_t jxl_frame_sample_width(const jxl_frame_header *fh, uint32_t upsampling);
uint32_t jxl_frame_sample_height(const jxl_frame_header *fh, uint32_t upsampling);
uint32_t jxl_frame_color_width(const jxl_frame_header *fh);
uint32_t jxl_frame_color_height(const jxl_frame_header *fh);
uint32_t jxl_frame_group_dim(const jxl_frame_header *fh);
uint32_t jxl_frame_num_groups(const jxl_frame_header *fh);
uint32_t jxl_frame_num_lf_groups(const jxl_frame_header *fh);
uint32_t jxl_frame_groups_per_row(const jxl_frame_header *fh);
uint32_t jxl_frame_lf_groups_per_row(const jxl_frame_header *fh);

/* ===================================================================== */
/* doc -- top level state                                                 */
/* ===================================================================== */

struct jxl_doc {
    jxl_ctx *ctx;
    const uint8_t *data;
    size_t len;
    jxl_container container;

    jxl_size_header size;         /* codestream dimensions                */
    jxl_image_metadata meta;

    uint8_t *icc;                 /* decoded ICC profile, or NULL         */
    size_t icc_len;

    /* Byte offset (in container.cs) just past the image header, where the
       first frame starts. */
    size_t first_frame_off;
    size_t first_frame_bitpos;    /* bit offset of first frame            */
};

#endif /* JXL_INTERNAL_H */
