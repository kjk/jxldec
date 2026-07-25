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
/* A length-prefixed UTF-8 name; NULL when empty. Caller frees. */
char *jxl_read_name(jxl_ctx *ctx, jxl_br *br);
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
void jxl_dec_begin(jxl_dec *dec, jxl_br *br);
void jxl_dec_free(jxl_dec *dec);
uint32_t jxl_dec_read(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx);
uint32_t jxl_dec_read_mult(jxl_dec *dec, jxl_br *br, uint32_t ctx_idx,
                           uint32_t dist_multiplier);
/* Same, but the caller already resolved the context to a cluster (the MA
   tree stores cluster indices directly in its leaves). */
uint32_t jxl_dec_read_clustered(jxl_dec *dec, jxl_br *br, uint32_t cluster,
                                uint32_t dist_multiplier);
int jxl_dec_finalize(jxl_dec *dec);

int jxl_read_clusters(jxl_ctx *ctx, jxl_br *br, uint32_t num_dist,
                      uint8_t *clusters, uint32_t *num_clusters_out);
int jxl_read_permutation(jxl_ctx *ctx, jxl_dec *dec, jxl_br *br, uint32_t size,
                         uint32_t skip, uint32_t *out);

/* ===================================================================== */
/* modular -- MA trees, predictors, transforms, channel decoding           */
/* ===================================================================== */

typedef struct {
    uint32_t p1, p2, p3a, p3b, p3c, p3d, p3e;
    uint32_t w0, w1, w2, w3;
} jxl_wp_header;

/* Predictor ids, as encoded in MA tree leaves. */
typedef enum {
    JXL_PRED_ZERO = 0,
    JXL_PRED_WEST,
    JXL_PRED_NORTH,
    JXL_PRED_AVG_W_N,
    JXL_PRED_SELECT,
    JXL_PRED_GRADIENT,
    JXL_PRED_SELF_CORRECTING,
    JXL_PRED_NORTH_EAST,
    JXL_PRED_NORTH_WEST,
    JXL_PRED_WEST_WEST,
    JXL_PRED_AVG_W_NW,
    JXL_PRED_AVG_N_NW,
    JXL_PRED_AVG_N_NE,
    JXL_PRED_AVG_ALL
} jxl_predictor;

typedef struct {
    int32_t property;     /* -1 marks a leaf */
    int32_t value;
    uint32_t left, right;
    uint8_t cluster;
    uint8_t predictor;
    int32_t offset;
    uint32_t multiplier;
} jxl_ma_node;

typedef struct {
    jxl_ma_node *nodes;
    uint32_t count;
    uint32_t root;
    jxl_dec dec;          /* histograms for the sample stream */
    int valid;
} jxl_ma_config;

int jxl_ma_config_read(jxl_ctx *ctx, jxl_br *br, jxl_ma_config *ma,
                       size_t node_limit);
void jxl_ma_config_free(jxl_ctx *ctx, jxl_ma_config *ma);

/* A channel: a view (pointer + stride) into some owning buffer. */
typedef struct {
    int32_t *data;
    size_t stride;         /* in int32_t units */
    uint32_t w, h;
    int hshift, vshift;
    uint32_t ow, oh;       /* size before the shift was applied */
} jxl_mchan;

typedef struct {
    int horizontal, in_place;
    uint32_t begin_c, num_c;
} jxl_sq_param;

typedef struct {
    int kind;              /* 0 = RCT, 1 = palette, 2 = squeeze */
    uint32_t begin_c;
    uint32_t rct_type;
    uint32_t num_c, nb_colours, nb_deltas;
    uint8_t d_pred;
    jxl_sq_param *sp;
    uint32_t nsp;
    /* palette runtime state */
    int32_t *pal_buf;      /* nb_colours * num_c, owned */
    jxl_mchan pal;
    jxl_mchan *saved;      /* member channels removed from the list */
    uint32_t nsaved;
} jxl_transform;

typedef struct {
    jxl_mchan *chans;
    uint32_t n;
    uint32_t cap;
    uint32_t nb_meta;
} jxl_chanlist;

typedef struct {
    int use_global_tree;
    jxl_wp_header wp;
    uint32_t ntransforms;
    jxl_transform *transforms;
} jxl_modular_header;

/* Declared channel geometry, before any transform. */
typedef struct {
    uint32_t w, h;
    int hshift, vshift;
} jxl_mchan_spec;

typedef struct {
    jxl_ctx *ctx;
    jxl_modular_header header;
    jxl_ma_config *ma;         /* global config, or &local */
    jxl_ma_config local;
    int has_local;
    uint32_t group_dim;
    uint32_t bit_depth;

    jxl_mchan *base;           /* declared channels, each owning a buffer */
    uint32_t nbase;
    int32_t **bufs;
    uint32_t nbufs;
} jxl_modular;

/* Reads the modular header (transforms + optional local MA tree) and
   allocates the declared channels. nspecs == 0 makes an empty image. */
int jxl_modular_init(jxl_ctx *ctx, jxl_modular *m, jxl_br *br,
                     const jxl_mchan_spec *specs, uint32_t nspecs,
                     jxl_ma_config *global_ma, uint32_t group_dim,
                     uint32_t bit_depth);
/* Same, but over channel views that already exist (a group's sub-stream). */
int jxl_modular_init_over(jxl_ctx *ctx, jxl_modular *m, jxl_br *br,
                          const jxl_mchan *chans, uint32_t nchans,
                          jxl_ma_config *global_ma, uint32_t group_dim,
                          uint32_t bit_depth);
void jxl_modular_free(jxl_ctx *ctx, jxl_modular *m);

/* Applies every transform to the channel list, producing the list of
   channels that are actually coded. */
int jxl_modular_transform_channels(jxl_ctx *ctx, jxl_modular *m,
                                   jxl_chanlist *cl);
/* Undoes every transform, writing back into the declared channels. */
int jxl_modular_inverse(jxl_ctx *ctx, jxl_modular *m, jxl_chanlist *cl);

/* Decodes one modular stream into the given channel list. */
int jxl_modular_decode(jxl_ctx *ctx, jxl_modular *m, jxl_chanlist *cl,
                       jxl_br *br, uint32_t stream_idx);

void jxl_chanlist_free(jxl_ctx *ctx, jxl_chanlist *cl);
int jxl_chanlist_push(jxl_ctx *ctx, jxl_chanlist *cl, const jxl_mchan *ch);

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
/* The VarDCT block grid, rounded up to a whole number of subsampled blocks. */
uint32_t jxl_frame_blocks_w(const jxl_frame_header *fh);
uint32_t jxl_frame_blocks_h(const jxl_frame_header *fh);

/* ===================================================================== */
/* dct                                                                    */
/* ===================================================================== */

/* In-place separable 2D DCT over a w x h block (stride in floats).
   inverse == 0 is the analysis transform, 1 the synthesis transform. */
void jxl_dct_2d(float *data, size_t stride, int w, int h, int inverse);
float jxl_scale_f(int c, int logb);

/* ===================================================================== */
/* vardct                                                                 */
/* ===================================================================== */

/* Varblock transform types, in bitstream order. */
typedef enum {
    JXL_TR_DCT8 = 0, JXL_TR_HORNUSS, JXL_TR_DCT2, JXL_TR_DCT4, JXL_TR_DCT16,
    JXL_TR_DCT32, JXL_TR_DCT16X8, JXL_TR_DCT8X16, JXL_TR_DCT32X8,
    JXL_TR_DCT8X32, JXL_TR_DCT32X16, JXL_TR_DCT16X32, JXL_TR_DCT4X8,
    JXL_TR_DCT8X4, JXL_TR_AFV0, JXL_TR_AFV1, JXL_TR_AFV2, JXL_TR_AFV3,
    JXL_TR_DCT64, JXL_TR_DCT64X32, JXL_TR_DCT32X64, JXL_TR_DCT128,
    JXL_TR_DCT128X64, JXL_TR_DCT64X128, JXL_TR_DCT256, JXL_TR_DCT256X128,
    JXL_TR_DCT128X256, JXL_TR_COUNT
} jxl_transform_type;

void jxl_tr_select_size(int tr, uint32_t *bw, uint32_t *bh);
void jxl_tr_matrix_size(int tr, uint32_t *w, uint32_t *h);
int jxl_tr_matrix_index(int tr);
int jxl_tr_order_id(int tr);
int jxl_tr_need_transpose(int tr);

typedef struct {
    uint32_t global_scale;
    uint32_t quant_lf;
} jxl_quantizer;

typedef struct {
    uint32_t colour_factor;
    float base_correlation_x, base_correlation_b;
    uint32_t x_factor_lf, b_factor_lf;
} jxl_lf_chan_corr;

typedef struct {
    uint32_t nqf;
    uint32_t qf_thresholds[16];
    uint32_t nlf[3];
    int32_t lf_thresholds[3][16];
    uint8_t *block_ctx_map;
    uint32_t block_ctx_map_len;
    uint32_t num_block_clusters;
} jxl_hf_block_ctx;

/* Slots are materialized on demand. The largest is 256x256 floats per
   channel, and a frame typically uses a handful of transform sizes, so
   building all 17 up front dominates the decode of a small image. */
struct jxl_dq_encoding;
typedef struct {
    float *matrix[17][3];
    float *matrix_tr[17][3];
    struct jxl_dq_encoding *enc;   /* 17 parsed, not-yet-built encodings */
} jxl_dequant_matrices;

typedef struct {
    uint16_t *order[13][3];    /* NULL means "use the natural order" */
    jxl_dec dist;
    int have_dist;
} jxl_hf_pass;

/* Per-8x8-block varblock state. dct_select is JXL_BLK_* or a transform id. */
#define JXL_BLK_UNINIT   0xff
#define JXL_BLK_OCCUPIED 0xfe

typedef struct {
    uint8_t dct_select;
    int32_t hf_mul;
} jxl_block_info;

typedef struct {
    int32_t *x_from_y, *b_from_y;
    uint32_t cfl_w, cfl_h;
    jxl_block_info *block_info;
    uint32_t bw, bh;
    float *epf_sigma;
    int have;
} jxl_hf_meta;

/* Natural (zig-zag-ish) coefficient orders, materialized on demand. */
typedef struct {
    uint16_t *order[13];
} jxl_natural_orders;

const uint16_t *jxl_natural_order(jxl_ctx *ctx, jxl_natural_orders *no,
                                  int order_id);
void jxl_natural_orders_free(jxl_ctx *ctx, jxl_natural_orders *no);

void jxl_quantizer_read(jxl_br *br, jxl_quantizer *q);
/* xyb: whether the frame is XYB-encoded, which decides the default Y-to-B
   correlation. */
void jxl_lf_chan_corr_read(jxl_br *br, jxl_lf_chan_corr *c, int xyb);
/* Per-channel subsampling shifts implied by the frame's jpeg_upsampling. */
void jxl_jpeg_upsampling_shifts(const uint32_t ju[3], int idx, int *hs,
                                int *vs);
/* Doubles a subsampled chroma plane in place along the requested axes. */
void jxl_chroma_upsample(float *p, uint32_t w, uint32_t h, size_t stride,
                         int hs, int vs, uint32_t out_w, uint32_t out_h);
/* (Cb, Y, Cr) -> (R, G, B), in place. */
void jxl_ycbcr_to_rgb(float *cb, float *y, float *cr, size_t n);
int jxl_hf_block_ctx_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_block_ctx *bc);
void jxl_hf_block_ctx_free(jxl_ctx *ctx, jxl_hf_block_ctx *bc);

int jxl_dequant_matrices_read(jxl_ctx *ctx, jxl_br *br,
                              jxl_dequant_matrices *dm, uint32_t bit_depth,
                              uint32_t num_lf_groups,
                              jxl_ma_config *global_ma);
void jxl_dequant_matrices_free(jxl_ctx *ctx, jxl_dequant_matrices *dm);
/* Builds the slot serving transform `tr`, if it is not built already. */
int jxl_dequant_matrices_ensure(jxl_ctx *ctx, jxl_dequant_matrices *dm, int tr);

int jxl_hf_pass_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_pass *hp,
                     const jxl_hf_block_ctx *bc, uint32_t num_hf_presets,
                     jxl_natural_orders *no);
void jxl_hf_pass_free(jxl_ctx *ctx, jxl_hf_pass *hp);

/* Decodes one pass group's HF coefficients into three float planes (the
   quantized values are stored as int32 bit patterns). */
typedef struct {
    uint32_t num_hf_presets;
    const jxl_hf_block_ctx *bc;
    const jxl_block_info *block_info;
    uint32_t bi_w, bi_h;
    size_t bi_stride;
    uint32_t jpeg_upsampling[3];
    const jxl_mchan *lf_quant[3];   /* NULL when there is no LF image */
    jxl_hf_pass *pass;
    uint32_t coeff_shift;
    jxl_natural_orders *no;
} jxl_hf_coeff_params;

int jxl_write_hf_coeff(jxl_ctx *ctx, jxl_br *br,
                       const jxl_hf_coeff_params *params, float *out[3],
                       size_t stride[3]);

void jxl_hf_meta_free(jxl_ctx *ctx, jxl_hf_meta *m);
int jxl_hf_meta_read(jxl_ctx *ctx, jxl_br *br, jxl_hf_meta *m,
                     uint32_t num_lf_groups, uint32_t lf_group_idx,
                     uint32_t lf_width, uint32_t lf_height,
                     const uint32_t jpeg_upsampling[3], uint32_t bit_depth,
                     jxl_ma_config *global_ma, const jxl_epf *epf,
                     uint32_t quantizer_global_scale);

void jxl_copy_lf_dequant(float *dst, size_t dstride, const jxl_mchan *src,
                         const jxl_quantizer *q, float m_lf,
                         int extra_precision);
int jxl_adaptive_lf_smoothing(jxl_ctx *ctx, float *plane[3], uint32_t width,
                              uint32_t height, size_t stride,
                              const float m_lf[3], const jxl_quantizer *q);
void jxl_cfl_lf(float *x, float *y, float *b, uint32_t w, uint32_t h,
                size_t stride, const jxl_lf_chan_corr *corr);
void jxl_cfl_hf(float *cx, float *cy, float *cb, size_t stride, uint32_t gw,
                uint32_t gh, const int32_t *x_from_y, const int32_t *b_from_y,
                uint32_t cfl_stride, const jxl_lf_chan_corr *corr);
void jxl_dequant_varblock(float *coeff, size_t stride, int tr, int32_t hf_mul,
                          int channel, const jxl_dequant_matrices *dm,
                          const jxl_quantizer *q, float qm_scale,
                          float quant_bias, float quant_bias_numerator);
void jxl_transform_varblock(float *coeff, size_t stride, int tr);
void jxl_fill_varblock_lf(float *coeff, size_t stride, int tr,
                          const float *lf, size_t lf_stride, uint32_t lf_x,
                          uint32_t lf_y);

/* ===================================================================== */
/* filter -- gaborish and the edge-preserving filter                      */
/* ===================================================================== */

int jxl_apply_gabor(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                    size_t stride, const float weights[3][2]);
int jxl_apply_epf(jxl_ctx *ctx, float *plane[3], uint32_t w, uint32_t h,
                  size_t stride, const float *sigma, uint32_t sigma_stride,
                  const jxl_epf *epf);

/* ===================================================================== */
/* color -- XYB and transfer functions                                    */
/* ===================================================================== */

void jxl_xyb_to_linear(float *x, float *y, float *b, size_t n,
                       const float opsin_inv[9], const float opsin_bias[3],
                       float intensity_target);
void jxl_linear_to_tf(float *v, size_t n, const jxl_colour_encoding *enc,
                      float intensity_target);
/* The opsin inverse matrix adjusted for the image's primaries/white point
   (and collapsed to luminance weights for grayscale output). */
void jxl_opsin_matrix_for(const jxl_image_metadata *meta, float out[9]);

/* ===================================================================== */
/* patch -- rectangles blended in from a reference frame                  */
/* ===================================================================== */

typedef enum {
    JXL_PATCH_NONE = 0,
    JXL_PATCH_REPLACE,
    JXL_PATCH_ADD,
    JXL_PATCH_MUL,
    JXL_PATCH_BLEND_ABOVE,
    JXL_PATCH_BLEND_BELOW,
    JXL_PATCH_MULADD_ABOVE,
    JXL_PATCH_MULADD_BELOW
} jxl_patch_mode;

typedef struct {
    uint8_t mode;
    uint32_t alpha_channel;
    int clamp;
} jxl_patch_blend;

typedef struct {
    int32_t x, y;
    jxl_patch_blend *blending;   /* num_extra + 1 entries */
} jxl_patch_target;

typedef struct {
    uint32_t ref_idx;
    uint32_t x0, y0, width, height;
    jxl_patch_target *targets;
    uint32_t ntargets;
} jxl_patch_ref;

typedef struct {
    jxl_patch_ref *refs;
    uint32_t n;
    uint32_t nblend;
} jxl_patches;

int jxl_patches_read(jxl_ctx *ctx, jxl_br *br, const jxl_image_metadata *meta,
                     const jxl_frame_header *fh, jxl_patches *out);
void jxl_patches_free(jxl_ctx *ctx, jxl_patches *p);

/* ===================================================================== */
/* spline -- smooth colored strokes drawn over the frame                  */
/* ===================================================================== */

typedef struct {
    int64_t *px, *py;
    uint32_t npoints;
    int32_t xyb_dct[3][32];
    int32_t sigma_dct[32];
} jxl_quant_spline;

typedef struct {
    jxl_quant_spline *splines;
    uint32_t n;
    int32_t quant_adjust;
} jxl_splines;

int jxl_splines_read(jxl_ctx *ctx, jxl_br *br, const jxl_frame_header *fh,
                     jxl_splines *out);
void jxl_splines_free(jxl_ctx *ctx, jxl_splines *sp);

/* ===================================================================== */
/* noise -- regenerated photon/film noise                                 */
/* ===================================================================== */

typedef struct {
    float lut[8];
} jxl_noise_params;

int jxl_noise_params_read(jxl_br *br, jxl_noise_params *np);

/* ===================================================================== */
/* decode -- frame decoding into float planes                             */
/* ===================================================================== */

typedef struct {
    float *data;
    uint32_t w, h;
    size_t stride;      /* in floats */
} jxl_fplane;

/* One decoded frame: color planes first, then the extra channels. Planes
   can differ in size when a channel is subsampled. */
typedef struct {
    jxl_ctx *ctx;
    jxl_fplane *plane;
    uint32_t nplane;
    uint32_t ncolor;
    uint32_t w, h;      /* the frame's color-plane dimensions */
} jxl_fimage;

int jxl_fimage_alloc(jxl_ctx *ctx, jxl_fimage *img, uint32_t nplane);
int jxl_fplane_alloc(jxl_ctx *ctx, jxl_fplane *p, uint32_t w, uint32_t h);
void jxl_fimage_free(jxl_ctx *ctx, jxl_fimage *img);

/* State carried across the frames of one document: the reference slots a
   frame can blend from, and the low-resolution image an LF frame leaves
   behind for the next frame to use as its LF. */
typedef struct {
    jxl_fimage refs[4];
    int refs_valid[4];
    jxl_fimage lf_image;
    int lf_valid;
    /* Noise is seeded from how many frames have been shown so far. */
    uint32_t visible_frames;
    uint32_t invisible_frames;
} jxl_frame_state;

void jxl_frame_state_free(jxl_ctx *ctx, jxl_frame_state *st);

int jxl_apply_patches(jxl_ctx *ctx, jxl_fimage *img, const jxl_patches *p,
                      const jxl_image_metadata *meta, jxl_fimage refs[4],
                      const int refs_valid[4]);
int jxl_blend_frame(jxl_ctx *ctx, jxl_fimage *canvas, const jxl_fimage *frame,
                    const jxl_frame_header *fh, const jxl_image_metadata *meta);
int jxl_fimage_blank_like(jxl_ctx *ctx, jxl_fimage *out, const jxl_fimage *like,
                          uint32_t w, uint32_t h);
int jxl_fimage_copy(jxl_ctx *ctx, jxl_fimage *dst, const jxl_fimage *src);
int jxl_render_splines(jxl_ctx *ctx, jxl_fimage *img, const jxl_splines *sp,
                       const jxl_frame_header *fh, float corr_x, float corr_b);
int jxl_render_noise(jxl_ctx *ctx, jxl_fimage *img, const jxl_noise_params *np,
                     const jxl_frame_header *fh, uint32_t visible_frames,
                     uint32_t invisible_frames, float corr_x, float corr_b);

/* apply_ct == 0 leaves the frame in its pre-color-transform (XYB) form, which
   is what a frame saved "before CT" -- every LF frame -- must store. */
int jxl_frame_decode(jxl_ctx *ctx, jxl_doc *doc, const jxl_frame_header *fh,
                     const jxl_toc *toc, jxl_frame_state *st, int apply_ct,
                     jxl_fimage *out);

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
    int frame_count;              /* displayed frames; 0 until counted    */
};

#endif /* JXL_INTERNAL_H */
