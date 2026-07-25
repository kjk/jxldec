# PROGRESS

Milestone log for the C JPEG XL decoder. Newest first.

## Status summary

`bun cmd/tests.ts -all` — 736 corpus files (73 libjxl testdata images × 10 cjxl
presets, plus the `.jxl` files that ship with libjxl):

```
607/736 ok, 129 failed
```

"ok" means byte-identical to `djxl file.jxl out.pam`, or within one 8-bit step
for the float (VarDCT) paths.

| area | state |
|---|---|
| container, headers, ICC | done, verified against `jxlinfo` |
| entropy coding (prefix, ANS, LZ77, context maps) | done |
| Modular lossless (all effort levels, palette, squeeze, RCT) | **byte-exact** |
| Modular lossy (XYB) | within 1 |
| VarDCT, 8-bit sRGB | within 1 |
| VarDCT, progressive (LF frames) | within 1 |
| gaborish, EPF | done |
| non-sRGB primaries (BT.2020, P3) | done (Bradford-adapted matrix) |
| images with an embedded ICC profile | **missing** — the enum fields are unused, so the conversion is wrong |
| patches / reference frames | **missing** — decodes the wrong image |
| splines, noise | **missing** — errors out |
| animation (frames after the first) | **missing** — only frame 0 |
| YCbCr / JPEG transcode frames | **missing** — errors out |

### What "pixel perfect" means here
Modular is integer arithmetic, so it is byte-identical to libjxl and any
mismatch is a bug. VarDCT is float: libjxl, jxl-oxide and this decoder all use
different IDCT factorizations and different `powf` approximations, so exact
equality is not achievable and libjxl's own conformance testing uses a
tolerance. Our target there is max abs difference ≤ 1 on 8-bit output, which
is met for sRGB content.

## Known-cause failures

1. **ICC-profile images** (`*_v4_krita`, `*_acescg_*`): when `want_icc` is set
   the enumerated primaries/transfer fields are absent and the real color space
   lives in the embedded profile. We decode the profile bytes but do not
   interpret them, so the color transform uses sRGB defaults.
   Enumerated non-sRGB primaries (BT.2020, P3, DCI) *are* handled:
   `jxl_opsin_matrix_for` folds the sRGB→target conversion (Bradford-adapted
   through XYZ D50) into the opsin inverse matrix, matching libjxl's
   `OutputEncodingInfo::SetFromMetadata`. Linear-transfer 16-bit images can
   still differ by a few counts at gamut edges, where out-of-range values clip.
2. **Patches** (`ellipses`, `grayscale_patches` at effort ≥ 7): the encoder
   emits a `ReferenceOnly` frame plus a main frame with the patches flag. We
   decode and store the reference frame but do not blit patches, and we bail
   out on the patches flag.
3. **Splines / noise**: `frame: patches/splines/noise are not implemented yet`.

## Log

### VarDCT + multi-frame
- Frame iteration now decodes non-displayed frames (LF, reference-only) for
  their side effects and returns the requested *displayed* frame. LF frames
  (progressive DC) feed the next frame's LF image directly, skipping CfL and
  adaptive smoothing.
- Bug found: `jxl_read_toc` never called `jxl_dec_begin` after parsing the
  permutation decoder, so every permuted TOC (which is what `--progressive`
  produces) decoded to garbage offsets.
- Lossy Modular stores XYB as `(Y, X, B)` integers that must be scaled by the
  LF dequant weights (`m_*_lf / 128`), with `B += Y` first — not as [0,1]
  samples.

### VarDCT core
- Ported quantizer, dequant matrices (all 7 encoding modes incl. raw Modular
  matrices), HF block contexts, coefficient orders, HF coefficient decoding,
  per-varblock dequant, chroma-from-luma, and all 27 inverse transforms.
- The `natural order` generator keeps libjxl's `x < lbw && y < lbw` quirk
  (`lbw` twice, not `lbh`) — "fixing" it breaks large transforms.
- libjxl's sRGB encode is a rational-polynomial approximation, not `powf`;
  `color.c` ports it so the 8-bit output rounds the same way.

### Modular
- Byte-exact against `djxl` for lossless at effort 1/3/7/9, single- and
  multi-group, grayscale/RGB/RGBA, 8- and 16-bit, and `--responsive=1`.
- Two bugs worth remembering: the ANS end-of-stream signature is `0x130000`
  (not `0x13000`), and MA tree leaves store an *already clustered* index, so
  reading them must bypass the context→cluster map.

### Scaffolding
- djvudec-style layout: one internal header, `cmd/*.ts` build/test scripts,
  `dist/` amalgamation, MSVC + clang builds, ASan harness.
- The corpus is generated: `cmd/png.ts` decodes libjxl's testdata PNGs in
  TypeScript (libpng is deliberately not built into the oracle), `cmd/corpus.ts`
  encodes them with `cjxl` across 10 presets.
