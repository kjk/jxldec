# PROGRESS

Milestone log for the C JPEG XL decoder. Newest first.

## Status summary

`bun cmd/tests.ts -all` — 821 corpus files (libjxl's testdata images × 11 cjxl
presets, its JPEGs transcoded, plus the `.jxl` files that ship with libjxl):

```
758/821 ok, 63 failed
```

"ok" means byte-identical to `djxl file.jxl out.pam`, or within one 8-bit step
for the float (VarDCT) paths. **No file fails to decode**: every one of the 63
is a lossy VarDCT path differing by 2-10 counts out of 255 on isolated pixels,
which is float divergence, not a bug (see below).

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
| images with an embedded ICC profile | done |
| patches / reference frames | **byte-exact** |
| splines, noise | done (noise is approximate, see below) |
| animation (frames after the first) | done |
| YCbCr / JPEG transcode frames | done, all subsampling modes |

### What "pixel perfect" means here
Modular is integer arithmetic, so it is byte-identical to libjxl and any
mismatch is a bug. VarDCT is float: libjxl, jxl-oxide and this decoder all use
different IDCT factorizations and different `powf` approximations, so exact
equality is not achievable and libjxl's own conformance testing uses a
tolerance. Our target there is max abs difference ≤ 1 on 8-bit output, which
is met for sRGB content.

## Remaining differences

1. **Residual VarDCT float divergence** — 63 files, max 2-10 counts on
   scattered pixels, mean under 0.3. Inherent to the differing IDCT
   factorizations; not worth chasing further.
2. **Noise** reproduces approximately (max 5, mean 0.47 on a photon-noise
   file). The XorShift128+ RNG and the 5x5 high-pass match jxl-oxide's
   structure but a small difference remains, uninvestigated.

## Performance

`bun cmd/bench.ts` links the `dist/` amalgamation and libjxl's static
libraries into one process and times both, single-threaded. On a random
sample we are ~3.7x libjxl overall: Modular 2-3x, VarDCT 1-5x with occasional
outliers. libjxl is AVX2; we are scalar C, so a small constant factor is
expected. Splines are the slowest path by a wide margin.

## Log

### YCbCr, ICC, and lazy dequant matrices
- JPEG-transcoded files (YCbCr frames) decode, including every chroma
  subsampling mode. The subtleties: the block grid rounds up to a whole
  number of subsampled blocks, so LF group rects are measured in blocks, not
  pixels; a subsampled channel lives in the top-left corner of each plane, so
  the per-group coefficient and LF-quant *views* start at the shifted block
  origin (getting this wrong decodes plausibly but wrongly, mean 4/255);
  chroma-from-luma is skipped entirely when anything is subsampled.
- `want_icc` images output **linear sRGB**. Recovering the real primaries and
  transfer curve from the profile is colorimetrically right but is not what
  libjxl does — without a CMS it falls back to linear sRGB and hands the
  profile to the caller. Matching libjxl took eight test profiles from
  max diff 74-169 to max 1.
- Grayscale XYB leaves three identical planes; dropping two is required or
  the extra-channel plane indices are off by two and alpha reads luma.
- Dequant matrices are built on demand. Materializing all 17 slots (including
  256x256) cost 8ms on files libjxl decodes in 0.2ms.

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
