# AGENTS.md — working on the C JPEG XL decoder

Plain-C, decode-only JPEG XL (`.jxl`) library, ported from **jxl-oxide** (Rust)
and verified pixel-for-pixel against **libjxl**. API style follows
**jbig2dec** / the sibling project **djvudec** (`../../djvudec`).

## Goal / scope
Decode-only, single-threaded. The caller hands us the entire `.jxl` file
up-front as an in-memory buffer. We provide:
- signature detection + image info (size, bit depth, channels, animation)
- decoded pixels for any frame, in 8- or 16-bit gray/gray+alpha/RGB/RGBA
- the embedded ICC profile, when present

No encoder. No JPEG reconstruction (`jbrd` boxes are located but not applied).
No multithreading — `jxl_ctx` is not shared across threads.

The consumer that drives the feature set is **SumatraPDF**
(`C:\Users\kjk\src\sumatrapdf`, `src/JxlReader.cpp`): it needs
`JxlSignatureCheck` (→ `jxl_signature_check`), basic info for a size query
(→ `jxl_decode_size`), and the first frame as RGBA8 (→ `jxl_frame_render` +
`jxl_ctx_set_bgr` so no swizzle pass is needed).

## Reference checkouts (local)
- `deps/libjxl` — the reference C++ decoder AND the oracle. `deps/libjxl-build`
  holds cjxl/djxl/jxlinfo built by `cmd/get-deps.ts` (cmake + ninja +
  clang-cl). Consult `lib/jxl/*.cc` for exact tables and constants.
- `deps/jxl-oxide` — Rust decoder; the implementation this port follows most
  closely. Its crate layout maps almost 1:1 onto our `src/` modules.
- `JXLDEC_LIBJXL=<dir>` points at an existing libjxl checkout instead of
  cloning (a directory junction at `deps/libjxl` works too).
- Spec: ISO/IEC 18181-1. The **code** is the more definitive reference.

`deps/` is gitignored.

### libpng is deliberately not built
libjxl's bundled-libpng cmake glue can't find the bundled zlib's generated
`zconf.h`, so `-DJPEGXL_BUNDLE_LIBPNG=OFF`. cjxl and djxl read/write PNM
natively, which is all the oracle needs; `cmd/png.ts` decodes the testdata
PNGs to PNM in TypeScript. Don't re-investigate this.

## Build & test
- `bun cmd/build.ts` — fetch deps, build the decoder + `jxl_test`. **MSVC is
  the default on Windows** (`out/msvc/jxl_test_msvc.exe`); `-clang` builds
  with clang (`out/clang/jxl_test_clang.exe`). `-clean` wipes `out/`.
  `bun cmd/build.ts asan` builds the clang+ASan harness.
- `bun cmd/tests.ts <-all | -rand N | file.jxl ...>` — the test driver:
  builds, then decodes each corpus file with both our decoder and `djxl`
  and compares the PNM output.
- `bun cmd/bench.ts <selection> [-runs N]` — times our decode against libjxl's,
  both in one process, single-threaded, best of N. Windows/MSVC only: it links
  libjxl's static libraries, which need `-MD` and `-DJXL_STATIC_DEFINE`.
- `bun cmd/prof.ts <file.jxl> [-runs N]` — sampling profile of *our* decoder
  alone on one file, via the sibling `../samply` (build it once with
  `cd ../samply && bun cmd/build.ts -release`). It builds `jxl_prof` with
  `-Zi` into `out/prof/` and passes samply `-print-agent`, which prints the
  top self-time functions, the hot source lines and the heaviest call path.
  Needs Administrator rights and `xperf.exe` from the Windows ADK.
- `bun cmd/build-dist.ts` — regenerate the `dist/jxl.c` + `dist/jxl.h`
  amalgamation and verify it compiles with every available toolchain.

**Script convention** (from djvudec): every script that operates on `.jxl`
files does nothing by default — it prints its options plus the available
corpus file count. Select work explicitly: `file.jxl ...`, `-rand N`, `-all`.

## Corpus
libjxl's `testdata/` holds only a handful of `.jxl` files; the corpus is
*generated* by encoding the testdata source images with `cjxl` across a matrix
of settings (VarDCT/modular, lossless/lossy, progressive, effort levels), cached
under `deps/corpus/`. See `cmd/corpus.ts`. `JXL_SPECS=<dir>` overrides the
corpus with any directory of `.jxl` files.

## Source map (`src/` → jxl-oxide crate)
| C file | jxl-oxide |
|---|---|
| `core.c` | (context/allocator plumbing; no counterpart) |
| `bitread.c` | `jxl-bitstream/src/bitstream.rs` |
| `container.c` | `jxl-bitstream/src/container/*` |
| `headers.c` | `jxl-image/src/lib.rs`, `color.rs` |
| `icc.c` | `jxl-color/src/icc/decode.rs` |
| `coding.c` | `jxl-coding/*` (prefix, ans, permutation, lib) |
| `frame.c` | `jxl-frame/src/header.rs`, `filter.rs`, `data/toc.rs` |
| `modular.c` | `jxl-modular/*` (ma, predictor, transform, image) |
| `vardct.c` | `jxl-frame/src/data/{lf_global,lf_group,hf_global,pass_group}.rs` |
| `dct.c` | `jxl-vardct` transforms |
| `filter.c` | gabor/EPF application |
| `color.c` | `jxl-color/src/{xyb,tf,convert}.rs` |
| `render.c` | `jxl-render` (frame blending, upsampling, output) |
| `doc.c` | `jxl-oxide/src/lib.rs` (top-level API) |
| `tables.c` | constant tables (upsampling kernels, quant weights) |

All internal declarations live in the single `src/jxl_internal.h`, one labeled
section per module. Every source file includes just that one header.

## Format cheat-sheet
- **Container**: bare codestream starts `FF 0A`; the ISOBMFF form starts with
  the 12-byte JXL signature box, and the codestream is in one `jxlc` box or
  split across `jxlp` boxes (each with a 4-byte index prefix; high bit = last).
- **Codestream header order**: `FF 0A`, SizeHeader, ImageMetadata,
  **CustomTransformData**, [ICC], ZeroPadToByte, frames. CustomTransformData
  (opsin matrix + custom upsampling weights) is a *separate* bundle read even
  when ImageMetadata says "all default" — easy to get wrong.
- **Bit order**: LSB-first within each byte; `u(n)` assembles LSB-first.
- **U32(a,b,c,d)**: a 2-bit selector picks one of four `offset + u(nbits)`
  alternatives. `U64` has its own escalating encoding.
- **Entropy coding**: a distribution set = context→cluster map + one
  hybrid-uint config per cluster + one histogram per cluster (prefix codes OR
  ANS, never mixed) + optional LZ77. The ANS alias table must be built with
  LIFO underfull/overfull stacks exactly like libjxl's `InitAliasTable`.
  ANS streams end with state == `0x130000`.
- **Frame**: header + TOC. TOC sections are LfGlobal, LfGroup*, HfGlobal,
  then GroupPass* — unless there is a single group and a single pass, in which
  case there is exactly one section covering everything.
  Our `jxl_toc` stores **absolute byte offsets into the codestream buffer**.

## Frames
A codestream is a sequence of frames. Only *displayed* frames (normal type,
and either `is_last` or a non-zero duration) are visible; LF frames
(progressive DC) and reference-only frames are decoded for their side effects.
`decode_frame_planes` in `render.c` walks the chain, keeping a
`jxl_frame_state` with four reference slots and the most recent LF image.
Frames saved "before CT" (every LF frame) skip the XYB -> color-space step.

## Methodology
Reference-oracle verification: decode with `jxl_test -out x.pam` and with
`djxl file.jxl x.pam`, compare. Modular (integer) paths must be **byte-exact**;
VarDCT paths are float, so they are gated on RMS **and** peak together (`-rms`
0.6, `-tol` 3, in 8-bit steps), the shape libjxl's own conformance checker
uses -- libjxl, jxl-oxide and this decoder use different IDCT factorizations
and different `powf` approximations, so exact equality is not achievable. Do
not gate on peak alone: one sample then decides a megapixel file, and the
resulting noise hid a real loop-filter edge bug for a long time. When a
tolerance failure looks like float divergence, check *where* the outliers are
before concluding it -- clustering by column, row or block position is what
distinguishes a bug from rounding. Work incrementally and keep `PROGRESS.md`
current; it lists the exact pass rate and every known gap.

### Gotchas already paid for (do not re-discover)
- The ANS end-of-stream signature is `0x130000`.
- MA tree leaves hold an already-clustered index; reading them must bypass the
  context->cluster map (`jxl_dec_read_clustered`).
- Every entropy stream needs `jxl_dec_begin` after `jxl_dec_init` -- missing it
  on the TOC permutation decoder silently corrupts every permuted TOC, which
  is what `--progressive` produces.
- `CustomTransformData` is read even when ImageMetadata says "all default".
- The natural-order generator keeps libjxl's `x < lbw && y < lbw` quirk (`lbw`
  twice, not `lbh`).
- libjxl encodes sRGB with a rational-polynomial approximation, not `powf`.
- Lossy Modular stores XYB as `(Y, X, B)` integers scaled by `m_*_lf / 128`,
  with `B += Y` applied first.
- `want_icc` images decode to **linear sRGB**, not to the profile's real color
  space. That is what libjxl does without a CMS; reconstructing the primaries
  from the profile makes the output *less* like `djxl`, not more.
- Grayscale XYB produces three identical planes. Two must be dropped, or the
  extra-channel plane indices are off by two and alpha reads the luma plane.
- With chroma subsampling the block grid rounds up to a whole number of
  subsampled blocks (`jxl_frame_blocks_w/h`), and LF group rects are measured
  in **blocks**, not pixels.
- A subsampled channel occupies the top-left corner of its full-size plane, so
  every per-group view into the coefficient and LF-quant planes must start at
  the *shifted* block origin. Getting this wrong still decodes -- it just
  produces a plausible, wrong image (mean 4/255).
- Chroma-from-luma is skipped entirely when anything is subsampled, and the
  default Y-to-B correlation is 1.0 only for XYB frames (0 otherwise).
- Public names are `JXLDEC_`-prefixed (`JXLDEC_FORMAT_RGB24`, `JXLDEC_SIG_*`)
  precisely so `jxl.h` and libjxl's `jxl/decode.h` can share a translation
  unit -- `test/jxl_bench.c` relies on that. Internal-only uppercase names
  keep the plain `JXL_` prefix (`JXL_ERR`, `JXL_TR_*`, `JXL_MIN`); they never
  reach a header a caller sees, so the two prefixes also mark the public/
  internal boundary. Do not add a new `JXL_*` name to `src/jxl.h`.
- Dequant matrices are built lazily. Do not make them eager again: all 17
  slots cost ~8ms, which is the entire decode of a small image.
- The spline `max_distance` follows **libjxl**, not jxl-oxide: jxl-oxide uses
  `max_color` where libjxl takes `log(max_color)`, inflating the draw radius
  and the pixel count for output that is identical at 8 bits. Do not "restore"
  the jxl-oxide form when diffing against that crate.
- `epf_pass` and `jxl_render_splines` are written so the per-sample float
  operation order matches what the obvious per-channel loop would produce.
  Reordering an accumulation there changes pixels; the interior fast path and
  the mirroring border path in `epf_pass` must stay in step.

**Do not commit automatically.** Make and verify changes, but leave them in the
working tree; only run `git commit` when the user asks. Never commit
`dist/jxl.c` / `dist/jxl.h` — the user regenerates and commits those manually.
