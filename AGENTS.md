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
- `bun cmd/bench.ts <selection>` — times our decode against libjxl's.
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

## Methodology
Reference-oracle verification: decode with `jxl_test` and with `djxl`, compare
PNM output. Modular (integer) paths must be **bit-exact**; VarDCT paths are
float and are compared with a tight tolerance. Work incrementally and keep
`PROGRESS.md` current.

**Do not commit automatically.** Make and verify changes, but leave them in the
working tree; only run `git commit` when the user asks. Never commit
`dist/jxl.c` / `dist/jxl.h` — the user regenerates and commits those manually.
