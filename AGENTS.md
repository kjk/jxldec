# AGENTS.md — working on the C JPEG XL decoder

Plain-C, decode-only JPEG XL (`.jxl`) library, ported from **jxl-oxide** (Rust)
and verified pixel-for-pixel against **libjxl**. API style follows
**jbig2dec** / the sibling project **djvudec** (`../djvudec`).

## Goal / scope
Decode-only, single-threaded. The caller hands us the entire `.jxl` file
up-front as an in-memory buffer. We provide:
- signature detection + image info (size, bit depth, channels, animation)
- decoded pixels for any frame, in 8- or 16-bit gray/gray+alpha/RGB/RGBA
- the embedded ICC profile, when present

No encoder. No JPEG reconstruction (`jbrd` boxes are skipped, like every
other non-codestream box).
No multithreading — `jxl_ctx` is not shared across threads. The one exception is `jxl_request_abort`, which another thread may call while a render runs (atomic epoch, see `jxl_render_begin` / `jxl_aborted` in jxl_internal.h).

The consumer that drives the feature set is **SumatraPDF**
(`sumatrapdf/src/shared/JxlReader.cpp`). It decodes the first frame two ways:
`jxl_decode` to `JXLDEC_FORMAT_RGBA32` with `jxl_ctx_set_bgr` (so no swizzle
pass is needed), and `jxl_doc_open` + `jxl_doc_info` + `jxl_frame_render_info`
+ `jxl_frame_render_into` straight into its own RGB24/RGBA32 buffer. Both turn
on `jxl_ctx_set_srgb_output`.

## Reference checkouts (local)
- `deps/libjxl` — the reference C++ decoder AND the oracle. `deps/libjxl-build`
  holds cjxl/djxl/jxlinfo built by `cmd/get-deps.ts` (cmake + ninja +
  clang-cl). Consult `lib/jxl/*.cc` for exact tables and constants.
- `deps/jxl-oxide` — Rust decoder; the implementation this port follows most
  closely. Its crate layout maps almost 1:1 onto our `src/` modules. Where it
  and libjxl disagree, **libjxl wins** — it is the oracle, and every
  disagreement found so far (see the gotchas) was jxl-oxide being wrong for a
  file no encoder emits by default.
- `deps/conformance` — github.com/libjxl/conformance, cloned by
  `cmd/get-deps.ts`. Two dozen hand-made files for what `cjxl` cannot emit:
  layers and every blend mode, cropped animation frames, spot colours, CMYK,
  float samples, delta palettes, loop filters on a Modular frame, extra
  channels with their own bit depth. Part of the corpus.
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
  `-no-deps` skips cloning/building libjxl (local compile-only checks).
  `bun cmd/build.ts asan` builds the clang+ASan harness.
- `bun cmd/tests.ts <-all | -rand N | file.jxl ...>` — the test driver:
  builds, then decodes each corpus file with both our decoder and `djxl`
  and compares the PAM output. The last lines say how many files were
  byte-exact, how many passed on tolerance and the worst peak among them.
- `bun cmd/tests.ts <selection> -all-frames` — the same, but against
  `jxl_ref` (`test/jxl_ref.c`: libjxl itself, dumping every coalesced frame)
  and for **every** frame of each file. `djxl` writes only frame 0 of an
  animation to PAM, so this is the only thing that verifies blending,
  reference slots and cropped frames; it also covers the three files `djxl`
  cannot write comparably (CMYK and spot colour come out with extra PAM
  planes, float samples not at all). MSVC only — it links libjxl's static
  libraries, like the benchmark. Run it after touching `walk_frames`,
  `jxl_blend_frame` or anything about extra channels.
- `bun cmd/test-oversize-render.ts [-clang] [-asan]` — unit regression for
  the output-stride overflow guard (`jxl_output_buffer_size`); no corpus
  needed.
- `bun cmd/test-abort.ts [file.jxl ...] [-clang] [-asan]` — `jxl_request_abort`
  (`test/abort_test.c`). The allocator callback requests the abort at chosen
  allocations across a render, so it is deterministic; every aborted render
  must fail, free everything, and leave the doc rendering the same pixels.
  A threaded pass then aborts from a second thread. Runs on a default file
  set (test/data plus a few corpus files, missing ones skipped). Run it after
  touching anything on a render's error or abort path. Both unit tests are
  built by `buildUnitTest` in `cmd/build.ts`; a new one is a `test/<name>.c`
  and a few lines of script.
- `bun cmd/coverage.ts [-rand N] [-cpu N] [-all-fns] [-reuse]` — decodes the
  corpus under clang source-based coverage and prints per-file line coverage
  plus, more usefully, every function that never ran. Use it before assuming a
  feature works: each decoder bug found so far lived in a path nothing
  exercised, and a function on that list is one nothing verifies against
  libjxl. `-reuse` re-reports from the profile already on disk (the decode is
  the slow part, ~3 min).
- `bun cmd/fuzz.ts -check` — replay every known reproducer once and exit:
  `fuzz/crashes/` plus two external sets already sitting in `deps/`, libjxl's
  OSS-Fuzz testcases and jxl-oxide's fuzz findings (about 120 inputs,
  seconds). Run
  it after touching a parser; it is the cheap half of the fuzzing story.
- `bun cmd/fuzz.ts [-jobs N] [-runs N] [-repro FILE] [-minimize]` — libFuzzer
  + ASan over `test/fuzz_target.c`, seeded from the corpus *and* those external
  reproducers on first run.
  `cmd/tests.ts` only ever feeds the decoder *valid* files, and a valid file
  takes no error path — `jxl_errorf`, which every `JXL_ERR` funnels into, never
  executes across the whole corpus. This covers what happens when the bytes
  lie. `fuzz/corpus/` is the checkpoint (gitignored): Ctrl-C to stop, rerun to
  resume. Artifacts land in `fuzz/crashes/`, which **is** tracked — replay one
  with `-repro`.
- `bun cmd/fuzz-afl.ts` — same harness via **AFL++** (macOS/*nix; needs
  `brew install afl++`). Shares `fuzz/corpus/` with libFuzzer; AFL state is in
  `fuzz/afl-out/` (gitignored). On exit (and with `-import`), queue/crashes
  merge into the shared dirs. On macOS run once: `sudo afl-system-config`.
- `bun cmd/bench.ts <selection> [-runs N]` — times our decode against libjxl's,
  both in one process, single-threaded, best of N. Windows/MSVC only: it links
  libjxl's static libraries, which need `-MD` and `-DJXL_STATIC_DEFINE`.
- `bun cmd/prof.ts <file.jxl> [-runs N] [-hz N] [-vs]` — sampling profile of
  *our* decoder alone on one file, via the sibling **winperf** (see
  "Windows sampling profiles" below).
  It builds `jxl_prof` with `-Zi` into `out/prof/` and passes winperf
  `-print-agent`, which prints the top self-time functions, the hot source
  lines and the heaviest call path. `-vs` profiles ours *and* libjxl in one
  process with both symbolized, by linking the benchmark harness against a
  debug-info libjxl (built on demand into `deps/libjxl-dbg`); that is how to
  see which side of a ratio is slow.
- `bun cmd/build-dist.ts` — regenerate the `dist/jxl.c` + `dist/jxl.h`
  amalgamation and verify it compiles with every available toolchain:
  host clang, host MSVC, MSVC x86 (`Host*/x86/cl` when the PATH cl is x64),
  and mingw-w64 gcc **without** `-mxsave` (catches the `_xgetbv` always_inline
  trap that broke SumatraPDF's Wine CI). Optional `-require-mingw` /
  `-require-msvc-x86` fail the script if that compiler is missing (used by CI).

**Script convention** (from djvudec): every script that operates on `.jxl`
files does nothing by default — it prints its options plus the available
corpus file count. Select work explicitly: `file.jxl ...`, `-rand N`, `-all`.
The one exception is `cmd/coverage.ts`, which decodes the whole corpus when
given no `-rand N`.

**Helper scripts are Bun TypeScript** — every one, saved or temporary: the
`cmd/*.ts` tools, and also throwaway scripts for a single investigation
(editing sources, comparing hashes, summarising benchmark output, driving a
timing loop). Run them with `bun`. No Python, no multi-line shell scripts;
a one-line shell command is fine.

### Windows sampling profiles (winperf)

The profiling harness vendors `test/winperf_control.h`, so a **winperf**
checkout is not required to compile. To record profiles, first look for the
private winperf repository at `..\winperf`. If it is absent, try to clone it
there:

```
git clone https://github.com/kjk/winperf ..\winperf
cd ..\winperf && bun cmd/build.ts -release
```

The clone requires access to the private repository. The vendored header is a
copy of `..\winperf\client\winperf_control.h` plus a three-line "vendored
from" note at the top; re-copy it before relying on newer winperf features.

```
bun cmd/prof.ts <file.jxl> [-runs N] [-hz N] [-vs]
```

Builds `out/prof/jxl_prof.exe` (MSVC `-O2 -Ob3 -Zi -Oy-`, no LTCG — LTCG
inlines across the whole amalgamation and blurs attribution) and records it
under winperf with `-print-agent`. Needs the **Windows Performance Toolkit**
(`xperf.exe` from the ADK) and **Administrator rights** (UAC prompt).

`jxl_prof` brackets each decode with `winperf_profile_start/stop` section
marks, so reading the file and process startup are dropped rather than
diluting the profile; the report says how many regions it saw
(`profiling windows: 40 region(s)` for `-runs 40`) which is the quick check
that the marks are being honoured. The marks are no-ops when the process is
not running under `winperf record`. Override the winperf binary with
`JXLDEC_WINPERF`. Give winperf an **absolute path** to the exe — it fails to
attach to a relative one and the trace fills with unrelated system processes
instead of erroring.

Manual equivalent:

```
..\winperf\out\rel64\winperf.exe record -i 4000 -o out\prof\winperf.etl -print-agent -- out\prof\jxl_prof.exe -runs 40 deps\corpus\gen\flower.v_icc.jxl
```

(This tool used to be called `samply`; older PROGRESS.md entries name it that.)

## Corpus
libjxl's `testdata/` holds only a handful of `.jxl` files; the corpus is
*generated* by encoding the testdata source images with `cjxl` across a matrix
of settings (VarDCT/modular, lossless/lossy, progressive, effort levels), cached
under `deps/corpus/`. See `cmd/corpus.ts`. On top of that come the `.jxl`
files libjxl ships and the conformance testcases. `JXL_SPECS=<dir>` overrides
the corpus with any directory of `.jxl` files.

A preset in `cmd/corpus.ts` can carry its own `peakTol` / `rmsTol`. Only the
two HDR ones do, and the comment there says exactly why (PQ's slope at black
turns float noise into tens of 8-bit steps). Do not reach for that to make a
failing preset pass: every other preset is held to the default limits.

`test/data/` holds fixtures no tool here can produce; its README says where
each came from and what it covers (two were made with libjxl's encoder API,
`test/gen_gray_spot.c`; one is a third-party file with custom dequant
tables, CC BY 4.0). They are part of the corpus.

When a decoder bug turns up in a path the corpus does not reach, add the
`cjxl` flag that reaches it as a preset (`m_rct7` and `m_p13` exist for
exactly that reason) rather than only fixing the bug.

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
| `decode.c` | `jxl-render/src/{render,modular}.rs` (per-frame decode: LfGlobal, Modular streams, filters, colour transform) |
| `patch.c` | `jxl-frame/src/data/patch.rs` |
| `spline.c` | `jxl-frame/src/data/spline.rs`, `jxl-render/src/features/spline.rs` |
| `noise.c` | `jxl-frame/src/data/noise.rs`, `jxl-render/src/features/noise.rs` |
| `upsample.c` | `jxl-render/src/features/upsampling.rs` (2x/4x/8x) |
| `vardct.c` | `jxl-frame/src/data/{lf_global,lf_group,hf_global,pass_group}.rs` |
| `dct.c` | `jxl-vardct` transforms |
| `filter.c` | gabor/EPF application |
| `color.c` | `jxl-color/src/{xyb,tf,convert}.rs` |
| `render.c` | `jxl-render` (frame walk, blending, output, orientation) |
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
`walk_frames` in `render.c` (called through `decode_frame_planes`) walks the chain, keeping a
`jxl_frame_state` with four reference slots and the most recent LF image.
Frames saved "before CT" (every LF frame) skip the XYB -> color-space step.

Compositing follows libjxl's pipeline (`dec_cache.cc`), and the order matters:
1. A normal frame that is not a plain full-frame replace of every channel is
   blended onto a canvas built from the reference slot each channel's own
   `BlendingInfo.source` names. **A slot nothing was saved to reads as zeros**
   (transparent black) — the previously displayed frame is never a fallback.
2. The result is saved to `save_as_reference` only if the frame can be
   referenced: not last, and `duration == 0 || save_as_reference != 0`
   (`FrameHeader::CanBeReferenced`). A displayed animation frame naming slot 0
   is *not* saved.
3. Reference-only frames and `save_before_ct` frames are saved as decoded:
   frame-sized, unblended, pre-colour-transform. That is what patches read.
4. Spot colours are mixed into the displayed frame last, after blending, and
   never into a saved reference. For a plain XYB frame (no blending, not
   saved after the colour transform) that means in **linear light**, before
   the transfer curve, because libjxl only applies the curve early when
   blending or a reference needs it (`jxl_frame_state.spot_linear`). A gray
   image is mixed with the ink's red component.

Without an animation header there is exactly one displayed frame, however
many layers the file has, so `jxl_doc_frame_count` answers 1 without walking.
`jxl_doc_frame_info` and the count walk read frame headers and TOCs only
(`img == NULL` in `walk_frames`).

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
- Extra-channel `BlendingInfo.source` is conditional on that channel's **own**
  mode, and the patch alpha-channel symbol on `num_extra_channels > 1` — both
  libjxl's rule, both different in jxl-oxide.
- The palette channel is `nb_colours + nb_deltas` wide (libjxl's encoder
  writes `nb_colours` with the deltas already subtracted).
- A permutation-only RCT (rct_type 7, 14, ...) has to move samples. Swapping
  `jxl_mchan` views is invisible to everything that reads
  `jxl_modular.base[]`, and breaks squeeze's adjacency assumption.
- `fh->ec_upsampling[i]` is the coded factor `<< dim_shift`, i.e. the
  channel's real size relative to the image. Extra channels are upsampled
  together with colour only when every one shares the colour factor
  (`late_ec_upsample`); otherwise each goes first, alone, before patches.
- Each extra channel is scaled by its **own** bit depth, and a float channel
  holds IEEE bit patterns (`modular_sample_to_float`), not counts.
- Gaborish and EPF apply to Modular frames too, with the constant
  `epf_sigma_for_modular`; a gray frame goes through as three identical
  channels (`modular_loop_filters`).
- HLG needs the inverse OOTF across all three channels before the per-channel
  curve; PQ is scaled by `intensity_target / 10000` and evaluated with
  libjxl's rational polynomial, because the exact `powf` form differs by
  several steps at black.
- A file with `have_preview` carries a whole extra frame before the first
  real one; `jxl_doc_open` steps over it.
- `jxl_dec_init` frees its own partial state on failure, so a caller can
  simply return.
- The AVX2+FMA kernels (DCT, dequant, EPF) use approximate reciprocals and
  are **not** bit-identical to scalar; the SSE2 ones are. `-DJXL_NO_AVX2` is
  the reproducible build, and `JXL_HAVE_BATCH_DCT8` is what keeps a
  `*_FORCE_SCALAR` build off the batched path.
- `-DJXL_DEBUG_MODULAR` compiles in the `JXL_DEBUG_TREE` / `JXL_DEBUG_TRACK`
  stderr dumps. Without it the library never reads the environment.
- `epf_pass` and `jxl_render_splines` are written so the per-sample float
  operation order matches what the obvious per-channel loop would produce.
  Reordering an accumulation there changes pixels; the interior fast path and
  the mirroring border path in `epf_pass` must stay in step.
- Every EPF kernel has a `gray` form (one channel, distance folded in with all
  three channel scales) that must stay bit-identical to the three-channel form
  on equal channels. Change both together; the check is that
  `grayscale_public_university` renders the same before and after.

**Do not commit automatically.** Make and verify changes, but leave them in the
working tree; only run `git commit` when the user asks. Never commit
`dist/jxl.c` / `dist/jxl.h` — the user regenerates and commits those manually.
