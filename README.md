# jxldec — a plain-C JPEG XL decoder

A decode-only [JPEG XL](https://fileformats.dev/jxl) library, in the spirit of
[djvudec](https://github.com/kjk/djvudec):

* plain C, no dependencies (vs. libjxl's C++ plus highway, brotli and skcms)
* simpler API — see [`src/jxl.h`](src/jxl.h)
* simple to integrate: copy `dist/jxl.h` and `dist/jxl.c` into your project.
  That is `src/*` amalgamated into a single translation unit, like sqlite
* single-threaded by design
* verified against libjxl on a corpus of 1547 `.jxl` files: libjxl's test
  images encoded 21 ways, plus the JPEG XL conformance testcases

## API
Sketch (full documentation in [`src/jxl.h`](src/jxl.h)):
```c
jxl_ctx *ctx = jxl_ctx_new(NULL, NULL, on_error, NULL);
if (jxl_signature_check(data, len) == JXLDEC_SIG_CODESTREAM) { ... }

jxl_doc *doc = jxl_doc_open(ctx, data, len);   /* data must outlive doc */
jxl_image_info info;
jxl_doc_info(doc, &info);                      /* size, depth, alpha, ... */

jxl_ctx_set_bgr(ctx, 1);                       /* emit BGRA for a Windows DIB */
jxl_image *img = jxl_frame_render(doc, 0, JXLDEC_FORMAT_RGBA32);
/* or render straight into your own buffer: */
jxl_frame_render_into(doc, 0, JXLDEC_FORMAT_RGBA32, dst, stride);
```

## Build & test
Requires `clang` (or MSVC), `bun`, `cmake`, `ninja` and `git`.
`cmd/get-deps.ts` clones libjxl (the reference decoder *and* the oracle) and
jxl-oxide into `deps/`, then builds `cjxl`/`djxl`/`jxlinfo`. `build.ts` and
`tests.ts` call it automatically.

```
bun cmd/get-deps.ts        # clone deps + build the libjxl tools
bun cmd/build.ts           # build the decoder + jxl_test (MSVC by default)
bun cmd/build.ts -clang    # ... with clang instead
bun cmd/tests.ts -all      # generate the corpus, then verify against djxl
bun cmd/tests.ts -rand 20  # ... on 20 random corpus files
bun cmd/build-dist.ts      # regenerate the dist/ amalgamation
```

`jxl_test` CLI (jbig2dec-flavored):
```
jxl_test -info in.jxl
jxl_test -frames in.jxl
jxl_test -out out.pam in.jxl
```
The output is binary PAM, the same thing `djxl in.jxl out.pam` writes, so the
two compare byte for byte. Other options: `-frame N` renders frame N instead
of the first, `-fmt N` asks for a `jxl_format` by its numeric value (default
0, `JXLDEC_FORMAT_NATIVE`), `-bgr` turns on `jxl_ctx_set_bgr` and `-srgb`
turns on `jxl_ctx_set_srgb_output`. With no option it prints `-info`.

## Status
See [PROGRESS.md](PROGRESS.md) for the full feature matrix. In short,
`bun cmd/tests.ts -all` decodes 1547 corpus files and compares each against
`djxl`: **all 1547 match — 538 byte-exact, the rest inside the tolerance.**
`bun cmd/tests.ts -all -all-frames` then checks *every frame* of every file
against libjxl itself (`djxl` writes only the first frame of an animation):
also 1547 of 1547.

* **Byte-exact** against libjxl: Modular lossless (every effort level,
  palette, squeeze, RCT, every predictor), plus patches and reference frames.
* **Within one 8-bit step**: VarDCT including progressive/LF frames, and
  lossy Modular (XYB). Two conformance files reach two steps.
* Also working: gaborish and EPF (on Modular frames too), splines, synthetic
  photon noise, 2x/4x/8x upsampling, all eight EXIF orientations, embedded
  ICC profiles, non-sRGB primaries (P3 and Rec.2100, Bradford-adapted),
  YCbCr/JPEG-transcoded frames in every chroma subsampling mode, layers and
  animation with every blend mode and cropped frames, spot colours, extra
  channels with their own bit depth, and lossless float samples.
* Transfer functions: sRGB, 709, HLG (with its OOTF) and PQ. PQ is verified
  only loosely: its slope at black turns the ~1e-5 of float noise between two
  correct XYB decoders into tens of 8-bit steps, so that preset alone is held
  to a much wider limit (see `cmd/corpus.ts`).

The non-byte-exact files are lossy paths where roughly a quarter of the
samples land one count either side of libjxl's. libjxl, jxl-oxide and this
decoder use different IDCT factorizations and different `powf` approximations,
so exact equality is not achievable — libjxl's own conformance testing uses a
tolerance too, and like it we gate on RMS and peak together rather than peak
alone (`-rms`, default 0.6; `-tol`, default 3).

Not implemented: encoding, JPEG reconstruction (`jbrd` boxes are skipped), and
multithreading. Two things are rejected with an error rather than decoded
wrongly: an extra channel whose upsampling factor differs from the colour
channels', and a spot colour on a grayscale image. A preview frame is skipped,
not decoded — untested, since no available file has one.

## Performance
`bun cmd/bench.ts -all` links the `dist/` amalgamation and libjxl's static
libraries into one process and times both single-threaded, best of N. Over the
1245 files the corpus had when it was last measured, this decoder takes
**about 1.10x** libjxl's decode time (1.0983x in a best-of-five sweep). libjxl is AVX2 throughout; the hot
loops here use runtime-selected SSE2/AVX2 kernels (noise, upsampling, EPF,
gaborish, both DCT passes, YCbCr and XYB conversion, the sRGB transfer
function and the output quantiser), so a constant factor is expected. Each
keeps a scalar twin; the SSE2 kernels are bit-identical to theirs, the
AVX2+FMA ones (DCT, dequantiser, EPF) use approximate reciprocals and are not.
PROGRESS.md records what closed the gap and, just as usefully, which
optimizations were measured and thrown away.

## How it was made
An AI-assisted port of [jxl-oxide](https://github.com/tirr-c/jxl-oxide) (Rust),
cross-checked against [libjxl](https://github.com/libjxl/libjxl) for exact
tables and constants, and verified against libjxl's own `djxl`.

## License
MIT; see [LICENSE.md](LICENSE.md). jxldec is a port of jxl-oxide
(MIT OR Apache-2.0) and contains tables and constants from libjxl
(BSD-3-Clause, plus libjxl's patent grant); LICENSE.md carries their notices.
