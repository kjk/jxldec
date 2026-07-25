# jxldec — a plain-C JPEG XL decoder

A decode-only JPEG XL library, in the spirit of
[djvudec](https://github.com/kjk/djvudec):

* plain C, no dependencies (vs. libjxl's C++ plus highway, brotli and skcms)
* simpler API — see [`src/jxl.h`](src/jxl.h)
* simple to integrate: copy `dist/jxl.h` and `dist/jxl.c` into your project.
  That is `src/*` amalgamated into a single translation unit, like sqlite
* single-threaded by design
* verified against libjxl on a generated corpus of hundreds of `.jxl` files

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
two compare byte for byte.

## Status
See [PROGRESS.md](PROGRESS.md) for the full feature matrix. In short,
`bun cmd/tests.ts -all` decodes 1113 corpus files and compares each against
`djxl`: **1113 match — 387 byte-exact, and no sample in the other 726 differs
by more than one 8-bit step.**

* **Byte-exact** against libjxl: Modular lossless (every effort level,
  palette, squeeze, RCT), plus patches and reference frames.
* **Within one 8-bit step**: VarDCT including progressive/LF frames, and
  lossy Modular (XYB), for sRGB content.
* Also working: gaborish and EPF, splines, synthetic photon noise, 2x/4x/8x
  upsampling, all eight EXIF orientations, animation, embedded ICC profiles,
  and YCbCr/JPEG-transcoded frames in every chroma subsampling mode.
* Implemented but **unverified**: non-sRGB primaries (BT.2020, P3) and the PQ
  and HLG transfer functions. `bun cmd/coverage.ts` shows that code never
  runs — every corpus file declares sRGB primaries, so nothing checks it
  against `djxl`.

The 726 non-byte-exact files are lossy VarDCT paths where roughly a quarter of
the samples land one count either side of libjxl's. libjxl, jxl-oxide and this
decoder use different IDCT factorizations and different `powf` approximations,
so exact equality is not achievable — libjxl's own conformance testing uses a
tolerance too, and like it we gate on RMS and peak together rather than peak
alone (`-rms`, default 0.6; `-tol`, default 3).

Not implemented: encoding, JPEG reconstruction (`jbrd` boxes are located but
not applied), and multithreading. Two extra-channel cases are rejected with an
error rather than decoded wrongly: a channel whose upsampling factor differs
from the colour channels', and a spot-colour channel (which has to be mixed
into the colour channels). Neither is producible by `cjxl`, so neither can be
verified against `djxl` yet.

## Performance
`bun cmd/bench.ts -all` links the `dist/` amalgamation and libjxl's static
libraries into one process and times both single-threaded, best of N. Over the
corpus this decoder takes **2.33x** libjxl's decode time, down from 3.22x
(measured over the 821 files that predate the `v_noise` preset). libjxl is AVX2 and this is scalar C, so a constant factor is
expected; PROGRESS.md records what closed the gap and, just as usefully, which
optimizations were measured and thrown away.

## How it was made
An AI-assisted port of [jxl-oxide](https://github.com/tirr-c/jxl-oxide) (Rust),
cross-checked against [libjxl](https://github.com/libjxl/libjxl) for exact
tables and constants, and verified against libjxl's own `djxl`.
