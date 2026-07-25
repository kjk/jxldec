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
if (jxl_signature_check(data, len) == JXL_SIG_CODESTREAM) { ... }

jxl_doc *doc = jxl_doc_open(ctx, data, len);   /* data must outlive doc */
jxl_image_info info;
jxl_doc_info(doc, &info);                      /* size, depth, alpha, ... */

jxl_ctx_set_bgr(ctx, 1);                       /* emit BGRA for a Windows DIB */
jxl_image *img = jxl_frame_render(doc, 0, JXL_FORMAT_RGBA32);
/* or render straight into your own buffer: */
jxl_frame_render_into(doc, 0, JXL_FORMAT_RGBA32, dst, stride);
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
See [PROGRESS.md](PROGRESS.md) for the current pass rate and the list of
features that are still missing. In short: Modular (lossless) decoding is
byte-exact against libjxl; VarDCT matches within one 8-bit step for sRGB
content; patches, splines, noise, animation and non-sRGB primaries are not
implemented yet.

## How it was made
An AI-assisted port of [jxl-oxide](https://github.com/tirr-c/jxl-oxide) (Rust),
cross-checked against [libjxl](https://github.com/libjxl/libjxl) for exact
tables and constants, and verified against libjxl's own `djxl`.
