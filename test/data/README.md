# test/data

Fixtures for features no encoder in `deps/` will produce. They are part of the
corpus (`cmd/corpus.ts`), so `bun cmd/tests.ts -all` checks them like any
generated file.

| file | what it covers | origin |
|---|---|---|
| `gray_spot_lossless.jxl` | spot colour on a grayscale image, Modular | `test/gen_gray_spot.c` (libjxl's encoder API) |
| `gray_spot_lossy.jxl` | the same, lossy XYB: the ink is mixed in linear light | `test/gen_gray_spot.c` |
| `custom_dequant_identity.jxl` | custom dequant tables for the IDENTITY (Hornuss) transform, encoding mode 1 | see below |

## custom_dequant_identity.jxl

`cjxl` always writes default dequantization tables, so nothing it generates
reaches the custom-table branches of `jxl_dequant_matrices_read`. This file
does: it signals its own IDENTITY weights, which libjxl scales by 64 after
reading. Before that scale was applied here, every IDENTITY block decoded as a
saturated 8x8 speck.

It was contributed as the reproducer in
[kjk/jxldec#1](https://github.com/kjk/jxldec/pull/1) by MadameMinty
(originally named `pac-sat160.jxl`).

The image is a page of the webcomic **Pepper&Carrot** by **David Revoy**
(<https://www.peppercarrot.com>), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It was modified:
saturation raised to 160% (`magick -modulate 100,160`), then encoded to JPEG XL
with the Rust crate `jixel` 0.4.2 (`Speed::Slow`, quality 80), which signals
custom IDENTITY weights on content it classifies as saturated.
