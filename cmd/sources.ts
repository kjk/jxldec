// sources.ts -- the decoder's translation units, in amalgamation order.
// Shared by build.ts and build-dist.ts (a separate module so the two don't
// form an import cycle).
export const SRCS = [
  "src/core.c",
  "src/tables.c",
  "src/bitread.c",
  "src/container.c",
  "src/headers.c",
  "src/coding.c",
  "src/icc.c",
  "src/frame.c",
  "src/modular.c",
  "src/patch.c",
  "src/spline.c",
  "src/noise.c",
  "src/upsample.c",
  "src/decode.c",
  "src/vardct.c",
  "src/dct.c",
  "src/filter.c",
  "src/color.c",
  "src/render.c",
  "src/doc.c",
];
