// test-oversize-render.ts -- regression for the jxl_frame_render output-stride
// integer overflow (render.c).
//
//   bun cmd/test-oversize-render.ts            build + run (msvc on Windows)
//   bun cmd/test-oversize-render.ts -clang     build + run with clang
//   bun cmd/test-oversize-render.ts -asan      build + run under clang+ASan
//   bun cmd/test-oversize-render.ts -clean     rebuild from scratch
//
// Compiles the decoder sources + test/oversize_render.c into a standalone exe
// and runs it. The test calls jxl_output_buffer_size directly (see that file
// for why an end-to-end crafted .jxl cannot drive the bug under a budget).
import { $ } from "bun";
import { buildUnitTest, defaultUseClang } from "./build";

export async function testOversizeRender(
  useClang = defaultUseClang,
  asan = false,
  clean = false,
): Promise<void> {
  const exe = await buildUnitTest("oversize_render", { useClang, asan, clean });
  await $`${exe}`;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const useClang = args.includes("-clang") || defaultUseClang;
  await testOversizeRender(useClang, args.includes("-asan"), args.includes("-clean"));
}
