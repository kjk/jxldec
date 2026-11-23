// test-abort.ts -- regression test for jxl_request_abort (test/abort_test.c).
//
//   bun cmd/test-abort.ts               build + run on the default file set
//   bun cmd/test-abort.ts file.jxl ...  ... on these files instead
//   options: -clang, -asan, -clean      (as for test-oversize-render.ts)
//
// The test aborts renders at chosen points (from inside the allocator, so it
// is deterministic) and from a second thread, and checks that an aborted
// render fails, leaks nothing, and leaves the doc rendering correctly.
//
// Unlike the corpus scripts this runs by default: it is a unit test, and its
// default files cover the shapes abort has to cut through -- a multi-group
// VarDCT frame, Modular, layers, a cropped animation, progressive LF frames.
// Files that are not there (deps/ not fetched) are skipped; the two under
// test/data/ always are.
import { $ } from "bun";
import { existsSync } from "fs";
import { join } from "path";
import { buildUnitTest, defaultUseClang } from "./build";

const ROOT = join(import.meta.dir, "..");
const DEFAULT_FILES = [
  "test/data/custom_dequant_identity.jxl",   // VarDCT, 35 groups
  "test/data/gray_spot_lossless.jxl",        // Modular, extra channel
  "deps/conformance/testcases/blendmodes/input.jxl",       // five layers
  "deps/conformance/testcases/animation_icos4d/input.jxl", // cropped animation
  "deps/conformance/testcases/progressive/input.jxl",      // LF frames
  "deps/corpus/gen/flower_small.rgb.m_e7.jxl",             // Modular lossless
];

const args = process.argv.slice(2);
const named = args.filter((a) => !a.startsWith("-"));
const files = (named.length ? named : DEFAULT_FILES.map((f) => join(ROOT, f)))
  .filter((f) => {
    if (existsSync(f)) return true;
    console.log(`skip ${f} (not present)`);
    return false;
  });
if (files.length === 0) {
  console.error("test-abort: no input files");
  process.exit(2);
}

const exe = await buildUnitTest("abort_test", {
  useClang: args.includes("-clang") || defaultUseClang,
  asan: args.includes("-asan"),
  clean: args.includes("-clean"),
});
const r = await $`${exe} ${files}`.nothrow();
process.exit(r.exitCode ?? 1);
