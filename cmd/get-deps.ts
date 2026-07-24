// get-deps.ts -- fetch the reference checkouts and build the libjxl oracle.
//
//   bun cmd/get-deps.ts
//
// Clones into deps/ (skipped if already present):
//   deps/libjxl     -- the reference decoder + its testdata submodule. Doubles
//                      as the oracle (cjxl/djxl/jxlinfo) and the source of the
//                      corpus images. JXLDEC_LIBJXL=<dir> points at an existing
//                      checkout instead of cloning (a junction/symlink at
//                      deps/libjxl works too).
//   deps/jxl-oxide  -- Rust JPEG XL decoder; the implementation this port
//                      follows most closely. Read-only reference.
// Exported as getDeps() so build.ts / tests.ts can self-provision.
// deps/ is gitignored.
import { $ } from "bun";
import { existsSync, mkdirSync } from "fs";
import { join } from "path";

const ROOT = `${import.meta.dir}/..`;
export const DEPS_DIR = join(ROOT, "deps");
export const LIBJXL_DIR = process.env.JXLDEC_LIBJXL || join(DEPS_DIR, "libjxl");
export const LIBJXL_BUILD = join(DEPS_DIR, "libjxl-build");
export const JXL_OXIDE_DIR = join(DEPS_DIR, "jxl-oxide");

export const isWindows = process.platform === "win32";
const exe = (name: string) => (isWindows ? `${name}.exe` : name);

/** Path to a libjxl reference tool (cjxl / djxl / jxlinfo). */
export function refTool(name: string): string {
  return join(LIBJXL_BUILD, "tools", exe(name));
}

export async function getDeps(): Promise<void> {
  mkdirSync(DEPS_DIR, { recursive: true });

  if (!existsSync(LIBJXL_DIR)) {
    console.log("deps: cloning libjxl (with submodules; several minutes)");
    await $`git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/libjxl/libjxl ${LIBJXL_DIR}`;
  }
  if (!existsSync(JXL_OXIDE_DIR)) {
    console.log("deps: cloning jxl-oxide");
    await $`git clone --depth 1 https://github.com/tirr-c/jxl-oxide ${JXL_OXIDE_DIR}`;
  }
  await buildRef();
}

// Configure + build cjxl/djxl/jxlinfo once into deps/libjxl-build. These are
// the oracle: djxl decodes to PNM for comparison, cjxl generates the corpus.
// libpng is deliberately NOT bundled (its cmake glue can't find the bundled
// zlib's generated zconf.h); PNG input is handled by cmd/png.ts instead, and
// PNM covers everything the oracle needs.
export async function buildRef(): Promise<void> {
  if (existsSync(refTool("cjxl")) && existsSync(refTool("djxl"))) return;
  console.log("deps: building libjxl reference tools (one-time, slow)...");
  const cc = isWindows ? "clang-cl" : "clang";
  const cxx = isWindows ? "clang-cl" : "clang++";
  await $`cmake -S ${LIBJXL_DIR} -B ${LIBJXL_BUILD} -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${cc} -DCMAKE_CXX_COMPILER=${cxx}
    -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF
    -DJPEGXL_ENABLE_TOOLS=ON -DJPEGXL_ENABLE_BENCHMARK=OFF
    -DJPEGXL_ENABLE_EXAMPLES=OFF -DJPEGXL_ENABLE_MANPAGES=OFF
    -DJPEGXL_ENABLE_SJPEG=OFF -DJPEGXL_ENABLE_OPENEXR=OFF
    -DJPEGXL_ENABLE_JNI=OFF -DJPEGXL_ENABLE_DOXYGEN=OFF
    -DJPEGXL_ENABLE_PLUGINS=OFF -DJPEGXL_BUNDLE_LIBPNG=OFF
    -DJPEGXL_ENABLE_SKCMS=ON`.quiet();
  await $`cmake --build ${LIBJXL_BUILD} --target cjxl djxl jxlinfo`;
  console.log("deps: libjxl tools ready");
}

if (import.meta.main) {
  await getDeps();
  const { corpusSummary } = await import("./corpus");
  console.log(`deps: ready; ${corpusSummary()}`);
}
