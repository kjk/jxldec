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
import { existsSync, mkdirSync, readFileSync, rmSync } from "fs";
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

// A cmake cache remembers the compiler and the option values it was first
// configured with, and re-running with different -D flags does not reliably
// reset everything derived from them, so a build directory left behind by a
// differently-configured attempt fails in confusing ways. Returns true when
// the existing cache matches what we are about to ask for, so that an
// interrupted build resumes instead of starting over.
//
// This also catches the wreckage of the newline-in-$-template bug noted at
// configure() below: that ran cmake with no -D flags at all, leaving a cache
// holding cl.exe and BUILD_TESTING=ON, which fails with the thoroughly
// misleading "PNG library is required by some tests".
function cacheIsUsable(cc: string): boolean {
  const cache = join(LIBJXL_BUILD, "CMakeCache.txt");
  if (!existsSync(cache)) return false;
  let text: string;
  try {
    text = readFileSync(cache, "utf8");
  } catch {
    return false;
  }
  const entry = (key: string) =>
    text.match(new RegExp(`^${key}:[^=]*=(.*)$`, "m"))?.[1]?.trim() ?? "";
  if (!/^(OFF|NO|FALSE|0)$/i.test(entry("BUILD_TESTING"))) return false;
  const cached = entry("CMAKE_C_COMPILER").replaceAll("\\", "/");
  return cached.toLowerCase().includes(`/${cc.toLowerCase()}`);
}

export async function buildRef(): Promise<void> {
  if (existsSync(refTool("cjxl")) && existsSync(refTool("djxl"))) return;
  console.log("deps: building libjxl reference tools (one-time, slow)...");
  const cc = isWindows ? "clang-cl" : "clang";
  const cxx = isWindows ? "clang-cl" : "clang++";

  if (existsSync(LIBJXL_BUILD) && !cacheIsUsable(cc)) {
    console.log("deps: discarding a stale deps/libjxl-build cache");
    rmSync(LIBJXL_BUILD, { recursive: true, force: true });
  }

  // One array, not a multi-line template: bun's $ parses a newline inside the
  // template as a command separator, so a wrapped `cmake ... \n -DFOO=bar`
  // silently configures with *no* -D flags at all and then tries to run
  // `-DFOO=bar` as a command. The resulting default configuration (MSVC,
  // Debug, BUILD_TESTING=ON) is what fails, not the build tree.
  const flags = [
    "-DCMAKE_BUILD_TYPE=Release",
    `-DCMAKE_C_COMPILER=${cc}`, `-DCMAKE_CXX_COMPILER=${cxx}`,
    "-DBUILD_TESTING=OFF", "-DBUILD_SHARED_LIBS=OFF",
    "-DJPEGXL_ENABLE_TOOLS=ON", "-DJPEGXL_ENABLE_BENCHMARK=OFF",
    "-DJPEGXL_ENABLE_EXAMPLES=OFF", "-DJPEGXL_ENABLE_MANPAGES=OFF",
    "-DJPEGXL_ENABLE_SJPEG=OFF", "-DJPEGXL_ENABLE_OPENEXR=OFF",
    "-DJPEGXL_ENABLE_JNI=OFF", "-DJPEGXL_ENABLE_DOXYGEN=OFF",
    "-DJPEGXL_ENABLE_PLUGINS=OFF", "-DJPEGXL_BUNDLE_LIBPNG=OFF",
    "-DJPEGXL_ENABLE_SKCMS=ON",
  ];
  const configure = () =>
    $`cmake -S ${LIBJXL_DIR} -B ${LIBJXL_BUILD} -G Ninja ${flags}`.quiet();

  // cmake reports what actually went wrong on stderr ("PNG library is required
  // by some tests", "No CMAKE_C_COMPILER could be found"); stdout only carries
  // the -- progress chatter. Printing stdout alone hides every real diagnostic.
  const tail = (e: unknown) => {
    const err = e as { stdout?: unknown; stderr?: unknown };
    const out = [String(err.stdout ?? ""), String(err.stderr ?? "")]
      .join("\n").trimEnd();
    return out.split("\n").slice(-25).join("\n");
  };

  try {
    await configure();
  } catch (e) {
    // The cache-shape check above cannot anticipate every way a build tree
    // goes stale, so a failed configure earns exactly one clean retry.
    if (!existsSync(join(LIBJXL_BUILD, "CMakeCache.txt"))) {
      console.error(tail(e));
      throw e;
    }
    console.log("deps: cmake configure failed; retrying with a clean build dir");
    console.error(tail(e));
    rmSync(LIBJXL_BUILD, { recursive: true, force: true });
    try {
      await configure();
    } catch (e2) {
      console.error(tail(e2));
      throw e2;
    }
  }
  await $`cmake --build ${LIBJXL_BUILD} --target cjxl djxl jxlinfo`;
  console.log("deps: libjxl tools ready");
}

if (import.meta.main) {
  await getDeps();
  const { corpusSummary } = await import("./corpus");
  console.log(`deps: ready; ${corpusSummary()}`);
}
