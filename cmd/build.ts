// build.ts -- build driver for the C JPEG XL decoder (`bun cmd/build.ts`).
//
//   bun cmd/build.ts          fetch deps + build jxl_test (MSVC default)
//   bun cmd/build.ts -clang   build with clang instead of MSVC
//   bun cmd/build.ts -clean   delete out/ first (full rebuild)
//   bun cmd/build.ts asan     clang + AddressSanitizer harness
//   bun cmd/build.ts cov      clang + coverage instrumentation (see coverage.ts)
//   bun cmd/build.ts fuzz     clang + libFuzzer + ASan (see fuzz.ts)
//   bun cmd/build.ts fuzz-afl afl-clang-fast harness (see fuzz-afl.ts; *nix)
//
// Objects and exes land in out/msvc/, out/clang/ or out/clang_asan/. The build
// is incremental: a source is recompiled only when newer than its object, and
// the exe is relinked only when an object is newer. build() returns the exe
// path. Verification lives in tests.ts.
import { $ } from "bun";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "fs";
import { resolve as resolvePath } from "path";
import { DIST_C, DIST_H, ensureDist } from "./build-dist";
import { buildRefDebug, getDeps } from "./get-deps";
import { SRCS } from "./sources";

// Forward slashes: Bun's shell treats backslashes as escapes.
const ROOT = `${import.meta.dir}/..`.replaceAll("\\", "/");
const OUT_ROOT = `${ROOT}/out`;

export const isWindows = process.platform === "win32";
export const isMac = process.platform === "darwin";

const outDir = (useClang: boolean) =>
  `${OUT_ROOT}/${useClang ? "clang" : "msvc"}`;

function binName(base: string): string {
  return isWindows ? `${base}.exe` : base;
}

const INTERNAL_H = `${ROOT}/src/jxl_internal.h`;
const PUBLIC_H = `${ROOT}/src/jxl.h`;


const objBase = (src: string) => src.replace(/^src\//, "").replace(/\.c$/, "");

function needsRebuild(output: string, ...inputs: string[]): boolean {
  if (!existsSync(output)) return true;
  const outMtime = statSync(output).mtimeMs;
  for (const input of inputs) {
    if (!existsSync(input)) return true;
    if (statSync(input).mtimeMs > outMtime) return true;
  }
  return false;
}

/** Echo a compiler/link command, then run it (stdout progress tracking). */
async function runCmd(cmd: string, cwd?: string): Promise<void> {
  console.log(cwd ? `+ cd ${cwd} && ${cmd}` : `+ ${cmd}`);
  const shell = $`${{ raw: cmd }}`;
  if (cwd) await shell.cwd(cwd);
  else await shell;
}

export function cleanBuildOutput(): void {
  rmSync(OUT_ROOT, { recursive: true, force: true });
}

// On Windows the default toolchain is MSVC; -clang forces clang.
export const defaultUseClang = !isWindows;

// MSVC cl.exe flags (use '-' not '/' -- Bun's shell mangles backslashes).
// -W4 -WX: high warning level + warnings as errors.
export const MSVC_CL_COMMON = `-nologo -O2 -Ob3 -GL -MT`;
export const JXLDEC_MSVC_CL_C =
  `${MSVC_CL_COMMON} -W4 -WX -std:c11 -D_CRT_SECURE_NO_WARNINGS`;

export const JXLDEC_CLANG_C_WARN =
  "-Wall -Wextra -Wuninitialized -Wconditional-uninitialized -Winit-self -Werror";
const JXLDEC_CLANG_C_STD = "-std=c11";

/** clang flags for our C sources. */
export function clangCFlags(opt = "-g -O3", win = isWindows): string {
  const crt = win ? " -D_CRT_SECURE_NO_WARNINGS" : "";
  return `${JXLDEC_CLANG_C_STD} ${opt} ${JXLDEC_CLANG_C_WARN}${crt}`;
}

const harnessExeName = (useClang: boolean) =>
  binName(`jxl_test_${useClang ? "clang" : "msvc"}`);

type CompileUnit = { src: string; obj: string; label: string };

function cUnits(dir: string, ext: string): CompileUnit[] {
  return [
    ...SRCS.map((s) => ({
      src: `${ROOT}/${s}`,
      obj: `${dir}/${objBase(s)}.${ext}`,
      label: s,
    })),
    {
      src: `${ROOT}/test/jxl_test.c`,
      obj: `${dir}/jxl_test.${ext}`,
      label: "test/jxl_test.c",
    },
  ];
}

async function buildClang(opt = "-g -O3", dir = outDir(true),
                          exeBase = `jxl_test_clang`,
                          extraFlags = ""): Promise<string> {
  const exePath = `${dir}/${binName(exeBase)}`;
  mkdirSync(dir, { recursive: true });
  const units = cUnits(dir, "o");
  for (const u of units) {
    if (!needsRebuild(u.obj, u.src, INTERNAL_H, PUBLIC_H)) continue;
    await runCmd(
      `clang ${extraFlags} ${clangCFlags(opt)} -I${ROOT}/src -c -o ${u.obj} ${u.src}`,
    );
  }
  const objs = units.map((u) => u.obj);
  if (needsRebuild(exePath, ...objs)) {
    const libm = isWindows ? "" : " -lm";
    await runCmd(`clang ${extraFlags} ${objs.join(" ")} -o ${exePath}${libm}`);
  }
  return exePath;
}

async function buildMsvc(): Promise<string> {
  if (!isWindows) throw new Error("MSVC build requires Windows");
  const dir = outDir(false);
  const exePath = `${dir}/${harnessExeName(false)}`;
  mkdirSync(dir, { recursive: true });

  const units = cUnits(dir, "obj");
  const clC = `${JXLDEC_MSVC_CL_C} -Isrc -Fo${dir}/ -c`;
  for (const u of units) {
    if (!needsRebuild(u.obj, u.src, INTERNAL_H, PUBLIC_H)) continue;
    const rel = u.src.startsWith(`${ROOT}/`) ? u.src.slice(ROOT.length + 1) : u.src;
    await runCmd(`cl ${clC} ${rel}`, ROOT);
  }
  const objs = units.map((u) => u.obj);
  if (needsRebuild(exePath, ...objs)) {
    await runCmd(
      `cl -nologo ${objs.join(" ")} -Fe:${exePath} -link -LTCG`,
      ROOT,
    );
  }
  return exePath;
}

/** Build the decoder + test harness. Returns the exe path. */
export async function build(useClang = defaultUseClang): Promise<string> {
  if (!useClang && !isWindows) {
    throw new Error("MSVC build requires Windows; use clang elsewhere");
  }
  const name = harnessExeName(useClang);
  const exePath = `${outDir(useClang)}/${name}`;
  const anyUnit = cUnits(outDir(useClang), useClang ? "o" : "obj").some((u) =>
    needsRebuild(u.obj, u.src, INTERNAL_H, PUBLIC_H),
  );
  if (!anyUnit && existsSync(exePath)) {
    console.log(`${name} up to date`);
    return exePath;
  }
  console.log(`building ${name} (${useClang ? "clang" : "msvc"})...`);
  const exe = useClang ? await buildClang() : await buildMsvc();
  console.log(`built ${name}`);
  return exe;
}

// clang + AddressSanitizer harness -> out/clang_asan/.
const ASAN_DIR = `${OUT_ROOT}/clang_asan`;

export async function buildAsan(): Promise<string> {
  const exe = await buildClang("-g -O1", ASAN_DIR, "jxl_test_asan",
                               "-fsanitize=address");
  if (isWindows) await copyAsanRuntimeDll(ASAN_DIR);
  return exe;
}

// clang + libFuzzer + ASan -> out/fuzz/. Unlike the other variants this one
// links test/fuzz_target.c instead of test/jxl_test.c, since libFuzzer brings
// its own main(). No external libraries are involved, so there is nothing here
// like heicdec's /MD import shim.
const FUZZ_DIR = `${OUT_ROOT}/fuzz`;
export const FUZZ_EXE = `${FUZZ_DIR}/${binName("jxl_fuzz")}`;

// Apple's clang ships without libFuzzer; Homebrew's has it.
function findFuzzClang(): string {
  const env = process.env.CLANG || process.env.CC || process.env.FUZZ_CC;
  if (env && existsSync(env)) return env;
  if (isMac) {
    for (const c of ["/opt/homebrew/opt/llvm/bin/clang",
                     "/usr/local/opt/llvm/bin/clang"]) {
      if (existsSync(c)) return c;
    }
  }
  return "clang";
}

export async function buildFuzz(clean = false): Promise<string> {
  mkdirSync(FUZZ_DIR, { recursive: true });
  const target = `${ROOT}/test/fuzz_target.c`;
  const units = [
    ...SRCS.map((s) => ({ src: `${ROOT}/${s}`, obj: `${FUZZ_DIR}/${objBase(s)}.o` })),
    { src: target, obj: `${FUZZ_DIR}/fuzz_target.o` },
  ];
  if (clean) {
    for (const u of units) rmSync(u.obj, { force: true });
    rmSync(FUZZ_EXE, { force: true });
  }

  const cc = findFuzzClang();
  // -O1 keeps ASan traces readable; the fuzzer is not a benchmark.
  const flags = `-fsanitize=address,fuzzer ${clangCFlags("-g -O1")}`;
  const stale = units.filter((u) => needsRebuild(u.obj, u.src, INTERNAL_H, PUBLIC_H));
  const objs = units.map((u) => u.obj);
  if (stale.length === 0 && !needsRebuild(FUZZ_EXE, ...objs) && existsSync(FUZZ_EXE)) {
    if (isWindows) await copyAsanRuntimeDll(FUZZ_DIR);
    console.log("jxl_fuzz up to date");
    return FUZZ_EXE;
  }
  try {
    for (const u of stale) {
      await runCmd(`${cc} ${flags} -I${ROOT}/src -c -o ${u.obj} ${u.src}`);
    }
    if (needsRebuild(FUZZ_EXE, ...objs)) {
      // ASan inflates stack frames and the Modular/VarDCT paths nest deeply;
      // Windows' default 1MB is tight once the fuzzer starts shrinking inputs.
      const stack = isWindows ? " -Wl,/STACK:8388608" : "";
      const libs = isWindows ? "" : " -lpthread -lm";
      await runCmd(`${cc} -fsanitize=address,fuzzer ${objs.join(" ")}` +
                   `${stack}${libs} -o ${FUZZ_EXE}`);
    }
  } catch (e) {
    if (isMac && cc === "clang") {
      console.error("\nfuzz build failed: Apple's clang has no libFuzzer.");
      console.error("  brew install llvm   (then re-run; it is auto-detected)\n");
    }
    throw e;
  }
  if (isWindows) await copyAsanRuntimeDll(FUZZ_DIR);
  console.log("built jxl_fuzz");
  return FUZZ_EXE;
}

/* ----- AFL++ target (driven by cmd/fuzz-afl.ts; macOS / *nix) ----- */

const FUZZ_AFL_DIR = `${OUT_ROOT}/fuzz-afl`;
export const FUZZ_AFL_EXE = `${FUZZ_AFL_DIR}/jxl_afl`;

/** Prefer Homebrew AFL++ wrappers (afl-clang-fast → Homebrew LLVM). */
export function findAflCc(): string {
  const env = process.env.AFL_CC || process.env.AFL_CLANG_FAST;
  if (env && existsSync(env)) return env;
  for (const c of [
    "/opt/homebrew/bin/afl-clang-fast",
    "/usr/local/bin/afl-clang-fast",
    "/opt/homebrew/bin/afl-cc",
    "/usr/local/bin/afl-cc",
  ]) {
    if (existsSync(c)) return c;
  }
  return "afl-clang-fast";
}

export function aflTool(name: string): string | null {
  const envKey = name.toUpperCase().replaceAll("-", "_");
  const env = process.env[envKey];
  if (env && existsSync(env)) return env;
  for (const base of ["/opt/homebrew/bin", "/usr/local/bin"]) {
    const p = `${base}/${name}`;
    if (existsSync(p)) return p;
  }
  return Bun.which(name);
}

// AFL++ target: same LLVMFuzzerTestOneInput harness as libFuzzer, but built
// with afl-clang-fast -fsanitize=fuzzer (AFL++ swaps that for libAFLDriver.a
// + persistent-mode shared-memory input). Optional ASan via AFL_USE_ASAN.
export async function buildAflFuzz(
  opts: { clean?: boolean; asan?: boolean } = {},
): Promise<string> {
  if (isWindows) {
    throw new Error(
      "AFL++ fuzzing is not supported on Windows; use bun cmd/fuzz.ts (libFuzzer)",
    );
  }
  const clean = !!opts.clean;
  const asan = opts.asan !== false; // default on
  mkdirSync(FUZZ_AFL_DIR, { recursive: true });

  const cc = findAflCc();
  const ccOk =
    existsSync(cc) ||
    !!Bun.which(cc) ||
    cc === "afl-clang-fast" ||
    cc === "afl-cc";
  if (!ccOk && !Bun.which("afl-clang-fast") && !Bun.which("afl-cc")) {
    throw new Error(
      "afl-clang-fast not found. On macOS: brew install afl++\n" +
        "  (uses Homebrew LLVM for instrumentation)",
    );
  }

  const testSrc = `${ROOT}/test/fuzz_target.c`;
  const units: { src: string; obj: string }[] = [
    ...SRCS.map((s) => ({
      src: `${ROOT}/${s}`,
      obj: `${FUZZ_AFL_DIR}/${objBase(s)}.o`,
    })),
    { src: testSrc, obj: `${FUZZ_AFL_DIR}/fuzz_target.o` },
  ];

  const stampWant = `asan=${asan ? 1 : 0};cc=${cc}`;
  const stampPath = `${FUZZ_AFL_DIR}/.afl_features`;
  const stampPrev = existsSync(stampPath)
    ? readFileSync(stampPath, "utf8").trim()
    : "";
  const stampChanged = stampPrev !== stampWant;

  if (clean || stampChanged) {
    for (const u of units) rmSync(u.obj, { force: true });
    rmSync(FUZZ_AFL_EXE, { force: true });
  }

  const headers = [INTERNAL_H, PUBLIC_H];
  const objStale = (u: { src: string; obj: string }) =>
    needsRebuild(u.obj, u.src, ...headers);
  const staleObj = units.some(objStale);
  const staleExe = needsRebuild(FUZZ_AFL_EXE, ...units.map((u) => u.obj));
  if (!staleObj && !staleExe && existsSync(FUZZ_AFL_EXE) && !stampChanged) {
    console.log("jxl_afl up to date");
    return FUZZ_AFL_EXE;
  }

  // -O1 for readable ASan traces; -fsanitize=fuzzer → AFL++ libAFLDriver.
  // Do not also pass -fsanitize=address: AFL_USE_ASAN adds it.
  const cflags =
    `-fsanitize=fuzzer ${clangCFlags("-g -O1", false)} -I${ROOT}/src`;
  const env: Record<string, string> = {
    ...process.env,
    // Strip -Werror noise from AFL plugin passes on some LLVM versions.
    AFL_LLVM_NO_ERROR: process.env.AFL_LLVM_NO_ERROR ?? "1",
    AFL_QUIET: process.env.AFL_QUIET ?? "1",
  };
  if (asan) env.AFL_USE_ASAN = "1";
  else delete env.AFL_USE_ASAN;

  console.log(
    `building jxl_afl (afl-clang-fast+fuzzer${asan ? "+asan" : ""})...`,
  );
  try {
    for (const u of units) {
      if (!objStale(u) && !stampChanged) continue;
      await $`${cc} ${{ raw: cflags }} -c -o ${u.obj} ${u.src}`
        .cwd(ROOT)
        .env(env);
    }
    const objs = units.map((u) => u.obj);
    if (needsRebuild(FUZZ_AFL_EXE, ...objs) || stampChanged) {
      await $`${cc} -fsanitize=fuzzer ${{ raw: objs.join(" ") }} -lpthread -lm -o ${FUZZ_AFL_EXE}`
        .cwd(ROOT)
        .env(env);
    }
  } catch (e) {
    console.error("\nAFL++ fuzz build failed.");
    console.error("Install AFL++ (and LLVM) via Homebrew:");
    console.error("  brew install afl++");
    console.error(
      "afl-clang-fast must be on PATH (or set AFL_CC=/path/to/afl-clang-fast).\n",
    );
    throw e;
  }
  writeFileSync(stampPath, stampWant);
  console.log("built jxl_afl");
  return FUZZ_AFL_EXE;
}

// clang + source-based coverage instrumentation -> out/clang_cov/.
// -O0 so that no inlining folds one function's lines into another's and a
// never-executed branch cannot be optimised away before it is counted.
const COV_DIR = `${OUT_ROOT}/clang_cov`;

export async function buildCoverage(): Promise<string> {
  return buildClang("-g -O0", COV_DIR, "jxl_test_cov",
                    "-fprofile-instr-generate -fcoverage-mapping");
}

// The exe links the DYNAMIC asan runtime, which lives in clang's resource dir
// (not on PATH); without a copy next to the exe the loader fails before main.
export async function copyAsanRuntimeDll(dir: string): Promise<void> {
  const dllName = "clang_rt.asan_dynamic-x86_64.dll";
  const dst = resolvePath(dir, dllName);
  if (existsSync(dst)) return;
  const proc = Bun.spawnSync(["clang", "-print-resource-dir"]);
  const resDir = proc.stdout.toString().trim();
  const src = resolvePath(resDir, "lib/windows", dllName);
  if (!existsSync(src)) {
    console.warn(`warning: ${src} not found; ${dllName} must be on PATH`);
    return;
  }
  copyFileSync(src, dst);
}

// Build the harness from the dist/ amalgamation (single translation unit).
export async function buildBench(useClang = defaultUseClang): Promise<string> {
  await ensureDist();
  const dir = outDir(useClang);
  const exePath = `${dir}/${harnessExeName(useClang)}`;
  mkdirSync(dir, { recursive: true });
  const ext = useClang ? "o" : "obj";
  const libObj = `${dir}/jxl.${ext}`;
  const testObj = `${dir}/jxl_test.${ext}`;

  if (useClang) {
    if (needsRebuild(libObj, DIST_C, DIST_H)) {
      await runCmd(`clang ${clangCFlags()} -I${ROOT}/dist -c -o ${libObj} ${DIST_C}`);
    }
    if (needsRebuild(testObj, `${ROOT}/test/jxl_test.c`, DIST_H, INTERNAL_H)) {
      await runCmd(
        `clang ${clangCFlags()} -I${ROOT}/dist -I${ROOT}/src -c -o ${testObj} ${ROOT}/test/jxl_test.c`,
      );
    }
    if (needsRebuild(exePath, libObj, testObj)) {
      const libm = isWindows ? "" : " -lm";
      await runCmd(`clang ${libObj} ${testObj} -o ${exePath}${libm}`);
    }
  } else {
    if (needsRebuild(libObj, DIST_C, DIST_H)) {
      await runCmd(`cl ${JXLDEC_MSVC_CL_C} -Idist -Fo${libObj} -c dist/jxl.c`, ROOT);
    }
    if (needsRebuild(testObj, `${ROOT}/test/jxl_test.c`, DIST_H, INTERNAL_H)) {
      await runCmd(
        `cl ${JXLDEC_MSVC_CL_C} -Idist -Isrc -Fo${testObj} -c test/jxl_test.c`,
        ROOT,
      );
    }
    if (needsRebuild(exePath, libObj, testObj)) {
      await runCmd(
        `cl -nologo ${libObj} ${testObj} -Fe:${exePath} -link -LTCG`,
        ROOT,
      );
    }
  }
  return exePath;
}

// libjxl's static libraries, in link order, for the benchmark harness.
function libjxlLibs(): string[] {
  const b = `${ROOT}/deps/libjxl-build`;
  return [
    `${b}/lib/jxl.lib`,
    `${b}/lib/jxl_cms.lib`,
    `${b}/third_party/brotli/brotlidec.lib`,
    `${b}/third_party/brotli/brotlienc.lib`,
    `${b}/third_party/brotli/brotlicommon.lib`,
    `${b}/third_party/highway/hwy.lib`,
  ];
}

/** Build jxl_bench: our amalgamation and libjxl in one process, MSVC only
 *  (libjxl is built as C++ with the MSVC runtime). */
/* The benchmark harness linked against the debug-info libjxl, so a profile
   attributes libjxl's own time to its own functions instead of one opaque
   module. Same harness, so both decoders run the same work in one process
   under identical conditions -- which is the only way to compare them
   honestly. Used by cmd/prof.ts -vs. */
export async function buildBenchHarnessDbg(): Promise<string> {
  if (!isWindows) throw new Error("the benchmark harness needs the MSVC toolchain");
  await ensureDist();
  await buildRefDebug();
  const dir = `${OUT_ROOT}/prof`;
  const exePath = `${dir}/${binName("jxl_bench_dbg")}`;
  const inc = `${ROOT}/deps/libjxl/lib/include`;
  const d = `${ROOT}/deps/libjxl-dbg`;
  mkdirSync(dir, { recursive: true });
  const clMd = `${JXLDEC_MSVC_CL_C.replace("-MT", "-MD")} -Zi -Fd${dir}/`;
  const libObj = `${dir}/dist_dbg.obj`;
  const benchObj = `${dir}/bench_dbg.obj`;
  if (needsRebuild(libObj, DIST_C, DIST_H)) {
    await runCmd(`cl ${clMd} -Idist -Fo${libObj} -c dist/jxl.c`, ROOT);
  }
  if (needsRebuild(benchObj, `${ROOT}/test/jxl_bench.c`, DIST_H)) {
    await runCmd(
      `cl ${clMd} -DJXL_STATIC_DEFINE -Idist -I${inc} -I${d}/lib/include -Fo${benchObj} -c test/jxl_bench.c`,
      ROOT,
    );
  }
  const libs = [
    `${d}/lib/jxl.lib`, `${d}/lib/jxl_cms.lib`,
    `${d}/third_party/brotli/brotlidec.lib`,
    `${d}/third_party/brotli/brotlienc.lib`,
    `${d}/third_party/brotli/brotlicommon.lib`,
    `${d}/third_party/highway/hwy.lib`,
  ];
  if (needsRebuild(exePath, libObj, benchObj, ...libs)) {
    await runCmd(
      `cl -nologo ${libObj} ${benchObj} -Fe:${exePath} -link -DEBUG ${libs.join(" ")}`,
      ROOT,
    );
  }
  return exePath;
}

export async function buildBenchHarness(): Promise<string> {
  if (!isWindows) throw new Error("the benchmark harness needs the MSVC toolchain");
  await ensureDist();
  const dir = outDir(false);
  const exePath = `${dir}/${binName("jxl_bench")}`;
  const inc = `${ROOT}/deps/libjxl/lib/include`;
  const incBuild = `${ROOT}/deps/libjxl-build/lib/include`;
  mkdirSync(dir, { recursive: true });

  // libjxl links the dynamic CRT, so the harness objects must too (-MD).
  const clMd = JXLDEC_MSVC_CL_C.replace("-MT", "-MD");
  const libObj = `${dir}/jxl_dist.obj`;
  const benchObj = `${dir}/jxl_bench.obj`;
  if (needsRebuild(libObj, DIST_C, DIST_H)) {
    await runCmd(`cl ${clMd} -Idist -Fo${libObj} -c dist/jxl.c`, ROOT);
  }
  if (needsRebuild(benchObj, `${ROOT}/test/jxl_bench.c`, DIST_H)) {
    await runCmd(
      `cl ${clMd} -DJXL_STATIC_DEFINE -Idist -I${inc} -I${incBuild} -Fo${benchObj} -c test/jxl_bench.c`,
      ROOT,
    );
  }
  const libs = libjxlLibs();
  if (needsRebuild(exePath, libObj, benchObj, ...libs)) {
    await runCmd(
      `cl -nologo ${libObj} ${benchObj} -Fe:${exePath} -link -LTCG ${libs.join(" ")}`,
      ROOT,
    );
  }
  return exePath;
}

/** Build jxl_prof: our decoder alone, with debug info so a sampling profiler
 *  can symbolize it. Objects live in out/prof/ so the release build's
 *  no-debug-info objects are not clobbered. */
export async function buildProfHarness(): Promise<string> {
  if (!isWindows) throw new Error("the profiling harness is MSVC-only for now");
  await ensureDist();
  const dir = `${OUT_ROOT}/prof`;
  const exePath = `${dir}/${binName("jxl_prof")}`;
  mkdirSync(dir, { recursive: true });

  // -Zi for line-level symbols; -GL (whole-program opt) is dropped because it
  // inlines across the whole amalgamation and blurs the profile attribution.
  const clZi = JXLDEC_MSVC_CL_C.replace("-GL", "-Zi -Oy-");
  const libObj = `${dir}/jxl_dist.obj`;
  const profObj = `${dir}/jxl_prof.obj`;
  if (needsRebuild(libObj, DIST_C, DIST_H)) {
    await runCmd(`cl ${clZi} -Fd${dir}/ -Idist -Fo${libObj} -c dist/jxl.c`, ROOT);
  }
  if (needsRebuild(profObj, `${ROOT}/test/jxl_prof.c`, DIST_H,
                   `${ROOT}/test/winperf_control.h`)) {
    await runCmd(
      `cl ${clZi} -Fd${dir}/ -Idist -Fo${profObj} -c test/jxl_prof.c`,
      ROOT,
    );
  }
  if (needsRebuild(exePath, libObj, profObj)) {
    await runCmd(
      `cl -nologo ${libObj} ${profObj} -Fe:${exePath} -link -DEBUG`,
      ROOT,
    );
  }
  return exePath;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.includes("-clean")) cleanBuildOutput();
  await getDeps();
  const useClang = args.includes("-clang") || defaultUseClang;
  if (args.includes("asan")) await buildAsan();
  else if (args.includes("cov")) await buildCoverage();
  else if (args.includes("fuzz-afl")) await buildAflFuzz();
  else if (args.includes("fuzz")) await buildFuzz();
  else await build(useClang);
}
