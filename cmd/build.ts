// build.ts -- build driver for the C JPEG XL decoder (`bun cmd/build.ts`).
//
//   bun cmd/build.ts          fetch deps + build jxl_test (MSVC default)
//   bun cmd/build.ts -clang   build with clang instead of MSVC
//   bun cmd/build.ts -clean   delete out/ first (full rebuild)
//   bun cmd/build.ts asan     clang + AddressSanitizer harness
//
// Objects and exes land in out/msvc/, out/clang/ or out/clang_asan/. The build
// is incremental: a source is recompiled only when newer than its object, and
// the exe is relinked only when an object is newer. build() returns the exe
// path. Verification lives in tests.ts.
import { $ } from "bun";
import { copyFileSync, existsSync, mkdirSync, rmSync, statSync } from "fs";
import { resolve as resolvePath } from "path";
import { DIST_C, DIST_H, ensureDist } from "./build-dist";
import { getDeps } from "./get-deps";
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

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.includes("-clean")) cleanBuildOutput();
  await getDeps();
  const useClang = args.includes("-clang") || defaultUseClang;
  if (args.includes("asan")) await buildAsan();
  else await build(useClang);
}
