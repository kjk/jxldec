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
import { existsSync, mkdirSync, rmSync, statSync } from "fs";
import {
  clangCFlags,
  copyAsanRuntimeDll,
  defaultUseClang,
  isWindows,
  JXLDEC_MSVC_CL_C,
} from "./build";
import { SRCS } from "./sources";

const ROOT = `${import.meta.dir}/..`.replaceAll("\\", "/");
const TEST_SRC = `${ROOT}/test/oversize_render.c`;
const HEADERS = [`${ROOT}/src/jxl_internal.h`, `${ROOT}/src/jxl.h`];

const objBase = (src: string) => src.replace(/^src\//, "").replace(/\.c$/, "");

function needsRebuild(output: string, ...inputs: string[]): boolean {
  if (!existsSync(output)) return true;
  const out = statSync(output).mtimeMs;
  for (const inp of inputs) {
    if (!existsSync(inp) || statSync(inp).mtimeMs > out) return true;
  }
  return false;
}

type Unit = { src: string; obj: string };

function units(dir: string, ext: string): Unit[] {
  return [
    ...SRCS.map((s) => ({ src: `${ROOT}/${s}`, obj: `${dir}/${objBase(s)}.${ext}` })),
    { src: TEST_SRC, obj: `${dir}/oversize_render.${ext}` },
  ];
}

async function buildClang(dir: string, asan: boolean): Promise<string> {
  const exe = `${dir}/${isWindows ? "oversize_render.exe" : "oversize_render"}`;
  mkdirSync(dir, { recursive: true });
  const san = asan ? "-fsanitize=address " : "";
  const cflags = asan ? clangCFlags("-g -O1") : clangCFlags();
  const us = units(dir, "o");
  for (const u of us) {
    if (!needsRebuild(u.obj, u.src, ...HEADERS)) continue;
    await $`clang ${{ raw: san }}${{ raw: cflags }} -I${ROOT}/src -c -o ${u.obj} ${u.src}`;
  }
  const objs = us.map((u) => u.obj);
  if (needsRebuild(exe, ...objs)) {
    const libm = isWindows ? "" : " -lm";
    await $`clang ${{ raw: san }}${{ raw: objs.join(" ") }} -o ${exe}${{ raw: libm }}`;
  }
  if (asan && isWindows) await copyAsanRuntimeDll(dir);
  return exe;
}

async function buildMsvc(dir: string): Promise<string> {
  const exe = `${dir}/oversize_render.exe`;
  mkdirSync(dir, { recursive: true });
  const us = units(dir, "obj");
  for (const u of us) {
    if (!needsRebuild(u.obj, u.src, ...HEADERS)) continue;
    const rel = u.src.startsWith(`${ROOT}/`) ? u.src.slice(ROOT.length + 1) : u.src;
    await $`cl ${{ raw: JXLDEC_MSVC_CL_C }} -Isrc -Fo${dir}/ -c ${{ raw: rel }}`.cwd(ROOT);
  }
  const objs = us.map((u) => u.obj);
  if (needsRebuild(exe, ...objs)) {
    await $`cl -nologo ${{ raw: objs.join(" ") }} -Fe:${exe} -link -LTCG`.cwd(ROOT);
  }
  return exe;
}

export async function testOversizeRender(
  useClang = defaultUseClang,
  asan = false,
  clean = false,
): Promise<void> {
  if (asan) useClang = true;
  const dir = `${ROOT}/out/${asan ? "clang_asan" : useClang ? "clang" : "msvc"}`;
  if (clean) {
    for (const u of units(dir, useClang ? "o" : "obj")) rmSync(u.obj, { force: true });
    rmSync(`${dir}/${isWindows ? "oversize_render.exe" : "oversize_render"}`, {
      force: true,
    });
  }
  const exe = useClang ? await buildClang(dir, asan) : await buildMsvc(dir);
  await $`${exe}`;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const useClang = args.includes("-clang") || defaultUseClang;
  await testOversizeRender(useClang, args.includes("-asan"), args.includes("-clean"));
}
