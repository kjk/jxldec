// build-dist.ts -- produce an SQLite-style single-file amalgamation in dist/.
//
//   bun cmd/build-dist.ts
//
// Emits:
//   dist/jxl.h  -- the public API header (verbatim copy of src/jxl.h)
//   dist/jxl.c  -- the entire decoder as one translation unit: the public
//                  header, then the internal header, then every src/*.c,
//                  concatenated with the local #include "jxl.h" /
//                  "jxl_internal.h" lines stripped; comments and trailing
//                  whitespace removed.
//
// A consumer drops both files into their tree and compiles jxl.c like any
// other source. Verified with every available toolchain before finishing.
// This only works because no two source files share a file-local (static)
// symbol name -- keep it that way or the single-unit build breaks.
import { $ } from "bun";
import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "fs";
import { join } from "path";
import { clangCFlags, JXLDEC_MSVC_CL_C, isWindows } from "./build";
import { SRCS } from "./sources";

const ROOT = `${import.meta.dir}/..`.replaceAll("\\", "/");
const SRC = join(ROOT, "src");
const DIST = join(ROOT, "dist");

export const DIST_MODULES = SRCS.map((s) => s.replace(/^src\//, ""));
export const DIST_H = join(DIST, "jxl.h");
export const DIST_C = join(DIST, "jxl.c");

export function distInputPaths(): string[] {
  return [
    join(SRC, "jxl.h"),
    join(SRC, "jxl_internal.h"),
    ...DIST_MODULES.map((name) => join(SRC, name)),
  ];
}

export function distOutdated(): boolean {
  if (!existsSync(DIST_H) || !existsSync(DIST_C)) return true;
  const outMtime = Math.min(statSync(DIST_H).mtimeMs, statSync(DIST_C).mtimeMs);
  for (const input of distInputPaths()) {
    if (!existsSync(input)) return true;
    if (statSync(input).mtimeMs > outMtime) return true;
  }
  return false;
}

// Drop a local-quote include of one of the named headers (system <...>
// includes are left untouched -- they have their own guards).
function stripIncludes(text: string, headers: string[]): string {
  const re = new RegExp(
    `^[ \\t]*#[ \\t]*include[ \\t]+"(?:${headers.join("|")})"[ \\t]*\\r?\\n`,
    "gm",
  );
  return text.replace(re, "");
}

// Remove // and /* */ comments; leaves string/char literal contents intact.
function stripCComments(code: string): string {
  let out = "";
  let i = 0;
  const n = code.length;
  while (i < n) {
    const c = code[i];
    const next = i + 1 < n ? code[i + 1] : "";
    if (c === '"' || c === "'") {
      const quote = c;
      out += c;
      i++;
      while (i < n) {
        if (code[i] === "\\" && i + 1 < n) {
          out += code[i] + code[i + 1];
          i += 2;
        } else if (code[i] === quote) {
          out += code[i];
          i++;
          break;
        } else {
          out += code[i];
          i++;
        }
      }
      continue;
    }
    if (c === "/" && next === "/") {
      i += 2;
      while (i < n && code[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && next === "*") {
      i += 2;
      while (i + 1 < n && !(code[i] === "*" && code[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  return out
    .replace(/\n{3,}/g, "\n\n")
    .split(/\r?\n/)
    .map((line) => line.replace(/[ \t]+$/, ""))
    .join("\n");
}

async function runCmd(cmd: string, cwd?: string): Promise<number> {
  console.log(cwd ? `+ cd ${cwd} && ${cmd}` : `+ ${cmd}`);
  const shell = $`${{ raw: cmd }}`.nothrow();
  if (cwd) await shell.cwd(cwd);
  const r = await shell;
  return r.exitCode ?? 1;
}

async function haveCompiler(name: string): Promise<boolean> {
  const probe = isWindows ? $`where ${name}` : $`which ${name}`;
  const r = await probe.nothrow().quiet();
  return r.exitCode === 0;
}

async function verifyDistCompile(toolchain: "clang" | "msvc"): Promise<boolean> {
  if (toolchain === "clang") {
    const obj = "dist/jxl_verify_clang.o";
    const rc = await runCmd(`clang ${clangCFlags("-O1")} -c dist/jxl.c -o ${obj}`, ROOT);
    if (existsSync(join(DIST, "jxl_verify_clang.o"))) rmSync(join(DIST, "jxl_verify_clang.o"));
    return rc === 0;
  }
  const obj = "dist/jxl_verify_msvc.obj";
  const rc = await runCmd(`cl ${JXLDEC_MSVC_CL_C} -c dist/jxl.c -Fo${obj}`, ROOT);
  if (existsSync(join(DIST, "jxl_verify_msvc.obj"))) rmSync(join(DIST, "jxl_verify_msvc.obj"));
  return rc === 0;
}

export async function buildDist(): Promise<void> {
  mkdirSync(DIST, { recursive: true });

  const publicHeader = readFileSync(join(SRC, "jxl.h"), "utf8");
  writeFileSync(DIST_H, publicHeader);

  const parts: string[] = [publicHeader.trimEnd() + "\n"];
  const internal = readFileSync(join(SRC, "jxl_internal.h"), "utf8");
  parts.push(stripIncludes(internal, ["jxl\\.h"]).trimEnd() + "\n");
  for (const name of DIST_MODULES) {
    const code = readFileSync(join(SRC, name), "utf8");
    parts.push(stripIncludes(code, ["jxl_internal\\.h", "jxl\\.h"]).trimEnd() + "\n");
  }

  const amalgamated = stripCComments(parts.join("\n"));
  writeFileSync(DIST_C, amalgamated);
  console.log(`wrote dist/jxl.h (${publicHeader.split("\n").length} lines)`);
  console.log(
    `wrote dist/jxl.c (${amalgamated.split("\n").length} lines, ${DIST_MODULES.length} modules)`,
  );

  const toolchains: ("clang" | "msvc")[] = [];
  if (await haveCompiler("clang")) toolchains.push("clang");
  else {
    console.error("amalgamation verify: clang not found");
    process.exit(1);
  }
  if (isWindows && (await haveCompiler("cl"))) toolchains.push("msvc");

  let failed = false;
  for (const tc of toolchains) {
    console.log(`verifying dist/jxl.c (${tc})...`);
    if (await verifyDistCompile(tc)) console.log(`amalgamation compiles cleanly (${tc})`);
    else {
      console.error(`amalgamation FAILED to compile (${tc})`);
      failed = true;
    }
  }
  if (failed) process.exit(1);
}

export async function ensureDist(): Promise<void> {
  if (distOutdated()) {
    console.log("dist/ outdated, regenerating...");
    await buildDist();
  } else {
    console.log("dist/ up to date");
  }
}

if (import.meta.main) {
  await buildDist();
}
