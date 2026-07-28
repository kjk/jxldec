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

async function firstOnPath(name: string): Promise<string | null> {
  const probe = isWindows ? $`where ${name}` : $`which ${name}`;
  const r = await probe.nothrow().quiet();
  if (r.exitCode !== 0) return null;
  const line = r.stdout.toString().trim().split(/\r?\n/)[0]?.trim();
  return line || null;
}

/** Resolve HostX64/x86/cl.exe next to the PATH cl (x64 host) for a Win32 compile. */
async function findMsvcClX86(): Promise<string | null> {
  const cl = await firstOnPath("cl");
  if (!cl) return null;
  const norm = cl.replaceAll("\\", "/");
  if (/\/x86\/cl\.exe$/i.test(norm)) return cl;
  const x86 = norm.replace(/\/x64\/cl\.exe$/i, "/x86/cl.exe");
  if (x86 !== norm && existsSync(x86)) return x86;
  return null;
}

/** Prefer posix-threaded mingw (matches Sumatra Wine CI); fall back to plain. */
async function findMingwGcc(): Promise<string | null> {
  for (const name of [
    "x86_64-w64-mingw32-gcc-posix",
    "x86_64-w64-mingw32-gcc",
  ]) {
    const p = await firstOnPath(name);
    if (p) return p;
  }
  return null;
}

function rmIfExists(...paths: string[]): void {
  for (const p of paths) {
    if (existsSync(p)) rmSync(p);
  }
}

type VerifyVariant = {
  name: string;
  /** false = skip (compiler missing); true = must pass */
  required: boolean;
  run: () => Promise<boolean>;
};

// Compile-only checks for consumer toolchains that have bitten us before:
//   msvc      -- default host MSVC (x64 on CI windows)
//   msvc-x86  -- Win32: _BitScanReverse64 is not available there
//   clang     -- host clang
//   mingw     -- cross gcc without -mxsave: _xgetbv needs inline asm
function distVerifyVariants(): VerifyVariant[] {
  return [
    {
      name: "clang",
      required: true,
      run: async () => {
        if (!(await haveCompiler("clang"))) return false;
        const obj = join(DIST, "jxl_verify_clang.o");
        const rc = await runCmd(
          `clang ${clangCFlags("-O1")} -c dist/jxl.c -o ${obj}`,
          ROOT,
        );
        rmIfExists(obj);
        return rc === 0;
      },
    },
    {
      name: "msvc",
      required: isWindows,
      run: async () => {
        if (!(await haveCompiler("cl"))) return false;
        const obj = join(DIST, "jxl_verify_msvc.obj");
        const rc = await runCmd(
          `cl ${JXLDEC_MSVC_CL_C} -c dist/jxl.c -Fo${obj}`,
          ROOT,
        );
        rmIfExists(obj);
        return rc === 0;
      },
    },
    {
      name: "msvc-x86",
      required: false,
      run: async () => {
        const clx86 = await findMsvcClX86();
        if (!clx86) return false;
        const obj = join(DIST, "jxl_verify_msvc_x86.obj");
        // -WX so an undeclared _BitScanReverse64 (C4013) fails the build.
        const rc = await runCmd(
          `"${clx86}" ${JXLDEC_MSVC_CL_C} -c dist/jxl.c -Fo${obj}`,
          ROOT,
        );
        rmIfExists(obj);
        return rc === 0;
      },
    },
    {
      name: "mingw",
      required: false,
      run: async () => {
        const gcc = await findMingwGcc();
        if (!gcc) return false;
        const obj = join(DIST, "jxl_verify_mingw.o");
        // No -mxsave: gcc's _xgetbv is always_inline and needs that target
        // flag unless we use inline asm (the path consumers / Wine CI use).
        const rc = await runCmd(
          `"${gcc}" -std=c11 -O1 -w -D_CRT_SECURE_NO_WARNINGS -c dist/jxl.c -o ${obj}`,
          ROOT,
        );
        rmIfExists(obj);
        return rc === 0;
      },
    },
  ];
}

/** Compile dist/jxl.c with every available toolchain variant. */
export async function verifyDist(opts: { requireMingw?: boolean; requireMsvcX86?: boolean } = {}): Promise<void> {
  const variants = distVerifyVariants();
  let failed = false;
  let ran = 0;
  for (const v of variants) {
    const force =
      (opts.requireMingw && v.name === "mingw") ||
      (opts.requireMsvcX86 && v.name === "msvc-x86");
    const available =
      v.name === "clang"
        ? await haveCompiler("clang")
        : v.name === "msvc"
          ? isWindows && (await haveCompiler("cl"))
          : v.name === "msvc-x86"
            ? !!(await findMsvcClX86())
            : v.name === "mingw"
              ? !!(await findMingwGcc())
              : false;

    if (!available) {
      if (v.required || force) {
        console.error(`amalgamation verify: ${v.name} required but not found`);
        failed = true;
      } else {
        console.log(`amalgamation verify: skip ${v.name} (compiler not found)`);
      }
      continue;
    }

    console.log(`verifying dist/jxl.c (${v.name})...`);
    ran++;
    if (await v.run()) console.log(`amalgamation compiles cleanly (${v.name})`);
    else {
      console.error(`amalgamation FAILED to compile (${v.name})`);
      failed = true;
    }
  }
  if (ran === 0) {
    console.error("amalgamation verify: no compilers available");
    failed = true;
  }
  if (failed) process.exit(1);
}

export async function buildDist(opts: { requireMingw?: boolean; requireMsvcX86?: boolean } = {}): Promise<void> {
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

  await verifyDist(opts);
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
  const args = process.argv.slice(2);
  await buildDist({
    requireMingw: args.includes("-require-mingw"),
    requireMsvcX86: args.includes("-require-msvc-x86"),
  });
}
