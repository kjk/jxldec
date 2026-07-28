// tests.ts -- the test driver: decode every selected corpus file with both
// our decoder and libjxl's djxl, and compare the PAM output.
//
//   bun cmd/tests.ts <-all | -rand N | file.jxl ...> [-clang] [-cpu N]
//                    [-preset a,b] [-asan] [-tol N] [-rms X]
//
// Modular (integer) paths must be byte-identical. VarDCT paths are float, so
// a small per-sample difference is expected; a file passes when it is under
// both thresholds, RMS (-rms, default 0.6) and peak (-tol, default 3), in
// 8-bit steps. See the note above PEAK_TOL for why both are needed.
//
// Files are tested in parallel, one worker per CPU, each with private temp
// files. Per-file lines print in completion order. Output is bench-style:
// directory header when the dir changes, then
//   time max rms mean basename [status] : size bytes : n/m
// Status (FAIL / DIFF / skip) is only printed when the file is not a pass.
import { existsSync, mkdirSync, readFileSync, rmSync, statSync } from "fs";
import { cpus, tmpdir } from "os";
import { basename, dirname, isAbsolute, join, relative } from "path";
import { build, buildAsan, defaultUseClang } from "./build";
import { corpusSummary, selectFiles } from "./corpus";
import { getDeps, refTool } from "./get-deps";

const ROOT = dirname(import.meta.dir);
const argv = process.argv.slice(2);
const useClang = argv.includes("-clang") || defaultUseClang;
const useAsan = argv.includes("-asan");
// Two thresholds, both required, in the shape libjxl's own conformance
// checker uses (tools/conformance/conformance.py CompareNPY: an RMSE limit
// and a peak limit, never peak alone).
//
// Peak alone does not work as the primary gate: a single sample decides a
// whole megapixel file, so the old peak<=1 default failed 63 files whose
// mean-error distribution was indistinguishable from the 377 that passed,
// and left 289 of those passes sitting exactly on the threshold. RMS is what
// separates "different IDCT factorisation" from "wrong". The peak is kept as
// a second gate because a handful of very wrong samples can hide under a
// good RMS -- which is exactly what the loop-filter edge bug did.
//
// 3 rather than 1: with that bug fixed no corpus file exceeds a peak of 1,
// so 3 catches any gross single-pixel breakage while leaving the headroom
// that made peak<=1 flip on harmless rounding.
const tolIdx = argv.indexOf("-tol");
const PEAK_TOL = tolIdx >= 0 ? parseFloat(argv[tolIdx + 1]) : 3;
// 0.6 is picked off the measured distribution, not by taste. Across the 440
// non-byte-exact corpus files RMS runs dense up to 0.505 and then stops: when
// every differing sample is off by exactly one count, rms = sqrt(fraction
// differing), so pure rounding noise saturates near 0.5 and cannot go higher.
// 0.6 clears that ceiling while staying well under 1.0, which is what a
// systematic one-count shift across the whole image would score.
const rmsIdx = argv.indexOf("-rms");
const RMS_TOL = rmsIdx >= 0 ? parseFloat(argv[rmsIdx + 1]) : 0.6;
const cpuIdx = argv.indexOf("-cpu");
const NCPU = cpuIdx >= 0 ? parseInt(argv[cpuIdx + 1]) : Math.max(1, cpus().length - 1);

await getDeps();

const files = selectFiles(
  `usage: bun cmd/tests.ts <selection> [options]
selection (required; default prints this help):
  file.jxl ...    test the given files
  -rand N         test N randomly selected corpus files
  -all            test every corpus file
options:
  -preset a,b     restrict the generated corpus to these presets
  -clang          build/test with clang instead of MSVC
  -asan           build and run the clang+AddressSanitizer harness
  -cpu N          worker count (default: cores - 1)
  -tol N          max allowed peak abs difference, 8-bit steps (default 3)
  -rms X          max allowed RMS difference, 8-bit steps (default 0.6);
                  a file must pass both. 16-bit output is normalised, so
                  either threshold means the same thing at either depth

${corpusSummary()}`,
  // Every flag that takes a value, so its value is not mistaken for a
  // filename. -tol was missing from this list, which made `-tol 3 file.jxl`
  // fail with "no such file: 3".
  ["-rand", "-cpu", "-preset", "-tol", "-rms"],
);

const EXE = useAsan ? await buildAsan() : await build(useClang);
const DJXL = refTool("djxl");

type Pam = { header: string; width: number; height: number; depth: number;
             maxval: number; data: Uint8Array };

function parsePam(buf: Uint8Array): Pam | null {
  const text = new TextDecoder("latin1").decode(buf.subarray(0, 200));
  const end = text.indexOf("ENDHDR\n");
  if (!text.startsWith("P7") || end < 0) return null;
  const header = text.slice(0, end + 7);
  const num = (key: string) => {
    const m = header.match(new RegExp(`^${key} (\\d+)$`, "m"));
    return m ? parseInt(m[1]) : 0;
  };
  return {
    header,
    width: num("WIDTH"),
    height: num("HEIGHT"),
    depth: num("DEPTH"),
    maxval: num("MAXVAL"),
    data: buf.subarray(header.length),
  };
}

type Cmp = {
  same: boolean;
  maxDiff: number;      // in the file's own sample units
  meanDiff: number;
  rmsDiff: number;
  scale: number;        // sample units per 8-bit step (1 or 257)
  note?: string;
};

function compare(ours: Uint8Array, ref: Uint8Array): Cmp {
  if (ours.length === ref.length && Buffer.compare(Buffer.from(ours), Buffer.from(ref)) === 0) {
    return { same: true, maxDiff: 0, meanDiff: 0, rmsDiff: 0, scale: 1 };
  }
  const a = parsePam(ours);
  const b = parsePam(ref);
  if (!a || !b) {
    return { same: false, maxDiff: 255, meanDiff: 255, rmsDiff: 255, scale: 1,
             note: "unparsable PAM" };
  }
  if (a.width !== b.width || a.height !== b.height || a.depth !== b.depth) {
    return {
      same: false, maxDiff: 255, meanDiff: 255, rmsDiff: 255, scale: 1,
      note: `geometry ${a.width}x${a.height}x${a.depth} vs ${b.width}x${b.height}x${b.depth}`,
    };
  }
  const n = Math.min(a.data.length, b.data.length);
  let maxDiff = 0, sum = 0, sumSq = 0, count = 0;
  if (a.maxval > 255) {
    for (let i = 0; i + 1 < n; i += 2) {
      const va = (a.data[i] << 8) | a.data[i + 1];
      const vb = (b.data[i] << 8) | b.data[i + 1];
      const d = Math.abs(va - vb);
      if (d > maxDiff) maxDiff = d;
      sum += d;
      sumSq += d * d;
      count++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const d = Math.abs(a.data[i] - b.data[i]);
      if (d > maxDiff) maxDiff = d;
      sum += d;
      sumSq += d * d;
      count++;
    }
  }
  // 16-bit output has 257 sample steps per 8-bit step; compare like for like.
  const scale = a.maxval > 255 ? 257 : 1;
  return {
    same: false,
    maxDiff,
    meanDiff: count ? sum / count : 0,
    rmsDiff: count ? Math.sqrt(sumSq / count) : 0,
    scale,
  };
}

type Result = {
  file: string;
  ok: boolean;
  ms: number;
  /** Peak / RMS / mean abs sample diff in 8-bit steps; omitted if no compare. */
  max8?: number;
  rms8?: number;
  mean8?: number;
  /** Only set when not a pass (FAIL / DIFF / skip). */
  status?: string;
};

async function testOne(file: string, slot: number): Promise<Result> {
  const tmp = join(tmpdir(), "jxldec-tests", String(slot));
  mkdirSync(tmp, { recursive: true });
  const ourPam = join(tmp, "our.pam");
  const refPam = join(tmp, "ref.pam");
  for (const p of [ourPam, refPam]) if (existsSync(p)) rmSync(p);

  const t0 = performance.now();
  const ourProc = Bun.spawnSync({
    cmd: [EXE, "-out", ourPam, file],
    stdout: "pipe",
    stderr: "pipe",
  });
  const ms = performance.now() - t0;
  const refProc = Bun.spawnSync({
    cmd: [DJXL, file, refPam],
    stdout: "ignore",
    stderr: "pipe",
  });

  const refOk = refProc.exitCode === 0 && existsSync(refPam);
  const ourOk = ourProc.exitCode === 0 && existsSync(ourPam);
  const ourErr = ourProc.stderr.toString().trim().split("\n")[0] ?? "";

  if (!refOk) {
    // libjxl itself can't decode it: not our problem, count as a skip.
    return { file, ok: true, ms, status: "skip (djxl failed)" };
  }
  if (!ourOk) return { file, ok: false, ms, status: `FAIL ${ourErr}` };

  const cmp = compare(new Uint8Array(readFileSync(ourPam)),
                      new Uint8Array(readFileSync(refPam)));
  if (cmp.same) {
    return { file, ok: true, ms, max8: 0, rms8: 0, mean8: 0 };
  }
  if (cmp.note) {
    return { file, ok: false, ms, status: `DIFF ${cmp.note}` };
  }
  // Tolerances are expressed in 8-bit steps regardless of the output depth.
  const max8 = cmp.maxDiff / cmp.scale;
  const mean8 = cmp.meanDiff / cmp.scale;
  const rms8 = cmp.rmsDiff / cmp.scale;
  if (rms8 <= RMS_TOL && max8 <= PEAK_TOL) {
    return { file, ok: true, ms, max8, rms8, mean8 };
  }
  const why = rms8 > RMS_TOL ? `rms>${RMS_TOL}` : `peak>${PEAK_TOL}`;
  return { file, ok: false, ms, max8, rms8, mean8, status: `DIFF ${why}` };
}

/** Path relative to ROOT with forward slashes (absolute if outside the tree). */
function displayPath(file: string): string {
  const rel = relative(ROOT, file);
  return (rel.startsWith("..") || isAbsolute(rel) ? file : rel).replaceAll(
    "\\",
    "/",
  );
}

function pathDir(file: string): string {
  const d = dirname(displayPath(file));
  return d === "" ? "." : d;
}

function pathBase(file: string): string {
  return basename(displayPath(file));
}

function colNum(n: number | undefined, width: number, digits: number): string {
  if (n === undefined) return "-".padStart(width);
  return n.toFixed(digits).padStart(width);
}

/** Exact byte count with thousands separators (e.g. 13,234). */
function formatBytesExact(n: number): string {
  return n.toLocaleString("en-US");
}

let lastDir = "";

function printResult(r: Result, n: number, total: number) {
  const dir = pathDir(r.file);
  if (dir !== lastDir) {
    console.log(dir);
    lastDir = dir;
  }
  const time = `${r.ms.toFixed(0)} ms`.padStart(8);
  const max = colNum(r.max8, 6, 2);
  const rms = colNum(r.rms8, 6, 2);
  const mean = colNum(r.mean8, 6, 2);
  const name = pathBase(r.file);
  const status = r.status ? ` ${r.status}` : "";
  const size = formatBytesExact(statSync(r.file).size);
  console.log(
    `${time} ${max} ${rms} ${mean} ${name}${status} : ${size} bytes : ${n}/${total}`,
  );
}

let nextIndex = 0;
let done = 0;
let failures = 0;
const failed: string[] = [];

console.log(
  `max = peak |sample diff| (8-bit steps); rms/mean = over all samples` +
    `  (pass: max<=${PEAK_TOL} and rms<=${RMS_TOL})`,
);
console.log(
  `${"time".padStart(8)} ${"max".padStart(6)} ${"rms".padStart(6)} ${"mean".padStart(6)} file`,
);

async function worker(slot: number) {
  for (;;) {
    const i = nextIndex++;
    if (i >= files.length) return;
    const r = await testOne(files[i], slot);
    done++;
    if (!r.ok) {
      failures++;
      failed.push(files[i]);
    }
    printResult(r, done, files.length);
  }
}

const t0 = performance.now();
await Promise.all(Array.from({ length: Math.min(NCPU, files.length) }, (_, i) => worker(i)));
const secs = (performance.now() - t0) / 1000;

console.log(
  `\n${files.length - failures}/${files.length} ok, ${failures} failed (${secs.toFixed(1)}s)`,
);
if (failures) {
  console.log("failing files:");
  for (const f of failed.slice(0, 40)) console.log(`  ${f}`);
  if (failed.length > 40) console.log(`  ... and ${failed.length - 40} more`);
}
process.exit(failures ? 1 : 0);
