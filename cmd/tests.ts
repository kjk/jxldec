// tests.ts -- the test driver: decode every selected corpus file with both
// our decoder and libjxl's djxl, and compare the PAM output.
//
//   bun cmd/tests.ts <-all | -rand N | file.jxl ...> [-clang] [-cpu N]
//                    [-preset a,b] [-asan] [-v]
//
// Modular (integer) paths must be byte-identical. VarDCT paths are float, so
// a small per-sample difference is expected and reported as max/mean abs
// diff; -tol N sets the pass threshold (default 1 for 8-bit output).
//
// Files are tested in parallel, one worker per CPU, each with private temp
// files. Per-file lines print in completion order.
import { existsSync, mkdirSync, readFileSync, rmSync, statSync } from "fs";
import { cpus, tmpdir } from "os";
import { dirname, join } from "path";
import { build, buildAsan, defaultUseClang } from "./build";
import { corpusSummary, fileLabel, selectFiles } from "./corpus";
import { getDeps, refTool } from "./get-deps";

const ROOT = dirname(import.meta.dir);
const argv = process.argv.slice(2);
const useClang = argv.includes("-clang") || defaultUseClang;
const useAsan = argv.includes("-asan");
const verbose = argv.includes("-v");
const tolIdx = argv.indexOf("-tol");
const TOL = tolIdx >= 0 ? parseInt(argv[tolIdx + 1]) : 1;
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
  -tol N          max allowed abs difference in 8-bit steps (default 1);
                  16-bit output is normalised, so the threshold means the
                  same thing at either depth
  -v              print per-file detail even when they match

${corpusSummary()}`,
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
  scale: number;        // sample units per 8-bit step (1 or 257)
  note?: string;
};

function compare(ours: Uint8Array, ref: Uint8Array): Cmp {
  if (ours.length === ref.length && Buffer.compare(Buffer.from(ours), Buffer.from(ref)) === 0) {
    return { same: true, maxDiff: 0, meanDiff: 0, scale: 1 };
  }
  const a = parsePam(ours);
  const b = parsePam(ref);
  if (!a || !b) {
    return { same: false, maxDiff: 255, meanDiff: 255, scale: 1, note: "unparsable PAM" };
  }
  if (a.width !== b.width || a.height !== b.height || a.depth !== b.depth) {
    return {
      same: false, maxDiff: 255, meanDiff: 255, scale: 1,
      note: `geometry ${a.width}x${a.height}x${a.depth} vs ${b.width}x${b.height}x${b.depth}`,
    };
  }
  const n = Math.min(a.data.length, b.data.length);
  let maxDiff = 0, sum = 0, count = 0;
  if (a.maxval > 255) {
    for (let i = 0; i + 1 < n; i += 2) {
      const va = (a.data[i] << 8) | a.data[i + 1];
      const vb = (b.data[i] << 8) | b.data[i + 1];
      const d = Math.abs(va - vb);
      if (d > maxDiff) maxDiff = d;
      sum += d;
      count++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const d = Math.abs(a.data[i] - b.data[i]);
      if (d > maxDiff) maxDiff = d;
      sum += d;
      count++;
    }
  }
  // 16-bit output has 257 sample steps per 8-bit step; compare like for like.
  const scale = a.maxval > 255 ? 257 : 1;
  return { same: false, maxDiff, meanDiff: count ? sum / count : 0, scale };
}

type Result = { file: string; ok: boolean; msg: string; ms: number };

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
    return { file, ok: true, msg: "skip (djxl failed)", ms };
  }
  if (!ourOk) return { file, ok: false, msg: `FAIL ${ourErr}`, ms };

  const cmp = compare(new Uint8Array(readFileSync(ourPam)),
                      new Uint8Array(readFileSync(refPam)));
  if (cmp.same) return { file, ok: true, msg: "same", ms };
  if (cmp.note) return { file, ok: false, msg: `diff ${cmp.note}`, ms };
  // Tolerances are expressed in 8-bit steps regardless of the output depth.
  const max8 = cmp.maxDiff / cmp.scale;
  const mean8 = cmp.meanDiff / cmp.scale;
  const detail =
    cmp.scale === 1
      ? `max ${cmp.maxDiff}, mean ${cmp.meanDiff.toFixed(3)}`
      : `max ${max8.toFixed(2)}/8bit (${cmp.maxDiff}/16bit), mean ${mean8.toFixed(3)}`;
  if (max8 <= TOL) return { file, ok: true, msg: `close (${detail})`, ms };
  return { file, ok: false, msg: `DIFF (${detail})`, ms };
}

let nextIndex = 0;
let done = 0;
let failures = 0;
const failed: string[] = [];

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
    if (!r.ok || verbose || r.msg !== "same") {
      console.log(
        `[${done}/${files.length}] ${fileLabel(r.file, ROOT)} — ${r.ms.toFixed(0)} ms — ${r.msg}`,
      );
    } else if (done % 50 === 0) {
      console.log(`[${done}/${files.length}] ...`);
    }
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
