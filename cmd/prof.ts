// prof.ts -- sampling profile of one decode on Windows via ../winperf
// (-vs profiles ours and libjxl together).
//
//   bun cmd/prof.ts <file.jxl> [-runs N] [-hz N] [-vs]
//
// Builds jxl_prof (our amalgamation only, with -Zi so the profiler can
// symbolize it), then records it under winperf and prints winperf's agent
// report: top self-time functions, hot source lines, and the heaviest call
// path.
//
// Windows only. winperf drives xperf (Windows Performance Toolkit, from the
// ADK) and needs Administrator rights (UAC prompt).
//
// The harness brackets its decode loop with winperf section marks
// (test/winperf_control.h), so samples taken while reading the file or
// starting up are dropped instead of diluting the profile.
//
// Build winperf once:
//   cd ../winperf && bun cmd/build.ts -release
// Or clone it if missing:
//   git clone https://github.com/kjk/winperf ..\winperf
import { existsSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { buildBenchHarnessDbg, buildProfHarness, isWindows } from "./build";

const ROOT = dirname(import.meta.dir);
// JXLDEC_WINPERF overrides; otherwise try the usual sibling checkout paths.
const WINPERF = (() => {
  const env = process.env.JXLDEC_WINPERF;
  if (env && existsSync(env)) return env;
  for (const p of [
    "../winperf/out/rel64/winperf.exe",
    "../winperf/out/dbg64/winperf.exe",
    "../winperf/out/winperf.exe",
  ]) {
    const abs = resolve(ROOT, p);
    if (existsSync(abs)) return abs;
  }
  return resolve(ROOT, "../winperf/out/rel64/winperf.exe");
})();

const argv = process.argv.slice(2);
const flagVal = (name: string, dflt: string) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const RUNS = flagVal("-runs", "10");
const HZ = flagVal("-hz", "4000");
// Only these take a value; treating every flag as if it did would swallow
// the filename after a boolean one like -vs.
const VALUE_FLAGS = ["-runs", "-hz"];
const files = argv.filter(
  (a, i) => !a.startsWith("-") && !VALUE_FLAGS.includes(argv[i - 1] ?? ""));

if (!isWindows || files.length !== 1) {
  console.error(`usage: bun cmd/prof.ts <file.jxl> [-runs N] [-hz N] [-vs]
  -runs N   decodes of the file inside the profiled process (default 10)
  -hz N     sampling rate (default 4000)
  -vs       profile ours *and* libjxl together, both symbolized (builds a
            debug-info libjxl the first time, which is slow)

Windows only: winperf records with xperf and needs Administrator rights
(and the Windows Performance Toolkit from the ADK).

Build winperf:    cd ../winperf && bun cmd/build.ts -release
Clone if missing: git clone https://github.com/kjk/winperf ..\\winperf`);
  process.exit(2);
}
if (!existsSync(WINPERF)) {
  console.error(`prof: ${WINPERF} not found -- build it with
  cd ../winperf && bun cmd/build.ts -release
Or clone first:
  git clone https://github.com/kjk/winperf ..\\winperf`);
  process.exit(2);
}

const file = resolve(files[0]);
if (!existsSync(file)) {
  console.error(`prof: no such file: ${files[0]}`);
  process.exit(1);
}

// -vs profiles the benchmark harness instead, which decodes with *both*
// decoders in one process. Linked against a debug-info libjxl, so libjxl's
// own functions are attributed rather than showing up as one opaque module --
// that is what makes "where are we slower than libjxl" answerable. It carries
// no section marks, so that profile covers the whole process.
const vs = argv.includes("-vs");
const EXE = vs ? await buildBenchHarnessDbg() : await buildProfHarness();
const outDir = resolve(ROOT, "out/prof");
mkdirSync(outDir, { recursive: true });
// Keep the .etl and the Firefox profile JSON inside out/, which is gitignored.
const OUT = resolve(outDir, "winperf.etl");

console.log(`prof: ${WINPERF}`);
console.log(`  exe:  ${EXE}`);
console.log(`  work: -runs ${RUNS} ${file}`);
console.log(`  out:  ${OUT}`);

const proc = Bun.spawnSync({
  // Absolute paths: winperf silently fails to attach to a relative one, which
  // shows up as a trace full of unrelated system processes rather than an
  // error.
  cmd: [WINPERF, "record", "-i", HZ, "-o", OUT, "-print-agent",
        "--", resolve(EXE), "-runs", RUNS, file],
  stdout: "inherit",
  stderr: "inherit",
  cwd: ROOT,
});
process.exitCode = proc.exitCode ?? 1;
