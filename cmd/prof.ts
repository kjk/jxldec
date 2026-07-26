// prof.ts -- profile our decoder on one file with ../samply (-vs profiles
// ours and libjxl together).
//
//   bun cmd/prof.ts <file.jxl> [-runs N] [-hz N]
//
// Builds jxl_prof (our amalgamation only, with -Zi so the profiler can
// symbolize it), then records it under samply and prints samply's agent
// report: top self-time functions, hot source lines, and the heaviest call
// path. Windows only -- samply drives xperf, which needs the Windows
// Performance Toolkit and Administrator rights.
import { existsSync } from "fs";
import { dirname, resolve } from "path";
import { buildBenchHarnessDbg, buildProfHarness, isWindows } from "./build";

const ROOT = dirname(import.meta.dir);
// JXLDEC_SAMPLY overrides; otherwise try the two places it usually lives.
const SAMPLY = (() => {
  const env = process.env.JXLDEC_SAMPLY;
  if (env && existsSync(env)) return env;
  for (const p of ["../samply/out/rel64/samply.exe",
                   "../exp/samply/out/rel64/samply.exe"]) {
    const abs = resolve(ROOT, p);
    if (existsSync(abs)) return abs;
  }
  return resolve(ROOT, "../samply/out/rel64/samply.exe");
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
  console.error(`usage: bun cmd/prof.ts <file.jxl> [-runs N] [-hz N]
  -runs N   decodes of the file inside the profiled process (default 10)
  -hz N     sampling rate (default 4000)
  -vs       profile ours *and* libjxl together, both symbolized (builds a
            debug-info libjxl the first time, which is slow)
Windows only: samply records with xperf and needs Administrator rights.`);
  process.exit(2);
}
if (!existsSync(SAMPLY)) {
  console.error(`prof: ${SAMPLY} not found -- build it with
  cd ../samply && bun cmd/build.ts -release`);
  process.exit(2);
}

// -vs profiles the benchmark harness instead, which decodes with *both*
// decoders in one process. Linked against a debug-info libjxl, so libjxl's
// own functions are attributed rather than showing up as one opaque module --
// that is what makes "where are we slower than libjxl" answerable.
const vs = argv.includes("-vs");
const EXE = vs ? await buildBenchHarnessDbg() : await buildProfHarness();
// Keep the .etl and the Firefox profile JSON inside out/, which is gitignored.
const OUT = `${ROOT}/out/prof/samply.etl`;
const proc = Bun.spawnSync({
  // Absolute paths: samply silently fails to attach to a relative one, which
  // shows up as a trace full of unrelated system processes.
  cmd: [SAMPLY, "record", "-i", HZ, "-o", OUT, "-print-agent",
        "--", resolve(EXE), "-runs", RUNS, resolve(files[0])],
  stdout: "inherit",
  stderr: "inherit",
  cwd: ROOT,
});
process.exitCode = proc.exitCode ?? 1;
