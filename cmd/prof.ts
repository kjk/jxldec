// prof.ts -- profile our decoder on one file with ../samply.
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
import { buildProfHarness, isWindows } from "./build";

const ROOT = dirname(import.meta.dir);
const SAMPLY = resolve(ROOT, "../samply/out/rel64/samply.exe");

const argv = process.argv.slice(2);
const flagVal = (name: string, dflt: string) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
};
const RUNS = flagVal("-runs", "10");
const HZ = flagVal("-hz", "4000");
const files = argv.filter((a, i) => !a.startsWith("-") && !argv[i - 1]?.startsWith("-"));

if (!isWindows || files.length !== 1) {
  console.error(`usage: bun cmd/prof.ts <file.jxl> [-runs N] [-hz N]
  -runs N   decodes of the file inside the profiled process (default 10)
  -hz N     sampling rate (default 4000)
Windows only: samply records with xperf and needs Administrator rights.`);
  process.exit(2);
}
if (!existsSync(SAMPLY)) {
  console.error(`prof: ${SAMPLY} not found -- build it with
  cd ../samply && bun cmd/build.ts -release`);
  process.exit(2);
}

const EXE = await buildProfHarness();
// Keep the .etl and the Firefox profile JSON inside out/, which is gitignored.
const OUT = `${ROOT}/out/prof/samply.etl`;
const proc = Bun.spawnSync({
  cmd: [SAMPLY, "record", "-i", HZ, "-o", OUT, "-print-agent",
        "--", EXE, "-runs", RUNS, files[0]],
  stdout: "inherit",
  stderr: "inherit",
  cwd: ROOT,
});
process.exitCode = proc.exitCode ?? 1;
