// bench.ts -- benchmark our decoder against libjxl.
//
//   bun cmd/bench.ts <file.jxl ... | -rand N | -all> [-runs N] [-preset a,b]
//
// Builds jxl_bench, which links the dist/ amalgamation and libjxl's static
// libraries into one process, then decodes each file with both. Both produce
// 8-bit interleaved samples of the first frame; libjxl runs single-threaded,
// the only configuration we implement. The best of -runs decodes is reported
// per decoder, since the fastest run is the least perturbed one.
//
// Windows/MSVC only: libjxl is C++ and links against the MSVC runtime.
import { dirname } from "path";
import { buildBenchHarness, isWindows } from "./build";
import { corpusSummary, selectFiles } from "./corpus";
import { getDeps } from "./get-deps";

const ROOT = dirname(import.meta.dir);
const argv = process.argv.slice(2);
const runsIdx = argv.indexOf("-runs");
const RUNS = runsIdx >= 0 ? argv[runsIdx + 1] : "3";

if (!isWindows) {
  console.error("bench: the harness links libjxl's MSVC build; Windows only.");
  process.exit(2);
}

await getDeps();

const files = selectFiles(
  `usage: bun cmd/bench.ts <selection> [options]
selection (required; default prints this help):
  file.jxl ...    benchmark the given files
  -rand N         benchmark N randomly selected corpus files
  -all            benchmark every corpus file
options:
  -runs N         decodes per file per decoder, best wins (default 3)
  -preset a,b     restrict the generated corpus to these presets

${corpusSummary()}`,
  ["-rand", "-runs", "-preset"],
);

const EXE = await buildBenchHarness();
const proc = Bun.spawnSync({
  cmd: [EXE, "-runs", RUNS, ...files],
  stdout: "inherit",
  stderr: "inherit",
  cwd: ROOT,
});
process.exit(proc.exitCode ?? 1);
