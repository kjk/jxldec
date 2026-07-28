// bench.ts -- benchmark our decoder against libjxl.
//
//   bun cmd/bench.ts <file.jxl ... | -rand N | -all> [-runs N] [-preset a,b] [-bgra]
//
// Builds jxl_bench, which links the dist/ amalgamation and libjxl's static
// libraries into one process, then decodes each file with both. The selection
// goes over in a list file: the whole corpus does not fit in a Windows
// command line. Both produce
// 8-bit interleaved samples of the first frame; libjxl runs single-threaded,
// the only configuration we implement. The best of -runs decodes is reported
// per decoder, since the fastest run is the least perturbed one.
//
// Default output (from jxl_bench): directory header lines, then
//   libjxl   jxldec     diff    %diff basename.jxl : N bytes
// (+ = jxldec slower). Last data line is sum of best times, label "total".
//
// Windows/MSVC only: libjxl is C++ and links against the MSVC runtime.
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { dirname, isAbsolute, join, relative, resolve } from "path";
import { buildBenchHarness, isWindows } from "./build";
import { corpusSummary, selectFiles } from "./corpus";
import { getDeps } from "./get-deps";

const ROOT = dirname(import.meta.dir);
const argv = process.argv.slice(2);
const runsIdx = argv.indexOf("-runs");
const RUNS = runsIdx >= 0 ? argv[runsIdx + 1] : "3";
const BGRA = argv.includes("-bgra");

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
  -bgra           request BGRA8 from ours and RGBA8 from libjxl

Default: dir headers, then libjxl jxldec diff %diff basename (+ = jxldec slower);
  ends with a "total" line (sum of best libjxl vs sum of best jxldec).

${corpusSummary()}`,
  ["-rand", "-runs", "-preset"],
);

const EXE = await buildBenchHarness();
const tmp = mkdtempSync(join(tmpdir(), "jxldec-bench-"));
const listFile = join(tmp, "files.txt");
const displayPath = (file: string): string => {
  const absolute = resolve(file);
  const rel = relative(ROOT, absolute);
  return (rel.startsWith("..") || isAbsolute(rel) ? absolute : rel)
    .replaceAll("\\", "/");
};
writeFileSync(listFile, files.map(displayPath).join("\n") + "\n");
try {
  const proc = Bun.spawnSync({
    cmd: [EXE, "-runs", RUNS, ...(BGRA ? ["-bgra"] : []),
          "-list", listFile],
    stdout: "inherit",
    stderr: "inherit",
    cwd: ROOT,
  });
  process.exitCode = proc.exitCode ?? 1;
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
