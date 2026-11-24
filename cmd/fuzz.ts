// fuzz.ts -- coverage-guided fuzzing of the JPEG XL decoder (libFuzzer).
//
//   bun cmd/fuzz.ts                    whole files: build, seed, fuzz until killed
//   bun cmd/fuzz.ts -target entropy    one component on its own (see below)
//   bun cmd/fuzz.ts -jobs 8            8 parallel workers sharing the corpus
//   bun cmd/fuzz.ts -runs N            stop after N inputs (a bounded pass)
//   bun cmd/fuzz.ts -msan              under MemorySanitizer (Linux)
//   bun cmd/fuzz.ts -check             replay every known reproducer, every target
//   bun cmd/fuzz.ts -repro FILE        replay one artifact (with -target if not doc)
//   bun cmd/fuzz.ts -minimize          shrink the target's corpus to a covering set
//
// Why: cmd/tests.ts checks that *valid* files decode to the same pixels as
// libjxl, and a valid file never takes an error path -- jxl_errorf, which
// every JXL_ERR in the decoder funnels into, does not execute once across the
// whole corpus (see cmd/coverage.ts). This covers the other half: what the
// decoder does when the bytes lie to it.
//
// Targets (test/fuzz_*.c; cmd/build.ts FUZZ_TARGETS):
//   doc      whole files through the public API, with the TOC-aware custom
//            mutator (test/fuzz_mutator.c) and the container dictionary.
//   headers  container, image header, ICC, every frame header and TOC; no
//            pixels. Microseconds per input.
//   icc      the ICC stream alone.
//   entropy  one entropy-coded stream: histograms, context map, LZ77, symbols.
//   modular  one Modular stream: transforms, MA trees, predictors, samples.
//   diff     differential against libjxl (Linux/macOS): aborts on different
//            geometry or frame count, any integer-path sample, or a gross
//            pixel error elsewhere -- wrong output, not just crashes.
// The component targets run thousands of inputs a second where whole files
// run a few; their seeds are cut from the corpus at the bit each component
// starts (test/fuzz_seeds.c).
//
// Sanitizers: ASan everywhere, with UBSan on top off Windows so every run --
// not only crash replay -- stops on undefined behaviour; -msan builds
// MemorySanitizer instead (Linux; reads of uninitialised memory, which ASan
// does not see). On Linux ASan includes LeakSanitizer, so a leak on an error
// path fails -check there even though the Windows build cannot see it.
//
// fuzz/corpus (doc, shared with cmd/fuzz-afl.ts) and fuzz/corpus-<target> are
// the checkpoints: stop with Ctrl-C, resume by running again. Artifacts land
// in fuzz/crashes (doc) or fuzz/crashes/<target>, which are tracked.
//
// No extra install on Windows: libFuzzer and ASan ship with the VS-bundled
// clang (or LLVM clang on PATH). On macOS use Homebrew's llvm. On
// Debian/Ubuntu, clang needs libclang-rt-<N>-dev for libFuzzer; the first run
// also builds the libjxl tools (cmake + ninja) to generate the seed corpus.
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync } from "fs";
import { join, resolve } from "path";
import { buildFuzz, FUZZ_TARGETS, isWindows, type FuzzSanitizer,
         type FuzzTarget } from "./build";
import { externalSeedFiles, seedComponentCorpusIfEmpty,
         seedCorpusIfEmpty } from "./fuzz-seeds";
import { getDeps } from "./get-deps";

const ROOT = resolve(import.meta.dir, "..");
const FUZZ = join(ROOT, "fuzz");
const SEEDS = join(FUZZ, "seeds");
const DICT = join(ROOT, "test", "fuzz.dict");

function usage(): never {
  console.error(
`usage: bun cmd/fuzz.ts [options]
  (no args)      whole-file target: build, seed if empty, fuzz until killed
  -target T      ${FUZZ_TARGETS.join(" | ")}   (default doc)
  -msan          MemorySanitizer instead of ASan+UBSan (Linux only)
  -jobs N        run N parallel workers sharing the corpus
  -runs N        stop after N inputs instead of running until killed
  -time S        stop after S seconds (libFuzzer -max_total_time)
  -max-len N     max input size in bytes (default 262144 for doc and diff,
                 65536 for the component targets)
  -check         replay every known reproducer through its target and exit
  -repro FILE    replay a single artifact through -target and exit
  -minimize      shrink -target's corpus to a minimal covering set
  -reseed        discard -target's corpus and seed it afresh first
  -h, --help`);
  process.exit(2);
}

const args = process.argv.slice(2);
let target: FuzzTarget = "doc";
let san: FuzzSanitizer = "asan";
let jobs = 1;
let runs = 0;
let seconds = 0;
let maxLen = 0;
let repro = "";
let minimize = false;
let check = false;
let reseed = false;

function intArg(v: string | undefined, name: string): number {
  const n = Number.parseInt(v ?? "", 10);
  if (!Number.isFinite(n) || n < 1) {
    console.error(`${name} requires a positive integer`);
    process.exit(2);
  }
  return n;
}

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === "-target") {
    const t = args[++i] as FuzzTarget;
    if (!FUZZ_TARGETS.includes(t)) usage();
    target = t;
  } else if (a === "-msan") san = "msan";
  else if (a === "-jobs") jobs = intArg(args[++i], "-jobs");
  else if (a === "-runs") runs = intArg(args[++i], "-runs");
  else if (a === "-time") seconds = intArg(args[++i], "-time");
  else if (a === "-max-len") maxLen = intArg(args[++i], "-max-len");
  else if (a === "-repro") repro = args[++i] ?? usage();
  else if (a === "-minimize") minimize = true;
  else if (a === "-check") check = true;
  else if (a === "-reseed") reseed = true;
  else if (a === "-h" || a === "--help") usage();
  else usage();
}

const isComponent = (t: FuzzTarget) => t === "icc" || t === "entropy" || t === "modular";
const corpusDir = (t: FuzzTarget) => join(FUZZ, t === "doc" ? "corpus" : `corpus-${t}`);
const crashDir = (t: FuzzTarget) => t === "doc" ? join(FUZZ, "crashes") : join(FUZZ, "crashes", t);
if (maxLen === 0) maxLen = target === "doc" || target === "diff" ? 262144 : 65536;

// Every artifact kind libFuzzer writes on a finding, plus AFL imports
// (content-hashed names with an afl- prefix from cmd/fuzz-afl.ts -import).
const isArtifact = (name: string) =>
  /^(crash|timeout|oom|leak|slow-unit|afl)-/.test(name);

const listArtifacts = (t: FuzzTarget) => {
  const d = crashDir(t);
  return existsSync(d)
    ? readdirSync(d).filter((n) => isArtifact(n) && statSync(join(d, n)).isFile())
    : [];
};

// Targets this platform can build: diff needs libjxl's non-MSVC static
// libraries, MSan needs Linux.
const available = (t: FuzzTarget) => !(t === "diff" && (isWindows || san === "msan"));

// Runs in fuzz/logs/<target>: with -jobs, libFuzzer writes fuzz-N.log to its
// working directory, and targets fuzzed side by side must not share one.
// Every path passed to it is absolute.
async function run(exe: string, argv: string[]): Promise<number> {
  const cwd = join(FUZZ, "logs", target);
  mkdirSync(cwd, { recursive: true });
  const proc = Bun.spawn([exe, ...argv], {
    cwd, stdout: "inherit", stderr: "inherit",
  });
  return await proc.exited;
}

mkdirSync(FUZZ, { recursive: true });

if (repro) {
  const exe = await buildFuzz(target, san);
  const path = resolve(repro);
  if (!existsSync(path)) {
    console.error(`no such file: ${path}`);
    process.exit(1);
  }
  console.log(`replaying ${path} through ${target}`);
  process.exit(await run(exe, [path]));
}

// -check: every known reproducer, through the target that found it, once.
// No mutation, so it is seconds rather than hours -- a regression gate for
// the crashes already fixed plus the ones other decoders found.
if (check) {
  await getDeps();
  let bad = 0, total = 0;
  for (const t of FUZZ_TARGETS) {
    if (!available(t)) continue;
    const groups: { label: string; files: string[] }[] = [
      { label: t === "doc" ? "fuzz/crashes" : `fuzz/crashes/${t}`,
        files: listArtifacts(t).map((f) => join(crashDir(t), f)) },
    ];
    // Other decoders' reproducers are whole files: the doc and diff targets
    // take them as they are.
    if (t === "doc" || t === "diff") {
      for (const { dir, files } of externalSeedFiles()) {
        groups.push({
          label: dir.replace(/\\/g, "/").replace(/^.*\/deps\//, "deps/") +
                 (t === "doc" ? "" : ` (${t})`),
          files,
        });
      }
    }
    if (groups.every((g) => g.files.length === 0)) continue;
    const exe = await buildFuzz(t, san);
    for (const g of groups) {
      let failed = 0;
      for (const f of g.files) {
        total++;
        const p = Bun.spawnSync({ cmd: [exe, f], stdout: "ignore", stderr: "pipe" });
        if (p.exitCode !== 0) {
          failed++;
          bad++;
          console.log(`  CRASH ${f}`);
          const err = p.stderr.toString().split("\n");
          const line = err.find((l) => l.startsWith("SUMMARY:") || l.startsWith("fuzz_diff:"));
          if (line) console.log(`        ${line.trim()}`);
        }
      }
      console.log(`${g.label}: ${g.files.length - failed}/${g.files.length} clean`);
    }
  }
  console.log(bad === 0
    ? `\nall ${total} reproducer(s) clean`
    : `\n${bad} of ${total} reproducer(s) still crash`);
  process.exit(bad === 0 ? 0 : 1);
}

const exe = await buildFuzz(target, san);
const CORPUS = corpusDir(target);
const CRASHES = crashDir(target);
if (reseed) {
  // Start over from the seeds: what a corpus grown under an older harness
  // or a larger -max-len should get. (The component seeds are re-cut too.)
  console.log(`discarding ${CORPUS}`);
  rmSync(CORPUS, { recursive: true, force: true });
  if (isComponent(target)) rmSync(SEEDS, { recursive: true, force: true });
}
mkdirSync(CORPUS, { recursive: true });
mkdirSync(CRASHES, { recursive: true });

if (minimize) {
  const tmp = `${CORPUS}.min`;
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  console.log(`minimizing ${CORPUS} (libFuzzer -merge=1)...`);
  const rc = await run(exe, ["-merge=1", `-max_len=${maxLen}`, tmp, CORPUS]);
  if (rc !== 0) {
    console.error(`merge failed (${rc}); leaving corpus untouched`);
    rmSync(tmp, { recursive: true, force: true });
    process.exit(rc);
  }
  const before = readdirSync(CORPUS).length;
  rmSync(CORPUS, { recursive: true, force: true });
  renameSync(tmp, CORPUS);
  console.log(`corpus minimized: ${before} -> ${readdirSync(CORPUS).length} inputs`);
  process.exit(0);
}

// First run: seed. The whole-file corpus comes from the test corpus (shared
// with cmd/fuzz-afl.ts); headers and diff read it as a second, read-only
// corpus directory, so they need it too; the component targets get inputs
// cut from the corpus by the seed extractor.
const extraDirs: string[] = [];
if (!isComponent(target)) {
  const docCorpus = corpusDir("doc");
  if (!existsSync(docCorpus) || readdirSync(docCorpus).length === 0) {
    await getDeps();
    seedCorpusIfEmpty(docCorpus, 262144);
  }
  if (target !== "doc") extraDirs.push(docCorpus);
} else if (readdirSync(CORPUS).length === 0) {
  await getDeps();
  await seedComponentCorpusIfEmpty(target, CORPUS, SEEDS);
}

const before = new Set(listArtifacts(target));

// Swallow SIGINT here so the child (which shares the console and gets the
// signal too) exits first and the summary below still prints.
process.on("SIGINT", () => {});

const fuzzArgs = [
  CORPUS,
  ...extraDirs,
  `-artifact_prefix=${CRASHES}/`,
  `-max_len=${maxLen}`,
  "-rss_limit_mb=4096",
  `-timeout=${isComponent(target) || target === "headers" ? 10 : 25}`,
  "-report_slow_units=20",
  "-print_final_stats=1",
  ...(target === "doc" || target === "headers" || target === "diff" ? [`-dict=${DICT}`] : []),
  ...(runs > 0 ? [`-runs=${runs}`] : []),
  ...(seconds > 0 ? [`-max_total_time=${seconds}`] : []),
  ...(jobs > 1 ? [`-jobs=${jobs}`, `-workers=${jobs}`] : []),
];

console.log(`fuzzing ${target} (${san}, ${jobs} ${jobs === 1 ? "process" : "workers"}); ` +
            `corpus=${CORPUS}\n  Ctrl-C to stop; rerun to resume.`);
const rc = await run(exe, fuzzArgs);

const fresh = listArtifacts(target).filter((f) => !before.has(f));
if (fresh.length > 0) {
  console.log(`\n${fresh.length} new artifact(s) in ${CRASHES}:`);
  for (const f of fresh) {
    console.log(`  ${f}`);
    console.log(`    reproduce: ${exe} ${join(CRASHES, f)}`);
    console.log(`    or:        bun cmd/fuzz.ts${target === "doc" ? "" : ` -target ${target}`}` +
                ` -repro ${join(CRASHES, f)}`);
  }
  process.exit(1);
}
process.exit(rc);
