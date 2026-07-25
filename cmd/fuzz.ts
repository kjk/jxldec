// fuzz.ts -- coverage-guided fuzzing of the JPEG XL decoder (libFuzzer).
//
//   bun cmd/fuzz.ts              build, seed corpus if empty, fuzz until killed
//   bun cmd/fuzz.ts -jobs 8      run 8 parallel workers sharing the corpus
//   bun cmd/fuzz.ts -runs N      stop after N inputs (for a bounded CI pass)
//   bun cmd/fuzz.ts -check       replay every known reproducer once (fast gate)
//   bun cmd/fuzz.ts -repro FILE  replay one crash artifact and exit
//   bun cmd/fuzz.ts -minimize    shrink the corpus to a minimal covering set
//
// Why: cmd/tests.ts checks that *valid* files decode to the same pixels as
// libjxl, and a valid file never takes an error path -- jxl_errorf, which
// every JXL_ERR in the decoder funnels into, does not execute once across the
// whole corpus (see cmd/coverage.ts). This covers the other half: what the
// decoder does when the bytes lie to it.
//
// fuzz/corpus IS the checkpoint. Stop with Ctrl-C, resume by running again;
// libFuzzer reloads the corpus and continues. Artifacts land in fuzz/crashes,
// each reproducible with `out/fuzz/jxl_fuzz.exe <file>`.
//
// No extra install on Windows: libFuzzer and ASan ship with the VS-bundled
// clang (or LLVM clang on PATH). On macOS use Homebrew's llvm.
import { copyFileSync, existsSync, mkdirSync, readdirSync, renameSync,
         rmSync, statSync } from "fs";
import { basename, join, resolve } from "path";
import { buildFuzz, FUZZ_EXE } from "./build";
import { corpusFiles } from "./corpus";
import { getDeps, JXL_OXIDE_DIR, LIBJXL_DIR } from "./get-deps";

const ROOT = resolve(import.meta.dir, "..");
const FUZZ = join(ROOT, "fuzz");
const CORPUS = join(FUZZ, "corpus");
const CRASHES = join(FUZZ, "crashes");

// Other people's crash reproducers, which are worth far more per byte than
// anything we generate: each is a minimised input that already broke a JPEG XL
// decoder. Both live in deps/, which get-deps already fetches, so they are
// referenced rather than copied into this repo.
//   libjxl   -- OSS-Fuzz testcases against djxl_fuzzer
//   jxl-oxide -- regression findings, each named for the bug it caught
//                (dequant_matrix_band, hf_varblock_across_group, icc_parse_oob)
const EXTERNAL_SEEDS = [
  join(LIBJXL_DIR, "testdata", "oss-fuzz"),
  join(JXL_OXIDE_DIR, "crates", "jxl-oxide-tests", "tests", "fuzz_findings"),
];


function usage(): never {
  console.error(
`usage: bun cmd/fuzz.ts [options]
  (no args)      build, seed corpus if empty, fuzz until killed (resumes)
  -jobs N        run N parallel workers sharing the corpus
  -runs N        stop after N inputs instead of running until killed
  -max-len N     max input size in bytes (default 1000000)
  -check         replay every known reproducer once and exit (fast gate)
  -repro FILE    replay a single crash artifact and exit
  -minimize      shrink the corpus to a minimal covering set
  -h, --help`);
  process.exit(2);
}

const args = process.argv.slice(2);
let jobs = 1;
let runs = 0;
let maxLen = 1000000;
let repro = "";
let minimize = false;
let check = false;

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
  if (a === "-jobs") jobs = intArg(args[++i], "-jobs");
  else if (a === "-runs") runs = intArg(args[++i], "-runs");
  else if (a === "-max-len") maxLen = intArg(args[++i], "-max-len");
  else if (a === "-repro") repro = args[++i] ?? usage();
  else if (a === "-minimize") minimize = true;
  else if (a === "-check") check = true;
  else if (a === "-h" || a === "--help") usage();
  else usage();
}

// Every artifact kind libFuzzer writes on a finding; all replay the same way.
const isArtifact = (name: string) =>
  /^(crash|timeout|oom|leak|slow-unit)-/.test(name);

const listArtifacts = () =>
  existsSync(CRASHES) ? readdirSync(CRASHES).filter(isArtifact) : [];

async function run(argv: string[]): Promise<number> {
  const proc = Bun.spawn([FUZZ_EXE, ...argv], {
    cwd: FUZZ, stdout: "inherit", stderr: "inherit",
  });
  return await proc.exited;
}

const exe = await buildFuzz();
mkdirSync(CORPUS, { recursive: true });
mkdirSync(CRASHES, { recursive: true });

if (repro) {
  const path = resolve(repro);
  if (!existsSync(path)) {
    console.error(`no such file: ${path}`);
    process.exit(1);
  }
  console.log(`replaying ${path}`);
  process.exit(await run([path]));
}

// -check: replay every known reproducer once and stop. No mutation, so it is
// seconds rather than hours -- a regression gate for the crashes we have
// already fixed plus the ones other decoders found.
if (check) {
  await getDeps();
  const groups: { label: string; files: string[] }[] = [
    { label: "fuzz/crashes", files: listArtifacts().map((f) => join(CRASHES, f)) },
    ...EXTERNAL_SEEDS.filter(existsSync).map((dir) => ({
      label: dir.replace(/\\/g, "/").replace(/^.*\/deps\//, "deps/"),
      files: readdirSync(dir).map((n) => join(dir, n))
        .filter((f) => statSync(f).isFile()),
    })),
  ];
  let bad = 0, total = 0;
  for (const g of groups) {
    let failed = 0;
    for (const f of g.files) {
      total++;
      const p = Bun.spawnSync({ cmd: [FUZZ_EXE, f], stdout: "ignore", stderr: "pipe" });
      if (p.exitCode !== 0) {
        failed++;
        bad++;
        console.log(`  CRASH ${f}`);
        const line = p.stderr.toString().split("\n")
          .find((l) => l.startsWith("SUMMARY:"));
        if (line) console.log(`        ${line.trim()}`);
      }
    }
    console.log(`${g.label}: ${g.files.length - failed}/${g.files.length} clean`);
  }
  console.log(bad === 0
    ? `\nall ${total} reproducer(s) clean`
    : `\n${bad} of ${total} reproducer(s) still crash`);
  process.exit(bad === 0 ? 0 : 1);
}

if (minimize) {
  const tmp = join(FUZZ, "corpus.min");
  rmSync(tmp, { recursive: true, force: true });
  mkdirSync(tmp, { recursive: true });
  console.log("minimizing corpus (libFuzzer -merge=1)...");
  const rc = await run(["-merge=1", "corpus.min", "corpus"]);
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

// First run: seed from the real corpus so mutation starts from valid
// codestreams rather than random bytes. The generated corpus has unique
// names by construction, but the shipped .jxl files can collide with them.
if (readdirSync(CORPUS).length === 0) {
  await getDeps();
  let n = 0, skipped = 0, ext = 0;
  const take = (f: string, prefix: string) => {
    if (statSync(f).size > maxLen) { skipped++; return false; }
    let dest = join(CORPUS, prefix + basename(f));
    if (existsSync(dest)) dest = join(CORPUS, `${prefix}${n}_${basename(f)}`);
    copyFileSync(f, dest);
    return true;
  };
  for (const f of corpusFiles()) if (take(f, "")) n++;
  for (const dir of EXTERNAL_SEEDS) {
    if (!existsSync(dir)) continue;
    for (const name of readdirSync(dir)) {
      const f = join(dir, name);
      if (statSync(f).isFile() && take(f, "ext_")) ext++;
    }
  }
  console.log(`seeded corpus with ${n} file(s) + ${ext} external reproducer(s)` +
              (skipped ? `; ${skipped} skipped as larger than -max-len` : ""));
}

const before = new Set(listArtifacts());

// Swallow SIGINT here so the child (which shares the console and gets the
// signal too) exits first and the summary below still prints.
process.on("SIGINT", () => {});

const fuzzArgs = [
  "corpus",
  "-artifact_prefix=crashes/",
  `-max_len=${maxLen}`,
  "-rss_limit_mb=4096",
  "-timeout=25",
  "-report_slow_units=20",
  "-print_final_stats=1",
  ...(runs > 0 ? [`-runs=${runs}`] : []),
  ...(jobs > 1 ? [`-jobs=${jobs}`, `-workers=${jobs}`] : []),
];

console.log(`fuzzing (${jobs} ${jobs === 1 ? "process" : "workers"}); ` +
            `corpus=${CORPUS}\n  Ctrl-C to stop; rerun to resume.`);
const rc = await run(fuzzArgs);

const fresh = listArtifacts().filter((f) => !before.has(f));
if (fresh.length > 0) {
  console.log(`\n${fresh.length} new crash artifact(s) in fuzz/crashes:`);
  for (const f of fresh) {
    console.log(`  ${f}`);
    console.log(`    reproduce: ${exe} ${join(CRASHES, f)}`);
    console.log(`    or:        bun cmd/fuzz.ts -repro ${join("fuzz", "crashes", f)}`);
  }
  process.exit(1);
}
process.exit(rc);
