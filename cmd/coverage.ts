// coverage.ts -- which decoder code does the corpus never reach?
//
//   bun cmd/coverage.ts              every corpus file
//   bun cmd/coverage.ts -rand 200    a sample, for a quick look
//   bun cmd/coverage.ts -cpu N       worker count (default: cores - 1)
//   bun cmd/coverage.ts -all-fns     list every uncovered function, not the top 40
//   bun cmd/coverage.ts -reuse       re-report from the profile already on disk
//   bun cmd/coverage.ts -dir fuzz/corpus   every file in a directory instead
//                                    (repeatable; any names, valid or not)
//
// Builds the clang coverage variant, decodes the corpus with it, merges the
// profiles and prints two things: per-file line coverage, and the functions
// that never ran at all. The second list is the point. Every decoder bug found
// so far -- the noise seed, the loop-filter edge, missing upsampling, spot
// colour -- lived in a path nothing in the corpus exercised, and each was found
// by guessing where those were. This measures it instead.
//
// A function listed here is not necessarily broken. It means nothing verifies
// it against libjxl, which is the state every one of those bugs was in.
//
// -dir points it at a fuzzer's corpus (fuzz/corpus, the whole-file target's)
// to find the fuzzer's blind spots: code the mutated inputs never reach is
// where seeds are missing. Invalid inputs are fine; a decode that fails
// still counts what it ran.
import { existsSync, mkdirSync, readdirSync, rmSync, statSync } from "fs";
import { cpus, tmpdir } from "os";
import { dirname, join } from "path";
import { buildCoverage } from "./build";
import { corpusFiles } from "./corpus";
import { getDeps } from "./get-deps";

const ROOT = dirname(import.meta.dir);
const argv = process.argv.slice(2);
const cpuIdx = argv.indexOf("-cpu");
const NCPU = cpuIdx >= 0 ? parseInt(argv[cpuIdx + 1]) : Math.max(1, cpus().length - 1);
const randIdx = argv.indexOf("-rand");
const allFns = argv.includes("-all-fns");
if (argv.includes("-h") || argv.includes("--help")) {
  console.log("usage: bun cmd/coverage.ts [-rand N] [-dir DIR] [-cpu N] [-all-fns] [-reuse]");
  process.exit(0);
}

const PROF_DIR = join(ROOT, "out", "clang_cov", "profraw");
const MERGED = join(ROOT, "out", "clang_cov", "corpus.profdata");

await getDeps();
const EXE = await buildCoverage();

const dirs: string[] = [];
for (let i = 0; i < argv.length; i++) if (argv[i] === "-dir" && argv[i + 1]) dirs.push(argv[++i]);
let files = dirs.length === 0 ? corpusFiles() : dirs.flatMap((d) => {
  if (!existsSync(d)) {
    console.error(`coverage: no such directory: ${d}`);
    process.exit(2);
  }
  return readdirSync(d).map((n) => join(d, n)).filter((f) => statSync(f).isFile());
});
if (dirs.length) console.log(`(${files.length} file(s) from ${dirs.join(", ")})`);
if (randIdx >= 0) {
  const n = parseInt(argv[randIdx + 1] ?? "");
  if (!(n > 0)) {
    console.error("usage: bun cmd/coverage.ts [-rand N] [-dir DIR] [-cpu N] [-all-fns] [-reuse]");
    process.exit(2);
  }
  // Deterministic stride rather than a shuffle: a sample that changes between
  // runs would make the uncovered list move for no reason.
  const total = files.length;
  const step = Math.max(1, Math.floor(total / n));
  files = files.filter((_, i) => i % step === 0).slice(0, n);
  console.log(`(${files.length} of ${total} corpus files)`);
}

// Decoding the corpus under -O0 instrumentation is the slow part by far, so
// allow the report to be re-run against the profile already on disk.
const reuse = argv.includes("-reuse");
if (reuse && !existsSync(MERGED)) {
  console.error(`coverage: -reuse but no profile at ${MERGED}`);
  process.exit(1);
}

if (!reuse) {
rmSync(PROF_DIR, { recursive: true, force: true });
mkdirSync(PROF_DIR, { recursive: true });

// %m pools the profiles by binary signature instead of writing one file per
// process, which keeps 1000+ decodes down to a handful of files.
const profPattern = join(PROF_DIR, "cov_%m.profraw");

let nextIndex = 0;
let done = 0;
let failed = 0;

async function worker(slot: number) {
  const tmp = join(tmpdir(), "jxldec-cov", String(slot));
  mkdirSync(tmp, { recursive: true });
  const outPam = join(tmp, "out.pam");
  for (;;) {
    const i = nextIndex++;
    if (i >= files.length) return;
    const r = Bun.spawnSync({
      cmd: [EXE, "-out", outPam, files[i]],
      stdout: "ignore",
      stderr: "ignore",
      env: { ...process.env, LLVM_PROFILE_FILE: profPattern },
    });
    // A decoder error is fine here: the point is which code ran, and the
    // corpus deliberately contains frames we refuse (spot colour, mixed
    // extra-channel upsampling). Only note them so the count is not a
    // surprise.
    if (r.exitCode !== 0) failed++;
    done++;
    if (done % 100 === 0) console.log(`  ${done}/${files.length} decoded`);
  }
}

const t0 = performance.now();
console.log(`coverage: decoding ${files.length} files with ${NCPU} worker(s)...`);
await Promise.all(Array.from({ length: NCPU }, (_, i) => worker(i)));
console.log(`coverage: ${done} decoded (${failed} returned an error) in ` +
            `${((performance.now() - t0) / 1000).toFixed(1)}s`);

}

const raws = reuse ? [] : readdirSync(PROF_DIR).filter((f) => f.endsWith(".profraw"))
  .map((f) => join(PROF_DIR, f));
if (!reuse && raws.length === 0) {
  console.error("coverage: no .profraw written -- is the build instrumented?");
  process.exit(1);
}

const run = (cmd: string[]) => {
  const r = Bun.spawnSync({ cmd, stdout: "pipe", stderr: "pipe" });
  if (r.exitCode !== 0) {
    console.error(`coverage: ${cmd[0]} failed: ${r.stderr.toString().trim()}`);
    process.exit(1);
  }
  return r.stdout.toString();
};

if (!reuse) run(["llvm-profdata", "merge", "-sparse", ...raws, "-o", MERGED]);

const srcs = readdirSync(join(ROOT, "src"))
  .filter((f) => f.endsWith(".c")).map((f) => join(ROOT, "src", f));

console.log("\n=== line coverage by file ===");
console.log(run(["llvm-cov", "report", EXE, `-instr-profile=${MERGED}`, ...srcs]).trimEnd());

// The actionable half: functions with zero executions.
const json = run(["llvm-cov", "export", EXE, `-instr-profile=${MERGED}`, ...srcs]);
type Fn = { name: string; count: number; filenames: string[] };
const data = JSON.parse(json) as { data: { functions: Fn[] }[] };
const fns = data.data.flatMap((d) => d.functions ?? []);

// llvm-cov names a static function "file.c:fn". A static inline in a header
// gets one entry per translation unit that includes it, so collapse those to
// one line -- eighteen copies of jxl_unpack_signed is noise, not a finding.
const publicHeader = Bun.file(join(ROOT, "src", "jxl.h"));
const publicSrc = await publicHeader.text();
const isPublic = (bare: string) =>
  new RegExp(`\\b${bare.replace(/[^\w]/g, "")}\\s*\\(`).test(publicSrc);

const seen = new Map<string, { bare: string; file: string; tus: number }>();
for (const f of fns) {
  if (f.count !== 0) continue;
  // Relative to ROOT, not "everything after the first /src/": this repo
  // lives under a path that itself contains /src/, which turned the test
  // harness into "src/jxldec/test/jxl_test.c" and filed it under decoder code.
  const abs = (f.filenames[0] ?? "").replaceAll("\\", "/");
  const file = abs.startsWith(ROOT.replaceAll("\\", "/") + "/")
    ? abs.slice(ROOT.replaceAll("\\", "/").length + 1) : abs;
  if (!file.startsWith("src/")) continue;
  const bare = f.name.includes(":") ? f.name.slice(f.name.indexOf(":") + 1) : f.name;
  const key = `${file}|${bare}`;
  const hit = seen.get(key);
  if (hit) hit.tus++;
  else seen.set(key, { bare, file, tus: 1 });
}
const dead = [...seen.values()]
  .sort((a, b) => a.file.localeCompare(b.file) || a.bare.localeCompare(b.bare));
const api = dead.filter((f) => isPublic(f.bare));
const internal = dead.filter((f) => !isPublic(f.bare));

const printGroup = (title: string, list: typeof dead, note: string) => {
  console.log(`\n=== ${title}: ${list.length} ===`);
  if (note) console.log(`  ${note}`);
  if (list.length === 0) { console.log("  (none)"); return; }
  const show = allFns ? list : list.slice(0, 40);
  let lastFile = "";
  for (const f of show) {
    if (f.file !== lastFile) { console.log(`  ${f.file}`); lastFile = f.file; }
    console.log(`      ${f.bare}${f.tus > 1 ? `  (inline, unused in ${f.tus} TUs)` : ""}`);
  }
  if (!allFns && list.length > show.length) {
    console.log(`  ... and ${list.length - show.length} more (-all-fns for all)`);
  }
};

printGroup(`decoder code ${dirs.length ? "these inputs" : "the corpus"} never reach${dirs.length ? "" : "es"}`, internal,
           "each of this run's bugs lived in a path exactly like these");
printGroup("public API the harness never calls", api,
           "expected for entry points jxl_test.c has no use for");

console.log(`\nprofile: ${MERGED}`);
console.log(`drill in with:  llvm-cov show ${EXE} -instr-profile=${MERGED} src/<file>.c`);
