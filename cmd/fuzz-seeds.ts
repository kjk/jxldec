// fuzz-seeds.ts -- the seed corpus both fuzzers start from (cmd/fuzz.ts,
// libFuzzer; cmd/fuzz-afl.ts, AFL++). They share fuzz/corpus/, so they share
// how it is filled.
import { copyFileSync, existsSync, mkdirSync, openSync, readdirSync, readSync,
         closeSync, rmSync, statSync, writeFileSync } from "fs";
import { basename, join } from "path";
import { buildFuzzSeeds } from "./build";
import { corpusFiles } from "./corpus";
import { JXL_OXIDE_DIR, LIBJXL_DIR } from "./get-deps";

// Other people's crash reproducers, which are worth far more per byte than
// anything we generate: each is a minimised input that already broke a JPEG XL
// decoder. Both live in deps/, which get-deps already fetches, so they are
// referenced rather than copied into this repo.
//   libjxl    -- OSS-Fuzz testcases against djxl_fuzzer
//   jxl-oxide -- regression findings, each named for the bug it caught
//                (dequant_matrix_band, hf_varblock_across_group, icc_parse_oob)
export const EXTERNAL_SEEDS = [
  join(LIBJXL_DIR, "testdata", "oss-fuzz"),
  join(JXL_OXIDE_DIR, "crates", "jxl-oxide-tests", "tests", "fuzz_findings"),
];

/** The regular files directly inside each existing EXTERNAL_SEEDS dir. */
export function externalSeedFiles(): { dir: string; files: string[] }[] {
  return EXTERNAL_SEEDS.filter(existsSync).map((dir) => ({
    dir,
    files: readdirSync(dir).map((n) => join(dir, n))
      .filter((f) => statSync(f).isFile()),
  }));
}

function readPrefix(f: string, n: number): Buffer {
  const fd = openSync(f, "r");
  try {
    const buf = Buffer.alloc(n);
    const got = readSync(fd, buf, 0, n, 0);
    return buf.subarray(0, got);
  } finally {
    closeSync(fd);
  }
}

/**
 * Fill an empty corpus dir from the test corpus and the external reproducers,
 * so mutation starts from valid codestreams rather than random bytes. Does
 * nothing if the dir already has files (it is the fuzzers' checkpoint).
 *
 * A file larger than maxLen goes in as its first maxLen bytes ("trunc_"):
 * the headers, the first frame's TOC and its first sections, which is where
 * the parsers are, at a size that runs hundreds of times a second instead of
 * once. libFuzzer would cut it to -max_len on load anyway; doing it here
 * means AFL++, which refuses oversized seeds, gets the same set. The
 * generated corpus has unique names by construction, but the shipped .jxl
 * files and the reproducers can collide with it, so a clash gets a numeric
 * prefix.
 */
export function seedCorpusIfEmpty(corpusDir: string, maxLen: number): void {
  mkdirSync(corpusDir, { recursive: true });
  if (readdirSync(corpusDir).length > 0) return;
  let n = 0, ext = 0, truncated = 0;
  const take = (f: string, prefix: string) => {
    const big = statSync(f).size > maxLen;
    if (big) prefix = "trunc_" + prefix;
    let dest = join(corpusDir, prefix + basename(f));
    if (existsSync(dest)) dest = join(corpusDir, `${prefix}${n + ext}_${basename(f)}`);
    if (big) {
      writeFileSync(dest, readPrefix(f, maxLen));
      truncated++;
    } else {
      copyFileSync(f, dest);
    }
  };
  for (const f of corpusFiles()) { take(f, ""); n++; }
  for (const { files } of externalSeedFiles()) {
    for (const f of files) { take(f, "ext_"); ext++; }
  }
  console.log(`seeded corpus with ${n} file(s) + ${ext} external reproducer(s)` +
              (truncated ? `; ${truncated} cut to their first ${maxLen} bytes` : ""));
}

/** The component targets whose seeds test/fuzz_seeds.c cuts out of files. */
export type ComponentTarget = "icc" | "entropy" | "modular";

/**
 * Fill an empty component-target corpus with real inputs for that component:
 * the corpus decoded by the seed extractor (cmd/build.ts buildFuzzSeeds),
 * which writes every ICC stream, entropy-coded stream and single-group
 * GlobalModular section it meets, each from the bit it starts at. One
 * extraction serves all three targets; it is kept under fuzz/seeds/.
 */
export async function seedComponentCorpusIfEmpty(
  target: ComponentTarget, corpusDir: string, seedsDir: string,
): Promise<void> {
  mkdirSync(corpusDir, { recursive: true });
  if (readdirSync(corpusDir).length > 0) return;
  const kinds = ["icc", "entropy", "modular"];
  if (!kinds.every((k) => existsSync(join(seedsDir, k)))) {
    const exe = await buildFuzzSeeds();
    rmSync(seedsDir, { recursive: true, force: true });
    for (const k of kinds) mkdirSync(join(seedsDir, k), { recursive: true });
    const files = corpusFiles();
    console.log(`extracting component seeds from ${files.length} corpus file(s)...`);
    // In batches: a Windows command line is limited to 32K characters.
    for (let i = 0; i < files.length; i += 64) {
      const p = Bun.spawnSync({ cmd: [exe, seedsDir, ...files.slice(i, i + 64)],
                                stdout: "ignore", stderr: "ignore" });
      if (p.exitCode !== 0) console.log(`  seed extractor exited ${p.exitCode} on a batch`);
    }
  }
  const src = join(seedsDir, target);
  let n = 0;
  for (const f of readdirSync(src)) {
    copyFileSync(join(src, f), join(corpusDir, f));
    n++;
  }
  console.log(`seeded ${target} corpus with ${n} input(s) from ${src}`);
}
