// fuzz-seeds.ts -- the seed corpus both fuzzers start from (cmd/fuzz.ts,
// libFuzzer; cmd/fuzz-afl.ts, AFL++). They share fuzz/corpus/, so they share
// how it is filled.
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "fs";
import { basename, join } from "path";
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

/**
 * Fill an empty corpus dir from the test corpus and the external reproducers,
 * so mutation starts from valid codestreams rather than random bytes. Does
 * nothing if the dir already has files (it is the fuzzers' checkpoint).
 * Files larger than maxLen are left out. The generated corpus has unique names
 * by construction, but the shipped .jxl files and the reproducers can collide
 * with it, so a clash gets a numeric prefix.
 */
export function seedCorpusIfEmpty(corpusDir: string, maxLen: number): void {
  mkdirSync(corpusDir, { recursive: true });
  if (readdirSync(corpusDir).length > 0) return;
  let n = 0, ext = 0, skipped = 0;
  const take = (f: string, prefix: string): boolean => {
    if (statSync(f).size > maxLen) {
      skipped++;
      return false;
    }
    let dest = join(corpusDir, prefix + basename(f));
    if (existsSync(dest)) dest = join(corpusDir, `${prefix}${n + ext}_${basename(f)}`);
    copyFileSync(f, dest);
    return true;
  };
  for (const f of corpusFiles()) if (take(f, "")) n++;
  for (const { files } of externalSeedFiles()) {
    for (const f of files) if (take(f, "ext_")) ext++;
  }
  console.log(`seeded corpus with ${n} file(s) + ${ext} external reproducer(s)` +
              (skipped ? `; ${skipped} skipped as larger than -max-len` : ""));
}
