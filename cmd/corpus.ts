// corpus.ts -- the test/bench corpus.
//
// libjxl's testdata/ carries only a handful of .jxl files, so the corpus is
// *generated*: every source image in testdata is encoded with cjxl across a
// matrix of settings that covers the format's feature space (modular vs
// VarDCT, lossless vs lossy, effort levels, squeeze, progressive, alpha).
// Results are cached in deps/corpus/ and regenerated only when missing.
//
// JXL_SPECS=<dir> overrides the corpus with any directory of .jxl files
// (scanned recursively), the way DJVU_SPECS does in djvudec.
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from "fs";
import { basename, dirname, isAbsolute, join, relative } from "path";
import { DEPS_DIR, LIBJXL_DIR, refTool } from "./get-deps";
import { pngToPam } from "./png";

const CORPUS_DIR = join(DEPS_DIR, "corpus");
const SRC_DIR = join(CORPUS_DIR, "src");
const GEN_DIR = join(CORPUS_DIR, "gen");

// ProPhoto is wide enough that cjxl keeps the profile verbatim instead of
// matching it to an enum.
const ICC_PROFILE = join(
  LIBJXL_DIR, "testdata", "external", "Compact-ICC-Profiles", "profiles",
  "ProPhoto-v4.icc",
);

/** cjxl settings, one per generated variant. */
export const PRESETS: { name: string; args: string[] }[] = [
  { name: "m_e1", args: ["-d", "0", "-e", "1"] },
  { name: "m_e3", args: ["-d", "0", "-e", "3"] },
  { name: "m_e7", args: ["-d", "0", "-e", "7"] },
  { name: "m_e9", args: ["-d", "0", "-e", "9"] },
  { name: "m_resp", args: ["-d", "0", "-e", "7", "--responsive=1"] },
  { name: "v_e3", args: ["-d", "1", "-e", "3"] },
  { name: "v_d1", args: ["-d", "1", "-e", "7"] },
  { name: "v_d05", args: ["-d", "0.5", "-e", "7"] },
  { name: "v_prog", args: ["-d", "1", "-e", "7", "--progressive"] },
  { name: "lm_d1", args: ["-d", "1", "-e", "7", "-m", "1"] },
  // An ICC profile cjxl cannot reduce to the enumerated colour fields, so the
  // codestream carries the profile itself and the decoder takes the
  // want_icc path.
  { name: "v_icc", args: ["-d", "1", "-e", "7", "-x", `icc_pathname=${ICC_PROFILE}`] },
  // Sets JXL_FF_NOISE in the frame header, which nothing else here does:
  // testdata ships no noisy image and no other preset asks for one, so the
  // noise synthesis path had no regression coverage at all.
  { name: "v_noise", args: ["-d", "1", "-e", "7", "--photon_noise_iso=3200"] },
  // Coded at 1/2 and 1/4 resolution and upsampled by the decoder. cjxl
  // downsamples the input itself, so the output stays the source's size --
  // no blow-up in the corpus. Both are needed: the 2x weight table has a
  // single quadrant (H=1), so only 4x and up exercise the quadrant mirroring
  // in build_kernel.
  { name: "v_rs2", args: ["-d", "1", "-e", "7", "--resampling=2"] },
  { name: "v_rs4", args: ["-d", "1", "-e", "7", "--resampling=4"] },
];

function walk(dir: string, pred: (name: string) => boolean): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p, pred));
    else if (pred(name)) out.push(p);
  }
  return out;
}

/** Source images shipped in libjxl's testdata, converted to PAM once. */
export function sourceImages(): string[] {
  mkdirSync(SRC_DIR, { recursive: true });
  const pngs = walk(join(LIBJXL_DIR, "testdata"), (n) =>
    n.toLowerCase().endsWith(".png"),
  );
  // testdata carries the same basename in more than one directory:
  // external/wesaturate/500px and .../64px hold *different* images under
  // three shared names. Keying the converted .pam by basename alone made
  // both collapse onto one file, so the 64px image was never tested and the
  // 500px one was tested (and counted) twice. Qualify a colliding name with
  // its parent directory; unique names keep the short form, so the cached
  // corpus does not have to be regenerated wholesale.
  const seen = new Map<string, number>();
  for (const png of pngs) {
    const b = basename(png).toLowerCase();
    seen.set(b, (seen.get(b) ?? 0) + 1);
  }
  const taken = new Set<string>();
  const out: string[] = [];
  for (const png of pngs) {
    let stem = basename(png).replace(/\.png$/i, "");
    if ((seen.get(basename(png).toLowerCase()) ?? 0) > 1) {
      stem = `${basename(dirname(png))}_${stem}`;
    }
    let name = `${stem}.pam`;
    for (let i = 2; taken.has(name.toLowerCase()); i++) name = `${stem}_${i}.pam`;
    taken.add(name.toLowerCase());
    const pam = join(SRC_DIR, name);
    if (!existsSync(pam)) {
      try {
        pngToPam(png, pam);
      } catch (e) {
        console.warn(`corpus: skipping ${relative(LIBJXL_DIR, png)}: ${e}`);
        continue;
      }
    }
    out.push(pam);
  }
  return out.sort();
}

/** Returns null when the file was written, else cjxl's complaint. */
function runCjxl(src: string, dst: string, args: string[]): string | null {
  const r = Bun.spawnSync({
    cmd: [refTool("cjxl"), src, dst, ...args],
    stdout: "ignore",
    stderr: "pipe",
  });
  if (r.exitCode === 0 && existsSync(dst)) return null;
  const lines = r.stderr.toString().trim().split("\n")
    .map((l) => l.trim()).filter((l) => l.length > 0);
  // cjxl's last line is the summary failure ("EncodeImageJXL() failed."); the
  // line before it is the one that says why.
  return lines.length > 1 ? lines[lines.length - 2] : (lines[0] ?? `exit ${r.exitCode}`);
}

/** Generates (once) and returns every corpus .jxl file. */
export function corpusFiles(presets = PRESETS.map((p) => p.name)): string[] {
  if (process.env.JXL_SPECS) {
    return walk(process.env.JXL_SPECS, (n) => n.toLowerCase().endsWith(".jxl")).sort();
  }
  mkdirSync(GEN_DIR, { recursive: true });
  const srcs = sourceImages();
  const out: string[] = [];
  // A source cjxl cannot encode used to `continue` in silence, so the corpus
  // quietly shrank and nothing said which coverage had gone. Collect the
  // reasons and print them: a file that stops being testable is news.
  //
  // The seven this currently reports for v_icc are the grayscale sources --
  // JxlEncoderSetICCProfile() rejects an RGB profile for them. Not worth
  // working around: the only gray profile in Compact-ICC-Profiles is sGrey,
  // and cjxl reduces that to the enumerated Grayscale/D65/sRGB fields, so it
  // never takes the want_icc path the preset exists to exercise.
  const skipped: string[] = [];
  let generated = 0;
  for (const src of srcs) {
    const base = basename(src).replace(/\.pam$/, "");
    for (const preset of PRESETS) {
      if (!presets.includes(preset.name)) continue;
      const dst = join(GEN_DIR, `${base}.${preset.name}.jxl`);
      if (!existsSync(dst)) {
        const err = runCjxl(src, dst, preset.args);
        if (err) {
          skipped.push(`${base}.${preset.name}: ${err}`);
          continue;
        }
        generated++;
        if (generated % 25 === 0) console.log(`corpus: generated ${generated} files...`);
      }
      out.push(dst);
    }
  }
  // Transcoded JPEGs are the only source of YCbCr frames, and testdata's set
  // covers every chroma subsampling mode.
  //
  // Two of testdata's 21 never make it: flower_small.cmyk.jpg, because this
  // cjxl build cannot read a CMYK JPEG at all ("Getting pixel data failed",
  // with or without --allow_jpeg_reconstruction 0 / --lossless_jpeg=0), and
  // flower.png.im_q85_rgb_subsample_blue.jpg, which its encoder rejects. Both
  // are libjxl-side limits, not ours -- they are reported, not worked around.
  for (const jpg of walk(join(LIBJXL_DIR, "testdata"), (n) =>
    /\.jpe?g$/i.test(n),
  ).sort()) {
    const dst = join(GEN_DIR, `${basename(jpg).replace(/\.jpe?g$/i, "")}.jpeg.jxl`);
    if (!existsSync(dst)) {
      const err = runCjxl(jpg, dst, []);
      if (err) {
        skipped.push(`${relative(LIBJXL_DIR, jpg).replaceAll("\\", "/")}: ${err}`);
        continue;
      }
      generated++;
    }
    out.push(dst);
  }

  // The .jxl files that ship with libjxl exercise things cjxl won't emit
  // (splines, animation, odd containers).
  out.push(...walk(join(LIBJXL_DIR, "testdata", "jxl"), (n) =>
    n.toLowerCase().endsWith(".jxl"),
  ));
  if (generated) console.log(`corpus: generated ${generated} file(s)`);
  if (skipped.length) {
    console.warn(`corpus: ${skipped.length} source(s) cjxl could not encode:`);
    for (const s of skipped) console.warn(`  ${s}`);
  }
  return out.sort();
}

export function pickRandom(files: string[], n: number): string[] {
  const shuffled = [...files];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, Math.max(0, Math.min(n, shuffled.length)));
}

// Shared file selection: explicit paths, -rand N, or -all. With none of
// those, print usageText and exit 2 (the djvudec convention).
export function selectFiles(usageText: string, valueFlags = ["-rand", "-cpu", "-preset"]): string[] {
  const argv = process.argv.slice(2);
  const explicit = argv.filter(
    (a, i) => !a.startsWith("-") && !valueFlags.includes(argv[i - 1] ?? ""),
  );
  const pi = argv.indexOf("-preset");
  const presets = pi >= 0 ? argv[pi + 1].split(",") : undefined;

  if (argv.includes("-all")) return corpusFiles(presets);
  const ri = argv.indexOf("-rand");
  if (ri >= 0) {
    const n = parseInt(argv[ri + 1] ?? "");
    if (!(n > 0)) {
      console.log(usageText);
      process.exit(2);
    }
    const all = corpusFiles(presets);
    const picked = pickRandom(all, n);
    console.log(`(${picked.length} random of ${all.length} corpus files)`);
    return picked;
  }
  if (explicit.length > 0) {
    for (const f of explicit) {
      if (!existsSync(f)) {
        console.error(`no such file: ${f}`);
        process.exit(1);
      }
    }
    return explicit;
  }
  console.log(usageText);
  process.exit(2);
}

export function fmtBytesHuman(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

export function fileLabel(f: string, root: string): string {
  let rel = relative(root, f);
  if (rel.startsWith("..") || isAbsolute(rel)) rel = f;
  rel = rel.replaceAll("\\", "/");
  return `${rel} (${fmtBytesHuman(statSync(f).size)})`;
}

export function corpusSummary(): string {
  const src = process.env.JXL_SPECS
    ? `JXL_SPECS=${process.env.JXL_SPECS}`
    : `deps/corpus (generated from libjxl testdata; ${PRESETS.length} presets)`;
  const dir = process.env.JXL_SPECS ?? GEN_DIR;
  const n = existsSync(dir)
    ? walk(dir, (x) => x.toLowerCase().endsWith(".jxl")).length
    : 0;
  return `${n} .jxl file(s) available from ${src}`;
}

if (import.meta.main) {
  const files = corpusFiles();
  console.log(`${files.length} corpus files in ${GEN_DIR}`);
  writeFileSync(join(CORPUS_DIR, "files.txt"), files.join("\n") + "\n");
}
