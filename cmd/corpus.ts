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
import { existsSync, lstatSync, mkdirSync, openSync, readSync, closeSync,
         readdirSync, statSync, writeFileSync } from "fs";
import { basename, dirname, join, relative, resolve } from "path";
import { CONFORMANCE_DIR, DEPS_DIR, LIBJXL_DIR, refTool } from "./get-deps";
import { pngToPam } from "./png";

const CORPUS_DIR = join(DEPS_DIR, "corpus");
const SRC_DIR = join(CORPUS_DIR, "src");
const GEN_DIR = join(CORPUS_DIR, "gen");
const EXIF_DIR = join(CORPUS_DIR, "exif");

// A minimal little-endian TIFF carrying nothing but tag 274, Orientation.
// Nothing in testdata has a non-default orientation and cjxl has no flag for
// one, so the only way to cover the writer's orientation fixup is to hand it
// the Exif blob ourselves (`cjxl -x exif=...`). libjxl reads it with
// IsExif + FindExifTagPosition (lib/jxl/base/exif.h), which needs the raw
// TIFF -- not the 4-byte-prefixed form a JXL Exif *box* carries.
function exifOrientationBlob(orientation: number): string {
  mkdirSync(EXIF_DIR, { recursive: true });
  const path = join(EXIF_DIR, `orientation${orientation}.bin`);
  if (!existsSync(path)) {
    const b = Buffer.alloc(26);
    b.write("II", 0, "latin1");
    b.writeUInt16LE(42, 2);            // TIFF magic
    b.writeUInt32LE(8, 4);             // offset of IFD0
    b.writeUInt16LE(1, 8);             // one tag in IFD0
    b.writeUInt16LE(274, 10);          // Orientation
    b.writeUInt16LE(3, 12);            // type SHORT
    b.writeUInt32LE(1, 14);            // count
    b.writeUInt16LE(orientation, 18);  // value, in the 4-byte value field
    b.writeUInt32LE(0, 22);            // no next IFD
    writeFileSync(path, b);
  }
  return path;
}

// Spreads the eight orientations across the sources instead of spending eight
// presets on them: unapply_orientation is a switch with a case per value, and
// one preset this way exercises every arm. Keyed on the name so a given file
// always gets the same one however the source set changes.
function orientationFor(src: string): number {
  const name = basename(src);
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return (h % 8) + 1;
}

const iccProfile = (name: string) => join(
  LIBJXL_DIR, "testdata", "external", "Compact-ICC-Profiles", "profiles",
  `${name}.icc`,
);
// ProPhoto is wide enough that cjxl keeps the profile verbatim instead of
// matching it to an enum. That is what makes it useful for the want_icc path
// -- and also what makes it useless for the colour-conversion code, since on
// that path the decoder falls back to linear sRGB by design.
const ICC_PROFILE = iccProfile("ProPhoto-v4");

/** cjxl settings, one per generated variant. `peakTol` / `rmsTol` replace
 *  the test driver's default limits for that preset's files; `alphaOnly`
 *  restricts a preset to the sources that have an alpha channel. */
export const PRESETS:
    { name: string; args: string[] | ((src: string) => string[]);
      peakTol?: number; rmsTol?: number; alphaOnly?: boolean }[] = [
  { name: "m_e1", args: ["-d", "0", "-e", "1"] },
  { name: "m_e3", args: ["-d", "0", "-e", "3"] },
  { name: "m_e7", args: ["-d", "0", "-e", "7"] },
  { name: "m_e9", args: ["-d", "0", "-e", "9"] },
  { name: "m_resp", args: ["-d", "0", "-e", "7", "--responsive=1"] },
  // Two Modular paths no other preset reaches, each of which hid a wrong-
  // pixels bug until it was asked for by name: an RCT that only permutes the
  // channels (-C 7 is type 0, permutation 1), and the AvgAll predictor
  // (-P 13), which the weighted-predictor fast path has to step around.
  { name: "m_rct7", args: ["-d", "0", "-e", "7", "-C", "7"] },
  { name: "m_p13", args: ["-d", "0", "-e", "5", "-P", "13"] },
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
  // Non-sRGB primaries, which nothing covered until now: everything else here
  // comes from PAM, which carries no colour information, so cjxl tags it sRGB
  // -- names like P3-sRGB-color-bars describe their *content*, not their
  // declared encoding. Unlike ProPhoto these two are narrow enough for cjxl to
  // match against the enumerated primaries rather than embedding the profile,
  // which is exactly what puts the decoder through jxl_opsin_matrix_for's
  // non-sRGB branch (get_primaries, primaries_to_xyz, adapt_to_xyz_d50).
  // Rec2020-v4 additionally declares transfer function 709, covering tf_bt709.
  { name: "v_p3", args: ["-d", "1", "-e", "7", "-x",
                         `icc_pathname=${iccProfile("DisplayP3-v4")}`] },
  { name: "v_2020", args: ["-d", "1", "-e", "7", "-x",
                           `icc_pathname=${iccProfile("Rec2020-v4")}`] },
  // The two HDR transfer functions. PAM carries no colour information, so
  // cjxl has to be told; it then declares PQ / HLG on an XYB frame, which is
  // what takes the decoder through tf_pq and the HLG inverse OOTF.
  //
  // Both get wider limits, PQ much wider, and it is worth being exact about
  // why. A channel that should be zero comes out of XYB -> RGB as zero plus
  // float noise: the opsin inverse has entries around +-10, so two correct
  // implementations disagree by ~1e-5 there, on either side of zero. PQ maps
  // 1e-5 of full scale to ~18 8-bit steps (its slope at black is that steep)
  // and a negative value to 0. On a photo that is a handful of samples per
  // image (peak ~9, RMS 0.49 like every other lossy preset); on testdata's
  // synthetic colour bars, where a third of the image is a saturated primary
  // with two channels at zero, it is whole regions (peak ~23, RMS up to
  // 1.7). So these limits only catch a PQ that is grossly wrong -- which is
  // still more than nothing, and is what the preset is for. HLG's square
  // root at black does the same thing far more mildly.
  { name: "v_pq", peakTol: 32, rmsTol: 2,
    args: ["-d", "1", "-e", "7", "-x", "color_space=RGB_D65_202_Rel_PeQ"] },
  { name: "v_hlg", peakTol: 16,
    args: ["-d", "1", "-e", "7", "-x", "color_space=RGB_D65_202_Rel_HLG"] },
  // An extra channel stored at a different resolution from the colour
  // channels, which libjxl upsamples on its own, before patches, rather than
  // together with them. One with full-size colour, one where both are
  // downsampled by different factors. Only meaningful with an alpha channel,
  // and on a source without one these would just repeat other presets.
  { name: "m_ec2", alphaOnly: true,
    args: ["-d", "0", "-e", "4", "--ec_resampling=2"] },
  { name: "v_rs2_ec8", alphaOnly: true,
    args: ["-d", "1", "-e", "4", "--resampling=2", "--ec_resampling=8"] },
  // EXIF orientation, which the writer undoes. The eight values are spread
  // across the sources by orientationFor, so this one preset reaches every
  // arm of unapply_orientation's switch, including the four that transpose.
  {
    name: "v_orient",
    args: (src) => [
      "-d", "1", "-e", "7",
      "-x", `exif=${exifOrientationBlob(orientationFor(src))}`,
    ],
  },
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

/** The limits a generated file's preset asks for, where it asks for any. */
export function tolFor(file: string): { peak?: number; rms?: number } {
  const base = basename(file);
  for (const p of PRESETS) {
    if (base.endsWith(`.${p.name}.jxl`)) return { peak: p.peakTol, rms: p.rmsTol };
  }
  return {};
}

// The JPEG XL conformance suite (github.com/libjxl/conformance): two dozen
// hand-made files covering what cjxl will not emit -- layers and every blend
// mode, cropped animation frames, spot colours, CMYK, float samples, delta
// palettes, loop filters on a Modular frame, extra channels with their own
// bit depth. The `*_5` testcases are symlinks to their Level 10 twins (plain
// text stubs in a Windows checkout), so they are left out.
function conformanceFiles(): string[] {
  const root = join(CONFORMANCE_DIR, "testcases");
  return walk(root, (n) => n === "input.jxl").filter((f) => {
    const st = lstatSync(f);
    return !st.isSymbolicLink() && st.size > 64;
  });
}

// Files whose first frame `djxl x.jxl x.pam` writes in a shape we cannot
// compare: it appends every extra channel as a further PAM plane, so
// cmyk_layers comes out as five (RGB, K, alpha), spot as six and the gray
// spot fixtures as two, where we render colour (+ alpha) with the ink mixed
// in. `tests.ts -all-frames` checks them against libjxl directly instead.
const DJXL_UNCOMPARABLE = [
  "/testcases/cmyk_layers/", "/testcases/spot/", "/test/data/gray_spot_",
];
export function skipWithDjxl(file: string): boolean {
  // Absolute, so a relative path given on the command line matches too.
  const f = resolve(file).replaceAll("\\", "/");
  return DJXL_UNCOMPARABLE.some((t) => f.includes(t));
}

// Fixtures checked into the repo: files no tool here can produce, made once
// with libjxl's encoder API (see test/gen_gray_spot.c).
const FIXTURE_DIR = join(import.meta.dir, "..", "test", "data");

/** Whether a PAM source has an alpha channel, from its TUPLTYPE line. */
function pamHasAlpha(pam: string): boolean {
  const fd = openSync(pam, "r");
  try {
    const buf = Buffer.alloc(128);
    const n = readSync(fd, buf, 0, 128, 0);
    return buf.subarray(0, n).toString("latin1").includes("_ALPHA");
  } finally {
    closeSync(fd);
  }
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
      if (preset.alphaOnly && !pamHasAlpha(src)) continue;
      const dst = join(GEN_DIR, `${base}.${preset.name}.jxl`);
      if (!existsSync(dst)) {
        const args = typeof preset.args === "function"
          ? preset.args(src) : preset.args;
        const err = runCjxl(src, dst, args);
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
  out.push(...conformanceFiles());
  out.push(...walk(FIXTURE_DIR, (n) => n.toLowerCase().endsWith(".jxl")));
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

export function corpusSummary(): string {
  const src = process.env.JXL_SPECS
    ? `JXL_SPECS=${process.env.JXL_SPECS}`
    : `deps/corpus (generated from libjxl testdata; ${PRESETS.length} presets)`;
  // The set corpusFiles() returns, counted without generating anything: the
  // cached gen/ files plus the .jxl files that ship in libjxl's testdata.
  const isJxl = (x: string) => x.toLowerCase().endsWith(".jxl");
  const n = process.env.JXL_SPECS
    ? walk(process.env.JXL_SPECS, isJxl).length
    : walk(GEN_DIR, isJxl).length +
      walk(join(LIBJXL_DIR, "testdata", "jxl"), isJxl).length +
      conformanceFiles().length + walk(FIXTURE_DIR, isJxl).length;
  return `${n} .jxl file(s) available from ${src}`;
}

if (import.meta.main) {
  const files = corpusFiles();
  console.log(`${files.length} corpus files (generated ones in ${GEN_DIR})`);
}
