# PROGRESS

Milestone log for the C JPEG XL decoder. Newest first.

## Status summary

`bun cmd/tests.ts -all` — 1245 corpus files (libjxl's testdata images × 17 cjxl
presets, its JPEGs transcoded, plus the `.jxl` files that ship with libjxl):

```
1245/1245 ok, 0 failed
```

"ok" means byte-identical to `djxl file.jxl out.pam`, or inside the float
tolerance for the VarDCT paths. 387 files are byte-exact; across the other 858
no sample differs from libjxl by more than one 8-bit step.

| area | state |
|---|---|
| container, headers, ICC | done, verified against `jxlinfo` |
| entropy coding (prefix, ANS, LZ77, context maps) | done |
| Modular lossless (all effort levels, palette, squeeze, RCT) | **byte-exact** |
| Modular lossy (XYB) | within 1 |
| VarDCT, 8-bit sRGB | within 1 |
| VarDCT, progressive (LF frames) | within 1 |
| gaborish, EPF | done |
| non-sRGB primaries (P3, Rec.2100) | within 1 (Bradford-adapted matrix) |
| transfer functions: sRGB, 709 | within 1 |
| transfer functions: PQ, HLG | **unverified** — cjxl here cannot emit a lossy frame declaring them |
| images with an embedded ICC profile | done |
| patches / reference frames | **byte-exact** |
| splines, noise | within 1 |
| 2x/4x/8x upsampling (`--resampling`) | within 1 |
| EXIF orientation, all 8 values | within 1 |
| animation (frames after the first) | done |
| YCbCr / JPEG transcode frames | done, all subsampling modes |

### What "pixel perfect" means here
Modular is integer arithmetic, so it is byte-identical to libjxl and any
mismatch is a bug. VarDCT is float: libjxl, jxl-oxide and this decoder all use
different IDCT factorizations and different `powf` approximations, so exact
equality is not achievable and libjxl's own conformance testing uses a
tolerance. Following it, a VarDCT file passes on RMS **and** peak together
(`-rms` 0.6, `-tol` 3 in 8-bit steps), not peak alone: a single sample should
not decide a megapixel file. Max abs difference ≤ 1 currently holds on every
corpus file anyway.

## Remaining differences

1. **VarDCT float divergence** — about a quarter of the samples in a lossy
   file land one count either side of libjxl's, RMS ~0.5. Inherent to the
   differing IDCT factorizations; not worth chasing further.

   This used to read "63 files, max 2-10 counts, not worth chasing". That was
   wrong: the long tail was a loop-filter edge bug, not float divergence. See
   the entry below.
2. ~~**Noise** reproduces approximately.~~ Fixed: it was the RNG seed being
   advanced one frame too late, not an approximation. Noise now lands inside
   the same one-count tolerance as every other VarDCT path. See the log.

## Performance

`bun cmd/bench.ts` links the `dist/` amalgamation and libjxl's static
libraries into one process and times both, single-threaded. Over the whole
1245-file corpus we are **2.31x libjxl** (2.74x before the SSE2 work below;
2.33x over the smaller 821-file corpus that predates the `v_noise`, `v_rs*`,
`v_orient`, `v_p3` and `v_2020` presets, which are lossy paths and pull the
average up). libjxl is AVX2 throughout; we are scalar C apart from the SSE2
hot loops (noise, upsampling, EPF, gaborish and both DCT passes), so a
constant factor is expected. Each of those keeps a scalar twin and is checked
bit-identical against it -- see the log.

`bun cmd/prof.ts <file.jxl>` profiles our decoder alone through the sibling
`../samply` and prints samply's `-print-agent` report (top self-time
functions, hot source lines, heaviest call path). That is how the numbers
below were found.

The worst ratios among files that take libjxl more than 5ms are now the
`v_rs2` set at 5.8-6.1x, whose upsampling filter is genuinely 25 taps per
output sample. Files below ~1ms sit at up to 9x purely on fixed setup cost.

### The EPF and spline rewrites
Three hot loops were doing the same work three or more times over.

- **`epf_pass`** was 78% of the decode of every VarDCT file with EPF on. Step
  0 visits 12 kernel taps x 3 channels x 5 SAD offsets per sample, and each of
  those 180 iterations called `jxl_mirror` **four** times (720 mirror loops
  per sample), recomputed the centre-pixel footprint 36 times instead of once,
  and divided by sigma once per tap. Now: the tap and footprint offsets are
  tabulated once per pass; samples whose whole footprint is inside the image
  take a mirror-free path using fixed offsets off the centre sample; the
  reciprocal of sigma is hoisted per sample; the SAD loop runs channel-outer
  like libjxl's, so a channel's footprint comes from one plane with its centre
  samples live in registers. The three closing divisions became one reciprocal
  and three multiplies, which is what libjxl does. Every reordering preserves
  the float operation order, so the output is bit-identical.
- **`jxl_render_splines`** was 93% of `splines.jxl`. The channel loop was
  *outermost*, so the `sqrt` and the two `spline_erf` calls ran three times per
  pixel. Moving it innermost, hoisting `0.25 * values[c] * sigma` out, and
  clamping each sample's bounding box once instead of bounds-checking every
  pixel cut that file from 2935ms to 1020ms.
- **`max_distance`** now uses libjxl's formula, `sqrt(-2*sigma^2*(log(0.1)*3 -
  log(max_color)))`. jxl-oxide uses `max_color` where libjxl takes its log,
  which inflates the draw radius (3.7 sigma vs 2.1 sigma at the 0.01 floor) and
  so the pixel count, for output that is identical after 8-bit quantization.
- **`write_pixels`**, the final float-to-integer pass for every format, called
  `unapply_orientation` and `plane_sample` per sample even though both are
  loop-invariant when there is no orientation fixup and no subsampled plane.
  That case now walks precomputed row pointers.
- `jxl_apply_epf`'s three scratch planes were `calloc`ed and immediately
  overwritten; they are `jxl_malloc`ed now and copied back row by row so the
  row padding never travels.

Measured effect, best of 3, ours vs libjxl:

| file | before | after |
|---|---|---|
| `splines.jxl` | 2935ms, 9.5x | 1020ms, 3.4x |
| `splines.v_e3.jxl` | 663ms, 17.3x | 256ms, 6.9x |
| `colorful_chessboards.v_e3.jxl` | 166ms, 15.9x | 66ms, 5.9x |
| `P3-sRGB-color-bars.v_e3.jxl` | 124ms, 16.0x | 50ms, 6.4x |
| `flower.v_e3.jxl` | 572ms, 13.2x | 229ms, 5.4x |
| corpus total | 47.6s, 3.22x | 38.6s, 2.64x |

All 821 files decode to the same bytes as before: the same 63 files differ
from `djxl`, with the same max and mean deviation on each.

### What is left
The profile on `P3-sRGB-color-bars.v_d1` is now flat -- no single function
above 14%, where `dct_1d` alone was 27% before the DCT work:

    dct_1d_v4          14.0%   both DCT passes, 4 lanes
    epf_pass           11.3%
    jxl_linear_to_tf   11.0%   a powf per sample
    memset_repstos      8.4%   calloc, see below
    write_pixels        8.2%
    jxl_xyb_to_linear   6.1%

Nothing here is an outlier any more, so the next round is several ~10% items
rather than one big one. `memset_repstos` is the cheapest: `jxl_fplane_alloc`
uses `calloc` and most callers overwrite the plane completely, the same waste
already removed from the noise and gaborish scratch buffers -- but it needs
checking caller by caller, since a plane that is only partly written would
start leaking heap contents instead of reading zeros. `jxl_linear_to_tf` is a
`powf` per sample and would want a polynomial approximation, which would *not*
be bit-identical and so needs the tolerance argument rather than a diff.

Still not taken on EPF: the kernel taps are symmetric pairs and the SAD is
symmetric under swapping the two footprints, so `dist(x, y, k) ==
dist(x+kx, y+ky, -k)`. Caching six SADs per sample in a rolling three-row
window would halve the SAD work. The `sigma < 0.3` early-out complicates it:
a skipped sample's neighbours still want its cached values.

## Log

### Both DCT passes, four lanes at a time
With the loop filters vectorised, `dct_1d` was 27% self and `jxl_dct_2d` 31%
inclusive -- the single biggest cost by a wide margin.

The column pass came almost free. It already had to gather each column out of
the row-major block, and **four consecutive floats from a row are exactly the
four columns' values at that row**, so the gather *is* the transpose: load
four columns into the lanes, run the transform, store back. `dct_1d_v4` and
`dct4_v4` mirror their scalar twins expression for expression, including how
the additions associate, so every lane performs the same operations in the
same order and the result is bit-identical. Divisions by 2 and 4 become
multiplies by 0.5 and 0.25, exact for powers of two.

The row pass needed a real transpose, since rows are contiguous --
`_MM_TRANSPOSE4_PS` in, the same kernel, transpose out -- but the arithmetic
each lane sees is unchanged, so it stays bit-identical too.

One thing worth recording: `dct_1d` has a hand-written `n == 8` case whose
expressions differ from what the general recursion produces for the same
length (`(a-b)/2*sec` against `(a-b)*sec/2`, among others). `dct_1d_v4` has no
such case and takes the general path at 8, which is the length that matters
most. Those *should* agree -- scaling by a power of two is exact, so the
rounding commutes -- but "should" is what `-DJXL_DCT_FORCE_SCALAR` is for, and
140/140 files came out identical.

    P3-sRGB-color-bars.v_d1   39.7ms  4.81x -> 31.6ms  3.80x
    flower.v_d1              177.6ms  3.82x -> 155.8ms 3.39x
    flower.v_prog            185.8ms  3.27x -> 162.8ms 2.86x
    splines.v_e3             171.7ms  4.53x -> 163.1ms 4.32x

Corpus total **2.43x -> 2.31x**, 40.3s -> 38.1s. Over the three rounds of
this work: 2.74x -> 2.31x, 48.4s -> 38.1s, a fifth of the wall clock gone.
1245/1245 ok, ASan clean, 115 fuzz reproducers clean.

### SSE2 in the two loop filters, proved bit-identical
With the noise stage fixed, a samply profile of `P3-sRGB-color-bars.v_d1`
(`bun cmd/prof.ts`, which now also finds samply under `../exp/samply`) put
`epf_pass` at 6.8% self and **`jxl_apply_gabor` at 16.7%** -- and `dct_1d`
above both. EPF's share was low only because the file is small; disabling EPF
outright showed it was still **half the decode** of the slow files
(`P3-sRGB-color-bars.v_d1` 0.14s -> 0.07s, `splines.v_e3` 0.30s -> 0.15s).

Both filters are now SSE2, vectorised **across x rather than across taps**.
That choice is what makes the result bit-identical: each lane executes the
same operations in the same order as the scalar code, so there is no tolerance
question and the scalar path remains the reference. `_mm_div_ps` rather than
`_mm_rcp_ps` for the same reason.

- **`epf_pass`**: four samples per iteration when the whole quad is in the
  mirror-free interior. Sigma blocks are 8 wide and the quad is 4-aligned, so
  `x/8` is constant across it; `border_sad_mul` applies only to lanes 0 and 7
  of a block, which is two fixed patterns.
- **`jxl_apply_gabor`** had the same shape as the noise and upsampling bugs:
  `sample_clamped`, four branches, called eight times per sample even in the
  interior. Row pointers are now clamped once per row and the interior needs
  no clamping at all. Its scratch was `calloc`ed and fully overwritten, too.

`-DJXL_EPF_FORCE_SCALAR` builds the scalar side alone so the claim can be
checked rather than asserted, and it earned its keep immediately: the first
gaborish vector version paired the loads as `(n+s)+(w+e)` where the scalar
expression associates `((n+s)+w)+e`. Float addition is not associative, so 18
of 80 files differed in the last bit. Fixed, then 120/120 identical.

    P3-sRGB-color-bars.v_d1   50.9ms  6.57x -> 39.7ms  4.81x
    splines.v_e3             256.8ms  6.80x -> 171.7ms 4.53x
    flower.v_d1              244.3ms  5.27x -> 177.6ms 3.82x
    flower.v_prog            248.2ms  4.33x -> 185.8ms 3.27x

`jxl_apply_gabor` drops from 16.7% self to 2.3%. Corpus total **2.61x ->
2.43x**, 43.2s -> 40.3s. 1245/1245 ok, ASan clean, 115 fuzz reproducers clean.

### The noise convolution was doing 25 taps where 10 would do
Re-benchmarking after the corpus grew put `v_noise` at the top on both axes:
worst ratio (5.99x) and the largest share of our total decode time (8.4%).
Subtracting the same image without noise isolates the stage -- `flower` cost us
257ms of noise against libjxl's 14ms, `splines` 313ms against 15.5ms. Roughly
**18-21x slower than libjxl on that stage alone**, against ~5x everywhere else.

Three compounding problems, all in the 5x5 high-pass:

- **The box is separable and nothing exploited it.** A 5x5 box sum is a
  horizontal 5-tap followed by a vertical one: ten adds a sample instead of
  twenty-five multiply-adds.
- **`noise_mirror` -- a loop -- ran for every tap of every sample**, five for
  the rows and twenty-five for the columns, including deep in the interior
  where the answer is always `x+dx`. Exactly the mistake `epf_pass` had.
  Rows now mirror once per row, and each row splits into an interior span that
  needs no mirroring and two short edges that do.
- **Six frame-sized planes were `calloc`ed** and immediately overwritten --
  24 bytes per pixel of zeroing thrown away. Groups tile the frame and each
  writes its whole extent, so `jxl_malloc` is enough.

The two hot loops are SSE2, which is baseline on x64 -- no runtime dispatch,
and the lanes are summed in the same order as the scalar path, four at a time.

    flower.v_noise    509.7ms  7.97x -> 262.4ms  4.46x
    splines.v_noise   600.9ms  9.54x -> 337.9ms  5.51x
    P3-color-bars     106.7ms  9.82x ->  57.5ms  4.92x
    v_noise preset    5.99x, 4086ms -> 3.61x, 2335ms

The stage itself is now ~25ms against libjxl's 13.5ms on `flower`: from 18x
down to 1.85x, and `flower.v_noise` is now *faster relative to libjxl* than
the same image without noise.

The same treatment on `upsample.c`, which had the identical per-tap mirroring
and is the innermost loop of the whole filter, took `v_rs2` from 5.96x to
4.90x and `v_rs4` from 5.17x to 4.07x; its 25-tap dot product is SSE2 now too.
Less dramatic because that filter is genuinely 25 taps per *output* sample and
is not separable -- each of the N*N positions has its own kernel -- so what is
left is arithmetic libjxl also does, just 8 lanes wide to our 4.

Corpus total: **2.74x -> 2.61x**, our wall clock 48.4s -> 43.2s. 1245/1245 ok,
ASan clean, all 115 fuzz reproducers clean.

### Importing other decoders' crashes
Two JPEG XL projects ship their crash reproducers, and both were already on
disk under `deps/`:

- `deps/libjxl/testdata/oss-fuzz` -- 44 ClusterFuzz testcases against
  `djxl_fuzzer`, already minimised.
- `deps/jxl-oxide/crates/jxl-oxide-tests/tests/fuzz_findings` -- 63 regression
  findings, each *named for the bug it caught*: `dequant_matrix_band`,
  `hf_varblock_across_group`, `ec_upsampling`, `icc_parse_oob`,
  `hybrid_integer_bits`, and so on. That naming is a map of where a JPEG XL
  decoder actually breaks.

**All 115 pass, including our own eight.** The check is not vacuous: 93 of the
106 external files get far enough to parse a header and 11 decode completely,
so they are reaching real frame-decoding and error-handling code rather than
bouncing off the signature check.

They are referenced from `deps/`, not copied in -- `get-deps` already fetches
both repos. `bun cmd/fuzz.ts -check` replays the lot in seconds as a
regression gate, and the first fuzz run now seeds from them as well, which
matters more than the check: each is a minimised input that already broke a
decoder, so they are far better mutation starting points than anything cjxl
emits.

Also looked at, and worth recording so nobody repeats the search:
`libjxl/jxl-rs` and `imazen/zenjxl-decoder` keep no checked-in crash
artifacts (both use ClusterFuzzLite, whose corpora live server-side). jxl-rs
does ship the 38 official conformance images under
`jxl/resources/test/conformance_test_images`, which is a better feature corpus
than ours in places -- but not for the PQ/HLG gap: 21 declare sRGB, 2 declare
709, one a gamma curve, and the remaining 14 carry an ICC profile instead.
`tf_pq` and `tf_hlg` stay dark.

### Fuzzing, and the first crash it found
`cmd/tests.ts` only ever feeds the decoder *valid* files, and a valid file
takes no error path: `jxl_errorf`, which every `JXL_ERR` in the decoder
funnels into, does not execute once across all 1245 corpus files. Everything
the decoder does when the bytes lie to it was unverified.

`bun cmd/fuzz.ts` is libFuzzer + ASan over `test/fuzz_target.c`, modelled on
`../heicdec`'s. The target opens the input, reads info and the ICC profile,
then renders up to four frames in two formats. It installs a budgeted
allocator (512 MB live per input): a crafted size header can declare an
enormous canvas, and the decoder has no pixel limit of its own, so bounding
allocation makes those inputs fail through the library's own
allocation-failure paths -- more error handling under test -- instead of
tripping libFuzzer's RSS limit and stopping the run. The corpus seeds from the
1245 real files, so mutation starts from valid codestreams.

It found a crash within minutes, and the same one eight times:

    jxl_image_metadata_free   headers.c:387
    jxl_doc_close             doc.c:52
    jxl_doc_open              doc.c:44        <- the *failure* path

`image_metadata_read` assigned `meta->num_extra` straight from the bitstream
and only allocated `ec_info` afterwards. Both early returns in between -- the
"too many extra channels" rejection and the calloc failure -- left num_extra
non-zero with ec_info still NULL, and `jxl_doc_open`'s cleanup calls
`jxl_doc_close`, which walks num_extra entries of that null array. A read of
address 0x18, the offset of `name` in the first entry.

The count is now published only once the array behind it exists, and the free
refuses to trust num_extra on its own, since its caller sees metadata
abandoned at whatever point parsing gave up. All eight artifacts replay clean;
they are kept in `fuzz/crashes/` (tracked) as regression seeds. 1245/1245 still
ok, ASan clean.

Worth noting what this says about the corpus: a null-deref reachable from
`jxl_doc_open` on a 62-byte input survived 1245 conformance files and every
ASan run over them, because none of them is malformed.

### Non-sRGB primaries, finally exercised -- and correct
Acting on what the coverage run below reported. The obstacle was never the
decoder: it was that every corpus source is a PAM, which carries no colour
information, so cjxl tags everything sRGB. `v_icc` did not help because
ProPhoto is *too* wide -- cjxl embeds it verbatim rather than matching an enum,
and on that want_icc path `doc.c` deliberately resets primaries to sRGB to
match libjxl's CMS-less fallback.

The fix is to pick profiles narrow enough for cjxl to enumerate. DisplayP3-v4
lands as `Primaries: P3`, Rec2020-v4 as `Primaries: Rec.2100` with
`Transfer function: 709`, which between them put the decoder through
`jxl_opsin_matrix_for`'s non-sRGB branch and `tf_bt709`. Two presets, `v_p3`
and `v_2020`, 66 sources each -- the seven grayscale ones reject an RGB
profile, exactly as they do for `v_icc`, and the skip report now says so.

`color.c` goes from 30.3% to 66.9% of regions and 35.9% to 80.9% of lines;
uncovered functions across the decoder drop from 21 to 15. All 132 files match
`djxl` at max 1, so the Bradford-adapted matrix was right all along -- it had
simply never run.

`tf_pq` and `tf_hlg` are still dark and stay that way for now. They need a
*lossy* frame declaring PQ or HLG: `jxl_linear_to_tf` is only reached on the
XYB path, so testdata's `pq_gradient.jxl` does not do it (grayscale lossless,
samples already in the target space). Compact-ICC-Profiles has no PQ or HLG
profile, and this cjxl cannot read PNG at all -- libpng is deliberately not
built -- so the tagged HDR sources in testdata are out of reach too.

Worth recording for whoever picks this up: `Rec2020-g24-v4.icc` reaches the
`tf_have_gamma` branch and passes, but runs hotter than the rest of the corpus
(max 0-3 over 14 sources against max 1 for sRGB), so a preset for it may sit
on the peak threshold rather than under it.

### Measuring the untested paths instead of guessing at them
Every decoder bug in this run -- the noise seed, the loop-filter edge, missing
upsampling, spot colour -- lived in a path nothing in the corpus exercised, and
each was found by guessing where such paths were. `bun cmd/coverage.ts` builds
a clang source-coverage variant (`build.ts cov`, `-O0` so inlining cannot fold
one function's lines into another's), decodes the corpus under it, and reports
per-file line coverage plus every function that never ran.

Over all 1113 files: **83.8% of lines, 91.0% of functions, 21 functions never
executed.** The full list is in the tool's output; the one that matters:

**`color.c` is 30.3% covered, and the entire non-sRGB colour path is dark** --
`get_primaries`, `get_white_point`, `primaries_to_xyz`, `adapt_to_xyz_d50`,
`mat3_inv`, `tf_pq`, `tf_hlg`, `tf_bt709`. The status table above claimed
non-sRGB primaries were done and Bradford-adapted. The code is there; nothing
runs it. Every corpus file declares `Primaries: sRGB` -- including
`P3-sRGB-color-bars`, `R2020-sRGB-red` and `phu1or_alfann24_2020_g1`, whose
names describe their *content*, not their declared encoding. They are built
from PAM, which carries no colour information, so cjxl tags them sRGB. The
`v_icc` preset does not help either: on the want_icc path `doc.c` deliberately
forces primaries back to sRGB, matching libjxl's CMS-less fallback. So the
claim has been resting on nothing, and the table now says so.

Other unexercised paths worth naming: `read_dct_params` and `read_fixed` in
`vardct.c` (custom dequant matrices), `preview_header_read` and `read_customxy`
in `headers.c`, `pred_nee` and `props_get_extra` in `modular.c`.

Two things the tool got wrong at first and no longer does: it filed the test
harness under decoder code, because it took "everything after the first
`/src/`" as the project-relative path and this checkout lives under a path that
itself contains `/src/`; and it listed a header's static inlines once per
translation unit, so one unused inline read as sixteen findings.

### EXIF orientation: untested, and correct
The first path this run that turned out fine. `unapply_orientation` had never
been exercised: nothing in libjxl's testdata carries a non-default
orientation, and the corpus is built from PAM, which has nowhere to put one.

cjxl has no orientation flag, but it will take a raw Exif blob via
`-x exif=<file>`, and libjxl reads it with `IsExif` + `FindExifTagPosition`
(`lib/jxl/base/exif.h`) -- which wants the bare little-endian TIFF, not the
4-byte-prefixed payload a JXL Exif *box* carries (that form is rejected). A
26-byte blob holding nothing but tag 274 is enough, and `exifOrientationBlob`
generates one per value.

All eight match `djxl` at max 1, transposes included -- and they are genuinely
being applied by both sides, not ignored by both: 510x532 comes out 532x510
for orientations 5 through 8. Orientations 6 and 8 share those dimensions and
differ only in pixel order, so the comparison would catch them being swapped.

One preset rather than eight: `orientationFor` spreads the values across the
sources by a hash of the filename, so `v_orient` reaches every arm of the
switch (10 sources at 1, 6-13 at each of the rest) for 73 files instead of
584. Stable per name, so a file keeps its orientation as the source set moves.

### The corpus stopped hiding what it could not encode
`corpusFiles` did `if (!runCjxl(...)) continue;`, so a source cjxl refused
vanished from the corpus with no signal at all -- the same silent-coverage-loss
shape as the basename collision below. It now collects cjxl's own reason for
each failure and prints them. Nine, immediately:

- **Seven grayscale sources fail `v_icc`** (`JxlEncoderSetICCProfile() failed`)
  -- an RGB profile cannot be attached to a gray image, so that preset has
  been covering 66 sources, not 73, for as long as it has existed. Not worth
  working around: the only gray profile in Compact-ICC-Profiles is sGrey, and
  cjxl reduces it to the enumerated Grayscale/D65/sRGB fields, so it never
  takes the want_icc path the preset exists to exercise.
- **`flower_small.cmyk.jpg`** -- this cjxl build cannot read a CMYK JPEG at
  all ("Getting pixel data failed", with or without
  `--allow_jpeg_reconstruction 0` or `--lossless_jpeg=0`), so there is no CMYK
  coverage to be had from it.
- **`flower.png.im_q85_rgb_subsample_blue.jpg`** -- its encoder rejects it.

All three are libjxl-side limits rather than ours. The point is not that they
are fixable; it is that the corpus is now 1040 files *and says so* when it
would otherwise have been more.

### Spot colour is refused, not guessed at
`headers.c` parses the four spot-colour floats and nothing consumed them, so a
file with a spot channel would have decoded with the colour silently unmixed
-- the same shape as the upsampling bug. libjxl mixes it after the colour
transform (`stage_spot.cc`), and the whole operation is

    mix = spot[3] * s[x];   p[c][x] = mix * spot[c] + (1 - mix) * p[c][x]

for the three colour channels. Ten lines, deliberately not written: nothing in
libjxl's testdata carries a spot channel and cjxl cannot emit one, so it would
be untested code that merely looks handled -- which is precisely what the two
entries below were. `jxl_frame_decode` refuses such a frame instead. Every
other extra-channel type (depth, selection mask, CFA, thermal, black) is
carried through as a plane, which is what libjxl does with it too; only
kSpotColor gets a compositing stage there.

### Upsampling was missing entirely, and failed silently
`cjxl --resampling=2|4|8` codes the frame at 1/2, 1/4 or 1/8 resolution and
leaves the decoder to upsample. We never did. `CustomTransformData` was parsed,
weights and all (`headers.c`), and `fh->upsampling` was used to size extra
channels -- but no filter existed, so the frame stayed at its coded size.
`walk_frames` then saw `tmp.w != doc->size.width`, concluded the frame was a
crop, and blended the half-size image into the corner of a blank full-size
canvas. Output was the right dimensions, exit code 0, and **max 255, rms 168,
mean 150** -- three quarters blank. Silently wrong, which is the worst way to
be wrong.

`src/upsample.c` implements it: each input sample expands to an NxN block, and
each output sample is a 5x5 weighted sum of the input neighbourhood using a
different set of 25 weights per position in the block, clamped to the min and
max of that neighbourhood (which is what keeps it from ringing at edges). The
codestream stores only one quadrant of the weight set and only its upper
triangle, since the kernel is symmetric under both reflections and under
transposition; `build_kernel` expands it. The stage runs between splines and
noise, where libjxl's pipeline puts it, so noise is synthesised at full
resolution.

2x, 4x and 8x all land at max 1 against `djxl`, alpha included. Two presets
lock it down: `v_rs2` and `v_rs4`. Both are needed -- at 2x the weight table
is a single quadrant (H=1), so the quadrant mirroring only gets exercised from
4x up. cjxl downsamples the input itself, so neither preset inflates the
corpus.

Not implemented: an extra channel whose upsampling factor differs from the
colour channels'. libjxl handles it with a separate, earlier pass
(`!late_ec_upsample`), and our extra-channel plane sizing is wrong for it, so
`jxl_frame_decode` refuses the frame instead of misplacing the channel.
`--resampling=1 --ec_resampling=2` is the way to produce one.

1040/1040 ok, 0 failed. ASan clean.

### Noise was seeded one frame too early, and untested
Two problems, the second of which hid the first.

**No coverage.** `PROGRESS.md` claimed noise "reproduces approximately (max 5,
mean 0.47 on a photon-noise file)" and called it uninvestigated -- but nothing
exercised it: testdata ships no noisy image and no preset asked for one, so
that figure came from an ad-hoc file and no regression could have been caught.
`cjxl --photon_noise_iso=3200` sets `JXL_FF_NOISE`, so a `v_noise` preset costs
one line. It failed 47 of 73 immediately, peaking at 55 counts -- an order of
magnitude worse than the claim.

**The bug.** `walk_frames` counted `visible_frames` / `invisible_frames` after
rendering each frame. libjxl advances them in `InitFrame`, *before* decoding
(`dec_frame.cc`), so the first visible frame synthesises its noise with
`visible_frame_index` already 1. Every single-frame image was therefore seeded
0 where libjxl seeds `1 << 32`.

Everything downstream of the seed was already right, which is exactly why it
was hard to see: the XorShift128+ state init, `Fill`, `BitsToFloat`, the 5x5
high-pass and the 1/128-vs-127/128 channel mix all match libjxl line for line.
So the noise had the correct distribution and no relation to libjxl's. On a
flat 2048x2048 gray image the two fields agreed on standard deviation to 0.3%
(4.131 vs 4.145), agreed on the 5x5 autocorrelation to three decimals, agreed
on the 0.98 cross-channel correlation -- and correlated with each other at
-0.0025, at every spatial shift. Chasing the magnitude would never have found
it; only asking what the field's *statistics* said, versus its *values*, did.
Advancing the counters before the decode takes the correlation to 0.9928, the
remainder being ordinary float rounding.

`v_noise` now passes 73/73, peak 1, and the corpus is `1040/1040 ok, 0 failed`.
ASan clean.

### Three testdata images were never being tested
`sourceImages()` named each converted `.pam` after the source PNG's basename
alone, but testdata ships three names in two directories:
`external/wesaturate/500px/` and `.../64px/` hold *different* images called
`cvo9xd_keong_macan_srgb8.png`, `tmshre_riaphotographs_srgb8.png` and
`u76c0g_bliznaca_srgb8.png`. Both copies collapsed onto one `.pam`, so the
64px image was never converted and the 500px one was enumerated twice per
preset -- 821 paths, 788 distinct. That is why the failure list used to print
each of those files twice.

A colliding basename is now qualified with its parent directory
(`64px_u76c0g_bliznaca_srgb8.pam`); unique names keep the short form, so an
existing corpus is not regenerated wholesale. The corpus is still 821 files
but all 821 are distinct, and the 33 that were shadowed are real coverage:
15 byte-exact, 18 within one count. All pass.

Worth noting the 64px images are 64x64 -- a multiple of 8, so they would have
been the control that made the loop-filter edge bug obvious. Being silently
dropped, they weren't.

### The loop filters ran over the block padding, not the image
Every VarDCT file whose width or height is not a multiple of 8 had a wrong
last column and last row. `decode.c` called `jxl_apply_gabor` and
`jxl_apply_epf` with `vd.pw x vd.ph` -- the size rounded up to whole 8x8
blocks -- so on a 500-wide image the filters treated x=500..503, the partial
block's padding, as image content. At x=499 gaborish read a padding sample as
its east neighbour where libjxl mirrors: its render pipeline allocates and
mirrors at `frame_dimensions_.xsize_upsampled`, the real size
(`render_pipeline/simple_render_pipeline.cc`). Passing `color_w, color_h` with
the stride still `vd.pw` is the whole fix.

Found by asking where the outliers actually were rather than how big they
were. On `u76c0g_bliznaca_srgb8.v_icc` (500x500), 76 of the 80 samples off by
more than one sat in column 499; on `cvo9xd_keong_macan_srgb8.v_icc`, 61 of
112 sat in row 499. The control was `vgqcws_vin_srgb8` at 64x64 -- a multiple
of 8 -- with zero outliers, and `flower_small.rgb` at 510x532 put 115 in
column 509 and 46 in row 531.

The corpus had pointed at the `v_icc` preset, which was a red herring: `v_icc`
differs from `v_d1` only by an embedded ProPhoto profile, and on the want_icc
path `doc.c` falls back to linear output, where the same float discrepancy
maps to a bigger integer step. It was the most sensitive detector of the bug,
not the cause -- `v_d1` on the same image had 109 of 110 outliers in the same
column, quietly under the old tolerance.

Peak error over the whole corpus drops from 10 to 1: `821/821 ok, 0 failed`,
from 758/821. ASan clean.

### Splitting the MA node so a big tree fits in cache
`ma_get_leaf` is **72% of a lossy-Modular decode** -- by far the hottest
function left. The cost is not arithmetic: `flower_alpha.lm_d1` has a
1901-node tree, and at 28 bytes per node that is 53KB walked as a chain of
dependent loads, so it misses L1 at nearly every level. Only `property`,
`value`, `left` and `right` are read during a walk; `cluster`, `predictor`,
`offset` and `multiplier` are read once, at the leaf. Splitting those four out
into a separate `jxl_ma_leaf` array leaves a 16-byte node -- four to a cache
line -- and since the node array is breadth-first, the decision nodes a walk
actually passes through sit in its first half.

Measured interleaved A/B, 5 runs each, two passes, ours (libjxl column flat
throughout, so this is not machine drift):

| file | tree | before | after |
|---|---|---|---|
| `flower_alpha.lm_d1` | 1901 nodes, 53KB | 1166ms, 4.11x | 1096ms, 3.93x |
| `flower.lm_d1` | large | 593ms, 3.71x | 569ms, 3.60x |
| `flower_alpha.m_e9` | <=1175 nodes | 1207ms, 1.83x | 1169ms, 1.78x |
| `flower_alpha.m_e3` | 67 nodes, 1.9KB | 819ms, 2.23x | 816ms, 2.22x |
| corpus total | | 2.73x | 2.71x |

The last row of the table is the point: a 67-node tree is 1.9KB and was
already L1-resident, so halving the node does nothing for it. That is what the
cache explanation predicts and what a "fewer instructions" explanation does
not, and it is the reason to trust the other rows. Corpus-wide the effect is
small (~0.8%) because large-tree files are a minority; `ma_get_leaf` fell from
72.4% to 69.1% self on `flower_alpha.lm_d1`, 11025 to 9582 samples.

### One child index instead of two: 16 bytes -> 12
The rebuild feeds its FIFO with strictly decreasing indices and pops two per
decision node, so a node's two children are **always** neighbours, right
second. That is structural, and measuring it agreed: `right == left + 1` on
3665 of 3665 decision nodes across 120 corpus files. So only the left index is
stored and the walk derives the right one as `child + 1`, which also makes the
step branchless (`child + (v <= value)`). `jxl_ma_config_read` rejects a tree
where the invariant does not hold rather than trusting it silently.

Interleaved A/B, 5 runs, two passes, libjxl column flat throughout:

| file | 16-byte | 12-byte |
|---|---|---|
| `flower_alpha.lm_d1` | 1157ms | 1130ms |
| `flower.lm_d1` | 601ms | 586ms |
| `flower_alpha.m_e9` | 1249ms | 1208ms |
| `flower_alpha.m_e3` (control) | 852ms | 847ms |
| total of the five | 4549ms | 4454ms (-2.1%) |

### Two things tried on the MA walk that did not work
Both were measured interleaved A/B, 5 runs, two passes, and both were
reverted. The walk as it stands looks like a local optimum.

**Three levels per entry, instead of two.** The obvious follow-on: if one load
per two levels beat one per level, one per three should be better again. It is
**6% slower** (2.20x, 2.22x against 2.08x, 2.09x on the five-file set). Eager
evaluation is what makes the flattening pay, and it does not scale: covering
three levels needs 7 tests to advance 3 levels (2.33 per level) where two
levels need 3 to advance 2 (1.5 per level), and only 3 of those 7 are ever on
the taken path. Enough wasted evaluation to turn the loop from latency-bound
into throughput-bound. Two levels balances the two effects; do not "finish the
job" by going wider.

**Padding the entry from 28 to 32 bytes.** At 28 bytes an entry straddles a
64-byte cache line about 44% of the time, which looks like an easy fix. It is
a wash: `flower.lm_d1` and `splines.lm_d1` gain ~5%, `m_e9` and `m_e3` lose
~2-3%, netting to nothing (2.08x, 2.07x against 2.06x, 2.07x). Not worth 14%
more memory for no consistent direction.

### Reading all three tests up front, not two of them in sequence
Flattening halved the *loads*, but the chain that was left ran
`eval -> select -> eval -> load`: picking the second test only after the first
had decided made the second property read depend on the first result, so two
L1 loads that could have issued together were serialised.

Both candidate tests are already in the entry, so read all three properties up
front and select afterwards. The three loads are independent and issue
together; the chain per entry becomes one property read plus the node load.
Evaluating the not-taken branch's property is safe -- `props_get_extra`
bounds-checks and returns 0 for anything out of range -- and cheap, because
almost every property is a `cache[]` hit.

This is worth more than the flattening it builds on. Interleaved A/B, 5 runs,
two passes; absolute times drifted between passes here, so read the ratio
column, which is measured against libjxl in the same process:

| file | select-then-read | read-then-select |
|---|---|---|
| `flower_alpha.lm_d1` | 3.35x, 3.33x | 2.67x, 2.67x |
| `flower.lm_d1` | 3.30x, 3.20x | 2.53x, 2.61x |
| `splines.lm_d1` | 4.71x, 4.87x | 3.83x, 3.80x |
| `flower_alpha.m_e9` | 1.59x, 1.55x | 1.42x, 1.41x |
| five-file total | 2.44x, 2.40x | 2.08x, 2.11x (-13.6%) |
| corpus total | 2.54x | **2.33x** |

`ma_get_leaf` drops to 5627 samples and 57.4% self, from 8225 and 65.7%.
The lesson generalises: in this walk the binding constraint has been the
*length of the dependency chain* every time, not the instruction count and not
the footprint.

### Two tree levels per entry: the walk's chain, halved
The two shrinks above made the chain *narrower*. This one makes it *shorter*,
and it is worth more than both together. Average leaf depth is 16.7 (max 31)
on `flower_alpha.lm_d1`, 10.9 on `flower.lm_d1`, 5.1 on the small `m_e3`
trees, so the worst file serialised ~17 dependent loads per sample.

Each entry now holds two levels: its own test plus the test at whichever child
that first test selects. The four grandchildren sit contiguously at
`child + 0..3` as LL, LR, RL, RR, so the second level picks
`child + 2*o0 + o1` and one load advances two levels. A leaf carries its
payload inline instead of pointing into a side array, which also removes the
last indirection of every walk. A child that is a leaf has no test to
contribute: its slot gets property 0 (the channel index -- always defined,
always cheap) with a split of `INT32_MAX`, so the comparison is
unconditionally false and both of its grandchild slots hold that same leaf.

Interleaved A/B, 5 runs, two passes, libjxl column flat throughout:

| file | one level | two levels |
|---|---|---|
| `flower_alpha.lm_d1` | 1126ms, 3.90x | 991ms, 3.40x |
| `flower_alpha.m_e9` | 1203ms, 1.72x | 1099ms, 1.55x |
| `flower.lm_d1` | 581ms, 3.54x | 549ms, 3.26x |
| `splines.lm_d1` | 666ms, 5.31x | 637ms, 4.92x |
| `flower_alpha.m_e3` | 854ms, 2.21x | 821ms, 2.12x |
| total of the five | 4430ms | 4097ms (-7.5%) |
| corpus total | 2.71x | **2.54x** |

`ma_get_leaf` is 8225 samples, from 9374 before this change and 11025 at the
start of the three. Unlike the two shrinks, this one also helps the 67-node
`m_e3` trees, which is the expected signature of a load-*count* win rather
than a footprint one.

### Correction: node size was not the mechanism
The two entries below explain their gains as cache-footprint effects. A later
experiment says that is wrong, so do not reason from it. Padding the 12-byte
node back out to 20 with a field nothing ever reads -- pure footprint, no
extra loads, no extra work -- costs **nothing measurable** (3726/3760ms padded
against 3770/3758ms unpadded over four alternating runs). At these sizes the
array footprint is simply not the binding constraint.

The gains were real and repeatable, but they came from the fields that were
*removed being fields the walk actually loaded* -- `right` every step, and the
leaf payload riding along in the same access -- not from the array getting
smaller. That is also why the flattening above pays so much better: it attacks
the number of dependent loads, which is what actually binds. When optimizing
this walk, count loads, not bytes.

### MA tree walk: a leaf is now always a leaf (not a speedup)
`ma_get_leaf` carried a step counter that `break`s out of the walk once it
exceeds the node count, which returns a **decision** node to the caller --
which then reads `cluster`, `multiplier` and `predictor` off a node that has
none, decoding garbage instead of failing. The termination condition belongs
at parse time: the rebuild in `jxl_ma_config_read` fills each node from a
queue holding only indices greater than that node's, so children always point
forwards and every walk strictly increases the index. That invariant is now
checked once, after the rebuild, and a tree violating it is rejected. The walk
itself lost the guard, the `property == 0 / == 1` special cases (`props_compute`
writes both into `cache[0..1]` with the rest) and the `props_get` wrapper.

This was attempted as an optimization and **is not one**. Interleaved A/B over
the three heaviest Modular files, 7 runs each: 1086/1093ms with the change,
1090/1098ms without -- inside the run-to-run spread of either version alone,
and the per-file ratios against libjxl (the in-process control) are flat. MSVC
was already hoisting the two static-property tests and the guard increment out
of the loop. Keep the change for the correctness and the simpler walk; do not
expect time back, and do not re-profile this loop hoping for it.

### EPF and spline hot loops
See **Performance** above: three loops were repeating per-channel work that
does not depend on the channel, and `epf_pass` was calling `jxl_mirror` 720
times per sample. 3.22x -> 2.64x over the corpus, bit-identical output.

### YCbCr, ICC, and lazy dequant matrices
- JPEG-transcoded files (YCbCr frames) decode, including every chroma
  subsampling mode. The subtleties: the block grid rounds up to a whole
  number of subsampled blocks, so LF group rects are measured in blocks, not
  pixels; a subsampled channel lives in the top-left corner of each plane, so
  the per-group coefficient and LF-quant *views* start at the shifted block
  origin (getting this wrong decodes plausibly but wrongly, mean 4/255);
  chroma-from-luma is skipped entirely when anything is subsampled.
- `want_icc` images output **linear sRGB**. Recovering the real primaries and
  transfer curve from the profile is colorimetrically right but is not what
  libjxl does — without a CMS it falls back to linear sRGB and hands the
  profile to the caller. Matching libjxl took eight test profiles from
  max diff 74-169 to max 1.
- Grayscale XYB leaves three identical planes; dropping two is required or
  the extra-channel plane indices are off by two and alpha reads luma.
- Dequant matrices are built on demand. Materializing all 17 slots (including
  256x256) cost 8ms on files libjxl decodes in 0.2ms.

### VarDCT + multi-frame
- Frame iteration now decodes non-displayed frames (LF, reference-only) for
  their side effects and returns the requested *displayed* frame. LF frames
  (progressive DC) feed the next frame's LF image directly, skipping CfL and
  adaptive smoothing.
- Bug found: `jxl_read_toc` never called `jxl_dec_begin` after parsing the
  permutation decoder, so every permuted TOC (which is what `--progressive`
  produces) decoded to garbage offsets.
- Lossy Modular stores XYB as `(Y, X, B)` integers that must be scaled by the
  LF dequant weights (`m_*_lf / 128`), with `B += Y` first — not as [0,1]
  samples.

### VarDCT core
- Ported quantizer, dequant matrices (all 7 encoding modes incl. raw Modular
  matrices), HF block contexts, coefficient orders, HF coefficient decoding,
  per-varblock dequant, chroma-from-luma, and all 27 inverse transforms.
- The `natural order` generator keeps libjxl's `x < lbw && y < lbw` quirk
  (`lbw` twice, not `lbh`) — "fixing" it breaks large transforms.
- libjxl's sRGB encode is a rational-polynomial approximation, not `powf`;
  `color.c` ports it so the 8-bit output rounds the same way.

### Modular
- Byte-exact against `djxl` for lossless at effort 1/3/7/9, single- and
  multi-group, grayscale/RGB/RGBA, 8- and 16-bit, and `--responsive=1`.
- Two bugs worth remembering: the ANS end-of-stream signature is `0x130000`
  (not `0x13000`), and MA tree leaves store an *already clustered* index, so
  reading them must bypass the context→cluster map.

### Scaffolding
- djvudec-style layout: one internal header, `cmd/*.ts` build/test scripts,
  `dist/` amalgamation, MSVC + clang builds, ASan harness.
- The corpus is generated: `cmd/png.ts` decodes libjxl's testdata PNGs in
  TypeScript (libpng is deliberately not built into the oracle), `cmd/corpus.ts`
  encodes them with `cjxl` across 10 presets.
