# PROGRESS

Milestone log for the C JPEG XL decoder. Newest first.

## Status summary

`bun cmd/tests.ts -all` — 821 corpus files (libjxl's testdata images × 11 cjxl
presets, its JPEGs transcoded, plus the `.jxl` files that ship with libjxl):

```
821/821 ok, 0 failed
```

"ok" means byte-identical to `djxl file.jxl out.pam`, or inside the float
tolerance for the VarDCT paths. 381 files are byte-exact; across the other 440
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
| non-sRGB primaries (BT.2020, P3) | done (Bradford-adapted matrix) |
| images with an embedded ICC profile | done |
| patches / reference frames | **byte-exact** |
| splines, noise | done (noise is approximate, see below) |
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
2. **Noise** reproduces approximately (max 5, mean 0.47 on a photon-noise
   file). The XorShift128+ RNG and the 5x5 high-pass match jxl-oxide's
   structure but a small difference remains, uninvestigated.

## Performance

`bun cmd/bench.ts` links the `dist/` amalgamation and libjxl's static
libraries into one process and times both, single-threaded. Over the whole
821-file corpus we are **2.33x libjxl** (was 3.22x, then 2.64x). libjxl is
AVX2; we are scalar C, so a constant factor is expected.

`bun cmd/prof.ts <file.jxl>` profiles our decoder alone through the sibling
`../samply` and prints samply's `-print-agent` report (top self-time
functions, hot source lines, heaviest call path). That is how the numbers
below were found.

The worst ratios among files that take libjxl more than 5ms are now
`P3-sRGB-color-bars.v_*` at 6.4-6.6x and `flower.v_prog` at 4.5x. Files
below ~1ms sit at up to 9x purely on fixed setup cost.

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
`epf_pass` is still ~48% of a VarDCT decode, and it is now arithmetic-bound
rather than overhead-bound. Two further options, neither taken:
- The kernel taps are symmetric pairs and the SAD is symmetric under swapping
  the two footprints, so `dist(x, y, k) == dist(x+kx, y+ky, -k)`. Caching six
  SADs per sample in a rolling three-row window would halve the SAD work. The
  `sigma < 0.3` early-out complicates it: a skipped sample's neighbours still
  want its cached values.
- libjxl runs this loop 8 samples wide under AVX2. Matching that needs
  intrinsics, which would end the scalar-C property.

## Log

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
