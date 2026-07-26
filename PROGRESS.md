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
1245-file corpus we are **1.91x libjxl** (2.74x before the SSE2 work below;
2.33x over the smaller 821-file corpus that predates the `v_noise`, `v_rs*`,
`v_orient`, `v_p3` and `v_2020` presets, which are lossy paths and pull the
average up). libjxl is AVX2 throughout; we are scalar C apart from the SSE2
hot loops (noise, upsampling, EPF, gaborish, both DCT passes, the XYB
inverse, the sRGB transfer function and the output quantiser), plus an AVX2
`epf_pass` chosen at runtime, so a constant factor is expected. Each of those
keeps a scalar twin and is checked bit-identical against it -- see the log.

`bun cmd/prof.ts <file.jxl>` profiles our decoder alone through the sibling
`../winperf` and prints its `-print-agent` report (top self-time functions,
hot source lines, heaviest call path). That is how the numbers below were
found. The tool was called `samply` until it was renamed, so log entries
below that date the rename refer to it by the old name; the invocation and
the report format are the same.

The worst ratios among files that take libjxl more than 5ms are the `v_rs2`
set, whose upsampling filter is genuinely 25 taps per output sample. Modular
is the strongest area -- `m_e9` is 1.25x and `m_e3` 1.65x. Files below ~1ms
sit at up to 9x purely on fixed setup cost.

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
The profile on `P3-sRGB-color-bars.v_d1` is flat -- `dct_1d` alone used to be
27%, and nothing is above ~14% now. Everything with a straightforward,
provably-bit-identical vector form has been done.

What is left is qualitatively different from the rounds above:

- **AVX2 beyond `epf_pass`.** The dispatch machinery exists now, but widening
  a kernel only paid for the one that is genuinely compute-bound; see the log
  for the measurements that killed the others.
- **`epf_pass` SAD symmetry.** The taps are symmetric pairs and the SAD is
  symmetric under swapping the two footprints, so `dist(x, y, k) ==
  dist(x+kx, y+ky, -k)`. Caching six SADs per sample in a rolling three-row
  window would halve the SAD work. The `sigma < 0.3` early-out complicates
  it: a skipped sample's neighbours still want its cached values.
- **Multithreading**, which is not implemented at all and would dwarf both.

### Unrolling the predictor's four-lane steps
`sc_predict` is the biggest single item at 25% self, so the line-level
profile is worth reading rather than guessing at: summing four `uint32`s
(`for (i = 0; i < 4; i++) sum_weights += weight[i];`) was **6.7% of the whole
decode** on its own. Three adds cannot cost that unless each one is a reload,
so `err_sum[]`, `weight[]` and `wn[]` became named locals and every
constant-trip loop over them was unrolled.

It is worth about 1% (`m_e3` 1.63x -> 1.61x), not the 6.7% the line suggested
-- so that line was mostly where the latency chain surfaces, not a memory
stall. Kept anyway: it is no less readable, and the profile no longer points
at a phantom.

The vector approaches were considered and rejected on inspection rather than
tried. The weight lane needs a per-lane variable shift and a 65-entry table
lookup: SSE2 has neither, and while AVX2 has `_mm_srlv_epi32` and
`_mm_i32gather_epi32`, a four-element gather costs more than four pipelined
scalar loads on every microarchitecture this would run on. The horizontal
sums are four elements -- an SSE2 reduction is more instructions and more
latency than three adds.

Corpus **2.02x -> 1.99x**, the first time under 2x.

## Log

### The fused-loop rewrite: the shadow rows are not redundant
libjxl decodes a Modular channel in one function with the neighbour tracking,
the property fill, the tree walk and the ANS read all inlined together. Ours
splits that across `props_compute`, `ma_get_leaf`, `jxl_dec_read_clustered`,
`predict_sample` and `pred_record`. Fusing them is the obvious remaining
structural idea, and the row-indexed rewrite of the *fixed-leaf* track earlier
had already won 11% on `m_e1` doing exactly this.

The part worth attacking is `pred_record`'s scratch. It keeps two width-sized
rows and copies every decoded sample into one of them -- a second store per
sample, plus a pointer swap per row -- to hold values the channel already
has. Reading the code, the two buffers are cleverer than they look: within a
row, `curr_row[x-2]` is the current row's already-written west-west sample
while `curr_row[x]` is still row *y-2*, so one buffer serves both `pred_ww`
and `pred_nn`. Both are nevertheless in the channel, at `[y][x-2]` and
`[y-2][x]`, so the buffers can be dropped and every neighbour indexed
directly.

Done, byte-identical over all 1242 corpus files -- and **3-4% slower**,
everywhere:

| preset | shadow rows p1 / p2 | channel-indexed p1 / p2 |
|---|---|---|
| `m_resp` | 1.337x / 1.330x | 1.389x / 1.386x |
| `lm_d1` | 1.335x / 1.331x | 1.390x / 1.394x |
| `m_e9` | 1.209x / 1.205x | 1.252x / 1.245x |
| `m_e7` | 1.234x / 1.221x | 1.272x / 1.270x |
| `v_prog` | 1.880x / 1.907x | 1.948x / 1.972x |
| `m_e3` | 1.326x / 1.321x | 1.356x / 1.345x |

(The absolute times overlap here because libjxl's own column drifted 6.7%
between runs; the ratio is what to read, since both decoders run interleaved
in the same process. Six presets, two passes, no overlap in any of them.)

The scratch rows were buying **locality**, not just bookkeeping. Two
width-sized buffers stay in L1 together; the channel's rows y, y-1 and y-2
are a stride apart, so on a wide channel each is a different set of lines,
and the index needs a multiply where the buffer needed a pointer add. The
extra store per sample is cheaper than the three scattered loads it saves.

Which also explains why the same transformation *did* pay on the fixed-leaf
track: there the predictors only reach n/w/nw, so it touches rows y and y-1 --
adjacent and hot -- and it removed a per-sample switch and the whole state
machine along with them. Here row y-2 is needed too and nothing else got
simpler.

Taken with force-inlining the Modular chain having measured 8.7% slower
earlier, the conclusion is that our split into separate functions is not what
costs us against libjxl. `ma_get_leaf` at ~4 dependent loads per sample,
`sc_predict`, and the ANS read are each near their own local optimum, and
fusing them does not make any of them do less work.

### v_prog: no v_prog-specific problem, and the MA walk is already short
`v_prog` at 1.93x was the worst remaining ratio and the one preset never
profiled. It turns out to have nothing of its own wrong with it. The `-vs`
profile is the same shape as `m_resp`'s:

| | |
|---|---|
| libjxl `DecodeModularChannelMAANS<0>` | 19.3% |
| ours `ma_get_leaf` | **14.2%** |
| ours `sc_predict` | 9.7% |
| ours `dec_read_symbol` | 6.3% |
| ours `jxl_modular_decode` (self) | 3.6% |
| ours `sc_record` | 2.6% |

36.4% against libjxl's 19.3% for the identical work. `v_prog` is simply a
VarDCT file whose Modular part -- the LF image, decoded once per pass --
dominates, so it inherits the Modular gap rather than having one of its own.
There is no progressive-specific work being repeated.

Three things checked, all negative:

**The fold threshold, again.** `v_prog`'s global tree is 1693 entries and its
channels go down to 32x64 = 2048 samples, so the rebuild looked nearly 1:1
with the samples it serves -- a much worse ratio than the `m_resp` case that
had already been swept. Sweeping it anyway: 1.932x at the current threshold,
1.932x / 1.942x / 1.964x / 1.967x as it rises. Flat then worse, same as
before. The earlier sweep not covering `v_prog` was a real gap in that test;
covering it changes nothing.

**Cache pressure from the big tree.** 1693 entries is 47KB, past L1, which
would explain a slow walk. But folding already collapses it: **1693 -> 185
for a 32x64 channel, 377 for 128x256**. The tree actually walked is 5-10KB.
Not the mechanism.

**Walk length.** Instrumenting `ma_get_leaf`: 4.20 entries per walk on
`v_prog`, 2.95 on `m_resp`, 5.16 on `m_e9`. Each entry covers two tree
levels, so that is 6-10 levels for trees of 185-377 entries -- about what a
balanced tree of that size costs, and there is no anomaly to remove.

So `ma_get_leaf` is at a local optimum given its shape: roughly four
dependent loads per sample, already two levels per load, already walking a
folded tree that fits L1. Closing the rest of this gap means what libjxl
does -- one fully inlined per-sample loop specialised per tree shape, with
the walk, the predictor and the ANS read fused -- which is a much larger
change than any single optimisation left on the list.

### The vertical squeeze was walking the image a column at a time
Two ideas came out of profiling `m_resp`, the biggest remaining loss. Only
the second one was right.

**The specialisation guard, which was wrong.** `m_resp`'s streams carry 20
channels of 32x64, so 2048 samples against a ~389-entry tree rebuild -- that
looked like the per-channel folding paying for itself many times over on tiny
channels, and the guard (`w*h >= nflat`) looked far too loose. Sweeping the
multiplier says otherwise:

| threshold | `lm_d1` | `m_resp` | `m_e3` |
|---|---|---|---|
| `x1` (current) | **1.480x** | 1.416x | 1.332x |
| `x8` | 1.545x | 1.406x | 1.324x |
| `x64` | 2.199x | 1.440x | 1.314x |
| never fold | 2.879x | 1.597x | 1.316x |

Raising it only makes things worse, and `lm_d1` collapses -- folding pays even
on a 2048-sample channel, because it shortens *every* walk and the rebuild is
one pass. The guard is already in the right place.

**The vertical squeeze, which was.** `squeeze_inverse_v` was 4.4% of `m_resp`
against libjxl's ~1.5%, and the reason is memory, not arithmetic. It walked
one column at a time: for a 2268-wide channel that is a strided gather
touching a separate cache line for all 1512 rows, then the next column comes
back for the same lines once they have been evicted, and the stores go back
out the same way.

The lifting step down a column is genuinely serial -- `top` feeds the next
row's tendency -- so this cannot be vectorised across y. But neighbouring
columns are completely independent, so a strip of 16 of them runs in lockstep:
64 bytes, so every cache line fetched is used in full, and the copy in, the
tendency pass and the stores back all run sequentially. `squeeze_tendency`
itself is left alone; it has int64 arithmetic and a divide by 12 that SSE2
would not enjoy, and it did not need touching.

| preset | per-column p1 / p2 | strip-16 p1 / p2 |
|---|---|---|
| `lm_d1` | 1.483x / 1.451x (1877/1707ms) | **1.318x / 1.335x (1568/1608ms)** |
| `m_resp` | 1.411x / 1.405x (2806/2635ms) | **1.346x / 1.333x (2503/2522ms)** |
| `m_e3` | 1.332x / 1.320x | 1.331x / 1.331x (no squeeze) |

The horizontal squeeze needs no equivalent -- it already runs along rows, so
its accesses were contiguous to begin with.

The arithmetic is untouched, so output is byte-identical to the previous
commit across all 1242 corpus files. Corpus **1.48x -> 1.46x**. ASan clean,
115 reproducers clean, amalgamation compiles.

### Upsampling eight wide, and one thing that looked obvious and was not
`v_rs2` and `v_rs4` were the worst ratios left, and a `-vs` profile of
`P3-sRGB-red.v_rs2` made the comparison unusually direct, because both
decoders name the same stage:

| | |
|---|---|
| ours `up_block4` | **26.4%** |
| libjxl `UpsamplingStage::ProcessRowImpl<2>` | **6.2%** |

Four times slower at the identical filter. Reading libjxl's version, the
algorithm is the same -- 25 taps per output, vectorised across x -- and it
differs in three ways: it runs eight lanes on AVX2, it uses FMA, and it sums
into **three** accumulators rather than one.

The three accumulators looked like the interesting one. Twenty-five products
summed into a single register is a 25-deep chain of dependent adds, and at
~4 cycles each that is 100 cycles of pure latency per output. Splitting it
three ways (and taking libjxl's exact grouping, so the rounding would match
the reference too) should have been most of the gap.

It is a **wash**: 700/683ms against 691/693ms on `v_rs2`. The reason is that
the chain was never exposed -- the `ox` loop already runs N independent
outputs over the same taps, so the out-of-order engine had plenty to overlap
the chain with. Reverted; it was 25 lines of unrolled code for nothing.

The gap was simply lanes. `up_block8` is `up_block4` widened, and only the
store needed thought: each vector holds one output column for eight input
samples while the row wants them interleaved, so splitting each accumulator
into its two 128-bit halves turns the problem back into exactly the
four-sample interleave that already existed, applied twice -- no 8-wide
shuffle network, and the N == 2 unpack and N == 4 transpose are reused as-is.
No FMA, so all three builds still agree bit for bit.

| preset | SSE2 4-wide p1 / p2 | AVX2 8-wide p1 / p2 |
|---|---|---|
| `v_rs2` | 2.405x / 2.417x (681/689ms) | **2.055x / 2.020x (587/585ms)** |
| `v_rs4` | 2.310x / 2.315x (438/445ms) | **1.828x / 1.854x (354/355ms)** |

Corpus **1.49x -> 1.48x**. Scalar, SSE2-only and AVX2 builds diffed
byte-identical over all 1242 corpus files, verdicts unchanged, ASan clean,
115 reproducers clean, amalgamation compiles.

### Caching the WP lookup table on the tree
The WP-error table is 16384 tree walks to build, and it was built **per
channel**. That is fine for one big channel and bad for the way frames are
actually laid out: `flower_alpha.m_e3` decodes 54 group streams of 4 channels
each, so it built 216 copies of the same table. `flower_alpha.v_e3` builds 54
(one alpha channel per group, each 256x256 -- exactly the size threshold, so
the table cost a quarter of the samples it served).

The table is a function of the WP error alone whenever no real node tests the
channel or stream index, which is exactly what `ma_tests_const_props` already
reports for the folding pass. Both files come back `fold=0`, so one table
serves every channel of every stream, and it now lives on the `jxl_ma_config`
and is built once.

That condition is doing double duty and both halves matter. It is what makes
the table channel-independent, and it is also what guarantees `cma == ma`, so
the leaf pointers the table stores point into `ma->flat` rather than into a
per-channel specialised tree that gets freed at the end of the channel.
Caching a table of dangling pointers is the obvious way to get this wrong.

| preset | rebuild p1 / p2 | cached p1 / p2 |
|---|---|---|
| `m_e3` (ours, ms) | 2858.6 / 2864.0 | **2745.3 / 2738.2** |
| `v_d1` (ours, ms) | 919.5 / 942.8 | 910.9 / 956.1 |

`m_e3` is a clean 4.2% with libjxl's column flat at 2070-2122ms. The VarDCT
presets do not move, and the reason is visible in the same debug line that
gave `fold`: `m_e3` carries four channels per stream against `v_e3`'s one, so
it had four times the redundant builds to remove. Predicting this would help
the alpha-heavy VarDCT files was wrong -- 54 rebuilds saved is real work, but
it is small next to the entropy decode those files are actually spending
their time in.

Output byte-identical to the previous commit across all 1242 corpus files.
Corpus stays at **1.49x** -- the gain is concentrated in one preset and does
not move the total. ASan clean, 115 reproducers clean.

`JXL_DEBUG_TRACK` now also prints `fold` and the first channel's dimensions,
which is what made the four-channels-versus-one explanation visible.

### Restructuring the fixed-leaf loop the way libjxl does
Reading `encoding.cc` again for the shape rather than the tracks: libjxl does
not just specialise on the *tree*, it specialises on the **predictor**, and
its fast tracks index the channel's own rows. Its "Gradient very fast track"
is a plain double loop with `left`, `top` and `topleft` read out of `r[x-1]`
and the row above, and the guess spelled out inline.

Ours resolved the leaf once and then still ran, per sample, a
`predict_sample` switch on a runtime predictor plus `pred_record` -- a state
machine maintaining n/w/nw and swapping a pair of scratch rows. All of that
to produce values that were already sitting in the channel: the scratch rows
hold exactly what the channel rows hold, both written from the same `value`,
and row y+1 is untouched until row y is done.

So the fixed-leaf track now switches on the predictor *outside* the loop and
indexes the channel directly. The edge cases have to be reproduced exactly,
and they turn out to be the same convention libjxl uses: `pred_record` leaves
n = w = nw equal to the previous sample on row 0 and equal to `prev_row[0]` at
the start of every later row, which is `left` falling back to `top` and
`top`/`topleft` falling back to `left`. Predictors that reach further than
n/w/nw -- north-east, west-west, the averages over them -- keep the state
machine, which knows how to find those.

| preset | state machine p1 / p2 | row-indexed p1 / p2 |
|---|---|---|
| `m_e1` | 1.921x / 1.937x | **1.724x / 1.722x** |
| `lm_d1` | 1.507x / 1.522x | **1.458x / 1.451x** |
| `m_e3` | 1.348x / 1.357x | 1.346x / 1.343x |
| `m_resp` | 1.402x / 1.402x | 1.398x / 1.402x |
| `v_e3` | 1.856x / 1.856x | 1.883x / 1.890x |

`m_e1` is 11% faster -- 922/946ms down to 837/832ms. `v_e3` reads worse but
is not: ours spans 1042-1070ms on *both* sides and libjxl's own column swings
561-577ms across the same runs, so that column is inside the noise. Only the
presets that actually take the fixed-leaf track move, which is the check that
the gain is where it claims to be.

The values computed are identical, only their source changed, so output is
byte-identical to the previous commit across all 1242 corpus files. Corpus
**1.51x -> 1.49x**. ASan clean, 115 reproducers clean, amalgamation compiles.

### Making the weighted predictor's bit-scans branchless: measured, reverted
`sc_predict` is still the single biggest item in a VarDCT-with-alpha decode
(13.5% on `flower_alpha.v_e3` -- the alpha channel is Modular-coded, and its
streams already take the WP lookup-table track, so the predictor itself is
the irreducible part). Its hottest line is the four `sc_weight_one` calls at
2.9%, and each contains a branch around a bit-scan:

```c
uint32_t shift = v > 1 ? jxl_floor_log2_u64(v) : 0;
```

`floor_log2(v | 1)` produces exactly the same value for every input -- 0 for
both 0 and 1 -- and removes the branch. Same trick applies to the `log_weight`
computation a few lines down. Four of these run per sample.

It does not pay. Interleaved, five runs, two passes, with libjxl's column flat
at 550-554ms throughout:

| preset | branch p1 / p2 | branchless p1 / p2 |
|---|---|---|
| `v_e3` | 1.849x / 1.843x | **1.900x / 1.891x** (worse) |
| `v_prog` | 1.905x / 1.934x | 1.925x / 1.947x (worse) |
| `m_e3` | 1.349x / 1.342x | 1.341x / 1.337x (better) |
| `m_e9` | 1.198x / 1.196x | 1.188x / 1.192x (better) |
| `m_e7` | 1.215x / 1.210x | 1.212x / 1.210x |

Small gain on the Modular presets, a real 2.3% loss on `v_e3` -- ours went
1024ms -> 1045ms while libjxl did not move, so it is the change and not the
machine. Reverted.

The explanation is that the branch was *predictable*. Error sums are usually
small, so `v <= 1` most of the time and the branch skipped the bit-scan
entirely; forcing `BSR` to run always puts its latency on the dependency
chain into `shift`, which then feeds both the table index and the final shift.
A well-predicted branch that skips a latency chain beats a branchless form
that cannot. Worth remembering before reaching for the next `| 1`.

### The weighted predictor was round-tripping through memory
With `m_e1`'s prefix decoding fixed, the biggest remaining losses were `m_e3`
(807ms) and `m_resp` (716ms), both ANS-coded. A `-vs` profile of
`flower_alpha.m_e3` put the whole thing in one place:

| | |
|---|---|
| libjxl `DecodeModularChannelMAANS<0>` | 41.9% -- the whole Modular decode |
| ours `sc_predict` | **21.7%** |
| ours `dec_read_symbol` | 8.2% |
| ours `ma_get_leaf` | 6.1% |
| ours `sc_record` | 5.7% |

`sc_predict` plus `sc_record` is 27.4% -- the weighted predictor is more than
half of our Modular time.

`sc_predict` computes four sub-predictions and a prediction, writes each
straight into the caller's `jxl_sc_result`, and then reads them back a few
lines later for the weighted sum and the clamp. Because they go out through a
pointer the compiler has to assume they might alias the `jxl_sc_pred` input,
so every one of those stores came straight back as a reload -- once per
sample. Keeping them in locals and storing once at the end is the same fix
the weight lanes in this function already had (they were unrolled out of
stack arrays for exactly this reason, and that was worth 6.7% at the time).

| preset | via `out` p1 / p2 | locals p1 / p2 |
|---|---|---|
| `m_e3` | 1.419x / 1.390x | **1.353x / 1.343x** |
| `m_e7` | 1.276x / 1.243x | **1.216x / 1.217x** |
| `m_e9` | 1.247x / 1.225x | **1.197x / 1.200x** |
| `m_resp` | 1.423x / 1.391x | 1.397x / 1.400x |
| `lm_d1` | 1.546x / 1.476x | 1.509x / 1.533x (noise) |

The three that move are the ones that lean hardest on the weighted
predictor; `lm_d1` barely uses it and its numbers are inside the run-to-run
spread. The arithmetic is untouched, so output is byte-identical to the
previous commit across all 1242 corpus files.

Corpus **1.52x -> 1.51x**. ASan clean, 115 reproducers clean.

### The bit reader was refilling on every peek
`m_e1` was the worst ratio among the files that lose real time -- 2.24x, 590ms
across the preset -- which is odd, because it takes the *cheapest* Modular
track there is: its tree is five nodes, folds to a single leaf, and skips the
property fill entirely. So the cost had to be in the entropy decode, and a
`-vs` profile (ours and libjxl in one process, both symbolized) said exactly
that:

| | |
|---|---|
| libjxl `DecodeModularChannelMAANS<1>` | **25.6%** -- the whole Modular decode |
| ours `pfx_read` | 22.3% |
| ours `read_uint` | 12.6% |
| ours `jxl_br_peek` | 4.4% |

`jxl_br_peek` appearing as its own entry is the tell: it was an out-of-line
call per decoded symbol. And it called `jxl_br_refill` *unconditionally* --
loading eight bytes, shifting and OR-ing them into the buffer even when the
buffer already held 56 bits and the refill would add nothing.

Both fixed together: `peek`, `consume` and `read` moved into the header as
inline functions, and the refill happens only when `nbits < n`. That is safe
for the obvious reason -- peeking n bits reads only the low n of the buffer,
and a refill only ever adds bits *above* position `nbits`, so skipping it
while `nbits >= n` returns the identical value. A short stream still refills
and still sets the sticky error.

| preset | out-of-line p1 / p2 | inlined p1 / p2 |
|---|---|---|
| `m_e1` | 2.243x / 2.248x | **1.904x / 1.905x** |
| `m_e3` | 1.388x / 1.398x | 1.385x / 1.400x |
| `m_e7` | 1.243x / 1.253x | 1.245x / 1.247x |
| `m_resp` | 1.399x / 1.392x | 1.389x / 1.387x |
| `lm_d1` | 1.506x / 1.508x | 1.500x / 1.498x |

Only `m_e1` moves, and that is the point rather than a disappointment: it is
the preset that codes with **prefix** (Huffman) codes, where `pfx_read` calls
peek and consume per symbol. The others use ANS, whose inner loop does its own
bit handling and never went through `jxl_br_peek`. Which says where to look
next.

Corpus **1.54x -> 1.52x**. Verdicts unchanged, ASan clean, 115 reproducers
clean, amalgamation still compiles.

### Packing output pixels without shuffling
`write_pixels` was 8.0% of a VarDCT decode -- about 12 cycles a pixel for
what is a quantise and a store. The quantise was already vectorised; the
store was not. Four pixels came back from `quantize4` as four separate
`uint32` arrays and were then written component by component, through a
per-pixel chain of tests on `gray`, `bgr`, `has_alpha`, `wide` and `ncomp`
that are fixed for the entire image.

The two common 8-bit RGB formats need no shuffling at all to pack. Each
lane's components are already 0..255 in separate 32-bit lanes, so
`r | g<<8 | b<<16 | a<<24` lays out a whole pixel per lane and four pixels
become one store. RGBA32 stores those sixteen bytes directly.

RGB24 has no fourth component to absorb the alpha slot, so each pixel is
written as a **4-byte** store whose top byte lands on the next pixel and is
overwritten by it. The loop stops four pixels short of the row so the final
overhang stays inside the row, and the scalar tail rewrites it. ASan over
the corpus is the check that the bound is right.

Measured interleaved, five runs, two passes:

| preset | scalar p1 / p2 | packed p1 / p2 |
|---|---|---|
| `jpeg` | 1.973x / 1.984x | **1.855x / 1.856x** |
| `v_icc` | 1.708x / 1.734x | **1.642x / 1.641x** |
| `v_d1` | 1.775x / 1.761x | **1.703x / 1.706x** |

The packed path produces the same bytes as the scalar one, so output should
be exactly unchanged, and all 1242 corpus files decode byte-identically to
the previous commit's binary. Corpus **1.58x -> 1.54x**. ASan clean, 115
reproducers clean.

### Two buffers that were being zeroed for nothing
`memset_repstos` sat at 7.6% of a VarDCT decode, and the coefficient planes
-- the obvious suspect, 41MB a frame -- genuinely need their zeros, because
the HF scatter writes only the non-zero coefficients and accumulates them
with `+=`. But 41MB is about 2% of that decode, not 7.6%, so the arithmetic
did not add up and it was worth counting rather than assuming. Instrumenting
`jxl_calloc` to report every allocation over 64KB gave 65MB zeroed per
decode:

| | |
|---|---|
| 3 x 13.7MB | coefficient planes -- **required** |
| 1 x 10.3MB | the output image buffer |
| 3 x 4.0MB | the LZ77 windows |

The other 23MB is waste. `write_pixels` covers every byte of the output
buffer -- its stride is exactly `width * bpp` and it writes all of them --
so zeroing it first is pure cost. And the LZ77 window never reads a slot it
has not written: a copy reads `window[copy_pos]` for `copy_pos` in
`[num_decoded - distance, num_decoded)`, `distance` is clamped to
`num_decoded` immediately above, and every index below `num_decoded` was
written by the store at the end of `jxl_dec_read_clustered`. That holds for
malformed streams too, so the output stays deterministic rather than
depending on what the allocator handed back.

Both arguments are the kind that are easy to get wrong, so neither is left
as an argument: `-DJXL_POISON_UNINIT` fills both buffers with 0xCD, and that
build was diffed against the zero-filled one over the whole corpus. No file
changed. (A first attempt at this checked nothing -- it passed an env var
the test runner does not read, so it silently re-ran the ordinary build.)

Measured interleaved, five runs, two passes:

| preset | calloc p1 / p2 | malloc p1 / p2 |
|---|---|---|
| `v_icc` | 1.891x / 1.892x | **1.698x / 1.713x** |
| `jpeg` | 2.076x / 2.076x | **1.983x / 1.993x** |
| `v_d1` | 1.826x / 1.834x | **1.764x / 1.777x** |

Corpus **1.60x -> 1.58x**. Verdicts unchanged, ASan clean, 115 reproducers
clean.

### Gaborish in place, with two saved rows
`memcpy_repmovs` was 6.0% of a VarDCT decode even after the coefficient
planes stopped being copied out. Most of what was left was gaborish: it
filled a full w*h scratch plane and then copied the whole thing back, three
full passes over the image per channel, all of it through DRAM.

None of that is needed. The filter writes row y from rows y-1, y and y+1 of
the *input*, and row y+1 has not been written yet when row y is produced --
so it can be read straight out of the plane. Only rows y-1 and y need to
survive being overwritten, and two saved rows is enough to run the whole
filter in place: save row y into a two-row ring just before writing over it,
and row y-1 is still in the other half from the previous iteration. A
row-sized buffer that stays in L1 replaces a full-image scratch plane.

Measured interleaved, five runs, two passes:

| preset | scratch p1 / p2 | in-place p1 / p2 |
|---|---|---|
| `v_icc` | 1.934x / 1.909x | **1.869x / 1.875x** |
| `v_d1` | 1.881x / 1.879x | **1.816x / 1.822x** |
| `jpeg` | 2.064x / 2.046x | 2.067x / 2.063x |

`jpeg` does not move because those files mostly do not run gaborish at all,
which is a reasonable check that the measurement is attributing the gain to
the right thing.

The arithmetic is untouched -- only where its inputs are read from -- so the
output should be *exactly* what it was, and that was checked rather than
assumed: all 1242 corpus files decode byte-identically to the previous
commit's binary. Corpus **1.60x**. ASan clean, 115 reproducers clean.

### The DCT column pass, eight wide
With dequantisation and chroma-from-luma vectorised, `dct_1d_v4` was left
holding 16.5% of a VarDCT decode -- the largest single item and, unlike the
memset beside it, actual arithmetic.

It was already four lanes wide. The reason to widen only the *column* pass is
structural: that pass gathers a whole row per step, and four consecutive
floats from a row are exactly four columns' values at that row, so the gather
already is the transpose and widening it is a wider load and nothing else.
The row pass would need a real 8x8 transpose in and out to match. On the 8x8
blocks that dominate a VarDCT frame, this turns the two column passes into
one.

`dct4_v8` and `dct_1d_v8` mirror the four-lane pair expression for
expression, so every lane performs the same operations in the same order as
the scalar code. No FMA, for the usual reason: fusing would round once where
the other two round twice.

Measured interleaved, five runs, two passes -- because the first full-corpus
sweep came back 1.64x against 1.63x and it was worth establishing whether
that was the change or the noise. It was the noise:

| preset | SSE2 pass 1 / 2 | AVX2 pass 1 / 2 |
|---|---|---|
| `v_icc` | 2.011x / 2.017x | **1.922x / 1.916x** |
| `v_d1` | 1.932x / 1.948x | **1.873x / 1.868x** |
| `jpeg` | 2.084x / 2.079x | **2.045x / 2.063x** |

Corpus **1.61x** (best of five; the `-runs 3` sweeps this session read about
a point higher and vary by about that much between runs). Scalar, SSE2-only
and AVX2 builds all diffed byte-identical over the 1242 corpus files,
verdicts against libjxl unchanged, ASan clean, 115 reproducers clean.

### Two VarDCT loops: a divide per sample, and a scalar dequantiser
With the presets clustered around 2x and no single file standing out, the
next targets came from the shared VarDCT core rather than any one preset.
`jxl_dequant_varblock` was 7.6% of a decode and `jxl_cfl_hf` 4.1%.

**Chroma-from-luma** was the sillier of the two. Its two correlation factors
are indexed by `x / 64`, so they are constant across each run of 64 columns,
but the loop recomputed both per sample -- an int-to-float and a **divide**
each, to produce a value that had not changed in 63 iterations. Walking the
row in runs of 64 hoists them out, and what remains is two multiply-
accumulates over contiguous floats, which vectorise directly.

**Dequantisation** is a per-coefficient convert-from-int, a biased rounding
correction, and two multiplies. The correction picks between `v * bias` and
`v - numerator / v` on `|v| <= 1`. Vectorised, both legs are computed and
selected bitwise -- and the divide in the discarded leg is by zero exactly
when the small leg is taken, so it is an infinity that the select throws
away rather than anything that can reach the output.

| preset | before | after |
|---|---|---|
| `jpeg` | 2.28x | **2.10x** |
| `v_icc` | 2.09x | **2.01x** |
| `v_d1` | 2.02x | **1.96x** |

Corpus **1.66x -> 1.63x**, the first time under 1.65x. Scalar and SSE2 builds
diffed byte-identical over all 1242 corpus files (`-DJXL_VARDCT_FORCE_SCALAR`),
verdicts against libjxl unchanged, ASan clean, 115 reproducers clean.

The 41MB of `memset` per frame that shows up alongside these is *not* a
target: the HF coefficient scatter writes only the non-zero coefficients and
accumulates them with `+=` across passes, so the plane genuinely has to start
at zero.

### Handing the coefficient planes over instead of copying them out
The VarDCT coefficient planes are padded out to whole 8x8 blocks, so they are
wider than the image. Cropping them into the output planes was a row-by-row
`memcpy` of the whole image, three times per frame -- 41MB on a 2268x1512
image -- plus three full-image allocations to copy into. `memcpy_repmovs` was
**9.4%** of a VarDCT decode and `memset_repstos` another 6.0%, the two of them
second only to the DCT.

None of it was necessary. A `jxl_fplane` already carries its own stride, so
the crop is nothing more than a smaller `w` and `h` over the same buffer: the
plane can simply take ownership of the coefficient allocation and keep the
block padding as stride. No copy, no second allocation, and the pages are
already warm.

What made this a real change rather than a one-liner is that two things
downstream assumed a plane's rows were contiguous -- `jxl_ycbcr_to_rgb` and
the `jxl_xyb_to_linear` / `jxl_linear_to_tf` pair were each handed
`w * h` samples as one run. Both now go a row at a time, which costs nothing
measurable: a row is thousands of samples, far more than the vector loops
inside them need to amortise a call. Everything else in the pipeline -- the
loop filters, upsampling, splines, noise, `write_pixels` -- was already
reading `plane.stride` and needed no change.

Every VarDCT preset moved, which is the point of it:

| preset | before | after |
|---|---|---|
| `jpeg` | 2.28x | **2.28x** |
| `v_prog` | 2.32x | **2.22x** |
| `v_icc` | 2.24x | **2.09x** |
| `v_e3` | 2.20x | **2.08x** |
| `v_p3` | 2.17x | **2.04x** |
| `v_d1` | 2.17x | **2.02x** |
| `v_2020` | 2.14x | **2.01x** |

Corpus **1.70x -> 1.66x**. Verdicts against libjxl unchanged, ASan clean --
which is the check that matters here, since the fix moves ownership of an
allocation and a stale free would show up as a double free. 115 reproducers
clean.

### The 709 curve was paying for precision libjxl does not use
`v_2020` sat at 2.61x while `v_p3` -- same kind of file, same pipeline --
sat at 2.17x. The only difference is the transfer function: P3 files declare
sRGB, and sRGB already had libjxl's polynomial-and-table fast path, while
Rec.2020 files declare **709**, which was a `powf(v, 0.45f)` per sample.
The profile put `powf` at **23.6%** of the whole decode, the hottest thing
in the file by a factor of three, with `jxl_linear_to_tf` at 34.1% inclusive.

libjxl does not call `powf` there either. Its own encode path evaluates the
curve with two rational polynomials -- `FastLog2f` and `FastPow2f` in
`base/fast_math-inl.h`, together about 3e-5 relative error -- so the exact
`powf` was not buying accuracy against the reference, it was buying
*disagreement* with it. Porting the same pair, scalar and four lanes wide,
makes the output closer to libjxl rather than further away.

Against an 8-bit output step of 1/255 there are four orders of magnitude to
spare, and the corpus agrees: every `v_2020` file reports byte-for-byte the
same error statistics against libjxl as before the change (`max 1, rms
0.499` and so on), and no verdict anywhere in the corpus moved.

| | before | after |
|---|---|---|
| `v_2020` | 2.61x | **2.14x** |
| `v_p3` (control) | 2.17x | 2.17x |

Which lands `v_2020` on top of `v_p3`, as it should be -- the transfer
function was the whole gap.

Only the 709 curve is switched. The gamma, DCI and PQ paths still call
`powf`: nothing in the corpus exercises them, so the change could not have
been measured or verified there, and an unverifiable change to numerics is
not worth making. Corpus **1.72x -> 1.70x**; scalar and SSE2 builds diffed
bit-identical, ASan clean, 115 reproducers clean.

### Upsampling: across x, not across the taps
`v_rs2` at 3.82x and `v_rs4` at 3.23x were the worst remaining ratios, and
`jxl_upsample_plane` was 36% of a resampled file's decode -- still 35.8%
after the easy fixes below, so the shape of the loop was the problem, not
its details.

The filter expands each input sample to an NxN block, every output of which
is a 5x5 weighted sum of the input neighbourhood, then clamps it to the min
and max of that neighbourhood. Written per output sample that is: gather 25
taps into a scratch array, reduce them to a min and a max, then do NxN
25-tap dot products each ending in a horizontal reduction.

Vectorising **across x** instead changes all of it. The tap a lane wants is
`srow[py][x + px - 2]`, so four consecutive x are four consecutive floats:
the neighbourhood is never gathered, each tap is one unaligned load feeding
four outputs at once, and the horizontal reduction disappears -- one
accumulator per output column, 25 products added in plain scalar order. The
clamp bounds come from five sliding 5-wide min/max windows over the same
loads. Each vector then holds one output column for four input samples, so
the row wants them interleaved: N=2 is an unpack pair, N=4 a 4x4 transpose.

Two smaller fixes first, both worth having on their own: the 25-element min
and max was 48 scalar compares per input sample, and the two innermost loops
re-tested `dx >= out_w` and `dy >= out_h` per output sample when they can
only fire on the last partial block. Together 3.82x -> 3.51x and
3.23x -> 2.93x; the restructure took it the rest of the way.

| | before | +bounds/minmax | +across-x |
|---|---|---|---|
| `v_rs2` | 3.82x | 3.51x | **2.81x** |
| `v_rs4` | 3.23x | 2.93x | **2.63x** |

The wide path handles the interior; the first and last two columns, the last
partial block, and any plane too narrow for it still go one sample at a time.
Those helpers are now plain scalar, which is a **correctness** improvement as
much as a simplification: the hand-vectorised 25-tap dot product they used to
run summed four partial accumulators and reduced them, an association the
scalar loop does not use, so the scalar and vector builds had never actually
agreed. Diffing the two builds over the corpus found 56 files differing.
With the interior on the wide path and the border scalar, all 1242 agree
byte for byte -- which is what `-DJXL_UPSAMPLE_FORCE_SCALAR` now exists to
check.

Corpus **1.76x -> 1.72x**. Verdicts against libjxl unchanged, ASan clean,
115 reproducers clean.

### The spline splat, four and eight lanes wide
With the Modular work done, the single worst file in the corpus was
`splines.jxl`: 743ms lost at 4.30x, 2.5% of the whole corpus in one 81-byte
file. The profile was unambiguous -- `jxl_render_splines` was **87.6%** of
all samples, 84% of it on two source lines:

```
   45.8%  factor = spline_erf((half + 0.354f) * inv_sigma) -
                   spline_erf((half - 0.354f) * inv_sigma);
   38.5%  rows[c][x] += vs[c] * ff;
```

Per pixel that is a `sqrtf`, two rational-approximation `erf`s (a divide
each), and three read-modify-writes. Pure arithmetic over x with no
cross-lane dependency, which is the case where vectorising is bit-identical
rather than merely close: each lane runs the identical operations in the
identical order and rounds the same way. So `spline_erf` gained a 4-lane and
an 8-lane twin and the splat loop runs AVX2, then SSE2, then a scalar tail.

Two details worth keeping. The sign restore selects on `x < 0` rather than
copying x's sign bit, because `-0.0f` has the sign bit set but must take the
*positive* branch to match the scalar `x < 0.0f ? -result : result`. And no
FMA anywhere: fusing would round once where the scalar rounds twice.

Unlike the element-wise kernels -- where AVX2 measured a net negative and was
removed -- this loop is divide- and sqrt-bound, so the wider lanes pay:
421ms -> 306ms and 422ms -> 300ms, interleaved, two passes.

| | `splines.jxl` |
|---|---|
| before | 969ms, 4.30x |
| SSE2 | 370ms, 1.68x |
| AVX2 | **256ms, 1.15x** |

Corpus **1.80x -> 1.76x**. All three paths -- scalar, SSE2-only and AVX2 --
were diffed against each other over all 1242 corpus files and produce
byte-identical output; that is what `-DJXL_SPLINE_FORCE_SCALAR` and
`-DJXL_NO_AVX2` exist for. 1245/1245, ASan clean, 115 reproducers clean.

### Folding the constant tests out of the tree, per channel
Properties 0 and 1 are the channel and the stream index. Neither changes
while a channel is being decoded, so every test on them has an answer before
the first sample is read -- and `lm_d1` turned out to be almost nothing but
those tests. Instrumenting the walk:

| stream | entries/walk | of which constant |
|---|---|---|
| `lm_d1` main | 11.96 (~24 tests) | **20.52** |
| `lm_d1` others | 3.13 | 4.94 |
| `m_resp` | 4.51 | 1.52 |
| `m_e7` | 4.85 | 0.51 |
| `m_e9` | 5.07 | 0.76 |

85% of `lm_d1`'s longest walk was re-deciding, per sample, a question fixed
for the whole channel. So each channel now rebuilds the flat tree with those
folded away, and walks that instead. Afterwards:

| | entries/walk |
|---|---|
| `lm_d1` main | 11.96 -> **1.60** |
| `lm_d1` others | 3.13 -> 0.46 |
| `m_resp` | 4.51 -> 3.73 |

The folding has to happen *inside* the flattening, not as a pass over the
binary tree first. An entry packs two levels, so removing a node shifts which
node lands at which level, and only the flattening knows that. `ma_flatten`
therefore grew a `fold` flag and calls `ma_fold` on each node it is about to
place; `ma_fold` walks past runs of constant tests, terminating for the same
reason the sample walk does -- child indices are already known to increase.
The binary tree and the leaf array are kept in `jxl_ma_config` for this,
where before they were freed once the flat form existed.

Entries/walk under 1 is the interesting part of that table: it means most
walks now end at the root because the whole tree collapsed to a leaf for that
channel. That is the existing fixed-leaf track, so its test changed from "the
tree only tests properties 0 and 1" to "what is left after folding is a
single leaf" -- strictly more cases, since a tree that tests real properties
elsewhere can still collapse down the branch one channel takes. Worth 1.65x
-> 1.53x on `lm_d1` by itself, on top of the folding.

Two guards. The rebuild is O(tree) against one walk per sample, so a channel
needs more samples than the tree has entries to be worth it. And a tree that
never tests properties 0 or 1 must skip the rebuild entirely -- which cannot
be asked of `ma_props_mask`, because flattening pads the slot of a child that
is a leaf with property 0 and an unreachable split, setting bit 0 on nearly
every tree. `ma_tests_const_props` asks the binary tree instead, where every
node is a real test. (An earlier note here read "property 0 is in every mask"
as if that meant something; it was this padding.)

| preset | before | after |
|---|---|---|
| `lm_d1` | 2.81x | **1.55x** |
| `m_resp` | 1.62x | **1.44x** |
| `v_d1` | 2.14x | 2.15x (tests neither, unchanged) |

Corpus **1.92x -> 1.80x**. 1245/1245 with per-file verdicts unchanged,
ASan clean, 115 reproducers clean, and a bounded fuzz run that found nothing.

### The property mask for `lm_d1`: measured, did not pay, reverted
The two specialised tracks each require the tree to test almost nothing, so
the obvious generalisation was to compute *only* the properties the tree
actually tests, whatever they are. `ma_props_mask` already reports that set.
Of the sixteen properties only three cost more than a couple of ALU ops --
12, 13 and 14 read a neighbour out of the previous or current row behind an
edge test -- so those were the ones to skip. `JXL_DEBUG_TRACK`, added here,
says which streams could benefit:

| preset | mask | tree | track |
|---|---|---|---|
| `m_e1` | `0x1` | 5 | fixed leaf |
| `m_e3` | `0x8001` | 85 | WP lookup table |
| `m_e7` | `0xbe01` | 541 | general, tests 12 and 13 |
| `m_e9` | `0xffff` | 853 | general, tests everything |
| `m_resp` | `0x1f1` | 57 / 489 | general, **skips 12-14** |
| `lm_d1` | `0x1f3` | 289 | general, **skips 12-14** |
| `v_d1` | `0x80c7` | 77 | general, **skips 12-14** |

So `lm_d1` should have skipped all three. It gained nothing: **2.811x ->
2.803x**, with `m_resp` 1.624x -> 1.619x and `v_d1` unchanged. Interleaved
A/B, 5 runs, libjxl column flat throughout.

Written first as three `mask & bit ? load : 0` expressions it was *slower*
(2.842x): the compiler picks cmov, which performs the load anyway and adds
the mask arithmetic on top. Rewriting it as one branch on a flag that is
constant for the whole channel -- which does genuinely skip the loads --
recovered that and no more. Both forms reverted.

The lesson is about where the time is, not about the mask. Those three reads
hit rows that are already in L1 with branches that predict perfectly, so they
were never costing anything. What `lm_d1` actually spends its time on is
walking a **289-node tree** per sample, and the way to make that cheaper is
to make the walk shorter, not the property fill thinner. Property 0 is in
every mask in the table above and properties 0 and 1 -- channel and stream
index -- are constant for a whole channel, so the nodes testing them could be
partial-evaluated away per channel, pruning whole subtrees off the top. That
is the generalisation of the fixed-leaf track worth trying; this one was not.

### Specialising the Modular inner loop on the tree's shape
On branch `modular-fused-loop`. The `-vs` profile said libjxl does the whole
per-sample Modular pipeline in one function at 35.5% where ours takes five at
54.5%, and force-inlining made that *worse*, so the difference is not call
overhead. Reading `encoding.cc` shows what it actually is: libjxl keeps
**specialised inner loops** picked by the shape of the MA tree
(`is_gradient_only`, `is_wp_only`), and in those it replaces the tree walk
with a precomputed `context_lookup[]` table and computes **one** property
instead of sixteen.

Measuring our own trees first, rather than guessing which specialisation is
worth writing:

    m_e1   props=0x1     only property 0 (channel)
    m_e3   props=0x8001  channel + property 15 (the WP error)
    m_e7   props=0x9e01  channel + 9..12 + WP
    m_e9   props=0xfffd  nearly everything

Property 0 is the channel index and 1 the stream index, both constant for a
whole channel. So when the tree tests nothing else, **the leaf is constant for
the channel** -- resolve it once and the tree walk leaves the per-sample loop.
And since `predict_sample` only reads the property struct for the
self-correcting predictor, with no weighted predictor in play the whole
sixteen-property fill goes as well: `props_compute` and `ma_get_leaf` both
disappear for trees shaped like `m_e1`'s.

    flower_alpha.m_e1   292.5ms  3.10x -> 233.3ms  2.27x   (-20%)
    flower.m_e1                            171.6ms  2.24x

Corpus **1.99x -> 1.96x**. 1245/1245 ok with Modular still byte-exact, ASan
clean, 115 fuzz reproducers clean.

**The WP table, the second track.** When the tree tests nothing but channel,
stream and the weighted-predictor error, the leaf is a function of one value,
so it can be tabulated: 16384 entries from the clamped error, built once per
channel, replacing the walk with an array index. `sc_predict` still runs --
the predictor's state is sequential and libjxl computes it too -- but the walk
and fifteen of the sixteen property slots leave the loop.

The table is only valid if clamping the index cannot change a decision. The
walk tests `v > split`, so clamping to HI agrees with the truth when
`split < HI`, and to LO when `split >= LO`; `ma_wp_lut_ok` checks every split
tested against property 15 and refuses the track otherwise, which is the same
bail-out libjxl's `TreeToLookupTable` has. Building it costs 16384 walks, so
it is only taken on channels with several times that many samples.

    flower.m_e3   449.4ms  1.61x -> 347.2ms  1.25x   (-23%)

Both tracks together, by preset:

    m_e1   2.41x     m_e3   1.42x     m_e7   1.31x
    m_e9   1.27x     m_resp 1.65x     lm_d1  2.86x

Corpus **1.99x -> 1.91x** over the two tracks. What is left in Modular is
`sc_predict` itself, which neither decoder can avoid, and `lm_d1` at 2.86x --
lossy Modular, whose tree tests properties 0,1,4..8 and so takes neither
track.

### Profiling libjxl next to ourselves
`bun cmd/prof.ts -vs <file>` profiles the *benchmark* harness rather than the
standalone one, so both decoders run the same work in one process, and links
it against a second libjxl built RelWithDebInfo (`deps/libjxl-dbg`, built on
demand). libjxl's own functions are then attributed by name instead of showing
up as one opaque module, which is what makes "where are we slower" answerable
rather than a guess.

Two things had to be fixed to get a usable trace. samply silently fails to
attach when handed a **relative** exe path -- the symptom is a trace full of
unrelated system processes and a duration far shorter than the run, which is
easy to misread as "the profiler does not work here". And `prof.ts` treated
every flag as taking a value, so `-vs <file>` swallowed the filename; the same
bug `tests.ts` had with `-tol`.

On `flower.m_e3`, 61k samples over 16s:

    35.5%  jxl::detail::DecodeModularChannelMAANS<0>   (libjxl, all of it)
    25.3%  sc_predict                                  (ours)
    11.0%  ma_get_leaf
     7.5%  ans_read_symbol
     5.9%  jxl_modular_decode (self)
     4.8%  sc_record
           ------
    35.5%  libjxl total vs 54.5% ours -- 1.53x, matching the measured 1.65x

The structural difference is visible: libjxl templates its whole per-sample
Modular pipeline into **one** function, where ours is five with a call each.
That looked like the explanation, so I forced the three hot ones inline --
and it was 8.7% *slower* (`m_e3` 1.65x -> 1.76x). Call overhead is not the
cost; the bloat is worse than the calls. Reverted.

What did help, slightly, came out of reading `sc_predict` line by line
afterwards: `div_lookup()` rebuilt a 65-entry table behind an init check on
every call, and it is called once per sample. As a `static const` table that
is a call and a branch per sample gone -- worth 1-3% on Modular files
(`lm_d1` 2.42x -> 2.35x), and nothing at corpus level.

### A software bit-scan was 37% of a Modular decode
Every profile so far had been taken on a VarDCT file, so all the SIMD work
went there -- while Modular is about 60% of the corpus by time. Profiling
`flower.m_e3` instead put `sc_predict` at **37.2% self**, `ma_get_leaf` at
22.6% and `sc_record` at 8.0%: the weighted predictor, not anything that had
been touched.

Inside `sc_predict`:

    uint64_t v = ((uint64_t)err_sum[i] + 1) >> 5;
    uint32_t shift = 0;
    while (v > 1) { v >>= 1; shift++; }

That is `floor(log2(v))` written as a loop, and it runs **five times per
sample** -- four in the weight loop, once for `log_weight`. `err_sum` is a
sum of accumulated prediction errors, so the loop can run twenty-odd
iterations. On a 3.4MP three-channel image that is on the order of a billion
iterations to compute something the machine does in one instruction.

`jxl_floor_log2_u64` uses `_BitScanReverse64` on MSVC and `__builtin_clzll`
elsewhere, with the loop kept as a portable fallback. It computes the same
value, so nothing about the output changes -- and Modular is byte-exact
against libjxl, so the corpus would have caught it instantly if it did.

    flower.m_e3    556.3ms  1.94x -> 453.3ms  1.65x   (-18%)
    flower.m_e9                      575.0ms  1.25x
    flower.lm_d1                     377.5ms  2.42x

Corpus **2.18x -> 2.02x**, 39.0s -> 33.3s. The single biggest step of the
whole effort, and it was neither SIMD nor an algorithm change -- just a loop
that should never have been a loop. The same shape as `noise_mirror`,
`sample_clamped` and the per-tap mirroring in `epf_pass`: the recurring bug
in this codebase is computing something per sample that the hardware or a
hoist could do once.

1245/1245 ok, ASan clean, 115 fuzz reproducers clean.

### AVX2 with runtime dispatch -- and why only one kernel kept it
`jxl_has_avx2()` in `core.c` does the detection: CPUID for the feature bit,
plus OSXSAVE and XCR0, because a CPU that reports AVX2 under a kernel that
does not save the upper halves would corrupt them across a context switch.
Cached after the first call.

The kernels sit beside their SSE2 twins in the same translation unit, so
`dist/jxl.c` stays one file. MSVC accepts AVX2 intrinsics without
`/arch:AVX2`; clang takes `__attribute__((target("avx2")))` per function.
Neither needs a separate object with different flags, and neither lets the
compiler emit AVX2 into code that has not been guarded. `-DJXL_NO_AVX2`
forces SSE2, which is how the two are diffed: 313 files, identical. No
`_mm256_fmadd_ps` anywhere -- a fused multiply-add rounds once where the
scalar code rounds twice.

**The interesting part is what got removed.** I widened the easy things
first -- the XYB inverse, `tf_srgb`, the output quantiser -- because they are
simple element-wise loops. Measured against an otherwise identical SSE2
build, ten decodes, best of three:

    file                     sse2    all-avx2        epf-only
    P3-sRGB-color-bars.v_d1  0.51s   0.49s (-3.9%)   0.48s (-5.9%)
    splines.v_e3             1.84s   1.64s (-10.9%)  1.58s (-14.1%)
    flower.v_d1              1.72s   1.67s (-2.9%)   1.66s (-3.5%)
    flower.v_prog            1.86s   1.76s (-5.4%)   1.74s (-6.5%)
    flower.m_e3              6.19s   6.27s (+1.3%)   6.23s (+0.6%)

Every file is faster with the element-wise kernels *gone*. They are
memory-bandwidth-bound, so a wider vector buys nothing, and `quantize8` is
called once per eight pixels and returns straight into SSE and scalar store
code -- an AVX-to-SSE transition each time. So only `epf_pass` keeps an AVX2
path: 12 kernel taps x 5 SAD offsets x 3 channels per sample out of an
L1-resident window, which is the one place in the decoder that is genuinely
compute-bound. An octet also suits it better than the quad does, since sigma
blocks are 8 wide and `border_sad_mul` then applies to lanes 0 and 7 only.

Corpus-wide this is small -- **2.19x -> 2.18x** -- because the corpus is
Modular-heavy and Modular never runs EPF. The gain is real but concentrated:
3.5-14% on VarDCT files, nothing elsewhere. 1245/1245 ok, ASan clean, 115
fuzz reproducers clean.

### Not zeroing planes that get overwritten, and vectorising the quantiser
The last two single-digit items on the profile, and the first one was the
interesting one because the obvious version of it is wrong.

`memset_repstos` at 8.4% was `jxl_fplane_alloc`'s `calloc` clearing a frame's
worth of planes that the very next loop overwrites. The tempting fix -- swap
the allocator to `jxl_malloc` -- would have been a **security bug**, not just
a risky optimisation: `jxl_fimage_blank_like` allocates a canvas and writes
*nothing*, relying on the zeros, so blank frames would have started showing
whatever the heap last held. Checking the six callers individually, four fill
their plane completely (the VarDCT assembly, the two Modular ones, and
`jxl_fimage_copy`, plus the upsampler's destination) and one does not. So the
uninitialised version is a separate entry point, `jxl_fplane_alloc_uninit`,
and the zeroing one stays the default.

`-DJXL_FPLANE_ALWAYS_ZERO` turns it back into `calloc`, which converts "every
sample really is written" from an argument into a diff: any plane read before
being written would show up as garbage against zeros. 210 files, identical.

`write_pixels` at 8.2% is eight formats crossed with orientation and BGR
order, so vectorising the *packing* would have been a lot of code for a
single-digit share. Its time goes on `quantize` -- a multiply, an add, two
compares and a truncating convert, per component per pixel -- so only that is
vectorised, four samples at a time into a small array that the existing scalar
store loop reads. All the format handling stays in one place.

    P3-sRGB-color-bars.v_d1   28.0ms  3.40x -> 27.2ms  3.30x
    flower.v_d1              145.2ms  3.10x -> 139.6ms 3.04x
    flower.m_e3 (Modular)                      557.0ms 1.99x

Corpus **2.23x -> 2.19x**, 37.1s -> 36.2s. Across all six rounds: 2.74x ->
2.19x, 48.4s -> 36.2s, a quarter of the wall clock, every step verified
against a scalar twin. 1245/1245 ok, ASan clean, 115 fuzz reproducers clean.

### The colour conversion, four lanes at a time
`jxl_linear_to_tf` at 11% and `jxl_xyb_to_linear` at 6.1% both looked like
transcendental-function costs and were not. `tf_srgb` is already libjxl's
polynomial-and-table approximation rather than a `powf`, and the `cbrtf` in
the XYB inverse is already hoisted out of the loop. Both loop bodies are pure
arithmetic, which means they vectorise **bit-identically** -- no approximation
argument needed, and no tolerance risk.

The XYB inverse is a straight four-at-a-time rewrite. `tf_srgb` needed one
concession: its 16-entry power table is indexed by the exponent, so those four
lookups are extracted, done scalar and reassembled -- still far cheaper than
the cubic polynomial around them, which is the bulk of the work.

Worth noting how the mistake in this one surfaced. The SSE2 detection block
sat *below* `jxl_xyb_to_linear` in the file, so its `#ifdef` was false and the
vector path silently compiled to nothing -- a build that succeeds, tests that
pass, and no speedup. Only the benchmark showed it. Moved above both uses.

    P3-sRGB-color-bars.v_d1   31.6ms  3.80x -> 28.0ms  3.40x
    flower.v_d1              155.8ms  3.39x -> 145.2ms 3.10x
    flower.v_prog            162.8ms  2.86x -> 152.3ms 2.65x

Corpus **2.31x -> 2.23x**, 38.1s -> 37.1s. Over the four rounds: 2.74x ->
2.23x, 48.4s -> 37.1s, a quarter of the wall clock gone, every step verified
bit-identical against its scalar twin. 1245/1245 ok, ASan clean, 115 fuzz
reproducers clean.

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
