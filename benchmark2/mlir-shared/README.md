# `mlir-shared/` — shared harness for the two MLIR lanes

Code shared by the `mlir-linalg` (M2, tensor/linalg surface) and `mlir-low`
(M1, memref/affine surface) toolchain lanes. Neither lane directory owns this
code; both import it.

Pinned toolchain (manifest §5, owner-approved 2026-09-09): **LLVM 21.1.0** at
`~/dev/croqtile/extern/llvm-project/bin`. Override with `MLIR_LLVM_ROOT`.
Note `mlir-cpu-runner` was renamed **`mlir-runner`** in this release.

## Modules

| file | role |
|---|---|
| `mlirbench.py` | toolchain driver: pipelines, `classify()`, assert counting, schema validation, `RecordWriter` |
| `compose.py` | `Case` model, small/full dims, dynamic-slot binding, numpy reference oracle, `checksums()`, `tolerance()` |
| `emit.py` | mutation-agnostic tensor-level linalg emitter + three-checksum oracle |
| `emit_low.py` | memref/affine-level emitter for the `mlir-low` lane (M1), tiled with a real boundary guard |
| `mutate.py` | mutation by `Case` perturbation (never by editing emitted text) |
| `_validate.py` | E1 gate: do the clean kernels compose, compile, run, and match the oracle? |
| `_validate_mutants.py` | M2 on the linalg surface: does every injected defect actually manifest? |
| `_validate_m1.py` | M1 on the memref/affine surface: same question, per-mutant manifestation check |
| `_validate_rtv.py` | RTV instrumentation census on **both** surfaces, split kernel guards vs oracle guards |

## M2 injection census — and why it is 50, not 40

`mutation-specs.md` §0 fixes **N = 40 per class**, and §2 numbers M2 as **five**
specs. Two counting rules follow from that, and they are not the same number:

* **injection** = one `(category, shape, spec)` triple → this is S1's
  `n_injected`. Spec 5 counts **once**.
* **record** = one `(category, shape, mutant, rtv-mode)` tuple. Spec 5 is written
  as "partial write (omitted tail tile) **/** duplicate write (overlapping tile)"
  — two manifestations of one injection — so it emits **two** records.

Both variants are kept because both are real defects: each lands in the
`never`/`corrupts` row (survived the verifier *and* RTV, oracle proves the output
wrong), which is exactly the silent-bug residue the benchmark measures. Dropping
one to make the arithmetic tidy would delete a genuine finding.

### Level split

The schema's `mutant.level` enum (`{"1","2"}`) carries §5's split, so the
overshoot is attributable per-record rather than only in aggregate:

| level | categories | injections |
|---|---|---|
| 1 (§5 minimal set) | `layer_normalization`, `matmul`, `concat` | 30 |
| 2 (§5 additions) | `elemwise_add`, `softmax` | 20 |
| | **total** | **50** |

5 specs × 5 categories × 2 shapes = **50 injections**, against §0's target of 40.

**The +10 is deliberate, not a counting error.** Level-1 alone is 30, *under* the
target; adding both §5 level-2 M2 categories overshoots it. The owner was shown
that following §5's level-2 list literally breaks the exact-40 alignment and chose
it anyway, so the overshoot is recorded here rather than silently trimmed. Do not
"fix" it by dropping a category or a spec-5 variant without re-asking.

Of the 50 injections, **6 are `n/a`** (surface cannot express the spec):
`matmul` M2.3 (no binary elementwise op), `softmax` M2.1 and M2.3 (no secondary
operand, no binary elementwise op) — each across both shapes. §6 requires `n/a`
be counted and **never re-balanced** and never merged into "detected", so
applicable = 44.

Records emitted: 50 injections → 54 mutants (spec 5 doubles) → **120 records**
(× 2 RTV modes, which manifest §5.1 requires be reported separately, never
merged).

## The three-checksum oracle — and why two was a real bug

`compose.checksums()` returns **(sum, sum-of-squares, position-weighted sum)**.
The third is not redundant, and adding it was a *bug fix*, not a refinement.

`sum` and `sum-of-squares` are both **permutation-invariant**. Any M1 defect that
merely *reorders* values therefore matched the reference exactly and was scored
`noop` — a false success specs §7.1 requires be discarded, silently shrinking N.
This was latent in **both** lanes; M2 never exposed it because shape defects change
the value *multiset*, not just its arrangement.

Measured on `relu/static`, with the output dumped via `func.call @printMemref2dF32`:

| kernel | output | sum | sum-of-squares | verdict under 2 checksums |
|---|---|---|---|---|
| clean | `[[0,5,3],[1,0,0]]` | 9 | 35 | — |
| M1.2 off-by-one | `[[5,3,1],[0,0,4.6e-44]]` | **9** | **35** | `noop` ← **false success** |
| M1.3 negative-index | `[[0,0,5],[3,1,0]]` | **9** | **35** | `noop` ← **false success** |
| M1.4 transposed-stride | `[[0,1,2.4e-41],[5,0,0]]` | 6 | 26 | `corrupts` ✓ |

The fix weights each element by its flat position,
$w(i) = ((i \cdot 41) \bmod 127) - 63$, breaking the symmetry (M1.2: 9 → −37).

**`_W_MOD = 127` is chosen by proof, not by sweep.** The largest small-size output
is 48 elements (`concat`/dyn), so with `_W_MOD > 48` every weight in the level-1
gate is **distinct**, which makes the weighted sum **injective on value
placements** — no permutation of a level-1 output can evade it. An earlier
`_W_MOD = 17` still missed 87 of 20,000 random permutations, because weights then
repeat with period 17.

Both oracles (`emit.emit_oracle` and `emit_low.emit_oracle_low`) duplicate this
comparison logic and **must be changed together**; a fix applied to only one
silently makes the two lanes incomparable.

## Tolerance calibration — measured, not guessed

`compose.tolerance(nel, ref_value)` is the single source of truth, used by both
oracles. It has two terms because the kernel's f32 reduction accumulates error in
two regimes:

* **random walk** — `_TOL_REL * sqrt(nel) * max(1,|ref|)`, with `_TOL_REL = 1e-5`.
* **monotone drift** — `_TOL_DRIFT * nel * eps_f32 * max(1,|ref|)`, with
  `_TOL_DRIFT = 0.5`.

Two separate bugs motivated the two terms:

1. **The comparison must be relative.** The original absolute form
   (`1e-2 * sqrt(nel)`) was ~1000× too loose and let a real defect through:
   `softmax/static` with a dropped boundary mask shifts sum-of-squares by 1.35e-2,
   under the 2.83e-2 absolute budget → scored `noop`. Measured on the same kernels
   the largest *clean* discrepancy is **8.2e-7 relative**, so the defect sits
   ~4500× above the noise floor and a relative tolerance separates them cleanly.

2. **`sqrt(nel)` is wrong for monotone-sign sums.** A checksum whose summands are
   all positive — sum-of-squares always is — does not random-walk. The accumulator
   grows monotonically, and once it passes $2^{24}$ f32 stops representing integers
   exactly, so error grows **linearly** in `nel`. Measured at full size,
   `elemwise_add` (12,582,912 elements) drifts **3.2e7** on a sum-of-squares of
   4.3e8 — a 7.5e-2 relative error against a sqrt-only budget of 1.5e7, which
   **fails a clean kernel** (margin 0.47). The drift term restores margin to 10.5.

The drift term is negligible where it does not matter: at `nel = 8` it is 4.8e-7
against the walk term's 2.8e-5, so small-size sensitivity is unchanged.

**Consequence worth stating plainly:** at full size the sum-of-squares check on a
large-magnitude category is drift-limited and nearly blind. Detection there rests
on the `sum` and position-weighted checks, whose summands cancel and stay tight
(measured drift 0.0 on every full-size category). The three checks are OR-ed for
exactly this reason.

## M1 injection census — 48, and the mode-dependent defect

M1's six specs (§1) have **no sub-variants**, so on this lane one injection is
exactly one record per `(category, shape, spec)` — unlike M2, where spec 5 emits
two variants counting as one injection.

| | |
|---|---|
| level-1 categories (§5's M1 minimal set) | `relu`, `transpose`, `softmax`, `layer_normalization` |
| injections | 6 specs × 4 categories × 2 shapes = **48** |
| §0 target N per class | 40 |
| delta | **+8** (owner-approved 2026-09-09) |
| `n/a` cells (§6) | **0** — `transposed-stride` did not raise `NotExpressible` on any of the four at small size |
| records emitted | 96 (× 2 RTV modes) |

The +8 is the same kind of deliberate overshoot as M2's +10 and is **accepted as an
auditable over-count** under specs §5.1: N = 40 is a target, not a cap, and the
over-count is guarded by cross-surface uniformity (any lane following §1's six
specs × §5's four categories × 2 shapes lands on 48). No redesign to force it back
to exactly 40.

### The manifestation check is per *mutant*, not per record

RTV-off and RTV-on are two **measurements of the same injected defect** (manifest
§5.1 requires both be reported separately and never merged). So specs §7.1's
question — "did this mutant actually corrupt anything?" — must be answered by
aggregating across both modes. A mutant is a false success only when it is `noop`
under **both**. When exactly one mode reports `noop`, the defect is real but
mode-dependent: that is a **finding**, listed separately, not a harness bug.
(Owner ruling 2026-09-09, specs §7.1: record as a finding, do **not** redesign the
kernel to force output-observability.)

The measured instance is `relu` M1.1 (dropped boundary mask), at both shapes. The
tiled loop runs one iteration past the extent, so the guardless read and write
both address `inp[i, extent]` / `out[i, extent]`. In row-major layout that wraps
to `[i+1, 0]` — and because relu is elementwise, `max(x[i+1,0], 0)` is exactly the
value the clean kernel writes there. **When the write lands idempotently** the
output is bit-identical and no output-comparison oracle at any tolerance can see
it; RTV's bounds check still does.

**But this is undefined behavior, not a deterministic noop.** The out-of-bounds
write can instead corrupt the heap (→ `abort`) or wrap an index so a loop bound is
never reached (→ `hang`), and which happens depends on heap layout. A single
`classify()` therefore samples the UB once and any finding derived from it is not
reproducible. Measured over **30 runs** of the same lowered module, small-size:

| cell | measured distribution | p(noop) |
|---|---|---|
| `relu/dyn` M1.1, RTV-**off** | 30× `never/noop` | **1.000** (deterministic) |
| `relu/dyn` M1.1, RTV-**on** | 30× `runtime/corrupts` | 0.000 (deterministic) |
| `relu/static` M1.1, RTV-**on** | 30× `runtime/corrupts` | 0.000 (deterministic) |
| `relu/static` M1.1, RTV-**off** | 19× `never/noop`, 11× `runtime/corrupts` | **0.633** ← coin flip |

That study covers **`relu` only** — it was scoped to the cell that motivated the
repeat count, so it is not a census of nondeterminism on this surface. Sampling the
committed `mutants.jsonl`, a fresh `./run.sh all`, and two `_validate_m1.py` runs
finds **four** cells that have produced a mixed distribution at least once (all
RTV-off, all M1). Each row pools every draw of that cell made so far:

| cell | draws | pooled noop | $p(\text{noop})$ | $p^{16}$ |
|---|---|---|---|---|
| `relu/static` M1.1 | 5 | 50/94 | **0.532** | **4.1e-5** |
| `transpose/dynamic` M1.1 | 3 | 3/48 | 0.0625 | 5.4e-20 |
| `transpose/static` M1.1 | 3 | 2/48 | 0.0417 | 8.3e-23 |
| `layer_norm/dynamic` M1.2 | 3 | 1/48 | 0.0208 | 1.3e-27 |

So **four cells are nondeterministic, but only one is nondeterministic enough to
matter.** The reduction rule (`reduce_verdicts`) is conservative — any `runtime`
run wins, and any `corrupts` beats `noop` — so a recorded cell flips only when
*every* one of the N runs lands in the weaker bucket. For the three
`transpose`/`layer_norm` cells $p(\text{noop})$ is ~1/16 or less, making that
$\lesssim 10^{-19}$: they are nondeterministic in the strict sense and stable in
every practical sense. `relu/static` M1.1 RTV-off, at $p\approx0.53$, is the one
real exposure, and it is what `N=16` is sized against.

That matters because the record tally is an artifact: `relu/static` M1.1 RTV-off
lands in the `never/noop` bucket only if *every* run lands noop, i.e. with
probability $p^N$. At the old default `N=5` that is **4.3%** at the pooled $p$
(10.2% at the 30-run study's $p=0.633$) — roughly one validation run in ten to
twenty-five would have silently shifted a tally cell (54/1/41 → 54/2/40, counting
`never`/`corrupts`, `never`/`noop`, `runtime`/`corrupts` over all 96 records).
`_validate_m1.py` therefore defaults to `N_REPEAT=16` (env `M1_REPEAT`).

Note that this churns the *artifact* without moving the *gate*. A mutant counts as a
`noop` false success (specs §7.1) only when it is noop in every run of **every**
mode, and `relu/static` M1.1 RTV-**on** is deterministically 16/16
`runtime/corrupts` — so the validator's exit code is reproducible at any `N`, and
what `N=16` actually protects is the committed `distribution` text, not the verdict.

No aggregate moves in any of these cases. All four cells reduce to the same
`outcome`/`manifest` pair in every sample, and `stats.json` was byte-identical
across the re-run: the reduction is what makes the aggregate stable even though the
raw draw is not.

**The residual risk at `N=16` is an order of magnitude, not a precise figure.**
$p(\text{noop})$ is itself only known by sampling, and five independent draws of
this cell disagree (the validator runs in a tempdir, so its draws are genuinely
separate from any `run.sh` draw, and the two validator runs are separate from each
other). The pool below is a **snapshot as of 2026-09-10**; every further run of
this cell adds a draw and will shift $p$ slightly, which is itself the point — the
figure is an estimate, not a constant:

| draw | runs | $p(\text{noop})$ | implied $p^{16}$ |
|---|---|---|---|
| 30-run study (above) | 19/30 | 0.633 | 6.7e-4 |
| committed `mutants.jsonl` | 12/16 | 0.750 | 1.0e-2 |
| `_validate_m1.py` run #1 | 7/16 | 0.438 | 1.8e-6 |
| fresh `./run.sh all` | 7/16 | 0.438 | 1.8e-6 |
| `_validate_m1.py` run #2 | 5/16 | 0.3125 | 8.3e-9 |
| **pooled** | **50/94** | **0.532** | **4.1e-5** |

Pooling all 94 runs gives $p=0.532$ with a Wilson 95% interval of $[0.432, 0.630]$,
so the flip risk at $N=16$ is **4.1e-5 (~1 in 24,000) at the point estimate, with a
95% interval spanning ~1.5e-6 (~1 in 685,000) to ~6.1e-4 (~1 in 1600)**. Raising `N`
narrows this, but the honest reading is "roughly 1 in 20,000, plausibly anywhere
from 1 in 1600 to 1 in 685,000" — quoting any single figure (the 6.6e-4 an earlier
revision of this lane carried, derived from one 30-run draw) overstates the
precision. What is *not* uncertain is the direction: `N=5` was ~4-10%, and `N=16` is
at least two orders of magnitude better, about three at the point estimate.

| N | 5 | 8 | 12 | 16 |
|---|---|---|---|---|
| P(tally cell flips) at pooled p=0.532 | 4.3e-2 | 6.4e-3 | 5.1e-4 | **4.1e-5** |

The extra wall-clock is not spent only on that one cell — the lane repeats *every*
(mutant, mode) cell, and three others are nondeterministic too — but it is only
*needed* for that one. The other three have $p(\text{noop})\le1/16$, so their
flip probability is $\lesssim10^{-19}$ and no realistic `N` would change their
recorded outcome.

`_validate_m1.py` runs each (mutant, mode) `N_REPEAT` times via
`mlirbench.classify_repeat` and reduces the verdict over the measured distribution
(`reduce_verdicts`: any `corrupts` wins). The **gate is reproducible at any N** — a
mutant is a false success only when *every run of every mode* is noop, and RTV-on
always corrupts, so `all_noop_every_run` can never trigger here. Findings are listed
under `MODE-DEPENDENT DEFECT(S)` with their measured distribution (e.g. `relu/static
M1.1: RTV-off [8x never/noop, 4x runtime/corrupts], RTV-on [12x runtime/corrupts]`)
rather than a hardcoded idempotent-write story, and the validator exits 0. The
finding *set* still grows with N — deeper sampling surfaces rarer noop events (4 at
N=5, 7 at N=12) — while the record tally stayed byte-identical across the N=5 and
N=12 runs. That is the intended split: the artifact is reproducible, the
characterization is honestly reported as a sample.

## S12 — external sanitizer supplement (real ASan, measured)

Owner ruling: S12 is a **real measurement**, not `n/a`. The chain is

```
mlir-opt <bare pipeline> | mlir-translate --mlir-to-llvmir
  | patch sanitize_address + target triple/datalayout
  | opt -passes=asan | clang -fsanitize=address | run | parse report
```

Four defects in that chain had to be found and fixed before any number it produced
could be trusted; all four are recorded in `mlirbench.py` and in
`/memories/mlir-asan-instrumentation.md`. The short version: `clang` never
instruments a `.ll` input (instrumentation must come from `opt -passes=asan`);
LLVM's ASan pass only instruments functions carrying the `sanitize_address`
attribute, which `mlir-translate` never emits; `mlir-translate` emits no
`target triple`/`datalayout`, so ASan computes a wrong shadow offset and a genuine
overflow surfaces as `SEGV on unknown address`; and a native link needs both runner
`.so`s **plus** `LD_LIBRARY_PATH` set to `$LLVM_LIB` (rc=127 without it is a
*loader* failure, not a kernel crash).

S12 runs with **RTV off**. RTV's own `cf.assert` bounds checks would abort before
the faulty access and make ASan silent — the measurement would then be of RTV, not
of the external checker.

### The false-negative gate

Zero coverage is not a result. `run_asan` counts
`call void @__asan_report_{load,store}(?:\d+|N)` sites in the instrumented IR and
returns `stage="instrument"` with `exercised=False` when the count is 0. Grepping
`__asan_load4` is **not** sufficient — those appear as declarations even when
nothing was instrumented. `stats` refuses to report a run containing such a record:
a zero-coverage record is split by stage into `rejected_before_run` (legitimate —
the verifier killed it, no binary ever existed) or `not_instrumented` (hard failure
— the chain is broken and the run is not green). Measured coverage on the mutants
that did run: 7–13 sites on `low`, 23–44 on `linalg`.

### Measured results

| lane | class | flagged ∧ exercised | total | split |
|---|---|---|---|---|
| `mlir-low` | M1 | **38** | 48 | 38 flagged, 10 genuine misses |
| `mlir-linalg` | M2 | **0** | 54 | 34 rejected before run, 20 ran clean |

The two rows are the whole point of S12 and they must not be averaged. On `low` the
injected defects are **memory** faults — an out-of-bounds index really does leave
the allocation — so an external checker sees 38 of 48. On `linalg` they are
**shape** faults: 34 of 54 never reach a binary at all because the verifier rejects
the type contract at lowering, and the 20 that do run are all M2.5
(partial-write / duplicate-write), which produce a wrong *result* while every
access stays inside its allocation. A memory checker is structurally blind to those.
That is the ledger-minus-sanitizer gap, and it matches what `iree` shows.

### The 10 `low` misses, audited individually

All 10 carry `instrumented` 7 or 13, so all are genuine sanitizer negatives rather
than instrumentation failures.

* **8× M1.6** (zero-stride / empty-range), 2 each on `layer_normalization`, `relu`,
  `softmax`, `transpose`. The mutant performs **no out-of-bounds access at all** —
  a zero stride or an empty range keeps every address inside the buffer. There is
  nothing for ASan to report. The output is still wrong, so the mutant records
  `outcome=never, manifest=corrupts`.
* **2× M1.4** (transposed-stride), `relu` **dynamic** and `transpose` **dynamic**
  only. Static M1.4 on both *is* flagged (`heap-buffer-overflow, 0B after end of
  region`).

### M1.4 is size-dependent — and the rule is exact

`transposed-stride` swaps the **last two read indices** (`emit_low.py`:
`idx[k], idx[k-1] = idx[k-1], idx[k]`). A swap of two indices only leaves the
allocation when the two extents it swaps between **differ**; on equal extents the
permuted address is still a legal address. So the miss set is a function of the
shape table, not of the mutation, and it changes between sizes. Measured at both:

| category | small dims | trailing equal? | small M1.4 | full dims | trailing equal? | full M1.4 |
|---|---|---|---|---|---|---|
| `relu` | `(2,3)` | no | static **flagged**, dyn clean | `(32,512,8,8)` | **yes** | both clean |
| `softmax` | `(2,4)` | no | both **flagged** | `(16,512,8,8)` | **yes** | both clean |
| `transpose` | `(2,3)` | no | static **flagged**, dyn clean | `(32,64)` | no | both **flagged** |
| `layer_normalization` | `(2,2,4)` | no | both **flagged** | `(32,64,128)` | no | both **flagged** |

All 16 cells follow the rule with no exceptions. The dynamic cells need one more
step: `DYNAMIC_SLOT` binds `relu`/`softmax`/`transpose`'s **axis 0** to
`_DYN_VALUE = 3`, so at small size `relu`'s `(2,3)` becomes a runtime-**square**
`3x3` and `transpose`'s becomes `3x3` too — equal trailing extents, hence clean,
while their static forms still have `(2,3)` and are flagged. At full size `relu`
and `softmax` are square in the *static* table already (`8,8`), so both shapes go
clean; `transpose` stays `(32,64)` and both shapes stay flagged.

This is the same category of finding as M1.6: **a corruption that never leaves the
allocation is invisible to any memory checker.** It is reported as a finding, not
patched away by choosing dims that force the defect to be observable, because
forcing output-observability would distort the kernel — the same reasoning as owner
decision 3 below. The consequence for the paper is that S12's M1 flagged count is
**38/48 at small size and 36/48 at full size**; the committed artifact reports the
small-size figure and the size-dependence is documented here rather than hidden.

### Full-size validation

The kernel gate is committed at **both** sizes, matching the `iree` lane (311 small
+ 311 full). Full-size extents are ~1000× larger (`elemwise_add` is
`32×512×768` ≈ 12.6M elements), so this is a real exposure test, not a repeat:

| check | small | full |
|---|---|---|
| `linalg` e2 | 14/14 green | 14/14 green |
| `low` e2 | 8/8 green | 8/8 green |
| `linalg` minimal | 120 rec / 50 inj, §5.1 OK | 120 rec / 50 inj, §5.1 OK |
| `low` minimal | 96 rec / 48 inj, §5.1 OK | 96 rec / 48 inj, §5.1 OK |
| `linalg` s12 | 54 sanitized, reconciled | 54 sanitized, reconciled |
| `low` s12 | 38 flagged / 10 miss | 36 flagged / 12 miss |

The full-size `low` minimal was run with `M1_REPEAT=1` rather than the committed
`16`, because 16 repeats at full size is hours of wall-clock and the repeat count
exists to sample *nondeterminism*, not to test size correctness. Its RTV split
therefore reads `never:46, runtime:2` against the committed `never:45, runtime:3`:
the one nondeterministic cell that can actually flip (`relu/static` M1.1 RTV-off,
p(noop)=0.532 pooled over 94 runs — see the risk table above) sampled once instead
of 16 times. That is the expected consequence of N=1, not a size effect — the
injection census, the §5.1 expectation check, and every deterministic cell are
unchanged. The committed artifact keeps `M1_REPEAT=16` at small size.

### S9 is size-invariant on `linalg` but NOT on `low`

This was expected to hold on both lanes and does not. Measured RTV kernel guards,
summed over each category's static+dynamic clean kernels:

| category | `low` small | `low` full | `linalg` small | `linalg` full |
|---|---|---|---|---|
| `relu` | 20 | **32** | 6 | 6 |
| `softmax` | 26 | **38** | 30 | 30 |
| `transpose` | 20 | 20 | 6 | 6 |
| `layer_normalization` | 42 | 42 | 48 | 48 |
| `matmul` | — | — | 14 | 14 |
| `concat` | — | — | 36 | 36 |
| `elemwise_add` | — | — | 10 | 10 |
| **S9 total** | **108** | **132** | **150** | **150** |

The cause is the difference between the two surfaces, and it is structural rather
than a defect. On `low` the kernels are **hand-tiled**: RTV instruments every
`memref.load`/`store` inside the loop nest, so the guard count tracks the nest
depth. `FULL_DIMS` makes `relu` `(32,512,8,8)` and `softmax` `(16,512,8,8)` —
rank 4, against rank 2 at small size — so the nest is two levels deeper and each
level contributes guards. On `linalg` the body is a `linalg.generic` whose access
structure is fixed by its indexing maps, so the number of instrumented accesses does
not grow with rank; `relu` stays at 3 guards per shape whether the tensor is rank 2
or rank 4.

**The committed S9 is the small-size figure** (108 / 150), matching the independent
`_validate_rtv.py` census in the table below. `e3` is run at small size by
`cmd_all`, and `remainder.jsonl` carries no size field — so a reviewer re-running
`./run.sh e3 --full` on the `low` lane would get 132 and must not read the
difference as a regression. This is the second measurement on this lane (after M1.4)
whose value depends on the shape table, and both are documented rather than pinned
away by choosing size-invariant dims.

The S12 `instrumented` site counts are IR-structural and unchanged across sizes.

The mutation battery and sanitizer supplement are measured at **one** size
(`census.battery_size` / `census.s12_size` in `stats.json`), following `iree`, whose
`mutants.jsonl` carries no size field at all. `totals.kernels` therefore spans both
sizes while `census.n_records` spans one — the difference is not a shortfall.

### The RTV contrast that makes the comparison fair

Manifest §5.1 requires both RTV modes be reported, and the two lanes behave
oppositely. These are **record** counts (each injection × shape × RTV mode), not
injection counts:

| lane | RTV off | RTV on | effect |
|---|---|---|---|
| `low` M1 | `never:45, runtime:3` | `never:10, runtime:38` | **flips dramatically** |
| `linalg` M2 | `compile:34, never:20, n/a:6` | identical | **no change at all** |

Bare MLIR on an out-of-bounds `memref.load` silently returns garbage with exit 0, so
on `low` RTV is doing essentially all of the detection work — 35 records move from
undetected to caught. On `linalg` RTV adds nothing, because the verifier has already
rejected the shape contract before RTV ever runs. Reporting only the RTV-on column
would therefore flatter `low` and say nothing about `linalg`; reporting only RTV-off
would understate `low` by an order of magnitude. Both are required.

Do not confuse the `linalg` record count with S1's `n_never: 10`. S1 reports at the
**injection** level, and M2.5 emits two variants (partial-write, duplicate-write)
that count as one injection, so its 20 records reduce to 10. The 60 `linalg`
RTV-off records reduce to the 50 injections S1 reports (34 compile + 10 never +
6 n/a).

## Two invariants that are easy to break

1. **A mutant is always checked against the *unmutated* case's reference**
   (`emit_kernel(mcase, ref_case=clean, ...)`). Comparing a mutant against its own
   mutated reference makes every mutant look correct.
2. **S9 counts kernel guards only.** RTV also instruments this harness's own
   checksum oracle, whose loads/stores are equally real memref accesses. Every
   emitted op therefore carries a `loc` — `loc("kernel")` or `loc("oracle")` —
   because RTV bakes the location into the assert message, which is the only
   surviving attribution channel (`lower-affine` and `one-shot-bufferize` strip
   `loc` from IR *structure*). Use `count_kernel_asserts()`, never the raw total:
   counting oracle guards as kernel guards inflated S9 by ~17% overall and by 67%
   on `relu`.

## Known LLVM-21 constraints

Recorded in full in `mlirbench.py` and the module docstrings. The ones that bite
most often: `arith.maxf` → `arith.maximumf`; `memref.print` is gone (use a
`func.func private @printMemref2dF32` with the **ranked** type — unranked
segfaults); `tensor.fill` does not exist (use `linalg.fill`); `expand-strided-metadata`
is **required** before `finalize-memref-to-llvm` or `tensor.concat`'s strided
`memref.subview` fails to convert; `loc("name")` must be the **last** thing on an
op statement.

Affine-map expressions use **infix** operators: `affine_map<(d0) -> (d0 ceildiv 2 * 2)>`
parses, `ceildiv(d0, 2)` does **not** — and the error is misleading
(`use of undeclared identifier` pointing at the bare `d0`), because MLIR never
recognized the call and fell back to resolving `d0` as an SSA name.

MLIR float literals **require a decimal point**: `1e-5` fails with
`custom op 'e' is unknown`, `1.0e-5` parses. Python's `repr` omits the dot
whenever the shortest round-trip form uses an exponent (`repr(1e-5)` → `"1e-05"`),
so `f"{v!r}"` interpolation is unsafe — **always use `emit.f32_lit()`**.

To dump a memref mid-module use **`func.call`, not `llvm.call`** — the latter
rejects a memref operand (`must be variadic of LLVM dialect-compatible type`). The
dump declaration's memref type must match the buffer's declared type **exactly**,
including dynamic dims (`memref<?x3xf32>`, not `memref<3x3xf32>`).

Two harness-level traps that cost real debugging time:

* **Never reuse a Python variable name across two `Emitter.new()` calls in one
  emitted expression.** `new()` increments a counter, so
  `bad = e.new("t"); e.emit(f"{bad} = arith.ori {bad}, ...")` after a prior
  `bad = e.new("t")` renders the *fresh* name on both sides →
  `%t82 = arith.ori %t82, ...` → `operand #0 does not dominate this use`. This
  broke all 14 linalg kernels at once when the third checksum was added.
* **When scraping `mlir-runner` stdout for memref data, exclude the trailing
  return value.** `--entry-point-result=i32` makes the runner print the i32 on its
  own line after the dump, so a naive number regex picks it up as an extra data
  element (every parsed array came back one long).

Assert counting is **stage-dependent**: count `cf.assert` at MLIR level, or unique
`assert_msg_N` globals in lowered IR. The MLIR count is always exactly 1 more than
the lowered count (CSE merges two identical messages).

## Measured RTV guard census (feeds S9)

`_validate_rtv.py` walks both surfaces. The kernel/oracle split is the point: S9
counts **kernel guards only**, and the two lanes are reported separately, never
summed (§7).

| surface | clean kernels | Σ kernel guards | Σ oracle guards |
|---|---|---|---|
| `linalg` | 14 | **150** | 42 |
| `low` | 8 | **108** | 78 |

Every decoded assert message across all 22 kernels is a **bounds** check
(`^ out-of-bounds access`, `^ subview runs out-of-bounds along dimension N`,
`^ offset N is out-of-bounds`). There are **zero** alignment checks — the
independent confirmation behind the M3.2 ruling that alignment contributes nothing
to MLIR's runtime verification.

The `low` surface carries proportionally far more oracle guards (78 against 108
kernel guards) because its oracle reduces at memref level with an explicit
`affine.for` nest, so every load in all three reductions is instrumented
individually. That is exactly the inflation S9 must exclude.

Adding the third checksum reduction grew the *oracle* guard count but left
Σ kernel guards on `linalg` at exactly **150** — unchanged from the measurement
taken before the oracle went three-way. That is the evidence the `loc` discipline
keeps harness instrumentation out of S9.

## Owner decisions (resolved 2026-09-09)

Four questions were raised; all four are now ruled and the specs updated. They are
kept here as the audit trail for why the lane's numbers look the way they do.

1. **Contract arithmetic — 160 was stale; the target is 120.** §0 and manifest §4
   both said "N = 40 per class → **160 total**", but there are only **3** classes
   (3 × 40 = 120). 160 = 4 × 40, dating from a 4-class layout that predates §0's
   fold of iteration-validity into M1. **Both strings now read 120.**
2. **M1 = 48 (delta +8) — approved.** Same treatment as M2's owner-approved +10:
   an auditable over-count guarded by cross-surface uniformity, accepted as long
   as every surface shows the same M1 enumeration and the over-count is itself
   documented. **No redesign to force it back to exactly 40.** Recorded in specs
   §5.1 alongside M2's +10.
3. **`relu` M1.1 mode-dependent defect — record as a finding, do NOT redesign.**
   An elementwise map cannot force the defect to be output-observable without
   distorting the kernel; the mode-dependence is itself the honest result.
   Recorded in specs §7.1. `_validate_m1.py` lists it under
   `MODE-DEPENDENT DEFECT(S)` and exits 0.
4. **`mutation-specs.md` §7's dangling "§11.1" — fixed upstream.** Retargeted to
   "owner plan §11 item 1" in submodule commit `d0805aa`. The per-mutant (not
   per-record) rule is now stated in-repo at specs **§7.1**, and every `§11.1`
   citation in this lane has been retargeted to `§7.1`.
