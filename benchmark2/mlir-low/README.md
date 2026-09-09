# `mlir-low` lane — provenance & re-derivation

Bare-MLIR baseline at the **memref/affine** surface: kernels are hand-tiled with
explicit `affine.for` nests, `memref.load`/`store`, and an explicit boundary guard.
This is the lower of the two MLIR surfaces — what a compiler produces *after*
lowering — and it is where the M1 index-and-stride defect class actually lives.

All recorded numbers live in `results/mlir-low/*.jsonl` + `stats.json`. They are
**generated**, never hand-edited. To re-derive every number (CPU only — this lane
JITs on the host and has **no GPU dependency**):

```bash
# 0. pinned toolchain: LLVM 21.1.0 at $MLIR_LLVM_ROOT
#    (default $HOME/dev/croqtile/extern/llvm-project)
benchmark2/mlir-low/run.sh setup

# 1. full lifecycle -> results/mlir-low/*
benchmark2/mlir-low/run.sh all
#   = setup -> e2(small) -> minimal -> s12 -> e3 -> collect -> stats

# 2. the kernel gate is committed at BOTH sizes (matching iree's 311+311)
benchmark2/mlir-low/run.sh e2 --small
benchmark2/mlir-low/run.sh e2 --full

# or, per stage:
#   run.sh minimal [--small|--full]   E1 mutants  -> raw/mutants.jsonl
#   run.sh e2      [--small|--full]   kernel gate -> raw/kernels.jsonl
#   run.sh e3      [--small|--full]   expressibility + remainder
#   run.sh s12     [--small|--full]   ASan        -> raw/sanitizer.jsonl
#   run.sh collect                    raw/*.jsonl -> results/mlir-low/
#   run.sh stats                                  -> results/mlir-low/stats.json
```

`e4`/`e5` print "choreo only, not applicable". `--level2` is accepted by `minimal`
but has no effect here: M2 is `n/a` on this surface, so there is no level-2 set to
widen into.

Per-record `settings_hash` + `kernel_hash` + `toolchain_version` are carried on every
record (manifest §8). `raw/` **is** committed (unlike `iree`, which gitignores it),
so every committed number traces directly to committed raw JSON — satisfying
homework-check I4 outright rather than by provenance note.

**Reproducibility caveat.** Re-running `./run.sh all` reproduces every committed
artifact byte-for-byte *except* the `distribution` strings on the nondeterministic
cells. Four cells have produced a mixed distribution at least once (all RTV-off, all
M1): `relu/static` M1.1, `transpose/static` M1.1, `transpose/dynamic` M1.1, and
`layer_norm/dynamic` M1.2. That field records a 16-sample draw from genuinely
undefined behaviour, so its split varies run to run — e.g. `relu/static` reads
`12x never/noop, 4x runtime/corrupts` in the committed file and `7x/9x` in a fresh
run. Every one of these cells reduces to the *same* `outcome`/`manifest` pair under
the lane's conservative reduction rule (any `runtime` wins; any `corrupts` beats
`noop`), so no aggregate in `stats.json` moves — it was byte-identical across the
re-run. Only the raw distribution text moves, on up to four lines. A dirty `git
diff` confined to those `distribution` strings after a re-run is expected and is
**not** a regression. The `mlir-linalg` lane has no such cell and is
byte-reproducible throughout: its 120 M2 records are all `compile` (68, verifier
rejects a shape mutant — compilation is deterministic) or `never`/`corrupts` (40,
in-bounds wrong output, not undefined behaviour) or `n/a` (12), with **zero**
`runtime` outcomes, so nothing samples UB and it runs at `n_repeat=1`. That is a
structural argument, not a repeated-sample one: an M2 shape-contract defect either
fails the verifier or stays within its allocation, unlike M1.1's dropped boundary
mask whose out-of-bounds write is genuinely UB.

## Census arithmetic (§5.1 condition 2)

| | |
|---|---|
| injection specs | M1.1–M1.6 (6 specs) |
| categories (§5's M1 minimal set) | `relu`, `transpose`, `softmax`, `layer_normalization` |
| shapes | static + dynamic |
| **injections** | 6 × 4 × 2 = **48** |
| §0 target N per class | 40 |
| delta | **+8** (owner-approved 2026-09-09, auditable over-count — never trimmed) |
| mutants | **48** — M1 specs have no sub-variants, so one injection is exactly one mutant |
| records | **96** — 48 × 2 RTV modes |
| `n/a` cells | **0** — `transposed-stride` raised no `NotExpressible` on any of the four |

M2 and M3 are `n/a` on this surface (`n_na: 40` each): M2's specs are tensor-level
shape-contract defects, which do not exist once everything is a `memref`, and M3 is
hardware-specific with no GPU here. S1 therefore reports **M1 only** for this lane.

`minimal` checks its census against a **pinned literal**, not a value derived from the
categories it iterated — see `../mlir-linalg/README.md` for the tautology this
prevents. This lane's `categories` and `battery_cats` happen to coincide (all four
composed categories are M1 targets), but the separation is still enforced so the
check cannot silently degrade if the composed set grows.

## Measured results (small size, committed)

| statistic | value |
|---|---|
| kernel gate | **16/16 green** (8 small + 8 full), 0 gate failures |
| S1 M1 | `n_injected: 48, n_compile: 0, n_runtime: 38, n_never: 10, n_na: 0` |
| S8 | `elem {yes:4}`, `shape {no:4}`, `loop {no:4}`, `hw {no:4}` |
| S9 | **108** kernel guards (layer_normalization 42, softmax 26, relu 20, transpose 20) |
| S12 M1 | `flagged_and_exercised: 38` of 48 — 38 flagged, 10 genuine misses |

`n_compile: 0` is the headline difference from `mlir-linalg`. At memref level there
is **no shape contract left to check** — every operand is a bare pointer with a
layout, so nothing rejects a wrong extent at lowering. All 48 injections reach a
binary and run. Detection is therefore entirely a *runtime* phenomenon here, which is
why RTV matters so much on this surface and not at all on the other.

## The RTV contrast — why both columns are mandatory

| | RTV off | RTV on | effect |
|---|---|---|---|
| `low` M1 | `never:45, runtime:3` | `never:10, runtime:38` | **35 records flip** |
| `linalg` M2 | `compile:34, never:20, n/a:6` | identical | **no change at all** |

Bare MLIR on an out-of-bounds `memref.load` **silently returns garbage with exit 0**.
That is the fairness requirement manifest §5.1 exists to enforce: RTV is opt-in, so
the honest baseline is what happens without it. On this lane RTV does essentially all
of the detection work. Reporting only the RTV-on column would flatter this lane;
reporting only RTV-off would understate it by an order of magnitude.

## S9 is size-dependent on this lane

Measured RTV kernel guards, summed over each category's static+dynamic clean kernels:

| category | small | full |
|---|---|---|
| `relu` | 20 | **32** |
| `softmax` | 26 | **38** |
| `transpose` | 20 | 20 |
| `layer_normalization` | 42 | 42 |
| **S9 total** | **108** | **132** |

The cause is structural, not a defect. These kernels are **hand-tiled**, so RTV
instruments every `memref.load`/`store` inside the loop nest and the guard count
tracks nest depth. `FULL_DIMS` raises `relu` to `(32,512,8,8)` and `softmax` to
`(16,512,8,8)` — rank 4 against rank 2 at small size — so the nest is two levels
deeper. `transpose` and `layer_normalization` keep their rank and their counts.

**The committed S9 is the small-size figure, 108**, matching the independent
`_validate_rtv.py` census. `e3` runs at small size under `all`, and
`remainder.jsonl` carries no size field, so `e3-detail.json` records `size` and a
`size_caveat` — a reviewer re-running `./run.sh e3 --full` would get 132 and must not
read that as a regression. The `mlir-linalg` lane's S9 *is* size-invariant at 150,
because a `linalg.generic`'s instrumented access count is fixed by its indexing maps
rather than by rank.

## S12 — external sanitizer (real ASan)

Owner ruling: a **real measurement**, not `n/a`. ASan is applied natively on the host
CPU via `mlir-opt | mlir-translate | opt -passes=asan | clang -fsanitize=address`,
with RTV **off** (RTV's own `cf.assert` would abort before the faulty access and make
the sanitizer silent). The chain, its four LLVM-21 defects, and the false-negative
gate are documented in `../mlir-shared/README.md`.

Result: **38 of 48 flagged** (`heap-buffer-overflow`), with all 48 exercised and
`instrumented` 7–13 sites each. This lane is the positive control for S12: the M1
defects really are memory faults, so an external checker sees most of them — against
`mlir-linalg`'s 0 of 54, where the defects are shape faults.

### The 10 misses, audited individually

All 10 carry `instrumented` > 0, so all are genuine sanitizer negatives rather than
instrumentation failures.

* **8× M1.6** (zero-stride / empty-range) — 2 each on `layer_normalization`, `relu`,
  `softmax`, `transpose`, at both shapes. The mutant performs **no out-of-bounds
  access at all**: a zero stride or an empty range keeps every address inside the
  buffer. There is nothing for ASan to report. The output is still wrong, so the
  mutant records `outcome=never, manifest=corrupts`.
* **2× M1.4** (transposed-stride) — `relu` **dynamic** and `transpose` **dynamic**
  only. Static M1.4 on both *is* flagged.

### M1.4 is size-dependent, and the rule is exact

`transposed-stride` swaps the **last two read indices**. A swap only leaves the
allocation when the two extents it swaps between **differ**; on equal extents the
permuted address is still legal. So the miss set is a function of the shape table,
not of the mutation:

| category | small dims | trailing equal? | small M1.4 | full dims | trailing equal? | full M1.4 |
|---|---|---|---|---|---|---|
| `relu` | `(2,3)` | no | static **flagged**, dyn clean | `(32,512,8,8)` | **yes** | both clean |
| `softmax` | `(2,4)` | no | both **flagged** | `(16,512,8,8)` | **yes** | both clean |
| `transpose` | `(2,3)` | no | static **flagged**, dyn clean | `(32,64)` | no | both **flagged** |
| `layer_normalization` | `(2,2,4)` | no | both **flagged** | `(32,64,128)` | no | both **flagged** |

All 16 cells follow the rule with no exceptions. The dynamic cells need one more
step: `DYNAMIC_SLOT` binds axis 0 of `relu`/`softmax`/`transpose` to
`_DYN_VALUE = 3`, so at small size `relu`'s `(2,3)` becomes a runtime-**square**
`3x3` and `transpose`'s becomes `3x3` too — equal trailing extents, hence clean,
while their static forms keep `(2,3)` and are flagged.

Consequence for the paper: **S12's M1 flagged count is 38/48 at small size and 36/48
at full size.** The committed artifact reports the small-size figure and the
size-dependence is documented here rather than hidden. This is the same category of
finding as M1.6 — *a corruption that never leaves the allocation is invisible to any
memory checker* — and it is reported as a finding, not patched away by choosing dims
that force observability, because that would distort the kernel (the same reasoning
as owner decision 3 in `../mlir-shared/README.md`).

## Full-size validation

Full-size extents are ~1000× larger (`relu` is `32×512×8×8` ≈ 1.05M elements), so
this is a real exposure test rather than a repeat:

| check | small | full |
|---|---|---|
| e2 | 8/8 green | 8/8 green |
| minimal | 96 rec / 48 inj, §5.1 OK | 96 rec / 48 inj, §5.1 OK |
| s12 | 48 sanitized, reconciled, 38 flagged | 48 sanitized, reconciled, 36 flagged |
| S9 | 108 | 132 |

The full-size `minimal` was run with `M1_REPEAT=1` rather than the committed `16`,
because 16 repeats at full size is hours of wall-clock and the repeat count exists to
sample *nondeterminism*, not to test size correctness. Its RTV split therefore reads
`never:46, runtime:2` against the committed `never:45, runtime:3`: the one
nondeterministic cell that can actually flip (`relu/static` M1.1 RTV-off) sampled
once instead of 16 times.
That cell is a genuine coin flip — the committed small-size record logged
`12x never/noop, 4x runtime/corrupts` (p(noop)=0.75), the separate 30-run study in
`../mlir-shared/README.md` measured 0.633, and three later draws measured 0.44,
0.44 and 0.31 — so which bucket it lands in varies run to run by construction. That is the
expected consequence of N=1, not a size effect: the injection census, the §5.1
expectation check, and every deterministic cell are unchanged. The committed artifact
keeps `M1_REPEAT=16` at small size. Pooling every sampled run of that cell
available as of 2026-09-10 gives $p(\text{noop})=0.532$ (50/94 across five
independent draws), so the risk of the tally cell flipping on a re-run is
**4.1e-5 (~1 in 24,000) at the point estimate** — but $p$ is itself only known by
sampling and the five draws disagree, so the honest 95% interval spans ~1 in 1600
to ~1 in 685,000. See `../mlir-shared/README.md` for the full table. What is not
uncertain is the direction: `N=5` was ~4-10%, and `N=16` is at least two orders of
magnitude better, about three at the point estimate.

Three other cells (`transpose/static` and `transpose/dynamic` M1.1,
`layer_norm/dynamic` M1.2, all RTV-off) have also produced a mixed distribution at
least once, but at $p(\text{noop})\le1/16$ their flip probability is
$\lesssim10^{-19}$ — nondeterministic in the strict sense, stable in every practical
sense, and none of them moved the RTV split at N=1.

## Breadth gap — FLAGGED FOLLOW-UP for the coordinator

Manifest §1 counts **15** categories. This lane composes **4**: `relu`, `transpose`,
`softmax`, `layer_normalization` — the M1 minimal set from mutation-specs §5.

**Missing 11:** `batch_norm`, `concat`, `conv2d`, `elemwise_add`, `embedding`,
`gelu`, `matmul`, `max_pool2d`, `reduce_mean`, `reshape`, `sigmoid`.

S8 and the kernel gate are emitted over the composed subset **only**. The 11 missing
categories have no emitter on this surface yet, so their expressibility is
**UNMEASURED, not "no"**. Fabricating a `no` row for them would understate the SOTA
baseline and manufacture a result the lane did not obtain.

> **S8's denominator here is 4, not 15, so this lane's S8 row is NOT directly
> comparable to the `triton`/`iree` rows whose denominator is 15.** The gap is
> recorded in `stats.json`'s `scope` block and restated here so it cannot be missed
> when the rows are assembled into a table.

Owner ruling (2026-09-10): *ship green now at the composed subset; document the
breadth gap.* The lane runs end-to-end and passes every gate; closing the gap to 15
is a separate, flagged follow-up.

## Shape evidence basis

All four composed categories record `diagnostic-only (no shape diagnostic)` in
`e3-detail.json`, and `shape_compile_rejections` is empty. That is consistent rather
than a gap: M2 is `n/a` on this surface, so **no shape mutant was ever injected
here**, and the only assert class measured is `^ out-of-bounds access` ×42. S8's
`shape {no:4}` for this lane therefore means "this surface has no shape contract to
check", which is exactly what `n_compile: 0` in S1 says from the other direction.

There are **zero** alignment diagnostics on either MLIR surface — the independent
confirmation behind the M3.2 ruling that alignment contributes nothing to MLIR's
runtime verification.
