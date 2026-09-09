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

So exactly **one** cell is nondeterministic. That matters because the record tally
is an artifact: `relu/static` M1.1 RTV-off lands in the `never/noop` bucket only if
*every* run lands noop, i.e. with probability $p^N$. At the old default `N=5` that
is **10.2%** — roughly one validation run in ten would have silently shifted a
tally cell (54/1/41 → 53/2/42). `_validate_m1.py` therefore defaults to
`N_REPEAT=16` (env `M1_REPEAT`), which puts the risk at **6.6e-4** (~1 in 1500):

| N | 5 | 8 | 12 | 16 |
|---|---|---|---|---|
| P(tally cell flips) | 1.0e-1 | 2.6e-2 | 4.2e-3 | **6.6e-4** |

The extra wall-clock is spent only on that one cell — every other (mutant, mode) is
deterministic, so its N runs finish in milliseconds.

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
