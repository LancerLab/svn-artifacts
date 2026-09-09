# `mlir-linalg` lane — provenance & re-derivation

Bare-MLIR SOTA baseline at the **tensor/linalg** surface: kernels are composed as
`linalg.generic` / `linalg.matmul` / `tensor.concat` over `tensor` operands, then
bufferized. This is the higher of the two MLIR surfaces — the one a user actually
writes — and it is the row the paper compares `choreo` against.

All recorded numbers live in `results/mlir-linalg/*.jsonl` + `stats.json`. They are
**generated**, never hand-edited. To re-derive every number (CPU only — this lane
JITs on the host and has **no GPU dependency**, unlike `iree`/`triton`):

```bash
# 0. pinned toolchain: LLVM 21.1.0 at $MLIR_LLVM_ROOT
#    (default $HOME/dev/croqtile/extern/llvm-project)
benchmark2/mlir-linalg/run.sh setup

# 1. full lifecycle -> results/mlir-linalg/*
benchmark2/mlir-linalg/run.sh all
#   = setup -> e2(small) -> minimal --level2 -> s12 -> e3 -> collect -> stats

# 2. the kernel gate is committed at BOTH sizes (matching iree's 311+311)
benchmark2/mlir-linalg/run.sh e2 --small
benchmark2/mlir-linalg/run.sh e2 --full

# or, per stage:
#   run.sh minimal [--level2] [--small|--full]  E1 mutants  -> raw/mutants.jsonl
#   run.sh e2      [--small|--full]             kernel gate -> raw/kernels.jsonl
#   run.sh e3      [--small|--full]             expressibility + remainder
#   run.sh s12     [--small|--full]             ASan        -> raw/sanitizer.jsonl
#   run.sh collect                              raw/*.jsonl -> results/mlir-linalg/
#   run.sh stats                                            -> results/mlir-linalg/stats.json
```

`e4`/`e5` print "choreo only, not applicable" — they measure the DSL's own
scheduling surface, which bare MLIR does not have.

Per-record `settings_hash` + `kernel_hash` + `toolchain_version` are carried on
every record (manifest §8), so each number traces to a build. `raw/` **is**
committed here (unlike `iree`, which gitignores it): the MLIR lanes produce small
text artifacts rather than `.vmfb` binaries and compile logs, so committing them
makes every committed number directly traceable to committed raw JSON without a
re-run. That satisfies homework-check I4's traceability requirement outright rather
than by provenance note.

## Census arithmetic (§5.1 condition 2)

| | |
|---|---|
| injection specs | M2.1–M2.5 (5 specs) |
| level-1 categories | `layer_normalization`, `matmul`, `concat` → 30 injections |
| level-2 categories | `elemwise_add`, `softmax` → 20 injections |
| shapes | static + dynamic |
| **injections** | 5 × 5 × 2 = **50** |
| §0 target N per class | 40 |
| delta | **+10** (owner-approved 2026-09-09, auditable over-count — never trimmed) |
| mutants | **54** — M2.5 emits two variants (partial-write, duplicate-write) |
| records | **120** — 54 mutants × 2 RTV modes, plus the n/a cells |

M1 and M3 are `n/a` on this surface (`n_na: 40` each): M1's specs are memref/affine
index-and-stride defects, which do not exist at tensor level, and M3 is
hardware-specific with no GPU here. S1 therefore reports **M2 only** for this lane.

`minimal` checks its own census against a **pinned literal** (`expected_injected`),
not against a value derived from the categories it iterated. An earlier revision
computed the expectation from the composed-category list, which made the check
tautological — it printed "expected 70 injections OK" for a census that violated the
owner-approved arithmetic, because it iterated 7 composed categories where M2 only
injects into 5. **An expectation computed from the thing being measured cannot fail,
and therefore cannot protect anything.**

## Measured results (small size, committed)

| statistic | value |
|---|---|
| kernel gate | **28/28 green** (14 small + 14 full), 0 gate failures |
| S1 M2 | `n_injected: 50, n_compile: 34, n_runtime: 0, n_never: 10, n_na: 6` |
| S8 | `elem {yes:7}`, `shape {yes:5, no:2}`, `loop {no:7}`, `hw {no:7}` |
| S9 | **150** kernel guards (concat 36, elemwise_add 10, layer_normalization 48, matmul 14, relu 6, softmax 30, transpose 6) |
| S12 M2 | `flagged_and_exercised: 0` of 54 — 34 rejected before run, 20 ran clean |

**M2 detection is entirely at the verifier.** 34 of 50 injections never reach a
binary: the linalg verifier rejects the shape contract at lowering. RTV adds
nothing at all — the RTV-off and RTV-on censuses are byte-identical
(`compile:34, never:20, n/a:6` both ways). That is itself the finding, and it is the
exact opposite of the `mlir-low` lane, where RTV moves 35 records from undetected to
caught. Both columns are required by manifest §5.1; reporting either alone would
misrepresent one of the two surfaces.

S9's 150 is **size-invariant** on this surface: a `linalg.generic`'s instrumented
access count is fixed by its indexing maps, not by tensor rank. (The `mlir-low`
lane's S9 is *not* size-invariant, because its kernels are hand-tiled — see that
lane's README.)

## S12 — external sanitizer (real ASan)

Owner ruling: a **real measurement**, not `n/a`. ASan is applied natively on the host
CPU via `mlir-opt | mlir-translate | opt -passes=asan | clang -fsanitize=address`,
with RTV **off** (RTV's own `cf.assert` would abort before the faulty access and make
the sanitizer silent). The chain, its four LLVM-21 defects, and the false-negative
gate are documented in `../mlir-shared/README.md`.

Result: **0 of 54 flagged.** This is not a null result and not a failure — it splits
into two structurally different halves:

* **34 rejected before run.** The verifier killed the shape contract at lowering, so
  no binary ever existed and there was nothing to instrument. Recorded as
  `rejected_before_run`, *not* as `not_instrumented` — conflating the two would
  report 34 false negatives where there are none.
* **20 ran clean under real coverage.** All are M2.5 (partial-write /
  duplicate-write), `instrumented` 23–44 sites each. The output is genuinely wrong
  (`outcome=never, manifest=corrupts`) but **every access stays inside its
  allocation**, so no memory checker can see it.

That 20 is the S12 residue the paper reports: shape faults are not memory faults. It
is the ledger-minus-sanitizer gap, and it matches what `iree` shows.

## Breadth gap — FLAGGED FOLLOW-UP for the coordinator

Manifest §1 counts **15** categories. This lane composes **7**:
`concat`, `elemwise_add`, `layer_normalization`, `matmul`, `relu`, `softmax`,
`transpose`.

**Missing 8:** `batch_norm`, `conv2d`, `embedding`, `gelu`, `max_pool2d`,
`reduce_mean`, `reshape`, `sigmoid`.

S8 and the kernel gate are emitted over the composed subset **only**. The 8 missing
categories have no emitter on this surface yet, so their expressibility is
**UNMEASURED, not "no"**. Fabricating a `no` row for them would understate the SOTA
baseline and manufacture a result the lane did not obtain.

> **S8's denominator here is 7, not 15, so this lane's S8 row is NOT directly
> comparable to the `triton`/`iree` rows whose denominator is 15.** The gap is
> recorded in `stats.json`'s `scope` block and restated here so it cannot be missed
> when the rows are assembled into a table.

Owner ruling (2026-09-10): *ship green now at the composed subset; document the
breadth gap.* The lane runs end-to-end and passes every gate; closing the gap to 15
is a separate, flagged follow-up.

## Shape evidence basis

S8's `shape` column is not uniform in how well it is evidenced, and `e3-detail.json`
records which:

| category | basis |
|---|---|
| `concat` | `mutation+diagnostic` — a shape mutant was injected *and* RTV emits a shape diagnostic |
| `elemwise_add`, `layer_normalization`, `matmul`, `softmax` | `mutation-only` — shape compile rejections measured (16, 16, 12, 8) but no shape-specific assert message |
| `relu`, `transpose` | `diagnostic-only (no shape diagnostic)` — composed and RTV-guarded, but not M2 injection targets, so no shape mutant was ever injected |

A composed category that is not a mutation-battery target never had a shape mutant
injected, so its verdict rests on the RTV diagnostic alone. That is still a
measurement, but a weaker one, and it is recorded as such rather than silently
counted equal.

Assert diagnostic classes measured: `^ out-of-bounds access` ×104,
`^ subview runs out-of-bounds along dimension N` ×16, `^ offset N is out-of-bounds`
×16. There are **zero** alignment diagnostics on either MLIR surface — the
independent confirmation behind the M3.2 ruling.
