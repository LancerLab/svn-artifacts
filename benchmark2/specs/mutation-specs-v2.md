# Mutation specs v2 — expanded coverage

**Version:** 2.0 (2026-09-10).
**Supersedes:** `mutation-specs.md` v1 for *new* generation only. v1 is frozen and
remains the spec under which all results up to and including the ASPLOS-era
`stats.json` were produced.
**Spec id to record in `stats.json`:** `spec_version = "v2.0"`.

Every generated mutant's `stats.json` **must** carry `spec_version`. A table that
mixes v1 and v2 rows is otherwise undetectable.

Related: `svn/eurosys27/plan/mutation-supplement-plan.md` (rationale),
`svn/eurosys27/plan/acceptable-e2e-test.md` (acceptance condition C1).

---

## §0 Scope and target

Three mutation classes. Out of scope, named: **concurrency safety** and
**numeric correctness** (unchanged from v1).

| Class | Meaning |
|---|---|
| M1 | element-access (OOB, stride, tiling) |
| M2 | shape-compatibility (extent disagreement) |
| M3 | hardware-constraint (descriptor, atom, alignment) |

**Target: ≥ 35 admissible mutants per `(class × surface)` cell, goal 40.**

`admissible = injected ∧ ¬noop ∧ oracle-confirmed corruption` (see §7). Because
admissible is always below injected, §6 gives per-surface injection budgets, not
target counts.

**Why 35.** A cell at n=12 moves 8.3 pp per mutant; a cell at n=2 moves 50 pp.
Neither can support the rate comparison the paper makes. v1's `N = 40` assumed a
one-to-one relation between injected and admissible; in practice the noop rule
takes 20–70% of the population, so `N` must be stated on **admissible** count.

## §1 M1 — element-access (12 specs)

v1 ids M1.1–M1.6 are preserved verbatim.

| ID | Spec | Corrupts |
|---|---|---|
| M1.1 | dropped boundary mask | ✔ |
| M1.2 | `p#n` off-by-one | ✔ |
| M1.3 | negative index | ✔ |
| M1.4 | transposed / non-contiguous stride | ✔ |
| M1.5 | offset-view overrun | ✔ |
| M1.6 | zero-stride / empty range | ✘ **noop by construction** — retained for the miss audit, excluded from the admissible denominator (see §6.1) |
| M1.7 | reversed loop bound (upper < lower) → overrun instead of empty | ✔ |
| M1.8 | stride scaling (stride × 2) | ✔ |
| M1.9 | base offset applied **without** shrinking the extent | ✔ |
| M1.10 | tile-boundary rounding (floor vs ceil on the last tile) | ✔ |
| M1.11 | read-after-write aliasing overlap | ✔ |
| M1.12 | wrong loop variable used for a dimension (broadcast index reuse) | ✔ |

**M1.6 is reclassified.** `mlir-shared/README.md` already records 8 of `mlir-low`'s
10 misses as `M1.6` — "zero-stride, no OOB by construction". A spec that cannot
corrupt cannot produce an admissible mutant, so including it in the denominator
is a category error. Keep it in the corpus (the *non*-detection is the point), but
exclude it from the admissible N and report the non-detection separately.

**M1.7 is the complement of M1.6.** M1.6 tests an empty range that is *correctly*
empty; M1.7 tests a range that *should* be empty but overruns. Together they
distinguish "the guard was right" from "the guard was right by accident".

## §2 M2 — shape-compatibility (9 specs)

v1 ids M2.1–M2.5 preserved verbatim.

| ID | Spec | Corrupts |
|---|---|---|
| M2.1 | wrong leading extent for `scale` / secondary operand | ✔ |
| M2.2 | DMA src/dst extent disagreement | ✔ |
| M2.3 | binary op mismatched shapes | ✔ |
| M2.4 | wrong output leading extent | ✔ |
| M2.5 | partial write (omitted tail tile) / duplicate write (overlapping tile) | ✔ |
| M2.6 | two extents transposed | ✔ |
| M2.7 | reduced-rank view (a dimension dropped) | ✔ |
| M2.8 | broadcast extent set to 1 instead of N | ✔ |
| M2.9 | batch/group dimension swapped | ✔ |

## §3 M3 — hardware-constraint (6 specs, redesigned)

### §3.0 Why v1's M3 failed

v1's M3 had three specs — atom-divisibility, base-address alignment, shared-memory
tile size — and a **70% noop rate** (`\sys`: 40 generated → 28 discarded → 12
admissible). The cause is structural:

> **v1's M3 specs are resource-oriented; the §7 oracle is value-oriented.**

- `atom{N}` / `kplus{N}` (v1 s1) **do** corrupt values — the dot-product extent is
  truncated. This is the one working spec.
- `lhs1` / `rhs2` / `w1` / `i1` (v1 s2) **do not** corrupt values — shifting a base
  address by one element still reads valid, in-bounds memory in a scalar reference
  kernel.
- `local8` / `shared32/64/128` (v1 s3) **do not** corrupt values — oversizing an
  allocation changes resource usage, not results.

Two of three specs are invisible to the oracle. Separately, none of the three
targets the obligation families the compiler emits most (§3.1).

### §3.1 The obligation families the compiler actually emits

Extracted from the SM_86 corpus ledgers (`e2e/results/*/*.ledger.json`):

| Family | Obligation (truncated) | Occurrences |
|---|---|---|
| **A** | `the {1st,2nd,3rd} dim N < 16777216` | ~190 |
| **B** | `CeilTo128Byte({src,dst}_dim0_size * bpe) < 2^24` | 14 |
| **C** | `CeilTo128Byte(bpe * dst dim0) * dim1 * ... * dim4 < 4GB` | 7 |
| D | atom divisibility / alignment (v1 s1/s2 territory) | — |

M3 must target **A, B, and C**, which no v1 spec does.

### §3.2 Screening rule

A spec enters M3 only if it satisfies **both**:

1. **Constraint-real** — it violates an obligation the compiler actually emits.
2. **Value-observable** — with the check absent, the computed output differs.

Resource-only violations fail (2) and are moved to §4.

### §3.3 The six specs

| ID | Spec | Family | Mechanism if unchecked | Corrupts |
|---|---|---|---|---|
| M3.1 | contraction extent not divisible by the tensor-core atom (tail dropped) | D | tail K-elements silently dropped | ✔ (v1 s1, retained) |
| M3.2 | descriptor dimension ≥ 2^24 | A | descriptor index wraps → wrong region read | ✔ |
| M3.3 | TMA box byte-size ≥ 2^24 after 128-byte ceiling | B | box wraps → tail tile unwritten | ✔ |
| M3.4 | tensor footprint ≥ 4 GB (5-D product) | C | descriptor truncated → wrong slice | ✔ |
| M3.5 | swizzle-incompatible box shape | B | read arrives mis-ordered | ✔ |
| M3.6 | leading dimension not aligned to descriptor granularity, **on a vectorized access** | D | vector load straddles a descriptor boundary | ✔ |

**Implementation note for M3.2–M3.4 — do not allocate.** A descriptor bound is
violated by a **stride**, not a length. Use a view whose stride is 2^24 (family A),
whose 128-byte-ceiled row extent is ≥ 2^24 (family B), or whose 5-D extent product
is ≥ 4 GB while each individual extent stays small (family C). This keeps every
mutant within the existing run budget.

**M3.6 replaces v1 s2.** v1's `lhs1` was a noop because the reference `k_matmul`
is scalar — an unaligned scalar load is legal, so nothing observable changes. The
spec must be expressed against a **vectorized** access (TCLE `leaptr<__vector
float, 1>` in `matmul/common.hpp`) where alignment is load-bearing. Same intent,
now value-observable.

## §4 Launch-status class (reclassified out of M3)

Resource-exhaustion mutants do **not** corrupt values — they prevent launch. They
cannot use the value oracle and must not enter M3's admissible N.

| ID | Spec | Observed as |
|---|---|---|
| L1 | shared-memory tile exceeds device limit | launch rejected |
| L2 | thread-block / cluster extent exceeds device limit | launch rejected |

Moved here from v1 M3 s3 (`local8`, `shared32/64/128`). Report as *rejected at
entry* — a distinct outcome in `tab:rq2-bugs`, and the same population that E5a
currently mishandles (see D3 in
`svn/eurosys27/plan/data-integrity-blockers.md`).

## §5 Per-surface translation

Not every surface expresses every class. `n/a` is a finding, not a gap.

| Surface | M1 | M2 | M3 | Notes |
|---|---|---|---|---|
| `\sys` (choreo) | ✔ 12 | ✔ 9 | ✔ 6 | all three |
| Triton | ✔ 12 | n/a | ✔ 6 | |
| TileLang | — | — | — | **drop** (see supplement §6) |
| IREE | n/a | ✔ 9 | n/a | |
| MLIR-linalg | n/a | ✔ 9 | n/a | |
| MLIR-low | ✔ 12 | n/a | n/a | |

**M3 categories.** M3 currently uses only `{matmul, conv2d}`. Expand to the four
categories with MMA/TMA/resource-bound constructs and a value-observable
expression of §3.3:

```
matmul, conv2d, batch_norm, max_pool2d
```

`embedding` is a gather (no MMA/TMA) and `transpose` is pure data movement, so
neither can express families A–C. `relu`, `gelu`, `sigmoid`, `elemwise_add` are
elementwise and have no descriptor obligations at all.

**M3 × MLIR-low = n/a** remains a measured, owner-approved `n/a` (v1 §4.1).

## §6 Injection budgets

Budgets assume the **pre-fix** manifestation rate, so they are conservative.

| Surface | Class | Specs | Categories | Shapes | Injections | Expected admissible |
|---|---|---|---|---|---|---|
| choreo | M1 | 12 | 4 | 2 | 96 | ~72 |
| choreo | M2 | 9 | 3 | 2 | 54 | ~40 |
| choreo | M3 | 6 | 4 | 2 | 48 | ~36 |
| triton | M1 | 12 | 4 | 2 | 96 | ~72 |
| triton | M3 | 6 | 4 | 2 | 48 | ~36 |
| iree | M2 | 9 | 3 | 2 | 54 | ~40 |
| mlir-linalg | M2 | 9 | 6 | 1 | 54 | ≥ 40 |
| mlir-low | M1 | 12 | 4 | 1 | 48 | ≥ 40 |

M1 category set: `{layer_norm, softmax, relu, transpose}` → extended to
`{layer_norm, softmax, relu, transpose, max_pool2d, conv2d}`.
M2 category set: `{layer_norm, matmul, concat}` → extended to
`{layer_norm, matmul, concat, elemwise_add, softmax, batch_norm}`.

### §6.1 Enumeration arithmetic

Two cells over-count against the naive specs × categories × shapes product; both
are intentional, and both must be recorded in `stats.json`:

- `choreo M1`: M1.6 is excluded from the admissible denominator (§1) → nominal 96
  injections yield ~72 admissible, not ~96.
- `mlir-linalg M2`: 9 specs × 6 categories = 54 injections; M2.7 (reduced-rank) and
  M2.6 (extent transpose) are degenerate on `layer_norm` and `softmax` (rank-2 with
  no tile decomposition) → 6 injections drop to `n/a` → **48 injectable**.

## §7 Outcome taxonomy and oracle (unchanged)

Per mutant: `{compile, runtime, never, n/a}`, plus `noop` for a mutant the oracle
finds value-identical to the reference.

**The oracle is mandatory and value-based.** A mutant is admissible only if the
three-checksum oracle (see `mlir-shared/README.md`; `_W_MOD = 127`,
`_TOL_REL = 1e-5`, `_TOL_DRIFT = 0.5`) confirms the output differs. Where a lane
has no checksum oracle, a byte-exact comparison against the reference output is
required.

The `manifest ∈ {corrupts, noop}` field is recorded per mutant. `noop` mutants are
**retained in the corpus and in `stats.json`**, and **excluded** from the
admissible denominator.

**Why the oracle must stay value-based.** Relaxing it to admit resource-only
violations would make M3's numbers rise while making the paper's claim weaker —
the ledger's job is to catch *wrong answers*, so a mutant that yields the right
answer is not evidence of anything. §4 is the correct destination for those
mutants, not a relaxed oracle.

## §8 Reporting

Every `stats.json` must record:

```
spec_version       : "v2.0"
spec_ids_used      : e.g. ["M1.1", ..., "M1.12"]
n_injected         : total generated
n_discarded_noop   : oracle-value-identical
n_admissible       : n_injected - n_discarded_noop - n_na
n_na               : class not expressible on this surface
```

The paper's denominator is `n_admissible`. Any table cell with `n_admissible < 35`
must be marked under-powered, not printed as a rate.

## §9 Supersession

- v1 `specs/mutation-specs.md` — frozen. All existing `stats.json` were produced
  under it. Do not edit; it is the audit trail for the ASPLOS-era results.
- v2.0 (this file) — governs new generation.
- A future v3 must state which cells it invalidates.
