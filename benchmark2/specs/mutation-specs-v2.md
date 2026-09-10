# Mutation specs v2 — expanded coverage

**Version:** 2.1 (2026-09-10).
**Supersedes:** `mutation-specs.md` v1 for *new* generation only. v1 is frozen and
remains the spec under which all results up to and including the ASPLOS-era
`stats.json` were produced.
**Spec id to record in `stats.json`:** `spec_version = "v2.1"`.

v2.1 is **additive to v2.0**: every v2.0 spec id keeps its meaning, so existing
v2.0 rows stay valid and no cell is invalidated. v2.1 adds M1.13–M1.18,
M2.10–M2.14, M3.7–M3.12, L3–L10, the new class **M4**, and the
**runtime-check-level curve** (§9). Rationale and evidence:
`svn/eurosys27/plan/mutation-redesign.md`,
`svn/eurosys27/plan/hardware-constraint-inventory.md`.

Every generated mutant's `stats.json` **must** carry `spec_version`. A table that
mixes v1 and v2 rows is otherwise undetectable.

Related: `svn/eurosys27/plan/mutation-supplement-plan.md` (rationale),
`svn/eurosys27/plan/acceptable-e2e-test.md` (acceptance condition C1).

---

## §0 Scope and target

Mutation classes. Out of scope, named: **concurrency safety** and **numeric
correctness** (unchanged from v1).

| Class | Meaning | Compiler `UsageType` | Corpus obligations | Runtime @ `entry` |
|---|---|---|---:|---:|
| M1 | element-access (OOB, stride, tiling) | `ElementAccess` | 974 | **1 / 290** |
| M2 | shape-compatibility (extent disagreement) | `ShapeCompatibility` | 16 | 0 / 2 |
| M3 | hardware-constraint (descriptor, atom, alignment) | `HardwareConstraint` | 287 | 22 / 22 |
| **M4** | **iteration-validity (zero/negative bounds)** | **`LoopBound`** | 19 | **11 / 11** |

The four classes are **exactly** the compiler's four `UsageType`s
(`lib/assess.hpp:26-34`). v2.0 covered three of them; M4 (§9.1) closes the gap and
serves as the experiment's control. Counts are from the 25 ledgers in
`svn/eurosys27/e2e/results/*/`; see `mutation-redesign.md` §2.

**Target: ≥ 35 admissible mutants per `(class × surface)` cell, goal 40.**

`admissible = injected ∧ ¬noop ∧ oracle-confirmed corruption` (see §7). Because
admissible is always below injected, §6 gives per-surface injection budgets, not
target counts.

**Why 35.** A cell at n=12 moves 8.3 pp per mutant; a cell at n=2 moves 50 pp.
Neither can support the rate comparison the paper makes. v1's `N = 40` assumed a
one-to-one relation between injected and admissible; in practice the noop rule
takes 20–70% of the population, so `N` must be stated on **admissible** count.

## §1 M1 — element-access (18 specs)

v1 ids M1.1–M1.6 are preserved verbatim; v2.0 ids M1.7–M1.12 are unchanged.

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

*v2.1 additions — target the obligation families M1.1–M1.12 do not reach.*

| ID | Spec | Corrupts |
|---|---|---|
| M1.13 | symbolic-bound overrun (index beyond a bound expressed over a runtime parameter, e.g. `seq_len/k`) | ✔ |
| M1.14 | `chunkat` tile-coordinate over/underflow while the element index stays in bounds | ✔ |
| M1.15 | `dimof` index ≥ rank, with a non-constant index | ✔ |
| M1.16 | `select` factor out of range (`f >= count` or `f < 0`) | ✔ |
| M1.17 | 5th-index access on a rank-5 view | ✔ |
| M1.18 | index in range by the `interval` mechanism but out of range by the `canonical` one (or vice versa) | ✔ |

### §1.1 Why the additions

M1.1–M1.12 are *semantic* mutations against **constant** bounds — the population
that resolves to `static-true` (684 of 974 `ElementAccess` obligations). The
runtime population is elsewhere:

| mechanism | corpus | runtime | enabled @ `entry` | site |
|---|---:|---:|---:|---|
| subscript index (`of element access`) | 655 | 18 | 1 | `semacheck.cpp:610-637`, `:1614-1618` |
| tile coordinate (`chunkat`) | 302 | **272** | **0** | chunkat lowering |
| `dimof` rank | **0** | **0** | 0 | `shapeinfer.cpp:2663-2684` — static only |
| `select` factor | **0** | **0** | 0 | `semacheck.cpp:2032-2038` — never exercised |

**M1.14 is the load-bearing addition:** it is the only spec in the suite that
directly targets a check the compiler has written and then disabled. **M1.15 and
M1.17 are gap specs** — a miss is the finding, recorded as `C4_NOT_ASSESSED`, not
as a cost-filter suppression. Evidence: `mutation-redesign.md` §3.

**M1.6 is reclassified.** `mlir-shared/README.md` already records 8 of `mlir-low`'s
10 misses as `M1.6` — "zero-stride, no OOB by construction". A spec that cannot
corrupt cannot produce an admissible mutant, so including it in the denominator
is a category error. Keep it in the corpus (the *non*-detection is the point), but
exclude it from the admissible N and report the non-detection separately.

**M1.7 is the complement of M1.6.** M1.6 tests an empty range that is *correctly*
empty; M1.7 tests a range that *should* be empty but overruns. Together they
distinguish "the guard was right" from "the guard was right by accident".

## §2 M2 — shape-compatibility (14 specs)

v1 ids M2.1–M2.5 preserved verbatim; v2.0 ids M2.6–M2.9 unchanged.

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

*v2.1 additions — target the extent-vs-layout blind spot (§2.1).*

| ID | Spec | Corrupts |
|---|---|---|
| M2.10 | transpose permutation mutation on a **square** operand (extents stay equal, memory order changes) | ✔ |
| M2.11 | DMA to-buffer element-count undersize on an otherwise `LogicalEqual` path | ✔ |
| M2.12 | rank mismatch through `.pad` (overlap still satisfies `f + pad == t`) | ✔ |
| M2.13 | shape-equal / layout-unequal (every extent agrees, the affine map does not) | ✔ |
| M2.14 | matmul contraction-dim (K) mismatch masked by broadcast | ✔ |

### §2.1 Why the additions — extents are compared, layouts are not

`semacheck.cpp:1073` compares `f_shape.ValueAt(dim_values[i])` against
`t_shape.ValueAt(i)`: **extents only**. The comparison is permutation-aware via
`dim_values`, but extent-only in every case. Two consequences M2.1–M2.9 do not
exploit:

- **A square transpose is decision-invisible.** `transpose: dims{0,1}` and
  `dims{1,0}` are indistinguishable when both extents are equal — the check
  passes, the layout is wrong, the values differ (**M2.10**).
- **Shape-equal / stride-unequal is decision-invisible.** Any mutation preserving
  every extent and changing only the affine map passes `ShapeCompatibility` and
  corrupts the result (**M2.13**).

M2's `never` population is therefore not "the checker is incomplete" but "the
contract is under-specified". M2 is also **compile-time dominated** — 14 of 16
obligations in the corpus are `static-true`, and the `.pad` rank check is a hard
`Error1` (`semacheck.cpp:1081-1090`) that produces **no ledger row at all**
(M2.12). The `ShapeCompatibility` runtime population is unannotated — see §9.3.

Evidence: `mutation-redesign.md` §4.

## §3 M3 — hardware-constraint (12 specs, redesigned)

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

### §3.4 Descriptor-encoding and resource bounds

§3.3 covers **descriptor-value** bounds only. An audit of the compiler's full
assessable set (`lib/Target/GPU/gpu_adapt.hpp`, `gpu_target.hpp`,
`cute_codegen.cpp`) plus vendor limits across CC 1.x → 12.x finds four further
kinds of hardware constraint: **descriptor encoding** (alignment, swizzle, pad
fields, rank), **launch geometry**, **per-SM resources**, and **feature gating**.
Six additional value-observable specs (**M3.7–M3.12**, adopted in §3.5) and eight
launch/feature specs (**L3–L10**, adopted in §4) come from
`svn/eurosys27/plan/hardware-constraint-inventory.md`, together with three source
defects found during the audit.

### §3.5 The six additions (M3.7–M3.12)

Adopted into v2.1 from `hardware-constraint-inventory.md` §5.3.

| ID | Spec | Kind | Corrupts |
|---|---|---|---|
| M3.7 | TMA inner-box geometry not 128-bit aligned | K2 encoding | ✔ |
| M3.8 | DMA rank = 6 (outside the assessed `[1,5]`) | K2 encoding | ✔ |
| M3.9 | pad-field overrun (`dma.pad` / `padding_mid` beyond assessed range) | K2 encoding | ✔ |
| M3.10 | last-dim / rank-5 mid-padding violates `padding_mid[rank-1] == 0` | K2 encoding | ✔ |
| M3.11 | shared operand base not 128-byte aligned **on `sm_90`+** (a noop on `sm_86`) | K2 alignment | ✔ |
| M3.12 | shared tile exactly at the capacity bound (±1 KiB edge) | K4 bound edge | ✔ |

Cost rule extension (from `hardware-constraint-inventory.md` §5.4): violate a
bound by **stride, not length; do not allocate**. M3.12 is the sole exception —
it shrinks an allocation, and must never grow one.

**M3.11 is the only spec whose manifestation is architecture-dependent** — the
same mutation is a noop on `sm_86` (`GetMemAlignmentByte` SHARED = 16) and a
mis-read on `sm_90`+ (SHARED = 128). It therefore requires the harness to select
`-arch` per mutant; see §9.4 q3.

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

The class is **not closed at L1/L2**. Adopted into v2.1 from
`hardware-constraint-inventory.md` §5.5:

| ID | Spec | Observed as | Channel |
|---|---|---|---|
| L3 | `__launch_bounds__` understated vs actual block size | launch rejected | runtime |
| L4 | block extent not a multiple of 32 / of 128 | launch rejected | assessor |
| L5 | shared tile exceeds per-SM capacity | launch rejected | runtime |
| L6 | WGMMA used below `sm_90` | launch rejected | assessor |
| L7 | shared operand used outside WGMMA | launch rejected | assessor |
| L8 | unsupported MMA configuration | launch rejected | assessor |
| L9 | `mma.scale` operand is not an accumulator | launch rejected | assessor |
| L10 | cluster extent > 8 (non-portable) | launch rejected | runtime |

L3–L10 are **attribution-only**: they never enter an admissible denominator. Their
value is that they grow the `never`-attribution table (proving the thesis that a
miss is usually a *rejected launch*, not a wrong answer), and they are reported as
*rejected at entry* alongside L1/L2.

The class is split into **assessor-visible** rejects (L4–L9) and **runtime**
rejects (L3, L5, L10).

## §5 Per-surface translation

Not every surface expresses every class. `n/a` is a finding, not a gap.

| Surface | M1 | M2 | M3 | M4 | Notes |
|---|---|---|---|---|---|
| `\sys` (choreo) | ✔ 18 | ✔ 14 | ✔ 12 | ✔ 5 | all four classes |
| Triton | ✔ 18 | n/a | ✔ 12 | n/a | |
| TileLang | — | — | — | — | **drop** (see supplement §6) |
| IREE | n/a | ✔ 14 | n/a | ✔ 5 | |
| MLIR-linalg | n/a | ✔ 14 | n/a | ✔ 5 | |
| MLIR-low | ✔ 18 | n/a | n/a | n/a | |

**M4 on non-choreo surfaces (marked ✔ provisionally).** `LoopBound` obligations
are emitted by choreo's own checkers. On IREE / MLIR-linalg the same mutation is
expressible (a zero or negative dimension) but is *not* backed by a
`loop-bound`-tagged obligation, so the attribution differs. Confirm before
reporting; if the obligation is absent, M4 is a **host-only control** and the
other surfaces must show `n/a` (see §9.4 q1).

**M3 categories.** M3 uses `{matmul, conv2d}` → expanded to the four categories
with MMA/TMA/resource-bound constructs:

```
matmul, conv2d, batch_norm, max_pool2d
```

`embedding` is a gather (no MMA/TMA) and `transpose` is pure data movement, so
neither can express families A–C. `relu`, `gelu`, `sigmoid`, `elemwise_add` are
elementwise and have no descriptor obligations at all.

**M3 × MLIR-low = n/a** remains a measured, owner-approved `n/a` (v1 §4.1).

## §6 Injection budgets

Budgets assume the **pre-fix** manifestation rate, so they are conservative.
v2.1 rows are marked **＋**.

| Surface | Class | Specs | Categories | Shapes | Injections | Expected admissible |
|---|---|---|---|---|---|---|
| choreo | M1 | **18** | 6 | 2 | **132** | ~108 |
| choreo | M2 | **14** | 6 | 2 | **112** | ~60 |
| choreo | M3 | **12** | 4 | 2 | **96** | ~66 |
| choreo | **M4** | **5** | 4 | 2 | **40** | ~32 |
| triton | M1 | **18** | 6 | 2 | **132** | ~108 |
| triton | M3 | **12** | 4 | 2 | **96** | ~66 |
| iree | M2 | **14** | 6 | 2 | **112** | ~60 |
| iree | M4 | **5** | 4 | 2 | **40** | ~32 |
| mlir-linalg | M2 | **14** | 6 | 1 | **84** | ≥ 60 |
| mlir-linalg | M4 | **5** | 4 | 1 | **20** | ~16 |
| mlir-low | M1 | **18** | 6 | 1 | **108** | ≥ 60 |

M1 category set: `{layer_norm, softmax, relu, transpose}` → extended to
`{layer_norm, softmax, relu, transpose, max_pool2d, conv2d}`, plus a **`select`**
and a **rank-5** category to host M1.16 / M1.17 (`mutation-redesign.md` §9 q4).
M2 category set: `{layer_norm, matmul, concat}` → extended to
`{layer_norm, matmul, concat, elemwise_add, softmax, batch_norm}`.
M4 category set: `{layer_norm, softmax, matmul, ele_add}` — every category with a
tile loop.

**The `-rtc` curve multiplies the M1 row (§9.2).** M1's admissible count at
`-rtc=high` rises toward the 272 suppressed tile-coordinate obligations, which is
what makes the ≥35-per-cell target reachable on element-access cells without
inventing further specs. Budget the curve on the **reduced** grid (1 shape):
4 levels × 8 categories = 32 runs per surface per class.

### §6.1 Enumeration arithmetic

Cells that over-count against the naive specs × categories × shapes product; all
intentional, and all must be recorded in `stats.json`:

- `choreo M1`: M1.6 is excluded from the admissible denominator (§1) → nominal 108
  (18 specs × 6 categories) yield ~72–108 admissible, not a one-to-one product.
- `mlir-linalg M2`: M2.7 (reduced-rank) and M2.6 (extent transpose) are degenerate
  on `layer_norm` and `softmax` (rank-2 with no tile decomposition) → 14 × 6 = 84
  drops to **~72** injectable.
- `MLIR-low M1`: M1.16 (`select`) and M1.17 (rank-5) are not expressible →
  18 × 6 = 108 drops to **~84**.

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
spec_version       : "v2.1"
spec_ids_used      : e.g. ["M1.1", ..., "M1.18"]
n_injected         : total generated
n_discarded_noop   : oracle-value-identical
n_admissible       : n_injected - n_discarded_noop - n_na
n_na               : class not expressible on this surface
```

v2.1 adds four fields so that a miss can be attributed rather than merely counted
(§9.2, `mutation-redesign.md` §7):

```
rtc_level              : "entry" | "low" | "medium" | "high"
n_obligations_runtime  : from the ledger, per usage type
n_obligations_enabled  : from the ledger, per usage type
detection_channel      : pipeline stage that first rejects the mutant
                         (compile | assessor | runtime | never)
```

The paper's denominator is `n_admissible`. Any table cell with `n_admissible < 35`
must be marked under-powered, not printed as a rate.

## §9 v2.1 additions

### §9.1 M4 — iteration-validity (new class, 5 specs)

`UsageType::LoopBound` is the compiler's fourth safety category
(`lib/assess.hpp:31`) and was covered by **no** mutation class in v2.0. It is
emitted in two places, both forced to `ENTRY` cost so that the cost filter cannot
suppress them:

| Mechanism | Site | Corpus | Runtime enabled |
|---|---|---:|---:|
| with-in span validity (`zero is detected for the Nth dim of the mdspan …`) | `semacheck.cpp:907` | 19 | 11 / 11 |
| `parallelby` bound validity (`… should be greater than 0`) | `semacheck.cpp:721` | 0 | — |

| ID | Spec | Expected |
|---|---|---|
| M4.1 | `with-in` mdspan dim mutated to 0 | caught — control |
| M4.2 | `parallelby` bound mutated to 0 or negative | caught — control |
| M4.3 | `parallelby` bound symbolic and zero only at runtime | caught (entry-cost, hoisted) |
| M4.4 | bound > 0 but the iteration space is empty (zero-trip loop) | **noop by construction** — retained, excluded from the denominator |
| M4.5 | stride/step = 0 in an iteration | caught or noop |

**M4 is the experiment's control.** Every other class has an alternative
explanation for a miss — M1 the cost filter, M2 the extent-only contract, M3 the
assessable set. `LoopBound` has none: the obligation is present, emitted, and
free. A missed M4 mutant is therefore a genuine unsoundness, and a caught one
calibrates the harness and the oracle. Without M4 a low M1 detection rate cannot
be distinguished from a broken harness.

M4.4 is a deliberate **noop control**: if it is ever counted as admissible, the
oracle has regressed.

### §9.2 The runtime-check-level curve

`enabled` is a pure function of cost: `enabled = cost ≤ -rtc threshold`, with
`NONE < ENTRY < LOW < MEDIUM < HIGH` (`assert_site.cpp:18-20`; cost buckets at
`:123-132`; default threshold `ENTRY` at `context.hpp:549`). Rerunning the *same*
corpus at each threshold enumerates the detection/coverage curve **from
obligations the compiler already emits** — no compiler change:

| `-rtc` | threshold | obligations enabled | Δ vs entry |
|---|---:|---:|---:|
| `none` | 0 | 0 | −34 |
| `entry` *(default)* | 1 | **34** | — |
| `low` | 2 | **47** | +13 |
| `medium` | 3 | **79** | +45 |
| `high` / `all` | 4 | **323** | **+289 (9.5×)** |

Generate each cell **once**; run the grid at `entry`, `low`, `medium`, `high`.
Predicted curve by class:

| class | entry → high enabled | predicted curve |
|---|---|---|
| M1 | 1 → 290 | **steep** — the cost filter bites here |
| M2 | 0 → 0 | **flat** — compile-time dominated |
| M3 | 22 → 22 | **flat** — already entry-cost |
| M4 | 11 → 11 | **flat** — control |

A steep M1 curve against flat M2/M3/M4 curves *is* the paper's thesis, measured
rather than asserted: **the runtime-check cost filter, not the absence of checks,
is the dominant cause of missed value corruption, and its cost is class-specific.**

**Reporting rule.** Headline numbers are reported at **`-rtc=entry`** — the
shipping default, so the claim is about the released tool, not a tuned
configuration. The `-rtc=high` run is the **ceiling**: the detection rate
available if cost were not a constraint. Any cell whose `entry → high` delta is
negative is a bug.

### §9.3 Defect that must be fixed before M2 numbers are reported

Both runtime `ShapeCompatibility` records in the corpus are unannotated:

```json
{"function":"ele_add","loc":":1.1",
 "message":"The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.",
 "outcome":"runtime","usage":"shape-compat","dependence":"scalar-symbolic","mechanism":"canonical"}
```

Three problems in one record:

1. **No `cost` / `enabled`.** Every other runtime obligation carries them. `cost`
   is assigned unconditionally in `EstimateAssertions()`
   (`assert_site.cpp:450-453`), so its absence means the assertion is **not in
   `assessor.GetAssertions()`** — the entry was logged with
   `assertion_idx = SIZE_MAX`. The ledger therefore cannot say whether it was
   emitted, so **any attribution of an M2 miss to "suppressed" or "emitted" is
   currently unsupported.**
2. **`loc` is `":1.1"`** — empty file part; the obligation cannot be traced to a
   source line.
3. **`(dim: 1)` vs `(dim: 1)` reported as inconsistent.** With
   `dependence: scalar-symbolic` the check is conservative — it emits `runtime`
   for a pair it cannot *prove* equal. That is a warn-shaped check recorded as an
   error-shaped obligation, and it will fire on correct programs.

Until fixed, exclude `scalar-symbolic` shape-compat obligations from the
attribution table and say so.

**Cosmetic but confirmed:** `lib/context.hpp:455` annotates
`unclassified_total` with `// UsageType::ShapeCompatibility` — the wrong enum
value (copy-pasted from the next line). Harmless at runtime; fix it, because it is
the kind of thing that makes a reviewer distrust the whole stats block.

### §9.4 Open questions

1. **Adopt M4?** It changes the paper's structure from three classes to four. For:
   the taxonomy is the compiler's own, the control is scientifically necessary, and
   it is nearly free (entry-cost, already emitted). Against: three classes is a
   cleaner story. *Recommendation: adopt, and present it explicitly as the
   control.*
2. **Run the `-rtc` curve?** *Recommendation: yes — 4× the reduced grid, the only
   way to turn the `never` column into the thesis.*
3. **Does the harness express mutations at the `chunkat` tile-coordinate level?**
   M1.14 depends on it; if the harness only mutates element indices, M1.14 is not
   implementable and the cost filter becomes unmeasurable. **This is the blocking
   question.**
4. **Which category hosts M1.15 (`dimof`) and M1.17 (rank-5)?** If none, both are
   `n/a` on every surface and should be dropped rather than reported.
5. **Confirm M2's contract statement.** M2.10/M2.13 assert that the checker
   constrains extents, not layouts. That is a claim about the tool's
   specification, not its implementation, and needs owner confirmation before it
   appears in the paper.
6. **Can the harness select `-arch` per mutant?** M3.11 needs it (§3.5).

## §10 Supersession

- v1 `specs/mutation-specs.md` — frozen. All existing `stats.json` were produced
  under it. Do not edit; it is the audit trail for the ASPLOS-era results.
- v2.0 `specs/mutation-specs-v2.md` (prior revision of this file) — governs any
  results already produced under it. v2.1 is **additive**: no v2.0 spec id changed
  meaning and no v2.0 cell is invalidated. Cells generated under v2.0 remain valid
  and are distinguished by `spec_version`.
- v2.1 (this file) — governs new generation.
- A future v3 must state which cells it invalidates.
