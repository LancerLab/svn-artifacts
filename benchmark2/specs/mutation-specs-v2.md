# Mutation specs v2 — expanded coverage

**Version:** 2.1 (2026-09-10).
**Supersedes:** `mutation-specs.md` v1 for *new* generation only. v1 is frozen and
remains the spec under which all results up to and including the ASPLOS-era
`stats.json` were produced.
**Spec id to record in `stats.json`:** `spec_version = "v2.1"`.

v2.1 is **additive to v2.0**: every v2.0 spec id keeps its meaning, so existing
v2.0 rows stay valid and no cell is invalidated. v2.1 adds M1.13–M1.18,
M2.10–M2.14, M3.7–M3.12, L3–L10, the new class **M4**, the
**runtime-check-level curve** (§9.2), and the **coverage audit against the
empirical bug taxonomies** (§9.5). Rationale and evidence:
`svn/eurosys27/plan/mutation-redesign.md`,
`svn/eurosys27/plan/hardware-constraint-inventory.md`,
`svn/eurosys27/plan/real-world-bug-coverage.md`,
`svn/eurosys27/plan/mutation-reachability-audit.md`.

§9.4 records two resolutions: **M4 is adopted** and **tile-coordinate mutation is
expressible, so M1.14 is unblocked**. §9.5.0 states the **applicability rule** — a
mutation counts as a test only if the model can be made to hold the corrupted state
(expressible, independent, survives repair, attributable); otherwise it is **not
applicable**, with the model prohibition named and cited. §9.5.1–§9.5.3 apply it to
the three gap specs from the coverage audit: **M1.19 adopted** (realization (b)
narrow-carrier; already a latent defect), **M3.13 adopted but narrowed** (the
descriptor-rank pair is *derived* → N/A; only the swizzle/box/alignment triple, and
only states surviving the repair at `:10271`), and **M5 not applicable** (prohibition:
*absent feature* — choreo has no reuse mechanism). §9.5.4 adds a **fourth miss
mechanism — unrepresentable** — with M2.10/M2.13 as its exemplars. The reporting
denominator is **applicable / not-applicable / out-of-scope**, and N/A mutants are
never counted as missed. v2.1 is final: the owner's condition is answered, and all
four open decisions are resolved.

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

### §9.4 Resolutions and remaining questions

**Resolved (owner decisions):**

1. ~~**Adopt M4?**~~ **Adopted.** The class stands as the experiment's control, and
   is now independently attested: the empty-tensor / zero-dimension corner case is
   a named bug trigger in the torch.compile study
   (`real-world-bug-coverage.md` §2.2, §4.3). It was opened as a control question;
   external attestation makes it a coverage claim as well.
2. ~~**Does the harness express mutations at the `chunkat` tile-coordinate
   level?**~~ **Yes — answered.** Tile-coordinate mutation is expressible, so
   **M1.14 is unblocked** and the cost filter is measurable. The 272 suppressed
   `chunkat` obligations (§1.1) are reachable at `-rtc≥low`.
3. ~~**Which category hosts M1.15 (`dimof`) and M1.17 (rank-5)?**~~ **Resolved by
   §6**: M1 gains a `select` category and a rank-5 category. Both specs are
   retained as gap specs (`C4_NOT_ASSESSED` on the surfaces that cannot express
   them).

**Still open:**

5. **Run the `-rtc` curve?** *Recommendation: yes — 4× the reduced grid, the only
   way to turn the `never` column into the thesis.*
6. **Confirm M2's contract statement.** M2.10/M2.13 assert that the checker
   constrains extents, not layouts. That is a claim about the tool's
   specification, not its implementation, and needs owner confirmation before it
   appears in the paper.
7. **Can the harness select `-arch` per mutant?** M3.11 needs it (§3.5).
8. ~~**Does a two-phase (compile-A, then request-B) oracle exist on any surface?**~~
   **Moot** — M5 is rejected because choreo has no reuse mechanism for a two-phase
   oracle to exercise (§9.5.3).

### §9.5 Coverage audit vs. empirical bug taxonomies

Full analysis: `svn/eurosys27/plan/real-world-bug-coverage.md`. Summary only here,
because it is a *completeness* argument, not a generation rule.

Two empirical taxonomies now exist for this exact domain and are the completeness
oracle for the class suite: **TileCodegenBug** (Rathnasuriya et al., ISSTA 2026;
301 confirmed tile-codegen bugs, six root causes) and **torch.compile silence**
(Li et al.; 116 confirmed silent-correctness bugs). Diffing the 15 shape- or
resource-shaped categories against M1–M4 and L1–L10:

| Outcome | Count | Categories |
|---|---:|---|
| already covered | 11 | branch predication · IR transformation tile-drop · launch config · thread-block mapping · stride/layout address bugs · offset views · boundary-mask/tail · broadcast-vs-extent · resource sizing · per-arch feature gating · MMA divisibility |
| **adopted** | **2** of 3 gaps | index-carrier overflow (**M1.19**) · descriptor-to-descriptor consistency (**M3.13**, narrowed to the swizzle/box/alignment triple) |
| **not applicable** | **1** | decision reuse (**M5**) — choreo has no reuse mechanism; prohibition: *absent feature* |
| out of scope (named) | 4 families | numeric (RC5.1/5.2/5.3, LCG) · concurrency/ordering (RC1.2, RC4.3) · IR determinism (RC2.1) · front-end tracing (GSC non-computational) |

The excluded families are **~38% of the tile-bug corpus by count and ~19% of the
torch.compile corpus**, all on the two axes v1 §0 already names (numeric
correctness, concurrency). Quote the weight; do not say "out of scope" bare.

### §9.5.0 Applicability rule — what "not applicable" means

A mutation may be counted as a test of the compiler **only if all four hold**:

1. **Expressible** — the corrupted state is constructible through the model's own
   inputs and interfaces.
2. **Independent** — the model does not derive the two sides of the corrupted
   relation from one source.
3. **Survives repair** — if the model detects and repairs the conflict, the injected
   state is one the repair does *not* catch.
4. **Attributable** — a miss would be a defect of the compiler, not of our harness.

Otherwise the mutation is **not applicable (N/A)**: we name the *model prohibition*
that forbids it, cite the source, and report it in the denominator
(**applicable / N/A / out-of-scope**) — never as a miss, and never as a hole in the
suite. N/A is a claim about the programming model, so it is evidence, not an excuse.

| Prohibition | Meaning | Instance |
|---|---|---|
| **absent** | the feature does not exist in the model | M5 — no cache / kernel hash / specialization key anywhere in choreo |
| **derived** | both sides come from one source, so a contradiction is unrepresentable | M3.13 rank pair — `tma_inner_splits_`, one writer (`:10291`) two readers (`:4715`/`:5018`) |
| **repaired** | the model detects and fixes the conflict | M3.13 swizzle overflow, explicitly split at `cute_codegen.cpp:10271` — *only the sub-case the repair misses is injectable* |
| **harness-owned** | the corruption would be ours, not the compiler's | M5 as originally framed |

The rule is the dual of the miss taxonomy: a model either **derives** an invariant
(nothing to mutate, nothing to check) or **assumes** it. Assumed-and-expressible is
where the suite operates; assumed-and-*in*expressible is where silent bugs live
(§9.5.4). Stating the rule pre-empts "why did you not test X?" — X is not a state of
this model.

**What this validates (do not re-derive):** M2.10/M2.13 are the best-attested
shape bug class in both corpora — tile RC4.1 (35 bugs) and torch.compile "Memory
Layout Conflicts" (14 bugs, split into layout-metadata tracking and missing
layout-compatibility checks) are both extents-equal / layout-unequal failures,
i.e. decision-invisible under `semacheck.cpp:1073`'s extent-only comparison. And
the **top triggering input axis in both studies is the view — non-contiguous /
transposed / offset — not the value**, which is exactly M1.4/M1.5/M2.10/M2.13.
Recommend reporting those four as a named **view-family** sub-table in the
results.

#### §9.5.1 M1.19 — index-carrier overflow — **ADOPTED, applicable**

Real-world: Triton #832 — a valid flattened index exceeding INT_MAX, computed in
signed int32, overflows to a negative offset → illegal memory access. Every
existing M1 spec mutates the index **value or bound**; none mutates the index
**carrier / arithmetic width**.

**Reachability (verified in choreo — see `mutation-reachability-audit.md` §1).** The
artifact exists and is not derived away:

| Carrier | Evidence |
|---|---|
| tile base offset truncated to `int` | `cute_codegen.cpp:1757` emits `(int)(<coord> * <extent>)`, summed in `int` |
| TMA coordinates | `int32_t coord0/coord1` (`runtime/choreo_cute.h:164`) |
| TMA box shape | `uint32_t` entries (`cute_codegen.cpp:10296`) |
| accessor index | `spanned_view::operator[](int)`, `ArrayProxy::operator[](int)` (`runtime/choreo.h:764`) |

No guard bounds a flat offset (`EmitHostRuntimeCheck` tests `shape()` equality
only). And dimensions above INT_MAX are **legal**: `__inf__ = 2³²−1`
(`runtime/choreo.h:222`).

**Realizations:** (a) large-shape — `bos*H*K ≥ 2³¹` (needs ≥ 4 GiB for fp16);
(b) narrow-carrier — a single dim in `(2³¹, 2³²)`, legal under `__inf__` but unindexable
by the `int` accessor. **Selected: (b).** It needs no large allocation, and because
`__inf__` explicitly permits the extent, the injected shape is a *legal program* — so a
surviving mutant is unambiguously a compiler defect. A miss is the finding.

**Separate finding.** The `(int)` cast at `:1757` is a **latent defect in the shipped
compiler**, independent of any injection. Report it as a found defect, not as a
mutation result.

#### §9.5.2 M3.13 — descriptor consistency — **ADOPTED, narrowed**

Real-world: Triton #2658 — `WSMaterialization` mutates `num-warps` without updating
tensor layouts, violating `∏(warpsPerCta) == num-warps`; no pass validates the
invariant.

**Reachability (verified — see `mutation-reachability-audit.md` §2).** Only one of the
three candidate pairs is admissible:

| Pair | Verdict |
|---|---|
| TMA descriptor rank ↔ device coordinate arity | **Non-target — derived.** `tma_inner_splits_` is written once (`cute_codegen.cpp:10291`) and read by *both* device sites (`:4715` load, `:5018` store). A contradiction is unrepresentable; choreo avoids this by construction. |
| descriptor stride ↔ actual buffer layout | Realizable, but this **is M2.13** — report there, do not double-count. |
| **swizzle width ↔ box inner dim ↔ shared alignment** | **Admissible.** Three independently derived quantities: `swiz_bytes` from `SwizMode` (`:10264`), box inner bytes from `t_shape` (`:10268`), `SharedAlignmentBytes` (`:276`). An explicit repair for the box-vs-swizzle conflict exists (`:10271`) — evidence the conflicting state is reachable. |

**Spec (narrowed):** mutate the swizzle-mode / box-shape / shared-alignment triple so
each value stays individually legal but the conjunction is unsatisfiable, **and the
injected state survives the repair at `:10271`** (i.e. it must not be the
`inner_bytes > swiz_bytes` case the compiler already splits). Record the rank pair as an
explicit **non-target / N/A** (prohibition: *derived*, §9.5.0) with the reason —
disclosing it pre-empts a reviewer asking why we did not mutate the descriptor rank.

#### §9.5.3 M5 — decision reuse — **NOT APPLICABLE (no mechanism)**

Real-world: torch.compile graph caching (guard omits input shapes →
a cached graph reused for a shape never proven for it) and tile RC4.3 (Warp #639 —
the kernel hash omitted `block_dim`).

**Reachability: NO — see `mutation-reachability-audit.md` §3.** There is no reuse
mechanism to corrupt: no cache, no kernel hash, no specialization key, no guard
anywhere in `lib/`, `tools/`, `runtime/`, `co2ir/` (every `specializ` match is a C++
template specialization), and the only index/cache-flavoured runtime, `runtime/catz`,
is **excluded from the build** (`scripts/oss/oss_exclude_paths.txt`). choreo compiles
per configuration and emits a fresh `runtime_check` block for each.

Consequently a "reuse" mutation fails **R2** (there is no independent second source to
contradict) and **R4** (the corrupted artifact would be **our harness**, not the compiler
under test) of the §9.5.0 rule. Verdict: **not applicable**, prohibition *absent feature*.
A miss would prove nothing about the obligation set. The two-phase oracle question
(§9.4 q8) is therefore **moot**.

**Salvage (record, do not promote to a class).** `EmitHostRuntimeCheck`
(`cute_codegen.cpp:9950`) guards **static** dims (`:9966`) and **unbounded** dims
(`:9974`), but for **symbolic** dims emits only the cross-parameter equality check
(`VISym` → `ve_entries_map`, `:9981`) — a symbolic dim's own value is never guarded.
That is a genuine *single-compilation* "not-invalidated" instance and needs no
two-phase oracle. Report it as a narrowed M5 candidate or as a named limitation.

#### §9.5.4 New miss mechanism: **unrepresentable**

Fell out of the reachability audit and is more consequential than the three proposals
above. Full write-up: `mutation-reachability-audit.md` §4 and
`real-world-bug-coverage.md` §8.1.

`spanned_view` (`runtime/choreo.h:764`) holds `T* ptr` and `mdspan<Rank> dims` — **no
strides** — with a **public** constructor. The generated host prologue guards extents
only (four `runtime_check(...shape()[i] == 256)`, zero stride checks) while the kernel
bakes the layout it assumed (`__choreo_tma_0_strides[] = {512}`). The ATen adapter
`make_spanview(const at::Tensor&)` does guard `is_contiguous()`
(`runtime/choreo.h:1061`), but the public constructor does not, and the adapter is
`#ifdef`-gated.

So `spanned_view<T,R>(ptr_into_a_strided_buffer, {256,256})` passes every check and
yields silently wrong results. Because the obligation **cannot be expressed** in the
input type, **no `-rtc` value reaches it**:

| Mechanism | Reachable by raising `-rtc`? |
|---|---|
| not emitted to runtime | no |
| cost-suppressed | **yes — this is the curve** |
| not invalidated | no |
| **unrepresentable** | **no — not fixable by any threshold** |

M2.10/M2.13 — already the best-attested class in both empirical corpora — are the
canonical exemplars. This is the direct answer to "why not just raise the threshold?"

#### §9.5.5 Generator requirement (not a spec)

Both studies name the same triggering axes (tile F6; torch.compile
"operator-induced layout transformations"). The shape sweep **must** include
**singleton (1)**, **prime**, and **tile_size ± 1** extents; the view sweep
**must** include **non-contiguous, transposed, and offset** inputs. Without the
view axis the view-family specs (M1.4/M1.5/M2.10/M2.13) — the best-attested
class in the literature — are untested in our own experiment.

## §10 Supersession

- v1 `specs/mutation-specs.md` — frozen. All existing `stats.json` were produced
  under it. Do not edit; it is the audit trail for the ASPLOS-era results.
- v2.0 `specs/mutation-specs-v2.md` (prior revision of this file) — governs any
  results already produced under it. v2.1 is **additive**: no v2.0 spec id changed
  meaning and no v2.0 cell is invalidated. Cells generated under v2.0 remain valid
  and are distinguished by `spec_version`.
- v2.1 (this file) — governs new generation. §9.5.0 (**applicability rule**) is
  normative for *reporting*: a mutation whose corrupted state the model forbids is
  reported as **not applicable** with the prohibition cited, and is never counted as a
  miss. §9.5.1–§9.5.3 record the verdicts that follow from it.
- A future v3 must state which cells it invalidates.
