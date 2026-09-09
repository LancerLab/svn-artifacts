# Mutation specs (M1 / M2 / M3) + per-surface translation

> Authoritative, shared by all workers. Generated 2026-09-08 (Phase 0).
> Mutations are specified **conceptually** (a defect spec), so the *same defect*
> is re-expressed in each toolchain's own idiom. Equal counts are fixed at spec
> level; a mutant a surface cannot express is marked `n/a`, never re-balanced.

## 0. Fixed decisions (owner-confirmed 2026-09-08)

- **3 classes mutated:** M1 element-access, M2 shape-compatibility, M3
  hardware-constraint.
- **Iteration-validity is folded into M1** (its observable failures — zero/empty
  iteration space, zero stride — inject through M1's stride/layout spec).
- **Equal count: N = 40 per class → 120 total.** *(The "160" this line carried
  until 2026-09-09 was stale: it dated from a 4-class layout and predates the
  fold of iteration-validity into M1 above. 3 classes × 40 = 120. Owner decision
  recorded 2026-09-09.)* See §5.1 for how a lane's actual count relates to the
  40-per-class target.
- **Out of scope (named, auditable):** concurrency safety (races, missing
  barriers) and numeric correctness (mixed-precision casts, NaN/inf/signed-zero,
  accumulation drift). These are the other silent-bug families in cited related
  work; excluded by the paper's §ledger scope statement, not overlooked.

## 1. M1 — element-access

*Paper bug categories folded in: `input-dependent OOB` (58) + `stride/layout
error` (1).*

Defect specs (each is one mutant family; reproduce the same defect in the
surface's own idiom):

1. dropped boundary mask (omit the `mask`/guard on a `p#n` / `at(i)` access)
2. `p#n` off-by-one
3. negative index
4. transposed / non-contiguous stride
5. offset view (base + offset that overruns the tensor)
6. zero-stride / empty range (the observable residue of iteration-validity)

**Choreo (reference) syntax:** `a.at(i)`, chunked access `p#n`.

## 2. M2 — shape-compatibility

*Paper bug categories folded in: `dimension-mismatch` (43) + `wrong-output-shape`
(1).*

Defect specs:

1. caller passes `scale` (or a secondary operand) of the wrong leading extent
2. DMA source/destination extent disagreement
3. binary op on mismatched shapes
4. output tensor declared with a wrong leading extent
5. partial write (omitted tail tile) / duplicate write (overlapping tile)

**Choreo (reference) syntax:** DMA `copy`, binary ops, `span_as(...)`.

## 3. M3 — hardware-constraint

*Paper §ledger: `target ops → alignment, atom divisibility`. No RQ2 bug category,
but in scope by the ledger table and documented empirically by
`rathnasuriya2026tilebugs` (device-specific WGMMA shape-divisibility +
resource-allocation alignment).*

Defect specs:

1. MMA contraction dim not divisible by the tensor-core atom (`K % 16 ≠ 0`)
2. misaligned tensor-core / DMA base address
3. shared-memory tile exceeding the device limit

**Choreo (reference) syntax:** `mma`/`tma` alignment, atom divisibility.

## 4. Per-surface translation table (pin before dispatch)

The `parallel p by 0` defect has **no portable Triton/MLIR analogue**; each spec
is translated to the surface's own mechanism for the same conceptual defect:

| Defect class | choreo (ref) | Triton | MLIR — linalg | MLIR — low level | IREE |
|---|---|---|---|---|---|
| M1 OOB | `a.at(i)` | omit `tl.load` mask | n/a (no memory) | `affine`/`memref` index out of range | entry shape only |
| M2 shape | DMA extent mismatch | n/a (no cross-tensor contract) | **rank/shape mismatch** (primary target) | n/a | entry shape check |
| M3 hw | `mma`/`tma` alignment, atom divisibility | partial (`tl.dot` K divisibility) | n/a | **n/a** *(measured; was "partial")* | n/a |

### 4.1 M3 × MLIR-low = `n/a` (measured 2026-09-09, owner-approved)

The cell above previously read "partial (`vector`/`affine` alignment)". All three
M3 defect specs were probed on **LLVM 21.1.0** and none is checked by MLIR's
verifier, `generate-runtime-verification` (RTV), or `canonicalize`/`cse`:

| Spec | Probe | Result |
|---|---|---|
| M3.1 atom divisibility | `vector.contract` with K=13 (`vector<16x13xf16> × vector<13x16xf16>`) | verifier **rc=0**, RTV **0 asserts**; K=16 behaves identically. `convert-vector-to-nvgpu` is **not a registered pass** in this build, so no lowering step enforces an atom shape. |
| M3.2 misaligned base | `memref<64xf32, 4>` + store/load | verifier rc=0; RTV emits asserts, but every message is `^ out-of-bounds access` — a *bounds* check; the string "alignment" appears nowhere in the lowered IR. The control **without** the alignment attribute yields the **identical assert count** ⇒ alignment contributes nothing. |
| M3.3 shared-mem over limit | 800 KB workgroup allocation (H800 max ≈ 228 KB) | unreachable via MLIR: `memref.alloc` + `#gpu.address_space<workgroup>` lowers to **`llvm.call @malloc`** (never `addrspace(3)`); `gpu.alloc` is **"explicitly marked illegal"** under `convert-gpu-to-nvvm`; a hand-written `llvm.mlir.global {addr_space = 3}` is **dead-stripped** (800 KB and 256 B controls serialize byte-identically). |

Raw `ptxas -arch=sm_90` **does** reject the M3.3 case
(`uses too much shared data (0xc3500 bytes, 0xc000 max)`), but `ptxas` is a
downstream vendor assembler in the same category as compute-sanitizer — not part
of the MLIR pipeline under audit. Per §6/R4 this `n/a` is **C2 evidence** (the
surface cannot express the obligation class), never counted as "detected".

**MLIR/linalg split:** linalg is compared only against **M2 (shape)**; low-level
MLIR only against **M1 (memory)**. The two MLIR targets get **different,
non-overlapping mutation sets** and are reported separately (never summed into
one "MLIR" row).

## 5. Minimal coverage set (E1) vs full suite (E2/E3)

E1 (injection) runs over the **minimal coverage set** — the smallest operator
subset where every obligation class is exercised by ≥1 operator. The full
15-category suite is still composed once per toolchain for E2/E3; only the
*injection* step is scoped to this set.

| Class | Mutate on (minimal set) |
|---|---|
| M1 element-access | `layer_norm`, `softmax`, `relu`, `transpose` |
| M2 shape-compat | `layer_norm`, `matmul`, `concat` |
| M3 hw-constraint | `matmul`, `conv2d` |

Level-2 (add only after level-1 green + breadth work done, in priority order):
`max_pool2d`, `conv2d` (M1) → `elemwise_add`, `softmax` (M2) → `embedding`,
`batch_norm` (M1) → `batch_norm` (M3).

### 5.1 Enumeration arithmetic (owner ruling 2026-09-09)

§0's **N = 40 per class** is a *target*, not a cap. A lane's actual count is
`specs × categories × shapes`, and where that product exceeds 40 the over-count
is **accepted as an auditable over-count** rather than trimmed — trimming would
mean dropping a spec or a category, which is exactly what breaks cross-surface
uniformity.

| Lane | Class | Arithmetic | Count | Δ vs 40 |
|---|---|---|---|---|
| `mlir-linalg` | M2 | 5 specs × 5 categories × 2 shapes; spec 5 emits 2 variants | 50 injections → 54 mutants | **+10** (owner-approved) |
| `mlir-low` | M1 | 6 specs × 4 categories × 2 shapes | 48 injections → 48 mutants | **+8** (owner-approved 2026-09-09) |

Both deltas are accepted under the same two conditions:

1. **Every surface shows the same enumeration for the class.** The counts above
   are pure products of §1/§2's spec lists and §5's minimal category set, so any
   lane that follows this file lands on the same number. Uniformity is what makes
   the lanes comparable; a lane may not cherry-pick specs to hit 40.
2. **The over-count is itself documented and attributable.** Each record carries
   `level`, so level-1 vs level-2 is separable per record, and each lane's
   `results/README.md` states its own census arithmetic.

No lane redesigns its kernels to force a class back to exactly 40.

## 6. Outcome taxonomy (per mutant)

`outcome ∈ {compile, runtime, never, n/a}`; for choreo also a `stage ∈
{compile, runtime}` split (the `c`/`r` boundary follows the **static-vs-dynamic
shape** boundary of the obligation, not its class). `n/a` = surface cannot
express the class; never merged into "detected".

## 7. Ground-truth oracle (mandatory)

Every `never` (and every `runtime`) mutant must carry a **manifest check**:
run the unmutated vs mutated kernel on the reference input and confirm the output
differs (or the kernel never writes). `manifest ∈ {corrupts, noop}` — a `noop`
mutant is a false success and is discarded (owner plan §11 item 1).

### 7.1 The check is per *mutant*, and manifestation can be mode-dependent (owner ruling 2026-09-09)

Owner plan §11 item 1 asks the question **per mutant**: "did this injected defect
actually corrupt anything?" Where a lane reports several *measurement modes* for
one injected defect — the MLIR lanes report RTV-off and RTV-on as separate
records, never merged (manifest §5.1) — the modes are two measurements of the
**same** defect. So:

- `noop` under **every** mode ⇒ false success, discarded.
- `noop` under **exactly one** mode ⇒ the defect is real but only some modes can
  see it. That is a **finding**, recorded as such — not a harness bug, and not
  grounds for discarding the mutant or for redesigning the kernel.

**Measured instance** (`mlir-low`, `relu`, M1.1 dropped boundary mask). The
hand-tiled loop runs one iteration past the extent, so the guardless read and
write both address `inp[i, extent]` / `out[i, extent]`; in row-major layout that
wraps to `[i+1, 0]`. Because relu is elementwise, `max(x[i+1,0], 0)` is exactly
the value the clean kernel already writes there, so *when the out-of-bounds write
lands idempotently* the output is bit-identical and no output-comparison oracle at
any tolerance can see it; RTV's bounds check does.

This is **undefined behavior, not a deterministic noop**: the OOB write may instead
corrupt the heap (→ `abort`) or wrap an index so a loop bound is never reached
(→ `hang`), depending on heap layout. A single run therefore samples the UB once
and is not reproducible. Measured over **30 runs** of the same lowered module
(small size), three of the four cells are deterministic and exactly one is not:

| cell | measured distribution | p(noop) |
|---|---|---|
| `relu/dyn` M1.1, RTV-off | 30× `never/noop` | **1.000** |
| `relu/dyn` M1.1, RTV-on | 30× `runtime/corrupts` | 0.000 |
| `relu/static` M1.1, RTV-on | 30× `runtime/corrupts` | 0.000 |
| `relu/static` M1.1, RTV-off | 19× `never/noop`, 11× `runtime/corrupts` | **0.633** |

Because the record tally is an artifact, the nondeterministic cell has to be sized
against: it lands in the `never/noop` bucket only if *every* run lands noop, with
probability $p^N$. At `N=5` that is **10.2%** — about one validation run in ten
would silently shift a tally cell (54/1/41 → 53/2/42). `_validate_m1.py` therefore
defaults to `N_REPEAT=16` (env `M1_REPEAT`), giving **6.6e-4**; `N=12` gives
4.2e-3. The cost is confined to that one cell, since every other (mutant, mode) is
deterministic and its N runs finish in milliseconds.

`_validate_m1.py` runs each (mutant, mode) `N_REPEAT` times and reduces over the
distribution (any `corrupts` wins): a mutant is a false success only when *every
run of every mode* is noop, and RTV-on always corrupts, so the **gate is
reproducible at any N**. Verdict: `relu/dyn` `noop` at RTV-off / `corrupts` at
RTV-on (deterministic mode-dependent finding); `relu/static` mode-dependent **and**
nondeterministic, reported with its measured distribution. The finding *set* grows
with N (4 at N=5, 7 at N=12) as deeper sampling surfaces rarer noop events, while
the record tally stayed byte-identical across the N=5 and N=12 runs — the artifact
is reproducible, the characterization is honestly reported as a sample.

The kernel is **not** redesigned to make this defect output-observable. An
elementwise map cannot force that without distorting the kernel, and the
mode-dependence is itself the honest result — it is precisely the contrast
manifest §5.1 requires both RTV modes be reported for.
