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
- **Equal count: N = 40 per class → 160 total.**
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
| M3 hw | `mma`/`tma` alignment, atom divisibility | partial (`tl.dot` K divisibility) | n/a | partial (`vector`/`affine` alignment) | n/a |

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

## 6. Outcome taxonomy (per mutant)

`outcome ∈ {compile, runtime, never, n/a}`; for choreo also a `stage ∈
{compile, runtime}` split (the `c`/`r` boundary follows the **static-vs-dynamic
shape** boundary of the obligation, not its class). `n/a` = surface cannot
express the class; never merged into "detected".

## 7. Ground-truth oracle (mandatory)

Every `never` (and every `runtime`) mutant must carry a **manifest check**:
run the unmutated vs mutated kernel on the reference input and confirm the output
differs (or the kernel never writes). `manifest ∈ {corrupts, noop}` — a `noop`
mutant is a false success and is discarded (§11.1).
