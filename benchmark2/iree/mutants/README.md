# IREE mutant surface

IREE consumes linalg-on-tensors MLIR; its *entry* surface exposes only shape contracts (no memory-access or hardware-constraint construct). Per specs/mutation-specs.md §4 + plan §3.3:

- **M1 element-access** -> n/a (no interior access at this surface; any spec M1 mutant has no expressible IREE form).
- **M2 shape-compatibility** -> expressible as an *entry shape/contract variant*: the defect is re-expressed by changing an operand's declared entry extent (caller-side contract), the only place IREE can detect it.
- **M3 hw-constraint** -> n/a (no mma/tma/alignment construct).

A mutant is represented as `mutants/M2/<category>/<mutant_id>.json` (reference kernel + the mutated entry extent); the kernel text itself is unchanged because the mutation lives at the entry contract, not the body.

## Measured outcome (ground truth, numeric oracle)

- **Static extent** mutants (operand dim is a fixed int): IREE rejects at the host entry via `hal.buffer_view.assert` (`INVALID_ARGUMENT: shape dimension mismatch`) -> **runtime**.
- **Dynamic extent** mutants (operand dim is `?`): IREE silently computes a wrong result -> **never/corrupts** (verified by numeric diff against the reference output; see `mutant_oracle.py`).

## Family M2-a (`run.sh minimal --level2`)

Entry-extent edits on a secondary operand (`layer_norm` `gamma`/`beta`, matmul
`rhs`, `elemwise_add` `rhs`). Budgeted like every other family at **4 kernels x
2 realisations = 8** (`M2_A_KEEP` in `lane.py`): one dynamic `layer_norm` kernel
realises M2.19 (symbolic extent, escapes the entry check -> **never**), one
static `layer_norm` kernel realises M2.1 (-> **runtime**), `elemwise_add`
realises M2.3 and `matmul` realises M2.14. The fuller per-kernel battery (23
instances) is not committed.

## Families M2-b .. M2-h (`run.sh m2`)

Beyond the M2-a extent edits (`run.sh minimal`), `lane.py m2` materialises the
remaining entry-contract families. Every mutant leaves the kernel text
unchanged and edits only what the caller passes to the compiled module:

| family | specs | edit kinds | kernels |
| --- | --- | --- | --- |
| M2-b | M2.6, M2.9 | `swap` two extents | matmul, transpose, concat, conv2d |
| M2-c | M2.7, M2.16 | `rankdrop`, `rankadd` (size-1 insert) | matmul, transpose, softmax, reduce_mean |
| M2-d | M2.8, M2.21 | `set1`, `setv` broadcast extent | layer_normalization, batch_norm |
| M2-e | M2.10, M2.13 | `dataperm` (same shape, different memory order) | matmul, transpose, concat, batch_norm |
| M2-h | M2.17 | non-divisible dynamic split extent | reshape |

- M2-b/M2-c edits change rank or the extent order, so IREE rejects them at the
  entry -> **runtime/corrupts** (caught).
- M2-d/M2-e edits keep the operand rank and static extents, so IREE accepts the
  call and computes a wrong result -> **never/corrupts** (numeric oracle over
  raw-binary `@file.bin` inputs, since the ramp values exceed `ARG_MAX`).
- M2-h: the dynamic-split reshape kernels compute the split factor with a
  runtime floor division (`arith.divui flat_sz, static_factor`) and never check
  divisibility. A caller extent that is not a multiple is accepted (rc=0) and
  the tail elements are silently dropped -> **never/corrupts**.

## Not expressible at the entry surface

- **M2-f (M2.5) partial / duplicate write.** The write coverage is a property
  of the kernel body (e.g. `concat/11` writes its two inputs with explicit
  `tensor.insert_slice` extents that cover the whole output). No caller operand
  encodes tile coverage, so an omitted/overlapping tile cannot be re-expressed
  as an entry contract.
- **M2-g (M2.15) pad_low <-> pad_high swapped.** Padding is a compile-time
  literal in the kernel body (`conv2d/2`: `tensor.pad low[0,0,1,1] high[0,0,1,1]`).
  No caller operand encodes pad placement.
- **M2.18 reshape on a non-contiguous span.** Blocked by the suite: choreo
  records `realized: false`, `registry_status: pending`, `prohibition: absent`
  -- the base cases have no strided view for this defect.

n/a cells are never counted as detected (R4).

## Level-2 (mutation-specs §5)

- M1 rows (max_pool2d, conv2d, embedding, batch_norm) -> n/a.
- M3 row (batch_norm) -> n/a.
- M2 rows: `elemwise_add` expressible (binary-op/leading-extent) and added by `run.sh minimal --level2`; `softmax` is single-tensor -> n/a.

n/a cells are never counted as detected (R4).
