# IREE mutant surface

IREE consumes linalg-on-tensors MLIR; its *entry* surface exposes only shape contracts (no memory-access or hardware-constraint construct). Per specs/mutation-specs.md §4 + plan §3.3:

- **M1 element-access** -> n/a (no interior access at this surface; any spec M1 mutant has no expressible IREE form).
- **M2 shape-compatibility** -> expressible as an *entry shape/contract variant*: the defect is re-expressed by changing an operand's declared entry extent (caller-side contract), the only place IREE can detect it.
- **M3 hw-constraint** -> n/a (no mma/tma/alignment construct).

A mutant is represented as `mutants/M2/<category>/<mutant_id>.json` (reference kernel + the mutated entry extent); the kernel text itself is unchanged because the mutation lives at the entry contract, not the body.

## Measured outcome (ground truth, numeric oracle)

- **Static extent** mutants (operand dim is a fixed int): IREE rejects at the host entry via `hal.buffer_view.assert` (`INVALID_ARGUMENT: shape dimension mismatch`) -> **runtime**.
- **Dynamic extent** mutants (operand dim is `?`): IREE silently computes a wrong result -> **never/value-changing** (verified by numeric diff against the reference output; see `mutant_oracle.py`).

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
  entry -> **runtime/value-changing** (caught).
- M2-d/M2-e edits keep the operand rank and static extents, so IREE accepts the
  call and computes a wrong result -> **never/value-changing** (numeric oracle over
  raw-binary `@file.bin` inputs, since the ramp values exceed `ARG_MAX`).
- M2-h: the dynamic-split reshape kernels compute the split factor with a
  runtime floor division (`arith.divui flat_sz, static_factor`) and never check
  divisibility. A caller extent that is not a multiple is accepted (rc=0) and
  the tail elements are silently dropped -> **never/value-changing**.

## Families M2-f (`M2.5`) and M2-g (`M2.15`) -- authored mutation-only kernels

These two specs are defects in *which elements are written* while every declared
extent stays individually legal: M2.5 omits (partial) or duplicates (overlapping)
a tile, M2.15 moves padding between the two sides of a mirrored pad while
preserving the total length. No settings kernel encodes a coverage or
padding-placement contract, so a new mutation-only kernel is authored for each,
whose coverage / placement is carried by a dynamic control operand. The compiled
kernel is unchanged; only the caller's control extent moves. Authored under
`kernels/tile_write/` and `kernels/pad_shift/`, materialised by `lane.py m2`:

| family | spec | kernels | edit | outcome |
| --- | --- | --- | --- | --- |
| M2-f | M2.5 | `tile_write/{1..4}_tile` | control extent `n` covers the output; mutant `n-1` omits the tail tile, `n+1` wraps an overlapping tile onto row 0 | **never/value-changing** |
| M2-g | M2.15 | `pad_shift/{1..4}_pad` | `pad_low`/`pad_high` both `P`; mutant `(P+1, P-1)` or `(P-1, P+1)` shifts one row across the core, total preserved | **never/value-changing** |

4 kernels x 2 realisations = 8 each. Both keep the operand rank and static
output shape, so IREE accepts the call and returns a wrong result (numeric oracle
over raw-binary `@file.bin` inputs). The generic "small" feed is a valid
reference for both (`tile_write` wraps by modulo; `pad_shift`'s output rows are
`2*SMALL + C`), so compute-sanitizer stays silent -- these are shape-contract
defects, not memory faults.

## Not expressible at the entry surface

- **M2.18 reshape on a non-contiguous span.** Blocked by the suite: choreo
  records `realized: false`, `registry_status: pending`, `prohibition: absent`
  -- the base cases have no strided view for this defect.

n/a cells are never counted as detected (R4).

## Level-2 (mutation-specs §5)

- M1 rows (max_pool2d, conv2d, embedding, batch_norm) -> n/a.
- M3 row (batch_norm) -> n/a.
- M2 rows: `elemwise_add` expressible (binary-op/leading-extent) and added by `run.sh minimal --level2`; `softmax` is single-tensor -> n/a.

n/a cells are never counted as detected (R4).
