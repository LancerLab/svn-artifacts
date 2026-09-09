# IREE mutant surface

IREE consumes linalg-on-tensors MLIR; its *entry* surface exposes only shape contracts (no memory-access or hardware-constraint construct). Per specs/mutation-specs.md §4 + plan §3.3:

- **M1 element-access** -> n/a (no interior access at this surface; any spec M1 mutant has no expressible IREE form).
- **M2 shape-compatibility** -> expressible as an *entry shape/contract variant*: the defect is re-expressed by changing an operand's declared entry extent (caller-side contract), the only place IREE can detect it.
- **M3 hw-constraint** -> n/a (no mma/tma/alignment construct).

A mutant is represented as `mutants/M2/<category>/<mutant_id>.json` (reference kernel + the mutated entry extent); the kernel text itself is unchanged because the mutation lives at the entry contract, not the body.

## Measured outcome (ground truth, numeric oracle)

- **Static extent** mutants (operand dim is a fixed int): IREE rejects at the host entry via `hal.buffer_view.assert` (`INVALID_ARGUMENT: shape dimension mismatch`) -> **runtime**.
- **Dynamic extent** mutants (operand dim is `?`): IREE silently computes a wrong result -> **never/corrupts** (verified by numeric diff against the reference output; see `mutant_oracle.py`).

## Level-2 (mutation-specs §5)

- M1 rows (max_pool2d, conv2d, embedding, batch_norm) -> n/a.
- M3 row (batch_norm) -> n/a.
- M2 rows: `elemwise_add` expressible (binary-op/leading-extent) and added by `run.sh minimal --level2`; `softmax` is single-tensor -> n/a.

n/a cells are never counted as detected (R4).
