# Triton lane results (E1 L1+L2, E2, E3, S12) — 2026-09-09

Machine: 1× NVIDIA RTX 5060 Ti (sm_120), driver 580.95.05, CUDA 13.0,
Triton 3.8.0 (uv venv, torch-free: gpubuf.py ctypes/libcudart shim + numpy refs).

## S1 — detection matrix (bare Triton)

| Class | Injected | Compile | Runtime | Never | n/a |
|---|---|---|---|---|---|
| M1 element-access | 27 | 0 | 0 | 27 | 0 |
| M3 hw-constraint | 2 | 2 | 0 | 0 | 0 |

- M1: Triton detects nothing. All 27 mutants run to completion; every one is
  proven corrupting by the oracle (output diff or clobbered canary — the
  canary catches foreign-memory corruption that output comparison misses,
  e.g. relu dropped-mask: correct output, stomped guard region).
- M3: both JIT-rejected at compile time — `tl.dot` K-atom rule
  ("Input shapes should have M >= 1, N >= 1 and K >= 16") and the
  shared-memory budget (`OutOfResources: Required: 262144, limit: 101376`).

## S12 — compute-sanitizer supplement (over the same mutants)

| Class | Total | Flagged ∧ exercised |
|---|---|---|
| M1 | 27 | 24 |
| M3 | 2 | 0 (never reach device) |

The three M1 misses: `relu-f6` (empty range — nothing executes),
`max_pool2d-f2` (in-bounds wrong result — not a memory error),
`embedding-f2` (negative index lands in a valid page — no fault).

## E2 — breadth: 15/15 categories composed from settings/ and
reference-checked green (raw/kernels.jsonl). Expressibility per class:
M1 yes (manual masks), M2 no, M3 partial (JIT tile/SMEM rules),
loop yes (manual bounds) — raw/expressibility.jsonl.

## E3 — remainder: n/a (Triton generates no checks); results/e3_remainder.json.

## Notes / flags

- Schema gaps pending coordinator enum update (warnings in collect.py):
  `stage` lacks a not-detected value; `paper_category` lacks `hw`.
- Validation otherwise clean; raw/ JSONL preserved; collect is idempotent.
