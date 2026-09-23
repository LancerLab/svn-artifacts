# Triton lane results (E1 L1+L2, E2, E3, S12) — 2026-09-10 (sm_90 final host)

Machine: 2× NVIDIA H800 PCIe (sm_90), driver 590.48.01, CUDA 13.0 toolkit,
Triton 3.8.0 (uv venv, torch-free: gpubuf.py ctypes/libcudart shim + numpy refs).
Every record carries `"arch"`, auto-detected from the GPU the lane runs on
(the sm_90 host produced `sm_90`; the current dev box produces `sm_86`).
Outcomes were shown arch-invariant against a 2026-09-09 sm_120 dev run.

## Arch handling (updated 2026-09-24)

The sm_90 H800 host is no longer available. The lane now targets **sm_86**
(dev box) and **sm_120** (available host); `--arch sm_86|sm_120|auto` labels
records with the requested arch and `driver.py` refuses the run if it differs
from the physical GPU (`--allow-mismatch` only for compile-only checks):

```
benchmark2/triton/run.sh minimal --arch=sm_120   # run on the sm_120 host
benchmark2/triton/run.sh archcheck               # compile-only cross-arch proof
```

`archcheck` compiles one representative kernel per mutation-only mechanism for
each arch without a GPU and asserts the verdict is identical; current verdict
(`raw/archcheck.json`): non-pow2 box / oversized box / sub-16-B box / rank-6
descriptor all **refused** on sm_86 and sm_120, rank-1 descriptor and int32-dim
(M3.2, a silent defect) **lowered** on both — i.e. the frontend surfaces are
arch-invariant.

## S1 — detection matrix (bare Triton) — FINAL, full-size runs

| Class | Injected | Compile | Runtime | Never | n/a |
|---|---|---|---|---|---|
| M1 element-access | 27 | 0 | 2 | 25 | 0 |
| M3 hw-constraint | 2 | 2 | 0 | 0 | 0 |

Size split (plan §2.4/rec. 4): gates pass at both small and full; mutants run
at FULL_RAGGED (sizes.py — full magnitude, ragged boundary so mask defects
manifest; the dynamic-dim cases take any runtime extent). Two mutants are
size-sensitive: layer_norm-f3/softmax-f3 (negative index) run silent at small
size but fault (illegal access) at full — recorded as runtime at full.

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
