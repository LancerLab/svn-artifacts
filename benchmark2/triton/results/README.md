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

## S1 — detection matrix (bare Triton) — current corpus (sm_86, 2026-09-24)

The narrative below this section was written for the sm_90 host's 29-mutant
corpus. The current corpus is **176 mutants** (`results/stats.json` is
authoritative): M1 64, M3 56, M4 56, M2 n/a.

| Class | Injected | Compile | Runtime | Never | n/a |
|---|---|---|---|---|---|
| M1 element-access | 64 | 8 | 11 | 45 | 0 |
| M3 hw-constraint | 56 | 40 | 0 | 16 | 0 |
| M4 iteration-space | 56 | 8 | 0 | 48 | 0 |
| M2 shape/structure | 0 | 0 | 0 | 0 | 40 |

- M1: bare Triton catches nothing at runtime (the 11 `runtime` rows are child
  crashes with no emitted check → `never`); the 8 compile refusals are the
  rank/arity surface (M1-g). Every runnable M1 mutant is proven corrupting by
  the oracle (output diff or clobbered canary).
- M3: 40 JIT-refused (descriptor box/rank/atom/smem rules) = `ct-check`; 16 run
  end-to-end and miss (`never`).
- M4: 8 refused (`tl.static_range` zero step, M4-f); 48 run and miss.

## S12 — compute-sanitizer supplement (over the same 176 mutants)

| Class | Total | Flagged ∧ exercised |
|---|---|---|
| M1 | 64 | 36 |
| M3 | 56 | 0 (never reach the device) |
| M4 | 56 | 0 |

Re-measured 2026-09-24 on sm_86 over the full E1 corpus
(`sanitizer_corpus.py`, `compute-sanitizer --tool memcheck`). The sanitizer adds
**36 M1 detections** bare Triton misses — all `Invalid __global__` reads, i.e.
memory-safety coverage for over half the M1 cell. M3 flags are zero because the
mutants are JIT-refused before launch; M4 flags are zero because the
iteration-space defects are logic errors (no out-of-bounds access) so memcheck
has nothing to observe.

## E2 — breadth: 15/15 categories composed from settings/ and
reference-checked green (raw/kernels.jsonl). Expressibility per class:
M1 yes (manual masks), M2 no, M3 partial (JIT tile/SMEM rules),
loop yes (manual bounds) — raw/expressibility.jsonl.

## E3 — remainder: n/a (Triton generates no checks); results/e3_remainder.json.

## Notes / flags

- Schema gaps pending coordinator enum update (warnings in collect.py):
  `stage` lacks a not-detected value; `paper_category` lacks `hw`.
- Validation otherwise clean; raw/ JSONL preserved; collect is idempotent.
