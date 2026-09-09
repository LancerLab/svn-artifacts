# TileLang Worker — Onboarding Report (plan §8, mandatory feasibility gate)

**Date:** 2026-09-09 · **Lane:** `tilelang` (exploratory / rebuttal hold-back) ·
**Status:** onboarding; feasibility probe in flight.

## 1. My card (plan §8.1 + §1 + §11.7)

- **Surface:** TileLang DSL (tile-level program; `T.Kernel`, `T.copy`, `T.gemm`,
  `T.load`/`T.store` with explicit index/loop structure).
- **E1 mutations:** "per `coordinator` mapping" — §3.3 has no TileLang row. The
  surface is Triton-shaped (explicit indices + `T.gemm`), so I adopt the
  **Triton mapping as a documented proposal**: M1 expressible (explicit-index
  OOB), M3 partial (`T.gemm` K/tile divisibility), M2 n/a (no cross-tensor
  contract). Flagged to coordinator (Q1).
- **E2 expressibility:** shape-aware surface (partial static shape reasoning).
- **E3 remainder:** n/a (no generated safety checks — TileLang emits code, not
  checks), analogous to Triton.
- **Disclosure (§11.7):** exploratory hold-back lane; measured results recorded,
  disclosure decided at integration (never silent).

## 2. Feasibility inspection (hardest step first)

| # | Step | Status |
|---|---|---|
| F1 | TileLang install | **in flight** — 0.1.14 + apache-tvm-ffi 0.1.13.post3 installed (tsinghua mirror, 1.4 MB/s); full dep set (torch 2.14 + triton 3.8 + CUDA 13 toolkit) downloading. |
| F2 | sm_120 (Blackwell, RTX 5060 Ti) codegen | **pending** — `tilelang` `__init__` preloads `torch`, so the probe blocks on F1. TVM/TileLang 0.1.14 (early 2025) may predate sm_120 (top risk). |
| F3 | compute-sanitizer | yes — `/usr/local/cuda/bin/compute-sanitizer` (CUDA 13.0). |
| F4 | GPU | single GPU (16 GB); correctness lanes share (§12.6). |
| F5 | 15-category composition | elementwise/matmul/softmax feasible; conv2d/pool/embedding harder (early API). |

## 3. Lane risks (priority order)

1. **R-TL1 (sm_120):** TVM/TileLang 0.1.14 may not lower to Blackwell consumer.
   If so the lane blocks → record with evidence (rebuttal answer: "cannot target
   the eval hardware").
2. **R-TL2 (install weight):** torch + CUDA toolkit ~4 GB; was network-blocked on
   PyPI (~50 kB/s) until the tsinghua mirror (~1.4 MB/s) unblocked it.
3. **R-TL3 (mutation mapping unpinned):** I adopt Triton's M1/M3(partial) mapping
   as a proposal, documented (onboarding rule), not a silent guess.
4. **R-TL4 (early API):** some families (embedding gather, max_pool) may be
   unsupported → n/a with reason (evidence for C2).
5. **R-TL5 (disclosure):** exploratory; results must not silently enter the paper.

## 4. Questions for the coordinator

- **Q1:** pin the TileLang §3.3 row (M1 = explicit-index OOB, M2 = n/a, M3 =
  `T.gemm` divisibility — Triton-shaped).
- **Q2:** pin TileLang version (0.1.14 newest on PyPI).

## 5. Immediate next actions

1. Finish install; sm_120 codegen probe (trivial kernel).
2. On green: compose kernels (reuse `full_shapes.json`/`sizes.py`), gate, mutate
   (M1/M3), classify, collect, stats — mirroring the triton lane.
