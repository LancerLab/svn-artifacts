# CUTLASS/CuTe Worker — Onboarding Report (plan §8, mandatory feasibility gate)

**Date:** 2026-09-23 · **Lane:** `cutlass` (candidate / late-addition) ·
**Owner decision:** build a full 16-operator lane even though it misses the
2026-09-24 paper deadline; results are for the M3 (hardware-constraint) and
M4 (iteration-validity) comparison and may land post-deadline.
**Spec correction (authoritative):** `specs/mutation-specs-v2.md` §5/§6 do *not*
require all 16 categories: M3 runs on `{matmul, conv2d, batch_norm, max_pool2d}`
and M4 on `{layer_norm, softmax, matmul, ele_add}` (union = **7 operators**).
The lane implements the spec-required set first; the remaining 9 categories are
deferred to a post-deadline extension.

## 1. My card

- **Surface:** a CuTe/C++17 CUDA kernel compiled by `nvcc`. The program is the
  kernel source; the constraints at issue are CuTe/CUTLASS type-level invariants
  (MMA atom geometry, layout/copy partitioning, shared-memory footprint), CUDA
  launch limits, and the program's own iteration structure.
- **Toolchain pin:** CUTLASS **4.2.1** (croqtile `extern/cutlass`, submodule
  branch `v4.2.1`, `https://github.com/NVIDIA/cutlass.git`), CUDA toolkit
  **12.9.86** (`/usr/local/cuda-12.9`), arch `sm_86` (dev host RTX 3070 8 GB).
  Final-host arch differs (manifest §2: H800 sm_90) — every record carries the
  arch, and TMA/GMMA families are arch-gated on `sm_86`.
- **E1 mutations:** M3 `hw-constraint` and M4 `loop` (iteration-validity). M1/M2
  are not this lane's remit; where a mutation is genuinely M1/M2 it is recorded
  but not counted toward the M3/M4 denominators.
- **E2 expressibility (S8):** measured per category × obligation class.
- **Outcome mapping (§9.6):** `ct-check` = nvcc/CuTe `static_assert` or compile
  error caused by the injected state; `rt-check` = a CUDA runtime fault at launch
  or execution (illegal access, misaligned address, launch-config failure);
  `unchecked` = compiles and runs, silently wrong; `avoided` = CuTe derives or
  repairs the invariant; `unexpressible` = no legal program states the defect.
  A compile/run failure caused by the *harness* (malformed program) is a
  generator defect, never a result (§9.6 rule 1).
- **Disclosure:** candidate lane, arch-limited to `sm_86`; not integrated
  silently.

## 2. Feasibility inspection (hardest step first)

| # | Step | Result |
|---|---|---|
| F1 | CuTe compiles/runs on `sm_86`, CUDA 12.9 | **green** — `cute::Layout` + managed-memory kernel probe. |
| F2 | Library + headers reachable | **green** — CUTLASS 4.2.1 at croqtile `extern/cutlass`; `nvcc -std=c++17 -arch=sm_86 -I include -I tools/util/include`. |
| F3 | MMA-atom divisibility is caught by CuTe | **negative** — `TiledMMA::partition_A` on an indivisible tiling (24 vs atom-M 16) compiles cleanly and returns a rest/padded layout; no `static_assert`. M3.1 on this surface is `unchecked`, not `ct-check`. |
| F4 | Copy/vectorization violation yields `ct-check` | **confirmed** — a rank-1 extent (3) not divisible by the 128-bit (4×f32) copy atom trips `copy_atom.hpp:111` while the divisible control (4) compiles; clean M3.6 pair. |
| F5 | M4 zero-trip iteration | **green** — compiles, runs, `acc=0`; empty iteration space is a noop by construction (§9.6/§9.1 M4.4). |
| F6 | Faithful 16-operator CuTe kernels | **in flight** — this is the long pole; correctness is gated per operator against a CPU reference. |

## 3. Lane risks (priority order)

1. **R-CUT1 (surface permissiveness):** CuTe appears to type-check far less than
   choreo's assessor; the M3 finding may be dominated by `unchecked`. This is a
   real result, but it must be reported as "the surface does not detain X"
   (a `avoided`/`unchecked` claim), never as a detection rate.
2. **R-CUT2 (arch gating):** TMA/GMMA descriptor families (M3.2/M3.3/M3.5/M3.13)
   require `sm_90`; on the `sm_86` dev host they are compile-only or
   `unexpressible`. Records are arch-tagged; the final host must re-run.
3. **R-CUT3 (fidelity):** the operator set must be the same 15 categories the
   other lanes probe; a hand-written kernel that is not the operator would
   invalidate the comparison. Correctness is gated by a CPU reference.
4. **R-CUT4 (harness-owned failures):** a malformed mutant (our bug) must be
   classified as a generator defect and fixed, never counted.
5. **R-CUT5 (cost):** nvcc + CuTe compile times are seconds-to-minutes per
   mutant; the 16-category × mutation grid must be parallelized.

## 4. Questions for the coordinator

- **Q1:** confirm M3/M4-only remit for this lane, and that a surface that
  "does not detain" a constraint is reported `unchecked`/`avoided` (never a miss).
- **Q2:** confirm the arch disclosure wording for `sm_86` vs the `sm_90` final
  host, and whether TMA-family cells may be reported `unexpressible (arch)`.

## 5. Immediate next actions

1. Finish the common CuTe kernel skeleton (tile/vec/smem/atom/loop knobs) and
   the CPU reference + oracle for all 16 categories.
2. Implement the 16 operator kernels, gating each against the reference.
3. M3 + M4 mutation batteries → raw → collect → stats.

## 6. Status — vertical slice (2026-09-23)

**Done and verified:**

- **Operator layer (15 kernels, all CPU-gated):** the 7 spec-required
  operators (`elemwise_add`, `softmax`, `layer_normalization`, `matmul`,
  `conv2d`, `batch_norm`, `max_pool2d`) plus the 8 remaining shape-bearing
  categories (`relu`, `sigmoid`, `gelu`, `reshape`, `transpose`, `concat`,
  `embedding`, `reduce_mean`). Each compiles under
  `nvcc -std=c++17 -arch=sm_86 -O2` and its unmutated output passes
  `reference.py::gate` (max abs diff ≤ 9.6e-6).
- **Realizable mutation battery** (`lane.py`):
  - `M4.1` zero-trip loop (`-DLOOP=0`) → **`unchecked`** on all 15
    shape-bearing categories (silent corruption; no diagnostic) — a strong M4
    data point.
  - `M4.3` runtime-zero bound (`CUT_LOOP_RT=0`, bound well-formed at compile
    time) → **`unchecked`** on `elemwise_add`, `softmax`,
    `layer_normalization`, `matmul`. The runtime argument is passed to the
    kernel, so the compiler cannot fold it away.
  - `M4.5` zero K-tile stride (`-DSTEP=0`) → **`unchecked`** on `matmul`,
    `conv2d`.
  - `L1` dynamic shared over cap (`-DSMEM_ELT=32768`, 128 KB) →
    **`rt-check`** (`CUDA error: invalid argument`) on `elemwise_add`, recorded
    with the **`L`** path class (the limit is held by the driver, so
    `applicable=false`); `noop` on `matmul` (that kernel sizes smem from its
    tiles, so the knob is inert — the one discarded `n/a` control).
  - `M4.2` (parallelby 0/negative) overlaps the loop-extent knob; the negative
    variant is not representable here. `M4.4` (harness noop control) is not run
    in this slice.
- **v2.1-conforming records** (`mutrec.py`, the single writer): every record
  carries the base set (`toolchain, category, class, paper_category, mutant_id,
  level, outcome, stage, manifest`) plus the ported set (`spec_id, path_class,
  prohibition, applicable, spec_version`), with `stage` derived by
  `schema.records.stage_for` and every record validated before it is written.
  Lane outcome words map to the canonical vocabulary: `ct-check→compile`,
  `rt-check→runtime`, `unchecked→never`, `noop→n/a`. The corpus
  (`records.jsonl` 23, `records_m3.jsonl` 12) is committed, so a clone
  reproduces `stats.json` with no GPU.
- **`verify.py`** (also `run.sh verify`): re-validates every committed record
  against `record-schema.json`, checks `mutant_id` uniqueness, re-derives the
  S1 blocks from the records alone, and asserts axis agreement (no `L` in
  `S1_detection`; `S1_declared_uncompared` matches the axis).
- **Compile-only M3 probe battery** (`probes.py` + `kernels/probe_m3.cu`),
  12 control/mutant pairs compiled with `nvcc -c` (TMA/GMMA/vector pairs target
  `sm_90a`, the static-smem pair `sm_86`; never linked or run):
  - **`M3.5` swizzle-incompatible box shape → `ct-check`** — `Swizzle<7,4,3>`
    trips `cute/swizzle.hpp:63` "Unsupported layout swizzle"; the legal control
    compiles.
  - **`M3.6` vector divisibility → `ct-check`** — a rank-1 extent of 3 floats
    with a 128-bit (4×f32) copy atom trips `copy_atom.hpp:111` "Src/Dst
    partitioning does not match the instruction requirement"; extent 4 compiles.
  - **`M3.8` GMMA descriptor rank → `ct-check`** — a rank-3 smem tensor trips
    `mma_traits_sm90_gmma.hpp:202` "GMMA Descriptors can only be constructed on
    rank-2 tensors"; the canonical rank-2 layout compiles.
  - **`M3.1` atom-M divisibility → `unchecked`** — `partition_A` accepts a
    24-row tiling with a 16-wide atom; no `static_assert` (confirms F3).
  - **`M3.2` descriptor dim ≥ 2²⁴ → `unchecked`** — a 2²⁴-extent TMA tensor
    compiles.
  - **`M3.3` TMA box byte-size ≥ 2²⁴ → `unchecked`** — a 256³ box (64 MB)
    compiles.
  - **`M3.4` footprint ≥ 4 GB → `unchecked`** — a 2³²-byte product compiles.
  - **`M3.7` TMA inner box not 16 B aligned → `unchecked`** — an 8 B-inner box
    compiles (driver `cuTensorMapEncodeTiled` would check at runtime).
  - **`M3.11` shared base not 128 B aligned → `unchecked`** — a 4 B-offset smem
    base compiles.
  - **`M3.12` static shared over cap → `unchecked` (compile)** — 128 KB static
    shared compiles on `sm_86`; the limit is a launch-time (driver) check.
  - **`M3.13` swizzle vs box inner dim (SW128, 128 B vs 32 B) → `unchecked`** —
    both tile-to-shape variants compile; the mismatch is not statically detained.
  - **`M3.14` linear copy dim ≥ 2²⁴ → `unchecked`** — a `DefaultCopy` over
    2²⁴ elements compiles (no check on the linear-copy path).
- **Harness:** `lane.py` (base+mutants+collect), `probes.py`, `reference.py`
  (bit-exact `fill` + refs + tolerance gate), `collect.py` (spec-§8-shaped
  `stats.json`, merges both record streams), `mutrec.py`, `verify.py`, `run.sh`
  (setup/minimal/e2/e3/collect/stats/verify/all).
  `results/cutlass/stats.json`: M3 = 12 injected / 3 `compile` / 9 `never`;
  M4 = 21 `never`; L = 1 `runtime` / 1 `n/a`. Flagged
  `lane_phase: vertical-slice`, `complete: false`.

**Not expressible on this surface (not probe cells):**

- **`M3.9` / `M3.10`** (TMA pad-field overrun; rank-5 `padding_mid[rank-1]==0`)
  and **`M3.15`** (`.pad`-path dim ≥ 2²⁴): CuTe authors the `CUtensorMap`
  descriptor and has no pad operator, so the mutation cannot be injected by the
  kernel author — `unexpressible`.
- **`M3.16`** (TMA box inner alignment with a *symbolic* leading dim):
  `make_tma_copy` requires static box shapes, so the symbolic-dim variant is not
  representable — `unexpressible`.

**Still open:**

- The compile-only `unchecked` families may in fact be detained by the driver
  at launch (`cuTensorMapEncodeTiled` / launch config); that is unmeasured on
  the sm_86 dev host and needs the final H800 host.
- The 9 non-required categories are deferred.

**Repro:** `PY=/home/gxf/.tools/iree-dev-20260908-venv/bin/python bash run.sh all`
(runs base + battery + probes + collect + verify; ~12 min on the dev host).

**Canonical-vocabulary note:** §9.6 words used above are the lane's internal
outcome names; the committed records use the schema vocabulary
(`compile|runtime|never|n/a`) with `path_class` (`ct-check|rt-check|unchecked|
avoided|L`) and `prohibition` (`absent|observation|harness-owned|...`).
