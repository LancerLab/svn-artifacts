# benchmark2 — Phase-0 coordinator manifest

> Authoritative shared inputs for all workers. Every worker reads this first.
> Generated 2026-09-08 by the `coordinator`. Provenance: derived from
> `benchmark/` in the `svn-artifacts` repo.

## 1. Provenance & scope

- `settings/` is derived from `benchmark/choreo/*/` by `benchmark2/derive_settings.py`.
  It extracts **operator contracts only** — function signatures (tensor shapes),
  reference semantics, and per-case content-sha1 — **never the kernel source**
  (plan §2.2). Reviewers re-run `derive_settings.py` and diff the sha1s to confirm
  no source leaked.
- **15 operator categories** (owner-confirmed 2026-09-08): `batch_norm`, `concat`,
  `conv2d`, `elemwise_add`, `embedding`, `gelu`, `layer_normalization`, `matmul`,
  `max_pool2d`, `reduce_mean`, `relu`, `reshape`, `sigmoid`, `softmax`, `transpose`.
- `scripts` is the **harness directory, NOT a 16th operator** (owner-confirmed).
  Excluded from `settings/`.

## 2. Machine precondition (shared; NOT reproduced per worker)

| Item | Value |
|---|---|
| GPU | 2 × NVIDIA H800 PCIe, 80 GB VRAM each (81,559 MiB) |
| MPS | **off**; MIG disabled |
| Driver | 590.48.01 |
| CUDA | 13.1 |

Workers do **not** install system CUDA. They clone + build their toolchain at the
pinned commit on top of this base (§8.1). Toolkit/nvcc/driver are recorded here,
never rebuilt.

## 3. Pinned choreo binary + flags (real flag names)

- **binary**: `croqtile/build-release/choreo`
- `--dump-ledger=<path>`  *(hidden)* — dump the safety ledger (all assessed
  obligations + outcomes) as JSON. This is the paper's per-kernel ledger.
- `--stats`               — aggregate assertion/assessment statistics.
- `-sass` / `--show-assess` — per-assessment report (inputs, sites, hoists, cost).
- `-rtc=<level>`          — runtime-check level ∈ `{none, entry, low, medium, high,
  all}`. **`-rtc=none` = runtime checks off** (the plan's "`--runtime-checks off`").
  Also `--disable-runtime-check` and `--zero-cost` disable all runtime checks.
- Pinned suite invocation (from storyline-state): `--stats -es
  --max-local-mem-capacity=2000000 -t cute`.

### Flag-name mapping (plan → binary)

| Plan name | Real flag |
|---|---|
| `--dump-ledger` | `--dump-ledger=<path>` (hidden) |
| `--runtime-checks off` | `-rtc=none` |
| (aggregate stats) | `--stats` |

## 4. Mutation set (owner-confirmed 2026-09-08)

- Classes: **M1 element-access**, **M2 shape-compatibility**, **M3
  hardware-constraint**. Iteration-validity folded into M1; hw-constraint
  reinstated. Out of scope (named): concurrency safety, numeric correctness.
- **Equal count: N = 40 per class → 120 total.** *(was "160", stale from the
  pre-fold 4-class layout; owner decision 2026-09-09 — see
  `specs/mutation-specs.md` §0 and §5.1.)*
- Specs + §3.3 translation table: `specs/mutation-specs.md`.
- Minimal coverage set vs full suite split: `specs/mutation-specs.md` §5.

## 5. Toolchain versions (to be pinned by each worker's `setup`)

| Worker | Toolchain | Commit pinned at |
|---|---|---|
| `choreo` | croqtile | `build-release/choreo` (current; pin `git rev-parse HEAD` at `setup`) |
| `triton` | Triton | *(pin in `triton/run.sh setup`)* |
| `mlir-linalg` / `mlir-low` | LLVM/MLIR | **LLVM 21.1.0**, prebuilt at `~/dev/croqtile/extern/llvm-project/bin` (owner-approved 2026-09-09; see §5.1) |
| `iree` | IREE | `ce36167c3be514dd165a3ecff2d377cfa8eca0c9` (release `iree-3.12.0rc20260908`, 2026-09-08 main; wheel `iree_base_compiler`/`iree_base_runtime`) |
| `tilelang` (exploratory) | TileLang | *(pin in `tilelang/run.sh setup`)* |

Each worker records its exact clone commit + build flags in `manifest.md` **at
`setup` time** (idempotent, one-time).

**choreo lane (pinned by coordinator, 2026-09-10 — R-D6):**
- Data collected on source **`f2f238f`** (`binary_sha1_12 = c3ebb1654d1d`,
  `build-release/choreo`). The launch-rejection fix **`1fa4719` postdates the
  data** and is not reflected in any committed record.
- **Do not rebuild or re-run** E1–E5: a rebuild invalidates the arch sweep,
  ground-truth fixture, and E4/E5 artifacts, and breaks host-quiet for other
  lanes. The stale-binary flag in `toolchain_identity` is a disclosure, not a
  defect.

**iree lane (pinned by `iree` worker at `setup`, 2026-09-08):**
- Install: python venv (`/home/gxf/.tools/iree-dev-<build>-venv`) + `pip install
  iree_base_compiler` + `iree_base_runtime` from the release tag
  `iree-3.12.0rc20260908` (commit `ce36167c`); tools `iree-compile`,
  `iree-run-module` on `PATH`.
- Compile flags: `--iree-hal-target-backends=cuda --iree-cuda-target=<arch>`.
- **arch knob:** `IREE_CUDA_TARGET` (default `sm_120` on the current lane host;
  overridable `sm_86`/`sm_90` for the final host). sm_120 (and ≥ sm_89 in
  release v3.11.0) additionally need `--iree-cuda-target-features=+ptx87`
  (`IREE_CUDA_FEATURES`); **sm_90 currently fails codegen config on both
  v3.11.0 and 2026-09-08 main — do not pick an sm_90 final host for this lane
  without re-testing** (§8.1 feasibility finding, 2026-09-09).
  - **Root cause confirmed 2026-09-10 (re-test on H800):** `--iree-cuda-target=sm_90`
    fails with `error: missing GPU target in #hal.executable.target` (exit 74) at
    `linalg.generic`. `sm_89`/`sm_86`/`sm_80` compile fine; `sm_90`/`sm_90a` fail.
    The pinned commit `ce36167c` (releases `rc20260908` **and** `rc20260909`, same
    commit) has **no Hopper entry** in
    `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`
    → `getNVIDIAGPUTargetDetails()`: the StringSwitch lists `sm_60/61/62, sm_70/72,
    sm_75, sm_80/86/87, sm_89, sm_120, sm_121` but **skips `sm_90`** (and
    `normalizeNVIDIAGPUTarget` maps neither `hopper` nor `h100`). So
    `getCUDATargetDetails("sm_90")` returns null and no `#hal.executable.target`
    is materialized. The GPU-capability table is still missing Hopper as of IREE
    `main` (2026-09-10), so no nearby release fixes it.
- Lane run host (correctness, current): `garfee-ubuntu`, 1 × RTX 5060 Ti
  (sm_120), driver 580.95.05 / CUDA 13.0. **Differs from §2 machine
  precondition (2×H800)** — discrete E1/E2/E3 outcomes only; arch recorded per
  record so numbers can be re-derived on the final host.

### 5.1 MLIR lanes — pinned toolchain and measurement protocol

Owner rulings 2026-09-08/09 for `mlir-linalg` and `mlir-low`:

- **LLVM 21.1.0** from croqtile's `extern/` (`~/dev/croqtile/extern/llvm-project/bin`).
  The LLVM-18 assumption in `benchmark/../scripts/*.sh` (`~/mlir-local/usr/bin/mlir-opt-18`)
  is **abandoned** — that directory does not exist on this machine.
- **`mlir-cpu-runner` was renamed `mlir-runner`** in LLVM 21.
- **Report BOTH RTV-off and RTV-on** variants (owner: "both for now"). RTV =
  `generate-runtime-verification`. This is a fairness requirement: RTV is opt-in,
  and bare MLIR on an OOB `memref.load` **silently returns garbage with exit 0**.
  The two variants are separate rows/records, never merged.
- **Pass ordering is a correctness requirement, not a detail.** `lower-affine`
  must precede RTV (RTV instruments `memref.load`/`tensor.extract` but **not**
  `affine.load`: 0 asserts before, 1 after). `convert-index-to-llvm` must be in
  the lowering pipeline (RTV emits `index.bool.constant`).
- **Runtime protocol:** `stdbuf -o0 mlir-runner --entry-point-result=void
  --shared-libs=<libmlir_runner_utils.so>,<libmlir_c_runner_utils.so>` —
  comma-separated, and `stdbuf -o0` is mandatory because the assert message goes
  to **stdout** via `puts` before `abort()` and is otherwise lost. Distinguish a
  JIT-symbol-error exit 1 from a genuine detection (exit 134).

### 5.2 Small-input convention (plan §2.4 — delegated to the MLIR lanes)

Plan §2.4 assigns the small-input convention to the coordinator, but Phase 0 did
not pin one. Owner delegated it per-lane 2026-09-09 ("in different lane you may
have different tiling choices, where a unique small value does not fit").
Convention for `mlir-linalg` and `mlir-low`:

1. Preserve **rank** and the **static-vs-dynamic pattern** exactly, so small and
   full agree semantically (§2.4).
2. Replace every static extent with the smallest value that preserves op
   semantics (≥1; contraction dims equal on both operands for `matmul`/`conv2d`).
3. Bind every dynamic extent to a small constant at the entry, recorded per
   category in the lane's `raw/` provenance.
4. Keep **dtype unchanged** — f16 vs f32 matters for any alignment reasoning.

## 6. Device plan (§12.6)

Two GPUs (`--device 0`, `--device 1`). Correctness lanes (E1/E2/E3) share freely;
**E5 timing runs only under the exclusivity lock** `benchmark2/.gpu-lock`, with
`nvidia-smi` idle and `exclusive=true` recorded. The `qc` worker rejects any
`cost`/`residue`/`latency` record with `exclusive=false`.

## 7. Data-integrity gate (§7) — RESOLVED 2026-09-08

Owner ruled the 210-row `bug_detection_results.csv` (regenerated 2026-09-08)
authoritative, superseding the 103-suite. The paper (`main.tex`,
`tab:rq2-bugs`) is now written against the 210-suite:

| Axis | Authoritative (210-suite) |
|---|---|
| Total bugs | **210** |
| Class split | dim_mismatch 139, input_dep_oob 58, wrong_output 8, stride_error 5 |
| choreo (svn) resolution | **210 compile** / 0 launch / 0 undetected |
| MLIR | compile 80 / runtime 59 / undetected 71 |
| IREE | entry 80 / undetected 130 |

S11 integrity-register is populated from this suite. Workers may proceed
against this authoritative baseline; no new number is written unless it traces
to a `benchmark2/results/` JSON record.

## 8. Schema + statistics contract

- Raw record schema: `schema/record-schema.json` (one JSON object per observation).
- Statistics manifest (S1–S13, lane ownership): `schema/statistics-manifest.md`.
- Harness contract: `run.sh.template` (copy → `benchmark2/<toolchain>/run.sh`).

## 9. Reference `run.sh` skeleton

Copy `benchmark2/run.sh.template` to `benchmark2/<toolchain>/run.sh` and fill the
`run_*` bodies. Subcommands: `setup | minimal[--level2] | e2 | e3 | e4 | e5 |
collect | stats | all`, each taking `--small/--full` and `--device <i>`.
