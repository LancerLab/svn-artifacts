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
- **Equal count: N = 40 per class → 160 total.**
- Specs + §3.3 translation table: `specs/mutation-specs.md`.
- Minimal coverage set vs full suite split: `specs/mutation-specs.md` §5.

## 5. Toolchain versions (to be pinned by each worker's `setup`)

| Worker | Toolchain | Commit pinned at |
|---|---|---|
| `choreo` | croqtile | `build-release/choreo` (current; pin `git rev-parse HEAD` at `setup`) |
| `triton` | Triton | *(pin in `triton/run.sh setup`)* |
| `mlir-linalg` / `mlir-low` | LLVM/MLIR | *(pin in each `run.sh setup`)* |
| `iree` | IREE | `ce36167c3be514dd165a3ecff2d377cfa8eca0c9` (release `iree-3.12.0rc20260908`, 2026-09-08 main; wheel `iree_base_compiler`/`iree_base_runtime`) |
| `tilelang` (exploratory) | TileLang | *(pin in `tilelang/run.sh setup`)* |

Each worker records its exact clone commit + build flags in `manifest.md` **at
`setup` time** (idempotent, one-time).

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
- Lane run host (correctness, current): `garfee-ubuntu`, 1 × RTX 5060 Ti
  (sm_120), driver 580.95.05 / CUDA 13.0. **Differs from §2 machine
  precondition (2×H800)** — discrete E1/E2/E3 outcomes only; arch recorded per
  record so numbers can be re-derived on the final host.

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
