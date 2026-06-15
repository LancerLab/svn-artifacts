# SVN: Shape Value Numbering — CGO 2027 Artifact

This repository is the artifact for the paper:

> **SVN: Shape Value Numbering for Comprehensive and Practical Safety Assessment**
>
> Submitted to CGO 2027

It contains the benchmark suite, evaluation scripts, and build orchestration
to reproduce the results (RQ1–RQ4) presented in the paper.

## Repository Layout

```
.
├── choreo/                  # Choreo compiler (git submodule → GitHub)
├── benchmark/
│   ├── choreo/              # 310 Choreo (.co) benchmark cases (15 categories)
│   ├── mlir/                # MLIR linalg comparison cases
│   ├── memref/              # MLIR memref comparison cases
│   ├── iree/                # IREE comparison cases
│   ├── triton/              # Triton comparison cases
│   ├── bugs/                # Bug injection mutants (RQ2)
│   └── results/             # (generated locally, not committed)
├── scripts/                 # Data collection, plotting, and automation
│   ├── reproduce_all.sh     # ★ One-command reproduction script
│   ├── choreo_assertion_stats.py    # RQ1: assessment coverage & discharge
│   ├── bug_detection_eval.py        # RQ2: bug detection effectiveness
│   ├── choreo_compile_overhead.py   # RQ4: compile-time overhead
│   ├── choreo_runtime_entry.py      # RQ3: runtime assertion overhead
│   ├── visualize_results.py         # Terminal + HTML report generation
│   ├── collect_all_stats.py         # Cross-system comparison
│   └── ...
├── Makefile                 # Build targets
└── README.md                # This file
```

## Quick Start

### Prerequisites

| Tool       | Version   | Notes                                        |
|------------|-----------|----------------------------------------------|
| GCC / G++  | >= 9.0    | C++17 support required                       |
| CMake      | >= 3.16   | Build system                                 |
| Ninja      | any       | `ninja-build` package                        |
| Python     | >= 3.8    | For statistics and plotting scripts           |
| matplotlib | any       | Optional: for PNG figures and HTML report     |
| Git        | any       | Submodule checkout                            |
| flex/bison | >= 2.6/3.8| Auto-downloaded if missing (see below)        |
| CUDA       | >= 12.0   | Required for RQ3 (runtime overhead) + GPU tests |

**Flex and Bison** are auto-downloaded and compiled from source during CMake
configuration if the system versions are missing or too old.

### One-Command Reproduction

```bash
git clone --recursive https://github.com/LancerLab/svn-artifacts.git
cd svn-artifacts
bash scripts/reproduce_all.sh
```

This will:

1. Initialize the Choreo submodule and its dependencies (cutlass, gtest)
2. Build Choreo from source
3. Run compile-time tests (check + cli)
4. Collect RQ1 assessment statistics (310 cases × 15 categories)
5. Run RQ2 bug detection evaluation (210 injected bugs × 3 systems)
6. Measure RQ3 runtime assertion overhead (if CUDA GPU available)
7. Measure RQ4 compile-time overhead (153 symbolic cases)
8. Print a comparison table against the paper values
9. Generate an interactive HTML report (`benchmark/results/report.html`)

Results are written to `benchmark/results/`.

### Expected Wall-Clock Times

| Step                    | Approx. Time | Notes                        |
|-------------------------|--------------|------------------------------|
| Build Choreo            | ~1 min       | Parallel make                |
| Compile-time tests      | ~30 sec      | lit runner                   |
| RQ1: Assessment stats   | ~2 min       | 310 cases, parallel          |
| RQ2: Bug detection      | ~5 min       | 210 mutants × SVN + MLIR    |
| RQ3: Runtime overhead   | ~10 min      | GPU required, 7 reps/case   |
| RQ4: Compile overhead   | ~3 min       | 153 cases × 5 reps          |
| MLIR build (optional)   | ~30 min      | Full LLVM from source        |
| Visualization           | ~10 sec      | Report generation            |
| **Total (with GPU)**    | **~22 min**  | Excluding optional MLIR build|

### Output

The script produces:

- **Terminal**: Rich summary tables with per-category breakdowns for all RQs
- **`benchmark/results/report.html`**: Self-contained HTML with interactive Chart.js graphs
- **`benchmark/results/figures/`**: PNG figures for each RQ (requires matplotlib)
- **`benchmark/results/choreo_stats.csv`**: Raw RQ1 data
- **`benchmark/results/bug_detection_results.csv`**: Raw RQ2 data
- **`benchmark/results/choreo_runtime_entry.csv`**: Raw RQ3 data (if GPU available)
- **`benchmark/results/choreo_compile_overhead.csv`**: Raw RQ4 data

## Research Questions

### RQ1: Safety Assessment Coverage and Discharge

Evaluates the breadth (ACD: Assessment Coverage Density) and resolution
capability (ADR: Assessment Discharge Ratio) of each system.

| Metric            | Paper Value |
|-------------------|-------------|
| Total assessments | 12,592      |
| Static discharged | 11,753      |
| ADR               | 93.3%       |
| Cases compiled    | 310/310     |
| ACD               | 40.6/case   |

Comparison: MLIR generates 2,634 (ACD 8.5, ADR 62.9%), IREE generates 370
(ACD 1.2, ADR 0%), Triton generates 0 compiler assessments (manual only).

### RQ2: Bug Detection Effectiveness

Tests detection of 210 injected shape bugs across 4 classes:
1. Dimension mismatch (139 bugs)
2. Input-dependent OOB (58 bugs)
3. Wrong output shape (8 bugs)
4. Stride/layout error (5 bugs)

| System | Detected | BDE    | Resolution          |
|--------|----------|--------|---------------------|
| SVN    | 210/210  | 100%   | All compile-time    |
| MLIR   | 139/210  | 66.2%  | 80 static + 59 runtime |
| IREE   | 80/210   | 38.1%  | All entry-level     |

### RQ3: Runtime Assertion Cost

Measures execution-time overhead (RAO) at four assertion levels:
- **none**: baseline (no assertions)
- **entry**: host-side entry-point checks only
- **all (hoisted)**: full checks with assertion hoisting
- **all (no-hoist)**: full checks without hoisting

| Level          | Paper Avg | Paper Max |
|----------------|-----------|-----------|
| Entry          | <0.4%     | —         |
| All (hoisted)  | +1.8%     | +7.1%     |
| All (no-hoist) | +9.6%     | +92.6%    |

Hoisting delivers a 5.3x cost reduction.

### RQ4: Compile-Time Overhead

Measures SVN's frontend compilation cost on 153 symbolic-dimension cases.

| Metric    | Paper Value |
|-----------|-------------|
| CTO       | 4.7%        |
| Per-case  | ~3.7 ms     |

## Step-by-Step Reproduction

```bash
# 1. Build Choreo
make choreo-build

# 2. Run compile-time tests
make choreo-test

# 3. Collect assessment statistics (RQ1)
make choreo-stats

# 4. Run bug detection evaluation (RQ2)
python3 scripts/bug_detection_eval.py

# 5. (Requires CUDA GPU) Measure runtime overhead (RQ3)
export CUDA_HOME=/usr/local/cuda
export CUTE_HOME=$(pwd)/choreo/extern/cutlass
python3 scripts/choreo_runtime_entry.py --reps 7 --levels none,entry,all,all-nohoist

# 6. Measure compile-time overhead (RQ4)
make choreo-cto

# 7. Generate visualization
python3 scripts/visualize_results.py

# 8. (Optional) Cross-system comparison
make mlir-clone && make mlir-build
python3 scripts/collect_all_stats.py
```

### MLIR Baseline (optional)

The cross-system comparison (SVN vs MLIR vs IREE vs Triton) requires
building the MLIR tools:

```bash
make mlir-clone   # shallow-clone llvm-project release/22.x
make mlir-build   # build mlir-opt, mlir-translate, FileCheck (~30 min)
```

Then re-run `bash scripts/reproduce_all.sh` without `--skip-mlir`.

## Pinned Versions

| Component      | Version       | Source                                   |
|----------------|---------------|------------------------------------------|
| Choreo (SVN)   | cgo2027-eval  | `github.com/LancerLab/croqtile`         |
| LLVM/MLIR      | release/22.x  | `github.com/llvm/llvm-project`           |
| IREE           | v3.10.0       | pre-compiled or `scripts/fetch_mlir_baselines.sh` |
| Triton         | v3.6.0        | `scripts/fetch_mlir_baselines.sh`        |
| CUTLASS        | v4.2.1        | via Choreo submodule                     |
| GoogleTest     | latest        | via Choreo submodule                     |

## License

See individual component licenses. The benchmark cases and evaluation scripts
in this repository are provided for artifact evaluation purposes.
