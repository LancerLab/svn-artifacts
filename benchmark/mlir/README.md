# Root-level MLIR comparison artifact

This directory contains the root-level MLIR comparison slice for the paper artifact.

## Layout
- `manifest.csv`: maps Choreo benchmark inputs to MLIR inputs.
- `cases/`: standalone MLIR programs for the current public slice.
- `results/`: generated CSV summaries and plots.

## Current slice
The first public slice is CPU-first and focuses on symbolic-shape reasoning:
1. dynamic `matmul`,
2. attention `reshape`, and
3. an invalid `matmul` verifier failure.

## Workflow
From the repository root:

```bash
make mlir-clone
make mlir-build
source scripts/env.sh
make choreo-build
make compare
make plot
```

The comparison uses the Choreo checkout in [choreo](../../choreo) but all orchestration is rooted at the repository top level.
