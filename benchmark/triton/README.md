# Root-level Triton comparison artifact

This directory contains the root-level Triton comparison suite for the paper artifact.

## Layout
- `manifest.csv`: full-suite mapping from `benchmark/choreo` cases to concrete per-case Triton entrypoints.
- `status.csv`: full-suite tracker for operator-view parity, explicit assertion count, and validation status.
- `<category>/`: one Python file per benchmark case, e.g. `relu/1_bert_32x512x768_32x512x768.py`.  This mirrors the layout of `benchmark/choreo/<category>/`.
- `cases/`: generation-time helpers (`triton_case_common.py`, shared family modules).
- `results/`: generated CSV summaries.

## Current suite
The Triton suite mirrors the layout of `benchmark/choreo`: each category has its own subdirectory and each case is one Python file, e.g. `benchmark/triton/relu/1_bert_32x512x768_32x512x768.py`.

Each generated file is a direct standalone benchmark entity with its own host-side driver, explicit runtime assertions, and operator implementation. The files do not import any shared wrapper modules (`generated_case`, `unary_family`, `binary_family`) at runtime.

The goal is operator-view equivalence, not identical internal implementation. Where Choreo can infer internal assessments, the Triton suite writes those checks explicitly in Python so the explicit-assertion-burden metric remains meaningful.

## Workflow
From the repository root:

```bash
source scripts/env.sh
# Re-render standalone case files after Choreo source changes:
python scripts/render_triton_standalone_cases.py
# Regenerate manifest.csv / status.csv:
python scripts/generate_triton_benchmark_suite.py
# Run a single case (compile smoke):
python benchmark/triton/relu/1_bert_32x512x768_32x512x768.py --compile-only
# Run chunk-data correctness check:
python benchmark/triton/relu/1_bert_32x512x768_32x512x768.py --chunk-check
# Compare against Choreo:
make compare-triton
```

If the Triton Python environment is not already available, build it first with:

```bash
./scripts/build_mlir_baselines.sh
```

The generated standalone cases depend on PyTorch for tensor construction and execution. If Triton or PyTorch is unavailable in the selected Python environment, the case records `N/A` and preserves that capability boundary in `results/compare_results.csv`. Very large concrete cases can also report `N/A` when they exceed the local execution budget; the contract row still remains covered in `manifest.csv` and `status.csv`.

## Baseline policy
- The Triton cases should preserve the same input and output shape contracts as the mapped Choreo cases.
- Unsupported environments or unsupported dynamic-shape forms are recorded as `N/A` instead of benchmark failures.
- `status.csv` is the maintained tracker for parity and validation state.
- Re-render the standalone case tree after Choreo benchmark changes with `./scripts/render_triton_standalone_cases.py`.
- Use the `# EXPLICIT_ASSERTION:` marker inside generated case files to gather assertion statistics directly from the benchmark tree.