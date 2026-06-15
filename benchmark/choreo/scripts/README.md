# Benchmark Scripts

Evaluation scripts for the Choreo compiler's symbolic system.

## Scripts

### Performance Analysis
- `cost_eval_symbolic_system.sh` - Compare static vs dynamic compilation performance
- `test_cost_eval.sh` - Quick performance test with sample files

### Shape Resolution Analysis
- `shape_resolve_analysis.sh` - Analyze symbolic shape inference capabilities
- `test_shape_analysis.sh` - Quick shape analysis test

## Usage

```bash
# Performance benchmarks
./benchmark/scripts/cost_eval_symbolic_system.sh
./benchmark/scripts/test_cost_eval.sh

# Shape analysis
./benchmark/scripts/shape_resolve_analysis.sh
./benchmark/scripts/test_shape_analysis.sh
```

Results are saved to `benchmark/scripts/results/`.

## Runtime Sweep Tracking

The root-level runtime sweep lives in [scripts/run_choreo_runtime_sweep.sh](../../../scripts/run_choreo_runtime_sweep.sh).

### Sweep commands

```bash
# Run the full runtime sweep and update the local ledger.
bash scripts/run_choreo_runtime_sweep.sh

# Sweep one family only.
bash scripts/run_choreo_runtime_sweep.sh --family concat

# Recompute the freshness report without running cases.
bash scripts/run_choreo_runtime_sweep.sh --status
```

### Local ledger files

- `benchmark/choreo/scripts/results/runtime_sweep_status.tsv`: latest recorded status per case, including the tracked input fingerprint.
- `benchmark/choreo/scripts/results/runtime_sweep_freshness.tsv`: current fresh/stale view against the workspace files.

Cases are marked stale when either:
- they have never been swept, or
- the tracked fingerprint changed since the last recorded sweep.

The tracked fingerprint covers the case `.co` file, any family-local `common.h` or `common.hpp`, and the sweep driver itself.

## Performance Analysis Details

Compares static vs dynamic compilation modes:
- **Static**: `__STATIC_SHAPE__=1` (concrete dimensions)
- **Dynamic**: No flag (symbolic dimensions)

Measures compilation and execution time deltas with statistical analysis.

## Shape Analysis Details

Analyzes `choreo -i` output to categorize shape resolution:
- **Concrete**: `[32, 768]` - fully resolved dimensions
- **Symbolic**: `[32, ::seq_len, 768]` - symbolic dimensions
- **Partial**: `[32, ?, 768]` - unresolved dimensions

Reports success rates and resolution quality metrics.
