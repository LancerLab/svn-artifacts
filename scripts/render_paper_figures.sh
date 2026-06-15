#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${1:-$SCRIPT_DIR/../benchmark/mlir/results}"
INPUT_CSV="${INPUT_CSV:-$RESULTS_DIR/compare_results.csv}"
OUTPUT_DIR="${2:-$RESULTS_DIR/plots}"

[[ -f "$INPUT_CSV" ]] || {
  echo "Missing input CSV: $INPUT_CSV" >&2
  echo "Run make compare first." >&2
  exit 1
}

python3 "$SCRIPT_DIR/plot_results.py" --input "$INPUT_CSV" --output-dir "$OUTPUT_DIR"
echo "Rendered figures into $OUTPUT_DIR"
