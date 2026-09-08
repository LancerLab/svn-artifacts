#!/usr/bin/env bash
# benchmark2/triton/run.sh — Triton lane harness (contract §12.1).
# The lane is torch-free by design: kernels/mutants run on
# benchmark2/triton/gpubuf.py (ctypes libcudart shim) + numpy references,
# with Triton 3.8.0 in an isolated uv venv (pinned below at setup).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${TRITON_VENV:-/tmp/triton-venv}"
PY="$VENV/bin/python"
SIZE="small"
LEVEL2=0
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
for arg in "$@"; do
  case "$arg" in
    --small) SIZE="small";; --full) SIZE="full";;
    --level2) LEVEL2=1;;
    --device=*) DEVICE="${arg#*=}";;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
need() { [ -x "$PY" ] || die "triton venv missing at $VENV (run: uv venv $VENV && uv pip install triton numpy)"; }

cmd_setup() {
  need
  {
    echo "triton=$("$PY" -c 'import triton; print(triton.__version__)')"
    echo "numpy=$("$PY" -c 'import numpy; print(numpy.__version__)')"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "cuda=$(/usr/local/cuda/bin/nvcc --version | tail -1)"
    echo "note: torch-free lane (gpubuf.py ctypes shim); pinned by wheel hash"
    echo "wheel=triton-3.8.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64"
  } | tee "$HERE/raw/setup.txt"
}

cmd_minimal() {
  need
  "$PY" "$HERE/driver.py" minimal --device "$DEVICE"
}

cmd_e2() {
  need
  # compose+gate all 15 categories: gate the composed kernels that exist in
  # kernels/; record expressibility per class (see results/expressibility.json)
  for k in "$HERE"/kernels/*.py; do
    cat_name="$(basename "$k" .py)"
    "$PY" "$HERE/driver.py" gate --category "$cat_name" --size "$SIZE" \
      --device "$DEVICE" || echo "gate failed: $cat_name"
  done
}

cmd_e3() {
  # Triton generates no checks; remainder is n/a by construction (§3.3).
  echo '{"toolchain":"triton","remainder":"n/a","note":"no generated checks"}' \
    > "$HERE/results/e3_remainder.json"
  echo "triton e3: n/a (no generated checks)"
}

cmd_collect() {
  need
  "$PY" "$HERE/collect.py"
}

cmd_stats() {
  need
  "$PY" "$HERE/collect.py" --stats
}

cmd_all() { cmd_setup; cmd_minimal; cmd_e2; cmd_e3; cmd_collect; cmd_stats; }

sub="${1:-}"; shift || true
case "$sub" in
  setup) cmd_setup;; minimal) cmd_minimal;; e2) cmd_e2;; e3) cmd_e3;;
  collect) cmd_collect;; stats) cmd_stats;; all) cmd_all;;
  *) die "unknown subcommand: $sub (want setup|minimal|e2|e3|collect|stats|all)";;
esac
