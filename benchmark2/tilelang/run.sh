#!/usr/bin/env bash
# benchmark2/tilelang/run.sh — TileLang lane harness (contract §12.1).
# TileLang 0.1.14 in an isolated venv (tsinghua mirror); kernels are torch-I/O
# tile programs compiled by TVM; numeric oracle = numpy reference self-check.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${TILELANG_VENV:-/home/gxf/.tools/tilelang-venv}"
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
need() { [ -x "$PY" ] || die "tilelang venv missing at $VENV"; "$PY" -c "import tilelang, torch" 2>/dev/null || die "tilelang/torch not installed"; }

cmd_setup() {
  need
  {
    echo "tilelang=$("$PY" -c 'import tilelang; print(tilelang.__version__)')"
    echo "torch=$("$PY" -c 'import torch; print(torch.__version__)')"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  } | tee "$HERE/raw/setup.txt"
}

cmd_minimal() { need; "$PY" "$HERE/driver.py" minimal --device "$DEVICE"; }
cmd_e2()      { need; "$PY" "$HERE/driver.py" e2 --size "$SIZE" --device "$DEVICE"; }
cmd_e3()      { echo '{"toolchain":"tilelang","remainder":"n/a","note":"no generated checks"}' > "$HERE/results/e3_remainder.json"; echo "tilelang e3: n/a"; }
cmd_collect() { "$PY" "$HERE/collect.py"; }
cmd_stats()   { "$PY" "$HERE/collect.py" --stats; }
cmd_all()     { cmd_setup; cmd_minimal; cmd_e2; cmd_e3; cmd_collect; cmd_stats; }

sub="${1:-}"; shift || true
case "$sub" in
  setup) cmd_setup;; minimal) cmd_minimal;; e2) cmd_e2;; e3) cmd_e3;;
  collect) cmd_collect;; stats) cmd_stats;; all) cmd_all;;
  *) die "unknown subcommand: $sub";;
esac
