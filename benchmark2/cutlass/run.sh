#!/usr/bin/env bash
# ============================================================================
# benchmark2/cutlass/run.sh — CUTLASS/CuTe candidate lane harness.
# Contract: run -> raw/ artifacts; collect -> records/stats JSON.
# Phase: vertical slice (spec-required operators; M4 + L-class battery).
# ============================================================================
set -euo pipefail

TOOLCHAIN="$(basename "$(dirname "$(realpath "$0")")")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$ROOT/$TOOLCHAIN"
RAW="$HERE/raw"
RECORDS="$HERE/records.jsonl"
RESULTS="$ROOT/results/$TOOLCHAIN"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUTLASS="${CUTLASS_ROOT:-$ROOT/../croqtile/extern/cutlass}"
PY="${PY:-/home/gxf/.tools/iree-dev-20260908-venv/bin/python}"
export IREE_CUDA_TARGET=sm_86 IREE_CUDA_FEATURES=""

die() { echo "ERROR: $*" >&2; exit 1; }

cmd_setup() {
  [ -x "$NVCC" ] || die "nvcc not found at $NVCC"
  [ -d "$CUTLASS/include/cute" ] || die "CUTLASS/CuTe not found at $CUTLASS"
  echo "[$TOOLCHAIN] setup: $("$NVCC" --version | grep -o 'release [0-9.]*')"
  echo "[$TOOLCHAIN] cutlass: $CUTLASS"
}

cmd_minimal() {
  mkdir -p "$RAW"
  "$PY" "$HERE/lane.py" all --out "$RAW" --records "$RECORDS"
}

cmd_e2() {
  # expressibility sweep: base kernels already gate all spec-required operators.
  # M1 is now a LAUNCHED battery in records.jsonl (M1 minimal operator set x
  # small/alt extents, one spec per family a..h); the old compile-only probe
  # slice is retired.
  echo "[$TOOLCHAIN] e2: see 'base' + M1 battery in lane.py (15 operators gated)"
}

cmd_e3() {
  # M3 is now a LAUNCHED battery in records.jsonl (M3 spec-required operators x
  # small/alt extents, one spec per family a..h); the old compile-only
  # descriptor/TMA/atom probe slice is retired.
  echo "[$TOOLCHAIN] e3: see M3 battery in lane.py (matmul/conv2d/batch_norm/max_pool2d)"
}

cmd_collect() {
  "$PY" "$HERE/lane.py" collect --records "$RECORDS" --collect-out "$RESULTS"
  "$PY" "$HERE/gen_manifest.py" --records "$RECORDS" --out "$RAW/mutant_manifest.json"
}
cmd_stats()   { cmd_collect; }
cmd_verify()  { "$PY" "$HERE/verify.py"; }

cmd_all() { cmd_setup; cmd_minimal; cmd_e2; cmd_e3; cmd_collect; cmd_verify; }

usage() { echo "usage: $0 {setup|minimal|e2|e3|collect|stats|verify|all} [--small|--full] [--device i]"; }

SUB="${1:-}"; shift || true
SIZE="small"; DEVICE="${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
for a in "$@"; do
  case "$a" in
    --small) SIZE="small";; --full) SIZE="full";;
    --device) DEVICE="$2"; shift;; --device=*) DEVICE="${a#*=}";;
  esac
done

mkdir -p "$RAW" "$RESULTS"
case "$SUB" in
  setup) cmd_setup;; minimal) cmd_minimal;; e2) cmd_e2;; e3) cmd_e3;;
  collect) cmd_collect;; stats) cmd_stats;; verify) cmd_verify;; all) cmd_all;;
  *) usage; exit 1;;
esac
