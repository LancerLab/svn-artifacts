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
  "$PY" "$HERE/probes.py" "$RAW/probes" "$HERE/records_m3.jsonl"
}

cmd_e2() {
  # expressibility sweep: base kernels already gate all spec-required operators.
  echo "[$TOOLCHAIN] e2: see 'base' stage in lane.py (8 operators gated)"
}

cmd_e3() {
  # unconditional-guard remainder: M3 descriptor/TMA/atom probe battery.
  "$PY" "$HERE/probes.py" "$RAW/probes" "$HERE/records_m3.jsonl"
}

cmd_collect() { "$PY" "$HERE/lane.py" collect --records "$RECORDS" --collect-out "$RESULTS"; }
cmd_stats()   { cmd_collect; }

cmd_all() { cmd_setup; cmd_minimal; cmd_e2; cmd_e3; cmd_collect; }

usage() { echo "usage: $0 {setup|minimal|e2|e3|collect|stats|all} [--small|--full] [--device i]"; }

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
  collect) cmd_collect;; stats) cmd_stats;; all) cmd_all;;
  *) usage; exit 1;;
esac
