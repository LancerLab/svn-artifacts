#!/usr/bin/env bash
# ============================================================================
# benchmark2/mlir-low/run.sh — MLIR memref/affine lane (§12.1 contract).
#
# Toolchain pin: LLVM/MLIR 21.1.0 from croqtile's extern/ (manifest.md §5.1).
#   NOT the LLVM-18 assumption in benchmark/../scripts/*.sh — that directory
#   does not exist on this machine. `mlir-cpu-runner` was renamed `mlir-runner`
#   in LLVM 21.
#
# Surface: memref/affine, hand-tiled kernels with an explicit boundary guard.
# Assigned mutation class: M1 (element-access / out-of-bounds). M2 and M3 are
# n/a on this surface — see mlir-shared/lane.py SURFACES["low"]["na_classes"].
#
# DEVICE is accepted for template compatibility but IGNORED: this lane is a
# CPU JIT on the host, so there is no GPU, no .gpu-lock and no exclusive run.
# S12 therefore uses LLVM AddressSanitizer on a natively linked binary rather
# than compute-sanitizer, which needs a CUDA device.
#
# All logic lives in ../mlir-shared/lane.py so the two MLIR lanes cannot drift
# apart; this file is the §12.1 contract surface only.
# ============================================================================
set -euo pipefail

TOOLCHAIN="mlir-low"
SURFACE="low"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                  # benchmark2/
LANE="$ROOT/$TOOLCHAIN"
SHARED="$ROOT/mlir-shared"
RAW="$LANE/raw"
RESULTS="$ROOT/results/$TOOLCHAIN"
DEVICE="${DEVICE:-cpu}"

# Pinned toolchain (manifest.md §5.1). Override with MLIR_LLVM_ROOT to point at
# a different LLVM build; mlirbench.py resolves every binary from it.
MLIR_LLVM_ROOT="${MLIR_LLVM_ROOT:-$HOME/dev/croqtile/extern/llvm-project}"
export MLIR_LLVM_ROOT

# M1.1 (dropped boundary mask) injects undefined behavior, so each (mutant, RTV
# mode) is run M1_REPEAT times and the verdict reduced over the distribution.
# N=16 is sized against the one measured coin-flip cell; see mlir-shared/lane.py.
export M1_REPEAT="${M1_REPEAT:-16}"
export MLIR_RUN_TIMEOUT="${MLIR_RUN_TIMEOUT:-15}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$SHARED/lane.py" ]] || die "missing $SHARED/lane.py"

lane() { python3 "$SHARED/lane.py" --surface "$SURFACE" "$@"; }

cmd_setup() {
  # One-time: verify the pinned LLVM binaries resolve, and record the resolved
  # paths + ASan target triple in raw/setup.json. Idempotent. Never installs
  # anything — the LLVM build is a precondition (manifest.md §5.1).
  mkdir -p "$RAW" "$RESULTS"
  lane setup
}

cmd_minimal() {
  # E1: the M1 battery. Every M1 record is level-1 (mutation-specs.md §5's M1
  # minimal set is exactly this lane's four composed categories); --level2 is
  # accepted for contract compatibility but adds nothing here.
  local extra=()
  [[ -n "${LEVEL2:-}" ]] && extra+=(--level2)
  lane minimal "${extra[@]}" "$SIZEFLAG"
}

cmd_e2()   { lane e2 "$SIZEFLAG"; }
cmd_e3()   { lane e3 "$SIZEFLAG"; }
cmd_s12()  { lane s12 "$SIZEFLAG"; }
cmd_e4()   { echo "[$TOOLCHAIN] e4 — choreo only, not applicable"; }
cmd_e5()   { echo "[$TOOLCHAIN] e5 — choreo only, not applicable"; }
cmd_collect() { lane collect; }
cmd_stats()   { lane stats; }

cmd_all() {
  cmd_setup; cmd_e2; cmd_minimal; cmd_s12; cmd_e3; cmd_collect; cmd_stats
}

usage() {
  echo "usage: $0 {setup|minimal[--level2]|e2|e3|s12|e4|e5|collect|stats|all} [--small|--full] [--device i]"
  echo "  note: --device is accepted but ignored (CPU JIT; no GPU dependency)"
}

SUB="${1:-}"; shift || true
SIZE="small"; LEVEL2=""
for a in "$@"; do
  case "$a" in
    --small) SIZE="small";; --full) SIZE="full";;
    --level2) LEVEL2=1;;
    --device) shift;; --device=*);;
  esac
done
SIZEFLAG="--$SIZE"

mkdir -p "$RAW" "$RESULTS"
case "$SUB" in
  setup) cmd_setup;; minimal) cmd_minimal;; e2) cmd_e2;; e3) cmd_e3;;
  s12) cmd_s12;;
  e4) cmd_e4;; e5) cmd_e5;; collect) cmd_collect;; stats) cmd_stats;;
  all) cmd_all;; *) usage; exit 1;;
esac
