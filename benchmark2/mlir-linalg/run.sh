#!/usr/bin/env bash
# ============================================================================
# benchmark2/mlir-linalg/run.sh — MLIR tensor/linalg entry-surface lane (§12.1).
#
# Toolchain pin: LLVM/MLIR 21.1.0 from croqtile's extern/ (manifest.md §5.1).
#   NOT the LLVM-18 assumption in benchmark/../scripts/*.sh — that directory
#   does not exist on this machine. `mlir-cpu-runner` was renamed `mlir-runner`
#   in LLVM 21.
#
# Surface: tensor/linalg — the level at which a SOTA MLIR user writes a kernel,
# before any bufferization. Assigned mutation class: M2 (shape contract), which
# mutation-specs.md §4 calls this surface's PRIMARY target. M1 is n/a here (no
# addressable memory before lowering) and M3 is n/a (no device contract at the
# tensor level) — see mlir-shared/lane.py SURFACES["linalg"]["na_classes"].
#
# Device: the lane executes on the GPU (CUDA backend, MLIR cuda runner), the
# same substrate as the `low` lane. DEVICE is accepted for template
# compatibility; the actual backend is `MLIR_LINALG_BACKEND` (default `cuda`).
# S12 therefore uses `compute-sanitizer --tool memcheck` over the lowered cubin,
# the checker that shares the device execution model; a host ASan binary cannot
# observe a device access. Set MLIR_LINALG_BACKEND=cpu to reproduce the old
# host-ASan baseline.
#
# All logic lives in ../mlir-shared/lane.py so the two MLIR lanes cannot drift
# apart; this file is the §12.1 contract surface only.
# ============================================================================
set -euo pipefail

TOOLCHAIN="mlir-linalg"
SURFACE="linalg"
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

# The M2 battery is deterministic — the verifier either rejects a shape mutant or
# it does not — so it needs no repetition. Kept env-overridable for symmetry with
# the low lane, where M1.1's undefined behavior does require it.
export M2_REPEAT="${M2_REPEAT:-1}"
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
  # E1: the M2 battery. mutation-specs.md §5 splits M2 coverage into a level-1
  # minimal set (layer_normalization, matmul, concat) and level-2 additions
  # (elemwise_add, softmax). --level2 widens to all five, which is what §5.1's
  # enumeration arithmetic (50 injections -> 54 mutants) assumes.
  local extra=()
  [[ -n "${LEVEL2:-}" ]] && extra+=(--level2)
  lane minimal "${extra[@]}" "$SIZEFLAG"
}

cmd_manifest() {
  # raw/mutant_manifest.json — the M2 lottery's provenance (U-4 /
  # Q-mlir-linalg-3). Always --level2 so the plan describes the same battery as
  # the committed results (the level-1 subset would emit a different 30-cell
  # plan and silently disagree with raw/mutants.jsonl).
  lane manifest --level2 "$SIZEFLAG"
}

cmd_e2()   { lane e2 "$SIZEFLAG"; }
cmd_e3()   { lane e3 "$SIZEFLAG"; }
cmd_s12()  { lane s12 "$SIZEFLAG"; }
cmd_e4()   { echo "[$TOOLCHAIN] e4 — choreo only, not applicable"; }
cmd_e5()   { echo "[$TOOLCHAIN] e5 — choreo only, not applicable"; }
cmd_collect() { lane collect; }
cmd_stats()   { lane stats; }

cmd_all() {
  # `all` must reproduce the COMMITTED census, which is 50 injections over all
  # five M2 categories. Without --level2, cmd_minimal runs the level-1 set only
  # (layer_normalization, matmul, concat) and lands on 30 injections — still
  # green, still self-consistent, but not the numbers in results/. Setting the
  # flag here rather than relying on the caller to pass it means the documented
  # "run all to re-derive everything" path actually re-derives everything.
  LEVEL2=1
  cmd_setup; cmd_e2; cmd_minimal; cmd_manifest; cmd_s12; cmd_e3; cmd_collect; cmd_stats
}

usage() {
  echo "usage: $0 {setup|minimal[--level2]|manifest|e2|e3|s12|e4|e5|collect|stats|all} [--small|--full] [--device i]"
  echo "  note: --device is accepted but ignored; the backend is MLIR_LINALG_BACKEND (default cuda, GPU)"
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
  setup) cmd_setup;; minimal) cmd_minimal;; manifest) cmd_manifest;;
  e2) cmd_e2;; e3) cmd_e3;;
  s12) cmd_s12;;
  e4) cmd_e4;; e5) cmd_e5;; collect) cmd_collect;; stats) cmd_stats;;
  all) cmd_all;; *) usage; exit 1;;
esac
