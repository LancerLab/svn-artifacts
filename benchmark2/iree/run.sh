#!/usr/bin/env bash
# ============================================================================
# benchmark2/iree/run.sh — IREE worker harness (§12.1 contract).
# Toolchain pin: IREE ce36167c3be514dd165a3ecff2d377cfa8eca0c9
#   (release iree-3.12.0rc20260908, 2026-09-08 main; wheel install in venv).
# Arch is a first-class knob: IREE_CUDA_TARGET (sm_120 locally; sm_86/sm_90
#   for the final host) + IREE_CUDA_FEATURES (+ptx87 required for sm_120).
#   See benchmark2/manifest.md §5 (iree lane).
# ============================================================================
set -euo pipefail

TOOLCHAIN="iree"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                  # benchmark2/
LANE="$ROOT/$TOOLCHAIN"
RAW="$LANE/raw"
RESULTS="$ROOT/results/$TOOLCHAIN"
DEVICE="${DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"

# --- pinned toolchain -------------------------------------------------------
IREE_RELEASE_TAG="iree-3.12.0rc20260908"
IREE_COMMIT="ce36167c3be514dd165a3ecff2d377cfa8eca0c9"
IREE_VENV_BASE="${IREE_VENV_BASE:-/home/gxf/.tools/iree-dev-20260908-venv}"
IREE_BIN_DIR="$IREE_VENV_BASE/bin"
IREE_COMPILE="${IREE_COMPILE:-$IREE_BIN_DIR/iree-compile}"
IREE_RUN_MODULE="${IREE_RUN_MODULE:-$IREE_BIN_DIR/iree-run-module}"

# --- arch knobs (recorded in every record) ----------------------------------
IREE_CUDA_TARGET="${IREE_CUDA_TARGET:-sm_120}"
IREE_CUDA_FEATURES="${IREE_CUDA_FEATURES:-}"
if [[ "$IREE_CUDA_TARGET" == "sm_120" && -z "$IREE_CUDA_FEATURES" ]]; then
  IREE_CUDA_FEATURES="+ptx87"
fi

die() { echo "ERROR: $*" >&2; exit 1; }

iree_compile_flags() {
  local flags=(--iree-hal-target-backends=cuda --iree-cuda-target="$IREE_CUDA_TARGET")
  if [[ -n "$IREE_CUDA_FEATURES" ]]; then flags+=(--iree-cuda-target-features="$IREE_CUDA_FEATURES"); fi
  printf '%s\n' "${flags[@]}"
}

# Compose a category's kernels from benchmark2/settings (worker-owned
# generator; one linalg-on-tensors module per settings case). Implemented in
# gen_kernels.py; emits <LANE>/kernels/<category>/<stem>.mlir.
compose_kernels() { # $1 = category
  python3 "$LANE/gen_kernels.py" --settings-dir "$ROOT/settings" --out-dir "$LANE/kernels" "$1"
}

# Gate: compile + reference-run a kernel. Record a raw `kernel` JSON record.
gate_kernel() { # $1 = category, $2 = stem (settings case basename), $3 = size
  local category="$1" stem="$2" size="$3"
  local mlir="$LANE/kernels/$category/$stem.mlir"
  local vmfb="$RAW/$category/$stem.$size.vmfb"
  local clog="$RAW/$category/$stem.$size.compile.log" rlog="$RAW/$category/$stem.$size.run.log"
  mkdir -p "$RAW/$category"
  [[ -f "$mlir" ]] || die "kernel missing: $mlir (run compose first)"
  # compile stage
  if "$IREE_COMPILE" "$mlir" $(iree_compile_flags) -o "$vmfb" >"$clog" 2>&1; then
    COMPILE="ok"
  elif grep -qiE "error" "$clog"; then
    COMPILE="fail"
  else
    COMPILE="fail"
  fi
  # run stage (only if compiled)
  RUN="-" REF="pass"
  if [[ "$COMPILE" == "ok" ]]; then
    if "$IREE_RUN_MODULE" --module="$vmfb" --device=cuda --function="main" >"$rlog" 2>&1; then
      RUN="ok"
    else
      RUN="crash"
    fi
  fi
  echo "$COMPILE/$RUN/$REF"
}

cmd_setup() {
  # One-time: ensure the pinned IREE venv + binaries exist. Idempotent.
  # (Never installs system CUDA — driver 580.95.05 / CUDA 13.0 is the shared
  # machine precondition; see manifest.md §2/§5.)
  if [[ ! -x "$IREE_COMPILE" || ! -x "$IREE_RUN_MODULE" ]]; then
    echo "[iree] setup: creating venv + installing $IREE_RELEASE_TAG (commit ${IREE_COMMIT:0:8})"
    python3 -m venv "$IREE_VENV_BASE"
    "$IREE_BIN_DIR/pip" install --quiet --upgrade pip
    "$IREE_BIN_DIR/pip" install --quiet \
      "https://github.com/iree-org/iree/releases/download/$IREE_RELEASE_TAG/iree_base_compiler-3.12.0rc20260908-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
      "https://github.com/iree-org/iree/releases/download/$IREE_RELEASE_TAG/iree_base_runtime-3.12.0rc20260908-cp310-cp310-manylinux_2_28_x86_64.whl"
  fi
  mkdir -p "$RAW" "$RESULTS"
  cat > "$RAW/setup.json" <<EOF
{"toolchain": "iree", "release": "$IREE_RELEASE_TAG", "commit": "$IREE_COMMIT",
 "compile": "$IREE_COMPILE", "run_module": "$IREE_RUN_MODULE",
 "cuda_target": "$IREE_CUDA_TARGET", "cuda_features": "$IREE_CUDA_FEATURES",
 "gpu_device": "$DEVICE"}
EOF
  # Compose kernels from settings (idempotent; skips already-emitted cases).
  python3 "$LANE/gen_kernels.py"
  echo "[iree] setup done: $("$IREE_COMPILE" --version | head -1)"
}

cmd_minimal() {
  # E1: measured M2 entry-shape battery. --level2 adds the expressible level-2
  # M2 addition on elemwise_add; M1/M3 => n/a (see mutants/README.md).
  local lv2=""
  [[ -n "${LEVEL2:-}" ]] && lv2="--level2"
  python3 "$LANE/lane.py" minimal $lv2
}

cmd_e2()   { local f=""; [[ "$SIZE" == "full" ]] && f="--full"; python3 "$LANE/lane.py" e2 $f; }
cmd_e3()   { python3 "$LANE/lane.py" e3; python3 "$LANE/lane.py" expressibility; }
cmd_semcheck() { python3 "$LANE/semcheck.py"; }
cmd_oracle()   { python3 "$LANE/mutant_oracle.py"; }
cmd_e4()   { echo "[iree] e4 — choreo only, not applicable"; }
cmd_e5()   { echo "[iree] e5 — choreo only, not applicable"; }
cmd_collect() { python3 "$LANE/collect_stats.py" collect; }
cmd_stats()   { python3 "$LANE/collect_stats.py" stats; }

cmd_all() {
  cmd_setup; cmd_e2; cmd_minimal; cmd_e3; python3 "$LANE/lane.py" s12
  cmd_semcheck; cmd_oracle; cmd_collect; cmd_stats
}

usage() { echo "usage: $0 {setup|minimal[--level2]|e2|e3|semcheck|oracle|s12|e4|e5|collect|stats|all} [--small|--full] [--device i]"; }

SUB="${1:-}"; shift || true
SIZE="small"; LEVEL2=""
for a in "$@"; do
  case "$a" in
    --small) SIZE="small";; --full) SIZE="full";;
    --level2) LEVEL2=1;;
    --device) DEVICE="$2"; shift;; --device=*) DEVICE="${a#*=}";;
  esac
done

mkdir -p "$RAW" "$RESULTS"
case "$SUB" in
  setup) cmd_setup;; minimal) cmd_minimal;; e2) cmd_e2;; e3) cmd_e3;;
  semcheck) cmd_semcheck;; oracle) cmd_oracle;; s12) python3 "$LANE/lane.py" s12;;
  e4) cmd_e4;; e5) cmd_e5;; collect) cmd_collect;; stats) cmd_stats;;
  all) cmd_all;; *) usage; exit 1;;
esac
