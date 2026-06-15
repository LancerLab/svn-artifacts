#!/usr/bin/env bash
# Dump IREE pipeline IR for StableHLO benchmark cases.
#
# Supported levels:
#   --level=input              — linalg-on-tensors (post-stablehlo-conversion)
#   --level=flow               — dispatch-formation IR (tiling/fusion decisions)
#   --level=stream             — stream resource allocation IR
#   --level=executable-sources — tiled linalg inside hal.executable, just before
#                                GPU codegen (nearest readable "pre-GPU" level);
#                                requires --iree-hal-target-backends=llvm-cpu
#
# Output directories:
#   benchmark/stablehlo-iree/{level}/{category}/{stem}.{level}.mlir
#
# Usage:
#   scripts/run_stablehlo_iree.sh                           # all cases, flow level
#   scripts/run_stablehlo_iree.sh --level=executable-sources  # pre-GPU kernel IR
#   scripts/run_stablehlo_iree.sh --level=flow --category=matmul

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IREE_COMPILE="${IREE_COMPILE:-/home/gxf/.local/bin/iree-compile}"
LEVEL="flow"
CATEGORY=""
# Backend used only for levels that require it (executable-sources and above)
# Use cuda to align with Choreo's GPU benchmarks
BACKEND="cuda"

for arg in "$@"; do
  case "$arg" in
    --level=*)    LEVEL="${arg#--level=}" ;;
    --category=*) CATEGORY="${arg#--category=}" ;;
    --backend=*)  BACKEND="${arg#--backend=}" ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# Levels that require an explicit HAL target backend
NEEDS_BACKEND=0
case "$LEVEL" in
  executable-sources|executable-targets|hal) NEEDS_BACKEND=1 ;;
esac

OUT_DIR="$ROOT/benchmark/stablehlo-iree/${LEVEL}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$IREE_COMPILE" ]]; then
  echo "iree-compile not found at $IREE_COMPILE"
  echo "Install with: pip3 install iree-compiler"
  exit 1
fi

echo "Level: $LEVEL${NEEDS_BACKEND:+  (backend=$BACKEND)}"
echo "Output: $OUT_DIR/"
echo ""

pass=0; fail=0
for f in "$ROOT"/benchmark/stablehlo/*/*.mlir; do
  cat_name="$(basename "$(dirname "$f")")"
  if [[ -n "$CATEGORY" && "$cat_name" != "$CATEGORY" ]]; then
    continue
  fi
  stem="$(basename "$f" .mlir)"
  mkdir -p "$OUT_DIR/$cat_name"
  out="$OUT_DIR/$cat_name/${stem}.${LEVEL}.mlir"

  # Build iree-compile argument list
  compile_args=(
    --iree-input-type=stablehlo
    "--compile-to=$LEVEL"
  )
  if [[ $NEEDS_BACKEND -eq 1 ]]; then
    compile_args+=("--iree-hal-target-backends=$BACKEND")
  fi

  if "$IREE_COMPILE" "${compile_args[@]}" "$f" -o "$out" 2>/dev/null; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    echo "FAIL: $cat_name/$stem"
    "$IREE_COMPILE" "${compile_args[@]}" "$f" -o "$out" 2>&1 | tail -3 || true
  fi
done

echo ""
echo "IREE --compile-to=$LEVEL: PASS=$pass  FAIL=$fail"
echo "Output written to $OUT_DIR/"
