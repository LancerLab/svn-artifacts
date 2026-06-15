#!/usr/bin/env bash

if [[ -n "${SVN_TRITON_COMMON_SH:-}" ]] && declare -F resolve_triton_python >/dev/null 2>&1; then
  return 0 2>/dev/null || exit 0
fi
SVN_TRITON_COMMON_SH=1

set -euo pipefail

TRITON_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$TRITON_COMMON_DIR/mlir_common.sh"

resolve_triton_python() {
  if [[ -n "${TRITON_PYTHON:-}" ]] && [[ -x "$TRITON_PYTHON" ]]; then
    printf '%s\n' "$TRITON_PYTHON"
    return 0
  fi

  local candidate
  for candidate in \
    "$WORKSPACE_ROOT/external/triton-v3.6.0/.venv/bin/python" \
    "$WORKSPACE_ROOT/external/triton-py310/bin/python" \
    "$WORKSPACE_ROOT/external/miniforge3/bin/python"; do
    if [[ -x "$candidate" ]]; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("triton") else 1)
PY
      then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  die "Unable to locate a Python interpreter with Triton installed. Run ./scripts/build_mlir_baselines.sh first or set TRITON_PYTHON=/path/to/python."
}