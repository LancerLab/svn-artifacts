#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_workspace_layout
ensure_command cmake
ensure_command ninja
ensure_command python3

CONFIGURE_ONLY=0
INSTALL_AFTER_BUILD=0
LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-X86}"
LLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS:-ON}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(python3 - <<'PY'
import os
print(max(os.cpu_count() or 1, 1))
PY
)}"
#BUILD_TARGETS=(mlir-opt mlir-translate mlir-cpu-runner FileCheck)
BUILD_TARGETS=(mlir-opt mlir-translate FileCheck)

usage() {
  cat <<EOF
Usage: bash scripts/build_llvm_mlir.sh [options]

Options:
  --configure-only      Configure CMake but do not build targets.
  --install             Run 'cmake --install' after the build.
  --targets <list>      Semicolon-separated LLVM targets (default: ${LLVM_TARGETS_TO_BUILD}).
  --jobs <n>            Parallel build jobs (default: ${PARALLEL_JOBS}).
  --help                Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure-only)
      CONFIGURE_ONLY=1
      shift
      ;;
    --install)
      INSTALL_AFTER_BUILD=1
      shift
      ;;
    --targets)
      LLVM_TARGETS_TO_BUILD="$2"
      shift 2
      ;;
    --jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

mkdir -p "$MLIR_BUILD_DIR"

log_info "Configuring MLIR in $MLIR_BUILD_DIR"
cmake -S "$MLIR_SOURCE_DIR" -B "$MLIR_BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$MLIR_INSTALL_DIR" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS_TO_BUILD" \
  -DLLVM_ENABLE_ASSERTIONS="$LLVM_ENABLE_ASSERTIONS" \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_BUILD_TOOLS=ON

if [[ "$CONFIGURE_ONLY" -eq 1 ]]; then
  log_info "Configuration completed."
  exit 0
fi

log_info "Building MLIR targets: ${BUILD_TARGETS[*]}"
cmake --build "$MLIR_BUILD_DIR" --parallel "$PARALLEL_JOBS" --target "${BUILD_TARGETS[@]}"

if [[ "$INSTALL_AFTER_BUILD" -eq 1 ]]; then
  log_info "Installing MLIR into $MLIR_INSTALL_DIR"
  cmake --install "$MLIR_BUILD_DIR"
fi

log_info "MLIR build completed."
log_info "Tools available in: $(mlir_bin_dir)"
