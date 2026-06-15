#!/usr/bin/env bash

if [[ -n "${SVN_MLIR_COMMON_SH:-}" ]] && declare -F ensure_command >/dev/null 2>&1; then
  return 0 2>/dev/null || exit 0
fi
SVN_MLIR_COMMON_SH=1

set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$COMMON_DIR/.." && pwd)}"
export CHOREO_ROOT="${CHOREO_ROOT:-$WORKSPACE_ROOT/choreo}"
export LLVM_PROJECT_DIR="${LLVM_PROJECT_DIR:-$WORKSPACE_ROOT/llvm-project}"
export LLVM_BRANCH="${LLVM_BRANCH:-release/22.x}"
export MLIR_SOURCE_DIR="${MLIR_SOURCE_DIR:-$LLVM_PROJECT_DIR/llvm}"
export MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-$WORKSPACE_ROOT/build/llvm-release}"
export MLIR_INSTALL_DIR="${MLIR_INSTALL_DIR:-$MLIR_BUILD_DIR/install}"
export CHOREO_BUILD_DIR="${CHOREO_BUILD_DIR:-$CHOREO_ROOT/build-release}"
export CHOREO_BIN="${CHOREO_BIN:-$CHOREO_ROOT/choreo}"

log_info() {
  printf '[INFO] %s\n' "$*"
}

log_warn() {
  printf '[WARN] %s\n' "$*" >&2
}

log_error() {
  printf '[ERROR] %s\n' "$*" >&2
}

die() {
  log_error "$*"
  exit 1
}

ensure_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_workspace_layout() {
  [[ -d "$WORKSPACE_ROOT" ]] || die "Missing workspace root at $WORKSPACE_ROOT"
  [[ -d "$CHOREO_ROOT" ]] || die "Missing Choreo checkout at $CHOREO_ROOT"
  [[ -d "$LLVM_PROJECT_DIR" ]] || die "Missing llvm-project checkout at $LLVM_PROJECT_DIR"
}

mlir_bin_dir() {
  if [[ -x "$MLIR_INSTALL_DIR/bin/mlir-opt" ]]; then
    printf '%s\n' "$MLIR_INSTALL_DIR/bin"
  else
    printf '%s\n' "$MLIR_BUILD_DIR/bin"
  fi
}

llvm_cmake_dir() {
  if [[ -d "$MLIR_INSTALL_DIR/lib/cmake/llvm" ]]; then
    printf '%s\n' "$MLIR_INSTALL_DIR/lib/cmake/llvm"
  else
    printf '%s\n' "$MLIR_BUILD_DIR/lib/cmake/llvm"
  fi
}

mlir_cmake_dir() {
  if [[ -d "$MLIR_INSTALL_DIR/lib/cmake/mlir" ]]; then
    printf '%s\n' "$MLIR_INSTALL_DIR/lib/cmake/mlir"
  else
    printf '%s\n' "$MLIR_BUILD_DIR/lib/cmake/mlir"
  fi
}

resolve_choreo_bin() {
  if [[ -x "$CHOREO_BIN" ]]; then
    printf '%s\n' "$CHOREO_BIN"
    return 0
  fi

  if [[ -x "$CHOREO_BUILD_DIR/choreo" ]]; then
    printf '%s\n' "$CHOREO_BUILD_DIR/choreo"
    return 0
  fi

  die "Choreo binary not found. Build Choreo first (for example: make -C choreo release)."
}

resolve_mlir_tool() {
  local tool_name="$1"
  local candidate
  candidate="$(mlir_bin_dir)/$tool_name"
  if [[ -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  if command -v "$tool_name" >/dev/null 2>&1; then
    command -v "$tool_name"
    return 0
  fi

  die "Unable to locate $tool_name. Build MLIR first (for example: make mlir-build)."
}
