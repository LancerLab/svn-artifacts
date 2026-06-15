#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_workspace_layout

MLIR_BIN_DIR="$(mlir_bin_dir)"
export PATH="$MLIR_BIN_DIR:$CHOREO_BUILD_DIR:$PATH"
export LLVM_DIR="$(llvm_cmake_dir)"
export MLIR_DIR="$(mlir_cmake_dir)"
export FILECHECK_BIN="${FILECHECK_BIN:-$(resolve_mlir_tool FileCheck)}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cat <<EOF
This script is intended to be sourced:

  source scripts/env.sh

Environment prepared with:
  WORKSPACE_ROOT=$WORKSPACE_ROOT
  CHOREO_ROOT=$CHOREO_ROOT
  LLVM_PROJECT_DIR=$LLVM_PROJECT_DIR
  MLIR_BUILD_DIR=$MLIR_BUILD_DIR
  MLIR_BIN_DIR=$MLIR_BIN_DIR
  LLVM_DIR=$LLVM_DIR
  MLIR_DIR=$MLIR_DIR
EOF
fi
