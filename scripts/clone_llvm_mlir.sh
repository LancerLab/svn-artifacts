#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_command git

REMOTE_URL="${LLVM_REMOTE_URL:-https://github.com/llvm/llvm-project.git}"
CLONE_DEPTH="${LLVM_CLONE_DEPTH:-1}"

if [[ -d "$LLVM_PROJECT_DIR/.git" ]]; then
  log_info "llvm-project already exists at $LLVM_PROJECT_DIR"
  log_info "Current branch: $(git -C "$LLVM_PROJECT_DIR" rev-parse --abbrev-ref HEAD)"
  log_info "Current commit: $(git -C "$LLVM_PROJECT_DIR" rev-parse HEAD)"
  exit 0
fi

mkdir -p "$(dirname "$LLVM_PROJECT_DIR")"
log_info "Cloning $REMOTE_URL into $LLVM_PROJECT_DIR"
git clone --depth "$CLONE_DEPTH" --branch "$LLVM_BRANCH" "$REMOTE_URL" "$LLVM_PROJECT_DIR"
log_info "Cloned branch: $(git -C "$LLVM_PROJECT_DIR" rev-parse --abbrev-ref HEAD)"
log_info "Commit: $(git -C "$LLVM_PROJECT_DIR" rev-parse HEAD)"
