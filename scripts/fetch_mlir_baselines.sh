#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_command tar
ensure_command patch

if command -v curl >/dev/null 2>&1; then
  DOWNLOAD_TOOL=(curl -L --fail --retry 3 -o)
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD_TOOL=(wget -O)
else
  die "Missing required command: curl or wget"
fi

BASELINES_DIR="${BASELINES_DIR:-$WORKSPACE_ROOT/external}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$BASELINES_DIR/_downloads}"

IREE_REF="${IREE_REF:-v3.10.0}"
IREE_EXTRACT_DIR="$BASELINES_DIR/iree-${IREE_REF}"
IREE_TOPLEVEL="iree-${IREE_REF#v}"
IREE_GIT_DIR="${IREE_GIT_DIR:-$BASELINES_DIR/iree-git-${IREE_REF}}"
IREE_URL="${IREE_URL:-https://github.com/iree-org/iree.git}"

TRITON_REF="${TRITON_REF:-v3.6.0}"
TRITON_URL="${TRITON_URL:-https://github.com/triton-lang/triton/archive/refs/tags/${TRITON_REF}.tar.gz}"
TRITON_ARCHIVE="$ARCHIVE_DIR/triton-${TRITON_REF}.tar.gz"
TRITON_EXTRACT_DIR="$BASELINES_DIR/triton-${TRITON_REF}"
TRITON_TOPLEVEL="triton-${TRITON_REF#v}"
TRITON_PATCH_FILE="${TRITON_PATCH_FILE:-$WORKSPACE_ROOT/patches/triton-v3.6.0-baseline.patch}"

mkdir -p "$ARCHIVE_DIR" "$BASELINES_DIR"

download_archive() {
  local url="$1"
  local archive_path="$2"

  if [[ -f "$archive_path" ]]; then
    log_info "Archive already exists: $archive_path"
    return 0
  fi

  log_info "Downloading $url"
  "${DOWNLOAD_TOOL[@]}" "$archive_path" "$url"
}

prepare_iree_source() {
  if [[ -d "$IREE_GIT_DIR/third_party/flatcc/src/runtime" ]]; then
    log_info "Using existing complete IREE git checkout: $IREE_GIT_DIR" >&2
    printf '%s\n' "$IREE_GIT_DIR"
    return 0
  fi

  if command -v git >/dev/null 2>&1; then
    if [[ -d "$IREE_GIT_DIR/.git" ]]; then
      log_info "Refreshing IREE git checkout in $IREE_GIT_DIR" >&2
      git -C "$IREE_GIT_DIR" fetch --tags origin
    else
      log_info "Cloning IREE ${IREE_REF} into $IREE_GIT_DIR" >&2
      git clone --branch "$IREE_REF" --depth 1 "$IREE_URL" "$IREE_GIT_DIR"
    fi

    if [[ -f "$IREE_GIT_DIR/.gitmodules" ]]; then
      git -C "$IREE_GIT_DIR" submodule update --init --recursive
    fi

    if [[ -d "$IREE_GIT_DIR/third_party/flatcc/src/runtime" ]]; then
      printf '%s\n' "$IREE_GIT_DIR"
      return 0
    fi
  fi

  log_warn "Falling back to archive IREE source at $IREE_EXTRACT_DIR; this tree may be incomplete for full builds" >&2
  local archive_path="$ARCHIVE_DIR/iree-${IREE_REF}.tar.gz"
  download_archive "https://github.com/iree-org/iree/archive/refs/tags/${IREE_REF}.tar.gz" "$archive_path"
  extract_archive "$archive_path" "$IREE_EXTRACT_DIR" "$IREE_TOPLEVEL"
  printf '%s\n' "$IREE_EXTRACT_DIR"
}

extract_archive() {
  local archive_path="$1"
  local extract_dir="$2"
  local toplevel_dir="$3"

  if [[ -d "$extract_dir" ]]; then
    log_info "Extracted directory already exists: $extract_dir"
    return 0
  fi

  local temp_dir
  temp_dir="$(mktemp -d "$BASELINES_DIR/.extract.XXXXXX")"
  trap 'rm -rf "$temp_dir"' RETURN
  tar -xzf "$archive_path" -C "$temp_dir"

  if [[ ! -d "$temp_dir/$toplevel_dir" ]]; then
    die "Expected extracted directory $toplevel_dir inside $archive_path"
  fi

  mv "$temp_dir/$toplevel_dir" "$extract_dir"
  rm -rf "$temp_dir"
  trap - RETURN
  log_info "Extracted to $extract_dir"
}

apply_triton_patch_if_needed() {
  local extract_dir="$1"
  local stamp_file="$extract_dir/.svn-triton-baseline.patch-applied"

  if [[ ! -f "$TRITON_PATCH_FILE" ]]; then
    log_warn "Triton patch file not found at $TRITON_PATCH_FILE; using extracted sources as-is"
    return 0
  fi

  if [[ -f "$stamp_file" ]]; then
    return 0
  fi

  if [[ -f "$extract_dir/CMakeLists.txt" ]] && [[ -f "$extract_dir/setup.py" ]] && \
     grep -q 'TRITON_ENABLE_AMD_BACKEND' "$extract_dir/CMakeLists.txt" && \
     grep -q 'requested_backends = os.environ.get("TRITON_CODEGEN_BACKENDS"' "$extract_dir/setup.py"; then
    : > "$stamp_file"
    return 0
  fi

  log_info "Applying local Triton baseline patch"
  patch --forward -p4 -d "$extract_dir" < "$TRITON_PATCH_FILE"
  : > "$stamp_file"
}

IREE_DIR="$(prepare_iree_source)"

download_archive "$TRITON_URL" "$TRITON_ARCHIVE"
extract_archive "$TRITON_ARCHIVE" "$TRITON_EXTRACT_DIR" "$TRITON_TOPLEVEL"
apply_triton_patch_if_needed "$TRITON_EXTRACT_DIR"

cat <<EOF
IREE_DIR=$IREE_DIR
TRITON_DIR=$TRITON_EXTRACT_DIR
EOF