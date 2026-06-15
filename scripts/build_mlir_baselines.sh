#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_workspace_layout
ensure_command cmake
ensure_command ninja

BASELINES_DIR="${BASELINES_DIR:-$WORKSPACE_ROOT/external}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$BASELINES_DIR/miniforge3}"
BASELINE_PYTHON_DIR="${BASELINE_PYTHON_DIR:-$BASELINES_DIR/triton-py310}"
IREE_BUILD_DIR_BASELINE="${IREE_BUILD_DIR_BASELINE:-$BASELINES_DIR/iree-git-v3.10.0/build-cuda-baseline}"
CUDA_ARCH="${CUDA_ARCH:-86}"
TRITON_BACKENDS="${TRITON_BACKENDS:-nvidia}"
HOST_C_COMPILER="${HOST_C_COMPILER:-/usr/bin/gcc}"
HOST_CXX_COMPILER="${HOST_CXX_COMPILER:-/usr/bin/g++}"
TRITON_APPEND_CMAKE_ARGS_DEFAULT="-DTRITON_BUILD_UT=OFF -DBUILD_TESTING=OFF -DCMAKE_C_COMPILER=${HOST_C_COMPILER} -DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}"

if [[ ! -x "$MINIFORGE_ROOT/bin/python" ]]; then
  die "Missing Miniforge Python at $MINIFORGE_ROOT/bin/python"
fi
if [[ ! -x "$MINIFORGE_ROOT/bin/conda" ]]; then
  die "Missing conda at $MINIFORGE_ROOT/bin/conda"
fi
if [[ ! -x "$HOST_C_COMPILER" ]]; then
  die "Missing host C compiler at $HOST_C_COMPILER"
fi
if [[ ! -x "$HOST_CXX_COMPILER" ]]; then
  die "Missing host C++ compiler at $HOST_CXX_COMPILER"
fi

FETCH_OUTPUT="$($SCRIPT_DIR/fetch_mlir_baselines.sh | grep -E '^(IREE_DIR|TRITON_DIR)=')"
eval "$FETCH_OUTPUT"

TRITON_LLVM_ROOT="${TRITON_LLVM_ROOT:-$TRITON_DIR/.llvm-project}"
TRITON_LLVM_SRC="${TRITON_LLVM_SRC:-$TRITON_LLVM_ROOT/src}"
TRITON_LLVM_BUILD="${TRITON_LLVM_BUILD:-$TRITON_LLVM_ROOT/build}"
TRITON_LLVM_HASH_FILE="$TRITON_DIR/cmake/llvm-hash.txt"

if [[ ! -f "$TRITON_LLVM_HASH_FILE" ]]; then
  die "Missing Triton LLVM hash file at $TRITON_LLVM_HASH_FILE"
fi

TRITON_LLVM_COMMIT_HASH="${TRITON_LLVM_COMMIT_HASH:-$(tr -d '[:space:]' < "$TRITON_LLVM_HASH_FILE")}"

log_info "Using IREE source: $IREE_DIR"
log_info "Using Triton source: $TRITON_DIR"

resolve_triton_llvm_syspath() {
  if [[ -f "$MLIR_BUILD_DIR/include/llvm/Passes/PassPlugin.h" ]] && [[ -d "$MLIR_BUILD_DIR/lib/cmake/lld" ]]; then
    printf '%s\n' "$MLIR_BUILD_DIR"
    return 0
  fi

  if [[ -f "$TRITON_LLVM_BUILD/include/llvm/Passes/PassPlugin.h" ]] && [[ -d "$TRITON_LLVM_BUILD/lib/cmake/lld" ]]; then
    printf '%s\n' "$TRITON_LLVM_BUILD"
    return 0
  fi

  log_info "Building Triton-pinned LLVM into $TRITON_LLVM_BUILD" >&2
  mkdir -p "$TRITON_LLVM_ROOT"
  LLVM_COMMIT_HASH="$TRITON_LLVM_COMMIT_HASH" \
  LLVM_PROJECT_PATH="$TRITON_LLVM_SRC" \
  LLVM_BUILD_PATH="$TRITON_LLVM_BUILD" \
  LLVM_INSTALL_PATH="$TRITON_LLVM_BUILD" \
  "$TRITON_DIR/scripts/build-llvm-project.sh" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER="$HOST_C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$HOST_CXX_COMPILER" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_TARGETS_TO_BUILD=host\;NVPTX\;AMDGPU \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_ENABLE_PROJECTS=mlir\;llvm\;lld \
    -DCMAKE_INSTALL_PREFIX="$TRITON_LLVM_BUILD" \
    -B"$TRITON_LLVM_BUILD" "$TRITON_LLVM_SRC/llvm" >&2

  printf '%s\n' "$TRITON_LLVM_BUILD"
}

if [[ ! -d "$IREE_DIR/third_party/flatcc/src/runtime" ]]; then
  die "IREE source at $IREE_DIR is incomplete; expected flatcc runtime sources"
fi

log_info "Configuring IREE in $IREE_BUILD_DIR_BASELINE"
cmake -G Ninja -S "$IREE_DIR" -B "$IREE_BUILD_DIR_BASELINE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_ENABLE_THREADING=ON \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_HAL_DRIVER_CUDA=ON \
  -DIREE_HAL_DRIVER_VULKAN=OFF \
  -DIREE_TARGET_BACKEND_CUDA=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=OFF \
  -DIREE_TARGET_BACKEND_VMVX=OFF \
  -DIREE_TARGET_BACKEND_VULKAN_SPIRV=OFF \
  -DIREE_TARGET_BACKEND_METAL_SPIRV=OFF \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_INPUT_TORCH=OFF

if [[ -x "$IREE_BUILD_DIR_BASELINE/tools/iree-compile" ]]; then
  log_info "Reusing existing iree-compile at $IREE_BUILD_DIR_BASELINE/tools/iree-compile"
else
  log_info "Building iree-compile"
  cmake --build "$IREE_BUILD_DIR_BASELINE" --target iree-compile -j "${BASELINE_BUILD_JOBS:-$(nproc)}"
fi

if [[ ! -x "$IREE_BUILD_DIR_BASELINE/tools/iree-compile" ]]; then
  die "Expected iree-compile at $IREE_BUILD_DIR_BASELINE/tools/iree-compile"
fi

log_info "Preparing dedicated Triton Python at $BASELINE_PYTHON_DIR"
TRITON_PYTHON_BIN="$MINIFORGE_ROOT/bin/python"
USE_DEDICATED_TRITON_PYTHON="${USE_DEDICATED_TRITON_PYTHON:-0}"
if [[ -x "$BASELINE_PYTHON_DIR/bin/python" ]]; then
  TRITON_PYTHON_BIN="$BASELINE_PYTHON_DIR/bin/python"
elif [[ "$USE_DEDICATED_TRITON_PYTHON" == "1" ]]; then
  if "$MINIFORGE_ROOT/bin/conda" create -y -p "$BASELINE_PYTHON_DIR" python=3.10 pip; then
    TRITON_PYTHON_BIN="$BASELINE_PYTHON_DIR/bin/python"
  else
    log_warn "Falling back to Miniforge base Python because the dedicated python=3.10 environment could not be created"
  fi
else
  log_info "Using Miniforge base Python; set USE_DEDICATED_TRITON_PYTHON=1 to create a separate environment"
fi

"$TRITON_PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$TRITON_PYTHON_BIN" -m pip install cmake ninja pybind11

TRITON_LLVM_SYSPATH="$(resolve_triton_llvm_syspath)"
log_info "Using Triton LLVM syspath: $TRITON_LLVM_SYSPATH"

log_info "Installing Triton editable package"
TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-$TRITON_APPEND_CMAKE_ARGS_DEFAULT}"
CC="$HOST_C_COMPILER" \
CXX="$HOST_CXX_COMPILER" \
TRITON_BUILD_WITH_CLANG_LLD="${TRITON_BUILD_WITH_CLANG_LLD:-0}" \
TRITON_BUILD_PROTON="${TRITON_BUILD_PROTON:-OFF}" \
TRITON_BUILD_UT="${TRITON_BUILD_UT:-OFF}" \
TRITON_CODEGEN_BACKENDS="$TRITON_BACKENDS" \
BUILD_TESTING="${BUILD_TESTING:-OFF}" \
TRITON_APPEND_CMAKE_ARGS="$TRITON_APPEND_CMAKE_ARGS" \
LLVM_SYSPATH="${LLVM_SYSPATH:-$TRITON_LLVM_SYSPATH}" \
MAX_JOBS="${MAX_JOBS:-8}" \
  "$TRITON_PYTHON_BIN" -m pip install "$TRITON_DIR" --no-build-isolation

log_info "Validating Triton import"
"$TRITON_PYTHON_BIN" -c "import triton; print(triton.__file__)"

cat <<EOF
IREE_DIR=$IREE_DIR
IREE_BUILD_DIR=$IREE_BUILD_DIR_BASELINE
IREE_COMPILE=$IREE_BUILD_DIR_BASELINE/tools/iree-compile
TRITON_DIR=$TRITON_DIR
TRITON_PYTHON=$TRITON_PYTHON_BIN
EOF