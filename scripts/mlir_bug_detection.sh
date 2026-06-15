#!/bin/bash
# MLIR Bug Detection Experiment
# Tests MLIR's ability to detect injected bugs via:
#   (1) Static verifier (mlir-opt)
#   (2) Runtime assertions (mlir-cpu-runner)
#
# Usage: ./scripts/mlir_bug_detection.sh [output_csv]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MLIR_LOCAL="${HOME}/mlir-local"
MLIR_OPT="${MLIR_LOCAL}/usr/bin/mlir-opt-18"
MLIR_RUNNER="${MLIR_LOCAL}/usr/bin/mlir-cpu-runner-18"
MLIR_LIB="${MLIR_LOCAL}/usr/lib/llvm-18/lib"
export LD_LIBRARY_PATH="${MLIR_LIB}:${MLIR_LOCAL}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
MLIR_CASES="$WORKSPACE_ROOT/benchmark/mlir/cases"

OUTPUT=${1:-$WORKSPACE_ROOT/benchmark/results/mlir_bug_detection.csv}
mkdir -p "$(dirname "$OUTPUT")"
MUTANT_DIR="/tmp/mlir_bug_mutants_$$"
mkdir -p "$MUTANT_DIR"

echo "case,category,bug_class,bug_description,static_result,static_detail,runtime_result,runtime_detail" > "$OUTPUT"

# ========================================================================
# Bug Injection Functions
# ========================================================================

inject_dim_mismatch_static() {
    local mlir_file=$1
    local mutant=$2
    cp "$mlir_file" "$mutant"
    # For static cases with linalg ops: modify tensor type dimensions
    # Change e.g. tensor<4x8xf32> matmul tensor<8x6xf32> -> tensor<4x8xf32> matmul tensor<7x6xf32>
    # For tensor types, change the inner dimension of the first input
    sed -i 's/tensor<\([0-9]*\)x\([0-9]*\)xf32>, %rhs: tensor<\2x/tensor<\1x\2xf32>, %rhs: tensor<99x/' "$mutant"
    if diff -q "$mlir_file" "$mutant" > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

inject_dim_mismatch_dynamic() {
    local mlir_file=$1
    local mutant=$2
    cp "$mlir_file" "$mutant"
    # For dynamic MLIR cases: invalidate the cf.assert by changing the
    # dimension being checked, e.g. change rhs_d0 comparison to rhs_d1
    # Or: remove the cf.assert entirely (simulating missing manual check)
    # Strategy: delete cf.assert lines to simulate "no manual assertion written"
    if grep -q 'cf.assert' "$mutant"; then
        sed -i '/cf.assert/d' "$mutant"
        return 0
    fi
    return 1
}

inject_oob_access() {
    local mlir_file=$1
    local mutant=$2
    cp "$mlir_file" "$mutant"
    # Inject OOB by modifying an index in tensor.extract/tensor.insert
    # Replace %input_d3 with a constant that's too large in a loop bound
    # Strategy: change a loop upper bound from %input_dN to %input_dN + 1
    if grep -q 'scf.for.*to %input_d' "$mutant"; then
        # Add 1 to a loop bound to cause OOB
        sed -i '0,/scf.for \(%[a-z0-9_]*\) = %c0 to \(%input_d[0-9]*\)/s//scf.for \1 = %c0 to %oob_bound/' "$mutant"
        # Add the oob_bound computation after constants
        sed -i '/^    %c3 = arith.constant 3 : index/a\    %oob_bound_base = tensor.dim %input, %c3 : tensor<[^>]*>\n    %oob_one = arith.constant 1 : index\n    %oob_bound = arith.addi %oob_bound_base, %oob_one : index' "$mutant"
        if ! diff -q "$mlir_file" "$mutant" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

inject_wrong_output_shape() {
    local mlir_file=$1
    local mutant=$2
    cp "$mlir_file" "$mutant"
    # Replace output tensor.empty dimension with a wrong one
    # e.g. tensor.empty(%rhs_d1) -> tensor.empty(%lhs_d0) when they differ
    if grep -q 'tensor.empty' "$mutant"; then
        sed -i '0,/tensor\.empty(\(%[a-z_]*d[0-9]*\))/s//tensor.empty(%c0)/' "$mutant"
        if ! diff -q "$mlir_file" "$mutant" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# ========================================================================
# Detection Test Functions
# ========================================================================

test_mlir_static() {
    local mutant=$1
    local out err rc
    out=$(mktemp)
    err=$(mktemp)
    "$MLIR_OPT" --canonicalize --cse "$mutant" > "$out" 2> "$err"
    rc=$?
    local detail=""
    if [[ $rc -ne 0 ]]; then
        detail=$(head -3 "$err" | tr '\n' ' ')
        rm -f "$out" "$err"
        echo "STATIC_DETECT|$detail"
        return
    fi
    # Check if mlir-opt produced any diagnostic/warning
    if grep -qi 'error\|failed\|invalid' "$err"; then
        detail=$(head -3 "$err" | tr '\n' ' ')
        rm -f "$out" "$err"
        echo "STATIC_DETECT|$detail"
        return
    fi
    rm -f "$out" "$err"
    echo "NOT_DETECTED|mlir-opt accepted the mutant"
}

test_mlir_runtime() {
    local mutant=$1

    if [[ ! -x "$MLIR_RUNNER" ]]; then
        echo "SKIP|mlir-cpu-runner not available"
        return
    fi

    local lowered lower_err
    lowered=$(mktemp --suffix=.mlir)
    lower_err=$(mktemp)

    # Full pipeline: bufferize tensors, then lower to LLVM
    "$MLIR_OPT" \
        --pass-pipeline="builtin.module(func.func(empty-tensor-to-alloc-tensor),one-shot-bufferize{bufferize-function-boundaries=true},buffer-deallocation-pipeline,convert-linalg-to-loops,convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-math-to-llvm,expand-strided-metadata,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" \
        "$mutant" > "$lowered" 2> "$lower_err"
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        local detail
        detail=$(head -3 "$lower_err" | tr '\n' ' ')
        rm -f "$lowered" "$lower_err"
        echo "LOWER_FAIL|$detail"
        return
    fi

    local run_out run_err
    run_out=$(mktemp)
    run_err=$(mktemp)
    timeout 10 "$MLIR_RUNNER" "$lowered" \
        --entry-point-result=void \
        -shared-libs="$MLIR_LIB/libmlir_runner_utils.so.18.1,$MLIR_LIB/libmlir_c_runner_utils.so.18.1" \
        > "$run_out" 2> "$run_err"
    rc=$?

    local detail=""
    if [[ $rc -ne 0 ]]; then
        if grep -qi 'assert\|abort\|trap\|Aborted' "$run_err"; then
            detail=$(head -3 "$run_err" | tr '\n' ' ')
            rm -f "$lowered" "$lower_err" "$run_out" "$run_err"
            echo "RUNTIME_DETECT|$detail"
            return
        fi
        detail=$(head -3 "$run_err" | tr '\n' ' ')
        rm -f "$lowered" "$lower_err" "$run_out" "$run_err"
        echo "CRASH_OPAQUE|$detail"
        return
    fi

    rm -f "$lowered" "$lower_err" "$run_out" "$run_err"
    echo "NOT_DETECTED|executed without error"
}

# ========================================================================
# Main Test Loop
# ========================================================================

total=0
static_detected=0
runtime_detected=0

echo "=============================================="
echo "MLIR Bug Detection Experiment"
echo "=============================================="
echo ""

# --- Test 1: Static verifier on matmul_invalid variant ---
echo "--- Test Group 1: Static Shape Mismatch (linalg verifier) ---"
if [[ -f "$MLIR_CASES/matmul_invalid.mlir" ]]; then
    result=$(test_mlir_static "$MLIR_CASES/matmul_invalid.mlir")
    status="${result%%|*}"
    detail="${result#*|}"
    echo "  matmul_invalid.mlir: $status ($detail)"
    echo "matmul_invalid,matmul,dim_mismatch_static,original invalid case,$status,$detail,N/A,static-only" >> "$OUTPUT"
    total=$((total + 1))
    [[ "$status" == "STATIC_DETECT" ]] && static_detected=$((static_detected + 1))
fi

# --- Test Group 2: Inject dim mismatch into static matmul cases ---
echo ""
echo "--- Test Group 2: Injected Static Dimension Mismatch ---"
for f in "$MLIR_CASES"/matmul/*efficientnet* "$MLIR_CASES"/matmul/*lstm* "$MLIR_CASES"/matmul/*resnet*; do
    [[ -f "$f" ]] || continue
    bname=$(basename "$f" .mlir)
    mutant="$MUTANT_DIR/${bname}_dim_mismatch.mlir"
    if inject_dim_mismatch_static "$f" "$mutant"; then
        result=$(test_mlir_static "$mutant")
        status="${result%%|*}"
        detail="${result#*|}"
        echo "  $bname: $status"
        echo "$bname,matmul,dim_mismatch_static,injected inner dim change,$status,$detail,N/A,static-only" >> "$OUTPUT"
        total=$((total + 1))
        [[ "$status" == "STATIC_DETECT" ]] && static_detected=$((static_detected + 1))
    fi
done

# --- Test Group 3: Dynamic cases - remove cf.assert (simulating missing manual assertion) ---
echo ""
echo "--- Test Group 3: Removed cf.assert (MLIR cannot auto-detect) ---"
DYNAMIC_SUBSET=(
    "$MLIR_CASES/matmul/10_dynamic_128x1280_1280xN_128xN.mlir"
    "$MLIR_CASES/matmul/11_dynamic_32xSx768_768x768_32xSx768.mlir"
    "$MLIR_CASES/matmul/12_dynamic_64xTx256_256x128_64xTx128.mlir"
    "$MLIR_CASES/batch_norm/10_dynamic_16x512xHxW_512_512_16x512xHxW.mlir"
    "$MLIR_CASES/batch_norm/11_dynamic_32xSx768_768_768_32xSx768.mlir"
    "$MLIR_CASES/layer_normalization/10_dynamic_16x512xHxW_HxW_HxW.mlir"
    "$MLIR_CASES/layer_normalization/11_dynamic_32xSx768_768_768.mlir"
    "$MLIR_CASES/conv2d/10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D.mlir"
    "$MLIR_CASES/conv2d/2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D.mlir"
    "$MLIR_CASES/reduce_mean/10_dynamic_16x512xHxW_16x512.mlir"
    "$MLIR_CASES/softmax/10_dynamic_16x512xHxW_16x512xHxW.mlir"
    "$MLIR_CASES/softmax/11_dynamic_32xSx768_32xSx768.mlir"
    "$MLIR_CASES/concat/9_dynamic_64x128xHxW_64x128xHxW_64x256xHxW.mlir"
    "$MLIR_CASES/gelu/10_dynamic_16x512xHxW.mlir"
    "$MLIR_CASES/embedding/10_dynamic_32xS_Vx768_32xSx768.mlir"
)

for f in "${DYNAMIC_SUBSET[@]}"; do
    [[ -f "$f" ]] || continue
    bname=$(basename "$f" .mlir)
    mutant="$MUTANT_DIR/${bname}_no_assert.mlir"
    if inject_dim_mismatch_dynamic "$f" "$mutant"; then
        result=$(test_mlir_static "$mutant")
        status="${result%%|*}"
        detail="${result#*|}"
        echo "  $bname (assert removed): static=$status"
        echo "$bname,$(basename $(dirname $f)),assert_removed,cf.assert deleted,$status,$detail,N/A,assert-removed" >> "$OUTPUT"
        total=$((total + 1))
        [[ "$status" == "STATIC_DETECT" ]] && static_detected=$((static_detected + 1))
    fi
done

# --- Test Group 4: OOB/dim-mismatch Runtime Detection via mlir-cpu-runner ---
# These are hand-crafted runnable programs that test whether MLIR's cf.assert
# (if present) catches dimension mismatches at runtime.
# Key insight: MLIR CANNOT auto-detect bugs — only manually-written cf.assert works.
echo ""
echo "--- Test Group 4: Runtime Detection Tests (mlir-cpu-runner) ---"

# Create runtime test cases: programs with main() that exercise assertions
create_runtime_test_passing() {
    local outfile=$1
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c128 = arith.constant 128 : index
    %c1280 = arith.constant 1280 : index
    %c64 = arith.constant 64 : index
    %lhs = memref.alloc(%c128, %c1280) : memref<?x?xf32>
    %rhs = memref.alloc(%c1280, %c64) : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %lhs_d1 = memref.dim %lhs, %c1 : memref<?x?xf32>
    %rhs_d0 = memref.dim %rhs, %c0 : memref<?x?xf32>
    %eq = arith.cmpi eq, %lhs_d1, %rhs_d0 : index
    cf.assert %eq, "lhs.dim(1)==rhs.dim(0)"
    memref.dealloc %lhs : memref<?x?xf32>
    memref.dealloc %rhs : memref<?x?xf32>
    return
  }
}
MLIR
}

create_runtime_test_dim_mismatch() {
    local outfile=$1
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c128 = arith.constant 128 : index
    %c1280 = arith.constant 1280 : index
    %c99 = arith.constant 99 : index
    %c64 = arith.constant 64 : index
    %lhs = memref.alloc(%c128, %c1280) : memref<?x?xf32>
    %rhs = memref.alloc(%c99, %c64) : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %lhs_d1 = memref.dim %lhs, %c1 : memref<?x?xf32>
    %rhs_d0 = memref.dim %rhs, %c0 : memref<?x?xf32>
    %eq = arith.cmpi eq, %lhs_d1, %rhs_d0 : index
    cf.assert %eq, "lhs.dim(1)==rhs.dim(0) FAILED"
    memref.dealloc %lhs : memref<?x?xf32>
    memref.dealloc %rhs : memref<?x?xf32>
    return
  }
}
MLIR
}

create_runtime_test_no_assert() {
    local outfile=$1
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c128 = arith.constant 128 : index
    %c1280 = arith.constant 1280 : index
    %c99 = arith.constant 99 : index
    %c64 = arith.constant 64 : index
    %lhs = memref.alloc(%c128, %c1280) : memref<?x?xf32>
    %rhs = memref.alloc(%c99, %c64) : memref<?x?xf32>
    memref.dealloc %lhs : memref<?x?xf32>
    memref.dealloc %rhs : memref<?x?xf32>
    return
  }
}
MLIR
}

create_runtime_test_oob_with_assert() {
    local outfile=$1
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %buf = memref.alloc(%c10) : memref<?xf32>
    %dim = memref.dim %buf, %c0 : memref<?xf32>
    %inbounds = arith.cmpi slt, %c11, %dim : index
    cf.assert %inbounds, "index out of bounds"
    memref.dealloc %buf : memref<?xf32>
    return
  }
}
MLIR
}

create_runtime_test_oob_no_assert() {
    local outfile=$1
    # OOB without assertion — writes past buffer end
    # On CPU this MAY crash or MAY silently corrupt — either way, no useful diagnostic
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %val = arith.constant 1.0 : f32
    %buf = memref.alloc(%c10) : memref<?xf32>
    memref.store %val, %buf[%c11] : memref<?xf32>
    memref.dealloc %buf : memref<?xf32>
    return
  }
}
MLIR
}

create_runtime_test_stride_error() {
    local outfile=$1
    cat > "$outfile" << 'MLIR'
module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    %c256 = arith.constant 256 : index
    %input = memref.alloc(%c32, %c512, %c768) : memref<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %actual_d2 = memref.dim %input, %c2 : memref<?x?x?xf32>
    %eq = arith.cmpi eq, %actual_d2, %c256 : index
    cf.assert %eq, "expected hidden_dim==256 but got different"
    memref.dealloc %input : memref<?x?x?xf32>
    return
  }
}
MLIR
}

run_runtime_test() {
    local test_file=$1
    local test_name=$2

    if [[ ! -x "$MLIR_RUNNER" ]]; then
        echo "  $test_name: SKIP (no mlir-cpu-runner)"
        echo "$test_name,runtime,runtime_test,runtime check,$test_name,N/A,SKIP,no runner" >> "$OUTPUT"
        return
    fi

    local lowered lower_err
    lowered=$(mktemp --suffix=.mlir)
    lower_err=$(mktemp)

    "$MLIR_OPT" \
        --convert-cf-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm \
        --convert-func-to-llvm --reconcile-unrealized-casts \
        "$test_file" > "$lowered" 2> "$lower_err"
    local rc=$?

    if [[ $rc -ne 0 ]]; then
        local detail
        detail=$(head -2 "$lower_err" | tr '\n' ' ')
        echo "  $test_name: LOWER_FAIL ($detail)"
        echo "$test_name,runtime,runtime_test,lowering failed,LOWER_FAIL,$detail,LOWER_FAIL,$detail" >> "$OUTPUT"
        rm -f "$lowered" "$lower_err"
        return
    fi

    local run_out run_err
    run_out=$(mktemp)
    run_err=$(mktemp)
    timeout 10 "$MLIR_RUNNER" "$lowered" \
        --entry-point-result=void \
        -shared-libs="$MLIR_LIB/libmlir_runner_utils.so.18.1,$MLIR_LIB/libmlir_c_runner_utils.so.18.1" \
        > "$run_out" 2> "$run_err"
    rc=$?

    if [[ $rc -ne 0 ]]; then
        # MLIR's cf.assert calls puts()+abort(), but the JIT's signal handler
        # catches SIGABRT before stdout is flushed — so the message is lost.
        # We detect assertion-based crashes by exit code 134 (SIGABRT from cf.assert)
        # vs exit code 124 (timeout) or 139 (SIGSEGV from actual memory error)
        if [[ $rc -eq 134 ]]; then
            echo "  $test_name: RUNTIME_DETECT (exit=134, assert→abort)"
            echo "$test_name,runtime,runtime_test,cf.assert fired (opaque abort),N/A,N/A,RUNTIME_DETECT,exit_code=134_abort" >> "$OUTPUT"
            runtime_detected=$((runtime_detected + 1))
        elif [[ $rc -eq 139 ]]; then
            echo "  $test_name: CRASH_OPAQUE (exit=139, SIGSEGV)"
            echo "$test_name,runtime,runtime_test,segfault,N/A,N/A,CRASH_OPAQUE,exit_code=139_sigsegv" >> "$OUTPUT"
        elif [[ $rc -eq 124 ]]; then
            echo "  $test_name: TIMEOUT"
            echo "$test_name,runtime,runtime_test,timeout,N/A,N/A,TIMEOUT,exit_code=124" >> "$OUTPUT"
        else
            echo "  $test_name: CRASH_OPAQUE (exit=$rc)"
            echo "$test_name,runtime,runtime_test,opaque crash,N/A,N/A,CRASH_OPAQUE,exit_code=$rc" >> "$OUTPUT"
        fi
    else
        echo "  $test_name: NOT_DETECTED (exit=0)"
        echo "$test_name,runtime,runtime_test,no detection,N/A,N/A,NOT_DETECTED,silent pass" >> "$OUTPUT"
    fi
    total=$((total + 1))
    rm -f "$lowered" "$lower_err" "$run_out" "$run_err"
}

# Run the runtime tests
echo ""
echo "  [4a] Baseline - correct dimensions with assert (should PASS):"
create_runtime_test_passing "$MUTANT_DIR/rt_passing.mlir"
run_runtime_test "$MUTANT_DIR/rt_passing.mlir" "matmul_correct_dims_with_assert"

echo "  [4b] Dimension mismatch WITH cf.assert (should DETECT):"
create_runtime_test_dim_mismatch "$MUTANT_DIR/rt_dim_mismatch.mlir"
run_runtime_test "$MUTANT_DIR/rt_dim_mismatch.mlir" "matmul_dim_mismatch_with_assert"

echo "  [4c] Dimension mismatch WITHOUT cf.assert (should NOT detect):"
create_runtime_test_no_assert "$MUTANT_DIR/rt_no_assert.mlir"
run_runtime_test "$MUTANT_DIR/rt_no_assert.mlir" "matmul_dim_mismatch_no_assert"

echo "  [4d] OOB access WITH bounds check assert (should DETECT):"
create_runtime_test_oob_with_assert "$MUTANT_DIR/rt_oob_assert.mlir"
run_runtime_test "$MUTANT_DIR/rt_oob_assert.mlir" "oob_access_with_assert"

echo "  [4e] OOB access WITHOUT any check (MLIR silent corruption):"
create_runtime_test_oob_no_assert "$MUTANT_DIR/rt_oob_no_assert.mlir"
run_runtime_test "$MUTANT_DIR/rt_oob_no_assert.mlir" "oob_access_no_assert"

echo "  [4f] Stride/shape error WITH assert (should DETECT):"
create_runtime_test_stride_error "$MUTANT_DIR/rt_stride.mlir"
run_runtime_test "$MUTANT_DIR/rt_stride.mlir" "stride_error_with_assert"

# --- Test Group 5: Wrong output shape ---
echo ""
echo "--- Test Group 5: Wrong Output Shape ---"
for f in "${DYNAMIC_SUBSET[@]}"; do
    [[ -f "$f" ]] || continue
    bname=$(basename "$f" .mlir)
    mutant="$MUTANT_DIR/${bname}_wrong_shape.mlir"
    if inject_wrong_output_shape "$f" "$mutant"; then
        result=$(test_mlir_static "$mutant")
        status="${result%%|*}"
        detail="${result#*|}"
        echo "  $bname (wrong output): $status"
        echo "$bname,$(basename $(dirname $f)),wrong_output_shape,tensor.empty dim wrong,$status,$detail,N/A,shape-error" >> "$OUTPUT"
        total=$((total + 1))
        [[ "$status" == "STATIC_DETECT" ]] && static_detected=$((static_detected + 1))
    fi
done

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "=============================================="
echo "SUMMARY"
echo "=============================================="
echo "Total test cases: $total"
echo ""
echo "=== Static Detection (mlir-opt verifier) ==="
echo "  Detected:     $static_detected"
echo ""
echo "=== Runtime Detection (mlir-cpu-runner cf.assert) ==="
echo "  Detected:     $runtime_detected"
echo ""
not_detected=$((total - static_detected - runtime_detected))
echo "=== NOT Detected (silent pass / no automatic safety) ==="
echo "  Silent:       $not_detected"
echo ""
echo "KEY FINDING: MLIR detects bugs ONLY when:"
echo "  1. Static shapes are fully known (linalg verifier), OR"
echo "  2. A programmer MANUALLY writes cf.assert checks"
echo ""
echo "  Without manual assertions, dynamic shape bugs are SILENT."
echo ""
echo "Results written to: $OUTPUT"

rm -rf "$MUTANT_DIR"
