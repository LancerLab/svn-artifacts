#!/bin/bash
# Run linalg-level MLIR test cases via mlir-cpu-runner
# Tests both correct and mismatched cases with and without runtime verification
#
# Usage: ./scripts/run_linalg_suite.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MLIR_LOCAL="${HOME}/mlir-local"
MLIR_OPT="${MLIR_LOCAL}/usr/bin/mlir-opt-18"
MLIR_RUNNER="${MLIR_LOCAL}/usr/bin/mlir-cpu-runner-18"
MLIR_LIB="${MLIR_LOCAL}/usr/lib/llvm-18/lib"
export LD_LIBRARY_PATH="${MLIR_LIB}:${MLIR_LOCAL}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

RUN_DIR="$WORKSPACE_ROOT/benchmark/mlir/cases_linalg_run"
OUTPUT_DIR="$WORKSPACE_ROOT/benchmark/results"
OUTPUT_CSV="$OUTPUT_DIR/linalg_runtime_test.csv"
mkdir -p "$OUTPUT_DIR"

LOWER_PIPELINE="builtin.module(convert-linalg-to-loops,convert-scf-to-cf,convert-math-to-llvm,convert-cf-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-vector-to-llvm,reconcile-unrealized-casts)"
RTV_PIPELINE="builtin.module(convert-linalg-to-loops,generate-runtime-verification,convert-scf-to-cf,convert-math-to-llvm,convert-cf-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-vector-to-llvm,reconcile-unrealized-casts)"

echo "category,case_name,type,lower_ok,run_result_no_rtv,run_result_with_rtv,rtv_assert_count" > "$OUTPUT_CSV"

total=0
correct_pass=0
mismatch_detected=0
mismatch_crash=0
mismatch_silent=0

echo "=========================================="
echo "Linalg Runtime Test Suite"
echo "=========================================="

run_case() {
    local mlir_file=$1
    local pipeline=$2
    local tmp_ll
    tmp_ll=$(mktemp --suffix=.mlir)
    local tmp_err
    tmp_err=$(mktemp)

    "$MLIR_OPT" --pass-pipeline="$pipeline" "$mlir_file" > "$tmp_ll" 2> "$tmp_err"
    if [[ $? -ne 0 ]]; then
        rm -f "$tmp_ll" "$tmp_err"
        echo "LOWER_FAIL"
        return
    fi

    local run_out run_err
    run_out=$(mktemp)
    run_err=$(mktemp)
    timeout 10 "$MLIR_RUNNER" "$tmp_ll" --entry-point-result=void \
        -shared-libs="$MLIR_LIB/libmlir_runner_utils.so.18.1,$MLIR_LIB/libmlir_c_runner_utils.so.18.1" \
        > "$run_out" 2> "$run_err"
    local rc=$?
    rm -f "$tmp_ll" "$tmp_err"

    if [[ $rc -eq 0 ]]; then
        local output
        output=$(cat "$run_out" | head -1)
        rm -f "$run_out" "$run_err"
        echo "PASS:$output"
    elif [[ $rc -eq 134 ]]; then
        rm -f "$run_out" "$run_err"
        echo "ABORT"
    elif [[ $rc -eq 139 ]]; then
        rm -f "$run_out" "$run_err"
        echo "SIGSEGV"
    elif [[ $rc -eq 124 ]]; then
        rm -f "$run_out" "$run_err"
        echo "TIMEOUT"
    else
        rm -f "$run_out" "$run_err"
        echo "EXIT_$rc"
    fi
}

for cat_dir in "$RUN_DIR"/*/; do
    cat=$(basename "$cat_dir")
    echo ""
    echo "=== Category: $cat ==="

    for f in "$cat_dir"*.mlir; do
        [[ -f "$f" ]] || continue
        bname=$(basename "$f" .mlir)
        total=$((total + 1))

        case_type="correct"
        echo "$bname" | grep -qi "mismatch" && case_type="mismatch"

        # Check if it lowers
        lower_ok="yes"
        "$MLIR_OPT" --pass-pipeline="$LOWER_PIPELINE" "$f" > /dev/null 2>&1 || lower_ok="no"

        if [[ "$lower_ok" == "no" ]]; then
            echo "  [$case_type] $bname: LOWER_FAIL"
            echo "$cat,$bname,$case_type,no,LOWER_FAIL,LOWER_FAIL,0" >> "$OUTPUT_CSV"
            continue
        fi

        # Count runtime verification assertions
        rtv_count=$("$MLIR_OPT" --pass-pipeline="$RTV_PIPELINE" "$f" 2>/dev/null | grep -c "cf.assert") || rtv_count=0

        # Run without runtime verification
        result_no_rtv=$(run_case "$f" "$LOWER_PIPELINE")

        # Run with runtime verification
        result_with_rtv=$(run_case "$f" "$RTV_PIPELINE")

        if [[ "$case_type" == "correct" ]]; then
            echo "$result_no_rtv" | grep -q "^PASS" && correct_pass=$((correct_pass + 1))
        else
            if [[ "$result_with_rtv" == "ABORT" ]]; then
                mismatch_detected=$((mismatch_detected + 1))
            elif echo "$result_with_rtv" | grep -q "^PASS"; then
                mismatch_silent=$((mismatch_silent + 1))
            else
                mismatch_crash=$((mismatch_crash + 1))
            fi
        fi

        echo "  [$case_type] $bname: no_rtv=$result_no_rtv, with_rtv=$result_with_rtv ($rtv_count checks)"
        echo "$cat,$bname,$case_type,$lower_ok,$result_no_rtv,$result_with_rtv,$rtv_count" >> "$OUTPUT_CSV"
    done
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Total cases: $total"
echo ""
echo "Correct cases that PASS: $correct_pass"
echo ""
echo "Mismatch cases:"
echo "  Detected (ABORT with rtv): $mismatch_detected"
echo "  Crash (SIGSEGV without rtv): $mismatch_crash"
echo "  Silent (passed despite mismatch): $mismatch_silent"
echo ""
echo "Results: $OUTPUT_CSV"
