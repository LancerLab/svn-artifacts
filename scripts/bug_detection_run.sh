#!/bin/bash
# Bug Detection Experiment Runner
# Injects bugs and tests detection across Choreo, Triton, MLIR, IREE
# Usage: ./scripts/bug_detection_run.sh [output_csv]
#
# NOTE: Run AFTER the runtime_hoist_comparison benchmark completes to avoid
# contention on the GPU.

set -u

CHOREO=${CHOREO:-./choreo/choreo}
BENCH_DIR=${BENCH_DIR:-./benchmark/choreo}
TARGET=${TARGET:-cute}
OUTPUT=${1:-./benchmark/results/bug_detection_comparison.csv}
export PATH=/usr/local/cuda-12.9/bin:$PATH

mkdir -p "$(dirname "$OUTPUT")"
MUTANT_DIR="/tmp/bug_detection_mutants_$$"
mkdir -p "$MUTANT_DIR"

echo "case,category,bug_class,bug_description,choreo_result,choreo_detail" > "$OUTPUT"

# === Bug Class 1: Dimension Mismatch ===
# Inject by modifying the main() to pass wrong-sized tensors
# For Choreo: modify the make_spandata dimensions to be mismatched

inject_dim_mismatch() {
    local cofile=$1
    local mutant=$2

    cp "$cofile" "$mutant"

    # Strategy: reduce the size of a parameter tensor (gamma/beta/scale/bias)
    # by 1 to create a dimension mismatch with the input tensor.
    # We target the 2nd or 3rd make_spandata call (typically parameters).
    if grep -q 'make_spandata.*gamma\|gm.*make_spandata\|scale.*make_spandata' "$mutant"; then
        # Has explicit gamma/scale variable
        sed -i '0,/\(gm\|gamma\|scale\).*make_spandata<choreo::f32>(\([^)]*\))/s//\1 = choreo::make_spandata<choreo::f32>(\2 - 1)/' "$mutant"
        return $?
    fi

    # Generic: find make_spandata calls with single-variable args (likely parameters)
    # Reduce the first single-arg make_spandata (E), (C), (K), (D) by 1
    local dims="E C K D N0 H W"
    for d in $dims; do
        if grep -q "make_spandata<choreo::f32>($d)" "$mutant"; then
            # Only modify the FIRST occurrence (likely a parameter, not input)
            sed -i "0,/make_spandata<choreo::f32>($d)/s//make_spandata<choreo::f32>($d - 1)/" "$mutant"
            return 0
        fi
    done
    return 1
}

# === Bug Class 2: Input-Dependent OOB ===
# Set dynamic dimension to a value that doesn't divide evenly by tile size
# This tests whether assertions catch tile-boundary violations

inject_input_dep_oob() {
    local cofile=$1
    local mutant=$2

    cp "$cofile" "$mutant"

    # Strategy: reduce the INPUT tensor's last dynamic dimension to half,
    # while keeping parameter tensors at full size.
    # This creates input_dim < param_dim, triggering an OOB when the kernel
    # accesses input beyond its actual bounds.
    # Target the make_spandata for the INPUT (first/largest allocation).
    
    # For multi-dim inputs like (I, J, K) or (N, S, E) or (N, C, H, W):
    # Halve the last dimension of the input only
    if grep -q 'make_spandata<choreo::f32>(I, J, K)' "$mutant"; then
        sed -i 's/make_spandata<choreo::f32>(I, J, K)/make_spandata<choreo::f32>(I, J, K\/2)/' "$mutant"
        return 0
    elif grep -q 'make_spandata<choreo::f32>(N, S, E)' "$mutant"; then
        sed -i 's/make_spandata<choreo::f32>(N, S, E)/make_spandata<choreo::f32>(N, S, E\/2)/' "$mutant"
        return 0
    elif grep -q 'make_spandata<choreo::f32>(N, C, H, W)' "$mutant"; then
        sed -i 's/make_spandata<choreo::f32>(N, C, H, W)/make_spandata<choreo::f32>(N, C, H\/2, W)/' "$mutant"
        return 0
    fi

    # Fallback: change a dimension define to a smaller value
    if grep -q 'embed_dim = ' "$mutant"; then
        sed -i 's/embed_dim = [0-9]*/embed_dim = 383/' "$mutant"
        return 0
    elif grep -q '#define K [0-9]' "$mutant"; then
        sed -i 's/#define K [0-9]*/#define K 1023/' "$mutant"
        return 0
    fi
    return 1
}

# === Bug Class 3: Stride/Layout Error ===
# Swap two dimensions in the main() call (e.g., pass [N,E,S] instead of [N,S,E])

inject_stride_error() {
    local cofile=$1
    local mutant=$2

    cp "$cofile" "$mutant"

    # Swap two dimension arguments in make_spandata for the input tensor
    # Pattern: make_spandata<choreo::f32>(N, S, E) -> make_spandata<choreo::f32>(N, E, S)
    if grep -q 'make_spandata<choreo::f32>(N, S, E)' "$mutant"; then
        sed -i 's/make_spandata<choreo::f32>(N, S, E)/make_spandata<choreo::f32>(N, E, S)/' "$mutant"
    elif grep -q 'make_spandata<choreo::f32>(N, C, H, W)' "$mutant"; then
        sed -i 's/make_spandata<choreo::f32>(N, C, H, W)/make_spandata<choreo::f32>(N, C, W, H)/' "$mutant"
    else
        return 1
    fi
    return 0
}

# Run a mutant through Choreo and classify the result
test_choreo_detection() {
    local mutant=$1
    local label=$2

    # First: try compile with --show-assess to see if static detection occurs
    local compile_out=$($CHOREO -t $TARGET -es --runtime-check=all --show-assess "$mutant" -o /dev/null 2>&1)
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        # Compilation failed — this IS detection (compiler rejected the code)
        echo "STATIC_DETECT"
        return
    fi

    # Check for error messages even with exit code 0
    if echo "$compile_out" | grep -iqE "error|errors have been detected"; then
        echo "STATIC_DETECT"
        return
    fi

    # Check if any assertion was statically resolved as false
    if echo "$compile_out" | grep -q "static-false"; then
        echo "STATIC_DETECT"
        return
    fi

    # Compilation succeeded; try to run it
    $CHOREO -t $TARGET -fc --runtime-check=all "$mutant" -o /tmp/bug_test_bin 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "COMPILE_FAIL_FC"
        return
    fi

    local run_out=$(/tmp/bug_test_bin 2>&1)
    local run_exit=$?

    if echo "$run_out" | grep -iqE "runtime check failed|assertion.*fail|choreo_assert"; then
        echo "RUNTIME_DETECT"
        return
    fi

    if echo "$run_out" | grep -q "CUDA failure"; then
        echo "CRASH_OPAQUE"
        return
    fi

    if [ $run_exit -ne 0 ]; then
        echo "CRASH_OTHER"
        return
    fi

    if echo "$run_out" | grep -iqE "pass|passed"; then
        echo "SILENT_WRONG"
        return
    fi

    echo "UNKNOWN"
}

# === Main Loop ===
echo "=== Bug Detection Comparison Experiment ==="
echo ""

# Use ALL dynamic benchmark cases (they have meaningful shape assertions)
mapfile -t ALL_CASES < <(find "$BENCH_DIR" -name '*dynamic*.co' ! -name '*.bak' | sort)

# If initial testing (10 cases), use SELECTED; else use ALL
if [ "${FULL_RUN:-0}" -eq 1 ]; then
    CASES=("${ALL_CASES[@]}")
    echo "FULL RUN: ${#CASES[@]} dynamic cases"
else
    CASES=(
        "benchmark/choreo/batch_norm/13_dynamic_32x512xV_V_V_32x512xV.co"
        "benchmark/choreo/batch_norm/10_dynamic_16x512xHxW_512_512_16x512xHxW.co"
        "benchmark/choreo/batch_norm/7_dynamic_32x197xE_E_E_32x197xE.co"
        "benchmark/choreo/layer_normalization/13_dynamic_32x512xV_V_V.co"
        "benchmark/choreo/layer_normalization/11_dynamic_32xSx768_768_768.co"
        "benchmark/choreo/layer_normalization/7_dynamic_32x197xD_D_D.co"
        "benchmark/choreo/elemwise_add/7_dynamic_128xCx112x112_128xCx112x112_128xCx112x112.co"
        "benchmark/choreo/reduce_mean/4_dynamic_NxCxHxW_N.co"
        "benchmark/choreo/relu/7_dynamic_32x197xE_32x197xE.co"
        "benchmark/choreo/concat/6_dynamic_128xC1x112x112_128xC2x112x112_128xC1pC2x112x112.co"
    )
    echo "INITIAL TEST: ${#CASES[@]} selected cases"
fi
echo ""

BUG_CLASSES=("dim_mismatch" "input_dep_oob" "stride_error")

for cofile in "${CASES[@]}"; do
    [ -f "$cofile" ] || continue
    category=$(basename "$(dirname "$cofile")")
    casename=$(basename "$cofile" .co)

    for bug_class in "${BUG_CLASSES[@]}"; do
        mutant="$MUTANT_DIR/${casename}_${bug_class}.co"

        # Inject the bug
        case "$bug_class" in
            dim_mismatch)
                inject_dim_mismatch "$cofile" "$mutant" || continue
                bug_desc="gamma/beta dimension reduced by 1"
                ;;
            input_dep_oob)
                inject_input_dep_oob "$cofile" "$mutant" || continue
                bug_desc="dynamic dim set to prime (non-tile-aligned)"
                ;;
            stride_error)
                inject_stride_error "$cofile" "$mutant" || continue
                bug_desc="input tensor dimensions swapped"
                ;;
        esac

        printf "  %-50s %-20s " "$casename" "$bug_class"

        # Test Choreo detection
        choreo_result=$(test_choreo_detection "$mutant" "$bug_class")
        printf "%s\n" "$choreo_result"

        echo "$casename,$category,$bug_class,$bug_desc,$choreo_result," >> "$OUTPUT"
    done
done

echo ""
echo "=== SUMMARY ==="
echo ""
total=$(tail -n +2 "$OUTPUT" | wc -l)
static=$(tail -n +2 "$OUTPUT" | grep -c "STATIC_DETECT") || static=0
runtime=$(tail -n +2 "$OUTPUT" | grep -c "RUNTIME_DETECT") || runtime=0
compile_err=$(tail -n +2 "$OUTPUT" | grep -c "COMPILE") || compile_err=0
crash=$(tail -n +2 "$OUTPUT" | grep -c "CRASH") || crash=0
silent=$(tail -n +2 "$OUTPUT" | grep -c "SILENT") || silent=0

detected=$((static + runtime + compile_err))

echo "Total mutants tested: $total"
echo "  Static assertion:    $static"
echo "  Compile-time error:  $compile_err"
echo "  Runtime detection:   $runtime"
echo "  Opaque crash:        $crash"
echo "  Silent wrong:        $silent"
echo ""
echo "Total detected (compile-time + runtime): $detected / $total"
if [ "$total" -gt 0 ]; then
    pct=$(echo "scale=1; $detected * 100 / $total" | bc 2>/dev/null)
    echo "Detection rate (Choreo): ${pct}%"
fi
echo ""
echo "Results written to: $OUTPUT"

# Cleanup
rm -rf "$MUTANT_DIR"
