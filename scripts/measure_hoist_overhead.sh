#!/bin/bash
# Measure runtime overhead of hoisted assertions vs no assertions
# Compiles each case with --runtime-check=all and --runtime-check=none,
# runs both binaries, and compares kernel time.
#
# Usage: ./scripts/measure_hoist_overhead.sh [cases_file]

CHOREO=${CHOREO:-./choreo/choreo}
BENCH_DIR=${BENCH_DIR:-./benchmark/choreo}
TARGET=${TARGET:-cute}
REPS=${REPS:-5}
TMPDIR=${TMPDIR:-/tmp/hoist_bench_$$}

mkdir -p "$TMPDIR"

# Select cases with significant hoisting (>= 5 hoisted assertions)
CASES=(
    "softmax/7_dynamic_32x197xE_32x197xE"
    "softmax/6_dynamic_128xCx112x112_128xCx112x112"
    "softmax/3_attention_32xNx512x64_32xNx512x64"
    "batch_norm/10_dynamic_16x512xHxW_512_512_16x512xHxW"
    "layer_normalization/6_dynamic_128xCx112x112"
    "layer_normalization/5_dynamic_Bx1280xHxW"
    "max_pool2d/6_dynamic_Bx1280xHxW_Bx1280xHd2xWd2"
    "max_pool2d/4_dynamic_64xCxHxW_64xCxHd2xWd2"
    "reduce_mean/6_dynamic_128xCx112x112"
)

echo "case,n_entry,n_hoist,time_none_us,time_all_us,overhead_pct"

for casepath in "${CASES[@]}"; do
    cofile="$BENCH_DIR/$casepath.co"
    if [ ! -f "$cofile" ]; then
        echo "# SKIP: $cofile not found" >&2
        continue
    fi

    casename=$(basename "$casepath")
    cu_none="$TMPDIR/${casename}_none.cu"
    cu_all="$TMPDIR/${casename}_all.cu"
    bin_none="$TMPDIR/${casename}_none"
    bin_all="$TMPDIR/${casename}_all"

    # Compile with runtime-check=none
    $CHOREO -t "$TARGET" --runtime-check=none "$cofile" -o "$cu_none" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "# FAIL compile (none): $casename" >&2
        continue
    fi

    # Compile with runtime-check=all
    $CHOREO -t "$TARGET" --runtime-check=all "$cofile" -o "$cu_all" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "# FAIL compile (all): $casename" >&2
        continue
    fi

    # Get hoist counts
    assess=$($CHOREO -t "$TARGET" -es --runtime-check=all --show-assess "$cofile" -o /dev/null 2>&1)
    n_entry=$(echo "$assess" | grep -c 'ENTRY ')
    n_hoist=$(echo "$assess" | grep -c 'HOIST ')

    # Compile CUDA files
    nvcc -O2 -arch=sm_86 "$cu_none" -o "$bin_none" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "# FAIL nvcc (none): $casename" >&2
        continue
    fi
    nvcc -O2 -arch=sm_86 "$cu_all" -o "$bin_all" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "# FAIL nvcc (all): $casename" >&2
        continue
    fi

    # Run and measure (take median of REPS runs)
    times_none=()
    times_all=()
    for ((i=0; i<REPS; i++)); do
        t=$("$bin_none" 2>&1 | grep -oP '\d+\.?\d*(?=\s*(us|microseconds))')
        times_none+=("$t")
        t=$("$bin_all" 2>&1 | grep -oP '\d+\.?\d*(?=\s*(us|microseconds))')
        times_all+=("$t")
    done

    # Compute medians
    med_none=$(printf '%s\n' "${times_none[@]}" | sort -n | sed -n "$((REPS/2+1))p")
    med_all=$(printf '%s\n' "${times_all[@]}" | sort -n | sed -n "$((REPS/2+1))p")

    if [ -n "$med_none" ] && [ -n "$med_all" ] && [ "$med_none" != "0" ]; then
        overhead=$(echo "scale=2; ($med_all - $med_none) * 100 / $med_none" | bc 2>/dev/null)
    else
        overhead="N/A"
    fi

    echo "$casename,$n_entry,$n_hoist,$med_none,$med_all,$overhead"
done

rm -rf "$TMPDIR"
