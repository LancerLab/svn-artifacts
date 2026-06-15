#!/bin/bash
# Compare runtime overhead: baseline (no checks) vs hoisted vs un-hoisted assertions
# Usage: ./scripts/runtime_hoist_comparison.sh [output_csv]
# Run on the remote machine where choreo + CUDA is available.

set -u

CHOREO=${CHOREO:-./choreo/choreo}
BENCH_DIR=${BENCH_DIR:-./benchmark/choreo}
TARGET=${TARGET:-cute}
OUTPUT=${1:-./benchmark/results/runtime_hoist_comparison.csv}
REPS=${REPS:-5}
export PATH=/usr/local/cuda-12.9/bin:$PATH

mkdir -p "$(dirname "$OUTPUT")"

echo "case,category,n_hoist,est_cost_hoist,est_cost_nohoist,time_none_us,time_entry_us,time_hoist_us,time_nohoist_us,overhead_entry_pct,overhead_hoist_pct,overhead_nohoist_pct" > "$OUTPUT"

compile_and_run() {
    local cofile=$1
    local flags=$2
    local outbin=$3
    local reps=$4

    $CHOREO -t $TARGET -fc $flags "$cofile" -o "$outbin" 2>/dev/null || true
    if [ ! -f "$outbin" ] || [ ! -x "$outbin" ]; then
        echo "COMPILE_FAIL"
        return
    fi

    # Warmup run (first invocation loads driver/context)
    local warmup_out=$("$outbin" 2>&1)
    if echo "$warmup_out" | grep -q "CUDA failure"; then
        echo "CUDA_FAIL"
        return
    fi

    local times=()
    for ((r=1; r<=reps; r++)); do
        local out=$("$outbin" 2>&1)
        if ! echo "$out" | grep -iqE "pass|passed"; then
            echo "RUN_FAIL"
            return
        fi
        local t=$(echo "$out" | grep -oP '\d+(?=\s*(us|microseconds))' || echo "")
        if [ -z "$t" ]; then
            t=$(echo "$out" | grep -oP 'Execution time:\s*\K\d+' || echo "")
        fi
        [ -n "$t" ] && times+=("$t")
    done

    if [ ${#times[@]} -eq 0 ]; then
        echo "NO_TIME"
        return
    fi

    # Return median
    printf '%s\n' "${times[@]}" | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'
}

get_hoist_stats() {
    local cofile=$1
    local assess=$($CHOREO -t $TARGET -es --runtime-check=all --show-assess "$cofile" -o /dev/null 2>&1)
    local n_hoist=$(echo "$assess" | grep -c 'HOIST' || echo 0)
    local cost_hoist=$(echo "$assess" | grep 'HOIST' | grep -oP 'estimated=\K\d+' | awk '{s+=$1}END{print s+0}' || echo 0)

    local assess_no=$($CHOREO -t $TARGET -es --runtime-check=all --show-assess --disable-assert-hoist "$cofile" -o /dev/null 2>&1)
    local cost_nohoist=$(echo "$assess_no" | grep 'USE_SITE' | grep -oP 'estimated=\K\d+' | awk '{s+=$1}END{print s+0}' || echo 0)

    echo "$n_hoist $cost_hoist $cost_nohoist"
}

# Collect benchmark cases (all .co files excluding .bak)
mapfile -t cases < <(find "$BENCH_DIR" -name '*.co' ! -name '*.bak' | sort)

total=${#cases[@]}
echo "=== Runtime Hoist Comparison ==="
echo "Cases: $total, Reps per config: $REPS"
echo ""

success=0
fail=0

for ((idx=0; idx<total; idx++)); do
    cofile="${cases[$idx]}"
    category=$(basename "$(dirname "$cofile")")
    casename=$(basename "$cofile" .co)

    printf "[%3d/%d] %-60s " "$((idx+1))" "$total" "$casename"

    # Get hoisting stats
    read -r n_hoist cost_hoist cost_nohoist <<< "$(get_hoist_stats "$cofile")"

    if [ "$n_hoist" -eq 0 ]; then
        printf "SKIP (no hoisted assertions)\n"
        continue
    fi

    # Compile and run: no checks (baseline)
    t_none=$(compile_and_run "$cofile" "--runtime-check=none" "/tmp/bench_none_$$" "$REPS")
    if [[ "$t_none" == "CUDA_FAIL" ]]; then
        printf "SKIP (CUDA runtime error in baseline)\n"
        continue
    fi
    if [[ "$t_none" == *FAIL* ]] || [[ "$t_none" == "NO_TIME" ]]; then
        printf "FAIL (baseline: %s)\n" "$t_none"
        ((fail++))
        continue
    fi

    # Compile and run: entry-level checks only
    t_entry=$(compile_and_run "$cofile" "--runtime-check=entry" "/tmp/bench_entry_$$" "$REPS")
    if [[ "$t_entry" == "CUDA_FAIL" ]]; then
        printf "SKIP (CUDA runtime error with entry)\n"
        continue
    fi
    if [[ "$t_entry" == *FAIL* ]] || [[ "$t_entry" == "NO_TIME" ]]; then
        printf "FAIL (entry: %s)\n" "$t_entry"
        ((fail++))
        continue
    fi

    # Compile and run: with hoisting (default)
    t_hoist=$(compile_and_run "$cofile" "--runtime-check=all" "/tmp/bench_hoist_$$" "$REPS")
    if [[ "$t_hoist" == "CUDA_FAIL" ]]; then
        printf "SKIP (CUDA runtime error with hoist)\n"
        continue
    fi
    if [[ "$t_hoist" == *FAIL* ]] || [[ "$t_hoist" == "NO_TIME" ]]; then
        printf "FAIL (hoist: %s)\n" "$t_hoist"
        ((fail++))
        continue
    fi

    # Compile and run: without hoisting
    t_nohoist=$(compile_and_run "$cofile" "--runtime-check=all --disable-assert-hoist" "/tmp/bench_nohoist_$$" "$REPS")
    if [[ "$t_nohoist" == "CUDA_FAIL" ]]; then
        printf "SKIP (CUDA runtime error without hoist)\n"
        continue
    fi
    if [[ "$t_nohoist" == *FAIL* ]] || [[ "$t_nohoist" == "NO_TIME" ]]; then
        printf "FAIL (nohoist: %s)\n" "$t_nohoist"
        ((fail++))
        continue
    fi

    # Calculate overhead percentages
    overhead_entry=$(echo "scale=2; ($t_entry - $t_none) * 100 / $t_none" | bc 2>/dev/null || echo "N/A")
    overhead_hoist=$(echo "scale=2; ($t_hoist - $t_none) * 100 / $t_none" | bc 2>/dev/null || echo "N/A")
    overhead_nohoist=$(echo "scale=2; ($t_nohoist - $t_none) * 100 / $t_none" | bc 2>/dev/null || echo "N/A")

    echo "$casename,$category,$n_hoist,$cost_hoist,$cost_nohoist,$t_none,$t_entry,$t_hoist,$t_nohoist,$overhead_entry,$overhead_hoist,$overhead_nohoist" >> "$OUTPUT"

    printf "none=%s entry=%s(%s%%) hoist=%s(%s%%) nohoist=%s(%s%%)\n" "$t_none" "$t_entry" "$overhead_entry" "$t_hoist" "$overhead_hoist" "$t_nohoist" "$overhead_nohoist"
    ((success++))
done

echo ""
echo "=== SUMMARY ==="
echo "Success: $success, Failed: $fail, Total: $total"
echo ""

if [ -f "$OUTPUT" ] && [ $(wc -l < "$OUTPUT") -gt 1 ]; then
    tail -n +2 "$OUTPUT" | awk -F',' '
    BEGIN { n=0; sum_entry=0; sum_oh=0; sum_ohn=0; max_ohn=0 }
    {
        n++
        sum_entry += $10
        sum_oh += $11
        sum_ohn += $12
        if ($12 > max_ohn) { max_ohn = $12; worst = $1 }
    }
    END {
        if (n > 0) {
            printf "Avg overhead (entry):      %.2f%%\n", sum_entry/n
            printf "Avg overhead (hoisted):    %.2f%%\n", sum_oh/n
            printf "Avg overhead (un-hoisted): %.2f%%\n", sum_ohn/n
            printf "Max overhead (un-hoisted): %.2f%% (%s)\n", max_ohn, worst
        }
    }'
fi

echo ""
echo "Results written to: $OUTPUT"
