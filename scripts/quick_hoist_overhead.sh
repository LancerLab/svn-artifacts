#!/bin/bash
# Quick runtime overhead measurement for cases that compile and run successfully
# Uses choreo direct compilation (not -fc) and measures execution time

CHOREO=./choreo/choreo
BENCH_DIR=./benchmark/choreo
TARGET=cute
export PATH=/usr/local/cuda-12.9/bin:$PATH

echo "case,category,n_hoist,time_none_us,time_all_us,overhead_pct"

# Test a case: compile with none, compile with all, run both, compare
test_case() {
    local cofile=$1
    local casename=$(basename "$cofile" .co)
    local category=$(basename "$(dirname "$cofile")")

    # Get hoist count
    local assess=$($CHOREO -t $TARGET -es --runtime-check=all --show-assess "$cofile" -o /dev/null 2>&1)
    local n_hoist=$(echo "$assess" | grep -c 'HOIST ')

    # Compile both versions
    $CHOREO -t $TARGET --runtime-check=none "$cofile" -o /tmp/bench_none 2>/dev/null
    [ $? -ne 0 ] && return 1
    $CHOREO -t $TARGET --runtime-check=all "$cofile" -o /tmp/bench_all 2>/dev/null
    [ $? -ne 0 ] && return 1

    # Run none version
    local out_none=$(/tmp/bench_none 2>&1)
    echo "$out_none" | grep -q "Passed" || return 1
    local t_none=$(echo "$out_none" | grep -oP '\d+(?=\s*(microseconds|us))')

    # Run all version
    local out_all=$(/tmp/bench_all 2>&1)
    echo "$out_all" | grep -q "Passed" || return 1
    local t_all=$(echo "$out_all" | grep -oP '\d+(?=\s*(microseconds|us))')

    if [ -n "$t_none" ] && [ -n "$t_all" ] && [ "$t_none" -gt 0 ]; then
        local overhead=$(echo "scale=2; ($t_all - $t_none) * 100 / $t_none" | bc 2>/dev/null)
        echo "$casename,$category,$n_hoist,$t_none,$t_all,$overhead"
    fi
}

# Run on all dynamic cases from categories with hoisting
for cat in batch_norm layer_normalization softmax; do
    for f in "$BENCH_DIR/$cat"/*dynamic*.co; do
        [ -f "$f" ] || continue
        echo "$f" | grep -q '\.bak$' && continue
        test_case "$f" 2>/dev/null
    done
done
