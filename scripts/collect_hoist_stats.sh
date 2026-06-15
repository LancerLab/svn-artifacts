#!/bin/bash
# Collect assert-site hoisting statistics across all benchmark cases
# Usage: ./scripts/collect_hoist_stats.sh [output_csv]

CHOREO=${CHOREO:-./choreo/choreo}
BENCH_DIR=${BENCH_DIR:-./benchmark/choreo}
OUTPUT=${1:-./benchmark/results/hoist_stats.csv}
TARGET=${TARGET:-cute}

echo "case,category,total_assessed,static_true,static_false,runtime,entry,hoist,use_site,entry_cost_sum,hoist_cost_sum,use_site_cost_sum" > "$OUTPUT"

find "$BENCH_DIR" -name '*.co' ! -name '*.bak' | sort | while read -r cofile; do
    category=$(basename "$(dirname "$cofile")")
    casename=$(basename "$cofile" .co)

    assess_out=$($CHOREO -t "$TARGET" -es --runtime-check=all --show-assess "$cofile" -o /dev/null 2>&1)

    total_assessed=$(echo "$assess_out" | grep -oP '\(\K\d+(?= assessed)')
    static_true=$(echo "$assess_out" | grep -oP '\d+(?= static-true)')
    static_false=$(echo "$assess_out" | grep -oP '\d+(?= static-false)')
    runtime=$(echo "$assess_out" | grep -oP '\d+(?= runtime\))')

    n_entry=$(echo "$assess_out" | grep -c 'ENTRY ')
    n_hoist=$(echo "$assess_out" | grep -c 'HOIST ')
    n_usesite=$(echo "$assess_out" | grep -c 'USE_SITE')

    # Sum estimated costs by type
    entry_cost=$(echo "$assess_out" | grep 'ENTRY ' | grep -oP 'estimated=\K\d+' | awk '{s+=$1}END{print s+0}')
    hoist_cost=$(echo "$assess_out" | grep 'HOIST ' | grep -oP 'estimated=\K\d+' | awk '{s+=$1}END{print s+0}')
    usesite_cost=$(echo "$assess_out" | grep 'USE_SITE' | grep -oP 'estimated=\K\d+' | awk '{s+=$1}END{print s+0}')

    # Default empty values
    total_assessed=${total_assessed:-0}
    static_true=${static_true:-0}
    static_false=${static_false:-0}
    runtime=${runtime:-0}

    echo "$casename,$category,$total_assessed,$static_true,$static_false,$runtime,$n_entry,$n_hoist,$n_usesite,$entry_cost,$hoist_cost,$usesite_cost" >> "$OUTPUT"

    printf "  %-50s  assessed=%3s rt=%2s entry=%2s hoist=%2s use=%2s\n" "$casename" "$total_assessed" "$runtime" "$n_entry" "$n_hoist" "$n_usesite"
done

echo ""
echo "=== SUMMARY ==="
tail -n +2 "$OUTPUT" | awk -F',' '
BEGIN { n=0; sum_assessed=0; sum_rt=0; sum_entry=0; sum_hoist=0; sum_usesite=0; sum_hcost=0; sum_ecost=0 }
{
    n++
    sum_assessed += $3
    sum_rt += $6
    sum_entry += $7
    sum_hoist += $8
    sum_usesite += $9
    sum_ecost += $10
    sum_hcost += $11
}
END {
    printf "Cases: %d\n", n
    printf "Total assessed: %d\n", sum_assessed
    printf "Total runtime assertions: %d\n", sum_rt
    printf "  ENTRY: %d  (est. cost sum: %d)\n", sum_entry, sum_ecost
    printf "  HOIST: %d  (est. cost sum: %d)\n", sum_hoist, sum_hcost
    printf "  USE_SITE: %d\n", sum_usesite
    if (sum_hoist > 0) {
        printf "Hoist ratio (hoisted / non-entry): %.1f%%\n", sum_hoist * 100.0 / (sum_hoist + sum_usesite)
        printf "Avg cost reduction per hoisted assert: %.0f iterations saved\n", sum_hcost / sum_hoist
    }
}'
echo "Results written to: $OUTPUT"
