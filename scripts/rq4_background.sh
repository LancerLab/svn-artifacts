#!/usr/bin/env bash
# scripts/rq4_background.sh — RQ4 runtime-check overhead measurement
#
# Phase 1: Parallel compilation of all dynamic cases × 3 runtime-check levels
# Phase 2: Sequential GPU execution (5 reps each) to measure overhead
#
# Designed to run inside a persistent tmux session so it survives disconnects.
#
# Usage:
#   tmux new-session -d -s rq4 'bash scripts/rq4_background.sh 2>&1 | tee benchmark/results/rq4_log.txt'
#   tmux attach -t rq4          # to watch progress
#   # Ctrl-b d to detach and let it run
#
# Output:
#   benchmark/results/choreo_rq4_runtime.csv

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHOREO_BIN="${WORKSPACE_ROOT}/choreo/build-release/choreo"
CASES_DIR="${WORKSPACE_ROOT}/benchmark/choreo"
RESULTS_DIR="${WORKSPACE_ROOT}/benchmark/results"
WORK_DIR="${RESULTS_DIR}/rq4_work"
OUT_CSV="${RESULTS_DIR}/choreo_rq4_runtime.csv"
LOG_DIR="${WORK_DIR}/logs"
N_REPS=5
PARALLEL_JOBS=$(nproc)
RTC_LEVELS="none entry all"

# Ensure nvcc is on PATH
export PATH="/usr/local/cuda/bin:${PATH}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"

echo "============================================"
echo "RQ4 Runtime-Check Overhead Measurement"
echo "============================================"
echo "Choreo:      ${CHOREO_BIN}"
echo "Cases:       ${CASES_DIR}"
echo "Work dir:    ${WORK_DIR}"
echo "Output CSV:  ${OUT_CSV}"
echo "CPU jobs:    ${PARALLEL_JOBS}"
echo "GPU reps:    ${N_REPS}"
echo "RTC levels:  ${RTC_LEVELS}"
echo "Started at:  $(date)"
echo "============================================"
echo ""

# ----------------------------------------------------------------
# Phase 1: Parallel compilation
# ----------------------------------------------------------------
echo "=== PHASE 1: Parallel compilation ==="
COMPILE_START=$(date +%s)

# Build task list: only dynamic cases (contain __STATIC_SHAPE__ guard but not #define __STATIC_SHAPE__)
TASK_LIST="${WORK_DIR}/compile_tasks.txt"
> "${TASK_LIST}"

for co_file in $(find "${CASES_DIR}" -name "*.co" | sort); do
    if grep -q "#ifdef __STATIC_SHAPE__" "$co_file" && ! grep -q "#define __STATIC_SHAPE__" "$co_file"; then
        category=$(basename "$(dirname "$co_file")")
        case_name=$(basename "$co_file" .co)
        for level in ${RTC_LEVELS}; do
            out_sh="${WORK_DIR}/${category}__${case_name}__rtc_${level}.sh"
            echo "${co_file}|${category}|${case_name}|${level}|${out_sh}" >> "${TASK_LIST}"
        done
    fi
done

TOTAL_TASKS=$(wc -l < "${TASK_LIST}")
echo "Total compile tasks: ${TOTAL_TASKS}"
echo ""

# Compile function (one task per invocation)
compile_one() {
    local line="$1"
    IFS='|' read -r co_file category case_name level out_sh <<< "$line"
    local log="${LOG_DIR}/${category}__${case_name}__rtc_${level}.compile.log"

    if timeout 60 "${CHOREO_BIN}" "$co_file" -gs -t cute "--runtime-check=${level}" -o "$out_sh" > "$log" 2>&1; then
        if [ -s "$out_sh" ]; then
            echo "OK: ${category}/${case_name} rtc=${level}"
        else
            echo "EMPTY: ${category}/${case_name} rtc=${level}"
        fi
    else
        echo "FAIL: ${category}/${case_name} rtc=${level}"
    fi
}
export -f compile_one
export CHOREO_BIN LOG_DIR

# Run compilations in parallel using xargs
cat "${TASK_LIST}" | xargs -P "${PARALLEL_JOBS}" -I{} bash -c 'compile_one "$@"' _ {}

COMPILE_END=$(date +%s)
echo ""
echo "Phase 1 completed in $((COMPILE_END - COMPILE_START)) seconds"

# Count results
COMPILED_OK=$(find "${WORK_DIR}" -name "*.sh" -size +0c | wc -l)
echo "Successfully compiled: ${COMPILED_OK} / ${TOTAL_TASKS}"
echo ""

# ----------------------------------------------------------------
# Phase 2: Sequential GPU execution
# ----------------------------------------------------------------
echo "=== PHASE 2: Sequential GPU execution ==="
EXEC_START=$(date +%s)

# Write CSV header
echo "category,case_name,rtc_none_us,rtc_entry_us,rtc_all_us,notes" > "${OUT_CSV}"

# Get unique case list (category__case_name)
CASE_LIST=$(awk -F'|' '{print $2 "|" $3}' "${TASK_LIST}" | sort -u)
CASE_COUNT=$(echo "$CASE_LIST" | wc -l)
CASE_IDX=0

echo "$CASE_LIST" | while IFS='|' read -r category case_name; do
    CASE_IDX=$((CASE_IDX + 1))
    echo "[${CASE_IDX}/${CASE_COUNT}] ${category}/${case_name}"

    row="${category},${case_name}"
    notes=""

    for level in ${RTC_LEVELS}; do
        sh_file="${WORK_DIR}/${category}__${case_name}__rtc_${level}.sh"
        col_val="error"

        if [ -s "${sh_file}" ]; then
            times=()
            exec_ok=true
            for rep in $(seq 1 ${N_REPS}); do
                exec_log="${LOG_DIR}/${category}__${case_name}__rtc_${level}.exec_${rep}.log"
                if timeout 300 bash "${sh_file}" --execute > "${exec_log}" 2>&1; then
                    # Extract execution time in microseconds
                    t=$(grep "Execution time:" "${exec_log}" | head -1 | sed 's/.*: *\([0-9]*\).*/\1/')
                    if [ -n "$t" ]; then
                        times+=("$t")
                    else
                        exec_ok=false
                        notes="${notes}parse-${level}-rep${rep};"
                        break
                    fi
                else
                    exec_ok=false
                    notes="${notes}exec-${level}-rep${rep};"
                    break
                fi
            done

            if $exec_ok && [ ${#times[@]} -eq ${N_REPS} ]; then
                # Compute median using sort
                sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
                median_idx=$(( (N_REPS - 1) / 2 ))
                col_val="${sorted_times[$median_idx]}"
                echo "  rtc=${level}: ${col_val} us"
            else
                col_val="error"
                echo "  rtc=${level}: FAILED"
            fi
        else
            col_val="error"
            notes="${notes}gen-${level}-failed;"
            echo "  rtc=${level}: compile-failed"
        fi

        row="${row},${col_val}"
    done

    echo "${row},${notes}" >> "${OUT_CSV}"
done

EXEC_END=$(date +%s)
echo ""
echo "Phase 2 completed in $((EXEC_END - EXEC_START)) seconds"

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
echo ""
echo "============================================"
echo "RQ4 Complete!"
echo "============================================"
echo "CSV: ${OUT_CSV}"
echo "Total time: $((EXEC_END - COMPILE_START)) seconds"
echo ""

# Print quick stats
OK_ROWS=$(awk -F, 'NR>1 && $3!="error" && $4!="error" && $5!="error" {n++} END{print n+0}' "${OUT_CSV}")
TOTAL_ROWS=$(awk 'END{print NR-1}' "${OUT_CSV}")
echo "Successfully measured: ${OK_ROWS} / ${TOTAL_ROWS} cases"

if [ "${OK_ROWS}" -gt 0 ]; then
    echo ""
    echo "Entry-level overhead (entry vs none):"
    awk -F, 'NR>1 && $3!="error" && $4!="error" {
        none=$3; entry=$4;
        if (none > 0) { ovhd=(entry/none - 1)*100; printf "  %s/%s: %.3f%%\n", $1, $2, ovhd }
    }' "${OUT_CSV}" | sort -t: -k2 -n | tail -5
    echo "  ... (see full CSV for all entries)"
fi

echo ""
echo "Finished at: $(date)"
