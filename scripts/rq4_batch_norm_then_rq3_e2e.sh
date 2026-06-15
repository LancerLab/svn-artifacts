#!/usr/bin/env bash
# scripts/rq4_batch_norm_then_rq3_e2e.sh
#
# Two-phase background job:
#   Phase A: Rerun RQ4 for batch_norm family only (5 reps × 3 rtc levels)
#   Phase B: Measure end-to-end compilation time for all dynamic cases
#            comparing SVN dynamic vs static (full nvcc pipeline)
#            + measure nvcc backend time separately via --time flag
#
# Usage:
#   tmux new-session -d -s rq4rq3 \
#     'bash scripts/rq4_batch_norm_then_rq3_e2e.sh 2>&1 | tee benchmark/results/rq4rq3_log.txt'
#   tmux attach -t rq4rq3

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHOREO_BIN="${WORKSPACE_ROOT}/choreo/build-release/choreo"
CASES_DIR="${WORKSPACE_ROOT}/benchmark/choreo"
RESULTS_DIR="${WORKSPACE_ROOT}/benchmark/results"
export PATH="/usr/local/cuda/bin:${PATH}"

N_REPS=5

mkdir -p "${RESULTS_DIR}"

########################################################################
# PHASE A: RQ4 batch_norm rerun
########################################################################
echo "========================================================"
echo "PHASE A: RQ4 batch_norm rerun (${N_REPS} reps × 3 levels)"
echo "Started at: $(date)"
echo "========================================================"

RQ4_WORK="${RESULTS_DIR}/rq4_batch_norm_rerun"
RQ4_LOG_DIR="${RQ4_WORK}/logs"
RQ4_CSV="${RESULTS_DIR}/rq4_batch_norm_rerun.csv"
RTC_LEVELS="none entry all"
FAMILY="batch_norm"

mkdir -p "${RQ4_WORK}" "${RQ4_LOG_DIR}"

# Skip Phase A if results already exist
if [[ -s "${RQ4_CSV}" ]]; then
    echo "Phase A already complete (${RQ4_CSV} exists). Skipping to Phase B."
else
echo ""
echo "--- Phase A-1: Compiling batch_norm cases ---"
for co_file in "${CASES_DIR}/${FAMILY}"/*.co; do
    [[ -f "$co_file" ]] || continue
    # Only dynamic cases
    if grep -q "#ifdef __STATIC_SHAPE__" "$co_file" && ! grep -q "#define __STATIC_SHAPE__" "$co_file"; then
        case_name=$(basename "$co_file" .co)
        for level in ${RTC_LEVELS}; do
            out_sh="${RQ4_WORK}/${FAMILY}__${case_name}__rtc_${level}.sh"
            log="${RQ4_LOG_DIR}/${FAMILY}__${case_name}__rtc_${level}.compile.log"
            if timeout 60 "${CHOREO_BIN}" "$co_file" -gs -t cute "--runtime-check=${level}" -o "$out_sh" > "$log" 2>&1; then
                echo "  OK: ${case_name} rtc=${level}"
            else
                echo "  FAIL: ${case_name} rtc=${level}"
            fi
        done
    fi
done

# Phase A-2: Execute batch_norm cases (sequential, on GPU)
echo ""
echo "--- Phase A-2: Executing batch_norm cases (${N_REPS} reps) ---"
echo "category,case_name,rtc_none_us,rtc_entry_us,rtc_all_us,notes" > "${RQ4_CSV}"

for co_file in "${CASES_DIR}/${FAMILY}"/*.co; do
    [[ -f "$co_file" ]] || continue
    if ! grep -q "#ifdef __STATIC_SHAPE__" "$co_file" || grep -q "#define __STATIC_SHAPE__" "$co_file"; then
        continue
    fi
    case_name=$(basename "$co_file" .co)
    echo "  [${FAMILY}/${case_name}]"
    row="${FAMILY},${case_name}"
    notes=""

    for level in ${RTC_LEVELS}; do
        sh_file="${RQ4_WORK}/${FAMILY}__${case_name}__rtc_${level}.sh"
        col_val="error"

        if [ -s "${sh_file}" ]; then
            times=()
            exec_ok=true
            for rep in $(seq 1 ${N_REPS}); do
                exec_log="${RQ4_LOG_DIR}/${FAMILY}__${case_name}__rtc_${level}.exec_${rep}.log"
                if timeout 300 bash "${sh_file}" --execute > "${exec_log}" 2>&1; then
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
                sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
                median_idx=$(( (N_REPS - 1) / 2 ))
                col_val="${sorted_times[$median_idx]}"
                echo "    rtc=${level}: ${col_val} us (median of ${N_REPS})"
            else
                col_val="error"
                echo "    rtc=${level}: FAILED"
            fi
        else
            col_val="error"
            notes="${notes}gen-${level}-failed;"
            echo "    rtc=${level}: compile-failed"
        fi
        row="${row},${col_val}"
    done
    echo "${row},${notes}" >> "${RQ4_CSV}"
done

echo ""
echo "Phase A done at $(date)"
echo "Results: ${RQ4_CSV}"
cat "${RQ4_CSV}"
echo ""
fi  # end Phase A skip

########################################################################
# PHASE B: End-to-end compile time comparison (RQ3 enhancement)
########################################################################
echo "========================================================"
echo "PHASE B: End-to-end compile time (full nvcc pipeline)"
echo "Started at: $(date)"
echo "========================================================"

E2E_WORK="${RESULTS_DIR}/rq3_e2e_work"
E2E_LOG_DIR="${E2E_WORK}/logs"
E2E_CSV="${RESULTS_DIR}/choreo_e2e_compile_time.csv"
E2E_REPS=5

mkdir -p "${E2E_WORK}" "${E2E_LOG_DIR}"

# CSV header (only write if file doesn't exist or is empty)
if [[ ! -s "${E2E_CSV}" ]]; then
    echo "category,case_name,dynamic_e2e_ms,static_e2e_ms,e2e_overhead_pct,dynamic_nvcc_ms,static_nvcc_ms,nvcc_overhead_pct,dynamic_frontend_ms,static_frontend_ms,frontend_overhead_pct,notes" > "${E2E_CSV}"
fi

# measure_compile: compile one case and return wall-clock ms
# Usage: measure_compile <co_file> <extra_flags...>
# Outputs: total_ms nvcc_backend_ms
measure_compile() {
    local co_file="$1"
    shift
    local out_obj="${E2E_WORK}/$(basename "$co_file" .co).o"
    local nvcc_time_log="${E2E_WORK}/$(basename "$co_file" .co).nvcc_time.log"
    # Remove old artifacts
    rm -f "$out_obj" "$nvcc_time_log"

    # Full compile: choreo -> CUDA source -> nvcc -> .o
    # Use -c (compile without linking) + -t cute
    # The -v flag makes choreo print the invoked subcommands (nvcc etc.)
    local t0 t1
    t0=$(date +%s%N)
    if ! timeout 300 "${CHOREO_BIN}" "$co_file" -c -t cute "$@" -o "$out_obj" >/dev/null 2>"${nvcc_time_log}" ; then
        echo "error error error"
        return 0
    fi
    t1=$(date +%s%N)
    local total_ms=$(( (t1 - t0) / 1000000 ))

    # Try to extract nvcc backend time from verbose output or by measuring separately
    # Method: compile with -es (emit source only) to get frontend time,
    # then subtract from total to get backend time
    local es_out="${E2E_WORK}/$(basename "$co_file" .co).cu"
    rm -f "$es_out"
    local t2 t3
    t2=$(date +%s%N)
    if timeout 120 "${CHOREO_BIN}" "$co_file" -es -t cute "$@" -o "$es_out" >/dev/null 2>&1; then
        t3=$(date +%s%N)
        local frontend_ms=$(( (t3 - t2) / 1000000 ))
        local backend_ms=$(( total_ms - frontend_ms ))
        echo "${total_ms} ${backend_ms} ${frontend_ms}"
    else
        echo "${total_ms} error error"
    fi
}

median_of() {
    # Read space-separated values and return median
    local vals=("$@")
    local n=${#vals[@]}
    if [ "$n" -eq 0 ]; then echo "error"; return; fi
    local sorted=($(printf '%s\n' "${vals[@]}" | sort -n))
    local mid=$(( (n - 1) / 2 ))
    echo "${sorted[$mid]}"
}

CASE_COUNT=0
for cat_dir in "${CASES_DIR}"/*/; do
    [[ -d "$cat_dir" ]] || continue
    category=$(basename "$cat_dir")
    [[ "$category" == "scripts" ]] && continue

    for co_file in "$cat_dir"/*.co; do
        [[ -f "$co_file" ]] || continue
        # Only dynamic cases
        if grep -q "#ifdef __STATIC_SHAPE__" "$co_file" && ! grep -q "#define __STATIC_SHAPE__" "$co_file"; then
            CASE_COUNT=$((CASE_COUNT + 1))
        fi
    done
done
echo "Total dynamic cases to measure: ${CASE_COUNT}"
echo ""

IDX=0
for cat_dir in $(ls -d "${CASES_DIR}"/*/ 2>/dev/null | sort); do
    [[ -d "$cat_dir" ]] || continue
    category=$(basename "$cat_dir")
    [[ "$category" == "scripts" ]] && continue

    for co_file in $(ls "$cat_dir"/*.co 2>/dev/null | sort); do
        [[ -f "$co_file" ]] || continue
        if ! grep -q "#ifdef __STATIC_SHAPE__" "$co_file" || grep -q "#define __STATIC_SHAPE__" "$co_file"; then
            continue
        fi
        case_name=$(basename "$co_file" .co)
        IDX=$((IDX + 1))

        # Resume support: skip cases already in CSV
        if [[ -f "${E2E_CSV}" ]] && grep -q "^${category},${case_name}," "${E2E_CSV}"; then
            echo "[${IDX}/${CASE_COUNT}] ${category}/${case_name} (cached)"
            continue
        fi
        echo "[${IDX}/${CASE_COUNT}] ${category}/${case_name}"

        # Create static version of the source
        static_co="${E2E_WORK}/${category}__${case_name}__static.co"
        {
            echo "#define __STATIC_SHAPE__"
            cat "$co_file"
        } > "$static_co"

        notes=""
        dyn_total_times=()
        dyn_backend_times=()
        dyn_frontend_times=()
        sta_total_times=()
        sta_backend_times=()
        sta_frontend_times=()

        for rep in $(seq 1 ${E2E_REPS}); do
            # Dynamic compile
            read -r dtot dback dfront <<< "$(measure_compile "$co_file")"
            if [[ "$dtot" == "error" ]]; then
                notes="${notes}dyn-compile-fail-rep${rep};"
                break
            fi
            dyn_total_times+=("$dtot")
            [[ "$dback" != "error" ]] && dyn_backend_times+=("$dback")
            [[ "$dfront" != "error" ]] && dyn_frontend_times+=("$dfront")

            # Static compile
            read -r stot sback sfront <<< "$(measure_compile "$static_co")"
            if [[ "$stot" == "error" ]]; then
                notes="${notes}sta-compile-fail-rep${rep};"
                break
            fi
            sta_total_times+=("$stot")
            [[ "$sback" != "error" ]] && sta_backend_times+=("$sback")
            [[ "$sfront" != "error" ]] && sta_frontend_times+=("$sfront")
        done

        # Compute medians
        dyn_e2e=$(median_of "${dyn_total_times[@]+"${dyn_total_times[@]}"}")
        sta_e2e=$(median_of "${sta_total_times[@]+"${sta_total_times[@]}"}")
        dyn_nvcc=$(median_of "${dyn_backend_times[@]+"${dyn_backend_times[@]}"}")
        sta_nvcc=$(median_of "${sta_backend_times[@]+"${sta_backend_times[@]}"}")
        dyn_fe=$(median_of "${dyn_frontend_times[@]+"${dyn_frontend_times[@]}"}")
        sta_fe=$(median_of "${sta_frontend_times[@]+"${sta_frontend_times[@]}"}")

        # Compute overhead percentages
        if [[ "$dyn_e2e" != "error" && "$sta_e2e" != "error" && "$sta_e2e" -gt 0 ]]; then
            e2e_ovhd=$(awk "BEGIN{printf \"%.4f\", ($dyn_e2e/$sta_e2e - 1)*100}")
        else
            e2e_ovhd="error"
        fi
        if [[ "$dyn_nvcc" != "error" && "$sta_nvcc" != "error" && "$sta_nvcc" -gt 0 ]]; then
            nvcc_ovhd=$(awk "BEGIN{printf \"%.4f\", ($dyn_nvcc/$sta_nvcc - 1)*100}")
        else
            nvcc_ovhd="error"
        fi
        if [[ "$dyn_fe" != "error" && "$sta_fe" != "error" && "$sta_fe" -gt 0 ]]; then
            fe_ovhd=$(awk "BEGIN{printf \"%.4f\", ($dyn_fe/$sta_fe - 1)*100}")
        else
            fe_ovhd="error"
        fi

        echo "  e2e: dyn=${dyn_e2e}ms sta=${sta_e2e}ms ovhd=${e2e_ovhd}%"
        echo "  nvcc_backend: dyn=${dyn_nvcc}ms sta=${sta_nvcc}ms ovhd=${nvcc_ovhd}%"
        echo "  frontend: dyn=${dyn_fe}ms sta=${sta_fe}ms ovhd=${fe_ovhd}%"

        echo "${category},${case_name},${dyn_e2e},${sta_e2e},${e2e_ovhd},${dyn_nvcc},${sta_nvcc},${nvcc_ovhd},${dyn_fe},${sta_fe},${fe_ovhd},${notes}" >> "${E2E_CSV}"
    done
done

echo ""
echo "========================================================"
echo "PHASE B Complete at $(date)"
echo "Results: ${E2E_CSV}"
echo "========================================================"

# Quick summary
echo ""
echo "End-to-end overhead summary:"
awk -F, 'NR>1 && $5!="error" {n++; s+=$5} END{if(n>0) printf "  Mean E2E overhead: %.2f%% (%d cases)\n", s/n, n; else print "  No valid cases"}' "${E2E_CSV}"
awk -F, 'NR>1 && $8!="error" {n++; s+=$8} END{if(n>0) printf "  Mean nvcc-backend overhead: %.2f%% (%d cases)\n", s/n, n}' "${E2E_CSV}"
awk -F, 'NR>1 && $11!="error" {n++; s+=$11} END{if(n>0) printf "  Mean frontend overhead: %.2f%% (%d cases)\n", s/n, n}' "${E2E_CSV}"

echo ""
echo "All phases complete at $(date)"
