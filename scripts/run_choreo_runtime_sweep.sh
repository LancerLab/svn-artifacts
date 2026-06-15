#!/usr/bin/env bash

set -u

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR="$ROOT_DIR/build/runtime-check"
BENCH_ROOT="$ROOT_DIR/benchmark/choreo"
CHOREO_BIN="$ROOT_DIR/choreo/choreo"
TIMEOUT_SECS=${TIMEOUT_SECS:-45}
STATUS_DIR=${SWEEP_STATUS_DIR:-"$ROOT_DIR/benchmark/choreo/scripts/results"}
STATUS_FILE=${SWEEP_STATUS_FILE:-"$STATUS_DIR/runtime_sweep_status.tsv"}
FRESHNESS_FILE=${SWEEP_FRESHNESS_FILE:-"$STATUS_DIR/runtime_sweep_freshness.tsv"}
SWEEP_HOST=${SWEEP_HOST_NAME:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)}
FAMILY_FILTER=""
STATUS_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_choreo_runtime_sweep.sh [family]
       bash scripts/run_choreo_runtime_sweep.sh --family <family>
       bash scripts/run_choreo_runtime_sweep.sh --status [--family <family>]

Modes:
  default    Run the Choreo runtime sweep, write build/runtime-check/results.tsv,
             and update the local sweep ledger at
             benchmark/choreo/scripts/results/runtime_sweep_status.tsv.
  --status   Recompute the freshness report without running any cases. Cases are
             marked stale when they were never swept or when their tracked input
             fingerprint changed since the last recorded sweep.

Tracked inputs per case:
  - the .co source file
  - family-local common.h/common.hpp when present
  - scripts/run_choreo_runtime_sweep.sh
EOF
}

ensure_status_header() {
  mkdir -p "$OUT_DIR" "$STATUS_DIR"
  if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'family\tcase\tstatus\tupdated_at\tfingerprint\thost\tdependencies\n' > "$STATUS_FILE"
  fi
}

collect_dependencies() {
  local family=$1
  local src=$2
  local family_dir="$BENCH_ROOT/$family"

  printf '%s\n' "${src#$ROOT_DIR/}"
  if [[ -f "$family_dir/common.h" ]]; then
    printf '%s\n' "${family_dir#$ROOT_DIR/}/common.h"
  fi
  if [[ -f "$family_dir/common.hpp" ]]; then
    printf '%s\n' "${family_dir#$ROOT_DIR/}/common.hpp"
  fi
  printf '%s\n' "scripts/run_choreo_runtime_sweep.sh"
}

case_fingerprint() {
  local family=$1
  local src=$2
  local dependencies=()

  mapfile -t dependencies < <(collect_dependencies "$family" "$src")
  if [[ ${#dependencies[@]} -eq 0 ]]; then
    printf 'missing-dependencies\n'
    return 0
  fi

  (
    cd "$ROOT_DIR" || exit 1
    sha256sum "${dependencies[@]}" | sha256sum | awk '{print $1}'
  )
}

join_dependencies() {
  local family=$1
  local src=$2
  local dependencies=()
  local joined=""
  local dependency

  mapfile -t dependencies < <(collect_dependencies "$family" "$src")
  for dependency in "${dependencies[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=";"
    fi
    joined+="$dependency"
  done
  printf '%s\n' "$joined"
}

lookup_status_record() {
  local family=$1
  local case_name=$2

  awk -F '\t' -v family="$family" -v case_name="$case_name" '
    NR > 1 && $1 == family && $2 == case_name { record = $0 }
    END { if (record != "") print record }
  ' "$STATUS_FILE"
}

update_status_record() {
  local family=$1
  local case_name=$2
  local status=$3
  local fingerprint=$4
  local dependencies=$5
  local timestamp
  local tmp_file

  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  tmp_file=$(mktemp)

  awk -F '\t' -v family="$family" -v case_name="$case_name" '
    NR == 1 || !($1 == family && $2 == case_name) { print }
  ' "$STATUS_FILE" > "$tmp_file"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$family" "$case_name" "$status" "$timestamp" "$fingerprint" "$SWEEP_HOST" "$dependencies" >> "$tmp_file"
  mv "$tmp_file" "$STATUS_FILE"
}

write_freshness_report() {
  local family_dir
  local family
  local src
  local case_name
  local current_fingerprint
  local record
  local recorded_status
  local recorded_updated_at
  local recorded_fingerprint
  local recorded_host
  local recorded_dependencies
  local freshness
  local reason
  local fresh_count=0
  local stale_count=0

  ensure_status_header
  printf 'family\tcase\tfreshness\treason\tlast_status\tupdated_at\tcurrent_fingerprint\trecorded_fingerprint\thost\n' > "$FRESHNESS_FILE"

  for family_dir in "$BENCH_ROOT"/*; do
    [[ -d "$family_dir" ]] || continue
    family=$(basename "$family_dir")
    [[ "$family" == "scripts" ]] && continue
    if [[ -n "$FAMILY_FILTER" && "$family" != "$FAMILY_FILTER" ]]; then
      continue
    fi

    for src in "$family_dir"/*.co; do
      [[ -f "$src" ]] || continue
      case_name=$(basename "$src" .co)
      current_fingerprint=$(case_fingerprint "$family" "$src")
      record=$(lookup_status_record "$family" "$case_name")

      if [[ -z "$record" ]]; then
        freshness="stale"
        reason="never-swept"
        recorded_status="missing"
        recorded_updated_at="-"
        recorded_fingerprint="-"
        recorded_host="-"
      else
        IFS=$'\t' read -r _ _ recorded_status recorded_updated_at recorded_fingerprint recorded_host recorded_dependencies <<< "$record"
        if [[ "$current_fingerprint" == "$recorded_fingerprint" ]]; then
          freshness="fresh"
          reason="up-to-date"
        else
          freshness="stale"
          reason="inputs-changed"
        fi
      fi

      if [[ "$freshness" == "fresh" ]]; then
        fresh_count=$((fresh_count + 1))
      else
        stale_count=$((stale_count + 1))
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$family" "$case_name" "$freshness" "$reason" "$recorded_status" "$recorded_updated_at" "$current_fingerprint" "$recorded_fingerprint" "$recorded_host" >> "$FRESHNESS_FILE"
    done
  done

  cat "$FRESHNESS_FILE"
  printf '\nsummary\tfresh\t%d\nsummary\tstale\t%d\n' "$fresh_count" "$stale_count"
}

run_case() {
  local family=$1
  local src=$2
  local case_name
  local case_dir
  local script_path
  local gen_log
  local run_log
  local fingerprint
  local dependencies
  local status
  local run_status

  case_name=$(basename "$src" .co)
  case_dir="$OUT_DIR/$family"
  script_path="$case_dir/$case_name.sh"
  gen_log="$case_dir/$case_name.generate.log"
  run_log="$case_dir/$case_name.run.log"
  fingerprint=$(case_fingerprint "$family" "$src")
  dependencies=$(join_dependencies "$family" "$src")

  mkdir -p "$case_dir"

  if ! "$CHOREO_BIN" -gs -t cute "$src" -o "$script_path" >"$gen_log" 2>&1; then
    status="generate-fail"
    printf '%s\t%s\t%s\n' "$family" "$case_name" "$status" >> "$RESULTS_FILE"
    update_status_record "$family" "$case_name" "$status" "$fingerprint" "$dependencies"
    return 0
  fi

  /usr/bin/timeout "${TIMEOUT_SECS}s" bash "$script_path" --execute >"$run_log" 2>&1
  run_status=$?
  if [[ $run_status -ne 0 ]]; then
    if [[ $run_status -eq 124 ]]; then
      status="timeout"
    else
      status="run-fail"
    fi
    printf '%s\t%s\t%s\n' "$family" "$case_name" "$status" >> "$RESULTS_FILE"
    update_status_record "$family" "$case_name" "$status" "$fingerprint" "$dependencies"
    return 0
  fi

  status="ok"
  printf '%s\t%s\t%s\n' "$family" "$case_name" "$status" >> "$RESULTS_FILE"
  update_status_record "$family" "$case_name" "$status" "$fingerprint" "$dependencies"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family)
      FAMILY_FILTER=${2:-}
      shift 2
      ;;
    --status)
      STATUS_ONLY=1
      shift
      ;;
    --status-file)
      STATUS_FILE=${2:-}
      shift 2
      ;;
    --freshness-file)
      FRESHNESS_FILE=${2:-}
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$FAMILY_FILTER" ]]; then
        FAMILY_FILTER=$1
        shift
      else
        printf 'Unknown argument: %s\n\n' "$1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

ensure_status_header

if [[ $STATUS_ONLY -eq 1 ]]; then
  write_freshness_report
  exit 0
fi

printf 'family\tcase\tstatus\n' > "$RESULTS_FILE"

for family_dir in "$BENCH_ROOT"/*; do
  [[ -d "$family_dir" ]] || continue
  family=$(basename "$family_dir")
  [[ "$family" == "scripts" ]] && continue
  if [[ -n "$FAMILY_FILTER" && "$family" != "$FAMILY_FILTER" ]]; then
    continue
  fi

  for src in "$family_dir"/*.co; do
    [[ -f "$src" ]] || continue
    run_case "$family" "$src"
  done
done

write_freshness_report > /dev/null
cat "$RESULTS_FILE"