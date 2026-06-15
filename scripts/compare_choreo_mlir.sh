#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"

ensure_workspace_layout
ensure_command python3
ensure_command awk
ensure_command grep

CHOREO_EXE="$(resolve_choreo_bin)"
MLIR_OPT="$(resolve_mlir_tool mlir-opt)"
MANIFEST="${1:-$WORKSPACE_ROOT/benchmark/mlir/manifest.csv}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_ROOT/benchmark/mlir/results}"
TEMP_DIR="$RESULTS_DIR/tmp"
PIPELINE_FLAGS=(--canonicalize --cse)

mkdir -p "$RESULTS_DIR" "$TEMP_DIR"

RAW_CSV="$RESULTS_DIR/compare_results.csv"
SUMMARY_TXT="$RESULTS_DIR/summary.txt"

printf 'category,case_name,expected,choreo_status,choreo_class,choreo_total,choreo_resolved,choreo_concrete,choreo_symbolic,choreo_partial,choreo_time_ms,mlir_status,mlir_class,mlir_tensor_types,mlir_static_tensors,mlir_dynamic_tensors,mlir_dynamic_dims,mlir_time_ms,notes\n' > "$RAW_CSV"

trim() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}

classify_choreo() {
  local concrete="$1" symbolic="$2" partial="$3"
  if (( partial > 0 )); then
    printf 'partial\n'
  elif (( symbolic > 0 )); then
    printf 'symbolic\n'
  elif (( concrete > 0 )); then
    printf 'concrete\n'
  else
    printf 'unknown\n'
  fi
}

run_choreo_case() {
  local choreo_case="$1"
  local output_file="$2"

  local start end duration
  start=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  "$CHOREO_EXE" -i "$choreo_case" > "$output_file" 2>&1
  end=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  duration=$(( (end - start) / 1000000 ))

  python3 - "$output_file" "$duration" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
duration = int(sys.argv[2])
text = path.read_text()
total = resolved = concrete = symbolic = partial = 0
for line in text.splitlines():
    if re.match(r'^(Parameter|Symbol|Bounded|Future|Function):', line):
        total += 1
        m = re.search(r'Type:\s+(.+)$', line)
        if not m:
            continue
        type_info = m.group(1)
        shape = re.search(r'\[([^\]]+)\]', type_info)
        if not shape:
            continue
        resolved += 1
        dims = shape.group(1)
        if '?' in dims:
            partial += 1
        elif re.search(r'[A-Za-z_]', dims):
            symbolic += 1
        else:
            concrete += 1
status = 'success'
print(f"{status},{total},{resolved},{concrete},{symbolic},{partial},{duration}")
PY
}

run_mlir_case() {
  local mlir_case="$1"
  local expected="$2"
  local output_file="$3"
  local error_file="$4"

  local start end duration rc
  start=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  set +e
  "$MLIR_OPT" "${PIPELINE_FLAGS[@]}" "$mlir_case" > "$output_file" 2> "$error_file"
  rc=$?
  set -e
  end=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  duration=$(( (end - start) / 1000000 ))

  if [[ "$expected" == "expected-failure" ]]; then
    if [[ "$rc" -eq 0 ]]; then
      printf 'unexpected-success,unknown,0,0,0,0,%s\n' "$duration"
      return 0
    fi
    printf 'expected-failure,invalid,0,0,0,0,%s\n' "$duration"
    return 0
  fi

  if [[ "$rc" -ne 0 ]]; then
    printf 'failed,invalid,0,0,0,0,%s\n' "$duration"
    return 0
  fi

  python3 - "$output_file" "$duration" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
duration = int(sys.argv[2])
all_types = re.findall(r'tensor<[^>]+>', text)
static_tensors = 0
dynamic_tensors = 0
dynamic_dims = 0
for ty in all_types:
    q = ty.count('?')
    dynamic_dims += q
    if q == 0:
        static_tensors += 1
    else:
        dynamic_tensors += 1
if dynamic_tensors > 0:
    klass = 'dynamic'
elif static_tensors > 0:
    klass = 'concrete'
else:
    klass = 'unknown'
print(f"success,{klass},{len(all_types)},{static_tensors},{dynamic_tensors},{dynamic_dims},{duration}")
PY
}

while IFS=',' read -r category case_name choreo_case mlir_case expected notes; do
  [[ "$category" == "category" ]] && continue
  category=$(trim "$category")
  case_name=$(trim "$case_name")
  choreo_case=$(trim "$choreo_case")
  mlir_case=$(trim "$mlir_case")
  expected=$(trim "$expected")
  notes=$(trim "$notes")

  choreo_status="not-run"
  choreo_class="missing"
  choreo_total=0
  choreo_resolved=0
  choreo_concrete=0
  choreo_symbolic=0
  choreo_partial=0
  choreo_time_ms=0

  if [[ -n "$choreo_case" ]]; then
    choreo_abs="$choreo_case"
    if [[ "$choreo_abs" != /* ]]; then
      choreo_abs="$WORKSPACE_ROOT/$choreo_abs"
    fi
    if [[ -f "$choreo_abs" ]]; then
      choreo_out="$TEMP_DIR/${case_name}_choreo.txt"
      IFS=',' read -r choreo_status choreo_total choreo_resolved choreo_concrete choreo_symbolic choreo_partial choreo_time_ms <<< "$(run_choreo_case "$choreo_abs" "$choreo_out")"
      choreo_class="$(classify_choreo "$choreo_concrete" "$choreo_symbolic" "$choreo_partial")"
    else
      choreo_status="missing"
    fi
  fi

  mlir_abs="$mlir_case"
  if [[ "$mlir_abs" != /* ]]; then
    mlir_abs="$WORKSPACE_ROOT/$mlir_abs"
  fi
  [[ -f "$mlir_abs" ]] || die "Missing MLIR case: $mlir_abs"
  mlir_out="$TEMP_DIR/${case_name}_mlir.txt"
  mlir_err="$TEMP_DIR/${case_name}_mlir.err"
  IFS=',' read -r mlir_status mlir_class mlir_tensor_types mlir_static_tensors mlir_dynamic_tensors mlir_dynamic_dims mlir_time_ms <<< "$(run_mlir_case "$mlir_abs" "$expected" "$mlir_out" "$mlir_err")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$category" "$case_name" "$expected" \
    "$choreo_status" "$choreo_class" "$choreo_total" "$choreo_resolved" "$choreo_concrete" "$choreo_symbolic" "$choreo_partial" "$choreo_time_ms" \
    "$mlir_status" "$mlir_class" "$mlir_tensor_types" "$mlir_static_tensors" "$mlir_dynamic_tensors" "$mlir_dynamic_dims" "$mlir_time_ms" \
    "$notes" >> "$RAW_CSV"
done < "$MANIFEST"

python3 - "$RAW_CSV" "$SUMMARY_TXT" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

raw = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = list(csv.DictReader(raw.open()))
if not rows:
    summary.write_text('No rows collected.\n')
    raise SystemExit(0)

def avg(values):
    vals = [float(v) for v in values]
    return statistics.mean(vals) if vals else 0.0

choreo_success = sum(r['choreo_status'] == 'success' for r in rows)
mlir_success = sum(r['mlir_status'] == 'success' for r in rows)
mlir_expected_fail = sum(r['mlir_status'] == 'expected-failure' for r in rows)
summary.write_text(
    '\n'.join([
        'Choreo vs MLIR comparison summary',
        '=================================',
        f'Total cases: {len(rows)}',
        f'Choreo successful inference cases: {choreo_success}',
        f'MLIR successful pipeline cases: {mlir_success}',
        f'MLIR expected verifier failures: {mlir_expected_fail}',
        f'Average Choreo inference time (ms): {avg(r["choreo_time_ms"] for r in rows):.2f}',
        f'Average MLIR pipeline time (ms): {avg(r["mlir_time_ms"] for r in rows):.2f}',
    ]) + '\n'
)
PY

echo "Wrote $RAW_CSV"
echo "Wrote $SUMMARY_TXT"
