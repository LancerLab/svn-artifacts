#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./mlir_common.sh
source "$SCRIPT_DIR/mlir_common.sh"
# shellcheck source=./triton_common.sh
source "$SCRIPT_DIR/triton_common.sh"

ensure_workspace_layout
ensure_command awk
ensure_command grep

CHOREO_EXE="$(resolve_choreo_bin)"
TRITON_PYTHON_BIN="$(resolve_triton_python)"
MANIFEST="${1:-$WORKSPACE_ROOT/benchmark/triton/manifest.csv}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_ROOT/benchmark/triton/results}"
TEMP_DIR="$RESULTS_DIR/tmp"

mkdir -p "$RESULTS_DIR" "$TEMP_DIR"

RAW_CSV="$RESULTS_DIR/compare_results.csv"
SUMMARY_TXT="$RESULTS_DIR/summary.txt"

printf 'category,case_name,expected,choreo_status,choreo_class,choreo_total,choreo_resolved,choreo_concrete,choreo_symbolic,choreo_partial,choreo_time_ms,triton_status,triton_mode,triton_explicit_assertions,triton_input_shape,triton_output_shape,triton_time_ms,triton_notes,notes\n' > "$RAW_CSV"

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
        match = re.search(r'Type:\s+(.+)$', line)
        if not match:
            continue
        type_info = match.group(1)
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
print(f"success,{total},{resolved},{concrete},{symbolic},{partial},{duration}")
PY
}

run_triton_case() {
  local triton_case="$1"
  local runner_args="$2"
  local output_file="$3"
  local error_file="$4"

  local rc
  set +e
  "$TRITON_PYTHON_BIN" "$triton_case" $runner_args > "$output_file" 2> "$error_file"
  rc=$?
  set -e

  python3 - "$output_file" "$rc" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text().strip()
rc = int(sys.argv[2])
if not text:
    if rc == 2:
        print('n/a,unknown,0,,,0,No Triton summary emitted')
    else:
        print('failed,unknown,0,,,0,Missing Triton summary output')
    raise SystemExit(0)

summary = json.loads(text)
status = summary.get('status', 'failed')
mode = summary.get('mode', 'unknown')
explicit_assertions = summary.get('explicit_assertions', 0)
input_shape = summary.get('input_shape', '')
output_shape = summary.get('output_shape', '')
time_ms = summary.get('time_ms', 0)
notes = summary.get('notes', '')
print(f"{status},{mode},{explicit_assertions},{input_shape},{output_shape},{time_ms},{notes}")
PY
}

while IFS=',' read -r category case_name choreo_case triton_case expected runner_args notes || [[ -n "${category:-}" ]]; do
  [[ "$category" == "category" ]] && continue
  category=$(trim "$category")
  case_name=$(trim "$case_name")
  choreo_case=$(trim "$choreo_case")
  triton_case=$(trim "$triton_case")
  expected=$(trim "$expected")
  runner_args=$(trim "$runner_args")
  notes=$(trim "$notes")

  choreo_status="not-run"
  choreo_class="missing"
  choreo_total=0
  choreo_resolved=0
  choreo_concrete=0
  choreo_symbolic=0
  choreo_partial=0
  choreo_time_ms=0

  choreo_abs="$WORKSPACE_ROOT/$choreo_case"
  if [[ -f "$choreo_abs" ]]; then
    choreo_out="$TEMP_DIR/${case_name}_choreo.txt"
    IFS=',' read -r choreo_status choreo_total choreo_resolved choreo_concrete choreo_symbolic choreo_partial choreo_time_ms <<< "$(run_choreo_case "$choreo_abs" "$choreo_out")"
    choreo_class="$(classify_choreo "$choreo_concrete" "$choreo_symbolic" "$choreo_partial")"
  else
    choreo_status="missing"
  fi

  triton_abs="$WORKSPACE_ROOT/$triton_case"
  [[ -f "$triton_abs" ]] || die "Missing Triton case: $triton_abs"
  triton_out="$TEMP_DIR/${case_name}_triton.json"
  triton_err="$TEMP_DIR/${case_name}_triton.err"
  IFS=',' read -r triton_status triton_mode triton_explicit_assertions triton_input_shape triton_output_shape triton_time_ms triton_notes <<< "$(run_triton_case "$triton_abs" "$runner_args" "$triton_out" "$triton_err")"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$category" "$case_name" "$expected" \
    "$choreo_status" "$choreo_class" "$choreo_total" "$choreo_resolved" "$choreo_concrete" "$choreo_symbolic" "$choreo_partial" "$choreo_time_ms" \
    "$triton_status" "$triton_mode" "$triton_explicit_assertions" "$triton_input_shape" "$triton_output_shape" "$triton_time_ms" "$triton_notes" \
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
    vals = [float(v) for v in values if str(v) not in {'', 'None'}]
    return statistics.mean(vals) if vals else 0.0

triton_success = sum(r['triton_status'] == 'success' for r in rows)
triton_na = sum(r['triton_status'] == 'n/a' for r in rows)
summary.write_text(
    '\n'.join([
        'Choreo vs Triton comparison summary',
        '===================================',
        f'Total cases: {len(rows)}',
        f'Triton successful cases: {triton_success}',
        f'Triton N/A cases: {triton_na}',
        f'Average Triton explicit assertions per case: {avg(r["triton_explicit_assertions"] for r in rows):.2f}',
        f'Average Triton runtime (ms): {avg(r["triton_time_ms"] for r in rows):.2f}',
    ]) + '\n'
)
PY

echo "Wrote $RAW_CSV"
echo "Wrote $SUMMARY_TXT"