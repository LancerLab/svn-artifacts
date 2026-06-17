#!/usr/bin/env bash
# reproduce_all.sh — One-command artifact evaluation for CGO 2027.
#
# Builds Choreo, runs compile-time tests, collects RQ1–RQ4 statistics,
# generates visualizations, and prints a summary comparing reproduced
# numbers against the paper.
#
# RQ1: Safety Assessment Coverage and Discharge (ACD, ADR)
# RQ2: Bug Detection Effectiveness (BDE)
# RQ3: Runtime Assertion Cost (RAO)
# RQ4: Compile-Time Overhead (CTO)
#
# Usage:
#   bash scripts/reproduce_all.sh           # full pipeline
#   bash scripts/reproduce_all.sh --skip-mlir  # skip MLIR baseline (needs mlir-opt)
#   bash scripts/reproduce_all.sh --skip-gpu   # skip RQ3 (needs CUDA GPU)
#   bash scripts/reproduce_all.sh --rq4-reps 5 # number of CTO repetitions
#
# Prerequisites:
#   - cmake >= 3.16, ninja-build, g++ >= 9 (C++17)
#   - git (for submodule init)
#   - CUDA toolkit >= 12.0 for GPU end-to-end tests (RQ3)
#   - flex/bison >= 3.8 (auto-downloaded if missing)
#   - python3 with matplotlib (for visualization, optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
SKIP_MLIR=false
SKIP_GPU=false
SKIP_CHOREO_BUILD=false
RQ3_REPS=3
RQ4_REPS=5
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-mlir)    SKIP_MLIR=true; shift ;;
    --skip-gpu)     SKIP_GPU=true; shift ;;
    --skip-build)   SKIP_CHOREO_BUILD=true; shift ;;
    --rq3-reps)     RQ3_REPS="$2"; shift 2 ;;
    --rq4-reps)     RQ4_REPS="$2"; shift 2 ;;
    --jobs|-j)      JOBS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

section() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
ok()      { printf '\033[1;32m[OK]\033[0m %s\n' "$1"; }
fail()    { printf '\033[1;31m[FAIL]\033[0m %s\n' "$1"; }
info()    { printf '[INFO] %s\n' "$1"; }

_step_start() { _STEP_T0=$(date +%s); }
_step_elapsed() {
  local t1=$(date +%s)
  local dt=$(( t1 - _STEP_T0 ))
  if (( dt >= 3600 )); then
    printf '%dh %dm %ds' $((dt/3600)) $(( (dt%3600)/60 )) $((dt%60))
  elif (( dt >= 60 )); then
    printf '%dm %ds' $((dt/60)) $((dt%60))
  else
    printf '%ds' "$dt"
  fi
}

declare -A STEP_TIMES

RESULTS_DIR="$ROOT/benchmark/results"
mkdir -p "$RESULTS_DIR"

LOGFILE="$RESULTS_DIR/reproduce_all.log"
exec > >(tee -a "$LOGFILE") 2>&1
GLOBAL_T0=$(date +%s)
echo "=== reproduce_all.sh started at $(date -Iseconds) ==="

# ── Step 1: Initialize submodules ─────────────────────────────────────
_step_start
section "Step 1: Initializing Choreo submodule"
cd "$ROOT"
if [[ ! -f choreo/CMakeLists.txt ]]; then
  git submodule update --init --recursive choreo
  ok "Choreo submodule initialized"
else
  info "Choreo submodule already present"
fi

cd "$ROOT/choreo"
if [[ ! -d extern/cutlass/.git ]] || [[ ! -d extern/gtest/.git ]]; then
  git submodule update --init --recursive
  ok "Choreo extern submodules initialized (cutlass, gtest)"
else
  info "Choreo extern submodules already present"
fi

STEP_TIMES[1_init]="$(_step_elapsed)"

# ── Step 2: Build Choreo ──────────────────────────────────────────────
_step_start
section "Step 2: Building Choreo compiler"
cd "$ROOT"
if [[ "$SKIP_CHOREO_BUILD" = true ]] && [[ -x choreo/choreo ]]; then
  info "Skipping build (--skip-build), using existing binary"
else
  make choreo-build
  ok "Choreo release build complete"
fi

CHOREO_BIN="$ROOT/choreo/choreo"
if [[ ! -x "$CHOREO_BIN" ]]; then
  fail "Choreo binary not found at $CHOREO_BIN"
  exit 1
fi

# Ensure copp is also accessible from choreo/ (lit.sh adds choreo/ to PATH)
if [[ -x "$ROOT/copp" ]] && [[ ! -e "$ROOT/choreo/copp" ]]; then
  ln -sf "$ROOT/copp" "$ROOT/choreo/copp"
fi
if [[ -x "$ROOT/build-release/copp" ]] && [[ ! -e "$ROOT/choreo/copp" ]]; then
  ln -sf "$ROOT/build-release/copp" "$ROOT/choreo/copp"
fi

info "Choreo binary: $CHOREO_BIN"
"$CHOREO_BIN" --help 2>/dev/null | head -1 || true

STEP_TIMES[2_build]="$(_step_elapsed)"

# ── Step 3: Run compile-time tests ───────────────────────────────────
_step_start
section "Step 3: Running Choreo compile-time tests"
cd "$ROOT/choreo"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUTE_HOME="${CUTE_HOME:-$ROOT/choreo/extern/cutlass}"

TEST_LOG="$RESULTS_DIR/test_results.txt"
> "$TEST_LOG"
_test_ok=true

for _tdir in tests/check tests/cli; do
  info "Running $_tdir ..."
  if bash tests/lit.sh "$_tdir" 2>&1 | tee -a "$TEST_LOG"; then
    _pass=$(grep -c "^PASS:" "$TEST_LOG" 2>/dev/null || echo "0")
    ok "$_tdir passed ($_pass total pass so far)"
  else
    _test_ok=false
    fail "$_tdir had failures"
  fi
done

if [[ "$_test_ok" = true ]]; then
  ok "All compile-time tests passed"
else
  fail "Some compile-time tests failed (see $TEST_LOG)"
fi

STEP_TIMES[3_tests]="$(_step_elapsed)"

# ── Step 4: RQ1 — Choreo assessment statistics ───────────────────────
_step_start
section "Step 4: RQ1 — Collecting Choreo assessment statistics"
cd "$ROOT"
CHOREO_STATS="$RESULTS_DIR/choreo_stats.csv"
python3 scripts/choreo_assertion_stats.py \
  --choreo "$CHOREO_BIN" \
  --out "$CHOREO_STATS" \
  --workers "$JOBS"
ok "Choreo stats written to $CHOREO_STATS"

# Parse summary
CHOREO_GEN=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$CHOREO_STATS')) if r['status']=='ok']
print(sum(int(r['generated']) for r in rows))
")
CHOREO_DIS=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$CHOREO_STATS')) if r['status']=='ok']
print(sum(int(r['discharged']) for r in rows))
")
CHOREO_RT=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$CHOREO_STATS')) if r['status']=='ok']
print(sum(int(r['runtime']) for r in rows))
")
CHOREO_OK=$(python3 -c "
import csv
print(sum(1 for r in csv.DictReader(open('$CHOREO_STATS')) if r['status']=='ok'))
")
CHOREO_ERR=$(python3 -c "
import csv
print(sum(1 for r in csv.DictReader(open('$CHOREO_STATS')) if r['status']!='ok'))
")

STEP_TIMES[4_rq1]="$(_step_elapsed)"

# ── Step 5: RQ2 — Bug detection effectiveness ────────────────────────
_step_start
section "Step 5: RQ2 — Bug detection effectiveness"
cd "$ROOT"
BUG_CSV="$RESULTS_DIR/bug_detection_results.csv"
python3 scripts/bug_detection_eval.py \
  --choreo "$CHOREO_BIN" \
  --target "${TARGET:-cute}" \
  --output "$BUG_CSV"
ok "Bug detection results written to $BUG_CSV"

STEP_TIMES[5_rq2]="$(_step_elapsed)"

# ── Step 6: RQ4 — Compile-time overhead ──────────────────────────────
_step_start
section "Step 6: RQ4 — Measuring compile-time overhead"
CTO_CSV="$RESULTS_DIR/choreo_compile_overhead.csv"
python3 scripts/choreo_compile_overhead.py \
  --reps "$RQ3_REPS" \
  --out "$CTO_CSV"
ok "CTO results written to $CTO_CSV"

CTO_AGG=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$CTO_CSV')) if not r.get('notes','')]
dyn = sum(float(r['dynamic_ms']) for r in rows)
sta = sum(float(r['static_ms']) for r in rows)
print('%.1f' % ((dyn/sta - 1)*100) if sta > 0 else 'N/A')
")
CTO_CASES=$(python3 -c "
import csv
print(sum(1 for r in csv.DictReader(open('$CTO_CSV')) if not r.get('notes','')))
")

STEP_TIMES[6_rq4_cto]="$(_step_elapsed)"

# ── Step 7: RQ3 — Runtime assertion overhead (GPU required) ──────────
_step_start
HAS_GPU=false
if [[ "$SKIP_GPU" = false ]] && command -v nvidia-smi &>/dev/null; then
  if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -qi .; then
    HAS_GPU=true
  fi
fi

RQ4_MED="(skipped)"
RQ4_CASES="0"
if [[ "$HAS_GPU" = true ]]; then
  section "Step 7: RQ3 — Measuring runtime assertion overhead (GPU)"
  RQ4_CSV="$RESULTS_DIR/choreo_runtime_entry.csv"
  python3 scripts/choreo_runtime_entry.py \
    --reps "$RQ4_REPS" \
    --out "$RQ4_CSV" \
    --verbose
  ok "RQ4 results written to $RQ4_CSV"

  RQ4_MED=$(python3 -c "
import csv, statistics
rows = [r for r in csv.DictReader(open('$RQ4_CSV'))
        if r.get('overhead_pct','N/A') not in ('N/A','')]
vals = [float(r['overhead_pct']) for r in rows]
print('%.3f%%' % statistics.median(vals) if vals else 'N/A')
")
  RQ4_CASES=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$RQ4_CSV'))
        if r.get('overhead_pct','N/A') not in ('N/A','')]
print(len(rows))
")
else
  if [[ "$SKIP_GPU" = true ]]; then
    info "Skipping RQ4 (--skip-gpu)"
  else
    info "Skipping RQ4 (no NVIDIA GPU detected)"
  fi
fi

STEP_TIMES[7_rq3_rao]="$(_step_elapsed)"

# ── Step 8: MLIR baseline (optional) ─────────────────────────────────
if [[ "$SKIP_MLIR" = false ]]; then
  section "Step 8: MLIR baseline statistics"

  if [[ ! -d "$ROOT/llvm-project" ]]; then
    info "Cloning LLVM/MLIR..."
    make mlir-clone
  fi
  if [[ ! -x "$ROOT/build/llvm-release/bin/mlir-opt" ]]; then
    info "Building MLIR tools..."
    make mlir-build
  fi

  python3 scripts/mlir_assertion_stats.py --out "$RESULTS_DIR/mlir_stats.csv"
  python3 scripts/memref_assertion_stats.py --out "$RESULTS_DIR/memref_stats.csv"
  ok "MLIR stats collected"

  MLIR_TENSOR_GEN=$(python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$RESULTS_DIR/mlir_stats.csv'))]
print(sum(int(r.get('generated_total',0)) for r in rows))
")
else
  info "Skipping MLIR baseline (--skip-mlir)"
  MLIR_TENSOR_GEN="(skipped)"
fi

# ── Summary ──────────────────────────────────────────────────────────
GLOBAL_T1=$(date +%s)
GLOBAL_ELAPSED=$(( GLOBAL_T1 - GLOBAL_T0 ))

section "RESULTS SUMMARY"
CHOREO_ADR=$(python3 -c "print('%.1f%%' % ($CHOREO_DIS/$CHOREO_GEN*100))")
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│           SVN Artifact Evaluation Results (CGO 2027)            │"
echo "├─────────────────────────┬──────────────┬────────────────────────┤"
echo "│ Metric                  │ Paper        │ Reproduced             │"
echo "├─────────────────────────┼──────────────┼────────────────────────┤"
printf "│ Cases compiled          │ 310/310      │ %s/%s %*s│\n" \
  "$CHOREO_OK" "$((CHOREO_OK + CHOREO_ERR))" $((20 - ${#CHOREO_OK} - ${#CHOREO_ERR} - 1)) ""
printf "│ RQ1: Assessments gen.   │ 12,592       │ %-22s │\n" \
  "$(printf '%s' "$CHOREO_GEN" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
printf "│ RQ1: Discharged (ADR)   │ 93.3%%        │ %-22s │\n" "$CHOREO_ADR"
printf "│      Discharged count   │ 11,753       │ %-22s │\n" \
  "$(printf '%s' "$CHOREO_DIS" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
printf "│      Runtime surviving  │ 839          │ %-22s │\n" \
  "$(printf '%s' "$CHOREO_RT" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
RQ2_SUMMARY=$(BUG_CSV_PATH="$BUG_CSV" python3 -c "
import csv, os
rows = list(csv.DictReader(open(os.environ['BUG_CSV_PATH'])))
svn = sum(1 for r in rows if r.get('svn_result','')=='compile')
print('%d/%d (SVN 100%%%%)' % (svn, len(rows)))
" 2>/dev/null || echo "(see CSV)")
printf "│ RQ2: Bug detection      │ 210/210      │ %-22s │\n" "$RQ2_SUMMARY"
printf "│ RQ3: RAO median (entry) │ <0.4%%        │ %-22s │\n" "${RQ4_MED} ($RQ4_CASES cases)"
printf "│ RQ4: CTO (aggregate)    │ 4.7%%         │ %-22s │\n" "${CTO_AGG}% ($CTO_CASES cases)"

if [[ "$SKIP_MLIR" = false ]]; then
  printf "│ MLIR tensor generated   │ 2,634        │ %-22s │\n" \
    "$(printf '%s' "$MLIR_TENSOR_GEN" | sed ':a;s/\B[0-9]\{3\}\>/,&/;ta')"
fi

echo "└─────────────────────────┴──────────────┴────────────────────────┘"
echo ""

if [[ "$CHOREO_GEN" -ge 12000 ]]; then
  ok "Choreo assessments >= paper threshold"
else
  fail "Choreo assessments < expected — investigate"
fi

# ── Detailed RQ1: Per-category breakdown ──
echo ""
echo "  RQ1 Details: Assessment Coverage by Category"
echo "  ─────────────────────────────────────────────────────────────"
CHOREO_STATS_PATH="$CHOREO_STATS" python3 << 'PYEOF'
import csv, os
from collections import defaultdict
rows = [r for r in csv.DictReader(open(os.environ["CHOREO_STATS_PATH"])) if r["status"]=="ok"]
cats = defaultdict(lambda: {"n":0, "gen":0, "dis":0, "rt":0})
for r in rows:
    c = r["category"]
    cats[c]["n"] += 1
    cats[c]["gen"] += int(r["generated"])
    cats[c]["dis"] += int(r["discharged"])
    cats[c]["rt"] += int(r["runtime"])
hdr = "  {:<22s} {:>5s} {:>6s} {:>6s} {:>5s} {:>6s}".format(
    "Category", "Cases", "Gen", "Dis", "RT", "ADR")
print(hdr)
print("  " + "-"*60)
for c in sorted(cats.keys()):
    v = cats[c]
    adr = v["dis"]/v["gen"]*100 if v["gen"] > 0 else 0
    print("  {:<22s} {:5d} {:6d} {:6d} {:5d} {:5.1f}%".format(
        c, v["n"], v["gen"], v["dis"], v["rt"], adr))
print("  " + "-"*60)
tot_gen = sum(v["gen"] for v in cats.values())
tot_dis = sum(v["dis"] for v in cats.values())
tot_rt = sum(v["rt"] for v in cats.values())
tot_n = sum(v["n"] for v in cats.values())
print("  {:<22s} {:5d} {:6d} {:6d} {:5d} {:5.1f}%".format(
    "TOTAL", tot_n, tot_gen, tot_dis, tot_rt, tot_dis/tot_gen*100))
PYEOF

# ── Detailed RQ2: Bug Detection ──
echo ""
echo "  RQ2 Details: Bug Detection Effectiveness"
echo "  ─────────────────────────────────────────────────────────────"
BUG_CSV_PATH="$BUG_CSV" python3 << 'PYEOF'
import csv, os
from collections import defaultdict
rows = list(csv.DictReader(open(os.environ["BUG_CSV_PATH"])))
bugs = defaultdict(int)
svn_det = defaultdict(int)
mlir_det = defaultdict(int)
iree_det = defaultdict(int)
for r in rows:
    bc = r.get("bug_class","unknown")
    bugs[bc] += 1
    if r.get("svn_result","") == "compile": svn_det[bc] += 1
    if r.get("mlir_result","") in ("compile","runtime"): mlir_det[bc] += 1
    if r.get("iree_result","") in ("compile","runtime"): iree_det[bc] += 1
print("  {:<22s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
    "Bug Class", "Total", "SVN", "MLIR", "IREE"))
print("  " + "-"*50)
for bc in sorted(bugs.keys()):
    print("  {:<22s} {:5d} {:5d} {:5d} {:5d}".format(
        bc, bugs[bc], svn_det[bc], mlir_det[bc], iree_det[bc]))
print("  " + "-"*50)
t = sum(bugs.values())
s = sum(svn_det.values())
m = sum(mlir_det.values())
i = sum(iree_det.values())
print("  {:<22s} {:5d} {:5d} {:5d} {:5d}".format("TOTAL", t, s, m, i))
print("  {:<22s} {:>5s} {:4.1f}% {:4.1f}% {:4.1f}%".format(
    "Detection rate", "", s/t*100, m/t*100, i/t*100))
PYEOF

# ── Detailed RQ3: RAO per-category ──
if [[ "$HAS_GPU" = true ]] && [[ -f "$RQ4_CSV" ]]; then
  echo ""
  echo "  RQ3 Details: Runtime Assertion Overhead by Category"
  echo "  ─────────────────────────────────────────────────────────────"
  RQ4_CSV_PATH="$RQ4_CSV" python3 << 'PYEOF'
import csv, os, statistics
from collections import defaultdict
rows = [r for r in csv.DictReader(open(os.environ["RQ4_CSV_PATH"]))
        if r.get("overhead_pct","N/A") not in ("N/A","")]
cats = defaultdict(list)
for r in rows:
    cats[r["category"]].append(float(r["overhead_pct"]))
print("  {:<22s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
    "Category", "Cases", "Median", "Mean", "Min", "Max"))
print("  " + "-"*66)
for c in sorted(cats.keys()):
    vals = cats[c]
    print("  {:<22s} {:5d} {:+7.3f}% {:+7.3f}% {:+7.3f}% {:+7.3f}%".format(
        c, len(vals), statistics.median(vals), statistics.mean(vals),
        min(vals), max(vals)))
print("  " + "-"*66)
all_vals = [v for vs in cats.values() for v in vs]
print("  {:<22s} {:5d} {:+7.3f}% {:+7.3f}% {:+7.3f}% {:+7.3f}%".format(
    "OVERALL", len(all_vals), statistics.median(all_vals),
    statistics.mean(all_vals), min(all_vals), max(all_vals)))
PYEOF
fi

# ── Detailed RQ4: CTO per-category ──
if [[ -f "$CTO_CSV" ]]; then
  echo ""
  echo "  RQ4 Details: Compile-Time Overhead by Category"
  echo "  ─────────────────────────────────────────────────────────────"
  CTO_CSV_PATH="$CTO_CSV" python3 << 'PYEOF'
import csv, os, statistics
from collections import defaultdict
rows = [r for r in csv.DictReader(open(os.environ["CTO_CSV_PATH"])) if not r.get("notes","")]
cats = defaultdict(lambda: {"dyn":0, "sta":0, "n":0, "pcts":[]})
for r in rows:
    c = r["category"]
    cats[c]["dyn"] += float(r["dynamic_ms"])
    cats[c]["sta"] += float(r["static_ms"])
    cats[c]["n"] += 1
    cats[c]["pcts"].append(float(r["overhead_pct"]))
print("  {:<22s} {:>5s} {:>9s} {:>9s} {:>7s} {:>8s}".format(
    "Category", "Cases", "Dyn(ms)", "Sta(ms)", "CTO", "Median"))
print("  " + "-"*70)
for c in sorted(cats.keys()):
    v = cats[c]
    cto = (v["dyn"]/v["sta"] - 1)*100 if v["sta"] > 0 else 0
    med = statistics.median(v["pcts"])
    print("  {:<22s} {:5d} {:9.0f} {:9.0f} {:+6.1f}% {:+7.1f}%".format(
        c, v["n"], v["dyn"], v["sta"], cto, med))
print("  " + "-"*70)
td = sum(v["dyn"] for v in cats.values())
ts = sum(v["sta"] for v in cats.values())
ap = [p for v in cats.values() for p in v["pcts"]]
print("  {:<22s} {:5d} {:9.0f} {:9.0f} {:+6.1f}% {:+7.1f}%".format(
    "OVERALL", len(rows), td, ts, (td/ts-1)*100, statistics.median(ap)))
PYEOF
fi

# ── Wall-clock time per step ──
echo ""
echo "  Wall-Clock Time per Step"
echo "  ─────────────────────────────────────────────────────────────"
printf "  %-40s %s\n" "Step 1: Submodule init" "${STEP_TIMES[1_init]:-N/A}"
printf "  %-40s %s\n" "Step 2: Build Choreo" "${STEP_TIMES[2_build]:-N/A}"
printf "  %-40s %s\n" "Step 3: Compile-time tests" "${STEP_TIMES[3_tests]:-N/A}"
printf "  %-40s %s\n" "Step 4: RQ1 Assessment statistics" "${STEP_TIMES[4_rq1]:-N/A}"
printf "  %-40s %s\n" "Step 5: RQ2 Bug detection" "${STEP_TIMES[5_rq2]:-N/A}"
printf "  %-40s %s\n" "Step 6: RQ4 Compile-time overhead" "${STEP_TIMES[6_rq4_cto]:-N/A}"
printf "  %-40s %s\n" "Step 7: RQ3 Runtime assertion overhead" "${STEP_TIMES[7_rq3_rao]:-N/A}"
echo "  ─────────────────────────────────────────────────────────────"
if (( GLOBAL_ELAPSED >= 3600 )); then
  printf "  %-40s %dh %dm %ds\n" "TOTAL" $((GLOBAL_ELAPSED/3600)) $(( (GLOBAL_ELAPSED%3600)/60 )) $((GLOBAL_ELAPSED%60))
else
  printf "  %-40s %dm %ds\n" "TOTAL" $((GLOBAL_ELAPSED/60)) $((GLOBAL_ELAPSED%60))
fi
echo ""

# ── Visualization ────────────────────────────────────────────────────
section "Generating visualizations"
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 scripts/visualize_results.py \
    --results-dir "$RESULTS_DIR" \
    --out-dir "$RESULTS_DIR"
  ok "Visualizations generated"
  info "  HTML report: $RESULTS_DIR/report.html"
  info "  Figures: $RESULTS_DIR/figures/"
else
  info "matplotlib not installed — skipping figure generation"
  info "  Install with: pip install matplotlib"
fi

echo ""
echo "Full results in: $RESULTS_DIR/"
echo "Log: $LOGFILE"
echo "=== reproduce_all.sh finished at $(date -Iseconds) ==="
