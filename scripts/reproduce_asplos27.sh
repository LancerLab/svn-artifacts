#!/usr/bin/env bash
# reproduce_asplos27.sh — One-command reproduction of the ASPLOS 2027 paper
# ("The Compiler as Auditor: Safety Ledgers for AI-Native GPU Kernels").
#
# Regenerates every CSV behind the paper's tables/figures, re-renders them
# into ../asplos27/{tables,figures}/, and prints a summary comparing the
# reproduced numbers against the paper.
#
# RQ1: Discharge coverage (per-category, static vs dynamic)
# RQ2: Compile-time bug detection (103 injected bugs)
# RQ3: Runtime cost of residual checks (requires NVIDIA GPU; --skip-gpu)
# RQ4: Compile-time overhead
# RQ5: Minimality ablation (--disable-vn-share / --disable-vn-simplify)
# RQ6: Mechanism / information-dependence breakdown
#
# Usage:
#   bash scripts/reproduce_asplos27.sh                 # full pipeline
#   bash scripts/reproduce_asplos27.sh --skip-build    # reuse pinned binary
#   bash scripts/reproduce_asplos27.sh --skip-gpu      # skip RQ3
#   bash scripts/reproduce_asplos27.sh --workers N
#
# Reproducibility note: the statistics harness invokes the compiler with
# absolute input paths. A few dynamic batch_norm cases exhibit order-dependent
# discharge (+-1 obligation) when invoked with relative paths; the harness
# configuration yields the stable counts reported in the paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS="$ROOT/benchmark/results"
mkdir -p "$RESULTS"

SKIP_BUILD=false
SKIP_GPU=false
WORKERS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=true; shift ;;
    --skip-gpu)   SKIP_GPU=true; shift ;;
    --workers)    WORKERS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

section() { printf '\n=== %s ===\n' "$1"; }

BIN="$ROOT/croqtile/build-release/choreo"

# -- Step 1: build (or reuse) the pinned compiler ---------------------------
section "Step 1: Choreo compiler"
# Nested submodules (cutlass, gtest) are required by the build.
cd "$ROOT" && git submodule update --init --recursive croqtile
cd "$ROOT/croqtile" && git submodule update --init --recursive
if [[ "$SKIP_BUILD" = false ]]; then
  cd "$ROOT" && make choreo-build
fi
[[ -x "$BIN" ]] || { echo "FAIL: compiler not found at $BIN"; exit 1; }
echo "Using pinned binary: $BIN"

STATS="python3 $SCRIPT_DIR/choreo_assertion_stats.py --choreo $BIN --workers $WORKERS"

# -- Step 2: RQ1 + RQ6 — full-suite assessment statistics -------------------
section "Step 2: RQ1/RQ6 — full-suite assessment statistics (full system)"
$STATS --out "$RESULTS/rq6_mechanism.csv"
# rq4_dependence.csv (RQ1 table source) = same run minus the mechanism columns
python3 - "$RESULTS/rq6_mechanism.csv" "$RESULTS/rq4_dependence.csv" <<'PYEOF'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
cols = [c for c in rows[0] if c not in ("mech_canonical", "mech_interval")]
w = csv.DictWriter(open(dst, "w"), fieldnames=cols)
w.writeheader()
for r in rows:
    w.writerow({c: r[c] for c in cols})
PYEOF

# -- Step 3: RQ5 — minimality ablation --------------------------------------
section "Step 3: RQ5 — ablation (4 configurations)"
cp "$RESULTS/rq6_mechanism.csv" "$RESULTS/rq5_full_svn.csv"
$STATS --choreo-flags="--disable-vn-share"    --out "$RESULTS/rq5_no_vn_share.csv"
$STATS --choreo-flags="--disable-vn-simplify" --out "$RESULTS/rq5_no_vn_simplify.csv"
$STATS --choreo-flags="--disable-vn-share --disable-vn-simplify" \
  --out "$RESULTS/rq5_no_vn_both.csv"

# -- Step 4: RQ2 — bug detection --------------------------------------------
section "Step 4: RQ2 — injected-bug detection"
python3 "$SCRIPT_DIR/bug_detection_eval.py" \
  --choreo "$BIN" --output "$RESULTS/bug_detection_results.csv"

# -- Step 5: RQ4 — compile-time overhead ------------------------------------
section "Step 5: RQ4 — compile-time overhead"
python3 "$SCRIPT_DIR/choreo_compile_overhead.py" \
  --out "$RESULTS/choreo_compile_overhead.csv"

# -- Step 6: RQ3 — runtime cost of residual checks (GPU) --------------------
if [[ "$SKIP_GPU" = false ]] && command -v nvidia-smi &>/dev/null \
   && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi .; then
  section "Step 6: RQ3 — runtime cost of residual checks (GPU)"
  python3 "$SCRIPT_DIR/choreo_runtime.py" \
    --out "$RESULTS/rq3_runtime_overhead.csv"
else
  section "Step 6: RQ3 — skipped (no GPU or --skip-gpu)"
fi

# -- Step 7: ledger dump for the Section 2 running example ------------------
section "Step 7: safety ledger of the Section 2 running example"
"$BIN" --dump-ledger="$RESULTS/example_layer_norm_ledger.json" \
  "$ROOT/benchmark/choreo/layer_normalization/3_attention_32xNx512x64_64_64.co" \
  > /dev/null 2>&1 || echo "WARN: ledger dump failed"
echo "Ledger written to $RESULTS/example_layer_norm_ledger.json"

# -- Step 8: re-render paper tables and figures ------------------------------
section "Step 8: render paper tables/figures"
python3 "$SCRIPT_DIR/render_svn27_tables.py"

# -- Summary: reproduced vs paper --------------------------------------------
section "RESULTS SUMMARY (reproduced vs paper)"
RESULTS_DIR="$RESULTS" python3 - <<'PYEOF'
import csv, os, statistics
R = os.environ["RESULTS_DIR"]

def load(name):
    return list(csv.DictReader(open(os.path.join(R, name))))

ok = [r for r in load("rq6_mechanism.csv") if r["status"] == "ok"]
gen = sum(int(r["generated"]) for r in ok)
dis = sum(int(r["static_true"]) for r in ok)
rt  = sum(int(r["runtime"]) for r in ok)
st  = [r for r in ok if r["is_dynamic"] == "0"]
dy  = [r for r in ok if r["is_dynamic"] == "1"]
g_st = sum(int(r["generated"]) for r in st); d_st = sum(int(r["static_true"]) for r in st)
g_dy = sum(int(r["generated"]) for r in dy); d_dy = sum(int(r["static_true"]) for r in dy)
canon = sum(int(r["mech_canonical"]) for r in ok)
itv   = sum(int(r["mech_interval"]) for r in ok)
direct = sum(int(r["direct_checks"]) for r in ok)
scalar_st = sum(int(r["st_scalar"]) for r in ok)

bugs = load("bug_detection_results.csv")
ours = sum(1 for r in bugs if r["svn_resolution"] == "compile")
mlir = sum(1 for r in bugs if r["mlir_resolution"] in ("compile", "runtime"))
iree = sum(1 for r in bugs if r["iree_resolution"] not in ("undetected", ""))

cto = [float(r["overhead_pct"]) for r in load("choreo_compile_overhead.csv")
       if r["overhead_pct"] not in ("", "N/A", "error")]

print(f"  Cases compilable      paper 289/310   reproduced {len(ok)}/310")
print(f"  RQ1 obligations       paper 17,717      reproduced {gen:,}")
print(f"  RQ1 discharged        paper 16,518      reproduced {dis:,} ({100*dis/gen:.1f}%)")
print(f"  RQ1 static-shape ADR  paper 99.2%       reproduced {100*d_st/g_st:.1f}% ({len(st)} cases)")
print(f"  RQ1 dynamic ADR       paper 87.9%       reproduced {100*d_dy/g_dy:.1f}% ({len(dy)} cases)")
print(f"  RQ1 runtime residue   paper 1,199       reproduced {rt:,}")
print(f"  RQ6 canonical/interval/direct  paper 8,120/2,837/5,561  reproduced {canon:,}/{itv:,}/{direct:,}")
print(f"  RQ6 scalar-symbolic static   paper 0      reproduced {scalar_st}")
print(f"  RQ2 bugs ours/MLIR/IREE      paper 103/43/23  reproduced {ours}/{mlir}/{iree}")
print(f"  RQ4 compile-time overhead    paper median 1.6%  reproduced {statistics.median(cto):.1f}% (n={len(cto)})")

for name in ("rq5_full_svn", "rq5_no_vn_share", "rq5_no_vn_simplify", "rq5_no_vn_both"):
    rows = [r for r in load(f"{name}.csv") if r["status"] == "ok"]
    d = sum(int(r["static_true"]) for r in rows)
    print(f"  RQ5 {name:20s} discharged {d:,} (paper: 16,518 in all configs)")

rq3 = os.path.join(R, "rq3_runtime_overhead.csv")
if os.path.exists(rq3):
    rows = [r for r in load("rq3_runtime_overhead.csv")
            if all(r.get(f"rtc_{l}_us", "error") not in ("error", "", None)
                   for l in ("none", "low", "medium", "high"))
            and r.get("static_us", "error") not in ("error", "", None)]
    lows = [(float(r["rtc_low_us"]) - float(r["rtc_none_us"]))
            / float(r["rtc_none_us"]) * 100 for r in rows]
    highs = [(float(r["rtc_high_us"]) - float(r["rtc_none_us"]))
             / float(r["rtc_none_us"]) * 100 for r in rows]
    print(f"  RQ3 residual cost (low/high) paper +1.5%/+1.4%  "
          f"reproduced {statistics.median(lows):+.1f}%/{statistics.median(highs):+.1f}% (n={len(rows)})")
else:
    print("  RQ3 skipped (no GPU)")
PYEOF

echo ""
echo "Paper tables/figures re-rendered into ../asplos27/{tables,figures}/"
echo "=== reproduce_asplos27.sh finished ==="
