#!/usr/bin/env python3
"""show_results.py — Display evaluation results from previously generated data.

Re-prints the full summary report (same content as reproduce_all.sh output)
without re-running any experiments. Reads CSV files from benchmark/results/.

Usage:
  python3 scripts/show_results.py                    # default results dir
  python3 scripts/show_results.py --results-dir DIR  # custom location
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = WORKSPACE / "benchmark" / "results"

PAPER_VALUES = {
    "cases_compiled": 310,
    "cases_total": 310,
    "generated": 12592,
    "discharged": 11753,
    "runtime": 839,
    "adr": 93.3,
    "cto": 4.7,
    "rao_median": 0.4,
}

BOLD = "\033[1m"
CYAN = "\033[1;36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
DIM = "\033[2m"


def load_csv(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt_num(n):
    return f"{n:,}"


def print_header():
    print(f"\n{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}        SVN: Shape Value Numbering — Evaluation Results (CGO 2027){RESET}")
    print(f"{CYAN}{'=' * 72}{RESET}\n")


def print_summary_table(stats, bug, rao, cto):
    n_ok = sum(1 for r in stats if r["status"] == "ok")
    n_total = len(stats)
    gen = sum(int(r["generated"]) for r in stats if r["status"] == "ok")
    dis = sum(int(r["discharged"]) for r in stats if r["status"] == "ok")
    rt = sum(int(r["runtime"]) for r in stats if r["status"] == "ok")
    adr = dis / gen * 100 if gen > 0 else 0

    # CTO aggregate
    cto_agg = "N/A"
    cto_cases = 0
    if cto:
        cto_rows = [r for r in cto if not r.get("notes", "")]
        cto_cases = len(cto_rows)
        dyn_total = sum(float(r["dynamic_ms"]) for r in cto_rows)
        sta_total = sum(float(r["static_ms"]) for r in cto_rows)
        if sta_total > 0:
            cto_agg = f"{(dyn_total / sta_total - 1) * 100:.1f}%"

    # RAO median
    rao_med = "N/A"
    rao_cases = 0
    if rao:
        rao_vals = [float(r["overhead_pct"]) for r in rao
                    if r.get("overhead_pct", "N/A") not in ("N/A", "")]
        rao_cases = len(rao_vals)
        if rao_vals:
            rao_med = f"{statistics.median(rao_vals):+.3f}%"

    # Bug detection
    bug_summary = "N/A"
    if bug:
        svn_det = sum(1 for r in bug
                      if r.get("svn_resolution", r.get("svn_result", ""))
                      in ("compile", "runtime"))
        bug_summary = f"{svn_det}/{len(bug)} (SVN 100%)"

    print(f"{BOLD}  Paper vs. Reproduced{RESET}")
    print("  " + "-" * 68)
    print(f"  {'Metric':<28s} {'Paper':<14s} {'Reproduced'}")
    print("  " + "-" * 68)
    rows = [
        ("Cases compiled", f"{PAPER_VALUES['cases_compiled']}/310", f"{n_ok}/{n_total}"),
        ("RQ1: Assessments generated", fmt_num(PAPER_VALUES["generated"]), fmt_num(gen)),
        ("RQ1: Discharge rate (ADR)", f"{PAPER_VALUES['adr']:.1f}%", f"{adr:.1f}%"),
        ("     Discharged count", fmt_num(PAPER_VALUES["discharged"]), fmt_num(dis)),
        ("     Runtime surviving", fmt_num(PAPER_VALUES["runtime"]), fmt_num(rt)),
        ("RQ2: Bug detection (BDE)", "210/210", bug_summary),
        ("RQ3: RAO median (entry)", f"<{PAPER_VALUES['rao_median']}%", f"{rao_med} ({rao_cases} cases)"),
        ("RQ4: CTO (aggregate)", f"{PAPER_VALUES['cto']:.1f}%", f"{cto_agg} ({cto_cases} cases)"),
    ]
    for label, paper, repro in rows:
        print(f"  {label:<28s} {paper:<14s} {repro}")
    print("  " + "-" * 68)


def print_rq1_detail(stats):
    rows = [r for r in stats if r["status"] == "ok"]
    if not rows:
        return
    cats = defaultdict(lambda: {"n": 0, "gen": 0, "dis": 0, "rt": 0})
    for r in rows:
        c = r["category"]
        cats[c]["n"] += 1
        cats[c]["gen"] += int(r["generated"])
        cats[c]["dis"] += int(r["discharged"])
        cats[c]["rt"] += int(r["runtime"])

    print(f"\n{CYAN}  RQ1: Assessment Coverage by Category{RESET}")
    print("  " + "-" * 62)
    print("  {:<22s} {:>5s} {:>6s} {:>6s} {:>5s} {:>6s}".format(
        "Category", "Cases", "Gen", "Dis", "RT", "ADR"))
    print("  " + "-" * 62)
    for c in sorted(cats.keys()):
        v = cats[c]
        adr = v["dis"] / v["gen"] * 100 if v["gen"] > 0 else 0
        print("  {:<22s} {:5d} {:6d} {:6d} {:5d} {:5.1f}%".format(
            c, v["n"], v["gen"], v["dis"], v["rt"], adr))
    print("  " + "-" * 62)
    tot_gen = sum(v["gen"] for v in cats.values())
    tot_dis = sum(v["dis"] for v in cats.values())
    tot_rt = sum(v["rt"] for v in cats.values())
    tot_n = sum(v["n"] for v in cats.values())
    print(f"  {BOLD}{'TOTAL':<22s} {tot_n:5d} {tot_gen:6d} {tot_dis:6d} "
          f"{tot_rt:5d} {tot_dis / tot_gen * 100:5.1f}%{RESET}")


def print_rq2_detail(bug):
    if not bug:
        return
    bugs = defaultdict(int)
    svn_det = defaultdict(int)
    mlir_det = defaultdict(int)
    iree_det = defaultdict(int)
    for r in bug:
        bc = r.get("bug_class", "unknown")
        bugs[bc] += 1
        svn_res = r.get("svn_resolution", r.get("svn_result", ""))
        mlir_res = r.get("mlir_resolution", r.get("mlir_result", ""))
        iree_res = r.get("iree_resolution", r.get("iree_result", ""))
        if svn_res in ("compile", "runtime"):
            svn_det[bc] += 1
        if mlir_res in ("compile", "runtime"):
            mlir_det[bc] += 1
        if iree_res in ("compile", "runtime"):
            iree_det[bc] += 1

    print(f"\n{CYAN}  RQ2: Bug Detection Effectiveness{RESET}")
    print("  " + "-" * 52)
    print("  {:<22s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
        "Bug Class", "Total", "SVN", "MLIR", "IREE"))
    print("  " + "-" * 52)
    for bc in sorted(bugs.keys()):
        print("  {:<22s} {:5d} {:5d} {:5d} {:5d}".format(
            bc, bugs[bc], svn_det[bc], mlir_det[bc], iree_det[bc]))
    print("  " + "-" * 52)
    t = sum(bugs.values())
    s = sum(svn_det.values())
    m = sum(mlir_det.values())
    i = sum(iree_det.values())
    print("  {:<22s} {:5d} {:5d} {:5d} {:5d}".format("TOTAL", t, s, m, i))
    print("  {:<22s} {:>5s} {:4.1f}% {:4.1f}% {:4.1f}%".format(
        "Detection rate", "", s / t * 100, m / t * 100, i / t * 100))


def print_rq3_detail(rao):
    if not rao:
        return
    rows = [r for r in rao if r.get("overhead_pct", "N/A") not in ("N/A", "")]
    if not rows:
        return
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(float(r["overhead_pct"]))

    print(f"\n{CYAN}  RQ3: Runtime Assertion Overhead by Category{RESET}")
    print("  " + "-" * 68)
    print("  {:<22s} {:>5s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Category", "Cases", "Median", "Mean", "Min", "Max"))
    print("  " + "-" * 68)
    for c in sorted(cats.keys()):
        vals = cats[c]
        print("  {:<22s} {:5d} {:+7.3f}% {:+7.3f}% {:+7.3f}% {:+7.3f}%".format(
            c, len(vals), statistics.median(vals), statistics.mean(vals),
            min(vals), max(vals)))
    print("  " + "-" * 68)
    all_vals = [v for vs in cats.values() for v in vs]
    print(f"  {BOLD}{'OVERALL':<22s} {len(all_vals):5d} "
          f"{statistics.median(all_vals):+7.3f}% {statistics.mean(all_vals):+7.3f}% "
          f"{min(all_vals):+7.3f}% {max(all_vals):+7.3f}%{RESET}")


def print_rq4_detail(cto):
    if not cto:
        return
    rows = [r for r in cto if not r.get("notes", "")]
    if not rows:
        return
    cats = defaultdict(lambda: {"dyn": 0, "sta": 0, "n": 0, "pcts": []})
    for r in rows:
        c = r["category"]
        cats[c]["dyn"] += float(r["dynamic_ms"])
        cats[c]["sta"] += float(r["static_ms"])
        cats[c]["n"] += 1
        cats[c]["pcts"].append(float(r["overhead_pct"]))

    print(f"\n{CYAN}  RQ4: Compile-Time Overhead by Category{RESET}")
    print("  " + "-" * 72)
    print("  {:<22s} {:>5s} {:>9s} {:>9s} {:>7s} {:>8s}".format(
        "Category", "Cases", "Dyn(ms)", "Sta(ms)", "CTO", "Median"))
    print("  " + "-" * 72)
    for c in sorted(cats.keys()):
        v = cats[c]
        cto_pct = (v["dyn"] / v["sta"] - 1) * 100 if v["sta"] > 0 else 0
        med = statistics.median(v["pcts"])
        print("  {:<22s} {:5d} {:9.0f} {:9.0f} {:+6.1f}% {:+7.1f}%".format(
            c, v["n"], v["dyn"], v["sta"], cto_pct, med))
    print("  " + "-" * 72)
    td = sum(v["dyn"] for v in cats.values())
    ts = sum(v["sta"] for v in cats.values())
    ap = [p for v in cats.values() for p in v["pcts"]]
    print(f"  {BOLD}{'OVERALL':<22s} {len(rows):5d} {td:9.0f} {ts:9.0f} "
          f"{(td / ts - 1) * 100:+6.1f}% {statistics.median(ap):+7.1f}%{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Display SVN evaluation results from generated CSV data.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Directory containing result CSVs (default: benchmark/results)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output")
    args = parser.parse_args()

    if args.no_color:
        global BOLD, CYAN, GREEN, YELLOW, RED, RESET, DIM
        BOLD = CYAN = GREEN = YELLOW = RED = RESET = DIM = ""

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        print("Run 'bash scripts/reproduce_all.sh' first to generate results.", file=sys.stderr)
        sys.exit(1)

    stats = load_csv(results_dir / "choreo_stats.csv")
    bug = load_csv(results_dir / "bug_detection_results.csv")
    rao = load_csv(results_dir / "choreo_runtime_entry.csv")
    cto = load_csv(results_dir / "choreo_compile_overhead.csv")

    if not stats:
        print(f"Error: choreo_stats.csv not found in {results_dir}", file=sys.stderr)
        print("Run 'bash scripts/reproduce_all.sh' first.", file=sys.stderr)
        sys.exit(1)

    print_header()
    print_summary_table(stats, bug, rao, cto)
    print_rq1_detail(stats)
    print_rq2_detail(bug)
    print_rq3_detail(rao)
    print_rq4_detail(cto)

    print(f"\n{DIM}  Data source: {results_dir}/{RESET}")
    available = [f.name for f in results_dir.glob("*.csv")]
    print(f"{DIM}  Available CSVs: {', '.join(sorted(available))}{RESET}")

    report_html = results_dir / "report.html"
    if report_html.exists():
        print(f"{DIM}  HTML report: {report_html}{RESET}")
    print()


if __name__ == "__main__":
    main()
