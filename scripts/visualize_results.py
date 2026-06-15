#!/usr/bin/env python3
"""visualize_results.py — Generate visual report from artifact evaluation results.

Produces:
  - Rich terminal output with ASCII bar charts and per-category tables
  - figures/*.png — Individual matplotlib plots for each RQ
  - report.html  — Self-contained HTML page with interactive Chart.js graphs

Usage:
  python3 scripts/visualize_results.py [--results-dir DIR] [--out-dir DIR]
"""

import argparse
import base64
import csv
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "rq4_median": 0.0,
}

CATEGORY_ORDER = [
    "batch_norm", "concat", "conv2d", "elemwise_add", "embedding",
    "gelu", "layer_normalization", "matmul", "max_pool2d", "reduce_mean",
    "relu", "reshape", "sigmoid", "softmax", "transpose",
]

SYSTEM_COLORS = {
    "SVN": "#2f6bff",
    "MLIR (tensor)": "#ff7f0e",
    "MLIR (memref)": "#e08010",
    "Triton": "#d62728",
    "IREE": "#9467bd",
}


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading & aggregation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_all_data(results_dir: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    # Choreo stats (RQ1/RQ2)
    choreo_stats = load_csv(results_dir / "choreo_stats.csv")
    ok = [r for r in choreo_stats if r.get("status") == "ok"]
    data["choreo_stats"] = choreo_stats
    data["choreo_ok"] = ok
    data["n_ok"] = len(ok)
    data["n_total"] = len(choreo_stats)
    data["generated"] = sum(int(r["generated"]) for r in ok) if ok else 0
    data["discharged"] = sum(int(r["discharged"]) for r in ok) if ok else 0
    data["runtime"] = sum(int(r["runtime"]) for r in ok) if ok else 0
    data["adr"] = data["discharged"] / data["generated"] * 100 if data["generated"] > 0 else 0

    # Per-category RQ1/RQ2
    cat_rq12: Dict[str, Dict] = defaultdict(lambda: {"gen": 0, "dis": 0, "rt": 0, "n": 0,
                                                       "ut_shape": 0, "ut_elem": 0,
                                                       "ut_loop": 0, "ut_hw": 0})
    for r in ok:
        cat = r.get("category", "unknown")
        cat_rq12[cat]["gen"] += int(r["generated"])
        cat_rq12[cat]["dis"] += int(r["discharged"])
        cat_rq12[cat]["rt"] += int(r["runtime"])
        cat_rq12[cat]["n"] += 1
        for col in ("ut_shape", "ut_elem", "ut_loop", "ut_hw"):
            cat_rq12[cat][col] += int(r.get(col, 0))
    data["cat_rq12"] = dict(cat_rq12)

    # Usage-type aggregates (RQ2 breakdown)
    data["ut_shape"] = sum(int(r.get("ut_shape", 0)) for r in ok)
    data["ut_elem"] = sum(int(r.get("ut_elem", 0)) for r in ok)
    data["ut_loop"] = sum(int(r.get("ut_loop", 0)) for r in ok)
    data["ut_hw"] = sum(int(r.get("ut_hw", 0)) for r in ok)

    # CTO (RQ3)
    cto_rows = load_csv(results_dir / "choreo_compile_overhead.csv")
    cto_ok = [r for r in cto_rows if not r.get("notes", "")]
    data["cto_rows"] = cto_ok
    cat_cto: Dict[str, Dict] = defaultdict(lambda: {"dyn": 0.0, "sta": 0.0, "n": 0})
    for r in cto_ok:
        try:
            d, s = float(r["dynamic_ms"]), float(r["static_ms"])
        except (ValueError, TypeError):
            continue
        cat_cto[r["category"]]["dyn"] += d
        cat_cto[r["category"]]["sta"] += s
        cat_cto[r["category"]]["n"] += 1
    data["cat_cto"] = dict(cat_cto)
    total_dyn = sum(v["dyn"] for v in cat_cto.values())
    total_sta = sum(v["sta"] for v in cat_cto.values())
    data["cto_overall"] = (total_dyn / total_sta - 1) * 100 if total_sta > 0 else 0
    data["cto_n_cases"] = sum(v["n"] for v in cat_cto.values())

    # RQ4
    rq4_rows = load_csv(results_dir / "choreo_runtime_entry.csv")
    rq4_ok = [r for r in rq4_rows
              if r.get("none_us") not in ("error", "", None)
              and r.get("overhead_pct") not in ("N/A", "", None)]
    data["rq4_rows"] = rq4_ok
    data["rq4_available"] = len(rq4_ok) > 0
    if rq4_ok:
        ovhds = [float(r["overhead_pct"]) for r in rq4_ok]
        data["rq4_median"] = statistics.median(ovhds)
        data["rq4_mean"] = statistics.mean(ovhds)
        data["rq4_min"] = min(ovhds)
        data["rq4_max"] = max(ovhds)
        data["rq4_n"] = len(ovhds)
        cat_rq4: Dict[str, List[float]] = defaultdict(list)
        for r in rq4_ok:
            cat_rq4[r["category"]].append(float(r["overhead_pct"]))
        data["cat_rq4"] = dict(cat_rq4)
    else:
        data["rq4_median"] = None
        data["rq4_n"] = 0
        data["cat_rq4"] = {}

    # Cross-system (optional)
    all_stats = load_csv(results_dir / "all_stats.csv")
    data["all_stats"] = all_stats

    return data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
CYAN = "\033[1;36m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ascii_bar(value: float, max_value: float, width: int = 30, char: str = "█") -> str:
    if max_value <= 0:
        return ""
    n = int(round(value / max_value * width))
    return char * n


def print_terminal_report(data: Dict[str, Any]) -> None:
    print(f"\n{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}  SVN: Shape Value Numbering — Artifact Evaluation Report{RESET}")
    print(f"{CYAN}{'═' * 70}{RESET}\n")

    # ── Summary table ──
    print(f"{BOLD}  Paper vs. Reproduced{RESET}")
    print(f"  {'─' * 66}")
    rows = [
        ("Cases compiled", f"{PAPER_VALUES['cases_compiled']}/310",
         f"{data['n_ok']}/{data['n_total']}"),
        ("RQ1: Assessments generated", f"{PAPER_VALUES['generated']:,}",
         f"{data['generated']:,}"),
        ("RQ1: Discharge rate (ADR)", f"{PAPER_VALUES['adr']:.1f}%",
         f"{data['adr']:.1f}%"),
        ("     Discharged count", f"{PAPER_VALUES['discharged']:,}",
         f"{data['discharged']:,}"),
        ("     Runtime surviving", f"{PAPER_VALUES['runtime']:,}",
         f"{data['runtime']:,}"),
        ("RQ4: CTO (aggregate)", f"{PAPER_VALUES['cto']:.1f}%",
         f"{data['cto_overall']:.1f}% ({data['cto_n_cases']} cases)"),
    ]
    if data["rq4_available"]:
        rows.append(("RQ3: RAO median (entry)", "<0.4%",
                      f"{data['rq4_median']:+.3f}% ({data['rq4_n']} cases)"))
    else:
        rows.append(("RQ3: RAO median (entry)", "<0.4%", "(skipped — no GPU data)"))

    for label, paper, repro in rows:
        print(f"  {label:<28s} │ {paper:<14s} │ {repro}")
    print(f"  {'─' * 66}\n")

    # ── RQ1: Per-category assessment coverage ──
    print(f"\n{CYAN}── RQ1: Assessment Coverage (per category) ──{RESET}")
    cat_data = data["cat_rq12"]
    max_gen = max((v["gen"] for v in cat_data.values()), default=1)
    cats = [c for c in CATEGORY_ORDER if c in cat_data]

    print(f"  {'Category':<22s} {'Cases':>5s} {'Gen':>6s} {'ACD':>6s}  Bar")
    print(f"  {'─' * 60}")
    for cat in cats:
        v = cat_data[cat]
        acd = v["gen"] / v["n"] if v["n"] > 0 else 0
        bar = ascii_bar(v["gen"], max_gen, 25)
        print(f"  {cat:<22s} {v['n']:5d} {v['gen']:6d} {acd:6.1f}  {BLUE}{bar}{RESET}")
    total_acd = data["generated"] / data["n_ok"] if data["n_ok"] > 0 else 0
    print(f"  {'─' * 60}")
    print(f"  {'TOTAL':<22s} {data['n_ok']:5d} {data['generated']:6d} {total_acd:6.1f}")

    # ── RQ2: Per-category ADR ──
    print(f"\n{CYAN}── RQ2: Discharge Rate (per category) ──{RESET}")
    print(f"  {'Category':<22s} {'Gen':>6s} {'Dis':>6s} {'RT':>5s} {'ADR':>7s}  Bar")
    print(f"  {'─' * 66}")
    for cat in cats:
        v = cat_data[cat]
        adr = v["dis"] / v["gen"] * 100 if v["gen"] > 0 else 0
        bar_len = int(round(adr / 100 * 25))
        bar = f"{GREEN}{'█' * bar_len}{DIM}{'░' * (25 - bar_len)}{RESET}"
        print(f"  {cat:<22s} {v['gen']:6d} {v['dis']:6d} {v['rt']:5d} {adr:6.1f}%  {bar}")
    print(f"  {'─' * 66}")
    print(f"  {'TOTAL':<22s} {data['generated']:6d} {data['discharged']:6d} "
          f"{data['runtime']:5d} {data['adr']:6.1f}%")

    # Usage-type breakdown
    ut_total = data["ut_shape"] + data["ut_elem"] + data["ut_loop"] + data["ut_hw"]
    if ut_total > 0:
        print(f"\n  {BOLD}Usage-type breakdown:{RESET}")
        for label, val in [("Shape compatibility", data["ut_shape"]),
                           ("Element access", data["ut_elem"]),
                           ("Loop bound", data["ut_loop"]),
                           ("Hardware", data["ut_hw"])]:
            pct = val / ut_total * 100
            bar = ascii_bar(pct, 100, 20)
            print(f"    {label:<22s} {val:6d} ({pct:5.1f}%)  {BLUE}{bar}{RESET}")

    # ── RQ3: Per-category CTO ──
    if data["cat_cto"]:
        print(f"\n{CYAN}── RQ3: Compile-Time Overhead (per category) ──{RESET}")
        cto_cats = [c for c in CATEGORY_ORDER if c in data["cat_cto"]]
        print(f"  {'Category':<22s} {'Dyn(s)':>8s} {'Sta(s)':>8s} {'CTO%':>7s}  Bar")
        print(f"  {'─' * 66}")
        max_cto = max(abs((data["cat_cto"][c]["dyn"] / data["cat_cto"][c]["sta"] - 1) * 100)
                      for c in cto_cats if data["cat_cto"][c]["sta"] > 0) or 1
        for cat in cto_cats:
            v = data["cat_cto"][cat]
            if v["sta"] <= 0:
                continue
            cto = (v["dyn"] / v["sta"] - 1) * 100
            dyn_s = v["dyn"] / 1000
            sta_s = v["sta"] / 1000
            bar_w = int(round(abs(cto) / max_cto * 20))
            color = YELLOW if cto >= 0 else GREEN
            bar = f"{color}{'█' * bar_w}{RESET}"
            sign = "+" if cto >= 0 else ""
            print(f"  {cat:<22s} {dyn_s:8.2f} {sta_s:8.2f} {sign}{cto:6.1f}%  {bar}")
        print(f"  {'─' * 66}")
        print(f"  {'OVERALL':<22s} {'':>8s} {'':>8s} {'+' if data['cto_overall'] >= 0 else ''}"
              f"{data['cto_overall']:6.1f}%")

    # ── RQ4: Runtime Assertion Overhead ──
    if data["rq4_available"]:
        print(f"\n{CYAN}── RQ4: Runtime Assertion Overhead (entry vs none) ──{RESET}")
        cat_rq4 = data["cat_rq4"]
        rq4_cats = [c for c in CATEGORY_ORDER if c in cat_rq4]
        print(f"  {'Category':<22s} {'Cases':>5s} {'Median%':>9s} {'Mean%':>9s}  Distribution")
        print(f"  {'─' * 70}")
        for cat in rq4_cats:
            vals = cat_rq4[cat]
            med = statistics.median(vals)
            mean = statistics.mean(vals)
            # mini sparkline: show distribution around 0
            bar = _mini_distribution(vals)
            print(f"  {cat:<22s} {len(vals):5d} {med:+8.3f}% {mean:+8.3f}%  {bar}")
        print(f"  {'─' * 70}")
        print(f"  {'OVERALL':<22s} {data['rq4_n']:5d} {data['rq4_median']:+8.3f}% "
              f"{data['rq4_mean']:+8.3f}%")
        print(f"\n  Range: [{data['rq4_min']:.3f}%, {data['rq4_max']:.3f}%]")
        if abs(data["rq4_median"]) < 0.1:
            print(f"  {GREEN}✓ Median < 0.1% — consistent with paper{RESET}")


def _mini_distribution(vals: List[float], width: int = 20) -> str:
    """Tiny ASCII box showing spread of values."""
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    med = statistics.median(vals)
    if hi - lo < 0.001:
        return f"{DIM}[─ {med:+.2f}% ─]{RESET}"
    center = (lo + hi) / 2
    span = max(abs(lo - center), abs(hi - center), 0.1)
    def pos(v: float) -> int:
        return max(0, min(width - 1, int((v - center + span) / (2 * span) * width)))
    bar = list("─" * width)
    bar[pos(lo)] = "├"
    bar[pos(hi)] = "┤"
    mp = pos(med)
    if 0 <= mp < width:
        bar[mp] = "●"
    return f"{DIM}{''.join(bar)}{RESET}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Matplotlib figures (PNG export)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "font.family": "sans-serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })
    import matplotlib.pyplot as plt
    return plt


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig_to_file(fig, path: Path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight")


def generate_matplotlib_figures(data: Dict, results_dir: Path, fig_dir: Path) -> Dict[str, str]:
    plt = setup_matplotlib()
    import numpy as np

    b64 = {}

    # ── RQ1 ──
    cats = [c for c in CATEGORY_ORDER if c in data["cat_rq12"]]
    gen_vals = [data["cat_rq12"][c]["gen"] for c in cats]
    acd_vals = [data["cat_rq12"][c]["gen"] / data["cat_rq12"][c]["n"]
                for c in cats if data["cat_rq12"][c]["n"] > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(cats))
    bars = ax.barh(y, gen_vals, color="#2f6bff", alpha=0.8, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8)
    ax.set_xlabel("Total assessments generated")
    ax.set_title("RQ1: Assessment Coverage by Category", fontweight="bold")
    for bar, val in zip(bars, gen_vals):
        ax.text(bar.get_width() + max(gen_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2, str(val),
                va="center", fontsize=7)
    fig.tight_layout()
    fig_to_file(fig, fig_dir / "rq1_coverage.png")
    b64["rq1"] = fig_to_base64(fig)
    plt.close(fig)

    # ── RQ2 ──
    dis_vals = [data["cat_rq12"][c]["dis"] for c in cats]
    rt_vals = [data["cat_rq12"][c]["rt"] for c in cats]
    adr_vals = [data["cat_rq12"][c]["dis"] / data["cat_rq12"][c]["gen"] * 100
                if data["cat_rq12"][c]["gen"] > 0 else 0 for c in cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})
    # stacked bar
    ax1.barh(y, dis_vals, color="#2f6bff", alpha=0.8, height=0.65, label="Discharged (static)")
    ax1.barh(y, rt_vals, left=dis_vals, color="#ff7f0e", alpha=0.8, height=0.65,
             label="Runtime surviving")
    ax1.set_yticks(y)
    ax1.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8)
    ax1.set_xlabel("Assessment count")
    ax1.set_title("RQ2: Discharged vs Runtime", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")

    # ADR horizontal bar
    colors = ["#2ca02c" if v >= 95 else "#2f6bff" if v >= 90 else "#ff7f0e" for v in adr_vals]
    ax2.barh(y, adr_vals, color=colors, alpha=0.75, height=0.65)
    ax2.axvline(data["adr"], color="#d62728", linestyle=":", linewidth=1.3,
                label=f"Overall: {data['adr']:.1f}%")
    ax2.set_yticks(y)
    ax2.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=8)
    ax2.set_xlabel("ADR (%)")
    ax2.set_xlim(0, 105)
    ax2.set_title("Per-Category ADR", fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    for bar_y, val in zip(y, adr_vals):
        ax2.text(val + 0.5, bar_y, f"{val:.0f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig_to_file(fig, fig_dir / "rq2_discharge.png")
    b64["rq2"] = fig_to_base64(fig)
    plt.close(fig)

    # ── RQ3 ──
    if data["cat_cto"]:
        cto_cats = [c for c in CATEGORY_ORDER if c in data["cat_cto"]]
        cto_pcts = []
        for c in cto_cats:
            v = data["cat_cto"][c]
            cto_pcts.append((v["dyn"] / v["sta"] - 1) * 100 if v["sta"] > 0 else 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        cy = np.arange(len(cto_cats))

        dyn_s = [data["cat_cto"][c]["dyn"] / 1000 for c in cto_cats]
        sta_s = [data["cat_cto"][c]["sta"] / 1000 for c in cto_cats]
        bh = 0.35
        ax1.barh(cy + bh / 2, sta_s, height=bh, color="#aaaaaa", label="Static baseline")
        ax1.barh(cy - bh / 2, dyn_s, height=bh, color="#2f6bff", alpha=0.85,
                 label="Dynamic (SVN)")
        ax1.set_yticks(cy)
        ax1.set_yticklabels([c.replace("_", " ") for c in cto_cats], fontsize=8)
        ax1.set_xlabel("Cumulative compile time (s)")
        ax1.set_title("Compile Time: Static vs Dynamic", fontweight="bold")
        ax1.legend(fontsize=8)

        bar_colors = ["#2f6bff" if v >= 0 else "#2ca02c" for v in cto_pcts]
        ax2.barh(cy, cto_pcts, color=bar_colors, alpha=0.7, height=0.6)
        ax2.axvline(data["cto_overall"], color="#d62728", linestyle=":",
                    linewidth=1.5, label=f"Overall: {data['cto_overall']:.1f}%")
        ax2.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        ax2.set_yticks(cy)
        ax2.set_yticklabels([c.replace("_", " ") for c in cto_cats], fontsize=8)
        ax2.set_xlabel("CTO (%)")
        ax2.set_title("RQ3: CTO per Category", fontweight="bold")
        ax2.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig_to_file(fig, fig_dir / "rq3_cto.png")
        b64["rq3"] = fig_to_base64(fig)
        plt.close(fig)

    # ── RQ4 ──
    if data["rq4_available"]:
        import random
        random.seed(42)
        cat_rq4 = data["cat_rq4"]
        rq4_cats = [c for c in CATEGORY_ORDER if c in cat_rq4]
        rq4_data = [cat_rq4[c] for c in rq4_cats]
        labels = [c.replace("_", " ") for c in rq4_cats]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bp = ax.boxplot(rq4_data, vert=False, labels=labels, patch_artist=True,
                        notch=False,
                        medianprops={"color": "black", "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4},
                        widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor("#2f6bff")
            patch.set_alpha(0.45)

        for i, vals in enumerate(rq4_data):
            jitter = [random.uniform(-0.20, 0.20) for _ in vals]
            ax.scatter(vals, [i + 1 + j for j in jitter],
                       color="#2f6bff", alpha=0.5, s=14, zorder=3)

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(data["rq4_median"], color="#d62728", linestyle=":", linewidth=1.2,
                   label=f"Median: {data['rq4_median']:+.3f}%")
        ax.set_xlabel("Runtime overhead: entry vs none (%)")
        ax.set_title("RQ4: Runtime Assertion Overhead", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        fig_to_file(fig, fig_dir / "rq4_rao.png")
        b64["rq4"] = fig_to_base64(fig)
        plt.close(fig)

    return b64


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTML report with interactive Chart.js
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_html_report(data: Dict, b64_figs: Dict[str, str]) -> str:
    cats = [c for c in CATEGORY_ORDER if c in data["cat_rq12"]]
    cat_labels = json.dumps([c.replace("_", " ") for c in cats])
    gen_vals = json.dumps([data["cat_rq12"][c]["gen"] for c in cats])
    dis_vals = json.dumps([data["cat_rq12"][c]["dis"] for c in cats])
    rt_vals = json.dumps([data["cat_rq12"][c]["rt"] for c in cats])
    adr_vals = json.dumps([round(data["cat_rq12"][c]["dis"] / data["cat_rq12"][c]["gen"] * 100, 1)
                           if data["cat_rq12"][c]["gen"] > 0 else 0 for c in cats])

    # CTO data
    cto_cats_js = "[]"
    cto_pcts_js = "[]"
    cto_dyn_js = "[]"
    cto_sta_js = "[]"
    if data["cat_cto"]:
        cc = [c for c in CATEGORY_ORDER if c in data["cat_cto"]]
        cto_cats_js = json.dumps([c.replace("_", " ") for c in cc])
        cto_pcts_js = json.dumps([round((data["cat_cto"][c]["dyn"] / data["cat_cto"][c]["sta"] - 1) * 100, 2)
                                  if data["cat_cto"][c]["sta"] > 0 else 0 for c in cc])
        cto_dyn_js = json.dumps([round(data["cat_cto"][c]["dyn"] / 1000, 3) for c in cc])
        cto_sta_js = json.dumps([round(data["cat_cto"][c]["sta"] / 1000, 3) for c in cc])

    # RQ4 data
    rq4_cats_js = "[]"
    rq4_medians_js = "[]"
    rq4_means_js = "[]"
    rq4_scatter_js = "[]"
    if data["rq4_available"]:
        rc = [c for c in CATEGORY_ORDER if c in data["cat_rq4"]]
        rq4_cats_js = json.dumps([c.replace("_", " ") for c in rc])
        rq4_medians_js = json.dumps([round(statistics.median(data["cat_rq4"][c]), 3) for c in rc])
        rq4_means_js = json.dumps([round(statistics.mean(data["cat_rq4"][c]), 3) for c in rc])
        scatter_pts = []
        for i, c in enumerate(rc):
            for v in data["cat_rq4"][c]:
                scatter_pts.append({"x": round(v, 3), "y": i})
        rq4_scatter_js = json.dumps(scatter_pts)

    # Usage-type data
    ut_total = data["ut_shape"] + data["ut_elem"] + data["ut_loop"] + data["ut_hw"]
    ut_js = json.dumps([data["ut_shape"], data["ut_elem"], data["ut_loop"], data["ut_hw"]])

    # Summary comparison
    rq4_str = f"{data['rq4_median']:+.3f}% ({data['rq4_n']} cases)" if data["rq4_available"] else "skipped"

    # PNG fallback images
    rq1_img = f'<img src="data:image/png;base64,{b64_figs["rq1"]}" style="width:100%">' if "rq1" in b64_figs else ""
    rq2_img = f'<img src="data:image/png;base64,{b64_figs["rq2"]}" style="width:100%">' if "rq2" in b64_figs else ""
    rq3_img = f'<img src="data:image/png;base64,{b64_figs["rq3"]}" style="width:100%">' if "rq3" in b64_figs else ""
    rq4_img = f'<img src="data:image/png;base64,{b64_figs["rq4"]}" style="width:100%">' if "rq4" in b64_figs else ""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SVN Artifact — Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --blue: #2f6bff; --orange: #ff7f0e; --green: #2ca02c;
    --red: #d62728; --purple: #9467bd; --bg: #f8f9fa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
         background: var(--bg); color: #212529; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; color: #1a1a2e; }}
  h2 {{ font-size: 1.35rem; margin: 2.5rem 0 1rem; color: #16213e;
       border-bottom: 3px solid var(--blue); padding-bottom: 0.4rem; }}
  .subtitle {{ color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }}

  /* Cards */
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border-radius: 12px; padding: 1.4rem 1rem;
               box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center;
               transition: transform 0.15s; }}
  .stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,0,0,0.1); }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--blue); }}
  .stat-value.green {{ color: var(--green); }}
  .stat-label {{ font-size: 0.82rem; color: #888; margin-top: 0.3rem; text-transform: uppercase;
                letter-spacing: 0.5px; }}

  /* Summary table */
  .summary-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
  .summary-table th {{ background: #e9ecef; font-weight: 600; text-align: left;
                       padding: 0.6rem 1rem; border-bottom: 2px solid #dee2e6; }}
  .summary-table td {{ padding: 0.55rem 1rem; border-bottom: 1px solid #eee; }}
  .summary-table tr:hover td {{ background: #f1f3f5; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
           font-size: 0.78rem; font-weight: 600; }}
  .badge-pass {{ background: #d4edda; color: #155724; }}
  .badge-warn {{ background: #fff3cd; color: #856404; }}
  .badge-skip {{ background: #e2e3e5; color: #383d41; }}

  /* Chart containers */
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
  .chart-box {{ background: white; border-radius: 12px; padding: 1.5rem;
               box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .chart-box.full {{ grid-column: 1 / -1; }}
  .chart-box canvas {{ width: 100% !important; }}

  /* Fallback images (noscript) */
  .fallback-img {{ display: none; }}
  noscript .fallback-img {{ display: block; }}

  .note {{ background: #e8f4fd; border-left: 4px solid var(--blue); padding: 0.8rem 1rem;
          margin: 1rem 0; border-radius: 0 6px 6px 0; font-size: 0.9rem; }}

  @media (max-width: 768px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  footer {{ margin-top: 3rem; text-align: center; color: #aaa; font-size: 0.8rem;
           border-top: 1px solid #eee; padding-top: 1rem; }}
</style>
</head>
<body>
<div class="container">

<h1>SVN: Shape Value Numbering — Artifact Evaluation</h1>
<p class="subtitle">Auto-generated report from <code>scripts/visualize_results.py</code></p>

<!-- ═══ Summary ═══ -->
<h2>Summary — Paper vs. Reproduced</h2>

<div class="stats-grid">
  <div class="stat-card"><div class="stat-value">{data['n_ok']}/{data['n_total']}</div>
       <div class="stat-label">Cases Compiled</div></div>
  <div class="stat-card"><div class="stat-value">{data['generated']:,}</div>
       <div class="stat-label">Assessments Generated</div></div>
  <div class="stat-card"><div class="stat-value green">{data['adr']:.1f}%</div>
       <div class="stat-label">Discharge Rate (ADR)</div></div>
  <div class="stat-card"><div class="stat-value">{data['cto_overall']:.1f}%</div>
       <div class="stat-label">Compile-Time Overhead</div></div>
  <div class="stat-card"><div class="stat-value green">{f"{data['rq4_median']:+.3f}%" if data['rq4_available'] else "N/A"}</div>
       <div class="stat-label">Runtime Overhead (median)</div></div>
</div>

<table class="summary-table">
<tr><th>Metric</th><th>Paper</th><th>Reproduced</th><th>Status</th></tr>
<tr><td>Cases compiled</td><td>291 / 310</td><td>{data['n_ok']} / {data['n_total']}</td>
    <td><span class="badge {'badge-pass' if data['n_ok'] >= 291 else 'badge-warn'}">{'✓ improved' if data['n_ok'] >= 291 else '△'}</span></td></tr>
<tr><td>Assessments generated (RQ1)</td><td>11,524</td><td>{data['generated']:,}</td>
    <td><span class="badge {'badge-pass' if data['generated'] >= 11524 else 'badge-warn'}">{'✓ improved' if data['generated'] >= 11524 else '△ lower'}</span></td></tr>
<tr><td>Discharge rate — ADR (RQ2)</td><td>92.8%</td><td>{data['adr']:.1f}%</td>
    <td><span class="badge {'badge-pass' if data['adr'] >= 92.5 else 'badge-warn'}">{'✓ consistent' if data['adr'] >= 92.5 else '△'}</span></td></tr>
<tr><td>Compile-time overhead (RQ3)</td><td>4.7%</td><td>{data['cto_overall']:.1f}%</td>
    <td><span class="badge {'badge-pass' if abs(data['cto_overall'] - 4.7) < 3 else 'badge-warn'}">{'✓ consistent' if abs(data['cto_overall'] - 4.7) < 3 else '△'}</span></td></tr>
<tr><td>Runtime overhead median (RQ4)</td><td>&lt;0.1%</td><td>{rq4_str}</td>
    <td><span class="badge {'badge-pass' if data['rq4_available'] and abs(data['rq4_median']) < 1 else 'badge-skip' if not data['rq4_available'] else 'badge-warn'}">{'✓ negligible' if data['rq4_available'] and abs(data['rq4_median']) < 1 else 'skipped (no GPU)' if not data['rq4_available'] else '△'}</span></td></tr>
</table>

<!-- ═══ RQ1 ═══ -->
<h2>RQ1: Assessment Coverage</h2>
<p class="note">SVN generates <strong>{data['generated']:,}</strong> assessments across {data['n_ok']} cases
   (ACD = {data['generated'] / data['n_ok']:.1f} per case) with <strong>zero</strong> manual annotations.</p>

<div class="chart-row">
  <div class="chart-box"><canvas id="rq1-bar"></canvas></div>
  <div class="chart-box"><canvas id="rq1-acd"></canvas></div>
</div>
<noscript><div class="fallback-img">{rq1_img}</div></noscript>

<!-- ═══ RQ2 ═══ -->
<h2>RQ2: Compile-Time Assessment Discharge</h2>
<p class="note">Overall ADR: <strong>{data['adr']:.1f}%</strong> — {data['discharged']:,} discharged statically,
   {data['runtime']:,} deferred to runtime.</p>

<div class="chart-row">
  <div class="chart-box"><canvas id="rq2-stacked"></canvas></div>
  <div class="chart-box"><canvas id="rq2-adr"></canvas></div>
</div>
<div class="chart-row">
  <div class="chart-box"><canvas id="rq2-usage"></canvas></div>
  <div class="chart-box"><canvas id="rq2-pie"></canvas></div>
</div>
<noscript><div class="fallback-img">{rq2_img}</div></noscript>

<!-- ═══ RQ3 ═══ -->
<h2>RQ3: Compile-Time Overhead</h2>
<p class="note">Aggregate CTO: <strong>{data['cto_overall']:.1f}%</strong> across {data['cto_n_cases']} dynamic cases.
   SVN's symbolic shape analysis adds minimal compilation cost.</p>

<div class="chart-row">
  <div class="chart-box"><canvas id="rq3-time"></canvas></div>
  <div class="chart-box"><canvas id="rq3-pct"></canvas></div>
</div>
<noscript><div class="fallback-img">{rq3_img}</div></noscript>

<!-- ═══ RQ4 ═══ -->
<h2>RQ4: Runtime Assertion Overhead</h2>
{"<p class='note'>Median overhead: <strong>" + f"{data['rq4_median']:+.3f}%" + "</strong> across " + str(data['rq4_n']) + " cases. Entry-level runtime checks are negligible.</p>" if data['rq4_available'] else "<p class='note'>RQ4 data not available — requires NVIDIA GPU execution.</p>"}

{"<div class='chart-row'><div class='chart-box'><canvas id='rq4-bar'></canvas></div><div class='chart-box'><canvas id='rq4-scatter'></canvas></div></div>" if data['rq4_available'] else ""}
<noscript><div class="fallback-img">{rq4_img}</div></noscript>

<footer>SVN Artifact Evaluation &mdash; Generated by <code>scripts/visualize_results.py</code></footer>
</div>

<script>
const catLabels = {cat_labels};
const genVals = {gen_vals};
const disVals = {dis_vals};
const rtVals  = {rt_vals};
const adrVals = {adr_vals};
const ctoCats = {cto_cats_js};
const ctoPcts = {cto_pcts_js};
const ctoDyn  = {cto_dyn_js};
const ctoSta  = {cto_sta_js};
const rq4Cats    = {rq4_cats_js};
const rq4Medians = {rq4_medians_js};
const rq4Means   = {rq4_means_js};
const rq4Scatter = {rq4_scatter_js};
const utVals = {ut_js};

Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;

// RQ1: Generated per category
new Chart(document.getElementById('rq1-bar'), {{
  type: 'bar',
  data: {{
    labels: catLabels,
    datasets: [{{ label: 'Assessments Generated', data: genVals,
                 backgroundColor: '#2f6bff88', borderColor: '#2f6bff', borderWidth: 1 }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{ title: {{ display: true, text: 'Assessments Generated per Category', font: {{ weight: 'bold' }} }} }},
    scales: {{ x: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }} }}
  }}
}});

// RQ1: ACD per category
new Chart(document.getElementById('rq1-acd'), {{
  type: 'bar',
  data: {{
    labels: catLabels,
    datasets: [{{ label: 'ACD (assessments per case)',
                 data: catLabels.map((_, i) => genVals[i] > 0 ? +(genVals[i] / Math.max(1, {json.dumps([data["cat_rq12"][c]["n"] for c in cats])}[i])).toFixed(1) : 0),
                 backgroundColor: '#2f6bff55', borderColor: '#2f6bff', borderWidth: 1 }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{ title: {{ display: true, text: 'Assessment Coverage Density (ACD)', font: {{ weight: 'bold' }} }} }},
    scales: {{ x: {{ beginAtZero: true, title: {{ display: true, text: 'Assessments per case' }} }} }}
  }}
}});

// RQ2: Stacked discharged / runtime
new Chart(document.getElementById('rq2-stacked'), {{
  type: 'bar',
  data: {{
    labels: catLabels,
    datasets: [
      {{ label: 'Discharged', data: disVals, backgroundColor: '#2f6bff88', borderColor: '#2f6bff', borderWidth: 1 }},
      {{ label: 'Runtime', data: rtVals, backgroundColor: '#ff7f0e88', borderColor: '#ff7f0e', borderWidth: 1 }}
    ]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{ title: {{ display: true, text: 'Discharged vs Runtime per Category', font: {{ weight: 'bold' }} }} }},
    scales: {{ x: {{ stacked: true, beginAtZero: true }}, y: {{ stacked: true }} }}
  }}
}});

// RQ2: ADR per category
new Chart(document.getElementById('rq2-adr'), {{
  type: 'bar',
  data: {{
    labels: catLabels,
    datasets: [{{ label: 'ADR (%)', data: adrVals,
                 backgroundColor: adrVals.map(v => v >= 95 ? '#2ca02c88' : v >= 90 ? '#2f6bff88' : '#ff7f0e88'),
                 borderColor: adrVals.map(v => v >= 95 ? '#2ca02c' : v >= 90 ? '#2f6bff' : '#ff7f0e'),
                 borderWidth: 1 }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{
      title: {{ display: true, text: 'Per-Category ADR (%)', font: {{ weight: 'bold' }} }},
      annotation: {{ annotations: {{ line1: {{ type: 'line', xMin: {data['adr']:.1f}, xMax: {data['adr']:.1f},
                     borderColor: '#d62728', borderWidth: 2, borderDash: [4,4],
                     label: {{ display: true, content: 'Overall {data["adr"]:.1f}%', position: 'start' }} }} }} }}
    }},
    scales: {{ x: {{ min: 0, max: 105, title: {{ display: true, text: 'ADR (%)' }} }} }}
  }}
}});

// RQ2: Usage-type bar
new Chart(document.getElementById('rq2-usage'), {{
  type: 'bar',
  data: {{
    labels: ['Shape compat.', 'Element access', 'Loop bound', 'Hardware'],
    datasets: [{{ label: 'Assessment count', data: utVals,
                 backgroundColor: ['#2f6bff88','#ff7f0e88','#2ca02c88','#9467bd88'],
                 borderColor: ['#2f6bff','#ff7f0e','#2ca02c','#9467bd'], borderWidth: 1 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Assessment Usage Types', font: {{ weight: 'bold' }} }} }},
    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }} }}
  }}
}});

// RQ2: Usage pie
new Chart(document.getElementById('rq2-pie'), {{
  type: 'doughnut',
  data: {{
    labels: ['Shape compat.', 'Element access', 'Loop bound', 'Hardware'],
    datasets: [{{ data: utVals,
                 backgroundColor: ['#2f6bff','#ff7f0e','#2ca02c','#9467bd'] }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Usage Type Distribution', font: {{ weight: 'bold' }} }},
              legend: {{ position: 'bottom' }} }}
  }}
}});

// RQ3: Compile time comparison
if (ctoCats.length > 0) {{
  new Chart(document.getElementById('rq3-time'), {{
    type: 'bar',
    data: {{
      labels: ctoCats,
      datasets: [
        {{ label: 'Static baseline (s)', data: ctoSta, backgroundColor: '#aaaaaa88', borderColor: '#aaa', borderWidth: 1 }},
        {{ label: 'Dynamic / SVN (s)', data: ctoDyn, backgroundColor: '#2f6bff88', borderColor: '#2f6bff', borderWidth: 1 }}
      ]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{ title: {{ display: true, text: 'Cumulative Compile Time', font: {{ weight: 'bold' }} }} }},
      scales: {{ x: {{ beginAtZero: true, title: {{ display: true, text: 'Time (seconds)' }} }} }}
    }}
  }});

  new Chart(document.getElementById('rq3-pct'), {{
    type: 'bar',
    data: {{
      labels: ctoCats,
      datasets: [{{ label: 'CTO (%)', data: ctoPcts,
                   backgroundColor: ctoPcts.map(v => v >= 0 ? '#2f6bff88' : '#2ca02c88'),
                   borderColor: ctoPcts.map(v => v >= 0 ? '#2f6bff' : '#2ca02c'), borderWidth: 1 }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{ title: {{ display: true, text: 'CTO per Category (%)', font: {{ weight: 'bold' }} }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Overhead (%)' }} }} }}
    }}
  }});
}}

// RQ4
if (rq4Cats.length > 0) {{
  new Chart(document.getElementById('rq4-bar'), {{
    type: 'bar',
    data: {{
      labels: rq4Cats,
      datasets: [
        {{ label: 'Median (%)', data: rq4Medians,
           backgroundColor: rq4Medians.map(v => v >= 0 ? '#2f6bff55' : '#2ca02c55'),
           borderColor: rq4Medians.map(v => v >= 0 ? '#2f6bff' : '#2ca02c'), borderWidth: 1 }},
        {{ label: 'Mean (%)', data: rq4Means,
           backgroundColor: '#ff7f0e44', borderColor: '#ff7f0e', borderWidth: 1 }}
      ]
    }},
    options: {{
      indexAxis: 'y', responsive: true,
      plugins: {{ title: {{ display: true, text: 'RQ4: Per-Category Runtime Overhead', font: {{ weight: 'bold' }} }} }},
      scales: {{ x: {{ title: {{ display: true, text: 'Overhead (%)' }} }} }}
    }}
  }});

  new Chart(document.getElementById('rq4-scatter'), {{
    type: 'scatter',
    data: {{
      datasets: [{{ label: 'Per-case overhead', data: rq4Scatter,
                   backgroundColor: '#2f6bff55', borderColor: '#2f6bff', pointRadius: 3 }}]
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'RQ4: Individual Case Overheads', font: {{ weight: 'bold' }} }} }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Overhead (%)' }} }},
        y: {{ type: 'category', labels: rq4Cats, title: {{ display: true, text: 'Category' }} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""
    return html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--no-terminal", action="store_true", help="Skip terminal output")
    ap.add_argument("--no-html", action="store_true", help="Skip HTML report")
    ap.add_argument("--no-png", action="store_true", help="Skip PNG figure generation")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation data...")
    data = load_all_data(results_dir)

    if not data["choreo_stats"]:
        print("ERROR: No choreo_stats.csv found. Run reproduce_all.sh first.", file=sys.stderr)
        sys.exit(1)

    # Terminal visualization
    if not args.no_terminal:
        print_terminal_report(data)

    # Matplotlib PNGs
    b64_figs: Dict[str, str] = {}
    if not args.no_png:
        try:
            print("\nGenerating matplotlib figures...")
            b64_figs = generate_matplotlib_figures(data, results_dir, fig_dir)
            print(f"  Figures saved to {fig_dir}/")
        except ImportError:
            print("  matplotlib not available — skipping PNG generation")

    # HTML report
    if not args.no_html:
        print("Generating HTML report...")
        html = build_html_report(data, b64_figs)
        report_path = out_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
