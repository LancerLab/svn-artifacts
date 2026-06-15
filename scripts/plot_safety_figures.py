#!/usr/bin/env python3
"""plot_safety_figures.py — Generate safety-metric figures for the SVN paper.

Produces:
  fig0-annotation-burden.pdf  — stacked bar: manual vs auto-generated per system
  fig-rtc-overhead.pdf        — box plot: runtime overhead per --runtime-check level
                                 (only generated if choreo_runtime.csv has the new
                                  5-column format with actual numeric data)

Usage:
  python3 scripts/plot_safety_figures.py [--results-dir DIR] [--figures-dir DIR]
"""

import argparse
import csv
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = WORKSPACE / "benchmark" / "results"
DEFAULT_FIGURES = WORKSPACE / "benchmark" / "results" / "figures"

SYSTEM_COLORS = {
    "SVN":         "#2f6bff",
    "MLIR\n(tensor)": "#ff7f0e",
    "MLIR\n(memref)": "#e08010",
    "Triton":       "#d62728",
    "IREE":         "#9467bd",
}

# Order is bottom-to-top in the horizontal bar chart.
# Grouped by annotation style: manual-heavy at bottom, zero-manual at top.
ANNOTATION_DATA = [
    # (label,           manual, auto)
    ("Triton",          1945,   0),
    ("MLIR\n(memref)",  135,    1161),
    ("MLIR\n(tensor)",  272,    2362),
    ("IREE",            0,      370),
    ("SVN",            0,      11524),
]

# Number of successfully compiled cases per system (used for ACD normalisation)
CASE_COUNTS = {
    "SVN":             310,
    "MLIR\n(tensor)":   310,
    "MLIR\n(memref)":   154,
    "Triton":           310,
    "IREE":             310,
}


def require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        # Embed fonts as TrueType (Type 42) in PDF/PS for crisp vector output
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"]  = 42
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc
    return plt, mpatches


def plot_annotation_burden(out_dir: Path):
    """Stacked horizontal bar chart only — the companion table is rendered
    as a proper LaTeX tabular in the manuscript's figure environment."""
    plt, mpatches = require_matplotlib()

    labels     = [d[0] for d in ANNOTATION_DATA]
    manual_abs = [d[1] for d in ANNOTATION_DATA]
    auto_abs   = [d[2] for d in ANNOTATION_DATA]
    cases      = [CASE_COUNTS[lab] for lab in labels]

    total_abs  = [m + a for m, a in zip(manual_abs, auto_abs)]
    auto_acd   = [a / c for a, c in zip(auto_abs,   cases)]
    manual_acd = [m / c for m, c in zip(manual_abs, cases)]
    total_acd  = [ta + ma for ta, ma in zip(auto_acd, manual_acd)]

    import numpy as np
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.22)

    bar_w = 0.55
    x = np.arange(len(labels))

    ax.bar(x, auto_acd,   width=bar_w, color="#2f6bff",
           label="Compiler-generated")
    ax.bar(x, manual_acd, width=bar_w, bottom=auto_acd, color="#d62728",
           label="Manual")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Assessments per case (ACD)", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(total_acd) * 1.12)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=7.5)

    ax.legend(loc="upper left",
              ncol=1, fontsize=7.5, framealpha=0.85,
              handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

    out = out_dir / "fig0-annotation-burden"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def _load_rtc_data(results_dir: Path):
    """Load choreo_runtime.csv and return per-case overhead ratios.

    Returns dict: level -> list of overhead fractions (rtc_X / rtc_none - 1).
    Returns None if data is not yet available.
    """
    csv_path = results_dir / "choreo_runtime.csv"
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    # Check if new 5-column format
    sample = rows[0]
    if "rtc_none_us" not in sample:
        print("  [skip] choreo_runtime.csv is in old format; skipping RTC figure")
        return None

    levels = ["low", "medium", "high"]
    overheads = {lvl: [] for lvl in levels}

    for row in rows:
        # Skip error rows
        try:
            none_us = float(row["rtc_none_us"])
        except (ValueError, TypeError):
            continue
        if none_us <= 0:
            continue

        for lvl in levels:
            col = f"rtc_{lvl}_us"
            try:
                v = float(row[col])
                overhead = (v - none_us) / none_us * 100.0  # percent
                overheads[lvl].append(overhead)
            except (ValueError, TypeError, KeyError):
                pass

    if not any(overheads.values()):
        return None
    return overheads


def plot_rtc_overhead(out_dir: Path, results_dir: Path):
    """Box plot: overhead of each --runtime-check level vs none."""
    overheads = _load_rtc_data(results_dir)
    if overheads is None:
        print("  [skip] RTC data not yet available; skipping fig-rtc-overhead")
        return

    plt, _ = require_matplotlib()

    levels = ["low", "medium", "high"]
    data = [overheads[lvl] for lvl in levels]

    # Sample statistics for annotation
    for lvl, d in zip(levels, data):
        if d:
            print(f"  rtc_{lvl}: n={len(d)} median={statistics.median(d):.2f}% "
                  f"p75={sorted(d)[int(len(d)*0.75)]:.2f}%")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bp = ax.boxplot(data, labels=["low", "medium", "high"],
                    patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})

    colors = ["#aec7e8", "#ffbb78", "#ff9896"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Overhead vs. --runtime-check=none (%)", fontsize=9)
    ax.set_xlabel("--runtime-check level", fontsize=9)
    ax.set_title("Runtime Assertion Overhead per Check Level", fontsize=10, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = out_dir / "fig-rtc-overhead"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def plot_cto_overhead(out_dir: Path, results_dir: Path):
    """Grouped horizontal bar chart: per-category total dynamic vs static compile
    time, derived from choreo_compile_overhead.csv.  Per-category sums are more
    stable than per-file medians for the short-compile cases."""
    csv_path = results_dir / "choreo_compile_overhead.csv"
    if not csv_path.exists():
        print("  [skip] choreo_compile_overhead.csv not found; skipping CTO figure")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Aggregate per category (sum of valid rows only)
    cat_data = {}
    for row in rows:
        cat = row["category"]
        try:
            dyn = float(row["dynamic_ms"])
            sta = float(row["static_ms"])
        except (ValueError, TypeError):
            continue
        if cat not in cat_data:
            cat_data[cat] = {"dyn": 0.0, "sta": 0.0, "n": 0}
        cat_data[cat]["dyn"] += dyn
        cat_data[cat]["sta"] += sta
        cat_data[cat]["n"] += 1

    if not cat_data:
        print("  [skip] no usable CTO data")
        return

    # Sort categories by total static time (ascending → bottom-to-top in horiz bar)
    categories = sorted(cat_data, key=lambda c: cat_data[c]["sta"])

    # Convert ms → seconds for readability
    dyn_s = [cat_data[c]["dyn"] / 1000.0 for c in categories]
    sta_s = [cat_data[c]["sta"] / 1000.0 for c in categories]

    # Pretty category labels
    label_map = {
        "batch_norm": "batch_norm", "conv2d": "conv2d",
        "elemwise_add": "elemwise_add", "embedding": "embedding",
        "gelu": "gelu", "layer_normalization": "layer_norm",
        "matmul": "matmul", "max_pool2d": "max_pool2d",
        "reduce_mean": "reduce_mean", "relu": "relu",
        "reshape": "reshape", "sigmoid": "sigmoid",
        "softmax": "softmax", "transpose": "transpose",
        "concat": "concat",
    }
    labels = [label_map.get(c, c) for c in categories]

    plt, _ = require_matplotlib()

    n = len(categories)
    y = list(range(n))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.14)

    ax.barh([yi + bar_h / 2 for yi in y], sta_s, height=bar_h,
            color="#aaaaaa", label="Static baseline")
    ax.barh([yi - bar_h / 2 for yi in y], dyn_s, height=bar_h,
            color="#2f6bff", label="Dynamic (SVN)", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cumulative compile time per category (s)", fontsize=9)
    ax.set_title("Compile-Time Overhead of SVN", fontsize=10, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate overall CTO from the per-case measurement
    total_dyn = sum(cat_data[c]["dyn"] for c in categories)
    total_sta = sum(cat_data[c]["sta"] for c in categories)
    total_n   = sum(cat_data[c]["n"]   for c in categories)
    overall_cto = (total_dyn / total_sta - 1) * 100.0 if total_sta > 0 else 0.0
    ax.text(0.97, 0.04,
            f"Overall CTO: {overall_cto:.1f}%\n({total_n} dynamic cases, 5 reps median)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    ax.legend(loc="lower right", fontsize=8, framealpha=0.85,
              handlelength=1.2, handletextpad=0.5)

    out = out_dir / "fig-cto-overhead"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def plot_rq4_entry_overhead(out_dir: Path, results_dir: Path):
    """Box+strip plot: per-case runtime overhead of --runtime-check=entry vs none,
    broken down by operator category.  Data from choreo_runtime_entry.csv."""
    csv_path = results_dir / "choreo_runtime_entry.csv"
    if not csv_path.exists():
        print("  [skip] choreo_runtime_entry.csv not found; skipping RQ4 figure")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Filter to valid rows
    ok = [r for r in rows
          if r.get("none_us") not in ("error", "", None)
          and r.get("overhead_pct") not in ("N/A", "", None)]

    if not ok:
        print("  [skip] no valid RQ4 data")
        return

    # Category order: sort by per-category median overhead
    cats_all = sorted(set(r["category"] for r in ok))
    cat_medians = {}
    cat_data = {}
    for cat in cats_all:
        vals = [float(r["overhead_pct"]) for r in ok if r["category"] == cat]
        cat_medians[cat] = statistics.median(vals)
        cat_data[cat] = vals

    # Sort categories by median overhead ascending
    cats = sorted(cats_all, key=lambda c: cat_medians[c])

    label_map = {
        "elemwise_add": "elemwise_add", "gelu": "gelu",
        "max_pool2d": "max_pool2d", "relu": "relu",
        "reshape": "reshape", "sigmoid": "sigmoid",
        "softmax": "softmax", "transpose": "transpose",
        "batch_norm": "batch_norm", "matmul": "matmul",
        "concat": "concat", "conv2d": "conv2d",
        "embedding": "embedding", "layer_normalization": "layer_norm",
        "reduce_mean": "reduce_mean",
    }

    plt, _ = require_matplotlib()
    import random
    random.seed(42)

    n = len(cats)
    y = list(range(n))
    data = [cat_data[c] for c in cats]
    labels = [label_map.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.16)

    # Horizontal box plot
    bp = ax.boxplot(data, vert=False, labels=labels,
                    patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.4},
                    widths=0.55)

    color = "#2f6bff"
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Jitter strip
    import numpy as np
    for i, vals in enumerate(data):
        jitter = np.random.uniform(-0.20, 0.20, len(vals))
        ax.scatter(vals, [i + 1 + j for j in jitter],
                   color=color, alpha=0.5, s=14, zorder=3)

    # Zero line
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Overall median annotation
    all_ovs = [float(r["overhead_pct"]) for r in ok]
    overall_med = statistics.median(all_ovs)
    ax.axvline(overall_med, color="#d62728", linestyle=":", linewidth=1.2,
               label=f"Overall median: {overall_med:.2f}%")

    ax.set_xlabel("Runtime overhead of entry-level assertions vs. no assertions (%)",
                  fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)

    out = out_dir / "fig-rq4-entry-overhead"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                        help="Directory containing *_stats.csv files")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES),
                        help="Output directory for figure PDFs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating safety figures...")
    plot_annotation_burden(figures_dir)
    plot_rtc_overhead(figures_dir, results_dir)
    plot_cto_overhead(figures_dir, results_dir)
    plot_rq4_entry_overhead(figures_dir, results_dir)
    print("Done.")


if __name__ == "__main__":
    main()
