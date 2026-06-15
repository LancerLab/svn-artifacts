#!/usr/bin/env python3
"""plot_rq5_rtcost.py — Generate runtime assertion cost bar chart (Figure 9).

Matches the style of fig0-annotation-burden.pdf and fig1-overall-totals.pdf:
same figsize, font sizes, grid, legend style.

Data source: benchmark/results/runtime_hoist_comparison.csv (if available),
otherwise falls back to hardcoded data from the experiment.
"""

import csv
import statistics
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = WORKSPACE / "benchmark" / "results"
DEFAULT_FIGURES = WORKSPACE / "latex" / "cgo27-svn" / "figures"

HARDCODED_DATA = {
    "batch_norm":          {"entry": 0.78, "hoist": 1.81,  "nohoist": 27.11},
    "concat":              {"entry": -0.46, "hoist": 0.63,  "nohoist": 5.56},
    "layer_normalization": {"entry": 0.14, "hoist": 2.08,  "nohoist": 0.08},
    "reduce_mean":         {"entry": 0.04, "hoist": 0.15,  "nohoist": 0.02},
    "transpose":           {"entry": -0.04, "hoist": 7.13,  "nohoist": -0.95},
}


def load_from_csv(results_dir: Path):
    """Try loading from runtime_hoist_comparison.csv (v2 format with entry column)."""
    hoist_csv = results_dir / "runtime_hoist_comparison.csv"
    if not hoist_csv.exists():
        return None

    hoist_rows = []
    with open(hoist_csv) as f:
        hoist_rows = list(csv.DictReader(f))

    cats = {}
    for r in hoist_rows:
        try:
            cat = r["category"]
            none_t = float(r["time_none_us"])
            if none_t <= 0:
                continue

            if "time_entry_us" in r and r["time_entry_us"]:
                entry_t = float(r["time_entry_us"])
                entry_oh = 100.0 * (entry_t - none_t) / none_t
            elif "overhead_entry_pct" in r and r["overhead_entry_pct"]:
                entry_oh = float(r["overhead_entry_pct"])
            else:
                entry_oh = 0.0

            hoist_t = float(r["time_hoist_us"])
            nohoist_t = float(r["time_nohoist_us"])
            hoist_oh = 100.0 * (hoist_t - none_t) / none_t
            nohoist_oh = 100.0 * (nohoist_t - none_t) / none_t
        except (ValueError, KeyError):
            continue

        if cat not in cats:
            cats[cat] = {"entry": [], "hoist": [], "nohoist": []}
        cats[cat]["entry"].append(entry_oh)
        cats[cat]["hoist"].append(hoist_oh)
        cats[cat]["nohoist"].append(nohoist_oh)

    if not cats:
        return None

    result = {}
    for cat, vals in cats.items():
        result[cat] = {
            "entry": statistics.mean(vals["entry"]) if vals["entry"] else 0,
            "hoist": statistics.mean(vals["hoist"]) if vals["hoist"] else 0,
            "nohoist": statistics.mean(vals["nohoist"]) if vals["nohoist"] else 0,
        }
    return result


def plot_rtcost(data: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import numpy as np

    categories = sorted(data.keys())
    labels = [c.replace("_", " ") for c in categories]
    entry_vals = [data[c]["entry"] for c in categories]
    hoist_vals = [data[c]["hoist"] for c in categories]
    nohoist_vals = [data[c]["nohoist"] for c in categories]

    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.24)

    x = np.arange(len(categories))
    bar_w = 0.22

    ax.bar(x - bar_w, entry_vals, width=bar_w, color="#2f6bff",
           edgecolor="#1a4abf", linewidth=0.4, label="Entry")
    ax.bar(x, hoist_vals, width=bar_w, color="#66BB6A",
           edgecolor="#388E3C", linewidth=0.4, label="All (hoisted)")
    ax.bar(x + bar_w, nohoist_vals, width=bar_w, color="#EF5350",
           edgecolor="#C62828", linewidth=0.4, label="All (no-hoist)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("RAO (%) vs. none baseline", fontsize=7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0, color="k", linewidth=0.4)
    ax.tick_params(axis="y", labelsize=6.5)

    ax.legend(loc="upper right", ncol=3, fontsize=6.5, framealpha=0.85,
              handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    out = out_dir / "fig-rtcost"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()

    data = load_from_csv(args.results_dir)
    if data is None:
        print("  [info] Using hardcoded data (CSV not found)")
        data = HARDCODED_DATA

    plot_rtcost(data, args.figures_dir)


if __name__ == "__main__":
    main()
