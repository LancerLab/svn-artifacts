#!/usr/bin/env python3
"""plot_rq1_percategory.py — Per-category ACD and ADR grouped bar chart.

Shows SVN vs MLIR (linalg) side-by-side for each operator category.
Two panels: top = ACD (assessments/case), bottom = ADR (%).

Output: figures/fig-rq1-percategory.pdf
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = WORKSPACE / "benchmark" / "results" / "all_stats.csv"
DEFAULT_FIGURES = WORKSPACE / "latex" / "cgo27-svn" / "figures"

CATEGORY_ORDER = [
    "relu", "sigmoid", "gelu", "elemwise_add", "batch_norm",
    "layer_normalization", "softmax", "reduce_mean",
    "matmul", "conv2d",
    "concat", "transpose",
    "embedding", "max_pool2d",
]

CATEGORY_LABELS = {
    "relu": "relu", "sigmoid": "sigm", "gelu": "gelu",
    "elemwise_add": "add", "batch_norm": "bn",
    "layer_normalization": "ln", "softmax": "smax",
    "reduce_mean": "rmean", "matmul": "mm", "conv2d": "conv",
    "concat": "cat", "transpose": "trans",
    "embedding": "emb", "max_pool2d": "pool",
}

COLOR_SVN = "#2f6bff"
COLOR_MLIR = "#4CAF50"


def load_percategory(csv_path: Path):
    """Load CSV, compute per-category ACD and ADR for choreo and mlir."""
    cats = defaultdict(lambda: defaultdict(lambda: {"gen": 0, "dis": 0, "n": 0}))

    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") not in ("ok", ""):
                continue
            system = r.get("system", "")
            if system == "memref":
                continue
            if system not in ("choreo", "mlir"):
                continue
            cat = r.get("category", "")
            try:
                gen = int(r.get("generated_total", 0))
                dis = int(r.get("discharged", 0))
            except (ValueError, TypeError):
                continue
            cats[cat][system]["gen"] += gen
            cats[cat][system]["dis"] += dis
            cats[cat][system]["n"] += 1

    result = {}
    for cat in CATEGORY_ORDER:
        if cat not in cats:
            continue
        svn = cats[cat].get("choreo", {"gen": 0, "dis": 0, "n": 0})
        mlir = cats[cat].get("mlir", {"gen": 0, "dis": 0, "n": 0})
        svn_acd = svn["gen"] / svn["n"] if svn["n"] else 0
        svn_adr = svn["dis"] / svn["gen"] * 100 if svn["gen"] else 0
        mlir_acd = mlir["gen"] / mlir["n"] if mlir["n"] else 0
        mlir_adr = mlir["dis"] / mlir["gen"] * 100 if mlir["gen"] else 0
        result[cat] = {
            "svn_acd": svn_acd, "svn_adr": svn_adr,
            "mlir_acd": mlir_acd, "mlir_adr": mlir_adr,
        }
    return result


def plot_percategory(data: dict, out_dir: Path):
    cats = [c for c in CATEGORY_ORDER if c in data]
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]
    n = len(cats)

    svn_acd = [data[c]["svn_acd"] for c in cats]
    mlir_acd = [data[c]["mlir_acd"] for c in cats]
    svn_adr = [data[c]["svn_adr"] for c in cats]
    mlir_adr = [data[c]["mlir_adr"] for c in cats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.33, 2.8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.94, bottom=0.14)

    x = np.arange(n)
    bar_w = 0.35

    ax1.bar(x - bar_w / 2, svn_acd, width=bar_w, color=COLOR_SVN,
            edgecolor="k", linewidth=0.3, label="SVN")
    ax1.bar(x + bar_w / 2, mlir_acd, width=bar_w, color=COLOR_MLIR,
            edgecolor="k", linewidth=0.3, label="MLIR (lg)")
    ax1.set_ylabel("ACD", fontsize=7)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.tick_params(axis="y", labelsize=6.5)
    ax1.legend(loc="upper right", ncol=2, fontsize=6, framealpha=0.85,
               handlelength=1.0, handletextpad=0.4)

    ax2.bar(x - bar_w / 2, svn_adr, width=bar_w, color=COLOR_SVN,
            edgecolor="k", linewidth=0.3)
    ax2.bar(x + bar_w / 2, mlir_adr, width=bar_w, color=COLOR_MLIR,
            edgecolor="k", linewidth=0.3)
    ax2.set_ylabel("ADR (%)", fontsize=7)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.tick_params(axis="y", labelsize=6.5)
    ax2.set_ylim(0, 105)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")

    out = out_dir / "fig-rq1-percategory"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES))
    args = parser.parse_args()

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating per-category ACD/ADR chart...")
    data = load_percategory(Path(args.input))
    plot_percategory(data, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
