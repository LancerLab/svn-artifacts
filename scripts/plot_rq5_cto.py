#!/usr/bin/env python3
"""plot_rq5_cto.py — Generate compile-time overhead bar chart.

Matches the style of fig0/fig1/fig-rtcost: same figsize, font sizes, grid.
Data: per-category CTO (%) from the compile-time overhead experiment.
"""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES = WORKSPACE / "latex" / "cgo27-svn" / "figures"

CTO_DATA = {
    "batch_norm":   {"n": 19, "cto": 3.0},
    "softmax":      {"n": 11, "cto": 4.2},
    "conv2d":       {"n":  6, "cto": 4.0},
    "elemwise_add": {"n": 10, "cto": 3.8},
    "max_pool2d":   {"n": 10, "cto": 12.9},
    "concat":       {"n": 13, "cto": 2.3},
    "matmul":       {"n": 10, "cto": 2.2},
    "relu":         {"n": 11, "cto": 3.7},
    "embedding":    {"n": 10, "cto": 2.3},
    "gelu":         {"n": 11, "cto": 4.6},
    "sigmoid":      {"n":  9, "cto": 2.0},
    "transpose":    {"n": 11, "cto": 4.7},
    "layer_norm":   {"n":  5, "cto": 19.4},
    "reduce_mean":  {"n":  6, "cto": 7.1},
    "reshape":      {"n": 11, "cto": 0.0},
}

AGGREGATE_CTO = 4.7


def plot_cto(out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import numpy as np

    categories = sorted(CTO_DATA.keys(), key=lambda c: CTO_DATA[c]["cto"], reverse=True)
    labels = [c.replace("_", " ") for c in categories]
    cto_vals = [CTO_DATA[c]["cto"] for c in categories]

    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.28)

    x = np.arange(len(categories))
    bar_w = 0.55

    bars = ax.bar(x, cto_vals, width=bar_w, color="#2f6bff",
                  edgecolor="#1a4abf", linewidth=0.4)

    ax.axhline(AGGREGATE_CTO, color="#d62728", linewidth=1.0, linestyle="--",
               label=f"Aggregate: {AGGREGATE_CTO}%")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.5, rotation=35, ha="right")
    ax.set_ylabel("CTO (%)", fontsize=7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0, max(cto_vals) * 1.12)

    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85,
              handlelength=1.5, handletextpad=0.4)

    out = out_dir / "fig-cto"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    plot_cto(args.figures_dir)


if __name__ == "__main__":
    main()
