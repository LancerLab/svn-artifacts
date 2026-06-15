#!/usr/bin/env python3
"""plot_rq1_combined.py — Generate combined RQ1 figure: generation vs discharge per system.

Each system gets two grouped bars:
  - Generation bar (stacked): compiler-generated (blue) + manual (orange)
  - Discharge bar (stacked, hatched): static-discharged (green) + runtime remaining (red)

Output: figures/fig-rq1-generation-discharge.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES = WORKSPACE / "latex" / "cgo27-svn" / "figures"

DATA = [
    # (label, compiler_gen, manual, discharged, runtime)
    ("SVN",         12592,  0,     11753, 839),
    ("MLIR\n(linalg)", 2362, 272,  1657,  977),
    ("IREE",        370,    0,     0,     370),
    ("Triton",      0,      1945,  0,     1945),
]

COLOR_COMP_GEN   = "#2f6bff"   # blue — compiler-generated assessments
COLOR_MANUAL     = "#FF9800"   # orange — manual/user annotations
COLOR_DISCHARGED = "#4CAF50"   # green — statically discharged
COLOR_RUNTIME    = "#d62728"   # red — residual runtime assertions


def plot_generation_discharge(out_dir: Path):
    labels = [d[0] for d in DATA]
    comp_gen   = np.array([d[1] for d in DATA], dtype=float)
    manual     = np.array([d[2] for d in DATA], dtype=float)
    discharged = np.array([d[3] for d in DATA], dtype=float)
    runtime    = np.array([d[4] for d in DATA], dtype=float)

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.35
    gap = 0.04

    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.18)

    x_gen = x - bar_w / 2 - gap / 2
    x_dis = x + bar_w / 2 + gap / 2

    # Generation bars (solid fill)
    ax.bar(x_gen, comp_gen, width=bar_w, color=COLOR_COMP_GEN,
           edgecolor="k", linewidth=0.3, label="Compiler-generated")
    ax.bar(x_gen, manual, width=bar_w, bottom=comp_gen, color=COLOR_MANUAL,
           edgecolor="k", linewidth=0.3, label="Manual annotation")

    # Discharge bars (hatched for B&W distinguishability)
    ax.bar(x_dis, discharged, width=bar_w, color=COLOR_DISCHARGED,
           edgecolor="k", linewidth=0.3, hatch="..", label="Discharged (static)")
    ax.bar(x_dis, runtime, width=bar_w, bottom=discharged, color=COLOR_RUNTIME,
           edgecolor="k", linewidth=0.3, hatch="..", label="Residual (runtime)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Generated / discharged count", fontsize=7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=6.5)

    ymax = max(comp_gen + manual) * 1.12
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.legend(loc="upper right", ncol=2, fontsize=5.5, framealpha=0.85,
              handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    out = out_dir / "fig-rq1-generation-discharge"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES))
    args = parser.parse_args()

    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating RQ1 combined figure...")
    plot_generation_discharge(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
