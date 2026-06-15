#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/plot_assessment_histograms.py

Read the unified benchmark/results/all_stats.csv and render publication-quality
figures comparing assessment metrics across Choreo, MLIR, Memref, IREE and
Triton.

Figures produced
----------------

Figure 1 – overall_totals.pdf / .png
  Grouped-bar chart: total generated vs discharged vs runtime per system.

Figure 2 – resolution_rate_by_system.pdf / .png
  Bar chart: overall compile-time discharge rate (%) per system.

Figure 3 – per_category_generated.pdf / .png
  Stacked grouped bar per category × system for total generated assertions.

Figure 4 – per_category_resolution.pdf / .png
  Heat-map of discharge rate per (category, system).

Figure 5 – generated_distribution.pdf / .png
  Box plot of per-case generated_total counts per system.

Usage
-----
  python scripts/plot_assessment_histograms.py          # uses default paths
  python scripts/plot_assessment_histograms.py \\
      --input benchmark/results/all_stats.csv \\
      --out-dir benchmark/results/figures
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    sys.exit("matplotlib and numpy are required:  pip install matplotlib numpy")

WORKSPACE_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT  = WORKSPACE_ROOT / "benchmark/results/all_stats.csv"
DEFAULT_CHOREO = WORKSPACE_ROOT / "benchmark/results/choreo_stats.csv"
DEFAULT_OUT    = WORKSPACE_ROOT / "benchmark/results/figures"

# Display order — matches RQ1 (bottom-to-top: Triton, MLIR memref, MLIR tensor, IREE, SVN)
SYSTEM_ORDER  = ["triton", "memref", "mlir", "iree", "choreo"]
SYSTEM_LABELS = {
    "choreo": "SVN",
    "mlir":   "MLIR (tensor)",
    "memref": "MLIR (memref)",
    "iree":   "IREE",
    "triton": "Triton",
}
SYSTEM_COLORS = {
    "choreo": "#2196F3",   # blue
    "mlir":   "#4CAF50",   # green
    "memref": "#8BC34A",   # light-green
    "iree":   "#FF9800",   # orange
    "triton": "#F44336",   # red
}

CATEGORY_ORDER = [
    "relu", "sigmoid", "gelu", "elemwise_add", "batch_norm",
    "layer_normalization", "softmax", "reduce_mean",
    "matmul", "conv2d",
    "concat", "reshape", "transpose",
    "embedding", "max_pool2d",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


import re as _re

_DYN_PATTERN = _re.compile(r'(?:\d|x)([A-Z])(?:\d|x)')

def _is_dynamic_case(case_name: str) -> bool:
    """Infer whether a case has symbolic/dynamic dimensions from its name."""
    if 'dynamic' in case_name.lower():
        return True
    return bool(_DYN_PATTERN.search(case_name))


def load_data(csv_path: Path):
    """Load unified CSV; return list of dicts, only 'ok' rows with numeric data."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("status", "ok")).lower() not in {"ok", ""}:
                continue
            gen = _safe_int(r.get("generated_total", 0))
            dis = _safe_int(r.get("discharged", 0))
            rt  = _safe_int(r.get("runtime", dis))   # fallback
            if rt == 0 and dis == 0 and gen == 0:
                # runtime may be missing; infer from gen - discharged
                rt = gen - dis
            rate = _safe_float(r.get("resolution_rate", 0))
            case = r.get("case_name", "")
            rows.append(dict(
                system   = r.get("system", ""),
                category = r.get("category", ""),
                case_name= case,
                generated= gen,
                discharged=dis,
                runtime  = rt,
                rate     = rate,
                is_dynamic= _is_dynamic_case(case),
            ))
    return rows


def aggregate_by_system(rows):
    agg = {s: dict(generated=0, discharged=0, runtime=0, n=0) for s in SYSTEM_ORDER}
    for r in rows:
        s = r["system"]
        if s not in agg:
            continue
        agg[s]["generated"]  += r["generated"]
        agg[s]["discharged"] += r["discharged"]
        agg[s]["runtime"]    += r["runtime"]
        agg[s]["n"]          += 1
    return agg


def aggregate_by_system_dynamic(rows):
    """Return {system: {'all': {...}, 'static': {...}, 'dynamic': {...}}}."""
    agg = {}
    for s in SYSTEM_ORDER:
        agg[s] = {
            cat: dict(generated=0, discharged=0, runtime=0, n=0)
            for cat in ("all", "static", "dynamic")
        }
    for r in rows:
        s = r["system"]
        if s not in agg:
            continue
        dyncat = "dynamic" if r.get("is_dynamic") else "static"
        for cat in ("all", dyncat):
            agg[s][cat]["generated"]  += r["generated"]
            agg[s][cat]["discharged"] += r["discharged"]
            agg[s][cat]["runtime"]    += r["runtime"]
            agg[s][cat]["n"]          += 1
    return agg


def aggregate_by_category(rows):
    """Return {category: {system: {generated, discharged, runtime}}}."""
    agg: dict = defaultdict(lambda: defaultdict(lambda: dict(generated=0, discharged=0, runtime=0, n=0)))
    for r in rows:
        s = r["system"]
        c = r["category"]
        agg[c][s]["generated"]  += r["generated"]
        agg[c][s]["discharged"] += r["discharged"]
        agg[c][s]["runtime"]    += r["runtime"]
        agg[c][s]["n"]          += 1
    return dict(agg)


def per_case_generated(rows):
    """Return {system: [generated_total per case]}."""
    d = defaultdict(list)
    for r in rows:
        d[r["system"]].append(r["generated"])
    return d


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use hyphens in filenames to match LaTeX \includegraphics references
    name_hyph = name.replace("_", "-")
    for ext in ("pdf", "png"):
        path = out_dir / f"{name_hyph}.{ext}"
        fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _systems_present(systems_in_data: set) -> list[str]:
    return [s for s in SYSTEM_ORDER if s in systems_in_data]


# ---------------------------------------------------------------------------
# Figure 1: overall totals — grouped stacked bars (all / dynamic / static)
# ---------------------------------------------------------------------------

def plot_overall_totals(agg_dyn, out_dir: Path, systems: list[str]):
    """Horizontal stacked bars: discharged (blue) + runtime (red) = generated.
    Three sub-bars per system: All, Dynamic, Static, each with distinct hatch.
    Order matches RQ1 (bottom → top): Triton, MLIR(memref), MLIR(tensor), IREE, SVN.
    """
    subcats = ["all", "dynamic", "static"]
    subcat_labels = {"all": "All", "dynamic": "Dyn", "static": "Sta"}
    subcat_hatches = {"all": "", "dynamic": "//", "static": ".."}
    n_sys = len(systems)
    n_sub = len(subcats)
    bar_h = 0.16
    group_gap = 0.08

    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.18)

    yticks = []
    ylabels = []

    # Exclude systems with zero discharge (IREE, Triton) — the table shows them
    systems = [s for s in systems
               if agg_dyn[s]["all"]["discharged"] > 0]
    n_sys = len(systems)

    for i, s in enumerate(systems):
        base_y = i * (n_sub * bar_h + group_gap)
        for j, sc in enumerate(subcats):
            y_pos = base_y + j * bar_h
            d = agg_dyn[s][sc]
            dis_v = d["discharged"]
            rt_v  = d["runtime"]
            hatch = subcat_hatches[sc]
            ax.barh(y_pos, dis_v, height=bar_h * 0.9, color="#66BB6A",
                    edgecolor="k", linewidth=0.3, hatch=hatch)
            ax.barh(y_pos, rt_v, height=bar_h * 0.9, left=dis_v, color="#EF5350",
                    edgecolor="k", linewidth=0.3, hatch=hatch)
            # Small label on the left side of each sub-bar
            if j == 0:
                yticks.append(base_y + bar_h)
                ylabels.append(SYSTEM_LABELS.get(s, s))
            # Annotate total on the right
            total = dis_v + rt_v
            if total > 0:
                ax.text(total + 80, y_pos, f"{subcat_labels[sc]}",
                        va="center", fontsize=6, color="#555")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Assessment count", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylim(-0.3, n_sys * (n_sub * bar_h + group_gap) - group_gap + bar_h)

    # Legend: discharged/runtime + hatch explanation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#66BB6A", edgecolor="k", label="Discharged"),
        Patch(facecolor="#EF5350", edgecolor="k", label="Runtime"),
        Patch(facecolor="white", edgecolor="k", hatch="", label="All"),
        Patch(facecolor="white", edgecolor="k", hatch="//", label="Dynamic"),
        Patch(facecolor="white", edgecolor="k", hatch="..", label="Static"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              framealpha=0.85, ncol=2)

    savefig(fig, out_dir, "fig1_overall_totals")
    print(f"  → fig1_overall_totals")


# ---------------------------------------------------------------------------
# Figure 2: resolution rate by system
# ---------------------------------------------------------------------------

def plot_resolution_rate(agg_sys, out_dir: Path, systems: list[str]):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(systems))
    rates = []
    for s in systems:
        g = agg_sys[s]["generated"]
        d = agg_sys[s]["discharged"]
        rates.append((d / g * 100) if g > 0 else 0.0)

    colors = [SYSTEM_COLORS.get(s, "#888") for s in systems]
    bars = ax.bar(x, rates, color=colors, edgecolor="k", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems], fontsize=11)
    ax.set_ylabel("Compile-time discharge rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Compile-time assessment resolution rate per system", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    savefig(fig, out_dir, "fig2_resolution_rate")
    print(f"  → fig2_resolution_rate")


# ---------------------------------------------------------------------------
# Figure 3: per-category generated totals (grouped bar)
# ---------------------------------------------------------------------------

def plot_per_category_generated(agg_cat, out_dir: Path, systems: list[str]):
    cats = [c for c in CATEGORY_ORDER if c in agg_cat]
    cats += sorted(c for c in agg_cat if c not in CATEGORY_ORDER)

    n_cats = len(cats)
    n_sys  = len(systems)
    w = 0.8 / n_sys
    x = np.arange(n_cats)

    fig, ax = plt.subplots(figsize=(max(12, n_cats * 1.2), 5))
    for i, s in enumerate(systems):
        vals = [agg_cat.get(c, {}).get(s, {}).get("generated", 0) for c in cats]
        offset = (i - n_sys / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=SYSTEM_LABELS.get(s, s),
               color=SYSTEM_COLORS.get(s, "#888"), edgecolor="k", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Total assessments generated", fontsize=11)
    ax.set_title("Assessments generated per category × system", fontsize=12)
    ax.legend(fontsize=9, ncol=n_sys)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    savefig(fig, out_dir, "fig3_per_category_generated")
    print(f"  → fig3_per_category_generated")


# ---------------------------------------------------------------------------
# Figure 4: heat-map of discharge rate per (category, system)
# ---------------------------------------------------------------------------

def plot_heatmap_resolution(agg_cat, out_dir: Path, systems: list[str]):
    cats = [c for c in CATEGORY_ORDER if c in agg_cat]
    cats += sorted(c for c in agg_cat if c not in CATEGORY_ORDER)

    # Only show systems that actually discharge (exclude IREE/Triton)
    shown_systems = [s for s in systems if s in ("choreo", "mlir", "memref")]

    # Build matrix: rows = categories, cols = shown systems
    # Use -1 to indicate "no assessments generated" (e.g., reshape in Choreo)
    matrix = np.full((len(cats), len(shown_systems)), -1.0)
    for ci, cat in enumerate(cats):
        for si, s in enumerate(shown_systems):
            d = agg_cat.get(cat, {}).get(s, {})
            gen = d.get("generated", 0)
            dis = d.get("discharged", 0)
            if gen > 0:
                matrix[ci, si] = (dis / gen * 100)

    # Custom colormap: -1 → light gray, 0..100 → RdYlGn
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    cmap_copy = cmap.copy()
    cmap_copy.set_under("#e0e0e0")  # gray for N/A

    fig, ax = plt.subplots(figsize=(max(4, len(shown_systems) * 1.8), max(5, len(cats) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap_copy, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Discharge rate (%)")

    ax.set_xticks(np.arange(len(shown_systems)))
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in shown_systems], fontsize=10)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_title("Compile-time discharge rate (%) per category", fontsize=12)

    for ci in range(len(cats)):
        for si in range(len(shown_systems)):
            v = matrix[ci, si]
            if v < 0:
                ax.text(si, ci, "N/A", ha="center", va="center",
                        fontsize=7, color="#666", fontstyle="italic")
            else:
                color = "black" if 20 < v < 80 else "white"
                ax.text(si, ci, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    savefig(fig, out_dir, "fig4_heatmap_resolution")
    print(f"  → fig4_heatmap_resolution")


# ---------------------------------------------------------------------------
# Figure 5: SVN per-usage-type discharge analysis
# ---------------------------------------------------------------------------

USAGE_TYPE_ORDER = ["shape", "elem", "loop", "hw"]
USAGE_TYPE_LABELS = {
    "shape": "Shape-\ncompat.",
    "elem":  "Element-\naccess",
    "loop":  "Loop-\nbound",
    "hw":    "Target-\nconstraint",
}


def aggregate_usage_types(rows):
    """Aggregate per-usage-type totals and runtime for choreo only."""
    tots = {t: 0 for t in USAGE_TYPE_ORDER}
    rts  = {t: 0 for t in USAGE_TYPE_ORDER}
    for r in rows:
        if r["system"] != "choreo" or r["generated"] == 0:
            continue
        for t in USAGE_TYPE_ORDER:
            tots[t] += r.get(f"ut_{t}", 0)
            rts[t]  += r.get(f"rt_{t}", 0)
    return tots, rts


def plot_usage_type_discharge(rows, out_dir: Path):
    """Two independent subfigures saved as fig5a (pie) and fig5b (bars),
    placed side-by-side in LaTeX to avoid any label overlap."""
    tots, rts = aggregate_usage_types(rows)

    # Render order: elem, loop, shape (avoids pie-label / title collision)
    render_order = [t for t in ["elem", "loop", "shape", "hw"] if tots.get(t, 0) > 0]
    if not render_order:
        print("  [skip] fig5_usage_type_discharge: no per-type data")
        return

    type_colors = {"shape": "#4CAF50", "elem": "#2196F3", "loop": "#FF9800", "hw": "#9C27B0"}
    dis_color = "#66BB6A"   # green – consistent with fig1
    rt_color  = "#EF5350"

    type_pretty = {
        "shape": "Shape-compat.",
        "elem":  "Element-access",
        "loop":  "Loop-bound",
        "hw":    "Target-constr.",
    }

    total_all = sum(tots[t] for t in render_order)
    n_types = len(render_order)

    # ---- fig5a: pie chart of type distribution ----
    fig_pie, ax_pie = plt.subplots(figsize=(2.5, 2.5))
    sizes = [tots[t] for t in render_order]
    colors = [type_colors[t] for t in render_order]
    labels = [f"{type_pretty[t]}\n{tots[t]:,} ({tots[t]/total_all*100:.1f}%)"
              for t in render_order]
    wedges, texts = ax_pie.pie(
        sizes, labels=labels, colors=colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="k", linewidth=0.5),
        labeldistance=1.25, textprops={"fontsize": 7})
    ax_pie.set_title("Distribution", fontsize=9, fontweight="bold", pad=8)
    savefig(fig_pie, out_dir, "fig5a_usage_pie")
    print(f"  → fig5a_usage_pie")

    # ---- fig5b: vertical stacked bars (discharged + runtime) ----
    fig_bar, ax_bar = plt.subplots(figsize=(2.5, 2.5))
    x_pos = np.arange(n_types)
    dis_vals = [tots[t] - rts[t] for t in render_order]
    rt_vals  = [rts[t] for t in render_order]

    ax_bar.bar(x_pos, dis_vals, width=0.55, color=dis_color,
               edgecolor="k", linewidth=0.5, label="Discharged")
    ax_bar.bar(x_pos, rt_vals, width=0.55, bottom=dis_vals, color=rt_color,
               edgecolor="k", linewidth=0.5, label="Runtime")

    # Annotate ADR above each bar
    for i, t in enumerate(render_order):
        adr = (tots[t] - rts[t]) / tots[t] * 100 if tots[t] else 0
        ax_bar.text(i, tots[t] + total_all * 0.015,
                    f"{adr:.1f}%", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([type_pretty[t] for t in render_order], fontsize=7)
    ax_bar.set_ylabel("Assessment count", fontsize=7)
    ax_bar.set_title("Discharge Ratio", fontsize=9, fontweight="bold", pad=8)
    ax_bar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_bar.tick_params(axis="y", labelsize=7)
    ax_bar.set_ylim(0, max(tots[t] for t in render_order) * 1.22)
    ax_bar.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    savefig(fig_bar, out_dir, "fig5b_usage_bars")
    print(f"  → fig5b_usage_bars")


# ---------------------------------------------------------------------------
# Figure 6: stacked bar — discharged vs runtime per system
# ---------------------------------------------------------------------------

def plot_stacked_discharge(agg_sys, out_dir: Path, systems: list[str]):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(systems))
    w = 0.55

    dis_vals = [agg_sys[s]["discharged"] for s in systems]
    rt_vals  = [agg_sys[s]["runtime"]    for s in systems]

    bars_dis = ax.bar(x, dis_vals, w, label="Discharged at compile time",
                      color="#1565C0", edgecolor="k", linewidth=0.5)
    bars_rt  = ax.bar(x, rt_vals, w, bottom=dis_vals,
                      label="Runtime assertions",
                      color="#EF5350", edgecolor="k", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems], fontsize=11)
    ax.set_ylabel("Assertion count", fontsize=11)
    ax.set_title("Compile-time vs runtime assertions (generated total)", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    savefig(fig, out_dir, "fig6_stacked_discharge")
    print(f"  → fig6_stacked_discharge")


def load_choreo_usage_types(csv_path: Path):
    """Load choreo_stats.csv and return rows with per-usage-type fields."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("status", "")).lower() != "ok":
                continue
            rows.append(dict(
                system="choreo",
                category=r.get("category", ""),
                case_name=r.get("case_name", ""),
                generated=_safe_int(r.get("generated", 0)),
                ut_shape=_safe_int(r.get("ut_shape", 0)),
                ut_elem=_safe_int(r.get("ut_elem", 0)),
                ut_loop=_safe_int(r.get("ut_loop", 0)),
                ut_hw=_safe_int(r.get("ut_hw", 0)),
                rt_shape=_safe_int(r.get("rt_shape", 0)),
                rt_elem=_safe_int(r.get("rt_elem", 0)),
                rt_loop=_safe_int(r.get("rt_loop", 0)),
                rt_hw=_safe_int(r.get("rt_hw", 0)),
            ))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--systems", nargs="+", default=None,
                    choices=SYSTEM_ORDER,
                    help="Which systems to include (default: all present in CSV)")
    args = ap.parse_args(argv)

    if not args.input.exists():
        sys.exit(
            f"Input CSV not found: {args.input}\n"
            "Run:  python scripts/collect_all_stats.py  first."
        )

    print(f"Loading {args.input} …")
    rows = load_data(args.input)
    if not rows:
        sys.exit("No valid rows found in input CSV.")

    present = {r["system"] for r in rows}
    systems = args.systems if args.systems else _systems_present(present)
    print(f"Systems: {systems}")

    agg_sys = aggregate_by_system(rows)
    agg_dyn = aggregate_by_system_dynamic(rows)
    agg_cat = aggregate_by_category(rows)
    per_c   = per_case_generated(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {args.out_dir} …")

    plot_overall_totals(agg_dyn, args.out_dir, systems)
    plot_resolution_rate(agg_sys, args.out_dir, systems)
    plot_per_category_generated(agg_cat, args.out_dir, systems)
    plot_heatmap_resolution(agg_cat, args.out_dir, systems)

    # Per-usage-type discharge (Choreo only, from separate CSV)
    choreo_rows = load_choreo_usage_types(DEFAULT_CHOREO)
    if choreo_rows:
        plot_usage_type_discharge(choreo_rows, args.out_dir)

    plot_stacked_discharge(agg_sys, args.out_dir, systems)

    print("Done.")


if __name__ == "__main__":
    main()
