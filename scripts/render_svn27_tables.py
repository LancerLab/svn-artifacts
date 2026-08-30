#!/usr/bin/env python3
"""scripts/render_svn27_tables.py

Generate the LaTeX tables and figures for the ASPLOS 2027 paper
("The Compiler as Auditor") from the artifact result CSVs.

Inputs  (benchmark/results/): rq4_dependence.csv, rq6_mechanism.csv,
        rq5_{full_svn,no_vn_share,no_vn_simplify,no_vn_both}.csv,
        bug_detection_results.csv, rq3_runtime_overhead.csv,
        choreo_compile_overhead.csv
Outputs (svn/asplos27/tables/*.tex, svn/asplos27/figures/*.pdf)

Usage: python3 scripts/render_svn27_tables.py
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"
PAPER = ROOT.parent / "asplos27"
TABDIR = PAPER / "tables"
FIGDIR = PAPER / "figures"


def load(name):
    p = RESULTS / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def write(name, text):
    TABDIR.mkdir(parents=True, exist_ok=True)
    (TABDIR / name).write_text(text)
    print(f"wrote {TABDIR / name}")


# ---------------------------------------------------------------- RQ1
def table_rq1_category():
    rows = [r for r in load("rq4_dependence.csv") if r["status"] == "ok"]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Per-category discharge. ADR = statically discharged / "
        r"generated obligations.}",
        r"\label{tab:rq1-category}",
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Category & Cases & Gen. & Disch. & ADR & Runtime\\",
        r"\midrule",
    ]
    tot = [0, 0, 0, 0]
    for cat in sorted(by_cat):
        sub = by_cat[cat]
        g = sum(int(r["generated"]) for r in sub)
        d = sum(int(r["discharged"]) for r in sub)
        rt = sum(int(r["runtime"]) for r in sub)
        cat_tex = cat.replace("_", chr(92) + "_")
        lines.append(
            f"{cat_tex} & {len(sub)} & {g:,} & {d:,} & "
            f"{d/g*100:.1f}\\% & {rt:,} \\\\")
        tot[0] += len(sub); tot[1] += g; tot[2] += d; tot[3] += rt
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & {tot[0]} & {tot[1]:,} & {tot[2]:,} & "
        f"{tot[2]/tot[1]*100:.1f}\\% & {tot[3]:,} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    write("rq1_category.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- RQ2
def table_rq2_bugs():
    rows = load("bug_detection_results.csv")
    by_cls = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        e = by_cls[r["bug_class"]]
        e[0] += 1
        e[1] += r["svn_resolution"] == "compile"
        e[2] += r["mlir_resolution"] in ("compile", "runtime")
        e[3] += r["iree_resolution"] in ("compile", "entry")
    pretty = {"dim_mismatch": "Dimension mismatch",
              "input_dep_oob": "Input-dependent OOB",
              "wrong_output": "Wrong output shape",
              "stride_error": "Stride/layout error"}
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Detection of injected bugs (detections per class). "
        r"Ours detects all at compile time; MLIR detects 23 at compile time "
        r"and 20 more only at runtime; IREE detects 23 at runtime entry.}",
        r"\label{tab:rq2-bugs}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Bug class & Injected & Ours & MLIR & IREE\\",
        r"\midrule",
    ]
    tot = [0, 0, 0, 0]
    for cls in sorted(by_cls):
        n, s, m, i = by_cls[cls]
        lines.append(f"{pretty.get(cls, cls)} & {n} & {s} & {m} & {i} \\\\")
        tot[0] += n; tot[1] += s; tot[2] += m; tot[3] += i
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & {tot[0]} & \\textbf{{{tot[1]}}} & {tot[2]} & "
        f"{tot[3]} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    write("rq2_bugs.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- RQ3
def table_rq3_runtime():
    rows = load("rq3_runtime_overhead.csv")
    ok = [r for r in rows
          if all(r.get(f"rtc_{l}_us", "error") not in ("error", "", None)
                 for l in ("none", "low", "medium", "high"))
          and r.get("static_us", "error") not in ("error", "", None)]
    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Residual runtime-check cost: median end-to-end kernel time "
        r"per category (3 runs per variant), normalized to the no-checks "
        r"variant.}",
        r"\label{tab:rq3-runtime}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Category & none ($" + chr(92) + r"mu$s) & low ovh. & high ovh. "
        + chr(92)*2,
        r"\midrule",
    ]
    all_low, all_high, all_static = [], [], []
    for cat in sorted(by_cat):
        sub = by_cat[cat]
        n = statistics.median(float(r["rtc_none_us"]) for r in sub)
        lo = statistics.median(float(r["rtc_low_us"]) for r in sub)
        hi = statistics.median(float(r["rtc_high_us"]) for r in sub)
        st = statistics.median(float(r["static_us"]) for r in sub)
        all_low.append((float(r["rtc_low_us"]) - float(r["rtc_none_us"]))
                       / float(r["rtc_none_us"]) * 100 for r in sub)
        all_high.append((float(r["rtc_high_us"]) - float(r["rtc_none_us"]))
                        / float(r["rtc_none_us"]) * 100 for r in sub)
        cat_tex = cat.replace("_", chr(92) + "_")
        lines.append(f"{cat_tex} & {n:,.0f} & {(lo-n)/n*100:+.1f}\\% & "
                     f"{(hi-n)/n*100:+.1f}\\% " + chr(92)*2)
    flat = lambda xs: [x for sub in xs for x in sub]
    lines += [
        r"\midrule",
        f"\\textbf{{Median over cases}} & & "
        f"{statistics.median(flat(all_low)):+.1f}\\% & "
        f"{statistics.median(flat(all_high)):+.1f}\\% " + chr(92)*2,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    write("rq3_runtime.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- RQ5
def table_rq5_ablation():
    configs = [("Full system", "rq5_full_svn.csv"),
               ("$-$ value-number sharing", "rq5_no_vn_share.csv"),
               ("$-$ VN-based simplification", "rq5_no_vn_simplify.csv"),
               ("$-$ both", "rq5_no_vn_both.csv")]
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Minimality ablation (310 cases): removing the "
        r"value-numbering layer does not change the discharge outcome.}",
        r"\label{tab:rq5-ablation}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Configuration & Generated & Discharged & ADR\\",
        r"\midrule",
    ]
    for label, fname in configs:
        rows = [r for r in load(fname) if r["status"] == "ok"]
        g = sum(int(r["generated"]) for r in rows)
        d = sum(int(r["discharged"]) for r in rows)
        lines.append(f"{label} & {g:,} & {d:,} & {d/g*100:.2f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("rq5_ablation.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- RQ6
def table_rq6_mechanism():
    rows = [r for r in load("rq6_mechanism.csv") if r["status"] == "ok"]
    S = lambda c: sum(int(r[c]) for r in rows)
    dis = S("discharged")
    entries = [
        ("Canonical normalization", "constant-only", S("mech_canonical")),
        ("Interval over bounded types", "block/thread structure",
         S("mech_interval")),
        ("Direct static checks", "compiler-internal", S("direct_checks")),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{Discharge by mechanism and information dependence.}",
        r"\label{tab:rq6-mechanism}",
        r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
        r"Mechanism & Dependence & Discharged & Share\\",
        r"\midrule",
    ]
    for mech, dep, n in entries:
        lines.append(f"{mech} & {dep} & {n:,} & {n/dis*100:.1f}\\% \\\\")
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & & {dis:,} & 100\\% \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    write("rq6_mechanism.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- figure
def fig_mechanism_per_category():
    rows = [r for r in load("rq6_mechanism.csv") if r["status"] == "ok"]
    by_cat = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        e = by_cat[r["category"]]
        e[0] += int(r["mech_canonical"])
        e[1] += int(r["mech_interval"])
        e[2] += int(r["direct_checks"])
        e[3] += int(r["runtime"])
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    import numpy as np

    cats = sorted(by_cat)
    data = np.array([by_cat[c] for c in cats], dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    shares = data / totals * 100.0
    labels = ["canonical (const)", "interval (bounded types)",
              "direct checks", "runtime residue"]
    colors = ["#4c9bd4", "#e2824e", "#7fbf7f", "#c44e52"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = np.zeros(len(cats))
    for i in range(4):
        ax.barh(cats, shares[:, i], left=left, label=labels[i],
                color=colors[i])
        left += shares[:, i]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of obligations (%)")
    ax.invert_yaxis()
    ax.legend(ncol=2, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_mechanism_per_category.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    table_rq1_category()
    table_rq2_bugs()
    table_rq3_runtime()
    table_rq5_ablation()
    table_rq6_mechanism()
    fig_mechanism_per_category()
