#!/usr/bin/env python3
"""scripts/render_e1_artifacts.py

Render the E1 (detection-completeness) artifacts for the ASPLOS 2027 paper
("The Compiler as Auditor") from the authoritative benchmark2 lane stats:

  1. tables/rq2_bugs.tex   -- E1 cross-pipeline detection matrix (Table).
  2. figures/fig_e1_detection.tex -- E1 grouped bar chart (pgfplots).

Data source (R-D5/R-D6; do NOT re-run the benchmark, see
benchmark2/manifest.md):
  benchmark2/results/<lane>/stats.json   S1_detection{,_matrix}, per class:
      {n_injected, n_compile, n_runtime, n_never, n_na}
  benchmark2/triton/results/stats.json   (triton lane lives one level up)

The system-name macro is ``\\sys`` (== \\textsc{KernAudit}); the table and the
figure legend use ``\\sys`` so the rendered name always matches the paper.
The literal string "choreo" never appears in the output.

Usage: python3 scripts/render_e1_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # svn/svn-artifacts
B2 = ROOT / "benchmark2"
PAPER = ROOT.parent / "asplos27"                       # svn/asplos27
TABDIR = PAPER / "tables"
FIGDIR = PAPER / "figures"

CLASSES = ["M1", "M2", "M3"]

# lane id -> (display label in LaTeX, path to stats.json, S1 key)
LANES = {
    "choreo": (r"\sys", B2 / "results/choreo/stats.json", "S1_detection_matrix"),
    "triton": ("Triton", B2 / "triton/results/stats.json", "S1_detection_matrix"),
    "mlir-linalg": ("MLIR-linalg", B2 / "results/mlir-linalg/stats.json",
                    "S1_detection"),
    "mlir-low": ("MLIR-low", B2 / "results/mlir-low/stats.json", "S1_detection"),
    "iree": ("IREE", B2 / "results/iree/stats.json", "S1_detection"),
}

# Row order within each class block of the table (our system first, bold).
TABLE_ROW_ORDER = {
    "M1": ["choreo", "triton", "mlir-low", "iree"],
    "M2": ["choreo", "mlir-linalg", "iree", "triton"],
    "M3": ["choreo", "triton", "iree"],
}


def load_matrix(lane):
    """Return {class: dict(n_injected, n_compile, n_runtime, n_never, n_na)}."""
    label, path, key = LANES[lane]
    d = json.load(open(path))
    s1 = d[key]
    # choreo's matrix nests under "per_class"
    if "per_class" in s1:
        s1 = s1["per_class"]
    out = {}
    for cls in CLASSES:
        e = s1.get(cls, {})
        out[cls] = {
            "n_injected": int(e.get("n_injected", 0)),
            "n_compile": int(e.get("n_compile", 0)),
            "n_runtime": int(e.get("n_runtime", 0)),
            "n_never": int(e.get("n_never", 0)),
            # different lanes spell the n/a key differently
            "n_na": int(e.get("n_na", e.get("n_n/a", 0))),
        }
    return out


# ------------------------------------------------------------------ table
def render_table():
    mats = {lane: load_matrix(lane) for lane in LANES}

    def row(cls, lane):
        label = LANES[lane][0]
        if lane == "choreo":
            label = r"\textbf{" + label + "}"
        m = mats[lane][cls]
        return (f"{cls} & {label} & {m['n_injected']} & {m['n_compile']} & "
                f"{m['n_runtime']} & {m['n_never']} & {m['n_na']} \\\\")

    lines = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\caption{E1 bug-detection matrix by mutation class and toolchain. "
        r"Injected = oracle-confirmed corruptions; Compile = refuted "
        r"statically; Launch = caught at entry before device execution; "
        r"Never = not detected; n/a = class the surface cannot express. "
        r"Before-device detection is Compile $+$ Launch.}",
        r"\label{tab:rq2-bugs}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Class & Toolchain & Injected & Compile & Launch & Never & n/a\\",
        r"\midrule",
    ]
    for i, cls in enumerate(CLASSES):
        for lane in TABLE_ROW_ORDER[cls]:
            lines.append(row(cls, lane))
        if i < len(CLASSES) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.4em}\begin{minipage}{0.98\linewidth}\raggedright"
        r"\footnotesize Classes: M1 = element-access (OOB / stride), "
        r"M2 = shape-compatibility (dim mismatch), M3 = hardware-constraint "
        r"(tile / alignment). n/a marks a class the surface cannot express "
        r"(Triton has no cross-tensor contract; IREE does not lower the "
        r"class; MLIR lanes expose only one dialect's worth of constructs). "
        r"Source: S1 detection matrices in each lane's \texttt{stats.json}."
        r"\end{minipage}",
        r"\end{table}",
    ]
    TABDIR.mkdir(parents=True, exist_ok=True)
    out = TABDIR / "rq2_bugs.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


# ------------------------------------------------------------------ figure
def render_figure():
    """Grouped bar chart: per system, per class, before-device detection.

    For each (system, class) a bar shows the before-device detection rate
    (Compile + Launch)/Injected in percent; systems that cannot express a
    class are omitted from that group. This is the visual form of Table
    rq2-bugs.
    """
    mats = {lane: load_matrix(lane) for lane in LANES}
    # legend / bar order: our system first
    order = ["choreo", "triton", "mlir-linalg", "mlir-low", "iree"]
    labels = {l: LANES[l][0] for l in order}

    tex = []
    # pgfplots grouped bars: x = class, one bar per system, height = share of
    # injected mutants detected before device (Compile+Launch)/Injected in %.
    # Normalizing to a percentage makes the systems directly comparable even
    # though their injected counts differ (e.g. MLIR-linalg M2 injects 50 vs
    # our 30). Systems that cannot express a class are omitted from that group.
    # Okabe-Ito colorblind-safe palette (ASPLOS CFP requires it). Define the
    # named colors OUTSIDE figure/tikzpicture: \definecolor is illegal inside
    # the pgfplots axis environment.
    tex.append(r"\definecolor{e1blue}{RGB}{0,114,178}")
    tex.append(r"\definecolor{e1orange}{RGB}{230,159,0}")
    tex.append(r"\definecolor{e1green}{RGB}{0,158,115}")
    tex.append(r"\definecolor{e1vermil}{RGB}{213,94,0}")
    tex.append(r"\begin{figure}[t]")
    tex.append(r"\centering")
    tex.append(r"\begin{tikzpicture}")
    tex.append(r"\begin{axis}[")
    tex.append(r"  ybar,")
    tex.append(r"  bar width=11pt,")
    tex.append(r"  width=\columnwidth, height=5.2cm,")
    tex.append(r"  ymin=0, ymax=118,")
    tex.append(r"  ytick={0,25,50,75,100},")
    tex.append(r"  xlabel={mutation class},")
    tex.append(r"  ylabel={before-device detection (\%)},")
    tex.append(r"  symbolic x coords={M1,M2,M3},")
    tex.append(r"  xtick=data,")
    tex.append(r"  legend style={at={(0.5,1.02)},anchor=south,")
    tex.append(r"    legend columns=5,font=\scriptsize,draw=none},")
    tex.append(r"  ymajorgrids=true, grid style={gray!20},")
    tex.append(r"  % raw detected/injected count above each bar: keeps small-n"
               r" cells honest (e.g. 2/2 reads 100\% but is tiny).")
    tex.append(r"  point meta=explicit symbolic,")
    tex.append(r"  nodes near coords," )
    tex.append(r"  every node near coord/.append style={font=\tiny,"
               r"anchor=south},")
    tex.append(r"]")
    colors = {
        "choreo": "black",
        "triton": "e1blue",
        "mlir-linalg": "e1orange",
        "mlir-low": "e1green",
        "iree": "e1vermil",
    }
    for lane in order:
        coords = []
        for cls in CLASSES:
            m = mats[lane][cls]
            if m["n_injected"] > 0:  # omit classes the surface cannot express
                bd = m["n_compile"] + m["n_runtime"]
                pct = 100.0 * bd / m["n_injected"]
                # (class, pct) [detected/injected label]
                coords.append(f"({cls},{pct:.1f}) [{bd}/{m['n_injected']}]")
        if not coords:
            continue
        tex.append(r"\addplot+[ybar,fill=" + colors[lane] +
                   ",draw=none] coordinates {" + " ".join(coords) + "};")
        tex.append(r"\addlegendentry{" + labels[lane] + "}")
    tex.append(r"\end{axis}")
    tex.append(r"\end{tikzpicture}")
    tex.append(r"\caption{E1 before-device detection rate (Compile $+$ Launch, "
               r"as a share of each toolchain's injected mutants) by mutation "
               r"class and toolchain; the raw detected/injected count labels "
               r"each bar. A bar appears only where the toolchain's surface "
               r"can express the class; \sys is the only pipeline that "
               r"detects in all three.}")
    tex.append(r"\label{fig:e1-detection}")
    tex.append(r"\end{figure}")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "fig_e1_detection.tex"
    out.write_text("\n".join(tex) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    render_table()
    render_figure()
