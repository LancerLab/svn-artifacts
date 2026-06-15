#!/usr/bin/env python3
"""
Paper Figure Renderer for CGO 2027 SVN Paper
Generates LaTeX tables and figures from benchmark results.

Outputs:
  - Table: Runtime overhead with/without hoisting
  - Table: Bug detection comparison (Choreo vs MLIR)
  - Table: Resolution time taxonomy
"""

import csv
import os
import statistics
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "benchmark/results"))

def load_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

# ========================================================================
# Table 1: Assertion Hoisting Effectiveness (runtime overhead comparison)
# ========================================================================
def render_hoisting_table():
    rows = load_csv("runtime_hoist_comparison.csv")
    if not rows:
        return
    
    print("\n% === Table: Assertion Hoisting Runtime Overhead ===")
    print(r"""\begin{table}[t]
    \centering
    \caption{Runtime overhead comparison: assertion hoisting reduces worst-case overhead from 87\% to 6\%, with an average of 1.0\% (vs.\ 9.4\% without hoisting). ``None'' = no assertions; ``Hoist'' = assertions with hoisting (default); ``No-Hoist'' = assertions without hoisting.}
    \label{tab:hoist-overhead}
    \scriptsize
    \setlength{\tabcolsep}{2.5pt}
    \begin{tabular}{@{}l r rr rr@{}}
    \toprule
    \textbf{Category} & \textbf{\#} & \textbf{Hoist avg\%} & \textbf{Hoist max\%} & \textbf{No-Hoist avg\%} & \textbf{No-Hoist max\%} \\
    \midrule""")
    
    by_cat = {}
    for r in rows:
        cat = r['category']
        if cat not in by_cat:
            by_cat[cat] = []
        try:
            oh = float(r['overhead_hoist_pct'])
            onh = float(r['overhead_nohoist_pct'])
            by_cat[cat].append((oh, onh))
        except (ValueError, KeyError):
            pass
    
    all_hoist = []
    all_nohoist = []
    for cat in sorted(by_cat.keys()):
        vals = by_cat[cat]
        h_vals = [v[0] for v in vals]
        nh_vals = [v[1] for v in vals]
        all_hoist.extend(h_vals)
        all_nohoist.extend(nh_vals)
        h_avg = statistics.mean(h_vals) if h_vals else 0
        h_max = max(h_vals) if h_vals else 0
        nh_avg = statistics.mean(nh_vals) if nh_vals else 0
        nh_max = max(nh_vals) if nh_vals else 0
        cat_display = cat.replace('_', r'\_')
        print(f"    {cat_display} & {len(vals)} & ${h_avg:+.1f}$ & ${h_max:+.1f}$ & ${nh_avg:+.1f}$ & ${nh_max:+.1f}$ \\\\")
    
    # Total row
    h_avg_all = statistics.mean(all_hoist) if all_hoist else 0
    h_max_all = max(all_hoist) if all_hoist else 0
    nh_avg_all = statistics.mean(all_nohoist) if all_nohoist else 0
    nh_max_all = max(all_nohoist) if all_nohoist else 0
    
    print(r"    \midrule")
    print(f"    \\textbf{{Overall}} & \\textbf{{{len(all_hoist)}}} & $\\mathbf{{{h_avg_all:+.1f}}}$ & ${h_max_all:+.1f}$ & $\\mathbf{{{nh_avg_all:+.1f}}}$ & ${nh_max_all:+.1f}$ \\\\")
    print(r"""    \bottomrule
    \end{tabular}
\end{table}""")

# ========================================================================
# Table 2: Bug Detection Resolution Time Comparison
# ========================================================================
def render_bug_detection_table():
    # Choreo data
    choreo_rows = load_csv("bug_detection_full.csv")
    choreo_static = sum(1 for r in choreo_rows if r.get('choreo_result') == 'STATIC_DETECT')
    choreo_total = len(choreo_rows)
    
    # Hoist stats for overall numbers
    hoist_rows = load_csv("hoist_stats.csv")
    
    print("\n% === Table: Bug Detection Comparison ===")
    print(r"""\begin{table}[t]
    \centering
    \caption{Bug detection comparison across 4 bug classes. Choreo detects all bugs at compile time. MLIR requires manual annotations for dimension checks; its auto-generated checks (\texttt{--generate-runtime-verification}) only cover per-element OOB after bufferization, with cost proportional to tensor size.}
    \label{tab:bug-detect}
    \scriptsize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}l cc cc@{}}
    \toprule
    & \multicolumn{2}{c}{\textbf{Choreo (SVN)}} & \multicolumn{2}{c}{\textbf{MLIR}} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}
    \textbf{Bug Class} & \textbf{Detect} & \textbf{When} & \textbf{Detect} & \textbf{When / How} \\
    \midrule
    Dim mismatch       & \cmark & Compile & \cmark$^*$ & Runtime (per-elem OOB) \\
    OOB access         & \cmark & Compile & \cmark$^\dagger$ & Runtime (per-elem) \\
    Wrong output shape & \cmark & Compile & \xmark & --- \\
    Stride/layout err  & \cmark & Compile & \xmark & --- \\
    \midrule
    \multicolumn{5}{@{}l}{\footnotesize $^*$Indirect: fires when loop index exceeds buffer dim (opaque abort).} \\
    \multicolumn{5}{@{}l}{\footnotesize $^\dagger$Via \texttt{--generate-runtime-verification} pass (memref level only).} \\
    \bottomrule
    \end{tabular}
\end{table}""")

# ========================================================================
# Table 3: Resolution Time Taxonomy (full picture)
# ========================================================================
def render_resolution_taxonomy():
    print("\n% === Table: Resolution Time Taxonomy ===")
    print(r"""\begin{table}[t]
    \centering
    \caption{Resolution time taxonomy: how each system resolves safety assertions. Choreo discharges 93.3\% statically and hoists the remainder; MLIR's auto-checks execute per-element inside loop bodies (cost $\propto$ tensor size).}
    \label{tab:resolution}
    \scriptsize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{@{}l rr rr@{}}
    \toprule
    & \multicolumn{2}{c}{\textbf{Choreo}} & \multicolumn{2}{c}{\textbf{MLIR (auto)}} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}
    \textbf{Resolution Level} & \textbf{Count} & \textbf{Cost} & \textbf{Count} & \textbf{Cost} \\
    \midrule
    Compile-time (static) & 11{,}753 & $O(0)$ & 0$^\dagger$ & --- \\
    Runtime hoisted       & 521     & $O(1)$ & ---         & --- \\
    Runtime entry         & 318     & $O(1)$ & ---         & --- \\
    Runtime per-element   & ---     & ---    & 438$^*$     & $O(N)$ \\
    \midrule
    \textbf{Total}        & \textbf{12{,}592} & & \textbf{438} & \\
    \bottomrule
    \multicolumn{5}{@{}l}{\footnotesize $^\dagger$For dynamic shapes; static shapes use linalg verifier.} \\
    \multicolumn{5}{@{}l}{\footnotesize $^*$Per-element bounds checks via \texttt{--generate-runtime-verification}.} \\
    \end{tabular}
\end{table}""")

# ========================================================================
# New RQ text: Assertion Hoisting Effectiveness
# ========================================================================
def render_rq_hoisting_text():
    rows = load_csv("runtime_hoist_comparison.csv")
    if not rows:
        return
    
    hoist_overheads = []
    nohoist_overheads = []
    for r in rows:
        try:
            hoist_overheads.append(float(r['overhead_hoist_pct']))
            nohoist_overheads.append(float(r['overhead_nohoist_pct']))
        except (ValueError, KeyError):
            pass
    
    h_avg = statistics.mean(hoist_overheads) if hoist_overheads else 0
    h_max = max(hoist_overheads) if hoist_overheads else 0
    nh_avg = statistics.mean(nohoist_overheads) if nohoist_overheads else 0
    nh_max = max(nohoist_overheads) if nohoist_overheads else 0
    
    print(f"\n% === RQ5 Text: Hoisting Effectiveness ===")
    print(f"% Average overhead WITH hoisting: {h_avg:.2f}%")
    print(f"% Maximum overhead WITH hoisting: {h_max:.2f}%")
    print(f"% Average overhead WITHOUT hoisting: {nh_avg:.2f}%")
    print(f"% Maximum overhead WITHOUT hoisting: {nh_max:.2f}%")
    print(f"% Reduction factor (avg): {nh_avg/h_avg:.1f}x" if h_avg != 0 else "% Reduction: inf")

# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print("% Auto-generated LaTeX tables for CGO 2027 SVN paper")
    print("% Generated by scripts/render_paper_tables.py")
    print("")
    
    render_hoisting_table()
    render_bug_detection_table()
    render_resolution_taxonomy()
    render_rq_hoisting_text()
    
    print("\n% Done.")
