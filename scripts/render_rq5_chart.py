#!/usr/bin/env python3
"""
Generate bar chart data for RQ5: Runtime Assertion Cost across 4 levels.
Levels: none → entry → all(hoisted) → all(no-hoist)

Uses data from:
  - choreo_rq4_runtime.csv: none, entry, all
  - runtime_hoist_comparison.csv: none, hoist, nohoist

Outputs:
  - LaTeX pgfplots bar chart code for manuscript
  - Summary statistics
"""

import csv
import os
import statistics
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS_DIR", "benchmark/results"))


def load_csv(name):
    path = RESULTS / name
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    hoist_rows = load_csv("runtime_hoist_comparison.csv")
    rq4_rows = load_csv("choreo_rq4_runtime.csv")

    # Build merged data: for cases in hoist_comparison, find entry time from rq4
    hoist_by_case = {}
    for r in hoist_rows:
        case = r['case']
        try:
            hoist_by_case[case] = {
                'category': r['category'],
                'none': float(r['time_none_us']),
                'hoist': float(r['time_hoist_us']),
                'nohoist': float(r['time_nohoist_us']),
            }
        except (ValueError, KeyError):
            pass

    rq4_by_case = {}
    for r in rq4_rows:
        case = r['case_name']
        try:
            rq4_by_case[case] = {
                'none': float(r['rtc_none_us']),
                'entry': float(r['rtc_entry_us']),
                'all': float(r['rtc_all_us']),
            }
        except (ValueError, KeyError):
            pass

    # Merge: use hoist data's none/hoist/nohoist, rq4 data's entry
    # Normalize entry relative to hoist data's none baseline
    merged = []
    for case, hd in hoist_by_case.items():
        if case in rq4_by_case:
            rd = rq4_by_case[case]
            # Normalize: entry_overhead = (entry - none_rq4) / none_rq4
            # Then project onto hoist_none: entry_projected = hoist_none * (1 + entry_overhead)
            if rd['none'] > 0:
                entry_overhead = (rd['entry'] - rd['none']) / rd['none']
            else:
                entry_overhead = 0
            entry_projected = hd['none'] * (1 + entry_overhead)

            merged.append({
                'case': case,
                'category': hd['category'],
                'none': hd['none'],
                'entry': entry_projected,
                'hoist': hd['hoist'],
                'nohoist': hd['nohoist'],
            })

    if not merged:
        print("ERROR: No merged data found")
        return

    # Calculate overheads relative to 'none'
    entry_ohs = []
    hoist_ohs = []
    nohoist_ohs = []

    for m in merged:
        if m['none'] > 0:
            entry_ohs.append(100 * (m['entry'] - m['none']) / m['none'])
            hoist_ohs.append(100 * (m['hoist'] - m['none']) / m['none'])
            nohoist_ohs.append(100 * (m['nohoist'] - m['none']) / m['none'])

    # Per-category averages
    cats = {}
    for m in merged:
        cat = m['category']
        if cat not in cats:
            cats[cat] = {'entry': [], 'hoist': [], 'nohoist': []}
        if m['none'] > 0:
            cats[cat]['entry'].append(100 * (m['entry'] - m['none']) / m['none'])
            cats[cat]['hoist'].append(100 * (m['hoist'] - m['none']) / m['none'])
            cats[cat]['nohoist'].append(100 * (m['nohoist'] - m['none']) / m['none'])

    print("% === RQ5 Bar Chart: Runtime Assertion Cost ===")
    print(f"% Data from {len(merged)} benchmark cases")
    print(f"% Entry overhead: avg {statistics.mean(entry_ohs):.1f}%, max {max(entry_ohs):.1f}%")
    print(f"% Hoist overhead: avg {statistics.mean(hoist_ohs):.1f}%, max {max(hoist_ohs):.1f}%")
    print(f"% NoHoist overhead: avg {statistics.mean(nohoist_ohs):.1f}%, max {max(nohoist_ohs):.1f}%")
    print()

    # Output pgfplots bar chart
    print(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=\columnwidth,
    height=4.5cm,
    ybar,
    bar width=4pt,
    ylabel={Overhead (\%)},
    symbolic x coords={""", end="")
    cat_names = sorted(cats.keys())
    print(",".join(cat_names), end="")
    print(r"""},
    xtick=data,
    x tick label style={rotate=30, anchor=east, font=\scriptsize},
    ylabel style={font=\small},
    ymin=-5,
    legend style={at={(0.5,1.05)}, anchor=south, legend columns=3, font=\scriptsize},
    enlarge x limits=0.15,
    every axis plot/.append style={fill opacity=0.8},
]""")

    # Entry bars
    print(r"\addplot[fill=blue!30, draw=blue!60] coordinates {", end="")
    for cat in cat_names:
        avg = statistics.mean(cats[cat]['entry']) if cats[cat]['entry'] else 0
        cat_esc = cat.replace('_', r'\_')
        print(f"({cat},{avg:.1f})", end=" ")
    print(r"};")

    # Hoist bars
    print(r"\addplot[fill=green!40, draw=green!60] coordinates {", end="")
    for cat in cat_names:
        avg = statistics.mean(cats[cat]['hoist']) if cats[cat]['hoist'] else 0
        cat_esc = cat.replace('_', r'\_')
        print(f"({cat},{avg:.1f})", end=" ")
    print(r"};")

    # NoHoist bars
    print(r"\addplot[fill=red!40, draw=red!60] coordinates {", end="")
    for cat in cat_names:
        avg = statistics.mean(cats[cat]['nohoist']) if cats[cat]['nohoist'] else 0
        cat_esc = cat.replace('_', r'\_')
        print(f"({cat},{avg:.1f})", end=" ")
    print(r"};")

    print(r"""
\legend{Entry, All (hoisted), All (no-hoist)}
\end{axis}
\end{tikzpicture}
\caption{Runtime assertion overhead by category. ``Entry'' = entry-point-only checks; ``All (hoisted)'' = full assertions with hoisting (default); ``All (no-hoist)'' = without hoisting. Hoisting reduces worst-case overhead from $87\%$ to $6\%$.}
\label{fig:rtcost}
\end{figure}""")

    print()
    print("% === Summary Table ===")
    print(f"% {'Level':<20} {'Avg %':>8} {'Max %':>8} {'Median %':>10}")
    print(f"% {'Entry':<20} {statistics.mean(entry_ohs):>8.1f} {max(entry_ohs):>8.1f} {statistics.median(entry_ohs):>10.1f}")
    print(f"% {'All (hoisted)':<20} {statistics.mean(hoist_ohs):>8.1f} {max(hoist_ohs):>8.1f} {statistics.median(hoist_ohs):>10.1f}")
    print(f"% {'All (no-hoist)':<20} {statistics.mean(nohoist_ohs):>8.1f} {max(nohoist_ohs):>8.1f} {statistics.median(nohoist_ohs):>10.1f}")


if __name__ == "__main__":
    main()
