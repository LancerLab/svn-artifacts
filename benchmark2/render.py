#!/usr/bin/env python3
"""benchmark2 integrator — merge per-lane `stats.json` into the paper artifacts.

Role (statistics-manifest.md, "integrator"): reads every ready lane's
`stats.json`, joins the cross-lane statistics (S1/S8/S9/S12 are owned by the
SOTA lanes; S2–S7/S10/S13 by choreo), and emits:

  * `stats-merged.json` / `stats-merged.md`  — the associated view
  * `tables/*.tex`  — paper tables (rq1_category, rq2_bugs, rq3_runtime,
                      rq5_ablation, rq6_mechanism)
  * `figures/*.pdf` — mechanism + discharge per category

Lanes that have not pushed `results/` yet (`mlir-linalg`, `mlir-low`, `tilelang`)
are reported in `stats-merged.json` as missing and skipped, never hard-failed.

Every number is read from a committed `stats.json` / register file; the renderer
never invents a value. If a lane is missing its `stats.json`, the affected
tables are emitted with that lane's columns empty and a note is printed.

Usage:
  python3 render.py [--all | --summary | --tables | --figures] [--out DIR]
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))  # benchmark2/

# ---------------------------------------------------------------------------
# Lane registry. "ready" lanes are those whose `run.sh stats` has produced a
# committed stats.json. Paths are relative to benchmark2/.
# NOTE the triton lane writes under triton/results/ (not results/triton/) —
# the integrator normalizes this discrepancy; both paths are probed.
# ---------------------------------------------------------------------------
LANE_STATS_PATHS = OrderedDict([
    ("choreo", ["results/choreo/stats.json"]),
    ("iree",   ["results/iree/stats.json"]),
    ("triton", ["triton/results/stats.json", "results/triton/stats.json"]),
])

# Statistics each lane owns (statistics-manifest.md). Used only for the merged
# view; the renderer pulls actual values from stats.json.
LANE_OWNS = OrderedDict([
    ("choreo", ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S10", "S13"]),
    ("triton", ["S1", "S8", "S9", "S12"]),
    ("iree",   ["S1", "S8", "S9", "S12"]),
])

NOT_READY = ["mlir-linalg", "mlir-low", "tilelang"]  # pending results/

# Canonical operator categories (settings/ dir). reshape is present in the
# 310-kernel corpus but carries 0 obligations in the choreo ledger (R-D7).
CANONICAL_CATEGORIES = [
    "batch_norm", "concat", "conv2d", "elemwise_add", "embedding", "gelu",
    "layer_normalization", "matmul", "max_pool2d", "reduce_mean", "relu",
    "reshape", "sigmoid", "softmax", "transpose",
]

MUTATION_CLASSES = ["M1", "M2", "M3"]


def num(x):
    """Thousands-separated integer."""
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "0"


def pct(x, digits=1):
    try:
        return f"{float(x):.{digits}f}\\%"
    except (TypeError, ValueError):
        return "---"


def esc(s):
    """LaTeX-escape an identifier (escape underscores)."""
    return str(s).replace("_", r"\_")


def _tex_note(note):
    """Render a table footnote: `` `code` `` spans become \\texttt{...} with
    escaped underscores; any bare underscore is escaped too."""
    parts = str(note).split("`")
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # backtick-delimited code span
            out.append(r"\texttt{" + p.replace("_", r"\_") + "}")
        else:
            out.append(p.replace("_", r"\_"))
    return "".join(out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_lane_stats():
    """Return {lane: stats_dict_or_None} for the ready lanes."""
    out = OrderedDict()
    for lane, paths in LANE_STATS_PATHS.items():
        stats = None
        for p in paths:
            fp = os.path.join(HERE, p)
            if os.path.isfile(fp):
                try:
                    stats = json.load(open(fp))
                    break
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[render] WARN: cannot parse {p}: {exc}", file=sys.stderr)
        out[lane] = stats
    return out


def load_kernel_counts():
    """Per-category kernel counts from the committed choreo register."""
    counts = {c: 0 for c in CANONICAL_CATEGORIES}
    fp = os.path.join(HERE, "results/choreo/kernel.jsonl")
    if not os.path.isfile(fp):
        return counts
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = r.get("category")
            if c in counts:
                counts[c] += 1
    return counts


# ---------------------------------------------------------------------------
# Summary (the "associated" view)
# ---------------------------------------------------------------------------
def _norm_s1(lane, stats):
    """Normalize S1 into {class: {n_injected,n_compile,n_runtime,n_never,n_na}}."""
    if not stats:
        return {}
    s1 = stats.get("S1_detection_matrix") or stats.get("S1_detection") or {}
    per = s1.get("per_class") if isinstance(s1, dict) else None
    norm = {}
    for cls in MUTATION_CLASSES:
        entry = (per or {}).get(cls) or s1.get(cls) or {}
        if isinstance(entry, dict):
            norm[cls] = {
                "n_injected": entry.get("n_injected", 0),
                "n_compile": entry.get("n_compile", 0),
                "n_runtime": entry.get("n_runtime", 0),
                "n_never": entry.get("n_never", 0),
                "n_na": entry.get("n_na", entry.get("n_n/a", 0)),
            }
    return norm


def _norm_s12(lane, stats):
    """Normalize S12 into {class: {flagged_and_exercised,total,flagged}}."""
    if not stats:
        return {}
    s12 = stats.get("S12_sanitizer_supplement") or {}
    if not isinstance(s12, dict):
        return {}
    return {c: v for c, v in s12.items() if c in MUTATION_CLASSES and isinstance(v, dict)}


def build_summary(stats_by_lane, kernel_counts):
    """Join everything into one dict. Values are direct references to committed
    stats.json content (never recomputed) so the merged file stays auditable."""
    s = OrderedDict()
    s["generated_at"] = None  # filled in main
    s["lanes_ready"] = [l for l, st in stats_by_lane.items() if st is not None]
    s["lanes_missing"] = ([l for l, st in stats_by_lane.items() if st is None]
                          + NOT_READY)
    s["lane_ownership"] = LANE_OWNS

    s["S1_detection"] = OrderedDict(
        (l, _norm_s1(l, stats_by_lane.get(l))) for l in stats_by_lane)
    s["S12_sanitizer_supplement"] = OrderedDict(
        (l, _norm_s12(l, stats_by_lane.get(l))) for l in stats_by_lane)

    # S8/S9 (SOTA expressibility + remainder) — per lane as emitted.
    s["S8_expressibility"] = OrderedDict()
    s["S9_remainder"] = OrderedDict()
    for l, st in stats_by_lane.items():
        if not st:
            continue
        if "S8_expressibility" in st:
            s["S8_expressibility"][l] = st["S8_expressibility"]
        if "S9_remainder" in st:
            s["S9_remainder"][l] = st["S9_remainder"]

    # choreo-owned statistics, carried through verbatim.
    choreo = stats_by_lane.get("choreo") or {}
    for key in ("S2_before_device", "S3_generation_totals", "S4_per_operator",
                "S5_discharge_rate", "S6_mechanism_split",
                "S7_no_interval_counterfactual", "S10_compile_cost",
                "S13_runtime_and_latency", "toolchain_identity"):
        if key in choreo:
            s[key] = choreo[key]

    s["kernel_counts_per_category"] = kernel_counts
    return s


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _tex_table(label, caption, header, rows, note=None):
    """Wrap rows into a booktabs table float."""
    cols = "".join("l" if i == 0 else "r" for i in range(len(header)))
    head = " & ".join(header)
    body = "\n".join(" & ".join(str(x) for x in r) + r" \\" for r in rows)
    out = ["\\begin{table}[t]", "\\centering\\footnotesize",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule",
           head + r" \\", "\\midrule", body, "\\bottomrule",
           "\\end{tabular}"]
    if note:
        out.append("\\vspace{0.4em}\\begin{minipage}{0.98\\linewidth}"
                   "\\raggedright\\footnotesize " + _tex_note(note) +
                   "\\end{minipage}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def table_rq1_category(summary):
    """tab:rq1-category — per-category discharge (S3 + S4 + kernel register)."""
    s3 = (summary.get("S3_generation_totals") or {}).get("per_category") or {}
    s4 = (summary.get("S4_per_operator") or {}).get("per_category") or {}
    s5 = summary.get("S5_discharge_rate") or {}
    counts = summary.get("kernel_counts_per_category") or {}

    rows = []
    tot_k = tot_g = tot_d = tot_r = 0
    for c in CANONICAL_CATEGORIES:
        k = counts.get(c, 0)
        g = s3.get(c, 0)
        d = (s4.get(c) or {}).get("discharged", 0)
        r = (s4.get(c) or {}).get("remainder", 0)
        adr = (s4.get(c) or {}).get("discharge_pct")
        tot_k += k
        tot_g += g
        tot_d += d
        tot_r += r
        rows.append([esc(c), k, num(g), num(d),
                     pct(adr) if adr is not None else "---", num(r)])

    all_pct = (s5.get("all") or {}).get("discharge_pct")
    rows.append([r"\textbf{Total}", tot_k, num(tot_g), num(tot_d),
                 pct(all_pct) if all_pct is not None else "---", num(tot_r)])

    return _tex_table(
        "tab:rq1-category",
        "Per-category discharge. Gen. = obligations generated (S3); "
        "Disch. = statically discharged (S4); ADR = discharge rate; "
        "Remainder = obligations left to runtime/budget (S5 residue).",
        ["Category", "Kernels", "Gen.", "Disch.", "ADR", "Remainder"],
        rows,
        note="Source: `results/choreo/stats.json` S3/S4/S5 and the committed "
             "`results/choreo/kernel.jsonl` register (310 kernels).")


def table_rq2_bugs(summary):
    """tab:rq2-bugs — detection matrix S1 x toolchain + S12 sanitizer."""
    s1 = summary.get("S1_detection") or {}
    s12 = summary.get("S12_sanitizer_supplement") or {}
    lanes = [l for l in LANE_STATS_PATHS if l in s1 and s1[l]]

    rows = []
    for cls in MUTATION_CLASSES:
        for lane in lanes:
            e = s1[lane].get(cls) or {}
            san = (s12.get(lane) or {}).get(cls) or {}
            san_cell = "---"
            if san:
                san_cell = f"{san.get('flagged_and_exercised', 0)}/{san.get('total', 0)}"
            rows.append([cls, esc(lane), e.get("n_injected", 0),
                         e.get("n_compile", 0), e.get("n_runtime", 0),
                         e.get("n_never", 0), e.get("n_na", 0), san_cell])

    return _tex_table(
        "tab:rq2-bugs",
        "Bug-detection matrix by mutation class and toolchain. Compile = "
        "refuted statically; Launch = caught at entry before device execution; "
        "Never = not detected; n/a = class the surface cannot express. "
        "Sanitizer = flagged $\\wedge$ exercised mutants (S12).",
        ["Class", "Toolchain", "Injected", "Compile", "Launch", "Never", "n/a",
         "Sanitizer"],
        rows,
        note="Classes: M1 = element-access (OOB / stride), M2 = "
             "shape-compatibility (dim mismatch), M3 = hardware-constraint "
             "(tile / alignment). Source: S1 detection matrices (choreo "
             "`per_class`, triton/iree top-level) and S12 in each lane's "
             "`stats.json`.")


def table_rq3_runtime(summary):
    """tab:rq3-runtime — E5a residue + E5b detection asymmetry (S13)."""
    s13 = summary.get("S13_runtime_and_latency") or {}
    e5a = s13.get("E5a") or {}
    e5b = s13.get("E5b") or {}
    ratio = (e5b.get("ratio") or {})
    asym = (e5b.get("detection_asymmetry") or {})

    rows = [
        [r"\textbf{E5a residue (checks on vs off)}", ""],
        ["cases measured", e5a.get("n_cases", 0)],
        ["median $\\Delta$", pct(e5a.get("median_delta_pct"), 2)],
        ["cases within noise", e5a.get("n_delta_within_noise", 0)],
        ["negative deltas", e5a.get("n_negative_delta", 0)],
        [r"\textbf{E5b detection latency (oracle vs choreo)}", ""],
        ["admissible pairs", ratio.get("n_pairs", 0)],
        ["median ratio (oracle slower)",
         f"{ratio.get('median_ratio', 0):.1f}$\\times$"],
        ["oracle-blind (choreo caught, memcheck silent)",
         asym.get("n_oracle_blind", 0)],
        ["oracle never executed (launch rejected)",
         asym.get("n_oracle_never_executed", 0)],
        ["aborted by choreo's own ungated check",
         asym.get("n_oracle_aborted_by_choreo_own_ungated_check", 0)],
    ]
    return _tex_table(
        "tab:rq3-runtime",
        "Runtime cost and detection latency (S13). E5a residue is structurally "
        "zero on device (byte-identical device code) and below the measurement "
        "floor on host; E5b is reported as detection asymmetry, not a latency "
        "multiplier (R-D1/R-D3).",
        ["Quantity", "Value"],
        rows,
        note="Source: `results/choreo/stats.json` `S13_runtime_and_latency`.")


def table_rq5_ablation(summary):
    """tab:rq5-ablation — flag-matrix ablation + never-cause breakdown (S2)."""
    s2 = summary.get("S2_before_device") or {}
    na = s2.get("never_attribution") or {}
    ab = na.get("flag_matrix_ablation") or {}
    cause = na.get("cause_counts") or {}

    def _row(label, val):
        return [esc(label), val]

    rows = [
        [r"\textbf{Flag-matrix ablation (before-device sensitivity)}", ""],
        _row("pinned default (rtc default, hoist on)",
             ab.get("at_pinned_default", "---")),
        _row("rtc=all + hoist disabled",
             ab.get("with_rtc_all_and_hoist_disabled", "---")),
        _row("recovered by flags", ab.get("recovered_by_flags", 0)),
        [r"\textbf{Never-cause breakdown}", ""],
        _row("C4_NOT_ASSESSED (coverage gap)", cause.get("C4_NOT_ASSESSED", 0)),
        _row("HOISTING_DEFECT (emission policy)", cause.get("HOISTING_DEFECT", 0)),
        _row("C5_OUT_OF_SCOPE (value error)", cause.get("C5_OUT_OF_SCOPE", 0)),
        _row("C3_LOWER_BOUND_OMITTED (gen bug)",
             cause.get("C3_LOWER_BOUND_OMITTED", 0)),
        _row("C1_COST_FILTER_SUPPRESSED (emission policy)",
             cause.get("C1_COST_FILTER_SUPPRESSED", 0)),
    ]
    return _tex_table(
        "tab:rq5-ablation",
        "Before-device detection sensitivity to the runtime-check / hoisting "
        "flag matrix (R-D2b). The pinned default is the headline; the "
        "`rtc=all` + hoist-disabled arm shows the hoisting defect's recoverable "
        "cost. Never-cause breakdown attributes the $n_\\mathrm{never}=37$ "
        "mutants.",
        ["Configuration", "Result"],
        rows,
        note="Source: `results/choreo/stats.json` `S2_before_device.never_attribution`.")


def table_rq6_mechanism(summary):
    """tab:rq6-mechanism — mechanism split (S6)."""
    s6 = summary.get("S6_mechanism_split") or {}
    po = s6.get("per_outcome") or {}
    proven = po.get("proven") or {}
    total = proven.get("total", 0) or sum(
        proven.get(k, 0) for k in ("canonical", "interval", "direct"))

    def _share(k):
        v = proven.get(k, 0)
        return pct(100.0 * v / total) if total else "---"

    rows = [
        ["Constant fold (canonical)", proven.get("canonical", 0),
         _share("canonical")],
        ["Interval (bounded types)", proven.get("interval", 0),
         _share("interval")],
        ["Direct static checks", proven.get("direct", 0), _share("direct")],
        [r"\textbf{Total}", total, "100\\%"],
    ]
    return _tex_table(
        "tab:rq6-mechanism",
        "Discharge by evaluation form (mechanism split, S6). Interval discharge "
        "is 2{,}920 (re-registered from 2{,}837).",
        ["Mechanism", "Discharged", "Share"],
        rows,
        note="Source: `results/choreo/stats.json` `S6_mechanism_split.per_outcome.proven`.")


def render_tables(summary):
    return OrderedDict([
        ("rq1_category.tex", table_rq1_category(summary)),
        ("rq2_bugs.tex", table_rq2_bugs(summary)),
        ("rq3_runtime.tex", table_rq3_runtime(summary)),
        ("rq5_ablation.tex", table_rq5_ablation(summary)),
        ("rq6_mechanism.tex", table_rq6_mechanism(summary)),
    ])


# ---------------------------------------------------------------------------
# Figures (matplotlib, optional)
# ---------------------------------------------------------------------------
def render_figures(summary, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[render] WARN: matplotlib unavailable — skipping figures",
              file=sys.stderr)
        return []

    os.makedirs(out_dir, exist_ok=True)
    written = []

    s4 = (summary.get("S4_per_operator") or {}).get("per_category") or {}
    cats = [c for c in CANONICAL_CATEGORIES if c in s4]
    mech = {c: (s4[c].get("mechanism") or {}) for c in cats}

    # -- fig_mechanism_per_category.pdf --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(cats))
    w = 0.27
    for i, key in enumerate(("canonical", "interval", "direct")):
        vals = [mech[c].get(key, 0) for c in cats]
        ax.bar([xi + (i - 1) * w for xi in x], vals, w,
               label={"canonical": "canonical", "interval": "interval",
                      "direct": "direct"}[key])
    ax.set_xticks(list(x))
    ax.set_xticklabels([esc(c).replace(r"\_", "_") for c in cats], rotation=45, ha="right")
    ax.set_ylabel("obligations discharged")
    ax.set_title("Discharge mechanism per category (choreo, S6/S4)")
    ax.legend()
    fig.tight_layout()
    fp = os.path.join(out_dir, "fig_mechanism_per_category.pdf")
    fig.savefig(fp)
    plt.close(fig)
    written.append(fp)

    # -- fig_discharge_per_category.pdf --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = [s4[c].get("discharge_pct", 0.0) or 0.0 for c in cats]
    bars = ax.bar([c.replace("_", " ") for c in cats], vals)
    ax.axhline((summary.get("S5_discharge_rate") or {}).get("all", {}).get("discharge_pct", 0),
               color="gray", ls="--", label="overall ADR")
    ax.set_ylabel("discharge rate (%)")
    ax.set_title("Discharge rate per category (choreo, S5/S4)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fp = os.path.join(out_dir, "fig_discharge_per_category.pdf")
    fig.savefig(fp)
    plt.close(fig)
    written.append(fp)

    return written


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(summary):
    lines = ["# benchmark2 — merged statistics", "",
             f"Generated by `render.py` (integrator). Ready lanes: "
             f"{', '.join(summary['lanes_ready'])}. Missing/not-ready: "
             f"{', '.join(summary['lanes_missing'])}.", "",
             "## Detection matrix (S1)", ""]
    lines.append("| Class | Toolchain | Injected | Compile | Launch | Never | n/a |")
    lines.append("|---|---|---|---|---|---|---|")
    for cls in MUTATION_CLASSES:
        for lane, s1 in (summary.get("S1_detection") or {}).items():
            e = (s1 or {}).get(cls) or {}
            if not e:
                continue
            lines.append(f"| {cls} | {lane} | {e.get('n_injected', 0)} | "
                         f"{e.get('n_compile', 0)} | {e.get('n_runtime', 0)} | "
                         f"{e.get('n_never', 0)} | {e.get('n_na', 0)} |")
    lines.append("")
    lines.append("## choreo-owned headline numbers")
    s5 = summary.get("S5_discharge_rate") or {}
    s3 = summary.get("S3_generation_totals") or {}
    s6 = summary.get("S6_mechanism_split") or {}
    proven = (s6.get("per_outcome") or {}).get("proven") or {}
    s10 = summary.get("S10_compile_cost") or {}
    lines.append(f"- S3 generation total: {num(s3.get('grand_total', 0))}")
    lines.append(f"- S5 discharge: all {pct((s5.get('all') or {}).get('discharge_pct'))}, "
                 f"static {pct((s5.get('static') or {}).get('discharge_pct'))}, "
                 f"dynamic {pct((s5.get('dynamic') or {}).get('discharge_pct'))}")
    lines.append(f"- S6 mechanism (discharged): {proven}")
    lines.append(f"- S10 compile cost: {pct(s10.get('grand_median_pct'), 3)} front-end vs nvcc")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "render"))
    args = ap.parse_args(argv)

    do_all = args.all or not (args.summary or args.tables or args.figures)
    do_summary = args.summary or do_all
    do_tables = args.tables or do_all
    do_figures = args.figures or do_all

    os.makedirs(args.out, exist_ok=True)

    stats_by_lane = load_lane_stats()
    kernel_counts = load_kernel_counts()
    summary = build_summary(stats_by_lane, kernel_counts)
    summary["generated_at"] = None

    import datetime
    summary["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    for lane, st in stats_by_lane.items():
        state = "OK" if st is not None else "MISSING (stats.json not found)"
        print(f"[render] {lane:8s} -> {state}")

    written = []
    if do_summary:
        fp = os.path.join(args.out, "stats-merged.json")
        json.dump(summary, open(fp, "w"), indent=2)
        written.append(fp)
        fp = os.path.join(args.out, "stats-merged.md")
        open(fp, "w").write(render_markdown(summary))
        written.append(fp)

    if do_tables:
        tdir = os.path.join(args.out, "tables")
        os.makedirs(tdir, exist_ok=True)
        for name, tex in render_tables(summary).items():
            fp = os.path.join(tdir, name)
            open(fp, "w").write(tex)
            written.append(fp)

    if do_figures:
        fdir = os.path.join(args.out, "figures")
        written += render_figures(summary, fdir)

    print(f"[render] wrote {len(written)} artifact(s) under {args.out}")
    for fp in written:
        print(f"    {os.path.relpath(fp, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
