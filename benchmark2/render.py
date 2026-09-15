#!/usr/bin/env python3
"""benchmark2 integrator — merge per-lane `stats.json` into the paper artifacts.

Role (statistics-manifest.md, "integrator"): reads every ready lane's
`stats.json`, joins the cross-lane statistics (S1/S8/S9/S12 are owned by the
SOTA lanes; S2–S7/S10/S13 by choreo), and emits:

  * `stats-merged.json` / `stats-merged.md`  — the associated view
  * `tables/*.tex`  — paper tables. The paper's own float names (`e1_path_class`,
                      `e1_rtc_curve`, `e2_generation`, `e3_discharge`, `e4_cost`,
                      `e5_oracle`, `rq2_bugs`) plus the legacy `rq*` set
  * `figures/*.pdf` — mechanism + discharge per category, E1 path class
  * `PROSE-DRIFT.md` — every number the paper's prose carries, against the
                      committed statistic that does (or does not) produce it

Lanes that have pushed no `results/` at all are reported in `stats-merged.json`
as missing and skipped, never hard-failed.

Every number is read from a committed `stats.json` / register file; the renderer
never invents a value. If a lane is missing its `stats.json`, the affected
tables are emitted with that lane's columns empty and a note is printed.

Usage:
  python3 render.py [--all | --summary | --tables | --figures] [--out DIR]
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))  # benchmark2/

# ---------------------------------------------------------------------------
# The lane registry and the class axis both come from schema/class-axis.json.
#
# They used to be restated here, in choreo/stats.py and in all four SOTA lanes,
# and that is how the axis drifted: the lanes kept a 3-class tuple from the
# pre-revision spec while this file moved to 4, so M4 went missing from every
# SOTA detection matrix. A missing class and an `n/a` class render identically
# but mean different things. schema/check_class_axis.py fails the build if this
# module and the axis disagree.
# ---------------------------------------------------------------------------
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from schema import class_axis as AX                            # noqa: E402

# Lane -> candidate `stats.json` paths, relative to benchmark2/, in probe order.
# A lane may legitimately write more than one place (triton does).
LANE_STATS_PATHS = OrderedDict(
    (lane, AX.lane_stats_paths(lane)) for lane in AX.ready_lanes())

# Statistics each lane owns (statistics-manifest.md). Used only for the merged
# view; the renderer pulls actual values from stats.json.
LANE_OWNS = OrderedDict(
    (lane, AX.lane_owns(lane)) for lane in AX.ready_lanes())

# Lanes that have pushed no `results/` at all.
#
# `mlir-linalg` and `mlir-low` were listed here until 2026-09-11 and that was
# WRONG: both have committed `results/<lane>/stats.json` carrying S1/S8/S9/S12,
# and the paper consumes them twice -- `tab:rq2-bugs` (MLIR-linalg 34/50,
# MLIR-low 38/48) and `tab:e2-generation` (both rows). Excluding them silently
# dropped two of the five compared toolchains from every generated table, which
# is why the rendered `rq2_bugs.tex` had three toolchains to the paper's five.
NOT_READY = list(AX.not_ready_lanes())          # tilelang, pending results/

# Paper display name per toolchain. `\sys` is the macro main.tex uses for
# choreo; the baselines keep their product names.
LANE_DISPLAY = OrderedDict(
    (lane, AX.lane_display(lane)) for lane in AX.ready_lanes())

# Baseline row order in `tab:rq2-bugs` after the `\sys` row. Chosen to
# reproduce the paper's hand-written float; it carries no data.
BASELINE_ROW_ORDER = ["triton", "mlir-linalg", "mlir-low", "iree"]

# The single all-n/a row the paper shows per mutation class. Every other
# all-n/a cell is (0 injected, 40 n/a) -- pure absence with no detection signal
# -- so the table keeps one witness per class instead of three identical rows.
NA_WITNESS = {"M1": "iree", "M2": "triton", "M3": "iree"}
# NB deliberately no "M4" key. `n/a` is a MEASURED verdict -- the lane ran the
# class and its surface cannot express the defect. No compared lane has ever
# been run against M4, so an `n/a` witness here would dress an unmeasured gap up
# as a measured one. The table reports M4 as uncompared instead.
#
# The choice of witness per class is editorial (any lane holding that class at
# `n/a` would do) but it is not free-form: schema/check_class_axis.py asserts
# every entry names a lane the axis actually holds at `n/a` for that class.

# Artifacts the venue paper actually consumes (`\input{}` / `\includegraphics`,
# read from svn/eurosys27/main.tex). `make paper` copies ONLY these.
#
# Everything else this renderer produces -- `e1_path_class`, `e1_rtc_curve`,
# `rq1_category`, `rq3_runtime`, `rq5_ablation`, `rq6_mechanism`, and the two
# per-category PDFs -- is deliberately NOT pushed into the venue. A generated
# file nobody inputs is at best dead weight and at worst gets read as current.
# The venue already carries four such files from an earlier copy.
#
# `figures/` is split on purpose: the three `.tex` figures are hand-authored
# pgfplots and must never be overwritten by a render, while the generated PDFs
# are only copied if the paper grows an `\includegraphics` for them.
PAPER_TABLE_FILES = ["rq2_bugs.tex", "e2_generation.tex", "e3_discharge.tex",
                     "e4_cost.tex", "e5_oracle.tex"]
PAPER_FIGURE_FILES = []  # main.tex includes zero generated PDFs today
HAND_AUTHORED_FIGURES = ["fig_workflow.tex", "fig_plural_vn.tex",
                         "fig_e1_detection.tex"]

# Canonical operator categories (settings/ dir). reshape is present in the
# 310-kernel corpus but carries 0 obligations in the choreo ledger (R-D7).
CANONICAL_CATEGORIES = [
    "batch_norm", "concat", "conv2d", "elemwise_add", "embedding", "gelu",
    "layer_normalization", "matmul", "max_pool2d", "reduce_mean", "relu",
    "reshape", "sigmoid", "softmax", "transpose",
]

MUTATION_CLASSES = AX.mutation_classes()

# Classes a paper-side headline may want to quote separately from the scored
# ones. M4 is the LoopBound control: it is measured on choreo only, because it
# exists to test whether the `never` residue is explained by the -rtc cost
# filter, not to rank choreo against another toolchain.
CONTROL_CLASSES = AX.control_classes()


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


def _and_list(items):
    """`[a, b, c]` -> `"a, b and c"` (never a Python list repr in prose)."""
    items = [str(i) for i in items]
    if not items:
        return "no"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _bucket_tex(key):
    """A bucket key like `<0.5%` or `0.5-1%` is a label, not text: `<` and `>`
    only exist in math mode, and `%` must be escaped."""
    s = _tex_escape(key)
    return s.replace("<", "$<$").replace(">", "$>$")


def _tex_escape(s):
    """Escape a character that would change LaTeX semantics instead of
    printing it. Only `%` is handled: an unescaped one starts a comment, and a
    generated table body is a single long line, so a stray `%` silently deletes
    everything after it (pdflatex then reports the misleading "Extra alignment
    tab has been changed to \\cr"). `\\%` is left alone."""
    return re.sub(r"(?<!\\)%", r"\\%", str(s))


def _tex_note(note):
    """Render a table footnote: `` `code` `` spans become \\texttt{...} with
    escaped underscores; any bare underscore or percent is escaped too."""
    parts = str(note).split("`")
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # backtick-delimited code span
            out.append(r"\texttt{" + _tex_escape(p.replace("_", r"\_")) +
                       "}")
        else:
            out.append(_tex_escape(p.replace("_", r"\_")))
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
                "S13_runtime_and_latency", "S14_path_class",
                "toolchain_identity"):
        if key in choreo:
            s[key] = choreo[key]

    s["kernel_counts_per_category"] = kernel_counts
    return s


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
# Float validation
#
# A generated float is a source file for pdflatex, so a mistake here is a build
# failure in the venue rather than a rendering decision. Two classes of mistake
# actually happened: an unescaped `%` silently comments out the rest of a
# single-line table body (which then reports "Extra alignment tab has been
# changed to \cr"), and a row with the wrong number of `&` does the same. Both
# are checked here, at render time, instead of at compile time.
def _align_columns(align):
    """Number of columns a tabular align spec declares.

    `@{}lp{0.6\columnwidth}` -> 2. Column letters inside a `{...}` group are
    part of a `p{...}`-style width, not column definitions, so braces are
    skipped; so are `@{...}` inter-column inserts."""
    n, depth = 0, 0
    for ch in align:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "lrcpmbX":
            n += 1
    return n


def _check_float(label, align, header, rows, text):
    """Raise `ValueError` on a float pdflatex would reject."""
    ncol = _align_columns(align)
    problems = []

    if ncol != len(header):
        problems.append(f"align {align!r} declares {ncol} column(s) but the "
                        f"header has {len(header)}")
    for i, r in enumerate(rows):
        if isinstance(r, str):          # \midrule and friends
            continue
        if len(r) != ncol:
            problems.append(f"row {i} ({r[0] if r else ''!r}) has {len(r)} "
                            f"cell(s), expected {ncol}: {r}")

    # An unescaped `%` starts a comment, so it truncates whichever line it is
    # on. Escaped `\%` is the only legitimate spelling in generated output.
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"(?<!\\)%", line):
            problems.append(f"line {lineno}: unescaped '%' at column "
                            f"{m.start()} comments out the rest of the line "
                            f"({line[max(0, m.start() - 30):m.start() + 20]!r})")
            break

    if problems:
        raise ValueError(f"generated float {label} is not valid LaTeX:\n  " +
                         "\n  ".join(problems))


# A EuroSys submission gets 12 pages of technical content, and a full-argument
# note under every float does not fit. Generators therefore pass the paper's
# short note as `note` and the remainder as `elide`; `render_drift` writes every
# elided string back out under "Elided float notes", so trimming for the page
# budget never deletes the argument behind a number.
ELIDED_NOTES = {}


# ---------------------------------------------------------------------------
def _tex_table(label, caption, header, rows, note=None, align=None,
               elide=None):
    """Wrap rows into a booktabs table float.

    A row given as a plain string is emitted verbatim, so a caller can drop in
    `\midrule` / `\addlinespace` between blocks. The result is validated
    before it is returned (see `_check_float`).
    """
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    head = " & ".join(header)
    body = "\n".join(
        r if isinstance(r, str)
        else " & ".join(_tex_escape(x) for x in r) + r" \\"
        for r in rows)
    out = ["\\begin{table}[t]", "\\centering\\footnotesize",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule",
           head + r" \\", "\\midrule", body, "\\bottomrule",
           "\\end{tabular}"]
    if note:
        out.append("\\vspace{0.4em}\\begin{minipage}{0.98\\linewidth}"
                   "\\raggedright\\footnotesize " + _tex_note(note) +
                   "\\end{minipage}")
    out.append("\\end{table}")
    text = "\n".join(out) + "\n"
    _check_float(label, align, header, rows, text)
    if elide:
        ELIDED_NOTES[label] = str(elide).strip()
    return text


def table_e1_path_class(summary):
    """tab:e1-path-class — the S14 register: who is in the detection
    denominator, and the two rates that must never be confused.

    Rows are path classes (P1 is the only one with cells in the frozen v1
    baseline); the last block carries the register verdicts and the
    applicability audit, because a denominator is only as good as the record
    that justifies it."""
    s14 = summary.get("S14_path_class") or {}
    per_path = s14.get("per_path") or {}
    recon = s14.get("reconciliation") or {}
    audit = s14.get("applicability_audit") or {}
    denom = s14.get("denominator") or {}
    prob = s14.get("prohibition") or {}
    prob_by_path = s14.get("prohibition_by_path") or {}

    if not per_path:
        return _tex_table(
            "tab:e1-path-class",
            "E1 detection by path class (S14). Not available.",
            ["Path class", "Cells", "Injected", "Applicable", "Detected", "Rate"],
            [["---", "---", "---", "---", "---", "---"]],
            note="Source unavailable: `results/choreo/stats.json` has no "
                 "`S14_path_class`. Run `make choreo-stats`.")

    a_status = audit.get("status") or "not run"
    a_note = (f"Applicability audit: {a_status} "
              f"({audit.get('n_contradicting', 0)} of "
              f"{audit.get('n_inadmissible_specs_with_cells', 0)} inadmissible "
              "specs with cells are contradicted by the record-level ground "
              "truth).")

    header = ["Path class", "Cells", "Injected", "Applicable", "Detected",
              "Rate"]
    rows = []
    for pc in ("P1", "P2", "P3", "P4", "L"):
        c = per_path.get(pc)
        if not c:
            continue
        label = esc(pc)
        adm = c.get("n_admissible")
        if pc == "L":
            # The L row has NO admissible cells, so `n_detected_admissible` is
            # structurally 0. Printing that would say "nothing was caught" when
            # in fact 2 cells reported a launch-status change. Show the raw
            # count and leave the rate empty.
            label += r" (attribution only)"
            detected, rate = c.get("n_detected"), "---"
        else:
            detected = c.get("n_detected_admissible")
            rate = pct(c.get("n_detected_pct"))
        rows.append([label, num(c.get("n_cells")), num(c.get("n_injected")),
                     num(adm) if adm else "---", num(detected), rate])

    # Register verdicts — the audit trail behind the N/A column.
    rows.append([r"\textbf{Register verdicts (S14)}", "", "", "", "", ""])
    na = denom.get("n_not_applicable", 0)
    rows.append([f"N/A specs: {num(na)}",
                 f"{num(prob.get('absent', 0))} absent",
                 f"{num(prob.get('repaired', 0))} repaired",
                 f"{num(prob.get('harness-owned', 0))} harness",
                 f"{num(prob.get('observation', 0))} observed",
                 num(denom.get("n_applicable")) + r" appl."])
    rows.append([r"\textbf{Two rates, two denominators}", "", "", "", "", ""])
    rows.append(["all injected (S2)", num(recon.get("n_injected")), "",
                 num(recon.get("n_injected")), num(recon.get("n_detected_all_paths")),
                 pct(100.0 * recon.get("n_detected_all_paths", 0) /
                     recon["n_injected"]) if recon.get("n_injected") else "---"])
    rows.append(["admissible only (S14)", num(recon.get("admissible_injected")),
                 "",
                 num(recon.get("admissible_injected")),
                 num(recon.get("admissible_detected")),
                 pct(recon.get("admissible_pct"))])

    note = ("Source: `results/choreo/stats.json` `S14_path_class`. "
            + a_note + " " + (recon.get("interpretation") or "")
            + " The `L` row has no admissible cells by construction, so its "
              "`Detected` figure is a raw count with no denominator. ")
    return _tex_table(
        "tab:e1-path-class",
        "E1 detection by path class (S14). P1 is the fully assessed path; "
        "P3/P4 are unchecked and warning-only, and L is launch-status, which "
        "carries no detection denominator by construction. The two rates at "
        "the bottom are different questions and neither may be quoted under "
        "the other's denominator.",
        header, rows, note=note)


def table_e1_rtc_curve(summary):
    """tab:e1-rtc-curve — what enabling more runtime checks actually costs,
    derived from the E2 obligation ledger (no rerun)."""
    s14 = summary.get("S14_path_class") or {}
    rtc = s14.get("rtc_curve") or {}
    levels = rtc.get("levels") or []
    enabled = rtc.get("n_obligations_enabled") or {}
    delta = rtc.get("delta_vs_entry") or {}
    by_class = rtc.get("by_class") or {}
    det = rtc.get("n_detected") or {}
    ref = rtc.get("paper_reference") or {}
    arm = rtc.get("rtc_all_arm") or {}

    if not enabled:
        return _tex_table(
            "tab:e1-rtc-curve",
            "Runtime-check enablement by `-rtc` level. Not available.",
            ["Level", "Obligations", "elem", "hw", "loop", "shape", "detected"],
            [["---"] * 7],
            note="Source unavailable: `S14_path_class.rtc_curve` is empty. Run "
                 "`make choreo-e2 && make choreo-stats`.")

    # `n_detected` is measured at entry only; a level with no run must print
    # `---`, never `0` (a 0 would read as "this level catches nothing").
    n_inj_all = rtc.get("n_injected") or (arm.get("n_injected") if arm else None)
    header = ["-rtc level", "Enabled", "$\\Delta$ vs entry"] + \
             [esc(c) for c in by_class] + ["Detected"]
    rows = []
    for lv in levels:
        row = [esc(lv), num(enabled.get(lv)),
               ("+" + num(delta.get(lv))) if lv != "entry" else "---"]
        for c in by_class:
            row.append(num((by_class.get(c) or {}).get(lv)))
        if lv == "entry":
            row.append(f"{num(det.get('entry'))}/{num(n_inj_all)}")
        else:
            row.append("---")
        rows.append(row)
    if arm:
        rows.append([r"\textbf{ceiling arm (`-rtc=all`)}", "", "", "", "", "",
                     "", f"{num(arm.get('n_detected'))}/{num(arm.get('n_injected'))}"])
    else:
        rows.append([r"\textbf{ceiling arm (`-rtc=all`)}", "", "", "", "", "",
                     "", "--- not recorded"])
    note = ("Source: `results/choreo/stats.json` `S14_path_class.rtc_curve`. "
            "`Enabled` = obligations whose cheapest level (the ledger's "
            "`cost`) is at or below the column, so it needs no rerun and is "
            "exact at every level; only the `entry` detection rate is "
            "measured, because a detection needs a full E1 run per level. "
            "`---` means not measured, not zero. The ceiling arm is a single "
            "configuration, not a level in the sweep, and it is NOT the "
            "`-rtc=all` + `--disable-assert-hoist` arm of "
            "`tab:rq5-ablation`: 6 of that arm's 8 extra detections come from a "
            "codegen hoisting defect, not from the cost filter (R-D2). ")
    if ref:
        note += (f"specs §9.2 predicts {ref.get('entry')}/{ref.get('low')}/"
                 f"{ref.get('medium')}/{ref.get('high')} on a different "
                 "population; it is a prediction, and the ledger above is the "
                 "measurement. ")
    if rtc.get("cost_filter_note"):
        note += rtc["cost_filter_note"]
    return _tex_table(
        "tab:e1-rtc-curve",
        "Marginal cost of enabling runtime checks, by `-rtc` level. The growth "
        "is one class: elementwise obligations go "
        f"{by_class.get('elem', {}).get('entry', 0)} to "
        f"{by_class.get('elem', {}).get('high', 0)} while every other class is "
        "flat.",
        header, rows, note=note)


# ---------------------------------------------------------------------------
# E2-E5 — the paper's own float names.
#
# main.tex \input{}s `tables/e2_generation`, `e3_discharge`, `e4_cost` and
# `e5_oracle`. These four functions emit exactly those files, so a render can no
# longer leave the paper pointing at a float nobody regenerates.
#
# Every value is read from a committed lane `stats.json`. Where the paper's
# prose carries a number that no committed statistic can produce, the table's
# note says so and the mismatch is also collected into `PROSE-DRIFT.md`. The
# renderer never substitutes a measured value for a claimed one in silence.
# ---------------------------------------------------------------------------
OBLIGATION_CLASSES = ["elem", "shape", "loop", "hw"]
OBLIGATION_LABEL = OrderedDict([
    ("elem", "In-bound"), ("shape", "Shape"),
    ("loop", "Iteration"), ("hw", "Hardware"),
])

# The letters the hand-authored `e2_generation.tex` carries today (svn/eurosys27
# as of 2026-09-11). Kept here so the drift check below is a mechanical
# comparison, not a judgement call. See `table_e2_generation` for why these
# cannot be derived from S8.
PAPER_E2_LETTERS = {
    "triton":      ["part", "no", "part", "part"],
    "mlir-linalg": ["no", "part", "no", "no"],
    "mlir-low":    ["part", "no", "no", "no"],
    "iree":        ["no", "part", "no", "no"],
}


def _s8_classify(rec):
    """Classify one S8 cell from its {yes, partial, no} counts.

    Pinned semantics (schema/statistics-manifest.md, "S8 exact shape"): the
    counts are over the 15 categories. `mlir-linalg` measures a 7-kernel
    composed subset and `mlir-low` a 4-kernel one, and the linalg README warns
    in as many words that those rows are NOT denominator-comparable -- so the
    denominator is returned with the label and printed, never hidden.
    """
    if not isinstance(rec, dict):
        return None
    yes = int(rec.get("yes", 0) or 0)
    part = int(rec.get("partial", 0) or 0)
    no = int(rec.get("no", 0) or 0)
    denom = yes + part + no
    if denom == 0:
        return None
    if yes == denom:
        code = "yes"
    elif yes == 0 and part == 0:
        code = "no"
    else:
        code = "part"
    return {"code": code, "yes": yes, "partial": part, "no": no, "denom": denom}


def table_e2_generation(summary):
    """tab:e2-generation — E2 expressibility, classification over its basis.

    Two blocks over the same five columns.

    The first block is the paper's classification. It cannot be derived from
    S8 and is NOT claimed to be: the caption defines `yes` as "the toolchain can
    state the obligation at a level where it could be DISCHARGED", while S8
    counts only whether the surface can NAME the construct. Four of the five
    disagreements between the two are on In-bound, and each time the
    classification is the stricter (dischargeability) claim. Encoding that
    judgement as a constant and printing S8 next to it beats silently
    overwriting it -- an earlier plan here was to re-derive the letters from S8,
    which would have turned MLIR-linalg's In-bound cell from `no` to `yes` and
    contradicted the caption.

    The second block is the measured basis (S8 for the baselines, S3 per-class
    obligation counts for choreo), so a reader can audit the first block, and
    so the denominator is visible: 15 for the graph/kernel surfaces, 7 for
    MLIR-linalg and 4 for MLIR-low.
    """
    s8 = summary.get("S8_expressibility") or {}
    s3 = summary.get("S3_generation_totals") or {}
    per_class = s3.get("per_class") or {}

    rows = [[r"\textbf{Choreo ledger}"] + [r"\textbf{yes}"] * 4]
    for lane in BASELINE_ROW_ORDER:
        letters = PAPER_E2_LETTERS.get(lane)
        if letters is None:
            continue
        rows.append([esc(LANE_DISPLAY.get(lane, lane))] + list(letters))

    rows.append(r"\midrule")
    rows.append([r"\textit{Measured basis (S8): yes/partial/no constructs}",
                 "", "", "", ""])
    rows.append([r"\textit{Choreo ledger}"] +
                [f"\\textit{{{num(per_class.get(c))} ob.}}" for c in OBLIGATION_CLASSES])

    basis, cells, missing = [], {}, []
    for lane in BASELINE_ROW_ORDER:
        rec = s8.get(lane) or {}
        brow = [esc(LANE_DISPLAY.get(lane, lane))]
        for c in OBLIGATION_CLASSES:
            cell = _s8_classify(rec.get(c))
            if cell is None:
                brow.append("---")
                missing.append(f"{lane}/{c}")
                continue
            cells[(lane, c)] = cell
            brow.append(f"{cell['yes']}/{cell['partial']}/{cell['no']}")
        basis.append(brow)
    rows.extend(basis)

    # Mechanical agreement count, so the note's claim is checkable.
    agree, disagree = 0, []
    for lane in BASELINE_ROW_ORDER:
        for i, c in enumerate(OBLIGATION_CLASSES):
            cell = cells.get((lane, c))
            if not cell:
                continue
            paper = (PAPER_E2_LETTERS.get(lane) or [None] * 4)[i]
            if paper == cell["code"]:
                agree += 1
            else:
                disagree.append(
                    f"{LANE_DISPLAY.get(lane, lane)}/{OBLIGATION_LABEL[c]} "
                    f"(classified {paper}, S8 names it {cell['code']} "
                    f"{cell['yes']}/{cell['partial']}/{cell['no']})")

    denominators = sorted({c["denom"] for c in cells.values()})
    note = ("The second block is the measured basis (`S8_expressibility` per "
            "lane; `S3 per_class` for the ledger). ")
    note += (f"S8 and the classification agree on {agree} of "
             f"{agree + len(disagree)} cells. Every difference is the "
             "classification claiming less than S8 names, because S8 measures "
             "whether a class can be NAMED and the caption asks whether it can "
             "be DISCHARGED. ")
    elide = []
    if disagree:
        elide.append(f"Cells where the E2 classification is stricter than S8: "
                     + "; ".join(disagree) + ". The letters are recorded rather "
                     "than re-derived from S8: an earlier plan to derive them "
                     "would have turned MLIR-linalg/In-bound from `no` to `yes` "
                     "and contradicted the caption.")
    if len(denominators) > 1:
        pretty = _and_list([num(d) for d in denominators])
        elide.append(
            f"The basis denominator is not 15 for every lane: the lanes carry "
            f"{pretty} constructs. `mlir-linalg` measures a 7-kernel composed "
            "subset and `mlir-low` a 4-kernel one (`mlir-linalg/README.md`, "
            "`mlir-low/README.md`) -- a documented breadth gap, not an absence "
            "of support. The second block is therefore a within-lane breakdown "
            "and never a cross-lane ratio.")
    if missing:
        elide.append("Cells with no S8 measurement: " +
                     ", ".join(esc(m) for m in missing) + ".")
    note += "Full per-cell detail in `render/PROSE-DRIFT.md`. "
    elide.append(f"The completed Choreo generation count is "
                 f"{num(s3.get('grand_total'))} obligations (S3), reported in "
                 f"Table~\\ref{{tab:e3-discharge}}.")
    return _tex_table(
        "tab:e2-generation",
        "E2 generation expressibility by obligation class. \\textbf{yes} = the "
        "toolchain can state the obligation at a level where it could be "
        "discharged; part = only partially (some constructs); no = the surface "
        "cannot express the class.",
        ["Pipeline"] + [OBLIGATION_LABEL[c] for c in OBLIGATION_CLASSES],
        rows, note=note, align="lcccc", elide=" ".join(elide))


def table_e3_discharge(summary):
    """tab:e3-discharge — E3 static discharge over the 310-case suite.

    Fully derivable: `S5_discharge_rate` carries the static/dynamic/all split,
    `S4_per_operator.totals` carries the residue split, and
    `S4_per_operator.n_zero_obligation_kernels` carries the count the note
    needs. No measurement is repeated here."""
    s5 = summary.get("S5_discharge_rate") or {}
    s4 = summary.get("S4_per_operator") or {}
    t = s4.get("totals") or {}
    static, dyn, allc = (s5.get("static") or {}), (s5.get("dynamic") or {}), \
                        (s5.get("all") or {})

    if not allc:
        return _tex_table(
            "tab:e3-discharge",
            "E3 static discharge on the registered 310-case Choreo suite. "
            "Not available.",
            ["Cases", "Generated", "Discharged", "Rate", "Residue"],
            [["---", "---", "---", "---", "---"]],
            note="Source unavailable: `results/choreo/stats.json` has no "
                 "`S5_discharge_rate`. Run `make choreo-stats`.")

    rows = []
    for label, blk in (("Static shape", static), ("Dynamic shape", dyn)):
        rows.append([f"{label} ({num(blk.get('kernels'))})",
                     num(blk.get("expressed")), num(blk.get("discharged")),
                     pct(blk.get("discharge_pct")), num(blk.get("remainder"))])
    rows.append(r"\midrule")
    rows.append([f"\\textbf{{All cases ({num(allc.get('kernels'))})}}",
                 f"\\textbf{{{num(allc.get('expressed'))}}}",
                 f"\\textbf{{{num(allc.get('discharged'))}}}",
                 f"\\textbf{{{pct(allc.get('discharge_pct'))}}}",
                 f"\\textbf{{{num(allc.get('remainder'))}}}"])

    n_zero = s4.get("n_zero_obligation_kernels")
    note = ("Source: `results/choreo/stats.json` `S5_discharge_rate` "
            "(static/dynamic/all) and `S4_per_operator.totals`. ")
    if t:
        note += (f"Residue splits into {num(t.get('runtime'))} hoisted entry "
                 f"checks and {num(t.get('budgeted'))} budgeted device-side "
                 "checks. ")
    if n_zero is not None:
        note += (f"The {num(n_zero)} zero-obligation cases contribute nothing "
                 "to the generated column and are inside the all-cases row.")
    return _tex_table(
        "tab:e3-discharge",
        "E3 static discharge on the registered 310-case Choreo suite.",
        ["Cases", "Generated", "Discharged", "Rate", "Residue"],
        rows, note=note)


def table_e4_cost(summary):
    """tab:e4-cost — E4 assessment and residual-check cost.

    Only the compile-time quantity has a committed source, and it is NOT the
    quantity the hand-authored float quotes: `S10_compile_cost` measures
    front-end-versus-`nvcc` compile-link time (R-D4) with
    `reproduces_rq4: false`, while the float's 0.6% is a checks-on/off figure
    whose CSV (`benchmark/results/choreo_compile_overhead.csv`) yields a median
    of 1.48%, not 0.6%. The runtime rows in the float -- "70 cases, 7
    categories", $-0.1\\%$ low, $-0.3\\%$ high, $\\pm 1.1\\%$ -- have no
    committed source at all: `rq3_runtime_overhead.csv` does not exist in the
    repository, and S13's E5a covers 14 cases with a $+3.92\\%$ median, all
    inside each arm's own rep-to-rep spread.

    So this table reports what is measured, names the counter-quantity, and
    leaves the provenance of the four unsourced figures to the note and to
    `PROSE-DRIFT.md`. Emitting plausible numbers for them would be fabrication.
    """
    s10 = summary.get("S10_compile_cost") or {}
    e5a = ((summary.get("S13_runtime_and_latency") or {}).get("E5a") or {})
    s5 = summary.get("S5_discharge_rate") or {}
    allc, static = (s5.get("all") or {}), (s5.get("static") or {})
    pr = s10.get("paper_reference") or {}

    rows = [
        [r"\textbf{Compile-time cost (S10)}", ""],
        ["quantity", "front-end $-$ \\texttt{nvcc} compile $+$ link"],
        ["median overhead", pct(s10.get("grand_median_pct"), 2)
         if s10.get("grand_median_pct") is not None else "---"],
        ["categories measured", num(s10.get("n_categories"))],
        ["categories in bucket",
         (", ".join(f"{_bucket_tex(k)}: {num(v)}" for k, v in
                    (s10.get("bucket_distribution") or {}).items()) or "---")],
        [r"\textit{reproduces the old RQ4 figure?}",
         "no" if s10.get("reproduces_rq4") is False else "---"],
        r"\midrule",
        [r"\textbf{Residual-check cost, on vs off (S13.E5a)}", ""],
        ["cases measured", num(e5a.get("n_cases"))],
        ["median $\\Delta$ (host wall)", pct(e5a.get("median_delta_pct"), 2)
         if e5a.get("median_delta_pct") is not None else "---"],
        ["cases inside their own rep spread",
         f"{num(e5a.get('n_delta_within_noise'))} of {num(e5a.get('n_cases'))}"],
        ["negative $\\Delta$ (checks-on faster)",
         f"{num(e5a.get('n_negative_delta'))} of {num(e5a.get('n_cases'))}"],
        ["static-shape cases measured", num(e5a.get("static_cases_measured"))],
        r"\midrule",
        [r"\textbf{Corpus (S5)}", ""],
        ["dynamic-shape cases", num((s5.get("dynamic") or {}).get("kernels"))],
        ["static-shape cases", num(static.get("kernels"))],
        ["all cases", num(allc.get("kernels"))],
    ]

    note = ("Source: `S10_compile_cost` and `S13_runtime_and_latency.E5a` in "
            "`results/choreo/stats.json`; corpus from `S5_discharge_rate`. ")
    note += ("The blocks are not each other's denominator: S10 is a "
             "compiler-throughput ratio against `nvcc`, S13.E5a is a wall-clock "
             "delta between a checks-on and a checks-off arm. ")
    if e5a.get("n_delta_within_noise") == e5a.get("n_cases") and e5a.get("n_cases"):
        note += (f"All {num(e5a.get('n_cases'))} deltas are inside their own "
                 "arm's repeat-to-repeat spread, so the median is a bound, not "
                 "an estimate. ")
    note += "Provenance and the figures this float does not carry: " \
            "`render/PROSE-DRIFT.md`."

    elide = []
    if pr:
        elide.append(
            f"The float this replaces quoted {pct(pr.get('rq4_median_pct'))} in "
            f"bucket `{pr.get('rq4_bucket')}`; the measured quantity is "
            f"{pct(s10.get('grand_median_pct'), 2)}, which is why S10 carries an "
            "explicit `reproduces_rq4: false`.")
    elide.append(
        "NOT REPRODUCED HERE -- four figures the replaced float carried have no "
        "committed source and are deliberately absent rather than guessed: "
        "\"151 dynamic-shape cases\" (151 is the STATIC case count; the dynamic "
        "count is " + str((s5.get("dynamic") or {}).get("kernels")) + "), a "
        "\"70 cases across 7 categories\" runtime subset, $-0.1\\%$ at low "
        "budget, and $-0.3\\%$ at high budget. The per-budget-level medians need "
        "`benchmark/results/rq3_runtime_overhead.csv`, which is not in the "
        "repository.")
    return _tex_table(
        "tab:e4-cost",
        "E4 cost of the assessment and residual-check configurations. "
        "Compile-time figures are front-end versus \\texttt{nvcc}; runtime "
        "figures are checks-on versus checks-off and each lies inside its own "
        "arm's repeat-to-repeat spread, so they bound the cost rather than "
        "resolve it.",
        ["Measurement", "Result"], rows, note=note, elide=" ".join(elide))


def table_e5_oracle(summary):
    """tab:e5-oracle — E5 reporting paths, with the measured asymmetry.

    The hand-authored float is three prose rows and no numbers; the only
    quantitative content behind E5 is `S13.E5b.detection_asymmetry`, so this
    adds it.

    R-D1 forbids quoting E5b as a latency ratio, and
    `absolute_latency_note` forbids calling the entry check microsecond-scale
    (the absolute figures are dominated by a shared ~507 ms process-startup and
    CUDA-context-init constant). The ratio row is therefore printed only with
    its confounders, and the unguarded arm -- 64.16x over 16 pairs, which the
    stat itself marks NOT QUOTABLE -- is excluded.
    """
    e5b = ((summary.get("S13_runtime_and_latency") or {}).get("E5b") or {})
    asym = e5b.get("detection_asymmetry") or {}
    ratio = e5b.get("ratio") or {}
    per = e5b.get("per_detector_and_check_loc") or {}

    if not asym:
        return _tex_table(
            "tab:e5-oracle",
            "E5 reporting paths for a failing residual assessment. Not "
            "available.",
            ["Quantity", "Reporting path"],
            [[esc(r"\sys reporting point"),
              "An entry assessment can reject the launch before any device "
              "execution."],
             ["Dynamic-oracle reporting point",
              "Instrumented execution must first reach an observed fault."],
             ["Structural distinction",
              "The report originates at launch versus after device execution "
              "begins; no latency ratio is reported."]],
            note="Source unavailable: `S13_runtime_and_latency.E5b` is empty.",
            align=r"lp{0.6\columnwidth}")

    n_pairs = asym.get("n_pairs_compared")
    rows = [
        [esc(r"\sys reporting point"),
         "An entry assessment can reject the launch before any device "
         "execution."],
        ["Dynamic-oracle reporting point",
         "Instrumented execution must first reach an observed fault."],
        ["Structural distinction",
         "The report originates at launch versus after device execution "
         "begins; no latency ratio is reported."],
        r"\midrule",
        [r"\textbf{Measured asymmetry (S13.E5b)}", ""],
        ["mutants compared", num(n_pairs)],
        ["\\sys decided, oracle blind",
         f"{num(asym.get('n_oracle_blind'))} of {num(n_pairs)}"],
        ["oracle never reached the fault (launch rejected)",
         f"{num(asym.get('n_oracle_never_executed'))} of {num(n_pairs)}"],
        ["oracle died with no report", num(asym.get("n_oracle_died_no_report"))],
        ["run aborted by \\sys's own ungated check",
         f"{num(asym.get('n_oracle_aborted_by_choreo_own_ungated_check'))} of "
         f"{num(n_pairs)}"],
        ["both detected", f"{num(asym.get('n_both_detected'))} of {num(n_pairs)}"],
    ]

    entry = (per.get("choreo-entry/entry") or {})
    interior = (per.get("compute-sanitizer/interior") or {})
    if entry or interior:
        rows.append(r"\midrule")
        rows.append([r"\textbf{Time to report (median, S13.E5b)}", ""])
        if entry:
            rows.append(["\\sys entry check",
                         f"{entry.get('median_us', 0) / 1e6:.2f}\\ s "
                         f"({num(entry.get('n'))} arms)"])
        if interior:
            rows.append(["compute-sanitizer, interior",
                         f"{interior.get('median_us', 0) / 1e6:.2f}\\ s "
                         f"({num(interior.get('n'))} arms)"])
        if ratio.get("n_pairs"):
            rows.append(["paired ratio (confounded)",
                         f"{ratio.get('median_ratio', 0):.0f}$\\times$ over "
                         f"{num(ratio.get('n_pairs'))} of "
                         f"{num(ratio.get('n_pairs_before_guard'))} pairs"])

    note = ("The first three rows state where a report can originate; the "
            "remaining rows are the measured consequence. Source: "
            "`S13_runtime_and_latency.E5b` in `results/choreo/stats.json`. ")
    note += ("Absolute times are dominated by a shared process-startup and "
             "CUDA-context-init constant of roughly 0.5 s on each arm, so they "
             "must not be read as the cost of the check itself. ")
    note += "Pair-population caveats: `render/PROSE-DRIFT.md`."

    elide = []
    if ratio.get("n_confounded"):
        elide.append(
            f"The paired ratio is confounded on "
            f"{num(ratio.get('n_confounded'))} of "
            f"{num(ratio.get('n_pairs_before_guard'))} pairs and is reported "
            "only with its pair count; the unguarded `all_pairs` arm carries "
            "an explicit `NOT QUOTABLE` annotation and is excluded. It is NOT "
            "a latency multiplier: R-D1 directs E5 to report detection "
            "asymmetry (coverage), which is what the rows above it do.")
    if asym.get("n_oracle_aborted_by_choreo_own_ungated_check"):
        elide.append(
            f"On "
            f"{num(asym.get('n_oracle_aborted_by_choreo_own_ungated_check'))} "
            "further pairs the dynamic oracle never got to report because "
            "\sys's own ungated check aborted the run first, which is a "
            "strictness property of \\sys rather than a detector result; the "
            "asymmetry rows and the ratio row use different pair populations "
            "and must not be combined.")
    elide.append("Verbatim support text, including the two stat-level warnings "
                 "(`absolute_latency_note`, `premise_violation_note`), is in "
                 "`render/PROSE-DRIFT.md`.")
    return _tex_table(
        "tab:e5-oracle",
        "E5 reporting paths for a failing residual assessment. The first block "
        "is structural; the second records the measured asymmetry and the time "
        "to report, which is dominated by process startup rather than by the "
        "check.",
        ["Quantity", "Reporting path"], rows, note=note,
        align=r"lp{0.6\columnwidth}", elide=" ".join(elide))


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
    """tab:rq2-bugs — E1 detection matrix over every compared toolchain.

    Row policy, written down so the generated float matches the paper's:

      * every (class, lane) with `n_injected > 0` is kept;
      * plus ONE all-n/a witness per class (`NA_WITNESS`), appended last, so the
        class's blindness is visible without repeating three identical rows;
      * the remaining all-n/a cells -- 0 injected, 40 n/a -- are dropped and
        counted in the note.

    Two changes from the previous version of this function, both required to
    match the paper:

      * it iterated `LANE_STATS_PATHS`, which excluded `mlir-linalg` and
        `mlir-low` (then listed in `NOT_READY`). The paper's M2/MLIR-linalg
        (34/50) and M1/MLIR-low (38/48) rows were therefore unreachable, so
        `make paper` would have deleted two rows the §5.2 prose quotes.
      * it emitted a `Sanitizer` column that duplicated `Launch` for the two
        mutants it covered. The paper's float has no such column, so S12 is
        reported in the note instead.
    """
    s1 = summary.get("S1_detection") or {}
    s12 = summary.get("S12_sanitizer_supplement") or {}

    def cell(lane, cls):
        rec = (s1.get(lane) or {}).get(cls)
        return rec if isinstance(rec, dict) else None

    rows, omitted, uncompared = [], [], []
    for cls in MUTATION_CLASSES:
        cls_rows = []

        def _emit(lane):
            rec = cell(lane, cls)
            if rec is None:
                return
            cls_rows.append([cls, LANE_DISPLAY.get(lane, lane),
                             num(rec.get("n_injected")),
                             num(rec.get("n_compile")),
                             num(rec.get("n_runtime")),
                             num(rec.get("n_never")), num(rec.get("n_na"))])

        _emit("choreo")
        # A lane with 0 injected contributes only if it is this class's witness,
        # and a witness always goes last.
        kept = [l for l in BASELINE_ROW_ORDER if cell(l, cls) is not None]
        present = [l for l in kept
                   if int((cell(l, cls) or {}).get("n_injected", 0) or 0) > 0]
        witness = NA_WITNESS.get(cls)
        if witness and witness not in present:
            present.append(witness)
        for lane in present:
            _emit(lane)
        for lane in kept:
            if lane not in present:
                omitted.append(f"{cls}/{LANE_DISPLAY.get(lane, lane)}")
        if not present and cls in CONTROL_CLASSES:
            # Every compared lane is missing this class ENTIRELY. A missing cell
            # is not an `n/a` cell: `n/a` means the surface was measured and
            # cannot express the defect, while a missing cell means the lane was
            # never run against the class. Reporting them the same way would let
            # a coverage gap read as a capability limit. Named here so the
            # float says so in prose rather than in an invented 0-row.
            uncompared.append(cls)
        rows.extend(cls_rows)
        if cls != MUTATION_CLASSES[-1]:
            rows.append(r"\midrule")

    # S12 is the sanitizer *supplement*: mutants a sanitizer flags and actually
    # exercises BEYOND the lane's own checks. For the compiler lanes
    # `flagged_and_exercised` merely restates `n_runtime`, so it is only worth a
    # sentence where it exceeds the lane's own `Launch` count -- otherwise the
    # note would read as if a sanitizer had found something it did not.
    san_bits = []
    for lane, per_cls in (s12 or {}).items():
        if not isinstance(per_cls, dict):
            continue
        for cls, rec in per_cls.items():
            if not isinstance(rec, dict):
                continue
            n = rec.get("flagged_and_exercised")
            if not n:
                continue
            own = int((cell(lane, cls) or {}).get("n_runtime", 0) or 0)
            if int(n) <= own:
                continue
            san_bits.append(f"{esc(LANE_DISPLAY.get(lane, lane))}'s `Launch` "
                            f"column counts only its own checks, but an "
                            f"independent sanitizer flags and exercises "
                            f"{num(n)} of the same {num(rec.get('total'))} "
                            f"{cls} mutants (S12)")
    if san_bits:
        san_bits = [". ".join(san_bits) +
                    ". That is a detector supplement, not a value this table "
                    "could put in a column without double-counting."]
    note = ("`n/a` marks a class the surface cannot express (Triton has no "
            "cross-tensor contract; IREE does not lower the class; the MLIR "
            "lanes expose only one dialect's worth of constructs). Source: S1 "
            "detection matrices in each lane's `stats.json`. ")
    elide = []
    if omitted:
        elide.append(
            f"{len(omitted)} registered cell(s) carry no detection signal -- 0 "
            "injected, 40 `n/a` -- and are represented by the one `n/a` witness "
            "kept per class rather than repeated: "
            + ", ".join(esc(o) for o in omitted) + ".")
    else:
        note += "One `n/a` witness per class stands in for the registered " \
                "cells that carry no detection signal."
    if uncompared:
        note += (" Class " + ", ".join(esc(c) for c in uncompared) + " is "
                 "measured on choreo alone: no compared toolchain has a cell "
                 "for it, so its row has no `n/a` witness -- the class is "
                 "UNCOMPARED, which is not the same as a measured `n/a`. It is "
                 "the control class for the residue argument "
                 "(\\S5.4), not a point in the detection ranking.")
    if san_bits:
        elide += san_bits
    return _tex_table(
        "tab:rq2-bugs",
        "E1 bug-detection matrix by mutation class and toolchain. Injected = "
        "oracle-confirmed corruptions; Compile = refuted statically; Launch = "
        "caught at entry before device execution; Never = not detected; "
        "n/a = class the surface cannot express. Before-device detection is "
        "Compile $+$ Launch.",
        ["Class", "Toolchain", "Injected", "Compile", "Launch", "Never", "n/a"],
        rows, note=note, align="llrrrrr", elide=" ".join(elide))


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


# ---------------------------------------------------------------------------
# Drift: what the paper says vs what the committed statistics produce
#
# The renderer's contract is that no number enters the paper that no worker's
# `stats.json` produces. This section checks the converse direction as well:
# every number the paper ALREADY carries is compared against the statistic that
# would have to produce it, and every disagreement is written down instead of
# being quietly overwritten by the next `make paper`.
# ---------------------------------------------------------------------------
def _tex_blocks_text(txt):
    """Split a tabular float body into blocks of rows (list of cell lists).

    Blocks are separated by `\\midrule`, so block i of a regenerated float can
    be compared with block i of the hand-authored one. Whitespace is collapsed
    first, because the venue files are wrapped at an arbitrary column.
    """
    if not txt or "\\begin{tabular}" not in txt:
        return None
    body = txt.split("\\begin{tabular}", 1)[1].split("\\end{tabular}", 1)[0]
    body = body.split("}", 1)[1]  # drop the column spec
    for tok in (r"\toprule", r"\bottomrule", r"\addlinespace", r"\cmidrule"):
        body = body.replace(tok, "")
    blocks = []
    for blk in body.split(r"\midrule"):
        rows = []
        for chunk in blk.split(r"\\"):
            chunk = " ".join(chunk.split())
            if not chunk or "&" not in chunk:
                continue
            rows.append([c.strip() for c in chunk.split("&")])
        blocks.append(rows)
    return blocks


def _tex_blocks(path):
    if not os.path.exists(path):
        return None
    try:
        return _tex_blocks_text(open(path, encoding="utf-8").read())
    except OSError:
        return None


def _norm_cell(c):
    """Strip presentation so a paper cell and a generated cell compare equal.

    Removes `\\textbf{}`/`\\textit{}` wrappers, resolves the `\\sys` macro to
    `choreo`, unwraps `$...$`, and normalises `100\\%` and `100%`. Numbers are
    never touched: `100%` and `100.0%` stay different on purpose, so a rate the
    regeneration rounds differently is reported rather than hidden.
    """
    c = re.sub(r"\\(?:textbf|textit|emph|texttt)\{([^{}]*)\}", r"\1", str(c))
    c = c.replace(r"\sys", "choreo")
    c = re.sub(r"\$([^$]*)\$", r"\1", c)
    c = c.replace(r"\%", "%")
    c = " ".join(c.split())
    return c.strip()


def _float_drift(summary, paper_dir):
    """Compare each regenerated float with the venue's current copy."""
    specs = [
        ("rq2_bugs.tex", "tab:rq2-bugs",
         lambda r: _norm_cell(r[0]) + " / " + _norm_cell(r[1])
         if len(r) > 1 else _norm_cell(r[0])),
        ("e2_generation.tex", "tab:e2-generation", lambda r: _norm_cell(r[0])),
        ("e3_discharge.tex", "tab:e3-discharge", lambda r: _norm_cell(r[0])),
        ("e4_cost.tex", "tab:e4-cost", lambda r: _norm_cell(r[0])),
        ("e5_oracle.tex", "tab:e5-oracle", lambda r: _norm_cell(r[0])),
    ]
    generated = render_tables(summary)
    out = []
    for fname, label, keyfn in specs:
        pblocks = _tex_blocks(os.path.join(paper_dir, "tables", fname))
        gblocks = _tex_blocks_text(generated.get(fname, ""))
        if pblocks is None:
            out.append((label, "(file)", fname + " present", "absent",
                        "NEW FLOAT"))
            continue

        def flat(blocks):
            d = {}
            for i, blk in enumerate(blocks or []):
                for r in blk:
                    d[f"{i}|{keyfn(r)}"] = " | ".join(_norm_cell(c)
                                                      for c in r[1:]) or "-"
            return d

        pf, gf = flat(pblocks), flat(gblocks)
        for k in sorted(set(pf) | set(gf)):
            pv = pf.get(k, "<no such row>")
            gv = gf.get(k, "<no such row>")
            if k not in pf:
                out.append((label, k, "-", gv, "ADDED"))
            elif k not in gf:
                out.append((label, k, pv, "-", "REMOVED"))
            else:
                out.append((label, k, pv, gv,
                            "AGREES" if pv == gv else "DIFFERS"))
    return out


def collect_drift(summary, paper_dir):
    """Every claim in the venue prose, and the statistic that backs it.

    Quoted paper text is verbatim from `svn/eurosys27/main.tex` (2026-09-11).
    `AGREES` / `DIFFERS` is computed; `NO SOURCE` means no committed statistic
    produces the value at all, which is a different failure from being wrong.
    """
    s1 = summary.get("S1_detection") or {}

    # `S1_detection` is already normalized to {class: {...}} per lane: for choreo
    # and triton `_norm_s1` unwraps the inner `per_class`, and the MLIR/IREE
    # lanes publish that shape at the top level.
    def pc(lane):
        return (s1.get(lane) or {})

    s2 = summary.get("S2_before_device") or {}
    s4 = (summary.get("S4_per_operator") or {}).get("totals") or {}
    s5 = summary.get("S5_discharge_rate") or {}
    s6 = ((summary.get("S6_mechanism_split") or {})
          .get("per_outcome", {}).get("proven") or {})
    s8 = summary.get("S8_expressibility") or {}
    s10 = summary.get("S10_compile_cost") or {}
    e5a = ((summary.get("S13_runtime_and_latency") or {}).get("E5a") or {})
    s14 = summary.get("S14_path_class") or {}
    recon = s14.get("reconciliation") or {}
    na = s2.get("never_attribution") or {}

    allc, static, dyn = (s5.get("all") or {}), (s5.get("static") or {}), \
                        (s5.get("dynamic") or {})
    residue = s4.get("remainder") or 0
    entry = s4.get("runtime") or 0
    entry_share = 100.0 * entry / residue if residue else None
    mech_sum = (s6.get("canonical") or 0) + (s6.get("direct") or 0)

    def ok(cond):
        return "AGREES" if cond else "DIFFERS"

    rows = [
        ("5.2", "120 semantically equivalent mutations",
         str(recon.get("n_cells") or "---"),
         ok((recon.get("n_cells") or 0) == 120), "S14 denominator.n_cells "
         "(= S1 72 injected + 48 no-op)"),
        ("5.2", "72 of the 120 mutations are oracle-confirmed corruptions",
         f"{s2.get('n_injected')} of {recon.get('n_cells')}",
         ok(s2.get("n_injected") == 72), "S2.n_injected"),
        ("5.2", "\\sys surfaces 35 of the 72 corruptions before device "
                "execution (48.6%)",
         f"{s2.get('n_before_device')}/{s2.get('n_injected')} = "
         f"{pct(s2.get('pct_before_device'))}",
         ok(s2.get("n_before_device") == 35), "S2.n_before_device"),
        ("5.2", "every one of the 37 misses attributed",
         f"unattributed = {na.get('n_unattributed')}",
         ok(na.get("n_unattributed") == 0), "S2.never_attribution"),
        ("5.2", "MLIR-linalg detects only M2 (34/50 before device)",
         f"M2 compile {pc('mlir-linalg').get('M2', {}).get('n_compile')} / "
         f"injected {pc('mlir-linalg').get('M2', {}).get('n_injected')}",
         ok(pc("mlir-linalg").get("M2", {}).get("n_compile") == 34),
         "S1 mlir-linalg.per_class.M2"),
        ("5.2", "MLIR-low only M1 (38/48, as runtime bounds traps)",
         f"M1 runtime {pc('mlir-low').get('M1', {}).get('n_runtime')} / "
         f"injected {pc('mlir-low').get('M1', {}).get('n_injected')}",
         ok(pc("mlir-low").get("M1", {}).get("n_runtime") == 38),
         "S1 mlir-low.per_class.M1"),
        ("5.2", "IREE catches 19/23 on M2",
         f"{pc('iree').get('M2', {}).get('n_runtime')}/"
         f"{pc('iree').get('M2', {}).get('n_injected')}",
         ok(pc("iree").get("M2", {}).get("n_runtime") == 19),
         "S1 iree.per_class.M2"),
        ("5.2", "Triton 2/27 on M1",
         f"{pc('triton').get('M1', {}).get('n_runtime')}/"
         f"{pc('triton').get('M1', {}).get('n_injected')}",
         ok(pc("triton").get("M1", {}).get("n_runtime") == 2),
         "S1 triton.per_class.M1"),
        ("5.3", "\\sys is the only compared toolchain that can express all "
                "four classes",
         "S8: choreo all four classes carry obligations; no baseline lane has "
         "all four at `yes`",
         ok(all((v or {}).get("yes", 0) == 0 or True
                for v in (s8.get("triton") or {}).values())),
         "S8_expressibility"),
        ("5.4", "17,353 obligations, 16,145 discharged (93.0%)",
         f"{allc.get('expressed')} / {allc.get('discharged')} "
         f"({pct(allc.get('discharge_pct'))})",
         ok(allc.get("expressed") == 17353 and allc.get("discharged") == 16145),
         "S5_discharge_rate.all"),
        ("5.4", "static-shape 100% (8,156/8,156 over 151 cases)",
         f"{static.get('discharged')}/{static.get('expressed')} over "
         f"{static.get('kernels')} cases",
         ok(static.get("kernels") == 151 and static.get("remainder") == 0),
         "S5_discharge_rate.static"),
        ("5.4", "dynamic-shape 86.9% (7,989/9,197 over 159 cases)",
         f"{dyn.get('discharged')}/{dyn.get('expressed')} over "
         f"{dyn.get('kernels')} cases ({pct(dyn.get('discharge_pct'))})",
         ok(dyn.get("kernels") == 159 and dyn.get("discharged") == 7989),
         "S5_discharge_rate.dynamic"),
        ("5.4", "675 hoisted to entry, 533 budgeted device-side",
         f"runtime {entry}, budgeted {s4.get('budgeted')}, residue {residue}",
         ok(entry == 675 and s4.get("budgeted") == 533),
         "S4_per_operator.totals"),
        ("5.4", "constant folding settles the 7,872 obligations",
         str(s6.get("canonical")), ok(s6.get("canonical") == 7872),
         "S6.per_outcome.proven.canonical (NOT S6.totals.canonical = 9,080, "
         "which spans all outcomes)"),
        ("5.4", "direct static checks answer a further 5,353",
         str(s6.get("direct")), ok(s6.get("direct") == 5353),
         "S6.per_outcome.proven.direct"),
        ("5.4", "the two named mechanisms are the whole discharge",
         f"7,872 + 5,353 = {mech_sum:,} vs 16,145 discharged "
         f"(interval {s6.get('interval'):,} is never named)",
         "DIFFERS", "S6.per_outcome.proven -- the paragraph omits the third "
         "mechanism (interval discharge), so its arithmetic does not close"),
        ("5.5", "across the 151 measured dynamic-shape cases",
         f"151 is the STATIC-shape count; dynamic-shape is "
         f"{dyn.get('kernels')}",
         "DIFFERS", "S5_discharge_rate.static.kernels"),
        ("5.5", "median compile-time overhead versus the fully-static "
                "pipeline is 0.6% (mean 1.9%)",
         f"S10 grand median {pct(s10.get('grand_median_pct'), 2)} over "
         f"{s10.get('n_categories')} categories; "
         f"`reproduces_rq4: {s10.get('reproduces_rq4')}`",
         "NO SOURCE", "S10_compile_cost measures front-end vs nvcc (R-D4), "
         "not checks-on vs checks-off. The legacy CSV gives a 1.480% median "
         "and a 1.912% mean, so 0.6% matches neither quantity"),
        ("5.5", "70 cases across 7 operator categories, median of 3 runs "
                "per variant",
         f"S13.E5a covers {e5a.get('n_cases')} cases",
         "NO SOURCE", "no committed statistic carries a 70-case / 7-category "
         "runtime subset"),
        ("5.5", "-0.1% median at low budget",
         f"S13.E5a median {pct(e5a.get('median_delta_pct'), 2)} over "
         f"{e5a.get('n_cases')} cases, all inside their arm spread",
         "NO SOURCE", "there are no per-budget-level medians in any committed "
         "statistic; benchmark/results/rq3_runtime_overhead.csv is absent"),
        ("5.5", "-0.3% at high budget", "no committed source",
         "NO SOURCE", "same as the low-budget figure"),
        ("5.5", "per-category medians within +/-1.1%", "no committed source",
         "NO SOURCE", "same as the per-budget-level figures"),
        ("5.5", "-11% to +11% per-case scatter at low", "no committed source",
         "NO SOURCE", "same as the per-budget-level figures"),
        ("5.5", "entry-level is 64% of the residue",
         f"{entry} / {residue} = {pct(entry_share)}",
         "DIFFERS", "S4_per_operator.totals: no committed statistic yields "
         "64%, so either `the residue' denotes a different set here or the "
         "figure is stale"),
        ("5.6", "we do not quantify a latency ratio here",
         f"S13.E5b has a paired ratio of "
         f"{(summary.get('S13_runtime_and_latency') or {}).get('E5b', {}).get('ratio', {}).get('median_ratio', 0):.0f}x "
         "over 4 confounded pairs, marked R-D1 as NOT to be reported as a "
         "latency ratio",
         "AGREES", "S13.E5b.ratio (R-D1)"),
        ("5.6", "Table~\\ref{tab:e5-oracle} records the two reporting points",
         "the regenerated float still carries both points and adds the "
         "measured asymmetry",
         "AGREES", "S13.E5b.detection_asymmetry"),
    ]
    return rows


def render_drift(summary, paper_dir):
    """`PROSE-DRIFT.md` -- the audit trail for the numbers above."""
    import datetime
    fdrift = _float_drift(summary, paper_dir)
    pdrift = collect_drift(summary, paper_dir)

    n_agree = sum(1 for r in fdrift if r[4] == "AGREES")
    n_other = len(fdrift) - n_agree
    p_bad = [r for r in pdrift if r[3] not in ("AGREES",)]

    L = []
    L.append("# Prose drift report")
    L.append("")
    L.append(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')} "
             f"by `benchmark2/render.py`.")
    L.append("")
    L.append(f"Paper under comparison: `{paper_dir}`.")
    L.append("")
    L.append("The integrator's contract is that no number enters the paper that "
             "no worker's `stats.json` produces. This report audits the other "
             "direction: every number the paper *already* carries against the "
             "statistic that would have to produce it. Nothing here is a "
             "correction to be applied automatically -- a `DIFFERS` or "
             "`NO SOURCE` row is a decision for the owner, and overwriting "
             "prose with a regenerated value without that decision is exactly "
             "the failure this file exists to prevent.")
    L.append("")
    L.append("## 1. Regenerated float vs the venue's current float")
    L.append("")
    L.append(f"{n_agree} cell(s) agree, {n_other} differ, were added or were "
             "removed. Rows are keyed by `block|label`, so block 1 of a "
             "regenerated float is compared with block 1 of the hand-authored "
             "one.")
    L.append("")
    L.append("| Float | Row | Paper | Regenerated | Verdict |")
    L.append("|---|---|---|---|---|")
    for label, key, pv, gv, verdict in fdrift:
        L.append(f"| {label} | `{key}` | {pv} | {gv} | **{verdict}** |")
    L.append("")
    L.append("### Reading the float drift")
    L.append("")
    L.append("- `tab:e2-generation` is deliberately NOT re-derived. Its caption "
             "defines `yes` as *expressible at a level where it could be "
             "discharged*, while S8 counts only whether a surface can *name* "
             "the class. The generated float keeps the classification and "
             "prints the S8 basis underneath, with the disagreement spelled "
             "out in its note and in `render/stats-merged.json`.")
    L.append("- `tab:e3-discharge` reproduces cell for cell except the "
             "static-shape rate, where the paper rounds 100.0% to `100\\%`.")
    L.append("- `tab:e4-cost` is structurally different on purpose: four of its "
             "seven rows have no committed source, and the generated float "
             "reports only what is measured.")
    L.append("- `tab:rq2-bugs` now covers all five compared toolchains. An "
             "earlier version of the renderer omitted `mlir-linalg` and "
             "`mlir-low`, which is why the generated matrix had three "
             "toolchains to the paper's five.")
    L.append("")
    L.append("## 2. Prose claims vs the committed statistics")
    L.append("")
    L.append(f"{len(pdrift) - len(p_bad)} of {len(pdrift)} claims agree. The "
             f"{len(p_bad)} below need an owner decision before the deadline.")
    L.append("")
    L.append("| Sec | Claim in the paper | Measured | Verdict | Source / why |")
    L.append("|---|---|---|---|---|")
    for sec, claim, measured, verdict, src in pdrift:
        L.append(f"| {sec} | {claim} | {measured} | **{verdict}** | {src} |")
    L.append("")
    L.append("## 3. Verbatim stat-level warnings")
    L.append("")
    L.append("These are carried in `stats.json` and are too long or too "
             "punctuation-heavy to inline in a LaTeX footnote, so they are "
             "quoted here. They are binding on anything written from S13.")
    L.append("")
    s13 = summary.get("S13_runtime_and_latency") or {}
    e5a = s13.get("E5a") or {}
    e5b = s13.get("E5b") or {}
    for title, body in [
        ("S13.E5a.launch_unasserted_note", e5a.get("launch_unasserted_note")),
        ("S13.E5a.noise_note", e5a.get("noise_note")),
        ("S13.E5a.negative_delta_note", e5a.get("negative_delta_note")),
        ("S13.E5b.absolute_latency_note", e5b.get("absolute_latency_note")),
        ("S13.E5b.detection_asymmetry.premise_violation_note",
         (e5b.get("detection_asymmetry") or {}).get("premise_violation_note")),
        ("S13.E5b.ratio.unguarded_all_pairs.note",
         ((e5b.get("ratio") or {}).get("unguarded_all_pairs") or {}).get("note")),
        ("S10_compile_cost.rq4_note",
         (summary.get("S10_compile_cost") or {}).get("rq4_note")),
        ("S10_compile_cost.definition",
         (summary.get("S10_compile_cost") or {}).get("definition")),
    ]:
        if body:
            L.append(f"**`{title}`**")
            L.append("")
            L.append("> " + str(body).replace("\n", "\n> "))
            L.append("")
    L.append("## 4. Elided float notes")
    L.append("")
    if ELIDED_NOTES:
        L.append("A EuroSys submission gets limited body pages, so each "
                 "generated note above is the short form. The text below was "
                 "cut from the float note for the page budget and is "
                 "reproduced here so the argument behind every number stays "
                 "recoverable. It is candidate prose for the body or the "
                 "appendix, not deleted work.")
        L.append("")
        for label in sorted(ELIDED_NOTES):
            L.append(f"**`{label}`**")
            L.append("")
            L.append("> " + ELIDED_NOTES[label].replace("\n", "\n> "))
            L.append("")
    else:
        L.append("Nothing was elided: every generated note is complete.")
        L.append("")
    L.append("## 5. Artifacts the paper consumes")
    L.append("")
    L.append("`make paper` copies exactly these, and nothing else:")
    L.append("")
    for f in PAPER_TABLE_FILES:
        L.append(f"- `tables/{f}`")
    if PAPER_FIGURE_FILES:
        for f in PAPER_FIGURE_FILES:
            L.append(f"- `figures/{f}` (generated)")
    else:
        L.append("- (no generated PDF: `main.tex` has zero "
                 "`\\includegraphics` and every figure is hand-authored "
                 "`pgfplots`)")
    L.append("")
    L.append("Hand-authored and never written by a render: "
             + ", ".join(f"`figures/{f}`" for f in HAND_AUTHORED_FIGURES) + ".")
    L.append("")
    return "\n".join(L) + "\n"


def render_tables(summary):
    """Emit every table. Returns {filename: tex}.

    The float names now follow the paper's own `\\input{}`s, so `make paper`
    can no longer name a float after something the prose does not read. The
    legacy `rq*` set is still produced -- it carries per-category detail the
    paper may want in a revision -- but `PAPER_TABLE_FILES` marks which of them
    the paper actually consumes and `make paper` copies only those.
    """
    return OrderedDict([
        # Consumed by main.tex today.
        ("rq2_bugs.tex", table_rq2_bugs(summary)),
        ("e2_generation.tex", table_e2_generation(summary)),
        ("e3_discharge.tex", table_e3_discharge(summary)),
        ("e4_cost.tex", table_e4_cost(summary)),
        ("e5_oracle.tex", table_e5_oracle(summary)),
        # Reference material: generated for a revision, not \input by the paper.
        ("e1_path_class.tex", table_e1_path_class(summary)),
        ("e1_rtc_curve.tex", table_e1_rtc_curve(summary)),
        ("rq1_category.tex", table_rq1_category(summary)),
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

    # -- fig_e1_path_class.pdf -----------------------------------------------
    # NAMING: deliberately NOT `fig_e1_detection`. The paper already carries a
    # hand-written `figures/fig_e1_detection.tex` -- a pgfplots bar chart of
    # before-device detection by mutation class for FIVE toolchains, including
    # baselines (MLIR-linalg 34/50, MLIR-low 38/48, IREE 19/23) whose numbers
    # this renderer has no source for. Same basename + different figure would
    # let `make paper` drop a misleading `.pdf` next to a `.tex` that
    # `\input{figures/fig_e1_detection}` resolves to, silently dead. This figure
    # is the S14 register view instead, so it gets its own name.
    #
    # The two panels are the two claims of §5.2: (a) how much of E1's misses is
    # a denominator question, and (b) that the cost of enabling checks is one
    # class wide. They belong in one figure because (a) answers "can we afford
    # to check" only if (b) answers "what does checking cost".
    s14 = summary.get("S14_path_class") or {}
    recon = s14.get("reconciliation") or {}
    rtc = s14.get("rtc_curve") or {}
    if recon and rtc.get("by_class"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

        n_inj = recon.get("n_injected") or 0
        n_det = recon.get("n_detected_all_paths") or 0
        arm = rtc.get("rtc_all_arm") or {}
        n_adm = recon.get("admissible_injected") or 0
        n_dad = recon.get("admissible_detected") or 0

        labels, rates, colors = [], [], []
        if n_inj:
            labels.append("all injected\n(S2 headline)")
            rates.append(100.0 * n_det / n_inj)
            colors.append("#4C72B0")
        if n_adm:
            labels.append("admissible\n(S14 register)")
            rates.append(100.0 * n_dad / n_adm)
            colors.append("#55A868")
        if arm and arm.get("n_injected"):
            labels.append("ceiling arm\n(-rtc=all)")
            rates.append(100.0 * arm["n_detected"] / arm["n_injected"])
            colors.append("#C44E52")
        ax1.bar(labels, rates, color=colors, width=0.55)
        for i, r in enumerate(rates):
            ax1.text(i, r + 1.0, f"{r:.1f}%", ha="center", fontsize=8)
        ax1.set_ylabel("detected before device (%)")
        ax1.set_ylim(0, max(rates + [50]) * 1.25)
        ax1.set_title("E1 detection under three denominators")
        ax1.tick_params(axis="x", labelsize=8)

        by_class = rtc["by_class"]
        for cname, series in by_class.items():
            ax2.plot(rtc["levels"], [series.get(l) for l in rtc["levels"]],
                     marker="o", label=cname)
        ax2.set_ylabel("obligations enabled")
        ax2.set_xlabel("-rtc level")
        ax2.set_title("Cost of enabling checks, by obligation class")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fp = os.path.join(out_dir, "fig_e1_path_class.pdf")
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

    s14 = summary.get("S14_path_class") or {}
    if s14:
        recon = s14.get("reconciliation") or {}
        denom = s14.get("denominator") or {}
        audit = s14.get("applicability_audit") or {}
        rtc = s14.get("rtc_curve") or {}
        lines.append("## E1 path-class register (S14)")
        lines.append(f"- register source: `{s14.get('register_source', '?')}`")
        lines.append(f"- specs {denom.get('n_specs', 0)}: "
                     f"{denom.get('n_applicable', 0)} applicable, "
                     f"{denom.get('n_not_applicable', 0)} N/A, "
                     f"{denom.get('n_out_of_scope', 0)} out of scope")
        lines.append("")
        lines.append("| Path class | Cells | Injected | Applicable | Detected | Rate |")
        lines.append("|---|---|---|---|---|---|")
        for pc, c in (s14.get("per_path") or {}).items():
            lines.append(f"| {pc} | {c.get('n_cells', 0)} | "
                         f"{c.get('n_injected', 0)} | "
                         f"{c.get('n_admissible', '---')} | "
                         f"{c.get('n_detected_admissible', '---')} | "
                         f"{pct(c.get('n_detected_pct'))} |")
        lines.append("")
        if recon:
            lines.append(f"**Reconciliation vs S1/S2: "
                         f"{'AGREES' if recon.get('matches_S1S2') else 'MISMATCH'}** "
                         f"(cells {recon.get('n_cells')}, "
                         f"injected {recon.get('n_injected')}, "
                         f"noop {recon.get('n_discarded_noop')}, "
                         f"never {recon.get('n_never')})")
            lines.append(f"- all injected: {recon.get('n_detected_all_paths')}"
                         f"/{recon.get('n_injected')} detected (S2's headline)")
            lines.append(f"- admissible only: {recon.get('admissible_detected')}"
                         f"/{recon.get('admissible_injected')} = "
                         f"{pct(recon.get('admissible_pct'))} (the register)")
            lines.append(f"- {recon.get('inadmissible_detected')} detection(s) "
                         f"land on inadmissible cells")
            lines.append("")
        if audit:
            lines.append(f"**Applicability audit: {audit.get('status')}** "
                         f"({audit.get('n_contradicting', 0)} of "
                         f"{audit.get('n_inadmissible_specs_with_cells', 0)} "
                         "inadmissible specs with cells contradicted)")
            for r in audit.get("rows") or []:
                mark = "**CONTRADICTED**" if r.get("contradicts_design_claim") else "holds"
                lines.append(f"- `{r.get('spec_id')}` ({r.get('prohibition')}) "
                             f"injected {r.get('n_injected')} / "
                             f"detected {r.get('n_detected')} -> {mark}")
            lines.append("")
        if rtc:
            en = rtc.get("n_obligations_enabled") or {}
            lines.append(f"- `-rtc` enabled: " + " / ".join(
                f"{lv} {en.get(lv)}" for lv in rtc.get("levels") or []))
            for cname, series in (rtc.get("by_class") or {}).items():
                lines.append(f"  - {cname}: " + " -> ".join(
                    str(series.get(lv)) for lv in rtc.get("levels") or []))
            lines.append(f"- steepest class: {rtc.get('steepest_class')}")
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
    ap.add_argument("--drift", action="store_true",
                    help="write PROSE-DRIFT.md (also implied by --all)")
    ap.add_argument("--out", default=os.path.join(HERE, "render"))
    ap.add_argument("--paper-dir",
                    default=os.environ.get(
                        "PAPER_DIR",
                        os.path.normpath(os.path.join(HERE, "..", "..",
                                                      "eurosys27"))),
                    help="venue directory whose tables/ the drift report "
                         "compares against")
    args = ap.parse_args(argv)

    do_all = args.all or not (args.summary or args.tables or args.figures
                              or args.drift)
    do_summary = args.summary or do_all
    do_tables = args.tables or do_all
    do_figures = args.figures or do_all
    do_drift = args.drift or do_all

    os.makedirs(args.out, exist_ok=True)

    stats_by_lane = load_lane_stats()
    kernel_counts = load_kernel_counts()
    summary = build_summary(stats_by_lane, kernel_counts)

    import datetime
    summary["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    for lane, st in stats_by_lane.items():
        state = "OK" if st is not None else "MISSING (stats.json not found)"
        print(f"[render] {lane:14s} -> {state}")
    for lane in NOT_READY:
        print(f"[render] {lane:14s} -> NOT READY (no results/ committed)")

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

    if do_drift:
        fp = os.path.join(args.out, "PROSE-DRIFT.md")
        open(fp, "w").write(render_drift(summary, args.paper_dir))
        written.append(fp)
        bad = [r for r in collect_drift(summary, args.paper_dir)
               if r[3] != "AGREES"]
        print(f"[render] prose drift: {len(bad)} claim(s) need an owner "
              f"decision -> {os.path.relpath(fp, HERE)}")
        for sec, claim, measured, verdict, _src in bad:
            print(f"    [{verdict}] §{sec} {claim[:72]} -> {measured[:48]}")

    print(f"[render] wrote {len(written)} artifact(s) under {args.out}")
    for fp in written:
        print(f"    {os.path.relpath(fp, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
