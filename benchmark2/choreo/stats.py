#!/usr/bin/env python3
"""Aggregate `benchmark2/results/choreo/*.jsonl` into `stats.json`
(S1-S7, S10, S13, S14).

--------------------------------------------------------------------------
WHAT THIS IS
--------------------------------------------------------------------------
`run.sh stats` -> here. The lane contract in schema/statistics-manifest.md says
the choreo worker owns S1-S7, S10 and S13, and that "the integrator writes no
number into main.tex that is not derivable from some worker's stats.json". So
this file is the last step before a number becomes a paper claim, and its job is
to make every claim traceable or explicitly untraceable.

S14 is ADDITIVE and is a v2.1 arrival, so it is not in that lane list. It
registers the path-class model: how many specs choreo was FAIR to ask about,
and for the rest, the recorded reason it was not asked. It must never feed back
into S1/S2 -- specs/expansion-workflow.md §7 reserves any change to the
detection denominator for v3 with owner sign-off, and the frozen E1 baseline
depends on that.

--------------------------------------------------------------------------
INPUT: the COLLECTED records, not the raw lane output
--------------------------------------------------------------------------
stats.py reads `benchmark2/results/choreo/*.jsonl` -- the projection collect.py
validated against schema/record-schema.json -- and NOT `raw/e2_ledger.json` or
`raw/e4_compile_cost.json`.

That ordering is deliberate. collect.py is the only place schema conformance is
enforced, so reading its output means a malformed lane cannot reach a paper
number without first failing loudly there. Reading raw/ directly would let a
lane emit an out-of-enum `outcome` and have stats.py silently bucket it as
"other", producing a total that no longer adds up -- the exact failure the
manifest's cross-cutting-fields rule exists to prevent.

Consequence: `run.sh collect` must precede `run.sh stats`. cmd_all does this,
twice (once before E5, once after), because E5 adds residue/latency records.

--------------------------------------------------------------------------
THE RULE THIS FILE IS BUILT AROUND: never emit a zero for missing data
--------------------------------------------------------------------------
A statistic whose lane has not run is emitted as

    {"status": "not_run", "reason": "...", "expected_records": "mutant"}

and NEVER as a table of zeros. The two are indistinguishable to a reader who
only sees the number, and the difference is the whole content of the claim:
"S2 = 0 before-device detections" is a catastrophic result, while "S2 was not
measured" is a schedule fact. plan §4.2 makes the same ruling for toolchains
that cannot express a class -- recorded as "not suitable for generation
comparison", "not given a fabricated zero" -- and this file applies it to
missing lanes.

The same rule covers S7, which is not missing but UNMEASURABLE: no flag in the
pinned build disables interval reasoning (-rtc=none, --disable-runtime-check and
-zero-cost are all ledger-identical; they gate guard EMISSION, not assessment).
S7 is emitted as {"status": "not_measurable"} with that reason and with the
inherited 93.2%->77.2% figure labelled as inherited, so the integrator can see
it was never re-derived here rather than discovering it at review time.

--------------------------------------------------------------------------
PROVENANCE
--------------------------------------------------------------------------
Every statistic carries `toolchain_version` and the `settings_hash` values it
was computed from. `settings_hash` is per-CATEGORY in this suite (one
`settings/<cat>.md` per operator), so a grand-total statistic legitimately has
several; those are listed rather than collapsed to one, because collapsing them
would hide which operator settings the total spans.

`toolchain_version` is the commit the BINARY was built from, per toolchain.py --
not the checkout HEAD. If the two disagree, `toolchain_stale` is set here too,
so the integrator sees it without having to open the run log.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

# NOTE: no `import re` here. Every regex this file uses is an ALIAS of run_e5's
# (see the block above `launch_status`). Keeping a local `re` import invited the
# reader to assume local patterns still existed, which is how the two divergent
# copies of `sanitizer_verdict` went unnoticed in the first place.
import toolchain
import run_e5          # noqa: E402  ONE definition of the launch/sanitizer
                       # classifier, shared with the instrument that produced
                       # the data. See the note at `sanitizer_verdict` below.

HERE = os.path.dirname(os.path.realpath(__file__))
B2 = os.path.dirname(HERE)                       # benchmark2/
RESULTS = os.path.join(B2, "results", "choreo")
RAW = os.path.join(HERE, "raw")

TOOLCHAIN = "choreo"

# Which record type each statistic needs. Used to distinguish "not run" from
# "run and empty" -- the former means the lane never produced its file.
NEEDS = {
    "S1": "mutant", "S2": "mutant",
    "S3": "obligation", "S4": "obligation", "S5": "obligation",
    "S6": "obligation", "S7": "obligation",
    "S10": "cost", "S13a": "residue", "S13b": "latency",
    "S14": "mutant",
}

# plan §6 outcome taxonomy. `never` is the one that matters: S2 asserts it is 0.
OUTCOMES = ("compile", "runtime", "never", "n/a")
CLASSES = ("M1", "M2", "M3")
OBL_CLASSES = ("elem", "shape", "loop", "hw")
OBL_OUTCOMES = ("proven", "refuted", "runtime", "budgeted")
MECHANISMS = ("canonical", "interval", "direct")

# The paper figures these statistics are checked against. Recorded so the report
# can print a comparison instead of leaving the reader to diff by hand. None of
# these is ever WRITTEN into stats.json as if measured -- they live under
# `paper_reference` and are labelled as such.
PAPER_REF = {
    "S3_grand_total": 17717,
    "S5_discharge_pct": 93.2,
    "S5_static_pct": 99.2,
    "S5_dynamic_pct": 87.9,
    "S4_layer_norm": {"expressed": 79, "discharged": 74, "runtime": 5, "dropped": 0},
    "S7_inherited": {"with_interval_pct": 93.2, "without_interval_pct": 77.2},
    "S10_rq4_median_pct": 0.6,
    "S2_abstract": "210/210",
}

S7_REASON = (
    "No flag in the pinned build disables interval reasoning. `-rtc=none`, "
    "`--disable-runtime-check` and `-zero-cost` are all ledger-identical "
    "(verified: 72 obligations, 5 runtime, on layer_normalization/3_attention) "
    "because they gate guard EMISSION, not assessment. The counterfactual "
    "therefore cannot be re-derived here; the 93.2%->77.2% figure is INHERITED "
    "from the existing ablation, not measured by this lane."
)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_jsonl(path):
    """Read a JSONL file. Returns [] if absent -- callers decide what that means."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"*** {os.path.basename(path)}:{i} is not valid JSON: {e}",
                      file=sys.stderr)
    return out


def load_all(results_dir):
    """Load every record type. Missing files yield empty lists, and `present`
    records which files existed at all -- that is what separates "lane not run"
    from "lane ran and found nothing"."""
    names = ("kernel", "mutant", "spec", "obligation", "sanitizer",
             "expressibility", "remainder", "cost", "residue", "latency")
    data, present = {}, {}
    for n in names:
        p = os.path.join(results_dir, f"{n}.jsonl")
        present[n] = os.path.exists(p)
        data[n] = load_jsonl(p)
    return data, present


def not_run(what, reason):
    """The shape emitted when a statistic cannot be computed. See the module
    docstring for why this is never a table of zeros."""
    return {"status": "not_run", "feeds": what, "reason": reason,
            "expected_records": None}


def prov(records, key="settings_hash"):
    """Sorted unique settings_hash / toolchain_version across a record set.

    Listed rather than collapsed: settings_hash is per-category, so a grand total
    spans several, and naming them is what lets a reviewer see which operator
    settings the total covers.
    """
    sh = sorted({r.get(key) for r in records if r.get(key) is not None})
    tv = sorted({r.get("toolchain_version") for r in records
                 if r.get("toolchain_version")})
    return {"settings_hash": sh, "toolchain_version": tv}


def pct(num, den):
    """Percentage, or None when the denominator is 0.

    Returning None rather than 0.0 matters: a 0% discharge rate over an empty
    set is not a measurement, and printing it would put a fabricated number in
    the report.
    """
    return round(100.0 * num / den, 4) if den else None


# --------------------------------------------------------------------------
# S1 / S2 -- detection matrix, from E1's `mutant` records
# --------------------------------------------------------------------------
def s1_s2(mutants, present):
    """S1 detection matrix + S2 before-device total.

    S1 is per class x toolchain: n_injected, n_compile, n_runtime, n_never, n_na.
    S2 is Sum(n_compile + n_runtime) per class.

    R-D2 (coordinator ruling 2026-09-10): the old `never = 0` acceptance criterion
    is **RETIRED**. It was a Phase-0 gate calibrated against a 210-kernel corpus
    and predates per-cause attribution. The criterion is now **no *unattributed*
    `never`** -- every never-detected mutant must carry one of the five recorded
    causes. A non-zero `never` is therefore not a failure; an unattributed one is.
    The per-cause table and the R-D2b flag-matrix ablation are emitted alongside so
    §5.2 ¶2 can be written from this block alone.

    `noop` mutants are excluded from n_injected. specs §11.1: a mutant whose own
    manifest check says it does not corrupt anything is a false success and is
    discarded, so counting it as an injection would credit choreo with detecting
    a non-bug. They are reported separately as n_discarded_noop so the exclusion
    is visible instead of silent.
    """
    if not present.get("mutant"):
        r = not_run("tab:rq2-bugs / abstract",
                    "E1 has not been run: results/choreo/mutant.jsonl is absent. "
                    "Run `run.sh minimal` first.")
        r["expected_records"] = "mutant"
        return r, r

    # A mutant is only countable if its ground truth is decided. specs §7 makes
    # the oracle mandatory for every `never` and every `runtime` outcome; an
    # undecidable manifest means the row cannot be interpreted at all.
    rows = collections.defaultdict(lambda: collections.Counter())
    undecidable = []
    for m in mutants:
        cls = m.get("class")
        if m.get("manifest") == "noop":
            rows[cls]["n_discarded_noop"] += 1
            continue
        if m.get("manifest") not in ("corrupts",):
            undecidable.append(m.get("mutant_id"))
        rows[cls]["n_injected"] += 1
        oc = m.get("outcome")
        if oc in OUTCOMES:
            rows[cls][f"n_{oc}"] += 1
        else:
            rows[cls]["n_unmapped_outcome"] += 1
        # `stage` is choreo-only (specs §6): where the detection happened.
        st = m.get("stage")
        if st:
            rows[cls][f"stage_{st}"] += 1

    # FEEDS REMAP, binding owner decision 2026-09-09 (experiment-redesign-plan.md
    # addendum §1): the paper's evaluation spine is now E1-E5, and S1+S12+S2 feed
    # the E1 detection matrix, which REPLACES `tab:rq2-bugs`. Labels below name the
    # E-axis and §5 subsection so the integrator regenerates the right float.
    s1 = {"status": "ok",
          "feeds": "E1 detection matrix (§5.2 ¶2; replaces tab:rq2-bugs)",
          "per_class": {}, "totals": collections.Counter()}
    for cls in CLASSES:
        c = rows.get(cls, collections.Counter())
        d = {k: int(v) for k, v in sorted(c.items())}
        # Every enum value gets a key, even at 0, so a missing class reads as
        # zero detections rather than as an absent row.
        for oc in OUTCOMES:
            d.setdefault(f"n_{oc}", 0)
        d["n_injected"] = int(c.get("n_injected", 0))
        s1["per_class"][cls] = d
        for k, v in d.items():
            s1["totals"][k] += v
    s1["totals"] = {k: int(v) for k, v in sorted(s1["totals"].items())}
    s1["undecidable_manifest_mutants"] = sorted(undecidable)
    s1.update(prov(mutants))

    # ---- S2 -------------------------------------------------------------
    t = s1["totals"]
    before_device = t.get("n_compile", 0) + t.get("n_runtime", 0)
    never = t.get("n_never", 0)
    s2 = {
        "status": "ok",
        "feeds": ('E1 detection matrix (§5.2 ¶2) — the "210/210" headline is '
                  'DROPPED (R-D2); criterion is now "no unattributed never"'),
        "n_injected": t["n_injected"],
        "n_before_device": before_device,
        "n_compile": t.get("n_compile", 0),
        "n_runtime": t.get("n_runtime", 0),
        "n_never": never,
        "n_na": t.get("n_na", 0),
        "n_discarded_noop": t.get("n_discarded_noop", 0),
        "pct_before_device": pct(before_device, t["n_injected"]),
        # RETIRED by R-D2, kept only so the old artifact stays comparable and so
        # the ruling's effect is auditable. This is NOT the acceptance criterion
        # any more and must not be printed as a pass/fail. See
        # `no_unattributed_never` below for the criterion that replaced it.
        "never_is_zero": never == 0,
        "never_is_zero_status": (
            "RETIRED (R-D2, 2026-09-10) — a non-zero `never` is no longer a "
            "failure; an UNATTRIBUTED one is. Do not report this as pass/fail."),
        "per_class_before_device": {
            cls: (s1["per_class"][cls].get("n_compile", 0)
                  + s1["per_class"][cls].get("n_runtime", 0))
            for cls in CLASSES if cls in s1["per_class"]
        },
    }
    s2.update(prov(mutants))

    # ---- R-D2: the criterion that replaced `never = 0` ------------------
    # `never_cause` is stamped onto the register record by collect.py from
    # raw/never_attribution.json (analyze_never.py). stats.py reads only the
    # register, so the criterion is evaluable from committed data alone.
    never_rows = [m for m in mutants
                  if m.get("outcome") == "never" and m.get("manifest") != "noop"]
    if never_rows:
        causes = collections.Counter(r.get("never_cause") for r in never_rows)
        unattributed = sorted(r.get("mutant_id") for r in never_rows
                              if not r.get("never_cause"))
        # The five recorded causes, grouped exactly as R-D2's §5.2 ¶2 wording does:
        # 25 out-of-scope/not-assessed-by-construction + 12 hoisting/config.
        OUT_OF_SCOPE_OR_NOT_ASSESSED = ("C4_NOT_ASSESSED", "C5_OUT_OF_SCOPE")
        TOOLCHAIN_OR_CONFIG = ("HOISTING_DEFECT", "C3_LOWER_BOUND_OMITTED",
                               "C1_COST_FILTER_SUPPRESSED")
        s2["never_attribution"] = {
            "criterion": "no unattributed `never` (R-D2, 2026-09-10; replaces the "
                         "retired `never = 0`)",
            "no_unattributed_never": not unattributed,
            "n_unattributed": len(unattributed),
            "unattributed_mutants": unattributed,
            "cause_counts": {k: int(v) for k, v in sorted(causes.items(),
                                                           key=lambda kv: -kv[1])
                             if k is not None},
            "n_out_of_scope_or_not_assessed": sum(
                int(causes.get(c, 0)) for c in OUT_OF_SCOPE_OR_NOT_ASSESSED),
            "n_toolchain_or_config": sum(
                int(causes.get(c, 0)) for c in TOOLCHAIN_OR_CONFIG),
            # R-D2b: the 2x2 flag matrix (-rtc x hoist) is an ABLATION, not a
            # re-pin. The pinned default stays the headline; this shows the
            # hoisting defect's recoverable cost.
            "flag_matrix_ablation": {
                "status": "R-D2b — report as sensitivity, pinned default is the "
                          "headline",
                "at_pinned_default": f"{before_device}/{t['n_injected']} = "
                                     f"{pct(before_device, t['n_injected'])}%",
                "with_rtc_all_and_hoist_disabled": "43/72 = 59.72%",
                "recovered_by_flags": 8,
                "source": "raw/c2_probe_groundtruth.json (s2_implication)",
            },
            "toolchain_bugs_to_log": {
                "HOISTING_DEFECT": "guard emitted but hoisted past the induction "
                                   "variable's reset — VACUOUS, confirmed by "
                                   "--disable-assert-hoist. R-D2: log as a "
                                   "toolchain bug, do NOT hide behind a flag.",
                "C3_LOWER_BOUND_OMITTED": "only one of the index's two bounds was "
                                          "generated — croqtile soundness bug.",
            },
        }
        s2["never_note"] = (
            f"{never} mutant(s) reached outcome=never. Under R-D2 this is NOT a "
            f"failure provided every one is attributed: "
            f"{len(unattributed)} unattributed. "
            f"{s2['never_attribution']['n_out_of_scope_or_not_assessed']} are "
            f"out-of-scope or not-assessed-by-construction and "
            f"{s2['never_attribution']['n_toolchain_or_config']} are attributable "
            f"to a hoisting defect plus cost-filter/lower-bound configuration."
        )
        s2["never_mutants"] = [
            {"mutant_id": m.get("mutant_id"), "class": m.get("class"),
             "category": m.get("category"),
             "detector": m.get("detector"),
             "manifest": m.get("manifest"),
             "never_cause": m.get("never_cause"),
             "never_cause_reason": m.get("never_cause_reason"),
             "note": m.get("note")}
            for m in never_rows
        ]
    else:
        s2["never_attribution"] = {
            "criterion": "no unattributed `never` (R-D2, 2026-09-10)",
            "no_unattributed_never": True, "n_unattributed": 0,
            "unattributed_mutants": [], "cause_counts": {},
            "note": "no `never` outcomes in this mutant set",
        }
    return s1, s2


# --------------------------------------------------------------------------
# S3-S7 -- generation, from E2's `obligation` records
# --------------------------------------------------------------------------
def s3(obligations):
    """Generation totals: per class, per category, and grand total.

    Feeds tab:gen-capability and the abstract's "17,717". The measured grand
    total is reported beside the paper figure rather than being adjusted to
    match it -- a delta here is a suite-version fact for the coordinator.
    """
    if not obligations:
        r = not_run("tab:gen-capability / abstract", "no obligation records")
        r["expected_records"] = "obligation"
        return r

    by_class = collections.Counter()
    by_cat = collections.Counter()
    by_class_cat = collections.defaultdict(collections.Counter)
    unmapped = collections.Counter()
    for o in obligations:
        cls, cat = o.get("class"), o.get("category")
        by_class[cls] += 1
        by_cat[cat] += 1
        by_class_cat[cat][cls] += 1
        if cls not in OBL_CLASSES:
            unmapped[cls] += 1

    grand = len(obligations)
    return {
        "status": "ok",
        "feeds": ('E2 generation completeness (§5.3 ¶1; tab:gen-capability, '
                  'abstract "17,353" — R-D5 re-registered from 17,717)'),
        "grand_total": grand,
        "per_class": {c: int(by_class.get(c, 0)) for c in OBL_CLASSES},
        "per_category": {c: int(n) for c, n in sorted(by_cat.items())},
        "per_category_per_class": {
            c: {k: int(by_class_cat[c].get(k, 0)) for k in OBL_CLASSES}
            for c in sorted(by_class_cat)
        },
        "unmapped_class_values": {str(k): int(v) for k, v in unmapped.items()},
        "paper_reference": {"grand_total": PAPER_REF["S3_grand_total"],
                            "delta": grand - PAPER_REF["S3_grand_total"]},
        **prov(obligations),
    }


def s4(obligations, kernels):
    """Per-operator ledger: expressed / discharged / runtime / budgeted / dropped.

    Feeds tab:per-operator.

    `discharged` = proven + refuted: an obligation choreo settled at compile
    time. `runtime` = a guard actually emitted. `budgeted` = resolved as needing
    a runtime guard but suppressed by the cost filter. `remainder` = runtime +
    budgeted = everything not discharged statically. `dropped` = expressed minus
    (discharged + remainder), which MUST be 0 -- a nonzero value means an
    obligation vanished between assessment and emission, i.e. silently unsafety.

    THE LAYER_NORM CRITERION IS PER-KERNEL, NOT PER-CATEGORY. The acceptance
    criterion pins `layer_norm` = 79/74/5/0, and 79 is the obligation count of
    ONE kernel (layer_normalization/3_attention: 74 proven + 1 runtime +
    4 budgeted, remainder 5, dropped 0) -- not the 1825 obligations of all 21
    layer_normalization kernels. So the criterion is checked against the single
    matching kernel, and the per-category table is reported separately. Checking
    it against the category aggregate would make the criterion fail forever, and
    a criterion that always fails trains the reader to ignore failures.

    Zero-obligation kernels are INCLUDED (as expressed=0). All 21 are reshape/*:
    a reshape is a pure span reinterpretation with no element access to bound, so
    there is genuinely nothing to assess. Showing them keeps "nothing to check"
    distinguishable from "skipped", and keeps the kernel count at 310 rather than
    silently dropping to 289.
    """
    if not obligations:
        r = not_run("tab:per-operator", "no obligation records")
        r["expected_records"] = "obligation"
        return r

    cats = collections.defaultdict(collections.Counter)
    kern = collections.defaultdict(collections.Counter)
    mech_by_cat = collections.defaultdict(collections.Counter)
    shape_by_cat = collections.defaultdict(collections.Counter)
    for o in obligations:
        c, k, oc = o.get("category"), o.get("kernel_id"), o.get("outcome")
        for agg, key in ((cats, c), (kern, k)):
            agg[key]["expressed"] += 1
            if oc in ("proven", "refuted"):
                agg[key]["discharged"] += 1
            agg[key][oc if oc in OBL_OUTCOMES else "unmapped"] += 1
        mech_by_cat[c][o.get("mechanism")] += 1
        shape_by_cat[c][o.get("shape_class")] += 1

    def ledger(n):
        expressed = n["expressed"]
        discharged = n.get("proven", 0) + n.get("refuted", 0)
        remainder = n.get("runtime", 0) + n.get("budgeted", 0)
        return {
            "expressed": expressed,
            "discharged": discharged,
            "proven": n.get("proven", 0),
            "refuted": n.get("refuted", 0),
            "runtime": n.get("runtime", 0),
            "budgeted": n.get("budgeted", 0),
            "remainder": remainder,
            "dropped": expressed - discharged - remainder,
            "unmapped_outcome": n.get("unmapped", 0),
            "discharge_pct": pct(discharged, expressed),
        }

    per_cat = {}
    for c in sorted(cats):
        per_cat[c] = ledger(cats[c])
        per_cat[c]["mechanism"] = {m: int(mech_by_cat[c].get(m, 0))
                                   for m in MECHANISMS}
        per_cat[c]["shape_class"] = {k: int(v)
                                     for k, v in sorted(shape_by_cat[c].items())
                                     if k}

    # Fold in zero-obligation kernels so the category table spans all 310.
    # `ledger(Counter())` yields an all-zeros row: Counter returns 0 for absent
    # keys, so an empty aggregate is a valid input rather than a KeyError.
    for c in per_cat:
        per_cat[c]["zero_obligation_kernels"] = []
    zero = []
    for k in kernels:
        kid, cat = k.get("kernel_id"), k.get("category")
        if kid in kern:
            continue
        zero.append(kid)
        row = per_cat.setdefault(cat, ledger(collections.Counter()))
        row.setdefault("zero_obligation_kernels", []).append(kid)
    for c in per_cat:
        per_cat[c]["zero_obligation_kernels"].sort()

    tot = collections.Counter()
    for d in per_cat.values():
        for kk in ("expressed", "discharged", "proven", "refuted", "runtime",
                   "budgeted", "remainder", "dropped", "unmapped_outcome"):
            tot[kk] += d[kk]

    # ---- the per-kernel layer_norm criterion ----------------------------
    want = PAPER_REF["S4_layer_norm"]   # expressed/discharged/runtime/dropped
    # `runtime` in the criterion is the REMAINDER (runtime + budgeted), so map it
    # onto the ledger's `remainder` field for the comparison.
    ln_check = None
    matches = [(kid, ledger(n)) for kid, n in kern.items()
               if n["expressed"] == want["expressed"]
               and kid.split("/")[0] in ("layer_normalization", "layer_norm")]
    if len(matches) == 1:
        kid, d = matches[0]
        got = {"expressed": d["expressed"], "discharged": d["discharged"],
               "runtime": d["remainder"], "dropped": d["dropped"]}
        ln_check = {
            "kernel_id": kid,
            "grain": "per-kernel (79 is one kernel's obligation count, not the "
                     "category's 1825)",
            "expected": want,
            "measured": got,
            "runtime_field_is_remainder": "runtime+budgeted = 1+4 = 5",
            "matches": all(got[k] == want[k] for k in want),
        }
    elif len(matches) > 1:
        ln_check = {"status": "ambiguous",
                    "reason": f"{len(matches)} layer_norm kernels have "
                              f"{want['expressed']} obligations",
                    "candidates": [m[0] for m in matches]}
    else:
        ln_check = {"status": "not_found",
                    "reason": f"no layer_norm kernel has exactly "
                              f"{want['expressed']} obligations",
                    "expected": want}

    return {
        "status": "ok",
        "feeds": ("E2 generation completeness (§5.3 ¶1; the worked "
                  "per-operator counts that anchor the running example)"),
        "per_category": per_cat,
        "totals": dict(tot),
        "n_kernels_with_obligations": len(kern),
        "n_zero_obligation_kernels": len(zero),
        "zero_obligation_kernel_ids": sorted(zero),
        "dropped_is_zero_everywhere": all(d["dropped"] == 0
                                          for d in per_cat.values()),
        "layer_norm_criterion": ln_check,
        **prov(obligations),
    }


def s5(obligations, kernels):
    """Discharge rate, split static-shape vs dynamic-shape.

    Feeds RQ1 and the "93.2%", "99.2%/87.9%" figures. The split is the point:
    a single blended rate hides that static kernels discharge essentially
    everything while dynamic ones leave a residue, which is the mechanism the
    paper's story rests on.

    `shape_class` comes from run_e2's conditional-aware signature classifier, not
    from the filename -- the suite's filenames do not reliably encode extents
    (conv2d/10_dynamic is named 32x128x112x112 but declares [8,128,16,16]).

    KERNEL COUNTS COME FROM `kernels`, NOT FROM THE OBLIGATIONS. The first cut
    derived the per-shape-class kernel count from the set of `kernel_id`s present
    in the obligation records, which silently dropped the 21 zero-obligation
    reshape/* kernels and reported static 141 / dynamic 148 = 289 instead of the
    true static 151 / dynamic 159 = 310. The PERCENTAGES are unaffected (a kernel
    with 0 obligations adds 0 to both numerator and denominator), but a reader who
    checks "141 + 148 = 289 ≠ 310" would rightly distrust the whole table. So the
    counts are taken from the kernel records, which carry `shape_class` on all 310,
    and the obligation aggregates are keyed by the same classifier.
    """
    if not obligations:
        r = not_run("RQ1", "no obligation records")
        r["expected_records"] = "obligation"
        return r

    agg = collections.defaultdict(lambda: collections.Counter())
    for o in obligations:
        sc = o.get("shape_class") or "unknown"
        agg[sc]["expressed"] += 1
        if o.get("outcome") in ("proven", "refuted"):
            agg[sc]["discharged"] += 1

    # Kernel counts by shape class, from the kernel records (all 310, including
    # the 21 with zero obligations).
    kern = collections.Counter()
    for k in kernels:
        kern[k.get("shape_class") or "unknown"] += 1

    def block(sc):
        a = agg.get(sc, collections.Counter())
        return {"kernels": int(kern.get(sc, 0)),
                "expressed": int(a.get("expressed", 0)),
                "discharged": int(a.get("discharged", 0)),
                "remainder": int(a.get("expressed", 0) - a.get("discharged", 0)),
                "discharge_pct": pct(a.get("discharged", 0),
                                     a.get("expressed", 0))}

    allb = {
        "kernels": len(kernels),
        "expressed": len(obligations),
        "discharged": sum(1 for o in obligations
                          if o.get("outcome") in ("proven", "refuted")),
    }
    allb["remainder"] = allb["expressed"] - allb["discharged"]
    allb["discharge_pct"] = pct(allb["discharged"], allb["expressed"])

    return {
        "status": "ok",
        "feeds": ('E3 discharge strength (§5.4 ¶1; RQ1 "93.04%", '
                  '"100.0%/86.87%", residue 1,208 — R-D5 re-registered from '
                  '93.2%/99.2%/87.9%/1,199)'),
        "static": block("static"),
        "dynamic": block("dynamic"),
        "all": allb,
        "paper_reference": {
            "all_pct": PAPER_REF["S5_discharge_pct"],
            "static_pct": PAPER_REF["S5_static_pct"],
            "dynamic_pct": PAPER_REF["S5_dynamic_pct"],
        },
        **prov(obligations),
    }


def s6(obligations):
    """Mechanism split per outcome: canonical / interval / direct.

    Feeds tab:rq6-mechanism. `direct` never appears in the raw ledger -- it is
    the stats-derived direct static checks that bypass the assessor, recovered by
    run_e2 as first-class records. Without them the per-class totals undercount
    (72 instead of 79 on layer_normalization/3_attention), so the mechanism
    split would silently omit the whole direct category.
    """
    if not obligations:
        r = not_run("tab:rq6-mechanism", "no obligation records")
        r["expected_records"] = "obligation"
        return r

    by_out = collections.defaultdict(collections.Counter)
    unmapped = collections.Counter()
    for o in obligations:
        oc, m = o.get("outcome"), o.get("mechanism")
        by_out[oc][m] += 1
        if m not in MECHANISMS:
            unmapped[str(m)] += 1

    per_outcome = {}
    for oc in OBL_OUTCOMES:
        c = by_out.get(oc, collections.Counter())
        per_outcome[oc] = {m: int(c.get(m, 0)) for m in MECHANISMS}
        per_outcome[oc]["total"] = sum(per_outcome[oc].values())
    # Any outcome value outside the enum still has to be visible.
    for oc, c in by_out.items():
        if oc not in OBL_OUTCOMES:
            per_outcome[f"UNMAPPED:{oc}"] = {m: int(c.get(m, 0))
                                             for m in MECHANISMS}

    totals = collections.Counter()
    for o in obligations:
        totals[o.get("mechanism")] += 1

    return {
        "status": "ok",
        # RQ6 is FOLDED INTO E3 (owner decision 2026-09-09): `tab:rq6-mechanism`
        # and `fig:mechanism` are CUT from the paper. The statistic is still
        # REQUIRED — it feeds one E3 sentence (0 of 16,518 static discharges used
        # pure scalar-symbolic reasoning). Prose only; do not build the float.
        "feeds": ("E3 mechanism-attribution ¶ (§5.4 ¶3, prose only — "
                  "tab:rq6-mechanism and fig:mechanism CUT)"),
        "per_outcome": per_outcome,
        "totals": {m: int(totals.get(m, 0)) for m in MECHANISMS},
        "unmapped_mechanism_values": {k: int(v) for k, v in unmapped.items()},
        **prov(obligations),
    }


def s7(obligations):
    """No-interval counterfactual. NOT MEASURABLE on the pinned build.

    Emitted as not_measurable with the reason and with the inherited figure
    labelled inherited. Reporting 93.2%->77.2% from this lane without that label
    would present an inherited number as a measured one.
    """
    interval = sum(1 for o in obligations if o.get("mechanism") == "interval")
    return {
        "status": "not_measurable",
        "feeds": ('E3 mechanism-attribution ¶ (§5.4 ¶3; "93.2%->77.2%" is '
                  'INHERITED, not re-derived — R-D5/S7 not_measurable on this '
                  'build; interval-discharged 2,920 per S6)'),
        "reason": S7_REASON,
        "inherited_from_paper": PAPER_REF["S7_inherited"],
        "what_would_be_needed": (
            "a build flag that disables interval/bounded-type reasoning in the "
            "assessor while leaving canonical normalization on. No such flag "
            "exists at this commit; -rtc/--disable-runtime-check/-zero-cost all "
            "produce a byte-identical ledger."
        ),
        # The upper bound on what the counterfactual could remove, measured: how
        # many obligations were discharged BY interval. If interval reasoning
        # were disabled, at most these would move out of `proven`.
        "interval_discharged_upper_bound": interval,
        "interval_pct_of_all": pct(interval, len(obligations)) if obligations else None,
        **(prov(obligations) if obligations else {}),
    }


# --------------------------------------------------------------------------
# S10 -- compile cost
# --------------------------------------------------------------------------
def s10(costs):
    """Median compile-overhead % per category + grand median. Feeds RQ4.

    A median without n and spread is not checkable, so each category carries
    n_kernels, min, max and mean alongside it. The per-kernel values are also
    listed: S10 is a median of MEDIANS, and a reviewer needs the underlying
    distribution to judge whether one 224-second nvcc outlier is driving a
    category (it is -- see the note this function emits).

    Records with `exclusive` false are flagged. plan §12.6 rule 2 says qc
    rejects any `cost` record carrying exclusive=false, so a run that produced
    them is inadmissible and this statistic says so instead of quietly
    aggregating rows that will be thrown away.
    """
    if not costs:
        r = not_run("RQ4", "no cost records; run `run.sh e4`")
        r["expected_records"] = "cost"
        return r

    # `exclusive` is a STRING enum in the schema ("true"/"false") but E4 wrote a
    # JSON boolean before collect.py coerced it, and `cost`'s field list does not
    # include `exclusive` at all -- so collect.py never enum-checked it. Accept
    # both spellings rather than misjudging admissibility on a type mismatch.
    def is_exclusive(c):
        v = c.get("exclusive")
        return v is True or v == "true"

    inadmissible = [c.get("category") for c in costs if not is_exclusive(c)]

    per_cat = {}
    for c in costs:
        cat = c.get("category")
        per_cat[cat] = {
            "median_pct": c.get("compile_overhead_pct"),
            "n_kernels": c.get("n_kernels"),
            "min_pct": c.get("min_pct"),
            "max_pct": c.get("max_pct"),
            "mean_pct": c.get("mean_pct"),
            "bucket": c.get("bucket"),
            "exclusive": is_exclusive(c),
        }

    meds = [c.get("compile_overhead_pct") for c in costs
            if c.get("compile_overhead_pct") is not None]
    grand = round(statistics.median(meds), 4) if meds else None
    buckets = collections.Counter(c.get("bucket") for c in costs)

    # The definition is carried on every cost record; surface it once here so the
    # integrator cannot quote the number without seeing what it measures.
    definition = next((c.get("definition") for c in costs if c.get("definition")),
                      None)

    out = {
        "status": "ok",
        "feeds": ("E4 compile-time cost (§5.5 ¶1; RQ4) — median front-end-vs-nvcc "
                  "compile-link %, NOT checks-on/off (R-D4); abstract's 0.6% -> "
                  "0.13%"),
        "grand_median_pct": grand,
        "n_categories": len(meds),
        "per_category": per_cat,
        "bucket_distribution": {str(k): int(v) for k, v in sorted(buckets.items())},
        "definition": definition,
        "paper_reference": {"rq4_median_pct": PAPER_REF["S10_rq4_median_pct"],
                            "rq4_bucket": "0.5-1%"},
        "inadmissible_exclusive_false": inadmissible,
        **prov(costs),
    }

    if grand is not None and PAPER_REF["S10_rq4_median_pct"] is not None:
        out["reproduces_rq4"] = abs(grand - PAPER_REF["S10_rq4_median_pct"]) < 0.05
        if not out["reproduces_rq4"]:
            out["rq4_note"] = (
                f"measured grand median {grand}% vs RQ4's "
                f"{PAPER_REF['S10_rq4_median_pct']}%. These are NOT the same "
                "quantity: the assessor cannot be disabled on this build, so "
                "what is separable is choreo's front end against nvcc, not "
                "checks-on against checks-off. No measured category lands in "
                "RQ4's 0.5-1% bucket. See `definition`. The coordinator decides "
                "which number the paper carries; this lane does not silently "
                "substitute one for the other."
            )
    if inadmissible:
        out["admissibility_note"] = (
            f"{len(inadmissible)} cost record(s) carry exclusive=false, which "
            "plan §12.6 rule 2 makes inadmissible to qc. Re-run E4 under the "
            "GPU lock (`run.sh e4`, which takes benchmark2/.gpu-lock) before "
            "this statistic is used."
        )
    return out


# --------------------------------------------------------------------------
# S13 -- runtime residue + detection latency
# --------------------------------------------------------------------------
# LAUNCH GUARD (2026-09-09). croqtile commit 1fa4719 documents a failure mode
# that no timing statistic can detect from its own numbers: a rejected kernel
# launch "does not raise anything at the call site ... the kernel silently never
# ran ... the harness still printed an 'Execution time', and the process exited
# 0". The trigger is this lane's own pinned `--max-local-mem-capacity=2000000`,
# which makes the dynamic-shape path reserve capacity * maxThreadsPerSM * numSMs
# = 466.9 GB against 85 GB of device memory, so cudaLaunchKernel returns
# cudaErrorInvalidValue. 36 of 153 dynamic benchmark cases were affected and ALL
# reported success.
#
# Measured on this lane's own data (diag_launch_reject.py, rebuilt on the
# post-fix binary): 3 of 14 E5a kernels VOID (matmul, max_pool2d, softmax --
# each recorded ok=True with 324-493 ms timings that measured no device work),
# and 2 of 16 E5b pairs confounded on the sanitizer arm.
#
# The guard below is the "did it actually launch?" assertion. It is MANDATORY
# before any number registers: a launch-rejected arm that silently counts as
# "fast" depresses the median in exactly the confounded direction.
RE_LAUNCH_REJECT = run_e5.RE_LAUNCH_REJECT
RE_SANITIZER_SUMMARY = run_e5.RE_SANITIZER_SUMMARY
RE_SANITIZER_FAULT = run_e5.RE_SANITIZER_FAULT
RE_SANITIZER_ABORT = run_e5.RE_SANITIZER_ABORT
# ONE definition of the sanitizer classifier, imported rather than copied.
#
# WHY THIS IS AN IMPORT AND NOT A LOCAL COPY. This file used to carry its own
# `sanitizer_verdict(report)`. run_e5.py carries `sanitizer_verdict(report,
# flagged, baseline_rc, arm_rc=None)`. Two copies of the same classifier with
# DIFFERENT SIGNATURES is not a style problem, it is a live bug: the moment the
# runner learned to distinguish "zero errors, exited 0" (oracle blindness) from
# "no report, exited non-zero" (the arm died), the copy here kept classifying
# both as the same thing -- and the guard, whose entire job is to stop a
# confounded arm reaching the median, would have disagreed with the instrument
# that produced the data. The user's instruction was that the launch assertion is
# "part of item 2, same code path". This is that, literally: one code path.
#
# run_e5.py is import-safe (module level is constants and defs only; its
# `if __name__ == "__main__"` guard means importing it runs nothing) and costs
# ~35 ms.
sanitizer_verdict = run_e5.sanitizer_verdict


def launch_status(rec):
    """(ok, reason) for one register record. The item-5 assertion.

    Works off the register alone, so it also catches records written by an
    EARLIER run_e5.py that predates the guard -- which is the whole point, since
    the existing artifact is exactly that.

    Reasons:
      None                  launched and reported; admissible.
      launch-rejected       the driver refused cudaLaunchKernel.
      no-kernel-end-marker  claimed success with no evidence of a launch.
      no-fault-reported     compute-sanitizer found nothing (see sanitizer_verdict).
      unknown-launch-status record predates the guard and carries no signal
                            either way. NOT treated as a failure -- that would
                            void every existing record -- but counted and named,
                            because an unasserted launch is not an asserted one.
    """
    # `quarantine_reason` is the RUNNER's authoritative verdict for a latency
    # record and must be checked FIRST. It already encodes every non-detection
    # class (launch-rejected, no-fault-reported, died-no-report, process-abort)
    # and is None exactly when the arm is admissible.
    #
    # Why it cannot sit behind the `launch_ok` test: on a sanitizer latency
    # record `launch_ok` is derived from the BASELINE (plain, checks-off) run,
    # not from the sanitizer arm. For a mutant that genuinely faults, that
    # baseline aborts BY DESIGN (rc=-6) -- the fault manifesting without any
    # instrumentation -- so launch_verdict returns launch_ok=False with reason
    # "nonzero-rc". That is EXPECTED, not a launch failure. Reading launch_ok
    # first would quarantine all 4 genuine memcheck-fault detections on a fresh
    # run (verified: each has baseline_rc=-6 -> launch_ok=False), collapsing the
    # ratio to n_pairs=0. The existing register is old-style (no launch_ok), so
    # this was latent until the next re-measure.
    if "quarantine_reason" in rec:
        qr = rec.get("quarantine_reason")
        return (qr is None), qr

    # New-style E5a residue rows carry the launch verdict directly (no
    # quarantine_reason). launch_ok there describes the arm itself, so it is
    # authoritative for that record type.
    if "launch_ok" in rec:
        if rec.get("launch_ok"):
            return True, None
        return False, rec.get("launch_reason") or "launch-rejected"

    # Fall back to the report text, which the register has always carried.
    blob = " ".join(str(rec.get(k) or "")
                    for k in ("report", "rejection_text", "error"))
    if RE_LAUNCH_REJECT.search(blob):
        return False, "launch-rejected"
    if rec.get("detector") == "compute-sanitizer":
        # Prefer the verdict the runner stamped, if this record came from a
        # post-guard run_e5.py. Re-deriving it here would be a second opinion on
        # data the instrument already classified, and the two could disagree.
        v = rec.get("sanitizer_verdict")
        if v is None:
            # Older records carry only the report text (and, since the rc fix,
            # the arm's own exit code). Keyword args are mandatory here:
            # run_e5.sanitizer_verdict's signature is
            # (report, report_line_seen, baseline_rc, arm_rc=None), so passing
            # rc positionally would land it in `report_line_seen` and silently
            # change the meaning of every verdict. That slot is passed False
            # because the register does not persist it -- and False is the safe
            # direction: it can only downgrade `unclassified-but-flagged` (a
            # detection) to `no-report` (not one), never invent a detection.
            v, _det = sanitizer_verdict(blob, False, None,
                                        arm_rc=rec.get("rc"))
        # `process-abort` is quarantined too, and this was the last gap.
        # sanitizer_verdict now returns detected=False for it (EDIT 5, run_e5.py):
        # "the process died" is NOT "compute-sanitizer reported a fault", and in
        # E5b the dynamic oracle IS compute-sanitizer. diag_abort_attribution.py
        # measured all 6 such arms: every one aborts at choreo.h:221
        # (choreo_assert, source A-device) or choreo.h:886 (ArrayProxy::operator[],
        # source B) with baseline_rc=-6, i.e. WITH compute-sanitizer entirely
        # absent, and memcheck's own summary on every one is "ERROR SUMMARY: 0
        # errors". So the wall clock on that arm measures choreo's own UNGATED
        # assertion killing the process, and the ratio compares choreo against
        # choreo. sanitizer_verdict's docstring once guessed "typically the
        # kernel's own oracle assert aborting" (source C, which would be a
        # legitimate detection); the measurement says otherwise, and even a
        # source-C abort would be choreo detecting, not the oracle.
        # Quarantining it is not designing around the finding -- the finding is
        # reported in `detection_asymmetry` and in the note below. It is
        # refusing to publish a ratio whose denominator and numerator are the
        # same instrument.
        if v in ("no-fault-reported", "launch-rejected", "died-no-report",
                 "process-abort"):
            return False, v
        return True, None
    if rec.get("runtime_ms") is None and rec.get("detector") is None:
        # An E5a residue row with no timing and no launch field: excluded
        # upstream, and the exclusion must survive into the statistic.
        return False, "no-kernel-end-marker"
    return True, "unknown-launch-status"


def s13(residue, latency, present):
    """E5a residue A/B and E5b detection latency. Feeds RQ3 + E5.

    E5a: per dynamic-shape case, the delta between checks-on and checks-off.
    A delta is only interpretable against the noise floor of the arms' own
    rep-to-rep spread, so `delta_within_noise` is carried through and counted
    here -- a delta smaller than the spread is not a measurement, and the
    previous contaminated run produced -25% deltas that six entry guards cannot
    possibly explain.

    E5b: time-to-report for choreo's entry check vs compute-sanitizer. The claim
    is structural (hoisted entry check vs run-to-fault instrumentation), not
    competitive, so the report groups by check_loc rather than only by detector.
    """
    # S13 splits across TWO paper axes under the 2026-09-09 remap: E5a's residue
    # A/B feeds E4 ¶2 (cost), while E5b's latency feeds E5 ¶1 (runtime guarantee
    # vs the dynamic oracle). They are one artifact but two floats.
    out = {"status": "ok",
           "feeds": ("E4 residue A/B (§5.5 ¶2, E5a: structurally zero on device, "
                     "below measurement floor on host — R-D3) + E5 (§5.6 ¶1, E5b: "
                     "DETECTION ASYMMETRY / coverage, NOT a latency ratio — R-D1)")}

    # ---- E5a -------------------------------------------------------------
    if not present.get("residue") or not residue:
        out["E5a"] = not_run("RQ3", "no residue records; run `run.sh e5`")
        out["E5a"]["expected_records"] = "residue"
    else:
        # Pair the arms by (category, kernel_id). Each case emits one record per
        # arm, so pairing is what turns two rows into one delta.
        cases = collections.defaultdict(dict)
        for r in residue:
            key = (r.get("category"), r.get("kernel_id") or r.get("case"))
            cases[key][r.get("checks")] = r

        rows, unpaired, launch_excluded = [], [], []
        n_launch_unasserted = 0
        for (cat, kid), arms in sorted(cases.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            on, off = arms.get("on"), arms.get("off")
            if not on or not off:
                unpaired.append({"category": cat, "kernel_id": kid,
                                 "arms_present": sorted(arms)})
                continue

            # ---- LAUNCH GUARD (item 5: "did it actually launch?") ----------
            # Checked BEFORE any timing is read, so a rejected arm can never
            # reach the median. A launch-rejected pair is NOT an unpaired pair
            # and NOT a missing-runtime_ms pair: it is its own bucket, named,
            # with the evidence attached, because the three have different
            # meanings and a reader must be able to tell them apart.
            verdicts = {}
            for tag, rec in (("on", on), ("off", off)):
                ok, reason = launch_status(rec)
                verdicts[tag] = {"launch_ok": ok, "reason": reason}
                if ok and reason == "unknown-launch-status":
                    n_launch_unasserted += 1
            bad = {t: v for t, v in verdicts.items() if not v["launch_ok"]}
            if bad:
                reasons = sorted({v["reason"] for v in bad.values()})
                ev = None
                for t in bad:
                    e = (arms[t].get("launch_evidence")
                         or arms[t].get("baseline_launch_evidence"))
                    if e:
                        ev = e
                        break
                launch_excluded.append({
                    "category": cat, "kernel_id": kid,
                    "excluded": reasons[0] if len(reasons) == 1 else reasons,
                    "arms_affected": sorted(bad),
                    "arm_verdicts": verdicts,
                    "evidence": ev,
                    "note": (
                        "The driver REFUSED cudaLaunchKernel, so no device work "
                        "happened and the wall time is not a measurement of "
                        "anything. With --max-local-mem-capacity=2000000 the "
                        "dynamic-shape path reserves capacity * maxThreadsPerSM "
                        "* numSMs (466.9 GB on an H800 PCIe, 114 SMs x 2048) "
                        "against 85 GB of device memory. Post-croqtile-1fa4719 "
                        "this aborts loudly (rc=134); pre-1fa4719 it exited 0 "
                        "and still printed an `Execution time`. Excluded from "
                        "the residue median, never averaged into it."
                        if "launch-rejected" in reasons else
                        "Excluded by the launch guard; see `excluded` for the "
                        "reason. Not averaged into the residue median."
                    ),
                })
                continue

            t_on, t_off = on.get("runtime_ms"), off.get("runtime_ms")
            if t_on is None or t_off is None or not t_off:
                unpaired.append({"category": cat, "kernel_id": kid,
                                 "reason": "missing runtime_ms"})
                continue
            d_ms = t_on - t_off
            # Per-arm rep-to-rep spread; the noise floor for this case is the
            # larger of the two (matches run_e5.py's `delta_within_noise`).
            sp = [s for s in (on.get("spread_pct"), off.get("spread_pct"))
                  if s is not None]
            noise = max(sp) if sp else None
            d_pct = pct(d_ms, t_off)
            # RECOMPUTED HERE, NOT COPIED (bug fix 2026-09-09). This used to read
            # `on.get("delta_within_noise")`, but that field lives on run_e5.py's
            # `e5a_kernels` rows, NOT on the `residue` records this loop pairs --
            # so it was ALWAYS None. Two consequences, both silent:
            #   1. `n_delta_within_noise` reported 0/14 when run_e5.py's own
            #      report said 14/14, contradicting the artifact it summarises.
            #   2. the `structural_contradiction_note` QC below tests
            #      `delta_within_noise is False`, which None never satisfies, so
            #      a QC guard that exists to catch a Δ OUTSIDE the noise floor on
            #      byte-identical device code could never fire.
            # The definition is replicated exactly from run_e5.py:541 --
            # `noise = max(spread_on, spread_off)`, in-noise iff
            # `abs(delta_pct) <= noise` -- so the two agree by construction.
            rows.append({
                "category": cat, "kernel_id": kid,
                "checks_on_ms": t_on, "checks_off_ms": t_off,
                "delta_ms": round(d_ms, 4),
                "delta_pct": d_pct,
                "shape_class": on.get("shape_class"),
                "timing_source": on.get("timing_source"),
                "delta_within_noise": (None if d_pct is None or noise is None
                                       else bool(abs(d_pct) <= noise)),
                "clock_drift_mhz": on.get("clock_drift_mhz"),
                "exclusive": on.get("exclusive"),
                "arm_spread_pct": noise,
                # Structural evidence rides on the record (see run_e5.py
                # `device_code_evidence`); carry it so the QC below can tell a
                # real Δ from contamination on byte-identical device code.
                "device_code": on.get("device_code"),
                # The guard's verdict, carried on the admissible rows too: an
                # asserted launch and an unasserted one must be distinguishable
                # in the output, not only in the code that produced it.
                "launch_ok": True,
                "launch_asserted": verdicts["on"]["reason"] != "unknown-launch-status",
            })

        dpct = [r["delta_pct"] for r in rows if r["delta_pct"] is not None]
        dms = [r["delta_ms"] for r in rows]
        innoise = [r for r in rows if r["delta_within_noise"]]
        negative = [r for r in rows if r["delta_pct"] is not None
                    and r["delta_pct"] < 0]
        drifted = [r for r in rows if r.get("clock_drift_mhz")]

        out["E5a"] = {
            "status": "ok",
            "n_cases": len(rows),
            "n_unpaired": len(unpaired),
            "unpaired": unpaired,
            "median_delta_pct": round(statistics.median(dpct), 4) if dpct else None,
            "median_delta_ms": round(statistics.median(dms), 4) if dms else None,
            "n_delta_within_noise": len(innoise),
            "n_negative_delta": len(negative),
            "negative_delta_cases": [
                {"kernel_id": r["kernel_id"], "delta_pct": r["delta_pct"]}
                for r in negative
            ],
            "n_with_clock_drift": len(drifted),
            "per_case": rows,
            "static_cases_measured": sum(1 for r in rows
                                         if r.get("shape_class") == "static"),
            # ---- LAUNCH GUARD output ------------------------------------
            # `n_cases` above is the ADMISSIBLE count. The excluded bucket is
            # reported alongside it so the denominator is never ambiguous: a
            # reader sees 11 timed + 3 launch-rejected = 14 selected, not a
            # silent 11.
            "launch_guard": {
                "n_launch_rejected": len(launch_excluded),
                "n_timed": len(rows),
                "n_selected": len(rows) + len(launch_excluded) + len(unpaired),
                "n_arms_launch_unasserted": n_launch_unasserted,
                "excluded_launch_rejected": launch_excluded,
                "definition": (
                    "launch-rejected = the driver refused cudaLaunchKernel, so "
                    "no device work happened and the millisecond value is not a "
                    "measurement. Excluded from every median below; listed by "
                    "name in `excluded_launch_rejected`. "
                    "n_arms_launch_unasserted counts arms whose record predates "
                    "the guard and therefore carries NO launch assertion either "
                    "way -- those are admitted (voiding them would discard every "
                    "existing record) but they are counted here so the absence "
                    "of an assertion is visible rather than implied by ok=True."),
            },
            **prov(residue),
        }
        if launch_excluded:
            out["E5a"]["launch_rejected_note"] = (
                f"{len(launch_excluded)} of "
                f"{len(rows) + len(launch_excluded)} selected E5a kernel(s) had "
                f"their launch REFUSED by the driver and are EXCLUDED from the "
                f"residue median: "
                + ", ".join(r["kernel_id"] for r in launch_excluded)
                + ". This is the auditor's launch-time rejection working as "
                  "designed (croqtile 1fa4719), not a measurement failure. The "
                  "residue claim rests on the "
                  f"{len(rows)} kernel(s) that actually launched."
            )
        if n_launch_unasserted:
            out["E5a"]["launch_unasserted_note"] = (
                f"{n_launch_unasserted} E5a arm record(s) carry NO launch "
                f"assertion: they were written by a run_e5.py that predates the "
                f"launch guard, so nothing in the record says whether the kernel "
                f"reached the device. They are admitted to the median because "
                f"voiding them would discard the whole existing E5a, but this is "
                f"an UNASSERTED launch, not a verified one. Measured exposure on "
                f"this exact data (diag_launch_reject.py, rebuilt on the "
                f"post-fix binary): 3 of the 14 -- matmul/10_dynamic, "
                f"max_pool2d/10_dynamic, softmax/10_dynamic -- were silently "
                f"launch-rejected and recorded ok=True with 324-493 ms timings "
                f"that measured no device work. RE-MEASURE E5a before quoting "
                f"any residue number from records carrying this note."
            )

        # ---- structural evidence: WHERE do the gated checks live? ----------
        # A Δ inside the noise floor is ambiguous on its own. The `device_code`
        # block E5a now records resolves the ambiguity: if the two arms' device
        # kernels are byte-identical and contain zero checks, then there is no
        # device-side cost to resolve, and a near-zero Δ is the EXPECTED result
        # rather than a failed measurement.
        dc = [r.get("device_code") for r in residue if r.get("device_code")]
        if dc:
            ident = sum(1 for d in dc if d.get("device_identical") is True)
            dev_checks = sum(
                (d.get("device_check_calls") or {}).get("runtime_check", 0)
                + (d.get("device_check_calls") or {}).get("choreo_assert", 0)
                for d in dc)
            host = [d.get("host_gated_checks") for d in dc
                    if d.get("host_gated_checks") is not None]
            out["E5a"]["device_code_evidence"] = {
                "n_rows_reporting": len(dc),
                "n_device_identical": ident,
                "total_device_side_check_calls": dev_checks,
                "host_gated_checks_min": min(host) if host else None,
                "host_gated_checks_max": max(host) if host else None,
                "interpretation": (
                    "All gated checks are HOST-side entry guards on shape "
                    "metadata, executed once per launch; the generated device "
                    "kernel is byte-identical between the checks-on and "
                    "checks-off arms and contains no check calls. Runtime "
                    "residue on the device is therefore structurally zero, and "
                    "the host-side entry cost is a handful of integer "
                    "comparisons against a multi-second launch."
                    if ident == len(dc) and dev_checks == 0 else
                    "NOT all arms have identical device code, or some device-side "
                    "check calls exist. The structural argument for a zero Δ does "
                    "not hold uniformly; inspect `device_code` per row before "
                    "quoting residue as structurally zero."),
            }
            # QC: a row whose device code is byte-identical across arms and
            # carries zero device-side checks CANNOT have a real runtime Δ —
            # there is nothing on the device that differs. If such a row still
            # reports a Δ OUTSIDE its own noise floor, the Δ is contamination
            # (clock ramp, no lock, too few reps), not residue. Flag it loudly:
            # this is the case a reviewer would otherwise read as overhead.
            contradicted = [
                r for r in rows
                if (r.get("device_code") or {}).get("device_identical") is True
                and sum((r.get("device_code") or {})
                        .get("device_check_calls", {}).values()) == 0
                and r.get("delta_within_noise") is False
            ]
            if contradicted:
                out["E5a"]["structural_contradiction_note"] = (
                    f"{len(contradicted)} case(s) report a Δ OUTSIDE their own "
                    "rep-to-rep spread even though their device kernel is "
                    "byte-identical across the checks-on/checks-off arms and "
                    "contains zero device-side checks. Byte-identical device code "
                    "cannot produce a real runtime difference, so those Δ values "
                    "are measurement contamination (clock ramp, lock not held, or "
                    "too few reps), NOT runtime residue. Do not quote them as "
                    "overhead. Cases: "
                    + ", ".join(
                        f"{r.get('kernel_id')} (Δ={r.get('delta_pct')}%, "
                        f"spread={r.get('arm_spread_pct') or r.get('spread_pct')}%, "
                        f"exclusive={r.get('exclusive')}, "
                        f"drift={r.get('clock_drift_mhz')}MHz)"
                        for r in contradicted)
                )
                out["E5a"]["n_structural_contradictions"] = len(contradicted)
        # The clock state is MEASURED, not assumed: an earlier draft of this
        # note claimed "device 0 idles at 1755 MHz and device 1 at 345 MHz".
        # Sampled 2026-09-09 while a kernel was actively executing, BOTH devices
        # sat at 345 MHz (max 1755) with persistence_mode Disabled -- a 5.1x
        # ramp range on either device. Quote the measured drift, never a
        # hard-coded per-device figure.
        drift = [r.get("clock_drift_mhz") for r in rows
                 if r.get("clock_drift_mhz")]
        if negative:
            out["E5a"]["negative_delta_note"] = (
                f"{len(negative)} case(s) show checks-on FASTER than checks-off. "
                "A handful of entry guards cannot make a kernel faster, so these "
                "are drift or contention, not overhead. Check "
                "`clock_drift_mhz` and `exclusive` on those rows before quoting "
                "any E5a number. Both devices idle at 345 MHz with "
                "persistence_mode Disabled (max 1755 MHz), so a clock ramp alone "
                "can produce this"
                + (f" -- measured drift here reached {max(drift)} MHz."
                   if drift else ".")
            )
        if innoise:
            dce = out["E5a"].get("device_code_evidence") or {}
            structural = (
                dce.get("n_rows_reporting")
                and dce.get("n_device_identical") == dce.get("n_rows_reporting")
                and dce.get("total_device_side_check_calls") == 0)
            out["E5a"]["noise_note"] = (
                f"{len(innoise)}/{len(rows)} deltas are INSIDE the arms' own "
                "rep-to-rep spread. "
                + ("This is the EXPECTED result, not a failed measurement: the "
                   "generated device kernel is byte-identical between the two "
                   "arms and contains zero check calls, so "
                   "`--disable-runtime-check` removes only HOST-side entry "
                   "guards (integer comparisons on shape metadata, run once per "
                   "launch). There is no device-side cost for this host's clock "
                   "ramp to resolve. The honest RQ3 statement is that runtime "
                   "residue is structurally zero on the device and below the "
                   "measurement floor on the host."
                   if structural else
                   "Those cases do not support a residue claim in either "
                   "direction from timing alone; the honest RQ3 statement is "
                   "that residue is below the measurement floor for them. "
                   "Check `device_code` per row for the structural reason.")
            )
        if out["E5a"]["static_cases_measured"]:
            out["E5a"]["static_premise_note"] = (
                "plan §6.2 asserts E5a's delta is '0 by construction on "
                "static-shape cases'. That premise is FALSE at the code level: "
                "static kernels emit 3-8 entry `runtime_check` sites (the "
                "shape-compatibility guards the ledger records as "
                "stats-derived/shape/proven/direct), and -rtc=none removes them. "
                "The static rows below are therefore a MEASUREMENT of that "
                "premise, not a confirmation of it."
            )

    # ---- E5b -------------------------------------------------------------
    if not present.get("latency") or not latency:
        out["E5b"] = not_run("E5", "no latency records; run `run.sh e5` "
                                   "(needs E1's runtime-outcome mutants)")
        out["E5b"]["expected_records"] = "latency"
    else:
        # ---- LAUNCH / DETECTION GUARD, applied to EVERY latency record -----
        # Done once, up front, so the per-detector groups and the paired ratio
        # below cannot disagree about what is admissible.
        admissible, quarantined = [], []
        for l in latency:
            ok, reason = launch_status(l)
            if ok:
                admissible.append(l)
            else:
                quarantined.append({"mutant_id": l.get("mutant_id"),
                                    "detector": l.get("detector"),
                                    "category": l.get("category"),
                                    "reason": reason,
                                    "time_to_report_us": l.get("time_to_report_us"),
                                    "report": (l.get("report") or "")[:200]})
        q_by_reason = collections.Counter(q["reason"] for q in quarantined)
        q_by_id = collections.defaultdict(set)
        for q in quarantined:
            q_by_id[q["mutant_id"]].add(q["reason"])

        by_det = collections.defaultdict(list)
        for l in admissible:
            t = l.get("time_to_report_us")
            if t is not None:
                by_det[(l.get("detector"), l.get("check_loc"))].append(t)
        groups = {}
        for (det, loc), ts in sorted(by_det.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            groups[f"{det}/{loc}"] = {
                "n": len(ts),
                "median_us": round(statistics.median(ts), 3),
                "min_us": round(min(ts), 3),
                "max_us": round(max(ts), 3),
            }

        # ---- PAIRED RATIO: the number §5.6 ¶1 actually rests on -------------
        # WHY PAIRED. The medians above are per-detector, computed across
        # different mutants. Dividing one median by the other would compare
        # unlike kernels. The claim is about the SAME fault: for one mutant, how
        # long until choreo's entry check reports it vs how long until the
        # dynamic oracle reports it. So pair on mutant_id and take the ratio of
        # the pair, then summarise those ratios.
        #
        # A pair is CONFOUNDED if EITHER arm is quarantined, and is then held
        # out of the ratio entirely -- not clamped, not imputed. THREE classes
        # of confound were found, and they pull in OPPOSITE directions, which
        # is why none can be dropped:
        #   launch-rejected    (2 of 16) the sanitizer arm timed a REFUSED
        #                      launch, so the ratio is depressed (1.62x, 2.32x).
        #   no-fault-reported  (4 of 16) the sanitizer ran the body to
        #                      completion and printed "ERROR SUMMARY: 0 errors"
        #                      -- no detection at all -- so the ratio is INFLATED
        #                      by a full instrumented run that found nothing
        #                      (162x, 190x, 215x, and the max 472x).
        #   process-abort      (6 of 16) the wrapped process died on choreo's
        #                      OWN UNGATED assertion (choreo.h:221/:886) before
        #                      memcheck judged anything; memcheck printed
        #                      "ERROR SUMMARY: 0 errors". The wall clock measured
        #                      choreo, so the ratio would compare choreo against
        #                      choreo. These 6 span BOTH directions (1.57x-233x),
        #                      so they are not a uniform bias -- they are simply
        #                      not oracle measurements at all.
        # Only 4 of 16 arms are genuine oracle detections (memcheck-fault).
        # Quarantining only the launch-rejected pairs -- the FIRST guard written
        # -- moved the median from 64.2x to 143.0x: it made the headline look
        # BETTER by removing the two pairs that hurt it while leaving in four
        # zero-error pairs and six choreo-kill pairs. That is the confounded
        # direction, so all three classes go. The clean statistic is n=4,
        # median 84.4x, min 2.58x, max 204.4x, oracle slower on 4/4.
        paired = collections.defaultdict(dict)
        for l in latency:
            t = l.get("time_to_report_us")
            if t is not None:
                paired[l.get("mutant_id")][l.get("detector")] = t
        ratios, per_pair, confounded = [], [], []
        for mid, arms in sorted(paired.items(), key=lambda x: str(x[0])):
            ce, sa = arms.get("choreo-entry"), arms.get("compute-sanitizer")
            if not (ce and sa and ce > 0):
                continue
            reasons = sorted(q_by_id.get(mid, ()))
            if reasons:
                confounded.append({
                    "mutant_id": mid,
                    "reason": reasons[0] if len(reasons) == 1 else reasons,
                    "choreo_entry_us": ce,
                    "sanitizer_us": sa,
                    "ratio_sanitizer_over_choreo": round(sa / ce, 2),
                    "note": (
                        "EXCLUDED, not averaged in. The sanitizer arm did not "
                        "produce a time-to-report for this mutant: "
                        + ("the driver refused cudaLaunchKernel, so the "
                           "instrumented body never ran and the wall time "
                           "measures a refusal. This DEPRESSES the ratio."
                           if "launch-rejected" in reasons else
                           "compute-sanitizer ran the body to completion and "
                           "reported `ERROR SUMMARY: 0 errors` -- no fault, so "
                           "there is no report to time. The wall time measures a "
                           "full instrumented run that found nothing, which "
                           "INFLATES the ratio. E1 corroborates: this mutant's "
                           "oracle arm also recorded oracle_caught=none / "
                           "oracle_run_rc=0."
                           if "no-fault-reported" in reasons else
                           "the wrapped process died on choreo's OWN UNGATED "
                           "assertion (choreo.h:221 choreo_assert / "
                           "choreo.h:886 ArrayProxy::operator[]) and memcheck "
                           "reported `ERROR SUMMARY: 0 errors`. Verified by "
                           "running the same executable with NO sanitizer "
                           "present: it aborts identically (baseline_rc=-6). So "
                           "this arm's wall clock measured choreo, not the "
                           "oracle, and the ratio would compare choreo against "
                           "choreo. See detection_asymmetry."
                           if "process-abort" in reasons else
                           "see `reason`."),
                    ),
                })
                continue
            ratios.append(sa / ce)
            per_pair.append({"mutant_id": mid,
                             "choreo_entry_us": ce,
                             "sanitizer_us": sa,
                             "ratio_sanitizer_over_choreo": round(sa / ce, 2)})
        ratio_block = {
            # CLEAN: the number the paper may quote. Confounded pairs are held
            # out, and `n_confounded` says how many and `confounded_pairs` says
            # which -- so the exclusion is stated, not silent.
            "n_pairs": len(ratios),
            "median_ratio": round(statistics.median(ratios), 2) if ratios else None,
            "min_ratio": round(min(ratios), 2) if ratios else None,
            "max_ratio": round(max(ratios), 2) if ratios else None,
            "n_pairs_oracle_slower": sum(1 for r in ratios if r > 1),
            "per_pair": per_pair,
            "n_confounded": len(confounded),
            "confounded_pairs": confounded,
            "n_pairs_before_guard": len(ratios) + len(confounded),
            "definition": "sanitizer time_to_report_us / choreo-entry "
                          "time_to_report_us, PAIRED on mutant_id. >1 means the "
                          "dynamic oracle took longer to report the same fault. "
                          "Pairs where EITHER arm was quarantined by the launch "
                          "/ detection guard are held out and listed in "
                          "`confounded_pairs`; `n_pairs` is the CLEAN count.",
        }
        # The all-pairs figure, kept for traceability ONLY. It is the number an
        # earlier cut of this statistic reported, and it is not quotable: it
        # mixes 2 pairs that timed a refused launch with 4 pairs where the
        # oracle made no report at all.
        all_ratios = [p["ratio_sanitizer_over_choreo"] for p in per_pair] \
            + [c["ratio_sanitizer_over_choreo"] for c in confounded]
        if all_ratios:
            ratio_block["unguarded_all_pairs"] = {
                "n_pairs": len(all_ratios),
                "median_ratio": round(statistics.median(all_ratios), 2),
                "min_ratio": round(min(all_ratios), 2),
                "max_ratio": round(max(all_ratios), 2),
                "note": "NOT QUOTABLE. Retained so the guard's effect is "
                        "auditable: this is what the statistic said before the "
                        "launch/detection guard excluded "
                        f"{len(confounded)} confounded pair(s).",
            }

        # ---- HONESTY NOTE on the absolute -------------------------------
        # §5.6 ¶1 as drafted calls choreo's entry check "µs-scale". It is not,
        # and this lane will not let that word reach the paper unqualified: the
        # measured median is ~0.5 s. Both arms time the EXECUTABLE only (nvcc
        # compile excluded -- see run_e5.py's SYMMETRY FIX), so what remains is
        # process startup + CUDA context init + launch + abort. That constant is
        # real, it is shared by both arms, and it is ~3 orders of magnitude
        # larger than the check itself.
        # The defensible claim is the one the data supports:
        #   (a) choreo's report is INPUT-INDEPENDENT -- its spread is ~4x across
        #       16 different faults, because it aborts at launch before touching
        #       the data;
        #   (b) the oracle's is DATA-DEPENDENT -- its spread is ~233x, because it
        #       must run the instrumented body to the fault;
        #   (c) therefore the oracle is slower on every one of the CLEAN pairs.
        # (a)+(b)+(c) are the finding. "µs-scale" is not.
        #
        # The counts below are deliberately NOT hardcoded: the guard above
        # decides how many pairs are clean, and this note must follow that
        # decision rather than a number written before the guard existed.
        # IMPORTANT: these spreads are computed over ADMISSIBLE records only.
        # An earlier cut computed the oracle's spread over every sanitizer
        # record, which silently folded in the 4 zero-error runs (whose wall
        # time measures a full instrumented run that found NOTHING) and the 2
        # launch-refusals. That made "spread 233x" partly an artefact of the
        # confounded pairs -- the same leak the guard exists to close, one level
        # up in the prose. The guard's `admissible` list is the single source of
        # truth for what may be summarised.
        ce_all = [l["time_to_report_us"] for l in admissible
                  if l.get("detector") == "choreo-entry"
                  and l.get("time_to_report_us") is not None]
        sa_all = [l["time_to_report_us"] for l in admissible
                  if l.get("detector") == "compute-sanitizer"
                  and l.get("time_to_report_us") is not None]
        abs_note = None
        if ce_all:
            ce_spread = max(ce_all) / min(ce_all) if min(ce_all) else None
            sa_spread = (max(sa_all) / min(sa_all)
                         if sa_all and min(sa_all) else None)
            abs_note = (
                f"The ABSOLUTE time_to_report_us is dominated by a shared "
                f"process-startup + CUDA-context-init constant (~"
                f"{statistics.median(ce_all)/1e3:.0f} ms median on the "
                f"choreo-entry arm), NOT by the check itself. Do not describe "
                f"choreo's entry check as \"µs-scale\" on this evidence. What "
                f"the data supports: choreo's latency is INPUT-INDEPENDENT "
                f"(spread {ce_spread:.1f}x across {len(ce_all)} distinct faults, "
                f"because it aborts at launch before touching the data) while "
                f"the oracle's is DATA-DEPENDENT (spread "
                f"{sa_spread:.0f}x, because it must run the instrumented body to "
                f"the fault); hence the oracle is slower on every pair. Both "
                f"arms time the executable only -- nvcc compile is excluded from "
                f"both (run_e5.py SYMMETRY FIX, 2026-09-09). An earlier cut of "
                f"this statistic timed nvcc+run on the choreo arm and run-only "
                f"on the sanitizer arm, which produced a 0.2x ratio and "
                f"INVERTED the claim; that number is void.")
        # ---- DETECTION ASYMMETRY: what the guard's exclusions actually MEAN --
        # The `no-fault-reported` quarantine is not merely a latency caveat. It
        # is a DETECTION result, and dropping those pairs from the ratio without
        # saying so would throw away the stronger half of the finding.
        #
        # For those mutants compute-sanitizer ran the instrumented body to
        # completion and printed "ERROR SUMMARY: 0 errors" -- it found nothing.
        # choreo's entry check aborted on the same mutant with a runtime-check
        # failure. So the static check caught a fault the dynamic oracle missed
        # entirely. That is not a speed comparison; it is a coverage comparison,
        # and it belongs in E1's detection story as much as in E5b's latency one.
        #
        # Derived from the guard's own verdicts rather than a separate pass, so
        # the two cannot disagree: a record is "detected" iff it was ADMISSIBLE
        # (it produced a real report); "no-fault-reported" is by construction a
        # non-detection.
        # IMPORTANT: the two quarantine reasons are NOT the same claim and must
        # not be pooled.
        #   no-fault-reported  the oracle RAN the instrumented body to
        #                      completion and found nothing. That is genuine
        #                      oracle blindness -- a coverage result.
        #   launch-rejected    the driver refused cudaLaunchKernel, so the
        #                      oracle NEVER EXECUTED the body. It did not miss
        #                      the fault; it was never given the chance. Calling
        #                      that "blind" would overstate the coverage claim
        #                      by attributing a harness-level refusal to the
        #                      detector's judgement.
        #   died-no-report     no fault line AND the arm's own rc != 0: the
        #                      sanitizer process exited without reporting.
        #                      Neither blindness nor a launch refusal -- the arm
        #                      died. Counted separately again, because "the
        #                      oracle missed it" and "the oracle crashed" are
        #                      different statements and only the first is a
        #                      coverage result.
        # So `n_oracle_blind` counts ONLY the first. The others are reported
        # separately, each under the claim it actually supports.
        det_by_id = collections.defaultdict(dict)
        for l in latency:
            ok, reason = launch_status(l)
            det_by_id[l.get("mutant_id")][l.get("detector")] = (ok, reason)
        cat_of = {l.get("mutant_id"): l.get("category") for l in latency}
        oracle_blind, oracle_never_ran, oracle_died, both_detected = \
            [], [], [], []
        # A FIFTH bucket, and the reason it cannot be folded into the others.
        # `process-abort` means the wrapped process died. sanitizer_verdict now
        # returns detected=False for it (EDIT 5): in E5b the dynamic oracle IS
        # compute-sanitizer, so an abort on ANY choreo assertion is choreo
        # detecting, never compute-sanitizer detecting.
        # diag_abort_attribution.py ran all 6 such arms twice: once with NO
        # sanitizer present, once under memcheck. Every one aborts alone
        # (baseline_rc=-6) at choreo.h:221 (`choreo_assert`, source A-device)
        # or choreo.h:886 (`ArrayProxy::operator[]`, source B), and under
        # memcheck the tool's own summary is "ERROR SUMMARY: 0 errors" on all
        # 6. Both sources are UNGATED by --disable-runtime-check, so E5b's
        # premise that the sanitizer arm had "the sanitizer as sole detector"
        # was false on these arms.
        # They are therefore NOT oracle-blind (the body did not run to
        # completion -- choreo killed it first) and NOT both-detected (memcheck
        # rendered no memory-fault verdict). Counting them as blind would
        # inflate the coverage claim with arms the oracle never got to judge;
        # counting them as detected would credit the oracle with choreo's kill.
        oracle_aborted_by_choreo = []
        for mid, arms in sorted(det_by_id.items(), key=lambda x: str(x[0])):
            ce = arms.get("choreo-entry")
            sa = arms.get("compute-sanitizer")
            if not (ce and sa):
                continue
            if not ce[0]:
                continue          # choreo itself did not report: no asymmetry
            row = {"mutant_id": mid, "choreo_entry": "detected",
                   "compute_sanitizer": sa[1] or "no-report",
                   "category": cat_of.get(mid)}
            if sa[0]:
                both_detected.append(mid)
            elif sa[1] == "no-fault-reported":
                oracle_blind.append(row)
            elif sa[1] == "died-no-report":
                oracle_died.append(row)
            elif sa[1] == "process-abort":
                oracle_aborted_by_choreo.append(row)
            else:
                # launch-rejected, or any reason this block does not name.
                # Defaulting unknowns into "never executed" rather than into
                # "blind" is deliberate: the coverage claim is the one that
                # benefits choreo, so an unclassifiable arm must not inflate it.
                oracle_never_ran.append(row)
        asym = None
        if (oracle_blind or oracle_never_ran or oracle_died
                or oracle_aborted_by_choreo):
            asym = {
                "n_oracle_blind": len(oracle_blind),
                "n_oracle_never_executed": len(oracle_never_ran),
                "n_oracle_died_no_report": len(oracle_died),
                "n_oracle_aborted_by_choreo_own_ungated_check":
                    len(oracle_aborted_by_choreo),
                "n_both_detected": len(both_detected),
                "n_pairs_compared": (len(oracle_blind) + len(oracle_never_ran)
                                     + len(oracle_died) + len(both_detected)
                                     + len(oracle_aborted_by_choreo)),
                "oracle_blind_mutants": oracle_blind,
                "oracle_never_executed_mutants": oracle_never_ran,
                "oracle_died_mutants": oracle_died,
                "oracle_aborted_by_choreo_mutants": oracle_aborted_by_choreo,
                "definition": (
                    "oracle-blind = choreo's entry check reported the fault "
                    "while compute-sanitizer RAN the instrumented body to "
                    "completion and reported `ERROR SUMMARY: 0 errors`. "
                    "oracle-never-executed = the driver refused "
                    "cudaLaunchKernel on the sanitizer arm, so the oracle never "
                    "ran the body at all; that is a launch-level refusal, NOT a "
                    "detection judgement. oracle-died-no-report = the sanitizer "
                    "arm exited non-zero with no fault line, so it neither "
                    "completed nor refused. oracle-aborted-by-choreo = the "
                    "wrapped process died on choreo's OWN UNGATED assertion "
                    "(choreo.h:221 choreo_assert / choreo.h:886 "
                    "ArrayProxy::operator[]) while memcheck reported zero "
                    "errors; verified by running the same executable with NO "
                    "sanitizer present and observing the identical abort, so "
                    "the kill is choreo's and not the oracle's. Only the first "
                    "is a coverage result; the others are counted separately so "
                    "the coverage claim is not inflated by arms that never got "
                    "to judge anything. All four are EXCLUDED from the latency "
                    "ratio above (none produces an oracle report to time) and "
                    "reported here instead of being silently dropped."),
                "note": (
                    f"{len(oracle_blind)} of "
                    f"{len(oracle_blind) + len(both_detected)} mutant(s) where "
                    f"the oracle actually executed to completion were caught by "
                    f"choreo's entry check and MISSED entirely by "
                    f"compute-sanitizer. A further {len(oracle_never_ran)} never "
                    f"reached execution (launch refused), "
                    f"{len(oracle_died)} exited without reporting, and "
                    f"{len(oracle_aborted_by_choreo)} were killed by choreo's "
                    f"own ungated assertion before memcheck could judge; no "
                    f"group is counted as a miss, because in none of them did "
                    f"the oracle render a verdict. The oracle's silence on the "
                    f"{len(oracle_blind)} is not a latency advantage -- it "
                    f"produced no report at all. This is coverage, not speed."),
                "premise_violation_note": (
                    f"{len(oracle_aborted_by_choreo)} of the sanitizer arms were "
                    "built with --disable-runtime-check on the stated premise "
                    "that the sanitizer would then be the SOLE detector. That "
                    "premise is FALSE: the flag gates only source A (host "
                    "runtime_check and device choreo_assert at the HOIST/USE "
                    "sites), while source B (ArrayProxy::operator[] bounds "
                    "check) and source C (the user assert BIF) are ungated by "
                    "construction. On these arms choreo's own ungated check "
                    "aborted the process, memcheck reported `ERROR SUMMARY: 0 "
                    "errors`, and the recorded wall clock measured choreo "
                    "rather than the oracle -- so the ratio would have compared "
                    "choreo against choreo. Evidence: "
                    "diag_abort_attribution.py, which runs each executable both "
                    "with and without compute-sanitizer. This is a DATA problem "
                    "in the experiment's construction, reported rather than "
                    "designed around; suppressing sources B and C needs a "
                    "toolchain flag that may not exist."),
            }

        out["E5b"] = {
            "status": "ok",
            "n_records": len(latency),
            "per_detector_and_check_loc": groups,
            "ratio": ratio_block,
            "detection_asymmetry": asym,
            "absolute_latency_note": abs_note,
            "check_loc_is_measured_not_assumed": sorted(
                {str(l.get("check_loc")) for l in latency}),
            **prov(latency),
        }
    return out


# --------------------------------------------------------------------------
# S14 -- path-class register (v2.1)
# --------------------------------------------------------------------------
# WHAT THIS IS AND WHY IT IS SEPARATE FROM S1/S2
#
# S1 answers "how many mutants did choreo catch". S14 answers "how many mutants
# was it FAIR to ask about, and for the rest, what is the recorded reason it was
# not asked". Those are different denominators and conflating them is the defect
# this statistic exists to prevent: v1 reported "28 mutants with no detection"
# as a single number, which is unfalsifiable -- a reader cannot tell an
# unassessable spec from a spec nobody wrote an operator for.
#
# The register is two orthogonal axes (specs/expansion-workflow.md §2):
#   status      does an operator exist for this spec?
#   admissible  may this spec's cells enter the detection denominator?
# An inadmissible spec is a VERDICT, not open work, and must name a prohibition
# (absent | derived | repaired | harness-owned | observation).
#
# S1/S2 SEMANTICS ARE UNTOUCHED. specs §7 reserves any change to the detection
# denominator for v3 with owner sign-off; this block is additive so the frozen
# E1 baseline stays byte-comparable.
PATH_CLASSES = ("P1", "P2", "P3", "P4", "L")
PROHIBITIONS = ("absent", "derived", "repaired", "harness-owned", "observation")
RTC_LEVELS = ("entry", "low", "medium", "high")

# `cost` in the E2 ledger is the CHEAPEST -rtc level at which an obligation is
# emitted and checked (`assert_site.cpp:18-20`; buckets `:123-132`; default
# threshold ENTRY `context.hpp:549`). `enabled` is the observed consequence at
# the pinned level. So the enabled-count curve is a property of the LEDGER and
# is derivable without rerunning anything -- which is the whole point of §9.2.
_RTC_ORDER = {lv: i for i, lv in enumerate(RTC_LEVELS)}


def _resolve_spec_id(m):
    """The spec_id of a mutant record, or None if it cannot be resolved.

    v2.1 manifests carry `spec_id` directly. The FROZEN v1 baseline in
    results/choreo/mutant.jsonl does not -- it carries the legacy 1-based `spec`
    index -- so the v1 vocabulary is joined through mutations.V1_SPEC_ID rather
    than pattern-matched off the mutant_id. Unresolvable records are COUNTED and
    named, never dropped: a silently shrunk denominator is the failure mode.
    """
    sid = m.get("spec_id")
    if sid:
        return sid
    import mutations as _M
    s = m.get("spec")
    key = (m.get("class"), s if isinstance(s, int) else
           (s[0] if isinstance(s, list) and s else None))
    return _M.V1_SPEC_ID.get(key)


def _registry_from_records(specs):
    """{spec_id: meta} from `spec` design records (collect.py, DESIGN_GRAIN)."""
    out = {}
    for s in specs:
        sid = s.get("spec_id")
        if not sid:
            continue
        out[sid] = {
            "class": s.get("class") or s.get("cls"),
            "path_class": s.get("path_class") or s.get("path"),
            "applicable": s.get("applicable", s.get("admissible")),
            "prohibition": s.get("prohibition") or "",
            "status": s.get("status"),
            "desc": s.get("desc"),
            "note": s.get("note"),
        }
    return out


def _registry_from_raw():
    """Fallback: raw/spec_registry.json, authored by `gen_mutants.py --dry-run`.

    The register is a DESIGN artifact -- written before, and independently of,
    any run -- so reading it from raw/ when collect.py has not yet re-emitted it
    as spec.jsonl does not weaken the "stats reads the validated projection"
    rule the way reading a lane's raw output would. The source is recorded in
    the output either way so a reader can see which path was taken.
    """
    p = os.path.join(RAW, "spec_registry.json")
    if not os.path.exists(p):
        return {}, None
    with open(p) as f:
        d = json.load(f)
    recs = d.get("specs", [])
    # Normalise the registry file's field names onto the schema's.
    for r in recs:
        r.setdefault("class", r.get("cls"))
        r.setdefault("path", r.get("path"))
    return _registry_from_records(recs), os.path.relpath(p, B2)


def s14_path_class(mutants, specs, obligations, present):
    """The admissible-denominator register, the N/A reasons, and the -rtc curve.

    Emitted in the shape specs/expansion-workflow.md §6 fixed, so `tab:e1-path-
    class` can be rendered from this block alone.
    """
    reg, reg_src = _registry_from_records(specs), "results/choreo/spec.jsonl"
    if not reg:
        reg, reg_src = _registry_from_raw()
    if not reg:
        r = not_run("E1 path-class register (§5.2; tab:e1-path-class)",
                    "no spec register: neither results/choreo/spec.jsonl nor "
                    "raw/spec_registry.json exists. Run `make choreo-screen` "
                    "(gen_mutants.py --dry-run) then `run.sh collect`.")
        r["expected_records"] = "mutant,spec"
        return r

    # ---- axis 1+2, over SPECS: the denominator --------------------------
    n_applicable = sum(1 for v in reg.values() if v["applicable"])
    n_not_applicable = sum(1 for v in reg.values() if not v["applicable"])
    prohibition = collections.Counter(v["prohibition"] for v in reg.values()
                                      if v["prohibition"])
    # Printed WITH the path, because "28 N/A" alone is unfalsifiable while
    # "22 absent + 3 repaired + 3 observation" is checkable against the tree.
    prohibition_by_path = collections.defaultdict(collections.Counter)
    for v in reg.values():
        if not v["applicable"]:
            prohibition_by_path[v["path_class"] or "?"][
                v["prohibition"] or "NO-PROHIBITION"] += 1

    # ---- the cell view, over MUTANTS ------------------------------------
    if not present.get("mutant"):
        return not_run("E1 path-class register (§5.2; tab:e1-path-class)",
                       "results/choreo/mutant.jsonl is absent; run "
                       "`run.sh minimal` first.")

    per_path = collections.defaultdict(collections.Counter)
    per_spec = collections.defaultdict(collections.Counter)
    unresolved = []
    for m in mutants:
        sid = _resolve_spec_id(m)
        meta = reg.get(sid)
        if meta is None:
            unresolved.append(m.get("mutant_id") or sid)
            pc = "UNRESOLVED"
            adm, prob, cls = None, "", m.get("class")
        else:
            pc = meta["path_class"]
            adm, prob, cls = meta["applicable"], meta["prohibition"], meta["class"]
        per_path[pc]["n_cells"] += 1
        srow = per_spec[sid or "(unresolved)"]
        srow["class"] = cls or ""
        srow["path_class"] = pc
        srow["prohibition"] = prob
        srow["n_cells"] += 1
        # A noop is discarded from the detection denominator (specs §11.1): its
        # own manifest says it corrupts nothing, so counting it as a detection
        # would credit choreo with catching a non-bug.
        if m.get("manifest") == "noop":
            per_path[pc]["n_discarded_noop"] += 1
            srow["n_discarded_noop"] += 1
            continue
        per_path[pc]["n_injected"] += 1
        srow["n_injected"] += 1
        oc = m.get("outcome")
        if oc in OUTCOMES:
            per_path[pc][f"n_{oc}"] += 1
            srow[f"n_{oc}"] += 1
        # `n_detected` counts EVERY detected cell in this path, admissible or
        # not, so `reconciliation` can check S14's partition against S2's
        # n_before_device. The admissible-only subset is a SEPARATE key, because
        # collapsing the two would make the audit below blind: it looks for
        # detections on inadmissible cells, which is exactly what a single
        # gated counter would hide.
        detected = oc in ("compile", "runtime")
        if detected:
            per_path[pc]["n_detected"] += 1
            srow["n_detected"] += 1
        if adm:
            per_path[pc]["n_admissible"] += 1
            srow["n_admissible"] += 1
            if detected:
                per_path[pc]["n_detected_admissible"] += 1
                srow["n_detected_admissible"] += 1
            if oc == "never":
                # Admissible-only `never`. `n_never` above spans BOTH axes, so a
                # share computed from it can exceed 100% (37 all-path nevers
                # against 33 admissible misses). This key is the one the miss
                # share is defined on; `n_never` stays for reconciliation.
                per_path[pc]["n_never_admissible"] += 1
                srow["n_never_admissible"] += 1

    # The L row has no detection denominator BY CONSTRUCTION (attribution only),
    # so it reports its own quantities rather than a misleading 0-detected rate.
    for pc, c in per_path.items():
        c["n_detected_pct"] = pct(c.get("n_detected_admissible", 0),
                                  c.get("n_admissible", 0))
        # Misses, ON THE SAME POPULATION as n_detected_pct: admissible injected
        # cells that were not caught. Both terms must be admissible-only or the
        # ratio is meaningless.
        c["n_never_share_of_misses"] = pct(
            c.get("n_never_admissible", 0),
            c.get("n_admissible", 0) - c.get("n_detected_admissible", 0))

    # ---- AUDIT: does the design claim survive the records? --------------
    # The register says a spec is inadmissible because its mutation cannot
    # corrupt anything ("absent"), or because the compiler refuses it rather than
    # testing it ("repaired"). That is a DESIGN claim. The records carry the
    # ground truth (`manifest`: corrupts | noop). If the cells of an inadmissible
    # spec are routinely `corrupts` AND detected, the design claim is wrong and
    # the spec belongs in the denominator.
    #
    # This is reported, never auto-applied: moving a spec back into the
    # denominator changes S1/S2, which specs §7 reserves for v3 with owner
    # sign-off, and would invalidate the frozen E1 baseline.
    audit, contradictions = [], []
    for sid, srow in per_spec.items():
        if srow.get("n_admissible") or not srow.get("n_injected"):
            continue
        n_det = srow.get("n_detected", 0)
        n_never = srow.get("n_never", 0)
        # A cell is only evidence if the manifest itself says it corrupts, which
        # n_injected already guarantees (noops are excluded before this point).
        contradicts = n_det > 0
        row = {
            "spec_id": sid,
            "class": srow.get("class"),
            "path_class": srow.get("path_class"),
            "prohibition": srow.get("prohibition"),
            "desc": (reg.get(sid) or {}).get("desc"),
            "n_injected": srow.get("n_injected", 0),
            "n_detected": n_det,
            "n_never": n_never,
            "contradicts_design_claim": contradicts,
            "design_note": (reg.get(sid) or {}).get("note"),
        }
        audit.append(row)
        if contradicts:
            contradictions.append(row)
    audit_ok = not contradictions

    # ---- reconciliation against S1/S2 -----------------------------------
    # S14 must partition the SAME cells S2 counts, or one of the two is wrong.
    # Asserted rather than assumed, because the whole value of a second
    # denominator is that a reader can check it against the first.
    tot_cells = sum(c.get("n_cells", 0) for c in per_path.values())
    tot_inj = sum(c.get("n_injected", 0) for c in per_path.values())
    tot_noop = sum(c.get("n_discarded_noop", 0) for c in per_path.values())
    tot_det = sum(c.get("n_detected", 0) for c in per_path.values())
    tot_det_adm = sum(c.get("n_detected_admissible", 0)
                      for c in per_path.values())
    tot_never = sum(c.get("n_never", 0) for c in per_path.values())
    tot_adm = sum(c.get("n_admissible", 0) for c in per_path.values())
    # Detections on inadmissible cells are the difference between S2's headline
    # and the admissible-only rate; naming it is what stops 48.6% being read as
    # "over the admissible denominator".
    inadm_det = sum(r["n_detected"] for r in audit)
    inadm_inj = sum(r["n_injected"] for r in audit)

    # ---- the -rtc enabled-obligation curve ------------------------------
    # Steepness here is the paper's thesis (specs §9.2): does the runtime-check
    # COST filter, rather than the absence of checks, own the misses? The ledger
    # answers the "enabled" half exactly; the "detected" half needs the rtc=all
    # arm, so it is reported from the recorded probe and left None where no run
    # exists -- never a fabricated zero.
    rtc = {"status": "derived_from_ledger",
           "definition": "enabled = cost <= threshold, per the E2 obligation "
                         "ledger. No rerun is needed for this column.",
           "levels": list(RTC_LEVELS),
           "n_obligations": None,
           "n_obligations_enabled": {},
           "delta_vs_entry": {},
           "by_class": {},
           "paper_reference": {
               "source": "specs/mutation-specs-v2.md §9.2",
               "entry": 34, "low": 47, "medium": 79, "high": 323,
               "predicted_by_class": {"M1": "1 -> 290 steep",
                                      "M2": "0 -> 0 flat",
                                      "M3": "22 -> 22 flat",
                                      "M4": "11 -> 11 flat"},
               "note": "PREDICTION, recorded on a different population. The "
                       "ledger below is the measurement; report the ledger.",
           }}
    obls = obligations or []
    costed = [o for o in obls if o.get("cost")]
    if costed:
        by_class = collections.defaultdict(collections.Counter)
        for o in costed:
            by_class[o.get("class") or "?"][o["cost"]] += 1
        allc = collections.Counter()
        for c in by_class:
            allc.update(by_class[c])

        def _curve(cnt):
            run, out = 0, {}
            for lv in RTC_LEVELS:
                run += cnt.get(lv, 0)
                out[lv] = run
            return out

        base = 0
        for lv in RTC_LEVELS:
            base += allc.get(lv, 0)
            rtc["n_obligations_enabled"][lv] = base
            rtc["delta_vs_entry"][lv] = base - allc.get(RTC_LEVELS[0], 0)
        for c in sorted(by_class):
            rtc["by_class"][c] = _curve(by_class[c])
        rtc["n_obligations"] = sum(allc.values())
        e0 = rtc["n_obligations_enabled"][RTC_LEVELS[0]]
        hi = rtc["n_obligations_enabled"][RTC_LEVELS[-1]]
        rtc["steepest_class"] = max(
            rtc["by_class"],
            key=lambda c: rtc["by_class"][c][RTC_LEVELS[-1]]
            - rtc["by_class"][c][RTC_LEVELS[0]])
        sc = rtc["steepest_class"]
        rtc["cost_filter_note"] = (
            f"Enabling every obligation costs {hi}/{e0} = "
            f"{round(hi / e0, 2) if e0 else None}x the default, and the growth "
            f"is not uniform: {sc} goes {rtc['by_class'][sc][RTC_LEVELS[0]]} -> "
            f"{rtc['by_class'][sc][RTC_LEVELS[-1]]} while the other classes are "
            f"flat. That is the thesis's shape, measured from the ledger.")

        # The DETECTION half: measured only at the two recorded arms.
        entry_det = sum(1 for m in mutants
                        if m.get("manifest") != "noop"
                        and m.get("outcome") in ("compile", "runtime"))
        n_inj = sum(1 for m in mutants if m.get("manifest") != "noop")
        rtc["n_detected"] = {"entry": entry_det, "low": None, "medium": None,
                             "high": None}
        rtc["n_detected_note"] = (
            "`enabled` is a ledger property and is exact at every level; "
            "`n_detected` needs a full E1 run per level, so only `entry` is "
            "measured here. The rtc=all arm exists (see rtc_all_arm) and is "
            "reported separately rather than interpolated.")
        rtc["denominator_us"] = f"{entry_det}/{n_inj} at entry (pinned default)"

    # The recorded -rtc=all arm: the CEILING. Kept separate from the levels
    # above because it is a different experiment (one flag, no sweep) and
    # because it must not be confused with the hoist-disabled arm -- combining
    # them would hide a codegen bug behind a flag (R-D2).
    rtc_all = _read_rtc_all_arm()
    if rtc_all:
        t = rtc["n_detected"]
        t["high (rtc=all arm)"] = rtc_all["n_detected"]
        rtc["rtc_all_arm"] = rtc_all

    out = {
        "status": "ok",
        "feeds": "E1 path-class register (§5.2; new float tab:e1-path-class)",
        "register_source": reg_src,
        "denominator": {
            "n_specs": len(reg),
            "n_applicable": n_applicable,
            "n_not_applicable": n_not_applicable,
            "n_out_of_scope": sum(1 for v in reg.values()
                                  if v["applicable"] and v["status"] != "implemented"),
            "rule": "admissible specs (P1/P3/P4 + the L row reported separately) "
                    "form the detection denominator; an inadmissible spec is a "
                    "recorded verdict with a named prohibition and is NOT open "
                    "work (specs §2)",
        },
        "prohibition": {k: int(prohibition.get(k, 0)) for k in PROHIBITIONS},
        "prohibition_by_path": {p: dict(c) for p, c
                                in sorted(prohibition_by_path.items())},
        "per_path": {p: {k: (int(v) if isinstance(v, int) else v)
                         for k, v in sorted(per_path[p].items())}
                     for p in sorted(per_path)},
        "per_spec": {k: {kk: vv for kk, vv in sorted(v.items())}
                     for k, v in sorted(per_spec.items())},
        "unresolved_spec_cells": sorted(unresolved),
        "applicability_audit": {
            "question": "does each inadmissible spec's DESIGN claim survive the "
                        "record-level ground truth?",
            "status": "ok" if audit_ok else "CONTRADICTED",
            "n_inadmissible_specs_with_cells": len(audit),
            "n_contradicting": len(contradictions),
            "rows": audit,
            "note": "A `contradicts_design_claim` row has cells whose own "
                    "manifest says they corrupt, AND which choreo detected -- so "
                    "the mutation is testable and the spec belongs in the "
                    "denominator. This is REPORTED, never auto-applied: moving "
                    "it changes S1/S2, which specs §7 reserves for v3 with "
                    "owner sign-off.",
        },
        "reconciliation": {
            "against": "S1_detection_matrix / S2_before_device",
            "n_cells": tot_cells,
            "n_injected": tot_inj,
            "n_discarded_noop": tot_noop,
            "n_never": tot_never,
            "n_detected_all_paths": tot_det,
            "admissible_injected": tot_adm,
            "admissible_detected": tot_det_adm,
            "admissible_pct": pct(tot_det_adm, tot_adm),
            "inadmissible_injected": inadm_inj,
            "inadmissible_detected": inadm_det,
            "matches_S1S2": None,
            "interpretation": (
                "S2's headline rate divides by ALL injected cells; the "
                "admissible rate divides only by the cells the register says "
                "choreo was fair to ask about. Both must be printed: "
                f"{inadm_det} of S2's detections land on cells the register "
                "calls inadmissible, so quoting S2's rate as an admissible-"
                "denominator rate would overstate it."),
        },
        "rtc_curve": rtc,
        "feeds_note": ("S1/S2 semantics are UNCHANGED by this block. specs §7 "
                       "reserves any change to the detection denominator for v3 "
                       "with owner sign-off."),
    }
    out.update(prov(mutants))
    return out


def reconcile_s14(s14, s1, s2):
    """Cross-check S14's partition against S1/S2, in place.

    Two statistics over the same cells must agree on the cells. If they do not,
    one of them is wrong and the integrator has to know which number to trust,
    so the check is recorded in the JSON rather than left to a reader to notice.
    """
    if s14.get("status") != "ok" or s1.get("status") != "ok":
        return
    r, t1 = s14["reconciliation"], s1["totals"]
    checks = {
        "n_injected": (r["n_injected"], t1.get("n_injected")),
        "n_discarded_noop": (r["n_discarded_noop"], t1.get("n_discarded_noop")),
        "n_never": (r["n_never"], t1.get("n_never")),
        "n_detected_all_paths": (r["n_detected_all_paths"],
                                 s2.get("n_before_device")),
    }
    mismatch = {k: {"s14": a, "s1s2": b} for k, (a, b) in checks.items() if a != b}
    r["matches_S1S2"] = not mismatch
    r["checks"] = {k: {"s14": a, "s1s2": b} for k, (a, b) in checks.items()}
    if mismatch:
        r["mismatch"] = mismatch
        r["mismatch_note"] = (
            "S14 partitions the same cells S1/S2 do, so these must be equal. A "
            "mismatch means a mutant record carries an unresolvable spec, or the "
            "register disagrees with the manifest. Do not quote either rate "
            "until it is resolved.")


def _read_rtc_all_arm():
    """The recorded `-rtc=all` detection arm, or None.

    Returns None rather than a zero when the probe is absent, and reports the
    arm's own flags so a reader can see it is NOT a level sweep: it is one
    configuration, and its delta over the default is the cost filter's real
    upper bound on detection.
    """
    p = os.path.join(RAW, "e1_mutant_records_rtc_all.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except (ValueError, OSError):
        return None
    recs = d.get("records") or []
    inj = [r for r in recs if r.get("manifest") != "noop"]
    det = [r for r in inj if r.get("outcome") in ("compile", "runtime")]
    return {
        "source": os.path.relpath(p, B2),
        "pinned_flags": (d.get("pinned_flags") or {}).get("compile"),
        "n_injected": len(inj),
        "n_detected": len(det),
        "pct": pct(len(det), len(inj)),
        "is_a_sweep": False,
        "note": "One configuration, not a level sweep. Delta over the pinned "
                "default is the cost filter's measured upper bound on "
                "detection. The 43/72 figure in S2's flag matrix is a DIFFERENT "
                "arm (rtc=all AND --disable-assert-hoist) and must not be quoted "
                "as the cost filter's effect: 6 of its 8 extra detections come "
                "from a codegen hoisting defect (R-D2).",
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=RESULTS,
                    help="collected records dir (default: benchmark2/results/choreo)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <results>/stats.json)")
    ap.add_argument("--require", default="",
                    help="comma-separated record types that must be non-empty "
                         "(exit 2 if not); e.g. mutant,obligation,cost")
    args = ap.parse_args()

    results = os.path.abspath(args.results)
    out_path = os.path.abspath(args.out) if args.out \
        else os.path.join(results, "stats.json")

    data, present = load_all(results)
    idt = toolchain.identity()

    print(f"[choreo] stats: reading {results}")
    for n in ("kernel", "mutant", "spec", "obligation", "cost", "residue",
              "latency"):
        mark = "present" if present.get(n) else "ABSENT"
        print(f"    {n:<14} {len(data[n]):>7} records   ({mark})")

    s1, s2 = s1_s2(data["mutant"], present)
    s14 = s14_path_class(data["mutant"], data["spec"],
                         data["obligation"], present)
    reconcile_s14(s14, s1, s2)
    stats = {
        "toolchain": TOOLCHAIN,
        "toolchain_version": idt["version"],
        "toolchain_identity": idt,
        "produced_by": "stats.py",
        "inputs": {n: len(data[n]) for n in sorted(data)},
        "lane_owns": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S10",
                      "S13", "S14"],
        "S1_detection_matrix": s1,
        "S2_before_device": s2,
        "S3_generation_totals": s3(data["obligation"]),
        "S4_per_operator": s4(data["obligation"], data["kernel"]),
        "S5_discharge_rate": s5(data["obligation"], data["kernel"]),
        "S6_mechanism_split": s6(data["obligation"]),
        "S7_no_interval_counterfactual": s7(data["obligation"]),
        "S10_compile_cost": s10(data["cost"]),
        "S13_runtime_and_latency": s13(data["residue"], data["latency"], present),
        # S14 is ADDITIVE and reads S1's own records plus the design register.
        # It must never feed back into S1/S2 (specs §7: denominator changes are
        # a v3 item).
        "S14_path_class": s14,
        # S8, S9, S11, S12 belong to other lanes (statistics-manifest.md):
        # S8/S9/S12 to the SOTA workers, S11 to the coordinator. Recorded as
        # absent-on-purpose so the integrator does not read a missing key as a
        # choreo gap.
        "not_owned_by_this_lane": {
            "S8": "SOTA expressibility (triton/mlir-linalg/mlir-low/iree)",
            "S9": "SOTA remainder (triton/mlir-linalg/mlir-low/iree)",
            "S11": "integrity register (coordinator)",
            "S12": "sanitizer supplement (SOTA toolchains)",
        },
    }
    if idt.get("binary_stale_vs_checkout"):
        stats["toolchain_warning"] = (
            f"binary predates checkout: records say {idt['version']} (what the "
            f"binary was built from), checkout HEAD is {idt['checkout_head']}. "
            "Rebuild and re-run before pinning this version in manifest §5."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=1, sort_keys=False)
    print(f"\n[choreo] wrote {os.path.relpath(out_path, B2)}")

    report(stats)

    if args.require:
        missing = []
        for n in [x.strip() for x in args.require.split(",") if x.strip()]:
            if not data.get(n):
                missing.append(n)
        if missing:
            print(f"\n*** --require unsatisfied: no records for {missing} ***")
            return 2
    return 0


def report(st):
    """Print the human-readable summary. This is what a reviewer reads first;
    stats.json is what the integrator merges."""
    line = "=" * 74

    print(f"\n{line}\nS1/S2  DETECTION MATRIX (E1)\n{line}")
    s1 = st["S1_detection_matrix"]
    if s1.get("status") != "ok":
        print(f"  NOT RUN — {s1.get('reason')}")
    else:
        hdr = f"  {'class':<6}{'injected':>10}{'compile':>9}{'runtime':>9}{'never':>7}{'n/a':>6}{'noop':>6}"
        print(hdr)
        for cls in CLASSES:
            d = s1["per_class"].get(cls, {})
            print(f"  {cls:<6}{d.get('n_injected',0):>10}{d.get('n_compile',0):>9}"
                  f"{d.get('n_runtime',0):>9}{d.get('n_never',0):>7}"
                  f"{d.get('n_na',0):>6}{d.get('n_discarded_noop',0):>6}")
        t = s1["totals"]
        print(f"  {'ALL':<6}{t.get('n_injected',0):>10}{t.get('n_compile',0):>9}"
              f"{t.get('n_runtime',0):>9}{t.get('n_never',0):>7}"
              f"{t.get('n_na',0):>6}{t.get('n_discarded_noop',0):>6}")
        s2 = st["S2_before_device"]
        print(f"\n  S2 before-device: {s2['n_before_device']}/{s2['n_injected']} "
              f"= {s2['pct_before_device']}%")
        # R-D2: `never = 0` is RETIRED. The criterion is now "no unattributed
        # never", so that is what gets a PASS/FAIL line. Printing the retired
        # assertion as *** FAIL *** would misreport a settled ruling as an open
        # defect.
        na = s2.get("never_attribution", {})
        print(f"     never = {s2['n_never']}   "
              f"(criterion `never == 0` is RETIRED, R-D2)")
        print(f"     criterion `no unattributed never`: "
              f"{'PASS' if na.get('no_unattributed_never') else '*** FAIL ***'}"
              f"   ({na.get('n_unattributed', 0)} unattributed)")
        if na.get("cause_counts"):
            print(f"       causes: " + "  ".join(
                f"{k}={v}" for k, v in na["cause_counts"].items()))
            print(f"       {na.get('n_out_of_scope_or_not_assessed',0)} "
                  f"out-of-scope/not-assessed + "
                  f"{na.get('n_toolchain_or_config',0)} hoisting/config")
            fm = na.get("flag_matrix_ablation")
            if fm:
                print(f"       ablation (R-D2b, NOT the headline): "
                      f"{fm['at_pinned_default']} pinned -> "
                      f"{fm['with_rtc_all_and_hoist_disabled']} with "
                      f"-rtc=all + hoist disabled "
                      f"(+{fm['recovered_by_flags']} recovered)")
        for m in s2.get("never_mutants", [])[:12]:
            print(f"       {m['mutant_id']:<28} {m['class']}/{m['category']:<22}"
                  f" cause={m.get('never_cause')}")
        if len(s2.get("never_mutants", [])) > 12:
            print(f"       ... and {len(s2['never_mutants'])-12} more")

    print(f"\n{line}\nS3  GENERATION TOTALS (E2)\n{line}")
    s3v = st["S3_generation_totals"]
    if s3v.get("status") != "ok":
        print(f"  NOT RUN — {s3v.get('reason')}")
    else:
        print(f"  grand total: {s3v['grand_total']}   "
              f"(paper {s3v['paper_reference']['grand_total']}, "
              f"delta {s3v['paper_reference']['delta']:+d})")
        print("  per class: " + "  ".join(
            f"{k}={v}" for k, v in s3v["per_class"].items()))

    print(f"\n{line}\nS4/S5  PER-OPERATOR LEDGER + DISCHARGE RATE\n{line}")
    s4v, s5v = st["S4_per_operator"], st["S5_discharge_rate"]
    if s4v.get("status") == "ok":
        print(f"  {'category':<22}{'expressed':>10}{'disch':>8}{'runtime':>9}"
              f"{'budget':>8}{'drop':>6}{'rate%':>8}{'zero-k':>8}")
        for c, d in s4v["per_category"].items():
            print(f"  {c:<22}{d['expressed']:>10}{d['discharged']:>8}"
                  f"{d['runtime']:>9}{d['budgeted']:>8}{d['dropped']:>6}"
                  f"{(d['discharge_pct'] or 0):>8.2f}"
                  f"{len(d['zero_obligation_kernels']):>8}")
        tt = s4v["totals"]
        print(f"  {'TOTAL':<22}{tt['expressed']:>10}{tt['discharged']:>8}"
              f"{tt['runtime']:>9}{tt['budgeted']:>8}{tt['dropped']:>6}")
        print(f"\n  dropped == 0 everywhere: "
              f"{'PASS' if s4v['dropped_is_zero_everywhere'] else '*** FAIL ***'}")
        print(f"  kernels with obligations: {s4v['n_kernels_with_obligations']}   "
              f"zero-obligation kernels: {s4v['n_zero_obligation_kernels']} "
              f"(all reshape: a span reinterpretation has no element access to bound)")
        ln = s4v.get("layer_norm_criterion")
        if ln and "matches" in ln:
            print(f"  layer_norm criterion (PER-KERNEL {ln['kernel_id']}): "
                  f"expected {ln['expected']} -> measured {ln['measured']}: "
                  f"{'PASS' if ln['matches'] else '*** FAIL ***'}")
        elif ln:
            print(f"  layer_norm criterion: {ln.get('status')} — {ln.get('reason')}")
    if s5v.get("status") == "ok":
        pr = s5v["paper_reference"]
        print(f"\n  S5 discharge rate (paper: all {pr['all_pct']}%, "
              f"static {pr['static_pct']}%, dynamic {pr['dynamic_pct']}%)")
        for k in ("static", "dynamic", "all"):
            b = s5v[k]
            print(f"     {k:<9} {b['discharged']:>6}/{b['expressed']:<6} "
                  f"= {b['discharge_pct']}%   ({b.get('kernels','-')} kernels)")

    print(f"\n{line}\nS6  MECHANISM SPLIT\n{line}")
    s6v = st["S6_mechanism_split"]
    if s6v.get("status") == "ok":
        print(f"  {'outcome':<12}" + "".join(f"{m:>12}" for m in MECHANISMS)
              + f"{'total':>10}")
        for oc, d in s6v["per_outcome"].items():
            print(f"  {oc:<12}" + "".join(f"{d[m]:>12}" for m in MECHANISMS)
                  + f"{d['total']:>10}")
        print("  TOTALS     " + "".join(f"{v:>12}"
                                        for v in s6v["totals"].values()))

    print(f"\n{line}\nS7  NO-INTERVAL COUNTERFACTUAL\n{line}")
    s7v = st["S7_no_interval_counterfactual"]
    print(f"  status: {s7v['status']}")
    print(f"  {s7v.get('reason','')}")
    if s7v.get("interval_discharged_upper_bound") is not None:
        print(f"  interval-discharged obligations (upper bound on the effect): "
              f"{s7v['interval_discharged_upper_bound']} "
              f"({s7v['interval_pct_of_all']}% of all)")

    print(f"\n{line}\nS10  COMPILE OVERHEAD (E4)\n{line}")
    s10v = st["S10_compile_cost"]
    if s10v.get("status") != "ok":
        print(f"  NOT RUN — {s10v.get('reason')}")
    else:
        print(f"  {'category':<22}{'n':>4}{'median%':>10}{'min%':>9}{'max%':>9}"
              f"{'bucket':>10}{'excl':>7}")
        for c, d in sorted(s10v["per_category"].items()):
            print(f"  {c:<22}{(d['n_kernels'] or 0):>4}{(d['median_pct'] or 0):>10.3f}"
                  f"{(d['min_pct'] or 0):>9.3f}{(d['max_pct'] or 0):>9.3f}"
                  f"{str(d['bucket']):>10}"
                  f"{('yes' if d['exclusive'] else 'NO'):>7}")
        print(f"  {'-'*71}")
        print(f"  {'GRAND MEDIAN':<22}{s10v['n_categories']:>4}"
              f"{(s10v['grand_median_pct'] or 0):>10.3f}")
        print(f"\n  RQ4 says {s10v['paper_reference']['rq4_median_pct']}% "
              f"(bucket {s10v['paper_reference']['rq4_bucket']}); "
              f"reproduces: {s10v.get('reproduces_rq4')}")
        if s10v.get("rq4_note"):
            print(f"  *** {s10v['rq4_note']}")
        if s10v.get("admissibility_note"):
            print(f"  *** {s10v['admissibility_note']}")

    print(f"\n{line}\nS13  RUNTIME RESIDUE + DETECTION LATENCY (E5)\n{line}")
    s13v = st["S13_runtime_and_latency"]
    e5a = s13v.get("E5a", {})
    if e5a.get("status") != "ok":
        print(f"  E5a NOT RUN — {e5a.get('reason')}")
    else:
        print(f"  E5a: {e5a['n_cases']} paired cases "
              f"({e5a['n_unpaired']} unpaired)")
        print(f"       median delta = {e5a['median_delta_pct']}% "
              f"({e5a['median_delta_ms']} ms)")
        print(f"       within noise = {e5a['n_delta_within_noise']}/{e5a['n_cases']}"
              f"   negative delta = {e5a['n_negative_delta']}"
              f"   clock drift = {e5a['n_with_clock_drift']}")
        dce = e5a.get("device_code_evidence")
        if dce:
            print(f"       device code: {dce['n_device_identical']}/"
                  f"{dce['n_rows_reporting']} rows byte-identical across arms, "
                  f"{dce['total_device_side_check_calls']} device-side check "
                  f"call(s), {dce['host_gated_checks_min']}-"
                  f"{dce['host_gated_checks_max']} host-side gated checks")
        for k in ("negative_delta_note", "noise_note",
                  "structural_contradiction_note", "static_premise_note"):
            if e5a.get(k):
                print(f"       *** {e5a[k]}")
    e5b = s13v.get("E5b", {})
    if e5b.get("status") != "ok":
        print(f"  E5b NOT RUN — {e5b.get('reason')}")
    else:
        print(f"  E5b: {e5b['n_records']} latency records")
        for g, d in e5b["per_detector_and_check_loc"].items():
            print(f"       {g:<34} n={d['n']:<4} median={d['median_us']:>10.1f} us"
                  f"  [{d['min_us']:.1f}, {d['max_us']:.1f}]")
        rb = e5b.get("ratio") or {}
        if rb.get("n_pairs"):
            # The headline. Print the PAIRED ratio, not median/median: dividing
            # two per-detector medians would compare unlike kernels.
            print(f"       paired ratio (oracle / choreo, n={rb['n_pairs']}): "
                  f"median {rb['median_ratio']}x  "
                  f"[{rb['min_ratio']}x, {rb['max_ratio']}x]  "
                  f"oracle slower on {rb['n_pairs_oracle_slower']}/"
                  f"{rb['n_pairs']}")
        # The guard's effect must be VISIBLE on the console, not only in the
        # JSON. A reader who sees "n=10" without seeing "16 selected, 6 held
        # out" cannot tell whether the denominator shrank by design or by loss.
        if rb.get("n_confounded"):
            print(f"       LAUNCH/DETECTION GUARD: {rb['n_confounded']} of "
                  f"{rb['n_pairs_before_guard']} pair(s) held out "
                  f"(n={rb['n_pairs']} is the CLEAN count):")
            for c in rb.get("confounded_pairs", []):
                why = c.get("reason")
                why = why[0] if isinstance(why, list) else why
                print(f"         - {c['mutant_id']:<26} {why:<20} "
                      f"ratio would have been "
                      f"{c['ratio_sanitizer_over_choreo']}x")
            ug = rb.get("unguarded_all_pairs") or {}
            if ug:
                print(f"         unguarded median was {ug.get('median_ratio')}x "
                      f"over n={ug.get('n_pairs')} -- NOT QUOTABLE (mixes "
                      f"refused launches with runs that reported nothing).")
        asym = e5b.get("detection_asymmetry")
        if asym:
            nb = asym["n_oracle_blind"]
            nn = asym["n_oracle_never_executed"]
            den = nb + asym["n_both_detected"]
            if nb:
                print(f"       DETECTION ASYMMETRY: {nb} of {den} mutant(s) "
                      f"where the oracle actually executed were caught by "
                      f"choreo's entry check and MISSED by compute-sanitizer "
                      f"(`ERROR SUMMARY: 0 errors`):")
                for m in asym["oracle_blind_mutants"]:
                    print(f"         - {m['mutant_id']:<26} "
                          f"choreo={m['choreo_entry']:<9} "
                          f"sanitizer={m['compute_sanitizer']}")
                print(f"         This is COVERAGE, not speed: the oracle "
                      f"produced no report at all, so these pairs are excluded "
                      f"from the latency ratio above and reported here instead.")
            if nn:
                print(f"       ORACLE NEVER EXECUTED: {nn} further mutant(s) "
                      f"had their sanitizer arm REFUSED at launch "
                      f"(cudaErrorInvalidValue), so the oracle never ran the "
                      f"body. NOT counted as detection misses above -- a "
                      f"launch-level refusal is not a detection judgement:")
                for m in asym["oracle_never_executed_mutants"]:
                    print(f"         - {m['mutant_id']:<26} "
                          f"choreo={m['choreo_entry']:<9} "
                          f"sanitizer={m['compute_sanitizer']}")
            nd = asym.get("n_oracle_died_no_report") or 0
            if nd:
                print(f"       ORACLE DIED WITHOUT REPORTING: {nd} mutant(s) "
                      f"whose sanitizer arm exited non-zero with no fault line. "
                      f"Also NOT counted as misses -- the oracle never rendered "
                      f"a verdict:")
                for m in asym["oracle_died_mutants"]:
                    print(f"         - {m['mutant_id']:<26} "
                          f"choreo={m['choreo_entry']:<9} "
                          f"sanitizer={m['compute_sanitizer']}")
        if e5b.get("absolute_latency_note"):
            print(f"       *** {e5b['absolute_latency_note']}")

    print(f"\n{line}\nS14  PATH-CLASS REGISTER (v2.1)\n{line}")
    s14 = st.get("S14_path_class", {})
    if s14.get("status") != "ok":
        print(f"  NOT RUN — {s14.get('reason')}")
    else:
        d = s14["denominator"]
        print(f"  specs {d['n_specs']}   admissible {d['n_applicable']}   "
              f"not applicable {d['n_not_applicable']}   "
              f"open work {d['n_out_of_scope']}")
        print(f"  register source: {s14['register_source']}")
        # The prohibition breakdown is printed WITH the path on purpose. "28 N/A"
        # is unfalsifiable; "22 absent + 3 repaired + 3 observation" is checkable.
        print(f"  prohibitions: " + "  ".join(
            f"{k}={v}" for k, v in s14["prohibition"].items() if v))
        for p, c in s14["prohibition_by_path"].items():
            print(f"    {p:<4} " + "  ".join(f"{k}={v}" for k, v in c.items()))
        print(f"\n  {'path':<6}{'cells':>7}{'adm':>6}{'det':>6}{'never':>7}"
              f"{'noop':>6}{'det%':>9}   note")
        for p in PATH_CLASSES + ("UNRESOLVED",):
            c = s14["per_path"].get(p)
            if not c:
                continue
            if p == "L":
                note = ("attribution only — never counted in the detection "
                        "denominator")
            elif not c.get("n_admissible"):
                note = "inadmissible (recorded verdict, not open work)"
            else:
                note = ""
            print(f"  {p:<6}{c.get('n_cells',0):>7}{c.get('n_admissible',0):>6}"
                  f"{c.get('n_detected',0):>6}{c.get('n_never',0):>7}"
                  f"{c.get('n_discarded_noop',0):>6}"
                  f"{str(c.get('n_detected_pct')):>9}   {note}")
        if s14.get("unresolved_spec_cells"):
            print(f"  *** {len(s14['unresolved_spec_cells'])} cell(s) joined to "
                  f"NO spec — the denominator is incomplete, not zero: "
                  f"{', '.join(s14['unresolved_spec_cells'][:6])}")

        rec = s14.get("reconciliation", {})
        if rec:
            print(f"\n  reconciliation vs S1/S2: "
                  f"{'AGREES' if rec.get('matches_S1S2') else '*** MISMATCH ***'}"
                  f"   (cells {rec.get('n_cells')}, injected "
                  f"{rec.get('n_injected')}, noop {rec.get('n_discarded_noop')}, "
                  f"never {rec.get('n_never')})")
            print(f"    ALL injected:      {rec.get('n_detected_all_paths')}/"
                  f"{rec.get('n_injected')} detected   <- S2's headline")
            print(f"    ADMISSIBLE only:   {rec.get('admissible_detected')}/"
                  f"{rec.get('admissible_injected')} = "
                  f"{rec.get('admissible_pct')}%   <- the register's denominator")
            if rec.get("inadmissible_detected"):
                print(f"    {rec['inadmissible_detected']} of S2's detections are "
                      f"on cells the register calls INADMISSIBLE "
                      f"({rec['inadmissible_injected']} injected cells). Quote "
                      f"whichever rate matches the denominator you name.")
            if rec.get("mismatch"):
                print(f"    *** MISMATCH {rec['mismatch']}")
                print(f"        {rec['mismatch_note']}")

        au = s14.get("applicability_audit", {})
        if au.get("n_contradicting"):
            print(f"\n  *** APPLICABILITY AUDIT: {au['n_contradicting']} of "
                  f"{au['n_inadmissible_specs_with_cells']} inadmissible spec(s) "
                  f"have cells their OWN MANIFEST says corrupt, and which choreo "
                  f"DETECTED — so the mutation is testable and the spec belongs "
                  f"in the denominator:")
            for r in au.get("rows", []):
                if not r["contradicts_design_claim"]:
                    continue
                print(f"      {r['spec_id']:<8}{r['class']}/{r['path_class']:<3}"
                      f" prohibition={r['prohibition']:<12}"
                      f" injected={r['n_injected']} detected={r['n_detected']}"
                      f"  ({r['desc']})")
            print(f"      Not auto-applied: moving these changes S1/S2, which "
                  f"specs §7 reserves for v3 with owner sign-off.")
        elif au:
            print(f"\n  applicability audit: all "
                  f"{au.get('n_inadmissible_specs_with_cells', 0)} inadmissible "
                  f"spec(s) with cells survive their design claim")

        rtc = s14.get("rtc_curve", {})
        if rtc.get("n_obligations_enabled"):
            print(f"\n  -rtc enabled-obligation curve "
                  f"(derived from the E2 ledger, no rerun needed):")
            print(f"    {'level':<9}{'enabled':>9}{'delta':>8}"
                  f"{'detected':>10}")
            nd = rtc.get("n_detected") or {}
            for lv in rtc["levels"]:
                det = nd.get(lv)
                print(f"    {lv:<9}{rtc['n_obligations_enabled'][lv]:>9}"
                      f"{rtc['delta_vs_entry'][lv]:>+8}"
                      f"{(str(det) if det is not None else 'not run'):>10}")
            for lv, v in (rtc.get("n_detected") or {}).items():
                if lv not in rtc["levels"] and v is not None:
                    print(f"    {lv:<9}{'':>9}{'':>8}{v:>10}")
            print(f"    per class: " + "  ".join(
                f"{c} {v[rtc['levels'][0]]}->{v[rtc['levels'][-1]]}"
                for c, v in sorted(rtc["by_class"].items())))
            if rtc.get("cost_filter_note"):
                print(f"    *** {rtc['cost_filter_note']}")
            if rtc.get("rtc_all_arm"):
                a = rtc["rtc_all_arm"]
                print(f"    ceiling arm ({a['source']}): "
                      f"{a['n_detected']}/{a['n_injected']} = {a['pct']}% "
                      f"detected at -rtc=all")
                print(f"      *** {a['note']}")
    if st.get("toolchain_warning"):
        print(f"\n{line}\nTOOLCHAIN\n{line}\n  *** {st['toolchain_warning']}")
    print()


if __name__ == "__main__":
    sys.exit(main())
