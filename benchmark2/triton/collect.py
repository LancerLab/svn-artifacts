"""collect.py — parse benchmark2/triton/raw/*.jsonl into results/ JSON and
stats.json per benchmark2/schema/. Idempotent: re-parses raw/ every time.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RES = HERE / "results"
B2 = HERE.parent

# The mutation-class axis is NOT restated here. This file used to hold a bare
# `s1.setdefault("M2", ...)` fallback, which decided on its own which classes the
# lane had run -- and therefore never mentioned M4 at all, leaving it absent from
# the detection matrix with nothing saying whether that meant "run and
# inexpressible" or "never run". schema/class-axis.json is the one definition.
if str(B2) not in sys.path:
    sys.path.insert(0, str(B2))
from schema import class_axis as AX                    # noqa: E402
from schema import records as RS                       # noqa: E402

LANE_NAME = "triton"


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def validate(rec: dict, kind: str) -> list[str]:
    """Delegates to schema/records.py -- the ONE record validator.

    This file used to carry its own copy, which read `spec["fields"]` as
    required of every record and therefore rejected this lane's whole committed
    corpus (collected before v2.1) as invalid. See schema/records.py.
    """
    return RS.validate(rec, kind)


def main():
    stats_only = "--stats" in sys.argv
    mutants = load_jsonl(RAW / "mutants.jsonl")
    kernels = load_jsonl(RAW / "kernels.jsonl")

    expr = load_jsonl(RAW / "expressibility.jsonl")
    san = load_jsonl(RAW / "sanitizer.jsonl")
    all_errs = []
    for r in mutants:
        all_errs += validate(r, "mutant")
    for r in kernels:
        all_errs += validate(r, "kernel")
    for r in expr:
        all_errs += validate(r, "expressibility")
    for r in san:
        all_errs += validate(r, "sanitizer")
    fatal = list(dict.fromkeys(all_errs))
    for e in fatal:
        print("SCHEMA ERROR:", e)
    if fatal:
        sys.exit(1)

    # de-duplicate expressibility (e2 re-runs may have appended twice)
    seen = set()
    expr_dedup = []
    for r in expr:
        k = (r.get("category"), r.get("class"))
        if k in seen:
            continue
        seen.add(k)
        expr_dedup.append(r)
    expr = expr_dedup

    if not stats_only:
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "mutants.json").write_text(json.dumps(mutants, indent=1))
        (RES / "kernels.json").write_text(json.dumps(kernels, indent=1))
        (RES / "expressibility.json").write_text(json.dumps(expr, indent=1))
        (RES / "sanitizer.json").write_text(json.dumps(san, indent=1))
        print(f"collected {len(mutants)} mutants, {len(kernels)} kernel records")

    # S12: sanitizer supplement — flagged∧exercised per class
    s12 = {}
    for r in san:
        c = r["class"]
        s12.setdefault(c, {"flagged_and_exercised": 0, "total": 0,
                           "flagged": 0})
        s12[c]["total"] += 1
        if r.get("flagged") == "true":
            s12[c]["flagged"] += 1
            if r.get("exercised") == "true":
                s12[c]["flagged_and_exercised"] += 1
    # stats (schema/statistics-manifest.md): S1 detection matrix for this lane
    by_class = {}
    for r in mutants:
        c = r["class"]
        by_class.setdefault(c, Counter())
        by_class[c][r["outcome"]] += 1
    s1 = {c: {"n_injected": sum(v.values()),
              "n_compile": v.get("compile", 0),
              "n_runtime": v.get("runtime", 0),
              "n_never": v.get("never", 0),
              "n_na": v.get("n/a", 0)}
          for c, v in by_class.items()}
    # Classes this lane RAN and cannot express -> n/a with n_na = N_TARGET.
    # `M2` here is not a literal choice made by this file: it is
    # AX.lane_where("triton", "n/a"), and the reason is read back from the axis
    # rather than paraphrased, so the lane and the axis cannot disagree.
    for cls in AX.lane_where(LANE_NAME, "n/a"):
        s1.setdefault(cls, {
            "n_injected": 0, "n_compile": 0, "n_runtime": 0,
            "n_never": 0, "n_na": AX.n_target_per_class(),
            "note": (f"no expressible {cls} mutant on this surface; all spec "
                     f"N={AX.n_target_per_class()} mutants => n/a. "
                     f"{AX.na_reason(LANE_NAME, cls)}")})

    # Classes this lane was NEVER RUN against get no cell at all, and are
    # declared instead. An absent class and an n/a class mean different things.
    uncompared = AX.uncompared_classes(LANE_NAME)
    for cls in uncompared:
        if cls in s1:
            raise AssertionError(
                f"{LANE_NAME}: {cls} is declared `uncompared` in "
                f"schema/class-axis.json but raw/mutants.jsonl carries records "
                f"for it; either the lane ran it (update the axis) or the "
                f"records are mislabelled")
    s1_declared_uncompared = {
        "classes": sorted(uncompared),
        "reason": AX.uncompared_reason(),
        "note": ("No cell is emitted for these classes. A missing cell is NOT "
                 "an n/a cell: `n/a` means this surface was measured and cannot "
                 "express the defect, while a missing cell means the lane was "
                 "never run against the class."),
    } if uncompared else {}

    # T4: S8 = 4-class × {yes, partial, no} counts (schema + IREE shape).
    s8 = {}
    for r in expr:
        c = r["class"]
        s8.setdefault(c, {"yes": 0, "partial": 0, "no": 0})
        s8[c][r["expressible"]] += 1

    stats = {"toolchain": "triton",
             "S1_detection_matrix": s1,
             "S1_declared_uncompared": s1_declared_uncompared,
             "S1_class_axis": {
                 "source": "schema/class-axis.json",
                 "axis_version": AX.axis()["axis_version"],
                 "mutation_classes": AX.mutation_classes(),
                 "status": AX.lane_status(LANE_NAME),
             },
             "S8_expressibility": {c: v for c, v in sorted(s8.items())},
             "S9_remainder": "n/a (no generated checks)",
             "S12_sanitizer_supplement": s12}
    (RES / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    (RES / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
