#!/usr/bin/env python3
"""benchmark2/iree/collect_stats.py — `run.sh collect` + `run.sh stats`.

collect: copy raw jsonl records into results/iree/ (schema/record-schema.json).
stats:   aggregate into results/iree/stats.json the statistics the iree lane
         owns (schema/statistics-manifest.md): S8 (expressibility counts),
         S9 (remainder counts), plus the per-category kernel gate used by
         tab:per-operator / QC. S1/S12 are produced by `minimal` (E1).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # benchmark2/
LANE = ROOT / "iree"
RAW = LANE / "raw"
RESULTS = ROOT / "results" / "iree"


def load(name: str) -> list[dict]:
    p = RAW / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def cmd_collect():
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("kernels.jsonl", "expressibility.jsonl", "remainder.jsonl",
                 "mutants.jsonl", "sanitizer.jsonl"):
        rows = load(name)
        if not rows:
            continue
        out = RESULTS / name
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        n += len(rows)
    print(f"[iree/collect] {n} records -> {RESULTS}")


def cmd_stats():
    kernels = load("kernels.jsonl")
    expr = load("expressibility.jsonl")
    rem = load("remainder.jsonl")
    mutants = load("mutants.jsonl")

    s8 = {}
    for r in expr:
        key = r["class"]
        s8.setdefault(key, {"yes": 0, "partial": 0, "no": 0})
        s8[key][r["expressible"]] += 1

    # S1 detection matrix: only classes with measured mutants are counted;
    # classes without an expressible mutant (M1 elem / M3 hw on this surface)
    # are recorded as n/a per R4 (see S8 + specs/mutation-specs.md §4) and are
    # never counted as detected.
    s1 = {}
    for r in mutants:
        row = s1.setdefault(r["class"], {
            "n_injected": 0, "n_compile": 0, "n_runtime": 0,
            "n_never": 0, "n_na": 0})
        row["n_injected"] += 1
        row[f"n_{r['outcome']}"] += 1
    s1["M1"] = {"n_injected": 0, "n_compile": 0, "n_runtime": 0, "n_never": 0,
                "n_na": 0, "note": "no expressible element-access mutant on the "
                "iree linalg entry surface (§3.3); any spec M1 mutant => n/a (R4)"}
    s1["M3"] = {"n_injected": 0, "n_compile": 0, "n_runtime": 0, "n_never": 0,
                "n_na": 0, "note": "no expressible hw-constraint mutant on the "
                "iree linalg entry surface (§3.3); any spec M3 mutant => n/a (R4)"}

    s9 = {}
    for r in rem:
        s9[r["category"]] = {"unconditional_guards": r["unconditional_guards"],
                             "criterion_ref": r["criterion_ref"]}

    per_cat = defaultdict(Counter)
    for r in kernels:
        per_cat[r["category"]][r["compile"]] += 1
        per_cat[r["category"]][r.get("run", "-")] += 1

    stats = {
        "toolchain": "iree",
        "S1_detection": s1,
        "S8_expressibility": {cls: v for cls, v in sorted(s8.items())},
        "S9_remainder": {
            "per_category": {c: v["unconditional_guards"] for c, v in sorted(s9.items())},
            "total": sum(v["unconditional_guards"] for v in s9.values()),
            "criterion_ref": "entry dynamic dims (not statically foldable)",
        },
        "kernel_gate": {
            cat: {"compiled": c["ok"], "run_ok": c["ok"], "ref_pass": p}
            for cat, c in sorted(per_cat.items())
            for p in [sum(1 for r in kernels
                          if r["category"] == cat and r["ref_check"] == "pass")]
        },
        "totals": {
            "kernels": len(kernels),
            "compiled": sum(1 for r in kernels if r["compile"] == "ok"),
            "ran_ok": sum(1 for r in kernels if r.get("run") == "ok"),
            "ref_pass": sum(1 for r in kernels if r["ref_check"] == "pass"),
        },
        "note": ("Correctness only, sm_120 (dev build). Kernel gate = structural "
                 "ref-check. E1 S1: measured M2 entry-shape mutants; M1/M3 n/a "
                 "per §3.3 (iree linalg entry surface has no expressible "
                 "element-access/hw mutant). S12 (compute-sanitizer) pending."),
    }
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["totals"]))
    print(f"[iree/stats] S8 expressibility per class: "
          f"{ {k: v['yes']+v['partial'] for k, v in stats['S8_expressibility'].items()} }; "
          f"S9 total remainder (dynamic entry dims): "
          f"{stats['S9_remainder']['total']}")


if __name__ == "__main__":
    import sys
    {"collect": cmd_collect, "stats": cmd_stats}[sys.argv[1]]()
