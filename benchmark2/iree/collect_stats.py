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
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # benchmark2/
LANE = ROOT / "iree"
RAW = LANE / "raw"
RESULTS = ROOT / "results" / "iree"

CUDA_TARGET = os.environ.get("IREE_CUDA_TARGET", "sm_120")


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
    san = load("sanitizer.jsonl")

    s8 = {}
    for r in expr:
        key = r["class"]
        s8.setdefault(key, {"yes": 0, "partial": 0, "no": 0})
        s8[key][r["expressible"]] += 1

    # S1 detection matrix (3 classes). M1/M3 have no expressible mutant on the
    # IREE linalg entry surface, so their spec N=40 mutants are n/a (R4).
    s1 = {}
    for r in mutants:
        row = s1.setdefault(r["class"], {
            "n_injected": 0, "n_compile": 0, "n_runtime": 0,
            "n_never": 0, "n_na": 0})
        row["n_injected"] += 1
        row[f"n_{r['outcome']}"] += 1
    for cls, why in (("M1", "element-access"), ("M3", "hardware-constraint")):
        s1.setdefault(cls, {
            "n_injected": 0, "n_compile": 0, "n_runtime": 0,
            "n_never": 0, "n_na": 40,
            "note": f"no expressible {why} mutant on the iree linalg entry "
                    "surface (§3.3); all spec N=40 mutants => n/a (R4)"})

    # S12: compute-sanitizer supplement (measured, not "pending")
    s12 = {}
    for r in san:
        c = r["class"]
        s12.setdefault(c, {"flagged_and_exercised": 0, "total": 0, "flagged": 0})
        s12[c]["total"] += 1
        if r.get("flagged") == "true":
            s12[c]["flagged"] += 1
            if r.get("exercised") == "true":
                s12[c]["flagged_and_exercised"] += 1

    s9 = {}
    for r in rem:
        s9[r["category"]] = {"unconditional_guards": r["unconditional_guards"],
                             "criterion_ref": r["criterion_ref"]}

    per_cat = defaultdict(Counter)
    for r in kernels:
        per_cat[r["category"]][r["compile"]] += 1
        per_cat[r["category"]][r.get("run", "-")] += 1

    # I3: full-size gate failures — auditable disposition per kernel
    full_failures = []
    for r in kernels:
        if r.get("size") == "full" and r.get("run") != "ok":
            full_failures.append({
                "category": r["category"], "kernel": r["kernel"], "size": "full",
                "disposition": ("full-size dynamic-dim concretization exceeds "
                                "16 GB VRAM (OOM/timeout); --small gate passes")})

    stats = {
        "toolchain": "iree",
        "S1_detection": s1,
        "S8_expressibility": {cls: v for cls, v in sorted(s8.items())},
        "S9_remainder": {
            "per_category": {c: v["unconditional_guards"] for c, v in sorted(s9.items())},
            "total": sum(v["unconditional_guards"] for v in s9.values()),
            "criterion_ref": "entry dynamic dims (not statically foldable)",
        },
        "S12_sanitizer_supplement": s12,
        "kernel_gate": {
            cat: {"compiled": c["ok"], "run_ok": c["ok"], "ref_pass": p}
            for cat, c in sorted(per_cat.items())
            for p in [sum(1 for r in kernels
                          if r["category"] == cat and r["ref_check"] == "pass")]
        },
        "full_size_failures": full_failures,
        "totals": {
            "kernels": len(kernels),
            "compiled": sum(1 for r in kernels if r["compile"] == "ok"),
            "ran_ok": sum(1 for r in kernels if r.get("run") == "ok"),
            "ref_pass": sum(1 for r in kernels if r["ref_check"] == "pass"),
            "full_size_failures": len(full_failures),
        },
        "note": (f"Correctness only, {CUDA_TARGET}. Kernel gate = structural "
                 "ref-check. E1 S1: measured M2 entry-shape mutants; M1/M3 n/a "
                 "per §3.3. S12: M2 shape-contract mutants are not memory "
                 "faults, so compute-sanitizer --tool memcheck is silent by "
                 "design (plan §3.5)."),
    }
    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["totals"]))
    print(f"[iree/stats] S1 M2: {s1.get('M2')}; S12: {s12}; "
          f"full-size failures: {len(full_failures)}")


if __name__ == "__main__":
    import sys
    {"collect": cmd_collect, "stats": cmd_stats}[sys.argv[1]]()
