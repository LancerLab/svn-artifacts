"""collect.py — parse benchmark2/tilelang/raw/*.jsonl into results/ + stats.json
per benchmark2/schema/. Idempotent; mirrors the triton lane collector."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RES = HERE / "results"
B2 = HERE.parent


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def validate(rec: dict, kind: str) -> list[str]:
    schema = json.loads((B2 / "schema" / "record-schema.json").read_text())
    spec = schema["records"].get(kind)
    if not spec:
        return [f"unknown record kind {kind}"]
    errs = []
    for f in spec["fields"]:
        if f not in rec:
            errs.append(f"missing field {f}")
    for f, allowed in spec.get("enums", {}).items():
        if f in rec and str(rec[f]) not in allowed:
            errs.append(f"{f}={rec[f]} not in {allowed}")
    return errs


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
    for e in dict.fromkeys(all_errs):
        print("SCHEMA ERROR:", e)
    if all_errs:
        sys.exit(1)

    seen = set()
    expr_dedup = [r for r in expr
                  if not ((r.get("category"), r.get("class")) in seen
                          or seen.add((r.get("category"), r.get("class"))))]

    if not stats_only:
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "mutants.json").write_text(json.dumps(mutants, indent=1))
        (RES / "kernels.json").write_text(json.dumps(kernels, indent=1))
        (RES / "expressibility.json").write_text(json.dumps(expr_dedup, indent=1))
        (RES / "sanitizer.json").write_text(json.dumps(san, indent=1))
        print(f"collected {len(mutants)} mutants, {len(kernels)} kernels")

    s12 = {}
    for r in san:
        c = r["class"]
        s12.setdefault(c, {"flagged_and_exercised": 0, "total": 0, "flagged": 0})
        s12[c]["total"] += 1
        if r.get("flagged") == "true":
            s12[c]["flagged"] += 1
            if r.get("exercised") == "true":
                s12[c]["flagged_and_exercised"] += 1

    by_class = {}
    for r in mutants:
        c = r["class"]
        by_class.setdefault(c, Counter())
        by_class[c][r["outcome"]] += 1
    s1 = {c: {"n_injected": sum(v.values()), "n_compile": v.get("compile", 0),
              "n_runtime": v.get("runtime", 0), "n_never": v.get("never", 0),
              "n_na": v.get("n/a", 0)} for c, v in by_class.items()}
    s1.setdefault("M2", {"n_injected": 0, "n_compile": 0, "n_runtime": 0,
                         "n_never": 0, "n_na": 40,
                         "note": "no cross-tensor contract (§3.3)"})

    s8 = {}
    for r in expr_dedup:
        c = r["class"]
        s8.setdefault(c, {"yes": 0, "partial": 0, "no": 0})
        s8[c][r["expressible"]] += 1

    stats = {"toolchain": "tilelang",
             "S1_detection_matrix": s1,
             "S8_expressibility": {c: v for c, v in sorted(s8.items())},
             "S9_remainder": "n/a (no generated checks)",
             "S12_sanitizer_supplement": s12}
    (RES / "stats.json").parent.mkdir(parents=True, exist_ok=True)
    (RES / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
