#!/usr/bin/env python3
"""verify.py — self-contained verification of the CUTLASS/CuTe lane corpus.

Runs WITHOUT a GPU (it reads the committed `records*.jsonl` and
`results/cutlass/stats.json`). It answers four questions that a reader of the
stats file otherwise has to take on faith:

  1. Is every committed record a valid v2.1 mutant record?
     (`schema.records.validate`, the one reader of `record-schema.json`.)
  2. Are the records unique, one injection per `mutant_id`?
  3. Does `stats.json` S1 block re-derive exactly from the records?
     (no hand-edit can survive: the aggregation is recomputed here.)
  4. Does the corpus agree with the class axis -- `S1_detection` holds mutation
     classes only (no `L`), the `L` path class is reported under
     `S1_path_class`, and the uncompared declaration matches the axis?

Exit status is non-zero on any failure, so `run.sh verify` is CI-shaped.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH2 = os.path.dirname(HERE)
if BENCH2 not in sys.path:
    sys.path.insert(0, BENCH2)
from schema import records as REC                                # noqa: E402
from schema import class_axis as AX                              # noqa: E402

LANE = "cutlass"
# M1 and M3 are LAUNCHED batteries in records.jsonl, so the old compile-only
# `records_m1.jsonl`/`records_m3.jsonl` probe slices are retired (kept on disk
# for provenance only).
RECORDS = [os.path.join(HERE, "records.jsonl")]
STATS = os.path.join(BENCH2, "results", LANE, "stats.json")


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def aggregate(recs):
    """Recompute the S1 blocks from the records; the one definition."""
    def _cell():
        return dict(n_injected=0, n_compile=0, n_runtime=0, n_never=0, n_na=0)
    det = {}
    pathcls = {}
    for r in recs:
        if r.get("path_class") == "L":
            bucket = pathcls.setdefault("L", _cell())
        else:
            bucket = det.setdefault(r["class"], _cell())
        bucket["n_injected"] += 1
        o = r["outcome"]
        if o == "compile":
            bucket["n_compile"] += 1
        elif o == "runtime":
            bucket["n_runtime"] += 1
        elif o == "never":
            bucket["n_never"] += 1
        elif o == "n/a":
            bucket["n_na"] += 1
    return det, pathcls


def main():
    fails = []
    recs = []
    for p in RECORDS:
        if not os.path.exists(p):
            fails.append(f"missing corpus file {p}")
            continue
        recs += load(p)

    # 1. schema conformance
    for r in recs:
        errs = REC.validate(r, "mutant")
        if errs:
            fails.append(f"{r.get('mutant_id')}: {errs}")
    # 2. uniqueness
    ids = [r.get("mutant_id") for r in recs]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        fails.append(f"duplicate mutant_id(s): {dupes}")

    # 3. stats re-derivation
    if not os.path.exists(STATS):
        fails.append(f"missing {STATS}")
        stats = {}
    else:
        stats = json.load(open(STATS))
        det, pathcls = aggregate(recs)
        for got, key in ((det, "S1_detection"), (pathcls, "S1_path_class")):
            if stats.get(key) != got:
                fails.append(f"{key} does not re-derive from the records:\n"
                             f"    stats = {json.dumps(stats.get(key), sort_keys=True)}\n"
                             f"    corpus= {json.dumps(got, sort_keys=True)}")

    # 4. axis agreement
    classes = set(AX.mutation_classes())
    s1 = stats.get("S1_detection", {})
    extra = sorted(set(s1) - classes)
    if extra:
        fails.append(f"S1_detection carries non-mutation keys {extra}")
    want_uncmp = sorted(AX.uncompared_classes(LANE))
    got_uncmp = sorted(stats.get("S1_declared_uncompared", {}).get("classes", []))
    if got_uncmp != want_uncmp:
        fails.append(f"S1_declared_uncompared.classes = {got_uncmp}, "
                     f"axis says {want_uncmp}")
    bad_cls = sorted({r["class"] for r in recs} - classes)
    if bad_cls:
        fails.append(f"records carry classes outside {sorted(classes)}: {bad_cls}")
    if "L" in s1:
        fails.append("S1_detection carries the `L` path class as a mutation class")

    # report
    print(f"[verify] {len(recs)} records; "
          f"{len({r['spec_id'] for r in recs})} distinct specs; "
          f"mutant_id unique={not dupes}")
    if fails:
        print(f"[verify] FAIL ({len(fails)})")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("[verify] OK: schema-valid v2.1 records, unique, S1 re-derives, "
          "axis-consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
