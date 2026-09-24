#!/usr/bin/env python3
"""gen_manifest.py — emit cutlass/raw/mutant_manifest.json from records.jsonl.

The manifest is the provenance record the family-depth guard reads
(`schema/check_class_axis.py::_lane_depth`): it recomputes each family's
instance count from `mutants`, so the rows here ARE the corpus the guard sees.
Every mutant the lane emits is `selected` (there is no lottery in this lane),
and the launch-status rows are `attribution_only` (they leave the class cell by
design, like `choreo`'s M3-L rows). The accounting block closes by construction:
`candidates = selected + attribution`.

Usage:
    gen_manifest.py [--records records.jsonl] [--out raw/mutant_manifest.json]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH2 = os.path.dirname(HERE)
if BENCH2 not in sys.path:
    sys.path.insert(0, BENCH2)
from schema import class_axis as AX                                # noqa: E402
from schema import method_taxonomy as T                            # noqa: E402

LANE = "cutlass"
CLASSES = AX.mutation_classes()


def build(records_path):
    recs = [json.loads(l) for l in open(records_path) if l.strip()]
    mutants, attribution = [], []
    for r in recs:
        spec_id = r.get("spec_id", "")
        family = T.family_of(spec_id)
        shape = (r.get("mutation") or {}).get("grid", "small")
        row = {"mutant_id": r["mutant_id"], "spec_id": spec_id,
               "family": family, "category": r.get("category", ""),
               "shape": shape}
        if r.get("path_class") == "L":
            row["attribution_only"] = True
            attribution.append(row)
        else:
            mutants.append(row)

    depth: dict[str, int] = {}
    for row in mutants:
        f = row.get("family")
        if f:
            depth[f] = depth.get(f, 0) + 1
    families = [f for cls in CLASSES for f in T.families_of(cls)
                if not T.is_absent(f)]
    short = {f: T.N_PER_FAMILY - depth.get(f, 0)
             for f in families if depth.get(f, 0) != T.N_PER_FAMILY}

    selected, attrib = len(mutants), len(attribution)
    accounting = {"candidates": selected + attrib, "selected": selected,
                  "attribution": attrib, "dropped": 0, "na": 0, "skipped": 0,
                  "emitted": selected + attrib}
    return {
        "toolchain": LANE,
        "spec_version": "v2.1",
        "lane": LANE,
        "n_per_family": T.N_PER_FAMILY,
        "n_realisations": T.N_REALISATIONS,
        "n_kernels": T.N_KERNELS,
        "n_cells": 1,
        "target_instances": T.target(LANE),
        "level2": True,
        "accounting": accounting,
        "family_depth": depth,
        "family_short": short,
        "unrealised_specs": {cls: [] for cls in CLASSES},
        "registry": {},
        "mutants": mutants,
        "dropped": [],
        "na": [],
        "attribution": attribution,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(HERE, "records.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "raw",
                                                  "mutant_manifest.json"))
    a = ap.parse_args()
    man = build(a.records)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    acc = man["accounting"]
    print(f"[manifest] {a.out}: mutants={len(man['mutants'])} "
          f"attribution={acc['attribution']} short={man['family_short']}")


if __name__ == "__main__":
    main()
