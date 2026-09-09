#!/usr/bin/env python3
"""E1 mutant generator for the choreo lane.

Reads the base kernels from benchmark/choreo/<category>/, applies the mutation
operators in mutations.py (specs/mutation-specs.md §1-§3), and writes:

  benchmark2/choreo/mutants/<class>/<category>/<mutant_id>__<case>.co
  benchmark2/choreo/raw/mutant_manifest.json

The manifest is the provenance record for every mutant: class, spec number,
paper category, level, base path, mutant path, kernel_hash, mutant_hash,
settings_hash, the human-readable defect description, and the unified diff.
`collect` turns it into `mutant` records per schema/record-schema.json.

Selection rule (specs §0 requires N = 40 per class):
  mutations.py declares more candidates than 40 for M1 and M3 because each spec
  is instantiated at several defect magnitudes. To land exactly 40 per class we
  round-robin over the categories in the minimal coverage set, taking candidates
  in declaration order within each category. This keeps the per-category balance
  that specs §5 requires (every class exercised by >=1 operator) and is fully
  deterministic: the same mutations.py always yields the same 40.

Usage:
  gen_mutants.py [--level2] [--dry-run] [--report] [--out DIR]
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutations as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/
REPO = os.path.dirname(B2)                                 # svn-artifacts/
SUITE = os.path.join(REPO, "benchmark", "choreo")
OUT = os.path.join(HERE, "mutants")
MANIFEST = os.path.join(HERE, "raw", "mutant_manifest.json")

N_PER_CLASS = 40          # specs §0
TOOLCHAIN = "choreo"


def sha1_12(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def settings_hash(category):
    """sha1 of benchmark2/settings/<category>.md — the cross-cutting field the
    statistics manifest calls non-negotiable."""
    p = os.path.join(B2, "settings", f"{category}.md")
    return sha1_12(p) if os.path.exists(p) else None


def base_path(category, case):
    return os.path.join(SUITE, category, case + ".co")


def apply_transform(src, mut):
    """Apply every edit in order. Returns (new_src, None) or (None, reason)."""
    out = src
    for old, new, n in mut.edits:
        c = out.count(old)
        if c == 0:
            return None, f"pattern absent: {old[:70]!r}"
        if n is not None and c != n:
            return None, f"pattern matched {c}x, expected {n}x: {old[:70]!r}"
        out = out.replace(old, new)
    if out == src:
        return None, "noop"
    return out, None


def select(cands, n=N_PER_CLASS):
    """Deterministic round-robin over categories, declaration order within each.

    Returns (selected, dropped); both preserve declaration order.
    """
    if len(cands) <= n:
        return list(cands), []
    by_cat = collections.OrderedDict()
    for c in cands:
        by_cat.setdefault(c.category, []).append(c)
    queues = {k: list(v) for k, v in by_cat.items()}
    chosen, seen = [], set()
    while len(chosen) < n:
        progressed = False
        for cat in queues:
            if queues[cat] and len(chosen) < n:
                m = queues[cat].pop(0)
                chosen.append(m)
                seen.add(m.id)
                progressed = True
        if not progressed:
            break
    dropped = [c for c in cands if c.id not in seen]
    return chosen, dropped


def compose(level2=False):
    """Build the candidate list, then select N_PER_CLASS per class."""
    plan = {}
    for cls in ("M1", "M2", "M3"):
        cands = list(M.transforms_for(cls))
        if level2:
            # Level-2 categories are not enumerated in mutations.py yet; they are
            # added only after level-1 is green (specs §5).
            pass
        chosen, dropped = select(cands)
        plan[cls] = {"chosen": chosen, "dropped": dropped, "total": len(cands)}
    return plan


LOCAL_HDR_CACHE = {}


def local_headers(category):
    """Non-.co files a category's kernels #include with a bare relative path.

    choreo emits a .cu that does `#include "common.hpp"` and passes
    `-I<directory of the .co file>`, so the header must sit next to the mutant,
    not next to the base kernel. Categories observed to need this: conv2d and
    transpose (common.h), matmul and softmax (common.hpp).
    """
    if category in LOCAL_HDR_CACHE:
        return LOCAL_HDR_CACHE[category]
    d = os.path.join(SUITE, category)
    hdrs = []
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and not name.endswith(".co"):
                hdrs.append(p)
    LOCAL_HDR_CACHE[category] = hdrs
    return hdrs


def emit(plan, out_dir, dry_run=False):
    records = []
    skipped = []
    for cls in ("M1", "M2", "M3"):
        for mut in plan[cls]["chosen"]:
            bp = base_path(mut.category, mut.case)
            if not os.path.exists(bp):
                skipped.append((mut.id, f"base kernel missing: {bp}"))
                continue
            src = open(bp).read()
            new, why = apply_transform(src, mut)
            if new is None:
                skipped.append((mut.id, why))
                continue
            rel = os.path.join(cls, mut.category, f"{mut.id}__{mut.case}.co")
            dst = os.path.join(out_dir, rel)
            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    f.write(new)
                for h in local_headers(mut.category):
                    shutil.copy2(h, os.path.join(os.path.dirname(dst),
                                                 os.path.basename(h)))
            diff = "".join(difflib.unified_diff(
                src.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{mut.category}/{mut.case}.co",
                tofile=f"b/{mut.category}/{mut.case}.co", n=2))
            records.append({
                "toolchain": TOOLCHAIN,
                "class": cls,
                "spec": mut.spec,
                "paper_category": mut.paper_category,
                "mutant_id": mut.id,
                "category": mut.category,
                "case": mut.case,
                "level": 1 if mut.category in M.MINIMAL_SET[cls] else 2,
                "base_path": os.path.relpath(bp, REPO),
                "mutant_path": os.path.relpath(dst, REPO),
                "kernel_hash": sha1_12(bp),
                "mutant_hash": hashlib.sha1(new.encode()).hexdigest()[:12],
                "settings_hash": settings_hash(mut.category),
                "local_headers": [os.path.basename(h)
                                  for h in local_headers(mut.category)],
                "desc": mut.desc,
                "diff": diff,
            })
    return records, skipped


def report(plan, records, skipped):
    print(f"selected {len(records)} mutants  (skipped {len(skipped)})")
    print(f"{'class':<6}{'candidates':>11}{'selected':>10}{'dropped':>9}{'target':>8}")
    for cls in ("M1", "M2", "M3"):
        p = plan[cls]
        print(f"{cls:<6}{p['total']:>11}{len(p['chosen']):>10}"
              f"{len(p['dropped']):>9}{N_PER_CLASS:>8}")
    print("\nper class x category:")
    by = collections.Counter((r["class"], r["category"]) for r in records)
    for (cls, cat), n in sorted(by.items()):
        print(f"  {cls}/{cat:<22}{n:>3}")
    print("\nper class x spec:")
    bys = collections.Counter((r["class"], r["spec"]) for r in records)
    for (cls, sp), n in sorted(bys.items()):
        print(f"  {cls}/spec{sp:<3}{n:>3}")
    print("\nper class x paper_category:")
    bpc = collections.Counter((r["class"], r["paper_category"]) for r in records)
    for (cls, pc), n in sorted(bpc.items()):
        print(f"  {cls}/{pc:<16}{n:>3}")
    if skipped:
        print(f"\nskipped ({len(skipped)}):")
        cnt = collections.Counter(w.split(":")[0] for _, w in skipped)
        for w, n in cnt.most_common():
            print(f"  {n:>4}  {w}")
        for mid, w in skipped[:40]:
            print(f"    {mid}: {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level2", action="store_true",
                    help="also widen to the level-2 operator set (specs §5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="classify and report without writing any mutant")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()

    plan = compose(a.level2)
    records, skipped = emit(plan, a.out, dry_run=a.dry_run)
    report(plan, records, skipped)

    if a.dry_run:
        return 0 if not skipped else 1

    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump({"toolchain": TOOLCHAIN, "n_per_class": N_PER_CLASS,
                   "mutants": records}, f, indent=1)
    print(f"\nwrote {len(records)} mutants -> {os.path.relpath(a.out, REPO)}")
    print(f"manifest -> {os.path.relpath(a.manifest, REPO)}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
