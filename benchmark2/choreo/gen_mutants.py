#!/usr/bin/env python3
"""E1 mutant generator for the choreo lane.

Reads the base kernels from benchmark/choreo/<category>/, applies the mutation
operators in mutations.py (specs/mutation-specs-v2.md §1-§4, §9.6), and writes:

  benchmark2/choreo/mutants/<class>/<category>/<mutant_id>__<case>.co
  benchmark2/choreo/raw/mutant_manifest.json

The manifest is the provenance record for every mutant: class, spec number,
v2.1 spec id, check-path class, prohibition, admissibility, level, base path,
mutant path, kernel_hash, mutant_hash, settings_hash, the human-readable defect
description, and the unified diff. `collect` turns it into `mutant` records per
schema/record-schema.json.

GATE 2 -- path-class-aware selection (specs §9.6.1, specs/expansion-workflow.md
§3/§4). The old flat "40 per class" rule is wrong under v2.1 because budget
follows the PATH, not the importance of the feature:

  P1  assessed    full budget (N_P1_PER_CLASS per class), THEN re-run the whole
                  class at each -rtc level so the cost-threshold curve exists
  P3  unchecked   exactly ONE operator per (spec x category) cell -- no
                  threshold reaches this spec, so more instances buy nothing
  P4  warning     same as P3
  L   launch      one per (spec x category) cell, attribution-only
  P2  hard-error  NEVER generated; recorded as N/A (specs §9.5.0)

Selection is therefore: take every non-P1 cell first (mandatory), then
round-robin P1 candidates over the categories in the minimal coverage set,
taking candidates in declaration order within each. Fully deterministic: the
same mutations.py always yields the same corpus.

GATE 2 also cross-checks mutations.SPEC_REGISTRY against the operators that
actually loaded and REFUSES to run if an operator claims an unregistered spec.
A spec that is `pending` is printed, not hidden.

Usage:
  gen_mutants.py [--level2] [--rtc LEVEL] [--dry-run] [--report]
                 [--n-p1 N] [--out DIR] [--manifest FILE]
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
SPEC_REGISTRY_OUT = os.path.join(HERE, "raw", "spec_registry.json")

# ---- budget (specs §9.6.1) -------------------------------------------------
N_P1_PER_CLASS = 40       # P1 assessed: full budget, so the -rtc curve exists
N_CELLS = 1               # P3 / P4 / L: one injection per (spec x category)
N_PER_CLASS = N_P1_PER_CLASS   # retained: the pre-v2.1 name in the manifest

# ---- check paths that are MUTATABLE ---------------------------------------
MUTATABLE_PATHS = ("P1", "P3", "P4", "L")

# ---- the -rtc levels the curve is swept over (specs §9.6.1) ---------------
RTC_LEVELS = ("entry", "low", "medium", "high")

TOOLCHAIN = "choreo"
SPEC_VERSION = "v2.1"


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


def select(cands, n_p1=N_P1_PER_CLASS, n_cells=N_CELLS):
    """Path-class-aware, deterministic selection (specs §9.6.1).

    Stage 1 -- cell floor. One instance per (spec_id, category) cell for EVERY
    generatable spec. This is the v2.1 non-negotiable: a declared spec that
    produces no instance is indistinguishable from a spec that was forgotten,
    so the corpus must cover each cell at least once. For P3/P4/L the cell floor
    IS the whole budget (no threshold reaches those paths, so more instances buy
    nothing); for P1 it is a floor, not a ceiling.

    Stage 2 -- P1 budget. Round-robin over categories in declaration order until
    the class has n_p1 P1 instances. This is the budget the cost-threshold curve
    is swept with, so it is deliberately generous.

    P2 candidates are never chosen: the compiler repairs the state, so there is
    no test (specs §9.5.0).

    Returns (selected, dropped); both preserve declaration order.
    """
    # ---- stage 1: cell floor --------------------------------------------
    cells = collections.OrderedDict()
    for c in cands:
        if c.is_na:                      # P2 -- repaired, never generated
            continue
        cells.setdefault((c.spec_id, c.category), []).append(c)

    mandatory = set()
    for group in cells.values():
        for c in group[:n_cells]:
            mandatory.add(c.id)

    # ---- stage 2: P1 budget ---------------------------------------------
    p1 = [c for c in cands if c.needs_rtc_curve and not c.is_na]
    by_cat = collections.OrderedDict()
    for c in p1:
        by_cat.setdefault(c.category, []).append(c)
    queues = {k: list(v) for k, v in by_cat.items()}
    taken_p1 = len([c for c in p1 if c.id in mandatory])
    # NOTE: a pass over the queues may legitimately consume only candidates that
    # are already in the cell floor (they are the first entry of their cell, so
    # they sit at the head of their category queue). That is *consumption*, not
    # a stall, so the termination test must be "did any queue yield an item in
    # this pass", not "did the P1 count increase". Using the latter silently
    # truncated every class whose cell heads are P1 (M3 chose 10 of 47).
    while taken_p1 < n_p1:
        consumed = 0
        for cat in queues:
            if taken_p1 >= n_p1:
                break
            if not queues[cat]:
                continue
            m = queues[cat].pop(0)
            consumed += 1
            if m.id not in mandatory:
                mandatory.add(m.id)
                taken_p1 += 1
        if consumed == 0:            # every category queue is drained
            break

    chosen = [c for c in cands if c.id in mandatory]
    dropped = [c for c in cands if c.id not in mandatory]
    return chosen, dropped


def compose(level2=False):
    """Build the candidate list per class, then select.

    Level 2 is now real: `mutations.categories_for(cls, level2=True)` unions the
    level-2 category order into the class's coverage set, so any operator that
    declares a level-2 category is picked up. No such operator exists yet, so
    the widening is reported rather than silently a no-op.
    """
    plan = {}
    for cls in sorted(M.ALL):
        cands = list(M.transforms_for(cls))
        covered = set(M.categories_for(cls, level2=level2))
        missing = sorted({c.category for c in cands} - covered)
        if missing:
            # A registered operator on an uncovered category would be dropped
            # by selection without saying so -- refuse instead.
            raise ValueError(
                "class %s declares operators on categories outside the "
                "level-%d coverage set: %s (add them to MINIMAL_SET/LEVEL2_SET "
                "in mutations.py or drop the operator)"
                % (cls, 2 if level2 else 1, ", ".join(missing)))
        gen = [c for c in cands if not c.is_na]
        na = [c for c in cands if c.is_na]
        chosen, dropped = select(gen)
        plan[cls] = {"chosen": chosen, "dropped": dropped, "total": len(cands),
                     "na": na, "level2": level2}
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
    # A stale mutant from an earlier run is indistinguishable from a current one
    # once it is on disk, and `mutants/` is what E1 consumes -- an obsolete file
    # that still compiles would be injected and counted. Clear the tree so the
    # corpus is EXACTLY `chosen`, which is what makes the census meaningful.
    if not dry_run and os.path.isdir(out_dir):
        # Guard against a caller pointing --out at something else of theirs.
        if not os.path.abspath(out_dir).startswith(HERE + os.sep):
            raise ValueError(
                "refusing to clear --out %r: it is outside %r"
                % (out_dir, HERE))
        shutil.rmtree(out_dir)
    records = []
    skipped = []
    for cls in sorted(plan):
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
            # The level is a COVERAGE fact: level 1 is the minimal set, level 2
            # the widened set (specs §5). It is derived here, once, because
            # `as_meta()` also carries a `level` and `rec.update()` below would
            # otherwise clobber this with the constructor default.
            lvl = 1 if mut.category in M.MINIMAL_SET.get(cls, []) else 2
            rec = {
                "toolchain": TOOLCHAIN,
                "class": cls,
                "spec": mut.spec,
                "paper_category": mut.paper_category,
                "mutant_id": mut.id,
                "category": mut.category,
                "case": mut.case,
                "level": lvl,
                "base_path": os.path.relpath(bp, REPO),
                "mutant_path": os.path.relpath(dst, REPO),
                "kernel_hash": sha1_12(bp),
                "mutant_hash": hashlib.sha1(new.encode()).hexdigest()[:12],
                "settings_hash": settings_hash(mut.category),
                "local_headers": [os.path.basename(h)
                                  for h in local_headers(mut.category)],
                "desc": mut.desc,
                "diff": diff,
                "spec_version": SPEC_VERSION,
            }
            # ---- v2.1 attribution (specs §9.6, §9.5.0) -------------------
            # `as_meta()` supplies spec/spec_id/path_class/prohibition/
            # admissible/spec_admissible. It does NOT supply `applicable`: that
            # is collect.py's serialization of `admissible`, and `is_na` is a
            # DIFFERENT question (P2 only), so deriving `applicable` from
            # `is_na` here would stamp L1/L2/L4 and the noop controls as
            # applicable and quietly corrupt the denominator.
            rec.update(mut.as_meta())
            rec["level"] = lvl            # re-assert: as_meta carries a level
            rec["needs_rtc_curve"] = mut.needs_rtc_curve
            records.append(rec)
    return records, skipped


def report(plan, records, skipped, rtc=None):
    print(f"selected {len(records)} mutants  (skipped {len(skipped)})"
          + (f"  @ -rtc={rtc}" if rtc else ""))
    print(f"{'class':<6}{'candidates':>11}{'P1':>8}{'mandatory':>11}"
          f"{'N/A':>6}{'dropped':>9}{'budget':>8}")
    for cls in sorted(plan):
        p = plan[cls]
        n_p1 = len([c for c in p["chosen"] if c.needs_rtc_curve])
        n_cells = len(p["chosen"]) - n_p1
        print(f"{cls:<6}{p['total']:>11}{n_p1:>8}{n_cells:>11}"
              f"{len(p['na']):>6}{len(p['dropped']):>9}{N_P1_PER_CLASS:>8}")

    print("\nper class x category:")
    by = collections.Counter((r["class"], r["category"]) for r in records)
    for (cls, cat), n in sorted(by.items()):
        print(f"  {cls}/{cat:<22}{n:>3}")

    print("\nper class x spec_id  (v2.1 id, path class):")
    bys = collections.Counter((r["class"], r["spec_id"], r["path_class"])
                              for r in records)
    for (cls, sid, path), n in sorted(bys.items()):
        print(f"  {cls}/{sid:<8}{path:<4}{n:>3}")

    print("\nper class x paper_category:")
    bpc = collections.Counter((r["class"], r["paper_category"]) for r in records)
    for (cls, pc), n in sorted(bpc.items()):
        print(f"  {cls}/{pc:<16}{n:>3}")

    print("\nper path class (the point of v2.1):")
    bpath = collections.Counter(r["path_class"] for r in records)
    for path, n in sorted(bpath.items()):
        print(f"  {path:<4}{n:>4}")

    # ---- GATE 2: registry vs operators ------------------------------------
    print("\nGATE 2 -- registry cross-check (mutations.SPEC_REGISTRY):")
    s = M.registry_summary()
    print(f"  specs {s['n_specs']}  operators {sum(s['n_operators'].values())}"
          f"  status {s['status_counts']}  paths {s['path_counts']}")
    print(f"  admissible {s['admissible_counts'].get(True, 0)}"
          f"  inadmissible {s['admissible_counts'].get(False, 0)}"
          f"  by reason {s['prohibition_counts']}")
    if s["pending"]:
        print(f"  PENDING (admissible, operator not written) -- "
              f"{len(s['pending'])}:")
        for sid in s["pending"]:
            v = M.SPEC_REGISTRY[sid]
            print(f"    {sid:<8}{v['path']:<4}{v['desc'][:58]}")
    else:
        print("  PENDING: none -- every admissible spec has an operator")
    if s["na"]:
        print(f"  N/A (recorded, never generated) -- {len(s['na'])}:")
        for sid in s["na"]:
            v = M.SPEC_REGISTRY[sid]
            print(f"    {sid:<8}{v['path']:<4}"
                  f"{(v['prohibition'] or 'noop'):<13}{v['desc'][:46]}")

    if skipped:
        print(f"\nskipped ({len(skipped)}):")
        cnt = collections.Counter(w.split(":")[0] for _, w in skipped)
        for w, n in cnt.most_common():
            print(f"  {n:>4}  {w}")
        for mid, w in skipped[:40]:
            print(f"    {mid}: {w}")


def write_spec_registry(path):
    """Write one record per SPEC_REGISTRY entry to `raw/spec_registry.json`.

    This is the ONLY artifact that can carry an N/A verdict, because every other
    artifact is keyed on a mutant and an N/A verdict has no mutant. Without it
    the 44-applicable / 28-N-A denominator would be unauditable from committed
    data. It is written even on --dry-run: it records a DESIGN decision, not a
    mutant, so `dry-run` ("write no mutants") still holds.
    """
    specs = []
    for sid, v in sorted(M.SPEC_REGISTRY.items()):
        specs.append({
            "spec_id": sid,
            "cls": v["cls"],
            "path": v["path"],
            "admissible": bool(v["admissible"]),
            "prohibition": v["prohibition"],
            "status": v["status"],
            "desc": v["desc"],
            "note": v["note"],
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"toolchain": TOOLCHAIN,
                   "spec_version": SPEC_VERSION,
                   "summary": M.registry_summary(),
                   "specs": specs}, f, indent=1)
    return len(specs)


def main():
    global N_P1_PER_CLASS, N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--level2", action="store_true",
                    help="also widen to the level-2 category set (specs §5)")
    ap.add_argument("--rtc", choices=RTC_LEVELS, default=None,
                    help="stamp the -rtc level this corpus will be run at "
                         "(specs §9.6.1: the curve is a RUN parameter; P1 "
                         "classes are re-run at every level)")
    ap.add_argument("--n-p1", type=int, default=N_P1_PER_CLASS,
                    help="P1 budget per class (default %d)" % N_P1_PER_CLASS)
    ap.add_argument("--n-cells", type=int, default=N_CELLS,
                    help="injections per (spec x category) cell for P3/P4/L "
                         "(default %d)" % N_CELLS)
    ap.add_argument("--dry-run", action="store_true",
                    help="classify and report without writing any mutant")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--fail-on-pending", action="store_true",
                    help="exit 2 if an ADMISSIBLE spec has no operator. This "
                         "is the GATE 2 gate `make choreo-screen` runs: an "
                         "inadmissible spec is a verdict and never fails, but "
                         "an admissible spec without an operator is open work "
                         "and must not pass silently")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()

    N_P1_PER_CLASS, N_CELLS = a.n_p1, a.n_cells
    if a.level2:
        for cls, cats in M.LEVEL2_SET.items():
            for c in cats:
                if c not in M.MINIMAL_SET.get(cls, []):
                    M.MINIMAL_SET.setdefault(cls, []).append(c)

    plan = compose(a.level2)
    records, skipped = emit(plan, a.out, dry_run=a.dry_run)
    summary = M.registry_summary()
    n_specs = write_spec_registry(SPEC_REGISTRY_OUT)
    report(plan, records, skipped, rtc=a.rtc)
    print(f"spec registry -> {os.path.relpath(SPEC_REGISTRY_OUT, REPO)}"
          f"  ({n_specs} specs)")

    rc = 0
    if skipped:
        rc = 1
    if a.fail_on_pending and summary["pending"]:
        print("\nGATE 2 FAILED: %d admissible spec(s) have no operator: %s"
              % (len(summary["pending"]), ", ".join(summary["pending"])),
              file=sys.stderr)
        rc = 2
    if a.dry_run:
        return rc

    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump({"toolchain": TOOLCHAIN,
                   "spec_version": SPEC_VERSION,
                   "n_per_class": N_PER_CLASS,
                   "n_p1_per_class": N_P1_PER_CLASS,
                   "n_cells": N_CELLS,
                   "level2": a.level2,
                   "rtc": a.rtc,
                   "registry": summary,
                   "mutants": records}, f, indent=1)
    print(f"\nwrote {len(records)} mutants -> {os.path.relpath(a.out, REPO)}")
    print(f"manifest -> {os.path.relpath(a.manifest, REPO)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
