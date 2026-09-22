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

GATE 2 -- outcome-aware selection (specs §9.6.1, specs/expansion-workflow.md
§3/§4). The old flat "40 per class" rule is wrong under v2.1 because budget
follows the OUTCOME, not the importance of the feature:

  rt-check   up to N_PER_FAMILY per method family, THEN re-run the whole class
             at each -rtc level so the cost-threshold curve exists
  unchecked  exactly ONE operator per (spec x category) cell -- no threshold
             reaches this spec, so more instances buy nothing
  ct-check   same as unchecked
  L          one per (spec x category) cell, attribution-only
  avoided    NEVER generated; recorded as N/A (specs §9.5.0)

The budget is per **(class, method family)** -- that is the unit of requirement
(b) -- and within a family it is `N_kernels kernels x N_realisations
realisations`, so each category is capped at N_REALISATIONS. A per-CLASS budget
cannot express this: it lets a family with seven declarations absorb the budget
while a family with one gets a single instance, which is precisely how the
corpus passed "40 per class" while 19 of 31 families sat at `R_f < 2`.

Selection is therefore: take every non-rt-check cell first (mandatory), then
round-robin rt-check candidates over the family's categories, taking candidates in
declaration order within each, until the family holds N_PER_FAMILY. Fully
deterministic: the same mutations.py always yields the same corpus.

**Every candidate that is not chosen is persisted with a reason.** 59
declarations used to vanish silently, so "the corpus has 170 mutants" was not
auditable from the corpus itself: the arithmetic `selected + dropped + n/a ==
candidates` did not hold anywhere. Reasons come from a closed vocabulary (see
`select()`).

GATE 2 also cross-checks mutations.SPEC_REGISTRY against the operators that
actually loaded and REFUSES to run if an operator claims an unregistered spec.
A spec that is `pending` is printed, not hidden.

Usage:
  gen_mutants.py [--level2] [--rtc LEVEL] [--dry-run] [--report]
                 [--n-per-family N] [--n-realisations N] [--out DIR]
                 [--manifest FILE]
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
ORACLE_POLICY = os.path.join(HERE, "raw", "oracle_policy.json")

# The per-family budget lives in ONE place: schema/method-taxonomy.json. It is
# imported the same way choreo/mutations.py imports schema.class_axis, so both
# levels of the axis are read from their owner rather than restated here.
if B2 not in sys.path:
    sys.path.insert(0, B2)
from schema import method_taxonomy as T  # noqa: E402

# ---- budget ---------------------------------------------------------------
# N = N_KERNELS x N_REALISATIONS, per (class, method family), per in-scope
# lane. The factor that matters for selection is N_REALISATIONS: without a
# per-category cap, "8 instances for M2-a" would be satisfiable by 8 edits to
# one kernel, and the "4 kernels x 2 realisations" decomposition would be a
# claim about the prose rather than about the corpus.
N_PER_FAMILY = T.N_PER_FAMILY          # 8 = 4 kernels x 2 realisations
N_REALISATIONS = T.N_REALISATIONS      # 2 -- the cap per kernel (category)
N_CELLS = 1                            # unchecked/ct-check/L: one instance per (spec, cat)
LANE = "choreo"

# ---- check paths that are MUTATABLE ---------------------------------------
MUTATABLE_PATHS = ("rt-check", "unchecked", "ct-check", "L")

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


def oracle_usable_cases():
    """The `category/case` keys the calibrated oracle can decide (specs §7).

    E1 gives a mutant on a case whose UNMUTATED base already fails its own
    reference check the verdict `noop` no matter what the edit does: the base
    cannot pass, so the manifest field carries no information. Spending a
    family's budget on such a case therefore buys a corpus entry that can never
    count as a validated test, while a usable case with the same kernel could
    have been validated. Selection prefers usable cases for that reason; when
    no policy has been calibrated yet the preference is empty and the order is
    declaration order, exactly as before.
    """
    if not os.path.exists(ORACLE_POLICY):
        return frozenset()
    try:
        with open(ORACLE_POLICY) as f:
            policy = json.load(f).get("policy", {})
    except (OSError, ValueError):
        return frozenset()
    return frozenset(k for k, v in policy.items() if v.get("oracle_usable"))


def prefer_usable(cands):
    """Stable partition: oracle-usable cases first, declaration order within
    each group. A stable sort is what keeps a re-run a no-op."""
    usable = oracle_usable_cases()
    if not usable:
        return list(cands)
    return sorted(cands,
                  key=lambda c: f"{c.category}/{c.case}" not in usable)


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

def _match_floor(specs, kernels_of, capacity):
    """Give each spec one kernel slot, never exceeding `capacity` per kernel.

    Stage 1's floor is "every declared spec is realised at least once", and a
    spec is usually realisable on several kernels while some are realisable on
    exactly one. Greedy assignment in declaration order is therefore wrong: a
    flexible spec can take the only kernel a constrained spec has, leaving the
    constrained one unrealised while a slot sits free on another kernel. That
    failure is invisible -- the family simply looks one instance short, which is
    also what a missing operator looks like. Augmenting paths fix it exactly,
    and with at most four kernels and seven specs the O(specs x slots) search
    costs nothing.

    `capacity` is a `{kernel: n}` of free slots. Returns `{spec_id: kernel}`;
    a spec absent from the result had no kernel left anywhere, which is a fact
    about the family's declaration set and is reported, not guessed at.
    """
    slots = [(k, i) for k, n in capacity.items() for i in range(n)]
    holder = dict.fromkeys(slots)

    def augment(sid, seen):
        for k in kernels_of[sid]:
            for slot in (s for s in slots if s[0] == k):
                if slot in seen:
                    continue
                seen.add(slot)
                if holder[slot] is None or augment(holder[slot], seen):
                    holder[slot] = sid
                    return True
        return False

    for sid in specs:
        augment(sid, set())
    return {sid: slot[0] for slot, sid in holder.items() if sid is not None}

def select(cands, n_per_family=N_PER_FAMILY, n_realisations=N_REALISATIONS,
           n_cells=N_CELLS, lane=LANE):
    """Per-family, path-class-aware, deterministic selection (specs §9.6.1).

    The unit of requirement (b) is `(class, method family)`, so the budget is
    per FAMILY. Budgeting per class cannot express it -- that is the defect this
    function replaces, not a refinement of it. A per-class budget let one
    family absorb the whole allowance while another got a single instance,
    which is precisely how a corpus can satisfy "35 per class" while most
    families sit at one.

    N is a hard ceiling on a family, and it decomposes as
    `n_kernels categories x n_realisations instances per category`, so no single
    kernel can absorb the budget either.

    Stage 1 -- the two obligations that are not budget. Every declared spec in
    the family must be realised at least once, or "a spec nobody wrote" and "a
    spec that matches nothing" are indistinguishable. And every non-rt-check cell
    (`(spec_id, category)`) must be realised, because a missing injection on
    unchecked/ct-check/L is a silent noop rather than a surviving mutant. The floor is per
    SPEC, not per cell: demanding every rt-check cell would demand more instances
    than the family has slots wherever one spec is declared on several
    surfaces, and the first guarantee to break would be (b) itself. For unchecked/ct-check/L
    one per cell is the whole budget, not a floor -- nothing sweeps those paths,
    so a second instance buys nothing. A family whose obligations alone exceed N
    is REFUSED, never truncated: truncating drops a guarantee to balance a
    budget, when the budget is the thing the paper promises.

    Stage 2 -- fill. Round-robin over the family's categories until the family
    holds `n_per_family`, capped at `n_realisations` per category. A family that
    cannot reach `n_per_family` is left short and REPORTED, never topped up from
    a neighbouring family: a family whose realisations do not exist is a work
    item (W1), and borrowing instances to fill it would satisfy (b) by
    relabelling.

    avoided candidates are never chosen -- the compiler repairs the state, so there
    is no test (specs §9.5.0) -- and they are not "dropped" either: they are
    recorded N/A in `raw/spec_registry.json`.

    A candidate is never silently discarded. `dropped` carries
    `(candidate, reason)` with a reason from this closed vocabulary, and the
    reason is the binding constraint, not whichever check ran first:

      out of scope   the lane's surface cannot express the family (a
                     prohibition, see method_taxonomy.na_reason)
      cell budget    a non-rt-check cell is already covered; unchecked/ct-check/L take exactly one
                     instance per (spec_id, category) by construction
      category cap   the category already holds n_realisations in this family
      family budget  the family already holds n_per_family

    ATTRIBUTION-ONLY specs (`path: L`, prohibition `observation` -- M3-L's 7
    spec_ids) are returned separately in `attribution`. They own no family, so
    they are not evidence for requirement (b); they exist to grow the
    never-attribution table. They must be generated and must NOT be counted in
    a class cell, so putting them in `selected` would inflate M3's cell by 12
    and putting them in `dropped` would delete real data. `method-taxonomy.json`
    states the rule ("Kept for attribution, EXCLUDED from M3's 8 families and
    from M3's class cell") and this is where it is enforced.

    Returns `(selected, dropped, attribution)`. `selected` preserves declaration
    order; `attribution` is `{tag: [candidate, ...]}`. Every candidate is in
    exactly one of `selected`, `dropped`, `attribution`'s values, or the N/A set.
    """
    attr_tag = {s: tag for tag, ids in T.ATTRIBUTION_ONLY.items()
                for s in ids}

    by_family = collections.OrderedDict()
    by_attr = collections.OrderedDict()
    for c in cands:
        fam = T.family_of(c.spec_id)
        if c.spec_id in attr_tag:
            if fam is not None:
                raise ValueError(
                    "spec_id %r is declared both attribution-only (tag %r) and "
                    "a member of family %r. It cannot be both: the first makes "
                    "it invisible to requirement (b) and the second makes it "
                    "evidence for it, so counting it would give two different "
                    "answers depending on which list a reader used."
                    % (c.spec_id, attr_tag[c.spec_id], fam))
            by_attr.setdefault(attr_tag[c.spec_id], []).append(c)
            continue
        if fam is None:
            raise ValueError(
                "operator %r realises spec_id %r, which no method family owns "
                "and which is not declared attribution-only. Every generated "
                "instance must be evidence for exactly one family, or it would "
                "sit in the corpus without being counted anywhere -- the same "
                "shape as defect D1." % (c.id, c.spec_id))
        by_family.setdefault(fam, []).append(c)

    selected = []
    dropped = []

    for fam in sorted(by_family):
        members = by_family[fam]

        # ---- a family this lane cannot express is n/a, not short ----------
        if not T.in_scope(lane, fam):
            for c in members:
                dropped.append((c, "out of scope: %s cannot express %s "
                                   "(prohibition %s)"
                                   % (lane, fam, T.na_reason(lane, fam))))
            continue

        # ---- census the family's cells ------------------------------------
        # A cell is `(spec_id, category)` -- "one injection per cell" in the
        # docs. Cells are censused in declaration order so selection is
        # deterministic and a re-run is a no-op.
        cells = collections.OrderedDict()
        for c in members:
            if c.is_na:                  # avoided -- repaired, never generated
                continue
            cells.setdefault((c.spec_id, c.category), []).append(c)
        for group in cells.values():     # a decidable case first within a cell
            group[:] = prefer_usable(group)

        take = collections.OrderedDict()          # insertion-ordered id set
        cat_used = collections.Counter()
        cell_used = collections.Counter()

        def ceiling(c):
            # unchecked/ct-check/L: no threshold reaches those paths, so a second instance
            # in the same cell re-measures a constant. "One injection per cell"
            # is that path's WHOLE budget, not a floor -- and it is a ceiling
            # on the CELL, so a spec declared on two surfaces still gets two.
            # rt-check: a cell is a floor of one, but the curve is the point, so the
            # cell ceiling is the category ceiling.
            return n_cells if not c.needs_rtc_curve else n_realisations

        def room(c):
            return (len(take) < n_per_family
                    and cat_used[c.category] < n_realisations
                    and cell_used[(c.spec_id, c.category)] < ceiling(c))

        def put(c):
            take[c.id] = c
            cat_used[c.category] += 1
            cell_used[(c.spec_id, c.category)] += 1

        # ---- stage 1: the obligations that are not budget ------------------
        # Two things must happen before any budget is spent on depth:
        #
        #   * every declared spec in the family is realised at least once, or
        #     "a spec nobody wrote" and "a spec that matches nothing" look
        #     identical -- which is the whole reason GATE 2 exists;
        #   * every non-rt-check cell is realised, because on those paths a missing
        #     injection is a silent noop rather than a surviving mutant.
        #
        # The floor is on SPECS, not on (spec, category) cells. Demanding every
        # rt-check cell would demand more instances than the family has slots
        # wherever a spec is declared on several surfaces -- M1-a alone has 13
        # rt-check cells against a family budget of 8 -- and the first thing to break
        # would be requirement (b) itself. The observation is realised on one
        # surface is not a forgotten spec.
        by_spec = collections.OrderedDict()
        for key in cells:
            by_spec.setdefault(key[0], []).append(key)

        free = {k[1]: n_realisations for k in cells}

        # Non-rt-check obligations are per CELL and their kernel is not a choice, so
        # they are reserved first: a matching that spent their kernel on a
        # flexible spec could not give it back.
        for sid, keys in by_spec.items():
            if cells[keys[0]][0].needs_rtc_curve:
                continue
            for key in keys:
                if free[key[1]] <= 0:
                    continue          # unpaid; reported by `compose()` below
                free[key[1]] -= 1
                put(cells[key][0])

        # rt-check obligations are one instance per SPEC, and which kernel it lands
        # on is a choice, so it is solved as a matching rather than a loop.
        p1 = [sid for sid, keys in by_spec.items()
              if cells[keys[0]][0].needs_rtc_curve]
        kernels_of = {sid: list(dict.fromkeys(k[1] for k in by_spec[sid]))
                      for sid in p1}
        assigned = _match_floor(p1, kernels_of, free)
        for sid in p1:
            if sid in assigned:
                put(cells[(sid, assigned[sid])][0])

        # N is a ceiling, so the obligations must fit under it. They are placed
        # before the cap is consulted -- a guarantee that only holds when the
        # budget is not yet full is not a guarantee -- so the breach is caught
        # here rather than allowed to become the corpus's shape.
        if len(take) > n_per_family:
            raise ValueError(
                "family %s owes %d obligation instance(s) but N is %d: the "
                "declaration set and the budget disagree, and honouring one "
                "would silently break the other."
                % (fam, len(take), n_per_family))

        # ---- stage 2: fill the family to N ---------------------------------
        queues = collections.OrderedDict()
        for c in members:
            if c.is_na or c.id in take:
                continue
            queues.setdefault(c.category, []).append(c)
        for q in queues.values():        # a decidable case first within a kernel
            q[:] = prefer_usable(q)

        # A pass that takes nothing means every queue is drained or every
        # category is capped, so terminate on "did this pass make progress",
        # not on the instance count -- the count also rises when the head of a
        # queue was already taken, which is consumption, not a stall.
        while len(take) < n_per_family:
            progress = 0
            for cat in list(queues):
                if len(take) >= n_per_family:
                    break
                if cat_used[cat] >= n_realisations:
                    continue
                while queues[cat]:
                    c = queues[cat].pop(0)
                    if room(c):
                        put(c)
                        progress += 1
                        break
            if progress == 0:
                break

        # ---- account for every candidate this family did not take ---------
        # Exactly one reason each, from the closed vocabulary, and the reason
        # is the binding constraint -- a candidate is named by the first thing
        # that stopped it, not by whatever check happens to come first here.
        for c in members:
            if c.id in take:
                selected.append(c)
            elif c.is_na:
                pass                     # an N/A verdict, not a drop
            elif not c.needs_rtc_curve and ceiling(c) <= cell_used[
                    (c.spec_id, c.category)]:
                dropped.append((c, "cell budget: %s takes exactly one instance "
                                   "per (spec_id, category) and the cell "
                                   "(%s, %s) is already covered"
                                   % (c.path_class, c.spec_id, c.category)))
            elif cat_used[c.category] >= n_realisations:
                dropped.append((c, "category cap: %s already holds %d "
                                   "instance(s) in %s (n_realisations=%d)"
                                   % (c.category, cat_used[c.category], fam,
                                      n_realisations)))
            elif len(take) >= n_per_family:
                dropped.append((c, "family budget: %s already holds %d "
                                   "instances (n_per_family=%d)"
                                   % (fam, len(take), n_per_family)))
            else:
                # Unreachable: the fill loop stops only with every queue
                # drained or every category capped. If it is reached, one of
                # the reasons above is not the binding constraint and the
                # ledger would be lying, so say so rather than invent a label.
                raise ValueError(
                    "candidate %s in family %s was not taken and no reason in "
                    "the closed vocabulary applies: take=%d/%d, %s holds %d, "
                    "cell (%s, %s) holds %d"
                    % (c.id, fam, len(take), n_per_family, c.category,
                       cat_used[c.category], c.spec_id, c.category,
                       cell_used[(c.spec_id, c.category)]))

    # ---- attribution-only: cell floor and nothing more --------------------
    # They carry `path: L`, so no threshold reaches them and a second instance
    # per cell buys nothing. They are never budgeted against a family because
    # they are not evidence for one.
    attribution = collections.OrderedDict()
    for tag in sorted(by_attr):
        members = by_attr[tag]
        cells = collections.OrderedDict()
        for c in members:
            if c.is_na:
                continue
            cells.setdefault((c.spec_id, c.category), []).append(c)
        keep = set()
        for group in cells.values():
            for c in group[:n_cells]:
                keep.add(c.id)
        keep_list = []
        for c in members:
            if c.id in keep:
                keep_list.append(c)
            elif c.is_na:
                pass
            else:
                dropped.append((c, "cell budget: attribution-only %s takes "
                                   "exactly one instance per (spec_id, "
                                   "category), and (%s, %s) is covered"
                                   % (tag, c.spec_id, c.category)))
        if keep_list:
            attribution[tag] = keep_list

    return selected, dropped, attribution


def compose(level2=False, n_per_family=N_PER_FAMILY,
            n_realisations=N_REALISATIONS, lane=LANE):
    """Build the candidate list per class, then select, then say what is left.

    Level 2 is now real: `mutations.categories_for(cls, level2=True)` unions the
    level-2 category order into the class's coverage set, so any operator that
    declares a level-2 category is picked up. No such operator exists yet, so
    the widening is reported rather than silently a no-op.

    `family_short` is the honest part of the result. A family that ends below
    `n_per_family` is short because its realisations do not exist, and that is a
    W1 work item; reporting it here means `--dry-run` answers "what does W1 still
    have to write" without anyone recomputing it from the manifest by hand.

    `unrealised_specs` is likewise a coverage fact, not a selection outcome: a
    generatable spec with ZERO operators. Such a spec has no (spec_id, category)
    cell, so stage 1's cell floor cannot see it -- the floor covers the cells
    that exist, and a spec with no operator has none. That is the blind spot the
    floor was supposed to close, so it is closed here by name instead.
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
        chosen, dropped, attribution = select(
            gen, n_per_family=n_per_family,
            n_realisations=n_realisations, lane=lane)

        # Per-family ledger: how deep is each family, against what budget.
        # `family_of` never returns None here -- `select()` refuses a candidate
        # that owns no family -- but a None would silently become a Counter key,
        # so it is dropped rather than counted.
        depth = collections.Counter(
            f for f in (T.family_of(c.spec_id) for c in chosen) if f)
        fams = [f for f in T.families_of(cls) if T.in_scope(lane, f)]
        short = collections.OrderedDict(
            (f, n_per_family - depth.get(f, 0)) for f in fams
            if depth.get(f, 0) < n_per_family)

        have = {c.spec_id for c in cands}
        unrealised = sorted(s for f in T.families_of(cls)
                            for s in T.specs_of(f)
                            if s not in T.AVOIDED and s not in have)

        # A spec with operators that still got no instance. `select()` gives
        # every declared spec a floor instance whenever a kernel has room, so
        # this set is exactly the families whose obligations do not fit inside
        # N and the per-kernel cap -- M2-a today, where M2.19's two fixed cells
        # and M2.14's single available kernel crowd three specs into
        # layer_normalization's two slots. Derived here rather than passed back
        # from `select()` so there is one definition of "realised": a spec is
        # realised iff an instance of it is in the corpus.
        declared = collections.defaultdict(set)
        realised = collections.defaultdict(set)
        for c in gen:
            f = T.family_of(c.spec_id)
            if f:
                declared[f].add(c.spec_id)
        for c in chosen:
            realised[T.family_of(c.spec_id)].add(c.spec_id)
        overcrowded = collections.OrderedDict(
            (f, sorted(declared[f] - realised[f]))
            for f in T.families_of(cls)
            if declared[f] - realised[f])

        plan[cls] = {"chosen": chosen, "dropped": dropped, "total": len(cands),
                     "na": na, "level2": level2, "depth": depth,
                     "family_short": short, "unrealised_specs": unrealised,
                     "attribution": attribution,
                     "obligation_unmet": overcrowded}
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
        # Attribution-only candidates are emitted through exactly this path, so
        # they get the same hashes, diff and provenance as everything else. The
        # tag rides along in `attribution_only` so a consumer can subtract them
        # from a class cell without pattern-matching on spec_id.
        todo = [(m, None) for m in plan[cls]["chosen"]]
        todo += [(m, tag) for tag, group in plan[cls]["attribution"].items()
                 for m in group]
        for mut, attr in todo:
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
            # DIFFERENT question (avoided only), so deriving `applicable` from
            # `is_na` here would stamp L1/L2/L4 and the noop controls as
            # applicable and quietly corrupt the denominator.
            rec.update(mut.as_meta())
            rec["level"] = lvl            # re-assert: as_meta carries a level
            rec["needs_rtc_curve"] = mut.needs_rtc_curve
            rec["attribution_only"] = attr
            records.append(rec)
    return records, skipped


def report(plan, records, skipped, rtc=None, n_per_family=N_PER_FAMILY,
           n_realisations=N_REALISATIONS, lane=LANE):
    print(f"selected {len(records)} mutants  (skipped {len(skipped)})"
          + (f"  @ -rtc={rtc}" if rtc else ""))
    print(f"target: N = {n_per_family} per (class x method family) per "
          f"in-scope lane, as {n_per_family // n_realisations} kernels x "
          f"{n_realisations} realisations, lane={lane}")
    print(f"{'class':<6}{'candidates':>11}{'chosen':>8}{'rt-check':>8}{'cells':>7}"
          f"{'N/A':>6}{'dropped':>9}{'budget':>8}")
    for cls in sorted(plan):
        p = plan[cls]
        n_p1 = len([c for c in p["chosen"] if c.needs_rtc_curve])
        n_cells = len(p["chosen"]) - n_p1
        target = len([f for f in T.families_of(cls)
                      if T.in_scope(lane, f)]) * n_per_family
        print(f"{cls:<6}{p['total']:>11}{len(p['chosen']):>8}{n_p1:>6}"
              f"{n_cells:>7}{len(p['na']):>6}{len(p['dropped']):>9}"
              f"{target:>8}")

    # ---- the accounting identity, per class and in total -------------------
    # "selected" alone is not auditable: the corpus is chosen + dropped + n/a +
    # attribution-only, and nothing used to print the last three, so 59
    # declarations vanished with no arithmetic that could notice.
    tot = [0, 0, 0, 0, 0]
    for cls in sorted(plan):
        p = plan[cls]
        tot[0] += p["total"]
        tot[1] += len(p["chosen"])
        tot[2] += len(p["dropped"])
        tot[3] += len(p["na"])
        tot[4] += sum(len(g) for g in p["attribution"].values())
    print(f"\naccounting: candidates {tot[0]} = chosen(family) {tot[1]} + "
          f"attribution-only {tot[4]} + dropped {tot[2]} + n/a {tot[3]}")
    assert tot[0] == tot[1] + tot[2] + tot[3] + tot[4], (
        "the candidate ledger does not balance: %d != %d + %d + %d + %d"
        % (tot[0], tot[1], tot[4], tot[2], tot[3]))
    print(f"class cell (chosen only -- what requirement (b) counts): {tot[1]}")

    # ---- per-family depth against the budget ------------------------------
    depth = collections.Counter(
        f for f in (T.family_of(r["spec_id"]) for r in records) if f)
    print(f"\nper (class x method family) -- depth vs N={n_per_family}:")
    n_short = 0
    for cls in sorted(plan):
        for f in T.families_of(cls):
            if not T.in_scope(lane, f):
                continue
            d = depth.get(f, 0)
            mark = "" if d >= n_per_family else "  SHORT -%d" % (n_per_family - d)
            if d < n_per_family:
                n_short += 1
            print(f"  {f:<6}{T.name(f):<26}{d:>3}{mark}")
    print(f"  families short of N: {n_short}")

    print("\nwhy candidates were not chosen (closed vocabulary):")
    why = collections.Counter(
        reason.split(":")[0] for p in plan.values()
        for _, reason in p["dropped"])
    if not why:
        print("  (none -- every generatable candidate was chosen)")
    for k, v in why.most_common():
        print(f"  {v:>5}  {k}")

    print("\ndeclared but never realised (no operator at all):")
    any_unrealised = False
    for cls in sorted(plan):
        if plan[cls]["unrealised_specs"]:
            any_unrealised = True
            print(f"  {cls}: {plan[cls]['unrealised_specs']}")
    if not any_unrealised:
        print("  (none)")

    # Distinct from the list above and much more specific: these specs DO have
    # operators, and still got no instance, because the family's obligations do
    # not fit N with the per-kernel cap. A spec with operators that is silently
    # absent is the exact failure the family axis exists to prevent, so it is
    # named here rather than left to be noticed in a shortfall count.
    print("\nobligations not met within N (operators exist, no instance "
          "selected):")
    any_unmet = False
    for cls in sorted(plan):
        for f, sids in plan[cls]["obligation_unmet"].items():
            any_unmet = True
            print(f"  {f:<6}holds {depth.get(f, 0):>2} of N "
                  f"{n_per_family}; unrealised: {sids}")
    if not any_unmet:
        print("  (none -- every declared spec with an operator is realised)")

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
    global N_CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--level2", action="store_true",
                    help="also widen to the level-2 category set (specs §5)")
    ap.add_argument("--lane", default=LANE, choices=T.lanes(),
                    help="the lane whose scope model drives n/a families "
                         "(default %s)" % LANE)
    ap.add_argument("--rtc", choices=RTC_LEVELS, default=None,
                    help="stamp the -rtc level this corpus will be run at "
                         "(specs §9.6.1: the curve is a RUN parameter; rt-check "
                         "classes are re-run at every level)")
    ap.add_argument("--n-per-family", type=int, default=N_PER_FAMILY,
                    help="instances per (class x method family), from "
                         "schema/method-taxonomy.json (default %d)"
                         % N_PER_FAMILY)
    ap.add_argument("--n-realisations", type=int, default=N_REALISATIONS,
                    help="cap per kernel/category -- the 'realisations' half of "
                         "N = kernels x realisations (default %d)"
                         % N_REALISATIONS)
    ap.add_argument("--n-cells", type=int, default=N_CELLS,
                    help="injections per (spec x category) cell for unchecked/ct-check/L "
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
    ap.add_argument("--fail-on-short", action="store_true",
                    help="exit 3 if any in-scope family is below "
                         "N. This is the W1 gate, and it is opt-in on purpose: "
                         "today it fails by construction (19 of 31 families are "
                         "short), and a gate that a known-incomplete corpus "
                         "always fails is noise, not a gate. It becomes the "
                         "default once W1 closes.")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()

    N_CELLS = a.n_cells
    # `--level2` used to APPEND the level-2 categories into M.MINIMAL_SET, which
    # (a) mutated the one definition the rest of the module reads for coverage
    # and for the `level` stamp, so a level-2 run silently relabelled level-2
    # categories as level 1, and (b) made the widening order-dependent within a
    # process. compose() already passes `level2` to categories_for(); nothing
    # else needs to know.

    plan = compose(a.level2, n_per_family=a.n_per_family,
                   n_realisations=a.n_realisations, lane=a.lane)
    records, skipped = emit(plan, a.out, dry_run=a.dry_run)
    summary = M.registry_summary()
    n_specs = write_spec_registry(SPEC_REGISTRY_OUT)
    report(plan, records, skipped, rtc=a.rtc, n_per_family=a.n_per_family,
           n_realisations=a.n_realisations, lane=a.lane)
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
    short = sorted(f for cls in plan for f, _ in plan[cls]["family_short"].items())
    if a.fail_on_short and short:
        print("\nW1 GATE FAILED: %d in-scope families are below N=%d: %s"
              % (len(short), a.n_per_family, ", ".join(short)),
              file=sys.stderr)
        rc = 3
    if a.dry_run:
        return rc

    # Every candidate that is not a mutant is written down. The manifest is the
    # provenance record for the corpus, and a corpus records what it does not
    # contain as well as what it does: without `dropped` and `na` the identity
    # `candidates == chosen + dropped + n/a` is not recoverable from committed
    # data, so 59 vanished declarations were invisible until someone re-ran the
    # generator and diffed by hand.
    dropped_recs = []
    for cls in sorted(plan):
        for c, reason in plan[cls]["dropped"]:
            dropped_recs.append({
                "class": cls,
                "mutant_id": c.id,
                "spec_id": c.spec_id,
                "family": T.family_of(c.spec_id),
                "category": c.category,
                "case": c.case,
                "path_class": c.path_class,
                "reason": reason,
            })
    na_recs = [{"class": cls, "mutant_id": c.id, "spec_id": c.spec_id,
                "family": T.family_of(c.spec_id), "category": c.category,
                "case": c.case, "path_class": c.path_class,
                "prohibition": c.prohibition}
               for cls in sorted(plan) for c in plan[cls]["na"]]
    attr_recs = [{"class": cls, "tag": tag, "mutant_id": c.id,
                  "spec_id": c.spec_id, "category": c.category,
                  "case": c.case, "path_class": c.path_class}
                 for cls in sorted(plan)
                 for tag, group in plan[cls]["attribution"].items()
                 for c in group]
    family_depth = {f: n for f, n in collections.Counter(
        x for x in (T.family_of(r["spec_id"]) for r in records) if x).items()}

    # The ledger, over the whole candidate set. `selected` is what the family
    # axis decided; `emitted` is what reached the disk, and they differ by
    # `skipped` -- a selected mutant whose edit did not match the base kernel,
    # which is a GATE 2 failure and not a budget decision. Keeping both makes
    # `candidates == selected + attribution + dropped + na` checkable by a
    # reader of the manifest rather than only by a re-run of the generator.
    accounting = {
        "candidates": sum(p["total"] for p in plan.values()),
        "selected": sum(len(p["chosen"]) for p in plan.values()),
        "attribution": len(attr_recs),
        "dropped": len(dropped_recs),
        "na": len(na_recs),
        "skipped": len(skipped),
        "emitted": len(records),
    }
    assert accounting["candidates"] == (accounting["selected"]
                                        + accounting["attribution"]
                                        + accounting["dropped"]
                                        + accounting["na"]), accounting
    assert accounting["emitted"] == (accounting["selected"]
                                      + accounting["attribution"]
                                      - accounting["skipped"]), accounting

    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump({"toolchain": TOOLCHAIN,
                   "spec_version": SPEC_VERSION,
                   "lane": a.lane,
                   "n_per_family": a.n_per_family,
                   "n_realisations": a.n_realisations,
                   "n_kernels": a.n_per_family // a.n_realisations,
                   "n_cells": N_CELLS,
                   "target_instances": T.target(a.lane),
                   "level2": a.level2,
                   "rtc": a.rtc,
                   "accounting": accounting,
                   "family_depth": {f: family_depth.get(f, 0)
                                    for cls in sorted(plan)
                                    for f in T.families_of(cls)
                                    if T.in_scope(a.lane, f)},
                   "family_short": {f: n for cls in sorted(plan)
                                    for f, n in plan[cls]["family_short"].items()},
                   "unrealised_specs": {cls: plan[cls]["unrealised_specs"]
                                        for cls in sorted(plan)},
                   "obligation_unmet": {f: sids for cls in sorted(plan)
                                        for f, sids
                                        in plan[cls]["obligation_unmet"].items()},
                   "registry": summary,
                   "dropped": dropped_recs,
                   "na": na_recs,
                   "attribution": attr_recs,
                   "mutants": records}, f, indent=1)
    print(f"\nwrote {len(records)} mutants -> {os.path.relpath(a.out, REPO)}")
    print(f"manifest -> {os.path.relpath(a.manifest, REPO)}"
          f"  (+{len(attr_recs)} attribution-only, {len(dropped_recs)} dropped, "
          f"{len(na_recs)} n/a)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
