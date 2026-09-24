"""The ONE loader for the mutation method-family taxonomy.

`schema/method-taxonomy.json` owns level 2 of the taxonomy -- the 31 method
families inside the four classes, and which spec_ids realise each one.
`schema/class_axis.py` owns level 1. Nothing else may restate either list.

Run `python3 schema/method_taxonomy.py --check` to validate the partition.

WHY THIS EXISTS
---------------
Requirement (b) -- an equal per-lane method count -- is a claim about
`(class, method family)`, not about `spec_id`. Before this module the family
partition existed only in prose, in `plan/mutation-method-taxonomy.md`, so
nothing could fail when a lane quietly realised four methods where another
realised eight. Two defects were live and neither was detectable:

  D1  `M4.2` was the sole realisation of M4-b AND one of M4-a's two, so a
      single operator was evidence for two families and counted in two class
      cells. Resolved by the `split` entry: the 4 negative cases move to M4.6.
  D2  `spec.enums.status` had no `unwritten`, so the specs that are neither
      implemented nor scheduled emitted no record at all and their N/A
      verdicts were unauditable.

WHAT THE CHECKS MEAN
--------------------
`--check` enforces the invariants a family partition must have, and is the
body of gates `families.partition` and `families.no-shared-instance` of
`schema/check_class_axis.py`:

  families.partition
       the partition is a bijection: every declared spec_id lands in exactly
       ONE family or in an explicit exclusion (attribution_only, or the
       `reassigned`/`unassigned` lists). A spec_id in two families is what
       makes D1 possible.
  families.no-shared-instance
       no spec_id is the sole realisation of two families.
  plus every family has a non-empty spec_id list, N is 8, and the family names
  are unique within a class.

`spec_ids: []` is legal only for a family declared `prohibition: "absent"`:
it has no realisation the model can forbid, so it is withdrawn from the cell
rather than reported as work. No family carries that declaration today (M4-g,
the former case, was restored 2026-09-24), but the mechanism is kept for a
future withdrawal. R_f == 0 on a family that is NOT declared absent is still
reported, never hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TAXONOMY_PATH = os.path.join(HERE, "method-taxonomy.json")

# A spec_id that ships an operator but is deliberately NOT in a family.
# (spec_id -> why). `reassigned` is not here: a reassigned spec_id IS in a
# family, just a different one from the class it was originally declared under.
EXCLUDED: dict[str, str] = {}

with open(TAXONOMY_PATH, encoding="utf-8") as _f:
    _T = json.load(_f)

VERSION: str = _T["taxonomy_version"]
N_PER_FAMILY: int = _T["budget"]["N_per_family"]
# N = kernel_component x realisation_component. Both halves are asserted by
# `check()` against `N_per_family`, so a consumer may read the product and the
# factors from here rather than restate either -- `gen_mutants.py::select()`
# caps each category at N_REALISATIONS so that a selection of N per family
# really is "N_kernels kernels x N_REALISATIONS realisations", not N instances
# piled onto one kernel.
N_KERNELS: int = _T["budget"]["kernel_component"]
N_REALISATIONS: int = _T["budget"]["realisation_component"]
IN_SCOPE: dict[str, list[str]] = _T["in_scope_lanes"]
LANE_SCOPE: dict[str, dict[str, str]] = _T["lane_scope"]
DECLARED_TARGETS: dict[str, int] = _T["targets"]
MEASURED_TODAY: dict[str, int] = _T["measured_today"]
VIEW_MEMBERS: list[str] = _T["view_family"]["members"]
PROHIBITIONS: list[str] = _T["prohibitions"]
DEAD: list[str] = _T["dead_declarations"]["spec_ids"]
# Terminal: the model states no legal program carrying the defect, so no
# operator is owed. Unlike DEAD, writing one is not the remedy. Signed off
# 2026-09-24; see `unexpressible_declarations` in the taxonomy file.
UNEXPRESSIBLE: list[str] = _T["unexpressible_declarations"]["spec_ids"]
AVOIDED: list[str] = _T["avoided_never_generated"]["spec_ids"]
DEBT: dict[str, dict] = _T["known_debt"]["items"]
ATTRIBUTION_ONLY: dict[str, list[str]] = {
    k: v["spec_ids"] for k, v in _T["attribution_only"].items()
}

_FAM: dict[str, dict] = _T["families"]


def families() -> list[str]:
    """All family ids, class-major then alphabetical: M1-a..h, M2-a..h, ..."""
    return list(_FAM)


def classes() -> list[str]:
    """The classes that OWN families, in order. Not the class axis -- see below."""
    seen: list[str] = []
    for f in _FAM:
        c = _FAM[f]["class"]
        if c not in seen:
            seen.append(c)
    return seen


def families_of(cls: str) -> list[str]:
    return [f for f in _FAM if _FAM[f]["class"] == cls]


def name(family: str) -> str:
    return _FAM[family]["name"]


def edit(family: str) -> str:
    return _FAM[family]["edit"]


def is_view(family: str) -> bool:
    return bool(_FAM[family].get("view"))


def is_noop_control(family: str) -> bool:
    """M4-d. Its instances are generated but admissible by construction, so it
    is excluded from the admissible floor and from every coverage claim. It is
    NOT excluded from the budget."""
    return bool(_FAM[family].get("noop_control"))


def is_absent(family: str) -> bool:
    """A family the model declares with no prohibition anywhere: nothing can
    forbid the state, so NO operator it could carry may ever enter a
    denominator. It is a declaration, not work.

    No family is declared absent today: M4-g (degenerate pad) was the only one,
    and it was restored to the cell on 2026-09-24 (its realisations M4.7/M4.8
    are `rt-check`, and a mutation-only kernel may carry the state). The
    mechanism is kept for a future withdrawal. An absent family is excluded
    from its class cell and from every lane's target -- reporting it as an 0/8
    shortfall would send the reader after instances that cannot be a test
    (method_taxonomy --check owns the arithmetic; `m4.md` section 6 owns the
    decision).
    """
    return _FAM[family].get("prohibition") == "absent"


def specs_of(family: str) -> list[str]:
    return list(_FAM[family]["spec_ids"])


def family_of(spec_id: str) -> str | None:
    """The single family that owns this spec_id, or None if it is excluded.

    Raises if the spec_id is in more than one family: that is exactly the D1
    defect, and silently returning the first match would hide it.
    """
    hits = [f for f in _FAM if spec_id in _FAM[f]["spec_ids"]]
    if len(hits) > 1:
        raise ValueError(
            f"{spec_id} belongs to {len(hits)} families {hits}: every spec_id "
            f"must be a realisation of exactly one family (see split/M4.6).")
    return hits[0] if hits else None


def spec_ids() -> list[str]:
    """Every spec_id that is in exactly one family."""
    out: list[str] = []
    for f in _FAM:
        out.extend(_FAM[f]["spec_ids"])
    return out


def _generatable(spec_id: str, registry: dict) -> bool:
    """An operator exists and the spec is not structurally refused."""
    e = registry.get(spec_id, {})
    return e.get("status") == "implemented" and e.get("path") != "avoided"


def _refused(spec_id: str, registry: dict) -> bool:
    """Path AVOIDED -- the compiler refuses or repairs the state, so NO operator
    for this spec_id could ever produce an instance, now or later.

    This is deliberately NOT the same as `status != implemented`. A spec that
    is merely unwritten on an rt-check/unchecked path is work waiting to be done; a spec on
    AVOIDED is work that CANNOT be done under this declaration. Merging the two is
    what made M3-h (refused) read like M3-d (unwritten) -- they need opposite
    responses.
    """
    return registry.get(spec_id, {}).get("path") == "avoided"


def _admissible(spec_id: str, registry: dict) -> bool:
    """Generatable AND it can serve as a TEST (R3 survives repair, R4
    attributable, and the obligation is actually assessed)."""
    e = registry.get(spec_id, {})
    return _generatable(spec_id, registry) and e.get("admissible") is True


def r_f(family: str, registry: dict | None = None) -> int:
    """R_f -- realisation DEPTH: how many realised operators the family has.

    Counts spec_ids with status `implemented` that are not on AVOIDED. This is the
    number that must reach 2 before N = 8 is realisable.

    Pass `registry` (mutations.SPEC_REGISTRY) to measure; without it, returns
    the number of DECLARED spec_ids, which is an upper bound and is what this
    module can know on its own.
    """
    if registry is None:
        return len(specs_of(family))
    return sum(1 for s in specs_of(family) if _generatable(s, registry))


def r_a(family: str, registry: dict) -> int:
    """R_a -- ADMISSIBLE depth: how many realisations can be a TEST.

    R_f and R_a differ, and the difference is a finding, not noise. A family
    with R_f >= 1 and R_a == 0 ships operators that cannot witness anything.
    M4-d is the only such family today, and it is the noop control, so the gap
    is by design (`is_noop_control`). Reporting only R_f calls it a healthy 1.
    """
    return sum(1 for s in specs_of(family) if _admissible(s, registry))


def needs_new_realisation(family: str, registry: dict) -> bool:
    """R_f < 2 -- the family is short of the 4x2 budget."""
    return r_f(family, registry) < 2


def unrealisable_as_declared(family: str, registry: dict) -> bool:
    """Every declared spec is on AVOIDED, so no amount of writing these spec_ids
    yields a test. Distinguishes 'needs a NEW realisation' (M3-h, whose only
    declaration M3.12 is AVOIDED/repaired) from 'needs these written' (M3-d, whose
    M3.4 is rt-check and merely unwritten)."""
    specs = specs_of(family)
    if not specs:
        return False
    return all(_refused(s, registry) for s in specs)


def no_admissible_test(family: str, registry: dict) -> bool:
    """Has at least one operator, but none of them can be a test."""
    return r_f(family, registry) > 0 and r_a(family, registry) == 0


def no_admissible_declaration(family: str, registry: dict) -> bool:
    """Declares specs, but EVERY one of them is inadmissible.

    Differs from `unrealisable_as_declared`, which is about AVOIDED -- a spec the
    compiler refuses to generate. Here the spec may well be on rt-check and merely
    unwritten; it carries a prohibition (`absent`, usually) that means no
    obligation assesses it, so no operator on it may enter the denominator.

    This is the distinction that matters for triage. A family in this state was
    being counted as `R_f < 2, needs the declared realisation written`, which
    sends the work in exactly the wrong direction: writing the operators raises
    R_f, leaves R_a at 0, and adds nothing to the denominator. The remedy is a
    model change (give the state a prohibition) or an n/a declaration -- a
    design call, not a coding one.

    A family with NO specs at all returns False: that is `no_declaration`, a
    different defect with a different fix.
    """
    spec = specs_of(family)
    if not spec:
        return False
    return not any(registry.get(s, {}).get("admissible") for s in spec)


def n(family: str) -> int:
    """The budget. Always N_PER_FAMILY -- there is no min(2, R_f) fallback."""
    return N_PER_FAMILY


def thin_families(registry: dict) -> dict[str, int]:
    """Families whose R_f < 2, i.e. the work list. R_f == 0 is included."""
    return {f: r_f(f, registry) for f in _FAM if r_f(f, registry) < 2}


def summary(registry: dict) -> dict[str, list[str]]:
    """The work list, split by WHY each family is short.

    Four groups need four different actions and must not be merged:

      no_declaration  the family declares no spec at all -- invent one
      unrealisable_as_declared
                      every declared spec is on AVOIDED, so the compiler refuses the
                      state: a NEW realisation is needed, not the declared one
                      (M3-h)
      model_gap       every declared spec is inadmissible but NOT on AVOIDED -- the
                      model does not forbid the state (`absent`), so no
                      operator may enter the denominator however many are
                      written (M3-d, M3-e, M3-f). The remedy is a prohibition
                      or an n/a verdict, i.e. a design call
      thin            at least one declared spec is admissible, so writing a
                      further admissible realisation really is the work

    `thin` is deliberately the LAST resort: a family only lands there once every
    cheaper explanation has been ruled out, because "just write the spec" is
    the label that wastes a day when it is wrong.
    """
    out: dict[str, list[str]] = {"no_declaration": [],
                                 "unrealisable_as_declared": [],
                                 "model_gap": [], "noop_control": [],
                                 "thin": []}
    for f in _FAM:
        if is_absent(f):
            continue          # withdrawn by declaration, not a work item
        if not specs_of(f):
            out["no_declaration"].append(f)
            continue
        if unrealisable_as_declared(f, registry):
            out["unrealisable_as_declared"].append(f)
        elif no_admissible_declaration(f, registry):
            # A noop control must be inadmissible -- it is the baseline, not a
            # test -- so it lands here by construction. Bucketed separately so
            # this list reconciles exactly with the DEBT ledger, which exempts
            # it by the same predicate. A summary that prints a family the
            # gate does not report is how M4-d would look like a work item.
            if is_noop_control(f):
                out["noop_control"].append(f)
            else:
                out["model_gap"].append(f)
        elif r_f(f, registry) < 2:
            out["thin"].append(f)
    return out


def n_a_families(registry: dict, in_scope_only: bool = False) -> dict[str, str]:
    """Families that cannot produce an instance at all, with the prohibition.

    A family is n/a for a lane when none of its spec_ids can generate: every
    one is either not `implemented`, or sits on path AVOIDED, or carries a
    prohibition. This is a MEASURED verdict and it carries a reason -- the
    difference between "the lane cannot express this" and "nobody wrote it" is
    the reason string, and collapsing the two is what makes a coverage table
    lie.
    """
    out: dict[str, str] = {}
    for f in _FAM:
        if is_absent(f):
            out[f] = "absent"
            continue
        specs = specs_of(f)
        if not specs:
            out[f] = "unwritten"
            continue
        live = [s for s in specs
                if registry.get(s, {}).get("status") == "implemented"
                and registry.get(s, {}).get("path") != "avoided"
                and not registry.get(s, {}).get("prohibition")]
        if not live:
            r0 = registry.get(specs[0], {})
            out[f] = r0.get("prohibition") or "unwritten"
    return out


def lanes() -> list[str]:
    """Every lane named anywhere in the scope model, stable order."""
    seen: list[str] = []
    for ls in IN_SCOPE.values():
        for l in ls:
            if l not in seen:
                seen.append(l)
    return seen


def runs_class(lane: str, cls: str) -> bool:
    """Whether the lane runs this class AT ALL. Coarse -- use in_scope()."""
    return lane in IN_SCOPE.get(cls, [])


def in_scope(lane: str, family: str) -> bool:
    """Whether a lane is required to realise this family at N.

    Scope is per (class x lane x FAMILY), not per (class x lane). A lane that
    runs M3 still has M3 families its surface cannot express, and those are
    n/a with a reason rather than missing instances. Triton's M3 has 6
    in-scope families of 8; iree's M2 has 4 of 8.
    """
    cls = _FAM[family]["class"]
    if lane not in IN_SCOPE.get(cls, []):
        return False
    if is_absent(family):
        return False
    return family not in LANE_SCOPE.get(lane, {})


def na_reason(lane: str, family: str) -> str | None:
    """The prohibition for an out-of-scope (lane, family) cell, else None.

    A cell is n/a in one of two ways and they must not be collapsed:
      * the lane's surface cannot express the defect -> a prohibition;
      * the family has no implementation anywhere -> handled by
        na_families(), because it is a gap in the TAXONOMY, not in the lane.
    """
    if not runs_class(lane, _FAM[family]["class"]):
        return LANE_SCOPE.get(lane, {}).get(
            family, "absent")  # whole class out of scope
    return LANE_SCOPE.get(lane, {}).get(family)


def target(lane: str) -> int:
    """Planned instances for a lane: N x the families in scope for it."""
    return sum(N_PER_FAMILY for f in _FAM if in_scope(lane, f))


def families_in_scope(lane: str) -> list[str]:
    return [f for f in _FAM if in_scope(lane, f)]


def targets() -> dict[str, int]:
    return {l: target(l) for l in lanes()}


def label(family: str) -> str:
    """`M1-b stride / step*` -- the form the paper table and the ledger use."""
    star = "$^{\\ast}$" if is_view(family) else ""
    return f"{family} {name(family)}{star}"


# --------------------------------------------------------------------------
# Checks (the body of gates `families.partition` and
# `families.no-shared-instance`)
# --------------------------------------------------------------------------
def check(registry: dict | None = None, declared: list[str] | None = None,
          strict_debt: bool = False) -> tuple[list[str], list[str]]:
    """Return `(bad, debt)`. `bad` blocks; `debt` is real and ledgered.

    A finding is split, not silenced: every finding carries a machine key
    `<family>:<finding>`, and it becomes `debt` only if that key is listed in
    `known_debt` of the taxonomy file. An unlisted finding is `bad`, and a
    listed key that no longer fires is ALSO `bad` (stale entry) -- so debt can
    be neither quietly taken on nor quietly dropped.

    Set `strict_debt` to force debt back into `bad`, for the release gate.
    """
    bad: list[str] = []
    debt: list[str] = []
    debt_keys: set[str] = set()

    def report(family: str, finding: str, text: str) -> None:
        key = f"{family}:{finding}"
        if key in DEBT and not strict_debt:
            debt.append(text)
            debt_keys.add(key)
        else:
            bad.append(text)

    # Every family has a non-empty id list, a name, and a legal class.
    cls_axis = None
    try:
        import class_axis as _AX
        cls_axis = _AX.mutation_classes()
    except Exception:
        pass
    for f, d in _FAM.items():
        if not d.get("name"):
            bad.append(f"{f}: no name")
        if not d.get("edit"):
            bad.append(f"{f}: no structural edit")
        if d["class"] not in (cls_axis or classes()):
            bad.append(f"{f}: class {d['class']} is not on the class axis "
                       f"{cls_axis or classes()}")
        if not d["spec_ids"] and not is_absent(f):
            bad.append(f"{f}: no spec_ids and is not declared absent")

    # ---- the budget block is one arithmetic claim in several fields -------
    # It is stated as `N = kernel_component x realisation_component` and again
    # as a per-class cell size, and a consumer may legitimately read any of
    # them. When the JSON says 4 x 2 and N = 8 but nothing compares them, the
    # first edit to one factor silently re-defines the unit of requirement (b)
    # -- every headline count in the paper is a multiple of N, so the drift is
    # invisible in the numbers and total in the meaning. Checked here, once.
    if N_KERNELS * N_REALISATIONS != N_PER_FAMILY:
        bad.append(f"budget: {N_KERNELS} kernels x {N_REALISATIONS} "
                   f"realisations != N_per_family {N_PER_FAMILY}")
    for c, d in _T["budget"]["classes"].items():
        # An absent family is excluded from the cell: nothing it could carry
        # may enter the denominator, so counting it would demand work that
        # cannot be done and misstate the class's target.
        n_fam = len([f for f in families_of(c) if not is_absent(f)])
        if d["families"] != n_fam:
            bad.append(f"budget: class {c} declares {d['families']} families "
                       f"but the family axis has {n_fam}")
        if d["cell"] != n_fam * N_PER_FAMILY:
            bad.append(f"budget: class {c} cell {d['cell']} != {n_fam} "
                       f"families x {N_PER_FAMILY} = {n_fam * N_PER_FAMILY}")
    # `targets` claims to be derived from the scope model rather than written
    # down. It was written down and nothing compared it: `DECLARED_TARGETS`
    # was loaded and never read, so --check could not fail on a stale target
    # even though the file said it would. Derive and compare.
    for lane_, want in targets().items():
        got = DECLARED_TARGETS.get(lane_)
        if got != want:
            bad.append(f"budget: lane {lane_} declares target {got} but N x "
                       f"families-in-scope = {want}")
    tot = sum(targets().values())
    if DECLARED_TARGETS.get("total") != tot:
        bad.append(f"budget: total declares {DECLARED_TARGETS.get('total')} "
                   f"but the lanes sum to {tot}")

    # Family names unique within a class (two families with one name are one
    # family that was written twice).
    for c in classes():
        seen: dict[str, str] = {}
        for f in families_of(c):
            nm = name(f)
            if nm in seen:
                bad.append(f"{c}: families {seen[nm]} and {f} share the name "
                           f"{nm!r}")
            seen[nm] = f

    # families.partition: bijection. Each spec_id in exactly one family, or
    # explicitly out.
    seen_ids: dict[str, list[str]] = {}
    for f in _FAM:
        for s in specs_of(f):
            seen_ids.setdefault(s, []).append(f)
    for s, fs in sorted(seen_ids.items()):
        if len(fs) > 1:
            bad.append(f"families.partition {s} is a realisation of {len(fs)} families "
                       f"{fs}: one operator cannot be the evidence for two "
                       f"families (defect D1)")
    if declared is not None:
        accounted = set(seen_ids)
        for ids in ATTRIBUTION_ONLY.values():
            accounted |= set(ids)
        for s in AVOIDED:
            accounted.add(s)
        for s in _T["reassigned"]:
            accounted.add(s)
        for s in declared:
            if s not in accounted:
                bad.append(f"families.partition {s} is declared but in no family and in no "
                           f"exclusion: add it to `families` or to "
                           f"`reassigned`/`avoided_never_generated`/"
                           f"`attribution_only`")
        for bucket, ids in (("attribution_only", [i for v in
                             ATTRIBUTION_ONLY.values() for i in v]),
                            ("avoided_never_generated", AVOIDED)):
            for s in ids:
                if s in seen_ids:
                    bad.append(f"families.partition {s} is in {bucket} AND in "
                               f"family {seen_ids[s]}: it must leave the "
                               f"class cell")

    # families.no-shared-instance: no spec_id is the sole realisation of two
    # families.
    if registry is not None:
        singles: dict[str, list[str]] = {}
        for f in _FAM:
            impl = [s for s in specs_of(f) if _generatable(s, registry)]
            if len(impl) == 1:
                singles.setdefault(impl[0], []).append(f)
        for s, fs in sorted(singles.items()):
            if len(fs) > 1:
                bad.append(f"families.no-shared-instance {s} is the SOLE implemented realisation of "
                           f"{fs}: those families have no independent "
                           f"realisation (defect D1)")
        # A family whose EVERY declaration is inadmissible cannot be satisfied
        # by writing more operators: no operator it could ever carry enters the
        # denominator. That is either a model gap (`absent` -- the model does
        # not forbid the state, so there is no obligation to violate) or a
        # refusal (`repaired`), and the remedy is a design call. Reported
        # separately from "unwritten" because the two send the reader in
        # opposite directions.
        for f in _FAM:
            if not specs_of(f) or is_noop_control(f) or is_absent(f):
                continue
            if unrealisable_as_declared(f, registry):
                report(f, "unrealisable as declared",
                       f"families.admissibility {f} is unrealisable as "
                       f"declared: every spec_id "
                       f"{specs_of(f)} is on AVOIDED (refused/repaired), so it "
                       f"needs a NEW realisation, not the declared one")
            elif no_admissible_declaration(f, registry):
                report(f, "no admissible declaration",
                       f"families.admissibility {f} has NO admissible "
                       f"declaration: every spec_id "
                       f"{specs_of(f)} carries a prohibition, so no operator "
                       f"it could carry may enter the denominator -- this is "
                       f"a model gap, not unwritten work")

    # families.dead-vs-unwritten: `dead` and `unwritten` are DIFFERENT sets, and
    # the difference is
    # exactly the refused-that-still-exists specs.
    #
    #   dead_declarations -- the declared surface does not exist in the DSL,
    #                        so there is nothing to write until the DSL grows
    #                        it
    #   unwritten         -- no operator has been written (status == pending)
    #
    # They were briefly assumed to be equal, which made a 21-vs-23 gap look
    # like a bookkeeping bug. It is not: a spec can be AVOIDED (the compiler
    # repairs the state, so no mutation is generatable) while its declared
    # surface very much exists. Those specs are unwritten for a different
    # reason and need a different remedy, so folding them into
    # `dead_declarations` would send someone off to build a DSL feature that
    # is already there.
    if registry is not None:
        dead = set(_T["dead_declarations"]["spec_ids"])
        # Structural fact, NOT the `avoided_never_generated` reporting list. That
        # list is the specs whose ONLY defect is AVOIDED, so it deliberately omits
        # M3.12 (already reported by `unrealisable as declared`). Using it here
        # would demand that M3.12 be dead as well, which is the opposite of
        # true -- M3.12 exists and is exactly why M3-h needs a new realisation.
        avoided = {s for s, v in registry.items() if v.get("path") == "avoided"}
        unexp = set(_T["unexpressible_declarations"]["spec_ids"])
        # A terminal unexpressible spec is NOT unwritten: writing an operator is
        # not the remedy, so it must not enter the dead-vs-unwritten arithmetic
        # (it would otherwise look like a non-dead pending spec that must be
        # AVOIDED). Consistency with the registry is asserted so the two files
        # cannot disagree about the verdict.
        for s in sorted(unexp):
            e = registry.get(s)
            if e is None:
                bad.append(f"families.unexpressible {s} is declared "
                           f"unexpressible but is absent from SPEC_REGISTRY")
            elif e.get("status") == "implemented":
                bad.append(f"families.unexpressible {s} is declared "
                           f"unexpressible but ships an operator "
                           f"(status implemented): an unexpressible spec must "
                           f"not be generatable")
        unwritten = {s for s, v in registry.items()
                     if v.get("status") == "pending" and s not in unexp}
        if not dead <= unwritten:
            bad.append("families.dead-vs-unwritten dead_declarations is not a "
                       "subset of the "
                       "unwritten specs: %s is called dead but has an "
                       "operator"
                       % ", ".join(sorted(dead - unwritten)))
        extra = unwritten - dead
        expected = avoided - dead
        if extra != expected:
            bad.append(
                "families.dead-vs-unwritten an unwritten spec that is not "
                "`dead` must be AVOIDED "
                "(refused, surface present). Got %s, expected %s"
                % (sorted(extra), sorted(expected)))

    # families.ids-canonical: every spec_id is spelled canonically -- `M3.2`,
    # never `M3.02`.
    # The registry keys and the taxonomy declare the same specs in two files,
    # so a padding mismatch would make a spec look absent from one side while
    # present in the other: the partition would silently lose a member and the
    # budget would silently gain a slot. Cheap to assert, invisible to the eye.
    _all_ids = set()
    for _f in _FAM:
        _all_ids |= set(specs_of(_f))
    _all_ids |= set(_T["dead_declarations"]["spec_ids"])
    _all_ids |= set(UNEXPRESSIBLE)
    _all_ids |= set(AVOIDED)
    for _ids in ATTRIBUTION_ONLY.values():
        _all_ids |= set(_ids)
    _all_ids |= set(_T["reassigned"])
    for _s in sorted(_all_ids):
        _parts = _s.split(".")
        if len(_parts) != 2 or not _parts[1].isdigit():
            bad.append(f"families.ids-canonical {_s!r} is not a canonical spec_id (M<n>.<k>)")
        elif _parts[1] != str(int(_parts[1])):
            bad.append(f"families.ids-canonical {_s!r} has a leading zero; the registry spells "
                       f"it M{_parts[0][1:]}.{int(_parts[1])}, so the two "
                       f"files disagree and the spec looks absent from one")

    # Scope model must reproduce the declared targets. Hand-typed totals drift,
    # and this is the number the paper prints.
    for lane, want in DECLARED_TARGETS.items():
        if lane == "total":
            continue
        got = target(lane)
        if got != want:
            bad.append(f"target({lane}) computed {got} but the file declares "
                       f"{want}: the lane_scope entries and the totals "
                       f"disagree")
    got_total = sum(target(l) for l in lanes())
    if got_total != DECLARED_TARGETS.get("total", got_total):
        bad.append(f"target total computed {got_total} but the file declares "
                   f"{DECLARED_TARGETS['total']}")

    # A lane_scope entry for a family the lane's class is not even in.
    for lane, fams in LANE_SCOPE.items():
        if lane.startswith("$"):
            continue
        for f in fams:
            if f not in _FAM:
                bad.append(f"lane_scope[{lane}] names {f}, which is not a "
                           f"family")
            elif not runs_class(lane, _FAM[f]["class"]):
                bad.append(f"lane_scope[{lane}] marks {f} n/a but the lane "
                           f"does not run {_FAM[f]['class']} at all: the "
                           f"whole-class exclusion belongs in in_scope_lanes")

    # Illegal prohibitions in the scope model.
    for lane, fams in LANE_SCOPE.items():
        if lane.startswith("$") or not isinstance(fams, dict):
            continue
        for f, p in fams.items():
            if p not in PROHIBITIONS:
                bad.append(f"lane_scope[{lane}][{f}]: prohibition {p!r} is not "
                           f"in the closed set {PROHIBITIONS}")

    # Illegal prohibitions on a spec.
    for f in _FAM:
        for s in specs_of(f):
            p = (registry or {}).get(s, {}).get("prohibition", "")
            if p and p not in PROHIBITIONS:
                bad.append(f"{s}: prohibition {p!r} is not in the closed set "
                           f"{PROHIBITIONS}")

    # Stale debt: a ledgered finding that no longer fires means the debt was
    # paid (delete the entry) or the key stopped matching (the ledger is
    # drifting from the code). Either way the ledger is now lying, so it fails.
    if registry is not None and not strict_debt:
        for key in DEBT:
            if key not in debt_keys:
                bad.append(f"known_debt has a STALE entry {key!r}: it no "
                           f"longer fires, so delete it (the debt is paid) or "
                           f"fix the key (the ledger has drifted)")

    return bad, debt


def _report(registry: dict | None) -> None:
    print(f"method-family taxonomy {VERSION}  ({TAXONOMY_PATH})")
    print(f"  {len(_FAM)} families in {len(classes())} classes, "
          f"N = {N_PER_FAMILY} per family per in-scope lane")
    print(f"  view family  {' U '.join(VIEW_MEMBERS)}")
    print()
    for c in classes():
        fs = families_of(c)
        live = [f for f in fs if not is_absent(f)]
        cell = len(live) * N_PER_FAMILY
        absent = len(fs) - len(live)
        extra = f" ({absent} absent, not counted)" if absent else ""
        print(f"  {c}  {len(live)} families, cell {len(live)}x{N_PER_FAMILY}"
              f" = {cell}{extra}")
        for f in fs:
            spec = specs_of(f)
            if registry:
                rf, ra = r_f(f, registry), r_a(f, registry)
                depth = f"R_f={rf} R_a={ra}"
            else:
                depth = f"R_f<={len(spec)}    "
            flag = ""
            if is_view(f):
                flag = " [view]"
            if is_absent(f):
                flag = " [absent: no prohibition, not in the cell]"
            if is_noop_control(f):
                flag = " [noop control]"
            if registry and needs_new_realisation(f, registry) \
                    and not is_absent(f):
                if unrealisable_as_declared(f, registry):
                    flag += " **UNREALISABLE AS DECLARED**"
                elif no_admissible_declaration(f, registry):
                    flag += " **MODEL GAP: NO ADMISSIBLE DECL**"
                else:
                    flag += " **SHORT OF 2**"
            if registry and no_admissible_test(f, registry) \
                    and not is_noop_control(f) and not is_absent(f) \
                    and not no_admissible_declaration(f, registry):
                flag += " **NO ADMISSIBLE TEST**"
            ids = ", ".join(spec) or "(none)"
            print(f"    {f}  {depth:<14} {name(f):<24}{flag}")
            print(f"        {ids}")
        print()
    t = targets()
    print("  targets: " + "  ".join(f"{k} {v}" for k, v in t.items())
          + f"   total {sum(t.values())}")
    print("  measured today: " + "  ".join(
        f"{k} {MEASURED_TODAY.get(k, '?')}" for k in t)
        + f"   total {MEASURED_TODAY.get('total', '?')}")
    print("  delta: " + "  ".join(
        f"{k} {t[k] - MEASURED_TODAY.get(k, 0):+d}" for k in t)
        + f"   total {sum(t.values()) - MEASURED_TODAY.get('total', 0):+d}")
    print()
    print("  families IN SCOPE per lane (this is what the budget multiplies):")
    for l in lanes():
        n_out = 31 - len(families_in_scope(l))
        print(f"    {l:<12} {len(families_in_scope(l)):>2} in scope"
              f"   {n_out:>2} n/a")
    print()
    print(f"  attribution-only (NOT in a class cell): "
          f"{ {k: len(v) for k, v in ATTRIBUTION_ONLY.items()} }")
    print(f"  AVOIDED / never generated: {AVOIDED}")
    print(f"  dead declarations: {len(DEAD)}")
    if registry:
        thin = thin_families(registry)
        zero = {f: v for f, v in thin.items() if v == 0}
        s = summary(registry)
        print(f"  R_f < 2: {len(thin)} families "
              f"({len(thin) - len(zero)} at 1, {len(zero)} at 0)")
        print(f"    MODEL GAP -- every declared spec is inadmissible, so no "
              f"operator could enter the denominator: {sorted(s['model_gap'])}")
        print(f"    unrealisable as declared (every spec on AVOIDED): "
              f"{s['unrealisable_as_declared']}")
        print(f"    inadmissible by construction (noop control, not a work "
              f"item): {s['noop_control']}")
        print(f"    genuinely thin -- write the declared realisation: "
              f"{sorted(s['thin'])}")
        print(f"    has no declaration at all: {s['no_declaration']}")
        # Counted on R_a, not R_f: a realisation that cannot enter the
        # denominator is not a realisation. Counting on R_f is what let M3-d,
        # M3-e and M3-f read as ordinary writing work.
        need_a = sum(max(0, 2 - r_a(f, registry)) for f in _FAM)
        need_f = sum(max(0, 2 - r_f(f, registry)) for f in _FAM)
        print(f"  realisations to write: {need_a} (on R_a, the metric the "
              f"denominator uses) vs {need_f} on R_f")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="validate the partition and exit non-zero on drift")
    ap.add_argument("--with-registry", action="store_true",
                    help="measure R_f against mutations.SPEC_REGISTRY")
    ap.add_argument("--strict-debt", action="store_true",
                    help="treat ledgered coverage debt as a failure too, for "
                         "the release gate")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    registry = declared = None
    if args.with_registry or args.check:
        sys.path.insert(0, os.path.dirname(HERE))
        try:
            from choreo import mutations as _M       # type: ignore
            registry = _M.SPEC_REGISTRY
            declared = list(registry)
        except Exception as e:                        # pragma: no cover
            if args.with_registry:
                print(f"could not import choreo.mutations: {e}", file=sys.stderr)
                return 2

    if not args.quiet:
        _report(registry)

    bad, debt = check(registry, declared, strict_debt=args.strict_debt)
    if debt:
        print(f"\nDEBT {len(debt)} ledgered finding(s) -- real, tracked, "
              f"does not block:")
        for d in debt:
            print(f"  ~ {d}")
        print("  (fix these in W1; `--strict-debt` fails on them for release)")
    if bad:
        print(f"\nFAIL {len(bad)} finding(s):")
        for b in bad:
            print(f"  - {b}")
        return 1
    if not args.quiet:
        print("\ntaxonomy partition sound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
