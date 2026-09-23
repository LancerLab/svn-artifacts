#!/usr/bin/env python3
"""Fill dashboard -- planned vs implemented, per (lane, class, family).

The class x lane dashboard (`gen_dashboard.py`) answers "which classes are
behind".  This one answers the next question down: **which family, in which
lane, is short, and by how much** -- the view needed to decide what to write
next.

It is a read-only projection of `gen_worklist.py`'s output, so it carries that
generator's guarantees: nothing here is hand-maintained, and re-running it
after any corpus change refreshes every number.

Usage:
    python3 schema/gen_fill_dashboard.py [worklist.csv] [out.md]

    # or, from benchmark2/
    make fill-dashboard
"""
import csv
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_dashboard as GD  # noqa: E402  the scope model, not a second copy

LANES = GD.LANES
CLASSES = GD.CLASSES

# The state vocabulary is `gen_worklist.py`'s; see its header comment.
BLOCKED = "unattributed(no spec_id)"
CONFLICT = "UNCOMPARED-conflict"
OUT = "out-of-scope"
NO_DATA = "no-data"

# `method-taxonomy.target()`'s per-family budget. Not a tunable: it is
# `kernel_component x realisation_component` = 4 x 2, and lowering it here would
# only hide the kernel-supply shortfall the dashboard exists to show.
N = 8


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(row, key):
    """Unassignable rows (`unattributed`, `UNCOMPARED-conflict`) leave the
    counting columns empty rather than zero -- there is no measurement to
    report, and zero would be read as one. Treat as 0 for arithmetic."""
    v = (row.get(key) or "").strip()
    return int(v) if v else 0


def published(row, key):
    """True iff the column carries a measurement at all.

    Blank and `0` are different facts, and the difference decides whether a
    family reads as `saturated` or `unwritten`. A blank `ceiling` means nobody
    measured one; a zero means it was measured and is zero."""
    return bool((row.get(key) or "").strip())


def fams_all(rows):
    """Every family the taxonomy declares -- the plan's domain.

    Deliberately not `{r['family'] for r in rows}`: a family missing from the
    worklist because *no* lane has a row for it must still appear, or its
    absence reads as "nothing to do".
    """
    return list(GD.FAMILIES)


def in_scope(row):
    st = row["state"]
    return (st != OUT and not st.startswith("carved-out")
            and not st.startswith(NO_DATA))


def resolves_family(rows):
    """True iff this lane's rows carry a `spec_id`, and so a family.

    The test is `gen_worklist.py`'s own attribution state, NOT a shape test on
    the counts. An earlier version asked "is every `have` <= `need`?" and
    inferred "no family axis" whenever a count exceeded N. That was wrong twice
    over: `mlir-linalg` and `mlir-low` DO emit `spec_id`
    (`mlir-linalg-layer_normalization-static-M2.1-off`), and a family at 40
    against N=8 is *over-delivery*, not the absence of an axis. The heuristic
    hid 48 unwritten `mlir-linalg` instances behind a label that said they were
    not countable.

    `no-data` also fails the test: with the lane's raw file absent there is no
    instance list to resolve, so the lane has no family axis *here* even though
    it would have one on a checkout that carries its raw results.
    """
    return not any(r["state"].startswith(("unattributed", NO_DATA))
                   for r in rows)


def cell(row, resolves, plan):
    """One (lane x family) cell: implemented / planned, marked by state.

    `plan` comes from the shared scope model, never from the row: a lane's own
    rows cannot be trusted to know what the lane is supposed to cover, which is
    exactly how the `spec_id` and `uncompared` defects stayed hidden.
    """
    if not plan:
        return "out"
    if row is None:
        return "-"
    st = row["state"]
    if st == OUT or st.startswith("carved-out"):
        return "out"
    if st.startswith(NO_DATA):
        return "no data"
    if st.startswith(BLOCKED.split("(")[0]):
        return "blocked"
    if st == CONFLICT:
        return "conflict"
    if not resolves:
        return "n/a"
    have, need = num(row, "have"), plan
    if have >= need:
        # `over` is not a warning: N is a decomposition, not a maximum, so a
        # lane may legitimately land more instances on one family than the
        # budget names -- but it is never the same fact as "ok", and it is
        # never a reason to leave a sibling family at zero.
        return f"**{have}**/{need} over" if have > need else f"**{have}**/{need} ok"
    if not published(row, "ceiling"):
        # Nothing was measured, so nothing may be claimed about the supply.
        # Reporting `saturated` here was a real defect in this generator: a
        # blank `ceiling` compares as 0, so `have 0 >= ceiling 0` printed
        # `**0**/8 saturated` for `mlir-linalg` M2-b, i.e. it dressed six
        # families of unwritten code as a measured kernel-supply limit.
        return f"**{have}**/{need} unwritten"
    if have >= num(row, "ceiling"):
        return f"**{have}**/{need} saturated"
    return f"**{have}**/{need} short"


def main(argv):
    # <svn>/svn-artifacts/benchmark2/schema/this.py -> <svn>/eurosys27/plan/...
    handoff = (Path(__file__).resolve().parents[2].parent
               / "eurosys27" / "plan" / "m1-m4-handoff")
    src = Path(argv[1]) if len(argv) > 1 else handoff / "worklist.csv"
    dst = Path(argv[2]) if len(argv) > 2 else handoff / "FILL-DASHBOARD.md"
    rows = read(src)
    ix = {(r["lane"], r["class"], r["family"]): r for r in rows}

    fams = {}
    names = {}
    for r in rows:
        fams.setdefault(r["class"], [])
        if r["family"] not in fams[r["class"]]:
            fams[r["class"]].append(r["family"])
            names[r["family"]] = r["family_name"]
        if r["family_name"]:
            names[r["family"]] = r["family_name"]
    for c in fams:
        fams[c].sort()

    resolves = {}
    no_data = {}
    for lane in LANES:
        lane_rows = [r for r in rows if r["lane"] == lane]
        resolves[lane] = resolves_family(lane_rows)
        no_data[lane] = any(r["state"].startswith(NO_DATA) for r in lane_rows)

    # The plan comes from the taxonomy's scope model (`gen_dashboard`), not from
    # the worklist rows. Summing `need` over the rows that happen to exist would
    # reproduce the very defect this table is meant to expose: a lane whose rows
    # are unattributable would report a plan of zero, i.e. "nothing missing".
    plan = {(lane, fam): GD.planned(lane, fam)
            for lane in LANES for fam in fams_all(rows)}
    lane_plan = {lane: GD.model_total(lane) for lane in LANES}

    out = []
    out.append("# Fill dashboard -- implemented / planned, per family")
    out.append("")
    out.append(f"_Generated {date.today().isoformat()} by "
               "`schema/gen_fill_dashboard.py` from `worklist.csv`. Read-only "
               "projection; **do not hand-edit** -- re-run `make "
               "fill-dashboard`._")
    out.append("")
    out.append("The cell vocabulary, and nothing else is implied by it:")
    out.append("")
    out.append("| cell | meaning |")
    out.append("|---|---|")
    out.append("| **n**/8 `ok` | family is at its declared cell of 8 |")
    out.append("| **n**/8 `saturated` | at the **measured** ceiling for this "
               "lane; only a new *kernel surface* can raise it, never another "
               "operator |")
    out.append("| **n**/8 `short` | an operator could still be written |")
    out.append("| `blocked` | the lane's rows carry no `spec_id`, so no family "
               "can be attributed and there is no plan to compare against |")
    out.append("| `conflict` | in the taxonomy's scope but `uncompared` on the "
               "class axis -- an instrument disagreement, **not** a gap |")
    out.append("| `n/a` | the lane does not resolve its instances to a family "
               "(class-first composition) -- see section 5 |")
    out.append("| `no data` | the lane's raw results are absent from this "
               "checkout (it is gitignored), so its numbers are unknown here, "
               "**not** zero |")
    out.append("| `out` | the lane does not run this class |")
    out.append("")
    out.append(f"`planned` is `N_per_family = {N}` wherever the family is in a "
               "lane's scope. A family is never a flat count of eight: it is "
               "4 kernels x 2 realisations, which is why `saturated` is a "
               "distinct state from `short`.")
    out.append("")

    # ---------------------------------------------------------------- summary
    out.append("## 1. By lane")
    out.append("")
    out.append("| lane | families in scope | planned | implemented | short | "
               "saturated | family axis |")
    out.append("|---|---|---|---|---|---|---|")
    tot_p = tot_h = tot_s = 0
    for lane in LANES:
        rs = [r for r in rows if r["lane"] == lane and in_scope(r)]
        n_fam = sum(1 for f in fams_all(rows) if plan[(lane, f)])
        h = sum(num(r, "have") for r in rs)
        s = sum(1 for r in rs if resolves[lane]
                and num(r, "have") >= num(r, "ceiling")
                and num(r, "have") < plan[(lane, r["family"])])
        axis = ("**no data**" if no_data[lane]
                else "yes" if resolves[lane] else "**no**")
        p = lane_plan[lane]
        out.append(f"| `{lane}` | {n_fam} | {p} | **{h}** | {p - h} | {s} | {axis} |")
        tot_p += p
        tot_h += h
        tot_s += s
    out.append(f"| **all** | | **{tot_p}** | **{tot_h}** | **{tot_p - tot_h}** | "
               f"{tot_s} | |")
    out.append("")
    out.append("`planned` is the scope model's own answer (`method_taxonomy`), "
               "the same number `DASHBOARD.md` shows as a lane's target, so the "
               "two documents cannot disagree. `implemented` counts instances "
               "in the same unit, so the two are subtractable. It is not "
               "`DASHBOARD.md`'s `rows` column: the mlir lanes run each mutant "
               "twice (rtv `off`/`on`) and report two rows per instance.")
    out.append("")

    # ------------------------------------------------------------ per family
    for cls in CLASSES:
        out.append(f"## 2.{cls[-1]} `{cls}` -- by family")
        out.append("")
        out.append("| family | kind | " + " | ".join(f"`{l}`" for l in LANES) + " |")
        out.append("|---|---|" + "---|" * len(LANES))
        for fam in fams.get(cls, []):
            cs = [cell(ix.get((lane, cls, fam)), resolves[lane], plan[(lane, fam)])
                  for lane in LANES]
            if all(c == "out" for c in cs):
                continue
            out.append(f"| `{fam}` | {names.get(fam, '')} | " + " | ".join(cs) + " |")
        out.append("")

    # ----------------------------------------------------------- next actions
    out.append("## 3. Short families where an operator would actually help")
    out.append("")
    out.append("Two cases qualify, and they are different jobs. `unwritten` "
               "means no ceiling was ever measured because the spec has no "
               "mutation code -- the fix is to write that code. `short` means a "
               "ceiling exists and the corpus is below it -- the fix is an "
               "operator whose anchor reaches an unused kernel. `saturated` "
               "families are excluded: their supply is exhausted and only a new "
               "kernel surface moves them (section 4).")
    out.append("")
    help_ = [r for r in rows
             if r["state"] == "in-scope" and resolves[r["lane"]]
             and num(r, "have") < plan[(r["lane"], r["family"])]
             and (not published(r, "ceiling")
                  or num(r, "have") < num(r, "ceiling"))]
    help_.sort(key=lambda r: (r["lane"], r["class"], r["family"]))
    if not help_:
        out.append("**None.** No family has unwritten mutation code, and every "
                   "written family is at its measured ceiling. The remaining gap "
                   "is a kernel-suite question, not a corpus one.")
    else:
        out.append("| lane | class | family | have | ceiling | planned | "
                   "missing | work |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in help_:
            pl = plan[(r["lane"], r["family"])]
            ceil = (r["ceiling"] if published(r, "ceiling") else "--")
            kind = ("write the spec's mutation code"
                    if not published(r, "ceiling")
                    else "operator with an anchor on +%d kernel(s)"
                         % (num(r, "ceiling") - num(r, "have")))
            out.append(f"| `{r['lane']}` | {r['class']} | `{r['family']}` | "
                       f"{r['have']} | {ceil} | {pl} | "
                       f"**{pl - num(r, 'have')}** | {kind} |")
    out.append("")

    out.append("## 4. Families at their measured ceiling (kernel-supply bound)")
    out.append("")
    sat = [r for r in rows
           if r["state"] == "in-scope" and resolves[r["lane"]]
           and published(r, "ceiling")
           and num(r, "have") >= num(r, "ceiling")
           and num(r, "have") < plan[(r["lane"], r["family"])]]
    sat.sort(key=lambda r: (r["lane"], r["class"], r["family"]))
    if not sat:
        out.append("None.")
    else:
        out.append(f"{len(sat)} families. Each is at a **published ceiling** "
                   "(`have >= ceiling`), so no operator lifts it. For `choreo` "
                   "the ceiling is its candidate table -- only a new kernel "
                   "surface moves it. For other lanes it is a lane-owned "
                   "expressibility declaration (`schema/lane-ceilings.json`), "
                   "and the `blocker` names the missing surface. Either way the "
                   "fix is a new instrument, not another operator. The `kind` "
                   "column names *why* the capped slots are not writable: `u` "
                   "= unexpressible (no legal program states the defect), `a` "
                   "= avoided (the lane's model repairs or refuses it), or "
                   "`unchecked` = a legal program exists that the lane accepts "
                   "silently -- a **created** mutant, not a `u`.")
        out.append("")
        out.append("| lane | class | family | kind | have | ceiling | planned | short | blocker |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        KIND_ABBR = {"unexpressible": "u", "avoided": "a", "unchecked": "unchecked"}
        for r in sat:
            pl = plan[(r["lane"], r["family"])]
            kind = (r.get("ceiling_kind") or "").strip() or "--"
            kind = KIND_ABBR.get(kind, kind)
            out.append(f"| `{r['lane']}` | {r['class']} | `{r['family']}` | "
                       f"`{kind}` | {r['have']} | {r['ceiling']} | {pl} | "
                       f"{pl - num(r, 'have')} | {r['blocker']} |")
    out.append("")

    # ------------------------------------------------- no family axis lanes
    out.append("## 5. Lanes whose instances cannot be attributed")
    out.append("")
    out.append("A lane lands here when its records carry a `class` but no "
               "`spec_id`, so its instances can be counted per class and never "
               "per family, or when it gitignores its raw results so a fresh "
               "clone has no instance list at all. In both cases `implemented` "
               "above is not a measurement -- it is 0 because nothing could be "
               "attributed -- and the instances that do exist are named here so "
               "the shortfall is not overstated.")
    out.append("")
    out.append("| lane | why | planned | implemented | discarded instances | "
               "the fix |")
    out.append("|---|---|---|---|---|---|")
    why = {}
    fix = {}
    unattributed_lanes = []
    for lane in LANES:
        if resolves[lane]:
            continue
        unattributed_lanes.append(lane)
        if no_data[lane]:
            out.append(
                f"| `{lane}` | its raw results file is absent from this "
                f"checkout (gitignored, hundreds of MB), so there is no "
                f"instance list to attribute. | {lane_plan[lane]} | **n/a** | "
                f"**n/a** | re-run the lane on a checkout that has the raw "
                f"results, or commit a census the way `mlir-low` and "
                f"`mlir-linalg` do. |")
            continue
        rs = [r for r in rows if r["lane"] == lane and in_scope(r)]
        if not rs:
            continue
        # The rows carry the count of instances that exist but could not be
        # attributed: "lane has N <class> instance(s) but its raw rows carry no
        # spec_id". Surface it, because `implemented 0` otherwise reads as "this
        # lane ran nothing", when it in fact ran N per class. The sentence
        # repeats on every family row of a class, so count each class once --
        # summing the rows reported 72 for a lane that ran 9.
        disc = 0
        seen_cls = set()
        for r in rs:
            m = re.search(r"lane has (\d+) (\w+) instance", r.get("blocker", ""))
            if m and m.group(2) not in seen_cls:
                seen_cls.add(m.group(2))
                disc += int(m.group(1))
        out.append(f"| `{lane}` | {why.get(lane, '')} | {lane_plan[lane]} | "
                   f"**{sum(num(r, 'have') for r in rs)}** | **{disc}** | "
                   f"{fix.get(lane, '')} |")
    out.append("")
    if not unattributed_lanes:
        out.append("No lane's instances are unattributable in this checkout: "
                   "every lane that runs a class in scope resolves its rows to "
                   "a family, so section 3's short columns are real operator "
                   "work and not an instrument artifact.")
    else:
        names_s = ", ".join(f"`{l}`" for l in unattributed_lanes)
        out.append(f"`implemented 0` in these rows is not a dead lane. The "
                   f"lane(s) {names_s} do not resolve instances to a family "
                   f"(no `spec_id`, or no instance list in this checkout). Fix "
                   f"the instrument first -- but do not read the result as a "
                   f"smaller job. Once attributed, whatever families the surface "
                   f"actually covers will show up as short, and the rest as "
                   f"genuinely unwritten.")
    out.append("")

    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
