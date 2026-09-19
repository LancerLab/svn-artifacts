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


def fams_all(rows):
    """Every family the taxonomy declares -- the plan's domain.

    Deliberately not `{r['family'] for r in rows}`: a family missing from the
    worklist because *no* lane has a row for it (M4-g) must still appear, or its
    absence reads as "nothing to do".
    """
    return list(GD.FAMILIES)


def in_scope(row):
    st = row["state"]
    return st != OUT and not st.startswith("carved-out")


def resolves_family(rows):
    """True iff this lane's rows are a family partition.

    `choreo` composes family-first, so every instance lands in a family and the
    per-family count is bounded by N.  The mlir lanes compose class-first, so
    `have` can exceed N in one family and be 0 in its siblings -- that is a
    missing axis, not over-delivery.
    """
    if not any(num(r, "need") for r in rows):
        return False
    return all(num(r, "have") <= num(r, "need") for r in rows)


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
    if st.startswith(BLOCKED.split("(")[0]):
        return "blocked"
    if st == CONFLICT:
        return "conflict"
    if not resolves:
        return "n/a"
    have, need, ceil = num(row, "have"), plan, num(row, "ceiling")
    if have >= need:
        return f"**{have}**/{need} ok"
    if have >= ceil:
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
    for lane in LANES:
        resolves[lane] = resolves_family([r for r in rows
                                          if r["lane"] == lane and in_scope(r)])

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
        axis = "yes" if resolves[lane] else "**no**"
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
    help_ = [r for r in rows
             if r["state"] == "in-scope" and resolves[r["lane"]]
             and num(r, "have") < num(r, "ceiling")
             and num(r, "have") < plan[(r["lane"], r["family"])]]
    help_.sort(key=lambda r: (-(num(r, "ceiling") - num(r, "have")), r["lane"], r["family"]))
    if not help_:
        out.append("**None.** Every in-scope family is at its measured ceiling. "
                   "No further operator can raise any lane; the remaining gap is "
                   "a kernel-suite question, not a corpus one.")
    else:
        out.append("| lane | class | family | have | ceiling | planned | headroom |")
        out.append("|---|---|---|---|---|---|---|")
        for r in help_:
            pl = plan[(r["lane"], r["family"])]
            out.append(f"| `{r['lane']}` | {r['class']} | `{r['family']}` | "
                       f"{r['have']} | {r['ceiling']} | {pl} | "
                       f"**{num(r, 'ceiling') - num(r, 'have')}** |")
    out.append("")

    out.append("## 4. Families at their measured ceiling (kernel-supply bound)")
    out.append("")
    sat = [r for r in rows
           if r["state"] == "in-scope" and resolves[r["lane"]]
           and num(r, "have") >= num(r, "ceiling")
           and num(r, "have") < plan[(r["lane"], r["family"])]]
    sat.sort(key=lambda r: (r["lane"], r["class"], r["family"]))
    if not sat:
        out.append("None.")
    else:
        out.append(f"{len(sat)} families. Each needs a **new kernel surface**, "
                   "not another operator: the literal anchor its specs rewrite "
                   "does not occur on enough base kernels for "
                   "`kernel_component x realisation_component` to be reachable.")
        out.append("")
        out.append("| lane | class | family | have | ceiling | planned | short | blocker |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in sat:
            pl = plan[(r["lane"], r["family"])]
            out.append(f"| `{r['lane']}` | {r['class']} | `{r['family']}` | "
                       f"{r['have']} | {r['ceiling']} | {pl} | "
                       f"{pl - num(r, 'have')} | {r['blocker']} |")
    out.append("")

    # ------------------------------------------------- no family axis lanes
    out.append("## 5. Lanes with no family axis")
    out.append("")
    out.append("These lanes cannot be scored per family, for two different "
               "reasons. Neither is improved by writing operators.")
    out.append("")
    out.append("| lane | why | planned | implemented | the fix |")
    out.append("|---|---|---|---|---|")
    why = {
        "triton": "family-first in principle, but its rows carry no `spec_id`, "
                  "so nothing can be attributed to a family.",
        "iree": "same `spec_id` blocker as `triton`.",
        "mlir-low": "class-first composition: its M1 work lands on 4 "
                    "categories, not on M1's 8 families.",
        "mlir-linalg": "class-first composition, as `mlir-low`.",
    }
    fix = {
        "triton": "emit `spec_id` in the triton collector rows",
        "iree": "emit `spec_id` in the iree collector rows",
        "mlir-low": "add the missing M1 mutation kinds to "
                    "`mlir-shared/mutate.py`, then distribute by family",
        "mlir-linalg": "add the missing M2 mutation kinds to "
                       "`mlir-shared/mutate.py`, then distribute by family",
    }
    for lane in LANES:
        rs = [r for r in rows if r["lane"] == lane and in_scope(r)]
        if not rs or resolves[lane]:
            continue
        out.append(f"| `{lane}` | {why.get(lane, '')} | {lane_plan[lane]} | "
                   f"**{sum(num(r, 'have') for r in rs)}** | "
                   f"{fix.get(lane, '')} |")
        if all((r["state"].startswith(BLOCKED.split("(")[0])
                or r["state"] == CONFLICT) for r in rs):
            out.append("")
            out.append(f"> `{lane}`'s shortfall is **not work**. Its rows and "
                       "the class-axis taxonomy disagree about what the lane "
                       "covers, or the rows are unattributable. Resolve the "
                       "instrument before assigning anything.")
    out.append("")

    dst.write_text("\n".join(out) + "\n")
    print(f"wrote {dst} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
