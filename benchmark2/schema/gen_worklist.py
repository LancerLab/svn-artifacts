#!/usr/bin/env python3
"""Emit the M1-M4 handoff worklist as CSV -- one row per (lane, class, family).

Read-only, like `gen_dashboard.py`: it never writes into a corpus, manifest or
results file. It projects what is on disk into a spreadsheet a worker can own.

The interesting column is `fill_plan`. `N = 8` is not a lump: it decomposes as
`N_KERNELS (4) x N_REALISATIONS (2)`, so a family at 6 is not "3/4 done", it is
missing a whole kernel. `fill_plan` names the kernels to add and how many
instances each needs, in the order that keeps the kernel spread even.

All four classes are in scope. M2 and M3 were absent until now, which made the
worklist read as "M1 and M4 are the work" -- and, worse, hid the fact that M3's
coverage set holds 2 kernels where `N` needs 4, so part of M3's shortfall is not
assignable work at all. `blocker` states that ceiling per row.

    python3 schema/gen_worklist.py [out.csv]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCHEMA = BASE / "schema"
sys.path.insert(0, str(SCHEMA))
sys.path.insert(0, str(BASE))

import method_taxonomy as T  # noqa: E402

try:
    from choreo import mutations as MUT
except Exception:  # pragma: no cover
    MUT = None

# The paper repo is a sibling of svn-artifacts, and this file lives at
# svn-artifacts/benchmark2/schema/gen_worklist.py, so `BASE.parent.parent` is
# svn-artifacts and its parent holds `eurosys27`. Derived rather than hardcoded:
# a colleague who clones this repo on another machine has no /home/garfee/...
# path, and a default that only works on one machine is worse than no default.
# Mirrors the Makefile's `PAPER ?= $(ROOT)/../../eurosys27`.
#
# Only checked when the default is actually used -- an explicit path argument
# never touches this, and importing the module never fails.
PAPER_DEFAULT = BASE.parent.parent / "eurosys27"
OUT_DEFAULT = PAPER_DEFAULT / "plan" / "m1-m4-handoff" / "worklist.csv"

CLASSES = ["M1", "M2", "M3", "M4"]
LANES = ["choreo", "triton", "mlir-low", "mlir-linalg", "iree"]


def load(path: Path, default=None):
    if not path.is_file():
        return default
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    return json.loads(text)


TAX = load(SCHEMA / "method-taxonomy.json", {})
FAM = TAX.get("families", {})
IN_SCOPE = TAX.get("in_scope_lanes", {})
LANE_SCOPE = TAX.get("lane_scope", {})
N = T.N_PER_FAMILY
N_K = T.N_KERNELS
N_R = T.N_REALISATIONS
MINIMAL = MUT.MINIMAL_SET if MUT else {}
SPEC_FAM = {s: f for f, r in FAM.items() for s in r.get("spec_ids", [])}
REGISTRY = MUT.SPEC_REGISTRY if MUT else {}

# The class axis is the SECOND instrument. It states, per lane, whether a class
# is `measured`, `n/a`, `uncompared` or `not_ready`. A class that the method
# taxonomy puts in scope but the class axis holds at `uncompared` is a conflict
# between the two instruments, not a shortfall -- report it as such.
AX = load(SCHEMA / "class-axis.json", {})
AX_STATUS = {lane: dict(v.get("status", {}))
             for lane, v in (AX.get("lanes") or {}).items()}
AX_UNCMP = AX.get("uncompared_reason", "")


def generatable(sid):
    m = REGISTRY.get(sid)
    return bool(m) and (m.get("status") == "implemented"
                        and m.get("path") != "P2"
                        and m.get("prohibition") != "repaired")


def spec_health(family):
    specs = FAM.get(family, {}).get("spec_ids", [])
    rf = sum(1 for s in specs if generatable(s))
    ra = sum(1 for s in specs if generatable(s)
             and REGISTRY.get(s, {}).get("admissible"))
    return len(specs), rf, ra


def missing_surface():
    """spec_id -> the MISSING SURFACE sentence, for the blocker column."""
    out = {}
    for sid, r in REGISTRY.items():
        n = r.get("note", "") or ""
        if r.get("status") != "implemented" and "MISSING SURFACE" in n:
            out[sid] = n.split("MISSING SURFACE:")[-1].strip()
    return out


MSURF = missing_surface()

# ------------------------------------------------------------------ corpora

# choreo -- the only lane with a mutant_manifest.json, and therefore the only
# lane whose per-family depth the `m{1,4}.<family>.instances` guards enforce.
choreo_manifest = load(BASE / "choreo" / "raw" / "mutant_manifest.json", {})
depth = defaultdict(Counter)
for m in choreo_manifest.get("mutants", []):
    f = SPEC_FAM.get(m.get("spec_id"))
    if f:
        depth[f][m.get("category")] += 1
choreo_fam = Counter({f: sum(c.values()) for f, c in depth.items()})

# every other lane -- instance counts by (class, family) where the rows carry
# spec_id, else by class only. Rows are deduplicated the way the dashboard
# does, because the mlir lanes run each mutant twice (rtv off/on).
lane_cls = {L: Counter() for L in LANES}
lane_fam = {L: Counter() for L in LANES}
lane_rows = {L: 0 for L in LANES}
for L in LANES:
    if L == "choreo":
        lane_cls[L] = Counter()
        for f, n in choreo_fam.items():
            lane_cls[L][T._FAM[f]["class"]] += n
        lane_fam[L] = choreo_fam
        lane_rows[L] = len(choreo_manifest.get("mutants", []))
        continue
    rows = load(BASE / L / "raw" / "mutants.jsonl", [])
    lane_rows[L] = len(rows)
    seen = set()
    for r in rows:
        key = (r.get("spec_id"), r.get("category"), r.get("shape"), r.get("kernel"))
        if key in seen:
            continue
        seen.add(key)
        f = SPEC_FAM.get(r.get("spec_id"))
        if f:
            lane_fam[L][f] += 1
            lane_cls[L][T._FAM[f]["class"]] += 1
        else:
            # No spec_id: the row can be attributed to a class but not a
            # family. Record it so the shortfall is not silently overstated.
            lane_cls[L][r.get("class") or "?"] += 1


# ------------------------------------------------------------------ fill plan


def fill_plan(cls, have: Counter, total: int) -> str:
    """Which kernels to add, in the order that evens the spread.

    `N = N_KERNELS x N_REALISATIONS`: every kernel should reach N_R instances
    before a new kernel is started, and a family needs N_KERNELS kernels. So
    fill the present-but-thin kernels first, then open new ones.
    """
    if total >= N:
        return ""
    want = N
    order = list(MINIMAL.get(cls, []))
    steps = []
    cur = dict(have)
    # Present-but-thin kernels first, then new ones. The sort key is
    # `(is_new, count)`, not `count`: a kernel the family does not use has
    # count 0, and a plain ascending sort puts it ahead of a kernel sitting at
    # 1 of 2 -- starting a new kernel while a thin one is still open, which is
    # the exact move this function exists to avoid. It also produced plans that
    # were one kernel less even than the family could afford: M2-a topped up to
    # `conv2d+1` when `relu+1` closes it to a perfectly flat `2,2,2,2`.
    for k in sorted(order, key=lambda k: (0 if cur.get(k, 0) else 1,
                                          cur.get(k, 0))):
        if sum(cur.values()) >= want:
            break
        c = cur.get(k, 0)
        if c >= N_R:
            continue
        add = min(N_R - c, want - sum(cur.values()))
        if add <= 0:
            continue
        steps.append("%s+%d" % (k, add))
        cur[k] = c + add
    return " ".join(steps)


def family_ceiling(cls: str, fam: str) -> int:
    """The most instances this family can hold, measured from its candidates.

    `N` is a ceiling *and* a decomposition, and three caps stack under it. Only
    the first is a budget; the other two are properties of the declaration set
    and of the candidate table, so no re-run of `select()` can lift them:

      * `n_realisations` (2) -- a family holds at most 2 instances per category;
      * the CELL ceiling -- a `(spec_id, category)` cell holds one instance, or
        two when that spec is P1, because there the curve is the point;
      * the candidate table -- a cell cannot supply more instances than it has
        candidates, so a cell with one incumbent supplies one, not two.

    `select()` takes exactly this maximum. For all 31 families the measured
    corpus equals this number, so wherever `ceiling == have` the supply is
    exhausted and only a NEW operator moves the count -- `fill_plan` names the
    kernel to add to, but it is an ideal spread and does not promise a
    candidate exists. Read the two together: `have 4 / ceiling 4 / plan
    matmul+1` means "the shape wants matmul, and there is no matmul left".
    """
    if MUT is None:
        return 0
    cells = defaultdict(lambda: defaultdict(int))   # category -> spec -> count
    per = {}                                        # (category, spec) -> 2|1
    for c in MUT.ALL.get(cls, []):
        if MUT.family_of(c.spec_id) != fam:
            continue
        cells[c.category][c.spec_id] += 1
        per[(c.category, c.spec_id)] = N_R if c.needs_rtc_curve else 1
    total = 0
    for cat, by_spec in cells.items():
        supply = sum(min(per[(cat, sid)], n) for sid, n in by_spec.items())
        total += min(N_R, supply)
    return min(N, total)


def kernel_ceiling(cls) -> int:
    """The most instances a family of `cls` can hold, given its coverage set.

    `N` is not a lump: it decomposes as `N_KERNELS (4) x N_REALISATIONS (2)`,
    and `select()` caps every `(family, category)` pair at `N_REALISATIONS`. So
    a class whose coverage set holds fewer than `N_KERNELS` kernels cannot reach
    `N` however many operators are written -- the ceiling is a property of the
    COVERAGE SET, not of the corpus, and no amount of work on operators lifts
    it. M3 is the live case: `MINIMAL_SET["M3"]` is `[matmul, conv2d]`, so its
    per-family ceiling is 4 and its class ceiling is 32, not 64.

    Saying this in the `blocker` column is the point. Without it a `short` of
    50 reads as 50 assignable instances, and the worker finds the wall instead
    of being told about it.
    """
    return len(MINIMAL.get(cls, [])) * N_R


def blocker(cls, family, rf, ra):
    parts = []
    if kernel_ceiling(cls) < N:
        order = list(MINIMAL.get(cls, []))
        parts.append(
            "CEILING: %s's coverage set holds %d kernel(s) (%s), but N=%d "
            "decomposes as %d kernels x %d realisations, and select() caps "
            "every (family, kernel) at %d. So this family cannot exceed %d "
            "and %s's class cell cannot exceed %d, however many operators are "
            "written -- the ceiling is in the COVERAGE SET, not the corpus. "
            "Closing this row needs MINIMAL_SET[%s] widened (mutations.py) or "
            "an accepted, stated deviation. Do NOT report %d."
            % (cls, len(order), ", ".join(order), N, N_K, N_R, N_R,
               kernel_ceiling(cls), cls,
               len([f for f in FAM if FAM[f]["class"] == cls])
               * kernel_ceiling(cls), cls, N))
    if rf == 0:
        parts.append("R_f=0: no generatable spec_id. Needs a NEW spec, not "
                     "operators. Design call.")
    elif ra == 0:
        parts.append("R_a=0: declarations exist but none is admissible, so "
                     "nothing can enter the denominator. Design call.")
    if cls == "M1" and family in ("M1-d", "M1-g"):
        gaps = [s for s, why in MSURF.items()
                if SPEC_FAM.get(s) == family and s not in FAM.get(family, {})
                .get("spec_ids", [])]
        pend = [s for s in FAM.get(family, {}).get("spec_ids", [])
                if REGISTRY.get(s, {}).get("status") != "implemented"]
        if pend:
            parts.append("pending spec(s) %s blocked on MISSING SURFACE: %s"
                         % (", ".join(pend),
                            MSURF.get(pend[0], "see registry note")))
    if not parts:
        # Nothing structural stands in the way: the row is short on operators
        # and `fill_plan` says which kernels they go on. Say so, so an empty
        # blocker is never mistaken for "no plan".
        bad = [s for s in FAM.get(family, {}).get("spec_ids", [])
               if not generatable(s)]
        if bad:
            parts.append("declared spec(s) %s have no generatable operator, "
                         "so the family's ceiling is below N until they do."
                         % ", ".join(bad))
    return " ".join(parts)


rows = []
for cls in CLASSES:
    fams = sorted(f for f in FAM if FAM[f]["class"] == cls)
    for lane in LANES:
        ax = (AX_STATUS.get(lane, {}) or {}).get(cls)
        # The lane has instances of this class but no family can be assigned,
        # because its rows carry no spec_id. Reporting 0 for every family would
        # understate the lane's real position, so say so instead.
        unattributed = (lane not in ("choreo",) and not lane_fam[lane]
                        and lane_cls[lane].get(cls, 0) > 0)
        for f in fams:
            if not T.runs_class(lane, cls):
                state = "out-of-scope"
            elif ax == "uncompared":
                # in the taxonomy's scope, absent from the axis. A conflict.
                state = "UNCOMPARED-conflict"
            elif unattributed:
                state = "unattributed(no spec_id)"
            elif LANE_SCOPE.get(lane, {}).get(f):
                state = "carved-out(%s)" % LANE_SCOPE[lane][f]
            else:
                state = "in-scope"

            nspec, rf, ra = spec_health(f)
            have = lane_fam[lane].get(f, 0)
            if state != "in-scope":
                short = 0
            else:
                short = max(0, N - have)
            rows.append({
                "lane": lane,
                "class": cls,
                "family": f,
                "family_name": FAM[f].get("name", ""),
                "state": state,
                "specs": nspec,
                "R_f": rf,
                "R_a": ra,
                "have": have,
                "need": N if state == "in-scope" else 0,
                "short": short,
                # The supply, not the obligation. `short` is `N - have`; this
                # is what the family can actually reach today. When the two
                # differ the row is corpus work, not a CPU run.
                "ceiling": (family_ceiling(cls, f) if lane == "choreo" else ""),
                "unattributed_rows": (
                    "" if state == "in-scope" and lane not in lane_fam
                    else ""),
                "fill_plan": fill_plan(cls, depth.get(f, {}), have)
                             if lane == "choreo" else "",
                "blocker": (blocker(cls, f, rf, ra) if short else "") or
                           ("in_scope_lanes[%s] lists this lane but "
                            "class-axis holds it at `uncompared` (declared "
                            "scope decision, not a gap). Resolve the two "
                            "instruments before assigning work."
                            % cls if state == "UNCOMPARED-conflict" else
                            "lane has %d %s instance(s) but its raw rows carry "
                            "no spec_id, so no family can be assigned. First "
                            "task: emit spec_id (and family) on every row, "
                            "then re-derive this column."
                            % (lane_cls[lane].get(cls, 0), cls)
                            if state.startswith("unattributed") else ""),
                "guarded": "yes" if (lane == "choreo" and state == "in-scope")
                           else "no",
            })


def main():
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        if not PAPER_DEFAULT.is_dir():
            raise SystemExit(
                "cannot find the paper repo: %s does not exist.\n"
                "This script writes its CSV into the paper tree, so it needs "
                "the paper repo checked out as a sibling of svn-artifacts.\n"
                "Pass an explicit path instead:\n"
                "  python3 schema/gen_worklist.py /path/to/worklist.csv"
                % PAPER_DEFAULT)
        out = OUT_DEFAULT
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["lane", "class", "family", "family_name", "state", "specs", "R_f",
            "R_a", "have", "ceiling", "need", "short", "fill_plan", "blocker",
            "guarded"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    tot = defaultdict(int)
    conflict = defaultdict(int)
    unattr = defaultdict(int)
    for r in rows:
        if r["state"] == "in-scope":
            tot[r["lane"]] += r["short"]
        elif r["state"] == "UNCOMPARED-conflict":
            conflict[r["lane"]] += 1
        elif r["state"].startswith("unattributed"):
            unattr[r["lane"]] += 1
    print("wrote %s (%d rows)" % (out, len(rows)))
    print("in-scope shortfall: " + ", ".join(
        "%s=%d" % (l, tot[l]) for l in LANES if tot[l]) +
        "  TOTAL=%d" % sum(tot.values()))
    if conflict:
        print("taxonomy-in-scope but class-axis `uncompared` (CONFLICT, not "
              "work): " +
              ", ".join("%s=%d families" % (l, n)
                        for l, n in conflict.items()))
    if unattr:
        print("BLOCKED, cannot be assigned: no spec_id on the lane's rows: " +
              ", ".join("%s=%d families" % (l, n)
                        for l, n in unattr.items()))


if __name__ == "__main__":
    main()
