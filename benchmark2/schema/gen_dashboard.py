#!/usr/bin/env python3
"""Generate DASHBOARD.md -- the single progress view over the mutation benchmark.

READ ONLY. Never writes into a corpus, manifest or results file. Safe to re-run
at any time; the dashboard is a projection of what is on disk today.

Sources (all under svn-artifacts/):
  benchmark2/schema/method-taxonomy.json      families, budget, targets, lane scope, debt
  benchmark2/schema/class-axis.json           classes, coverage, per-lane status
  benchmark2/choreo/mutations.py              SPEC_REGISTRY + ALL (operator inventory)
  benchmark2/choreo/raw/mutant_manifest.json  choreo materialised instances
  benchmark2/{triton,iree}/raw/mutants.jsonl  per-lane instances + results
  benchmark2/{mlir-low,mlir-linalg}/raw/minimal-census.json   injected vs target
  benchmark2/schema/dashboard-log.md          dated progress entries (hand-maintained)

ONE COMPOSITION MODEL, for every lane. `N_per_family = 8`, so
  planned(lane, class) = 8 x (families of that class in scope for that lane)
which is `method_taxonomy.target()` -- check-enforced against `target` below.
A lane's class cell differs from the declared cell for exactly two reasons:
  (1) the class's family count -- M4 declares 7 families, so its cell is 56
      in every lane, not 64;
  (2) that lane's `lane_scope` carve-outs -- each removes one family, worth
      exactly 8. choreo has none; triton drops M3-b/M3-g (M3 64 -> 48);
      iree drops M2-e/f/g/h (M2 64 -> 32).
The five lanes sum to `targets.total` exactly (choreo 64+64+64+56 = 248,
triton 64+48+56 = 168, mlir-low 64+56 = 120, mlir-linalg 64+56 = 120,
iree 32+56 = 88; total 744). Reproduce that sum and the model is right.

SEPARATE AND WEAKER: `class-axis.json` `n_target_per_class = 40`. That is the
mlir collectors' own internal per-class gate, NOT a budget -- mlir-low clears it
(48 >= 40) while still sitting under its M1 cell of 64. Never use it as a
coverage denominator. The mlir lanes are additionally not distributed across
their class's 8 families (mlir-low's M1 work lands on 4 categories), which is
why section 4 prints them as raw counts rather than /8.

THREE STAGES, never interchangeable:
  TARGET     declared.   What the plan promises.
  MATERIAL   on disk.    Mutant files / instances that exist.
  MEASURED   results.    Rows carrying an outcome (a run happened).

Usage:  python3 schema/gen_dashboard.py [out.md]
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent            # benchmark2/
sys.path.insert(0, str(BASE))

DEADLINE = date(2026, 9, 24)
CLASSES = ["M1", "M2", "M3", "M4"]
LANES = ["choreo", "triton", "mlir-low", "mlir-linalg", "iree"]
PER_CLASS_LANES = {"mlir-low", "mlir-linalg"}
OUT_DEFAULT = Path("/home/garfee/croq-paper-plan/svn/eurosys27/DASHBOARD.md")


def load_json(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception as e:                                # noqa: BLE001
        print("  ! cannot read %s: %s" % (p, e), file=sys.stderr)
        return {}


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


# ------------------------------------------------------------------ sources

TAX = load_json(BASE / "schema" / "method-taxonomy.json")
try:
    from choreo import mutations as MUT                     # noqa: E402
except Exception as e:                                       # noqa: BLE001
    print("  ! cannot import choreo.mutations: %s" % e, file=sys.stderr)
    MUT = None

MANIFEST = load_json(BASE / "choreo" / "raw" / "mutant_manifest.json")

FAMILIES = TAX.get("families", {})
BUDGET = TAX.get("budget", {})
N_PER_FAMILY = BUDGET.get("N_per_family", 8)
TARGETS = TAX.get("targets", {})
IN_SCOPE = TAX.get("in_scope_lanes", {})
LANE_SCOPE = TAX.get("lane_scope", {})
MEASURED_TODAY = TAX.get("measured_today", {})
KNOWN_DEBT = TAX.get("known_debt", {})

FAM_CLASS = {f: r["class"] for f, r in FAMILIES.items()}
SPEC_FAM = {}
for _f, _r in FAMILIES.items():
    for _s in _r.get("spec_ids", []):
        SPEC_FAM.setdefault(_s, _f)

REGISTRY = MUT.SPEC_REGISTRY if MUT else {}
OPERATORS = MUT.ALL if MUT else {}


def fam_sort(f):
    c = FAM_CLASS.get(f, "M9")
    return (CLASSES.index(c) if c in CLASSES else 9, f)


def generatable(sid):
    m = REGISTRY.get(sid)
    return bool(m) and (m.get("status") == "implemented"
                        and m.get("path") != "P2"
                        and m.get("prohibition") != "repaired")


def spec_health(family):
    specs = FAMILIES.get(family, {}).get("spec_ids", [])
    rf = sum(1 for s in specs if generatable(s))
    ra = sum(1 for s in specs if generatable(s)
             and REGISTRY.get(s, {}).get("admissible"))
    return specs, rf, ra


def planned(lane, family):
    cls = FAM_CLASS.get(family)
    if not cls or lane not in IN_SCOPE.get(cls, []):
        return 0
    if LANE_SCOPE.get(lane, {}).get(family) in ("derived", "absent"):
        return 0
    return N_PER_FAMILY


def model_total(lane):
    """The scope model's own answer for a lane, recomputed from first
    principles: 8 x (families of an in-scope class that the lane does not
    carve out). Must equal the declared `targets[lane]`; if it does not, the
    dashboard's denominators are wrong, not the taxonomy's."""
    return sum(planned(lane, f) for f in FAMILIES)


# ------------------------------------------------------- choreo (family model)

choreo_fam = dict(MANIFEST.get("family_depth", {}) or {})
choreo_material = len(MANIFEST.get("mutants", []))
choreo_by_cls = Counter()
for f, n in choreo_fam.items():
    choreo_by_cls[FAM_CLASS.get(f, "?")] += n

# ------------------------------------------------- family lanes (triton/iree)

fam_rows = {L: load_jsonl(BASE / L / "raw" / "mutants.jsonl") for L in LANES}
# triton/iree rows carry `class` but no `spec_id`, so no family attribution.
fam_lane_cls = {L: Counter(r.get("class") for r in fam_rows[L])
                for L in ("triton", "iree")}

# ------------------------------------------- per-class lanes (mlir-low/linalg)

mlir = {}
for L in PER_CLASS_LANES:
    cen = load_json(BASE / L / "raw" / "minimal-census.json")
    rows = fam_rows[L]
    by_fam = defaultdict(int)
    seen = set()
    for r in rows:
        key = (r.get("spec_id"), r.get("category"), r.get("shape"), r.get("kernel"))
        if key in seen:
            continue
        seen.add(key)
        f = SPEC_FAM.get(r.get("spec_id"))
        if f:
            by_fam[f] += 1
    # The census labels the whole lane with ONE class, but a lane can carry a
    # spec that was re-homed into another class -- mlir-low ships M1.6, whose
    # registry class is M4 (family M4-d). Attribute by spec_id -> family ->
    # class so the class split is real, not the census's label.
    by_cls = Counter()
    for f, n in by_fam.items():
        by_cls[FAM_CLASS.get(f, "?")] += n
    mlir[L] = {
        "census_class": cen.get("class"),
        "rows": len(rows),
        "injected": cen.get("n_injected") or len(seen),
        "target_per_class": cen.get("n_target_per_class", 40),
        "categories": cen.get("categories", []),
        "by_fam": dict(by_fam),
        "by_cls": dict(by_cls),
    }

# ---------------------------------------------------------------- rendering


def mark(impl, plan, ok="\u2705", no="\u274c"):
    if plan == 0:
        return "\u2014"
    if impl >= plan:
        return "**%d/%d** %s" % (impl, plan, ok)
    if impl == 0:
        return "%d/%d %s" % (impl, plan, no)
    return "%d/%d" % (impl, plan)


def main():
    today = date.today()
    days_left = (DEADLINE - today).days
    L = []
    w = L.append

    w("# Mutation-benchmark dashboard")
    w("")
    w("_Generated %s by `schema/gen_dashboard.py` \u2014 a read-only projection of "
      "what is on disk. Deadline %s (**T\u2212%d**)._" %
      (today.isoformat(), DEADLINE.isoformat(), days_left))
    w("")
    w("Three stages, never interchangeable: **TARGET** (declared) \u2192 "
      "**MATERIAL** (exists) \u2192 **MEASURED** (a run happened).")
    w("")
    w("> **One composition model, for every lane.** "
      "`N_per_family = 8` and `planned(lane, class) = 8 \u00d7 (families of "
      "that class in scope for that lane)`. This is `method_taxonomy.target()` "
      "and it is check-enforced. The five lanes sum to %d." %
      TARGETS.get("total", 0))
    w(">")
    w("> A lane's class cell differs from the declared cell for exactly two "
      "reasons, and only two: **(1)** the class's own family count "
      "(M4 has 7 families, so M4's cell is 56 everywhere, not 64), and "
      "**(2)** that lane's `lane_scope` carve-outs, each worth exactly 8. "
      "With no carve-outs an in-scope cell is the full declared cell.")
    w(">")
    w("> **Separate and weaker: `n_target_per_class = 40`** "
      "(`class-axis.json`). That is the mlir collectors' own internal "
      "per-class gate, not a budget. `mlir-low` clears it (48 \u2265 40) while "
      "still being under its M1 cell of 64. It is never the coverage "
      "denominator here. The mlir lanes are also not distributed across "
      "their class's 8 families \u2014 `mlir-low`'s M1 work lands on 4 "
      "categories \u2014 which is why \u00a74 shows them as raw counts.")
    w("")

    # ------------------------------------------------------------- 1. top line
    w("## 1. Where we are")
    w("")
    w("| | count |")
    w("|---|---|")
    w("| Families declared | %d |" % len(FAMILIES))
    w("| Declared target, all lanes | %d |" % TARGETS.get("total", 0))
    w("| Choreo instances on disk | %d |" % choreo_material)
    w("| Operators written | %d |" % sum(len(v) for v in OPERATORS.values()))
    w("| Choreo classes at/over cell | %d / 4 |" % sum(
        1 for c in CLASSES
        if choreo_by_cls[c] >= BUDGET.get("classes", {}).get(c, {}).get("cell", 64)))
    w("")

    # ------------------------------------------------------------- 2. per lane
    w("## 2. Per lane")
    w("")
    w("| lane | model | in scope | target | instances | rows | \u0394 |")
    w("|---|---|---|---|---|---|---|")
    for lane in LANES:
        cls = [c for c in CLASSES if lane in IN_SCOPE.get(c, [])]
        tgt = TARGETS.get(lane, 0)
        if lane == "choreo":
            model, inst = "family", choreo_material
            rows = MEASURED_TODAY.get(lane, 0)
        elif lane in PER_CLASS_LANES:
            m = mlir[lane]
            model, inst, rows = "family", m["injected"], m["rows"]
        else:
            model, inst = "family", len(fam_rows[lane])
            rows = MEASURED_TODAY.get(lane, len(fam_rows[lane]))
        d = tgt - inst
        dd = "**%d to go**" % d if d > 0 else ("done" if d == 0 else "+%d" % -d)
        w("| `%s` | %s | %s | %d | %d | %d | %s |" %
          (lane, model, "/".join(cls), tgt, inst, rows, dd))
    w("")
    w("`instances` is deduplicated; `rows` is what `measured_today` counts. "
      "They diverge because the mlir lanes run each mutant **twice** (rtv "
      "`off`/`on`): 96 rows = 48 instances, 120 rows = 50. **Do not read "
      "`rows` as coverage.**")
    w("")
    _mism = [l for l in LANES if model_total(l) != TARGETS.get(l, 0)]
    if _mism:
        w("> \u26a0 **Scope model does not reproduce the declared targets** for: "
          "%s. Every denominator below is suspect. Reconcile "
          "`method_taxonomy.target()` against `targets` before trusting this "
          "file." % ", ".join("`%s`" % x for x in _mism))
    else:
        w("**Scope model reconciles.** `8 \u00d7 families-in-scope` reproduces "
          "all five declared targets exactly (%s = %d). So a cell below "
          "differs from its class's declared cell only by the lane's "
          "`lane_scope` carve-outs." % (" + ".join(
              str(TARGETS.get(l, 0)) for l in LANES), TARGETS.get("total", 0)))
    w("")

    # --------------------------------------------------------- 3. class x lane
    w("## 3. Per class \u00d7 lane (instances / that lane's own budget)")
    w("")
    w("| class | cell | " + " | ".join("`%s`" % x for x in LANES) + " |")
    w("|---|---|" + "---|" * len(LANES))
    for cls in CLASSES:
        fams = [f for f in FAMILIES if FAM_CLASS.get(f) == cls]
        cell = BUDGET.get("classes", {}).get(cls, {}).get("cell", len(fams) * 8)
        cells = []
        for lane in LANES:
            if lane not in IN_SCOPE.get(cls, []):
                cells.append("n/a")
            elif lane == "choreo":
                cells.append(mark(choreo_by_cls[cls], cell))
            elif lane in PER_CLASS_LANES:
                m = mlir[lane]
                n = m["by_cls"].get(cls, 0)
                cells.append(mark(n, cell))
            else:
                plan = sum(planned(lane, f) for f in fams)
                cells.append(mark(fam_lane_cls[lane].get(cls, 0), plan))
        w("| **%s** | %d | %s |" % (cls, cell, " | ".join(cells)))
    w("")
    w("**Bottom line: M4 is empty in every lane except choreo and one cell of "
      "`mlir-low`.** `mlir-low`'s M4 is 8/56 \u2014 and it is `M4-d` only, "
      "arriving via `M1.6`, which the registry re-homed into M4. `mlir-low` "
      "has no other M4 family and no other class; `mlir-linalg` has no M4 at "
      "all. `triton` has 2 of 48 M3 and 0 of 56 M4; `iree` 0 of 56 M4.")
    w("")
    w("**A lane is not its census label.** `mlir-low`'s census says "
      "`class: M1`, but 8 of its 48 instances are class M4 (`M1.6` "
      "\u2192 `M4-d`). The split above is by `spec_id` \u2192 family "
      "\u2192 class, not by the census label. Watch for the same re-homing "
      "in any lane carrying `M1.6` or `M1.7`.")
    w("")
    w("`cell` is from `method-taxonomy.json` `budget` (8 families \u00d7 8; M4 "
      "is 7 \u00d7 8) and is the denominator for **every** lane. Where a lane "
      "shows less than the cell the difference is its `lane_scope` "
      "carve-outs \u2014 `triton` M3 48 = 64 \u2212 2\u00d78, `iree` M2 "
      "32 = 64 \u2212 4\u00d78.")
    w("")

    # ------------------------------------------------------------- 4. families
    w("## 4. Per family")
    w("")
    w("| family | class | name | specs | `R_f` | `R_a` | plan | choreo | triton | mlir-low | mlir-linalg | iree |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(FAMILIES, key=fam_sort):
        cls = FAM_CLASS.get(f, "?")
        nm = FAMILIES[f].get("name", "")
        specs, rf, ra = spec_health(f)
        cells = []
        for lane in LANES:
            p = planned(lane, f)
            if lane == "choreo":
                cells.append(mark(choreo_fam.get(f, 0), p))
            elif lane in PER_CLASS_LANES:
                n = mlir[lane]["by_fam"].get(f, 0)
                cells.append(str(n) if n else "\u2014")
            else:
                cells.append("?" if p else "\u2014")
        w("| `%s` | %s | %s | %d | **%d** | %d | %d | %s |" %
          (f, cls, nm, len(specs), rf, ra, planned("choreo", f), " | ".join(cells)))
    w("")
    w("*`mlir-low` / `mlir-linalg` columns are raw instance counts, not `/8` "
      "\u2014 those lanes are built as *spec \u00d7 category \u00d7 shape "
      "\u00d7 kernel* and are not distributed across the class's 8 families "
      "(`mlir-low`'s M1 work lands on 4 categories). Their budget is still "
      "the family model; see the note at the top.*")
    w("")
    w("**Reading `R_f` and `R_a`:**")
    w("")
    rf0 = [f for f in sorted(FAMILIES, key=fam_sort) if spec_health(f)[1] == 0]
    ra0 = [f for f in sorted(FAMILIES, key=fam_sort)
           if spec_health(f)[1] > 0 and spec_health(f)[2] == 0]
    w("- **`R_f = 0`** \u2014 no generatable declaration. **Writing operators "
      "cannot fix this**; it needs a new spec on a P1 path. %d families: %s." %
      (len(rf0), ", ".join("`%s`" % f for f in rf0)))
    w("- **`R_f > 0` but `R_a = 0`** \u2014 declarations exist and are "
      "generatable, but none is *admissible*, so nothing can enter the "
      "denominator. Reads half-done while being zero. %d families: %s." %
      (len(ra0), ", ".join("`%s`" % f for f in ra0)))
    w("- **`R_a < R_f`** \u2014 declared specs outnumber admissible ones, so the "
      "ceiling is `R_a`-driven.")
    w("")

    # ------------------------------------------------------------- 5. registry
    if REGISTRY:
        w("## 5. Declaration status (`SPEC_REGISTRY`)")
        w("")
        w("| class | specs | implemented | pending | admissible | inadmissible |")
        w("|---|---|---|---|---|---|")
        for cls in CLASSES:
            sids = [s for s in REGISTRY if s.startswith(cls + ".")]
            impl = [s for s in sids if REGISTRY[s].get("status") == "implemented"]
            adm = [s for s in sids if REGISTRY[s].get("admissible")]
            w("| **%s** | %d | %d | %d | %d | %d |" %
              (cls, len(sids), len(impl), len(sids) - len(impl), len(adm),
               len(sids) - len(adm)))
        w("")
        w("| class | operators | admissible | inadmissible |")
        w("|---|---|---|---|")
        for cls in CLASSES:
            ops = OPERATORS.get(cls, [])
            a = sum(1 for m in ops if m.admissible)
            w("| **%s** | %d | %d | %d |" % (cls, len(ops), a, len(ops) - a))
        w("")
        w("An operator that is written but inadmissible is **inert**: it can "
          "never be selected, so it inflates the operator count without moving "
          "any instance count. M3 is where this bites hardest \u2014 53 "
          "operators, only 33 admissible.")
        w("")

    # ------------------------------------------------------------- 6. blockers
    items = KNOWN_DEBT.get("items", {})
    if items:
        w("## 6. Blockers \u2014 design calls, not coding")
        w("")

        def clip(s, n=150):
            s = " ".join(str(s or "").split())
            return s[:n - 1] + "\u2026" if len(s) > n else s

        w("| key | observed | must NOT be printed as | remedy |")
        w("|---|---|---|---|")
        it = items.items() if isinstance(items, dict) else enumerate(items)
        for k, v in it:
            if not isinstance(v, dict):
                w("| `%s` | %s | | |" % (k, v))
                continue
            w("| `%s` | %s | %s | %s |" %
              (k, clip(v.get("observed")), clip(v.get("must_not_print_as")),
               clip(v.get("fix"), 190)))
        w("")
        w("Every remedy ends in the same words: *\u201cDeciding which is a "
          "design call, not a coding one.\u201d* No quantity of operators "
          "clears these.")
        w("")

    # ----------------------------------------------------------------- 7. log
    log = BASE / "schema" / "dashboard-log.md"
    if log.exists():
        w("---")
        w("")
        w(log.read_text().strip())
        w("")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    out.write_text("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (out, len(L)))


if __name__ == "__main__":
    main()
