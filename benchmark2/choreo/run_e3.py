#!/usr/bin/env python3
"""E3 — discharge strength via *remainders* (choreo side).

Plan §5.2: because the SOTA toolchains keep no generation count, E3 cannot
compute "resolved / generated" for them. It measures **remainders** instead —
the obligations left unresolved — which is observable everywhere. For choreo the
remainder is *explicit and accounted*:

    "1,199 of 17,717 (6.8%) are runtime-materialized or budgeted-not-emitted,
     0 silently dropped."

This script derives that sentence from the E2 ledger sweep rather than asserting
it. It is the evidence for C3 (discharge strength) and for the `dropped = 0`
column of `tab:per-operator` (S4).

FEEDS  S4 (per-operator ledger: expressed / discharged / runtime / budgeted /
           dropped), and the choreo column of the remainder accounting.
       S9 (`remainder` records) is owned by the SOTA workers per the statistics
       manifest; choreo emits the same record shape for its own column so the
       integrator can merge uniformly.

--------------------------------------------------------------------------
THE PINNED CRITERION (plan R5: "must be pinned to a documented IR criterion")
--------------------------------------------------------------------------
R5 assigns ownership of the rule to the `coordinator`. The criterion applied
here for choreo, stated so a reviewer can check it against the raw ledger:

    An obligation is a REMAINDER iff the ledger records `outcome: runtime`,
    i.e. it was not resolved at compile time (neither `static-true` nor
    `static-false`).

    A remainder splits by whether a guard was actually emitted:
      enabled = true   →  RUNTIME-MATERIALIZED. The cost filter kept it; a real
                          `runtime_check(...)` call exists in the generated code.
      enabled = false  →  BUDGETED-NOT-EMITTED. Accounted for in the ledger,
                          suppressed by the cost filter, no code emitted.

    DROPPED  =  total − (proven + refuted + runtime + budgeted)
    and the paper's claim is that this is 0 for every kernel. It is checked per
    kernel here, not assumed: a nonzero value anywhere is reported loudly.

The split is well-defined because of a measured invariant over the full
311-kernel sweep: `enabled == (cost == "entry")` with 0 exceptions across all
12,167 obligations. Only entry-cost checks survive the filter.

Note the asymmetry with the SOTA criterion: for MLIR-low the remainder is "an
unconditional bounds guard left after canonicalization" — a *syntactic* residue
in the IR. For choreo it is a *ledger field*. choreo's remainder is smaller and
accounted precisely because the obligation was never lost sight of; that contrast
is the point of §5.2's "gen+resolve compatibility" argument.

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES NOT DO
--------------------------------------------------------------------------
It does not re-run choreo. It consumes `raw/e2_ledger.json`. Re-deriving the
ledger here would duplicate ~311 invocations and, worse, could diverge from the
E2 records that S3/S4/S5/S6 are computed from. Every number in S4 must trace to
the same raw sweep. Pass --e2 to point at a different E2 output.

It does not implement S7 (the no-interval counterfactual, §5.3). No
interval-disable flag exists in the pinned choreo build — see the note printed
at the end of the run. S7 is inherited, not re-derived here.

E3 touches no GPU. exclusive=false.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark2/choreo
B2 = os.path.dirname(HERE)                                 # benchmark2/
REPO = os.path.dirname(B2)                                 # svn-artifacts/

RAW = os.path.join(HERE, "raw")
E2_DEFAULT = os.path.join(RAW, "e2_ledger.json")
OUT = os.path.join(RAW, "e3_remainder.json")

TOOLCHAIN = "choreo"

# The pinned criterion, recorded verbatim on every emitted record so the
# integrator and any reviewer can see which rule produced the count (R5).
CRITERION_REF = (
    "choreo-ledger-outcome: remainder iff ledger outcome == 'runtime' "
    "(not resolved at compile time); split by `enabled` — true = "
    "runtime-materialized guard emitted, false = budgeted-not-emitted "
    "(suppressed by the cost filter); dropped = total - (proven + refuted + "
    "runtime + budgeted), asserted 0 per kernel. Criterion owner: coordinator "
    "(plan R5)."
)

OUTCOMES = ("proven", "refuted", "runtime", "budgeted")


def rollup(obligations):
    """{outcome: n} plus the mechanism split, for one scope."""
    r = {k: 0 for k in OUTCOMES}
    mech = {"canonical": 0, "interval": 0, "direct": 0, "other": 0}
    for o in obligations:
        oc = o.get("outcome")
        if oc in r:
            r[oc] += 1
        else:
            r.setdefault("unmapped:" + str(oc), 0)
            r["unmapped:" + str(oc)] += 1
        m = o.get("mechanism")
        mech[m if m in mech else "other"] += 1
    r["total"] = sum(r[k] for k in OUTCOMES)
    r["mechanism"] = mech
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e2", default=E2_DEFAULT,
                    help="E2 ledger sweep to consume (default raw/e2_ledger.json)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--size", default="small", choices=["small", "full"])
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    a = ap.parse_args()

    a.e2 = os.path.abspath(a.e2)
    a.out = os.path.abspath(a.out)

    if not os.path.exists(a.e2):
        print(f"ERROR: no E2 sweep at {a.e2}", file=sys.stderr)
        print("       run `benchmark2/choreo/run.sh e2` first — E3 consumes it",
              file=sys.stderr)
        return 2

    data = json.load(open(a.e2))
    kernels = data.get("kernels", [])
    ok = [k for k in kernels if k.get("compile") == "ok"]
    print(f"[choreo] E3: remainder accounting over {len(ok)}/{len(kernels)} "
          f"kernels from {os.path.relpath(a.e2, REPO)}")
    print(f"[choreo] criterion: {CRITERION_REF}")

    # ------------------------------------------------- per-kernel accounting ----
    per_kernel = []
    dropped_kernels = []
    unmapped_kernels = []
    unreconciled = []
    contradictions = []

    for k in ok:
        obls = k.get("obligations", [])
        r = rollup(obls)

        # `dropped` is the claim under test: obligations the ledger neither
        # discharged nor accounted for. Computed against the kernel's own
        # reported total so a silent row loss is caught.
        reported_total = k.get("total_obligations", len(obls))
        dropped = reported_total - r["total"]

        # materialized vs budgeted. map_outcome() in run_e2.py already folded
        # the ledger's `enabled` field into the outcome enum, so read it back
        # from there rather than re-deriving — the two must never disagree:
        #   enabled=true  → outcome "runtime"  → a real runtime_check() was
        #                   emitted into the generated code (materialized)
        #   enabled=false → outcome "budgeted" → accounted for in the ledger,
        #                   suppressed by the cost filter, no code emitted
        # The direction was measured, not assumed: see map_outcome()'s docstring
        # for the -rtc=entry site count that pins it.
        emitted = r["runtime"]
        suppressed = r["budgeted"]

        rec = {
            "toolchain": TOOLCHAIN,
            "category": k["category"],
            "kernel_id": k["kernel_id"],
            "settings_hash": k.get("settings_hash"),
            "kernel_hash": k.get("kernel_hash"),
            "shape_class": k.get("shape_class"),
            "expressed": reported_total,
            "discharged": r["proven"] + r["refuted"],
            "proven": r["proven"],
            "refuted": r["refuted"],
            "runtime_materialized": emitted,
            "budgeted_not_emitted": suppressed,
            "remainder": emitted + suppressed,
            "dropped": dropped,
            "mechanism": r["mechanism"],
            "criterion_ref": CRITERION_REF,
        }
        per_kernel.append(rec)

        if dropped != 0:
            dropped_kernels.append((k["kernel_id"], dropped, reported_total,
                                    r["total"]))
        if any(kk.startswith("unmapped:") for kk in r):
            unmapped_kernels.append((k["kernel_id"],
                                     {kk: v for kk, v in r.items()
                                      if kk.startswith("unmapped:")}))
        if not k.get("reconciled"):
            unreconciled.append(k["kernel_id"])
        if k.get("shape_contradiction"):
            contradictions.append(k["kernel_id"])

    # ------------------------------------------------------ per-category ----
    by_cat = {}
    for r in per_kernel:
        c = by_cat.setdefault(r["category"], {
            "toolchain": TOOLCHAIN, "category": r["category"], "kernels": 0,
            "expressed": 0, "discharged": 0, "proven": 0, "refuted": 0,
            "runtime_materialized": 0, "budgeted_not_emitted": 0,
            "remainder": 0, "dropped": 0,
            "static": {"expressed": 0, "discharged": 0, "remainder": 0},
            "dynamic": {"expressed": 0, "discharged": 0, "remainder": 0},
            "mechanism": {"canonical": 0, "interval": 0, "direct": 0, "other": 0},
        })
        c["kernels"] += 1
        for f in ("expressed", "discharged", "proven", "refuted",
                  "runtime_materialized", "budgeted_not_emitted",
                  "remainder", "dropped"):
            c[f] += r[f]
        for m, v in r["mechanism"].items():
            c["mechanism"][m] = c["mechanism"].get(m, 0) + v
        sc = r["shape_class"] if r["shape_class"] in ("static", "dynamic") else None
        if sc:
            c[sc]["expressed"] += r["expressed"]
            c[sc]["discharged"] += r["discharged"]
            c[sc]["remainder"] += r["remainder"]

    for c in by_cat.values():
        c["criterion_ref"] = CRITERION_REF
        c["residue_pct"] = round(100.0 * c["remainder"] / c["expressed"], 2) \
            if c["expressed"] else 0.0
        c["discharge_pct"] = round(100.0 * c["discharged"] / c["expressed"], 2) \
            if c["expressed"] else 0.0

    # ----------------------------------------------------------- grand total ----
    G = {f: sum(c[f] for c in by_cat.values()) for f in
         ("kernels", "expressed", "discharged", "proven", "refuted",
          "runtime_materialized", "budgeted_not_emitted", "remainder", "dropped")}
    G["mechanism"] = {m: sum(c["mechanism"].get(m, 0) for c in by_cat.values())
                      for m in ("canonical", "interval", "direct", "other")}
    G["residue_pct"] = round(100.0 * G["remainder"] / G["expressed"], 2) \
        if G["expressed"] else 0.0
    G["discharge_pct"] = round(100.0 * G["discharged"] / G["expressed"], 2) \
        if G["expressed"] else 0.0

    out = {
        "toolchain": TOOLCHAIN,
        "toolchain_version": data.get("toolchain_version"),
        "produced_by": "e3",
        "consumes": os.path.relpath(a.e2, REPO),
        "size": a.size,
        "gpu_device": str(a.device),
        "exclusive": False,
        "criterion_ref": CRITERION_REF,
        "grand_total": G,
        "categories": [by_cat[c] for c in sorted(by_cat)],
        "kernels": sorted(per_kernel, key=lambda r: r["kernel_id"]),
        "integrity": {
            "kernels_ok": len(ok),
            "kernels_total": len(kernels),
            "dropped_nonzero": len(dropped_kernels),
            "unmapped_outcomes": len(unmapped_kernels),
            "unreconciled_e2": len(unreconciled),
            "shape_contradictions": len(contradictions),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(a.out, REPO)}")

    # ------------------------------------------------------------- report ----
    print("\n=== S4 per-operator ledger (expressed / discharged / runtime / "
          "budgeted / dropped) ===")
    print(f"{'category':<22}{'kern':>5}{'expressed':>11}{'discharged':>12}"
          f"{'runtime':>9}{'budgeted':>10}{'dropped':>9}{'residue%':>10}")
    for c in sorted(by_cat.values(), key=lambda x: x["category"]):
        print(f"{c['category']:<22}{c['kernels']:>5}{c['expressed']:>11}"
              f"{c['discharged']:>12}{c['runtime_materialized']:>9}"
              f"{c['budgeted_not_emitted']:>10}{c['dropped']:>9}"
              f"{c['residue_pct']:>10.2f}")
    print(f"{'-'*22}{'-'*5}{'-'*11}{'-'*12}{'-'*9}{'-'*10}{'-'*9}{'-'*10}")
    print(f"{'TOTAL':<22}{G['kernels']:>5}{G['expressed']:>11}"
          f"{G['discharged']:>12}{G['runtime_materialized']:>9}"
          f"{G['budgeted_not_emitted']:>10}{G['dropped']:>9}"
          f"{G['residue_pct']:>10.2f}")

    print(f"\n=== the §5.2 sentence, re-derived ===")
    print(f"  {G['remainder']} of {G['expressed']} ({G['residue_pct']}%) are "
          f"runtime-materialized or budgeted-not-emitted, "
          f"{G['dropped']} silently dropped")
    print(f"    runtime-materialized (guard emitted)   : "
          f"{G['runtime_materialized']}")
    print(f"    budgeted-not-emitted (cost-filtered) : "
          f"{G['budgeted_not_emitted']}")
    print(f"  discharge rate: {G['discharge_pct']}% "
          f"(proven {G['proven']} + refuted {G['refuted']})")
    print(f"  mechanism split: canonical {G['mechanism']['canonical']} / "
          f"interval {G['mechanism']['interval']} / "
          f"direct {G['mechanism']['direct']}")

    # static vs dynamic residue — the split §6 says must not be smeared
    print("\n=== residue by shape class (static must be ~0: §6) ===")
    for sc in ("static", "dynamic"):
        e = sum(c[sc]["expressed"] for c in by_cat.values())
        d = sum(c[sc]["discharged"] for c in by_cat.values())
        r = sum(c[sc]["remainder"] for c in by_cat.values())
        if not e:
            continue
        print(f"  {sc:<8} expressed={e:<7} discharged={d:<7} remainder={r:<6} "
              f"residue={100.0*r/e:5.2f}%")

    rc = 0
    if dropped_kernels:
        rc = 1
        print(f"\n*** INTEGRITY FAILURE: {len(dropped_kernels)} kernel(s) with "
              f"dropped != 0 — the '0 silently dropped' claim is FALSE ***")
        for kid, dr, tot, acc in dropped_kernels[:20]:
            print(f"    {kid}: dropped={dr} (reported_total={tot}, "
                  f"accounted={acc})")
    if unmapped_kernels:
        rc = 1
        print(f"\n*** {len(unmapped_kernels)} kernel(s) with an outcome outside "
              f"the schema enum ***")
        for kid, u in unmapped_kernels[:20]:
            print(f"    {kid}: {u}")
    if unreconciled:
        rc = 1
        print(f"\n*** {len(unreconciled)} kernel(s) failed E2 reconciliation; "
              f"their remainder counts are untrustworthy ***")
        for kid in unreconciled[:20]:
            print(f"    {kid}")
    if contradictions:
        print(f"\n*** {len(contradictions)} shape-class contradiction(s) "
              f"(static kernel emitting runtime obligations) ***")
        for kid in contradictions[:20]:
            print(f"    {kid}")

    if rc == 0:
        print("\nintegrity: PASS — every obligation accounted for, "
              "0 dropped, 0 unmapped outcomes")

    print("\nNOTE (S7, plan §5.3): the no-interval counterfactual "
          "(93.2%→77.2%) is NOT re-derived here. The pinned choreo build "
          "exposes no interval-disable flag; S7 is inherited from the prior "
          "benchmark and must be reconciled by the coordinator before it is "
          "reported as a benchmark2 number.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
