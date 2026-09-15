"""Build the cross-lane mutation correspondence ledger.

Why this exists
---------------
The four SOTA lanes (mlir-linalg, mlir-low, iree, triton) and the choreo lane
each hold a corpus, but nothing in the repo asserts that a given *defect* is
represented on more than one lane.  A reader comparing "choreo detected X%"
against "Triton detected Y%" is therefore comparing two different mutation
sets, not two detectors.  This script makes that fact checkable.

What it emits
-------------
`schema/mutation-correspondence.json` — one entry per choreo spec (the
register), with

  * the semantic mutation unit (the dedup key),
  * whether choreo realises it (and with how many mutants),
  * for each SOTA lane: `covered` / `n-a` / `gap` / `uncompared`, with the
    *evidence* (file + symbol) that justifies the verdict,
  * the N/A kind, because there are three different things currently spelled
    `prohibition="absent"`.

Every verdict is traceable to a source file.  Where a verdict is a judgement
(the choreo-spec -> SOTA-family mapping) the judgement is written down here
rather than left implicit, so it can be argued with.

Run:  python3 schema/build_correspondence.py [--check]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "choreo" / "raw" / "spec_registry.json"
MANIFEST = ROOT / "choreo" / "raw" / "mutant_manifest.json"
N_TARGET = 40

# --------------------------------------------------------------------------
# The N/A taxonomy.  This is the correction that motivates the audit: the
# spec registry spells all three of these "absent", but they are different
# claims with different evidential weight.
# --------------------------------------------------------------------------
NA_KINDS = {
    "delegation": (
        "The model does not check this because the responsibility is "
        "contractually the code-writer's.  Triton has no cross-tensor extent "
        "relation to violate because the kernel never declares one.  This is "
        "an apple-to-apple exclusion: comparing our rate against theirs would "
        "credit them for a check they never claimed."
    ),
    "constitutional": (
        "The surface has no construct of that kind, so the defect does not "
        "exist at that level of the stack.  IREE's entry surface carries no "
        "interior memory access, so there is nothing for an M1 mutation to "
        "corrupt.  This is a scope claim, not a detection claim."
    ),
    "suite-surface": (
        "OUR base kernels lack the surface the defect needs.  This is a hole "
        "in the benchmark, not a property of any programming model, and it "
        "must never be reported as n/a."
    ),
    "model-prohibited": (
        "The model repairs or rejects the construct before it can be "
        "observed, or the construct cannot exist (e.g. an empty iteration "
        "space with a positive bound)."
    ),
}

# --------------------------------------------------------------------------
# choreo spec -> SOTA lane realisation.
#
# The SOTA lanes were written against the v1 catalogue (M1.1-M1.6,
# M2.1-M2.5).  Each mapping below names the artefact that implements it, so a
# reader can open the file and disagree.
# --------------------------------------------------------------------------
TRITON_FAMILY = {
    "M1.1": ("triton/mutants/relu.py", "relu_nomask", 1),
    "M1.2": ("triton/mutants/relu.py", "relu_offbyone", 2),
    "M1.3": ("triton/mutants/relu.py", "relu_negidx", 3),
    "M1.4": ("triton/mutants/relu.py", "relu_badstride", 4),
    "M1.5": ("triton/mutants/relu.py", "relu_offsetview", 5),
    "M1.6": ("triton/mutants/relu.py", "relu_zerorange", 6),
    "M3.1": ("triton/mutants/matmul.py", "family 1 (K % 16 != 0)", 1),
    # README: "2 misaligned base / tile exceeding shared-memory budget".
    # That is the register's M3.11 (shared operand base not 128B aligned).
    "M3.11": ("triton/mutants/matmul.py", "family 2 (misaligned base)", 2),
}
MLIR_LOW_SPEC = {
    "M1.1": "drop-mask",
    "M1.2": "off-by-one",
    "M1.3": "negative-index",
    "M1.4": "transposed-stride",
    "M1.5": "offset-overrun",
    "M1.6": "zero-stride",
}
MLIR_LINALG_SPEC = {
    "M2.1": "bump leading extent of rhs/b/scale",
    "M2.2": "bump trailing axis of out",
    "M2.3": "bump trailing axis of bias/a/rhs",
    "M2.4": "bump leading axis of out",
    "M2.5": "partial-write / duplicate-write structural variant",
    # The lane labels these M2.1 ("bump leading extent of rhs"); on a matmul
    # kernel rhs's leading extent IS the contraction dim K, so the SITE is the
    # register's M2.14.  Recorded here rather than silently dropped.
    "M2.14": "M2.1 bump-leading-extent(rhs) applied to matmul -> K pm1",
}
IREE_SPEC = {
    "M2.1": "elemwise_add rhs d1 pm1; layer_norm beta/gamma len pm1",
    "M2.14": "iree/mutants/M2/matmul/iree-matmul-*-rhs-K pm1",
}

# Sites where the lane's own spec label differs from the register's but the
# edited site is the same.  Recorded so the ledger is auditable, and so nobody
# later "discovers" a phantom extra mutant.
CROSS_LABEL_SPEC = {
    "M2.14": {
        "mlir-linalg": "lane label M2.1 on matmul kernels",
        "iree": "lane label M2.1 site; iree names the axis K explicitly",
    },
}

# Verdicts that are a model property rather than a suite hole.
LANE_NA_KIND = {
    ("iree", "M1"): "constitutional",
    ("iree", "M3"): "constitutional",
    ("mlir-linalg", "M1"): "constitutional",
    ("mlir-linalg", "M3"): "constitutional",
    ("mlir-low", "M2"): "constitutional",
    ("mlir-low", "M3"): "constitutional",
    ("triton", "M2"): "delegation",
}

LANES = ("triton", "mlir-linalg", "mlir-low", "iree")

# Where each lane's committed stats.json lives. `triton` keeps its own
# results/ directory; the rest share results/<lane>/.
LANE_STATS = {
    "triton": ROOT / "triton" / "results" / "stats.json",
    "mlir-linalg": ROOT / "results" / "mlir-linalg" / "stats.json",
    "mlir-low": ROOT / "results" / "mlir-low" / "stats.json",
    "iree": ROOT / "results" / "iree" / "stats.json",
}

# The class -> obligation bijection. The axis states it; it is restated here
# ONLY to translate a lane's S8 (which is keyed by obligation) into a per-class
# capability. The axis remains the source of truth.
OBLIGATION_OF = {"M1": "elem", "M2": "shape", "M3": "hw", "M4": "loop"}

# The power floor: below this many injections a cell cannot support a rate.
POWER_FLOOR = 35
LANE_STATS_ALL = dict(
    LANE_STATS, choreo=ROOT / "results" / "choreo" / "stats.json")


def lane_cells() -> dict:
    """Per lane, how many mutants are ACTUALLY injected into each cell.

    The denominator health check.  A cell with fewer than POWER_FLOOR
    injections is annotated under-powered and must not be printed as a rate.
    Reads both the flat `S1_detection` and the nested `S1_detection_matrix`
    spellings, because the lanes drifted on the key name.
    """
    out: dict[str, dict[str, int]] = {}
    for lane, path in LANE_STATS_ALL.items():
        if not path.exists():
            out[lane] = {}
            continue
        d = json.loads(path.read_text())
        cells = d.get("S1_detection") or d.get("S1_detection_matrix") or {}
        if isinstance(cells, dict) and "per_class" in cells:
            cells = cells["per_class"]
        row: dict[str, int] = {}
        for cls, tally in cells.items():
            if isinstance(tally, dict) and tally.get("n_injected"):
                row[cls] = tally["n_injected"]
        out[lane] = row
    return out


# --------------------------------------------------------------------------
# Coverage vocabulary.  These are four DIFFERENT claims and the paper must not
# merge them: a lane that never wrote the mutant is not the same as a lane
# whose surface cannot express it, and neither is the same as our missing case.
# --------------------------------------------------------------------------
COVERAGE_KINDS = {
    "recorded": (
        "The lane's committed corpus contains a mutant for this spec. This is "
        "the only verdict that supports a head-to-head rate."
    ),
    "expressible-not-recorded": (
        "The lane's own measured expressibility (S8) says its surface CAN "
        "carry this obligation, but no mutant exists in its corpus. This is "
        "an open action item, NOT a limitation of the model -- and it must "
        "never be reported as n/a."
    ),
    "constitutional-n-a": (
        "The lane's surface has no construct of this kind, so the defect is "
        "not expressible at that level of the stack. A genuine model claim."
    ),
    "uncompared": (
        "The lane was never run against this class and has no cell for it. "
        "A missing cell is not an n/a cell; it is an unmeasured gap."
    ),
    "blocked-by-suite": (
        "OUR base kernels lack the construct this defect needs, so neither "
        "choreo nor any lane can generate the mutant.  This is not a lane's "
        "corpus gap and not a model claim -- it is a hole in the benchmark "
        "that must be filled with a new base case before the unit counts."
    ),
}


def lane_capability() -> dict:
    """Per lane, which obligation classes its own S8 says it can express.

    S8 is MEASURED for `elem` and `shape` (the lane generates real guards or
    rejects real mutants) and structural for `loop`/`hw`. Either way it is the
    lane's own declaration, not our guess, so it is the right thing to test a
    missing mutant against.
    """
    cap: dict[str, dict[str, str]] = {}
    for lane, path in LANE_STATS.items():
        if not path.exists():
            cap[lane] = {}
            continue
        s8 = json.loads(path.read_text()).get("S8_expressibility") or {}
        row = {}
        for obl, tally in s8.items():
            yes = tally.get("yes", 0)
            partial = tally.get("partial", 0)
            no = tally.get("no", 0)
            # A row is `yes` only if EVERY probed kernel expressed it; a mixed
            # row (yes>0 and no>0, e.g. mlir-linalg shape = 5/2) is `partial`,
            # never `yes`.
            if yes and not partial and not no:
                row[obl] = "yes"
            elif yes or partial:
                row[obl] = "partial"
            else:
                row[obl] = "no"
        cap[lane] = row
    return cap


def lane_s8_raw() -> dict:
    """The raw S8 tallies, carried into the ledger as the evidence for the
    capability verdicts so a reader can re-derive them without the lane."""
    out = {}
    for lane, path in LANE_STATS.items():
        if not path.exists():
            continue
        out[lane] = json.loads(path.read_text()).get("S8_expressibility") or {}
    return out


def choreo_operators() -> list[dict]:
    """Every choreo mutation operator, from the generator source.

    The manifest records what was generated; the operator list records what the
    generator CAN generate, which is what a dedup has to reason about. The two
    agree today, and this reads the source so a disagreement becomes visible
    instead of being averaged away.
    """
    import importlib.util
    src = ROOT / "choreo" / "mutations.py"
    spec = importlib.util.spec_from_file_location("choreo_mutations", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["choreo_mutations"] = mod
    spec.loader.exec_module(mod)
    out = []
    for cls, ops in mod.ALL.items():
        for m in ops:
            out.append({
                "class": cls,
                "spec_id": m.spec_id,
                "category": m.category,
                "case": m.case,
                "mutant_id": m.id,
                "desc": m.desc,
                "edits": [[o, n] for o, n, _ in m.edits],
            })
    return out


def _normalize(edits) -> tuple:
    """Collapse integer literals so constant-only variants collide."""
    return tuple(
        (re.sub(r"\b\d+\b", "N", o), re.sub(r"\b\d+\b", "N", n))
        for o, n in edits)


def load_choreo():
    reg = json.loads(REGISTRY.read_text())["specs"]
    man = json.loads(MANIFEST.read_text())["mutants"]
    return reg, man


def diff_body(diff: str | None) -> str:
    """The added/removed lines of a unified diff, blank lines dropped."""
    keep = []
    for line in (diff or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line[:1] in "+-":
            keep.append(line)
    return "\n".join(keep)


def dedup(ops):
    """Classify meaningless replication. Three kinds, not one.

    `exact-duplicate`
        Two specs inject the identical edit into the identical base kernel.
        Nothing distinguishes the two mutants, so one is redundant. Evidence:
        5 pairs, all `M1.5`/`M1.14` plus `M1.2`/`M1.9`.

    `constant-only-variant`
        One spec, one kernel, one RULE, several integer constants (K+2 / K+4 /
        K+12; 7 / 9 / 31 tiles). Same mutation, different value -- the owner's
        "index minus 1 vs index minus 2" case. Beyond the first, these re-test
        one claim and inflate the denominator; keep one and record the rest.

    `mechanism-mismatch`
        The edit does not perturb the relation the spec names (an extent change
        filed as a stride defect; a tile-coordinate move filed as an element-
        index move). This is a validity defect, not redundancy: the spec is not
        actually covered, so its mutants are evidence for a DIFFERENT spec.

    The test is case-aware throughout. The same edit on a different kernel is
    legitimate coverage -- a different shape exercises a different path -- so it
    is never flagged. Measured 2026-09-15: 22 of the 26 same-diff pairs are of
    that benign kind.
    """
    findings: list[dict] = []

    # ---- exact duplicates: same category + case + edit under 2+ specs ----
    exact: dict[tuple, list] = collections.defaultdict(list)
    for o in ops:
        exact[(o["category"], o["case"], tuple(map(tuple, o["edits"])))].append(o)
    for key, group in exact.items():
        specs = sorted({o["spec_id"] for o in group})
        if len(specs) < 2:
            continue
        findings.append({
            "kind": "exact-duplicate",
            "category": key[0],
            "case": key[1],
            "specs": specs,
            "mutant_ids": sorted(o["mutant_id"] for o in group),
            "edit": [list(e) for e in key[2]],
            "n_redundant": len(group) - 1,
            "disposition": (
                "one mutant per cluster is redundant; retire the spec whose "
                "declared mechanism the edit does NOT actually implement"
            ),
        })

    # Mutants already accounted for as exact duplicates, so the weaker
    # constant-only statement is not reported twice for the same mutants.
    exact_mutants = {m for f in findings if f["kind"] == "exact-duplicate"
                     for m in f["mutant_ids"]}

    # ---- constant-only variants: same spec + case + normalised edit ----
    const: dict[tuple, list] = collections.defaultdict(list)
    for o in ops:
        const[(o["spec_id"], o["category"], o["case"],
               _normalize(o["edits"]))].append(o)
    for key, group in const.items():
        if len(group) < 2:
            continue
        # A cluster already reported as an exact duplicate is not re-reported
        # here; the exact finding is the stronger statement.
        if all(o["mutant_id"] in exact_mutants for o in group):
            continue
        findings.append({
            "kind": "constant-only-variant",
            "spec": key[0],
            "category": key[1],
            "case": key[2],
            "mutant_ids": sorted(o["mutant_id"] for o in group),
            "edit": [list(e) for e in group[0]["edits"]],
            "n_redundant": len(group) - 1,
            "disposition": (
                "one rule, several constants: keep the first, record the rest "
                "as variants-of, unless a value is separately attested by a "
                "cited taxonomy"
            ),
        })

    # ---- cross-spec same-site: 2+ specs, one declaration, different constants
    # The same declaration perturbed by several specs with different constants.
    # Neither an exact duplicate (the constants differ) nor a single-spec
    # constant-only family (the spec_ids differ), so it needs its own kind:
    # these are competing claims on one site and must be adjudicated, not
    # counted as independent coverage.
    site: dict[tuple, list] = collections.defaultdict(list)
    for o in ops:
        site[(o["category"], o["case"], _normalize(o["edits"]))].append(o)
    exact_keys = {(f["category"], f["case"])
                  for f in findings if f["kind"] == "exact-duplicate"}
    for key, group in site.items():
        specs = sorted({o["spec_id"] for o in group})
        if len(specs) < 2:
            continue
        # One identical edit shared by many specs is the exact-duplicate case.
        if len({tuple(map(tuple, o["edits"])) for o in group}) == 1:
            continue
        findings.append({
            "kind": "cross-spec-same-site",
            "category": key[0],
            "case": key[1],
            "specs": specs,
            "mutant_ids": sorted(o["mutant_id"] for o in group),
            "site": key[2][0][1] if key[2] else None,
            "constants": sorted({c for o in group for pair in o["edits"]
                                 for c in (pair[0], pair[1])}),
            "n_redundant": len(specs) - 1,
            "disposition": (
                "several specs claim one declaration with different "
                "constants; adjudicate which spec owns the site and fold the "
                "others in as variants-of, or re-site them on a distinct "
                "declaration"
            ),
        })

    # ---- mechanism mismatch: the edit cannot implement the named relation ----
    # Explicit, spec-keyed rules.  A prose regex over `desc` was too fragile:
    # it silently stopped matching when a description was reworded.  Each entry
    # names the spec, the mutants that are mis-filed (None = all of them), and
    # the reason the edit cannot implement the declared mechanism.
    mismatch_rules = {
        "M1.4": (
            ("M1.s4.rl1.tile", "M1.s4.tp1.tile"),
            "declared as a non-contiguous/stride defect, but these edits change "
            "a declared EXTENT (`shared f32 [1,1,64,1]` -> `[1,1,65,1]`), "
            "which is a shape relation (M2).  As M1 mutants they are evidence "
            "for M2, not M1.",
        ),
        "M1.9": (
            None,
            "declared as a base offset without shrinking the extent, but the "
            "suite has no subspan-with-base construct (`span(d)` takes an "
            "extent only), so every realisation is an element-index "
            "perturbation: the spec is vacuously defined.",
        ),
    }
    for o in ops:
        rule = mismatch_rules.get(o["spec_id"])
        if not rule:
            continue
        only, why = rule
        if only is not None and o["mutant_id"] not in only:
            continue
        edits = o["edits"]
        if True:
            findings.append({
                "kind": "mechanism-mismatch",
                "spec": o["spec_id"],
                "category": o["category"],
                "case": o["case"],
                "mutant_ids": [o["mutant_id"]],
                "edit": [list(e) for e in edits],
                "n_redundant": 0,
                "why": why,
                "disposition": "re-realise against the named relation, or call "
                               "it a suite-surface gap and stop counting it",
            })
    return sorted(findings, key=lambda d: (d["kind"], str(d.get("spec") or d.get("specs"))))


def unit_of(spec):
    """The semantic mutation unit: class + operand x axis + direction + site.

    The registry carries the mechanism in `desc`; the operand/axis/site are
    recoverable from the mutant diffs.  For the ledger we key on the spec and
    record the mechanism string, which is the part a human must adjudicate.
    """
    return f"{spec['spec_id']} :: {spec['desc']}"


def build():
    reg, man = load_choreo()
    ops = choreo_operators()
    ops_by_spec = collections.defaultdict(list)
    for o in ops:
        ops_by_spec[o["spec_id"]].append(o)
    cap = lane_capability()

    by_spec = collections.defaultdict(list)
    for m in man:
        by_spec[m["spec_id"]].append(m)

    units = []
    lane_totals = collections.Counter()          # recorded
    lane_achievable = collections.Counter()      # recorded + expressible
    undecided = []
    for s in sorted(reg, key=lambda r: (r["cls"], int(r["spec_id"].split(".")[1]))):
        sid, cls = s["spec_id"], s["cls"]
        got = by_spec.get(sid, [])
        rec = {
            "unit": sid,
            "class": cls,
            "path_class": s["path"],
            "mechanism": s["desc"],
            "choreo": {
                "realized": bool(got),
                "n_mutants": len(got),
                "cases": sorted({m["case"] for m in got}),
                "admissible": bool(s["admissible"]),
                "registry_status": s["status"],
                "prohibition": s["prohibition"] or None,
                "note": s["note"] or None,
            },
            "lanes": {},
        }
        note = s["note"] or ""
        suite_gap = (not got) and ("MISSING SURFACE" in note)
        for lane in LANES:
            # Did this lane demonstrably realise the unit?  Check that FIRST.
            # A lane's committed mutant outranks our own suite-gap label: if a
            # SOTA lane ships a mutant for this site, the site is not blocked,
            # it is *we* who lack the base case (see M3.11 / triton family 2).
            recorded = (
                (lane == "triton" and sid in TRITON_FAMILY)
                or (lane == "mlir-low" and sid in MLIR_LOW_SPEC)
                or (lane == "mlir-linalg" and sid in MLIR_LINALG_SPEC)
                or (lane == "iree" and sid in IREE_SPEC)
            )
            if recorded:
                if lane == "triton":
                    f, sym, _fam = TRITON_FAMILY[sid]
                    ev = f"{f}::{sym}"
                elif lane == "mlir-low":
                    ev = f"mlir-shared/mutate.py Structural({MLIR_LOW_SPEC[sid]})"
                elif lane == "mlir-linalg":
                    ev = (f"mlir-shared/mutate.py M2 spec "
                          f"{sid.split('.')[1]} ({MLIR_LINALG_SPEC[sid]})")
                else:
                    ev = f"iree/mutants/M2/* ({IREE_SPEC[sid]})"
                rec["lanes"][lane] = {"status": "recorded", "evidence": ev}
                lane_totals[lane] += 1
                lane_achievable[lane] += 1
                continue

            # A spec whose base construct our own kernels lack cannot be
            # generated by us.  It is not a lane's corpus gap, so it takes
            # precedence over the lane's S8 verdict.
            if suite_gap:
                rec["lanes"][lane] = {
                    "status": "blocked-by-suite",
                    "annotation": COVERAGE_KINDS["blocked-by-suite"],
                    "evidence": ("choreo/raw/spec_registry.json note: "
                                 "MISSING SURFACE"),
                    "needs": "a new base case -- see the suite-gap register",
                }
                continue
            # Not recorded, not blocked.  Why?  The lane's own measured
            # expressibility is the authority, not our reading of the prose.
            obl = OBLIGATION_OF[cls]
            ability = cap.get(lane, {}).get(obl)
            if ability in ("yes", "partial"):
                rec["lanes"][lane] = {
                    "status": "expressible-not-recorded",
                    "declared_ability": ability,
                    "evidence": (f"{LANE_STATS[lane].relative_to(ROOT)} "
                                 f"S8_expressibility[{obl}] = {ability}"),
                    "note": ("the lane's own S8 says this obligation is "
                             "expressible, but no mutant for this spec exists "
                             "in its corpus -- an open action item, not a "
                             "limitation"),
                }
                lane_achievable[lane] += 1
            elif ability == "no":
                na_kind = LANE_NA_KIND.get((lane, cls), "constitutional")
                rec["lanes"][lane] = {
                    "status": "constitutional-n-a",
                    "kind": na_kind,
                    "annotation": NA_KINDS.get(na_kind, ""),
                    "evidence": (f"{LANE_STATS[lane].relative_to(ROOT)} "
                                 f"S8_expressibility[{obl}] = no"),
                    "needs": (
                        "nothing the lane can add: the construct does not "
                        "exist at this level of the stack.  Do not count this "
                        "as a miss; do not report it as an n/a rate either."
                    ),
                }
            else:
                rec["lanes"][lane] = {
                    "status": "uncompared",
                    "annotation": COVERAGE_KINDS["uncompared"],
                    "evidence": f"{lane}: no S8 row and no {cls} cell",
                    "needs": (
                        "the lane must be run against this class, or the axis "
                        "must declare the class out of scope"
                    ),
                }

        # The suite-hole correction: a registry entry whose own note says
        # MISSING SURFACE is not a model claim.
        if suite_gap:
            rec["na_kind"] = "suite-surface"
            rec["na_kind_correction"] = (
                "registry says prohibition='absent'; the note says MISSING "
                "SURFACE.  This is our gap, not the model's."
            )
        elif not got and (s["prohibition"] or "") == "absent":
            rec["na_kind"] = "model-prohibited"
        elif not got:
            rec["na_kind"] = f"model-{s['prohibition'] or 'unknown'}"
        rec["choreo"]["n_operators"] = len(ops_by_spec.get(sid, []))
        rec["choreo"]["n_distinct_realisations"] = len({
            (o["category"], o["case"], tuple(map(tuple, o["edits"])))
            for o in ops_by_spec.get(sid, [])})
        units.append(rec)

    dups = dedup(ops)

    choreo_realized = sum(1 for u in units if u["choreo"]["realized"])
    recorded_any = sum(1 for u in units
                       if any(v["status"] == "recorded"
                              for v in u["lanes"].values()))
    achievable_any = sum(
        1 for u in units
        if any(v["status"] in ("recorded", "expressible-not-recorded")
               for v in u["lanes"].values()))
    # The apples-to-apples denominator: of the specs choreo actually realises,
    # how many could a SOTA lane also realise?  `achievable_any` counts specs
    # choreo does NOT realise too, so dividing it by `choreo_realized` can
    # exceed 100% (it did: 72/49 = 146.9%).  Split the two questions.
    achievable_choreo = sum(
        1 for u in units if u["choreo"]["realized"]
        and any(v["status"] in ("recorded", "expressible-not-recorded")
                for v in u["lanes"].values()))
    suite_gaps = [u["unit"] for u in units if u.get("na_kind") == "suite-surface"]
    dup_specs = sorted({s for d in dups
                        for s in (d.get("specs") or [d.get("spec")])})

    # Denominator health (requirement: annotate what a lane cannot catch and
    # flag every cell below the power floor, so a rate is never printed on an
    # under-powered cell).
    cells = lane_cells()
    denoms = []
    for lane in ("choreo",) + LANES:
        for cls in ("M1", "M2", "M3", "M4"):
            n = cells.get(lane, {}).get(cls, 0)
            denoms.append({
                "lane": lane,
                "class": cls,
                "n_mutants": n,
                "power_floor": POWER_FLOOR,
                "status": ("absent" if n == 0
                           else "under-powered" if n < POWER_FLOOR
                           else "ok"),
                "annotation": (
                    "no cell: the lane was never run against this class -- "
                    "unmeasured, not n/a" if n == 0 else
                    f"{n} < {POWER_FLOOR}: a rate must not be reported from "
                    "this cell" if n < POWER_FLOOR else
                    "meets the power floor"
                ),
            })

    # Which lane x class the lane genuinely CANNOT catch, and why.  This is
    # the "annotate the unable-to-catch cases" register: it names the model
    # property behind each empty cell so an empty cell is never mistaken for
    # an unmeasured one.
    lane_cls_recorded = collections.Counter()
    for u in units:
        for lane, v in u["lanes"].items():
            if v["status"] == "recorded":
                lane_cls_recorded[(lane, u["class"])] += 1
    na_annotations = []
    for lane in LANES:
        for cls in ("M1", "M2", "M3", "M4"):
            if lane_cls_recorded[(lane, cls)]:
                continue
            kind = LANE_NA_KIND.get((lane, cls))
            if kind is None:
                continue          # absent cell with no declared reason =
                                  # unmeasured gap, already flagged in denoms
            na_annotations.append({
                "lane": lane,
                "class": cls,
                "kind": kind,
                "annotation": NA_KINDS.get(kind, ""),
                "cell_status": next(d["status"] for d in denoms
                                    if d["lane"] == lane and d["class"] == cls),
            })

    return {
        "generated_by": "schema/build_correspondence.py",
        "n_target_per_class": N_TARGET,
        "na_kinds": NA_KINDS,
        "coverage_kinds": COVERAGE_KINDS,
        "summary": {
            "n_specs": len(units),
            "choreo_realized": choreo_realized,
            "choreo_unrealized": len(units) - choreo_realized,

            # The headline the paper must use. `recorded` is what exists to
            # compare against today; `achievable` is what the lanes' own
            # measured expressibility says could exist.
            "recorded_by_a_sota_lane": recorded_any,
            "recorded_over_choreo": round(recorded_any / choreo_realized, 4),
            "recorded_over_all_specs": round(recorded_any / len(units), 4),
            "achievable_by_a_sota_lane": achievable_any,
            "achievable_over_all_specs": round(achievable_any / len(units), 4),
            "achievable_among_choreo_realized": achievable_choreo,
            "achievable_over_choreo": round(achievable_choreo / choreo_realized, 4),
            "choreo_only_and_achievable": achievable_choreo - recorded_any,
            "choreo_only_and_recorded": recorded_any,
            "constitutional_n_a_anywhere": sum(
                1 for u in units if any(
                    v["status"] == "constitutional-n-a"
                    for v in u["lanes"].values())),
            "lane_recorded_counts": dict(lane_totals),
            "lane_achievable_counts": dict(lane_achievable),
            # The honest per-lane gap: of what a lane's own surface can carry,
            # how much is actually in its corpus?  This is the number to quote;
            # the union-across-lanes metric saturates at 100% by construction
            # (every class has SOME lane that declares the obligation
            # expressible), so it is a plausibility check, not a rate.
            "lane_recorded_over_achievable": {
                lane: round(lane_totals[lane] / lane_achievable[lane], 4)
                for lane in LANES if lane_achievable[lane]},
            "union_metric_caveat": (
                "`achievable_by_a_sota_lane` is the union over lanes and reaches "
                "100% because each obligation class is declared expressible by "
                "at least one lane.  It answers 'could ANY lane express this?' "
                "and must not be quoted as a per-lane rate.  Quote "
                "`lane_recorded_over_achievable` instead."
            ),
            "n_duplicate_findings": len(dups),
            "n_exact_duplicate_mutants": sum(
                d["n_redundant"] for d in dups if d["kind"] == "exact-duplicate"),
            "n_cross_spec_same_site": sum(
                1 for d in dups if d["kind"] == "cross-spec-same-site"),
            "n_constant_only_mutants": sum(
                d["n_redundant"] for d in dups
                if d["kind"] == "constant-only-variant"),
            "n_mechanism_mismatches": sum(
                1 for d in dups if d["kind"] == "mechanism-mismatch"),
            "specs_with_duplicates": dup_specs,

            "n_under_powered_cells": sum(
                1 for d in denoms if d["status"] == "under-powered"),
            "n_absent_cells": sum(1 for d in denoms if d["status"] == "absent"),
            "suite_surface_gaps": suite_gaps,
            "n_suite_surface_gaps": len(suite_gaps),
            "undecided_lanes": undecided,
        },
        "denominators": denoms,
        "duplicates": dups,
        "lane_na_annotations": na_annotations,
        "s8_expressibility": lane_s8_raw(),
        "cross_label_spec": CROSS_LABEL_SPEC,
        "units": units,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the emitted file is stale")
    args = ap.parse_args()
    out = ROOT / "schema" / "mutation-correspondence.json"
    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.check:
        if not out.exists() or out.read_text() != text:
            print("STALE: schema/mutation-correspondence.json differs from a "
                  "fresh build.  Re-run without --check.", file=sys.stderr)
            return 1
        print("ok: correspondence ledger is current.")
        return 0
    out.write_text(text)
    s = payload["summary"]
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  specs                          {s['n_specs']}")
    print(f"  choreo realises                {s['choreo_realized']}")
    print()
    print("  -- the headline: recorded vs achievable --")
    print(f"  RECORDED by a SOTA lane        "
          f"{s['recorded_by_a_sota_lane']}/{s['n_specs']} = "
          f"{s['recorded_over_all_specs']:.1%} of all specs")
    print(f"    ... of choreo's {s['choreo_realized']} realized specs  "
          f"{s['recorded_by_a_sota_lane']}/{s['choreo_realized']} = "
          f"{s['recorded_over_choreo']:.1%}")
    print(f"  ACHIEVABLE (each lane's own S8)")
    print(f"    ... all specs                "
          f"{s['achievable_by_a_sota_lane']}/{s['n_specs']} = "
          f"{s['achievable_over_all_specs']:.1%}")
    print(f"    ... choreo's realized specs  "
          f"{s['achievable_among_choreo_realized']}/{s['choreo_realized']} = "
          f"{s['achievable_over_choreo']:.1%}")
    print(f"  choreo-only & recorded         {s['choreo_only_and_recorded']}")
    print(f"  choreo-only & achievable       "
          f"{s['choreo_only_and_achievable']}")
    print(f"  constitutional n/a anywhere    "
          f"{s['constitutional_n_a_anywhere']}")
    print(f"  suite-surface gaps             {s['n_suite_surface_gaps']}")
    print()
    print("  -- denominator health (power floor "
          f"{POWER_FLOOR}) --")
    print(f"  absent cells                   {s['n_absent_cells']}")
    print(f"  under-powered cells            {s['n_under_powered_cells']}")
    for d in payload["denominators"]:
        if d["status"] != "ok":
            n = d["n_mutants"]
            print(f"    {d['lane']:<13} {d['class']}  "
                  f"{'--' if n == 0 else n:>3}  {d['status']}")
    print()
    print("  -- dedup (real) --")
    print(f"  duplicate findings             {s['n_duplicate_findings']}")
    print(f"    exact-duplicate mutants      {s['n_exact_duplicate_mutants']}")
    print(f"    cross-spec same-site         {s['n_cross_spec_same_site']}")
    print(f"    constant-only redundant      {s['n_constant_only_mutants']}")
    print(f"    mechanism mismatches         {s['n_mechanism_mismatches']}")
    print(f"   specs involved                {', '.join(s['specs_with_duplicates'])}")
    for lane in sorted(s["lane_recorded_counts"]):
        print(f"    {lane:<14} recorded {s['lane_recorded_counts'][lane]:<3} "
              f"achievable {s['lane_achievable_counts'].get(lane, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
