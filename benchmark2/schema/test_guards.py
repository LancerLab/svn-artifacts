#!/usr/bin/env python3
"""Negative controls for the guards in `check_class_axis.py`.

A guard is only a guard if it can fail. Every case here mutates exactly one
input, runs ONE guard by id in a fresh interpreter, requires that the guard
fails naming the defect, and then restores the input byte-identically.

Three things about the shape of this file are deliberate.

  * **One guard per case, addressed by id.** `--guard <id>` selects exactly one
    guard (`_matches` is exact-or-prefix), so a case cannot pass because a
    sibling guard happened to notice. That is also why the ids are paths: a
    control is meaningless if it cannot name what it controlled.

  * **A fresh interpreter per case.** `method_taxonomy` reads its JSON at import
    and `check()` closes over that module state, so a mutation applied in-process
    would be judged against the value loaded a moment earlier. This is the same
    trap recorded for `choreo/__pycache__`: never let a gate judge a file it did
    not just read.

  * **The coverage assertion at the bottom.** Every id in `--list` must be either
    controlled here or listed in `UNCONTROLLED` with a reason. A new guard with
    no control fails this file, so the test suite cannot silently fall behind
    the registry -- which is the failure mode that let four `families.*` ids sit
    in `RESERVED` for a day while `--list` looked complete.

Run: python3 schema/test_guards.py [-v]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "schema" / "check_class_axis.py"
AXIS = ROOT / "schema" / "class-axis.json"
SCHEMA = ROOT / "schema" / "record-schema.json"
TAXONOMY = ROOT / "schema" / "method-taxonomy.json"
MANIFEST = ROOT / "choreo" / "raw" / "mutant_manifest.json"
REGISTRY = ROOT / "choreo" / "raw" / "spec_registry.json"
MUTATIONS = ROOT / "choreo" / "mutations.py"
GEN_MUTANTS = ROOT / "choreo" / "gen_mutants.py"
RENDER = ROOT / "render.py"
TRITON = ROOT / "triton" / "collect.py"
PATCH = ROOT / "schema" / "PATCH-v2.1.md"
# The plans live beside the artifacts tree, not inside it:
#   <svn>/eurosys27/plan/  and  <svn>/svn-artifacts/benchmark2/schema/
PLAN = ROOT.parent.parent / "eurosys27" / "plan"
DOC = PLAN / "plan-a-execution.md"


def run(guard: str) -> str:
    """Run one guard in a fresh interpreter; return its combined output."""
    p = subprocess.run([sys.executable, str(CHECKER), "--guard", guard],
                       cwd=str(ROOT), capture_output=True, text=True)
    return p.stdout + p.stderr


def registry() -> dict:
    d = json.loads(REGISTRY.read_text())
    return {s["spec_id"]: s for s in d["specs"]}


def generatable(spec_id: str) -> bool:
    e = registry().get(spec_id) or {}
    return e.get("status") == "implemented" and e.get("path") != "avoided"


# ---------------------------------------------------------------------------
# Mutations. Each takes the parsed document and edits it in place; the harness
# owns writing, running and restoring.
# ---------------------------------------------------------------------------
def m_second_family(doc):
    """`M1.1` becomes a realisation of two families."""
    fams = doc["families"]
    for f in fams:
        if f != "M1-a" and "M1.1" not in fams[f]["spec_ids"]:
            fams[f]["spec_ids"].append("M1.1")
            return
    raise AssertionError("no second family to add M1.1 to")


def m_shared_sole(doc):
    """Two families are left holding the same only generatable spec."""
    fams = doc["families"]
    singles = [(f, [s for s in fams[f]["spec_ids"] if generatable(s)])
               for f in sorted(fams)]
    singles = [(f, s) for f, s in singles if len(s) == 1]
    if len(singles) < 2:
        raise AssertionError(f"need two one-spec families, found {singles}")
    (_, (src,)), (dst, _) = singles[0], singles[1]
    fams[dst]["spec_ids"] = [src]
    return dst, src


def m_dead_with_operator(doc):
    """A spec called dead that does have an operator."""
    impl = [s for s in sorted(doc["dead_declarations"]["spec_ids"])]
    if not impl:
        raise AssertionError("dead_declarations is empty")
    # A spec that is dead AND implemented is the contradiction under test.
    live = [s for s in _all_declared(doc) if generatable(s)]
    doc["dead_declarations"]["spec_ids"] = sorted(set(impl) | {live[0]})
    return live[0]


def m_leading_zero(doc):
    """One spec_id is spelled `M1.01` where the registry says `M1.1`."""
    for f, d in doc["families"].items():
        for i, s in enumerate(d["spec_ids"]):
            head, _, tail = s.partition(".")
            if tail.isdigit() and len(tail) == 1:
                d["spec_ids"][i] = f"{head}.0{tail}"
                return s, d["spec_ids"][i]
    raise AssertionError("no one-digit spec_id to pad")


def m_clear_the_ledger(doc):
    """Empty `known_debt`, so every ledgered finding becomes a failure."""
    items = doc["known_debt"]["items"]
    n = len(items)
    doc["known_debt"]["items"] = {}
    return n


def m_break_accounting(doc):
    doc["accounting"]["selected"] += 1


def m_strip_a_drop_family(doc):
    """A drop of a spec that HAS a family, with the family removed."""
    for row in doc["dropped"]:
        if row.get("family"):
            row.pop("family")
            return row["mutant_id"]
    raise AssertionError("no drop carries a family")


def m_move_an_instance(doc):
    """One `M1-a` instance is re-pointed at a spec of another M1 family.

    Membership is read from the taxonomy and the edit is made to the manifest,
    which is the point: the two are separate documents, and a manifest row only
    has a family because the taxonomy says its `spec_id` has one.

    Both halves matter: the row has to leave `M1-a` (so the depth drops) and it
    has to land on a real spec (so the file is still a manifest). Moving it to a
    *sibling* spec inside `M1-a` would change nothing, which is the kind of
    no-op control this file is written to make impossible.
    """
    fams = json.loads(TAXONOMY.read_text())["families"]
    mine = set(fams["M1-a"]["spec_ids"])
    elsewhere: set[str] = set()
    for name, d in fams.items():
        if name.startswith("M1-") and name != "M1-a":
            elsewhere |= set(d["spec_ids"])
    away = sorted(elsewhere - mine)
    if not away:
        raise AssertionError("M1-a owns every M1 spec; no move is possible")
    for row in doc["mutants"]:
        if row.get("class") == "M1" and row.get("spec_id") in mine:
            was = row["spec_id"]
            row["spec_id"] = away[0]
            return was, away[0]
    raise AssertionError(f"no M1 mutant among {sorted(mine)} to move")


def m_declare_out_of_scope(doc):
    doc["lane_scope"].setdefault("choreo", {})["M1-a"] = "absent"


def m_illegal_na(doc):
    """`M1-a` is n/a for choreo with a reason that is not a prohibition, and
    holds nothing -- the only shape in which the reason can be judged."""
    doc["lane_scope"].setdefault("choreo", {})["M1-a"] = "bogus"


def m_unfamiliable_row(doc):
    """An emitted M1 row whose spec belongs to no M1 family."""
    row = json.loads(json.dumps(doc["mutants"][0]))
    row["class"], row["spec_id"] = "M1", "M3.17"
    row["attribution_only"] = None
    doc["mutants"].append(row)
    return row["mutant_id"]


def m_wrong_cell(doc):
    doc["budget"]["classes"]["M1"]["cell"] = 63


def m_obligation_collision(doc):
    """Two classes claim the same obligation."""
    cs = doc["classes"]
    cs[1]["obligation"] = cs[0]["obligation"]


def m_bad_class_enum(doc):
    """The `class` enum in the record schema admits an id the axis does not."""
    want = ["M1", "M2", "M3", "M4"]
    seen = {"hit": False}

    def walk(obj, under_obligation=False):
        if seen["hit"]:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, under_obligation or k == "obligation")
        elif isinstance(obj, list):
            if obj == want and not under_obligation:
                obj[0] = "M5"
                seen["hit"] = True
                return
            for v in obj:
                walk(v, under_obligation)

    walk(doc)
    if not seen["hit"]:
        raise AssertionError("no class enum in record-schema.json")


def _all_declared(doc) -> list[str]:
    out: list[str] = []
    for d in doc["families"].values():
        out += d["spec_ids"]
    return out


# --- text mutations: these are Python or Markdown, so they are edited as
# --- source. The checker never imports the files it reads here (it compiles
# --- `mutations.py` from source and `gen_mutants` is the one exception, whose
# --- edit changes the file's size and so invalidates its `.pyc`).
WITNESS = 'NA_WITNESS = {"M1": "iree", "M2": "triton", "M3": "iree"}'
M1_MINIMAL = ('"M1": ["layer_normalization", "softmax", "relu", "transpose",\n'
              '           "dma_rank5"],')
M1_LEVEL2 = '"M1": ["max_pool2d", "conv2d", "embedding", "batch_norm"],'
SUITE_LINE = 'SUITE = os.path.join(REPO, "benchmark", "choreo")'

# The two tokens the vocabulary guard exists to refuse are BUILT here, never
# written. `guards.vocabulary` scans Python too, so a test that spelled them out
# would be refused by the very guard it is testing -- correctly, and this file
# was. Spelling them would also make the control depend on a self-exemption,
# which is the hole the bounded budget exists to close.
STALE_ORDINAL = "G" + "99"
PHANTOM_ID = "families" + "." + "nonexistent-guard"


def _append_ordinal(text):
    return text + f"\nThe {STALE_ORDINAL} gate must pass before freeze.\n"


def _append_phantom_id(text):
    return text + f"\nSee `{PHANTOM_ID}`.\n"


def _replace_one(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f"expected 1 occurrence of {old[:40]!r}, "
                             f"found {text.count(old)}")
    return text.replace(old, new)


def t_witness_for_m4(text):
    return _replace_one(text, WITNESS, WITNESS[:-1] + ', "M4": "iree"}')


def t_restated_tuple(text):
    return text + '\n_SPARE_CLASSES = ["M1", "M2"]\n'


def t_lane_stops_reading_the_axis(text):
    return text.replace("class_axis", "klass_axis")


def t_superseded_claim(text):
    return text + '\nThe pair "M4", "L" was never a class list.\n'


def t_drop_a_constant(text):
    """Make a name the other guards read disappear from the registry.

    Every occurrence, not just the assignment: the control is "this name is not
    in mutations.py", and renaming only the definition would leave a NameError
    for the exec to trip over instead -- a crash, not a finding.
    """
    if text.count("MINIMAL_SET") < 2:
        raise AssertionError("MINIMAL_SET does not appear as both a "
                            "definition and a use")
    return text.replace("MINIMAL_SET", "MINIMAL_SETX")


def t_empty_m1_coverage(text):
    return _replace_one(_replace_one(text, M1_MINIMAL, '"M1": [],'),
                        M1_LEVEL2, '"M1": [],')


def t_suite_vanishes(text):
    return _replace_one(text, SUITE_LINE, SUITE_LINE.replace('"choreo"',
                                                             '"choreo-x"'))


def _as_text(doc) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Cases: (guard id, what it must refuse, mutation, expected substring)
# ---------------------------------------------------------------------------
CASES = [
    # --- the axis itself ---------------------------------------------------
    ("axis.self-consistency",
     "two classes claiming one obligation",
     ("json", AXIS, m_obligation_collision),
     "class<->obligation is not a bijection"),

    ("axis.schema-enums",
     "a record-schema enum that admits a class the axis does not",
     ("json", SCHEMA, m_bad_class_enum),
     "expected ['M1', 'M2', 'M3', 'M4']"),

    ("axis.restated-constants",
     "an `n/a` witness for a class no lane holds at n/a",
     ("text", RENDER, t_witness_for_m4),
     "has an M4 key"),

    ("axis.no-restated-tuples",
     "a class sequence hard-coded outside schema/",
     ("text", RENDER, t_restated_tuple),
     "restates a sequence of class ids"),

    ("lanes.declarations",
     "a lane that stopped reading the axis for its class set",
     ("text", TRITON, t_lane_stops_reading_the_axis),
     "does not import schema.class_axis"),

    ("docs.no-superseded-claim",
     "a document still presenting `L` as a mutation class",
     ("text", PATCH, t_superseded_claim),
     "carries a superseded axis claim"),

    ("mutants.registry",
     "a registry missing a name the other guards read",
     ("text", MUTATIONS, t_drop_a_constant),
     "has no ['MINIMAL_SET']"),

    ("mutants.edits-apply",
     "an operator whose declared base kernel is not in the suite",
     ("text", GEN_MUTANTS, t_suite_vanishes),
     "does not exist"),

    # --- per-class ---------------------------------------------------------
    ("m1.coverage",
     "a class left with an empty coverage set",
     ("text", MUTATIONS, t_empty_m1_coverage),
     "has an EMPTY coverage set"),

    ("m1.cell",
     "a class declaring fewer kernels than the budget needs",
     ("text", MUTATIONS, t_empty_m1_coverage),
     "declares only 0 categor"),

    # --- the taxonomy ------------------------------------------------------
    ("families.partition",
     "a spec_id filed under two families",
     ("json", TAXONOMY, m_second_family),
     "is a realisation of"),

    ("families.no-shared-instance",
     "two families left holding one realisation",
     ("json", TAXONOMY, m_shared_sole),
     "is the SOLE implemented realisation of"),

    ("families.dead-vs-unwritten",
     "a spec called dead that has an operator",
     ("json", TAXONOMY, m_dead_with_operator),
     "is not a subset of the unwritten specs"),

    ("families.ids-canonical",
     "an id padded where the registry does not pad it",
     ("json", TAXONOMY, m_leading_zero),
     "leading zero"),

    ("families.admissibility",
     "an admissibility gap with the ledger that excuses it removed",
     ("json", TAXONOMY, m_clear_the_ledger),
     "has NO admissible declaration"),

    ("families.select-conformance",
     "an accounting block that does not add up",
     ("json", MANIFEST, m_break_accounting),
     "does not close"),

    ("families.select-conformance",
     "a drop of a filed spec that names no family",
     ("json", MANIFEST, m_strip_a_drop_family),
     "is dropped with no family"),

    ("m1.a.instances",
     "a family one instance short of N",
     ("json", MANIFEST, m_move_an_instance),
     "holds 7 instances of 8"),

    ("m1.a.instances",
     "instances in a cell the lane declares out of scope",
     ("json", TAXONOMY, m_declare_out_of_scope),
     "is OUT of scope for this lane but the corpus holds"),

    ("m1.a.instances",
     "an n/a whose reason is not a prohibition",
     ("json", TAXONOMY, m_illegal_na),
     "is n/a with reason 'bogus'"),

    ("m1.instances",
     "a class cell that is not the sum of its families",
     ("json", MANIFEST, m_unfamiliable_row),
     "families hold"),

    ("m1.instances",
     "a declared cell that is not families x N",
     ("json", TAXONOMY, m_wrong_cell),
     "the budget declares the M1 cell as 63"),

    ("guards.vocabulary",
     "an ordinal re-introduced into a plan document",
     ("text", DOC, _append_ordinal),
     "is a position, not an identity"),

    ("guards.vocabulary",
     "a citation of an id that was never registered",
     ("text", DOC, _append_phantom_id),
     "is not a registered guard id"),

    ("guards.vocabulary",
     "a launch-geometry file that stopped saying what `G` means",
     ("text", PLAN / "hardware-constraint-inventory.md",
      lambda t: t.replace("launch geometry", "geometry")
                 .replace("launch constraint", "constraint")
                 .replace("launch-constraint", "constraint")),
     "never says what `G` means"),
]

# A missing manifest and a missing taxonomy are not expressible as a text edit,
# so they are handled separately below.
VACUOUS = [
    ("families.select-conformance", MANIFEST,
     "would pass vacuously"),
    # The SAME mutation, addressed to a different guard.
    #
    # `choreo` is the only lane with a corpus, so `family_instances` skips every
    # other lane in silence: its per-lane loop iterates `_lane_manifests()`,
    # which by design omits a lane that has not emitted one ("not_ready, not
    # wrong"). That silence is correct for a lane still unwritten, but it means
    # the branch that fires when NO corpus exists is the branch that is normally
    # unreachable -- and a guard cannot refuse data it never read. Without this
    # control, deleting the last manifest would turn all 31 cell guards into
    # 31 vacuous passes and nothing would say so.
    ("m1.a.instances", MANIFEST,
     "was never looked at"),
]

# Every id in `--list` must be accounted for. The ids that carry logic of their
# own have a mutation above. A per-family instance cell does not: all 31 are one
# function called with a different family, and its three branches (short of N,
# instances in an n/a cell, an n/a whose reason is not a prohibition) are
# controlled on `m1.a.instances`. Spelling that out here means adding a new
# piece of logic without a control fails this file.
SHARED_BODY = (
    "the 31 `m<C>.<family>.instances` cells share `family_instances`; its "
    "branches are controlled on m1.a.instances"
)

UNCONTROLLED: dict[str, str] = {
    f"m{c}.{x}.instances": SHARED_BODY
    for c in "1234" for x in "abcdefgh"
    if not (c == "1" and x == "a")
}
UNCONTROLLED.update({
    "m2.instances": SHARED_BODY,
    "m3.instances": SHARED_BODY,
    "m4.instances": SHARED_BODY,
    "m2.coverage": SHARED_BODY,
    "m3.coverage": SHARED_BODY,
    "m4.coverage": SHARED_BODY,
    "m2.cell": SHARED_BODY,
    "m4.cell": SHARED_BODY,
    # Red on the real tree, which is itself a demonstration that it is not
    # vacuous -- and in each case the live finding names the defect. An
    # isolated control would mean committing the drift the guard exists to
    # catch.
    "m3.cell": "already red on the real tree: M3 realises 2 of the 4 "
               "categories its cell of 64 needs",
    "lanes.stats-conformance": "already red on the real tree",
    "corpus.declared-files": "already red on the real tree",
    "mutants.no-duplicate-injection": "already red on the real tree",
})


def ids_from_checker() -> list[str]:
    out = subprocess.run([sys.executable, str(CHECKER), "--list"],
                         cwd=str(ROOT), capture_output=True, text=True)
    return [ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()]


def coverage() -> list[str]:
    """Ids in `--list` that no case here would catch."""
    controlled: set[str] = set()
    for gid_, _, _, _ in CASES:
        controlled.add(gid_)
    for gid_, _, _ in VACUOUS:
        controlled.add(gid_)
    missing = [i for i in ids_from_checker()
               if i not in controlled and i not in UNCONTROLLED]
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    fails: list[str] = []
    for gid_, what, (kind, path, mut), expect in CASES:
        orig = path.read_text()
        try:
            if kind == "json":
                doc = json.loads(orig)
                mut(doc)
                path.write_text(_as_text(doc))
            else:
                path.write_text(mut(orig))
            out = run(gid_)
        except Exception as exc:                 # noqa: BLE001 - report, keep going
            out = f"control raised {exc!r}"
        finally:
            path.write_text(orig)
        if path.read_text() != orig:
            fails.append(f"{gid_}: {path.name} was NOT restored byte-identically")
            continue
        if "FAIL" not in out or expect not in out:
            fails.append(f"{gid_} does not refuse {what}: expected "
                         f"{expect!r} in its output")
            if args.verbose:
                print(f"--- {gid_} / {what}\n{out}")
        else:
            print(f"ok   {gid_:<28} refuses {what}")

    for gid_, path, expect in VACUOUS:
        orig = path.read_text()
        stash = path.with_suffix(path.suffix + ".control")
        try:
            shutil.move(str(path), str(stash))
            out = run(gid_)
        finally:
            shutil.move(str(stash), str(path))
        if "FAIL" not in out or expect not in out:
            fails.append(f"{gid_} passes with no manifest at all: expected "
                         f"{expect!r} in its output")
        else:
            print(f"ok   {gid_:<28} refuses {expect}")

    # Every guard id must be reachable. A `--guard` that matches nothing exits
    # 2 and prints no FAIL, so this is the one way a control can lie.
    dead = [g for g, _, _, _ in CASES if "no guard matches" in run(g)]
    for g in dead:
        fails.append(f"{g} is not a registered guard id")
    missing = coverage()
    for m in missing:
        fails.append(f"{m} has no negative control and no reason to lack one")
    if not missing:
        print(f"ok   {'coverage':<28} all {len(ids_from_checker())} guard ids "
              f"are controlled or excused")

    print()
    if fails:
        print(f"FAIL {len(fails)} control(s):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"{len(CASES) + len(VACUOUS)} control(s) pass; every guard can fail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
