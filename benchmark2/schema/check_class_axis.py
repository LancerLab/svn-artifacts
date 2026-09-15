#!/usr/bin/env python3
"""Fail the build if any consumer's mutation-class axis disagrees with
`schema/class-axis.json`.

Run:  python3 schema/check_class_axis.py [-q]

WHY
---
The class axis was restated in six places and drifted. The SOTA lanes kept a
pre-revision 3-class dialect while choreo and the renderer moved to 4, so **M4
was absent from every SOTA lane's detection matrix** -- and absent is not the
same claim as `n/a`. This checker makes the six consumers provably agree, and
fails loudly when a lane's committed `stats.json` cannot be reconciled with what
the axis says the lane does.

The guards, in the order they run:

  G1  the axis is internally consistent (bijection class <-> obligation, no
      `L` among the classes, every lane's status map complete and legal).
  G2  `schema/record-schema.json`'s enums match the axis exactly.
  G3  the Python modules that restate a constant (`CLASSES`, `MUTATION_CLASSES`,
      `NA_WITNESS`, ...) restate it correctly.
  G4  no module outside `schema/` contains a hard-coded sequence of two or more
      class ids. This is the guard that matters: a restated *tuple* is how the
      3-class dialect survived. A single class id (`"M4"`) is fine -- that is a
      reference to one class, not a claim about the axis.
  G5  every lane that the axis holds at `n/a` carries the reason string, and
      every lane held at `uncompared` says so explicitly rather than leaving
      the class out by omission.
  G6  each lane's committed `stats.json` S1 block has exactly the classes the
      axis says it has, with `n/a` cells carrying `n_na == n_target` and
      `uncompared` classes carrying no cell at all.
  G7  the docs do not carry a superseded claim about the axis.
  G8  every declared corpus file exists, can be read, and carries the release
      and the v2.1 field set the axis records for it.
  G9  no lane injects the same perturbation twice, and the cross-lane
      correspondence ledger is current.

Exit status is 0 only if every guard passes.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))       # benchmark2/schema/
B2 = os.path.dirname(HERE)                              # benchmark2/
if B2 not in sys.path:
    sys.path.insert(0, B2)

from schema import class_axis as AX                     # noqa: E402
from schema import records as _records                 # noqa: E402

CLASSES = AX.mutation_classes()
SCORED = AX.scored_classes()
CONTROLS = AX.control_classes()
OBLIGATIONS = AX.obligation_classes()
PATHS = AX.path_classes()
STATUSES = ["measured", "n/a", "uncompared", "not_ready"]
N_TARGET = AX.n_target_per_class()

# The Python modules that restate an axis constant. Each entry is
# (file, constant-name) and the value must equal `expected`.
PY_CONSTANTS = [
    ("choreo/stats.py", "CLASSES", CLASSES),
    ("choreo/stats.py", "CONTROL_CLASSES", CONTROLS),
    ("choreo/stats.py", "OBL_CLASSES", OBLIGATIONS),
    ("choreo/stats.py", "PATH_CLASSES", PATHS),
    ("render.py", "MUTATION_CLASSES", CLASSES),
    ("render.py", "CONTROL_CLASSES", CONTROLS),
]

# A class id, or a quoted sequence of two or more of them (G4).
_MULTI_CLASS = re.compile(r'(["\'])(M[0-9])\1(?:\s*,\s*(["\'])(M[0-9])\3)+')

# Superseded claims. Each is (file, regex, why).
STALE = [
    ("schema/PATCH-v2.1.md", r'"M4"\s*,\s*"L"',
     "v2.1 added `L` to the class enum; the 2026-09-14 revision folded `L` into "
     "M3.17-M3.26 and demoted `L` to a path class."),
    ("schema/PATCH-v2.1.md", r'M4 and L became real mutation classes',
     "same supersession; `L` is not a mutation class."),
    ("specs/mutation-specs-v2.md", r'^\s*#+\s*4\.\s*Launch-status class',
     "§4 still presents `L` as a class. It is the M3.17-M3.26 launch-status "
     "*path*, classified P2."),
    ("mlir-shared/compose.py", r'#\s*"M1"\s*\|\s*"M2"\s*\|\s*"M3"',
     "compose.Mutation.klass was documented as a 3-class field; M4 is also a "
     "class (though no M4 mutant is composed here)."),
    ("schema/PATCH-v2.1.md", r'L` operators are generated',
     "same supersession."),
]


def _rel(p: str) -> str:
    return os.path.relpath(p, B2)


def _jsonl(path) -> list | None:
    """Records of a corpus file, or None if this gate cannot read one.

    Two on-disk shapes exist and both are legitimate:

    * JSON Lines -- one record per line, used by the SOTA lanes.
    * A single JSON object wrapping a list under one key, used by choreo:
      `{"records": [...]}` for run records and `{"mutants": [...]}` for the
      manifest. The manifest is a PLAN, not run records (it has no `outcome`
      or `stage`), which is why this file is declared as a corpus for its
      release vocabulary only.

    Returning None for anything else is deliberate. The choreo manifest was a
    single JSON object, and the first version of this helper read "no records"
    out of it -- so the gate passed both of choreo's declared files without
    having looked at either. A gate must never pass a file it did not read.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None

    lines = [l for l in text.splitlines() if l.strip()]
    try:
        return [json.loads(l) for l in lines] or None      # JSON Lines
    except ValueError:
        pass

    try:
        obj = json.loads("\n".join(lines))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    lists = [v for v in obj.values() if isinstance(v, list)]
    if len(lists) != 1:
        return None                # ambiguous: cannot say which list is records
    recs = lists[0]
    if not recs or not all(isinstance(r, dict) for r in recs):
        return None
    return recs


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _assignments(path: str) -> dict[str, object]:
    """Module-level assignments in a Python file, as literals.

    Parsed with `ast`, never imported: several of these modules do work at
    import time and importing them from a checker would be a side effect.

    A right-hand side that mentions `AX.` / `class_axis` is recorded as
    ``("__derived__", <source>)``. That is the *preferred* form -- the value is
    then computed from the axis and cannot drift -- so G3 passes it and leaves
    the checking to the module itself. A literal is still accepted, and still
    checked, because `NA_WITNESS` is an editorial choice rather than a value the
    axis determines.
    """
    src = _read(path)
    if not src:
        return {}
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as exc:                             # pragma: no cover
        return {"__syntax_error__": str(exc)}
    out: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name):
                continue
            rhs = ast.unparse(node.value)
            try:
                out[tgt.id] = ast.literal_eval(node.value)
            except ValueError:
                if "AX." in rhs or "class_axis" in rhs:
                    out[tgt.id] = ("__derived__", rhs)
                    continue
                # e.g. OrderedDict([...]) -- not a literal. Recover the keys of
                # the inner list-of-pairs so the registries can be checked.
                if (isinstance(node.value, ast.Call)
                        and node.value.args
                        and isinstance(node.value.args[0],
                                       (ast.List, ast.Tuple))):
                    pairs = []
                    for elt in node.value.args[0].elts:
                        if (isinstance(elt, (ast.Tuple, ast.List))
                                and elt.elts):
                            try:
                                pairs.append(ast.literal_eval(elt.elts[0]))
                            except ValueError:
                                pass
                    if pairs:
                        out[tgt.id] = pairs
    return out


def _is_derived(value: object) -> bool:
    return (isinstance(value, tuple) and len(value) == 2
            and value[0] == "__derived__")


# ---------------------------------------------------------------------------
# G1 -- the axis is internally consistent
# ---------------------------------------------------------------------------
def g1_axis() -> list[str]:
    bad: list[str] = []
    if CLASSES != ["M1", "M2", "M3", "M4"]:
        bad.append(f"mutation_classes is {CLASSES}, expected M1..M4 in order")
    if "L" in CLASSES:
        bad.append("`L` is in the class axis; it is a PATH class "
                   f"({AX.not_classes()['L']['meaning']})")
    if "L" not in PATHS:
        bad.append("`L` is not in path_classes")
    if set(CLASSES) & set(PATHS):
        bad.append(f"class ids collide with path ids: "
                   f"{sorted(set(CLASSES) & set(PATHS))}")
    if set(CLASSES) & set(OBLIGATIONS):
        bad.append(f"class ids collide with obligation ids: "
                   f"{sorted(set(CLASSES) & set(OBLIGATIONS))} -- these are "
                   f"two vocabularies, not one")
    if len(AX.class_ids()) != len(CLASSES):
        bad.append("`classes[]` does not have one entry per mutation class")
    for c in AX.axis()["classes"]:
        if c["obligation"] not in OBLIGATIONS:
            bad.append(f"classes[{c['id']}].obligation={c['obligation']!r} is "
                       f"not in obligation_classes")
    # bijection
    fwd = {c["id"]: c["obligation"] for c in AX.axis()["classes"]}
    if sorted(fwd.values()) != sorted(OBLIGATIONS):
        bad.append(f"class<->obligation is not a bijection: {fwd}")
    if set(CONTROLS) - set(CLASSES) or set(SCORED) | set(CONTROLS) != set(CLASSES):
        bad.append("role=control classes are not a subset of the class axis")
    if N_TARGET <= 0:
        bad.append(f"n_target_per_class={N_TARGET}")
    # lanes
    for lane in AX.lanes():
        st = AX.lane_status(lane)
        if sorted(st) != sorted(CLASSES):
            bad.append(f"lane {lane!r} status map covers {sorted(st)}, "
                       f"expected {sorted(CLASSES)}")
        for cls, s in st.items():
            if s not in STATUSES:
                bad.append(f"lane {lane!r} class {cls} has status {s!r}, "
                           f"not one of {STATUSES}")
        for cls in AX.lane_where(lane, "n/a"):
            try:
                if not AX.na_reason(lane, cls).strip():
                    bad.append(f"{lane}/{cls} is n/a with an empty reason")
            except KeyError as exc:
                bad.append(str(exc))
    return bad


# ---------------------------------------------------------------------------
# G2 -- record-schema.json enums
# ---------------------------------------------------------------------------
def _schema_enums(obj, prefix: str = "") -> list[tuple[str, str, list]]:
    found: list[tuple[str, str, list]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "enums" and isinstance(v, dict):
                for ek, ev in v.items():
                    found.append((f"{prefix}.enums.{ek}", ek, ev))
            else:
                found += _schema_enums(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _schema_enums(v, f"{prefix}[{i}]")
    return found


def g2_schema() -> list[str]:
    bad: list[str] = []
    path = os.path.join(HERE, "record-schema.json")
    try:
        schema = json.loads(_read(path))
    except json.JSONDecodeError as exc:                    # pragma: no cover
        return [f"schema/record-schema.json does not parse: {exc}"]
    expect = {"class": CLASSES, "paper_category": None,
              "path_class": PATHS, "prohibition": None}
    del expect["paper_category"], expect["prohibition"]
    seen_class_enum = False
    for where, key, values in _schema_enums(schema):
        if key == "class":
            # `obligation.enums.class` is the OTHER axis and must not be
            # mistaken for this one.
            if where.startswith("records.obligation"):
                if values != OBLIGATIONS:
                    bad.append(f"{where} = {values}, expected {OBLIGATIONS}")
                continue
            seen_class_enum = True
            if values != CLASSES:
                bad.append(f"{where} = {values}, expected {CLASSES}")
        elif key == "path_class":
            if values != PATHS:
                bad.append(f"{where} = {values}, expected {PATHS}")
        elif "L" in (values if isinstance(values, list) else []):
            # Any other enum admitting `L` is a class-shaped enum in disguise.
            if "path_class" not in where:
                bad.append(f"{where} admits \"L\": {values}")
    if not seen_class_enum:
        bad.append("record-schema.json declares no `class` enum at all")
    return bad


# ---------------------------------------------------------------------------
# G3 -- restated Python constants
# ---------------------------------------------------------------------------
def g3_constants() -> list[str]:
    bad: list[str] = []
    cache: dict[str, dict] = {}
    for rel, name, expected in PY_CONSTANTS:
        path = os.path.join(B2, rel)
        if rel not in cache:
            cache[rel] = _assignments(path)
        got = cache[rel].get(name)
        if isinstance(got, dict) and "__syntax_error__" in got:
            bad.append(f"{rel} does not parse: {got['__syntax_error__']}")
            continue
        if got is None:
            bad.append(f"{rel}:{name} not found")
            continue
        if _is_derived(got):
            continue          # computed from the axis: nothing to drift from
        if list(got) != list(expected):
            bad.append(f"{rel}:{name} = {list(got)}, expected {list(expected)}")

    # render.py's witness + registry maps are derived facts about the axis.
    r = _assignments(os.path.join(B2, "render.py"))
    witness = r.get("NA_WITNESS")
    if isinstance(witness, dict):
        if "M4" in witness:
            bad.append("render.py:NA_WITNESS has an M4 key; M4 is uncompared on "
                       "every compared lane, so an `n/a` witness would report an "
                       "unmeasured gap as a measured verdict")
        for cls, lane in witness.items():
            if cls not in SCORED:
                bad.append(f"render.py:NA_WITNESS keys {cls!r}, which is not a "
                           f"scored class ({SCORED})")
            elif cls in AX.control_classes():
                bad.append(f"render.py:NA_WITNESS keys control class {cls!r}")
            elif lane not in AX.lanes():
                bad.append(f"render.py:NA_WITNESS[{cls!r}]={lane!r} is not a lane")
            elif cls not in AX.lane_where(lane, "n/a"):
                bad.append(f"render.py:NA_WITNESS[{cls!r}]={lane!r}, but {lane} "
                           f"does not hold {cls} at n/a (it holds "
                           f"{AX.lane_status(lane)[cls]!r})")
    else:
        bad.append("render.py:NA_WITNESS not found")

    for name in ("LANE_STATS_PATHS", "LANE_OWNS"):
        lanes = r.get(name)
        if lanes is None:
            bad.append(f"render.py:{name} not found")
        elif _is_derived(lanes):
            continue
        elif sorted(lanes) != sorted(AX.ready_lanes()):
            bad.append(f"render.py:{name} covers {sorted(lanes)}, but the axis "
                       f"holds {sorted(AX.ready_lanes())} ready (and "
                       f"{sorted(AX.not_ready_lanes())} not ready)")

    not_ready = r.get("NOT_READY")
    if not_ready is None:
        bad.append("render.py:NOT_READY not found")
    elif not _is_derived(not_ready):
        if sorted(not_ready) != sorted(AX.not_ready_lanes()):
            bad.append(f"render.py:NOT_READY = {sorted(not_ready)}, but the axis "
                       f"marks {sorted(AX.not_ready_lanes())} not ready")
    return bad


# ---------------------------------------------------------------------------
# G4 -- no restated class tuple outside schema/
# ---------------------------------------------------------------------------
TUPLE_EXEMPT: set[tuple[str, int]] = set()


def g4_literals() -> list[str]:
    bad: list[str] = []
    for root, dirs, files in os.walk(B2):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".git", "schema", "results",
                                "extern", "build")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            for i, line in enumerate(_read(path).splitlines(), 1):
                m = _MULTI_CLASS.search(line)
                if not m or (path, i) in TUPLE_EXEMPT:
                    continue
                # A commented-out literal (including the historical one, quoted
                # in the note explaining why it was removed) is not a restatement.
                hash_at = line.find("#")
                if 0 <= hash_at < m.start():
                    continue
                bad.append(f"{_rel(path)}:{i} restates a sequence of class "
                           f"ids: {line.strip()!r}. Use "
                           f"schema.class_axis.mutation_classes() instead.")
    return bad


# ---------------------------------------------------------------------------
# G5 -- lanes declare their n/a and uncompared classes
# ---------------------------------------------------------------------------
# Lane -> the source files that must carry the declaration.
LANE_SOURCES = {
    "mlir-linalg": ["mlir-shared/lane.py"],
    "mlir-low": ["mlir-shared/lane.py"],
    "iree": ["iree/collect_stats.py", "iree/lane.py"],
    "triton": ["triton/collect.py"],
}


def g5_declarations() -> list[str]:
    """Each lane must DERIVE its `n/a` and `uncompared` sets from the axis.

    A literal is not evidence here. The drift that started all this was a file
    that named three classes and silently omitted the fourth, so this guard asks
    the opposite question: does the lane read the axis at all, and does it have
    a code path for a class it was never run against?
    """
    bad: list[str] = []
    for lane, sources in LANE_SOURCES.items():
        text = "\n".join(_read(os.path.join(B2, s)) for s in sources)
        label = ", ".join(sources)
        if "class_axis" not in text:
            bad.append(f"{lane}: {label} does not import schema.class_axis, so "
                       f"its class set is a restatement and will drift")
        for cls in AX.lane_where(lane, "n/a"):
            # Either the reason is local prose (the id appears somewhere) or it
            # is read back from the axis. Both are fine; neither is an omission.
            if f'"{cls}"' in text:
                continue
            if "na_reason" in text or "lane_where" in text:
                continue
            bad.append(f"{lane} holds {cls} at n/a but {label} neither names it "
                       f"nor reads AX.na_reason, so the class is unaccounted for")
        if AX.lane_where(lane, "uncompared") and "uncompared" not in text:
            bad.append(f"{lane} holds {AX.lane_where(lane, 'uncompared')} at "
                       f"uncompared but {label} has no uncompared code path; the "
                       f"class is then absent by omission, which is "
                       f"indistinguishable from a bug")
    return bad


# ---------------------------------------------------------------------------
# G6 -- committed stats.json conforms
# ---------------------------------------------------------------------------
LANE_STATS = {lane: AX.lane_stats_paths(lane) for lane in AX.lanes()}


def _load_s1(lane: str):
    key = AX.lane_stats_key(lane)
    for rel in LANE_STATS[lane]:
        path = os.path.join(B2, rel)
        if not os.path.exists(path):
            continue
        try:
            doc = json.loads(_read(path))
        except json.JSONDecodeError as exc:
            return None, rel, f"does not parse: {exc}"
        blk = doc.get(key)
        if blk is None:
            others = [k for k in ("S1_detection", "S1_detection_matrix")
                      if k in doc and k != key]
            if others:
                return None, rel, (f"has its S1 block under {others[0]!r}, but "
                                   f"the axis says this lane writes {key!r}")
            return None, rel, f"has no {key!r} block"
        if "per_class" in blk and isinstance(blk["per_class"], dict):
            return blk["per_class"], rel, None
        return {k: v for k, v in blk.items() if k in CLASSES}, rel, None
    return None, LANE_STATS[lane][0], "not present"


def _load_doc(lane: str):
    for rel in LANE_STATS[lane]:
        path = os.path.join(B2, rel)
        if os.path.exists(path):
            try:
                return json.loads(_read(path)), rel
            except json.JSONDecodeError:
                return None, rel
    return None, LANE_STATS[lane][0]


def g6_outputs() -> list[str]:
    bad: list[str] = []
    for lane in AX.lanes():
        st = AX.lane_status(lane)
        cells, rel, err = _load_s1(lane)
        if cells is None:
            if all(s == "not_ready" for s in st.values()):
                continue          # a lane with no results yet is not drift
            bad.append(f"{rel} ({lane}): {err}; the axis expects "
                       f"{ {c: s for c, s in st.items() if s != 'uncompared'} }")
            continue
        if not isinstance(cells, dict):
            bad.append(f"{rel} ({lane}): S1 block is not a mapping")
            continue
        extra = sorted(set(cells) - set(CLASSES))
        if extra:
            bad.append(f"{rel} ({lane}): S1 carries unknown class keys {extra}")
        for cls in CLASSES:
            want = st[cls]
            cell = cells.get(cls)
            if want == "uncompared":
                if cell is not None:
                    bad.append(f"{rel} ({lane}): {cls} is uncompared "
                               f"(the lane was never run against it) but S1 "
                               f"carries a cell {cell!r}; an unmeasured class "
                               f"must not be reported as a measurement")
                continue
            if cell is None:
                bad.append(f"{rel} ({lane}): {cls} is {want!r} per the axis but "
                           f"S1 has no cell for it. Regenerate the lane's "
                           f"stats; do not hand-edit.")
                continue
            if not isinstance(cell, dict):
                bad.append(f"{rel} ({lane}): {cls} cell is not a mapping")
                continue
            n_inj = cell.get("n_injected")
            n_na = cell.get("n_na")
            if want == "n/a":
                if n_na != N_TARGET:
                    bad.append(f"{rel} ({lane}): {cls} is n/a but n_na="
                               f"{n_na!r}, expected {N_TARGET}")
                if n_inj not in (0, None):
                    bad.append(f"{rel} ({lane}): {cls} is n/a but "
                               f"n_injected={n_inj!r}")
                if not str(cell.get("note", "")).strip():
                    bad.append(f"{rel} ({lane}): {cls} is n/a with no note "
                               f"explaining what the surface cannot express")
            elif want == "measured":
                if not n_inj:
                    bad.append(f"{rel} ({lane}): {cls} is measured but has no "
                               f"injections (n_injected={n_inj!r})")

        # The uncompared declaration. `uncompared` classes are absent from S1 by
        # design, so the declaration block is the ONLY place a reader can learn
        # that the lane was never run against them rather than that the surface
        # cannot express them.
        doc, drel = _load_doc(lane)
        want_uncmp = sorted(AX.uncompared_classes(lane))
        if doc is not None:
            decl = doc.get(AX.uncompared_key(), {})
            got_uncmp = sorted(decl.get("classes", []) if isinstance(decl, dict)
                               else [])
            if got_uncmp != want_uncmp:
                bad.append(f"{drel} ({lane}): {AX.uncompared_key()}.classes = "
                           f"{got_uncmp}, expected {want_uncmp}. A class the "
                           f"lane was never run against must be declared, not "
                           f"merely absent from S1.")
            if want_uncmp and not str(decl.get("reason", "")).strip():
                bad.append(f"{drel} ({lane}): {AX.uncompared_key()} carries no "
                           f"reason for {want_uncmp}")
    return bad


# ---------------------------------------------------------------------------
# G7 -- docs
# ---------------------------------------------------------------------------
def g7_docs() -> list[str]:
    bad: list[str] = []
    cache: dict[str, str] = {}
    for rel, pattern, why in STALE:
        if rel not in cache:
            cache[rel] = _read(os.path.join(B2, rel))
        if not cache[rel]:
            continue                       # absent file: not drift
        m = re.search(pattern, cache[rel], re.M)
        if m:
            line = cache[rel][:m.start()].count("\n") + 1
            bad.append(f"{rel}:{line} carries a superseded axis claim "
                       f"({m.group(0)!r}): {why}")
    return bad


# ---------------------------------------------------------------------------
# G8 -- corpus release
# ---------------------------------------------------------------------------
# The rule "required fields = the base set plus the fields the record's RELEASE
# adds" (schema/records.py) has one hole: `v1` is the default, so a v2.1 writer
# that forgets to stamp its records is indistinguishable from a v1 writer that
# is doing the right thing. Every v2.1 field would simply go missing and the
# record would validate as an older release -- silently.
#
# The lane's own declaration closes it. `lanes.<lane>.spec_version.corpus` says
# which release the lane's `raw/` files are. G8 reads those files and asserts
# every record in them is at that release. So the declaration is not a comment:
# it is the assertion that makes the default harmless. A lane that holds v1
# records and declares v2.1 fails here; a lane that stamps v2.1 records and
# declares v1 fails here too.
#
# What this does NOT do is decide whether preparing the port is worth the GPU
# time. It only makes the fork a stated fact rather than a trap.
def g8_corpus_release() -> list[str]:
    """The release label must be a measurement, not an aspiration.

    `records.py` reads `release = spec_version or "v1"`, so `v1` is the default
    and a v2.1 writer that forgets to stamp its records is indistinguishable
    from a v1 writer doing the right thing. Two things close that hole, and this
    gate checks both: the lane's own per-file declaration, and the measured
    field set of the file it points at.

    The declaration is PER FILE, not per lane, because `choreo`'s two files
    genuinely differ -- its manifest is v2.1 and carries four of the five v2.1
    fields, while `raw/e1_mutant_records.json` is still the v1 baseline. A
    lane-level list averaged those into one false statement, which is what the
    first version of this declaration did.

    A file may sit BELOW the lane's target release. That is real work, not a
    pass, so it is allowed only when `port` names it. A PARTIAL v2.1 port (the
    mlir lanes wrote `spec_id` but none of the three path fields) is likewise
    allowed only when declared -- and it is not something `spec_version` can be
    stamped onto, since stamping it v2.1 would demand all five fields and the
    records do not have them.
    """
    bad: list[str] = []
    B2p = Path(B2)
    v21 = set(_records.versioned_fields("mutant"))
    for lane in AX.ready_lanes():
        complaint = AX.lane_release_is_consistent(lane)
        if complaint:
            bad.append(complaint)
            continue
        target = AX.lane_corpus_release(lane)
        spec = AX.lane_spec_version(lane)
        for rel in AX.lane_corpus_paths(lane):
            path = B2p / rel
            if not path.exists():
                bad.append(f"{lane}: declared corpus {rel} is missing")
                continue
            recs = _jsonl(path)
            if recs is None:
                bad.append(
                    f"{lane}: {rel} is declared as a corpus but is not a "
                    f"records file this gate can read. A declaration the gate "
                    f"cannot check must not pass -- say what the file is, or "
                    f"stop declaring it.")
                continue

            declared = AX.lane_corpus_release_at(lane, rel)
            want = set(AX.lane_v21_fields_at(lane, rel))
            releases: dict[str, int] = {}
            shapes: dict[frozenset, int] = {}
            for r in recs:
                v = _records.spec_version_of(r)
                releases[v] = releases.get(v, 0) + 1
                s = frozenset(v21 & set(r))
                shapes[s] = shapes.get(s, 0) + 1

            if len(releases) > 1:
                bad.append(
                    f"{lane}: {rel} mixes releases "
                    f"({', '.join(f'{k}={v}' for k, v in sorted(releases.items()))}). "
                    f"A corpus has one release; a mixed one means part of it was "
                    f"regenerated and part was not, so any figure read from it "
                    f"spans two vocabularies.")
            elif declared not in releases:
                bad.append(
                    f"{lane}: declares {rel} at release {declared} but it holds "
                    f"{sorted(releases)}. The lane's evidence and its label must "
                    f"agree; a v1 corpus relabelled v2.1 to satisfy a gate is "
                    f"exactly what this check exists to prevent.")

            if len(shapes) > 1:
                detail = ", ".join(
                    f"{sorted(k) or 'none'}x{n}" for k, n in
                    sorted(shapes.items(), key=lambda kv: -kv[1]))
                bad.append(
                    f"{lane}: {rel} does not have ONE v2.1 field set -- {detail}. "
                    f"Part of this corpus was written by a ported writer and part "
                    f"was not, so a reader cannot tell which records can be read "
                    f"for the path axis. Regenerate the whole corpus, or drop the "
                    f"ported fields from the part that has them.")
                continue

            got = next(iter(shapes), frozenset())
            if got != want:
                bad.append(
                    f"{lane}: {rel} carries the v2.1 fields {sorted(got)} but the "
                    f"axis records {sorted(want)}. The declaration is a "
                    f"measurement, so a stale list here hides unported fields "
                    f"behind a release label that reads fine.")
                continue

            if got and got != v21 and AX.lane_corpus_kind_at(lane, rel) == "records":
                missing = sorted(v21 - got)
                if declared != "v1":
                    bad.append(
                        f"{lane}: {rel} carries a partial v2.1 vocabulary "
                        f"({sorted(got)}) but declares release {declared}. A "
                        f"partial port is a v1 corpus; it cannot be read for the "
                        f"path axis until {missing} are present.")
                elif not str(spec.get("port") or "").strip():
                    bad.append(
                        f"{lane}: {rel} carries a partial v2.1 vocabulary "
                        f"({sorted(got)}) with no `port` target recorded. An "
                        f"undeclared partial port is the silent half-stamp this "
                        f"gate exists to catch: name the writer that must gain "
                        f"{missing} in the axis's `port`, or drop the field.")
    return bad


def g9_correspondence() -> list[str]:
    """Two rules about *what* a lane injects, not about what it reports.

    The other eight guards check that a consumer agrees with the axis about
    vocabulary. None of them can see that a mutant is a meaningless copy,
    because a copy agrees with the axis perfectly. This guard adds the smallest
    rule that makes the corpus itself reviewable.

    1.  No two specs may inject the SAME perturbation into the same kernel.
        `build_correspondence.py` classifies four kinds; two are fatal: an
        `exact-duplicate` (a mutant is a meaningless copy) and a
        `mechanism-mismatch` (the edit cannot implement the relation the spec
        names). Measured 2026-09-15: **6 exact-duplicate mutants in 6 clusters**
        (`M1.14`/`M1.5` on five kernels -- every one of M1.14's five
        realisations is byte-identical to an M1.5 realisation -- and
        `M1.2`/`M1.9` on one) and **5 mechanism mismatches** (`M1.4`'s two
        `.tile` edits change an *extent* while the spec claims a *stride*; the
        three `M1.9` edits are element-index moves where the spec claims a
        *base offset*, a construct the suite does not have). The other two
        kinds are advisories, printed but not fatal, because they need an owner
        ruling: `cross-spec-same-site` (11 specs contest 5 declarations) and
        `constant-only-variant` (24 redundant mutants, 17 of them M3.1).

    2.  The committed correspondence ledger must equal a fresh build. The ledger
        records, per spec, which lanes realise it and with what evidence -- the
        claim a reader needs in order to know whether a cross-lane rate compares
        the same defects. If the manifest moves and the ledger does not, that
        claim goes stale silently, which is the failure mode this whole file
        exists to prevent.

    The guard does NOT fail on low cross-lane coverage. Coverage is a
    measurement, not a drift, and a gate that fails on a number forces the
    number to be edited instead of the code.
    """
    bad: list[str] = []
    try:
        from schema import build_correspondence as BC
    except Exception as exc:                      # pragma: no cover
        return [f"G9 cannot import the ledger builder: {exc!r}"]

    payload = BC.build()

    # Advisories: findings that need an owner ruling, not unambiguous defects.
    # Printed, never fatal -- the gate must not force a number to be edited.
    for d in payload["duplicates"]:
        if d["kind"] == "constant-only-variant":
            print(f"    G9 advisory: {d['spec']} re-injects one rule with "
                  f"{d['n_redundant'] + 1} constants in {d['category']} "
                  f"({', '.join(d['mutant_ids'])}).  Keep the first, record "
                  f"the rest as variants-of unless a cited taxonomy attests a "
                  f"value.")
        elif d["kind"] == "cross-spec-same-site":
            print(f"    G9 advisory: {', '.join(d['specs'])} contest one "
                  f"declaration in {d['category']} ({d['case']}): "
                  f"`{(d.get('site') or '')[:60]}`.  Adjudicate which spec "
                  f"owns the site, or re-site the others.")

    # Failures: a mutant that is a meaningless copy, or a spec whose edit
    # cannot implement the relation it names.
    for d in payload["duplicates"]:
        if d["kind"] == "exact-duplicate":
            bad.append(
                f"duplicate injection: {', '.join(d['specs'])} inject the same "
                f"perturbation into the same kernel ({d['category']} "
                f"{d['case']}, edit {d['edit']}): "
                f"{', '.join(d['mutant_ids'])}.  Two specs producing one "
                f"mutant inflate a cell's denominator without adding evidence. "
                f"{d['disposition']}.")
        elif d["kind"] == "mechanism-mismatch":
            bad.append(
                f"mechanism mismatch: {d['spec']} declares a relation its edit "
                f"cannot implement ({d['category']} {d['case']}): "
                f"{', '.join(d['mutant_ids'])}.  {d['why']}  {d['disposition']}.")

    ledger = Path(B2) / "schema" / "mutation-correspondence.json"
    want = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if not ledger.exists():
        bad.append(
            "schema/mutation-correspondence.json is missing. Run "
            "`python3 schema/build_correspondence.py`.")
    elif ledger.read_text() != want:
        bad.append(
            "schema/mutation-correspondence.json is stale: a fresh build "
            "differs. The ledger states which lanes realise which specs, so a "
            "stale one states that falsely. Re-run "
            "`python3 schema/build_correspondence.py` and read the diff.")
    return bad


CHECKS = [
    ("G1 axis self-consistency", g1_axis),
    ("G2 record-schema enums", g2_schema),
    ("G3 restated constants", g3_constants),
    ("G4 no restated class tuples", g4_literals),
    ("G5 lane n/a + uncompared declarations", g5_declarations),
    ("G6 committed stats.json conformance", g6_outputs),
    ("G7 docs", g7_docs),
    ("G8 corpus release", g8_corpus_release),
    ("G9 no duplicate injections + ledger current", g9_correspondence),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only print failures")
    ap.add_argument("--guard", action="append", default=None,
                    help="run only the named guard (repeatable), e.g. --guard G4")
    args = ap.parse_args(argv)

    wanted = set(args.guard or [])
    print(f"class-axis drift check against {_rel(AX.AXIS_PATH)}")
    print(f"  class axis      {CLASSES}")
    print(f"  obligation axis {OBLIGATIONS}")
    print(f"  path axis       {PATHS}")
    print(f"  controls        {CONTROLS}   scored {SCORED}   "
          f"n_target_per_class {N_TARGET}")
    for lane in AX.lanes():
        st = AX.lane_status(lane)
        print(f"  lane {lane:<12} " +
              " ".join(f"{c}={st[c]}" for c in CLASSES))
    print()

    total = 0
    for name, fn in CHECKS:
        if wanted and name.split()[0] not in wanted:
            continue
        bad = fn()
        if bad:
            total += len(bad)
            print(f"FAIL {name}")
            for b in bad:
                print(f"     - {b}")
        elif not args.quiet:
            print(f"ok   {name}")

    print()
    if total:
        print(f"{total} drift finding(s). The axis is "
              f"{_rel(AX.AXIS_PATH)}; fix the consumer, not the axis.")
        return 1
    print("no drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
