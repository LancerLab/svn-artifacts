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

Guards are named by the OBJECT they constrain, never by their position in
this file -- `axis.self-consistency`, `m3.cell`, `choreo.stats`,
`mutants.edits-apply`. `--list` prints the registry; `--guard m3` runs every
`m3.*` guard.

  axis.self-consistency    the axis is internally consistent (bijection
                           class <-> obligation, no `L` among the classes,
                           every lane's status map complete and legal).
  axis.schema-enums        `schema/record-schema.json`'s enums match the axis.
  axis.restated-constants  the modules that restate a constant (`CLASSES`,
                           `MUTATION_CLASSES`, `NA_WITNESS`, ...) restate it
                           correctly.
  axis.no-restated-tuples  no module outside `schema/` contains a hard-coded
                           sequence of two or more class ids. This is the guard
                           that matters: a restated *tuple* is how the 3-class
                           dialect survived. A single class id (`"M4"`) is
                           fine -- that is a reference to one class, not a
                           claim about the axis.
  lanes.declarations       every lane the axis holds at `n/a` carries the
                           reason string, and every lane held at `uncompared`
                           says so rather than leaving the class out by
                           omission.
  lanes.stats-conformance  each lane's committed `stats.json` S1 block has
                           exactly the classes the axis says it has, with `n/a`
                           cells carrying `n_na == n_target` and `uncompared`
                           classes carrying no cell at all.
  docs.no-superseded-claim the docs carry no superseded claim about the axis.
  corpus.declared-files    every declared corpus file exists, can be read, and
                           carries the release and the v2.1 field set the axis
                           records for it.
  mutants.no-duplicate-injection
                           no lane injects the same perturbation twice, and the
                           cross-lane correspondence ledger is current.
  mutants.registry         `choreo/mutations.py` loads and exposes the
                           registries the two guards below read.
  mutants.edits-apply      every operator's edit pattern applies to the base
                           kernel that operator declares.
  m1..m4.coverage          a class covers every category its own operators are
                           realised on, and each operator's class agrees with
                           the class its spec is registered under.
  m1..m4.cell              a class's declared cell is arithmetically reachable
                           under the budget. `m3.cell` is why this exists.
  guards.vocabulary        every guard-shaped token in the tree is a registered
                           id, and no guard is named by a position. This is the
                           guard that keeps the ids above closed.

Exit status is 0 only if every guard passes, and 2 if a `--guard` id matches
no registered guard -- a rename that silently ran zero guards and reported
success is the one failure this checker must not have.
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

# A class id, or a quoted sequence of two or more of them
# (`axis.no-restated-tuples`).
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
     "*outcome*, classified `avoided`."),
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
    then computed from the axis and cannot drift -- so
    `axis.restated-constants` passes it and leaves
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
# axis.self-consistency -- the axis is internally consistent
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
# axis.schema-enums -- record-schema.json enums
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
# axis.restated-constants -- restated Python constants
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
# axis.no-restated-tuples -- no restated class tuple outside schema/
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
# lanes.declarations -- lanes declare their n/a and uncompared classes
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
# lanes.stats-conformance -- committed stats.json conforms
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
# docs.no-superseded-claim -- docs
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
# corpus.declared-files -- corpus release
# ---------------------------------------------------------------------------
# The rule "required fields = the base set plus the fields the record's RELEASE
# adds" (schema/records.py) has one hole: `v1` is the default, so a v2.1 writer
# that forgets to stamp its records is indistinguishable from a v1 writer that
# is doing the right thing. Every v2.1 field would simply go missing and the
# record would validate as an older release -- silently.
#
# The lane's own declaration closes it. `lanes.<lane>.spec_version.corpus` says
# which release the lane's `raw/` files are. `corpus.declared-files` reads
# those files and asserts
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
    genuinely differ in KIND -- its manifest is a PLAN (it carries spec_id,
    path_class, prohibition and spec_version, but a plan records the in-process
    `admissible` and has no `applicable`), while `raw/e1_mutant_records.json`
    is the RECORD output and carries all five fields. A lane-level list
    averaged those into one false statement, which is what the first version of
    this declaration did.

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
    g = gid("mutants", "no-duplicate-injection")
    bad: list[str] = []
    try:
        from schema import build_correspondence as BC
    except Exception as exc:                      # pragma: no cover
        return [f"{g} cannot import the ledger builder: {exc!r}"]

    payload = BC.build()

    # Advisories: findings that need an owner ruling, not unambiguous defects.
    # Printed, never fatal -- the gate must not force a number to be edited.
    for d in payload["duplicates"]:
        if d["kind"] == "constant-only-variant":
            print(f"    {g} advisory: {d['spec']} re-injects one rule with "
                  f"{d['n_redundant'] + 1} constants in {d['category']} "
                  f"({', '.join(d['mutant_ids'])}).  Keep the first, record "
                  f"the rest as variants-of unless a cited taxonomy attests a "
                  f"value.")
        elif d["kind"] == "cross-spec-same-site":
            print(f"    {g} advisory: {', '.join(d['specs'])} contest one "
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


def _load_mutations():
    """Import `choreo/mutations.py` from SOURCE, never from a cached `.pyc`.

    A gate must never pass a file it did not read -- the same rule applies to
    judging one. `import choreo.mutations` consults
    `choreo/__pycache__/mutations.*.pyc` and accepts it when the recorded source
    mtime matches to the second, so a file written twice within one second (a
    negative-control harness that edits, runs, and restores) can be *judged* as
    if the edit were still in place. Observed 2026-09-17: four negative controls
    ran back to back, the fourth reported the third's failure, and a correctly
    restored file looked broken.

    Compiling the source directly is the only way to guarantee that what is
    reported and what is on disk are the same bytes. `sys.dont_write_bytecode`
    is irrelevant here -- we are removing the *read* of the cache, not the write.
    """
    import types
    path = Path(B2) / "choreo" / "mutations.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    src = path.read_text()
    pkg = "choreo"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [str(path.parent)]
        sys.modules[pkg] = p
    mod = types.ModuleType(f"{pkg}.mutations")
    mod.__file__ = str(path)
    mod.__package__ = pkg
    sys.modules[f"{pkg}.mutations"] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# How a guard is named
# ---------------------------------------------------------------------------
# A guard is named by the OBJECT it constrains, spelled the way the axis spells
# it -- never by the position it happens to hold in this file:
#
#     class M1      ->  m1.coverage  m1.cell
#     family M1-a   ->  m1.a.instances
#     lane choreo   ->  choreo.stats
#     instance      ->  mutants.edits-apply
#
# and the aggregate roots name the artifact that owns the invariant:
#
#     axis.*      schema/class-axis.json
#     families.*  schema/method-taxonomy.json :: families
#     mutants.*   choreo/mutants/, the emitted corpus
#     lanes.*     schema/class-axis.json :: lanes
#     corpus.*    the declared corpus files
#     docs.*      the plan documents
#     paper.*     tables/
#
# The ordinal this replaces was minted independently in three files, so it
# collided three ways. `G10` meant "class coverage" here and "family coverage"
# in `plan-a-execution.md`; `G11`/`G12` meant "edits apply"/"cell reachable"
# here and "family partition"/"select conformance" there; `method_taxonomy.py`
# had already emitted `G11`/`G13` for the latter; and
# `hardware-constraint-inventory.md` uses `G1`-`G7` for launch-geometry
# constraints. A position is not an identity -- a path is. `m3.cell` can only
# ever mean one thing.
ROOTS = ("axis", "families", "mutants", "lanes", "corpus", "docs", "paper")


def gid(*parts: str) -> str:
    """Join id parts the way the axis spells them: `("M1", "a")` -> `m1.a`."""
    return ".".join(str(p).lower() for p in parts)


def _bind(fn, arg):
    """Register a per-object guard without losing the id it is registered as."""
    return lambda: fn(arg)


_MUTATIONS = None
_MUTATIONS_ERROR: str | None = None


def _mutations_or_error():
    """Source-load `choreo/mutations.py`, remembering a failure.

    The per-class guards each need this module, but a load failure is not a
    per-class fact. It is reported once, by `mutants.registry`, instead of four
    times over -- and a guard that cannot read its input must not look like a
    guard that passed.
    """
    global _MUTATIONS, _MUTATIONS_ERROR
    if _MUTATIONS is None and _MUTATIONS_ERROR is None:
        try:
            _MUTATIONS = _load_mutations()
        except FileNotFoundError as exc:
            _MUTATIONS_ERROR = str(exc)
        except Exception as exc:
            # `mutations.py` asserts operator/registry agreement while building
            # `ALL`, so a disagreement arrives here. A finding, not a crash.
            _MUTATIONS_ERROR = f"cannot load choreo/mutations.py: {exc!r}"
    return _MUTATIONS, _MUTATIONS_ERROR


def mutants_registry() -> list[str]:
    """`choreo/mutations.py` loads and exposes what the other guards read.

    Named for the object, and separate from the guards that consume it, for one
    reason: every rule in this file is a statement about the registry, so a
    registry that cannot be read makes all of them vacuous. Reporting that once
    under its own id is the difference between "four guards passed" and "one
    guard failed and three proved nothing".
    """
    g = gid("mutants", "registry")
    M, err = _mutations_or_error()
    if M is None:
        return [f"{g} {err}"]
    needed = ("ALL", "MINIMAL_SET", "LEVEL2_SET", "SPEC_REGISTRY")
    missing = [n for n in needed if not hasattr(M, n)]
    if missing:
        return [f"{g} choreo/mutations.py has no {missing}. Every guard that "
                f"reads them would pass vacuously."]
    return []


def class_coverage(cls: str) -> list[str]:
    """A class must cover every category its own operators are realised on.

    Registered once per class, as `m1.coverage` .. `m4.coverage`, so that a
    finding names the class it is about and `--guard m4` runs exactly the
    class-scoped guards.

    This is the guard for a failure that is invisible from either end alone.
    `choreo/mutations.py` owns both the operator list and the per-class
    coverage set (`MINIMAL_SET` / `LEVEL2_SET`), so on 2026-09-17 the two
    disagreed without anybody noticing: 13 operators were re-homed from M1 to
    M4 (the empty-range and reversed-bound specs M1.6/M1.7 are M4 families),
    and while the registry and the family taxonomy both learned the new class,
    the coverage set did not. `M4` therefore claimed conv2d/relu/transpose while
    seven of its operators lived on layer_normalization/softmax.

    Nothing reported that. `gen_mutants.py` refused -- but only when someone ran
    it, and the refusal read like a missing-config error rather than a
    misclassification. The class cell was silently understated for as long as
    nobody regenerated.

    Stage 1 of selection is a floor per (spec_id, category): a declared spec
    that produces NO instance in its own class is indistinguishable from a
    forgotten spec. So the rule is exactly:

        for every operator o of class C:
            o.category in COVERAGE[C]

    plus the two facts that make the rule meaningful:

      * `o.cls` must equal the class the registry records for `o.spec_id`.
        `mutations.py` asserts this at import; this guard re-states it so the
        disagreement is reported as a finding rather than as a traceback.
      * every category named in a class's coverage set must exist in the
        suite, and a class that is not entirely n/a must have a non-empty set.
        An empty set makes the rule above vacuously true, which is how a guard
        becomes decorative.

    Deliberately NOT checked: whether the coverage set is *complete* for the
    paper's claim. Coverage is a design statement, and a gate that fails on a
    number forces the number to be edited. This guard only refuses a set that
    contradicts operators that already exist.
    """
    bad: list[str] = []
    g = gid(cls, "coverage")
    M, err = _mutations_or_error()
    if M is None:
        return []                  # reported once, by `mutants.registry`

    # The suite of base kernels lives beside benchmark2/, not inside it.
    suite = Path(B2).parent / "benchmark" / "choreo"
    on_disk = set()
    if suite.is_dir():
        on_disk = {p.name for p in suite.iterdir() if p.is_dir()}

    cover = list(M.MINIMAL_SET.get(cls, []))
    cover += [c for c in M.LEVEL2_SET.get(cls, []) if c not in cover]
    if not cover:
        return [
            f"{g} has an EMPTY coverage set. Every rule about categories is "
            f"then vacuously true, so the class could cover nothing and still "
            f"pass. Name the categories {cls} is measured on, or declare it "
            f"n/a."]
    if not on_disk:
        bad.append(
            f"{g} could not read the base-kernel suite at {suite}, so the "
            f"'category exists' rule did not run.")
    else:
        missing = sorted(set(cover) - on_disk)
        if missing:
            bad.append(
                f"{g} claims categories the suite does not have: {missing}. "
                f"Measured categories are {sorted(on_disk)}.")

    ops = list(M.ALL.get(cls, []))
    if not ops:
        bad.append(
            f"{g} read no operators of class {cls} from choreo/mutations.py. "
            f"Every rule above passed vacuously, so this guard has proven "
            f"nothing.")
    for m in ops:
        if m.category not in cover:
            bad.append(
                f"{g} mutation {m.id!r} is class {cls} but injects into "
                f"{m.category!r}, which is outside {cls}'s coverage set "
                f"{sorted(cover)}. The class cell is defined per "
                f"(class, category), so this instance belongs to no cell "
                f"the class claims -- either add {m.category!r} to "
                f"MINIMAL_SET[{cls!r}] (if {cls} really is measured there) "
                f"or re-home the operator.")
        reg = M.SPEC_REGISTRY.get(m.spec_id, {})
        want = reg.get("cls")
        if want and want != m.cls:
            bad.append(
                f"{g} mutation {m.id!r} declares class {m.cls} but spec "
                f"{m.spec_id} is registered as class {want}: the operator "
                f"and the registry disagree about which class cell this is "
                f"evidence for.")
    return bad


def mutants_edits_apply() -> list[str]:
    """Every operator's edits must apply to the kernel that operator declares.

    Registered as `mutants.edits-apply`: the object is the emitted corpus, and
    a finding names the operator that cannot fire.

    `choreo/mutations.py` is the registry of what the benchmark *can* inject,
    and "implemented" in the taxonomy means "an operator exists here" -- not
    "that operator can fire". Those two differ, and the difference is silent.

    `gen_mutants.py` does detect it: `apply_transform()` returns
    `(None, "pattern absent: ...")` and the instance is reported as *skipped*.
    But selection only ever applies the operators the budget chose, so an
    operator whose pattern never matches is judged only if the lottery picks
    it. Found 2026-09-17 by sweeping all 229 operators instead: 5 of them
    (2.2%) could not apply. `M1.s5.sm11.store` and `M1.14.sm11.store` declared
    `chunkat(i#p, _, _)` where the kernel says `chunkat(i#p, q, _)`;
    `M2.s2.mm11.tiles` declared the tile row `[1, 6, 6]` where the case-11
    kernel says `[1, 32, 32]`; and `M3.s3.cv2.shared512/1024` carried a second
    edit against a `make_spandata<choreo::f32>(32, 128, 1, 1)` line that conv2d
    case 11 never had. All five had been "implemented" for the project's whole
    life, and one of them only surfaced because a budget rewrite happened to
    select it.

    The rule, stated with the emitter's own predicate so the two cannot drift:

        for every operator o of every class:
            apply_transform(read(o.base_kernel), o) is not None

    A no-op body and a wrong occurrence count are the same defect wearing
    different clothes: the instance is emitted, the base kernel is unchanged,
    and the record claims an injection that never happened -- which reads as a
    detection miss, and lands in the miss column against the compiler.
    `apply_transform()` already refuses all three reasons, so this guard
    refuses them too.

    Deliberately NOT checked: whether the injected mutation is *detected* by
    the compiler, and whether the operator's declared class/path is right.
    Both need the toolchain (GATE 3/4) and the taxonomy (`axis.*`, `m*.coverage`);
    a guard that cannot run in a bare checkout must not pretend to have run.
    """
    g = gid("mutants", "edits-apply")
    bad: list[str] = []
    M, err = _mutations_or_error()
    if M is None:
        return []                  # reported once, by `mutants.registry`

    sys.path.insert(0, str(Path(B2) / "choreo"))
    try:
        import gen_mutants as GM
    except Exception as exc:
        return [f"{g} cannot import choreo/gen_mutants.py: {exc!r}"]

    ops = 0
    for cls in CLASSES:
        for m in M.ALL.get(cls, []):
            ops += 1
            path = Path(GM.base_path(m.category, m.case))
            if not path.exists():
                bad.append(
                    f"{g} mutation {m.id!r} declares category {m.category!r} "
                    f"case {m.case!r}, but {path} does not exist. The operator "
                    f"names a base kernel the suite does not have, so no "
                    f"instance it claims to inject can be measured at all.")
                continue
            _, why = GM.apply_transform(path.read_text(), m)
            if why is not None:
                bad.append(
                    f"{g} mutation {m.id!r} does not apply to {path.name}: "
                    f"{why}. It is counted as an implemented realisation but "
                    f"injects nothing: whenever the budget selects it the run "
                    f"reports the instance SKIPPED, and whenever it does not, "
                    f"the registry overstates what exists.")
    if not ops:
        bad.append(
            f"{g} read no operators at all from choreo/mutations.py. Every "
            f"rule above passed vacuously, so this guard has proven "
            f"nothing.")
    return bad


def class_cell(cls: str) -> list[str]:
    """A class cell must be ARITHMETICALLY reachable, not just declared.

    Registered once per class, as `m1.cell` .. `m4.cell`. `m3.cell` is the
    finding this guard exists for.

    The budget is `N = kernel_component x realisation_component` = `4 x 2` = 8
    instances per family, and `select()` enforces the second half as a cap of 2
    instances per CATEGORY per family. So the number of instances a family can
    reach is `2 x (categories the class can inject into)`, and a class can only
    reach its declared cell of `families x 8` if it can inject into at least
    `kernel_component` = 4 categories.

    Measured 2026-09-17: M1 has exactly 4 (layer_normalization/relu/softmax/
    transpose -- the model anchors), M2 has 6, M4 has 5. **M3 has 2**
    (matmul, conv2d). Its cell is therefore capped at `8 x 2 x 2 = 32` against a
    declared `64`, and the shortfall is arithmetic rather than effort: no amount
    of operator writing closes it, because the 3rd and 4th kernel do not exist.
    `MINIMAL_SET["M3"]` names only 2 categories for the same reason -- M3's
    specs are simply not realised anywhere else yet. This is W4's blocker, and
    W4's exit criterion, with a number on it.

    Both halves are checked, because they fail for different reasons and the fix
    differs: the DECLARATION (the coverage set reserves fewer than
    `kernel_component` categories) is a taxonomy edit, while the REALISATION
    (fewer than `kernel_component` categories actually carry a generatable
    operator) is suite or operator work. A class that declares 4 and realises 2
    is *late*, not *wrong* -- but it must not report a cell it cannot fill.

    Deliberately NOT checked: whether the extra categories are *appropriate*
    for the class (that the 4th M3 kernel really has MMA constructs). That is a
    design judgement this guard cannot make; it only refuses the claim when the
    arithmetic makes it false.
    """
    g = gid(cls, "cell")
    bad: list[str] = []
    M, err = _mutations_or_error()
    if M is None:
        return []                  # reported once, by `mutants.registry`
    sys.path.insert(0, str(Path(B2)))
    try:
        from schema import method_taxonomy as T
    except Exception as exc:
        return [f"{g} cannot load the taxonomy: {exc!r}"]

    reg = getattr(M, "SPEC_REGISTRY", None)
    if not reg:
        # `mutants.registry` reports the missing registry. Say nothing here so
        # the same fact is not reported five times.
        return []

    need = T.N_KERNELS
    declared = sorted(set(M.MINIMAL_SET.get(cls, []))
                      | set(M.LEVEL2_SET.get(cls, [])))
    realised = sorted({m.category for m in M.ALL.get(cls, [])
                       if T._generatable(m.spec_id, reg)})
    fams = len(T.families_of(cls))
    cell = fams * T.N_PER_FAMILY

    if len(declared) < need:
        reach = fams * min(T.N_PER_FAMILY, 2 * len(declared))
        bad.append(
            f"{g} declares only {len(declared)} categor"
            f"{'y' if len(declared) == 1 else 'ies'} {declared} but the "
            f"budget needs kernel_component = {need}. Its family budget of "
            f"{T.N_PER_FAMILY} decomposes as {need} kernels x "
            f"{T.N_REALISATIONS} instances-per-kernel, so the declared cell "
            f"of {cell} ({fams} families x {T.N_PER_FAMILY}) is capped at "
            f"{reach} by the missing kernel(s). Add the categories the "
            f"class is measured on, or restate this class's cell as "
            f"{reach} -- do not report {cell}.")
    if len(realised) < need:
        reach = fams * min(T.N_PER_FAMILY, 2 * len(realised))
        bad.append(
            f"{g} has generatable operators on only "
            f"{len(realised)} categor"
            f"{'y' if len(realised) == 1 else 'ies'} {realised}; the "
            f"budget needs {need}. `select()` caps a family at "
            f"{T.N_REALISATIONS} instances per category, so the family "
            f"budget {T.N_PER_FAMILY} -- and the class cell {cell} -- are "
            f"unreachable by construction: the cell computes to at most "
            f"{reach}. Either realise the class on "
            f"{need - len(realised)} more categor"
            f"{'y' if need - len(realised) == 1 else 'ies'}, or record the "
            f"class cell that the corpus can actually carry.")
    return bad


# ---------------------------------------------------------------------------
# families.* -- the partition the operators are filed under
# ---------------------------------------------------------------------------
# `m<C>.coverage` and `m<C>.cell` above ask whether the OPERATORS are filed
# correctly. These ask whether the PARTITION they are filed under holds: every
# spec_id in exactly one family, no family short an independent realisation, and
# every spec_id spelled the way the registry spells it.
#
# `schema/method_taxonomy.py::check()` is the body of all four. It runs once and
# its findings are bucketed by the id each finding names, so `--guard
# families.partition` runs that claim and not its three siblings. One guard id
# covering four claims is what made the old ordinals ambiguous; it is not a
# shortcut worth repeating.
_TAX = None
_TAXONOMY = None
_TAXONOMY_ERR: str | None = None


def _tax():
    """`schema.method_taxonomy`, imported once. `None` if it will not load."""
    global _TAX
    if _TAX is None:
        sys.path.insert(0, str(Path(B2)))
        try:
            from schema import method_taxonomy as _mod
            _TAX = _mod
        except Exception:
            _TAX = False
    return _TAX or None


def _taxonomy():
    """`(bad, debt)` from `method_taxonomy.check()`, memoized.

    Memoized because four guards read one result. The check is pure and reads
    two files; running it once per id would let sibling guards take different
    views of the same tree, which is the failure mode a partition guard exists
    to prevent.
    """
    global _TAXONOMY, _TAXONOMY_ERR
    if _TAXONOMY is None and _TAXONOMY_ERR is None:
        T = _tax()
        if T is None:
            _TAXONOMY_ERR = "schema/method_taxonomy.py will not import"
        else:
            M, err = _mutations_or_error()
            if M is None:
                _TAXONOMY_ERR = err
            else:
                try:
                    reg = getattr(M, "SPEC_REGISTRY", None) or {}
                    _TAXONOMY = T.check(reg, list(reg))
                except Exception as exc:
                    _TAXONOMY_ERR = f"the taxonomy check raised {exc!r}"
    return _TAXONOMY if _TAXONOMY is not None else ([], [])


# The ids `method_taxonomy.check()` emits. A finding that names none of them is a
# structural claim about the partition itself -- the budget arithmetic, a family
# with no name, two families sharing one -- so it belongs to
# `families.partition`, which is the guard for the partition as a whole.
TAXONOMY_IDS = ("families.partition", "families.no-shared-instance",
                "families.dead-vs-unwritten", "families.ids-canonical",
                "families.admissibility")


def _taxonomy_guard(want: str):
    """One `families.*` guard: `check()`'s findings that name this id.

    Reports `bad` and not `debt`, which is `method_taxonomy.py`'s own
    discipline: a finding listed in `known_debt` is real and tracked and exits
    0, so a guard that failed on it would just be re-litigating the ledger. The
    debt is still visible -- `method_taxonomy.py --check` prints it as `~`, and
    `--strict-debt` fails on it at the release gate. What this guard must catch
    is a finding the ledger has never seen.
    """

    def run() -> list[str]:
        if _TAXONOMY_ERR:
            return [f"{want} {_TAXONOMY_ERR}"]
        bad, _debt = _taxonomy()
        out: list[str] = []
        for msg in bad:
            head = msg.partition(" ")[0]
            if head in TAXONOMY_IDS:
                if head == want:
                    out.append(msg)
            elif want == TAXONOMY_IDS[0]:
                out.append(msg)
        return out

    return run


# ---------------------------------------------------------------------------
# families.select-conformance -- the emitter's accounting closes
# ---------------------------------------------------------------------------
def _lane_manifests() -> list[tuple[str, dict | str]]:
    """`(lane, manifest)` for every lane that has emitted a corpus.

    A lane with no manifest is `not_ready`, not wrong: the comparators run in
    W3 and only `choreo` has produced a corpus yet. A manifest that exists but
    will not parse is a string, because that IS wrong.
    """
    out: list[tuple[str, dict | str]] = []
    for lane in AX.lanes():
        path = Path(B2) / lane / "raw" / "mutant_manifest.json"
        if not path.is_file():
            continue
        try:
            out.append((lane, json.loads(path.read_text())))
        except Exception as exc:
            out.append((lane, f"{_rel(path)} will not parse: {exc!r}"))
    return out


def families_select_conformance() -> list[str]:
    """`select()`'s accounting closes, and no family is emitted short.

    The manifest is the only record of what the lottery did, and every count in
    the paper is downstream of it. `dropped` is the load-bearing key: before it
    existed, declarations disappeared between `select()` and the corpus, and
    nothing could tell a drop from a decision.
    """
    g = gid("families", "select-conformance")
    M, err = _mutations_or_error()
    if M is None:
        return []
    T = _tax()
    if T is None:
        return [f"{g} cannot load schema/method_taxonomy.py"]

    bad: list[str] = []
    mans = _lane_manifests()
    if not mans:
        return [f"{g} found no lane manifest under "
                f"{_rel(Path(B2) / '<lane>' / 'raw' / 'mutant_manifest.json')} "
                f"for any of {list(AX.lanes())}, so it would pass vacuously."]

    keys = ("candidates", "selected", "attribution", "dropped", "na",
            "skipped", "emitted")
    for lane, man in mans:
        if isinstance(man, str):
            bad.append(f"{g} {lane}: {man}")
            continue
        acc = man.get("accounting")
        if not isinstance(acc, dict):
            bad.append(f"{g} {lane}: its manifest has no `accounting` block, "
                       f"so no count in it can be checked.")
            continue
        missing = [k for k in keys if k not in acc]
        if missing:
            bad.append(f"{g} {lane}: its accounting has no {missing}.")
        else:
            parts = (acc["selected"] + acc["attribution"] + acc["dropped"]
                     + acc["na"] + acc["skipped"])
            if acc["candidates"] != parts:
                bad.append(
                    f"{g} {lane}: its accounting does not close -- candidates "
                    f"{acc['candidates']} != selected {acc['selected']} + "
                    f"attribution {acc['attribution']} + dropped "
                    f"{acc['dropped']} + na {acc['na']} + skipped "
                    f"{acc['skipped']} = {parts}. Note the drop count is NOT "
                    f"`candidates - selected`: the attribution_only mutants "
                    f"leave the lottery by a third route and must be counted, "
                    f"or every drop looks like {acc['candidates'] - acc['selected']} "
                    f"when it is {acc['dropped']}.")
            if acc["emitted"] != acc["selected"] + acc["attribution"]:
                bad.append(
                    f"{g} {lane}: emitted {acc['emitted']} != selected "
                    f"{acc['selected']} + attribution {acc['attribution']} = "
                    f"{acc['selected'] + acc['attribution']}. An emitted row "
                    f"is a selected one or an attributed one; there is no "
                    f"third source.")
        # The counters and the lists they summarise must agree in BOTH
        # directions. A counter with no rows is a decision nobody recorded; rows
        # with no counter are instances the cell arithmetic cannot see.
        for key, bucket in (("dropped", "dropped"), ("na", "na"),
                            ("attribution", "attribution")):
            rows = man.get(bucket)
            if isinstance(rows, list) and key in acc and len(rows) != acc[key]:
                bad.append(
                    f"{g} {lane}: accounting says `{key}` {acc[key]} but the "
                    f"`{bucket}` list holds {len(rows)}. One is the report and "
                    f"the other is the data; the corpus is built from the data.")

        if man.get("n_per_family") != T.N_PER_FAMILY:
            bad.append(
                f"{g} {lane}: the manifest is emitted at n_per_family "
                f"{man.get('n_per_family')} but the budget says "
                f"{T.N_PER_FAMILY}. N is one number with one definition "
                f"(`method-taxonomy.json::budget`); a lane may not carry its "
                f"own.")
        for key, want in (("n_kernels", T.N_KERNELS),
                          ("n_realisations", T.N_REALISATIONS)):
            if man.get(key) != want:
                bad.append(f"{g} {lane}: the manifest states {key} "
                           f"{man.get(key)} but the budget says {want}.")
        if man.get("target_instances") != T.target(lane):
            bad.append(
                f"{g} {lane}: the manifest targets "
                f"{man.get('target_instances')} instances but N x the "
                f"families in scope for it = {T.target(lane)}. A target that "
                f"is written down instead of derived goes stale the moment "
                f"scope moves.")

        # Every drop is a decision, and the decision must name its family as
        # the taxonomy names it -- otherwise the drop is filed under a family
        # that does not exist and the shortfall it explains is invisible.
        #
        # A family is required only where the taxonomy HAS one. A candidate for
        # an `attribution_only` spec (M3-L) is dropped out of a cell it was
        # never eligible for, so it correctly names no family -- 8 of the 123
        # drops are exactly that, and demanding a family for them would demand
        # that the attribution set be filed somewhere it must not be.
        for row in man.get("dropped") or []:
            if not row.get("reason"):
                bad.append(f"{g} {lane}: {row.get('mutant_id')} is dropped "
                           f"with no reason, so it cannot be told from a bug.")
            want = T.family_of(row.get("spec_id", ""))
            got = row.get("family")
            if want is None and got is None:
                pass        # attribution_only / avoided / reassigned: no cell to leave
            elif want is None and got is not None:
                bad.append(
                    f"{g} {lane}: {row.get('mutant_id')} is dropped as family "
                    f"{got}, but spec {row.get('spec_id')} belongs to no family "
                    f"(it is attribution_only, avoided or reassigned). A drop cannot "
                    f"explain a shortfall in a cell its spec is not in.")
            elif got is None:
                bad.append(
                    f"{g} {lane}: {row.get('mutant_id')} is dropped with no "
                    f"family, but spec {row.get('spec_id')} belongs to {want}. "
                    f"A drop that names no family cannot be reconciled against "
                    f"the cell it left.")
            elif got != want:
                bad.append(
                    f"{g} {lane}: {row.get('mutant_id')} is dropped as family "
                    f"{got} but spec {row.get('spec_id')} belongs to {want}. The "
                    f"drop explains a shortfall in the wrong family.")
    return bad


# ---------------------------------------------------------------------------
# m<C>.<family>.instances and m<C>.instances -- the cells themselves
# ---------------------------------------------------------------------------
def _lane_depth(man: dict) -> dict[str, int]:
    """Instances per family in one lane, recomputed from the emitted rows.

    Recomputed rather than read from `family_depth`, deliberately: that field is
    a cache, and a stale cache is exactly the shape a shortfall hides in. The
    two are compared by `families.select-conformance`, so a divergence is
    reported once, as a cache that went stale, instead of as a cell in the
    wrong place.
    """
    T = _tax()
    out: dict[str, int] = {}
    for row in man.get("mutants") or []:
        if row.get("attribution_only"):
            continue        # leaves the class cell by design (`attribution`)
        f = T.family_of(row.get("spec_id", "")) if T else None
        if f:
            out[f] = out.get(f, 0) + 1
    return out


def family_instances(family: str) -> list[str]:
    """One `m<C>.<family>.instances` cell: N instances, or an n/a with a reason.

    There is no third state. A cell at 5 is not "nearly there": N is the unit
    requirement (b) is stated in, so a family at 5 is 3 units missing from the
    denominator, and the paper's headline count is a multiple of N. When this
    guard is red it is not reporting a bug in the taxonomy -- it is printing the
    W1 work list, which is why it is worth having red.
    """
    T = _tax()
    if T is None:
        return []
    g = gid(family.replace("-", "."), "instances")
    cls = T._FAM[family]["class"]
    n = T.N_PER_FAMILY
    bad: list[str] = []
    seen_lane = False
    for lane, man in _lane_manifests():
        if isinstance(man, str):
            continue        # `families.select-conformance` reports this once
        if not T.runs_class(lane, cls):
            continue        # the whole class is out of this lane; not a cell
        seen_lane = True
        depth = _lane_depth(man).get(family, 0)
        if T.in_scope(lane, family):
            if depth != n:
                bad.append(
                    f"{g} {lane}: {family} holds {depth} instance"
                    f"{'' if depth == 1 else 's'} of {n}"
                    f"{'' if depth >= n else f', {n - depth} short'}. N is not "
                    f"a ceiling and a cell is not a budget to spend elsewhere: "
                    f"{n} decomposes as {T.N_KERNELS} kernels x "
                    f"{T.N_REALISATIONS} instances-per-kernel, so a family at "
                    f"{depth} is missing {n - depth} of the unit requirement "
                    f"(b) is stated in.")
        else:
            if depth:
                bad.append(
                    f"{g} {lane}: {family} is OUT of scope for this lane but "
                    f"the corpus holds {depth} of its instances. A cell that "
                    f"is n/a must be empty; instances here inflate the lane "
                    f"past the target it is measured against.")
            reason = T.na_reason(lane, family)
            legal = T._T["prohibitions"]
            if reason not in legal:
                bad.append(
                    f"{g} {lane}: {family} is n/a with reason {reason!r}, "
                    f"which is not one of {legal}. An n/a without a legal "
                    f"prohibition is indistinguishable from an unwritten "
                    f"family.")
    if not seen_lane:
        bad.append(f"{g} no lane that runs {cls} has an emitted corpus, so "
                   f"this cell was never looked at. Expected at least one of "
                   f"{T.IN_SCOPE.get(cls, [])}.")
    return bad


def class_instances(cls: str) -> list[str]:
    """One `m<C>.instances` cell: the class cell is the sum of its families.

    Not a second count of the same shortfalls -- `m<C>.<family>.instances`
    already reports those per family. This checks the AGGREGATION: that the
    class cell the paper prints is arithmetic over the families rather than a
    number that happens to sit next to them.
    """
    T = _tax()
    if T is None:
        return []
    g = gid(cls, "instances")
    fams = T.families_of(cls)
    cell = len(fams) * T.N_PER_FAMILY
    bad: list[str] = []
    seen = False
    for lane, man in _lane_manifests():
        if isinstance(man, str):
            continue
        if not T.runs_class(lane, cls):
            continue
        seen = True
        depth = _lane_depth(man)
        s = sum(depth.get(f, 0) for f in fams)
        emitted = sum(1 for r in man.get("mutants") or []
                      if r.get("class") == cls and not r.get("attribution_only"))
        if s != emitted:
            bad.append(
                f"{g} {lane}: the {cls} families hold {s} instances but "
                f"{emitted} {cls} mutants are emitted. An emitted mutant "
                f"counts in exactly one family of its own class, so the cell "
                f"and the rows cannot disagree.")
        declared = T._T["budget"]["classes"][cls]["cell"]
        if declared != cell:
            bad.append(
                f"{g} the budget declares the {cls} cell as {declared} but "
                f"{len(fams)} families x {T.N_PER_FAMILY} = {cell}.")
    if not seen:
        bad.append(f"{g} no lane that runs {cls} has an emitted corpus, so "
                   f"this cell was never looked at. Expected at least one of "
                   f"{T.IN_SCOPE.get(cls, [])}.")
    return bad


# ---------------------------------------------------------------------------
# guards.vocabulary -- the set of guard ids is closed
# ---------------------------------------------------------------------------
# The ids this file registers are the only names a guard may have, and this is
# the guard that says so. Two rules:
#
#   V1  no guard is named by a *position*. `\bG<number>\b` is an ordinal, and an
#       ordinal is minted locally -- on 2026-09-17 three separate files had each
#       minted "position 11" for a different object, and a fourth used G1..G7
#       for launch geometry. A position cannot be looked up; a path can.
#
#   V2  a token that *looks* like a guard id -- its root is one of ROOTS, a
#       class, or a lane -- must BE one. This is what makes a rename total: a
#       doc still saying `m1.coverage` after the id moved fails the build,
#       instead of reading as current. Without it the rename is a convention.
#
# Both rules are deliberately narrow, because a guard that cries wolf earns an
# exemption and then guards nothing. `_ID_TOKEN` only matches a dotted path, and
# only a token rooted in the guard vocabulary is judged at all -- so
# `stats.json`, `method-taxonomy.json`, `schema.records`, `e.g.`, `i.e.` and
# `Fig.3` are invisible to it. Two genuine namespaces are recognised by shape:
# `M1.12` is a declaration and `M1.a` is a family, neither of which is a guard.
#
# The one other `G<n>` vocabulary in the tree is launch geometry. Those live in
# `GEOMETRY_FILE` and must say what they are -- `launch-constraint G1` -- with
# a geometry word on the line, so the two cannot be confused again.

# The one other `G<n>` vocabulary in the tree is launch geometry. It keeps its
# letters -- the constraint, not the guard, is what `G` reads as there -- but the
# file that uses them must say so. Otherwise neither a reader nor this guard can
# tell `G3` the block-dimension cap from `G3` the restated-constants check, and
# that ambiguity is exactly how the guard ordinals came to collide.
GEOMETRY_FILE = "hardware-constraint-inventory.md"
GEOMETRY_DECLARATION = ("launch constraint", "launch-constraint",
                        "launch geometry", "launch-constraints")

# Ids the plan has reserved and this file does not implement yet. Naming them
# here lets a doc cite them without the vocabulary guard calling the citation
# stale, so a plan can be written before its gate.
#
# **Empty as of 2026-09-17.** Every id the plan reserved -- `families.partition`,
# `families.select-conformance`, `families.no-shared-instance`,
# `families.dead-vs-unwritten`, `families.ids-canonical` and the `m<C>.instances`
# / `m<C>.<family>.instances` cells -- now has a guard, so a hole here would be
# a lie rather than a plan. The mechanism stays because reserving an id is how
# the next one gets named before it is written; what must not stay is a
# reservation for something that already exists.
RESERVED: tuple[str, ...] = ()

_ID_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+")
_ORDINAL = re.compile(r"\bG[0-9]{1,2}\b")
# Namespaces that are NOT guards but have the same dotted shape, recognised by
# shape so that a legitimate use is not reported as a stale one.
# `M1.12`, `M1.19a`, `M1.14.sm1.store`, `M3.17-M3.26`, `M3.13-narrowed`: once
# the component after the class is numeric, the token is a declaration or an
# instance under one, never a guard -- a guard's second component is a word.
_DECLARATION = re.compile(r"^m[1-4]\.\d+[a-z0-9-]*(?:\..*)?$")   # M1.12, M1.14.sm1.store
_BARE_FAMILY = re.compile(r"^m[1-4]\.[a-z]{1,2}$")                # M1.a, M3.x
_INSTANCE = re.compile(r"^m[1-4]\.s[0-9]")                        # M1.s5.sm11.store
# A trailing filename component. `mutants.py` and `choreo.h` share a root with
# a real guard id and are not one.
_EXTENSIONS = frozenset((
    "py", "pyi", "sh", "bash", "json", "jsonl", "md", "tex", "sty", "bib",
    "txt", "cfg", "toml", "ini", "yaml", "yml", "csv", "tsv", "log", "out",
    "aux", "bbl", "blg", "fls", "fdb_latexmk", "pdf", "png", "svg",
    "c", "h", "hh", "hpp", "cc", "cpp", "cxx", "cu", "cuh", "co", "rs", "go",
    "npy", "npz", "pt", "so", "o", "a", "bin", "lock",
))

# `--guard` ids are the object's path in the axis. A token is judged only if its
# first component is one of these; anything else belongs to another vocabulary.
GUARD_ROOTS = frozenset(ROOTS) | frozenset(c.lower() for c in CLASSES)


def _id_status(tok: str, registered: set[str], lanes: set[str]) -> bool | None:
    """Judge one dotted token: `None` = not guard vocabulary, `False` = bad id.

    A lane may prefix an id -- `choreo.m1.a` is that family in that lane -- so a
    lane head is stripped and the remainder judged in its place.
    """
    head, _, tail = tok.partition(".")
    if tail.split(".")[-1] in _EXTENSIONS:
        return None
    if head in lanes:
        return False if not tail else _id_status(tail, registered, lanes=frozenset())
    if head not in GUARD_ROOTS:
        return None
    if tok in registered:
        return True
    if _DECLARATION.match(tok) or _BARE_FAMILY.match(tok) or _INSTANCE.match(tok):
        return True
    return False


_SCAN_EXTS = {".py", ".json", ".md", ".sh", ".tex", ".toml", ".cfg", ".txt"}
_SCAN_NAMES = {"Makefile", "makefile"}
# V2 judges *citations*, so it reads only files a citation can live in. A code
# file names its ids by construction -- a finding string is built from the id
# that `CHECKS` registers -- so scanning code for dotted tokens would only turn
# `mutants.append` into a finding.
_DOC_EXTS = {".md", ".tex"}
# Generated output and vendored code. `raw/` holds one run script per mutant and
# `mutants/` holds the emitted corpus; neither is edited and both are large.
_SCAN_SKIP = {"extern", "__pycache__", ".git", "node_modules", ".venv",
              "raw", "mutants", "results", "build", "_build", ".mypy_cache",
              ".pytest_cache"}
_SCAN_LIMIT = 8      # findings listed per file before the rest are counted
# This file has to name the old ordinals in order to forbid them, so V1 could
# not apply to it. That exemption is *budgeted* rather than open-ended, so a
# `G<n>` newly added to the code still trips the guard instead of hiding behind
# the file's own name. Raise this in the same commit that grows the record.
SELF_ORDINAL_BUDGET = 13


def _vocabulary_files() -> list[Path]:
    """The files that may carry a guard id: this project's code and its plans."""
    roots = [Path(B2), Path(B2).parent.parent / "eurosys27"]
    here = Path(__file__).name
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name == here:
                continue      # this file names the ordinals in order to forbid
                              # them, and registers them in `RESERVED`. V1's
                              # exemption here is checked in `guards_vocabulary`
                              # against `SELF_ORDINAL_BUDGET` and the comment rule.
            if _SCAN_SKIP & set(p.parts):
                continue
            if p.suffix not in _SCAN_EXTS and p.name not in _SCAN_NAMES:
                continue
            out.append(p)
    return out


def guards_vocabulary() -> list[str]:
    """Every guard-shaped token in the tree is a registered id.

    Registered means: named in this file's `CHECKS`, in `RESERVED`, or one of
    those behind a lane name -- `choreo.m1.a` is the family, `m1.a.instances` the
    reserved guard. Three namespaces share the dotted shape and are all
    recognised: `M1.12` a declaration, `M1.a` a family, `M1.s5.sm11.store` an
    instance. So are filenames, by their extension.

    V1 fails on a residual `G<n>` anywhere except in the launch-geometry file,
    which keeps those letters but must declare the namespace it keeps them for.
    This file is exempt so the collision can be described -- and that exemption
    is bounded two ways: by `SELF_ORDINAL_BUDGET`, and by a rule that an ordinal
    may sit only in a comment or in the launch-geometry message, never in code.
    It fails on reading zero files, because a scope that vanished would
    otherwise turn this into the vacuous pass it exists to prevent.
    """
    g = gid("guards", "vocabulary")
    registered = {i for i, _, _ in CHECKS} | set(RESERVED)
    lanes = set(AX.lanes())

    files = _vocabulary_files()
    if not files:
        return [f"{g} read no files at all, so it would pass vacuously. "
                f"Expected to scan {_rel(B2)} and "
                f"{_rel(Path(B2).parent.parent / 'eurosys27')}."]

    bad: list[str] = []

    me = Path(__file__)
    own: list[str] = []
    cited: list[int] = []
    for lineno, line in enumerate(me.read_text(errors="replace").splitlines(), 1):
        if not _ORDINAL.search(line):
            continue
        own += [f"{lineno}:{t}" for t in _ORDINAL.findall(line)]
        if not (line.lstrip().startswith("#")
                or any(d in line.lower() for d in GEOMETRY_DECLARATION)):
            cited.append(lineno)
    if len(own) > SELF_ORDINAL_BUDGET:
        bad.append(
            f"{g} this guard registry carries {len(own)} `G<n>` tokens, over "
            f"its documented budget of {SELF_ORDINAL_BUDGET}: "
            f"{', '.join(own)}. The self-exemption exists so the ordinal "
            f"collision can be described, not so code can cite a position. If "
            f"the record genuinely grew, raise SELF_ORDINAL_BUDGET in the same "
            f"commit; otherwise name the guard by its id.")
    for lineno in cited:
        bad.append(
            f"{g} {_rel(me)}:{lineno}: an ordinal outside a comment. In code the "
            f"only legitimate mention is the launch-geometry finding, which "
            f"names the namespace it means.")

    for path in files:
        text = path.read_text(errors="replace")
        docs = path.suffix in _DOC_EXTS
        geometry = path.name == GEOMETRY_FILE
        if geometry:
            if not any(d in text.lower() for d in GEOMETRY_DECLARATION):
                bad.append(
                    f"{g} {_rel(path)}: uses `G<n>` but never says what `G` "
                    f"means. A file that keeps those letters must declare the "
                    f"namespace it keeps them for -- `launch constraint G1`.")
        count = 0
        suppressed = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            msgs: list[str] = []
            if not geometry:
                for mo in _ORDINAL.finditer(line):
                    msgs.append(
                        f"`{mo.group(0)}` is a position, not an identity. A "
                        f"guard is named by the object it constrains; "
                        f"`--guard` takes an id from `--list`. If this is a "
                        f"launch-geometry constraint, move it to "
                        f"`{GEOMETRY_FILE}` -- or write it as "
                        f"`launch-constraint {mo.group(0)}` there.")
            if docs:
                for mo in _ID_TOKEN.finditer(line):
                    tok = mo.group(0).lower()
                    if _id_status(tok, registered, lanes) is not False:
                        continue
                    head = tok.partition(".")[0]
                    near = [i for i in sorted(registered)
                            if i.partition(".")[0] == head and i != tok]
                    hint = (f" Same namespace: {', '.join(near)}."
                            if near else "")
                    msgs.append(
                        f"`{mo.group(0)}` is not a registered guard id.{hint} "
                        f"Register it in `RESERVED` if the plan has reserved "
                        f"it, or fix the citation.")
            for m in msgs:
                if count >= _SCAN_LIMIT:
                    suppressed += 1
                    continue
                bad.append(f"{g} {_rel(path)}:{lineno}: {m}")
                count += 1
        if suppressed:
            bad.append(f"{g} {_rel(path)}: {suppressed} further finding(s) in "
                       f"this file, not listed.")
    return bad


# (id, help, fn). The id is a path into the axis, never an ordinal: see the
# "How a guard is named" note above `class_coverage`. `--guard m3` runs every
# guard whose id starts `m3.`; `--guard m3.cell` runs exactly one.
CHECKS = [
    (gid("axis", "self-consistency"),
     "the class axis is internally consistent", g1_axis),
    (gid("axis", "schema-enums"),
     "record-schema enums match the axis", g2_schema),
    (gid("axis", "restated-constants"),
     "modules that restate a constant restate it correctly", g3_constants),
    (gid("axis", "no-restated-tuples"),
     "no module outside schema/ hard-codes a class sequence", g4_literals),
    (gid("lanes", "declarations"),
     "n/a and uncompared lanes carry their reason", g5_declarations),
    (gid("lanes", "stats-conformance"),
     "committed stats.json agrees with the axis", g6_outputs),
    (gid("docs", "no-superseded-claim"),
     "the docs carry no superseded claim about the axis", g7_docs),
    (gid("corpus", "declared-files"),
     "every declared corpus file exists and carries its release", g8_corpus_release),
    (gid("mutants", "no-duplicate-injection"),
     "no lane injects the same perturbation twice; the ledger is current",
     g9_correspondence),
    (gid("mutants", "registry"),
     "choreo/mutations.py loads and exposes what the guards read",
     mutants_registry),
    (gid("mutants", "edits-apply"),
     "every operator's edits apply to its base kernel", mutants_edits_apply),
]
CHECKS += [
    (gid(cls, "coverage"),
     f"every {cls} operator injects into a category {cls} claims",
     _bind(class_coverage, cls))
    for cls in CLASSES
]
CHECKS += [
    (gid(cls, "cell"),
     f"{cls}'s declared cell is arithmetically reachable",
     _bind(class_cell, cls))
    for cls in CLASSES
]
# The `families.*` half of the method axis. `families.partition` is listed
# first because it is also the bucket for `check()`'s structural findings, which
# name no id of their own.
CHECKS += [
    (gid("families", "partition"),
     "every spec_id is a realisation of exactly one family",
     _taxonomy_guard(gid("families", "partition"))),
    (gid("families", "no-shared-instance"),
     "no spec_id is the sole realisation of two families",
     _taxonomy_guard(gid("families", "no-shared-instance"))),
    (gid("families", "dead-vs-unwritten"),
     "a dead declaration is a subset of the unwritten specs, and the rest are "
     "`avoided`", _taxonomy_guard(gid("families", "dead-vs-unwritten"))),
    (gid("families", "ids-canonical"),
     "every spec_id is spelled the way the registry spells it",
     _taxonomy_guard(gid("families", "ids-canonical"))),
    (gid("families", "admissibility"),
     "every family has an admissible declaration, or a ledgered model gap",
     _taxonomy_guard(gid("families", "admissibility"))),
    (gid("families", "select-conformance"),
     "the emitter's accounting closes and its manifest is current",
     families_select_conformance),
]
# One guard per (family x lane) cell and one per class cell. The per-family id
# is generated from the taxonomy rather than listed, because the taxonomy is the
# vocabulary it is written in: `m1.a.instances` is "the M1-a family, its
# instances", so a family that moved would move its id with it. The taxonomy
# writes the family `M1-a`; an id is a path, so it writes `m1.a`.
#
# If the taxonomy will not load there are no family ids to register, and the
# four `families.*` guards already report why. Registering none is better than
# registering ids nothing can check.
_TAXM = _tax()
if _TAXM is None:
    CHECKS += [
        (gid("families", "cells"),
         "every (family x lane) cell holds N instances or an n/a with a reason",
         lambda: [f"{gid('families', 'cells')} schema/method_taxonomy.py will "
                  f"not import, so no cell was looked at. A guard that cannot "
                  f"read its vocabulary must not pass."]),
    ]
else:
    CHECKS += [
        (gid(f.replace("-", "."), "instances"),
         f"every lane holds either {f}'s {_TAXM.N_PER_FAMILY} instances or an "
         f"n/a with a reason",
         _bind(family_instances, f))
        for f in _TAXM.families()
    ]
CHECKS += [
    (gid(cls, "instances"),
     f"{cls}'s cell is the sum of its families",
     _bind(class_instances, cls))
    for cls in CLASSES
]
# Last, and deliberately so: it judges the ids the list above registers, so it
# must be able to see all of them. It is the only guard whose subject is the
# other guards.
CHECKS += [
    (gid("guards", "vocabulary"),
     "every guard-shaped token in the tree is a registered id",
     guards_vocabulary),
]


def _matches(gid_: str, wanted: set[str]) -> bool:
    """`--guard m3` selects `m3.cell`; `--guard m3.cell` selects exactly it."""
    return any(gid_ == w or gid_.startswith(w + ".") for w in wanted)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="only print failures")
    ap.add_argument("--guard", action="append", default=None,
                    help="run only the named guard, by id or id prefix "
                         "(repeatable): --guard m3, --guard m3.cell")
    ap.add_argument("--list", action="store_true",
                    help="list every guard id and help line, then exit")
    args = ap.parse_args(argv)

    if args.list:
        for gid_, help_, _ in CHECKS:
            print(f"{gid_:<33} {help_}")
        return 0

    wanted = set(args.guard or [])
    # A `--guard` that matches nothing would otherwise run zero guards and
    # report success. That is the one way this checker can lie, and it is not
    # hypothetical: the ids it used to have are the ones the docs still cite.
    unknown = sorted(w for w in wanted
                     if not any(_matches(gid_, {w}) for gid_, _, _ in CHECKS))
    if unknown:
        print(f"no guard matches {unknown}. Known ids:")
        for gid_, help_, _ in CHECKS:
            print(f"  {gid_:<33} {help_}")
        return 2

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
    ran = 0
    for gid_, help_, fn in CHECKS:
        if wanted and not _matches(gid_, wanted):
            continue
        ran += 1
        bad = fn()
        if bad:
            total += len(bad)
            print(f"FAIL {gid_:<33} {help_}")
            for b in bad:
                print(f"     - {b}")
        elif not args.quiet:
            print(f"ok   {gid_:<33} {help_}")

    print()
    if total:
        print(f"{total} drift finding(s). The axis is "
              f"{_rel(AX.AXIS_PATH)}; fix the consumer, not the axis.")
        return 1
    print(f"no drift ({ran} guard(s) run).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
