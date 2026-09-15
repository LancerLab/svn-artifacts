#!/usr/bin/env python3
"""The canonical mutation-class axis, loaded from `schema/class-axis.json`.

WHY THIS FILE EXISTS
--------------------
The class axis used to be restated in six places -- `record-schema.json`'s
enums, `choreo/stats.py`, `render.py`, `mlir-shared/lane.py`, `iree/collect_stats.py`
and `triton/collect.py` -- each with its own list, its own order, and its own idea
of which classes exist. The lanes drifted to a 3-class dialect (`("M1","M2","M3")`)
left over from the spec revision that treated iteration-validity as folded into
M1, so **M4 was silently absent from every SOTA lane's detection matrix**. A
missing class and an `n/a` class are different claims, and the renderer could not
tell them apart because the lanes did not say which they meant.

Now the list lives in one JSON file and every consumer derives from it:

    import class_axis as AX
    AX.mutation_classes()      # ["M1","M2","M3","M4"]  -- the class axis
    AX.obligation_classes()    # ["elem","shape","loop","hw"] -- S8's axis
    AX.path_classes()          # ["P1","P2","P3","P4","L"] -- orthogonal
    AX.lane_status("iree")     # {"M1": "n/a", "M2": "measured", ...}

`schema/check_class_axis.py` fails the build if any consumer restates a literal
that disagrees with this file, so the drift cannot come back silently.

THE ONE RULE THAT MATTERS
-------------------------
`measured` / `n/a` / `uncompared` are three different things and must stay three:

  measured    -- the lane ran the class; S1 has a cell for it.
  n/a         -- the lane RAN the class and its surface cannot express the defect.
                 Measured verdict, carries a reason, n_na = n_target_per_class.
  uncompared  -- the lane was never run against the class. The lane emits NO
                 cell. Reporting this as a 0 or as `n/a` would dress a coverage
                 gap up as a capability limit (or as a detection).

`L` is a PATH, not a class: it appears in `path_class` and in no `class` enum.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent                # benchmark2/schema/
AXIS_PATH = HERE / "class-axis.json"


@lru_cache(maxsize=1)
def axis() -> dict:
    """The parsed axis file. Cached; the file is read once per process."""
    return json.loads(AXIS_PATH.read_text())


def mutation_classes() -> list[str]:
    """The class axis, in canonical order. Every S1/S12 table iterates this."""
    return list(axis()["mutation_classes"])


def obligation_classes() -> list[str]:
    """S8's axis (`elem|shape|loop|hw`), in canonical order.

    Same four classes as `mutation_classes()`, under the names the obligation
    and expressibility records use. The bijection is class -> obligation in
    `AXIS["classes"]`; `obligation_of()` resolves it.
    """
    return list(axis()["obligation_classes"])


def path_classes() -> list[str]:
    """`P1|P2|P3|P4|L`. Orthogonal to the class axis; `L` lives only here."""
    return list(axis()["path_classes"])


def class_ids() -> list[str]:
    return [c["id"] for c in axis()["classes"]]


def class_info(cls: str) -> dict:
    for c in axis()["classes"]:
        if c["id"] == cls:
            return c
    raise KeyError(f"unknown mutation class {cls!r}; "
                   f"known: {mutation_classes()}")


def obligation_of(cls: str) -> str:
    """`M1` -> `elem`, `M4` -> `loop`, ..."""
    return class_info(cls)["obligation"]


def class_of_obligation(obl: str) -> str:
    """`loop` -> `M4`. Inverse of `obligation_of`."""
    for c in axis()["classes"]:
        if c["obligation"] == obl:
            return c["id"]
    raise KeyError(f"unknown obligation class {obl!r}; "
                   f"known: {obligation_classes()}")


def control_classes() -> list[str]:
    """Classes that are controls, not scored points (M4)."""
    return [c["id"] for c in axis()["classes"] if c.get("role") == "control"]


def scored_classes() -> list[str]:
    """The complement of `control_classes()` -- what a headline rate may cover."""
    return [c for c in mutation_classes() if c not in control_classes()]


def n_target_per_class() -> int:
    return int(axis()["n_target_per_class"])


def lanes() -> list[str]:
    return list(axis()["lanes"])


def lane_status(lane: str) -> dict[str, str]:
    """`{class: status}` for one lane, in canonical order."""
    try:
        raw = axis()["lanes"][lane]["status"]
    except KeyError:
        raise KeyError(f"unknown lane {lane!r}; known: {lanes()}") from None
    return {c: raw[c] for c in mutation_classes()}


def lane_where(lane: str, status: str) -> list[str]:
    """Classes this lane holds at `status` (e.g. `lane_where("iree","n/a")`)."""
    return [c for c, s in lane_status(lane).items() if s == status]


def lane_display(lane: str) -> str:
    return axis()["lanes"][lane].get("display", lane)


def lane_stats_key(lane: str) -> str | None:
    """The key `lane` writes its S1 block under in its own `stats.json`.

    The lanes disagree today -- choreo and triton write `S1_detection_matrix`
    (their S1 block is nested under `per_class`), the three MLIR-side lanes and
    iree write `S1_detection` flat. `render.py` normalizes both into
    `canonical_stats_key()`. The spelling is recorded per lane so the difference
    is enumerated and checked rather than accidental.
    """
    return axis()["lanes"][lane].get("stats_key")


def canonical_stats_key() -> str:
    """The key the MERGED view and the paper tables use (`S1_detection`)."""
    return axis()["canonical_stats_key"]["key"]


def lane_stats_paths(lane: str) -> list[str]:
    """Where to look for `lane`'s `stats.json`, relative to benchmark2/.

    Ordered; the first that exists wins. A lane may legitimately write two
    places (triton does), which is why this is a list.
    """
    return list(axis()["lanes"][lane].get("stats_paths", []))


def lane_owns(lane: str) -> list[str]:
    """Statistics `lane` owns (statistics-manifest.md)."""
    return list(axis()["lanes"][lane].get("owns", []))


def ready_lanes() -> list[str]:
    """Lanes that have published results (everything but tilelang today)."""
    return [l for l in lanes() if axis()["lanes"][l].get("ready")]


def not_ready_lanes() -> list[str]:
    return [l for l in lanes() if not axis()["lanes"][l].get("ready")]


def na_reason(lane: str, cls: str) -> str:
    """Why `lane` reports `n/a` for `cls`. Raises if that pair is not `n/a`."""
    if lane_status(lane)[cls] != "n/a":
        raise KeyError(f"{lane}/{cls} is {lane_status(lane)[cls]!r}, not 'n/a'")
    try:
        return axis()["na_reasons"][lane][cls]
    except KeyError:
        raise KeyError(f"{lane} declares {cls} n/a but class-axis.json has no "
                       f"na_reasons entry for it") from None


def uncompared_reason() -> str:
    return axis()["uncompared_reason"]


def uncompared_key() -> str:
    """The stats.json key every lane writes to declare a class it never ran.

    Shape: ``{classes: [...], reason: str, note: str}``. An empty object means
    the lane has no uncompared class. The point is that a class missing from S1
    is a STATED scope decision, not a missing loop iteration.
    """
    return axis()["uncompared_key"]["key"]


def uncompared_declaration(lane: str) -> dict:
    """The declaration block `lane` is required to write."""
    return {
        "classes": sorted(uncompared_classes(lane)),
        "reason": uncompared_reason(),
    }


def uncompared_classes(lane: str) -> list[str]:
    return lane_where(lane, "uncompared")


def lane_spec_version(lane: str) -> dict:
    """`lane`'s corpus release: ``{corpus, corpus_paths, note, port}``.

    This is PROVENANCE, not a verdict. A lane declaring ``v1`` is not weaker
    than one declaring ``v2.1`` -- it is a statement about which fields its
    committed records can be read FOR. The v2.1 path-class vocabulary
    (`path_class`, `prohibition`, `applicable`) landed 2026-09-11, one day after
    every SOTA lane's `raw/` was last written, so those lanes hold v1 evidence
    and their records carry none of it. Recovering the path axis for them means
    regenerating a corpus, not re-parsing one.

    Without this field the fork was invisible: `record-schema.json` listed the
    v2.1 fields as required of every record, which retroactively invalidated
    every pre-existing lane and made four separate validators reject their own
    committed corpora. `schema/records.py` now resolves required fields per
    release, and gate G8 in `check_class_axis.py` checks each ready lane's
    records against the release declared here -- so a lane cannot silently hold
    records of a release it does not declare, and a v1 corpus cannot be
    mislabelled as v2.1 to make a gate pass.
    """
    return axis()["lanes"][lane]["spec_version"]


def lane_corpus_release(lane: str) -> str:
    """Shorthand for `lane_spec_version(lane)["corpus"]`."""
    return lane_spec_version(lane)["corpus"]


def lane_corpus_entry(lane: str, path: str) -> dict:
    """The declared release and v2.1 field set for one of `lane`'s files.

    Per-path, because one lane's files genuinely differ. `choreo` declares two:
    its manifest is `v2.1` and carries four of the five v2.1 fields, while
    `raw/e1_mutant_records.json` is still the `v1` baseline that predates them.
    A lane-level list would have averaged those two into one false statement --
    which is exactly what the first version of this declaration did.
    """
    for e in lane_spec_version(lane)["corpus_paths"]:
        if e["path"] == path:
            return e
    raise KeyError(f"{lane} declares no corpus at {path!r}")


def lane_corpus_paths(lane: str) -> list[str]:
    """The files whose release the lane declares, relative to benchmark2/.

    Note these are `raw/` inputs, not `results/`. The declaration is about what
    the lane MEASURED; a lane's `stats.json` is derived from them.
    """
    return [e["path"] for e in lane_spec_version(lane)["corpus_paths"]]


def lane_corpus_release_at(lane: str, path: str) -> str:
    """The release declared for ONE file. See `lane_corpus_entry`."""
    return lane_corpus_entry(lane, path)["release"]


def lane_corpus_kind_at(lane: str, path: str) -> str:
    """`"records"` or `"plan"` for ONE file. See `lane_corpus_entry`.

    A `plan` is a generation input -- choreo's manifest names the mutants it
    intends to build, and uses `admissible`, the in-process attribute name,
    where a v2.1 mutant RECORD must carry `applicable`, the serialized one. So
    the full mutant field set governs `records` files only; a `plan` is checked
    for the register vocabulary it shares with v2.1, and for nothing else.
    """
    return lane_corpus_entry(lane, path)["kind"]


def lane_v21_fields_at(lane: str, path: str) -> list[str]:
    """The v2.1 fields declared present in ONE file. See `lane_corpus_entry`."""
    return list(lane_corpus_entry(lane, path)["v21_fields_present"])


def lane_release_is_consistent(lane: str) -> str | None:
    """`None` if `lane`'s declared release is well formed, else the complaint.

    Checks the declaration against itself and against the axis's own vocabulary,
    and refuses a declaration that points at files which do not exist -- a path
    typo would otherwise make gate G8 pass by having nothing to check.

    A file may sit BELOW the lane's target release: the target is a statement of
    where the corpus is going, and the per-path `release` is where it is. That is
    not an error, but the gap must be named in `port`, so this returns a
    complaint for an unnamed gap -- otherwise "we intend to port this" would be
    indistinguishable from "we ported this".
    """
    sv = lane_spec_version(lane)
    target = sv["corpus"]
    if target not in SPEC_VERSIONS:
        return (f"{lane}: declares corpus release {target!r}, which is not one "
                f"of {list(SPEC_VERSIONS)}")
    if lane_status(lane).get(mutation_classes()[0]) == "not_ready":
        return None
    behind = []
    for e in sv["corpus_paths"]:
        if e["release"] not in SPEC_VERSIONS:
            return (f"{lane}: {e['path']} declares release {e['release']!r}, "
                    f"which is not one of {list(SPEC_VERSIONS)}")
        if not (HERE.parent / e["path"]).exists():
            return (f"{lane}: declares corpus {e['path']}, which does not exist. "
                    f"A declaration pointing at a missing file cannot be checked, "
                    f"so it must not pass silently")
        if e["release"] != target:
            behind.append(e["path"])
    if behind and not str(sv.get("port") or "").strip():
        return (f"{lane}: target release is {target} but {behind} are still at "
                f"an older release, and the declaration names no `port`. The gap "
                f"is real work; leaving it unnamed makes the target read as "
                f"already achieved")
    return None


#: Legal values of `lanes.<lane>.spec_version.corpus`. Mirrors
#: `schema/records.py` SPEC_VERSIONS; G3 checks the two agree.
SPEC_VERSIONS = ("v1", "v2.1")


def not_classes() -> dict[str, dict]:
    """`L` (a path) and `M5` (a rejected proposal) -- the two things that are
    NOT mutation classes, so a reader stops looking for them."""
    return {n["id"]: n for n in axis()["not_classes"]}


def assert_no_class_named(bad: str) -> None:
    """Guard for a class enum that must NOT contain `bad` (e.g. `L`)."""
    if bad in mutation_classes():
        raise AssertionError(f"{bad!r} must not be a mutation class: "
                             f"{not_classes().get(bad, {}).get('meaning', '')}")
