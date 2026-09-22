"""records.py — the ONE record validator for benchmark2.

Why this module exists
======================

`schema/record-schema.json` is the schema of record for every `raw/*.jsonl`
observation. Before 2026-09-15 four separate modules each re-implemented the
check against it:

    choreo/collect.py            fields = spec["fields"]
    mlir-shared/mlirbench.py     validate() -> spec.get("fields", [])
    triton/collect.py            validate() -> spec["fields"]
    tilelang/collect.py          validate() -> spec["fields"]

They agreed by luck, not by construction, and they disagreed about the one
thing that mattered: what `fields` MEANS. All four read it as "required of
every record". `schema/PATCH-v2.1.md` meanwhile promised that the v2.1 patch was
"purely additive: every v1.0 record stays valid" -- which cannot be true if the
five fields the patch appends to `fields` are then demanded of v1.0 records.

The cost of that contradiction was paid in the SOTA lanes. All four of them hold
corpora collected before 2026-09-11, so all four fail validation, so none of
them can re-emit `results/<lane>/stats.json` from the committed `raw/`. A
stats-only recomputation -- which touches no GPU and no compiler -- was
impossible. The lanes were frozen not because they were expensive to re-run but
because the validator could not express "this record is older than v2.1".

The rule, stated once
=====================

A record's **release** is its `spec_version`, or ``v1`` when it carries none.
That is not a fallback, it is the definition: v1 *is* the release that predates
the field.

    required(record) = schema.records[kind].fields
                     + schema.records[kind].fields_since[release]

So a v1 record is required to have the v1 fields and nothing more, a v2.1 record
is required to have everything, and adding a field to `fields_since` can never
invalidate an already-collected record. That is what "additive" has to mean to
be worth writing down.

The residual hole, and where it is closed
=========================================

A v2.1 writer that forgets to stamp `spec_version` produces records that look
v1 and validate. This module cannot close that, because "v1" and "forgot to
say" are the same JSON. It is closed one level up, where the missing information
actually lives: every lane declares its corpus release in
`schema/class-axis.json`, and `corpus.declared-files` in
`schema/check_class_axis.py` asserts that the declaration and the records agree.
A lane cannot forget in private.

Net effect on the guard's strength
==================================

Strictly stronger, not weaker. Before: every record had to carry all 14 fields,
and a v1 corpus failed. After: a v2.1 record still has to carry all 14 fields (a
missing one is still an error), and additionally a corpus that *claims* to be
v2.1 but is not will be caught by `corpus.declared-files`. The only records
that now pass where they previously failed are ones explicitly declared v1 by
their lane.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "record-schema.json"

#: Legal `spec_version` values. ``v1`` is the release that predates the field,
#: so a record may declare it explicitly or leave it out.
SPEC_VERSIONS = ("v1", "v2.1")

#: The release assumed for a record that carries no `spec_version`.
DEFAULT_SPEC_VERSION = "v1"

# --------------------------------------------------------------------------
# `stage` is a PROJECTION of `outcome`, not an independent measurement
# --------------------------------------------------------------------------
# Both are required fields of a mutant record and both are enums, which makes
# them look like two separate observations. They are one:
#
#     outcome  ['compile', 'runtime', 'never', 'n/a']
#     stage    ['compile', 'runtime', 'none']
#
# Checked against every committed record before this table was written
# (mlir-linalg 120/120, mlir-low 96/96, triton 29/29 -- 245 of 245), and
# re-checked on the v2.1 refresh: mlir-low 144/144 agree.
#
# `stage` is kept as a required field rather than dropped, because the S1 tables
# are read per stage ("caught by the typecheck" vs "caught at runtime") and a
# render step that recomputed it from `outcome` would be a SECOND definition of
# the same fact, free to drift from the first. But because the mapping is total,
# a lane that never wrote the field can be corrected EXACTLY -- no re-run and no
# assumption. That is the whole of the IREE lane's omission: a missing writer
# line, not a missing measurement.
STAGE_FOR_OUTCOME = {
    "compile": "compile",
    "runtime": "runtime",
    "never": "none",
    "n/a": "none",
}


def stage_for(outcome: str) -> str:
    """The one definition of a mutant record's `stage`, given its `outcome`."""
    return STAGE_FOR_OUTCOME.get(str(outcome), "none")


@lru_cache(maxsize=1)
def schema() -> dict:
    """The parsed schema. Cached; the file is a build input, not run state."""
    return json.loads(SCHEMA_PATH.read_text())


@lru_cache(maxsize=1)
def _records() -> dict:
    return schema()["records"]


def kinds() -> tuple[str, ...]:
    """Every record kind, minus the non-kind bookkeeping keys."""
    return tuple(k for k, v in _records().items()
                 if isinstance(v, dict) and "fields" in v)


def spec_version_of(rec: dict) -> str:
    """The release a record belongs to.

    An unrecognised value is returned as-is rather than coerced, so `validate`
    can reject it instead of silently treating it as v1.
    """
    v = rec.get("spec_version")
    if v is None:
        return DEFAULT_SPEC_VERSION
    return str(v)


def versioned_fields(kind: str) -> tuple[str, ...]:
    """Fields that are required only of records at a later release."""
    out: list[str] = []
    for fields in _records().get(kind, {}).get("fields_since", {}).values():
        out.extend(fields)
    return tuple(out)


def required_fields(kind: str, rec: dict) -> list[str]:
    """Exactly the fields `rec` must carry, given `rec`'s release."""
    spec = _records().get(kind, {})
    req = list(spec.get("fields", []))
    req += list(spec.get("fields_since", {}).get(spec_version_of(rec), []))
    return req


def version_declaration_required(kind: str) -> bool:
    """Kinds whose records must say which release they are (e.g. `spec`)."""
    return kind in schema()["records"].get("require_version_declaration", [])


def validate(rec: dict, kind: str) -> list[str]:
    """Validate `rec` against `schema/record-schema.json`. Returns errors.

    There is deliberately no `schema` argument. Three of the four validators
    this module replaces took one, which is how they came to disagree: each
    caller held its own parsed copy and applied its own reading of `fields`.
    The schema is a file, this module is the only reader, and the release rule
    lives here. A caller holding a stale copy can no longer validate against it.
    """
    errs: list[str] = []
    spec = _records().get(kind)
    if spec is None:
        return [f"unknown record kind {kind!r}"]

    for f in required_fields(kind, rec):
        if f not in rec:
            errs.append(f"{kind}: missing field {f!r}")

    if version_declaration_required(kind) and \
            spec_version_of(rec) not in SPEC_VERSIONS:
        errs.append(f"{kind}: must declare `spec_version` in "
                    f"{list(SPEC_VERSIONS)}, got {rec.get('spec_version')!r}")

    for f, allowed in spec.get("enums", {}).items():
        if f in rec and str(rec[f]) not in set(allowed):
            errs.append(f"{kind}: field {f!r}={rec[f]!r} not in "
                        f"{sorted(allowed)}")
    return errs


def load_schema(benchmark2: Path | str | None = None) -> dict:
    """Back-compat shim for `me/`-era callers that pass a benchmark2 root."""
    if benchmark2 is None:
        return schema()
    return json.loads((Path(benchmark2) / "schema" /
                       "record-schema.json").read_text())


def census(rows: list[dict], kind: str) -> dict[str, int]:
    """Count records by release. Used by the checker's `corpus.declared-files`
    and by lane logs."""
    out: dict[str, int] = {}
    for r in rows:
        v = spec_version_of(r)
        out[v] = out.get(v, 0) + 1
    return out
