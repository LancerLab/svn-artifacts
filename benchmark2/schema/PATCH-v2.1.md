# Schema patch — format 1.0 → 1.2 (additive), vocabulary v2.1

Applied in place to `schema/record-schema.json`. The patch is **additive**: every
v1.0 record stays valid, and no existing field changes meaning. That is a hard
requirement, not a preference — the committed E1 record
(`results/choreo/mutant.jsonl`) must re-validate byte-for-byte, because `S1`/`S2`
are published numbers. This note records the diff for the review packet; the
JSON is the schema of record.

> **Correction, 2026-09-15 — this claim was false as first applied.** The row
> below originally read `fields += <the five v2.1 fields>`. In this schema
> `fields` means *required of every record*, so that row did not add an optional
> column: it made five fields **mandatory** and thereby invalidated every record
> already committed, including all four SOTA lanes and the choreo E1 corpus. The
> patch contradicted its own header, and four separately-written validators
> (`choreo/collect.py`, `mlir-shared/mlirbench.py`, `triton/collect.py`,
> `tilelang/collect.py`) disagreed about it in four different ways.
>
> The patch is additive **now, and only because** required fields were split into
> two keys:
>
> - `fields` — the v1.0 base set, unchanged, 9 fields.
> - `fields_since` — `{"v2.1": [spec_id, path_class, prohibition, applicable,
>   spec_version]}`, the delta a record's **release** adds.
>
> A record's required set is `fields + fields_since[release]`, where
> `release = spec_version or "v1"`. So `v1` records remain valid with 9 fields
> and `v2.1` records must carry 14. `schema/records.py` is the single
> implementation of that rule; the four lane validators now delegate to it. A
> release is a statement about which fields a record *can be read for*, not a
> date and not a quality grade.

## `mutant`

| change | value | why |
|---|---|---|
| `fields` | **unchanged** — 9 fields | the v1.0 base set. This row replaces an earlier `fields +=`, which was not additive; see the correction above. |
| `fields_since` = | `{"v2.1": [`spec_id`, `path_class`, `prohibition`, `applicable`, `spec_version`]}` | the v2.1 vocabulary (`specs/expansion-workflow.md` §2) as a **release delta**. Without these a mutant cannot be placed in a path class, so `S14_path_class` cannot be computed from the record alone. Requiring them only of `v2.1` records is what keeps the patch additive. |
| `enums.class` += | `"M4"` | v2.1 added a fourth mutation class, the `LoopBound` control (host-only; see `schema/class-axis.json`). |
| `enums.path_class` = | `["P1","P2","P3","P4","L"]` | new enum. Note the asymmetry with `enums.class`: `L` is a **path** class and NOT a mutation class. |
| `enums.prohibition` = | `["","absent","derived","repaired","harness-owned","observation"]` | new enum; `""` for admissible specs. `observation` is the fifth kind, added when the `L` class was folded into the same registry — it does not come from specs §9.5.0 |
| `enums.applicable` = | `["true","false"]` | new enum (string, matching this schema's existing `exclusive`/`size` convention) |
| `enums.spec_version` = | `["v1","v2.1"]` | new enum. `v1` is a **real release**, not the absence of one: it is the default when the field is absent, which is what lets the four SOTA lanes' committed corpora validate unmodified. |
| `produced_by` += | `"screen"` | `make choreo-screen` emits `spec` records as well as failing the build |

`applicable` is the *serialized* form of `admissible` and is deliberately named
differently: `admissible` is the in-process Python attribute, and a record's
`applicable` is the verdict that travels to `stats.py`. Keeping one name for
both invited the exact bug the two-axis rule in the workflow doc guards against
(`status` vs `admissible`).

## `spec` (new record type)

One line per `SPEC_REGISTRY` entry, written by `make choreo-screen`. This is the
only record type that can represent a spec with **no** mutants, which is why it
exists: an N/A verdict must be auditable without a mutant to hang it on.

| field | meaning |
|---|---|
| `spec_id` | `M1.20`, `M3.14`, `L3`, … |
| `class` | `M1`–`M4`, `L` |
| `path_class` | `P1`–`P4`, `L` |
| `applicable` | `true`/`false` |
| `prohibition` | reason, when not applicable |
| `status` | `implemented` (an operator exists) / `pending` (admissible, unwritten) |
| `desc` | one-line statement of the corruption |
| `note` | free text; **must name the missing surface** when `prohibition == "absent"` |
| `spec_version` | `v2.1` |

`tilelang` and `iree` are the only lanes with no `spec_id` at all, so their
`spec_version` is absent throughout and they resolve to `v1`. `mlir-linalg` and
`mlir-low` carry `spec_id` but none of the three path fields; they too are `v1`,
and the axis records that partial state explicitly rather than stamping them
`v2.1`, which would demand fields the records do not have.

## `kernel`

Unchanged. The `-rtc` level is a property of the *run*, not of the kernel, so it
is carried in the run settings (`settings.py --rtc`) rather than added here.
