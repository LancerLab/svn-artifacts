# Schema patch — format 1.0 → 1.1 (additive), vocabulary v2.1

Applied in place to `schema/record-schema.json`. The patch is **purely
additive**: every v1.0 record stays valid, and no existing field changes
meaning. That is a hard requirement, not a preference — the committed E1 record
(`results/choreo/mutant.jsonl`) must re-validate byte-for-byte, because `S1`/`S2`
are published numbers. This note records the diff for the review packet; the
JSON is the schema of record.

## `mutant`

| change | value | why |
|---|---|---|
| `fields` += | `spec_id`, `path_class`, `prohibition`, `applicable`, `spec_version` | the v2.1 vocabulary (`specs/expansion-workflow.md` §2). Without these a mutant cannot be placed in a path class, so `S14_path_class` cannot be computed from the record alone. |
| `enums.class` += | `"M4"`, `"L"` | M4 and L became real mutation classes in v2.1. `L` operators are generated (attribution-only), so the enum must admit them even though they never enter the denominator. |
| `enums.path_class` = | `["P1","P2","P3","P4","L"]` | new enum |
| `enums.prohibition` = | `["","absent","derived","repaired","harness-owned","observation"]` | new enum; `""` for admissible specs. `observation` is the fifth kind, added when the `L` class was folded into the same registry — it does not come from specs §9.5.0 |
| `enums.applicable` = | `["true","false"]` | new enum (string, matching this schema's existing `exclusive`/`size` convention) |
| `enums.spec_version` = | `["v2.1"]` | new enum |
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

## `kernel`

Unchanged. The `-rtc` level is a property of the *run*, not of the kernel, so it
is carried in the run settings (`settings.py --rtc`) rather than added here.
