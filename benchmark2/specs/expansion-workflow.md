# Expansion workflow — spec → operator → table

Added 2026-09-11. Companion to `mutation-specs-v2.md` (the spec of record) and
`mutation-expansion-m1m2m3.md` (the derivation). This file is the **operating
manual**: it says what order things happen in, which artifact each step writes,
what the gate at each step is, and what must never be skipped.

Nothing here changes a number that has already been published. Every field this
document introduces is **additive**; see §7 for the one change that is *not*
additive and is therefore reserved.

---

## 1 The pipeline

```
                 ┌─ spec of record ──────────────────────────────────┐
                 │  specs/mutation-specs-v2.md  (§1–§4 class tables,   │
                 │  §9.6 path classes, §9.6.3 provisional)             │
                 └───────────────────────┬───────────────────────────┘
                                         │
   GATE 1  screen by path class          │  §3 below. P2 → N/A, stop.
                                         ▼
                 ┌─ mutations.py ────────────────────────────────────┐
                 │  SPEC_REGISTRY  (spec_id → class, path, class-of-  │
                 │  admissibility) + Mut operators (old,new,n) edits   │
                 └───────────────────────┬───────────────────────────┘
                                         │
   GATE 2  every operator resolves       │  gen_mutants.py --dry-run --report
                                         ▼
                 ┌─ gen_mutants.py ──────────────────────────────────┐
                 │  mutants/<class>/<category>/<id>__<case>.co        │
                 │  raw/mutant_manifest.json  (spec_id, path_class,    │
                 │  prohibition, admissible, settings_hash, diff …)   │
                 └───────────────────────┬───────────────────────────┘
                                         │
   GATE 3  oracle is calibrated          │  calibrate_oracle.py
                                         ▼
                 ┌─ run_e1.py ───────────────────────────────────────┐
                 │  raw/e1_*.json  → compile | runtime | never | n/a   │
                 └───────────────────────┬───────────────────────────┘
                                         │
                 ┌─ analyze_never.py ────────────────────────────────┐
                                         │
                                         ▼
                 ┌─ collect.py ──────────────────────────────────────┐
                 │  results/choreo/mutant.jsonl  (schema-validated)    │
                 └───────────────────────┬───────────────────────────┘
                                         │
                 ┌─ stats.py ────────────────────────────────────────┐
                 │  results/choreo/stats.json                          │
                 │    S1/S2  … unchanged (published contract)          │
                 │    S14    path-class denominator (NEW, additive)    │
                 └───────────────────────┬───────────────────────────┘
                                         │
                 ┌─ render.py + Makefile ────────────────────────────┐
                 │  render/tables/*.tex   render/figures/*.pdf         │
                 └───────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                     make paper  →  ../eurosys27/{tables,figures}/
```

`make paper` targets the **live** venue directory, `svn/eurosys27`. The paper was
retargeted from ASPLOS to EuroSys; `svn/asplos27` is the pre-retarget snapshot
and must not be written to. See §11 for what `make paper` can and cannot supply.

Run it with `make choreo-all`, or stage by stage: `make choreo-screen`,
`make choreo-minimal`, `make choreo-e1` (alias), `make choreo-never`,
`make choreo-collect`, `make choreo-stats`, `make render`, `make paper`.

---

## 2 The vocabulary this workflow depends on

Four things were added to the vocabulary by `mutation-specs-v2.md` §9.5.0 and
§9.6 and must be present on **every** mutant record before it can be reported.

| Term | Meaning | Where it lands |
|---|---|---|
| `spec_id` | v2.1 id, e.g. `M1.20`, `M3.14`, `M4.1`, `L3` | manifest → register → stats |
| `path_class` | `P1` assessed · `P2` hard-error · `P3` unchecked · `P4` warning-only · `L` launch-status | manifest → register → stats |
| `prohibition` | if not applicable, **why** (see below) | manifest → register → stats |
| `admissible` | `false` for `P2`, for noop-by-construction specs (M1.6, M4.4), and for every `L` spec | manifest → register → index |

`prohibition` has five values. Four come from §9.5.0; `observation` was added when
the launch-status class was folded into the same registry (`mutations.py`).

| Value | Meaning |
|---|---|
| `absent` | the suite cannot express the corrupted state — no source surface. **The entry must name the missing surface**, or the gap is not actionable |
| `derived` | the prohibition is not stated, only derived by the model from one source |
| `repaired` | the model **forbids** the state, so the compiler refuses it — `P2`, no test to run |
| `harness-owned` | the check lives in the harness (the E2 obligation suite), not the compiler |
| `observation` | attribution-only (`L` class): the mutant is generated, but its outcome can only ever be *rejected launch*, so it enters no admissible denominator |

**Mutatable surface = `P1 ∪ P3 ∪ P4`, minus the noop controls.** A spec whose
only governing path is `P2` is **not applicable** — a positive verdict, not a
gap. It must be *recorded* and must *not* be generated.

**Two orthogonal axes — do not merge them.** `status` (implemented/pending) says
whether an operator *exists*; `admissible` says whether the spec may enter the
denominator. M1.6, M3.6, M4.4, L1, L2 and L4 **are generated** (their
non-detection is the point) yet stay out of the denominator. Merging the axes
either loses the controls or inflates the denominator. `pending` therefore means
only *admissible and not yet written* — the open-work list.

---

## 3 GATE 1 — screen a new spec by path class

Before a spec may enter a generation run, answer four questions. **Any `no` means
the spec is not applicable**; record it with the prohibition name and stop.

1. **Expressible?** Can the corrupted state be built through the model's own
   inputs and interfaces? If not → *absent*.
2. **Independent?** Does the model derive the two sides of the corrupted relation
   from one source? If yes → *derived* (e.g. M3.13's descriptor rank pair:
   `tma_inner_splits_` is written once, `cute_codegen.cpp:10291`, and read by both
   device sites, `:4715`/`:5018`).
3. **Survives repair?** If the compiler detects and fixes the conflict, is the
   injected state one the repair misses? If not → *repaired* (e.g. M3.13's
   swizzle overflow, explicitly split at `cute_codegen.cpp:10271` — only the
   sub-case the repair misses is injectable).
4. **Attributable?** Would a miss be a defect of the compiler, not of our harness?
   If not → *harness-owned*.

Question 1 is the one that dominates in practice: of the 28 inadmissible specs,
**22 are `absent`**. Screening is therefore also a *survey of the base suite*,
not only of the compiler. Before writing any operator, grep the base suite for
the surface the spec needs — an operator that silently matches nothing is
indistinguishable from a spec nobody wrote.

Then name the **path class**:

| Path | Mechanism | Mutant can survive? | Miss mechanism | Budget |
|---|---|---|---|---|
| `P1` | `CreateAssessment` / `Assess` → a `UsageType` | yes | cost-suppressed (`-rtc`) | full N **+ the `-rtc` curve** |
| `P2` | `Error1` / `Error` — the compiler refuses | **no** | *repaired* → **N/A** | **zero — do not generate** |
| `P3` | no check on this path | yes | not emitted to runtime | **one injection per cell** |
| `P4` | `Warning` then continue | yes | not emitted to runtime | **one injection per cell** |

**Two obligation populations.** `StaticFail(pred, UsageType)`
(`shapeinfer.cpp:36-52`) is a *counter*, not an assessment creator: statically
true → counted; statically false → the caller raises `Error1`; **not statically
known → nothing**. A feature governed only by `StaticFail` is silent for every
runtime variation, so it has two disjoint mutant populations (static → `P2` noop;
symbolic → `P3` silent). A spec must separate them, or its noop rate is an
artefact of the split. This is the mechanism behind M1.15/M1.17 and the whole
`view`/`subspan` family.

**Budget rule, stated once.**
- `P1` cells: generate once, then **sweep `-rtc`** over `entry` (shipping
  default) / `low` / `medium` / `high`. Enabled obligations 34 / 47 / 79 / 323
  (9.5×).
- `P3`/`P4` cells: **one injection per `(spec × surface)` cell**. No obligation
  exists at any threshold, so a curve sweep spends budget to re-measure a
  constant.
- `P2` and noop-by-construction specs (M1.6, M3.6, M4.4): **zero injections
  as test cells**. They stay in the corpus only as named controls, so their
  `outcome` is *evidence of correct silence*, not a miss. `M1.6`/`M3.6`/`M4.4`
  are still generated (the non-detection is the point); what is zero is their
  standing in the denominator.
- `L` cells: **one injection per cell**, attribution-only. They may never be
  counted in `n_admissible` under any path class.

---

## 4 GATE 2 — operator contract

An operator is an entry in `choreo/mutations.py`. It is a `Mut` produced by the
`_m(...)` helper:

```python
_m(mid, cls, spec, paper_category, category, case, desc, *edits)
```

where each edit is `(old, new, n)`:

| field | meaning |
|---|---|
| `old` | exact literal text in the base kernel |
| `new` | replacement |
| `n` | **required occurrence count** (`None` = at least one, replace all) |

`n` is not decoration: `gen_mutants.py` refuses a mutant whose edit matched a
different number of times than declared. That refusal is GATE 2 — a spec that
silently matches zero times is worse than a spec that fails, because it produces
a `noop` that looks like a detection miss.

The v2.1 fields travel through the same call (see §6). Transforms stay keyed by
`(class, category, case)`, never by class alone.

`make choreo-screen` runs `gen_mutants.py --dry-run --report` and fails if any
spec in `SPEC_REGISTRY` is unsatisfied — i.e. if a declared spec produced neither
an operator nor an explicit N/A record.

---

## 5 The `-rtc` curve: a property of the run — and already in the ledger

The threshold is a *run* parameter, so a cell is generated once and the level is
stamped on the corpus:

```bash
make screen                       # GATE 2 first: registry + census
make choreo-minimal RTC=entry     # headline (shipping default)
make choreo-minimal RTC=low
make choreo-minimal RTC=medium
make choreo-minimal RTC=high      # ceiling
```

`RTC=` reaches `gen_mutants.py --rtc`, which stamps the level into every record;
without it a detection rate cannot be attributed to a configuration.

**Reporting rule.** Headline numbers are `-rtc=entry` — the claim is about the
released tool, not a tuned configuration. `-rtc=high` is the ceiling. Any cell
whose `entry → high` delta is negative is a bug, not a result.

**The `enabled` half needs no rerun, and it is already measured.** The E2
obligation ledger records `cost` — the cheapest level at which an obligation is
emitted and checked — so `enabled = count(cost ≤ threshold)` is a property of the
committed ledger:

| `-rtc` | enabled | Δ vs entry | elem | hw | loop | shape |
|---|---:|---:|---:|---:|---:|---:|
| `entry` *(default)* | 675 | — | 58 | 359 | 223 | 35 |
| `low` | 855 | +180 | 238 | 359 | 223 | 35 |
| `medium` | 936 | +261 | 319 | 359 | 223 | 35 |
| `high` | 1,151 | **+476 (1.71×)** | **534** | 359 | 223 | 35 |

The **shape** confirms the thesis: elementwise is the only steep class
(58 → 534, 9.2×) and every other class is flat. The **magnitude does not**:
`mutation-specs-v2.md` §9.2's table (34/47/79/323 = 9.5×) was measured on a
different population, so it must be re-labelled a *prediction* and the ledger
reported as the measurement. Note also that the predicted table below is keyed by
M-class while the ledger is keyed by access shape — state the mapping explicitly
when quoting one against the other. The prediction, as originally asserted:

| class | entry → high enabled | predicted curve |
|---|---|---|
| M1 | 1 → 290 | **steep** — the cost filter bites here |
| M2 | 0 → 0 | flat — compile-time dominated |
| M3 | 22 → 22 | flat — already entry-cost |
| M4 | 11 → 11 | flat — control |

A steep elementwise curve against flat loop/hw/shape curves *is* the paper's
thesis. It only becomes measurable once every mutant carries its path class,
which is why GATE 1 is step one and not a later annotation.

**The detection half does need a rerun, and only `entry` has one.** Detection at
`entry` is 35/72. The recorded `-rtc=all` arm gives 37/72, a delta of **+2**,
which equals the ledger's `C1_COST_FILTER_SUPPRESSED = 2` exactly. So the cost
filter owns 2 of 37 misses — *not* the dominant share. The `43/72` figure in
`S2`'s flag matrix is a different arm (`-rtc=all` **and** `--disable-assert-hoist`);
6 of its 8 extra detections come from the hoisting defect, and quoting it as the
cost filter's effect is an R-D2 violation. Levels with no run are reported as
absent, never as 0.

---

## 6 What each step writes

**`gen_mutants.py` → `raw/mutant_manifest.json`**, one record per mutant:

```
toolchain  class  spec  spec_id  path_class  prohibition  admissible  level
paper_category  mutant_id  category  case  base_path  mutant_path
kernel_hash  mutant_hash  settings_hash  local_headers  desc  diff
```

`spec` (the v1 integer) is **retained** so results produced under v1 stay
comparable. `spec_id` is the v2.1 id and is the field new tables use.

**`collect.py` → `results/choreo/mutant.jsonl`**, schema-validated against
`schema/record-schema.json`. The v2.1 fields are carried through unchanged; the
record type's `class` enum gains `M4` and `L`, and `outcome` keeps
`{compile, runtime, never, n/a}`.

**`stats.py` → `results/choreo/stats.json`**:

- `S1_detection_matrix`, `S2_before_device`, `S3…S13` — **unchanged semantics**.
  Existing published numbers must reproduce byte-for-byte. (Verified: after
  adding `S14`, every one of `S1…S13` is byte-identical to the committed
  baseline. The only other diffs are `inputs.spec`, `lane_owns`, and the
  live-read `toolchain_identity`.)
- `S14_path_class` — **new, additive.** The reporting denominator, as landed:

```
S14_path_class:
  register_source                 "results/choreo/spec.jsonl", else
                                  "choreo/raw/spec_registry.json"
  denominator:
    n_specs          = 72
    n_applicable     = admissible specs (the detection denominator)
    n_not_applicable = P2 ∪ NOOP_BY_CONSTRUCTION ∪ every recorded prohibition
    n_out_of_scope   = admissible ∧ status != implemented  (open work; must be 0)
  prohibition            {absent, derived, repaired, harness-owned, observation}
  prohibition_by_path    {L: {observation: 3, absent: 6}, P1: {absent: 12},
                          P2: {repaired: 3}, P3: {absent: 3}, P4: {absent: 1}}
  per_path:
    P1: {n_cells, n_injected, n_admissible, n_detected, n_detected_admissible,
         n_detected_pct, n_never, n_never_admissible,
         n_never_share_of_misses, n_discarded_noop}
    P2/P3/P4: {…}
    L:  {n_cells, n_injected, n_detected, n_discarded_noop}   # attribution-only
  per_spec:  M1.20 → {class, path_class, prohibition, n_cells, n_injected,
                      n_admissible, n_detected, n_detected_admissible, n_never,
                      n_discarded_noop}
  unresolved_spec_cells  [mutant_id, …]   # cells that joined NO spec
  applicability_audit
  reconciliation
  rtc_curve
```

`n_detected` counts **every** detected cell in that path; `n_detected_admissible`
counts only the ones the register admits, and `n_detected_pct` divides the
latter. Collapsing the two would blind the audit below, which looks precisely
for detections on *inadmissible* cells. `n_never_share_of_misses` is likewise
defined on **admissible-only** terms (`n_never_admissible` over admissible
misses); computing it from the all-path `n_never` yields a share above 100 %,
which is the same population error one level down.

**`applicability_audit` — the register checks its own claims.** Every
inadmissible spec is a *design* claim ("this mutation cannot corrupt anything" /
"the compiler refuses rather than tests it"). The records carry the ground truth
in `manifest: corrupts | noop`. `S14` cross-checks them and reports any spec
whose cells are `corrupts` **and** detected:

```
applicability_audit:
  status               "ok" | "CONTRADICTED"
  n_inadmissible_specs_with_cells   n_contradicting
  rows: [{spec_id, class, path_class, prohibition, desc, n_injected,
          n_detected, n_never, contradicts_design_claim, design_note}]
```

This is **reported, never auto-applied**: moving a spec back into the
denominator changes `S1`/`S2`, which §7 reserves for v3 with owner sign-off, and
would invalidate the frozen E1 baseline.

**`reconciliation` — S14 must partition the same cells as S1/S2.** Asserted, not
assumed, because the entire value of a second denominator is that a reader can
check it against the first:

```
reconciliation:
  n_cells  n_injected  n_discarded_noop  n_never   # must equal S1's totals
  n_detected_all_paths                             # must equal S2's n_before_device
  admissible_injected  admissible_detected  admissible_pct
  inadmissible_injected  inadmissible_detected
  matches_S1S2  checks  [mismatch, mismatch_note]
```

**`rtc_curve` — the enabled curve needs no rerun.** `cost` in the E2 ledger is
the *cheapest* `-rtc` level at which an obligation is emitted and checked
(`assert_site.cpp:18-20`; buckets `:123-132`; default `ENTRY` at
`context.hpp:549`), and `enabled` is the observed consequence at the pinned
level. So `enabled = count(cost ≤ threshold)` is a property of the ledger:

```
rtc_curve:
  n_obligations_enabled  {entry, low, medium, high}
  delta_vs_entry         {…}
  by_class               {elem|shape|loop|hw → {entry, low, medium, high}}
  steepest_class         cost_filter_note
  n_detected             {entry: measured, low/medium/high: null}   # NOT 0
  rtc_all_arm            the recorded -rtc=all configuration, or absent
  paper_reference        specs §9.2's predicted 34/47/79/323 table
```

Only `entry` has a measured `n_detected`, because a detection needs a full E1
run per level. The missing levels are emitted as `null` with a reason, never as
`0` — the same rule the module already applies to un-run lanes. The
`-rtc=all` arm is kept in its own block because it is **one configuration, not a
level sweep**, and because it must not be confused with the `rtc=all AND
--disable-assert-hoist` arm in `S2`'s flag matrix: 6 of that arm's 8 extra
detections come from a codegen hoisting defect (R-D2).

The `prohibition` breakdown is what makes the N/A count auditable, so `per_path`
and the breakdown must be printed together: "28 N/A" on its own is
unfalsifiable, while "28 N/A = 22 absent + 3 repaired + 3 observation" can be
checked against the registry.

**N/A is never a miss.** It is a claim about the programming model, and it is
evidence, not an excuse. It is reported *in* the denominator so a reviewer can
audit the exclusion — and `applicability_audit` is the mechanism that holds the
exclusion to the records.

**Both detection rates must be printed.** `S2`'s headline divides by all injected
cells; `S14`'s admissible rate divides only by the cells the register says choreo
was fair to ask about. They are different questions and neither may be quoted
under the other's denominator.

**The frozen baseline is joined, not re-collected.** `S14` reads the *design*
register. On the committed v1 corpus the records have no `spec_id`, so
`register_source` reports `choreo/raw/spec_registry.json` and the per-cell join
goes through `mutations.V1_SPEC_ID` (all 120 rows resolve; zero unmapped). That
is the supported path, not a degraded one:

- Re-running `collect.py` on the v1 records **fails by design** — it refuses a
  manifest whose rows carry no `spec_id`, because a partially path-classified
  register would make `S14` silently partial. So `make choreo-collect` is not
  idempotent across the v2.1 boundary; the frozen `results/` stay as they are.
- Regenerating the corpus and re-running E1 would produce a *different* corpus
  (the `11_dynamic` base kernel moved from `[1,6,6]` to `[1,32,32]`), which would
  invalidate the frozen `S1`/`S2` that every published number depends on.

Once a v2.1 run is banked, `results/choreo/spec.jsonl` exists and `S14` prefers
it. Until then the join is the correct answer, and `register_source` says which
one was used so a reader never has to guess.

**`render.py` → `render/tables/*.tex`** gains one table,
`tab:e1-path-class`, from `S14_path_class`: rows = `{applicable, N/A, out of
scope}`, columns = `{P1, P3, P4}`, plus the `entry → high` delta column.
The two detection rates from `reconciliation` (`35/72` and `27/60`) belong in
the caption: a reader who sees only one of them cannot tell which denominator
it is over.

---

## 7 The one reserved change

`mutation-specs-v2.md` §10:

> A future v3 must state which cells it invalidates. **One thing is reserved for
> v3:** the `P3`/`P4` specs of §9.6.2 change what a "miss" *means* (silence by
> construction rather than by cost), so the §7 oracle and the §9.5.0 denominator
> must both name the path class. That is a change to the reporting contract, not
> to the spec set, and needs owner sign-off.

Concretely: `S14_path_class` is additive and can be produced now. What may **not**
happen without sign-off is re-defining `n_admissible`, `n_never`, or
`pct_before_device` in `S1`/`S2` to be path-class-relative, or folding the `P3`
population into the same rate as the `P1` population. Until then, `S1`/`S2` keep
their published meaning and `S14` carries the new one, side by side.

---

## 8 Worked example — adding one spec end to end

Take **M3.14** — linear `.copy` with a dimension ≥ 2²⁴, the check is absent
(`gpu_adapt.hpp:320` `// omitted`).

1. **Screen.** Expressible ✔ (a `.copy` is a model construct). Independent ✔
   (the extent is the caller's input). Survives repair ✔ (there is no repair —
   there is no check). Attributable ✔. Path = **`P3`**, because the only thing
   that reads this path is the linear-copy lowering and it never calls
   `CheckDimSize`, unlike the other six cells of the DMA matrix. → *not* N/A.
2. **Budget.** `P3` ⇒ **one injection per cell**. No `-rtc` sweep.
3. **Operator.** Add to `M3` in `mutations.py`, on the static `matmul` base,
   violating the bound **by stride, not length** (cost rule):
   ```python
   _m("M3.14.mm1.copy2p24", "M3", 14, "dim-mismatch", "matmul",
      "1_bert_32x512x768_768x768_32x512x768",
      "linear .copy with a dim >= 2^24 reached by stride, check absent (P3)",
      ("dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;",
       "dma.copy lhs.chunkat(p#q, m_tile, k_tile) => local;"),  # + stride edit
      spec_id="M3.14", path_class="P3")
   ```
4. **GATE 2.** `make choreo-screen` — the edit must match its declared `n`.
5. **Generate + run.** `make choreo-minimal && make choreo-e1`.
6. **Collect + stats.** `make choreo-collect && make choreo-stats`; confirm the
   mutant appears under `S14_path_class.per_spec["M3.14"]` with
   `path_class: "P3"`, and that `S1`/`S2` are byte-identical to before.
7. **Report.** The `render` table shows it in the `P3` column. A miss here is a
   **finding** (the absent check), not a rate — report it beside F1 in §9.6.4,
   which points at the same `// omitted`.

Also record the **independent defect** (F1) regardless of the mutation outcome:
the omission is in the shipped checker and needs no injection to be true.

---

## 9 Checklist

- [ ] GATE 1 answered for the spec; path class named; prohibition named if N/A.
- [ ] `SPEC_REGISTRY` entry added (`spec_id`, `class`, `path_class`,
      `admissible`, `prohibition`, `desc`).
- [ ] Operator added with the v2.1 fields; every edit declares its occurrence `n`.
- [ ] `make choreo-screen` passes (no unsatisfied spec, no unexpected noop).
      It fails on any *admissible* spec without an operator, and on any
      inadmissible spec that does not name a `prohibition`.
- [ ] Budget honours the path: `P1` → curve, `P3`/`P4` → one per cell,
      `P2`/noop-control → zero.
- [ ] Bound violated by **stride, not length**; no allocation grown (M3.12 the
      sole exception, and it shrinks).
- [ ] `S1`/`S2` byte-identical after the run; `S14_path_class` present.
- [ ] Any defect found is logged in §9.6.4 / `DECISIONS-NEEDED.md`, not buried in
      a rate.
- [ ] `spec_version` stamped; v1 results left untouched.

---

## 10 Where the current gap is

For the record, as of 2026-09-11.

**Before this expansion** the registry did not exist: M1 carried v1 ids 1–6, M2
ids 1–5, M3 ids 1–3, and M4, L, the path class and applicability were absent
entirely. The committed E1 record (`injected 72 / compile 24 / runtime 11 /
never 37 / n/a 0`) is the *symptom*: 37 `never`, of which 19 are `C4_NOT_ASSESSED`
— exactly the `P3` population this workflow makes visible, and exactly what a
`P3`-aware budget would have predicted before a single injection.

**After this expansion.** `make choreo-screen` reports, from
`mutations.registry_summary()`:

| | specs | operators | admissible | N/A | with operator | unwritten |
|---|---|---|---|---|---|---|
| M1 | 21 | 73 | 16 | 5 | 17 | 4 |
| M2 | 20 | 64 | 18 | 2 | 18 | 2 |
| M3 | 16 | 51 | 6 | 10 | 7 | 9 |
| M4 | 5 | 7 | 4 | 1 | 4 | 1 |
| L | 10 | 1 | 0 | 10 | 3 | 7 |
| **total** | **72** | **196** | **44** | **28** | **49** | **23** |

`admissible 44 / inadmissible 28`, of which `absent 22 · repaired 3 ·
observation 3`. **`pending` is empty**: every admissible spec has an operator,
so the remaining 23 unwritten specs are *verdicts*, not open work. The census is
**137 selected / 0 skipped** — `0 skipped` is the real regression test, because a
malformed edit yields zero mutants and only the skip list reveals it.

Two classifications changed as a direct result of screening, and both are
findings rather than bookkeeping:

- **M3.12 and L5 → `P2` / repaired.** Both were written as resource-bound edges
  whose expected channel was a rejected launch. `lib/memcheck.hpp:106-123`
  (`CheckCtMemUsage`) raises **`Error1`** when compile-time `SHARED`/`LOCAL`
  usage exceeds `mem_capacity`, so the model *forbids* the state and the
  compiler refuses it: there is no test to run. A runtime channel exists only for
  a symbolic extent, which the suite's shared tiles do not have. This also
  explains the 28/40 M3 noops in the committed v1 baseline — v1 `M3 s3` was
  exactly this family, and `L1`/`L2` inherited its realization. `L1`/`L2` are
  kept verbatim because the frozen `S1`/`S2` define them; re-examining them is a
  v3 item, not a v2.1 one.
- **22 specs are `absent`, and 13 of them are `absent` for one reason: the base
  suite is too small for the surface.** There is no rank-5 tensor anywhere (so
  M1.17, M3.4, M3.8 are unreachable), no explicit swizzle (M3.5, M3.13), no
  explicit TMA box (M3.7), no explicit MMA/WGMMA (L6–L9), no cluster (L10), no
  `__launch_bounds__` (L3), and no `dimof`/`select` (M1.15, M1.16) — while the
  whole `M3.2`/`M3.3` family needs a strided view that the source suite never
  writes. Each entry names its missing surface in `note`, so closing one is a
  bounded task: add a base case, re-run GATE 1, write the operator.

  The one structural consequence is **`P4` has zero coverage** (M2.18 is the
  only `P4` spec, and it is `absent`). The warning-then-continue path is
  therefore unmeasured, and any claim about it must be marked as such.

  Note the framing: this concentration is a property of the *suite*, not of the
  compiler. The honest sentence in the paper is therefore "22 of the 72 specs
  are not expressible on the current 14 categories", never "22 of 72 are
  unchecked" -- the second reads as a defect count and is false.

### 10.1 What `S14` found as soon as it was computed

Four results from running `S14_path_class` against the frozen v1 baseline. All
four are *outputs of the register*, and two of them change numbers the paper
intends to print.

**1. The admissible denominator is 60, not 72.** `S14`'s partition agrees
exactly with `S1`/`S2` (`cells 120 · injected 72 · noop 48 · never 37`), but 12
of the 72 injected cells belong to specs the register calls **inadmissible**, and
**8 of `S2`'s 35 detections land on them**:

| denominator | rate |
|---|---|
| all injected (S2's headline) | **35/72 = 48.61 %** |
| admissible only (the register) | **27/60 = 45.00 %** |

Both must be printed, each under the denominator it names. Quoting 48.61 % as an
admissible-denominator rate overstates it by 3.6 points.

**2. The applicability audit contradicts 2 of the 3 inadmissible specs that have
cells.** A prohibition is a *design* claim; the manifest is the ground truth:

| spec | claim | records |
|---|---|---|
| `M1.6` "zero-stride / empty range" | `absent` — "noop by construction, there is no prohibition to enforce" | **7 injected, 6 detected** |
| `L1` "shared-memory tile exceeds the device limit" | `observation` — "observed as launch rejected" | **2 injected, 2 detected at compile** |
| `M3.6` leading dim unaligned on a vectorized access | `absent` — v1 realization is a noop on a scalar reference | 3 injected, 0 detected, 3 never — **holds** |

`M1.6`'s "noop by construction" is false for 7 of its 10 realizations (only the
three `sm1`/`sm11` cells are noops as claimed), and `L1` is caught statically by
the same `memcheck.hpp` `Error1` path that reclassified M3.12 — i.e. part of
`L1` is `P2`/repaired, not launch-status. Both are **reported, not applied**:
moving them changes `S1`/`S2`, which §7 reserves for v3 with owner sign-off, and
would invalidate the frozen baseline. This needs a decision (see
`DECISIONS-NEEDED.md`); it is the clearest evidence that a second denominator is
worth having, since neither contradiction is visible from `S1`/`S2` alone.

**3. The `-rtc` enabled curve is derivable from the committed ledger — no rerun
needed — and its shape confirms the thesis while its magnitude does not.** The
E2 ledger carries `cost` (cheapest level at which an obligation is checked), so
`enabled = count(cost ≤ threshold)`:

| `-rtc` | enabled | Δ vs entry | elem | hw | loop | shape |
|---|---:|---:|---:|---:|---:|---:|
| `entry` *(default)* | 675 | — | 58 | 359 | 223 | 35 |
| `low` | 855 | +180 | 238 | 359 | 223 | 35 |
| `medium` | 936 | +261 | 319 | 359 | 223 | 35 |
| `high` | 1151 | +476 | **534** | 359 | 223 | 35 |

The shape is exactly the prediction: **elementwise is the only steep class
(58 → 534, 9.2×), everything else is flat**. The *magnitude* is not: the spec's
§9.2 table (34/47/79/323 = 9.5×) was measured on a different population, so the
ledger is what must be reported, and §9.2's table should be re-labelled as a
prediction.

**4. The cost filter owns 2 of 37 misses, not the dominant share.** Detection at
`entry` is 35/72. The recorded `-rtc=all` arm gives **37/72** — a delta of
**+2**, which matches the ledger's `C1_COST_FILTER_SUPPRESSED = 2` exactly. The
`43/72` figure in `S2`'s flag matrix is a *different* arm (`-rtc=all` **and**
`--disable-assert-hoist`); 6 of its 8 extra detections come from the hoisting
defect. So `mutation-specs-v2.md` §9.2's claim that "the runtime-check cost
filter, not the absence of checks, is the dominant cause of missed value
corruption" is **not supported by the measured data** — the dominant cause is
`C4_NOT_ASSESSED` (19), i.e. *assessment coverage*, which is this workflow's own
subject. §9.2 needs rewording or its population re-specified.

**5. Pre-existing bug, reported not changed:** `S2`'s `n_na` is structurally
always 0. `S1` builds its keys as `f"n_{outcome}"` and `OUTCOMES` contains
`"n/a"`, so the key written is `n_n/a` and `S2`'s `t.get("n_na", 0)` can never
match it. The field is dead. Fixing it changes `S1`/`S2` output, so it is a
coordinator call.

---

## 11 The venue target — and what `make paper` can actually supply

### 11.1 The live directory is `eurosys27`, not `asplos27`

The paper was retargeted from ASPLOS to EuroSys. `svn/eurosys27/` is live;
`svn/asplos27/` is the pre-retarget snapshot and must not be written to.

| | `svn/asplos27/` | `svn/eurosys27/` |
|---|---|---|
| `Makefile` banner | "Build the **ASPLOS** 2027 SVN paper PDF" | "Build the **EuroSys** 2027 SVN paper PDF" + 12-page layout contract |
| `plan/` | review-report, review-state, storyline-state, l3/l4-evaluation, rw-handoff | the mutation design set cited by `mutation-specs-v2.md` §1 |
| `e2e/` | absent | present (`e2e/results/` gitignored) |
| `main.tex` | md5 `7afc011b` | md5 `7afc011b` — **not yet forked** |

`mutation-specs-v2.md` already cites `svn/eurosys27/plan/…` in ten places (and
`mutation-specs.md` in two more), plus `svn/eurosys27/e2e/results/*/` for the
measured numbers, so the *design* side moved before this workflow was written.
`PAPER` in the Makefile is the last piece that had not, and is now repointed.
Both `main.tex` files are still identical, so the split is currently by
*directory*, not yet by content — which is why the retarget is cheap to finish
and expensive to forget.

### 11.2 `make paper` copies unconditionally — and cannot produce everything

`make paper` does `cp render/tables/*.tex $(PAPER)/tables/` and
`cp render/figures/*.pdf $(PAPER)/figures/`. There is no diff, no merge, and no
check that a destination file was hand-edited. So before running it against a
venue directory, know what the paper actually `\input`s and what the renderer
actually emits:

| the paper `\input`s | renderer emits? | note |
|---|---|---|
| `tables/rq2_bugs` | **yes** | but the paper's copy has a **hand-edited caption** ("oracle-confirmed corruptions", "Before-device detection is Compile $+$ Launch") that the generated caption lacks — a re-render silently reverts it |
| `tables/e2_generation` | no | hand-authored qualitative table (`yes`/`part`/`no`). No numeric source in `stats.json` |
| `tables/e3_discharge` | no | hand-authored 2-row static/dynamic **aggregate**; `rq1_category` is a 15-row **per-category** breakdown, so the two are not interchangeable |
| `tables/e4_cost` | no | no renderer source |
| `tables/e5_oracle` | no | hand-authored prose reporting paths; contains no numbers at all |
| `figures/fig_e1_detection` | no | hand-written pgfplots bar chart over **five** toolchains, incl. baselines (MLIR-linalg 34/50, MLIR-low 38/48, IREE 19/23) whose numbers the renderer has no source for |

And the renderer emits three tables the paper never `\input`s — `rq1_category`,
`rq3_runtime`, `rq5_ablation` — plus `rq6_mechanism`, whose content the paper
carries as **prose** ("mechanism attribution (folded RQ6)").

### 11.3 The two rules that follow

1. **Never name a new artifact after an existing hand-written one.** This
   workflow's first S14 figure was briefly called `fig_e1_detection.pdf`, which
   is the *basename* of a different, hand-written figure. Had `make paper` run,
   `\input{figures/fig_e1_detection}` would still have resolved to the `.tex`,
   leaving a misleading `.pdf` beside it — dead, but available to mislead. It is
   now `fig_e1_path_class.pdf`. Check `ls $(PAPER)/tables $(PAPER)/figures`
   before naming anything.
2. **A renderer output is a source file the paper may have edited.** Treat
   `render/` as generated and the venue's `tables/`/`figures/` as
   hand-maintained, and diff before overwriting. Reconciling the `rq*` and `e*`
   sets — which are *not* a pure rename, per §11.2 — is a coordinator call; the
   renderer leaves the `rq*` set alone until it is made.

