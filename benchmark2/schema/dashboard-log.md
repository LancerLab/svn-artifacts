# Progress log

Dated entries. Newest first. `gen_dashboard.py` inlines this verbatim into
`DASHBOARD.md`, so an entry here survives every regeneration.

Keep each entry to: what changed, what number moved, what it proved.

---

## 2026-09-19 (rev. 4) — M3's cell is not a design call; the floor settles it

Was recorded as a decision for the owner: *"widen `MINIMAL_SET["M3"]`, or restate
the cell and record the deviation."* The measurements below remove the choice —
one branch is forced and the other does not exist.

- **The floor rejects every restatement, on measured data.** From
  `choreo/raw/mutant_manifest.json`, the admissible rate per class is M1 34/34 =
  100%, M2 26/26 = 100%, **M3 10/14 = 71.43%**, M4 21/29 = 72.41% (M3 counts
  family-assigned rows; over all 18 rows it is 55.56%). `floor_rationale` reasons
  from an *assumed* 45% attrition, but M3's is **better** than that — 28.6% — and
  it still fails: `admissible_floor_per_class = 35` needs a cell of
  `ceil(35 / 0.7143) = **49**`. Cell 64 gives 45.7 (**clears**), cell 48 gives
  **34.3 — short by 0.7**, cell 32 gives 22.9 — short by 12.1. The declared 64 is
  the only reachable value that clears the floor, so "restate as 48 or 32" is not
  a cheaper honest option, it is a floor failure. (M4 survives its smaller cell of
  56 at 40.6 because its rate is higher.)
- **Restating does not clear the guard either — verified, not inferred.**
  `m3.cell` checks `len(declared)` and `len(realised)` against
  `budget.kernel_component = 4` and derives the cell from `families x
  N_PER_FAMILY`; **it never reads `budget.classes.M3.cell`**. Setting that field
  to 48 and re-running `check_class_axis.py --guard m3.cell` reproduces the
  identical two findings. The guard's *"or record the class cell that the corpus
  can actually carry"* is permission to be honest in prose — the standing FAIL
  **is** the record.
- **So the remedy is a work item, and its size is known.** `declared` = 3
  (`matmul`, `conv2d`, `batch_norm`) and `realised` = 2 (`matmul`, `conv2d`),
  so **both halves must move** and the costly one is the second: a category added
  to `MINIMAL_SET` with no operator behind it changes the declaration, not the
  corpus. The suite realises **seven** kernels in total — `concat`, `conv2d`,
  `layer_normalization`, `matmul`, `relu`, `softmax`, `transpose` — and **M3's
  already-declared third, `batch_norm`, has no case anywhere** under
  `choreo/mutants/` (`max_pool2d` and `embedding`, which `M1`'s Level-2 order
  names, are equally absent). The `MINIMAL_SET` edit is the *last* step, not the
  first. Needs the DSL toolchain, so it is not CPU work.
- **`m3.cell` is the only red cell guard** — `m1.cell`, `m2.cell` and `m4.cell`
  all pass. M4 realises 5 categories (`conv2d`, `layer_normalization`, `relu`,
  `softmax`, `transpose`), so the same disease was suspected there and does not
  exist. Guards unchanged at **44**.

Docs corrected to match: `m3.md` §1.2 (new) and §4 (rewritten — "The design call"
becomes "and why it is not a choice"), §2 and §6, plus `HANDOFF.md` §3.1 and
`README.md`.

---

## 2026-09-19 (rev. 3) — the corpus is provably saturated, and two plans were wrong

- **`fill_plan` sorted the wrong way.** `sorted(order, key=lambda k:
  cur.get(k, 0))` put a kernel the family does not use (0) **ahead of** one
  sitting at 1 of 2, so it opened a new kernel while a thin one was still open —
  the exact opposite of the function's own docstring, and the reason `M2-a` was
  told to add `conv2d` when it has no `conv2d` operator at all. The key is now
  `(is_new, count)`. **14 plans changed**, all toward a flatter spread;
  `M2-a` is now `relu+1`, which closes it to a perfect `2,2,2,2`.
- **`worklist.csv` gained a `ceiling` column — the number that says whether a
  plan is even reachable.** `short` is `N - have` (the obligation). `ceiling` is
  what the candidate table can supply. Three caps stack under `N`: 2 per
  category; the CELL ceiling (one instance per `(spec_id, category)`, or two
  when the spec is P1); and the candidate count — a cell with one incumbent
  supplies one, not two. `family_ceiling()` computes exactly that.
- **It reproduces the measured corpus for all 31 families — every choreo row
  reads `have == ceiling`.** M1 34/34, M2 26/26, M3 14/14, M4 29/29. That is the
  mechanical answer to "can we not just re-run it to get more": where
  `ceiling == have` there is **no candidate left to take**, so CPU time cannot
  change the corpus and only a new operator can. It also settles the open
  `M2 = 34` vs `26` puzzle: `34` came from `Σ_categories min(2, candidates)`,
  which ignores the per-cell ceiling. `26` is the correct value.
- **Two documentation errors corrected in the handoff**, both caused by writing
  prose from `fill_plan` instead of from the candidate table: `M2-a` was
  described as `7 → 8` by a re-run (it is 7 today, and 7 is its ceiling), and
  M3's "reachable after the plans land" was given as `32` when only 5 of its 8
  families hold operators, so `5 × 4 = 20` is the fillable figure and the
  assignable distance is **6**, not 18. `m2.md` §2.1's "to reach 8, add" column
  was re-derived per family from measured cells.
- `make guards` unchanged at **44** — no guard consumes `worklist.csv`, so a
  more honest column moves no verdict.
- **Lesson.** A `fill_plan` derived from the class coverage set names kernels
  the family does not have. Verify a generated plan against the family's own
  candidate table before publishing it, and never print a plan without the
  ceiling beside it.

---

## 2026-09-19 (rev. 2) — the worklist now covers M1–M4, and M3 has a ceiling

- **`make worklist`: 75 → 155 rows. `in-scope shortfall` 97 → 233.** The
  generator's `CLASSES` was `["M1", "M4"]`, matching a handoff that shipped
  `m1.md`/`m4.md` and called M2/M3 "deferred". Extending it to all four classes
  found that **half the taxonomy was outside the worklist**, and that one of the
  two missing classes cannot be worked at all.
- **The finding: a class shortfall has two ceilings, and only one is authorable.**
  The *candidate* ceiling is how many `(spec_id, category)` operators exist.
  The *coverage-set* ceiling is `len(MINIMAL_SET[cls]) x N_REALISATIONS` —
  `select()` caps every `(family, category)` at `N_REALISATIONS`, so a class
  offering fewer than `N_KERNELS` kernels cannot reach `N` however many
  operators are written.
  - **`M2`: authorable.** `MINIMAL_SET["M2"]` holds 6 kernels where `N` needs 4,
    and `m2.cell` is `ok`. But `have 26 = reachable 26`: the class is saturated
    at its *existing operator supply*, so a CPU re-run emits the same 26 and the
    38-instance gap is closed by **writing operators**. "Burn CPU for M2" is a
    no-op.
  - **`M3`: not authorable.** `MINIMAL_SET["M3"] = [matmul, conv2d]` — 2 kernels
    where `N` needs 4. Per-family ceiling **4**, class ceiling **32**. `have 14 /
    reachable 14 / 32`. **Three numbers must not be conflated: declared 64,
    declared-coverage 48, realisable 32.** And 32 is itself below
    `admissible_floor_per_class = 35`, so M3 fails the floor by construction —
    `floor_rationale` rules out lowering `N` for exactly this reason.
- **`gen_worklist.py` changes:** (a) `CLASSES` covers all four; (b) new
  `kernel_ceiling(cls)`; (c) `blocker()` accumulates reasons instead of
  early-returning, and states the ceiling verbatim with *"Do NOT report 8"*, so a
  `short` of 50 is never read as 50 assignable instances. **No guard consumes
  `worklist.csv`** — this changes the report, not the audit.
- **Numbers that moved:** rows 75 → 155; total shortfall 97 → 233 (of M3's 50,
  **18** reachable). **Numbers that did not:** `make guards` is still **44**,
  group-level FAILs still **32**, corpus still 107 mutants, and every family
  depth is unchanged (`M2-a 7`, `M2-e 2`, `M3-a 4`, `M3-e 4`). A wider report is
  not a worse audit.
- **Also established, and worth keeping:** `m2.cell` `ok`, `m3.cell` FAIL,
  `m1.cell`/`m4.cell` `ok`; all 8 `m2.*` and all 8 `m3.*` instance guards are
  red, as are 6 of 7 `m4.*`. And **only `choreo` emits
  `raw/mutant_manifest.json`**, so every other lane's family depth is reported by
  `worklist.csv` and enforced by nothing.
- **Still open:** the `kind: "plan"` detail (below), and M3's cell — a design
  call, not a work item.

## 2026-09-19 (final)

- **`corpus.declared-files`: 46 → 44. The v2.1 declaration is now a measurement,
  not a label.** `run_e1.py`'s `V21_FIELDS` stopped at `spec_version` and never
  minted `applicable` — the fifth field of the *record* vocabulary
  (`records.versioned_fields("mutant")`), and the one field the manifest (a
  PLAN) cannot supply, because a plan carries the in-process `admissible`
  instead. So `996087b`'s regenerated corpus held four of five fields while
  stamping `spec_version: v2.1`: a partial port wearing a full-port label, which
  is exactly what the gate refuses. `stamp()` now mints `applicable` from the
  per-mutant `admissible`, else the spec's `spec_admissible` — the rule
  `collect.py:296` already used — and `python3 choreo/run_e1.py --stamp-only`
  backfilled all 107 rows. CPU only; that mode re-derives nothing.
  **Numbers that moved:** findings 46 → 44, `corpus.declared-files` details
  `3 → 1`. Group-level FAILs stay **32** — the group lost two details and kept a
  third, so a group delta need not show up in the group *count* either.
  **Proved:** reading the file rather than the label — 107/107 rows carry
  `applicable`, it agrees with the fallback rule on all 107, and **every
  non-v2.1 field is identical to a pre-edit copy** (zero verdict drift), so no
  M2/M3 number can have moved. Also proved that relabelling alone cannot work: a
  stamped partial port fails *whichever* release it declares.
  **Also fixed:** the axis note claimed the manifest carries its four plan fields
  on "all 170 mutants" — the measured count is **107**. A wrong literal, in the
  note that documents the measurement.
  **Still open, pre-existing, owner's call:** the gate cannot read
  `choreo/raw/mutant_manifest.json`, because `g8_corpus_release` only understands
  `kind: "records"` while the manifest is honestly declared `kind: "plan"`. That
  is the one remaining detail in the group. Teaching the gate the `plan` shape,
  or dropping the plan from `corpus_paths`, would take the baseline to 43.

## 2026-09-19 (later)

- **`996087b` closed five `mutants.edits-apply` findings and opened two
  `corpus.declared-files` ones. Baseline 49 → 46 — the number is the sum of
  both.** The repair is real: all five operators now apply (re-verified by hand
  against their base kernel), and **no measured number moved** — the corpus is
  still 107 mutants, and M2/M3's family depths are unchanged (`M2-a 7`, `M2-e 2`,
  `M3-a 4`, `M3-e 4`). But `raw/e1_mutant_records.json` was regenerated and now
  emits **v2.1** records (`path_class`, `prohibition`, `spec_id`, `spec_version`)
  while the lane still declares that file at **release v1**, and the axis records
  none of those fields. So `corpus.declared-files` went `1 → 3`. **Arithmetic:
  49 − 5 + 2 = 46**; group-level FAILs `33 → 32`. **Proved:** per-group parse of
  `make guards` before and after — the *only* changed groups are
  `mutants.edits-apply` (`FAIL 5` → `ok 0`) and `corpus.declared-files` (`FAIL 1`
  → `FAIL 3`). A count that moved by three is not "five fixed"; read the groups.
  **Owner action:** either bump the declared release to `v2.1` and record the
  fields in the axis, or stop declaring `e1_mutant_records.json` until ported.

- **Correction to the entry below.** It claimed the five non-applying operators
  make the M2/M3 instance counts "optimistic by four". **That was wrong**, and
  was checked the careless way — from the registry rather than from the corpus.
  See the amended note in that entry.

## 2026-09-19

- **The guard baseline moved 44 → 49, and the extra findings were M2/M3 registry
  debt — not a worse audit.**
  The count moved 44 → 49, which looks like a regression and is not. Commit
  `1161cee` added the `mutants.edits-apply` guard; it immediately found five
  operators whose literal `old` string no longer exists in their base kernel,
  because the base source moved in the corpus turnover. All five target
  `11_dynamic_32xSx768_768x768_32xSx768.co`: `M2.s3.mm11.swap` (spec `M2.3`,
  M2-a), `M2.s2.mm11.tiles` (`M2.2`, M2-a), `M2.13.mm11.affine` (`M2.13`,
  M2-e), `M3.s1.mm11.ktile31` (`M3.1`, M3-a), `M3.s2.mm11.rhs1` (`M3.6`,
  M3-e). **They inflated the registry, not the measurements** — corrected above:
  none was in the committed corpus, and each has a working sibling realising the
  same spec, so no instance count and no `R_f` moved. **Proved:** `make guards` at `1161cee~1` → 14 findings, i.e. the
  growth is guard instrumentation, not graded findings; `make test-guards` →
  27/27 controls pass at 49. **The audit did not get worse; the instrument got
  sharper.** Recorded in `DASHBOARD.md`'s handoff — M2/M3 are deferred.
  `plan/m1-m4-handoff/HANDOFF.md` §2.5 and §6.6.

- **A missing lane input no longer renders as a measured zero.** No number
  moved on a checkout that has the data. Cloning `gxf/croq-paper-plan` +
  `LancerLab/svn-artifacts` and running `make dashboard` produced
  `iree | 88 | 0 | 0 | **88 to go**` and `M2 iree 0/32 ❌` — numbers that were
  **not measured anywhere**, on the one artifact a colleague is told to read.
  Cause: `benchmark2/iree/.gitignore` ignores `raw/`, so
  `benchmark2/iree/raw/mutants.jsonl` is untracked while triton/mlir-low/
  mlir-linalg commit theirs; `load_jsonl` returns `[]` for an absent file, which
  is indistinguishable from a lane that measured nothing. The generator now
  resolves each lane's *own* source (`SOURCES` — `choreo` legitimately has no
  `raw/mutants.jsonl`, it reads `mutant_manifest.json`, so a shared filename
  pattern would have wrongly blanked choreo's real 107) and, when a source is
  absent, prints `n/a` + `no data in this checkout` instead of `0`, withholds
  the `❌` that implies a measured shortfall, adds a **"`n/a` is not zero"**
  note naming the ignored path, and makes the M4 prose claim about `iree`
  conditional. **Proved:** same generator, two checkouts — data present: 128
  lines, `iree 23`, `M2 iree 23/32`; data absent: 130 lines, `iree n/a`.
  Guard-neutral (49 findings either way; 27/27 controls pass).

- **Both generators' fallback output path is derived, not hardcoded.** No number
  moved. `gen_worklist.py` and `gen_dashboard.py` each carried an absolute
  `/home/garfee/...` default, so the M1/M4 handoff's *documented* command
  (`make worklist`) worked on exactly one machine. Now derived as
  `benchmark2/schema -> ../../.. /eurosys27`, mirroring the Makefile's
  `PAPER ?=`. The existence check runs only when the default is used, so
  importing either module is still side-effect-free (verified) and an explicit
  path argument still overrides everything. If the paper repo is absent the
  failure names the missing directory and the override, instead of creating a
  stray tree.

- **Output paths repointed at the live venue.** The vscode workspace root is
  `svn/asplos27`, so both generators defaulted into it — but `asplos27` is the
  **pre-retarget snapshot**: the paper is EuroSys now, every recent commit
  touches `eurosys27/` only, and `plan-a-execution.md` lives in
  `eurosys27/plan/`. The handoff had been written into the dead directory.
  Moved to `eurosys27/plan/m1-m4-handoff/`; `make dashboard` and `make worklist`
  now both default under the existing `PAPER ?=` knob
  (`OUT`/`OUT_WORKLIST ?= $(PAPER)/...`), so the venue is named in exactly one
  place. `asplos27/plan/README.md` is a signpost back. **Proved:**
  `make dashboard` → `eurosys27/DASHBOARD.md`, `make worklist` →
  `eurosys27/plan/m1-m4-handoff/worklist.csv` (75 rows), 97 in-scope shortfall.
  Nothing writes to the snapshot.

- **Added `make worklist` + `schema/gen_worklist.py`; issued the M1/M4 handoff.**
  `plan/m1-m4-handoff/{README,HANDOFF,m1,m4}.md`, `lanes/*.md`, `worklist.csv`
  (75 rows). Scope is M1+M4 only; M2/M3 deferred by instruction. The worklist is
  generated, not written: it reads the manifest, the lane `mutants.jsonl`, the
  taxonomy and the class axis, and refuses to invent a `fill_plan` for a family
  it cannot attribute.
- **The worklist immediately found two things the guard suite cannot see.**
  (a) Every row for `triton` and `iree` carries **no `spec_id`**, so no instance
  can be attributed to a family — 8 M1 families recorded as
  `unattributed(no spec_id)` rather than as a shortfall. The instance guards
  read only `{lane}/raw/mutant_manifest.json`, which **only `choreo` has**, so
  this has been invisible since the lanes were added. (b) `mlir-low`'s M4 is
  **not empty**: attributing its 48 rows by `spec_id -> family -> class` gives
  `M1 40 + M4 8`, the 8 arriving via `M1.6`, which the registry re-homed out of
  M1 into `M4-d`. Its census header says `class: M1` and counts 48; **a lane is
  not its census label.**
- **Assignable work is 97, not 377.** The first run reported `TOTAL=377` because
  it counted every non-choreo lane's M4 as a gap. But `class-axis.json` holds M4
  at **`uncompared`** in four lanes — *"the lane emits NO cell for it"* — while
  `method_taxonomy.json::in_scope_lanes[M4]` lists all five and `targets` bakes
  in 224 for the other four. The generator now distinguishes
  `UNCOMPARED-conflict` from `in-scope`, which drops the headline to
  **97** (choreo M1 30 + choreo M4 27 + mlir-low M1 40) and exposes the M4
  conflict as decision 1 of the handoff instead of silently assigning 280
  instances of phantom work to four lanes.
- **No guard reconciles the class axis with the taxonomy.** `check_class_axis.py`
  consumes `in_scope()` only in `family_width()` and never reads `lane_status`;
  `uncompared` is consumed elsewhere. The tree is internally consistent only
  because the guard set is blind at this seam. Flagged in the handoff, not
  fixed — it needs a decision, not an edit.
- **Corrected the dashboard's composition model. There is ONE model, not two.**
  I had been dividing the mlir lanes by `n_target_per_class = 40`. That is
  wrong, and it made `mlir-low` read `48/40 ✅` (i.e. "done") when it is
  `48/64` — still short of M1's cell. The mlir lanes are family-model lanes
  like everything else: `mlir-low` covers M1+M4 = 8+7 = 15 families × 8 = **120**,
  exactly its declared target. §3 now shows `mlir-low` M1 **48/64**,
  mlir-linalg M2 **50/64**, and both **0/56 ❌** on M4.
- **Derived the `targets` model from first principles and reconciled it.**
  `planned(lane, class) = 8 × (families of that class in scope for that lane)`.
  Class cells are the declared cell *except* for two things: M4's 7 families
  (so 56, not 64, in every lane) and each lane's `lane_scope` carve-outs, each
  worth exactly 8. Verified: choreo 64+64+64+56 = 248 · triton 64+48+56 = 168 ·
  mlir-low 64+56 = 120 · mlir-linalg 64+56 = 120 · iree 32+56 = 88 → **744**.
  The generator now recomputes this every run and shouts if it stops holding.
- **Located the real conflict.** `class-axis.json::n_target_per_class = 40` is
  the mlir collectors' own internal per-class gate, not a budget. It is the
  pre-existing `lanes.stats-conformance` FAIL: 40/class × 2 in-scope classes =
  80, against a declared 120. Two instruments, two numbers, never reconciled.
  Not fixed — needs a decision, not an edit.

## 2026-09-18

- **Added `M3.14.cv1.linearcopy`** to `choreo/mutations.py` — a second
  realisation of the absent `CheckDimSize` cell (defect F1) on conv2d.
  M3-b family depth **2 → 3**. Corpus regenerated, **106 → 107** mutants.
- **Proved the corpus turnover is not mine.** Diffing manifest id sets against
  `HEAD`: my change introduced exactly one new id; the other 79 deselections /
  15 new ids are the pre-existing uncommitted W2 per-family re-selection.
- **Measured M3's real ceiling: 14 of 64.** Six of eight families are already at
  their arithmetic ceiling. `MINIMAL_SET["M3"] == ['matmul','conv2d']` is a hard
  cap, and `compose()` raises on any M3 operator outside it.
- **Found M3.14 is mis-attributed.** Both its operators mutate a `chunkat`
  factor, which makes the copy `IsSlice` — and that cell *does* run
  `CheckDimSize`. The `// omitted` cell needs an untiled copy. `M3.6` sits in the
  same wrong cell and is already recorded inadmissible. Identified 53 untested
  `dma.copy w.span_as(Cout, K) => shared` sites that reach the genuinely silent
  cell — an honest F1 route.
- `make test-guards` → **27 controls pass**, all 61 guard ids controlled or
  excused. `make guards` → **44 drift findings** (unchanged).
- Added this dashboard.

## 2026-09-17

- Guard programme: `families.select-conformance`, `m<C>.<family>.instances`,
  `method_taxonomy.check()` wired into `CHECKS`; every new guard
  negative-controlled.
- Established that `MINIMAL_SET["M3"]` names only 2 categories, so M3's cell is
  64 declared but far fewer reachable.

## 2026-09-19

- Repaired `mutants.edits-apply` after the benchmark-case migration for the
  shared-memory compiler fix (croqtile PR #23): 5 operators on matmul case 11
  (`M2.s2.mm11.tiles`, `M2.s3.mm11.swap`, `M2.13.mm11.affine`,
  `M3.s1.mm11.ktile31`, `M3.s2.mm11.rhs1`) had literal `old` strings keyed to
  the pre-migration source (`=> local`, `[1, 32, 32]` tiling). Edit strings
  re-sited onto the migrated source (`=> shared`, `[32, 32, 32]`), mutation
  semantics unchanged.
- Corpus 107 -> 107 mutants (regenerated, not re-collected).
- `make guards`: 47 findings (was 50 on arrival; handoff baseline was 44 —
  the +6 predate this sitting and are upstream churn since the handoff).
  New from this sitting: none.
- `make test-guards`: 27 controls pass. GREEN.
- Not from this sitting, flagged for the coordinator: upstream baseline drift
  44 -> 47 between handoff issuance and this sitting.
