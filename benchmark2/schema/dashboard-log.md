# Progress log

Dated entries. Newest first. `gen_dashboard.py` inlines this verbatim into
`DASHBOARD.md`, so an entry here survives every regeneration.

Keep each entry to: what changed, what number moved, what it proved.

---

## 2026-09-19

- **The guard baseline is 49, and four of those findings are real M2/M3 debt.**
  The count moved 44 → 49, which looks like a regression and is not. Commit
  `1161cee` added the `mutants.edits-apply` guard; it immediately found five
  operators whose literal `old` string no longer exists in their base kernel,
  because the base source moved in the corpus turnover. All five target
  `11_dynamic_32xSx768_768x768_32xSx768.co`: `M2.s3.mm11.swap` (spec `M2.3`,
  M2-a), `M2.s2.mm11.tiles` (`M2.2`, M2-a), `M2.13.mm11.affine` (`M2.13`,
  M2-e), `M3.s1.mm11.ktile31` (`M3.1`, M3-a), `M3.s2.mm11.rhs1` (`M3.6`,
  M3-e). **Four are admissible and counted, so the M2 and M3 instance counts
  in this dashboard are optimistic by four** (M2 by three, M3 by one); the
  fifth sits in `M3-e`, which is already `0 of 8` admissible, so it overstates
  nothing. **Proved:** `make guards` at `1161cee~1` → 14 findings, i.e. the
  growth is guard instrumentation, not graded findings; `make test-guards` →
  27/27 controls pass at 49. **The audit did not get worse; the instrument got
  sharper.** Recorded in `DASHBOARD.md`'s handoff, not fixed — M2/M3 are
  deferred. `plan/m1-m4-handoff/HANDOFF.md` §2.5 and §6.6.

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
