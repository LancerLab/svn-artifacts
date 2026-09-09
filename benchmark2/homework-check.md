# benchmark2 — Homework Check (iree + triton + choreo lanes)

> Coordinator review, 2026-09-09 (last updated 2026-09-10). Cross-checks the
> committed `iree/`, `triton/`, and `choreo/` lane outputs against the shared
> contract (`manifest.md`, `specs/mutation-specs.md`, `schema/statistics-manifest.md`,
> `schema/record-schema.json`, and the `experiment-redesign-plan.md` §3.3–3.5 /
> §12 ownership table).
>
> **Verdict: all items resolved.** The `iree`/`triton` lanes closed every item in
> the close-out commit `8ca5568`; the coordinator then repaired a one-line
> markdown regression in `statistics-manifest.md` (`760bfbb`). The `choreo` lane
> landed coordinator rulings **R-D1…R-D7** in `88c574d` and is reviewed below
> (CH1–CH8). Per-item status is below; the original evidence + fix text is
> retained for audit.

## Resolution status (2026-09-10)

| Item | Status | Where |
|---|---|---|
| I1 — IREE S12 surfacing | ✅ resolved | `results/iree/stats.json` → `S12_sanitizer_supplement` |
| I2 — IREE S1 n/a accounting | ✅ resolved | M1/M3 rows → `n_na: 40` |
| I3 — IREE full-size gate failures | ✅ resolved | `full_size_failures[]` disposition |
| I4 — IREE raw/ traceability | ✅ resolved | `iree/README.md` provenance note |
| T1 — Triton S1 M2 row | ✅ resolved | M2 row → `n_na: 40` |
| T2 — Triton e3_remainder.json | ✅ resolved | file committed |
| T3 — Triton expressibility dedup | ✅ resolved | 120 → 60 lines |
| T4 — Triton S8 4-class re-key | ✅ resolved | `{elem, shape, loop, hw}` |
| C1 — stage enum | ✅ resolved | `["compile", "runtime", "none"]` |
| C2 — paper_category enum | ✅ resolved | adds `"hw"` |
| C3 — S8 format pin | ✅ resolved | `statistics-manifest.md` §"S8 exact shape" |
| CH1 — choreo register completeness | ✅ resolved | `results/choreo/*.jsonl` (17,858 records) |
| CH2 — choreo `never ≠ 0` attribution (R-D2) | ✅ resolved | `S2_before_device.no_unattributed_never: true` |
| CH3 — choreo E5b detection asymmetry (R-D1) | ✅ resolved | `S13…E5b.detection_asymmetry` |
| CH4 — choreo toolchain pin (R-D6) | ✅ resolved | `binary_sha1_12 c3ebb1654d1d` + `manifest.md` §5 |
| CH5 — choreo E5a below-floor (R-D3) | ✅ resolved | `S13…E5a.n_delta_within_noise: 14` |
| CH6 — choreo S3/S5/S6/S7/S10 re-registration (R-D4/R-D5) | ✅ resolved | `results/choreo/stats.json` |
| CH7 — choreo corpus integrity (R-D7) | ✅ resolved | reshape/14 safe; conv2d/10 excluded |
| CH8 — choreo raw/ provenance | ✅ resolved | `choreo/README.md` Provenance |

---

## IREE lane

### I1. S12 is still "pending" in `stats.json` — surface it (blocking)

- **Evidence:** `results/iree/stats.json` has **no `S12` key**; its `note` ends
  with `"S12 (compute-sanitizer) pending."` — yet the `s12` command exists
  (`iree/lane.py::cmd_s12`) and `results/iree/sanitizer.jsonl` already contains
  23 records (all `flagged:false, fault:none, exercised:false`).
- **Why this matters:** the measurement was in fact run; the paper table
  (`E1` sanitizer column, S12) reads only `stats.json`, so leaving it "pending"
  silently drops the result.
- **Fix:** add an `S12_sanitizer_supplement` key to `results/iree/stats.json`
  of the form
  `{"M2": {"flagged_and_exercised": 0, "total": 23, "flagged": 0}}`
  plus a note that M2 shape-contract mutants are **not memory faults**, so
  `compute-sanitizer --tool memcheck` is silent by design (plan §3.5). Remove
  the "pending" tail of the `note`.
- **Accept:** `stats.json` carries a concrete S12 value; `note` no longer says
  "pending".

### I2. S1 n/a accounting is self-contradictory (blocking)

- **Evidence:** `S1_detection.M1` and `.M3` both read `n_injected:0, n_na:0`
  while their `note` asserts *"any spec M1/M3 mutant => n/a (R4)."* A class the
  surface cannot express at all must be counted as n/a, not 0
  (`mutation-specs.md` §4: "n/a = surface cannot express the class").
- **Fix:** set `n_na = N` (the spec's 40 per class) for M1 and M3, and keep the
  n/a note. If the lane instead wants to record "not run," delete the n/a claim
  and state "not run" explicitly — the row must not assert n/a and report
  `n_na:0` at the same time.
- **Accept:** no S1 row is simultaneously `n_na:0` and "=> n/a".

### I3. Three full-size kernels fail the gate (blocking — document disposition)

- **Evidence:** `results/iree/kernels.jsonl` has 3 records with
  `run:"crash"`, `ref_check:"fail"`, `run_shape:""`, all `size:"full"`:
  - `conv2d/3_dynamic_16x256xHxW_256x256x3x3_16x256xHxW_S_P_D`
  - `conv2d/4_dynamic_32x128xHxW_256x128x3x3_32x256xHxW_S_P_D`
  - `matmul/6_dynamic_32x197xE_Ex3072_32x197x3072`
  The empty `run_shape` indicates the full-size dynamic-dim run did not
  concretize the `?` dims before launch.
- **Why this matters:** the definition-of-done requires "all 15 categories
  compose + gate"; 3 failures break it.
- **Fix:** determine whether each is (a) a harness bug (fix → pass) or (b) an
  accepted dynamic-dim edge case. For (b) mark it explicitly in
  `kernel_gate`/`note` with the reason, and adjust the `totals` so `ref_pass`
  isn't silently 619/622.
- **Accept:** every non-passing full-size kernel has a documented, auditable
  disposition in `stats.json`.

### I4. `raw/` is gitignored — confirm traceability (non-blocking)

- **Evidence:** `iree/.gitignore` excludes `raw/`, so the compile/run logs that
  back the 3 failing kernels (I3) and the `sanitizer.jsonl` provenance are not
  committed.
- **Fix:** state in `results/iree/` (or `iree/README.md`) how a reviewer
  re-derives `kernels.jsonl`/`sanitizer.jsonl` from `run.sh` subcommands, so the
  "every number traces to raw JSON" rule (manifest §8) stays verifiable.
- **Accept:** a one-paragraph provenance note is present.

---

## Triton lane

### T1. S1 matrix is missing the M2 row (blocking)

- **Evidence:** `results/triton/stats.json` → `S1_detection_matrix` has keys
  `["M1", "M3"]` only. S1 is defined "per class × toolchain" (3 classes).
- **Fix:** add
  `"M2": {"n_injected": 0, "n_compile": 0, "n_runtime": 0, "n_never": 0,
  "n_na": 40, "note": "no cross-tensor contract (§3.3)"}`.
- **Accept:** S1 carries all three classes; M2 is an explicit n/a row.

### T2. `results/e3_remainder.json` is referenced but not committed (blocking)

- **Evidence:** `run.sh::cmd_e3` writes
  `$HERE/results/e3_remainder.json`; `results/README.md` references it; the file
  is absent from the tree.
- **Fix:** run `./run.sh e3` (it emits the file) and commit
  `results/e3_remainder.json` (a one-line
  `{"toolchain":"triton","remainder":"n/a","note":"no generated checks"}`), or
  remove the dangling references if `stats.json`'s S9 is intended to be
  authoritative.
- **Accept:** every file referenced by `run.sh`/`README.md` exists.

### T3. `raw/expressibility.jsonl` is duplicated (non-blocking)

- **Evidence:** `raw/expressibility.jsonl` is **120 lines** = 60 records × 2
  (each category×class pair appears twice). 15 categories × 4 classes = 60.
- **Fix:** de-duplicate to 60 unique lines; confirm `results/expressibility.json`
  still holds exactly 60 objects.
- **Accept:** `wc -l raw/expressibility.jsonl` = 60; parsed array length 60.

### T4. S8 is keyed by M1/M2/M3 instead of the 4-class schema (blocking)

- **Evidence:** `results/triton/stats.json` → `S8_expressibility` uses
  `M1/M2/M3` prose, and has **no `loop` row** — but
  `raw/expressibility.jsonl` records `class ∈ {elem, shape, loop, hw}`
  (`record-schema.json` obligation/expressibility class enum), with `loop=yes`
  for all 15 categories. IREE's S8 already uses the 4-class form.
- **Fix:** re-key `S8_expressibility` to
  `{"elem": {"yes": 15}, "shape": {"no": 15}, "loop": {"yes": 15},
  "hw": {"partial": 15}}` (counts from `raw/expressibility.jsonl`), matching the
  schema and IREE's shape.
- **Accept:** Triton and IREE S8 use the same 4-class keys; `loop` is present.

---

## Coordinator (schema — affects both lanes)

### C1. `stage` enum lacks a not-detected value

- **Evidence:** Triton `results/mutants.json` emits `stage:"none"` (25 records),
  `stage:"runtime"` (2), `stage:"compile"` (2). `record-schema.json` `stage`
  enum is `["compile", "runtime"]` — `"none"` does not validate.
- **Fix:** add a not-detected value to the `stage` enum (e.g. `"none"` or
  `"not-detected"`), then re-validate both lanes' JSON.

### C2. `paper_category` enum lacks `hw`

- **Evidence:** Triton M3 mutants emit `paper_category:"hw"` (2 records);
  `record-schema.json` `paper_category` enum is
  `["dim-mismatch", "oob", "wrong-shape", "stride"]`.
- **Fix:** add `"hw"` to the enum (M3 = hardware-constraint), then re-validate.

### C3. Pin the S8 `stats.json` format

- **Evidence:** the two lanes emit structurally different S8 (`stats.json` prose
  vs 4-class counts), which blocks mechanical table generation.
- **Fix:** state in `schema/statistics-manifest.md` the exact S8 shape
  (4-class × `{yes, partial, no}` counts) and require both lanes to conform.

---

## choreo lane

> The choreo lane (`benchmark2/choreo/`, `benchmark2/results/choreo/`) owns
> **S1–S7, S10, S13** and closed its work in `88c574d` ("implement coordinator
> rulings R-D1…R-D7"). Those rulings are the coordinator's resolution of
> `choreo/DECISIONS-NEEDED.md` D1–D7; each is re-checked below against the
> committed register. Register counts — `cost.jsonl` (15), `kernel.jsonl` (310),
> `latency.jsonl` (32), `mutant.jsonl` (120), `obligation.jsonl` (17,353),
> `residue.jsonl` (28) — total **17,858 records**, matching
> `DECISIONS-NEEDED.md`'s "collect PASS (17,858 records, 0 schema violations)".

### CH1. Register is committed, schema-valid, and self-contained (blocking)

- **Evidence:** all six `.jsonl` register files are committed; their counts match
  the `inputs` block in `stats.json` (`cost 15, kernel 310, mutant 120,
  obligation 17353, latency 32, residue 28`). `stats.py` reads **only** the
  register (per `choreo/README.md` "Data flow"), so every `stats.json` number
  traces to a committed record.
- **Accept:** register committed and counts reconcile to `stats.json.inputs`; no
  `stats.json` value is sourced from gitignored `raw/`.

### CH2. `never ≠ 0` is retired; every `never` is attributed (R-D2) (blocking)

- **Evidence:** `S2_before_device` reports `n_injected 72`, `n_before_device 35`,
  `n_never 37`, `pct_before_device 48.6111`, with `no_unattributed_never: true`
  and cause counts `{C4_NOT_ASSESSED 19, HOISTING_DEFECT 6, C5_OUT_OF_SCOPE 6,
  C3_LOWER_BOUND_OMITTED 4, C1_COST_FILTER_SUPPRESSED 2}`. The flag-matrix
  ablation records `pinned 48.61%` vs `rtc_all+hoist-disabled 59.72%` as a
  sensitivity result, not a re-pin.
- **Why this matters:** the Phase-0 gate `never = 0` is retired (R-D2); the new
  criterion is "no *unattributed* never", and §5.2 ¶2 is reframed around 35/72
  before-device, not the dropped "210/210" headline.
- **Accept:** `no_unattributed_never: true`; the 5 cause buckets sum to 37; the
  48.6% pinned default is the headline (59.7% is ablation-only).

### CH3. E5b is detection asymmetry, not a latency multiplier (R-D1) (blocking)

- **Evidence:** `S13_runtime_and_latency.E5b` emits `detection_asymmetry`
  (`n_oracle_blind 4`, `n_oracle_never_executed 2`, `n_oracle_aborted_by_choreo 6`)
  and a guarded `ratio` of `n_pairs 4` (`median 84.45×`), with `n_confounded 12`
  and `unguarded_all_pairs` retained only under a `NOT QUOTABLE` marker. The
  `premise_violation_note` documents that `--disable-runtime-check` gates only
  assertion source A (sources B/C are ungated by construction).
- **Accept:** no `64×`/`472×` headline; `detection_asymmetry` is the publishable
  claim; the pre-guard ratio is explicitly non-quotable.

### CH4. Toolchain identity pinned; stale binary disclosed (R-D6) (blocking)

- **Evidence:** `stats.json.toolchain_identity` = `binary_sha1_12 c3ebb1654d1d`,
  source `f2f238f`, `checkout_head 1fa4719`, `binary_stale_vs_checkout: true`;
  the `toolchain_warning` surfaces the discrepancy. `manifest.md` §5 carries the
  coordinator pin ("data collected on `f2f238f`; `1fa4719` postdates the data").
- **Accept:** `binary_sha1_12` present and cited in `manifest.md` §5; the
  stale-binary flag is disclosed, not treated as a defect.

### CH5. E5a residue is "below the measurement floor", static premise dropped (R-D3) (blocking)

- **Evidence:** `S13.E5a` reports `n_cases 14`, `median_delta_pct 3.9162`,
  `n_delta_within_noise 14`, with every case `device_identical: true` and
  `device_check_calls: {choreo_assert: 0, runtime_check: 0}`; the `feeds` string
  reads "structurally zero on device, below measurement floor on host".
- **Accept:** no percentage residue headline; the static-shape "zero residue by
  construction" premise is dropped (0 static kernels were measured).

### CH6. S10/S7/S3/S5/S6 re-registered from the committed register (R-D4, R-D5) (blocking)

- **Evidence (all from `stats.json`):** S10 `grand_median_pct 0.1296`
  (front-end-vs-nvcc, `reproduces_rq4: false`); S7 `status: not_measurable`
  (interval counterfactual inherited, not re-derived); S3 `grand_total 17353`
  (`delta −364` vs 17,717); S5 `all 93.04%` (16,145/17,353), `static 100.0%`,
  `dynamic 86.87%`, `remainder 1208`; S6 `interval 2920`.
- **Accept:** no stale `0.6%`/`17,717`/`93.2%`/`1,199`/`2,837` figures remain
  quotable; S7 is labeled `not_measurable`, not "77.2%".

### CH7. Corpus integrity: reshape/14 safe, conv2d/10 excluded, 36 rejections disclosed (R-D7) (blocking)

- **Evidence:** `reshape/14` fix approved (0 obligations, corpus-neutral);
  `conv2d/10_dynamic` excluded with a note (288.0 KB shared `A` tile vs ~228 KB
  hardware max); the 36 silent launch-rejections are folded into the R-D6
  "postdates the data" limitation (pre-fix binary + `--max-local-mem-capacity`).
- **Accept:** the 2 failing kernels have documented dispositions; the 36 silent
  rejections are disclosed as a pre-fix artifact, not silently dropped.

### CH8. `raw/` provenance note present (non-blocking, I4 precedent)

- **Evidence:** `choreo/README.md` "Provenance" lists the committed raw JSONs and
  a regeneration table for the gitignored logs (mirroring the `iree` lane's I4).
- **Accept:** a reviewer can re-derive the register from `run.sh` subcommands.

---

## Out of scope of this check

- **mlir-linalg / mlir-low** lanes: only the shared harness (`mlir-shared/`) has
  landed; their **S1/S8/S9/S12 result outputs are still pending** (no
  `results/mlir-*` yet). A follow-up check is required when they push (see the
  "On the MLIR lanes" note in `choreo/DECISIONS-NEEDED.md`).
- **tilelang** is exploratory.

## Definition of done

✅ **iree + triton (2026-09-10).** All blocking items (I1, I2, I3, T1, T2, T4,
C1, C2, C3) closed in `8ca5568`; non-blocking items (I4, T3) closed in the same
commit; every number in `stats.json` traces to committed raw JSON; and `run.sh
all` completes with schema-valid output. The only follow-up was the `760bfbb`
markdown-row repair (doc-only).

✅ **choreo (2026-09-10).** All seven coordinator rulings (R-D1…R-D7) landed in
`88c574d`; CH1–CH8 confirm the register is committed and schema-valid,
`never ≠ 0` is retired with full per-cause attribution, E5b is reframed as
detection asymmetry (multiplier dropped), the toolchain is pinned to
`binary_sha1_12 c3ebb1654d1d` (`f2f238f`), and S3/S5/S6/S7/S10 are re-registered
from the committed register.

**Remaining gap:** the `mlir-linalg`/`mlir-low` **result outputs** are still
pending (only `mlir-shared/` has landed); they get their own homework check when
they push.
