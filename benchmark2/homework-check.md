# benchmark2 — Homework Check (iree + triton lanes)

> Coordinator review, 2026-09-09. Cross-checks the committed `iree/` and
> `triton/` lane outputs against the shared contract (`manifest.md`,
> `specs/mutation-specs.md`, `schema/statistics-manifest.md`,
> `schema/record-schema.json`, and the `experiment-redesign-plan.md` §3.3–3.5 /
> §12 ownership table).
>
> **Verdict: measurement work is substantially complete, but the lanes have not
> "done everything."** The items below are the blocking and non-blocking gaps.
> Each item states the evidence, the required fix, and the acceptance check.
> Close every `[ ]` before this lane is considered done.

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

## Out of scope of this check

- **choreo lane** (`benchmark2/choreo/`, `benchmark2/results/choreo/`) is still
  untracked local work and is not covered here; it owns S1–S7/S10/S13 and has
  its own acceptance criterion (`never = 0` per the 2026-09-09 binding addendum,
  item 5).
- **mlir-linalg / mlir-low** lanes do not exist yet (binding addendum item 5);
  **tilelang** is exploratory.

## Definition of done

All blocking items (I1, I2, I3, T1, T2, T4, C1, C2, C3) closed; non-blocking
items (I4, T3) either closed or explicitly waived by the coordinator; every
number in `stats.json` traces to committed raw JSON; and `run.sh all` completes
with schema-valid output.
