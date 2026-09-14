# Prose drift report

Generated 2026-09-11T19:22:58 by `benchmark2/render.py`.

Paper under comparison: `/home/garfee/croq-paper-plan/svn/svn-artifacts/benchmark2/../../eurosys27`.

The integrator's contract is that no number enters the paper that no worker's `stats.json` produces. This report audits the other direction: every number the paper *already* carries against the statistic that would have to produce it. Nothing here is a correction to be applied automatically -- a `DIFFERS` or `NO SOURCE` row is a decision for the owner, and overwriting prose with a regenerated value without that decision is exactly the failure this file exists to prevent.

## 1. Regenerated float vs the venue's current float

60 cell(s) agree, 0 differ, were added or were removed. Rows are keyed by `block|label`, so block 1 of a regenerated float is compared with block 1 of the hand-authored one.

| Float | Row | Paper | Regenerated | Verdict |
|---|---|---|---|---|
| tab:rq2-bugs | `0|llrrrrr@{}} Class / Toolchain` | Toolchain | Injected | Compile | Launch | Never | n/a | Toolchain | Injected | Compile | Launch | Never | n/a | **AGREES** |
| tab:rq2-bugs | `1|M1 / IREE` | IREE | 0 | 0 | 0 | 0 | 40 | IREE | 0 | 0 | 0 | 0 | 40 | **AGREES** |
| tab:rq2-bugs | `1|M1 / MLIR-low` | MLIR-low | 48 | 0 | 38 | 10 | 0 | MLIR-low | 48 | 0 | 38 | 10 | 0 | **AGREES** |
| tab:rq2-bugs | `1|M1 / Triton` | Triton | 27 | 0 | 2 | 25 | 0 | Triton | 27 | 0 | 2 | 25 | 0 | **AGREES** |
| tab:rq2-bugs | `1|M1 / choreo` | choreo | 30 | 9 | 4 | 17 | 0 | choreo | 30 | 9 | 4 | 17 | 0 | **AGREES** |
| tab:rq2-bugs | `2|M2 / IREE` | IREE | 23 | 0 | 19 | 4 | 0 | IREE | 23 | 0 | 19 | 4 | 0 | **AGREES** |
| tab:rq2-bugs | `2|M2 / MLIR-linalg` | MLIR-linalg | 50 | 34 | 0 | 10 | 6 | MLIR-linalg | 50 | 34 | 0 | 10 | 6 | **AGREES** |
| tab:rq2-bugs | `2|M2 / Triton` | Triton | 0 | 0 | 0 | 0 | 40 | Triton | 0 | 0 | 0 | 0 | 40 | **AGREES** |
| tab:rq2-bugs | `2|M2 / choreo` | choreo | 30 | 13 | 7 | 10 | 0 | choreo | 30 | 13 | 7 | 10 | 0 | **AGREES** |
| tab:rq2-bugs | `3|M3 / IREE` | IREE | 0 | 0 | 0 | 0 | 40 | IREE | 0 | 0 | 0 | 0 | 40 | **AGREES** |
| tab:rq2-bugs | `3|M3 / Triton` | Triton | 2 | 2 | 0 | 0 | 0 | Triton | 2 | 2 | 0 | 0 | 0 | **AGREES** |
| tab:rq2-bugs | `3|M3 / choreo` | choreo | 12 | 2 | 0 | 10 | 0 | choreo | 12 | 2 | 0 | 10 | 0 | **AGREES** |
| tab:e2-generation | `0|lcccc@{}} Pipeline` | In-bound | Shape | Iteration | Hardware | In-bound | Shape | Iteration | Hardware | **AGREES** |
| tab:e2-generation | `1|Choreo ledger` | yes | yes | yes | yes | yes | yes | yes | yes | **AGREES** |
| tab:e2-generation | `1|IREE` | no | part | no | no | no | part | no | no | **AGREES** |
| tab:e2-generation | `1|MLIR-linalg` | no | part | no | no | no | part | no | no | **AGREES** |
| tab:e2-generation | `1|MLIR-low` | part | no | no | no | part | no | no | no | **AGREES** |
| tab:e2-generation | `1|Triton` | part | no | part | part | part | no | part | part | **AGREES** |
| tab:e2-generation | `2|Choreo ledger` | 9,073 ob. | 4,477 ob. | 326 ob. | 3,477 ob. | 9,073 ob. | 4,477 ob. | 326 ob. | 3,477 ob. | **AGREES** |
| tab:e2-generation | `2|IREE` | 0/15/0 | 14/0/1 | 0/0/15 | 0/0/15 | 0/15/0 | 14/0/1 | 0/0/15 | 0/0/15 | **AGREES** |
| tab:e2-generation | `2|MLIR-linalg` | 7/0/0 | 5/0/2 | 0/0/7 | 0/0/7 | 7/0/0 | 5/0/2 | 0/0/7 | 0/0/7 | **AGREES** |
| tab:e2-generation | `2|MLIR-low` | 4/0/0 | 0/0/4 | 0/0/4 | 0/0/4 | 4/0/0 | 0/0/4 | 0/0/4 | 0/0/4 | **AGREES** |
| tab:e2-generation | `2|Measured basis (S8): yes/partial/no constructs` |  |  |  |  |  |  |  |  | **AGREES** |
| tab:e2-generation | `2|Triton` | 15/0/0 | 0/0/15 | 15/0/0 | 0/15/0 | 15/0/0 | 0/0/15 | 15/0/0 | 0/15/0 | **AGREES** |
| tab:e3-discharge | `0|lrrrr@{}} Cases` | Generated | Discharged | Rate | Residue | Generated | Discharged | Rate | Residue | **AGREES** |
| tab:e3-discharge | `1|Dynamic shape (159)` | 9,197 | 7,989 | 86.9% | 1,208 | 9,197 | 7,989 | 86.9% | 1,208 | **AGREES** |
| tab:e3-discharge | `1|Static shape (151)` | 8,156 | 8,156 | 100.0% | 0 | 8,156 | 8,156 | 100.0% | 0 | **AGREES** |
| tab:e3-discharge | `2|All cases (310)` | 17,353 | 16,145 | 93.0% | 1,208 | 17,353 | 16,145 | 93.0% | 1,208 | **AGREES** |
| tab:e4-cost | `0|lr@{}} Measurement` | Result | Result | **AGREES** |
| tab:e4-cost | `1|Compile-time cost (S10)` | - | - | **AGREES** |
| tab:e4-cost | `1|categories in bucket` | <0.5%: 15 | <0.5%: 15 | **AGREES** |
| tab:e4-cost | `1|categories measured` | 15 | 15 | **AGREES** |
| tab:e4-cost | `1|median overhead` | 0.13% | 0.13% | **AGREES** |
| tab:e4-cost | `1|quantity` | front-end - nvcc compile + link | front-end - nvcc compile + link | **AGREES** |
| tab:e4-cost | `1|reproduces the old RQ4 figure?` | no | no | **AGREES** |
| tab:e4-cost | `2|Residual-check cost, on vs off (S13.E5a)` | - | - | **AGREES** |
| tab:e4-cost | `2|cases inside their own rep spread` | 14 of 14 | 14 of 14 | **AGREES** |
| tab:e4-cost | `2|cases measured` | 14 | 14 | **AGREES** |
| tab:e4-cost | `2|median \Delta (host wall)` | 3.92% | 3.92% | **AGREES** |
| tab:e4-cost | `2|negative \Delta (checks-on faster)` | 6 of 14 | 6 of 14 | **AGREES** |
| tab:e4-cost | `2|static-shape cases measured` | 0 | 0 | **AGREES** |
| tab:e4-cost | `3|Corpus (S5)` | - | - | **AGREES** |
| tab:e4-cost | `3|all cases` | 310 | 310 | **AGREES** |
| tab:e4-cost | `3|dynamic-shape cases` | 159 | 159 | **AGREES** |
| tab:e4-cost | `3|static-shape cases` | 151 | 151 | **AGREES** |
| tab:e5-oracle | `0|lp{0.6\columnwidth}@{}} Quantity` | Reporting path | Reporting path | **AGREES** |
| tab:e5-oracle | `1|Dynamic-oracle reporting point` | Instrumented execution must first reach an observed fault. | Instrumented execution must first reach an observed fault. | **AGREES** |
| tab:e5-oracle | `1|Structural distinction` | The report originates at launch versus after device execution begins; no latency ratio is reported. | The report originates at launch versus after device execution begins; no latency ratio is reported. | **AGREES** |
| tab:e5-oracle | `1|choreo reporting point` | An entry assessment can reject the launch before any device execution. | An entry assessment can reject the launch before any device execution. | **AGREES** |
| tab:e5-oracle | `2|Measured asymmetry (S13.E5b)` | - | - | **AGREES** |
| tab:e5-oracle | `2|both detected` | 4 of 16 | 4 of 16 | **AGREES** |
| tab:e5-oracle | `2|choreo decided, oracle blind` | 4 of 16 | 4 of 16 | **AGREES** |
| tab:e5-oracle | `2|mutants compared` | 16 | 16 | **AGREES** |
| tab:e5-oracle | `2|oracle died with no report` | 0 | 0 | **AGREES** |
| tab:e5-oracle | `2|oracle never reached the fault (launch rejected)` | 2 of 16 | 2 of 16 | **AGREES** |
| tab:e5-oracle | `2|run aborted by choreo's own ungated check` | 6 of 16 | 6 of 16 | **AGREES** |
| tab:e5-oracle | `3|Time to report (median, S13.E5b)` | - | - | **AGREES** |
| tab:e5-oracle | `3|choreo entry check` | 0.51\ s (16 arms) | 0.51\ s (16 arms) | **AGREES** |
| tab:e5-oracle | `3|compute-sanitizer, interior` | 60.16\ s (4 arms) | 60.16\ s (4 arms) | **AGREES** |
| tab:e5-oracle | `3|paired ratio (confounded)` | 84\times over 4 of 16 pairs | 84\times over 4 of 16 pairs | **AGREES** |

### Reading the float drift

- `tab:e2-generation` is deliberately NOT re-derived. Its caption defines `yes` as *expressible at a level where it could be discharged*, while S8 counts only whether a surface can *name* the class. The generated float keeps the classification and prints the S8 basis underneath, with the disagreement spelled out in its note and in `render/stats-merged.json`.
- `tab:e3-discharge` reproduces cell for cell except the static-shape rate, where the paper rounds 100.0% to `100\%`.
- `tab:e4-cost` is structurally different on purpose: four of its seven rows have no committed source, and the generated float reports only what is measured.
- `tab:rq2-bugs` now covers all five compared toolchains. An earlier version of the renderer omitted `mlir-linalg` and `mlir-low`, which is why the generated matrix had three toolchains to the paper's five.

## 2. Prose claims vs the committed statistics

17 of 26 claims agree. The 9 below need an owner decision before the deadline.

| Sec | Claim in the paper | Measured | Verdict | Source / why |
|---|---|---|---|---|
| 5.2 | 120 semantically equivalent mutations | 120 | **AGREES** | S14 denominator.n_cells (= S1 72 injected + 48 no-op) |
| 5.2 | 72 of the 120 mutations are oracle-confirmed corruptions | 72 of 120 | **AGREES** | S2.n_injected |
| 5.2 | \sys surfaces 35 of the 72 corruptions before device execution (48.6%) | 35/72 = 48.6\% | **AGREES** | S2.n_before_device |
| 5.2 | every one of the 37 misses attributed | unattributed = 0 | **AGREES** | S2.never_attribution |
| 5.2 | MLIR-linalg detects only M2 (34/50 before device) | M2 compile 34 / injected 50 | **AGREES** | S1 mlir-linalg.per_class.M2 |
| 5.2 | MLIR-low only M1 (38/48, as runtime bounds traps) | M1 runtime 38 / injected 48 | **AGREES** | S1 mlir-low.per_class.M1 |
| 5.2 | IREE catches 19/23 on M2 | 19/23 | **AGREES** | S1 iree.per_class.M2 |
| 5.2 | Triton 2/27 on M1 | 2/27 | **AGREES** | S1 triton.per_class.M1 |
| 5.3 | \sys is the only compared toolchain that can express all four classes | S8: choreo all four classes carry obligations; no baseline lane has all four at `yes` | **AGREES** | S8_expressibility |
| 5.4 | 17,353 obligations, 16,145 discharged (93.0%) | 17353 / 16145 (93.0\%) | **AGREES** | S5_discharge_rate.all |
| 5.4 | static-shape 100% (8,156/8,156 over 151 cases) | 8156/8156 over 151 cases | **AGREES** | S5_discharge_rate.static |
| 5.4 | dynamic-shape 86.9% (7,989/9,197 over 159 cases) | 7989/9197 over 159 cases (86.9\%) | **AGREES** | S5_discharge_rate.dynamic |
| 5.4 | 675 hoisted to entry, 533 budgeted device-side | runtime 675, budgeted 533, residue 1208 | **AGREES** | S4_per_operator.totals |
| 5.4 | constant folding settles the 7,872 obligations | 7872 | **AGREES** | S6.per_outcome.proven.canonical (NOT S6.totals.canonical = 9,080, which spans all outcomes) |
| 5.4 | direct static checks answer a further 5,353 | 5353 | **AGREES** | S6.per_outcome.proven.direct |
| 5.4 | the two named mechanisms are the whole discharge | 7,872 + 5,353 = 13,225 vs 16,145 discharged (interval 2,920 is never named) | **DIFFERS** | S6.per_outcome.proven -- the paragraph omits the third mechanism (interval discharge), so its arithmetic does not close |
| 5.5 | across the 151 measured dynamic-shape cases | 151 is the STATIC-shape count; dynamic-shape is 159 | **DIFFERS** | S5_discharge_rate.static.kernels |
| 5.5 | median compile-time overhead versus the fully-static pipeline is 0.6% (mean 1.9%) | S10 grand median 0.13\% over 15 categories; `reproduces_rq4: False` | **NO SOURCE** | S10_compile_cost measures front-end vs nvcc (R-D4), not checks-on vs checks-off. The legacy CSV gives a 1.480% median and a 1.912% mean, so 0.6% matches neither quantity |
| 5.5 | 70 cases across 7 operator categories, median of 3 runs per variant | S13.E5a covers 14 cases | **NO SOURCE** | no committed statistic carries a 70-case / 7-category runtime subset |
| 5.5 | -0.1% median at low budget | S13.E5a median 3.92\% over 14 cases, all inside their arm spread | **NO SOURCE** | there are no per-budget-level medians in any committed statistic; benchmark/results/rq3_runtime_overhead.csv is absent |
| 5.5 | -0.3% at high budget | no committed source | **NO SOURCE** | same as the low-budget figure |
| 5.5 | per-category medians within +/-1.1% | no committed source | **NO SOURCE** | same as the per-budget-level figures |
| 5.5 | -11% to +11% per-case scatter at low | no committed source | **NO SOURCE** | same as the per-budget-level figures |
| 5.5 | entry-level is 64% of the residue | 675 / 1208 = 55.9\% | **DIFFERS** | S4_per_operator.totals: no committed statistic yields 64%, so either `the residue' denotes a different set here or the figure is stale |
| 5.6 | we do not quantify a latency ratio here | S13.E5b has a paired ratio of 84x over 4 confounded pairs, marked R-D1 as NOT to be reported as a latency ratio | **AGREES** | S13.E5b.ratio (R-D1) |
| 5.6 | Table~\ref{tab:e5-oracle} records the two reporting points | the regenerated float still carries both points and adds the measured asymmetry | **AGREES** | S13.E5b.detection_asymmetry |

## 3. Verbatim stat-level warnings

These are carried in `stats.json` and are too long or too punctuation-heavy to inline in a LaTeX footnote, so they are quoted here. They are binding on anything written from S13.

**`S13.E5a.launch_unasserted_note`**

> 28 E5a arm record(s) carry NO launch assertion: they were written by a run_e5.py that predates the launch guard, so nothing in the record says whether the kernel reached the device. They are admitted to the median because voiding them would discard the whole existing E5a, but this is an UNASSERTED launch, not a verified one. Measured exposure on this exact data (diag_launch_reject.py, rebuilt on the post-fix binary): 3 of the 14 -- matmul/10_dynamic, max_pool2d/10_dynamic, softmax/10_dynamic -- were silently launch-rejected and recorded ok=True with 324-493 ms timings that measured no device work. RE-MEASURE E5a before quoting any residue number from records carrying this note.

**`S13.E5a.noise_note`**

> 14/14 deltas are INSIDE the arms' own rep-to-rep spread. This is the EXPECTED result, not a failed measurement: the generated device kernel is byte-identical between the two arms and contains zero check calls, so `--disable-runtime-check` removes only HOST-side entry guards (integer comparisons on shape metadata, run once per launch). There is no device-side cost for this host's clock ramp to resolve. The honest RQ3 statement is that runtime residue is structurally zero on the device and below the measurement floor on the host.

**`S13.E5a.negative_delta_note`**

> 6 case(s) show checks-on FASTER than checks-off. A handful of entry guards cannot make a kernel faster, so these are drift or contention, not overhead. Check `clock_drift_mhz` and `exclusive` on those rows before quoting any E5a number. Both devices idle at 345 MHz with persistence_mode Disabled (max 1755 MHz), so a clock ramp alone can produce this -- measured drift here reached 1410 MHz.

**`S13.E5b.absolute_latency_note`**

> The ABSOLUTE time_to_report_us is dominated by a shared process-startup + CUDA-context-init constant (~507 ms median on the choreo-entry arm), NOT by the check itself. Do not describe choreo's entry check as "µs-scale" on this evidence. What the data supports: choreo's latency is INPUT-INDEPENDENT (spread 3.7x across 16 distinct faults, because it aborts at launch before touching the data) while the oracle's is DATA-DEPENDENT (spread 76x, because it must run the instrumented body to the fault); hence the oracle is slower on every pair. Both arms time the executable only -- nvcc compile is excluded from both (run_e5.py SYMMETRY FIX, 2026-09-09). An earlier cut of this statistic timed nvcc+run on the choreo arm and run-only on the sanitizer arm, which produced a 0.2x ratio and INVERTED the claim; that number is void.

**`S13.E5b.detection_asymmetry.premise_violation_note`**

> 6 of the sanitizer arms were built with --disable-runtime-check on the stated premise that the sanitizer would then be the SOLE detector. That premise is FALSE: the flag gates only source A (host runtime_check and device choreo_assert at the HOIST/USE sites), while source B (ArrayProxy::operator[] bounds check) and source C (the user assert BIF) are ungated by construction. On these arms choreo's own ungated check aborted the process, memcheck reported `ERROR SUMMARY: 0 errors`, and the recorded wall clock measured choreo rather than the oracle -- so the ratio would have compared choreo against choreo. Evidence: diag_abort_attribution.py, which runs each executable both with and without compute-sanitizer. This is a DATA problem in the experiment's construction, reported rather than designed around; suppressing sources B and C needs a toolchain flag that may not exist.

**`S13.E5b.ratio.unguarded_all_pairs.note`**

> NOT QUOTABLE. Retained so the guard's effect is auditable: this is what the statistic said before the launch/detection guard excluded 12 confounded pair(s).

**`S10_compile_cost.rq4_note`**

> measured grand median 0.1296% vs RQ4's 0.6%. These are NOT the same quantity: the assessor cannot be disabled on this build, so what is separable is choreo's front end against nvcc, not checks-on against checks-off. No measured category lands in RQ4's 0.5-1% bucket. See `definition`. The coordinator decides which number the paper carries; this lane does not silently substitute one for the other.

**`S10_compile_cost.definition`**

> compile_overhead_pct = 100 * t_choreo_frontend / (t_choreo_frontend + t_nvcc_compile_link); frontend = choreo -gs -t cute -kt <k> -es (parse + assess + emit CUDA, nvcc skipped); nvcc = the generated script's own --compile-link path with choreo's exact flags. The assessor cannot be disabled on this build (-rtc only gates emitted assertions; -rtc=none/-zero-cost/--disable-runtime-check give a byte-identical ledger), so a checks-on/checks-off delta is not obtainable and is NOT what this number reports.

## 4. Elided float notes

A EuroSys submission gets limited body pages, so each generated note above is the short form. The text below was cut from the float note for the page budget and is reproduced here so the argument behind every number stays recoverable. It is candidate prose for the body or the appendix, not deleted work.

**`tab:e2-generation`**

> Cells where the E2 classification is stricter than S8: Triton/In-bound (classified part, S8 names it yes 15/0/0); Triton/Iteration (classified part, S8 names it yes 15/0/0); MLIR-linalg/In-bound (classified no, S8 names it yes 7/0/0); MLIR-low/In-bound (classified part, S8 names it yes 4/0/0); IREE/In-bound (classified no, S8 names it part 0/15/0). The letters are recorded rather than re-derived from S8: an earlier plan to derive them would have turned MLIR-linalg/In-bound from `no` to `yes` and contradicted the caption. The basis denominator is not 15 for every lane: the lanes carry 4, 7 and 15 constructs. `mlir-linalg` measures a 7-kernel composed subset and `mlir-low` a 4-kernel one (`mlir-linalg/README.md`, `mlir-low/README.md`) -- a documented breadth gap, not an absence of support. The second block is therefore a within-lane breakdown and never a cross-lane ratio. The completed Choreo generation count is 17,353 obligations (S3), reported in Table~\ref{tab:e3-discharge}.

**`tab:e4-cost`**

> The float this replaces quoted 0.6\% in bucket `0.5-1%`; the measured quantity is 0.13\%, which is why S10 carries an explicit `reproduces_rq4: false`. NOT REPRODUCED HERE -- four figures the replaced float carried have no committed source and are deliberately absent rather than guessed: "151 dynamic-shape cases" (151 is the STATIC case count; the dynamic count is 159), a "70 cases across 7 categories" runtime subset, $-0.1\%$ at low budget, and $-0.3\%$ at high budget. The per-budget-level medians need `benchmark/results/rq3_runtime_overhead.csv`, which is not in the repository.

**`tab:e5-oracle`**

> The paired ratio is confounded on 12 of 16 pairs and is reported only with its pair count; the unguarded `all_pairs` arm carries an explicit `NOT QUOTABLE` annotation and is excluded. It is NOT a latency multiplier: R-D1 directs E5 to report detection asymmetry (coverage), which is what the rows above it do. On 6 further pairs the dynamic oracle never got to report because \sys's own ungated check aborted the run first, which is a strictness property of \sys rather than a detector result; the asymmetry rows and the ratio row use different pair populations and must not be combined. Verbatim support text, including the two stat-level warnings (`absolute_latency_note`, `premise_violation_note`), is in `render/PROSE-DRIFT.md`.

**`tab:rq2-bugs`**

> 4 registered cell(s) carry no detection signal -- 0 injected, 40 `n/a` -- and are represented by the one `n/a` witness kept per class rather than repeated: M1/MLIR-linalg, M2/MLIR-low, M3/MLIR-linalg, M3/MLIR-low. Triton's `Launch` column counts only its own checks, but an independent sanitizer flags and exercises 23 of the same 27 M1 mutants (S12). That is a detector supplement, not a value this table could put in a column without double-counting.

## 5. Artifacts the paper consumes

`make paper` copies exactly these, and nothing else:

- `tables/rq2_bugs.tex`
- `tables/e2_generation.tex`
- `tables/e3_discharge.tex`
- `tables/e4_cost.tex`
- `tables/e5_oracle.tex`
- (no generated PDF: `main.tex` has zero `\includegraphics` and every figure is hand-authored `pgfplots`)

Hand-authored and never written by a render: `figures/fig_workflow.tex`, `figures/fig_plural_vn.tex`, `figures/fig_e1_detection.tex`.

