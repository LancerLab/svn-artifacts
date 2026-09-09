# choreo lane — `benchmark2/choreo/`

Worker lane for the **choreo** toolchain. Owns statistics **S1–S7, S10, S13**
(see `../schema/statistics-manifest.md`); S8/S9/S11/S12 belong to other lanes.

> ⚠ **Open decisions blocking publication are listed in
> [`DECISIONS-NEEDED.md`](DECISIONS-NEEDED.md).** Two of them (D1, D2) invalidate
> numbers that were previously reported upward. Read that file before quoting any
> S13 figure.

## Layout

```
choreo/
  run.sh                 # entry point: setup|minimal|e2|e3|e4|e5|e5b|never|collect|stats|all
  gen_mutants.py         # M1/M2/M3 mutant generation -> mutants/ + raw/mutant_manifest.json
  mutations.py           # mutation operators
  run_e1.py .. run_e5.py # one driver per experiment
  analyze_never.py       # root-cause attribution for `never`-detected mutants
  calibrate_oracle.py    # §7 oracle calibration
  arch_sweep.py          # arch-sensitivity sweep (sm_86 vs native)
  arch_e5_probe.py       # E5 exposure probe for the arch question
  diag_abort_attribution.py  # who kills the E5b "process-abort" arms: memcheck or choreo?
  diag_launch_reject.py      # launch-rejection diagnostic
  collect.py             # raw/*.json -> ../results/choreo/*.jsonl (schema-validated)
  stats.py               # ../results/choreo/*.jsonl -> ../results/choreo/stats.json
  toolchain.py           # binary/checkout identity (provenance)
  gpuinfo.py             # host-quiet + device snapshot helpers
  mutants/               # 120 generated mutants (committed)
  raw/                   # per-experiment artifacts (small JSONs committed, logs ignored)
```

Published output lives in **`../results/choreo/`**: the register (`*.jsonl`) and
`stats.json`.

## Running

```bash
./run.sh all          # full pipeline; holds ../.gpu-lock for the timing lanes
./run.sh e5b          # latency half only, reusing the banked E5a checkpoint
./run.sh collect      # raw -> register (schema-validated)
./run.sh stats        # register -> stats.json
```

`--jobs` **must stay 1** for E4/E5: these are timing lanes and concurrent runs
contend for the device. The GPU lock is `../.gpu-lock`; a hand-made lock is never
auto-released and must be removed manually.

## Provenance — how a reviewer re-derives the committed numbers

`raw/` is **partly gitignored** (`.gitignore`), following the `iree` lane
precedent and `../homework-check.md` item **I4**, which accepts a gitignored
`raw/` provided this note exists.

**Committed** (small, and each backs a published number):
`raw/mutant_manifest.json`, `raw/e1_mutant_records.json`, `raw/e4_compile_cost.json`,
`raw/e5_runtime.json`, `raw/e5_runtime.e5a.json`, `raw/never_attribution.json`,
`raw/c2_probe_groundtruth.json`, `raw/arch_sweep.json`, `raw/arch_e5_probe.json`,
`raw/oracle_survey.json`, `raw/oracle_policy.json`, `raw/e3_remainder.json`.

**Ignored, and how to regenerate:**

| Ignored | Size | Regenerate with |
|---|---|---|
| `raw/e1_logs/` | 48 MB | `./run.sh minimal` (E1) |
| `raw/e2_logs/`, `raw/e2_ledger.json` | 20 MB | `./run.sh e2` |
| `raw/oracle_survey/`, `raw/oracle_calibration/` | 24 MB | `python3 calibrate_oracle.py` |
| `raw/e4_logs/`, `raw/e5_logs/` | — | `./run.sh e4` / `./run.sh e5` |
| `raw/probe_raw/` | — | `python3 arch_e5_probe.py` |
| `raw/*.log` | — | console transcripts of the above |
| `raw/*.pre-reproject.json`, `*.pre-exclusive.json`, `*.oldclassifier.json` | — | superseded snapshots, kept locally for audit only |

`raw/e2_ledger.json` (8.8 MB) is ignored because it is fully re-derived into the
**committed** register at `../results/choreo/obligation.jsonl` (17,353 records) by
`./run.sh e2 && ./run.sh collect`. Committing both would duplicate the corpus.

**Data flow:** `run_e*.py → raw/*.json → collect.py → ../results/choreo/*.jsonl →
stats.py → ../results/choreo/stats.json`. `stats.py` reads **only** the register,
never `raw/`, so every number in `stats.json` traces to a committed `.jsonl`.

## Known limitations (recorded, not hidden)

> **All open decisions are now resolved** — see
> [`DECISIONS-NEEDED.md`](DECISIONS-NEEDED.md) "COORDINATOR RULINGS — 2026-09-10"
> (R-D1…R-D7). The items below are the resolved findings, kept as the audit trail.

- **`never = 0` is RETIRED (R-D2).** 37 of 72 injected mutants are
  never-detected, all attributed by cause (25 out-of-scope/not-assessed + 12
  hoisting/config). §5.2 ¶2 reports **35/72 = 48.6% before device execution**
  with the per-cause table — it does **not** claim `never = 0`.
- **E5b's "sole detector" premise is FALSE (R-D1).** `--disable-runtime-check`
  gates only assertion source A; sources B and C are ungated by construction.
  §5.6 ¶1 is reframed around the **detection asymmetry** (choreo caught 4/4
  faults the oracle reported zero errors on), not a latency multiplier.
- **E5a measured 0 static kernels (R-D3).** §5.5 ¶2's premise that static shapes
  carry zero residue is untested here; the static claim is dropped and residue is
  stated as *structurally zero on the device, below the measurement floor on the
  host*.
- **`toolchain.py`'s mtime heuristic understates** the binary's source commit
  under rebuild-then-commit. `binary_sha1_12 = c3ebb1654d1d` (`f2f238f`) is the
  authoritative build; the `1fa4719` launch-rejection fix postdates the data (R-D6).

## Corpus exclusions and repairs (R-D7)

**R-D7a — `reshape/14` repaired.** `reshape/14_efficientnet_64x1280x7x7_64x62720.co`
used the `output = dma.copy … => global; return output.data;` idiom, which has no
`parallel` block. The front end accepts it (rc=0) but emits
`warning: dma.copy falls back to naive copy (participating thread count is unknown
or invalid)` and lowers the tensor declarations to **global scope**, so nvcc dies
with 15 errors (`identifier "output__buf__" is undefined`, `__syncthreads()` at
global scope, `cannot overload functions distinguished by return type alone`).
Repaired to the `f32 […] output; parallel p by 1 { dma.copy … => output; } return
output;` idiom already used by `reshape/3`–`13` (commit `16cb601`).

Verified after the repair: front end rc=0, nvcc rc=0, **`Test reshape14 Passed!`**
with an execution time printed, and the ledger still yields **0 obligations** —
identical to the already-repaired sibling `reshape/13`. The repair is therefore
**corpus-neutral**: the 17,353-obligation ledger and every committed statistic are
unaffected, and no register file changed. `reshape/14` is not a mutant base (no
reshape kernel is), so E1 is unaffected.

⚠ **The register still carries the pre-repair identity for this kernel.**
`results/choreo/kernel.jsonl` records `kernel_hash = 4f3494350bcd` and
`compile = "ok"`; the repaired source hashes to `fbf597590d40`. The register is
**deliberately not regenerated** — R-D7b forbids re-deriving the ledger this close
to deadline, and R-D6 forbids re-running. Note also that `compile = "ok"` was
always about the **ledger fast path** (`-gs --dump-ledger`, no nvcc), which never
exercised the failing nvcc stage; it was never a claim that the kernel builds an
executable.

⚠ **This is a defect class, not a single kernel.** `reshape/1`, `reshape/2`, and
`reshape/15`–`reshape/21` share the identical broken idiom — `reshape/1` and
`reshape/16` were confirmed to fail with the same 15 nvcc errors. R-D7a approved
repairing **`reshape/14` only**, so the other nine are left as-is and recorded
here. All ten carry **0 obligations**, so none of them affects any committed
statistic; they simply cannot produce an executable. Repairing them would be
corpus-neutral too, but is out of scope for this deadline.

**R-D7b — `conv2d/10_dynamic` excluded, not repaired.**
`conv2d/10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D.co` fails CuTe
codegen with 27 errors. The reported `static_assert` text —
*"On SM_86, must satisfy: CeilTo128Byte(bpe * dst dim0) * dim1 * dim2 * dim3 *
dim4 < 4GB"* — is **not the real violation**. Recomputed, the actual cause is the
`shared f32 [M,K] A` tile at 64 × 1152 × 4 = **294,912 B = 288.0 KB** against a
~228 KB hardware maximum.

Two further facts a reviewer must know:

- **The filename lies.** It advertises `112x112`, but the body is
  `[8, 128, 16, 16]` with stride 2. Benchmark `.co` filenames in this suite
  routinely disagree with their contents — always read the body.
- **It is retained in the frozen ledger.** It carries **89 obligations**
  (`kernel_hash d434e35e9aee`): 11 `runtime` (all canonical, all `enabled: true`)
  and 78 `proven` (16 canonical + 62 direct). Repairing it would change
  `kernel_hash` and therefore the 17,353 total, which R-D7b forbids. It is **not**
  a mutant base.

**R-D7c — the 36 silent launch rejections are documented, not audited as a class.**
Under the pinned pre-fix binary (`f2f238f`) with
`--max-local-mem-capacity=2000000`, the dynamic-shape path sizes the per-thread
local arena to full capacity (`lib/mem_reuse.cpp:430-434`), so the driver reserves
`capacity × maxThreadsPerSM × numSMs` = **466.9 GB** on H800 PCIe (114 × 2048)
against 85 GB available → `cudaErrorInvalidValue`, **regardless of the actual
launch dims**. **36 of 153 dynamic cases** were affected and **all reported
success**. The fix is `1fa4719` ("codegen: check launch errors after plain `<<<>>>`
kernel launches"), which **postdates the data** — this is the same disclosure as
R-D6, folded in here per the ruling.

These rejections affect **only the timing/measurement lanes (E4/E5)**, never the
static E2/E3 corpus: the ledger fast path does not launch a kernel. See
`diag_launch_reject.py` for the reproduction and `run_e5.py:384` for the
`launch_verdict()` guard that makes the rejection loud.

## Toolchain bugs to log (found by this lane, not fixed by it)

Recorded per R-D1/R-D2. These are **croqtile defects**, disclosed rather than
worked around; none is hidden behind a flag.

| # | Defect | Evidence | Ruling |
|---|---|---|---|
| 1 | **`choreo.h:221` fires despite `--disable-runtime-check`.** This site is `choreo_assert` = assertion source **A-device**, which the flag is supposed to gate. | E5b abort attribution; `diag_abort_attribution.py` | R-D1 |
| 2 | **Guard hoisted past the induction variable's reset** — 6 mutants. The guard is emitted but is vacuous, so the mutant is never detected. Confirmed by `--disable-assert-hoist`. | `never_cause = HOISTING_DEFECT` (6) | R-D2 — *log as a toolchain bug, do **not** hide behind a flag* |
| 3 | **Only one of an index's two bounds is generated** — 4 mutants. A croqtile **soundness** bug: the omitted bound is never checked. | `never_cause = C3_LOWER_BOUND_OMITTED` (4) | R-D2 |
| 4 | **Silent launch rejection** — 36 dynamic cases reported success while the driver refused the launch. | `1fa4719`; `diag_launch_reject.py` | R-D7c / R-D6 |

Bugs 2 and 3 are emitted programmatically by `stats.py` under
`S2_before_device.never_attribution.toolchain_bugs_to_log`, so they travel with the
register rather than living only in prose. Bugs 2 and 3 together account for **10 of the 12**
`never` mutants attributed to toolchain-or-configuration causes; the remaining 2
are `C1_COST_FILTER_SUPPRESSED`.
