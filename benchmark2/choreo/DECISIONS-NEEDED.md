# choreo lane — DECISIONS NEEDED (dispatcher / owner)

**Raised:** 2026-09-10 00:41 CST by the choreo worker.

> ## ⏰ DEADLINE: ~19 hours remain
> ASPLOS 2027 submission closes **September 9, 2026 AoE** = **2026-09-10 12:00
> UTC**. At the time of writing that is **19 h 18 m** away.
>
> **This makes D6 (rebuild + re-run E1–E5) almost certainly infeasible** — E1
> alone took hours, E2 produced 17,353 records, and the E5a re-measure is ~27 min
> of *exclusive* GPU that cannot overlap with anything. The realistic path is
> **D6b**: pin the paper explicitly to the binary that actually ran
> (`binary_sha1_12 c3ebb1654d1d`, source `f2f238f`) and state that the
> launch-rejection fix `1fa4719` postdates the data.
>
> **The worker needs a decision on D1 and D2 within hours, not days**, because
> both block prose in §5.2 and §5.6 and there is no time for a second round.

**Status of the lane:** code complete, `collect` PASS (17,858 records, 0 schema
violations), `stats.py` rc=0. **But five numbers previously reported upward are
now known to be wrong or unquotable**, and two acceptance criteria fail. Nothing
below was silently worked around — per `../README.md`'s onboarding rule (*"Ask
questions first — never silently improvise"*) and the owner's 2026-09-09
directive item 1 (*"it's a data problem, report it, don't design around it"*).

All figures are read from the committed `../results/choreo/stats.json`.

**Blocking order:** D1 and D2 block prose. D3–D5 are re-registrations that need a
decision only because they contradict numbers already in `main.tex`. D6 gates
whether *any* of it should be re-measured first.

---

## D1 — E5b's latency ratio is not quotable, and its premise is FALSE  ⚠ BLOCKING

### What was reported upward
The owner's 2026-09-09 directive item 3 prescribed the paper wording
*"median **64×** longer to first report, up to **472×**"*, replacing the L4
placeholder *"up to ~100×"*.

### Why both numbers are invalid
A launch/detection guard now quarantines **12 of the 16** E5b pairs:

| Quarantine class | n | Why it is not an oracle measurement |
|---|---|---|
| `no-fault-reported` | 4 | memcheck ran the body to completion and printed `ERROR SUMMARY: 0 errors`. **This class contains the 472× pair.** There is no oracle report to time. |
| `process-abort` | 6 | The process was killed by **choreo's own ungated assertion**, not by memcheck. Spans 1.57×–233×, so the bias is not even uniform. |
| `launch-rejected` | 2 | The driver refused `cudaLaunchKernel`; the oracle never ran the body. |

The truly admissible statistic is:

> **n = 4, median 84.45×, min 2.58×, max 204.39×, oracle slower on 4/4.**

`stats.json` carries the pre-guard figure under
`E5b.ratio.unguarded_all_pairs` (`n=16, median 64.16, max 472.13`) marked
**`NOT QUOTABLE`**, retained only so the guard's effect is auditable.

### The premise violation (the real finding)
E5b builds the sanitizer arm with `--disable-runtime-check` on the stated premise
that *"the sanitizer is the sole detector."* **That premise is false.** The flag
gates only assertion **source A**. Sources **B** (`ArrayProxy::operator[]` bounds
check, `runtime/choreo.h:886`) and **C** (the user `assert` BIF) are **ungated by
construction**.

`diag_abort_attribution.py` proves it: it runs each of the 6 executables **twice**
— once with no sanitizer present, once under memcheck. Every one aborts alone
(`baseline_rc = −6`) at `choreo.h:221` (`choreo_assert`) or `choreo.h:886`, and
under memcheck the tool's own summary is `ERROR SUMMARY: 0 errors` on all 6.

So on those arms the recorded wall clock measured **choreo**, and the ratio would
have compared choreo against choreo.

> **Secondary caveat for the toolchain owner:** `choreo.h:221` is source
> **A-device**, which the gating model says *is* disabled by
> `--disable-runtime-check` — yet it fired anyway. Either the gate does not cover
> every A-device emitter, or these asserts come from a path that never consults
> `DisableRuntimeCheck()`. **Do not assume the flag silences all of source A.**

### Decision required
1. **Re-frame §5.6 ¶1 around the detection asymmetry (D1a below) and drop the
   multiplier headline.** ← *worker recommendation.* It is the only E5b claim
   that survives, and n=4 is too small to headline a multiplier in any case.
2. **Re-run E5b with sources B and C actually suppressed.** This needs a
   toolchain flag that **may not exist**. If the owner can name one, this is the
   cleanest fix; the worker cannot find it.
3. **Restrict E5b** to mutants whose bases neither launch-reject nor abort on
   ungated checks — i.e. accept n=4 and publish it with the caveat.

### D1a — the salvaged result (publishable regardless of the above)
**choreo detected all 4 faults that compute-sanitizer ran to completion and
reported zero errors on** (`M2.s1.ln1.bias`, `M2.s1.ln1.caller.bias`,
`M2.s1.ln3.bias`, `M2.s1.mm1.rhs`). Of the 8 arms where the oracle actually
rendered a verdict, it missed 4 and caught 4.

This is a **coverage** claim, not a speed claim. It needs no oracle report to
time, so it is immune to every latency confound above. Emitted at
`S13_runtime_and_latency.E5b.detection_asymmetry`.

⚠ Note this **inverts** the intended story: §5.6 was written to show choreo is
*faster* than a dynamic oracle. What the data actually supports is that choreo
catches faults the oracle **cannot see at all**. That is a stronger claim but a
different one, and it needs an owner decision on framing.

---

## D2 — the `never = 0` acceptance criterion FAILS (37 of 72)  ⚠ BLOCKING

`S2_before_device`: `n_injected 72`, `n_before_device 35`, **`n_never 37`**,
`n_discarded_noop 48`, `pct_before_device 48.6111`, **`never_is_zero: false`**.

The binding 2026-09-09 addendum item 5 requires choreo `never = 0`. It is not
met, so **§5.2 ¶2's storyline is VOID** per `l3-evaluation-e1-e5.md`'s explicit
gate (*"if the choreo lane's current 37-never anomaly does not resolve, this
subsection's storyline is void and must be re-aligned (stop, do not write)"*).

### Root cause is already attributed (`raw/never_attribution.json`, `raw/c2_probe_groundtruth.json`)

| Cause | n |
|---|---|
| `C4_NOT_ASSESSED` | 19 |
| `C5_OUT_OF_SCOPE` | 6 |
| `HOISTING_DEFECT` | 6 |
| `C3_LOWER_BOUND_OMITTED` | 4 |
| `C1_COST_FILTER_SUPPRESSED` | 2 |

**8 of the 37 are recoverable by flags.** With `-rtc=all` plus hoisting disabled,
S2 becomes **43/72 = 59.72%** vs **35/72 = 48.61%** at the pinned default.

### Decision required
- **D2a:** Is `never = 0` still the acceptance criterion, or is it relaxed to a
  documented non-zero with per-cause attribution? The 6 `HOISTING_DEFECT` cases
  look like a genuine toolchain bug; the 19 `C4_NOT_ASSESSED` look like a scope
  question about what the assessor is supposed to cover.
- **D2b:** May the lane re-run E1 under `-rtc=all` + hoist-disabled (the 2×2 flag
  matrix) and report **both** configurations? That raises the headline from
  48.61% to 59.72% but changes the pinned-flag contract in `../manifest.md`,
  which is coordinator-owned.
- **D2c:** If `never = 0` stays mandatory, §5.2 ¶2 cannot be written at all and
  the L3 storyline needs re-alignment.

⚠ This is **independent** of the launch-rejection bug (D3): 0 of the 18
never-on-rejected-base mutants are in the 37-probe population. Fixing D3 does not
fix D2.

---

## D3 — E5a must be re-measured before any residue number is quoted

`E5a.launch_guard.n_arms_launch_unasserted = **28**`. All 28 arm records were
written by a `run_e5.py` that predates the launch guard, so nothing in the record
says whether the kernel reached the device.

Measured exposure on this exact data (`diag_launch_reject.py`, rebuilt on the
post-fix binary): **3 of the 14** kernels — `matmul/10_dynamic`,
`max_pool2d/10_dynamic`, `softmax/10_dynamic` — were **silently launch-rejected**
and recorded `ok=True` with **324–493 ms** timings that measured **no device
work**.

Root cause is upstream commit `1fa4719` (*"codegen: check launch errors after
plain `<<<>>>` kernel launches"*): with
`--max-local-mem-capacity=2000000` the dynamic-shape path sizes the per-thread
local arena to the full capacity, and the driver reserves
`capacity × maxThreadsPerSM × numSMs` = **466.9 GB** on an H800 PCIe (114 SMs ×
2048 threads) against 85 GB of device memory — **regardless of actual launch
dims**, so even a degenerate 2-thread launch is refused. Pre-`1fa4719` this was
silent; post-`1fa4719` it aborts loudly (rc=134).

Also note `E5a.noise_note`: **14/14 deltas are inside the arms' own rep-to-rep
spread**, and 6 cases show checks-on *faster* than checks-off (clock ramp; both
devices idle at 345 MHz with `persistence_mode Disabled`, measured drift reached
1410 MHz). The generated device kernel is **byte-identical** between arms and
contains zero check calls, so the honest statement is that runtime residue is
**structurally zero on the device and below the measurement floor on the host**.

### Decision required
- **D3a:** Approve the ~27 min exclusive-GPU re-measure on the post-fix binary
  (owner directive item 1 already asked for this; it is pending D6).
- **D3b:** Confirm the intended framing for §5.5 ¶2 given `noise_note` — a
  "below the measurement floor" result, not a percentage.
- **D3c:** §5.5 ¶2's premise that *"static-shape cases carry zero residue by
  construction"* is **false at the code level** — static kernels emit 3–8 entry
  `runtime_check` sites (`static_premise_note`). E5a measured **0 static
  kernels** (`--size small` selects dynamic only). Approve `--include-static`, or
  drop the static claim from §5.5 ¶2.

---

## D4 — S10 measures a different quantity than RQ4's 0.6%

`S10_compile_cost`: `grand_median_pct = **0.1296**`, `bucket_distribution =
{'<0.5%': 15}`, **`reproduces_rq4: false`**, `paper_reference = {rq4_median_pct:
0.6, rq4_bucket: "0.5-1%"}`.

The owner's directive item 3 called this an *"expected placeholder"* and said
0.13% is stronger for E4 ¶1. **It is stronger, but it is not the same
measurement**, and the lane will not substitute one for the other silently:

- RQ4's 0.6% was **checks-on vs checks-off**.
- S10's 0.1296% is **choreo front end vs nvcc compile-link**.

The assessor **cannot be disabled on this build** — `-rtc` only gates *emitted
assertions*, and `-rtc=none` / `-zero-cost` / `--disable-runtime-check` all give a
**byte-identical ledger**. So a checks-on/checks-off delta is not obtainable.
No measured category lands in RQ4's 0.5–1% bucket.

### Decision required
Which number does the paper carry, and does §5.5 ¶1 get rewritten to describe
front-end-vs-nvcc rather than checks-on-vs-off? The current L4 text implies the
latter quantity.

---

## D5 — S3/S5 re-registration (mechanical, but contradicts `main.tex`)

These are the *"expected placeholder"* items from directive item 3. They need no
judgement, but they touch 4+ sites in `main.tex` plus the abstract, so they are
listed for the sweep:

| Statistic | In paper now | Measured | Note |
|---|---|---|---|
| S3 grand total | **17,717** | **17,353** | `delta: −364` |
| S5 all | **93.2%** | **93.04%** | 16,145 / 17,353 discharged |
| S5 static | **99.2%** | **100.0%** | 8,156 / 8,156, remainder 0 |
| S5 dynamic | **87.9%** | **86.87%** | 7,989 / 9,197, remainder 1,208 |

⚠ The abstract's *"210/210"* and the intro/RQ2/conclusion suite-size numbers are
**provisional placeholders** built on a 210-kernel suite; the real corpus is 310
kernels / 17,353 obligations. `storyline-state.md` already records this.

### Decision required
Confirm the sweep, and confirm the residue count in §5.4 ¶1 changes from
**1,199** to **1,208**.

---

## D6 — the binary predates the checkout; rebuild invalidates everything  ⚠ GATES D1–D5

`toolchain_identity`: `binary_sha1_12 = c3ebb1654d1d`, `version = f2f238f…`,
`checkout_head = **1fa4719**…`, **`binary_stale_vs_checkout: true`**.

Every committed record says `f2f238f`, but the checkout HEAD is `1fa4719` — which
is *precisely the commit that makes silent launch rejection loud* (D3). So the
data was collected on a binary that **does not contain the fix that exposes the
bug**.

`stats.json`'s own `toolchain_warning` says: *"binary predates checkout… Rebuild
and re-run before pi[nning]"*.

A rebuild invalidates: every probe/cause finding, the ground-truth fixture, the
arch sweep, the E4 artifact, the E5 arch probe, and the E5b latency data.

⚠ **Known limitation of `toolchain.py`:** its mtime heuristic **understates** the
binary's source commit under rebuild-then-commit. The binary was built 14:57:15
and `1fa4719` was committed 15:04:31, so the heuristic returns `f2f238f` even
though the binary may already contain `1fa4719`'s content. `binary_sha1_12` is the
only field that pins which build actually ran. **This ambiguity should be
resolved before any re-measure**, or the re-measure will be attributed to the
wrong commit.

### Decision required
- **D6a:** Rebuild `croqtile` at `1fa4719` and re-run E1–E5. ⚠ **With ~19 h to
  the deadline this is almost certainly not achievable** — it would also invalidate
  the arch sweep, the ground-truth fixture, and the E4 artifact, all of which
  would need re-running too.
- **D6b:** ← *worker recommendation given the deadline.* **Pin the paper to
  `f2f238f` explicitly** and state that the launch-rejection fix `1fa4719`
  postdates the data. Cite `binary_sha1_12 c3ebb1654d1d` as the build that ran.
  This is honest, costs no GPU time, and is already what every committed record
  says.
- **D6c:** Note that `build-release/choreo` is a **shared path** — a rebuild also
  breaks host-quiet for any other worker's timing lane mid-run. With the deadline
  this close, a rebuild is a risk to *other* lanes' data, not just this one's.

---

## D7 — two `benchmark/choreo` kernels fail; one is corpus-neutral, one is not

| Kernel | Obligations in the frozen E2 ledger | Risk |
|---|---|---|
| `reshape/14_efficientnet_64x1280x7x7_64x62720.co` | **0** (static) | **SAFE** — corpus-neutral. No parallel block at all, so `__syncthreads()` is emitted at global scope. Exactly `16cb601`'s reshape/3-13 pattern; fix template in hand. |
| `conv2d/10_dynamic_32x128x112x112_256x128x3x3_32x256x56x56_S_P_D.co` | **89** (11 runtime / 62 direct), `kernel_hash d434e35e9aee` | **NEEDS A DECISION** — fixing it changes `kernel_hash` and therefore the 17,353 total in D5. |

⚠ The conv2d **filename lies**: the content is `[8, 128, 16, 16]` with stride 2,
not `112x112`. Recomputed footprint (stride 2 → Ho=Wo=8, M=64, K=1152): the
`shared f32 [M,K] A` tile is 64×1152×4 = **294,912 B = 288.0 KB** against a
~228 KB hardware max. That is the real violation, not the CuTe `< 4GB`
static_assert text. Neither kernel is a mutant base.

⚠ **The "2 failures" inventory UNDERCOUNTS.** Per `1fa4719`, **36 of 153**
dynamic benchmark cases were silently launch-rejected while reporting success.
They are not compile failures, so they never appeared in a failure list.

### Decision required
- **D7a:** Approve the `reshape/14` fix (safe, 0 obligations).
- **D7b:** Decide on `conv2d/10_dynamic` — fix it and re-derive the ledger
  (changing D5's 17,353), or exclude it with a note.
- **D7c:** Should the 36 silently-rejected dynamic cases be audited as a class?

---

## What is already done and needs no decision

- **Guards implemented and verified** (owner directive items 1, 2, 5):
  launch-rejection guard in `run_e5.py`; launch/detection guard in `stats.py`;
  `rc` + `verdict` persisted on both the detail and the register record;
  `n_confounded` and the clean median emitted side by side so exclusions are
  explicit rather than silent.
- **A latent guard bug found and fixed before it could fire:** `launch_status`
  tested `launch_ok` before `quarantine_reason`. On a sanitizer record `launch_ok`
  is derived from the **baseline** run, and a genuinely faulting mutant's baseline
  aborts *by design* (`rc=−6`) — so all 4 real detections would have been
  quarantined on the next re-measure, collapsing the ratio to **n_pairs = 0**.
  The committed register is old-style (no `launch_ok` key), so this was latent.
- **Classifier unified:** `stats.py` imports `run_e5.sanitizer_verdict` rather
  than duplicating it, so there is exactly one definition of the verdict classes
  (`stats.sanitizer_verdict is run_e5.sanitizer_verdict → True`). 9/9 unit cases
  pass.
- **`RE_SANITIZER` false positive fixed:** a bare `ERROR` alternative matched
  compute-sanitizer's own `========= ERROR SUMMARY: 0 errors` trailer, so
  zero-error reports were counted as detections.
- **The arch question is closed:** `sm_86` comes from choreo's codegen default,
  not the lane scripts. Static analysis is arch-independent (311-kernel sweep:
  0 errors, 309 label-only, 2 real differences, both `batch_norm`, opposite
  directions). 0 of 120 E1 mutants derive from the 2 arch-sensitive kernels. E5
  exposure measured nil (`raw/arch_e5_probe.json`: agrees 0, disagrees 0,
  both-inside-noise 2, errors 0). **Do not add `-arch=native`.** The E2/E3 corpus
  (17,353 records) is **valid — do not re-run it.**
- **`-rtc=none` ≡ `--disable-runtime-check`**, byte-identical (proved).
- **E3 integrity PASS**, `dropped = 0`.

---

## Suggested resolution order — revised for the ~19 h deadline

The order below assumes **D6b** (pin to `f2f238f`, no rebuild). If the owner
chooses D6a instead, everything else slips and the submission is at risk.

1. **D6b — decide now (minutes).** It gates everything: if there is no rebuild,
   the existing data stands as-is and the remaining decisions are about *framing*,
   not re-measurement.
2. **D2 — decide now (hours).** The top blocker per the owner's own directive
   item 4. §5.2 ¶2 cannot be written until it lands, and there is no time for a
   second round. **D2a is the real question**: is `never = 0` still mandatory?
   If yes, §5.2 ¶2 is void and the L3 storyline needs re-alignment *tonight*.
3. **D1 — decide now (hours).** §5.6's framing. Option 2 (suppress sources B/C)
   needs a toolchain flag that may not exist and is **not achievable in 19 h**;
   realistically it is option 1 (re-frame around the detection asymmetry) or
   option 3 (publish n=4 with the caveat).
4. **D3 — only if GPU time is free.** The ~27 min E5a re-measure is the one
   re-measurement that *might* still fit. But note `noise_note`: 14/14 deltas are
   inside the arms' own spread, so a re-measure is likely to reproduce "below the
   measurement floor" rather than produce a percentage. **D3b/D3c (framing) can be
   decided without re-measuring.**
5. **D4, D5 — prose-only, no GPU.** Both are re-registrations against the
   committed `stats.json` and can be swept as soon as D1–D3 land.
6. **D7 — defer.** `reshape/14` is safe but changes nothing published (0
   obligations). `conv2d/10_dynamic` would change the 17,353 total in D5 and
   **should not be touched this close to the deadline**.

The worker will not sweep prose into `main.tex` until D1–D5 are decided, per the
owner's instruction that prose is swept **only from a committed `stats.json`**.

### If no decision arrives in time
The defensible fallback, requiring no new measurement and no prose the data does
not support:
- **§5.2 ¶2:** report S2 as measured — **35/72 = 48.61% before device execution**
  — with the per-cause attribution table, and state plainly that `never ≠ 0`.
  Do **not** claim `never = 0`.
- **§5.6 ¶1:** lead with the **detection asymmetry** (choreo caught 4 of the 8
  faults the oracle actually judged, and all 4 that the oracle ran to completion
  and reported zero errors on). Drop the multiplier headline entirely; if a
  latency figure is wanted at all, use **n=4, median 84.45×** with the
  confound stated.
- **§5.5:** state residue as **structurally zero on the device, below the
  measurement floor on the host** (per `noise_note`), not as a percentage.
- **Abstract/intro/conclusion:** the *"210/210"* placeholder must go; the real
  corpus is **310 kernels / 17,353 obligations**.

---

# COORDINATOR RULINGS — 2026-09-10 (all decisions)

Reviewed against the updated design (`manifest.md` §5.1–5.2,
`specs/mutation-specs.md` §0/§4.1/§5.1/§7.1) and the committed
`../results/choreo/stats.json`. **Every ruling below is final; no re-measurement
is required and no prose that the data does not support is sanctioned.** The
worker's fallback framing (above) is ratified as the *actual* framing, not a
last resort.

## R-D6 — pin to `f2f238f`, no rebuild (D6b). **APPROVED.**

`binary_sha1_12 = c3ebb1654d1d` is the authoritative build identity; the
`toolchain.py` mtime heuristic is advisory only. State in `manifest.md` §5 that
the choreo data was collected on `f2f238f` and that the launch-rejection fix
`1fa4719` **postdates** the data. Do not rebuild — a rebuild invalidates the
arch sweep, ground-truth fixture, and E4/E5 artifacts, and breaks host-quiet for
other lanes. The stale-binary flag is a disclosure, not a defect.

## R-D2 — `never = 0` is RETIRED; report with per-cause attribution. **D2a.**

`never = 0` was a Phase-0 gate calibrated against a 210-kernel corpus and predates
per-cause attribution; it is **not** the acceptance criterion any more. The new
criterion is: **no *unattributed* `never`** — every `never` mutant must carry one
of the five recorded causes (`C4_NOT_ASSESSED`, `C5_OUT_OF_SCOPE`,
`HOISTING_DEFECT`, `C3_LOWER_BOUND_OMITTED`, `C1_COST_FILTER_SUPPRESSED`).

- §5.2 ¶2 is **rewritten, not voided**: "choreo catches **35/72 (48.6%)** before
  device execution (24 compile-time refutations + 11 launch-time guards); of the
  37 not caught, **25 are out-of-scope or not-assessed-by-construction** and **12
  are attributable to a hoisting defect + cost-filter/lower-bound configuration**."
- **D2b — yes, report the 2×2 flag matrix as an ablation** (`-rtc` × hoist), but
  the **pinned default (48.6%) remains the headline**; the 59.7% configuration is
  a sensitivity result that shows the hoisting defect's recoverable cost, not a
  re-pin. Log the 6 `HOISTING_DEFECT` cases as a toolchain bug to fix later — do
  **not** hide them behind a flag.
- **D2c — moot.** §5.2 ¶2 is written; the "210/210" headline is dropped everywhere
  (abstract, intro, RQ2, conclusion).

## R-D1 — re-frame §5.6 around the detection asymmetry (D1a). **APPROVED.**

The "64×/472×" multiplier headline is **dropped** — 12 of 16 pairs are confounded
(6 `process-abort` by choreo's own ungated assertion, 4 `no-fault-reported`, 2
`launch-rejected`). The publishable claim is **coverage, not speed**:

> choreo detected **all 4 faults** that compute-sanitizer ran to completion and
> reported zero errors on (`M2.s1.ln1.bias`, `M2.s1.ln1.caller.bias`,
> `M2.s1.ln3.bias`, `M2.s1.mm1.rhs`); a further **2** never reached the oracle
> (launch refused), and **6** were killed by choreo's own ungated assertion before
> memcheck judged. The oracle's silence is not a latency advantage — it produced
> no report at all.

This **inverts** the intended "faster than the oracle" story into "catches faults
the oracle cannot see at all" — a stronger claim, and it is the one the data
supports. If a latency figure is wanted at all, use **n=4, median 84.45×**
(min 2.58, max 204.39, oracle slower 4/4) with the confound stated.

Log the `choreo.h:221` finding (a source-**A**-device site firing despite
`--disable-runtime-check`) as a toolchain bug — do not assume the flag silences
all of source A.

## R-D3 — no E5a re-measure; below-floor framing; drop the static premise. **APPROVED.**

Consistent with R-D6 (no rebuild, no post-fix binary): **do not re-measure E5a.**
- **D3b — yes**: §5.5 ¶2 states residue as **structurally zero on the device**
  (byte-identical device code, 0 check calls) and **below the measurement floor
  on the host** (14/14 deltas inside rep-to-rep spread), *not* as a percentage.
- **D3c — drop the static claim**: 0 static kernels were measured (`--size small`
  selects dynamic only), so "static shapes carry zero residue by construction" is
  untested and must be removed or marked structural-not-measured.

## R-D4 — carry S10's 0.13%, reframe as front-end vs nvcc. **APPROVED.**

The 0.6% (checks-on vs checks-off) is **unobtainable** — `-rtc=none` /
`-zero-cost` / `--disable-runtime-check` all give a byte-identical ledger. E4 ¶1
carries **`grand_median_pct = 0.1296`** and is rewritten to describe
**front-end vs nvcc compile-link**, not checks-on vs checks-off. Update the
abstract's "0.6% median compile-time overhead" accordingly (0.13%).

## R-D5 — confirm the sweep, and it is larger than listed. **APPROVED (expanded).**

Re-register from `stats.json`; the worker's table is a subset. The full stale →
measured list (all in `main.tex`/abstract):

| Claim in paper | Measured | Source |
|---|---|---|
| "210/210 before device" | **drop** | S2 |
| 17{,}717 obligations | **17{,}353** | S3 |
| 16{,}518 discharged (93.2%) | **16{,}145 (93.04%)** | S5 |
| 99.2% static (8,341/8,410) | **100.0% (8,156/8,156)** | S5 |
| 87.9% dynamic (8,177/9,307) | **86.87% (7,989/9,197)** | S5 |
| residue 1{,}199 | **1{,}208** | S5 |
| 2{,}837 interval-discharged | **2{,}920** | S6 |
| "93.2%→77.2%, residue ×3.4" | **inherited, not measured** | S7 = `not_measurable` |
| 0.6% (mean 1.9%) compile | **0.13% front-end vs nvcc** | S10 |

⚠ The S7 no-interval counterfactual is **`not_measurable`** (no flag disables
interval reasoning; all three flags are ledger-identical). The "77.2%" figure is
inherited from the prior ablation, not re-derived — label it as such in §5.4 ¶3,
or drop it. The mechanism paragraph's "2,837" → "2,920" (S6 `interval: 2920`).

## R-D7 — approve reshape/14; exclude conv2d/10; document the 36. **APPROVED.**

- **D7a — approve** the `reshape/14` fix (0 obligations, corpus-neutral).
- **D7b — exclude** `conv2d/10_dynamic` with a note (the real violation is the
  288.0 KB shared `A` tile vs ~228 KB hardware max; the filename's `112x112` is
  wrong, content is `[8,128,16,16]`). Do **not** re-derive the 17,353 ledger this
  close to deadline.
- **D7c — document, do not audit-as-a-class.** The 36 silent rejections are an
  artifact of the pre-fix binary (`f2f238f`) + `--max-local-mem-capacity`; they
  affect only the timing/measurement lanes (E4/E5), not the static E2/E3 corpus.
  Fold into the R-D6 "postdates the data" limitation note.

## On the MLIR lanes

The landed work (`mlir-shared/` harness, §5.1 toolchain/protocol pin, §4.1
M3×MLIR-low = `n/a`, §5.1 enumeration arithmetic, §7.1 per-mutant manifestation)
is **accepted as-is** — all four owner decisions are already resolved in-repo. The
`mlir-linalg`/`mlir-low` **S1/S8/S9/S12 result outputs are still pending** (only
the shared harness has landed; there is no `results/mlir-*` yet). Flag when those
lanes push so they get the same homework check.
