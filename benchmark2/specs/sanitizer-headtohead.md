# Spec: sanitizer head-to-head on choreo's runtime subset (E6-RT)

> Commissioned 2026-09-24 (owner decision). Paper-side context: E6 is being
> retitled from "marginal value of external tooling" to a runtime-vs-runtime
> head-to-head — same defect instances, choreo's budgeted checks vs
> compute-sanitizer (memcheck), four axes: detection rate, time to report,
> cost, diagnostic level. The artifact already carries axes 2–4 (S13.E5b
> paired timing; E5a checks-on/off cost). This run supplies the missing
> axis-1 datum: memcheck's **standalone** per-family detection rate over
> choreo's runtime subset.

## Population

- Source: `benchmark2/results/choreo/mutant.jsonl` — records whose outcome is
  a runtime outcome (launch-refused or during-execution detection) **or**
  `never`. I.e., every injected, oracle-confirmed mutant that choreo did NOT
  refute at compile time. Snapshot at commissioning: 48 runtime + 43 never
  = 91 records; use the settled manifest, the corpus is still drifting.
- Exclusions follow the existing convention: no-op controls and undecidable
  bases are out (already excluded from the S1 denominators).
- Group results by class (M1–M4).

## Arms (paired per record, same inputs as the lane's own run)

1. **sanitizer arm**: run the mutant executable under
   `compute-sanitizer --tool memcheck` with choreo's OWN checks gated off
   (no choreo_assert / assertion traps). Rationale: S13.E5b lost 12 of 16
   pairs because the sanitizer-arm process died on choreo's own ungated
   assertions before memcheck could observe anything — a standalone sanitizer
   rate requires the oracle to get its own chance. Record: memcheck error
   reported (yes/no), first error type, ERROR SUMMARY count, wall clock,
   time-to-first-report if reported, process exit status.
2. **plain arm**: same executable, no sanitizer, checks gated off — wall
   clock only. Skip records already covered by an E5b plain arm; reuse those.

## Honest-exclusion discipline (same as E5b)

Any pair where the sanitizer arm cannot produce a verdict for reasons
unrelated to the mutant (harness abort, launcher failure, etc.) gets a
per-pair note and is excluded from rates, with counts by reason. Do not
average confounded arms into anything.

## Deliverables

- `benchmark2/results/choreo/sanitizer.jsonl`: one row per pair with the
  fields above.
- `stats.json`: new block (suggest `S12_sanitizer_headtohead`, following the
  existing S-block naming) with, per class and overall:
  `{n_runtime_subset, memcheck_flags, flag_rate, median_instrumented_wall_us,
  median_plain_wall_us, median_slowdown_ratio, n_time_to_report,
  median_time_to_report_us, n_excluded, exclusion_reasons}`.
- Note in the block how many of the `never` records memcheck flagged (this
  reproduces the E6 never-residue delta as a byproduct — expected ~0) and how
  many of the runtime-detected records it flagged (the contested ground where
  the other three axes carry the claim).

## Coordination

- Do not clobber the in-flight post-repair rerun; sequence after it or run
  against its settled manifest.
- The paper-side table generator for the retitled E6 will read this block;
  ping when landed so the E1/E5/E6 prose pass can run once over final numbers.
