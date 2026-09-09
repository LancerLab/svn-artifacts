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

- **`never = 0` acceptance criterion FAILS** — 37 of 72 injected mutants are
  never-detected. §5.2 ¶2 prose is **VOID** until D2 is decided.
- **E5b's "sole detector" premise is FALSE** — `--disable-runtime-check` gates
  only assertion source A; sources B and C are ungated by construction. See D1.
- **E5a measured 0 static kernels** (`--size small` selects dynamic only), so
  §5.5 ¶2's premise that static shapes carry zero residue is untested here.
  `--include-static` exists.
- **`toolchain.py`'s mtime heuristic understates** the binary's source commit
  under rebuild-then-commit. `binary_sha1_12` is the only field that pins which
  build ran.
