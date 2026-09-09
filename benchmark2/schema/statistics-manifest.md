# Statistics manifest (what `run.sh stats` must emit)

> Pinned 2026-09-08 (Phase 0); **amended 2026-09-10** per the coordinator rulings
> in `choreo/DECISIONS-NEEDED.md` (R-D1…R-D7). Each statistic is tagged with the
> paper table/number it feeds, so the integrator's mapping is mechanical. A
> worker's `stats.json` must reproduce the aggregates its lane owns; the
> integrator writes no number into `main.tex` that is not derivable from some
> worker's `stats.json`.

| # | Statistic | Aggregation | Feeds |
|---|---|---|---|
| S1 | detection matrix | per class × toolchain: `n_injected, n_compile, n_runtime, n_never, n_na` | `tab:rq2-bugs` |
| S12 | sanitizer supplement | per class × SOTA toolchain: `flagged ∧ exercised` mutants (Δ vs bare SOTA) | `tab:rq2-bugs` sanitizer column |
| S2 | choreo before-device | `Σ(n_compile + n_runtime)` per class; **every `never` attributed by cause** (the old `never = 0` criterion is retired, R-D2) | E1 detection matrix (§5.2 ¶2) — the "210/210" headline is dropped |
| S3 | generation totals | choreo obligations per class, per category, + grand total | `tab:gen-capability`, abstract "17,353" |
| S4 | per-operator ledger | per category: `expressed / discharged / runtime / budgeted / dropped(=0)` | `tab:per-operator`; `layer_norm`=79/74/5/0 |
| S5 | discharge rate | `Σ discharged / Σ generated`, **split static-shape vs dynamic-shape** | RQ1; "93.04%", "100.0%/86.87%" |
| S6 | mechanism split | per outcome: `canonical / interval / direct` counts | `tab:rq6-mechanism` |
| S7 | no-interval counterfactual | **`not_measurable`** on the pinned build (no flag disables interval reasoning); 93.2%→77.2% is **inherited**, not re-derived (R-D5) | "93.2%→77.2%" (inherited) |
| S8 | SOTA expressibility | per obligation class × toolchain: counts | `tab:gen-capability` |
| S9 | SOTA remainder | per toolchain: `Σ unconditional_guards` per category | `tab:per-operator` (SOTA cols) |
| S10 | compile cost | median **front-end-vs-nvcc compile-link** % (choreo) — not checks-on/off (R-D4) | RQ4 |
| S13 | runtime cost & latency | E5a: Δ(residue on/off) per dynamic-shape case; E5b: **detection asymmetry (coverage), not latency ratio** (R-D1) | RQ3 + E5 |
| S11 | integrity register | 310-kernel corpus 3-way split (compile / launch / never) re-derived from raw CSV — owner ruled authoritative 2026-09-08 | Phase-0 gate |

## S8 exact shape (pinned 2026-09-09)

`S8_expressibility` is keyed by the four obligation classes from
`record-schema.json` (`elem`, `shape`, `loop`, `hw`) and must emit, for each
class, the counts `{"yes": n, "partial": n, "no": n}` over the 15 categories.
Example (a surface that only expresses element-access):

```json
{"elem": {"yes": 15, "partial": 0, "no": 0},
 "shape": {"yes": 0, "partial": 0, "no": 15},
 "loop": {"yes": 0, "partial": 0, "no": 15},
 "hw": {"yes": 0, "partial": 0, "no": 15}}
```

## Non-negotiable cross-cutting fields

Every record, every worker: `settings_hash` + `kernel_hash` + toolchain version
from `manifest.md`. Without these a number cannot be traced to a build and is
treated as unverified by the `qc` worker.

## Lane ownership

| Worker | Statistics it owns |
|---|---|
| `choreo` | S1–S7, S10, S13 |
| `triton` | S1, S8, S9, S12 |
| `mlir-linalg` | S1, S8, S9, S12 |
| `mlir-low` | S1, S8, S9, S12 |
| `iree` | S1, S8, S9, S12 |
| `tilelang` (exploratory) | S1, S8, S9, S12 (disclosure decided at integration) |
| `integrator` | merges all `stats.json` → paper tables |
| `coordinator` | S11 (integrity register) |
