# benchmark2 — worker entry point

> Phase-0 shared inputs. **Read in this order before writing any code.**

1. [`manifest.md`](manifest.md) — pinned binary + flags, machine precondition,
   toolchain versions, device plan, and the §7 data-integrity gate (BLOCKING).
2. [`specs/mutation-specs.md`](specs/mutation-specs.md) — the M1/M2/M3 mutation
   specs, the per-surface translation table, the minimal coverage set, and the
   outcome taxonomy. **N = 40 per class → 160 total.**
3. [`settings/<category>.md`](settings/) — per-category operator contracts
   (shapes, reference semantics, provenance). This is the **only** source of
   truth for shapes/contracts; you compose your own kernel from it (§2.2).
4. [`schema/record-schema.json`](schema/record-schema.json) — the raw-record
   schema your `collect` must emit.
5. [`schema/statistics-manifest.md`](schema/statistics-manifest.md) — the S1–S13
   statistics your `stats` must reproduce (per-lane ownership).
6. [`run.sh.template`](run.sh.template) — copy to `benchmark2/<toolchain>/run.sh`
   and fill in the `run_*` bodies.

## Directory layout

```
benchmark2/
  settings/     # shared operator settings (the ONLY thing handed to workers)
  specs/        # M1/M2/M3 mutation specs + translation table (coordinator pins)
  schema/       # record-schema.json + statistics-manifest.md
  manifest.md   # pinned binary/flags, versions, GPU topology, device plan
  run.sh.template
  derive_settings.py   # reproducible settings extraction (provenance)
  results/      # per-toolchain JSON: 3-way outcomes + gen-capability + remainder
```

## Onboarding rule (every worker, every time)

Before implementing anything: (a) read this whole `benchmark2/` tree and the
experiment-redesign plan, (b) inspect feasibility of your own card — hardest step
first, (c) list your own-lane risks in priority order. **Ask questions first** —
never silently improvise, guess a mapping, or substitute an operator.

## Definition of done

`run.sh all` completes; all 15 categories compose + gate (small-input); every
assigned mutant is classified and schema-validated; `stats.json` reproduces your
lane's statistics (§12.4) and every number traces to raw JSON.
