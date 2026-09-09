#!/usr/bin/env python3
"""M1 mutant validation for the `mlir-low` lane: does each defect manifest?

Mirrors `_validate_mutants.py` (which covers M2 on the linalg surface) but drives
`emit_low.py` over the memref/affine surface. Per `mutation-specs.md` §4 the
MLIR-low translation of M1 is "`affine`/`memref` index out of range", and §7 makes
the ground-truth oracle mandatory: a mutant that compiles, runs, and still matches
the reference is a `noop` false success and must be discarded (§7.1), not counted
as a detection.

The reference checksums are always taken from the **unmutated** case, so a mutant
that silently compiles to something equivalent is caught rather than counted.

Every category is run in all four combinations of static/dynamic shape and
RTV-off/RTV-on, because manifest §5.1 requires the two RTV modes to be reported as
separate rows and §5.2 requires the dynamic pattern to be preserved.

THE MANIFESTATION CHECK IS PER MUTANT, NOT PER RECORD
----------------------------------------------------
RTV-off and RTV-on are two *measurements of the same injected defect*, so §7.1's
question -- "did this mutant actually corrupt anything?" -- is answered by
aggregating across both modes. A mutant is a false success only when it is `noop`
under **both**. When exactly one mode reports `noop` the defect is real but
mode-dependent, which is a *finding* rather than a harness bug, and is listed
separately rather than failing the run. Owner ruling 2026-09-09 (specs §7.1):
record it as a finding; do NOT redesign the kernel to force output-observability.

The measured instance is relu's dropped boundary mask (M1.1). The tiled loop runs
one iteration past the extent, so the guardless read and write both address
`inp[i, extent]` / `out[i, extent]`. In row-major layout that wraps to
`[i+1, 0]` -- and because relu is elementwise, `max(x[i+1,0], 0)` is exactly the
value the clean kernel writes there, so *when the write lands idempotently* the
output is bit-identical and no output-comparison oracle at any tolerance can see
it. RTV's bounds check still does.

BUT THIS IS UNDEFINED BEHAVIOR, NOT A DETERMINISTIC NOOP. The out-of-bounds write
can instead corrupt the heap (-> `abort`) or wrap an index so a loop bound is
never reached (-> `hang`), and which happens depends on heap layout, varying run
to run and shape to shape. Measured on small-size RTV-off over 30 runs: relu/dyn
M1.1 is a stable noop (30/30), while relu/static M1.1 is a coin flip at
p(noop) = 0.633 (19/30 noop, 11/30 heap-corruption abort). RTV-on is
deterministic at both shapes (30/30 runtime/corrupts). A single `classify()`
therefore *samples the UB once* and any finding derived from it is not
reproducible. This validator runs each (mutant, mode) `N_REPEAT` times via
`mlirbench.classify_repeat` and reports the measured distribution, so the
characterization is honest and stable across invocations. This is precisely the
contrast manifest §5.1 requires both modes be reported for.

INJECTION vs RECORD
-------------------
M1's six specs (§1) have no sub-variants, so here one injection is exactly one
record per (category, shape, spec) -- unlike M2, where spec 5 emits two variants
that count as one injection. The census still reports both numbers so the two
lanes stay directly comparable.

`n/a` cells (§6) are counted and never re-balanced. On this surface they arise
from `emit_low.NotExpressible` -- e.g. `transposed-stride` needs two axes to swap,
so a rank-1 trailing axis cannot express it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose as C
import emit_low as L
import mlirbench as B
import mutate as M

# mutation-specs.md §5's M1 minimal set is exactly the four categories this lane
# composes, so every M1 record here is level-1. The §5 level-2 M1 additions
# (max_pool2d, conv2d, embedding, batch_norm) have no low-level emitter yet.
LEVEL1_M1_CATS = L.LOW_CATS
LEVEL_OF_M1 = {c: "1" for c in L.LOW_CATS}

N_TARGET = C.N_TARGET_PER_CLASS

# M1.1's dropped-boundary-mask defect injects undefined behavior (an OOB write),
# so a single run samples one of {noop, abort, hang} non-reproducibly. Each
# (mutant, mode) is therefore run N_REPEAT times and the verdict reduced over the
# distribution (mlirbench.classify_repeat / reduce_verdicts).
#
# N_REPEAT is sized against the ONE cell whose recorded outcome can realistically
# flip, relu/static M1.1 at RTV-off: p(noop) = 0.532 pooled over 94 runs across
# five independent draws (30-run study 19/30, committed artifact 12/16, two
# validator runs 7/16 and 5/16, one fresh `run.sh all` 7/16).  Its tally cell
# reads never/noop only if EVERY run lands noop, i.e. with probability p^N:
#     N=5 -> 4.3e-2    N=8 -> 6.4e-3    N=12 -> 5.1e-4    N=16 -> 4.1e-5
# The 95% interval on that p is [0.432, 0.630], so the N=16 risk is honestly
# ~1.5e-6..6.1e-4 -- quote the interval, not the point estimate.  Three other
# cells (transpose/static + transpose/dynamic M1.1, layer_norm/dynamic M1.2, all
# RTV-off) have also produced a mixed distribution at least once, but at
# p(noop)<=1/16 their flip probability is <=1e-19, so they are nondeterministic
# in the strict sense and stable in every practical sense.  NOTE: the
# "MODE-DEPENDENT DEFECT(S)" this validator reports are a DIFFERENT set -- cells
# where RTV-off and RTV-on disagree -- and must not be conflated with the
# mixed-distribution cells.
# The per-run timeout is short because a
# correct kernel finishes in milliseconds, so a hang is a property of the defect,
# not of the workload.  Both are env-overridable (M1_REPEAT / M1_RUN_TIMEOUT) so a
# quick check or a deeper characterization can be run without editing the file.
# The *gate* (exit code) is reproducible at any N: a mutant is a false success
# only when every run of every mode is noop, and RTV-on always corrupts.
N_REPEAT = int(os.environ.get("M1_REPEAT", "16"))
RUN_TIMEOUT = int(os.environ.get("M1_RUN_TIMEOUT", "15"))


def _dist_str(census: dict) -> str:
    """Format a per-mode run distribution as a compact, stable string.

    Derived from the *measured* census rather than hardcoded prose, so the finding
    text reflects what actually happened over N_REPEAT runs (e.g. an M1.1 that
    splits noop/abort/hang) instead of asserting an idempotent-write story that is
    false for the static shape. Sorted for reproducibility.
    """
    parts = [f"{n}x {outcome}/{manifest}" for (outcome, manifest), n
             in sorted(census.items())]
    return ", ".join(parts)


def main() -> int:
    print(B.describe_toolchain())
    hdr = (f"{'category':22s} {'lv':2s} {'shape':7s} {'mutant':14s} {'rtv':4s} "
           f"{'outcome':8s} {'manifest':9s} distribution ({N_REPEAT} runs)")
    print(hdr)
    print("-" * len(hdr))

    counts: dict[tuple[str, str], int] = {}
    problems: list[str] = []
    # Real defects that only one RTV mode can see. Reported, not failed.
    notes: list[str] = []

    # injection key -> list of per-variant outcomes. M1 has one variant per spec,
    # so the list is always length 1; kept as a list so the `n/a` rule (an
    # injection is `n/a` only when *every* variant is) reads the same as in the M2
    # validator and cannot silently diverge.
    inj_outcomes: dict[tuple[str, str, str], list[str]] = {}
    n_records = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for cat in L.LOW_CATS:
            lv = LEVEL_OF_M1[cat]
            for dynamic in (False, True):
                shape = "dyn" if dynamic else "static"
                clean = C.make_case(cat, size="small", dynamic=dynamic)
                for mut in C.M1_SPECS:
                    mid = mut.mutant_id
                    ikey = (cat, shape, mut.spec_id)

                    try:
                        mcase, structural = M.apply(clean, mut)
                    except M.NotApplicable as exc:
                        for rtv in (False, True):
                            print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                                  f"{'on' if rtv else 'off':4s} "
                                  f"{'n/a':8s} {'-':9s} {exc}")
                            counts[("n/a", "-")] = counts.get(("n/a", "-"), 0) + 1
                            n_records += 1
                        inj_outcomes.setdefault(ikey, []).append("n/a")
                        continue
                    except AssertionError as exc:
                        problems.append(f"NOOP mutation {mid} on {cat}/{shape}: {exc}")
                        for rtv in (False, True):
                            print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                                  f"{'on' if rtv else 'off':4s} "
                                  f"{'NOOP-BUG':8s} {'-':9s} {exc}")
                        continue

                    # NotExpressible is raised by the emitter, not by apply(): the
                    # Case perturbation is well-formed, but this surface cannot
                    # realize the defect on this category's rank.
                    try:
                        src = L.emit_kernel_low(
                            mcase, ref_case=clean, structural=structural
                        )
                    except L.NotExpressible as exc:
                        for rtv in (False, True):
                            print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                                  f"{'on' if rtv else 'off':4s} "
                                  f"{'n/a':8s} {'-':9s} {exc}")
                            counts[("n/a", "-")] = counts.get(("n/a", "-"), 0) + 1
                            n_records += 1
                        inj_outcomes.setdefault(ikey, []).append("n/a")
                        continue

                    f = tmp / f"{cat}_{shape}_{mid}.mlir"
                    f.write_text(src)

                    # Compile once, run N_REPEAT times per mode, reduce over the
                    # distribution. M1.1 is UB, so a single sample is not
                    # reproducible; the canonical verdict and the census both come
                    # from mlirbench so the lanes cannot drift.
                    canon_manifests: list[str] = []
                    censuses: list[dict] = []
                    all_noop_every_run = True
                    for rtv in (False, True):
                        verdicts = B.classify_repeat(
                            f, tmp, "low", rtv, n=N_REPEAT, run_timeout=RUN_TIMEOUT
                        )
                        canonical, census = B.reduce_verdicts(verdicts)
                        canon_manifests.append(canonical.manifest or "-")
                        censuses.append(census)
                        if any(v.manifest != "noop" for v in verdicts):
                            all_noop_every_run = False
                        dist = _dist_str(census)
                        nondet = " [nondeterministic]" if len(census) > 1 else ""
                        print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                              f"{'on' if rtv else 'off':4s} {canonical.outcome:8s} "
                              f"{canonical.manifest or '-':9s} {dist}{nondet}")
                        # One record per (mutant, mode): the canonical verdict.
                        key = (canonical.outcome, canonical.manifest or "-")
                        counts[key] = counts.get(key, 0) + 1
                        n_records += 1
                        inj_outcomes.setdefault(ikey, []).append(canonical.outcome)

                    # §7.1's manifestation check is per *mutant*, not per record,
                    # and under UB not per *run*. RTV-off and RTV-on are two
                    # measurements of the same injected defect (manifest §5.1
                    # requires both be reported separately and never merged). The
                    # mutant is a false success only when *every run of every mode*
                    # is noop -- i.e. the defect never corrupted anything anywhere.
                    if all_noop_every_run:
                        problems.append(
                            f"NOOP {mid} on {cat}/{shape}: noop in every run of "
                            f"both RTV modes ({N_REPEAT} runs each) -- compiles, "
                            f"runs, and still matches the reference, so it is not a "
                            f"defect (§7.1)"
                        )
                    elif "noop" in canon_manifests or any(len(c) > 1 for c in censuses):
                        # A real defect whose visibility is mode-dependent and/or
                        # nondeterministic under UB. This is a finding, not a
                        # harness bug. The text is derived from the measured
                        # distributions rather than a hardcoded idempotent-write
                        # story, which is false for relu/static M1.1.
                        notes.append(
                            f"{mid} on {cat}/{shape}: RTV-off [{_dist_str(censuses[0])}]"
                            f", RTV-on [{_dist_str(censuses[1])}]"
                            f" ({N_REPEAT} runs/mode)"
                        )

    print()
    print("=== record tally (one per mutant x RTV mode) ===")
    for (outcome, manifest), n in sorted(counts.items()):
        print(f"  {outcome:8s} {manifest:9s} {n:4d}")
    print(f"  {'TOTAL':18s} {sum(counts.values()):4d}")

    print()
    print("=== M1 injection census (spec level; mutation-specs.md §0) ===")
    n_inj = len(inj_outcomes)
    n_na = sum(1 for outs in inj_outcomes.values() if all(o == "n/a" for o in outs))
    print(f"  level-1 ({', '.join(LEVEL1_M1_CATS)}): {n_inj} injections")
    print(f"  TOTAL injections      : {n_inj}")
    print(f"    n/a cells (§6: counted, never re-balanced): {n_na}")
    print(f"    applicable                                : {n_inj - n_na}")
    print(f"  §0 target N per class : {N_TARGET}")
    delta = n_inj - N_TARGET
    if delta == 0:
        print("  delta                 : 0 (lands exactly on N)")
    else:
        print(f"  delta                 : {delta:+d}")
    print(f"  records emitted       : {n_records}")

    if notes:
        print()
        print(f"=== {len(notes)} MODE-DEPENDENT DEFECT(S) (findings, not failures) ===")
        for n in notes:
            print(f"  - {n}")

    if problems:
        print()
        print(f"=== {len(problems)} PROBLEM(S) ===")
        for p in problems:
            print(f"  - {p}")
        return 1

    print()
    print("=== OK: no noop false successes ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
