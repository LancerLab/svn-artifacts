#!/usr/bin/env python3
"""Mutant validation: does each M2 defect actually manifest?

Per the requirement that mutants must *actually work*, a mutant is only usable if
it fails to compile, aborts at runtime, or runs clean but produces a provably
wrong output (`corrupts`). A mutant that runs clean and still matches the
reference is a `noop` false success and must be discarded (specs §7.1).

The reference checksums are always taken from the **unmutated** case, so a mutant
that silently compiles to something equivalent is caught rather than counted.

Every category is run in all four combinations of static/dynamic shape and
RTV-off/RTV-on, because manifest §5.1 requires the two RTV modes to be reported
as separate rows and §5.2 requires the dynamic pattern to be preserved.

TWO DIFFERENT COUNTS, deliberately kept apart
---------------------------------------------
`mutation-specs.md` §0 fixes N = 40 per class at **spec** level, while §2 numbers
M2 as five specs of which spec 5 names two manifestations ("partial write /
duplicate write"). So:

* an **injection** is one (category, shape, spec) triple -- this is what S1's
  `n_injected` reports, and spec 5 counts once;
* a **record** is one (category, shape, mutant, rtv-mode) tuple -- spec 5's two
  variants are two records, i.e. two measurements of one injection.

The census at the end of the run prints both, plus the `n/a` cells, which §6
requires be counted and never re-balanced.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose as C
import emit as E
import mlirbench as B
import mutate as M

# mutation-specs.md §5 splits the M2 coverage set into a level-1 minimal set and
# level-2 additions. The split lives in compose.py so this validator and the
# eventual record writer cannot drift apart; the schema enum for `level` is
# exactly {"1","2"}.
LEVEL1_M2_CATS = C.LEVEL1_M2_CATS
LEVEL2_M2_CATS = C.LEVEL2_M2_CATS
M2_CATS = C.M2_CATS
LEVEL_OF = C.LEVEL_OF

# §0's per-class injection target, used only to report the delta.
N_TARGET = C.N_TARGET_PER_CLASS

# The MLIR diagnostic is the interesting part of a compile error; the leading
# `path:line:col:` prefix is noise in a summary table.
_LOC = re.compile(r"^.*?\.mlir:\d+:\d+:\s*", re.MULTILINE)


def _detail(c: B.Classification) -> str:
    raw = c.abort_message or c.compile_error or c.notes or ""
    raw = _LOC.sub("", raw)
    return " ".join(raw.split())[:60]


def main() -> int:
    print(B.describe_toolchain())
    hdr = (f"{'category':22s} {'lv':2s} {'shape':7s} {'mutant':14s} {'rtv':4s} "
           f"{'outcome':8s} {'manifest':9s} detail")
    print(hdr)
    print("-" * len(hdr))

    counts: dict[tuple[str, str], int] = {}
    problems: list[str] = []

    # injection key -> list of per-variant outcomes, so an injection is `n/a`
    # only when *every* variant of that spec is inexpressible.
    inj_outcomes: dict[tuple[str, str, str], list[str]] = {}
    n_records = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for cat in M2_CATS:
            lv = LEVEL_OF[cat]
            for dynamic in (False, True):
                shape = "dyn" if dynamic else "static"
                clean = C.make_case(cat, size="small", dynamic=dynamic)
                for mut in C.M2_SPECS:
                    mid = mut.mutant_id
                    ikey = (cat, shape, mut.spec_id)

                    # Apply + emit ONCE per mutant; only classification depends on
                    # the RTV mode.
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
                        # A mutation that changes nothing would be recorded as a
                        # `noop` false success and silently shrink N (specs §7.1).
                        problems.append(f"NOOP mutation {mid} on {cat}/{shape}: {exc}")
                        for rtv in (False, True):
                            print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                                  f"{'on' if rtv else 'off':4s} "
                                  f"{'NOOP-BUG':8s} {'-':9s} {exc}")
                        continue

                    src = E.emit_kernel(mcase, ref_case=clean, structural=structural)
                    f = tmp / f"{cat}_{shape}_{mid}.mlir"
                    f.write_text(src)

                    for rtv in (False, True):
                        c = B.classify(f, tmp, "linalg", rtv)
                        if c.manifest == "noop":
                            problems.append(
                                f"NOOP {mid} on {cat}/{shape} (rtv="
                                f"{'on' if rtv else 'off'}): compiles, runs, and "
                                f"still matches the reference -- not a defect"
                            )
                        print(f"{cat:22s} {lv:2s} {shape:7s} {mid:14s} "
                              f"{'on' if rtv else 'off':4s} {c.outcome:8s} "
                              f"{c.manifest or '-':9s} {_detail(c)}")
                        key = (c.outcome, c.manifest or "-")
                        counts[key] = counts.get(key, 0) + 1
                        n_records += 1
                        inj_outcomes.setdefault(ikey, []).append(c.outcome)

    print()
    print("=== record tally (one per mutant x RTV mode) ===")
    for (outcome, manifest), n in sorted(counts.items()):
        print(f"  {outcome:8s} {manifest:9s} {n:4d}")
    print(f"  {'TOTAL':18s} {sum(counts.values()):4d}")

    # ---------------- injection census (spec level, feeds S1 n_injected) -------
    print()
    print("=== M2 injection census (spec level; mutation-specs.md §0) ===")
    for lv, cats in (("1", LEVEL1_M2_CATS), ("2", LEVEL2_M2_CATS)):
        n = sum(1 for (c, _s, _sp) in inj_outcomes if LEVEL_OF[c] == lv)
        print(f"  level-{lv} ({', '.join(cats)}): {n} injections")
    n_inj = len(inj_outcomes)
    n_na = sum(1 for outs in inj_outcomes.values() if all(o == "n/a" for o in outs))
    print(f"  TOTAL injections      : {n_inj}")
    print(f"    n/a cells (§6: counted, never re-balanced): {n_na}")
    print(f"    applicable                                : {n_inj - n_na}")
    print(f"  §0 target N per class : {N_TARGET}")
    delta = n_inj - N_TARGET
    if delta == 0:
        print("  delta                 : 0 (lands exactly on N)")
    else:
        n_lv1 = sum(1 for (c, _s, _sp) in inj_outcomes if LEVEL_OF[c] == "1")
        n_lv2 = n_inj - n_lv1
        print(f"  delta                 : {delta:+d}  <-- OVERSHOOT, must be "
              f"recorded explicitly, NOT silently trimmed")
        print(f"                          level-1 alone is {n_lv1} (under target); "
              f"level-2 adds {n_lv2}.")
        print("                          Owner approved following §5's level-2 list "
              "literally, knowing it breaks the exact-40 alignment. See "
              "mlir-shared/README.md.")
    print(f"  records emitted       : {n_records} "
          f"({n_inj} injections; spec 5 emits 2 variants, x2 RTV modes)")

    if problems:
        print()
        print(f"=== {len(problems)} PROBLEM(S) ===")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print()
    print("all mutants either detected or provably corrupt; no noop false successes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
