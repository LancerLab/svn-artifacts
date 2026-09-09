#!/usr/bin/env python3
"""Check that RTV actually instruments the clean kernels.

Manifest §5.1 requires RTV-off and RTV-on to be reported as separate rows. That
is only meaningful if RTV genuinely inserts checks. This measures, per category:

* total `cf.assert` at MLIR level (the stage where the op still exists),
* the split between guards on the **kernel under audit** (`loc("kernel")`) and
  guards on this harness's **checksum oracle** (`loc("oracle")`),
* unique `assert_msg_N` globals after full LLVM lowering,
* the decoded messages, to distinguish bounds checks from anything else.

The split matters. RTV instruments the oracle's loads and stores too, because
they are real memref accesses, but those guards say nothing about the operator
being audited. S9 (remainder = Σ unconditional_guards) must count kernel guards
only; folding the oracle's in inflates the remainder by roughly a third on relu.

Both surfaces are walked, because `statistics-manifest.md` gives `mlir-linalg`
and `mlir-low` each their own S9. Their numbers are reported separately and never
summed (mutation-specs.md §7): the low lane's kernels are hand-tiled with an
explicit boundary guard, so their guard structure is not comparable to the
linalg lane's.

These numbers feed S9 and S12.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose as C
import emit as E
import emit_low as L
import mlirbench as B

CATS = ["matmul", "relu", "softmax", "transpose", "concat", "layer_normalization", "elemwise_add"]

# (surface, categories, emitter). Both lanes own S9 (statistics-manifest.md), so
# the kernel/oracle guard split has to be measured on each surface separately --
# the low lane's kernels are hand-tiled with an explicit boundary guard, so their
# guard counts are structurally different from the linalg lane's and cannot be
# inferred from them.
SURFACES = [
    ("linalg", CATS, E.emit_kernel),
    ("low", L.LOW_CATS, L.emit_kernel_low),
]


def main() -> int:
    print(B.describe_toolchain())
    hdr = (f"{'surface':8s} {'category':22s} {'shape':7s} {'kernel':>7s} "
           f"{'oracle':>7s} {'total':>6s} {'lowered':>8s} {'run':>5s}")
    print(hdr)
    print("-" * len(hdr))

    msgs: Counter[str] = Counter()
    loc_msgs: Counter[str] = Counter()
    kernel_by_surface: dict[str, int] = {}
    rows_by_surface: dict[str, int] = {}
    problems: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for surface, cats, emit_fn in SURFACES:
            for cat in cats:
                for dynamic in (False, True):
                    shape = "dyn" if dynamic else "static"
                    case = C.make_case(cat, size="small", dynamic=dynamic)
                    work = tmp / f"{surface}_{cat}_{shape}"
                    work.mkdir(parents=True, exist_ok=True)
                    mlir = work / "kernel.mlir"
                    mlir.write_text(emit_fn(case))

                    # RTV-only pipeline: count cf.assert, attributed by location.
                    rtv_ir = work / "rtv.mlir"
                    r = B.run_mlir_opt(B.RTV_ONLY[surface], mlir, rtv_ir)
                    if not r.ok:
                        problems.append(
                            f"{surface}/{cat}/{shape}: RTV pipeline failed"
                        )
                        print(f"{surface:8s} {cat:22s} {shape:7s} RTV-PIPELINE-FAIL")
                        continue
                    by_loc = B.count_asserts_by_loc(rtv_ir)
                    n_kernel, n_total = B.count_kernel_asserts(rtv_ir)
                    n_oracle = by_loc.get("oracle", 0)
                    if "<untagged>" in by_loc:
                        problems.append(
                            f"{surface}/{cat}/{shape}: {by_loc['<untagged>']} "
                            "untagged assert(s) -- the emitter lost a loc tag"
                        )

                    # Full RTV-on pipeline: count lowered assert_msg globals.
                    ll = work / "kernel.ll.mlir"
                    f = B.run_mlir_opt(B.pipeline(surface, True), mlir, ll)
                    if not f.ok:
                        problems.append(f"{surface}/{cat}/{shape}: RTV-on lowering failed")
                        print(f"{surface:8s} {cat:22s} {shape:7s} {n_kernel:7d} "
                              f"{n_oracle:7d} {n_total:6d} LOWER-FAIL")
                        continue
                    n_low = B.count_asserts(ll, lowered=True)
                    # Count every decoded message, not just the unique ones: CSE
                    # collapses identical payloads into one global, so deduplicating
                    # here would under-count the guards that actually execute.
                    for m in B.assert_messages(ll):
                        msgs[m] += 1
                        lm = B._LOC_TAG.search(m)
                        loc_msgs[lm.group(1) if lm else "<untagged>"] += 1

                    # Confirm the RTV-on kernel still runs clean.
                    run = B.run_kernel(ll)
                    ret = B._parse_return(run.stdout)
                    ok = run.rc == 0 and ret == 0
                    if not ok:
                        problems.append(
                            f"{surface}/{cat}/{shape}: RTV-on run rc={run.rc} ret={ret}"
                        )
                    kernel_by_surface[surface] = (
                        kernel_by_surface.get(surface, 0) + n_kernel
                    )
                    rows_by_surface[surface] = rows_by_surface.get(surface, 0) + 1
                    print(f"{surface:8s} {cat:22s} {shape:7s} {n_kernel:7d} "
                          f"{n_oracle:7d} {n_total:6d} {n_low:8d} "
                          f"{'ok' if ok else 'FAIL':>5s}")

    print("-" * len(hdr))
    for surface, _, _ in SURFACES:
        print(f"Σ kernel guards, {surface:8s} ({rows_by_surface.get(surface, 0)} "
              f"clean kernels): {kernel_by_surface.get(surface, 0)}")
    print(f"Σ kernel guards, all surfaces: {sum(kernel_by_surface.values())}")
    print()
    print("NOTE: the two lanes' S9 remainders are reported separately and never "
          "summed (mutation-specs.md §7).")
    print()
    print("assert messages, aggregated by diagnostic class:")
    if not msgs:
        print("  (none decoded -- assert_messages() is broken)")
    # Keying on the raw message yields one bucket per assert, because each embeds
    # its own SSA names (%37, %12, %arg0) and dimension numbers. Aggregate on the
    # diagnostic line with numbers normalized: that is what distinguishes a
    # *bounds* guard from an *alignment* or *shape* guard, which is the analysis
    # that settled the M3 x mlir-low n/a ruling.
    classes: Counter[str] = Counter()
    for m, n in msgs.items():
        diag = next(
            (ln.strip() for ln in m.splitlines() if ln.strip().startswith("^")),
            "",
        )
        if not diag:
            classes["(no diagnostic line)"] += n
            continue
        classes[re.sub(r"\d+", "N", diag)] += n
    for k, n in classes.most_common():
        print(f"  {n:3d}x  {k}")

    print()
    print("assert messages, by location:")
    for k, n in loc_msgs.most_common():
        print(f"  {n:3d}x  loc({k})")
    print()
    print(f"problems ({len(problems)}):")
    for p in problems:
        print(f"  !! {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
