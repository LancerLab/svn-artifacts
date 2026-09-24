#!/usr/bin/env python3
"""collect.py — fold lane.py records into results/stats.json (spec §8 shape).

This lane is an sm_86 vertical slice: the M1/M2/M3/M4 launched batteries and the
L path-class control have been run, so `complete` reflects only the pending
sm_120 re-run. It is `ready` in the class axis with M1/M2/M3/M4 all `measured`.

Input records are the v2.1 shape written by mutrec.py: `outcome` is
`compile|runtime|never|n/a`, `class` is the mutation class (M1/M2/M3/M4) and
`path_class` says how the defect was met. `L` is a path class, not a mutation
class, so those records are routed to `S1_path_class`, never `S1_detection`.
"""

import json
import os
import sys

SPEC_VERSION = "v2.1"

# The mutation-class axis is the ONE definition (`schema/class-axis.json`), not a
# literal here. `axis.no-restated-tuples` forbids restating the class sequence
# outside `schema/`; the uncompared declaration is read from the axis so it
# follows the lane's status instead of drifting from it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmark2/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from schema import class_axis as AX                              # noqa: E402

LANE = "cutlass"

# A class the lane has never been run against is `uncompared` in the axis: it
# must be absent from S1 and declared here, NOT reported as `n/a`. The
# distinction is load-bearing (`check_class_axis` g6). All four classes are now
# measured, so this is empty and follows the axis instead of drifting from it.
UNCOMPARED_CLASSES = AX.uncompared_classes(LANE)


def main(records_path, out_dir):
    recs = []
    paths = [records_path]
    base = os.path.dirname(os.path.abspath(records_path))
    # M1 and M3 are now LAUNCHED batteries in records.jsonl (real operators x
    # extents), so the old compile-only `records_m1.jsonl`/`records_m3.jsonl`
    # probe slices are retired and not merged (kept on disk for provenance).
    for extra in ():
        p = os.path.join(base, extra)
        if p not in paths:
            paths.append(p)
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                recs += [json.loads(l) for l in f if l.strip()]

    # `n_na` is the canonical S1 key (`schema/statistics-manifest.md`, and the
    # key set `render.py::_norm_s1` reads); on this lane it counts the
    # instance-level inapplicable rows (outcome `n/a`, i.e. measured `avoid`: no
    # legal program states the defect). Those rows are excluded from detection
    # and enter no denominator. It is deliberately NOT spelled
    # `n_discarded_noop`: that name is choreo's S14/S2 "inert no-op" concept,
    # which is a different thing and is not an S1 cell field.
    def _cell():
        return dict(n_injected=0, n_compile=0, n_runtime=0, n_never=0, n_na=0)
    classes = {c: _cell() for c in AX.mutation_classes()}
    # `L` is a path class, not a mutation class: it must not appear in
    # `S1_detection` (which is keyed by M1..M4 only). It is reported separately.
    path_classes = {"L": _cell()}
    for r in recs:
        if r.get("path_class") == "L":
            bucket = path_classes["L"]
        else:
            bucket = classes.setdefault(r["class"], _cell())
        bucket["n_injected"] += 1
        o = r["outcome"]
        if o == "compile":
            bucket["n_compile"] += 1
        elif o == "runtime":
            bucket["n_runtime"] += 1
        elif o == "never":
            bucket["n_never"] += 1
        elif o == "n/a":
            bucket["n_na"] += 1

    stats = {
        "toolchain": "cutlass",
        "spec_version": SPEC_VERSION,
        "lane_phase": "vertical-slice",
        "complete": False,
        "host": {"gpu": "RTX 3070 8GB", "arch": "sm_86", "cuda": "12.9",
                 "cutlass": "v4.2.1",
                 "note": "dev host; manifest.md §2 pins 2xH800 sm_120"},
        "S1_detection": classes,
        "S1_path_class": path_classes,
        "S1_declared_uncompared": {
            "classes": UNCOMPARED_CLASSES,
            "reason": ("" if not UNCOMPARED_CLASSES else
                       "The CuTe surface can express these classes, but this "
                       "candidate lane has not been run against them: they are "
                       "unmeasured, not inexpressible, and carry no S1 cell."),
        },
        "S1_class_axis": {
            "source": "schema/class-axis.json",
            "axis_version": "v2.1",
            # legal axis vocabulary only (measured|n/a|uncompared|not_ready):
            # read from the axis so the lane-local copy cannot drift from it.
            "status": AX.lane_status(LANE),
            # lane-local phase detail; `S1_detection` holds the only cells.
            "sampled": {"M1": True, "M2": True, "M3": True, "M4": True},
        },
        "notes": [
            "vertical slice: launched M1/M2/M3/M4 batteries + L path-class control",
            "M4 is a CONFORMING 7-family battery of 56 instances (families "
            "M4-a..M4-g x 4 kernels x 2 realisations): the kernels are the "
            "spec-required M4_OPS (layer_normalization, softmax, matmul, "
            "elemwise_add) and the two realisations are the small/alt extent "
            "grids in sizes.py",
            "M4-a: M4.1 (with-in dim -> 0, LOOP=0) + M4.2 (parallel-by -> 0, "
            "PBOUND=0); M4-b: M4.6 negative bound (NBOUND); M4-c: M4.3 "
            "runtime-zero bound (env CUT_LOOP_RT); M4-f: M4.5 zero step "
            "(STEP=0) -- both extent realisations -> 8 each",
            "M4-d: M4.4 (empty iteration space) + M1.6 (stride-0 over an empty "
            "range) are neutral controls, no-op by construction (recorded "
            "`n/a`); M4-e: M1.7 reversed/overrunning bound (REVB); M4-g: "
            "M4.7/M4.8 degenerate padded extent, a mutation-only pad category "
            "(PADEXT) on all four kernels",
            "M4 outcomes at launch time: M4.8 empty padded extent detains at "
            "COMPILE time (CuTe static assert); M4.1/M4.6/M4.3/M4.5 and M4.7 "
            "compile and launch with output differing silently (`never`); the "
            "M4.2 zero parallel-by, the neutral controls and the elemwise_add "
            "zero step are `n/a`",
            "M3 is a CONFORMING 8-family battery of 64 instances (families "
            "M3-a..M3-h x 4 kernels x 2 realisations): the kernels are the "
            "spec-required M3_OPS (matmul, conv2d, batch_norm, max_pool2d) and "
            "the two realisations are the small/alt extent grids in sizes.py; "
            "one representative spec per family (M3.1/M3.2/M3.3/M3.4/M3.6/M3.8/"
            "M3.9/M3.12), applied to the operator's own element index",
            "M1 is a CONFORMING 8-family battery of 64 instances (families "
            "M1-a..M1-h x 4 kernels x 2 realisations): the kernels are "
            "MINIMAL_SET[\"M1\"] (layer_normalization, softmax, relu, "
            "transpose) and the two realisations are the small/alt extent grids; "
            "one representative spec per family (M1.2/M1.4/M1.9/M1.12/M1.14/"
            "M1.19/M1.15/M1.11), applied to the operator's own index arithmetic",
            "M2 is a CONFORMING 8-family battery of 64 instances (families "
            "M2-a..M2-h x 4 kernels x 2 realisations): the kernels are "
            "matmul, conv2d, layer_normalization, relu and the two realisations "
            "are the small/alt extent grids; one representative spec per family "
            "(M2.1/M2.6/M2.7/M2.8/M2.10/M2.5/M2.15/M2.17)",
            "n_na counts the instance-level inapplicable rows: outcome `n/a`, "
            "measured `avoid`, applicable=false, manifest `noop`. These are the "
            "neutral controls "
            "(M4.4/M1.6 empty-iteration no-op by construction, M1.7 reversed "
            "bound) and the specs whose defect has no legal statement on some "
            "operators (M1.19 32-bit carrier on ln/softmax/transpose; the "
            "launch-status control L1 on matmul). They are excluded from "
            "detection and enter no denominator (§9.6.1b rule 4).",
            "`L` (launch-status) is a path class, not a mutation class: its "
            "cells live under `S1_path_class`, never in `S1_detection`",
            "records are v2.1 mutant records (see cutlass/mutrec.py): outcome "
            "compile|runtime|never|n/a, class M1/M2/M3/M4, path_class carries "
            "the L distinction",
        ],
        "records": len(recs),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stats.json")
    # Preserve blocks this collector does not own (S8_expressibility, S9,
    # S12_method/S12_sanitizer_supplement) so `collect` is idempotent instead of
    # destructive: those are derived from sanitizer.json / probes, not from the
    # mutant records, and a regenerate must not silently drop them.
    if os.path.exists(path):
        try:
            with open(path) as f:
                prior = json.load(f)
            for k, v in prior.items():
                if k not in stats:
                    stats[k] = v
        except (json.JSONDecodeError, OSError):
            pass
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[collect] wrote {path}  ({len(recs)} records)")
    for c, v in list(classes.items()) + list(path_classes.items()):
        print(f"[collect] {c}: injected={v['n_injected']} "
              f"compile={v['n_compile']} runtime={v['n_runtime']} "
              f"never={v['n_never']} n_na={v['n_na']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "records.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "results")
