#!/usr/bin/env python3
"""collect.py — fold lane.py records into results/stats.json (spec §8 shape).

This lane is a VERTICAL SLICE: only the realizable mutation battery has been
run. `complete` is therefore false and the class axis is `partial`; downstream
renderers must not read these cells as a finished measurement.
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

# M1 (element access) and M2 (shape contract) are expressible on the CuTe
# surface but this candidate lane has never been run against them: when the axis
# marks them `uncompared` they are declared here, NOT reported as `n/a`. The
# distinction is load-bearing (`check_class_axis` g6): an unmeasured class must
# be absent from S1 and declared here, while an inexpressible class would be
# reported as `n/a` with a note.
UNCOMPARED_CLASSES = AX.uncompared_classes(LANE)


def class_of(spec_id):
    if spec_id.startswith("M4"):
        return "M4"
    if spec_id.startswith("M3"):
        return "M3"
    return "L"   # launch-status path class (record-schema `path_class`)


def main(records_path, out_dir):
    recs = []
    paths = [records_path]
    m3 = os.path.join(os.path.dirname(os.path.abspath(records_path)), "records_m3.jsonl")
    if m3 not in paths:
        paths.append(m3)
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                recs += [json.loads(l) for l in f if l.strip()]

    classes = {"M3": dict(n_injected=0, n_compile=0, n_runtime=0, n_never=0,
                          n_discarded_noop=0),
               "M4": dict(n_injected=0, n_compile=0, n_runtime=0, n_never=0,
                          n_discarded_noop=0)}
    # `L` is a path class, not a mutation class: it must not appear in
    # `S1_detection` (which is keyed by M1..M4 only). It is reported separately.
    path_classes = {"L": dict(n_injected=0, n_compile=0, n_runtime=0,
                              n_never=0, n_discarded_noop=0)}
    for r in recs:
        c = class_of(r["spec_id"])
        if c == "L":
            bucket = path_classes["L"]
        else:
            classes.setdefault(c, dict(n_injected=0, n_compile=0, n_runtime=0,
                                       n_never=0, n_discarded_noop=0))
            bucket = classes[c]
        bucket["n_injected"] += 1
        o = r["outcome"]
        if o == "ct-check":
            bucket["n_compile"] += 1
        elif o == "rt-check":
            bucket["n_runtime"] += 1
        elif o == "unchecked":
            bucket["n_never"] += 1
        elif o == "noop":
            bucket["n_discarded_noop"] += 1

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
            "reason": ("The CuTe surface can express element-access (M1) and "
                       "shape-contract (M2) mutations, but this candidate lane "
                       "has only been run against M3/M4. M1 and M2 are "
                       "unmeasured, not inexpressible: they carry no S1 cell."),
        },
        "S1_class_axis": {
            "source": "schema/class-axis.json",
            "axis_version": "v2.1",
            # legal axis vocabulary only (measured|n/a|uncompared|not_ready):
            # the lane is held at not_ready in the axis until it is promoted.
            "status": {"M1": "not_ready", "M2": "not_ready",
                       "M3": "not_ready", "M4": "not_ready"},
            # lane-local phase detail; `S1_detection` holds the only M3/M4 cells.
            "sampled": {"M1": False, "M2": False, "M3": True, "M4": True},
        },
        "notes": [
            "vertical slice: launch-time M4/L battery + compile-only M3 probe battery",
            "M4 outcomes are launch-time: M4.1 compile-time zero bound, M4.3 "
            "runtime-zero bound (env CUT_LOOP_RT), M4.5 zero step; all unchecked",
            "M4.1 zero-trip loop is applied to all 15 shape-bearing "
            "categories; M4.3 (runtime-zero bound) to the four operators "
            "whose kernels take a runtime loop argument",
            "M4.2 (parallelby 0/negative) overlaps the loop-extent knob; the "
            "negative variant is not representable. M4.4 is a harness noop "
            "control (see the `noop` outcome), not an admissible injection",
            "coverage extension: all 15 shape-bearing categories are gated and "
            "carry an M4.1 cell; dma_rank5 carries no shape and is not gated",
            "M3 outcomes are COMPILE-ONLY: `unchecked` means nvcc/CuTe did not "
            "detain the defect at compile time; runtime behaviour is unmeasured",
            "M3 ct-check probes: M3.5 swizzle, M3.6 vector divisibility, "
            "M3.8 GMMA descriptor rank",
            "M3 unchecked (compile) probes: M3.1/M3.2/M3.3/M3.4/M3.7/M3.11/"
            "M3.12/M3.13/M3.14; TMA/GMMA pairs target sm_90a, static-smem sm_86",
            "M3.9/M3.10 (TMA pad-field) and M3.15 (pad path) are unexpressible "
            "on CuTe: the descriptor/pad fields are authored by the library, "
            "not the kernel author",
            "M3.16 (symbolic leading dim) is unexpressible: make_tma_copy "
            "requires static box shapes",
            "n_discarded_noop counts admissible noop controls (e.g. M4.4)",
            "`L` (launch-status) is a path class, not a mutation class: its "
            "cells live under `S1_path_class`, never in `S1_detection`",
            "M1/M2 are declared uncompared in `S1_declared_uncompared`; they "
            "carry no S1 cell because the lane was never run against them",
        ],
        "records": len(recs),
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[collect] wrote {path}  ({len(recs)} records)")
    for c, v in list(classes.items()) + list(path_classes.items()):
        print(f"[collect] {c}: injected={v['n_injected']} "
              f"compile={v['n_compile']} runtime={v['n_runtime']} "
              f"never={v['n_never']} noop={v['n_discarded_noop']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "records.jsonl",
         sys.argv[2] if len(sys.argv) > 2 else "results")
