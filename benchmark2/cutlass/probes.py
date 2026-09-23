#!/usr/bin/env python3
"""probes.py — compile-only M3 probe battery for the CUTLASS/CuTe lane.

Compiles each (control, mutant) pair in kernels/probe_m3.cu with `nvcc -c` and
classifies the surface's compile-time behaviour (§9.6):

    control OK + mutant FAIL -> ct-check  (CuTe statically detains the defect)
    control OK + mutant OK   -> unchecked (surface does not detain it)
    control FAIL             -> generator-defect (excluded; fix the probe)

Only compile-time evidence is produced here; anything that compiles is *not*
claimed to be wrong at runtime — that is a separate launch-time question and is
recorded as `unchecked` (compile-time) with the caveat in the note.
"""

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA = "/usr/local/cuda-12.9/bin/nvcc"
CUTLASS = os.path.abspath(os.path.join(
    ROOT, "..", "..", "croqtile", "extern", "cutlass"))
INCLUDES = ["-I", os.path.join(CUTLASS, "include"),
            "-I", os.path.join(CUTLASS, "tools", "util", "include"),
            "-I", os.path.join(ROOT, "kernels")]
SRC = os.path.join(ROOT, "kernels", "probe_m3.cu")

# spec_id, control, mutant, arch, note
PROBES = [
    ("M3.1", 10, 11, "sm_90a",
     "atom-M divisibility: partition_A accepts a non-divisible extent"),
    ("M3.2", 20, 21, "sm_90a",
     "descriptor dim: 2^23 vs 2^24 tensor extent"),
    ("M3.3", 30, 31, "sm_90a",
     "TMA box byte-size: 256B vs 64MB (3D 256^3 box)"),
    ("M3.4", 40, 41, "sm_90a",
     "footprint: 2^30B vs 2^32B tensor product"),
    ("M3.5", 50, 51, "sm_90a",
     "swizzle B-field: Swizzle<3,4,3> vs Swizzle<7,4,3>"),
    ("M3.6", 60, 61, "sm_90a",
     "vector divisibility: extent 4 vs 3 with a 128-bit copy atom"),
    ("M3.7", 70, 71, "sm_90a",
     "TMA inner box: 128B vs 8B (not 16B aligned)"),
    ("M3.8", 80, 81, "sm_90a",
     "GMMA descriptor rank: rank-2 vs rank-3 smem tensor"),
    ("M3.11", 110, 111, "sm_90a",
     "shared operand base: 0B vs 4B offset (not 128B aligned)"),
    ("M3.12", 120, 121, "sm_86",
     "static shared tile: 16KB vs 128KB (over the sm_86 48KB static limit)"),
    ("M3.13", 130, 131, "sm_90a",
     "swizzle vs box inner dim: SW128 with 128B vs 32B inner"),
]


def compile_probe(pid, arch, objdir):
    out = os.path.join(objdir, f"probe_{pid}.o")
    err = os.path.join(objdir, f"probe_{pid}.err")
    cmd = [CUDA, "-std=c++17", f"-arch={arch}", "-O2"] + INCLUDES + \
          [f"-DPROBE={pid}", "-c", SRC, "-o", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(err, "w") as f:
        f.write(p.stderr)
    return p.returncode == 0, p.stderr


def first_error(stderr):
    for line in stderr.splitlines():
        if "error" in line or "static assertion" in line:
            return line.strip()[:200]
    return ""


def classify(spec_id, control_ok, mutant_ok, err):
    if not control_ok:
        return "generator-defect", first_error(err)
    if not mutant_ok:
        return "ct-check", first_error(err)
    return "unchecked", "control and mutant both compile"


def main(objdir=None, records_path=None):
    objdir = objdir or os.path.join(ROOT, "raw", "probes")
    os.makedirs(objdir, exist_ok=True)
    recs = []
    for spec_id, cid, mid, arch, note in PROBES:
        cok, cerr = compile_probe(cid, arch, objdir)
        mok, merr = compile_probe(mid, arch, objdir)
        outcome, detail = classify(spec_id, cok, mok, merr if not mok else cerr)
        r = {"spec_id": spec_id, "category": "probe", "mutation": {"control": cid, "mutant": mid},
             "arch": arch, "outcome": outcome, "detail": detail, "note": note}
        recs.append(r)
        print(f"[probe] {spec_id:6s} ctrl={cok!s:5s} mut={mok!s:5s} -> {outcome:16s} {note}")
    if records_path:
        with open(records_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None,
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "records_m3.jsonl"))
