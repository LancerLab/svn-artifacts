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

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import mutrec                                                   # noqa: E402

CUDA = "/usr/local/cuda-12.9/bin/nvcc"
CUTLASS = os.path.abspath(os.path.join(
    ROOT, "..", "..", "croqtile", "extern", "cutlass"))
INCLUDES = ["-I", os.path.join(CUTLASS, "include"),
            "-I", os.path.join(CUTLASS, "tools", "util", "include"),
            "-I", os.path.join(ROOT, "kernels")]
SRC = os.path.join(ROOT, "kernels", "probe_m3.cu")
SRC_M1 = os.path.join(ROOT, "kernels", "probe_m1.cu")

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
    ("M3.14", 140, 141, "sm_90a",
     "linear copy dim: 4096 vs 2^24 (no check on the linear-copy path)"),
]

# M1 (element access) probe slice: one realizable spec per family (a..h). On a
# hand-written CuTe kernel the index arithmetic is the author's, so every pair
# is expected to compile (unchecked) except M1-g, where CuTe's tuple `get`
# carries an "Index out of range" static_assert. This is the evidence that the
# surface detains rank/arity but none of the bound/stride/offset/tile axes.
M1_PROBES = [
    ("M1.2", 10, 11, "sm_86",
     "M1-a bound: p#n off-by-one reads one past the extent"),
    ("M1.4", 20, 21, "sm_86",
     "M1-b stride: transposed/non-contiguous stride, extents intact"),
    ("M1.9", 30, 31, "sm_86",
     "M1-c origin: base displaced 8 elements without shrinking the extent"),
    ("M1.12", 40, 41, "sm_86",
     "M1-d index: dimension addressed with the wrong loop variable"),
    ("M1.14", 50, 51, "sm_86",
     "M1-e tile coordinate: block/tile coords swapped, in-tile index legal"),
    ("M1.19", 60, 61, "sm_86",
     "M1-f carrier: 32-bit index cannot name a legal element of a 2^33 tensor"),
    ("M1.15", 70, 71, "sm_86",
     "M1-g rank: get<2> on a rank-2 shape (CuTe static_assert)"),
    ("M1.11", 80, 81, "sm_86",
     "M1-h overlap: two live tiles share 8 of 16 slots"),
]


def compile_probe(pid, arch, objdir, src=SRC):
    out = os.path.join(objdir, f"probe_{pid}.o")
    err = os.path.join(objdir, f"probe_{pid}.err")
    cmd = [CUDA, "-std=c++17", f"-arch={arch}", "-O2"] + INCLUDES + \
          [f"-DPROBE={pid}", "-c", src, "-o", out]
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


def run_battery(probes, src, tag, objdir, records_path):
    os.makedirs(objdir, exist_ok=True)
    with open(src, "rb") as f:
        probe_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    recs = []
    for spec_id, cid, mid, arch, note in probes:
        cok, cerr = compile_probe(cid, arch, objdir, src)
        mok, merr = compile_probe(mid, arch, objdir, src)
        outcome, detail = classify(spec_id, cok, mok, merr if not mok else cerr)
        if outcome == "generator-defect":
            # A control that does not compile is a defect in the probe, not in
            # the surface: excluded from the corpus (§9.6), fixed and re-run.
            print(f"[probe] {spec_id:6s} GENERATOR-DEFECT (excluded): {detail}")
            continue
        path_class = "ct-check" if outcome == "ct-check" else "unchecked"
        r = mutrec.make_record(
            spec_id=spec_id, category="probe",
            mutation={"control": cid, "mutant": mid},
            outcome=outcome, path_class=path_class, manifest="undecidable",
            prohibition="" if outcome == "ct-check" else "absent",
            applicable=True, mutant_id=f"cutlass-{tag}probe-{spec_id}-{mid}",
            detail=detail, kernel_hash=probe_hash,
            settings_hash=f"probe{cid}->{mid}", arch=arch, note=note)
        recs.append(r)
        print(f"[probe] {spec_id:6s} ctrl={cok!s:5s} mut={mok!s:5s} -> "
              f"{outcome:9s} {note}")
    if records_path:
        with open(records_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs


def main(objdir=None, records_path=None):
    return run_battery(
        PROBES, SRC, "m3",
        objdir or os.path.join(ROOT, "raw", "probes"),
        records_path or os.path.join(ROOT, "records_m3.jsonl"))


def main_m1(objdir=None, records_path=None):
    return run_battery(
        M1_PROBES, SRC_M1, "m1",
        objdir or os.path.join(ROOT, "raw", "probes_m1"),
        records_path or os.path.join(ROOT, "records_m1.jsonl"))


if __name__ == "__main__":
    import sys
    which = sys.argv[3] if len(sys.argv) > 3 else "m3"
    objdir = sys.argv[1] if len(sys.argv) > 1 else None
    rec = sys.argv[2] if len(sys.argv) > 2 else None
    if which == "m1":
        main_m1(objdir, rec)
    else:
        main(objdir, rec)
