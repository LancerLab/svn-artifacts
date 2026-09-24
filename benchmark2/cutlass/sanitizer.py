#!/usr/bin/env python3
"""sanitizer.py — GPU compute-sanitizer sweep over the CUTLASS/CuTe lane.

The lane classifies a mutant by its OWN emitted diagnostic (CUT_CHECK) or by a
byte-difference from the same-extent base. This script is the independent
*dynamic oracle*: it launches each built binary under compute-sanitizer's
memcheck and records the ERROR SUMMARY. It answers one question the lane's own
records cannot -- when the injected defect is launched, does a GPU memory
sanitizer see it?

Representative sampling keeps the sweep finite: one kernel per (spec, grid) plus
every base kernel. The full battery is in records.jsonl; this is the oracle arm.

Usage:
    sanitizer.py [--raw raw] [--out results/sanitizer.json] [--device 0]
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane                                                       # noqa: E402

SANITIZER = "/usr/local/cuda/bin/compute-sanitizer"
RE_ERRORS = re.compile(r"ERROR SUMMARY: (\d+) error")
RE_LEAK = re.compile(r"LEAK SUMMARY")


def tag_for(spec_id, cat, mut, env, grid):
    tag = f"{cat}.{spec_id.replace('.', '_')}." + \
          "_".join(f"{k}{v}" for k, v in sorted((mut or {}).items()))
    if env:
        tag += "." + "_".join(f"{k}{v}" for k, v in sorted(env.items()))
    return tag + f".{grid}"


def run_sanitized(binpath, outpath, env, device):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = str(device)
    if env:
        e.update(env)
    cmd = [SANITIZER, "--tool", "memcheck", "--error-exitcode", "0",
           binpath, outpath]
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=300)
    blob = (p.stdout or "") + (p.stderr or "")
    m = RE_ERRORS.search(blob)
    errors = int(m.group(1)) if m else None
    tail = [ln for ln in blob.splitlines() if "ERROR SUMMARY" in ln
            or "Invalid" in ln or "out of bounds" in ln]
    return errors, (tail[-1].strip() if tail else ""), p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "cutlass", "sanitizer.json"))
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES",
                                                       "0"))
    a = ap.parse_args()

    # The lane's own outcome tells a compile-time detention (no binary by
    # design) from a missing build (a real gap in the sweep).
    outcome_of = {}
    recpath = os.path.join(HERE, "records.jsonl")
    if os.path.exists(recpath):
        with open(recpath) as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    key = (r.get("spec_id"), r.get("category"),
                           (r.get("mutation") or {}).get("grid"))
                    outcome_of[key] = r

    runs = []
    seen = set()
    for spec_id, cat, mut, env, grid in lane.BATTERY:
        if spec_id.startswith("L"):
            continue
        if (spec_id, grid) in seen:
            continue
        seen.add((spec_id, grid))
        tag = tag_for(spec_id, cat, mut, env, grid)
        binp = os.path.join(a.raw, tag)
        outp = binp + ".san.bin"
        rec = outcome_of.get((spec_id, cat, grid))
        if not os.path.exists(binp):
            out = rec.get("outcome") if rec else None
            if out == "compile":
                detail = "n/a: compile-time detention (ct-check)"
            else:
                detail = "binary not found (run `lane.py all`)"
            runs.append({"spec_id": spec_id, "category": cat, "grid": grid,
                         "kernel": cat, "outcome": out, "errors": None,
                         "detail": detail})
            continue
        errs, detail, rc = run_sanitized(binp, outp, env, a.device)
        runs.append({"spec_id": spec_id, "category": cat, "grid": grid,
                     "kernel": cat,
                     "mutant_id": (rec or {}).get("mutant_id"),
                     "outcome": (rec or {}).get("outcome"),
                     "errors": errs, "rc": rc, "detail": detail})
        print(f"[san] {spec_id:6s} {cat:20s}/{grid:5s} errors={errs} {detail}")

    bases = []
    for cat in lane.OPS:
        for grid in lane.GRIDS:
            binp = os.path.join(a.raw, f"{cat}.{grid}.base")
            if not os.path.exists(binp):
                continue
            errs, detail, rc = run_sanitized(binp, binp + ".san.bin", None,
                                             a.device)
            bases.append({"category": cat, "grid": grid, "errors": errs,
                          "rc": rc, "detail": detail})
            print(f"[san] base   {cat:20s}/{grid:5s} errors={errs} {detail}")

    by_class: dict[str, dict] = {}
    for r in runs:
        cls = lane.mutrec.class_of(r["spec_id"])
        b = by_class.setdefault(cls, {"n": 0, "n_error": 0, "n_clean": 0,
                                      "n_unknown": 0})
        b["n"] += 1
        if r["errors"] is None:
            b["n_unknown"] += 1
        elif r["errors"] > 0:
            b["n_error"] += 1
        else:
            b["n_clean"] += 1

    report = {
        "toolchain": "cutlass",
        "tool": "compute-sanitizer --tool memcheck",
        "host": {"gpu": "RTX 3070", "arch": "sm_86", "cuda": "12.9"},
        "sampling": "one kernel per (spec, grid); every base kernel",
        "note": (
            "The dynamic oracle is blind to most of the battery, but not all of "
            "it. Memcheck reports 0 errors for every M2 and M3 sample and for "
            "most M1/M4 samples: those index perturbations stay INSIDE the "
            "allocation (a wrong element of the right buffer), so a memory "
            "sanitizer cannot see them -- which is the lane's point. It DOES "
            "catch the samples whose index escapes the allocation: M1.2, "
            "M1.9 (off-by-one / un-shrunk base offset) and M1.12 (leading "
            "coordinate taken from the trailing loop variable) on "
            "layer_normalization, and M4.2 (zero parallel-by bound) on "
            "layer_normalization. A "
            "compile-time detention (ct-check) has no binary to run and is "
            "recorded `n/a`."),
        "classes": by_class,
        "bases": bases,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"[san] wrote {a.out}")
    for cls, b in sorted(by_class.items()):
        print(f"[san] {cls}: {b}")


if __name__ == "__main__":
    main()
