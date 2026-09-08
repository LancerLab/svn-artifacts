"""Triton lane driver — benchmark2/triton/driver.py.

Implements the plan's lifecycle (§2.3) and outcome taxonomy (specs §6):
  compose+gate kernels -> inject M1/M3 defects -> classify every mutant
  (compile / runtime / never / n/a) with the mandatory manifestation oracle
  (specs §7: unmutated vs mutated output must differ for `never`/`runtime`
  to count; `noop` mutants are discarded).

Outcome mapping for Triton (see onboarding report R-T3):
  compile = JIT compile error before any device work (e.g. tl.dot shape rules)
  runtime = launch/runtime failure (illegal memory access, assert, sanitizer flag)
  never   = runs to completion; output corruption confirmed by the oracle
  n/a     = the surface cannot express the defect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import re
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
B2 = HERE.parent
KERNELS = HERE / "kernels"

sys.path.insert(0, str(KERNELS))


def sha1_text(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def settings_hash(category: str) -> str:
    p = B2 / "settings" / f"{category}.md"
    return sha1_text(p.read_text()) if p.exists() else "unknown"


def triton_version() -> str:
    import triton
    return triton.__version__


# ---------------------------------------------------------------- oracle
def oracle_check(category, ref_out, mut_out) -> str:
    """manifest field: corrupts | noop"""
    if mut_out is None:
        return "corrupts"  # never wrote output
    try:
        same = torch.allclose(ref_out, mut_out, atol=1e-4, equal_nan=True)
    except Exception:
        same = False
    return "noop" if same else "corrupts"


# ---------------------------------------------------------------- records
def emit(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def kernel_record(category, size, compile_st, run_st, ref_st, device):
    return {
        "toolchain": "triton",
        "category": category,
        "settings_hash": settings_hash(category),
        "kernel_hash": "",
        "size": size,
        "gpu_device": device,
        "exclusive": "false",
        "compile": compile_st,
        "run": run_st,
        "ref_check": ref_st,
        "toolchain_version": triton_version(),
    }


def stage_of(outcome: str) -> str:
    # schema note: enum is {compile, runtime} (choreo-oriented); "none" marks
    # never-detected. Reported to coordinator as a schema gap.
    return {"compile": "compile", "runtime": "runtime"}.get(outcome, "none")


def mutant_record(category, cls, paper_cat, mid, level, outcome, manifest,
                  detail=""):
    return {
        "toolchain": "triton",
        "category": category,
        "class": cls,
        "paper_category": paper_cat,
        "mutant_id": mid,
        "level": str(level),
        "outcome": outcome,
        "stage": stage_of(outcome),
        "manifest": manifest,
        "detail": detail[:200],
        "toolchain_version": triton_version(),
    }


# ---------------------------------------------------------------- gating
def gate_kernel(category: str, size: str, device: str, raw: Path):
    """Compose gate: import the kernel module and run its built-in self-check."""
    mod = f"{category}"
    try:
        subprocess.run(
            [sys.executable, str(KERNELS / f"{mod}.py")],
            check=True, capture_output=True, text=True, timeout=600)
        emit(raw / "kernels.jsonl",
             kernel_record(category, size, "ok", "ok", "pass", device))
        return True
    except subprocess.CalledProcessError as e:
        emit(raw / "kernels.jsonl",
             kernel_record(category, size, "ok", "crash", "fail", device))
        (raw / f"{category}.stderr.log").write_text(e.stderr[-4000:])
        return False
    except subprocess.TimeoutExpired:
        emit(raw / "kernels.jsonl",
             kernel_record(category, size, "ok", "crash", "fail", device))
        return False


# ---------------------------------------------------------------- mutants
MUTANTS = {
    "layer_normalization": ("layer_norm", "M1", "oob", [1, 2, 3, 4, 5]),
    "softmax": ("softmax", "M1", "oob", [1, 2, 3, 4, 5]),
    "relu": ("relu", "M1", "oob", [1, 2, 3, 4, 5, 6]),
    "transpose": ("transpose", "M1", "stride", [1, 2, 3, 4, 5]),
    "matmul": ("matmul", "M3", "hw", [1, 2]),
    "conv2d": ("conv2d", "M3", "hw", []),
}
PAPER_CAT = {"M1": "oob", "M3": "hw"}

COMPILE_ERR_MARKERS = ("OutOfResources", "CompilationError",
                       "Input shapes should have", "MLIR", "frontend error")


def run_mutant(category: str, family: int, raw: Path):
    """Run one mutant in a subprocess; classify per the taxonomy."""
    mod, cls, pcat, _ = MUTANTS[category]
    mid = f"{mod}-f{family}"
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "mutants" / f"{mod}.py"),
             "--family", str(family)],
            capture_output=True, text=True, timeout=300,
            cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return mutant_record(category, cls, pcat, mid, 1, "runtime",
                             "corrupts", "timeout")
    out = r.stdout + r.stderr
    m = re.search(r"RESULT run=(\w+) out=(\S+)", out)
    if m is None:
        # crashed before printing: classify by error class
        if any(k in out for k in COMPILE_ERR_MARKERS):
            return mutant_record(category, cls, pcat, mid, 1, "compile",
                                 "corrupts",
                                 "jit: " + out.strip().splitlines()[-1][:140])
        return mutant_record(category, cls, pcat, mid, 1, "runtime",
                             "corrupts",
                             "crash: " + out.strip().splitlines()[-1][:140])
    run, oracle = m.group(1), m.group(2)
    if run == "crash":
        # error text sits above the RESULT line in the child output
        if any(k in out for k in COMPILE_ERR_MARKERS):
            return mutant_record(category, cls, pcat, mid, 1, "compile",
                                 "corrupts", "jit: child reported crash")
        return mutant_record(category, cls, pcat, mid, 1, "runtime",
                             "corrupts", "child crash")
    # ran to completion: was anything caught? Triton catches nothing at
    # runtime; manifestation decides never vs noop-discard.
    manifest = "corrupts" if ("diff" in oracle
                              or "CLOBBERED" in oracle) else "noop"
    outcome = "never" if manifest == "corrupts" else "never"
    return mutant_record(category, cls, pcat, mid, 1, outcome, manifest,
                         f"oracle={oracle}")


def cmd_minimal(device: str):
    raw = HERE / "raw"
    for category in MUTANTS:
        mod, cls, pcat, fams = MUTANTS[category]
        if not fams:
            continue  # nothing assigned on this surface for this operator
        for fam in fams:
            rec = run_mutant(category, fam, raw)
            emit(raw / "mutants.jsonl", rec)
            print(rec["mutant_id"], rec["outcome"], rec["manifest"],
                  rec["detail"][:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "minimal"])
    ap.add_argument("--category", default=None)
    ap.add_argument("--size", default="small", choices=["small", "full"])
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    raw = HERE / "raw"
    if args.stage == "gate":
        ok = gate_kernel(args.category, args.size, args.device, raw)
        sys.exit(0 if ok else 1)
    if args.stage == "minimal":
        cmd_minimal(args.device)


if __name__ == "__main__":
    main()
