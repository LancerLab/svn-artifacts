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
            [sys.executable, str(KERNELS / f"{mod}.py"), "--size", size],
            check=True, capture_output=True, text=True, timeout=900)
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
# level-2 (plan §3.1 priority rows; run only via `minimal --level2`)
MUTANTS_L2 = {
    "max_pool2d": ("max_pool2d", "M1", "oob", [1, 2]),
    "conv2d": ("conv2d", "M1", "oob", [1]),
    "embedding": ("embedding", "M1", "oob", [1, 2]),
    "batch_norm": ("batch_norm", "M1", "oob", [1]),
}
PAPER_CAT = {"M1": "oob", "M3": "hw"}

COMPILE_ERR_MARKERS = ("OutOfResources", "CompilationError",
                       "Input shapes should have", "MLIR", "frontend error")


CURRENT_TABLE = MUTANTS


def run_mutant(category: str, family: int, raw: Path, size: str = "full"):
    """Run one mutant in a subprocess; classify per the taxonomy."""
    mod, cls, pcat, _ = CURRENT_TABLE[category]
    mid = f"{mod}-f{family}"
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "mutants" / f"{mod}.py"),
             "--family", str(family), "--size", size],
            capture_output=True, text=True, timeout=900,
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


def cmd_minimal(device: str, level2: bool = False, size: str = "full"):
    raw = HERE / "raw"
    table = MUTANTS_L2 if level2 else MUTANTS
    for category in table:
        mod, cls, pcat, fams = table[category]
        if not fams:
            continue  # nothing assigned on this surface for this operator
        for fam in fams:
            rec = run_mutant(category, fam, raw, size=size)
            rec["level"] = "2" if level2 else "1"
            rec["detail"] += f" size={size}"
            emit(raw / "mutants.jsonl", rec)
            print(rec["mutant_id"], rec["outcome"], rec["manifest"],
                  rec["detail"][:80])


# ---------------------------------------------------------------- e2
# Triton expressibility per obligation class (schema: expressibility record).
# Triton has no generated checks; every check is a user-written mask value.
TRITON_EXPRESSIBILITY = {
    "elem": "yes",     # masks express bounds, but are user values
    "shape": "no",     # no cross-tensor contract on the surface
    "loop": "yes",     # manual loop bounds
    "hw": "partial",   # tl.dot tile rules + smem budget checked at JIT only
}
CATEGORIES = ["batch_norm", "concat", "conv2d", "elemwise_add", "embedding",
              "gelu", "layer_normalization", "matmul", "max_pool2d",
              "reduce_mean", "relu", "reshape", "sigmoid", "softmax",
              "transpose"]
KERNEL_MOD = {"layer_normalization": "layer_norm", "elemwise_add":
              "elemwise_add", "reduce_mean": "reduce_mean", "max_pool2d":
              "max_pool2d", "batch_norm": "batch_norm"}


def cmd_e2(device: str, size: str):
    raw = HERE / "raw"
    # gate every composed category at BOTH sizes (plan rec. 4: small/full
    # pairing — same program, same reference check, both sizes recorded)
    for cat in CATEGORIES:
        mod = KERNEL_MOD.get(cat, cat)
        if not (KERNELS / f"{mod}.py").exists():
            continue
        for sz in ("small", "full"):
            gate_kernel(cat, sz, device, raw)
    # expressibility records per category per class
    for cat in CATEGORIES:
        for cls, val in TRITON_EXPRESSIBILITY.items():
            emit(raw / "expressibility.jsonl", {
                "toolchain": "triton", "category": cat, "class": cls,
                "expressible": val, "toolchain_version": triton_version()})


SANITIZER = "/usr/local/cuda/bin/compute-sanitizer"


def run_mutant_sanitizer(category: str, family: int, raw: Path):
    """S12 (§3.5): does compute-sanitizer memcheck flag what bare Triton
    missed? Run the mutant under the sanitizer; record flagged + exercised."""
    mod, cls, pcat, _ = CURRENT_TABLE[category]
    mid = f"{mod}-f{family}"
    try:
        r = subprocess.run(
            [SANITIZER, "--tool", "memcheck", sys.executable,
             str(HERE / "mutants" / f"{mod}.py"), "--family", str(family)],
            capture_output=True, text=True, timeout=600, cwd=str(HERE))
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"toolchain": "triton", "category": category, "class": cls,
                "mutant_id": mid, "flagged": "timeout", "fault": "none",
                "exercised": "unknown", "detail": str(e)[:120]}
    out = r.stdout + r.stderr
    flagged = ("Invalid __global__" in out or "Invalid __shared__" in out
               or "misaligned" in out.lower())
    fault = ("oob" if "Invalid __global__" in out
             else "misaligned" if "misaligned" in out.lower() else "none")
    m = re.search(r"RESULT run=(\w+) out=(\S+)", out)
    # exercised = the kernel body actually ran on device: either it completed
    # or the sanitizer observed the faulting access itself.
    exercised = "true" if (flagged or (m and m.group(1) == "ok")) else "false"
    return {"toolchain": "triton", "category": category, "class": cls,
            "mutant_id": mid, "flagged": str(flagged).lower(),
            "fault": fault, "exercised": exercised,
            "detail": ("sanitizer: " + out.strip().splitlines()[-1][:100])
                      if flagged else "clean"}


def cmd_sanitizer(device: str, level2: bool = False):
    raw = HERE / "raw"
    table = dict(MUTANTS)
    if level2:
        table.update(MUTANTS_L2)
    for category, (mod, cls, pcat, fams) in table.items():
        for fam in fams:
            rec = run_mutant_sanitizer(category, fam, raw)
            emit(raw / "sanitizer.jsonl", rec)
            print(rec["mutant_id"], "flagged=" + rec["flagged"],
                  rec["fault"], "exercised=" + rec["exercised"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "minimal", "e2", "sanitizer"])
    ap.add_argument("--category", default=None)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--level2", action="store_true")
    args = ap.parse_args()
    raw = HERE / "raw"
    if args.stage == "gate":
        ok = gate_kernel(args.category, args.size, args.device, raw)
        sys.exit(0 if ok else 1)
    if args.stage == "minimal":
        if args.level2:
            CURRENT_TABLE.clear()
            CURRENT_TABLE.update(MUTANTS_L2)
        cmd_minimal(args.device, level2=args.level2, size=args.size)
    if args.stage == "sanitizer":
        table = dict(MUTANTS)
        if args.level2:
            table.update(MUTANTS_L2)
        CURRENT_TABLE.clear()
        CURRENT_TABLE.update(table)
        cmd_sanitizer(args.device, level2=args.level2)
    if args.stage == "e2":
        cmd_e2(args.device, args.size)


if __name__ == "__main__":
    main()
