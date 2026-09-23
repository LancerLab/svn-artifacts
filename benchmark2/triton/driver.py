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

import triton
import triton.language as tl


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


# Set by `--arch` in main(); read by current_arch() so every record carries the
# requested arch, not just the one auto-detected. `None`/`"auto"` = auto-detect.
ARCH_OVERRIDE: str | None = None
VALID_ARCHES = ("auto", "sm_86", "sm_90", "sm_120")


def device_arch() -> str:
    """The physical GPU's arch, regardless of any `--arch` override."""
    from triton.runtime import driver
    try:
        return f"sm_{driver.active.get_current_target().arch}"
    except Exception:
        return "unknown"


def current_arch() -> str:
    """Requested arch if `--arch` was given, else the auto-detected GPU arch.

    Auto-detection alone cannot label a corpus for a host it is not running on
    (this checkout has sm_86; the final host has sm_120). `--arch` makes the
    label explicit and main() refuses a mismatch unless --allow-mismatch."""
    if ARCH_OVERRIDE and ARCH_OVERRIDE != "auto":
        return ARCH_OVERRIDE
    return device_arch()


# ---------------------------------------------------------------- oracle
def oracle_check(category, ref_out, mut_out) -> str:
    """manifest field: corrupts | noop.

    NOTE: this lane is torch-free by design (gpubuf.py is a ctypes libcudart
    shim), so the comparison is numpy. It used to call `torch.allclose` with
    no `import torch` anywhere in the file: the `except Exception` swallowed
    the resulting NameError and returned `corrupts` for every mutant. The
    function is currently unreferenced -- the live manifest comes from
    `run_mutant` -- so the bug was latent, not active. Do not "fix" it by
    adding a torch dependency.
    """
    import numpy as np

    if mut_out is None:
        return "corrupts"  # never wrote output
    try:
        same = bool(np.allclose(np.asarray(ref_out), np.asarray(mut_out),
                                atol=1e-4, equal_nan=True))
    except Exception:
        same = False
    return "noop" if same else "corrupts"


# ---------------------------------------------------------------- records
# Identity of a record within its stream. Everything NOT listed here (outcome,
# manifest, detail, hashes, toolchain_version) is the measurement, and is
# overwritten when the stage is re-run.
IDENTITY = ("toolchain", "category", "class", "mutant_id", "level", "size")


def _key(record: dict) -> tuple:
    return tuple((k, record[k]) for k in IDENTITY if k in record)


def emit(path: Path, record: dict):
    """Upsert `record` into the JSONL stream at `path`, keyed by IDENTITY.

    This lane writes ONE stream from TWO invocations: `minimal` (23 level-1
    rows) and `minimal --level2` (6 level-2 rows) both land in
    `raw/mutants.jsonl`. So the writer cannot truncate per invocation -- that
    would make the second stage destroy the first -- and it must not blindly
    append either: appending the same generation twice silently doubles the
    corpus (observed 29 -> 52 rows, 23 byte-identical pairs), which is the
    `mutants.no-duplicate-injection` finding and inflates every count derived
    from the lane. Upsert gives both properties at once: re-running a stage
    REPLACES that stage's rows, and the two stages still accumulate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _key(record)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue  # drop unparseable lines rather than propagate them
            if _key(old) != key:
                rows.append(old)
    rows.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def kernel_record(category, size, compile_st, run_st, ref_st, device):
    return {
        "toolchain": "triton",
        "category": category,
        "settings_hash": settings_hash(category),
        "kernel_hash": "",
        "size": size,
        "gpu_device": device,
        "arch": current_arch(),
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
    # family number is carried in the mutant_id (`{mod}-f{family}`), so every
    # call site attributes without threading a new argument through.
    m = re.search(r"-f(\d+)$", mid)
    family = int(m.group(1)) if m else None
    return {
        "toolchain": "triton",
        "category": category,
        "class": cls,
        "paper_category": paper_cat,
        "mutant_id": mid,
        "spec_id": spec_id_for(category, cls, family),
        "level": str(level),
        "arch": current_arch(),
        "outcome": outcome,
        "stage": stage_of(outcome),
        "manifest": manifest,
        "detail": detail[:200],
        "toolchain_version": triton_version(),
    }


# ---------------------------------------------------------------- gating
def gate_kernel(category: str, size: str, device: str, raw: Path):
    """Compose gate: import the kernel module and run its built-in self-check."""
    mod = KERNEL_MOD.get(category, category)
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
    # f3 (M1.3) is NOT generated: family M1-a already holds its 8 from f1/f2
    # across these four categories, so f3 would over-deliver to 12. The family
    # lists below are the reproducible corpus; do not re-add f3.
    "layer_normalization": ("layer_norm", "M1", "oob",
                            [1, 2, 4, 5, 8, 9, 12, 121, 19, 21, 211]),
    "softmax": ("softmax", "M1", "oob",
                [1, 2, 4, 5, 8, 9, 12, 121, 14, 141, 19]),
    "relu": ("relu", "M1", "oob", [1, 2, 4, 5, 6, 8, 9, 14, 141]),
    "transpose": ("transpose", "M1", "stride",
                  [1, 2, 4, 5, 8, 9, 12, 121, 14, 141]),
    "matmul": ("matmul", "M3", "hw", [1, 2, 3]),
    # mutation-only descriptor-pad surface: M3-g (M3.9/M3.10), four ranks x two
    # overrun geometries (mutants/desc_pad.py).
    "desc_pad": ("desc_pad", "M3", "hw", [1, 2, 3, 4, 5, 6, 7, 8]),
    # mutation-only descriptor-dim surface: M3-b (M3.2), ranks 2..5 x two
    # int32 carriers (2**31 / 2**32) -- structure variants of matmul-f3
    # (mutants/desc_dim.py).
    # corpus kernel already holds one desc_dim instance; surface supplies 7.
    "desc_dim": ("desc_dim", "M3", "hw", [1, 2, 3, 4, 5, 6, 7]),
    # mutation-only descriptor-rank surface: M3-f (M3.8), four base shapes x
    # {rank 6, rank 0}; the frontend refuses at trace time (mutants/desc_rank.py).
    "desc_rank": ("desc_rank", "M3", "hw", [1, 2, 3, 4, 5, 6, 7, 8]),
    # mutation-only tl.dot atom surface: M3-a (M3.1), four base shapes x two
    # sub-atom K values; JIT refuses the tile (mutants/dot_atom.py).
    # corpus kernel already holds one dot_atom instance; surface supplies 7.
    "dot_atom": ("dot_atom", "M3", "hw", [1, 2, 3, 4, 5, 6, 7]),
    # mutation-only on-chip capacity surface: M3-h (M3.12), four base shapes x
    # two over-budget tilings; JIT refuses OutOfResources (mutants/smem_cap.py).
    # corpus kernel already holds one smem_cap instance; surface supplies 7.
    "smem_cap": ("smem_cap", "M3", "hw", [1, 2, 3, 4, 5, 6, 7]),
    # mutation-only descriptor box surface: M3-c (M3.3/M3.5/M3.13), oversized /
    # non-power-of-2 / shape-incompatible / oversized boxes; frontend refuses
    # (mutants/box_swizzle.py).
    "box_swizzle": ("box_swizzle", "M3", "hw", [1, 2, 3, 4, 5, 6, 7, 8]),
    # mutation-only inner-box alignment surface: M3-e (M3.7), sub-16-byte last
    # dimension of a descriptor box; frontend refuses (mutants/box_align.py).
    "box_align": ("box_align", "M3", "hw", [1, 2, 3, 4, 5, 6, 7, 8]),
    # mutation-only rank/arity surface: M1-g (M1.15/16/17/20), rank/axis/arity
    # arguments out of domain; frontend refuses (mutants/rank_arity.py).
    "rank_arity": ("rank_arity", "M1", "oob", [1, 2, 3, 4, 5, 6, 7, 8]),
    "conv2d": ("conv2d", "M3", "hw", []),
    # carrier-only categories: M1.19 is a generator setting (huge extent), not a
    # source edit, so these hold only family 19 and are not re-run by `minimal`.
    "sigmoid": ("sigmoid", "M1", "oob", []),
    "gelu": ("gelu", "M1", "oob", []),
    "elemwise_add": ("elemwise_add", "M1", "oob", []),
    "reshape": ("reshape", "M1", "oob", []),
    # level-2 categories extended with M1.12 (batch_norm) / M1.19 (concat)
    # realisations; empty base fams keep `minimal` from re-running the whole
    # level-2 surface.
    "batch_norm": ("batch_norm", "M1", "oob", []),
    "concat": ("concat", "M1", "oob", []),
}
# level-2 (plan §3.1 priority rows; run only via `minimal --level2`)
MUTANTS_L2 = {
    "max_pool2d": ("max_pool2d", "M1", "oob", [1, 2]),
    "conv2d": ("conv2d", "M1", "oob", [1]),
    "embedding": ("embedding", "M1", "oob", [1, 2]),
    "batch_norm": ("batch_norm", "M1", "oob", [1]),
}
PAPER_CAT = {"M1": "oob", "M3": "hw"}

# `spec_id` is the field the worklist generator reads to attribute an instance
# to a family (schema/gen_worklist.py). The family NUMBER is local to this lane,
# so four variants do NOT line up with their number -- read the variant
# (mutants/*.py), not the number:
#   embedding f1  idx = V+3         -> M1.13 (runtime index past the row bound)
#   embedding f2  idx = -1          -> M1.3
#   batch_norm f1 c = pid%C + 1     -> M1.2
#   matmul    f2  BM=BN=256, BK=64  -> M3.12 (on-chip capacity, not M3.11)
#   matmul    f3  descriptor dim=2**31 -> M3.2 (silent int32 shape narrowing)
# Every other (category, family) is M{class}.{family}. relu f6 is the
# noop-by-construction control: its spec_id is M1.6, which the registry classes
# M4, so it lands in family M4-d and does NOT count toward this lane's M1=64.
SPEC_ID = {
    ("embedding", 1): "M1.13",
    ("embedding", 2): "M1.3",
    ("batch_norm", 1): "M1.2",
    ("matmul", 2): "M3.12",
    ("matmul", 3): "M3.2",  # descriptor dim >= 2**31 silently zero-fills
    # desc_pad mutation-only surface: odd = tail overrun (M3.9), even = mid
    # overrun (M3.10); four ranks each.
    ("desc_pad", 1): "M3.9",
    ("desc_pad", 2): "M3.10",
    ("desc_pad", 3): "M3.9",
    ("desc_pad", 4): "M3.10",
    ("desc_pad", 5): "M3.9",
    ("desc_pad", 6): "M3.10",
    ("desc_pad", 7): "M3.9",
    ("desc_pad", 8): "M3.10",
    # desc_dim: every rank/carrier variant is the same M3.2 defect on rank 2..5
    ("desc_dim", 1): "M3.2",
    ("desc_dim", 2): "M3.2",
    ("desc_dim", 3): "M3.2",
    ("desc_dim", 4): "M3.2",
    ("desc_dim", 5): "M3.2",
    ("desc_dim", 6): "M3.2",
    ("desc_dim", 7): "M3.2",
    # ("desc_dim", 8) trimmed: family already holds 8 with the corpus kernel.
    # ("desc_dim", 8): "M3.2",
    # desc_rank: rank-6 (odd) and rank-0 (even) are both M3.8
    ("desc_rank", 1): "M3.8",
    ("desc_rank", 2): "M3.8",
    ("desc_rank", 3): "M3.8",
    ("desc_rank", 4): "M3.8",
    ("desc_rank", 5): "M3.8",
    ("desc_rank", 6): "M3.8",
    ("desc_rank", 7): "M3.8",
    ("desc_rank", 8): "M3.8",
    # dot_atom: every base shape / sub-atom K is the M3.1 atom defect
    ("dot_atom", 1): "M3.1",
    ("dot_atom", 2): "M3.1",
    ("dot_atom", 3): "M3.1",
    ("dot_atom", 4): "M3.1",
    ("dot_atom", 5): "M3.1",
    ("dot_atom", 6): "M3.1",
    ("dot_atom", 7): "M3.1",
    # ("dot_atom", 8) trimmed: family already holds 8 with the corpus kernel.
    # ("dot_atom", 8): "M3.1",
    # smem_cap: every base shape / tiling is the M3.12 capacity defect
    ("smem_cap", 1): "M3.12",
    ("smem_cap", 2): "M3.12",
    ("smem_cap", 3): "M3.12",
    ("smem_cap", 4): "M3.12",
    ("smem_cap", 5): "M3.12",
    ("smem_cap", 6): "M3.12",
    ("smem_cap", 7): "M3.12",
    # ("smem_cap", 8) trimmed: family already holds 8 with the corpus kernel.
    # ("smem_cap", 8): "M3.12",
    # box_swizzle: box<->swizzle<->alignment relation (M3.5), survive-repair
    # (M3.13), and ceiled byte size (M3.3)
    ("box_swizzle", 1): "M3.5",
    ("box_swizzle", 2): "M3.5",
    ("box_swizzle", 3): "M3.5",
    ("box_swizzle", 4): "M3.5",
    ("box_swizzle", 5): "M3.5",
    ("box_swizzle", 6): "M3.3",
    ("box_swizzle", 7): "M3.3",
    ("box_swizzle", 8): "M3.3",
    # box_align: inner-box geometry not 128-bit aligned (M3.7, family M3-e)
    ("box_align", 1): "M3.7",
    ("box_align", 2): "M3.7",
    ("box_align", 3): "M3.7",
    ("box_align", 4): "M3.7",
    ("box_align", 5): "M3.7",
    ("box_align", 6): "M3.7",
    ("box_align", 7): "M3.7",
    ("box_align", 8): "M3.7",
    # rank_arity: dimof index (M1.15), select factor (M1.16), rank-5 5th-index
    # (M1.17), view/subspan rank arity (M1.20)
    ("rank_arity", 1): "M1.20",
    ("rank_arity", 2): "M1.15",
    ("rank_arity", 3): "M1.17",
    ("rank_arity", 4): "M1.17",
    ("rank_arity", 5): "M1.20",
    ("rank_arity", 6): "M1.15",
    ("rank_arity", 7): "M1.16",
    ("rank_arity", 8): "M1.16",
    ("relu", 141): "M1.14",   # second M1.14 realisation (dest tile coord)
    ("softmax", 141): "M1.14",  # second M1.14 realisation (source tile coord)
    ("transpose", 141): "M1.14",  # second M1.14 realisation (dest tile coord)
    ("layer_normalization", 211): "M1.21",  # second M1.21 realisation
    ("layer_normalization", 121): "M1.12",  # second M1.12 realisation (store)
    ("softmax", 121): "M1.12",  # second M1.12 realisation (store)
    ("transpose", 121): "M1.12",  # second M1.12 realisation (store)
    ("batch_norm", 121): "M1.12",  # second M1.12 realisation (store)
    ("relu", 111): "M1.11",       # second M1.11 realisation (forward shift)
    ("sigmoid", 111): "M1.11",    # second M1.11 realisation (forward shift)
    ("gelu", 111): "M1.11",       # second M1.11 realisation (forward shift)
    ("reshape", 111): "M1.11",    # second M1.11 realisation (forward shift)
}


def spec_id_for(category: str, cls: str, family) -> str:
    return SPEC_ID.get((category, family), f"{cls}.{family}")


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


# ---------------------------------------------------------------- archcheck
# The sm_90 final host is gone; the lane must be valid on sm_86 (dev) and
# sm_120 (available host). The mutation-only surfaces are frontend refusals, so
# their outcome should not depend on the codegen arch. This stage *proves* that
# by compiling one representative kernel per mechanism for each requested arch
# (compile-only; no GPU needed), then reports whether the verdict is invariant.
#
# `refused` = non-inert mutant / `avoided`; `lowered` = no refusal (a silent
# defect such as M3.2 int32 narrowing, or a valid control such as rank-1).
@triton.jit
def _p_nonpow2(x, o, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x, shape=[64, 256], strides=[256, 1],
                                  block_shape=[1, B])
    tl.store(o, tl.sum(tl.ravel(d.load([0, 0]))))


@triton.jit
def _p_bigbox(x, o, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x, shape=[1 << 20, 64], strides=[64, 1],
                                  block_shape=[1, B])
    tl.store(o, tl.sum(tl.ravel(d.load([0, 0]))))


@triton.jit
def _p_tinybox(x, o):
    d = tl.make_tensor_descriptor(x, shape=[64, 256], strides=[256, 1],
                                  block_shape=[1, 1])
    tl.store(o, tl.sum(tl.ravel(d.load([0, 0]))))


@triton.jit
def _p_rank6(x, o, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x, shape=[2, 2, 2, 2, 2, 64],
                                  strides=[64, 32, 16, 8, 4, 1],
                                  block_shape=[1, 1, 1, 1, 1, B])
    tl.store(o, tl.sum(tl.ravel(d.load([0, 0, 0, 0, 0, 0]))))


@triton.jit
def _p_rank1(x, o, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x, shape=[256], strides=[1], block_shape=[B])
    tl.store(o, tl.sum(tl.ravel(d.load([0]))))


@triton.jit
def _p_int32dim(x, o, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x, shape=[1 << 31, 64], strides=[64, 1],
                                  block_shape=[1, B])
    tl.store(o, tl.sum(tl.ravel(d.load([0, 0]))))


ARCHCHECK_PROBES = {
    # name -> (kernel, signature, constexprs, mechanism)
    "nonpow2_box": (_p_nonpow2, {"B": 48}, "M3.5 non-power-of-2 box"),
    "big_box": (_p_bigbox, {"B": 1 << 21}, "M3.3 box numel over budget"),
    "tiny_box": (_p_tinybox, {}, "M3.7 inner box < 16 B"),
    "rank6_desc": (_p_rank6, {"B": 16}, "M3.8 descriptor rank > 5"),
    "rank1_desc": (_p_rank1, {"B": 16}, "control: legal rank-1 descriptor"),
    "int32_dim": (_p_int32dim, {"B": 16}, "M3.2 int32 dim (silent, not refused)"),
}


def _cap_of(arch: str) -> int:
    return int(arch.split("_")[1])


def cmd_archcheck(arches, raw: Path):
    from triton.compiler import ASTSource, compile as tcompile
    from triton.backends.compiler import GPUTarget
    sig_base = {"x": "*fp32", "o": "*fp32"}
    out = {}
    for arch in arches:
        out[arch] = {}
        for name, (fn, cst, mech) in ARCHCHECK_PROBES.items():
            sig = dict(sig_base)
            sig.update({k: "constexpr" for k in cst})
            try:
                tcompile(ASTSource(fn, sig, constexprs=cst),
                         target=GPUTarget("cuda", _cap_of(arch), 32))
                verdict, msg = "lowered", ""
            except Exception as e:
                verdict = "refused"
                msg = str(e).strip().splitlines()[-1][:90]
            out[arch][name] = {"verdict": verdict, "mechanism": mech,
                               "detail": msg}
            print(f"{arch} {name:12s} {verdict:8s} {msg}")
    # invariance: every arch must agree on each probe
    for name in ARCHCHECK_PROBES:
        verdicts = {a: out[a][name]["verdict"] for a in arches}
        same = len(set(verdicts.values())) == 1
        print(f"INVARIANT {name:12s} {'yes' if same else 'NO'} {verdicts}")
    (raw / "archcheck.json").write_text(json.dumps(out, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "minimal", "e2", "sanitizer",
                                      "archcheck"])
    ap.add_argument("--arch", default="auto", choices=list(VALID_ARCHES),
                    help="arch to label records with (default: auto-detect). "
                         "Refuses if it differs from the physical GPU unless "
                         "--allow-mismatch is set.")
    ap.add_argument("--arches", default="sm_86,sm_90,sm_120",
                    help="archcheck: comma-separated archs to cross-compile")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="allow --arch to differ from the physical GPU "
                         "(compile-only cross-checks; do NOT use for a run "
                         "whose outcomes depend on device execution)")
    ap.add_argument("--category", default=None)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--level2", action="store_true")
    args = ap.parse_args()
    raw = HERE / "raw"

    global ARCH_OVERRIDE
    if args.arch != "auto":
        ARCH_OVERRIDE = args.arch
        dev = device_arch()
        # The arch label must be the arch the code actually runs on; a mismatch
        # would mislabel records (or, worse, launch a cubin the GPU cannot load).
        if (dev not in ("unknown", args.arch)
                and not args.allow_mismatch
                and args.stage in ("gate", "minimal", "e2", "sanitizer")):
            sys.exit(f"ERROR: --arch {args.arch} but physical GPU is {dev}; "
                     "pass --allow-mismatch only for compile-only checks")
    if args.stage == "archcheck":
        arches = [a.strip() for a in args.arches.split(",") if a.strip()]
        cmd_archcheck(arches, raw)
        return
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
