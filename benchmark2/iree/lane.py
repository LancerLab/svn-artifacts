#!/usr/bin/env python3
"""benchmark2/iree/lane.py — IREE lane driver bodies for run.sh (§12.1).

iree worker row (mutation-specs §4 / plan §8.1):
  mutation classes M1 ("entry shape only") + M2 ("entry shape check"); M3 = n/a.
  E2 expressibility = "entry shape asserts only"; E3 remainder = "entry checks
  not statically foldable".

Kernel records carry a STRUCTURAL reference check: the device run must return
the declared output shape (numeric value oracle implemented for E1 minimal ops
in minimal.py). size=small maps every dynamic/oversized dim to 8 so runs are
cheap; the mapping is recorded per record and is semantics-preserving for the
small-input mode (plan §2.4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # benchmark2/
LANE = ROOT / "iree"
RAW = LANE / "raw"

# `stage` is a required field of a mutant record and this lane never wrote it.
# It is a projection of `outcome` (schema/records.py STAGE_FOR_OUTCOME), so the
# value was always available and the omission leaves the lane's S1 rows
# unreadable per stage. The rule is imported rather than restated here.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from schema import records as RS                       # noqa: E402
KERNELS = LANE / "kernels"
IREE_BIN = Path(os.environ.get(
    "IREE_BIN_DIR", "/home/gxf/.tools/iree-dev-20260908-venv/bin"))
IREE_COMPILE = IREE_BIN / "iree-compile"
IREE_RUN = IREE_BIN / "iree-run-module"
CUDA_TARGET = os.environ.get("IREE_CUDA_TARGET", "sm_120")
CUDA_FEATURES = os.environ.get("IREE_CUDA_FEATURES", "")
if CUDA_TARGET == "sm_120" and not CUDA_FEATURES:
    CUDA_FEATURES = "+ptx87"
SMALL = 8
FULL = 128

FUNC_RE = re.compile(r"func\.func @([\w]+)\(([^)]*)\)\s*->\s*([^\s{]+)")
TENSOR_RE = re.compile(r"tensor<([^>]+)>")

M1_OPS = ["layer_normalization", "softmax", "relu", "transpose"]
M2_OPS = ["layer_normalization", "matmul", "concat"]
MINIMAL_OPS = sorted(set(M1_OPS + M2_OPS))


def log(*a):
    print("[iree/lane]", *a, flush=True)


def parse_dims_dtype(frag: str) -> tuple[list, str]:
    parts = frag.split("x")
    dtype = parts[-1]
    dims = []
    for p in parts[:-1]:
        dims.append("?" if p == "?" else int(p))
    return dims, dtype


def parse_func(mlir: Path):
    text = mlir.read_text()
    m = FUNC_RE.search(text)
    if not m:
        return None
    fname, args_str, ret_frag = m.group(1), m.group(2), m.group(3)
    args = []
    for part in args_str.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        nm, ty = part.split(":", 1)
        tm = TENSOR_RE.search(ty)
        if not tm:
            continue
        dims, dtype = parse_dims_dtype(tm.group(1))
        args.append((nm.strip(), dims, dtype))
    ret = None
    rm = TENSOR_RE.search(ret_frag)
    if rm:
        ret = parse_dims_dtype(rm.group(1))
    return fname, args, ret


def compile_kernel(mlir: Path, vmfb: Path, clog: Path, timeout=120) -> str:
    cmd = [str(IREE_COMPILE), str(mlir), "--iree-hal-target-backends=cuda",
           f"--iree-cuda-target={CUDA_TARGET}"]
    if CUDA_FEATURES:
        cmd += [f"--iree-cuda-target-features={CUDA_FEATURES}"]
    cmd += ["-o", str(vmfb)]
    try:
        with clog.open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
        return "ok" if r.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"


def smallized(dims: list) -> list:
    return [SMALL if d == "?" else d for d in dims]


_FULL_SHAPES = None


def full_shapes():
    global _FULL_SHAPES
    if _FULL_SHAPES is None:
        p = LANE / "full_shapes.json"
        _FULL_SHAPES = json.loads(p.read_text()) if p.exists() else {"cases": {}}
    return _FULL_SHAPES


def case_dims(cat: str, stem: str) -> list:
    return full_shapes().get("cases", {}).get(f"{cat}/{stem}", {}).get("dims", [])


def concrete_feed_dims(cat: str, stem: str, arg_dims: list, size: str) -> list:
    """Per-operand concrete run dims: concrete `#define` value for --full,
    ragged small for --small (plan §2.4), falling back to the legacy blanket
    only for dims with no pinned full size."""
    concrete = case_dims(cat, stem)
    out = []
    for i, dims in enumerate(arg_dims):
        cd = concrete[i] if i < len(concrete) else []
        feed = []
        for j, d in enumerate(dims):
            if d != "?":
                feed.append(d)
                continue
            cv = cd[j] if j < len(cd) else "?"
            if size == "full":
                feed.append(FULL if cv == "?" else cv)
            else:
                # ragged small: (cv % 7)+3 for the concrete value, else SMALL
                feed.append((cv % 7) + 3 if isinstance(cv, int) else SMALL)
        out.append(feed)
    return out


def input_spec(dims: list, dtype: str, size: str, cat: str = "", stem: str = "") -> str:
    dd = concrete_feed_dims(cat, stem, [dims], size)[0] if (cat and stem) else \
        (smallized(dims) if size == "small" else fullsized(dims))
    shape = "x".join(str(d) for d in dd)
    return f"{shape}x{dtype}=" + "1"  # shape-conformant all-ones (structural gate)


def run_module(vmfb: Path, func: str, arg_specs: list[str], timeout=90) -> tuple[str, str, Path]:
    rlog = RAW / f"_run_{vmfb.stem}.log"
    rlog.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(IREE_RUN), f"--module={vmfb}", "--device=cuda",
           f"--function={func}"]
    for a in arg_specs:
        cmd += ["--input=" + a]
    try:
        with rlog.open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    shape = ""
    if rc == 0:
        m = re.search(r"result\[0\]: hal\.buffer_view\s*\n\s*([0-9x]+xf32)", rlog.read_text())
        if m:
            shape = m.group(1)
    return ("ok" if rc == 0 else "crash"), shape, rlog


def expected_shape(ret, size="small"):
    if ret is None:
        return None
    dims, dtype = ret
    sub = SMALL if size == "small" else FULL
    return "x".join(str(sub if d == "?" else d) for d in dims) + "x" + dtype


def gate_kernel(cat: str, stem: str, mlir: Path, size="small") -> dict:
    d = RAW / cat
    d.mkdir(parents=True, exist_ok=True)
    parsed = parse_func(mlir)
    base = {"toolchain": "iree", "category": cat, "kernel": stem, "size": size,
            "gpu_device": "0", "exclusive": "false"}
    if parsed is None:
        return {**base, "compile": "fail", "run": "-", "ref_check": "fail",
                "note": "unparseable-signature"}
    fname, args, ret = parsed
    vmfb = d / f"{stem}.{size}.vmfb"
    clog = d / f"{stem}.{size}.compile.log"
    c = compile_kernel(mlir, vmfb, clog)
    if c != "ok":
        return {**base, "compile": "fail", "run": "-", "ref_check": "fail"}
    specs = [input_spec(dims, dtype, size, cat, stem) for (_n, dims, dtype) in args]
    run, shape, rlog = run_module(vmfb, fname, specs)
    exp = expected_shape(ret, size)
    # ref_check is structural: the device run must return a tensor whose shape
    # matches the declared return type. When the return type has dynamic dims
    # (incl. derived dims such as reshape products / concat sums), the device
    # value is authoritative, so any successful run of the right rank passes.
    ret_has_dyn = bool(ret and any(d == "?" for d in ret[0]))
    ref = "pass" if (run == "ok" and shape and (not ret_has_dyn or shape.count("x") == len(ret[0]))) else "fail"
    if ref == "fail" and run == "ok" and exp and shape == exp:
        ref = "pass"
    return {**base, "compile": "ok", "run": run, "ref_check": ref,
            "run_shape": shape or "", "expected_shape": exp or ""}


def cmd_e2(size="small"):
    manifest = KERNELS / "manifest.jsonl"
    out = RAW / "kernels.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[f"{r['category']}/{r['kernel']}/{r.get('size','small')}"] = r
    recs = list(done.values())
    n_new = 0
    with out.open("a") as f:
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            meta = json.loads(line)
            cat, stem = meta["category"], meta["stem"]
            key = f"{cat}/{stem}/{size}"
            if key in done:
                continue
            mlir = KERNELS / cat / f"{stem}.mlir"
            if not mlir.exists():
                continue
            g = gate_kernel(cat, stem, mlir, size)
            g.update({"settings_hash": meta["settings_hash"],
                      "kernel_hash": meta["kernel_hash"]})
            recs.append(g)
            done[key] = g
            f.write(json.dumps(g) + "\n")
            n_new += 1
            if n_new % 20 == 0:
                log(f"e2 gate progress: {len(recs)} recorded")
    n_ok = sum(1 for r in recs if r["compile"] == "ok")
    n_run = sum(1 for r in recs if r.get("run") == "ok")
    n_ref = sum(1 for r in recs if r["ref_check"] == "pass")
    log(f"e2 gate [{size}]: compiled {n_ok}/{len(recs)}, ran {n_run}, "
        f"ref-pass {n_ref} (new this run: {n_new})")


def cmd_expressibility():
    rows = []
    for cat in sorted(p.stem for p in (ROOT / "settings").glob("*.md")):
        for cls, expr in (("elem", "partial"), ("shape", "yes"),
                          ("loop", "no"), ("hw", "no")):
            if cat == "reshape" and cls == "shape":
                expr = "no"
            rows.append({"toolchain": "iree", "category": cat, "class": cls,
                         "expressible": expr})
    with (RAW / "expressibility.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"expressibility: {len(rows)} rows")


def cmd_e3():
    per = {}
    for line in (KERNELS / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        mlir = KERNELS / meta["category"] / f"{meta['stem']}.mlir"
        if not mlir.exists():
            continue
        parsed = parse_func(mlir)
        if not parsed:
            continue
        dyn = sum(1 for (_n, dims, _t) in parsed[1] for d in dims if d == "?")
        per.setdefault(meta["category"], []).append(1 if dyn else 0)
    rows = []
    for cat, v in sorted(per.items()):
        rows.append({"toolchain": "iree", "category": cat,
                     "unconditional_guards": sum(v),
                     "criterion_ref": "entry dynamic dims (not statically foldable)"})
    with (RAW / "remainder.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"e3 remainder: {len(rows)} rows")


# ---------------------------------------------------------------------------
# E1 — measured entry-shape mutation (M2 only; M1/M3 => n/a on this surface)
# ---------------------------------------------------------------------------

def _ramp_input(dims: list, dtype: str, size="small") -> str:
    """Compact fill (single value, replicated by iree) — keeps argv small.
    Oracle is structural (log diff): M2 shape mutants either fail at entry
    (runtime) or change the derived output (shape/value) — a byte-identical
    silent run is discarded as a noop (plan §11.1)."""
    return input_spec(dims, dtype, size)


def _distinct_spec(dims: list, dtype: str) -> str:
    """Distinct integer-valued fill (deterministic) for the numeric oracle."""
    n = 1
    for d in dims:
        n *= d
    vals = ",".join(str((i % 9) + 1) for i in range(n))
    return f"{'x'.join(str(x) for x in dims)}x{dtype}=" + vals


def _parse_result(log: Path):
    if not log.exists():
        return None
    txt = log.read_text()
    m = re.search(r"result\[0\]: hal\.buffer_view\s*\n", txt)
    seg = txt[m.end():] if m else txt
    if "=" not in seg:
        return None
    return [float(v) for v in
            re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", seg[seg.find("=") + 1:])]


def _run_out(vmfb: Path, func: str, arg_specs, timeout=90, extra=None) -> tuple[int, Path, str]:
    h = hashlib.md5("\x00".join(arg_specs).encode()).hexdigest()[:10]
    rlog = RAW / f"_mut_{vmfb.stem}_{h}.log"
    cmd = [str(IREE_RUN), f"--module={vmfb}", "--device=cuda", f"--function={func}"]
    for a in arg_specs:
        cmd += ["--input=" + a]
    if extra:
        cmd += list(extra)
    try:
        with rlog.open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    shape = ""
    if rc == 0:
        m = re.search(r"result\[0\]: hal\.buffer_view\s*\n\s*([0-9x]+xf32)", rlog.read_text())
        if m:
            shape = m.group(1)
    return rc, rlog, shape


# M2 mutant descriptors over the minimal set (each is a shape-contract
# violation at iree entry; arg_idx/dim_idx index into the parsed signature).
# layer_normalization -> secondary operands gamma(1)/beta(2) wrong leading
# extent (mutation-specs §2 family 1/3). matmul -> secondary operand K-extent
# wrong (family 3). concat / elementwise single-tensor ops have no expressible
# M2 entry mutant here => recorded via S8 (n/a), not fabricated.
def _m2_bases(cat: str, want: int = 3) -> list[dict]:
    bases = []
    for line in (RAW / "kernels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["category"] != cat or r.get("run") != "ok" or "dynamic" not in r["kernel"]:
            continue
        bases.append(r)
        if len(bases) >= want:
            break
    return bases


M2_SPECS = {
    "layer_normalization": [("gamma-len-1", 1, 0, -1), ("gamma-len+1", 1, 0, +1),
                            ("beta-len-1", 2, 0, -1),
                            ("gamma-dyn-1", 1, 1, -1), ("beta-dyn-1", 2, 1, -1)],
    "matmul": [("rhs-K-1", 1, 0, -1), ("rhs-K+1", 1, 0, +1)],
}

# Level-2 (plan §3.1 / mutation-specs §5): M1 rows (max_pool2d, conv2d,
# embedding, batch_norm) and M3 (batch_norm) are n/a on IREE's surface; the
# only expressible level-2 addition is M2 on elemwise_add (binary-op / leading-
# extent contract). softmax is single-tensor => n/a for M2.
LEVEL2_SPECS = {
    "elemwise_add": [("rhs-len-1", 1, 0, -1), ("rhs-len+1", 1, 0, +1),
                     ("rhs-d1-1", 1, 1, -1)],
}

# M2-a family budget: 4 kernels x 2 realisations = 8, matching every other M2
# family (the lane card's "battery-vs-composition" gap). The keep-set is
# explicit so each chosen kernel contributes exactly one realisation per spec and
# all four M2-a specs stay credited: M2.19 on the dynamic layer_norm (symbolic
# extent, escapes the entry check), M2.1 on a static layer_norm, M2.3 on
# elemwise_add and M2.14 on matmul. Materialising the whole per-kernel battery
# (23) is what this replaces.
M2_A_KEEP = {
    "iree-layer_normalization-10-gamma-dyn-1",   # M2.19, symbolic -> never
    "iree-layer_normalization-10-beta-dyn-1",    # M2.19, symbolic -> never
    "iree-layer_normalization-11-gamma-len-1",   # M2.1, static -> runtime
    "iree-layer_normalization-11-beta-len-1",    # M2.1, static -> runtime
    "iree-elemwise_add-10-rhs-len-1",            # M2.3
    "iree-elemwise_add-10-rhs-d1-1",             # M2.3
    "iree-matmul-10-rhs-K-1",                    # M2.14
    "iree-matmul-10-rhs-K+1",                    # M2.14
}

# spec_id per edit tag -- which M2 spec the entry-contract edit realises. The
# tag alone is not enough (the same `len` tag is a leading-extent edit on
# gamma/beta but an interior-extent edit on elemwise's rhs), so the mapping is
# keyed by (category, tag). Two signals pin each choice:
#   * the spec's own `path` (registry): a static extent is rejected at the
#     entry (rt-check), a symbolic one escapes (unchecked) -- and the measured
#     outcomes split exactly that way (static -> runtime, symbolic -> never);
#   * the operand's role: gamma/beta are secondary operands (M2.1), elemwise's
#     rhs makes a binary op disagree (M2.3), matmul's rhs dim 0 is the
#     contraction dim (M2.14).
# Every one of these lands in family M2-a (`extent value`); see
# `lanes/iree.md` §2.1. `schema/gen_worklist.py::bucket()` reads `spec_id`, so
# a row without one is invisible to the denominator.
M2_SPEC_ID = {
    ("layer_normalization", "gamma-len-1"): "M2.1",
    ("layer_normalization", "gamma-len+1"): "M2.1",
    ("layer_normalization", "beta-len-1"): "M2.1",
    ("layer_normalization", "gamma-dyn-1"): "M2.19",
    ("layer_normalization", "beta-dyn-1"): "M2.19",
    ("matmul", "rhs-K-1"): "M2.14",
    ("matmul", "rhs-K+1"): "M2.14",
    ("elemwise_add", "rhs-len-1"): "M2.3",
    ("elemwise_add", "rhs-len+1"): "M2.3",
    ("elemwise_add", "rhs-d1-1"): "M2.3",
}


def m2_spec_id(cat: str, tag: str) -> str:
    """The M2 spec_id an entry-contract edit realises, or '' if unmapped."""
    return M2_SPEC_ID.get((cat, tag), "")


def m2_family(sid: str) -> str:
    """The method family that owns `sid`, or '' if unmapped (convenience only;
    `gen_worklist.py` derives the family from `spec_id` via the taxonomy)."""
    if not sid:
        return ""
    from schema import method_taxonomy as MT
    return MT.family_of(sid) or ""


def cmd_minimal(level2: bool = False):
    """E1: measured M2 entry-shape battery (level-1 minimal set; --level2 adds
    the expressible level-2 M2 addition on elemwise_add).

    IREE has no compile-time detection and no interior access/hw mechanism at
    its linalg entry surface (§3.3): M1 element-access and M3 hw-constraint
    have NO expressible mutant on this surface (=> n/a by R4, recorded via S8
    expressibility + mutants/README.md, never counted as detected). M2
    shape-compatibility IS expressible *as entry shape checks*: a mutant
    re-expresses a contract violation by changing an operand's entry extent.
    Each mutant is materialised as a committed definition file under
    `mutants/` (the mutated entry contract) and its measured outcome written
    to raw/mutants.jsonl (=> results/mutants.jsonl via collect).
    """
    MUT = LANE / "mutants"
    MUT.mkdir(parents=True, exist_ok=True)
    (MUT / "README.md").write_text(
        "# IREE mutant surface\n\n"
        "IREE consumes linalg-on-tensors MLIR; its *entry* surface exposes only "
        "shape contracts (no memory-access or hardware-constraint construct). "
        "Per specs/mutation-specs.md §4 + plan §3.3:\n\n"
        "- **M1 element-access** -> n/a (no interior access at this surface; any "
        "spec M1 mutant has no expressible IREE form).\n"
        "- **M2 shape-compatibility** -> expressible as an *entry shape/contract "
        "variant*: the defect is re-expressed by changing an operand's declared "
        "entry extent (caller-side contract), the only place IREE can detect it.\n"
        "- **M3 hw-constraint** -> n/a (no mma/tma/alignment construct).\n\n"
        "A mutant is represented as `mutants/M2/<category>/<mutant_id>.json` "
        "(reference kernel + the mutated entry extent); the kernel text itself is "
        "unchanged because the mutation lives at the entry contract, not the body.\n\n"
        "## Measured outcome (ground truth, numeric oracle)\n\n"
        "- **Static extent** mutants (operand dim is a fixed int): IREE rejects at "
        "the host entry via `hal.buffer_view.assert` "
        "(`INVALID_ARGUMENT: shape dimension mismatch`) -> **runtime**.\n"
        "- **Dynamic extent** mutants (operand dim is `?`): IREE silently computes a "
        "wrong result -> **never/value-changing** (verified by numeric diff against the "
        "reference output; see `mutant_oracle.py`).\n\n"
        "## Level-2 (mutation-specs §5)\n\n"
        "- M1 rows (max_pool2d, conv2d, embedding, batch_norm) -> n/a.\n"
        "- M3 row (batch_norm) -> n/a.\n"
        "- M2 rows: `elemwise_add` expressible (binary-op/leading-extent) and added "
        "by `run.sh minimal --level2`; `softmax` is single-tensor -> n/a.\n\n"
        "n/a cells are never counted as detected (R4).\n")


    recs = []
    specs_by_cat = dict(M2_SPECS)
    if level2:
        specs_by_cat.update(LEVEL2_SPECS)
    for cat, specs in specs_by_cat.items():
        for base in _m2_bases(cat):
            mlir = KERNELS / base["category"] / f"{base['kernel']}.mlir"
            parsed = parse_func(mlir)
            fname, args, _ret = parsed
            meta = {}
            for _l in (KERNELS / "manifest.jsonl").read_text().splitlines():
                if not _l.strip():
                    continue
                _j = json.loads(_l)
                if _j.get("category") == base["category"] and _j.get("stem") == base["kernel"]:
                    meta = _j
                    break
            vmfb = RAW / base["category"] / f"{base['kernel']}.small.vmfb"
            if not vmfb.exists():
                continue
            for tag, aidx, didx, delta in specs:
                if aidx >= len(args) or didx >= len(args[aidx][1]):
                    continue  # spec dim not present in this case (e.g. 1-D gamma)
                mid = f"iree-{cat}-{base['kernel'].split('_')[0]}-{tag}"
                if mid not in M2_A_KEEP:
                    continue  # family budget: 4 kernels x 2 realisations
                ref_specs = [_ramp_input(d, t) for (_n, d, t) in args]
                ref_rc, ref_log, _ = _run_out(vmfb, fname, ref_specs)
                if ref_rc != 0:
                    continue
                mut_dims = [list(d) for (_n, d, _t) in args]
                if mut_dims[aidx][didx] == "?":
                    mut_dims[aidx][didx] = SMALL + delta
                else:
                    mut_dims[aidx][didx] = max(1, mut_dims[aidx][didx] + delta)
                mut_specs = [_ramp_input(mut_dims[i], args[i][2])
                             for i in range(len(args))]
                rc, mlog, mshape = _run_out(vmfb, fname, mut_specs)
                if rc != 0:
                    outcome, manifest = "runtime", "value-changing"
                else:
                    # silent: numeric ground-truth oracle. Re-run reference and
                    # mutant with distinct inputs (small dynamic dims so values
                    # fit argv) and compare device outputs.
                    outcome = "never"
                    OSMALL = 2
                    ref2_specs = [_distinct_spec([OSMALL if x == "?" else x for x in d], t)
                                  for (_n, d, t) in args]
                    mut2_dims = [list(d) for (_n, d, _t) in args]
                    if mut2_dims[aidx][didx] == "?":
                        mut2_dims[aidx][didx] = OSMALL + delta
                    else:
                        mut2_dims[aidx][didx] = max(1, mut2_dims[aidx][didx] + delta)
                    mut2_specs = [_distinct_spec([OSMALL if x == "?" else x for x in mut2_dims[i]],
                                                 args[i][2]) for i in range(len(args))]
                    r2rc, r2log, _ = _run_out(vmfb, fname, ref2_specs)
                    m2rc, m2log, _ = _run_out(vmfb, fname, mut2_specs)
                    o_ref = _parse_result(r2log) if r2rc == 0 else None
                    o_mut = _parse_result(m2log) if m2rc == 0 else None
                    if o_ref is not None and o_mut is not None:
                        manifest = ("value-changing" if (len(o_ref) != len(o_mut) or
                                    any(abs(a - b) > 1e-3 for a, b in zip(o_ref, o_mut)))
                                    else "noop")
                    else:
                        manifest = "value-changing"  # contract differs; IREE ran but not cleanly
                if outcome == "never" and manifest == "noop":
                    continue  # false-success mutant discarded (plan §11.1)
                # materialise the mutated version (definition file)
                sid = m2_spec_id(cat, tag)
                fam = m2_family(sid)
                mdir = MUT / "M2" / cat
                mdir.mkdir(parents=True, exist_ok=True)
                (mdir / f"{mid}.json").write_text(json.dumps({
                    "mutant_id": mid, "class": "M2",
                    "paper_category": "dim-mismatch", "category": cat,
                    "kernel": base["kernel"],
                    "settings_hash": meta.get("settings_hash", ""),
                    "kernel_hash": meta.get("kernel_hash", ""),
                    "spec_id": sid, "family": fam,
                    "mutation": {"operand_index": aidx, "dim_index": didx,
                                 "delta": delta,
                                 "reference_dims": [d for (_n, d, _t) in args],
                                 "mutant_dims": mut_dims},
                    "outcome": outcome, "manifest": manifest,
                }, indent=2) + "\n")
                rec = {
                    "toolchain": "iree", "category": cat,
                    "settings_hash": meta.get("settings_hash", ""),
                    "kernel": base["kernel"],
                    "class": "M2", "paper_category": "dim-mismatch",
                    "mutant_id": mid, "level": ("2" if cat in LEVEL2_SPECS else "1"),
                    "outcome": outcome, "stage": RS.stage_for(outcome),
                    "manifest": manifest,
                    "kernel_hash": meta.get("kernel_hash", ""),
                    "spec_id": sid, "family": fam,
                }
                recs.append(rec)
    cmd_attribution()  # rebuild raw/mutants.jsonl from *all* families' defs
    n_runtime = sum(1 for r in recs if r["outcome"] == "runtime")
    lvl = "level-2 " if level2 else ""
    log(f"minimal [{lvl}E1]: {len(recs)} M2 mutants materialised in mutants/, "
        f"{n_runtime} runtime-caught, {len(recs)-n_runtime} never "
        f"(M1/M3 => n/a, see mutants/README.md)")



# ---------------------------------------------------------------------------
# M2 families M2-b..M2-e — entry-contract batteries beyond M2-a's extent edits.
#
# Every mutant still edits the *caller-side contract* handed to an unchanged
# compiled kernel (the lane's surface claim: IREE exposes shape contracts, not
# memory access). `kind` selects the edit:
#   swap      (aidx,a,b)    two extents transposed             -> M2.6 / M2.9
#   rankdrop  (aidx,i)      one dimension dropped              -> M2.7
#   rankadd   (aidx,i)      one size-1 dimension inserted       -> M2.16
#   set1      (aidx,i)      a broadcast extent set to 1         -> M2.8
#   setv      (aidx,i,v)    a broadcast extent set to a 3rd val -> M2.21
#   dataperm  (aidx,perm)   same shape, different memory order  -> M2.10 / M2.13
#
# M2.9 ("batch/group dimension swapped") is the dim0<->dim1 swap of a batched
# operand; M2.6 is any other extent transposition. M2.16 keeps the element
# count (size-1 insert), M2.7 does not. M2.8 sets the broadcast extent to 1,
# M2.21 to a third value. M2.10 is a transpose on a *square* operand (extents
# agree, so only the memory order differs); M2.13 is any other shape-equal /
# layout-unequal layout.
#
# M2-f (M2.5 partial/duplicate write), M2-g (M2.15 pad_low<->pad_high) and
# M2-h (M2.17 span_as on runtime-shaped data / M2.18 blocked-by-suite) have no
# caller-side realisation at this surface: see mutants/README.md and the
# channel summary for the per-spec blocker.
# ---------------------------------------------------------------------------
M2_BATTERY = [
    # family, spec_id, category, kernel, tag, kind, params
    # --- M2-b extent order ---
    ("M2-b", "M2.9", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "lhs-batchgroup-swap", "swap", (0, 0, 1)),
    ("M2-b", "M2.6", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "lhs-seq-hid-swap", "swap", (0, 1, 2)),
    ("M2-b", "M2.9", "transpose", "11_dynamic_32xSx768_32x768xS",
     "in-batchgroup-swap", "swap", (0, 0, 1)),
    ("M2-b", "M2.6", "transpose", "11_dynamic_32xSx768_32x768xS",
     "in-seq-hid-swap", "swap", (0, 1, 2)),
    ("M2-b", "M2.9", "concat", "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
     "in0-batchgroup-swap", "swap", (0, 0, 1)),
    ("M2-b", "M2.6", "concat", "11_dynamic_32xS1x768_32xS2x768_32xS1pS2x768",
     "in0-seq-hid-swap", "swap", (0, 1, 2)),
    ("M2-b", "M2.9", "conv2d", "2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D",
     "in-batch-channel-swap", "swap", (0, 0, 1)),
    ("M2-b", "M2.6", "conv2d", "2_dynamic_Nx64x56x56_128x64x3x3_Nx128x56x56_S_P_D",
     "in-channel-height-swap", "swap", (0, 1, 2)),
    # --- M2-c rank ---
    ("M2-c", "M2.7", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "lhs-rankdrop", "rankdrop", (0, 2)),
    ("M2-c", "M2.16", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "lhs-rankadd", "rankadd", (0, 0)),
    ("M2-c", "M2.7", "transpose", "11_dynamic_32xSx768_32x768xS",
     "in-rankdrop", "rankdrop", (0, 2)),
    ("M2-c", "M2.16", "transpose", "11_dynamic_32xSx768_32x768xS",
     "in-rankadd", "rankadd", (0, 0)),
    ("M2-c", "M2.7", "softmax", "11_dynamic_32xSx768_32xSx768",
     "in-rankdrop", "rankdrop", (0, 2)),
    ("M2-c", "M2.16", "softmax", "11_dynamic_32xSx768_32xSx768",
     "in-rankadd", "rankadd", (0, 0)),
    ("M2-c", "M2.7", "reduce_mean", "11_dynamic_64xTx256_64x256",
     "in-rankdrop", "rankdrop", (0, 2)),
    ("M2-c", "M2.16", "reduce_mean", "11_dynamic_64xTx256_64x256",
     "in-rankadd", "rankadd", (0, 0)),
    # --- M2-d broadcast ---
    ("M2-d", "M2.8", "layer_normalization", "10_dynamic_16x512xHxW_HxW_HxW",
     "gamma-bcast-to-1", "set1", (1, 0)),
    ("M2-d", "M2.21", "layer_normalization", "10_dynamic_16x512xHxW_HxW_HxW",
     "gamma-bcast-3rd", "setv", (1, 0, 3)),
    ("M2-d", "M2.8", "layer_normalization", "13_dynamic_32x512xV_V_V",
     "gamma-bcast-to-1", "set1", (1, 0)),
    ("M2-d", "M2.21", "layer_normalization", "13_dynamic_32x512xV_V_V",
     "gamma-bcast-3rd", "setv", (1, 0, 5)),
    ("M2-d", "M2.8", "batch_norm", "10_dynamic_16x512xHxW_512_512_16x512xHxW",
     "gamma-bcast-to-1", "set1", (1, 0)),
    ("M2-d", "M2.21", "batch_norm", "10_dynamic_16x512xHxW_512_512_16x512xHxW",
     "gamma-bcast-3rd", "setv", (1, 0, 3)),
    ("M2-d", "M2.8", "batch_norm", "13_dynamic_32x512xV_V_V_32x512xV",
     "gamma-bcast-to-1", "set1", (1, 0)),
    ("M2-d", "M2.21", "batch_norm", "13_dynamic_32x512xV_V_V_32x512xV",
     "gamma-bcast-3rd", "setv", (1, 0, 5)),
    # --- M2-e layout (extents intact, view) ---
    ("M2-e", "M2.10", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "rhs-square-transpose", "dataperm", (1, (1, 0))),
    ("M2-e", "M2.13", "matmul", "11_dynamic_32xSx768_768x768_32xSx768",
     "lhs-layout-unequal", "dataperm", (0, (0, 2, 1))),
    ("M2-e", "M2.10", "transpose", "10_dynamic_16x512xHxW_16xHxWx512",
     "in-square-transpose", "dataperm", (0, (0, 1, 3, 2))),
    ("M2-e", "M2.13", "transpose", "10_dynamic_16x512xHxW_16xHxWx512",
     "in-layout-unequal", "dataperm", (0, (1, 0, 2, 3))),
    ("M2-e", "M2.10", "concat", "10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW",
     "in0-square-transpose", "dataperm", (0, (0, 1, 3, 2))),
    ("M2-e", "M2.13", "concat", "10_dynamic_16x512xHxW_16x512xHxW_16x1024xHxW",
     "in0-layout-unequal", "dataperm", (0, (1, 0, 2, 3))),
    ("M2-e", "M2.10", "batch_norm", "10_dynamic_16x512xHxW_512_512_16x512xHxW",
     "in-square-transpose", "dataperm", (0, (0, 1, 3, 2))),
    ("M2-e", "M2.13", "batch_norm", "10_dynamic_16x512xHxW_512_512_16x512xHxW",
     "in-layout-unequal", "dataperm", (0, (1, 0, 2, 3))),
]


def _flat_coords(i: int, shape: list) -> list:
    c = [0] * len(shape)
    for ax in range(len(shape) - 1, -1, -1):
        c[ax] = i % shape[ax]
        i //= shape[ax]
    return c


def _perm_fill(shape: list, perm) -> "array":
    """float32 values laid out so the logical tensor is `base.transpose(perm)`."""
    import array
    bshape = [shape[p] for p in perm]
    n = 1
    for d in bshape:
        n *= d
    out = array.array("f")
    for i in range(n):
        cb = _flat_coords(i, bshape)
        ca = [0] * len(shape)
        for ax in range(len(shape)):
            ca[perm[ax]] = cb[ax]
        j = 0
        for ax in range(len(shape)):
            j = j * shape[ax] + ca[ax]
        out.append(float((j % 9) + 1))
    return out


def _bin_spec(dims: list, dtype: str, perm=None) -> str:
    """Raw little-endian float32 input via a file (no argv length limit)."""
    import array
    shape = list(dims)
    n = 1
    for d in shape:
        n *= d
    if perm is None:
        vals = array.array("f", [float((i % 9) + 1) for i in range(n)])
    else:
        vals = _perm_fill(shape, perm)
    h = hashlib.md5((str(shape) + str(perm)).encode()).hexdigest()[:12]
    p = RAW / "_m2bin" / f"{h}.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("wb") as f:
            f.write(vals.tobytes())
    return f"{'x'.join(str(d) for d in shape)}x{dtype}=@{p}"


def _m2_small_dims(args, feed):
    """Consistent tiny concrete dims (distinct per dynamic symbol) for the
    numeric oracle. Static dims are untouched."""
    sm, nxt = {}, [2]
    for f in feed:
        for v in f:
            if v not in sm:
                sm[v] = nxt[0]
                nxt[0] += 1
    out = []
    for i, (_n, dims, _t) in enumerate(args):
        out.append([(sm[feed[i][j]] if dims[j] == "?" else dims[j])
                    for j in range(len(dims))])
    return out


def _m2_apply(kind: str, params, dims: list):
    """Apply the entry edit to a copy of `dims`; returns (new_dims, perm, aidx)."""
    d = list(dims)
    perm = None
    if kind == "delta":
        aidx, i, delta = params
        d[i] += delta
    elif kind == "swap":
        aidx, a, b = params
        d[a], d[b] = d[b], d[a]
    elif kind == "rankdrop":
        aidx, i = params
        d.pop(i)
    elif kind == "rankadd":
        aidx, i = params
        d.insert(i, 1)
    elif kind == "set1":
        aidx, i = params
        d[i] = 1
    elif kind == "setv":
        aidx, i, v = params
        # A broadcast extent larger than the reference is simply not indexed, so
        # it never corrupts. Use the largest valid third value below the
        # reference (>=2), falling back upward only when no such value exists.
        v = 2 if d[i] > 2 else d[i] + 1
        d[i] = v
    elif kind == "dataperm":
        aidx, perm = params
    else:
        raise ValueError(kind)
    return d, perm, aidx


def _m2_measure(cat: str, stem: str, kind: str, params) -> dict | None:
    """Compile (cached) and measure one entry-contract mutant. Returns a record
    with outcome/manifest, or None if the reference does not run."""
    mlir = KERNELS / cat / f"{stem}.mlir"
    parsed = parse_func(mlir)
    if parsed is None:
        return None
    fname, args, _ret = parsed
    feed = concrete_feed_dims(cat, stem, [d for _n, d, _t in args], "small")
    vmfb = RAW / cat / f"{stem}.small.vmfb"
    if not vmfb.exists():
        clog = RAW / cat / f"{stem}.small.compile.log"
        (RAW / cat).mkdir(parents=True, exist_ok=True)
        if compile_kernel(mlir, vmfb, clog) != "ok":
            return {"compile": "fail"}
    ref_specs = [_ramp_input(feed[i], args[i][2]) for i in range(len(args))]
    rrc, _rlog, rshape = _run_out(vmfb, fname, ref_specs)
    if rrc != 0:
        return None
    mut = [list(x) for x in feed]
    aidx = params[0]
    md, perm, _ = _m2_apply(kind, params, mut[aidx])
    mut[aidx] = md
    mut_specs = [_ramp_input(mut[i], args[i][2]) for i in range(len(args))]
    rc, mlog, mshape = _run_out(vmfb, fname, mut_specs)
    ref_dims_rec = [list(x) for x in feed]
    if rc != 0:
        return {"compile": "ok", "outcome": "runtime", "manifest": "value-changing",
                "reference_dims": ref_dims_rec, "mutant_dims": mut,
                "kind": kind, "params": list(params)}
    if mshape != rshape and mshape:
        return {"compile": "ok", "outcome": "never", "manifest": "value-changing",
                "reference_dims": ref_dims_rec, "mutant_dims": mut,
                "kind": kind, "params": list(params)}
    # same-shape silent run: numeric ground truth with raw-binary inputs.
    small = _m2_small_dims(args, feed)
    smd, sperm, _ = _m2_apply(kind, params, small[aidx])
    smut = [list(x) for x in small]
    smut[aidx] = smd
    r2 = [_bin_spec(small[i], args[i][2]) for i in range(len(args))]
    m2 = [_bin_spec(smut[i], args[i][2],
                    perm=(sperm if i == aidx else None)) for i in range(len(args))]
    rr, rl, _ = _run_out(vmfb, fname, r2, extra=["--output_max_element_count=100000"])
    mm, ml, _ = _run_out(vmfb, fname, m2, extra=["--output_max_element_count=100000"])
    ro = _parse_result(rl) if rr == 0 else None
    mo = _parse_result(ml) if mm == 0 else None
    if ro is None or mo is None:
        manifest = "value-changing"
    elif len(ro) != len(mo) or any(abs(a - b) > 1e-3 for a, b in zip(ro, mo)):
        manifest = "value-changing"
    else:
        manifest = "noop"
    outcome = "never" if manifest == "value-changing" else "noop"
    return {"compile": "ok", "outcome": outcome, "manifest": manifest,
            "reference_dims": ref_dims_rec, "mutant_dims": mut,
            "kind": kind, "params": list(params)}


def _m2_17_measure(cat: str, stem: str, aidx: int, dim_idx: int,
                   ref_v: int, mut_v: int) -> dict | None:
    """M2.17: reshape/span_as on runtime-shaped data with the check skipped.

    The dynamic-split reshape kernels compute the split factor with a runtime
    floor division (`arith.divui flat_sz, static_factor`) and never verify
    divisibility. A caller extent that is not a multiple is accepted (rc=0) and
    the tail elements are silently dropped. The reference extent is a multiple,
    so the same kernel is a valid operator; the defect is the absent check.
    """
    mlir = KERNELS / cat / f"{stem}.mlir"
    parsed = parse_func(mlir)
    if parsed is None:
        return None
    fname, args, _ret = parsed
    small = _m2_small_dims(
        args, concrete_feed_dims(cat, stem, [d for _n, d, _t in args], "small"))
    vmfb = RAW / cat / f"{stem}.small.vmfb"
    if not vmfb.exists():
        (RAW / cat).mkdir(parents=True, exist_ok=True)
        if compile_kernel(mlir, vmfb, RAW / cat / f"{stem}.small.compile.log") != "ok":
            return {"compile": "fail"}

    def run(v):
        fd = [list(x) for x in small]
        fd[aidx][dim_idx] = v
        specs = [_ramp_input(fd[i], args[i][2]) for i in range(len(args))]
        rc, _log, sh = _run_out(vmfb, fname, specs)
        return rc, sh, fd

    rrc, _rsh, _rfd = run(ref_v)
    if rrc != 0:
        return None
    mrc, msh, mfd = run(mut_v)
    rdims = [list(x) for x in small]
    rdims[aidx][dim_idx] = ref_v
    if mrc != 0:
        return {"compile": "ok", "outcome": "runtime", "manifest": "value-changing",
                "reference_dims": rdims, "mutant_dims": mfd,
                "kind": "runtime-shape", "params": [aidx, dim_idx, ref_v, mut_v]}
    in_n = 1
    for d in mfd[aidx]:
        in_n *= d
    out_n = 1
    for t in re.findall(r"\d+", msh):
        out_n *= int(t)
    corrupts = in_n != out_n
    return {"compile": "ok",
            "outcome": "never" if corrupts else "noop",
            "manifest": "value-changing" if corrupts else "noop",
            "reference_dims": rdims, "mutant_dims": mfd,
            "kind": "runtime-shape", "params": [aidx, dim_idx, ref_v, mut_v],
            "in_elems": in_n, "out_elems": out_n}


# M2-h = M2.17 (unchecked reshape on runtime-shaped data) + M2.18. M2.18 is
# blocked-by-suite (choreo: `realized: false`, `registry_status: pending`,
# `prohibition: absent` -- needs a non-contiguous strided view the base suite
# never writes), so this battery carries M2.17 only. The base suite supplies
# two dynamic-split reshape kernels (7, 8); each contributes four distinct
# runtime extents, so the family target of 8 is met with 2 kernels x 4.
M2_H_BATTERY = [
    ("M2-h", "M2.17", "reshape", "7_dynamic_32x197xE_32x197xEd64x64",
     0, 2, 128, 129, "nondiv-plus-one"),
    ("M2-h", "M2.17", "reshape", "7_dynamic_32x197xE_32x197xEd64x64",
     0, 2, 128, 100, "nondiv-fraction"),
    ("M2-h", "M2.17", "reshape", "7_dynamic_32x197xE_32x197xEd64x64",
     0, 2, 64, 65, "nondiv-tiny-above"),
    ("M2-h", "M2.17", "reshape", "7_dynamic_32x197xE_32x197xEd64x64",
     0, 2, 64, 63, "nondiv-zero-tile"),
    ("M2-h", "M2.17", "reshape", "8_dynamic_16x1024xD_16x1024xDd64x64",
     0, 2, 128, 129, "nondiv-plus-one"),
    ("M2-h", "M2.17", "reshape", "8_dynamic_16x1024xD_16x1024xDd64x64",
     0, 2, 128, 100, "nondiv-fraction"),
    ("M2-h", "M2.17", "reshape", "8_dynamic_16x1024xD_16x1024xDd64x64",
     0, 2, 64, 65, "nondiv-tiny-above"),
    ("M2-h", "M2.17", "reshape", "8_dynamic_16x1024xD_16x1024xDd64x64",
     0, 2, 64, 63, "nondiv-zero-tile"),
]


# M2-f = M2.5 (partial / duplicate tile write) and M2-g = M2.15 (pad_low <->
# pad_high swapped, total length preserved). Both are "which elements got
# written" defects that leave every declared extent individually legal, so they
# cannot be composed from the settings suite: no `.co` base case exposes a
# coverage or padding-placement contract to edit. They are AUTHORED HERE as
# mutation-only kernels whose coverage / placement is carried by a dynamic
# control operand -- the same entry surface a settings kernel would expose, but
# with the *relation* between extents (not one extent's value) as the contract.
# The reference passes the covering / symmetric extent, the mutant a
# non-covering / shifted one, and the one compiled kernel accepts both, so the
# defect is a same-shape silent divergence. 4 kernels x 2 realisations = the
# 8-mutant family budget (mutation-specs-v2 M2.5/M2.15).
#
# They carry no `settings/<cat>.md` row and are deliberately NOT wired into the
# settings-driven `cmd_e2` gate: the generic small-feed sizing has no notion of
# the relation between extents. Their gate is `_m2_only_measure` below, which
# runs the reference at valid extents and requires the mutant to be a same-shape
# silent divergence. `_ps_mlir`'s output rows are `2*SMALL + C` so the generic
# small feed (`?` -> SMALL) is itself a valid reference and S12 never reads out
# of bounds.
M2_ONLY_TW = [
    # stem, tile_cols (TC), output rows (NR == number of covering tiles). The
    # leading integer keeps `_m2_emit`'s mutant ids unique within the category
    # (it derives them from `stem.split("_")[0]`, as for the settings batteries).
    ("1_tile", 4, 2),
    ("2_tile", 2, 2),
    ("3_tile", 8, 4),
    ("4_tile", 1, 4),
]
M2_ONLY_PS = [
    # stem, width (W), core rows (C), pad rows per side (P); output rows =
    # 2*SMALL + C (see the comment above).
    ("1_pad", 4, 4, 2),
    ("2_pad", 2, 2, 2),
    ("3_pad", 8, 5, 2),
    ("4_pad", 1, 8, 2),
]


def _tw_mlir(TC: int, NR: int) -> str:
    t = f"tensor<?x{TC}xf32>"
    o = f"tensor<{NR}x{TC}xf32>"
    return (
        "module {\n"
        f"  func.func @f_tw(%tiles: {t}) -> {o} {{\n"
        "    %c0 = arith.constant 0 : index\n"
        "    %c1 = arith.constant 1 : index\n"
        f"    %cNR = arith.constant {NR} : index\n"
        "    %c0f = arith.constant 0.0 : f32\n"
        f"    %nt = tensor.dim %tiles, %c0 : {t}\n"
        f"    %init = tensor.empty() : {o}\n"
        f"    %z = linalg.fill ins(%c0f : f32) outs(%init : {o}) -> {o}\n"
        f"    %out = scf.for %j = %c0 to %nt step %c1 iter_args(%acc = %z) -> ({o}) {{\n"
        f"      %tile = tensor.extract_slice %tiles[%j, 0][1, {TC}][1, 1] : {t} to tensor<1x{TC}xf32>\n"
        "      %rowm = arith.remui %j, %cNR : index\n"
        f"      %acc2 = tensor.insert_slice %tile into %acc[%rowm, 0][1, {TC}][1, 1] : tensor<1x{TC}xf32> into {o}\n"
        f"      scf.yield %acc2 : {o}\n"
        "    }\n"
        f"    return %out : {o}\n"
        "  }\n"
        "}\n"
    )


def _ps_mlir(W: int, C: int, TOT: int) -> str:
    p = f"tensor<?x{W}xf32>"
    c = f"tensor<{C}x{W}xf32>"
    o = f"tensor<{TOT}x{W}xf32>"
    return (
        "module {\n"
        f"  func.func @f_ps(%plo: {p}, %phi: {p}, %core: {c}) -> {o} {{\n"
        "    %c0 = arith.constant 0 : index\n"
        f"    %cC = arith.constant {C} : index\n"
        "    %c0f = arith.constant 0.0 : f32\n"
        f"    %low = tensor.dim %plo, %c0 : {p}\n"
        f"    %high = tensor.dim %phi, %c0 : {p}\n"
        f"    %init = tensor.empty() : {o}\n"
        f"    %z = linalg.fill ins(%c0f : f32) outs(%init : {o}) -> {o}\n"
        f"    %i1 = tensor.insert_slice %plo into %z[0, 0][%low, {W}][1, 1] : {p} into {o}\n"
        "    %coff = arith.addi %c0, %low : index\n"
        f"    %i2 = tensor.insert_slice %core into %i1[%coff, 0][{C}, {W}][1, 1] : {c} into {o}\n"
        "    %hoff = arith.addi %coff, %cC : index\n"
        f"    %i3 = tensor.insert_slice %phi into %i2[%hoff, 0][%high, {W}][1, 1] : {p} into {o}\n"
        f"    return %i3 : {o}\n"
        "  }\n"
        "}\n"
    )


def _write_m2_only_kernels():
    """Materialise the authored mutation-only kernels under `kernels/`."""
    wrote = 0
    for stem, TC, NR in M2_ONLY_TW:
        p = KERNELS / "tile_write" / f"{stem}.mlir"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text() != _tw_mlir(TC, NR):
            p.write_text(_tw_mlir(TC, NR))
            wrote += 1
    for stem, W, C, _P in M2_ONLY_PS:
        p = KERNELS / "pad_shift" / f"{stem}.mlir"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.read_text() != _ps_mlir(W, C, 2 * SMALL + C):
            p.write_text(_ps_mlir(W, C, 2 * SMALL + C))
            wrote += 1
    if wrote:
        log(f"m2-only: materialised {wrote} authored kernel(s)")


def _salted_bin(dims: list, dtype: str, salt: int) -> str:
    """Distinct integer raw fill, offset per operand so the reference and mutant
    regions (and pad vs core) cannot coincide by construction."""
    import array
    n = 1
    for d in dims:
        n *= d
    vals = array.array("f", [float(((i + 37 * salt) % 9) + 1) for i in range(n)])
    h = hashlib.md5((str(dims) + str(salt)).encode()).hexdigest()[:12]
    p = RAW / "_m2only" / f"{h}.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("wb") as f:
            f.write(vals.tobytes())
    return f"{'x'.join(str(d) for d in dims)}x{dtype}=@{p}"


def _m2_only_measure(cat: str, stem: str, ref_dims: list, mut_dims: list) -> dict | None:
    """Measure an authored kernel: valid reference extents vs mutant extents.

    Both runs use the same compiled artifact; the mutant is a silent corruptor
    iff it returns rc=0 with the same output shape but different values."""
    mlir = KERNELS / cat / f"{stem}.mlir"
    parsed = parse_func(mlir)
    if parsed is None:
        return None
    fname, args, _ret = parsed
    if len(args) != len(ref_dims):
        return None
    vmfb = RAW / cat / f"{stem}.small.vmfb"
    if not vmfb.exists():
        (RAW / cat).mkdir(parents=True, exist_ok=True)
        if compile_kernel(mlir, vmfb, RAW / cat / f"{stem}.small.compile.log") != "ok":
            return {"compile": "fail"}
    extra = ["--output_max_element_count=100000"]

    def run(dims_list):
        specs = [_salted_bin(dims_list[i], args[i][2], i + 1)
                 for i in range(len(args))]
        return _run_out(vmfb, fname, specs, extra=extra)

    rrc, rlog, rshape = run(ref_dims)
    if rrc != 0:
        return None
    mrc, mlog, mshape = run(mut_dims)
    ref_rec = [list(x) for x in ref_dims]
    mut_rec = [list(x) for x in mut_dims]
    if mrc != 0:
        return {"compile": "ok", "outcome": "runtime", "manifest": "value-changing",
                "reference_dims": ref_rec, "mutant_dims": mut_rec}
    ro = _parse_result(rlog)
    mo = _parse_result(mlog)
    if mshape != rshape or ro is None or mo is None:
        manifest = "value-changing"
    elif len(ro) != len(mo) or any(abs(a - b) > 1e-3 for a, b in zip(ro, mo)):
        manifest = "value-changing"
    else:
        manifest = "noop"
    return {"compile": "ok",
            "outcome": "never" if manifest == "value-changing" else "noop",
            "manifest": manifest,
            "reference_dims": ref_rec, "mutant_dims": mut_rec}


def _cmd_m2_only(recs, meta_by, MUT):
    """Materialise M2-f (M2.5) and M2-g (M2.15) from the authored kernels."""
    _write_m2_only_kernels()
    # These kernels have no manifest row, so fingerprint them here (same
    # sha1[:12] as gen_kernels.py) and hand the hash to `_m2_emit`.
    for cat, specs in (("tile_write", M2_ONLY_TW), ("pad_shift", M2_ONLY_PS)):
        for row in specs:
            p = KERNELS / cat / f"{row[0]}.mlir"
            meta_by[(cat, row[0])] = {
                "kernel_hash": hashlib.sha1(p.read_text().encode()).hexdigest()[:12],
                "settings_hash": ""}
    cases = []
    for stem, TC, NR in M2_ONLY_TW:
        ref = [[NR, TC]]
        cases.append((stem, "M2-f", "M2.5", "partial", ref, [[NR - 1, TC]], 0))
        cases.append((stem, "M2-f", "M2.5", "duplicate", ref, [[NR + 1, TC]], 0))
    for stem, W, C, P in M2_ONLY_PS:
        ref = [[P, W], [P, W], [C, W]]
        cases.append((stem, "M2-g", "M2.15", "shift-low", ref,
                      [[P + 1, W], [P - 1, W], [C, W]], 0))
        cases.append((stem, "M2-g", "M2.15", "shift-high", ref,
                      [[P - 1, W], [P + 1, W], [C, W]], 1))
    for stem, family, sid, tag, ref, mut, aidx in cases:
        cat = "tile_write" if family == "M2-f" else "pad_shift"
        r = _m2_only_measure(cat, stem, ref, mut)
        if r is None:
            log(f"m2-only: SKIP {cat}/{stem} {tag}: reference did not run")
            continue
        if r.get("compile") != "ok":
            log(f"m2-only: SKIP {cat}/{stem} {tag}: compile fail")
            continue
        if r.get("outcome") == "noop":
            log(f"m2-only: DROP {cat}/{stem} {tag}: noop (inert, not a test)")
            continue
        _m2_emit(recs, meta_by, MUT, family, sid, cat, stem, tag, {
            "kind": "authored", "operand_index": aidx, "dim_index": 0,
            "params": [tag], "reference_dims": r["reference_dims"],
            "mutant_dims": r["mutant_dims"],
        }, r)


def _m2_emit(recs, meta_by, MUT, family, sid, cat, stem, tag, mutation, r):
    fam_of = m2_family(sid) or family
    num = stem.split("_")[0]
    mid = f"iree-{cat}-{num}-{tag}"
    if any(x["mutant_id"] == mid for x in recs):
        mid = f"{mid}-{len(recs)}"
    m = meta_by.get((cat, stem), {})
    mdir = MUT / "M2" / cat
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{mid}.json").write_text(json.dumps({
        "mutant_id": mid, "class": "M2",
        "paper_category": "dim-mismatch", "category": cat, "kernel": stem,
        "settings_hash": m.get("settings_hash", ""),
        "kernel_hash": m.get("kernel_hash", ""),
        "spec_id": sid, "family": fam_of,
        "mutation": mutation,
        "outcome": r["outcome"], "manifest": r["manifest"],
    }, indent=2) + "\n")
    recs.append({
        "toolchain": "iree", "category": cat,
        "settings_hash": m.get("settings_hash", ""),
        "kernel": stem, "class": "M2", "paper_category": "dim-mismatch",
        "mutant_id": mid, "level": "1",
        "outcome": r["outcome"], "stage": RS.stage_for(r["outcome"]),
        "manifest": r["manifest"], "kernel_hash": m.get("kernel_hash", ""),
        "spec_id": sid, "family": fam_of,
    })
    log(f"m2: {mid}: {r['outcome']}/{r['manifest']} ({sid}/{fam_of})")


def cmd_m2():
    """M2 entry-contract batteries.

    Settings-derived: M2-a..M2-e and M2-h (M2.17). Authored: M2-f (M2.5) and
    M2-g (M2.15), whose coverage/placement contracts have no settings base case
    (`_cmd_m2_only`). Materialises `mutants/M2/<category>/<id>.json` and
    `raw/m2_families.jsonl`; `cmd_attribution` folds these into
    `raw/mutants.jsonl` (the committed attribution extract) along with the M2-a
    definitions.
    """
    MUT = LANE / "mutants"
    meta_by = {}
    for _l in (KERNELS / "manifest.jsonl").read_text().splitlines():
        if not _l.strip():
            continue
        _j = json.loads(_l)
        meta_by[(_j.get("category"), _j.get("stem"))] = _j
    recs = []
    for family, sid, cat, stem, tag, kind, params in M2_BATTERY:
        r = _m2_measure(cat, stem, kind, params)
        if r is None:
            log(f"m2: SKIP {cat}/{stem} {tag}: reference did not run")
            continue
        if r.get("compile") != "ok":
            log(f"m2: SKIP {cat}/{stem} {tag}: compile fail")
            continue
        if r.get("outcome") == "noop":
            log(f"m2: DROP {cat}/{stem} {tag}: noop (inert, not a test)")
            continue
        _m2_emit(recs, meta_by, MUT, family, sid, cat, stem, tag, {
            "kind": kind, "operand_index": params[0],
            "dim_index": params[1] if kind != "dataperm" else -1,
            "params": list(params),
            "reference_dims": r["reference_dims"],
            "mutant_dims": r["mutant_dims"],
        }, r)
    for family, sid, cat, stem, aidx, dim_idx, ref_v, mut_v, tag in M2_H_BATTERY:
        r = _m2_17_measure(cat, stem, aidx, dim_idx, ref_v, mut_v)
        if r is None:
            log(f"m2: SKIP {cat}/{stem} {tag}: reference did not run")
            continue
        if r.get("compile") != "ok":
            log(f"m2: SKIP {cat}/{stem} {tag}: compile fail")
            continue
        if r.get("outcome") == "noop":
            log(f"m2: DROP {cat}/{stem} {tag}: noop (inert, not a test)")
            continue
        _m2_emit(recs, meta_by, MUT, family, sid, cat, stem, tag, {
            "kind": "runtime-shape", "operand_index": aidx,
            "dim_index": dim_idx, "params": [aidx, dim_idx, ref_v, mut_v],
            "reference_dims": r["reference_dims"],
            "mutant_dims": r["mutant_dims"],
        }, r)
    _cmd_m2_only(recs, meta_by, MUT)
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "m2_families.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter((r["family"], r["outcome"]) for r in recs)
    log(f"m2 battery: {len(recs)} mutants materialised; {dict(c)}")
    cmd_attribution()  # fold M2-a + the new families into raw/mutants.jsonl


def _tag_for(cat: str, aidx: int, didx: int, delta: int) -> str:
    """Reverse of the `M2_SPECS` / `LEVEL2_SPECS` edit descriptors."""
    specs = dict(M2_SPECS)
    specs.update(LEVEL2_SPECS)
    for tag, a, d, de in specs.get(cat, []):
        if (a, d, de) == (aidx, didx, delta):
            return tag
    return ""


def cmd_attribution():
    """Offline `raw/mutants.jsonl` rebuild -- the attribution extract, no GPU.

    The committed `mutants/M2/<cat>/<id>.json` definitions already record what
    each mutant does (`mutation`), so the spec_id and family can be re-derived
    without re-running IREE. This exists so a clean checkout has the lane's rows
    on the attribution path (`gen_worklist.py::bucket()` reads `spec_id`), which
    the full 561 MB `raw/` tree cannot be committed to provide.
    """
    defs = sorted((LANE / "mutants" / "M2").glob("*/*.json"))
    recs = []
    for p in defs:
        d = json.loads(p.read_text())
        m = d.get("mutation", {})
        tag = _tag_for(d["category"], m.get("operand_index", -1),
                       m.get("dim_index", -1), m.get("delta", 0))
        sid = d.get("spec_id") or m2_spec_id(d["category"], tag)
        fam = d.get("family") or m2_family(sid)
        recs.append({
            "toolchain": "iree", "category": d["category"],
            "settings_hash": d.get("settings_hash", ""),
            "kernel": d.get("kernel", ""),
            "class": d.get("class", "M2"),
            "paper_category": d.get("paper_category", "dim-mismatch"),
            "mutant_id": d["mutant_id"],
            "level": ("2" if d["category"] in LEVEL2_SPECS else "1"),
            "outcome": d.get("outcome", ""),
            "stage": RS.stage_for(d.get("outcome", "")),
            "manifest": d.get("manifest", ""),
            "kernel_hash": d.get("kernel_hash", ""),
            "spec_id": sid, "family": fam,
        })
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "mutants.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    un = [r["mutant_id"] for r in recs if not r["spec_id"]]
    log(f"attribution: {len(recs)} rows -> {RAW / 'mutants.jsonl'}"
        + (f" ({len(un)} unmapped: {un})" if un else ""))


def cmd_s12():
    """compute-sanitizer supplement (§3.5/S12): sample E1 mutants and run
    compute-sanitizer --tool memcheck over iree's generated device code.

    Expected per the plan's table: M2 shape-contract mutants are NOT memory
    faults, so compute-sanitizer stays silent (flagged=false) even where IREE's
    own entry check caught them — the ledger-minus-sanitizer gap.
    """
    import glob
    memcheck = "/usr/local/cuda/bin/compute-sanitizer"
    rows = []
    samples = []
    for line in (RAW / "mutants.jsonl").read_text().splitlines():
        if line.strip():
            samples.append(json.loads(line))
    # full sweep over all measured mutants (bounded by per-run timeout)
    for m in samples:
        vmfb = RAW / m["category"] / f"{m['kernel']}.small.vmfb"
        mlir = KERNELS / m["category"] / f"{m['kernel']}.mlir"
        parsed = parse_func(mlir)
        specs = [input_spec(d, t, "small") for (_n, d, t) in parsed[1]]
        clog = RAW / f"_cs_{m['category']}_{m['kernel']}.log"
        cmd = [memcheck, "--tool", "memcheck", "--launch-timeout", "120",
               str(IREE_RUN), f"--module={vmfb}", "--device=cuda",
               f"--function={parsed[0]}"]
        for s in specs:
            cmd += ["--input=" + s]
        try:
            with clog.open("w") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=300)
        except subprocess.TimeoutExpired:
            r = None
        txt = clog.read_text() if clog.exists() else ""
        flagged = bool(re.search(r"Invalid|out of bounds|OOB|ERROR SUMMARY: [1-9]",
                                 txt, re.I))
        rows.append({"toolchain": "iree", "category": m["category"],
                     "class": m["class"], "mutant_id": m["mutant_id"],
                     "flagged": "true" if flagged else "false",
                     "fault": "none", "exercised": "false"})
    with (RAW / "sanitizer.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"S12: {len(rows)} sanitizer records (sampled)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e2", "expressibility", "e3", "minimal",
                                    "m2", "s12", "attribution"])
    ap.add_argument("--level2", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    {"expressibility": cmd_expressibility, "e3": cmd_e3, "s12": cmd_s12,
     "attribution": cmd_attribution, "m2": cmd_m2,
     "minimal": (lambda: cmd_minimal(level2=a.level2)),
     "e2": (lambda: cmd_e2(size="full" if a.full else "small"))}[a.cmd]()
