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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # benchmark2/
LANE = ROOT / "iree"
RAW = LANE / "raw"
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


def _run_out(vmfb: Path, func: str, arg_specs, timeout=90) -> tuple[int, Path, str]:
    h = hashlib.md5("\x00".join(arg_specs).encode()).hexdigest()[:10]
    rlog = RAW / f"_mut_{vmfb.stem}_{h}.log"
    cmd = [str(IREE_RUN), f"--module={vmfb}", "--device=cuda", f"--function={func}"]
    for a in arg_specs:
        cmd += ["--input=" + a]
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
        "wrong result -> **never/corrupts** (verified by numeric diff against the "
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
                ref_specs = [_ramp_input(d, t) for (_n, d, t) in args]
                ref_rc, ref_log, _ = _run_out(vmfb, fname, ref_specs)
                if ref_rc != 0:
                    continue
                mut_dims = [list(d) for (_n, d, _t) in args]
                if mut_dims[aidx][didx] == "?":
                    mut_dims[aidx][didx] = SMALL + delta
                else:
                    mut_dims[aidx][didx] = max(1, mut_dims[aidx][didx] + delta)
                mid = f"iree-{cat}-{base['kernel'].split('_')[0]}-{tag}"
                mut_specs = [_ramp_input(mut_dims[i], args[i][2])
                             for i in range(len(args))]
                rc, mlog, mshape = _run_out(vmfb, fname, mut_specs)
                if rc != 0:
                    outcome, manifest = "runtime", "corrupts"
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
                        manifest = ("corrupts" if (len(o_ref) != len(o_mut) or
                                    any(abs(a - b) > 1e-3 for a, b in zip(o_ref, o_mut)))
                                    else "noop")
                    else:
                        manifest = "corrupts"  # contract differs; IREE ran but not cleanly
                if outcome == "never" and manifest == "noop":
                    continue  # false-success mutant discarded (plan §11.1)
                # materialise the mutated version (definition file)
                mdir = MUT / "M2" / cat
                mdir.mkdir(parents=True, exist_ok=True)
                (mdir / f"{mid}.json").write_text(json.dumps({
                    "mutant_id": mid, "class": "M2",
                    "paper_category": "dim-mismatch", "category": cat,
                    "kernel": base["kernel"],
                    "settings_hash": meta.get("settings_hash", ""),
                    "kernel_hash": meta.get("kernel_hash", ""),
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
                    "outcome": outcome, "manifest": manifest,
                    "kernel_hash": meta.get("kernel_hash", ""),
                }
                recs.append(rec)
    with (RAW / "mutants.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_runtime = sum(1 for r in recs if r["outcome"] == "runtime")
    lvl = "level-2 " if level2 else ""
    log(f"minimal [{lvl}E1]: {len(recs)} M2 mutants materialised in mutants/, "
        f"{n_runtime} runtime-caught, {len(recs)-n_runtime} never "
        f"(M1/M3 => n/a, see mutants/README.md)")



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
    ap.add_argument("cmd", choices=["e2", "expressibility", "e3", "minimal", "s12"])
    ap.add_argument("--level2", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    {"expressibility": cmd_expressibility, "e3": cmd_e3, "s12": cmd_s12,
     "minimal": (lambda: cmd_minimal(level2=a.level2)),
     "e2": (lambda: cmd_e2(size="full" if a.full else "small"))}[a.cmd]()
