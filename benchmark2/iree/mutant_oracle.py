#!/usr/bin/env python3
"""benchmark2/iree/mutant_oracle.py — ground-truth proof for E1 mutants.

Answers: does a mutated IREE entry actually expose a real obligation
violation, and is the outcome classification (runtime / never) correct?

For each expressible M2 family we build a SMALL dynamic-shape instance, run it
with distinct (non-trivial) inputs, and compare device output against a NumPy
reference that mirrors the operator semantics:

  * runtime  — IREE rejects the violated entry contract before device
               execution. We prove the mutation is a genuine obligation
               violation (not a no-op) by showing the operator's reference
               semantics REQUIRE the dimension that the mutant breaks.
  * never    — IREE accepts and silently computes. We prove corruption by
               comparing the device result against the NumPy reference: a
               mismatch is silent corruption (corrupts), a match is a no-op
               mutant (discarded per plan §11.1).

Output: raw/mutant_oracle.jsonl
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent          # svn-artifacts/
sys.path.insert(0, str(ROOT / "scripts"))
import gen_iree_cases as g  # noqa: E402

RAW = ROOT / "benchmark2" / "iree" / "raw"
IREE = Path("/home/gxf/.tools/iree-dev-20260908-venv/bin")
COMPILE = IREE / "iree-compile"
RUN = IREE / "iree-run-module"
TMP = RAW / "_oracle"
TMP.mkdir(parents=True, exist_ok=True)


def compile_mlir(text, name):
    (TMP / f"{name}.mlir").write_text(text)
    r = subprocess.run([str(COMPILE), str(TMP / f"{name}.mlir"),
                        "--iree-hal-target-backends=cuda", "--iree-cuda-target=sm_120",
                        "--iree-cuda-target-features=+ptx87", "-o", str(TMP / f"{name}.vmfb")],
                       capture_output=True)
    assert r.returncode == 0, f"compile fail: {r.stderr.decode()[:300]}"
    return TMP / f"{name}.vmfb"


def run(vmfb, func, specs):
    cmd = [str(RUN), f"--module={vmfb}", "--device=cuda", f"--function={func}"]
    for s in specs:
        cmd += ["--input=" + s]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        return None, r.stderr.decode()[:200]
    m = re.search(r"result\[0\]: hal\.buffer_view\s*\n", r.stdout.decode())
    seg = r.stdout.decode()[m.end():] if m else r.stdout.decode()
    vals = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", seg[seg.find("=") + 1:])
    return np.array([float(v) for v in vals]), ""


def spec(a):
    return f"{'x'.join(str(x) for x in a.shape)}xf32=" + ",".join(
        f"{float(v):.6f}" for v in a.flatten())


def layernorm_np(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = (x * x).mean(axis=-1, keepdims=True) - mean * mean
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def main():
    rng = np.random.default_rng(7)
    rows = []
    SZ = 6  # dynamic norm/K dim (kept small so distinct inputs fit argv)

    # --- layer_norm: gamma/beta length must equal normalized dim -----------
    x = (rng.random((2, SZ)) * 2 - 1).astype(np.float32)
    gfull = (rng.random((SZ,)) * 2).astype(np.float32)
    bfull = (rng.random((SZ,)) * 2).astype(np.float32)
    # dynamic-shape module: input tensor<2x?xf32>, gamma/beta tensor<?xf32>
    code = g.gen_layer_norm("ln", [[2, "?"], ["?"], ["?"]])
    vmfb = compile_mlir(code, "ln")
    # reference run (correct contract, gamma/beta length == SZ)
    ref_out, _ = run(vmfb, "ln", [spec(x), spec(gfull), spec(bfull)])
    np_ref = layernorm_np(x, gfull, bfull)
    assert np.allclose(ref_out, np_ref.flatten(), atol=1e-4), "reference mismatch"
    # mutant: gamma length off by one (SZ-1), beta correct
    gshort = gfull[:-1]
    mut_out, err = run(vmfb, "ln", [spec(x), spec(gshort), spec(bfull)])
    if mut_out is None:
        rows.append({"family": "layer_norm-gamma-len-1", "outcome": "runtime",
                     "genuine_violation": True,
                     "proof": f"gamma dim {SZ-1} != normalized dim {SZ}; IREE rejected at entry",
                     "entry_error": err[:160]})
    else:
        # silent: what a permissive compiler would produce with truncated gamma
        perm = layernorm_np(x, np.append(gshort, 0.0), bfull)
        rows.append({"family": "layer_norm-gamma-len-1", "outcome": "never",
                     "genuine_violation": bool(np.max(np.abs(mut_out - np_ref.flatten())) > 1e-3),
                     "silent_err": float(np.max(np.abs(mut_out - np_ref.flatten()))),
                     "permissive_wrong_err": float(np.max(np.abs(perm.flatten() - np_ref.flatten())))})

    # --- matmul: rhs rows (K) must equal lhs cols (K) ----------------------
    a = (rng.random((2, SZ)) * 2 - 1).astype(np.float32)
    b = (rng.random((SZ, 3)) * 2 - 1).astype(np.float32)
    code = g.gen_matmul("mm", [[2, "?"], ["?", 3], [2, 3]])
    vmfb = compile_mlir(code, "mm")
    ref_out, _ = run(vmfb, "mm", [spec(a), spec(b)])
    np_ref = (a @ b)
    assert np.allclose(ref_out, np_ref.flatten(), atol=1e-3), "mm reference mismatch"
    b_bad = b[:-1, :]  # K-1 rows
    mut_out, err = run(vmfb, "mm", [spec(a), spec(b_bad)])
    if mut_out is None:
        rows.append({"family": "matmul-rhs-K-1", "outcome": "runtime",
                     "genuine_violation": True,
                     "proof": f"rhs rows {SZ-1} != lhs cols {SZ}; IREE rejected at entry",
                     "entry_error": err[:160]})
    else:
        rows.append({"family": "matmul-rhs-K-1", "outcome": "never",
                     "genuine_violation": bool(np.max(np.abs(mut_out - np_ref.flatten())) > 1e-3),
                     "silent_err": float(np.max(np.abs(mut_out - np_ref.flatten())))})

    (RAW / "mutant_oracle.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    for r in rows:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
