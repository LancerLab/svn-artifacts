#!/usr/bin/env python3
"""benchmark2/iree/semcheck.py — value-level semantic correctness check.

Runs each operator template (scripts/gen_iree_cases.py) on the device with
deterministic non-trivial inputs and compares the result against a NumPy
reference that mirrors the generator's exact semantics. This is the *numeric*
oracle the structural gate does not provide.

Output: raw/semcheck.jsonl — {op, shapes, max_abs_err, pass}
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
IREE_BIN = Path("/home/gxf/.tools/iree-dev-20260908-venv/bin")
IREE_COMPILE = IREE_BIN / "iree-compile"
IREE_RUN = IREE_BIN / "iree-run-module"
TMP = RAW / "_semcheck"
TMP.mkdir(parents=True, exist_ok=True)


def compile_mlir(text: str, name: str) -> Path:
    mlir = TMP / f"{name}.mlir"
    vmfb = TMP / f"{name}.vmfb"
    mlir.write_text(text)
    r = subprocess.run([str(IREE_COMPILE), str(mlir),
                        "--iree-hal-target-backends=cuda",
                        "--iree-cuda-target=sm_120",
                        "--iree-cuda-target-features=+ptx87",
                        "-o", str(vmfb)],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile fail {name}: {r.stderr.decode()[:300]}")
    return vmfb


def run_func(vmfb: Path, func: str, arrays: dict) -> np.ndarray:
    cmd = [str(IREE_RUN), f"--module={vmfb}", "--device=cuda", f"--function={func}"]
    for nm, av in arrays.items():
        a = av[0] if isinstance(av, tuple) else av
        dt = av[1] if isinstance(av, tuple) else "f32"
        shape = "x".join(str(x) for x in a.shape)
        vals = ",".join(str(int(v)) if dt != "f32" else str(float(v))
                        for v in a.flatten())
        cmd += [f"--input={shape}x{dt}={vals}"]
    out = TMP / f"{vmfb.stem}.out"
    r = subprocess.run(cmd, capture_output=True)
    text = r.stdout.decode()
    if r.returncode != 0:
        raise RuntimeError(f"run fail {func}: {r.stderr.decode()[:300]}")
    m = re.search(r"result\[0\]: hal\.buffer_view\s*\n", text)
    seg = text[m.end():] if m else text
    eq = seg.find("=")
    vals = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", seg[eq + 1:])
    return np.array([float(v) for v in vals], dtype=np.float64)


# --- numpy references mirroring the generator semantics ---------------------
def ref_relu(x): return np.maximum(x, 0)
def ref_sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def ref_gelu(x):
    from math import erf
    e = np.array([erf(float(v) * 0.7071067811865476) for v in x.flatten()])
    return (0.5 * x * (1.0 + e.reshape(x.shape)))
def ref_elemwise(a, b): return a + b          # same-rank case
def ref_matmul(a, b): return a @ b
def ref_transpose(x): return x.transpose(0, 2, 1)
def ref_concat(a, b, axis=1): return np.concatenate([a, b], axis=axis)
def ref_batchnorm(x, gamma, beta, axis=1):
    return x * gamma.reshape([-1 if i == axis else 1 for i in range(x.ndim)]) + \
           beta.reshape([-1 if i == axis else 1 for i in range(x.ndim)])
def ref_layernorm(x, gamma, beta, norm_rank, eps=1e-5):
    batch_rank = x.ndim - norm_rank
    axes = tuple(range(batch_rank, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = (x * x).mean(axis=axes, keepdims=True) - mean * mean
    istd = 1.0 / np.sqrt(var + eps)
    gshape = [1] * x.ndim; bshape = [1] * x.ndim
    for i in range(norm_rank):
        gshape[batch_rank + i] = gamma.shape[i]
        bshape[batch_rank + i] = beta.shape[i]
    return (x - mean) * istd * gamma.reshape(gshape) + beta.reshape(bshape)
def ref_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
def ref_reduce_mean(x, axis=1):
    return x.mean(axis=axis, keepdims=False)
def ref_reshape(x, shape):
    return x.reshape(shape)
def ref_maxpool(x, stride=2):
    n, c, h, w = x.shape
    oh, ow = h // stride, w // stride
    out = np.zeros((n, c, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i*stride:(i+1)*stride,
                                 j*stride:(j+1)*stride].max(axis=(2, 3))
    return out
def ref_conv2d(x, f, stride=1):
    n, ci, h, w = x.shape
    co, ck, kh, kw = f.shape
    oh, ow = h - kh + 1, w - kw + 1
    out = np.zeros((n, co, oh, ow), dtype=np.float64)
    for nn in range(n):
        for fo in range(co):
            for oi in range(oh):
                for oj in range(ow):
                    out[nn, fo, oi, oj] = np.sum(
                        x[nn, :, oi:oi+kh, oj:oj+kw] * f[fo])
    return out


def check(op: str, code: str, func: str, arrays: dict[str, np.ndarray],
          ref: np.ndarray, tol=1e-3) -> dict:
    vmfb = compile_mlir(code, op)
    got = run_func(vmfb, func, arrays)
    ref_flat = np.asarray(ref).flatten()
    if got.shape != ref_flat.shape:
        return {"op": op, "pass": False, "max_abs_err": None,
                "note": f"shape {got.shape} vs {ref_flat.shape}"}
    err = float(np.max(np.abs(got - ref_flat)))
    return {"op": op, "pass": bool(err <= tol), "max_abs_err": err}


def main():
    rng = np.random.default_rng(0)
    X = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    Y = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    Xt = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    A2 = (rng.random((4, 5)) * 2 - 1).astype(np.float32)
    B2 = (rng.random((5, 6)) * 2 - 1).astype(np.float32)
    LN = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    g3 = (rng.random((4,)) * 2).astype(np.float32)
    b3 = (rng.random((4,)) * 2).astype(np.float32)
    C1 = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    C2 = (rng.random((2, 5, 4)) * 2 - 1).astype(np.float32)
    Yb = (rng.random((2, 1, 1)) * 2 - 1).astype(np.float32)
    BN = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    bng = (rng.random((3,)) * 2).astype(np.float32)
    bnb = (rng.random((3,)) * 2).astype(np.float32)
    SM = (rng.random((2, 3, 4)) * 2 - 1).astype(np.float32)
    Xp = (rng.random((1, 2, 4, 4)) * 2 - 1).astype(np.float32)
    Xc = (rng.random((1, 2, 5, 5)) * 2 - 1).astype(np.float32)
    Fc = (rng.random((3, 2, 3, 3)) * 2 - 1).astype(np.float32)
    Idx = (rng.integers(0, 5, (2, 3))).astype(np.int64)
    Tbl = (rng.random((5, 4)) * 2 - 1).astype(np.float32)
    Xe = (rng.random((2, 8, 8)) * 2 - 1).astype(np.float32)
    Xc2 = (rng.random((2, 3, 4, 4)) * 2 - 1).astype(np.float32)
    Xp2 = (rng.random((2, 3, 4, 5)) * 2 - 1).astype(np.float32)

    t = lambda *a: tuple(int(x) if isinstance(x, (int, np.integer)) else "?" for x in a)
    cases = [
        ("relu", g.gen_relu("t_relu", [[2, 3, 4]]), "t_relu",
         {"input": X}, ref_relu(X)),
        ("sigmoid", g.gen_sigmoid("t_sig", [[2, 3, 4]]), "t_sig",
         {"input": X}, ref_sigmoid(X)),
        ("gelu", g.gen_gelu("t_gelu", [[2, 3, 4]]), "t_gelu",
         {"input": X}, ref_gelu(X)),
        ("elemwise_add", g.gen_elemwise_add("t_add", [[2, 3, 4], [2, 3, 4], [2, 3, 4]]),
         "t_add", {"lhs": X, "rhs": Y}, ref_elemwise(X, Y)),
        ("elemwise_add_bcast", g.gen_elemwise_add("t_addb", [[2, 3, 4], [2, 1, 1], [2, 3, 4]]),
         "t_addb", {"lhs": X, "rhs": Yb}, ref_elemwise(X, np.broadcast_to(Yb, (2, 3, 4)))),
        ("matmul", g.gen_matmul("t_mm", [[4, 5], [5, 6], [4, 6]]),
         "t_mm", {"lhs": A2, "rhs": B2}, ref_matmul(A2, B2)),
        ("transpose", g.gen_transpose("t_tr", [[2, 3, 4], [2, 4, 3]]),
         "t_tr", {"input": Xt}, ref_transpose(Xt)),
        ("concat", g.gen_concat("t_cat", [[2, 3, 4], [2, 5, 4], [2, 8, 4]]),
         "t_cat", {"in0": C1, "in1": C2}, ref_concat(C1, C2, axis=1)),
        ("batch_norm", g.gen_batch_norm("t_bn", [[2, 3, 4], [3], [3], [2, 3, 4]]),
         "t_bn", {"input": BN, "gamma": bng, "beta": bnb},
         ref_batchnorm(BN, bng, bnb, axis=1)),
        ("layer_norm", g.gen_layer_norm("t_ln", [[2, 3, 4], [4], [4]]),
         "t_ln", {"input": LN, "gamma": g3, "beta": b3},
         ref_layernorm(LN, g3, b3, norm_rank=1)),
        ("softmax", g.gen_softmax("t_sm", [[2, 3, 4]]),
         "t_sm", {"input": SM}, ref_softmax(SM)),

        ("reduce_mean", g.gen_reduce_mean("t_rm", [[2, 3, 4], [2, 4]]),
         "t_rm", {"input": X}, ref_reduce_mean(X, axis=1)),
        ("reshape", g.gen_reshape("t_rs", [[2, 3, 4], [2, 12]]),
         "t_rs", {"input": X}, ref_reshape(X, (2, 12))),
        ("reshape_expand", g.gen_reshape("t_rse", [[2, 8, 8], [2, 2, 8, 4]]),
         "t_rse", {"input": Xe}, ref_reshape(Xe, (2, 2, 8, 4))),
        ("reshape_coll", g.gen_reshape("t_rsc", [[2, 3, 4, 4], [2, 6, 8]]),
         "t_rsc", {"input": Xc2}, ref_reshape(Xc2, (2, 6, 8))),
        ("reshape_perm", g.gen_reshape("t_rsp", [[2, 3, 4, 5], [2, 4, 5, 3]]),
         "t_rsp", {"input": Xp2}, Xp2.transpose(0, 2, 3, 1)),
        ("max_pool2d", g.gen_max_pool2d("t_mp", [[1, 2, 4, 4], [1, 2, 2, 2]]),
         "t_mp", {"input": Xp}, ref_maxpool(Xp, stride=2)),
        ("conv2d", g.gen_conv2d("t_cv", [[1, 2, 5, 5], [3, 2, 3, 3], [1, 3, 3, 3]]),
         "t_cv", {"input": Xc, "filter": Fc}, ref_conv2d(Xc, Fc, stride=1)),
        ("embedding", g.gen_embedding("t_emb", [[2, 3], [5, 4], [2, 3, 4]]),
         "t_emb", {"indices": (Idx, "i64"), "table": Tbl},
         Tbl[Idx.astype(int)]),
    ]

    rows = []
    for op, code, func, arrays, ref in cases:
        if code is None:
            rows.append({"op": op, "pass": False, "max_abs_err": None,
                         "note": "generator returned None"})
            continue
        try:
            rows.append(check(op, code, func, arrays, ref))
        except Exception as e:  # noqa: BLE001
            rows.append({"op": op, "pass": False, "max_abs_err": None,
                         "note": str(e)[:200]})
    out = RAW / "semcheck.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for r in rows:
        print(f"  {r['op']:16s} pass={r['pass']}  err={r['max_abs_err']}"
              + (f"  ({r['note']})" if r.get("note") else ""))
    n_ok = sum(1 for r in rows if r["pass"])
    print(f"[iree/semcheck] {n_ok}/{len(rows)} operators numerically correct")


if __name__ == "__main__":
    main()
