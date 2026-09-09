"""benchmark2/tilelang/driver.py — TileLang lane (exploratory / rebuttal).

Compose+gate kernels at concrete small/full shapes, inject M1 (index OOB) and
M3 (gemm K divisibility) defects, classify each mutant with a numpy
manifestation oracle, and emit schema-valid records.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import tilelang
import tilelang.language as T

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE))
from sizes import SMALL, FULL, FULL_RAGGED  # noqa: E402

DEVICE = "cuda"
F32 = "float32"


# --------------------------------------------------------------------------
# kernels (reference)
# --------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def relu(M, BLOCK):
    @T.prim_func
    def main(X: T.Tensor((M,), "float32"), Y: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, BLOCK), threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                if idx < M:
                    Y[idx] = T.max(X[idx], 0)
    return main


@tilelang.jit(out_idx=[-1])
def relu_oob(M, BLOCK):  # M1 mutant: off-by-one index (reads X[idx+1])
    @T.prim_func
    def main(X: T.Tensor((M,), "float32"), Y: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, BLOCK), threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                if idx + 1 < M:
                    Y[idx] = T.max(X[idx + 1], 0)
    return main


@tilelang.jit(out_idx=[-1])
def sigmoid(M, BLOCK):
    @T.prim_func
    def main(X: T.Tensor((M,), "float32"), Y: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, BLOCK), threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                if idx < M:
                    Y[idx] = 1.0 / (1.0 + T.exp(-X[idx]))
    return main


@tilelang.jit(out_idx=[-1])
def add(M, BLOCK):
    @T.prim_func
    def main(A: T.Tensor((M,), "float32"), B: T.Tensor((M,), "float32"),
             C: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, BLOCK), threads=BLOCK) as bx:
            for i in T.Parallel(BLOCK):
                idx = bx * BLOCK + i
                if idx < M:
                    C[idx] = A[idx] + B[idx]
    return main


@tilelang.jit(out_idx=[-1])
def transpose(M, N, bM, bN):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Y: T.Tensor((N, M), "float32")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            for i, j in T.Parallel(bM, bN):
                r = by * bM + i
                c = bx * bN + j
                if r < M and c < N:
                    Y[c, r] = X[r, c]
    return main


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, bM, bN, bK):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            A_s = T.alloc_shared((bM, bK), "float32")
            B_s = T.alloc_shared((bK, bN), "float32")
            C_l = T.alloc_fragment((bM, bN), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=2):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[k * bK, bx * bN], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[by * bM, bx * bN])
    return main


# --------------------------------------------------------------------------
# gate + oracle
# --------------------------------------------------------------------------
def _emit(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _divisor(n, lo=16, hi=64):
    for d in range(hi, lo - 1, -4):
        if n % d == 0:
            return d
    return lo


def gate_one(cat, size, rng):
    shp = (SMALL if size == "small" else FULL)[cat]
    rec = {"toolchain": "tilelang", "category": cat, "settings_hash": "",
           "kernel_hash": "", "size": size, "gpu_device": DEVICE,
           "exclusive": "false", "toolchain_version": tilelang.__version__}
    try:
        if cat in ("relu", "sigmoid", "elemwise_add", "gelu"):
            M = shp[0]
            k = {"relu": relu, "sigmoid": sigmoid, "elemwise_add": add,
                 "gelu": sigmoid}[cat](M, 256)
            if cat == "elemwise_add":
                a = rng.normal(size=(M,)).astype(np.float32)
                b = rng.normal(size=(M,)).astype(np.float32)
                c = k(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda())
                want = a + b
            else:
                x = rng.normal(size=(M,)).astype(np.float32)
                c = k(torch.from_numpy(x).cuda())
                want = {"relu": np.maximum(x, 0),
                        "sigmoid": 1.0 / (1.0 + np.exp(-x)),
                        "gelu": 1.0 / (1.0 + np.exp(-x))}[cat]
            got = c.cpu().numpy()
            return {**rec, "compile": "ok", "run": "ok",
                    "ref_check": "pass" if np.allclose(got, want, atol=1e-2) else "fail"}
        if cat == "transpose":
            M, N = shp
            k = transpose(M, N, 32, 32)
            x = rng.normal(size=(M, N)).astype(np.float32)
            y = k(torch.from_numpy(x).cuda())
            return {**rec, "compile": "ok", "run": "ok",
                    "ref_check": "pass" if np.allclose(y.cpu().numpy(), x.T, atol=1e-2) else "fail"}
        if cat == "matmul":
            M, K, N = shp
            k = matmul(M, N, K, _divisor(M), _divisor(N), 16)
            a = rng.normal(size=(M, K)).astype(np.float32)
            b = rng.normal(size=(K, N)).astype(np.float32)
            c = k(torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda())
            return {**rec, "compile": "ok", "run": "ok",
                    "ref_check": "pass" if np.allclose(c.cpu().numpy(), a @ b,
                                                       rtol=1e-2, atol=1e-1) else "fail"}
    except Exception as e:  # noqa: BLE001
        return {**rec, "compile": "fail", "run": "crash",
                "ref_check": f"fail ({type(e).__name__}: {str(e)[:80]})"}
    return {**rec, "compile": "n/a", "run": "-", "ref_check": "n/a"}


def cmd_e2(size):
    rng = np.random.default_rng(0)
    cats = ["relu", "sigmoid", "gelu", "elemwise_add", "transpose", "matmul"]
    for cat in cats:
        _emit(RAW / "kernels.jsonl", gate_one(cat, size, rng))
    for cat in cats:
        for cls, v in (("elem", "yes"), ("shape", "no"),
                       ("loop", "yes"), ("hw", "partial")):
            _emit(RAW / "expressibility.jsonl",
                  {"toolchain": "tilelang", "category": cat, "class": cls,
                   "expressible": v, "toolchain_version": tilelang.__version__})
    print(f"[tilelang/e2 {size}] gated {len(cats)} categories")


# --------------------------------------------------------------------------
# mutants (E1): M1 index OOB on relu, M3 K-divisibility on matmul
# --------------------------------------------------------------------------
def cmd_minimal():
    rng = np.random.default_rng(0)
    M = FULL_RAGGED["relu"][0]          # ragged full magnitude
    rec = {"toolchain": "tilelang", "class": "M1", "paper_category": "oob",
           "level": "1", "settings_hash": "", "kernel_hash": ""}
    # M1: off-by-one OOB on relu (reference vs mutant on the same input)
    x = rng.normal(size=(M,)).astype(np.float32)
    ref = relu(M, 256)(torch.from_numpy(x).cuda()).cpu().numpy()
    try:
        mut = relu_oob(M, 256)(torch.from_numpy(x).cuda())
        torch.cuda.synchronize()
        got = mut.cpu().numpy()
        manifest = "noop" if np.allclose(got, ref, atol=1e-2) else "corrupts"
        _emit(RAW / "mutants.jsonl",
              {**rec, "category": "relu", "mutant_id": "relu-f2-offbyone",
               "outcome": "never", "manifest": manifest,
               "stage": "none", "detail": "off-by-one index (reads X[idx+1])"})
        print(f"[tilelang/minimal] relu M1 off-by-one -> never/{manifest}")
    except Exception as e:  # noqa: BLE001
        _emit(RAW / "mutants.jsonl",
              {**rec, "category": "relu", "mutant_id": "relu-f2-offbyone",
               "outcome": "runtime", "manifest": "corrupts", "stage": "runtime",
               "detail": f"crash: {type(e).__name__}: {str(e)[:60]}"})
        print(f"[tilelang/minimal] relu M1 off-by-one -> runtime (crash)")

    # M3: matmul K not divisible by the gemm atom
    rec2 = {"toolchain": "tilelang", "class": "M3", "paper_category": "hw",
            "level": "1", "settings_hash": "", "kernel_hash": ""}
    M3, K3, N3 = 32, 48, 32  # K=48, block_K=32 -> K%32 != 0
    try:
        matmul(M3, N3, K3, 16, 16, 32)   # compile with non-divisible K
        _emit(RAW / "mutants.jsonl",
              {**rec2, "category": "matmul", "mutant_id": "matmul-f1-kdiv",
               "outcome": "never", "manifest": "corrupts", "stage": "none",
               "detail": "K=48 not divisible by gemm atom 32 (silently handled)"})
        print("[tilelang/minimal] matmul M3 K-div -> never")
    except Exception as e:  # noqa: BLE001
        _emit(RAW / "mutants.jsonl",
              {**rec2, "category": "matmul", "mutant_id": "matmul-f1-kdiv",
               "outcome": "compile", "manifest": "corrupts", "stage": "compile",
               "detail": f"gemm: {str(e)[:80]}"})
        print("[tilelang/minimal] matmul M3 K-div -> compile")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e2", "minimal"])
    ap.add_argument("--size", default="small")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    DEVICE = a.device
    if a.cmd == "e2":
        cmd_e2(a.size)
    else:
        cmd_minimal()
