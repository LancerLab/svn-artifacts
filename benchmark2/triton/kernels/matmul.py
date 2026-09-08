"""matmul — Triton composition from benchmark2/settings/matmul.md.
y = lhs @ rhs over the inner dim K. Tiled tl.dot kernel with fixed tiles.
Host side uses ../gpubuf.py (no torch).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK * N
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: GpuBuf, b: GpuBuf, M: int, K: int, N: int,
           bm: int = 64, bn: int = 64, bk: int = 32) -> GpuBuf:
    c = GpuBuf(M * N)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    matmul_kernel[grid](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk)
    sync()
    return c


def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    M, K, N = (SMALL if a.size == "small" else FULL)["matmul"]
    aa = randn(M * K); bb = randn(K * N, seed=1)
    got = matmul(GpuBuf.from_numpy(aa), GpuBuf.from_numpy(bb), M, K, N
                 ).to_host()
    want = (aa.reshape(M, K) @ bb.reshape(K, N)).ravel()
    assert np.allclose(got, want, atol=1e-1, rtol=1e-2), (M, K, N)
    print("ok", a.size, (M, K, N))
