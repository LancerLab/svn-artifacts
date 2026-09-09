#!/usr/bin/env python3
"""sm_120 codegen probe for the tilelang lane (feasibility gate F2)."""
import sys

import numpy as np
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def add_kernel(M, N, block_M, block_N, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = (
                    A[by * block_M + i, bx * block_N + j]
                    + B[by * block_M + i, bx * block_N + j])
    return main


def main():
    import torch
    M, N = 32, 64
    k = add_kernel(M, N, 32, 64)
    a = torch.ones(M, N, device="cuda")
    b = torch.ones(M, N, device="cuda")
    c = k(a, b)  # out_idx=[-1] => output allocated by the framework
    torch.cuda.synchronize()
    want = (a + b).cpu().numpy()
    got = c.cpu().numpy()
    ok = np.allclose(want, got, atol=1e-4)
    print(f"[tilelang/probe] sm_120 add compile+run: {'OK' if ok else 'FAIL'} "
          f"max_err={float(np.max(np.abs(want - got))):.2e}")
    print(f"[tilelang/probe] torch cuda arch: {torch.cuda.get_device_capability()}")


if __name__ == "__main__":
    main()
