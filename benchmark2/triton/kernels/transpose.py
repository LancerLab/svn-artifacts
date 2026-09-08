"""transpose — Triton composition from benchmark2/settings/transpose.md.
y = x.swapaxes(-1, -2) on the trailing two dims of a [B, M, N] input.
Host side uses ../gpubuf.py (no torch).
"""
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def transpose_kernel(x_ptr, y_ptr, B, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x_ptrs = x_ptr + pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    tile = tl.load(x_ptrs, mask=mask_mn)
    y_ptrs = y_ptr + pid_b * M * N + offs_n[:, None] * M + offs_m[None, :]
    mask_nm = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptrs, tl.trans(tile), mask=mask_nm)


def transpose(x: GpuBuf, B: int, M: int, N: int) -> GpuBuf:
    y = GpuBuf(x.n)
    grid = (B, triton.cdiv(M, 32), triton.cdiv(N, 32))
    transpose_kernel[grid](x, y, B, M, N, BLOCK_M=32, BLOCK_N=32)
    sync()
    return y


def reference(x: np.ndarray) -> np.ndarray:
    return np.swapaxes(x, -1, -2)


if __name__ == "__main__":
    for shape in [(32, 197, 768), (64, 100, 255), (16, 64, 33)]:
        B, M, N = shape
        x = randn(B * M * N)
        got = transpose(GpuBuf.from_numpy(x), B, M, N).to_host()
        want = reference(x.reshape(shape)).ravel()
        assert np.array_equal(got, want), (shape, "MISMATCH")
        print("ok", shape)
