"""relu — Triton composition from benchmark2/settings/relu.md.
y = max(x, 0); elementwise, output shape = input shape.
Reference-checked against a numpy host reference (no torch on this machine;
see ../gpubuf.py).
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
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def relu(x: GpuBuf) -> GpuBuf:
    y = GpuBuf(x.n)
    grid = (triton.cdiv(x.n, 1024),)
    relu_kernel[grid](x, y, x.n, BLOCK=1024)
    sync()
    return y


def reference(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    n = (SMALL if a.size == "small" else FULL)["relu"][0]
    x = randn(n)
    got = relu(GpuBuf.from_numpy(x)).to_host()
    assert np.allclose(got, np.maximum(x, 0.0)), n
    print("ok", a.size, n)
