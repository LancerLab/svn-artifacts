"""relu — Triton composition from benchmark2/settings/relu.md.
y = max(x, 0); elementwise, output shape = input shape.
Reference-checked against a numpy host reference (no torch on this machine;
see ../gpubuf.py).
"""
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
    for shape in [(64 * 1280 * 7 * 7,), (32 * 197 * 768,), (127 * 1023,)]:
        x = randn(shape[0])
        got = relu(GpuBuf.from_numpy(x)).to_host()
        want = reference(x)
        assert np.allclose(got, want), (shape, "MISMATCH")
        print("ok", shape)
