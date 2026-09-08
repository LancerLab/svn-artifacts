"""softmax — Triton composition from benchmark2/settings/softmax.md.
y = exp(x - max) / sum(exp(x - max)) over the trailing dim. Row-wise kernel.
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
def softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


def softmax(x: GpuBuf, n_rows: int, n_cols: int) -> GpuBuf:
    y = GpuBuf(x.n)
    block = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK=block)
    sync()
    return y


def reference(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    rows, cols = (SMALL if a.size == "small" else FULL)["softmax"]
    x = randn(rows * cols)
    got = softmax(GpuBuf.from_numpy(x), rows, cols).to_host()
    want = reference(x.reshape(rows, cols)).ravel()
    assert np.allclose(got, want, atol=1e-4), (rows, cols)
    print("ok", a.size, (rows, cols))
