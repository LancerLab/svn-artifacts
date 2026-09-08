"""layer_normalization — Triton composition from
benchmark2/settings/layer_normalization.md.
y = (x - mean)/sqrt(var+eps) * scale + bias; mean/var over the trailing dim.
Row-wise kernel (one program per row). Host side uses ../gpubuf.py (no torch).
"""
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def layer_norm_kernel(x_ptr, scale_ptr, bias_ptr, y_ptr, n_cols, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, diff * rstd * scale + bias,
             mask=mask)


def layer_norm(x: GpuBuf, scale: GpuBuf, bias: GpuBuf, n_rows: int,
               n_cols: int, eps: float = 1e-5) -> GpuBuf:
    y = GpuBuf(x.n)
    block = triton.next_power_of_2(n_cols)
    layer_norm_kernel[(n_rows,)](x, scale, bias, y, n_cols, eps, BLOCK=block)
    sync()
    return y


def reference(x: np.ndarray, s: np.ndarray, b: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * s + b


if __name__ == "__main__":
    for rows, cols in [(64 * 100, 256), (32 * 197, 768), (16 * 16, 1023)]:
        x = randn(rows * cols)
        s = randn(cols, seed=1)
        b = randn(cols, seed=2)
        got = layer_norm(GpuBuf.from_numpy(x), GpuBuf.from_numpy(s),
                         GpuBuf.from_numpy(b), rows, cols).to_host()
        want = reference(x.reshape(rows, cols), s, b).ravel()
        assert np.allclose(got, want, atol=1e-4), (rows, cols, "MISMATCH")
        print("ok", (rows, cols))
