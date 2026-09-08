"""reduce_mean — row mean over trailing dim (settings/reduce_mean.md)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def reduce_mean_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row, tl.sum(x, axis=0) / n_cols)


def reduce_mean(x, n_rows, n_cols):
    y = GpuBuf(n_rows)
    reduce_mean_kernel[(n_rows,)](x, y, n_cols,
                                  BLOCK=triton.next_power_of_2(n_cols))
    sync()
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    rows, cols = (SMALL if a.size == "small" else FULL)["reduce_mean"]
    x = randn(rows * cols)
    got = reduce_mean(GpuBuf.from_numpy(x), rows, cols).to_host()
    want = x.reshape(rows, cols).mean(axis=-1)
    assert np.allclose(got, want, atol=1e-4), (rows, cols)
    print("ok", a.size, (rows, cols))
