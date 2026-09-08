"""elemwise_add — y = a + b (settings/elemwise_add.md)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def add_kernel(a_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a + b, mask=mask)


def elemwise_add(a, b):
    y = GpuBuf(a.n)
    add_kernel[(triton.cdiv(a.n, 1024),)](a, b, y, a.n, BLOCK=1024)
    sync()
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    n = (SMALL if a.size == "small" else FULL)["elemwise_add"][0]
    x, b = randn(n), randn(n, seed=1)
    got = elemwise_add(GpuBuf.from_numpy(x), GpuBuf.from_numpy(b)).to_host()
    assert np.allclose(got, x + b), n
    print("ok", a.size, n)
