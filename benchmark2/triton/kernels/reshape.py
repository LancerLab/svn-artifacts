"""reshape — pure re-viewing copy (settings/reshape.md)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def reshape(x, n):
    y = GpuBuf(n)
    copy_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    sync()
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    n = (SMALL if a.size == "small" else FULL)["reshape"][0]
    x = randn(n)
    got = reshape(GpuBuf.from_numpy(x), n).to_host()
    assert np.array_equal(got, x), n
    print("ok", a.size, n)
