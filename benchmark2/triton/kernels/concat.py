"""concat — y = concat(a, b) along dim 1 (settings/concat.md)."""
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def concat_kernel(a_ptr, b_ptr, y_ptr, nA, nB, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nA + nB
    in_a = offs < nA
    va = tl.load(a_ptr + offs, mask=mask & in_a, other=0.0)
    vb = tl.load(b_ptr + offs - nA, mask=mask & ~in_a, other=0.0)
    tl.store(y_ptr + offs, tl.where(in_a, va, vb), mask=mask)


def concat(a, b):
    y = GpuBuf(a.n + b.n)
    concat_kernel[(triton.cdiv(a.n + b.n, 1024),)](a, b, y, a.n, b.n,
                                                   BLOCK=1024)
    sync()
    return y


if __name__ == "__main__":
    for na, nb in [(16 * 512 * 100, 16 * 512 * 55), (1023, 77)]:
        a, b = randn(na), randn(nb, seed=1)
        got = concat(GpuBuf.from_numpy(a), GpuBuf.from_numpy(b)).to_host()
        assert np.array_equal(got, np.concatenate([a, b])), (na, nb)
        print("ok", (na, nb))
