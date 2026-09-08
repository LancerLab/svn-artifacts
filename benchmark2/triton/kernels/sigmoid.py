"""sigmoid — y = 1/(1+e^-x) (settings/sigmoid.md)."""
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def sigmoid_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


def sigmoid(x):
    y = GpuBuf(x.n)
    sigmoid_kernel[(triton.cdiv(x.n, 1024),)](x, y, x.n, BLOCK=1024)
    sync()
    return y


if __name__ == "__main__":
    for n in [127 * 1023 + 5, 64 * 100 * 256]:
        x = randn(n)
        got = sigmoid(GpuBuf.from_numpy(x)).to_host()
        assert np.allclose(got, 1 / (1 + np.exp(-x)), atol=1e-6), n
        print("ok", n)
