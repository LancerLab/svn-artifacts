"""gelu — tanh approximation (settings/gelu.md)."""
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (tl.exp(2 * t) - 1) / (tl.exp(2 * t) + 1))
    tl.store(y_ptr + offs, y, mask=mask)


def gelu(x):
    y = GpuBuf(x.n)
    gelu_kernel[(triton.cdiv(x.n, 1024),)](x, y, x.n, BLOCK=1024)
    sync()
    return y


if __name__ == "__main__":
    for n in [32 * 512 * 768, 127 * 1023]:
        x = randn(n)
        got = gelu(GpuBuf.from_numpy(x)).to_host()
        t = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        want = 0.5 * x * (1 + np.tanh(t))
        assert np.allclose(got, want, atol=1e-5), n
        print("ok", n)
