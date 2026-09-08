"""batch_norm — per-channel normalize (settings/batch_norm.md).
y = (x - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c].
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
def bn_kernel(x_ptr, mean_ptr, var_ptr, s_ptr, b_ptr, y_ptr, CHW, C, HW, eps,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # one program per (n, c) plane
    n = pid // C
    c = pid % C
    offs = tl.arange(0, BLOCK)
    mask = offs < HW
    base = x_ptr + n * CHW + c * HW
    x = tl.load(base + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    s = tl.load(s_ptr + c)
    b = tl.load(b_ptr + c)
    y = (x - mean) / tl.sqrt(var + eps) * s + b
    tl.store(y_ptr + n * CHW + c * HW + offs, y, mask=mask)


def batch_norm(x, mean, var, scale, bias, N, C, H, W, eps=1e-5):
    y = GpuBuf(x.n)
    HW = H * W
    grid = (N * C,)
    bn_kernel[grid](x, mean, var, scale, bias, y, C * HW, C, HW, eps,
                    BLOCK=triton.next_power_of_2(HW))
    sync()
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    N, C, H, W = (SMALL if a.size == "small" else FULL)["batch_norm"]
    HW = H * W
    x = randn(N * C * HW)
    xs = x.reshape(N, C, HW)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    var = xs.var(axis=(0, 2)).astype(np.float32)
    s = randn(C, seed=1); b = randn(C, seed=2)
    got = batch_norm(GpuBuf.from_numpy(x), GpuBuf.from_numpy(mean),
                     GpuBuf.from_numpy(var), GpuBuf.from_numpy(s),
                     GpuBuf.from_numpy(b), N, C, H, W).to_host()
    want = ((xs - mean[None, :, None]) / np.sqrt(var[None, :, None] + 1e-5)
            * s[None, :, None] + b[None, :, None]).ravel()
    assert np.allclose(got, want, atol=1e-2), (N, C, H, W)
    print("ok", a.size, (N, C, H, W))
