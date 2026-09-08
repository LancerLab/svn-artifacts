"""max_pool2d — windowed max (settings/max_pool2d.md)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def pool_kernel(x_ptr, y_ptr, C, H, W, OH, OW, KS: tl.constexpr,
                BLOCK: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    oh = offs // OW
    ow = offs % OW
    mask = offs < OH * OW
    acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for kh in range(0, KS):
        for kw in range(0, KS):
            ih = oh * KS + kh
            iw = ow * KS + kw
            m = mask & (ih < H) & (iw < W)
            v = tl.load(x_ptr + pid_c * H * W + ih * W + iw, mask=m,
                        other=float("-inf"))
            acc = tl.maximum(acc, v)
    tl.store(y_ptr + pid_c * OH * OW + offs, acc, mask=mask)


def max_pool2d(x, C, H, W, k):
    OH, OW = H // k, W // k
    y = GpuBuf(C * OH * OW)
    pool_kernel[(C, triton.cdiv(OH * OW, 128))](x, y, C, H, W, OH, OW,
                                                KS=k, BLOCK=128)
    sync()
    return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="small")
    a = ap.parse_args()
    from sizes import SMALL, FULL
    C, H, W, k = (SMALL if a.size == "small" else FULL)["max_pool2d"]
    x = randn(C * H * W)
    got = max_pool2d(GpuBuf.from_numpy(x), C, H, W, k).to_host()
    xs = x.reshape(C, H // k, k, W // k, k)
    want = xs.max(axis=(2, 4)).ravel()
    assert np.array_equal(got, want), (C, H, W, k)
    print("ok", a.size, (C, H, W, k))
