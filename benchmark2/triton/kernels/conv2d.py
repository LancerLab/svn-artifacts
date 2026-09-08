"""conv2d — Triton composition from benchmark2/settings/conv2d.md.
y[b,co,oh,ow] = sum_{ci,kh,kw} x[b,ci,oh*sh+kh,ow*sw+kw] * w[co,ci,kh,kw].
Direct tiled composition (no im2col). Host side uses ../gpubuf.py (no torch).
"""
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, y_ptr, B, C, H, W, CO, KH, KW, OH, OW,
                  SH: tl.constexpr, SW: tl.constexpr,
                  BLOCK_CO: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    oh = offs_s // OW
    ow = offs_s % OW
    acc = tl.zeros((BLOCK_CO, BLOCK_S), dtype=tl.float32)
    for ci in range(0, C):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh * SH + kh
                iw = ow * SW + kw
                xmask = (offs_s[None, :] < OH * OW) & (ih[None, :] < H) & \
                        (iw[None, :] < W)
                xv = tl.load(x_ptr + pid_b * C * H * W + ci * H * W
                             + ih[None, :] * W + iw[None, :],
                             mask=xmask, other=0.0)
                wv = tl.load(w_ptr + offs_co[:, None] * (C * KH * KW)
                             + ci * KH * KW + kh * KW + kw,
                             mask=(offs_co[:, None] < CO), other=0.0)
                acc += xv * wv
    ymask = (offs_co[:, None] < CO) & (offs_s[None, :] < OH * OW)
    tl.store(y_ptr + pid_b * CO * OH * OW + offs_co[:, None] * OH * OW
             + offs_s[None, :], acc, mask=ymask)


def conv2d(x: GpuBuf, w: GpuBuf, B: int, C: int, H: int, W: int,
           CO: int, KH: int, KW: int, stride=(1, 1)) -> GpuBuf:
    SH, SW = stride
    OH, OW = (H - KH) // SH + 1, (W - KW) // SW + 1
    y = GpuBuf(B * CO * OH * OW)
    grid = (B, triton.cdiv(CO, 32), triton.cdiv(OH * OW, 64))
    conv2d_kernel[grid](x, w, y, B, C, H, W, CO, KH, KW, OH, OW,
                        SH=SH, SW=SW, BLOCK_CO=32, BLOCK_S=64)
    sync()
    return y


def reference(x: np.ndarray, w: np.ndarray, stride=(1, 1)) -> np.ndarray:
    B, C, H, W = x.shape
    CO, _, KH, KW = w.shape
    SH, SW = stride
    OH, OW = (H - KH) // SH + 1, (W - KW) // SW + 1
    y = np.zeros((B, CO, OH, OW), dtype=np.float32)
    for b in range(B):
        for co in range(CO):
            for i in range(OH):
                for j in range(OW):
                    y[b, co, i, j] = np.sum(
                        x[b, :, i * SH:i * SH + KH, j * SW:j * SW + KW]
                        * w[co])
    return y


if __name__ == "__main__":
    for shape, wshape in [((8, 16, 14, 14), (16, 16, 3, 3)),
                          ((4, 8, 12, 12), (8, 8, 1, 1))]:
        B, C, H, W = shape
        CO, _, KH, KW = wshape
        x = randn(B * C * H * W)
        w = randn(CO * C * KH * KW, seed=1)
        got = conv2d(GpuBuf.from_numpy(x), GpuBuf.from_numpy(w),
                     B, C, H, W, CO, KH, KW).to_host()
        want = reference(x.reshape(shape), w.reshape(wshape)).ravel()
        assert np.allclose(got, want, atol=1e-2), (shape, "MISMATCH")
        print("ok", shape)
