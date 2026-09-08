"""M1 level-2 mutant — see mutants/README.md."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync

CANARY = 4096

@triton.jit
def conv_nomask(x_ptr, w_ptr, y_ptr, B, C, H, W, CO, KH, KW, OH, OW,
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
                xv = tl.load(x_ptr + pid_b * C * H * W + ci * H * W
                             + (oh * 1 + kh)[None, :] * W + (ow + kw)[None, :])
                # M1.1: no input-boundary mask, no output mask
                wv = tl.load(w_ptr + offs_co[:, None] * (C * KH * KW)
                             + ci * KH * KW + kh * KW + kw)
                acc += xv * wv
    tl.store(y_ptr + pid_b * CO * OH * OW + offs_co[:, None] * OH * OW
             + offs_s[None, :], acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    args = ap.parse_args()
    from sizes import SMALL, FULL_RAGGED
    B, C, H, W, CO, KH, KW = (FULL_RAGGED if args.size == "full" else SMALL)["conv2d"]
    OH, OW = H - KH + 1, W - KW + 1
    x = randn(B * C * H * W)
    w = randn(CO * C * KH * KW, seed=1)
    yhost = np.full(B * CO * OH * OW + CANARY, -777.0, dtype=np.float32)
    yhost[B * CO * OH * OW:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    run = "ok"
    try:
        conv_nomask[(B, triton.cdiv(CO, 32), triton.cdiv(OH * OW, 64))](
            GpuBuf.from_numpy(x), GpuBuf.from_numpy(w), y,
            B, C, H, W, CO, KH, KW, OH, OW, BLOCK_CO=32, BLOCK_S=64)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        # reference: the gated correct kernel (verified vs numpy at small size;
        # a python loop reference is infeasible at full size)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "kernels"))
        from conv2d import conv2d as ref_conv2d
        refg = ref_conv2d(GpuBuf.from_numpy(x), GpuBuf.from_numpy(w),
                          B, C, H, W, CO, KH, KW).to_host()
        out_same = np.allclose(got[:B*CO*OH*OW], refg, atol=1e-2)
        canary_ok = np.array_equal(got[B*CO*OH*OW:],
                                   np.full(CANARY, 123.25, dtype=np.float32))
        out = ("same" if out_same else "diff") + \
              ("+canary-ok" if canary_ok else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
