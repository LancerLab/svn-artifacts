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
def bn_offlane(x_ptr, mean_ptr, var_ptr, s_ptr, b_ptr, y_ptr, CHW, C, HW, eps,
               BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C + 1          # M1: off-by-one channel -> reads next plane's stats
    offs = tl.arange(0, BLOCK)
    mask = offs < HW
    base = x_ptr + n * CHW + c * HW
    x = tl.load(base + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    s = tl.load(s_ptr + c)
    b = tl.load(b_ptr + c)
    tl.store(y_ptr + n * CHW + c * HW + offs,
             (x - mean) / tl.sqrt(var + eps) * s + b, mask=mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    args = ap.parse_args()
    from sizes import SMALL, FULL_RAGGED
    N, C, H, W = (FULL_RAGGED if args.size == "full" else SMALL)["batch_norm"]
    HW = H * W
    x = randn(N * C * HW)
    xs = x.reshape(N, C, HW)
    mean = xs.mean(axis=(0, 2)).astype(np.float32)
    var = xs.var(axis=(0, 2)).astype(np.float32)
    s = randn(C, seed=1); b = randn(C, seed=2)
    y = GpuBuf(N * C * HW)
    run = "ok"
    try:
        bn_offlane[(N * C,)](GpuBuf.from_numpy(x), GpuBuf.from_numpy(mean),
                             GpuBuf.from_numpy(var), GpuBuf.from_numpy(s),
                             GpuBuf.from_numpy(b), y, C * HW, C, HW, 1e-5,
                             BLOCK=triton.next_power_of_2(HW))
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        want = ((xs - mean[None, :, None]) / np.sqrt(var[None, :, None] + 1e-5)
                * s[None, :, None] + b[None, :, None]).ravel()
        out = "same" if np.allclose(got, want, atol=1e-3) else "diff"
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
