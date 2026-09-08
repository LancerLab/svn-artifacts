"""M1 level-2 mutant (plan §3.1 priority rows) — see mutants/README.md."""
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
def pool_nomask(x_ptr, y_ptr, C, H, W, OH, OW, KS: tl.constexpr,
                BLOCK: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    oh = offs // OW
    ow = offs % OW
    acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for kh in range(0, KS):
        for kw in range(0, KS):
            ih = oh * KS + kh
            iw = ow * KS + kw
            v = tl.load(x_ptr + pid_c * H * W + ih * W + iw)  # M1.1: no bounds
            acc = tl.maximum(acc, v)
    tl.store(y_ptr + pid_c * OH * OW + offs, acc)             # no mask


@triton.jit
def pool_offbyone(x_ptr, y_ptr, C, H, W, OH, OW, KS: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    oh = offs // OW
    ow = offs % OW
    mask = offs < OH * OW
    acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    for kh in range(0, KS + 1):                                # M1.2
        for kw in range(0, KS + 1):
            ih = oh * KS + kh
            iw = ow * KS + kw
            m = mask & (ih < H) & (iw < W)
            v = tl.load(x_ptr + pid_c * H * W + ih * W + iw, mask=m,
                        other=float("-inf"))
            acc = tl.maximum(acc, v)
    tl.store(y_ptr + pid_c * OH * OW + offs, acc, mask=mask)


FAMILIES = {1: pool_nomask, 2: pool_offbyone}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    args = ap.parse_args()
    C, H, W, k = 16, 14, 14, 3   # 14 % 3 != 0: ragged window boundary
    OH, OW = H // k, W // k
    x = randn(C * H * W)
    yhost = np.full(C * OH * OW + CANARY, -777.0, dtype=np.float32)
    yhost[C * OH * OW:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    run = "ok"
    try:
        FAMILIES[args.family][(C, triton.cdiv(OH * OW, 128))](
            GpuBuf.from_numpy(x), y, C, H, W, OH, OW, KS=k, BLOCK=128)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        xs = x.reshape(C, H, W)
        # pad-aware reference: max over the valid part of each window
        refp = np.stack([xs[:, i*k:min(i*k+k,H), j*k:min(j*k+k,W)]
                         .max(axis=(1, 2))
                         for i in range(OH) for j in range(OW)], axis=1).ravel()
        out_same = np.array_equal(got[:C*OH*OW], refp)
        canary_ok = np.array_equal(got[C*OH*OW:],
                                   np.full(CANARY, 123.25, dtype=np.float32))
        out = ("same" if out_same else "diff") + \
              ("+canary-ok" if canary_ok else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
