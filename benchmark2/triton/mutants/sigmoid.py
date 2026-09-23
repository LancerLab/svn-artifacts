"""M1 mutants over the sigmoid kernel (benchmark2/triton/kernels/sigmoid.py).
Interface: python3 sigmoid.py --family N  (see mutants/README.md)
"""
import argparse
import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def sigmoid_carrier(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # M1.19: the base index expression, unmutated -- the injected state is the
    # >2^31 extent (a generator setting), not a source edit. The flat index
    # `pid*BLOCK + arange` wraps negative in the tail programs.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-xf))
    tl.store(y_ptr + offs, y.to(tl.float16), mask=mask)


@triton.jit
def sigmoid_alias_rev(x_ptr, n, BLOCK: tl.constexpr):
    # M1.11: in-place read-after-write aliasing overlap (reversal).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(x_ptr + (n - 1 - offs), y, mask=mask)


@triton.jit
def sigmoid_alias_shift(x_ptr, n, BLOCK: tl.constexpr):
    # M1.11 second realisation: forward-shift overlap.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dst = offs + 1
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(x_ptr + dst, y, mask=dst < n)


FAMILIES = {19: sigmoid_carrier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full", "huge"])
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, sync
    import numpy as np

    if args.family == 19:
        # M1.19: element count > INT_MAX; f16 in-place keeps it to one buffer.
        from sizes import HUGE
        n = HUGE["sigmoid"][0]
        x = GpuBuf(n, np.zeros(n, dtype=np.float16))
        run = "ok"
        try:
            sigmoid_carrier[(triton.cdiv(n, 1024),)](x, x, n, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return

    if args.family in (11, 111):
        from gpubuf import randn
        from sizes import FULL_RAGGED
        n = FULL_RAGGED["sigmoid"][0]
        x0 = randn(n)
        x = GpuBuf.from_numpy(x0.copy())
        kern = sigmoid_alias_rev if args.family == 11 else sigmoid_alias_shift
        run = "ok"
        try:
            kern[(triton.cdiv(n, 1024),)](x, n, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        got = x.to_host() if run == "ok" else None
        if got is None:
            out = "none"
        else:
            ref = 1.0 / (1.0 + np.exp(-x0.astype(np.float32)))
            out = "same" if np.allclose(got, ref, atol=1e-3) else "diff"
        print(f"RESULT run={run} out={out}")
        return
    raise SystemExit(f"unimplemented family {args.family}")


if __name__ == "__main__":
    main()
