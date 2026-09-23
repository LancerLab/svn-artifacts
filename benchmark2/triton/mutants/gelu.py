"""M1 mutants over the gelu kernel (benchmark2/triton/kernels/gelu.py).
Interface: python3 gelu.py --family N  (see mutants/README.md)
"""
import argparse
import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def gelu_carrier(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # M1.19: the base index expression, unmutated -- the injected state is the
    # >2^31 extent (a generator setting), not a source edit. The flat index
    # `pid*BLOCK + arange` wraps negative in the tail programs.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (tl.exp(2 * t) - 1) / (tl.exp(2 * t) + 1))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def gelu_alias_rev(x_ptr, n, BLOCK: tl.constexpr):
    # M1.11: in-place read-after-write aliasing overlap (reversal).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (tl.exp(2 * t) - 1) / (tl.exp(2 * t) + 1))
    tl.store(x_ptr + (n - 1 - offs), y, mask=mask)


@triton.jit
def gelu_alias_shift(x_ptr, n, BLOCK: tl.constexpr):
    # M1.11 second realisation: forward-shift overlap.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    dst = offs + 1
    x = tl.load(x_ptr + offs, mask=mask)
    t = tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + (tl.exp(2 * t) - 1) / (tl.exp(2 * t) + 1))
    tl.store(x_ptr + dst, y, mask=dst < n)


FAMILIES = {19: gelu_carrier}


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
        n = HUGE["gelu"][0]
        x = GpuBuf(n, np.zeros(n, dtype=np.float16))
        run = "ok"
        try:
            gelu_carrier[(triton.cdiv(n, 1024),)](x, x, n, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return

    if args.family in (11, 111):
        from gpubuf import randn
        from sizes import FULL_RAGGED
        n = FULL_RAGGED["gelu"][0]
        x0 = randn(n)
        x = GpuBuf.from_numpy(x0.copy())
        kern = gelu_alias_rev if args.family == 11 else gelu_alias_shift
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
            xf = x0.astype(np.float32)
            ref = 0.5 * xf * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                            * (xf + 0.044715 * xf ** 3)))
            out = "same" if np.allclose(got, ref, atol=1e-2) else "diff"
        print(f"RESULT run={run} out={out}")
        return
    raise SystemExit(f"unimplemented family {args.family}")


if __name__ == "__main__":
    main()
