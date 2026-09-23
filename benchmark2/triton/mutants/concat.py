"""M1 carrier mutant over the concat kernel
(benchmark2/triton/kernels/concat.py).
Interface: python3 concat.py --family N  (see mutants/README.md)
"""
import argparse
import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def concat_carrier(a_ptr, b_ptr, y_ptr, nA, nB, BLOCK: tl.constexpr):
    # M1.19: base index expression, unmutated -- the injected state is the
    # >2^31 extent (a generator setting); the flat index `pid*BLOCK + arange`
    # wraps negative in the tail programs.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nA + nB
    in_a = offs < nA
    va = tl.load(a_ptr + offs, mask=mask & in_a, other=0.0)
    vb = tl.load(b_ptr + offs - nA, mask=mask & ~in_a, other=0.0)
    tl.store(y_ptr + offs, tl.where(in_a, va, vb), mask=mask)


FAMILIES = {19: concat_carrier}


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
        n = 2 ** 31 + 4096
        a = GpuBuf(n, np.zeros(n, dtype=np.float16))
        b = GpuBuf(1, np.zeros(1, dtype=np.float16))
        run = "ok"
        try:
            concat_carrier[(triton.cdiv(n, 1024),)](a, b, a, n, 0, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return
    raise SystemExit(f"unimplemented family {args.family}")


if __name__ == "__main__":
    main()
