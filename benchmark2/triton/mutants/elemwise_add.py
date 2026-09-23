"""M1 mutants over the elemwise_add kernel
(benchmark2/triton/kernels/elemwise_add.py).
Interface: python3 elemwise_add.py --family N  (see mutants/README.md)
"""
import argparse
import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def add_carrier(a_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
    # M1.19: the base index expression, unmutated -- the injected state is the
    # >2^31 extent (a generator setting), not a source edit. The flat index
    # `pid*BLOCK + arange` wraps negative in the tail programs.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, a + b, mask=mask)


FAMILIES = {19: add_carrier}


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
        n = HUGE["elemwise_add"][0]
        x = GpuBuf(n, np.zeros(n, dtype=np.float16))
        run = "ok"
        try:
            add_carrier[(triton.cdiv(n, 1024),)](x, x, x, n, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return
    raise SystemExit(f"unimplemented family {args.family}")


if __name__ == "__main__":
    main()
