"""M1 mutants over the relu kernel (benchmark2/triton/kernels/relu.py).
Interface: python3 relu.py --family N  (see mutants/README.md)
"""
import argparse
import sys
from pathlib import Path

import triton
import triton.language as tl


@triton.jit
def relu_nomask(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)                      # M1.1: no mask
    tl.store(y_ptr + offs, tl.maximum(x, 0.0))     # M1.1: no mask


@triton.jit
def relu_offbyone(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements + 1                   # M1.2: off by one
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_negidx(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs - 1, mask=mask)       # M1.3: negative index
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_badstride(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs + pid, mask=mask)     # M1.4: stride drift per block
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_offsetview(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + BLOCK + offs, mask=mask)   # M1.5: base advanced past tail
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_zerorange(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


FAMILIES = {1: relu_nomask, 2: relu_offbyone, 3: relu_negidx,
            4: relu_badstride, 5: relu_offsetview, 6: relu_zerorange}

CANARY = 4096  # guard floats after the output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, randn, sync
    import numpy as np
    from sizes import SMALL, FULL_RAGGED
    n = (FULL_RAGGED if args.size == "full" else SMALL)["relu"][0]
    x = randn(n)
    # y with a canary tail: foreign-memory corruption is observable
    yhost = np.full(n + CANARY, -777.0, dtype=np.float32)
    yhost[n:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    n_kernel = 0 if args.family == 6 else n
    grid = (triton.cdiv(max(n_kernel, 1), 1024),)
    run = "ok"
    try:
        FAMILIES[args.family][grid](GpuBuf.from_numpy(x), y, n_kernel,
                                    BLOCK=1024)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    ref = np.maximum(x, 0.0)
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        out_same = np.allclose(got[:n], ref, equal_nan=True)
        canary_intact = np.array_equal(got[n:], np.full(CANARY, 123.25,
                                                        dtype=np.float32))
        out = ("same" if out_same else "diff") +               ("+canary-ok" if canary_intact else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
