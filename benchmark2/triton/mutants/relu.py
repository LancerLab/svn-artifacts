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


@triton.jit
def relu_stride2(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    dst = pid * BLOCK + tl.arange(0, BLOCK)
    src = pid * BLOCK * 2 + tl.arange(0, BLOCK)     # M1.8: tile advances by 2
    x = tl.load(x_ptr + src, mask=src < n_elements, other=0.0)
    tl.store(y_ptr + dst, tl.maximum(x, 0.0), mask=dst < n_elements)


@triton.jit
def relu_baseoff1(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs + 1, mask=mask)        # M1.9: origin +1, extent intact
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_tile_load(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    src = offs + BLOCK                              # M1.14: source tile coord +1
    x = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_tile_store(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dst = offs + BLOCK                              # M1.14: dest tile coord +1
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + dst, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_carrier(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    # M1.19: the index is legal but its int32 carrier cannot name it -- the
    # same `pid*BLOCK + arange` expression the base kernel uses. No source
    # defect is injected; the injected state is the >2^31 extent (a generator
    # setting), and the flat index wraps negative in the tail program.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_alias_rev(x_ptr, n_elements, BLOCK: tl.constexpr):
    # M1.11: in-place read-after-write aliasing overlap -- the store lands on a
    # different program's load region (reversal) in the same buffer.
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + (n_elements - 1 - offs), tl.maximum(x, 0.0), mask=mask)


@triton.jit
def relu_alias_shift(x_ptr, n_elements, BLOCK: tl.constexpr):
    # M1.11 second realisation: forward-shift overlap (write offs+1).
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dst = offs + 1
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + dst, tl.maximum(x, 0.0), mask=dst < n_elements)


FAMILIES = {1: relu_nomask, 2: relu_offbyone, 3: relu_negidx,
            4: relu_badstride, 5: relu_offsetview, 6: relu_zerorange,
            8: relu_stride2, 9: relu_baseoff1,
            14: relu_tile_load, 141: relu_tile_store, 19: relu_carrier}

CANARY = 4096  # guard floats after the output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full", "huge"])
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, randn, sync
    import numpy as np

    if args.family == 19:
        # M1.19: element count > INT_MAX; f16 in-place keeps it to one buffer.
        from sizes import HUGE
        n = HUGE["relu"][0]
        x = GpuBuf(n, np.zeros(n, dtype=np.float16))
        run = "ok"
        try:
            relu_carrier[(triton.cdiv(n, 1024),)](x, x, n, BLOCK=1024)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return

    if args.family in (11, 111):
        # M1.11 in-place aliasing: same buffer for read and write.
        from sizes import FULL_RAGGED
        n = FULL_RAGGED["relu"][0]
        x0 = randn(n)
        x = GpuBuf.from_numpy(x0.copy())
        kern = relu_alias_rev if args.family == 11 else relu_alias_shift
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
            out_same = np.allclose(got, np.maximum(x0, 0.0), atol=1e-4,
                                   equal_nan=True)
            out = "same" if out_same else "diff"
        print(f"RESULT run={run} out={out}")
        return

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
