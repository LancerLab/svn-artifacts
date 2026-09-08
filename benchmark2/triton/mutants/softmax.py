"""M1 mutants over the softmax kernel (benchmark2/triton/kernels/softmax.py).
Interface: python3 softmax.py --family N  (see mutants/README.md)
"""
import argparse
import sys
import triton
from pathlib import Path
import triton.language as tl


@triton.jit
def sm_nomask(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + cols)                 # M1.1
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(y_ptr + row * n_cols + cols, e / tl.sum(e, axis=0))  # M1.1


@triton.jit
def sm_offbyone(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols + 1                                  # M1.2
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


@triton.jit
def sm_negidx(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols - 1, mask=mask,
                other=float("-inf"))                          # M1.3
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


@triton.jit
def sm_badstride(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * (n_cols + 2) + cols, mask=mask,
                other=float("-inf"))                          # M1.4
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


@triton.jit
def sm_offsetview(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + (row + 1) * n_cols + cols, mask=mask,
                other=float("-inf"))                          # M1.5
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


FAMILIES = {1: sm_nomask, 2: sm_offbyone, 3: sm_negidx, 4: sm_badstride,
            5: sm_offsetview}

CANARY = 4096


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
    rows, cols = (FULL_RAGGED if args.size == "full" else SMALL)["softmax"]
    x = randn(rows * cols)
    yhost = np.full(rows * cols + CANARY, -777.0, dtype=np.float32)
    yhost[rows * cols:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    block = triton.next_power_of_2(cols)
    run = "ok"
    try:
        FAMILIES[args.family][(rows,)](GpuBuf.from_numpy(x), y, cols,
                                       BLOCK=block)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        xs = x.reshape(rows, cols)
        e = np.exp(xs - xs.max(axis=-1, keepdims=True))
        ref = (e / e.sum(axis=-1, keepdims=True)).ravel()
        out_same = np.allclose(got[:rows * cols], ref, atol=1e-5,
                               equal_nan=True)
        canary_ok = np.array_equal(got[rows * cols:],
                                   np.full(CANARY, 123.25,
                                           dtype=np.float32))
        out = ("same" if out_same else "diff") + \
              ("+canary-ok" if canary_ok else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
