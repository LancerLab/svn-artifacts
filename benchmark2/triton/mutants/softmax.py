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


@triton.jit
def sm_stride2(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK) * 2                       # M1.8: column step 2
    mask = cols < n_cols
    x = tl.load(x_ptr + pid * n_cols + cols, mask=mask,
                other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + pid * n_cols + tl.arange(0, BLOCK), e / s,
             mask=tl.arange(0, BLOCK) < n_cols)


@triton.jit
def sm_baseoff1(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols + 1, mask=mask,
                other=float("-inf"))                          # M1.9: origin +1
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


@triton.jit
def sm_tile_store(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + pid * n_cols + cols, mask=mask,
                other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + (pid + 1) * n_cols + cols, e / s, mask=mask)  # M1.14


@triton.jit
def sm_tile_load(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + (pid + 1) * n_cols + cols, mask=mask,
                other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + pid * n_cols + cols, e / s, mask=mask)         # M1.14


@triton.jit
def sm_idxsub_load(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # M1.12: the row loop variable (extent rows > cols) is reused for the
    # column dimension; the address is block-materialised so the whole row
    # reads the single out-of-range row-scale offset.
    base = row * n_cols + row + tl.zeros([BLOCK], tl.int32)
    x = tl.load(x_ptr + base, mask=mask, other=float("-inf"))     # M1.12
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


@triton.jit
def sm_idxsub_store(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    # M1.12: same wrong-dimension reuse on the store (see `sm_idxsub_load`).
    base = row * n_cols + row + tl.zeros([BLOCK], tl.int32)
    tl.store(y_ptr + base, e / s, mask=mask)                      # M1.12


@triton.jit
def sm_carrier(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    # M1.19: no source defect; the injected state is a >2^31 element extent
    # whose row-major flat index `row*n_cols + cols` overflows int32.
    # f16 in-place, with the exp upcast (Triton 3.8 rejects fp16 exp) -- the
    # same upcast as the base sigmoid carrier; the index carrier is unchanged.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask,
                other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    tl.store(y_ptr + row * n_cols + cols, e / s, mask=mask)


FAMILIES = {1: sm_nomask, 2: sm_offbyone, 3: sm_negidx, 4: sm_badstride,
            5: sm_offsetview, 8: sm_stride2, 9: sm_baseoff1,
            12: sm_idxsub_load, 121: sm_idxsub_store,
            14: sm_tile_store, 141: sm_tile_load,
            19: sm_carrier}

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

    if args.family == 19:
        # M1.19 carrier: rows*cols = 2^31 + 4096, BLOCK = 4096.
        rows, cols = 524289, 4096
        n = rows * cols
        x = GpuBuf(n, np.zeros(n, dtype=np.float16))
        run = "ok"
        try:
            sm_carrier[(rows,)](x, x, cols, BLOCK=cols)
            sync()
        except Exception as e:
            print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
            run = "crash"
        print(f"RESULT run={run} out=none")
        return

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
