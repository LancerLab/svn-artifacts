"""M1 mutants over the layer_norm kernel
(benchmark2/triton/kernels/layer_norm.py).
Interface: python3 layer_norm.py --family N  (see mutants/README.md)
"""
import argparse
import sys
import triton
import triton.language as tl
from pathlib import Path


@triton.jit
def ln_nomask(x_ptr, s_ptr, b_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + cols)                # M1.1
    mean = tl.sum(x, axis=0) / n_cols
    d = x - mean
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    s = tl.load(s_ptr + cols)                               # M1.1
    b = tl.load(b_ptr + cols)                               # M1.1
    tl.store(y_ptr + row * n_cols + cols, d * rstd * s + b)  # M1.1


@triton.jit
def ln_offbyone(x_ptr, s_ptr, b_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols + 1                                # M1.2
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    s = tl.load(s_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, d * rstd * s + b, mask=mask)


@triton.jit
def ln_negidx(x_ptr, s_ptr, b_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols - 1, mask=mask, other=0.0)  # M1.3
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    s = tl.load(s_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, d * rstd * s + b, mask=mask)


@triton.jit
def ln_badstride(x_ptr, s_ptr, b_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * (n_cols + 1) + cols, mask=mask,
                other=0.0)                                    # M1.4
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    s = tl.load(s_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, d * rstd * s + b, mask=mask)


@triton.jit
def ln_offsetview(x_ptr, s_ptr, b_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + (row + 1) * n_cols + cols, mask=mask,
                other=0.0)                                    # M1.5
    mean = tl.sum(x, axis=0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / n_cols + eps)
    s = tl.load(s_ptr + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * n_cols + cols, d * rstd * s + b, mask=mask)


FAMILIES = {1: ln_nomask, 2: ln_offbyone, 3: ln_negidx, 4: ln_badstride,
            5: ln_offsetview}

CANARY = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, randn, sync
    import numpy as np
    rows, cols = 197, 769  # ragged cols
    x = randn(rows * cols)
    s = randn(cols, seed=1)
    b = randn(cols, seed=2)
    yhost = np.full(rows * cols + CANARY, -777.0, dtype=np.float32)
    yhost[rows * cols:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    block = triton.next_power_of_2(cols)
    run = "ok"
    try:
        FAMILIES[args.family][(rows,)](GpuBuf.from_numpy(x),
                                       GpuBuf.from_numpy(s),
                                       GpuBuf.from_numpy(b),
                                       y, cols, 1e-5, BLOCK=block)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        xs = x.reshape(rows, cols)
        mu = xs.mean(axis=-1, keepdims=True)
        var = xs.var(axis=-1, keepdims=True)
        ref = ((xs - mu) / np.sqrt(var + 1e-5) * s + b).ravel()
        out_same = np.allclose(got[:rows * cols], ref, atol=1e-4,
                               equal_nan=True)
        canary_ok = np.array_equal(got[rows * cols:],
                                   np.full(CANARY, 123.25,
                                           dtype=np.float32))
        out = ("same" if out_same else "diff") + \
              ("+canary-ok" if canary_ok else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
