"""M1 mutants over the transpose kernel
(benchmark2/triton/kernels/transpose.py).
Interface: python3 transpose.py --family N  (see mutants/README.md)
"""
import argparse
import sys
import triton
from pathlib import Path
import triton.language as tl


@triton.jit
def tr_nomask(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    t = tl.load(x_ptr + om[:, None] * N + on[None, :])            # M1.1
    tl.store(y_ptr + on[:, None] * M + om[None, :], tl.trans(t))  # M1.1


@triton.jit
def tr_offbyone(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    m = (om[:, None] < M + 1) & (on[None, :] < N)                  # M1.2
    t = tl.load(x_ptr + om[:, None] * N + on[None, :], mask=m)
    mo = (on[:, None] < N) & (om[None, :] < M + 1)
    tl.store(y_ptr + on[:, None] * M + om[None, :], tl.trans(t), mask=mo)


@triton.jit
def tr_negidx(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    m = (om[:, None] < M) & (on[None, :] < N)
    t = tl.load(x_ptr + om[:, None] * N + on[None, :] - 1, mask=m)  # M1.3
    mo = (on[:, None] < N) & (om[None, :] < M)
    tl.store(y_ptr + on[:, None] * M + om[None, :], tl.trans(t), mask=mo)


@triton.jit
def tr_badstride(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    m = (om[:, None] < M) & (on[None, :] < N)
    t = tl.load(x_ptr + om[:, None] * (N + 1) + on[None, :], mask=m)  # M1.4
    mo = (on[:, None] < N) & (om[None, :] < M)
    tl.store(y_ptr + on[:, None] * M + om[None, :], tl.trans(t), mask=mo)


@triton.jit
def tr_offsetview(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    m = (om[:, None] < M) & (on[None, :] < N)
    t = tl.load(x_ptr + N + om[:, None] * N + on[None, :], mask=m)   # M1.5
    mo = (on[:, None] < N) & (om[None, :] < M)
    tl.store(y_ptr + on[:, None] * M + om[None, :], tl.trans(t), mask=mo)


FAMILIES = {1: tr_nomask, 2: tr_offbyone, 3: tr_negidx, 4: tr_badstride,
            5: tr_offsetview}

CANARY = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, randn, sync
    import numpy as np
    M, N = 65, 127
    x = randn(M * N)
    yhost = np.full(M * N + CANARY, -777.0, dtype=np.float32)
    yhost[M * N:] = 123.25
    y = GpuBuf.from_numpy(yhost)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    run = "ok"
    try:
        FAMILIES[args.family][grid](GpuBuf.from_numpy(x), y, M, N,
                                    BM=32, BN=32)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        ref = x.reshape(M, N).T.ravel()
        out_same = np.array_equal(got[:M * N], ref)
        canary_ok = np.array_equal(got[M * N:],
                                   np.full(CANARY, 123.25,
                                           dtype=np.float32))
        out = ("same" if out_same else "diff") + \
              ("+canary-ok" if canary_ok else "+canary-CLOBBERED")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
