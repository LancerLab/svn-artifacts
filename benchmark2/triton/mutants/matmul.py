"""M3 mutants over the matmul kernel (benchmark2/triton/kernels/matmul.py).
Interface: python3 matmul.py --family N  (see mutants/README.md)

M3 families for Triton (specs §3 → §4 translation: partial, tl.dot rules):
 1 K not divisible by the tensor-core atom (tl.dot BK=8 -> JIT rejects)
 2 oversized tile / shared-memory budget exceeded (BM=BN=256, BK=64)
"""
import argparse
import sys
import triton
from pathlib import Path
import triton.language as tl


@triton.jit
def mm_katom(a_ptr, b_ptr, c_ptr, M, N, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * K + ok[None, :]
    bp = b_ptr + ok[:, None] * N + on[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        b = tl.load(bp, mask=(ok[:, None] < K) & (on[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        ap += BK
        bp += BK * N
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc,
             mask=(om[:, None] < M) & (on[None, :] < N))


FAMILIES = {1: dict(BM=32, BN=32, BK=8),    # K atom violation
            2: dict(BM=256, BN=256, BK=64)}  # shared-memory blowup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gpubuf import GpuBuf, randn, sync
    import numpy as np
    import numpy as np
    dt = np.float16 if args.family == 1 else np.float32
    M, K, N = 64, 48, 64
    a = randn(M * K).astype(dt)   # fp16 for the MMA atom constraint
    b = randn(K * N, seed=1).astype(dt)
    c = GpuBuf(M * N)
    cfg = FAMILIES[args.family]
    run = "ok"
    try:
        mm_katom[(triton.cdiv(M, cfg["BM"]), triton.cdiv(N, cfg["BN"]))](
            GpuBuf.from_numpy(a), GpuBuf.from_numpy(b), c, M, N, K, **cfg)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = c.to_host() if run == "ok" else None
    ref = (a.astype(np.float32).reshape(M, K)
           @ b.astype(np.float32).reshape(K, N)).ravel()
    out = "none" if got is None else (
        "same" if np.allclose(got, ref, atol=1e-1, rtol=1e-2,
                              equal_nan=True) else "diff")
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
