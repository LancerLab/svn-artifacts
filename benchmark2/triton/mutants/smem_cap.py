"""M3-h (on-chip capacity) — mutation-only oversized-tile kernels.

Interface: python3 smem_cap.py --family N --size full

The family's object is an on-chip allocation past the device limit. Triton
block sizes are `constexpr`, so the size is a source-level constant and the JIT
refuses an over-budget program (`OutOfResources`). Every instance is therefore a
created, non-inert mutant with a compile-refusal outcome (`avoided` shape): the
source states the defect, the compiler rejects it. Only the runtime-shaped
extent (M3.27/M3.28) is genuinely unavailable, because the tile extent is never
symbolic.

Four base shapes x two over-budget tilings (wide square tile / deep K tile).
Each tiling is pipelined (an accumulator K-loop), which is what materialises the
operand tiles in shared memory and trips the budget.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

# base (M, N, K) of the matmul; the tile tiling is what goes over budget
SHAPES = [(1024, 1024, 1024), (2048, 1024, 512), (1024, 2048, 512),
          (512, 512, 4096)]


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK * N
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    kernel = (family - 1) // 2
    variant = (family - 1) % 2
    M, N, K = SHAPES[kernel]
    if variant == 0:
        BM = BN = 256          # 256x256 tile: 128K smem > 100K budget
    else:
        BM = BN = 512          # 512x512 tile: 256K smem
    BK = 64
    a = GpuBuf(M * K, np.ones(M * K, dtype=np.float16))
    b = GpuBuf(K * N, np.ones(K * N, dtype=np.float16))
    c = GpuBuf(M * N, np.zeros(M * N, dtype=np.float16))
    try:
        mm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](a, b, c, M, N, K, BM, BN, BK)
        sync()
        print("RESULT run=ok out=same")   # must never happen: tile over budget
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        print("RESULT run=crash out=none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    ap.add_argument("--size", default="full", choices=["small", "full"])
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if not 1 <= args.family <= 8:
        raise SystemExit(f"family must be 1..8, got {args.family}")
    run_family(args.family)


if __name__ == "__main__":
    main()
