"""M3-c (byte / swizzle box) — mutation-only descriptor box-shape kernels.

Interface: python3 box_swizzle.py --family N --size full

The family's object is the TMA box byte size, or the box<->swizzle<->alignment
relation (M3.3 ceiled byte size >= 2^24, M3.5 swizzle-incompatible box, M3.13 a
state that must survive the compiler's own repair). `block_shape` IS a
source-level box descriptor in Triton, so the state is expressible: the frontend
refuses a non-power-of-2, shape-incompatible or oversized box at trace time.
Each instance is a created, non-inert mutant with a compile-refusal outcome
(`avoided`).

Eight forms (the last-dim < 16-byte forms live in `box_align.py`, spec M3.7):
  f1 block inner 48  (non-power-of-2)      M3.5
  f2 block inner 24  (non-power-of-2)      M3.5
  f3 block inner 20  (non-power-of-2)      M3.5
  f4 block inner 5   (non-power-of-2)      M3.5
  f5 block inner 6   (non-power-of-2)      M3.5
  f6 block inner 2^23 (32 MB box)          M3.3
  f7 block inner 2^22 (16 MB box)          M3.3
  f8 block inner 2^21 (8 MB box)           M3.3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


@triton.jit
def box_inner(x_ptr, out_ptr, B1: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[64, 256], strides=[256, 1],
                                  block_shape=[1, B1])
    v = d.load([0, 0])
    tl.store(out_ptr, tl.sum(tl.ravel(v)))


@triton.jit
def box2(x_ptr, out_ptr, B0: tl.constexpr, B1: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[64, 256], strides=[256, 1],
                                  block_shape=[B0, B1])
    v = d.load([0, 0])
    tl.store(out_ptr, tl.sum(tl.ravel(v)))


@triton.jit
def box_huge(x_ptr, out_ptr, B1: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[64, 1 << 24], strides=[1 << 24, 1],
                                  block_shape=[1, B1])
    v = d.load([0, 0])
    tl.store(out_ptr, tl.sum(tl.ravel(v)))


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    x = GpuBuf(1 << 26, np.ones(1 << 26, dtype=np.float32))
    out = GpuBuf(1, np.zeros(1, dtype=np.float32))
    if family in (1, 2, 3, 4):
        b1 = {1: 48, 2: 24, 3: 20, 4: 5}[family]
        call = lambda: box_inner[(1,)](x, out, b1)
    elif family == 5:
        call = lambda: box_inner[(1,)](x, out, 6)
    else:
        b1 = {6: 1 << 23, 7: 1 << 22, 8: 1 << 21}[family]
        call = lambda: box_huge[(1,)](x, out, b1)
    try:
        call()
        sync()
        print("RESULT run=ok out=same")   # must never happen: box illegal
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
