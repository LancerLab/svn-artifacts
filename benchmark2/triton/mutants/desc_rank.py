"""M3-f (descriptor rank) — mutation-only rank-variant kernels.

Interface: python3 desc_rank.py --family N --size full

The family's object is a descriptor rank outside the assessed [1, 5]. Triton
CAN state it: `make_tensor_descriptor` takes a source-level rank, and the
frontend refuses a bad one at trace time
(`semantic.py:1854` -> `Expected 1 <= ndim <= 5 but got 6`). That is a created,
non-inert mutant whose outcome is a compile refusal -- the `avoided`/`repaired`
shape of M3.8, NOT an unexpressible slot.

Four base shapes x two rank variants:
  odd family   -> rank 6 (one dim beyond the assessed maximum) -> M3.8
  even family  -> rank 0 (empty shape, below the assessed minimum) -> M3.8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

C = 64


@triton.jit
def r6(x_ptr, out_ptr, C: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[2, 2, 2, 2, 2, C],
                                  strides=[16 * C, 8 * C, 4 * C, 2 * C, C, 1],
                                  block_shape=[1, 1, 1, 1, 1, C])
    v = d.load([0, 0, 0, 0, 0, 0])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


@triton.jit
def r0(x_ptr, out_ptr, C: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[], strides=[],
                                  block_shape=[])
    v = d.load([])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    variant = (family - 1) % 2          # 0 -> rank 6, 1 -> rank 0
    kern = r6 if variant == 0 else r0
    x = GpuBuf(1 << 16, np.arange(1 << 16, dtype=np.float32))
    out = GpuBuf(C, np.full(C, -7.0, dtype=np.float32))
    try:
        kern[(1,)](x, out, C)
        sync()
        print("RESULT run=ok out=same")   # must never happen: the rank is bad
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
