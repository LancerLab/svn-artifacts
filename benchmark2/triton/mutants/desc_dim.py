"""M3-b (descriptor dim) — mutation-only descriptor kernels, ranks 2..5.

Interface: python3 desc_dim.py --family N --size full

The family's object is a descriptor dimension that reaches the carrier bound.
Triton narrows every descriptor shape entry to int32 with NO range assert
(`semantic.py:1872`; block pointers DO assert at `:1831`), so a dimension in
(2^31, 2^32) becomes negative, every offset tests out-of-bounds, and the read
silently zero-fills. This is the same defect `matmul-f3` (M3.2) realises on
rank 2; this module is the structure variants that carry it across ranks 3..5.

Each kernel is run over a faithful carrier and two mutated carriers, so the
compiled program is unchanged and the miss is a same-shape silent divergence:
  faithful  BIG = ROWS        -> the true row values
  mutated   BIG = 2**31       -> int32-negative carrier, zero fill
  mutated   BIG = 2**32       -> int32 wrap to 0
Family map: rank = 2 + (family-1)//2; variant = (family-1) % 2 (2**31 / 2**32).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

ROWS = 64
C = 64
ROW = 32


@triton.jit
def dim2(x_ptr, out_ptr, B0: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[B0, C], strides=[C, 1],
                                  block_shape=[1, C])
    v = d.load([R, 0])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


@triton.jit
def dim3(x_ptr, out_ptr, B0: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[B0, 2, C], strides=[2 * C, C, 1],
                                  block_shape=[1, 1, C])
    v = d.load([R, 0, 0])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


@triton.jit
def dim4(x_ptr, out_ptr, B0: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[B0, 2, 2, C],
                                  strides=[4 * C, 2 * C, C, 1],
                                  block_shape=[1, 1, 1, C])
    v = d.load([R, 0, 0, 0])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


@triton.jit
def dim5(x_ptr, out_ptr, B0: tl.constexpr, C: tl.constexpr, R: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[B0, 2, 2, 2, C],
                                  strides=[8 * C, 4 * C, 2 * C, C, 1],
                                  block_shape=[1, 1, 1, 1, C])
    v = d.load([R, 0, 0, 0, 0])
    tl.store(out_ptr + tl.arange(0, C), tl.reshape(v, [C]))


KERN = {2: dim2, 3: dim3, 4: dim4, 5: dim5}


def _run(rank, x, out, big):
    KERN[rank][(1,)](x, out, big, C, ROW)


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    rank = 2 + (family - 1) // 2
    variant = (family - 1) % 2
    big = 2 ** 31 if variant == 0 else 2 ** 32
    x = GpuBuf(1 << 16, np.arange(1 << 16, dtype=np.float32))
    ref_b = GpuBuf(C, np.full(C, -7.0, dtype=np.float32))
    mut_b = GpuBuf(C, np.full(C, -7.0, dtype=np.float32))
    run = "ok"
    try:
        _run(rank, x, ref_b, ROWS)     # faithful carrier
        _run(rank, x, mut_b, big)      # mutated carrier
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    if run != "ok":
        print("RESULT run=crash out=none")
        return
    ref = ref_b.to_host()
    got = mut_b.to_host()
    # the faithful carrier returns the true row; the mutated one silently
    # zero-fills. The oracle is the divergence, plus the zero-fill signature.
    differed = not np.array_equal(got, ref)
    zero_fill = bool(np.all(got == 0)) and not bool(np.all(ref == 0))
    out = "diff" if (differed and zero_fill) else "same"
    print(f"RESULT run=ok out={out}")


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
