"""M3-g (pad encoding) — mutation-only descriptor kernels.

Interface: python3 desc_pad.py --family N --size full   (see mutants/README.md)

The family is declared `M3.9`/`M3.10` on the DMA/TMA pad fields. Triton's
surface for those fields is the descriptor's `padding_option`: the fill mode the
descriptor applies when a block request runs past the tensor bound. The model
requires the out-of-range region to be zero; a non-zero fill is the defect
`mutation-specs-v2.md` §3.5 states ("non-zero where forbidden").

Four ranks (2..5) x two geometries:
  odd family   -> tail overrun   (the whole requested block lies past the
                  bound; every padded element is out of range) -> M3.9
  even family  -> mid overrun    (the block straddles the bound; a partial
                  block is padded) -> M3.10

Each kernel is the SAME shape with two constexpr `padding_option` values, so
the compiled program is unchanged and the defect is a same-shape silent
divergence in the fill value (the iree control-operand pattern). The reference
is the zero fill; the mutant is the nan fill.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

ROWS = 32
C = 64


@triton.jit
def pad2(x_ptr, out_ptr, R0: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
         P: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[R0, C], strides=[C, 1],
                                  block_shape=[2, C], padding_option=P)
    v = d.load([R, 0])
    tl.store(out_ptr + tl.arange(0, 2 * C), tl.reshape(v, [2 * C]))


@triton.jit
def pad3(x_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
         R: tl.constexpr, P: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[A, B, C],
                                  strides=[B * C, C, 1],
                                  block_shape=[1, 2, C], padding_option=P)
    v = d.load([0, R, 0])
    tl.store(out_ptr + tl.arange(0, 2 * C), tl.reshape(v, [2 * C]))


@triton.jit
def pad4(x_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
         C: tl.constexpr, R: tl.constexpr, P: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[A, B, D, C],
                                  strides=[B * D * C, D * C, C, 1],
                                  block_shape=[1, 1, 2, C], padding_option=P)
    v = d.load([0, 0, R, 0])
    tl.store(out_ptr + tl.arange(0, 2 * C), tl.reshape(v, [2 * C]))


@triton.jit
def pad5(x_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr, D: tl.constexpr,
         E: tl.constexpr, C: tl.constexpr, R: tl.constexpr, P: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[A, B, D, E, C],
                                  strides=[B * D * E * C, D * E * C, E * C, C, 1],
                                  block_shape=[1, 1, 1, 2, C], padding_option=P)
    v = d.load([0, 0, 0, R, 0])
    tl.store(out_ptr + tl.arange(0, 2 * C), tl.reshape(v, [2 * C]))


# family -> (kernel, callable(launch), rank)
def _launch(family, x, out, P):
    A = ROWS
    if family <= 2:                                    # rank 2
        R = A if family == 1 else A - 1
        pad2[(1,)](x, out, A, C, R, P)
    elif family <= 4:                                  # rank 3
        R = A if family == 3 else A - 1
        pad3[(1,)](x, out, 2, A, C, R, P)
    elif family <= 6:                                  # rank 4
        R = A if family == 5 else A - 1
        pad4[(1,)](x, out, 2, 2, A, C, R, P)
    else:                                              # rank 5
        R = A if family == 7 else A - 1
        pad5[(1,)](x, out, 2, 2, 2, A, C, R, P)


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    n = 2 * C
    x = GpuBuf(1 << 16, np.arange(1 << 16, dtype=np.float32))
    ref_b = GpuBuf(n, np.full(n, -7.0, dtype=np.float32))
    mut_b = GpuBuf(n, np.full(n, -7.0, dtype=np.float32))
    run = "ok"
    try:
        _launch(family, x, ref_b, "zero")   # faithful fill
        _launch(family, x, mut_b, "nan")    # mutated pad field
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    if run != "ok":
        print("RESULT run=crash out=none")
        return
    ref = ref_b.to_host()
    got = mut_b.to_host()
    # the manifestation oracle: the mutated fill must differ from the faithful
    # one, and the difference must be the injected non-zero (nan) pad.
    differed = not np.array_equal(got, ref, equal_nan=True)
    nan_fill = bool(np.isnan(got).any())
    out = "diff" if (differed and nan_fill) else "same"
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
