"""M4-g (degenerate pad) — mutation-only padded-extent kernels.

Interface: python3 pad_bound.py --family N --size full   (see mutants/README.md)

M4-g is the `LoopBound` family whose edit is a **padded extent that goes
negative or empty** -- the negative-padding trigger named by the torch.compile
study.  Triton authors its loops in Python, so the loop bound is a source value;
here that bound is not a literal but the *padded output extent*

    ext = N + 2 * PAD - KERNEL + 1

so the mutated field is the pad (`PAD`), and the bound degenerates as a
consequence.  A sufficiently negative pad makes `ext` negative (`M4.7`); a pad
that exactly cancels the input makes it zero (`M4.8`).  Either way
`range(0, ext)` is a legal empty loop, the kernel runs to completion, and `y`
is never written.

The distinction from `M4-b` (negative bound) is the edit, not the symptom: the
corrupted field is the padding, and the bound is derived from it.  On a lane
that does not author a padded extent (Triton has no conv/pool pad), this
mutation-only kernel is how the family is carried (channels/README.md).

Eight instances, four loop shapes x two pad degeneracies:

  idx 0..3  negpad  `PAD = -128` -> `ext = -2`  -> M4.7
  idx 4..7  emptypad `PAD = -127` -> `ext = 0`  -> M4.8

Outcome mapping: every instance runs to completion with `y` untouched
(`oracle=diff`) -> manifest `value-changing`, outcome `never` (Triton authors no
LoopBound check; silent).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

N_DEF = 256
KERNEL = 3
PAD_NEG = -128      # ext = 256 + 2*(-128) - 3 + 1 = -2   (M4.7)
PAD_EMPTY = -127    # ext = 256 + 2*(-127) - 3 + 1 =  0   (M4.8)


@triton.jit
def _work(x_ptr, y_ptr, offs, mask):
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(v, 0.0), mask=mask)


@triton.jit
def pb_kernel(x_ptr, y_ptr, N: tl.constexpr, PAD: tl.constexpr,
              KERNEL: tl.constexpr, SHAPE: tl.constexpr, BLOCK: tl.constexpr):
    """The loop bound is the padded output extent derived from `PAD`.  A
    negative or empty padded extent leaves the loop with zero iterations."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    r = tl.arange(0, BLOCK)
    ext = N + 2 * PAD - KERNEL + 1
    if SHAPE == 1:
        for k in range(0, ext):
            offs = base + k * BLOCK + r
            _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 2:
        for i in range(0, 2):
            for k in range(0, ext):
                offs = base + (i * 2 + k) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 3:
        for k in range(0, ext):
            for j in range(0, 2):
                offs = base + (k * 2 + j) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, ext):
            offs = base + k * BLOCK + r
            acc += tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        tl.store(y_ptr + base + r, acc, mask=base + r < N)


SHAPES = (1, 2, 3, 4)
BLOCKS = (64, 128)


def form_of(family: int):
    """(pad, shape, block) for a 1-based family index: four shapes x two pad
    degeneracies, laid out as contiguous ranges of 8."""
    idx = family - 1
    pad = PAD_NEG if idx // 4 == 0 else PAD_EMPTY
    return pad, SHAPES[idx % 4], BLOCKS[(idx // 4) % 2]


def run_family(family: int):
    from gpubuf import GpuBuf, sync
    pad, shape, block = form_of(family)
    x = GpuBuf.from_numpy(np.ones(N_DEF, dtype=np.float32))
    y = GpuBuf.from_numpy(np.full(N_DEF, -777.0, dtype=np.float32))
    grid = (triton.cdiv(N_DEF, block),)
    run = "ok"
    try:
        pb_kernel[grid](x, y, N_DEF, pad, KERNEL, shape, block)
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    if run != "ok":
        print("RESULT run=crash out=none")
        return
    got = y.to_host()
    ref = np.maximum(np.ones(N_DEF, dtype=np.float32), 0.0)
    out = "same" if np.allclose(got, ref, atol=1e-4, equal_nan=True) else "diff"
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
