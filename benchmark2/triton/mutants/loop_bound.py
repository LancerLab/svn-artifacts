"""M4 (iteration-space invalidity) — mutation-only loop-bound kernels.

Interface: python3 loop_bound.py --family N --size full   (see mutants/README.md)

M4 is the `LoopBound` class: the defect is always on the **iteration space**,
never on an address.  Triton authors its loops in Python (`range` /
`tl.static_range`), so the loop bound is a source value on this surface.  The
family's edit is a bound/step that is legal Python but leaves an empty (or
never-advancing) iteration space.

Six defect idioms, one per family group (47 instances total):

  group A  `M4-a` zero bound       `range(0, 0)`        legal-empty, silent
  group B  `M4-b` negative bound   `range(0, -1)`       legal-empty, silent
  group C  `M4-c` runtime-zero     `range(0, niter)`    niter = 0 at launch
  group D  `M4-d` empty space      `range(0, BLOCK)`    bound > 0, extent 0
  group E  `M4-e` reversed bound   `range(BLOCK, 0)`    empty, silent
  group F  `M4-f` zero step        `tl.static_range(0, BLOCK, 0)`
                                                       REFUSED at trace

Each group is realised as eight structure variants: four loop shapes (single
tile loop; 2-deep nest with the defect outside; 2-deep nest with the defect
inside; reduction loop) x two block sizes, exactly the "structure variants"
pattern the other lanes use to fill a family.

Outcome mapping:
  A/B/C/D/E  run to completion, output untouched  -> manifest `corrupts`,
             outcome `never` (the lane authors no LoopBound check; silent).
  F          the frontend raises `CompilationError` at trace (a `static_range`
             with step 0) -> manifest `corrupts`, outcome `compile`
             (`avoided`).  NOTE: plain `range(0, K, 0)` does NOT refuse -- it
             lowers to `scf.for ... step 0` and hangs (probed 2026-09-24); the
             refusal is specific to `tl.static_range`, which Python evaluates
             eagerly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


@triton.jit
def _work(x_ptr, y_ptr, offs, mask):
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(v, 0.0), mask=mask)


@triton.jit
def lb_kernel(x_ptr, y_ptr, N, LB: tl.constexpr, UB: tl.constexpr,
              SHAPE: tl.constexpr, BLOCK: tl.constexpr):
    """Empty-loop defects A (0,0), B (0,-1), E (BLOCK,0); empty space D
    (0,BLOCK with N=0).  `range(LB, UB)` is lowered to `scf.for`; an empty
    range yields zero iterations and the kernel runs to completion."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    r = tl.arange(0, BLOCK)
    if SHAPE == 1:
        for k in range(LB, UB):
            offs = base + k * BLOCK + r
            _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 2:
        for i in range(0, 2):
            for k in range(LB, UB):
                offs = base + (i * 2 + k) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 3:
        for k in range(LB, UB):
            for j in range(0, 2):
                offs = base + (k * 2 + j) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(LB, UB):
            offs = base + k * BLOCK + r
            acc += tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        tl.store(y_ptr + base + r, acc, mask=base + r < N)


@triton.jit
def lb_rtzero(x_ptr, y_ptr, N, niter, SHAPE: tl.constexpr, BLOCK: tl.constexpr):
    """Defect C: the bound is symbolic and is 0 only at run time."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    r = tl.arange(0, BLOCK)
    if SHAPE == 1:
        for k in range(0, niter):
            offs = base + k * BLOCK + r
            _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 2:
        for i in range(0, 2):
            for k in range(0, niter):
                offs = base + (i * 2 + k) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 3:
        for k in range(0, niter):
            for j in range(0, 2):
                offs = base + (k * 2 + j) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, niter):
            offs = base + k * BLOCK + r
            acc += tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        tl.store(y_ptr + base + r, acc, mask=base + r < N)


@triton.jit
def lb_step0(x_ptr, y_ptr, N, SHAPE: tl.constexpr, BLOCK: tl.constexpr):
    """Defect F: a step of 0 in an unrolled iteration.  `tl.static_range` is
    evaluated eagerly by the frontend, so Python raises `range() arg 3 must not
    be zero` and Triton wraps it in a `CompilationError`."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    r = tl.arange(0, BLOCK)
    if SHAPE == 1:
        for k in tl.static_range(0, BLOCK, 0):
            offs = base + k * BLOCK + r
            _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 2:
        for i in tl.static_range(0, 2):
            for k in tl.static_range(0, BLOCK, 0):
                offs = base + (i * 2 + k) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    elif SHAPE == 3:
        for k in tl.static_range(0, BLOCK, 0):
            for j in tl.static_range(0, 2):
                offs = base + (k * 2 + j) * BLOCK + r
                _work(x_ptr, y_ptr, offs, offs < N)
    else:
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in tl.static_range(0, BLOCK, 0):
            offs = base + k * BLOCK + r
            acc += tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        tl.store(y_ptr + base + r, acc, mask=base + r < N)


# family -> (defect, shape, block); 4 shapes x 2 block sizes per defect group,
# laid out as contiguous ranges of 8 (7 for M4-d).
GROUPS = [("A", 8), ("B", 8), ("C", 8), ("D", 7), ("E", 8), ("F", 8)]
SHAPES = (1, 2, 3, 4)
BLOCKS = (64, 128)


def form_of(family: int):
    """(group, shape, block) for a 1-based family index."""
    n = family
    for name, cnt in GROUPS:
        if n <= cnt:
            idx = n - 1
            return name, SHAPES[idx % 4], BLOCKS[(idx // 4) % 2]
        n -= cnt
    raise SystemExit(f"family out of range: {family}")


N_DEF = 256


def run_family(family: int):
    from gpubuf import GpuBuf, sync
    group, shape, block = form_of(family)
    x = GpuBuf.from_numpy(np.ones(N_DEF, dtype=np.float32))
    y = GpuBuf.from_numpy(np.full(N_DEF, -777.0, dtype=np.float32))
    grid = (triton.cdiv(N_DEF, block),)
    run = "ok"
    try:
        if group == "A":
            lb_kernel[grid](x, y, N_DEF, 0, 0, shape, block)
        elif group == "B":
            lb_kernel[grid](x, y, N_DEF, 0, -1, shape, block)
        elif group == "C":
            lb_rtzero[grid](x, y, N_DEF, 0, shape, block)
        elif group == "D":
            # bound is positive (BLOCK), the extent is 0 -> empty space
            lb_kernel[grid](x, y, 0, 0, block, shape, block)
        elif group == "E":
            lb_kernel[grid](x, y, N_DEF, block, 0, shape, block)
        elif group == "F":
            lb_step0[grid](x, y, N_DEF, shape, block)
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
    if not 1 <= args.family <= 47:
        raise SystemExit(f"family must be 1..47, got {args.family}")
    run_family(args.family)


if __name__ == "__main__":
    main()
