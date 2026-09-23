"""M1-g (rank / arity) — mutation-only rank- and arity-violating kernels.

Interface: python3 rank_arity.py --family N --size full

The family's object is a rank, axis or arity argument outside its domain
(M1.15 `dimof` index >= rank, M1.16 `select` factor out of range, M1.17
5th-index on a rank-5 view, M1.20 `view`/`subspan` rank arity). Triton HAS a
source-level rank/index/arity model -- `tl.permute`, `tl.reshape`,
`tl.expand_dims`, `tl.sum(axis=)`, `tl.trans`, `tl.split` all take a rank- or
axis-shaped argument -- so the state IS expressible; the frontend refuses it at
trace time. Each instance is therefore a created, non-inert mutant with a
compile-refusal outcome (`avoided`), not an unexpressible slot.

Eight forms over four specs:
  f1 permute dims arity != rank                 M1.20
  f2 permute dims not a permutation             M1.15
  f3 sum axis == rank                           M1.17
  f4 sum axis = -(rank+1)                       M1.17
  f5 reshape changes the element count          M1.20
  f6 expand_dims axis > rank                    M1.15
  f7 trans on a rank-1 tensor                   M1.16
  f8 tl.split last dim != 2                     M1.16
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

N = 256


@triton.jit
def f_permute_arity(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    a = tl.reshape(a, (16, 16))
    b = tl.permute(a, (0, 1, 2))
    tl.store(out_ptr + tl.arange(0, 256), tl.ravel(b))


@triton.jit
def f_permute_dup(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    a = tl.reshape(a, (16, 16))
    b = tl.permute(a, (0, 0))
    tl.store(out_ptr + tl.arange(0, 256), tl.ravel(b))


@triton.jit
def f_sum_rank(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    a = tl.reshape(a, (16, 16))
    b = tl.sum(a, axis=2)
    tl.store(out_ptr + tl.arange(0, 16), b)


@triton.jit
def f_sum_neg(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    a = tl.reshape(a, (16, 16))
    b = tl.sum(a, axis=-3)
    tl.store(out_ptr + tl.arange(0, 16), b)


@triton.jit
def f_reshape_count(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    b = tl.reshape(a, (8, 8))
    tl.store(out_ptr + tl.arange(0, 256), tl.ravel(b))


@triton.jit
def f_expand(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    b = tl.expand_dims(a, axis=2)
    tl.store(out_ptr + tl.arange(0, 512), tl.ravel(b))


@triton.jit
def f_trans1(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    b = tl.trans(a)
    tl.store(out_ptr + tl.arange(0, 256), b)


@triton.jit
def f_split(x_ptr, out_ptr):
    a = tl.load(x_ptr + tl.arange(0, 256))
    a = tl.reshape(a, (64, 4))
    p, q = tl.split(a)
    tl.store(out_ptr + tl.arange(0, 64), p)


KERN = {1: f_permute_arity, 2: f_permute_dup, 3: f_sum_rank, 4: f_sum_neg,
        5: f_reshape_count, 6: f_expand, 7: f_trans1, 8: f_split}


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    x = GpuBuf(N, np.arange(N, dtype=np.float32))
    out = GpuBuf(N, np.zeros(N, dtype=np.float32))
    try:
        KERN[family][(1,)](x, out)
        sync()
        print("RESULT run=ok out=same")   # must never happen: rank/arity illegal
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
