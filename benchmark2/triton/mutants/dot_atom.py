"""M3-a (tl.dot atom) — mutation-only dot kernels with atom-violating tiles.

Interface: python3 dot_atom.py --family N --size full

The family's object is a `tl.dot` tile that violates the tensor-core atom rule
(M/N/K at least one atom). Triton checks the rule at JIT time and refuses the
program, so every instance is a created, non-inert mutant whose outcome is a
compile refusal (`avoided` shape) -- not an unexpressible slot: the source
states the defect, the compiler is what rejects it.

Four base shapes x two violations:
  variant 0 -> K = 8  (below the f16 K atom)
  variant 1 -> K = 4  (below the f16 K atom)

M/N below the atom is NOT a violation on this surface -- the frontend pads a
short M/N (probe: M=8, N=8 both lower), so only the K atom is a defective state.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl

SHAPES = [(64, 64, 64), (128, 32, 32), (32, 128, 64), (64, 64, 128)]


@triton.jit
def dotk(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    kernel = (family - 1) // 2
    variant = (family - 1) % 2
    M, N, K = SHAPES[kernel]
    K = 8 if variant == 0 else 4    # below the f16 K atom
    n = max(M * K, K * N, M * N)
    a = GpuBuf(n, np.ones(n, dtype=np.float16))
    b = GpuBuf(n, np.ones(n, dtype=np.float16))
    c = GpuBuf(n, np.zeros(n, dtype=np.float16))
    try:
        dotk[(1,)](a, b, c, M, N, K)
        sync()
        print("RESULT run=ok out=same")   # must never happen: the tile is illegal
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
