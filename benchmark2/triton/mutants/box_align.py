"""M3-e (base / inner-box alignment) — mutation-only descriptor inner-box kernels.

Interface: python3 box_align.py --family N --size full

M3-e's defect class is "base or inner box not aligned to the required
granularity" (M3.6 unaligned leading dim on a vectorized access, M3.7 inner-box
geometry not 128-bit aligned, M3.11/M3.16 the arch-owned residues).  For the
inner-box half (M3.7) the state IS writable: the last dimension of a descriptor
`block_shape` is a source value, and Triton's frontend requires it to span at
least 16 bytes.  A sub-16-byte inner box is a created, non-inert mutant whose
outcome is a compile refusal (`avoided`).

Eight forms (dtype x inner block, all last-dim < 16 bytes):
  f1 f32 [1,1] -> 4 B    f5 f16 [1,1] -> 2 B
  f2 f32 [1,2] -> 8 B    f6 f16 [1,2] -> 4 B
  f3 f32 [2,1] -> 4 B    f7 f16 [2,1] -> 2 B
  f4 f32 [4,1] -> 4 B    f8 f16 [4,2] -> 4 B
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import triton
import triton.language as tl


@triton.jit
def box_align(x_ptr, out_ptr, B0: tl.constexpr, B1: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, shape=[64, 256], strides=[256, 1],
                                  block_shape=[B0, B1])
    v = d.load([0, 0])
    tl.store(out_ptr + tl.arange(0, 2), tl.ravel(v)[0:2])


FORMS = {
    1: ("f32", 1, 1), 2: ("f32", 1, 2), 3: ("f32", 2, 1), 4: ("f32", 4, 1),
    5: ("f16", 1, 1), 6: ("f16", 1, 2), 7: ("f16", 2, 1), 8: ("f16", 4, 2),
}


def run_family(family: int):
    from gpubuf import GpuBuf, sync, install_descriptor_allocator
    install_descriptor_allocator()
    dt, b0, b1 = FORMS[family]
    npdt = np.float32 if dt == "f32" else np.float16
    x = GpuBuf(1 << 22, np.ones(1 << 22, dtype=npdt))
    out = GpuBuf(2, np.zeros(2, dtype=np.float32))
    try:
        box_align[(1,)](x, out, b0, b1)
        sync()
        print("RESULT run=ok out=same")   # must never happen: box < 16 B illegal
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
