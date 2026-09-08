"""M1/M3 mutant variants for the Triton minimal set.

Each file exposes the same interface:

    python3 <file>.py --family N

runs the mutated kernel on an oracle-chosen input and prints exactly one line:

    RESULT run=<ok|crash> out=<same|diff|none>

The driver classifies the outcome from that line plus the exit code; the
manifestation oracle is applied per specs §7.

M1 families (specs/mutation-specs.md §1), translated to Triton idiom:
 1 dropped boundary mask   (mask removed from load+store)
 2 off-by-one bound        (mask offs < n+1)
 3 negative index          (load at offs-1 without guard)
 4 transposed/bad stride   (row stride off by one)
 5 offset view overrun     (base pointer advanced past the tail)
 6 zero-stride/empty range (degenerate extent reaches the kernel)

M3 families (specs §3) for matmul/conv2d:
 1 tl.dot contraction dim not divisible by the tensor-core atom (K % 16 != 0)
 2 misaligned base / tile exceeding shared-memory budget
"""
