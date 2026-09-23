"""sizes.py — small-shape table for the CUTLASS/CuTe lane.

Shapes are copied verbatim from `iree/sizes.py::SMALL` so that every lane probes
the same 15-category case set at the same small magnitudes (plan §2.4: small and
full must be the same program, only the extents differ). The provenance of each
value is the corresponding `benchmark2/settings/<category>.md` case signature.

FULL is the provenance case's own concrete dims; this lane's reduced candidate
budget runs the SMALL grid only and records the size in every record.

Dims are argument tuples for each category's wrapper:
  - elementwise / reshape / relu / sigmoid / gelu: a flat element list
    (the operator's logical shape is carried separately in `logical`).
  - matmul: (M, K, N)
  - conv2d: (B, C, H, W, CO, KH, KW)
  - max_pool2d: (C, H, W, k)
  - batch_norm: (N, C, H, W)
  - concat: (nA, nB)
  - embedding: (V, D, n_idx)
  - reduction family: (rows, cols)
  - transpose: (M, N)
"""

# logical shapes (per category) are the source of truth for the CPU reference;
# the flat arg tuple is what the host wrapper allocates.
SMALL = {
    "relu": (127 * 1023 + 5,),
    "sigmoid": (127 * 1023 + 5,),
    "gelu": (127 * 1023,),
    "elemwise_add": (127 * 1023 + 5,),
    "reshape": (127 * 1023 + 5,),
    "layer_normalization": (197, 769),      # rows, cols
    "softmax": (64, 1023),                  # rows, cols
    "reduce_mean": (16, 1023),              # rows, cols
    "transpose": (65, 127),                 # M, N
    "matmul": (64, 48, 64),                 # M, K, N
    "conv2d": (4, 8, 13, 13, 8, 3, 3),      # B, C, H, W, CO, KH, KW
    "max_pool2d": (16, 14, 14, 2),          # C, H, W, k
    "batch_norm": (4, 16, 9, 13),           # N, C, H, W
    "concat": (1023, 77),                   # nA, nB
    "embedding": (999, 128, 37),            # V, D, n_idx
}

# order fixed so generation/stat tables are stable
CATEGORIES = [
    "batch_norm", "concat", "conv2d", "elemwise_add", "embedding", "gelu",
    "layer_normalization", "matmul", "max_pool2d", "reduce_mean", "relu",
    "reshape", "sigmoid", "softmax", "transpose",
]


def shape(cat):
    return SMALL[cat]
