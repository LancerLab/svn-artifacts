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

# Second realisation grid. `N = 8` is `4 kernels x 2 realisations`; the two
# realisations must be the same program at different extents (plan §2.4). ALT is
# the reduced second grid: same operators, different concrete extents, small
# enough that the launched M4 mutants stay inside the run budget.
ALT = {
    "relu": (127 * 511 + 7,),
    "sigmoid": (127 * 511 + 7,),
    "gelu": (127 * 511,),
    "elemwise_add": (127 * 511 + 7,),
    "reshape": (127 * 511 + 7,),
    "layer_normalization": (129, 511),
    "softmax": (48, 511),
    "reduce_mean": (32, 511),
    "transpose": (31, 129),
    "matmul": (48, 32, 96),
    "conv2d": (2, 4, 15, 15, 4, 3, 3),
    "max_pool2d": (8, 16, 16, 2),
    "batch_norm": (2, 8, 11, 15),
    "concat": (511, 39),
    "embedding": (511, 96, 29),
}

# order fixed so generation/stat tables are stable
CATEGORIES = [
    "batch_norm", "concat", "conv2d", "elemwise_add", "embedding", "gelu",
    "layer_normalization", "matmul", "max_pool2d", "reduce_mean", "relu",
    "reshape", "sigmoid", "softmax", "transpose",
]

GRIDS = {"small": SMALL, "alt": ALT}


def shape(cat, which="small"):
    return GRIDS[which][cat]
