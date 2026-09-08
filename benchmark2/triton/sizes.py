"""sizes.py — small/full shape table for the Triton lane (plan §2.4).

- SMALL: ragged exploration shapes (chosen so boundary defects manifest).
- FULL: the original benchmark case's own concrete dims (extracted from the
  #define blocks / host allocations of the provenance case recorded in
  benchmark2/settings/<category>.md; see full_shapes.json for provenance).

Dims are argument tuples for each kernel's wrapper. Small and full must be
the same program (plan recommendation 4): only the shapes differ.
"""

SMALL = {
    "relu": (127 * 1023 + 5,),
    "sigmoid": (127 * 1023 + 5,),
    "gelu": (127 * 1023,),
    "elemwise_add": (127 * 1023 + 5,),
    "reshape": (127 * 1023 + 5,),
    "layer_normalization": (197, 769),          # rows, cols
    "softmax": (64, 1023),
    "reduce_mean": (16, 1023),
    "transpose": (65, 127),                     # M, N
    "matmul": (64, 48, 64),                     # M, K, N
    "conv2d": (4, 8, 13, 13, 8, 3, 3),          # B, C, H, W, CO, KH, KW
    "max_pool2d": (16, 14, 14, 2),              # C, H, W, k (H%k==0 for the simple ref)
    "batch_norm": (4, 16, 9, 13),               # N, C, H, W
    "concat": (1023, 77),                       # nA, nB
    "embedding": (999, 128, 37),                # V, D, n_idx
}

# full = provenance case's concrete dims (see full_shapes.json)
FULL = {
    "relu": (16 * 512 * 2 * 2,),                          # I,J,K,L = 16,512,2,2
    "sigmoid": (16 * 512 * 7 * 5,),                       # I,J,K,L
    "gelu": (16 * 512 * 19 * 19,),                        # I,J,H,W
    "elemwise_add": (16 * 512 * 8 * 3,),                  # I,J,K,L
    "reshape": (64 * 128 * 28 * 28,),                     # B,C,height,width
    "layer_normalization": (16 * 512, 128 * 27),          # rows=(I,J), cols=(K,L)
    "softmax": (16 * 512, 56 * 56),                       # rows=(I,J), cols=H*W
    "reduce_mean": (32 * 2048, 768),                      # (N,SEQ_LEN), H
    "transpose": (512, 64),                               # M=512, N=64 (scale_h/w)
    "matmul": (128, 1280, 1000),                          # M, K, N=H
    "conv2d": (64, 128, 32, 32, 128, 1, 1),  # provenance case 11 (static 1x1)
    "max_pool2d": (16, 35, 35, 5),                        # channels, h, w, k
    "batch_norm": (16, 512, 16, 16),                      # N, C, H, W
    "concat": (16 * 512 * 4 * 4, 16 * 512 * 4 * 4),       # (I,J1,K,L), (I,J2,K,L)
    "embedding": (999, 768, 32 * 128),                    # vocab, dim, batch*seq
}

# Mutant runs must *manifest* the defect (specs §7 oracle): a perfectly
# divisible full shape would make a dropped mask a no-op. Mutants therefore
# run at FULL_RAGGED: the FULL magnitudes with the boundary extent perturbed
# to a non-divisible value (the original cases are dynamic-shape; H/W/N are
# symbolic at compile time, so any runtime extent is faithful).
FULL_RAGGED = {
    "relu": (16 * 512 * 2 * 2 + 5,),
    "sigmoid": (16 * 512 * 7 * 5 + 3,),
    "gelu": (16 * 512 * 19 * 19 + 1,),
    "elemwise_add": (16 * 512 * 8 * 3 + 5,),
    "reshape": (64 * 128 * 28 * 28 + 3,),
    "layer_normalization": (16 * 512, 128 * 27 + 1),  # ragged cols
    "softmax": (16 * 512, 56 * 56 + 1),
    "reduce_mean": (32 * 2048, 768 + 1),
    "transpose": (512, 65),                            # ragged N
    "matmul": (128, 1280, 1001),                       # ragged N
    "conv2d": (32, 128, 112, 112, 256, 3, 3),          # K=3 on 112 is ragged
    "max_pool2d": (16, 35, 35, 3),                     # ragged k on the full extent
    "batch_norm": (16, 512, 16, 15),                   # ragged W
    "concat": (16 * 512 * 4 * 4 + 1, 16 * 512 * 4 * 4),
    "embedding": (999, 768, 32 * 128),
}
