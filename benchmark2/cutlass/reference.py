"""reference.py — CPU references for the CUTLASS/CuTe lane gate.

The GPU kernels generate their inputs deterministically with `cut_fill`
(kernels/common.cuh) from fixed seeds; this module reproduces exactly the same
inputs and computes the reference output. The lane's *gate* only asks that the
unmutated kernel agrees with this reference within tolerance; the mutation
oracle (mutation output vs unmutated output) needs no reference at all.

Seeds are mirrored from the kernels (X=12345, weights=67890, affine
params=222/333/444/555). Keep them in sync if a kernel's seeds change.
"""

import numpy as np

SEED_X = 12345
SEED_W = 67890
SEED_G = 222
SEED_B = 333
SEED_MU = 444
SEED_VAR = 555

ATOL = 1e-4
RTOL = 1e-4


def read_bin(path):
    """Load a raw float32 dump produced by cut_dump."""
    return np.fromfile(path, dtype=np.float32)


def fill(seed, n):
    """Bit-exact replica of kernels/common.cuh::cut_fill."""
    s = seed if seed else 1
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((np.int32(s >> 8) % 1000) / 1000.0)
    return out


def inputs(cat, shape):
    """Return {name: ndarray} for a category's declared kernel inputs."""
    if cat in ("elemwise_add",):
        (n,) = shape
        return {"A": fill(SEED_X, n), "B": fill(SEED_W, n)}
    if cat in ("relu", "sigmoid", "gelu", "reshape"):
        (n,) = shape
        return {"A": fill(SEED_X, n)}
    if cat == "transpose":
        M, N = shape
        return {"A": fill(SEED_X, M * N)}
    if cat == "concat":
        nA, nB = shape
        return {"A": fill(SEED_X, nA), "B": fill(SEED_W, nB)}
    if cat == "embedding":
        V, D, nidx = shape
        return {"T": fill(SEED_X, V * D), "I": fill(SEED_W, nidx)}
    if cat == "reduce_mean":
        rows, cols = shape
        return {"A": fill(SEED_X, rows * cols)}
    if cat in ("softmax",):
        rows, cols = shape
        return {"A": fill(SEED_X, rows * cols)}
    if cat in ("layer_normalization",):
        rows, cols = shape
        return {"A": fill(SEED_X, rows * cols), "G": fill(SEED_G, cols),
                "B": fill(SEED_B, cols)}
    if cat == "matmul":
        M, K, N = shape
        return {"A": fill(SEED_X, M * K), "B": fill(SEED_W, K * N)}
    if cat == "conv2d":
        B_, C, H, W, CO, KH, KW = shape
        return {"X": fill(SEED_X, B_ * C * H * W),
                "W": fill(SEED_W, CO * C * KH * KW)}
    if cat == "batch_norm":
        N, C, H, W = shape
        return {"X": fill(SEED_X, N * C * H * W), "G": fill(SEED_G, C),
                "B": fill(SEED_B, C), "Mu": fill(SEED_MU, C),
                "Va": fill(SEED_VAR, C)}
    if cat == "max_pool2d":
        C, H, W, K = shape
        return {"X": fill(SEED_X, C * H * W)}
    raise KeyError(cat)


def reference(cat, shape, a):
    """CPU reference output, flattened float32."""
    if cat == "elemwise_add":
        return (a["A"] + a["B"]).astype(np.float32)
    if cat == "relu":
        return np.maximum(a["A"], np.float32(0)).astype(np.float32)
    if cat == "sigmoid":
        x = a["A"].astype(np.float64)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if cat == "gelu":
        from math import erf
        x = a["A"].astype(np.float64)
        e = np.array([erf(float(v) * 0.7071067811865476) for v in x])
        return (0.5 * x * (1.0 + e)).astype(np.float32)
    if cat == "reshape":
        return a["A"].astype(np.float32).copy()
    if cat == "transpose":
        M, N = shape
        return a["A"].reshape(M, N).T.astype(np.float32).ravel()
    if cat == "concat":
        return np.concatenate([a["A"], a["B"]]).astype(np.float32)
    if cat == "embedding":
        V, D, nidx = shape
        idx = np.minimum((fill(SEED_W, nidx) * np.float32(V)).astype(np.int64),
                         V - 1)
        T = a["T"].reshape(V, D)
        return T[idx].astype(np.float32).ravel()
    if cat == "reduce_mean":
        rows, cols = shape
        x = a["A"].reshape(rows, cols).astype(np.float64)
        return x.mean(axis=1).astype(np.float32)
    if cat == "softmax":
        rows, cols = shape
        x = a["A"].reshape(rows, cols).astype(np.float64)
        m = x.max(axis=1, keepdims=True)
        e = np.exp(x - m)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32).ravel()
    if cat == "layer_normalization":
        rows, cols = shape
        x = a["A"].reshape(rows, cols).astype(np.float64)
        g = a["G"].astype(np.float64)
        b = a["B"].astype(np.float64)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        y = (x - mean) / np.sqrt(var + 1e-5) * g + b
        return y.astype(np.float32).ravel()
    if cat == "matmul":
        M, K, N = shape
        A = a["A"].reshape(M, K).astype(np.float64)
        B = a["B"].reshape(K, N).astype(np.float64)
        return (A @ B).astype(np.float32).ravel()
    if cat == "conv2d":
        B_, C, H, W, CO, KH, KW = shape
        OH, OW = H - KH + 1, W - KW + 1
        X = a["X"].reshape(B_, C, H, W).astype(np.float64)
        Wt = a["W"].reshape(CO, C, KH, KW).astype(np.float64)
        Y = np.zeros((B_, CO, OH, OW), dtype=np.float64)
        for co in range(CO):
            for oh in range(OH):
                for ow in range(OW):
                    Y[:, co, oh, ow] = np.sum(
                        X[:, :, oh:oh + KH, ow:ow + KW] * Wt[co], axis=(1, 2, 3))
        return Y.astype(np.float32).ravel()
    if cat == "batch_norm":
        N, C, H, W = shape
        X = a["X"].reshape(N, C, H, W).astype(np.float64)
        g = a["G"].astype(np.float64)[None, :, None, None]
        b = a["B"].astype(np.float64)[None, :, None, None]
        mu = a["Mu"].astype(np.float64)[None, :, None, None]
        va = a["Va"].astype(np.float64)[None, :, None, None]
        y = (X - mu) / np.sqrt(va + 1e-5) * g + b
        return y.astype(np.float32).ravel()
    if cat == "max_pool2d":
        C, H, W, K = shape
        x = a["X"].reshape(C, H // K, K, W // K, K).astype(np.float32)
        return x.max(axis=(2, 4)).astype(np.float32).ravel()
    raise KeyError(cat)


def gate(cat, got, ref):
    """Return (ok, max_abs_diff, max_rel_diff)."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        return False, float("inf"), float("inf")
    d = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), ATOL)
    return bool(np.all(d <= ATOL + RTOL * denom)), float(d.max()), float((d / denom).max())
