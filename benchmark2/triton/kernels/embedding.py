"""embedding — y[i] = table[idx[i]] (settings/embedding.md)."""
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync


@triton.jit
def embed_kernel(tab_ptr, idx_ptr, y_ptr, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # one program per index
    i = tl.load(idx_ptr + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    v = tl.load(tab_ptr + i * D + offs, mask=mask)
    tl.store(y_ptr + pid * D + offs, v, mask=mask)


def embedding(table, idx, n_idx, dim):
    y = GpuBuf(n_idx * dim)
    embed_kernel[(n_idx,)](table, idx, y, n_idx, dim,
                           BLOCK=triton.next_power_of_2(dim))
    sync()
    return y


class IntBuf(GpuBuf):
    pass


if __name__ == "__main__":
    V, D = 50257, 128
    table = randn(V * D)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, V, size=1000).astype(np.int32)
    # int32 device buffer
    ib = GpuBuf.__new__(IntBuf)
    ib.n = idx.size
    ib.dtype = np.dtype("int32")
    ib._esz = 4
    import ctypes
    from gpubuf import _rt, _ck
    ib.ptr = ctypes.c_void_p()
    _ck(_rt.cudaMalloc(ctypes.byref(ib.ptr), ctypes.c_size_t(idx.size * 4)),
        "cudaMalloc")
    ib.from_host(idx)
    got = embedding(GpuBuf.from_numpy(table), ib, idx.size, D).to_host()
    want = table.reshape(V, D)[idx].ravel()
    assert np.array_equal(got, want), "MISMATCH"
    print("ok", (V, D, idx.size))
