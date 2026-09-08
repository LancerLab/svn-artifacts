"""M1 level-2 mutant (plan §3.1 priority rows) — see mutants/README.md."""
import argparse
import sys
from pathlib import Path
import numpy as np
import triton
import triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpubuf import GpuBuf, randn, sync

CANARY = 4096

@triton.jit
def embed_kernel(tab_ptr, idx_ptr, y_ptr, N, D, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.load(idx_ptr + pid)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    v = tl.load(tab_ptr + i * D + offs, mask=mask)
    tl.store(y_ptr + pid * D + offs, v, mask=mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, required=True)
    args = ap.parse_args()
    import ctypes
    from gpubuf import _rt, _ck
    V, D = 256, 128
    table = randn(V * D)
    rng = np.random.default_rng(0)
    # family 1: idx holds V+3 (out of table range); family 2: idx = -1
    if args.family == 1:
        idx = np.array([V + 3], dtype=np.int32)
    else:
        idx = np.array([-1], dtype=np.int32)
    ib = GpuBuf.__new__(GpuBuf)
    ib.n = idx.size
    ib.dtype = np.dtype("int32")
    ib._esz = 4
    ib.ptr = ctypes.c_void_p()
    _ck(_rt.cudaMalloc(ctypes.byref(ib.ptr), ctypes.c_size_t(idx.size * 4)),
        "cudaMalloc")
    ib.from_host(idx)
    y = GpuBuf(D)
    run = "ok"
    try:
        embed_kernel[(idx.size,)](GpuBuf.from_numpy(table), ib, y, idx.size, D,
                                  BLOCK=triton.next_power_of_2(D))
        sync()
    except Exception as e:
        print(f"EXC {type(e).__name__}: {e}", file=sys.stderr)
        run = "crash"
    got = y.to_host() if run == "ok" else None
    if got is None:
        out = "none"
    else:
        bad_idx = int(idx[0]) % V  # what a wrapping read would return
        wrapped = table.reshape(V, D)[bad_idx]
        out = "same" if np.array_equal(got, wrapped) else "diff"
    print(f"RESULT run={run} out={out}")


if __name__ == "__main__":
    main()
