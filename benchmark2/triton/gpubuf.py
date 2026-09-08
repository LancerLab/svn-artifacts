"""gpubuf — minimal CUDA buffer shim for the Triton lane (no torch).

Wraps libcudart via ctypes: device alloc + H2D/D2H copies, numpy-backed host
side. A `GpuBuf` exposes `.data_ptr()` which is all Triton needs as a kernel
argument. Reference checks are computed on the host with numpy.

Why: this machine's PyPI throughput cannot carry torch (~3 GB). The lane's
kernels and mutants only need alloc/copy/compare — nothing torch-specific.
"""
from __future__ import annotations

import ctypes
import numpy as np

_rt = ctypes.CDLL("libcudart.so")


def _ck(rc: int, what: str):
    if rc != 0:
        err = _rt.cudaGetErrorString(ctypes.c_int(rc)).decode()
        raise RuntimeError(f"{what}: cudaError {rc} ({err})")


class GpuBuf:
    """1-D device buffer backed by a numpy host array (f32/f16)."""

    def __init__(self, n: int, init: np.ndarray | None = None):
        self.n = n
        self.dtype = np.dtype("float32")
        self._esz = 4
        if init is not None:
            self.dtype = init.dtype
            self._esz = init.dtype.itemsize
        self.ptr = ctypes.c_void_p()
        _ck(_rt.cudaMalloc(ctypes.byref(self.ptr),
                           ctypes.c_size_t(n * self._esz)), "cudaMalloc")
        if init is not None:
            assert init.size == n
            self.from_host(init)

    @classmethod
    def from_numpy(cls, a: np.ndarray) -> "GpuBuf":
        return cls(a.size, np.ascontiguousarray(a).ravel())

    def data_ptr(self) -> int:
        return self.ptr.value

    def from_host(self, a: np.ndarray):
        _ck(_rt.cudaMemcpy(self.ptr, a.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_size_t(self.n * self._esz), ctypes.c_int(1)),
            "cudaMemcpy H2D")  # 1 = cudaMemcpyHostToDevice

    def to_host(self) -> np.ndarray:
        out = np.empty(self.n, dtype=self.dtype)
        _ck(_rt.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), self.ptr,
                           ctypes.c_size_t(self.n * self._esz), ctypes.c_int(2)),
            "cudaMemcpy D2H")  # 2 = cudaMemcpyDeviceToHost
        return out

    def free(self):
        _rt.cudaFree(self.ptr)
        self.ptr = ctypes.c_void_p()


def randn(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def sync():
    _ck(_rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
