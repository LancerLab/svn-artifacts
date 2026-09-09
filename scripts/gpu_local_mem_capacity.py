#!/usr/bin/env python3
"""Compute a hardware-satisfiable --max-local-mem-capacity for this GPU.

Why this is needed
------------------
For dynamic shapes the compiler sizes the per-thread LOCAL arena to the full
`--max-local-mem-capacity` (croqtile/lib/mem_reuse.cpp:430-434), emitting e.g.

    alignas(16) unsigned char anon_2[1999964];   // ~2 MB *per thread*

CUDA must back every resident thread's local frame with device memory:

    reserve = capacity * maxThreadsPerSM * numSMs

On an H800 PCIe (114 SMs x 2048 threads, 85 GB) a 2 MB/thread frame needs
466.9 GB -- far more than the GPU has -- so the launch is rejected with
cudaErrorInvalidValue and the kernel NEVER RUNS.  Because the generated code
did not check the launch error, the harness still printed an "Execution time"
and the case was counted as a success.

cudaDeviceSetLimit(cudaLimitStackSize, N) fails for the same reason, so the
ceiling can be discovered empirically.  We take the largest power of two that
the driver accepts, then leave headroom for the tensors themselves.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The arena reservation competes with the actual tensor allocations, so do not
# let it consume the whole device.  0.75 is deliberately generous: the capacity
# must stay large enough that a case's real runtime local-memory need still
# passes the emitted `runtime_check(spm_size <= capacity)`, otherwise the case
# aborts on the host.  Prefer the largest value that satisfies BOTH the driver
# stack ceiling and the device-memory budget.
SAFETY_FRACTION = 0.75
# Resident threads the driver must be able to back with local memory.
MAX_THREADS_PER_SM = 2048
FALLBACK = 65536

_PROBE_SRC = r"""
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("0 0 0\n"); return 1; }
  size_t best = 0;
  for (size_t v = 1024; v <= (64ULL<<20); v *= 2) {
    cudaGetLastError();
    if (cudaDeviceSetLimit(cudaLimitStackSize, v) == cudaSuccess) best = v;
    else break;
  }
  printf("%zu %d %zu\n", best, p.multiProcessorCount, p.totalGlobalMem);
  return 0;
}
"""


def probe() -> tuple[int, int, int]:
    """Return (max_stack_bytes, num_sms, total_global_mem_bytes)."""
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    if not Path(nvcc).exists():
        return (0, 0, 0)
    with tempfile.TemporaryDirectory(prefix="cap_probe_") as td:
        src = Path(td) / "probe.cu"
        exe = Path(td) / "probe"
        src.write_text(_PROBE_SRC)
        try:
            b = subprocess.run([nvcc, "-O0", "-o", str(exe), str(src)],
                               capture_output=True, text=True, timeout=300)
            if b.returncode != 0:
                return (0, 0, 0)
            r = subprocess.run([str(exe)], capture_output=True, text=True,
                               timeout=120)
            best, sms, mem = r.stdout.split()
            return (int(best), int(sms), int(mem))
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return (0, 0, 0)


def safe_capacity() -> int:
    best, sms, mem = probe()
    if best <= 0:
        return FALLBACK
    # The driver already refused anything above `best`, so that is the hard
    # ceiling.  additionally cap it so the reservation stays a modest share
    # of device memory.
    if sms > 0 and mem > 0:
        by_mem = int(mem * SAFETY_FRACTION) // (sms * MAX_THREADS_PER_SM)
        # round down to a power of two
        p = 1
        while p * 2 <= by_mem:
            p *= 2
        best = min(best, p)
    return max(1024, best)


# Probing compiles and runs a CUDA binary, so cache it: every artifact script
# calls this once per case and must not pay that cost repeatedly.
_CACHED: int | None = None


def capacity() -> int:
    """Return the local-memory capacity to pass to choreo (cached).

    Override with CHOREO_MAX_LOCAL_MEM_CAPACITY to pin a value without
    re-probing (useful for reproducing a specific run or on a machine with
    no GPU/nvcc).
    """
    global _CACHED
    env = os.environ.get("CHOREO_MAX_LOCAL_MEM_CAPACITY")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    if _CACHED is None:
        _CACHED = safe_capacity()
    return _CACHED


def capacity_flag() -> str:
    """Return the choreo CLI flag, e.g. '--max-local-mem-capacity=262144'."""
    return f"--max-local-mem-capacity={capacity()}"


if __name__ == "__main__":
    cap = safe_capacity()
    if len(sys.argv) > 1 and sys.argv[1] == "--explain":
        best, sms, mem = probe()
        print(f"driver stack ceiling : {best} bytes", file=sys.stderr)
        print(f"SMs                  : {sms}", file=sys.stderr)
        print(f"device memory        : {mem/1e9:.1f} GB", file=sys.stderr)
        print(f"reservation at ceil  : "
              f"{best*sms*MAX_THREADS_PER_SM/1e9:.1f} GB", file=sys.stderr)
        print(f"chosen capacity      : {cap} bytes "
              f"({cap*sms*MAX_THREADS_PER_SM/1e9:.1f} GB reserved)",
              file=sys.stderr)
    print(cap)
