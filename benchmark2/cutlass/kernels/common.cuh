// common.cuh — shared host harness for the CUTLASS/CuTe lane kernels.
//
// Every operator kernel provides:
//   - a device functor `op(dst, src...)` or a kernel `k_<cat>(...)`,
//   - the shapes via `-D` macros from lane.py.
// This header provides the deterministic input init, the host launch, and the
// output dump so the oracle compares byte-exact against the CPU reference.
//
// Mutation knobs are compile-time macros with safe defaults (see lane.py):
//   TILE      elements per block-tile              (M3/M4 shape)
//   LOOP      tile-loop trip count                 (M4; -1 => full)
//   STEP      tile-loop stride                     (M4)
//   VEC       copy vector width in elements        (M3.6)
//   SALIGN    shared base alignment in elements    (M3.6/M3.11)
//   SMEM_ELT  dynamic shared bytes (elements)      (M3/L1)
//   ATOM_*    MMA atom geometry                    (M3.1)
//
// Runtime knob (M4.3): env CUT_LOOP_RT >= 0 overrides the loop bound on the
// host; it is passed to the kernel as `rt`. Base runs leave it unset (-1).
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

#ifndef TILE
#define TILE 256
#endif
#ifndef LOOP
#define LOOP -1   // -1 => derive the full trip count at runtime; 0 => zero-trip
#endif
#ifndef STEP
#define STEP 1
#endif
#ifndef PBOUND
#define PBOUND -1   // -1 => derive the parallel grid; >=0 => override (M4.2)
#endif
#ifndef REVB
#define REVB 0      // 1 => reversed/overrunning bound: upper loop one past (M1.7)
#endif
#ifndef NBOUND
#define NBOUND 0    // 1 => negative loop bound (M4.6)
#endif
#ifndef EMPTY
#define EMPTY 0     // M4.4: bound>0 over an empty inner space (neutral control)
#endif
#ifndef ZSTRIDE
#define ZSTRIDE 0   // M1.6: stride-0 traversal over an empty range (neutral)
#endif
#ifndef PADEXT
#define PADEXT 4    // M4.7/M4.8: padded extent (0 => empty, <0 => negative)
#endif
#ifndef VEC
#define VEC 4
#endif
#ifndef SALIGN
#define SALIGN 16
#endif
#ifndef SMEM_ELT
#define SMEM_ELT 0
#endif

// per-category concrete dims (defaults match sizes.py::SMALL)
#ifndef N_ELEM
#define N_ELEM (127 * 1023 + 5)
#endif
#ifndef TR_M
#define TR_M 65
#endif
#ifndef TR_N
#define TR_N 127
#endif
#ifndef CC_NA
#define CC_NA 1023
#endif
#ifndef CC_NB
#define CC_NB 77
#endif
#ifndef EM_V
#define EM_V 999
#endif
#ifndef EM_D
#define EM_D 128
#endif
#ifndef EM_N
#define EM_N 37
#endif
#ifndef RM_ROWS
#define RM_ROWS 16
#endif
#ifndef RM_COLS
#define RM_COLS 1023
#endif

// deterministic pseudo-random fill (matches reference.py::fill)
static inline void cut_fill(float* p, size_t n, uint32_t seed) {
  uint32_t s = seed ? seed : 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    p[i] = (float)((int32_t)(s >> 8) % 1000) / 1000.0f;
  }
}

// M4.4 / M1.6 neutral controls: a bound that is exercised over an empty
// iteration space. Both are no-ops by construction, so a correct oracle must
// record them as `noop` (path avoided); observing anything else is a bug in
// the oracle, not in the kernel.
__device__ __forceinline__ void cut_empty_probe() {
  volatile int sink = 0;
  for (int i = 0; i < EMPTY; ++i)
    for (int j = 0; j < 0; ++j) sink += j;
  (void)sink;
}
__device__ __forceinline__ void cut_zstride_probe() {
  volatile long sink = 0;
  for (long i = 0; i < ZSTRIDE; ++i) sink += i * 0;
  (void)sink;
}

// M4.7/M4.8 mutation-only pad category: a padded smem<->smem copy whose
// trailing extent is the compile-time `PADEXT`. PADEXT=0 is an empty padded
// extent; PADEXT<0 is a negative one. The scratch region is never read, so
// the probe cannot change the operator's output.
template <int P>
__device__ __forceinline__ void cut_pad_probe(float* scratch) {
  auto l = cute::make_layout(
      cute::make_shape(cute::Int<8>{}, cute::Int<P>{}),
      cute::make_stride(cute::Int<8>{}, cute::Int<1>{}));
  auto a = cute::make_tensor(cute::make_smem_ptr(scratch), l);
  auto b = cute::make_tensor(cute::make_smem_ptr(scratch + 128), l);
  cute::copy(a, b);
}

static inline int cut_env_int(const char* k, int dflt) {
  const char* v = getenv(k);
  return v && *v ? atoi(v) : dflt;
}

// dump a contiguous float buffer as raw little-endian bytes
static inline int cut_dump(const char* path, const void* data, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return 1;
  f.write(reinterpret_cast<const char*>(data), (std::streamsize)bytes);
  return f ? 0 : 1;
}

#define CUT_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(e_));             \
      return 2;                                                                \
    }                                                                          \
  } while (0)
