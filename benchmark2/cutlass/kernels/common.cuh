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
//   LOOP      tile-loop trip count                 (M4)
//   STEP      tile-loop stride                     (M4)
//   VEC       copy vector width in elements        (M3.6)
//   SALIGN    shared base alignment in elements    (M3.6/M3.11)
//   SMEM_ELT  dynamic shared bytes (elements)      (M3/L1)
//   ATOM_*    MMA atom geometry                    (M3.1)
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
#ifndef VEC
#define VEC 4
#endif
#ifndef SALIGN
#define SALIGN 16
#endif
#ifndef SMEM_ELT
#define SMEM_ELT 0
#endif

// deterministic pseudo-random fill (matches reference.py::fill)
static inline void cut_fill(float* p, size_t n, uint32_t seed) {
  uint32_t s = seed ? seed : 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    p[i] = (float)((int32_t)(s >> 8) % 1000) / 1000.0f;
  }
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
