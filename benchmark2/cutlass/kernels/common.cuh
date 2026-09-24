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
// `M1DEF` not `M1`: CuTe headers use `M1` as an identifier, so a macro of that
// name rewrites them.
#ifndef M1DEF
#define M1DEF 0     // M1 element-access defect, applied to the operator's own
                    // index arithmetic: 2 off-by-one, 4 bad stride, 9 displaced
                    // base, 12 wrong loop var, 14 swapped tile coord, 15 index
                    // >= rank, 19 narrow carrier, 11 read/write overlap.
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

// M3 hardware-constraint defect, applied to the operator's own element index
// `i` of extent `n`. `M3` holds the *spec number*, not the family letter, so
// the attribution reads directly off the macro. `M3` is a safe name (unlike
// `M1`, which CuTe uses as an identifier). Base (M3==0) is the identity.
//   1  M3-a  atom divisibility: the tail atom is dropped
//   2  M3-b  descriptor dim: the leading extent reaches past 2^24
//   3  M3-c  byte / swizzle box: the ceiled box wraps
//   4  M3-d  5-D descriptor footprint exceeds 4 GB (32-bit product wraps)
//   6  M3-e  base / inner box not aligned to the descriptor granularity
//   8  M3-f  descriptor rank outside the assessed [1,5]
//   9  M3-g  pad field overruns its assessed range
//   12 M3-h  on-chip tile exceeds the per-SM budget (index folds back)
#ifndef M3
#define M3 0
#endif

__device__ __forceinline__ long cut_m3_read(long i, long n) {
  long r = i;
  if (M3 == 1)  r = (i >= n / 2) ? (i - n / 2) : i; // a: dropped tail atom
  if (M3 == 2)  r = (i + 1) % n;                   // b: 2^24 descriptor stride
  if (M3 == 3)  r = (i + n / 2) % n;               // c: box byte wrap
  if (M3 == 4)  r = (i / 2) % n;                   // d: 5-D footprint wrap
  if (M3 == 6)  r = (i + 3) % n;                   // e: unaligned base
  if (M3 == 8)  r = (i / 2) % n;                   // f: rank-6 alias
  if (M3 == 9)  r = (i + 7) % n;                   // g: pad overrun
  if (M3 == 12) r = i % (n - n / 8);               // h: on-chip capacity fold
  return r;
}

// M2 shape-compatibility defect, applied to the operator's own element index
// `i` of extent `n`. `M2` holds the spec number. Base (M2==0) is the identity.
//   1  M2-a  wrong leading extent (secondary operand / scale)
//   6  M2-b  two extents transposed
//   7  M2-c  reduced-rank view (a dimension dropped)
//   8  M2-d  broadcast extent set to 1 instead of N
//   10 M2-e  transpose permutation on a square operand (layout, not extent)
//   5  M2-f  partial / duplicate write
//   15 M2-g  pad_low <-> pad_high swapped (total length preserved)
//   17 M2-h  runtime-shaped span (the size check is skipped)
#ifndef M2
#define M2 0
#endif

__device__ __forceinline__ long cut_m2_read(long i, long n) {
  long r = i;
  if (M2 == 1)  r = (i + 1) % n;                   // a: wrong leading extent
  if (M2 == 6)  r = (i / 2) % n;                   // b: extents transposed
  if (M2 == 7)  r = i % (n / 2);                   // c: reduced-rank view
  if (M2 == 8)  r = i - (i % 2);                   // d: broadcast extent 1
  if (M2 == 10) r = (i ^ 1) % n;                   // e: square transpose
  if (M2 == 5)  r = (i + n / 2) % n;               // f: partial/duplicate write
  if (M2 == 15) r = (i + n / 4) % n;               // g: pad fields swapped
  if (M2 == 17) r = (i + 3) % n;                   // h: runtime-shaped span
  return r;
}

// M1 element-access defect on a column index `k` of extent `cols`. `M1DEF`
// picks the defect; the base (M1DEF==0) is the identity.
__device__ __forceinline__ long cut_m1_read(long k, long cols) {
  long r = k;
  if (M1DEF == 4)  r = (k * 2) % cols;   // M1.4 non-unit stride
  if (M1DEF == 9)  r = k + 1;            // M1.9 displaced base
  if (M1DEF == 12) r = (k + 1) % cols;   // M1.12 wrong loop variable
  // M1.19: narrow carrier. Truncate the row-local index to 16 bits; it aliases
  // within the same 64K window, and the caller skips aliases past `cols`.
  if (M1DEF == 19) r = k & 0xFFFF;
  return r;
}
__device__ __forceinline__ bool cut_m1_bound(long k, long cols) {
  if ((M1DEF == 19) && (k & 0xFFFF) >= cols)       // M1.19 alias past extent
    return false;
  return (M1DEF == 2) ? (k <= cols) : (k < cols);   // M1.2 off-by-one
}
// M1.14: `off` is the tile/row coordinate; the defect reads the *next* one.
__device__ __forceinline__ long cut_m1_row(long r, long rows) {
  return (M1DEF == 14) ? ((r + 1) % rows) : r;      // swapped tile coordinate
}
// M1.12 (M1-d, "wrong loop variable for a dimension"): the leading (row)
// coordinate is taken from the trailing (column) loop variable `k`. Every
// assessed grid is wide (cols > rows), so `k * cols` runs past the row extent.
__device__ __forceinline__ long cut_m1_rowbase(long r, long rows, long k,
                                               long cols) {
  if (M1DEF == 12) return k * cols;              // M1.12 wrong loop variable
  return cut_m1_row(r, rows) * cols;
}

// M1-g / M1.15: an index at or past the rank. Only `get<2>` of a rank-2 shape
// is instantiated, and only when the M1 knob selects it, so the base kernel
// still compiles.
template <int M, bool Bad = (M == 15)> struct CutM1RankProbe;
template <int M> struct CutM1RankProbe<M, false> {
  __device__ __forceinline__ static void run() {}
};
template <int M> struct CutM1RankProbe<M, true> {
  __device__ __forceinline__ static void run() {
    auto s = cute::make_shape(cute::Int<4>{}, cute::Int<4>{});
    (void)cute::get<(M - 13)>(s);   // M==15 => get<2> of a rank-2 shape
  }
};

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
