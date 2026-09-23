// probe_m1.cu — COMPILE-ONLY M1 (element-access) probes for the CuTe surface.
//
// Each PROBE id is one half of a (control, mutant) pair, compiled with
// `nvcc -c` (never linked, never run) exactly like the M3 battery:
//   control OK  + mutant FAIL  -> ct-check   (CuTe statically detains the defect)
//   control OK  + mutant OK    -> unchecked  (the surface does not detain it)
//   control FAIL               -> generator defect (fix the probe; emit no cell)
//
// The point of this battery is the SHAPE of the answer: on a hand-written
// CuTe kernel the index/stride/bound/tile defects are the author's own index
// arithmetic, so only the rank/arity defect has a CuTe static_assert behind it.
//
// PROBE map:  10/11 M1.2  bound off-by-one          | 20/21 M1.4  transposed stride
//             30/31 M1.9  offset, extent not shrunk | 40/41 M1.12 wrong loop var
//             50/51 M1.14 tile coordinate swapped   | 60/61 M1.19 narrow carrier
//             70/71 M1.15 index >= rank             | 80/81 M1.11 tile overlap
#include <cute/tensor.hpp>
using namespace cute;

// ---- M1-a / M1.2: `p#n` off-by-one (bound reads one past the extent) -------
#if PROBE == 10 || PROBE == 11
void host_probe() {
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}));
  float acc = 0.f;
#if PROBE == 10
  for (int i = 0; i < 16; ++i) acc += t(i, 0);
#else
  for (int i = 0; i <= 16; ++i) acc += t(i, 0);
#endif
  (void)acc;
}
#endif

// ---- M1-b / M1.4: transposed / non-contiguous stride (extents intact) ------
#if PROBE == 20 || PROBE == 21
void host_probe() {
#if PROBE == 20
  auto L = make_layout(make_shape(Int<16>{}, Int<16>{}),
                       make_stride(Int<16>{}, Int<1>{}));
#else
  auto L = make_layout(make_shape(Int<16>{}, Int<16>{}),
                       make_stride(Int<1>{}, Int<16>{}));
#endif
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr), L);
  auto x = t(3, 5);
  (void)x;
}
#endif

// ---- M1-c / M1.9: base displaced without shrinking the extent --------------
#if PROBE == 30 || PROBE == 31
void host_probe() {
  auto L = make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{});
#if PROBE == 30
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr), L);
#else
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr) + 8, L);
#endif
  auto x = t(0, 0);
  (void)x;
}
#endif

// ---- M1-d / M1.12: wrong loop variable for a dimension (broadcast reuse) ---
#if PROBE == 40 || PROBE == 41
void host_probe() {
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<16>{}, Int<16>{}), GenRowMajor{}));
  float acc = 0.f;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
#if PROBE == 40
      acc += t(i, j);
#else
      acc += t(i, i);
#endif
    }
  (void)acc;
}
#endif

// ---- M1-e / M1.14: tile coordinate over/underflow (element index legal) ----
#if PROBE == 50 || PROBE == 51
void host_probe() {
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{}));
  int bx = 1, by = 2;
#if PROBE == 50
  auto tile = local_tile(t, make_shape(Int<16>{}, Int<16>{}), make_coord(by, bx));
#else
  auto tile = local_tile(t, make_shape(Int<16>{}, Int<16>{}), make_coord(bx, by));
#endif
  (void)tile;
}
#endif

// ---- M1-f / M1.19: index legal but the carrier cannot name it --------------
#if PROBE == 60 || PROBE == 61
void host_probe() {
  long long n = 1LL << 33;
  auto t = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(n), make_stride(1LL)));
#if PROBE == 60
  long long i = (1LL << 32) + 5;
#else
  int i = (int)((1LL << 32) + 5);
#endif
  auto x = t(i);
  (void)x;
}
#endif

// ---- M1-g / M1.15: index >= rank (the one CuTe static_assert) --------------
#if PROBE == 70 || PROBE == 71
void host_probe() {
  auto s = make_shape(Int<4>{}, Int<4>{});
#if PROBE == 70
  auto x = get<1>(s);
#else
  auto x = get<2>(s);
#endif
  (void)x;
}
#endif

// ---- M1-h / M1.11: concurrent accesses address the same live slot ----------
#if PROBE == 80 || PROBE == 81
void host_probe() {
  auto L = make_layout(make_shape(Int<16>{}), make_stride(Int<1>{}));
#if PROBE == 80
  auto a = make_tensor(make_gmem_ptr((float const*)nullptr) + 0,  L);
  auto b = make_tensor(make_gmem_ptr((float const*)nullptr) + 16, L);
#else
  auto a = make_tensor(make_gmem_ptr((float const*)nullptr) + 0,  L);
  auto b = make_tensor(make_gmem_ptr((float const*)nullptr) + 8,  L);
#endif
  (void)a; (void)b;
}
#endif
