// probe_m3.cu — COMPILE-ONLY M3 probes for the CuTe/CUTLASS surface.
//
// Each PROBE id is one half of a (control, mutant) pair. The pair is compiled
// with `nvcc -c` (never linked, never run):
//   control OK  + mutant FAIL  -> ct-check   (CuTe statically detains the defect)
//   control OK  + mutant OK    -> unchecked  (the surface does not detain it)
//   control FAIL               -> generator defect (fix the probe; emit no cell)
//
// TMA/GMMA probes target sm_90 compile-only: the dev host is sm_86, and the
// question is whether the *compiler* catches the defect, independent of launch.
//
// PROBE map:  10/11 M3.1 atom divisibility | 20/21 M3.2 descriptor dim
//             40/41 M3.4 footprint <4GB     | 50/51 M3.5 swizzle
//             80/81 M3.8 GMMA rank          | 120/121 L1 static smem
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
using namespace cute;

// ---- M3.1 contraction extent not divisible by the MMA atom ---------------
#if PROBE == 10 || PROBE == 11
__global__ void k_probe() {
  auto mma = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{});
  auto blk = mma.get_slice(0);
#if PROBE == 10
  auto A = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_shape(Int<32>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
#else
  auto A = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_shape(Int<24>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
#endif
  auto tAgA = blk.partition_A(A); (void)tAgA;
}
#endif

// ---- M3.2 descriptor dimension bound (dim < 2^24) -------------------------
#if PROBE == 20 || PROBE == 21
void host_probe() {
#if PROBE == 20
  auto g = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<8388608>{}, Int<8>{}), GenRowMajor{}));
#else
  auto g = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<16777216>{}, Int<8>{}), GenRowMajor{}));
#endif
  auto t = make_tma_copy(SM90_TMA_LOAD{}, g,
                         make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));
  (void)t;
}
#endif

// ---- M3.4 tensor footprint (< 4GB) ---------------------------------------
#if PROBE == 40 || PROBE == 41
void host_probe() {
#if PROBE == 40
  auto g = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<32768>{}, Int<32768>{}), GenRowMajor{}));
#else
  auto g = make_tensor(make_gmem_ptr((float const*)nullptr),
                       make_layout(make_shape(Int<65536>{}, Int<65536>{}), GenRowMajor{}));
#endif
  auto t = make_tma_copy(SM90_TMA_LOAD{}, g,
                         make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));
  (void)t;
}
#endif

// ---- M3.5 swizzle-incompatible box shape ---------------------------------
#if PROBE == 50 || PROBE == 51
__global__ void k_probe() {
#if PROBE == 50
  using Swz = Swizzle<3, 4, 3>;
#else
  using Swz = Swizzle<7, 4, 3>;
#endif
  auto l = composition(Swz{}, Layout<Shape<_8, _8>>{}); (void)l;
}
#endif

// ---- M3.8 GMMA descriptor rank outside the supported set -----------------
#if PROBE == 80 || PROBE == 81
__global__ void k_probe() {
#if PROBE == 80
  auto s = make_tensor(make_smem_ptr((float*)nullptr),
                       make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{}));
#else
  auto s = make_tensor(make_smem_ptr((float*)nullptr),
                       make_layout(make_shape(Int<8>{}, Int<8>{}, Int<8>{}), GenRowMajor{}));
#endif
  auto d = cute::SM90::GMMA::make_gmma_desc<Major::MN>(s); (void)d;
}
#endif

// ---- L1 / M3.12 static shared tile over the arch capacity ----------------
#if PROBE == 120 || PROBE == 121
#if PROBE == 120
__global__ void k_probe() { __shared__ float s[4096]; s[threadIdx.x] = 1.f; }
#else
__global__ void k_probe() { __shared__ float s[32768]; s[threadIdx.x] = 1.f; }
#endif
#endif

int main() {
#if PROBE == 20 || PROBE == 21 || PROBE == 40 || PROBE == 41
  host_probe();
#endif
  return 0;
}
