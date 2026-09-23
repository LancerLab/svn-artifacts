// elemwise_add.cu — CUTLASS/CuTe lane operator kernel (generated knob surface).
// y[i] = a[i] + b[i], tiled: gmem -> smem (CuTe copy) -> register -> gmem.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_ele_add(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C, long n, int rt) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  cut_empty_probe();
  cut_zstride_probe();
  __shared__ float cut_pad_scratch[512];
  cut_pad_probe<PADEXT>(cut_pad_scratch);
  float* sA = smem;
  float* sB = smem + TILE;
  const long stride = (long)TILE * (long)gridDim.x;
  int nl = (LOOP == -1) ? ((rt >= 0) ? rt : 1) : LOOP;
  if (NBOUND) nl = -1 - nl;   // M4.6 negative bound
  if (REVB) nl = nl + 1;      // M1.7 reversed bound
  for (int it = 0; it < nl; ++it) {
    const long off = (long)blockIdx.x * TILE + (long)it * STEP * stride;
    const long cnt = (n - off < TILE) ? (n - off) : TILE;
    if (cnt <= 0) continue;
    auto gA = make_tensor(make_gmem_ptr(A + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto tA = make_tensor(make_smem_ptr(sA), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto tB = make_tensor(make_smem_ptr(sB), make_shape(cnt),
                          make_stride(Int<1>{}));
    copy(gA, tA);
    copy(gB, tB);
    for (long k = threadIdx.x; k < cnt; k += blockDim.x)
      gC(k) = tA(k) + tB(k);
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long n = (long)N_ELEM;
  float *A, *B, *C;
  CUT_CHECK(cudaMallocManaged(&A, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&B, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, n * sizeof(float)));
  cut_fill(A, n, 12345u);
  cut_fill(B, n, 67890u);
  for (long i = 0; i < n; ++i) C[i] = 0.f;

  const int threads = 256;
  const long grid0 = (n + (long)TILE - 1) / (long)TILE;
  const long grid = (PBOUND >= 0) ? (long)PBOUND : grid0;   // M4.2
  const size_t smem = (size_t)((2 * TILE > SMEM_ELT) ? 2 * TILE : SMEM_ELT) * 4;
  const int rt = cut_env_int("CUT_LOOP_RT", -1);
  k_ele_add<<<(unsigned)grid, threads, smem>>>(A, B, C, n, rt);
  CUT_CHECK(cudaGetLastError());   // launch-config rejection (L class)
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(B));
  CUT_CHECK(cudaFree(C));
  return 0;
}
