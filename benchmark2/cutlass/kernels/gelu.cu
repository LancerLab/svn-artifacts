// gelu.cu — y[i] = 0.5 * a[i] * (1 + erf(a[i] / sqrt(2)))  (exact erf form).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__device__ __forceinline__ float gelu_f(float x) {
  return 0.5f * x * (1.f + erff(x * 0.7071067811865476f));
}

__global__ void k_gelu(const float* __restrict__ A, float* __restrict__ C,
                       long n) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  const long stride = (long)TILE * (long)gridDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)blockIdx.x * TILE + (long)it * STEP * stride;
    const long cnt = (n - off < TILE) ? (n - off) : TILE;
    if (cnt <= 0) continue;
    auto gA = make_tensor(make_gmem_ptr(A + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto tA = make_tensor(make_smem_ptr(smem), make_shape(cnt),
                          make_stride(Int<1>{}));
    copy(gA, tA);
    for (long k = threadIdx.x; k < cnt; k += blockDim.x)
      gC(k) = gelu_f(tA(k));
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long n = (long)N_ELEM;
  float *A, *C;
  CUT_CHECK(cudaMallocManaged(&A, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, n * sizeof(float)));
  cut_fill(A, n, 12345u);
  for (long i = 0; i < n; ++i) C[i] = 0.f;
  const int threads = 256;
  const long grid = (n + (long)TILE - 1) / (long)TILE;
  const size_t smem = (size_t)((TILE > SMEM_ELT) ? TILE : SMEM_ELT) * 4;
  k_gelu<<<(unsigned)grid, threads, smem>>>(A, C, n);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(C));
  return 0;
}
