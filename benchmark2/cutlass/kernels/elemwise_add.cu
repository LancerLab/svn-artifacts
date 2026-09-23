// elemwise_add.cu — CUTLASS/CuTe lane operator kernel (generated knob surface).
// y[i] = a[i] + b[i], tiled: gmem -> smem (CuTe copy) -> register -> gmem.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_ele_add(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C, long n) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  float* sA = smem;
  float* sB = smem + TILE;
  const long stride = (long)TILE * (long)gridDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
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
  const long grid = (n + (long)TILE - 1) / (long)TILE;
  const size_t smem = (size_t)((2 * TILE > SMEM_ELT) ? 2 * TILE : SMEM_ELT) * 4;
  k_ele_add<<<(unsigned)grid, threads, smem>>>(A, B, C, n);
  CUT_CHECK(cudaGetLastError());   // launch-config rejection (L class)
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(B));
  CUT_CHECK(cudaFree(C));
  return 0;
}
