// transpose.cu — y[n*M + m] = x[m*N + n]  (logical 2-D transpose).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_transpose(const float* __restrict__ A, float* __restrict__ C,
                            long M, long N) {
  const long total = M * N;
  const long gsz = (long)gridDim.x * blockDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x +
                  (long)it * STEP * gsz;
         i < total; i += gsz) {
      const long m = i / N, n = i % N;
      C[n * M + m] = A[i];
    }
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long M = (long)TR_M, N = (long)TR_N;
  const long n = M * N;
  float *A, *C;
  CUT_CHECK(cudaMallocManaged(&A, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, n * sizeof(float)));
  cut_fill(A, n, 12345u);
  for (long i = 0; i < n; ++i) C[i] = 0.f;
  const int threads = 256;
  const long grid = (n + threads - 1) / threads;
  k_transpose<<<(unsigned)grid, threads>>>(A, C, M, N);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(C));
  return 0;
}
