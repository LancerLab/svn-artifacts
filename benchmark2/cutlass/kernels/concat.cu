// concat.cu — C = [A ; B] concatenated on the flat leading axis.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_concat(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, long nA, long nB) {
  const long total = nA + nB;
  const long gsz = (long)gridDim.x * blockDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x +
                  (long)it * STEP * gsz;
         i < total; i += gsz)
      C[i] = (i < nA) ? A[i] : B[i - nA];
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long nA = (long)CC_NA, nB = (long)CC_NB;
  const long total = nA + nB;
  float *A, *B, *C;
  CUT_CHECK(cudaMallocManaged(&A, nA * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&B, nB * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, total * sizeof(float)));
  cut_fill(A, nA, 12345u);
  cut_fill(B, nB, 67890u);
  for (long i = 0; i < total; ++i) C[i] = 0.f;
  const int threads = 256;
  const long grid = (total + threads - 1) / threads;
  k_concat<<<(unsigned)grid, threads>>>(A, B, C, nA, nB);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, total * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(B));
  CUT_CHECK(cudaFree(C));
  return 0;
}
