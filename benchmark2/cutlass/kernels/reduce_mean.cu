// reduce_mean.cu — C[i] = mean_j X[i, j]  (reduce over the trailing axis).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_reduce_mean(const float* __restrict__ X, float* __restrict__ C,
                              long rows, long cols) {
  extern __shared__ __align__(SALIGN * 4) float red[];
  const long r = blockIdx.x;
  const float* x = X + r * cols;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    float acc = 0.f;
    for (long k = threadIdx.x; k < cols; k += blockDim.x) acc += x[k];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) C[r] = red[0] / (float)cols;
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long rows = (long)RM_ROWS, cols = (long)RM_COLS;
  float *X, *C;
  CUT_CHECK(cudaMallocManaged(&X, rows * cols * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, rows * sizeof(float)));
  cut_fill(X, rows * cols, 12345u);
  for (long i = 0; i < rows; ++i) C[i] = 0.f;
  const int threads = 256;
  const size_t smem = (size_t)threads * 4;
  k_reduce_mean<<<(unsigned)rows, threads, smem>>>(X, C, rows, cols);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, rows * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(X));
  CUT_CHECK(cudaFree(C));
  return 0;
}
