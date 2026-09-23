// layer_normalization.cu — y = (x-mean)/sqrt(var+eps)*gamma+beta over COLS.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__device__ void blk_sum(float* s) {
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
    __syncthreads();
  }
}

__global__ void k_layernorm(const float* __restrict__ A,
                            const float* __restrict__ G,
                            const float* __restrict__ B,
                            float* __restrict__ C, long rows, long cols) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  float* red = smem;
  const long r = blockIdx.x;
  const float* a = A + r * cols;
  float* c = C + r * cols;
  const int nchunks = (int)((cols + TILE - 1) / TILE);
  const int nl = (LOOP < 0) ? nchunks : LOOP;
  const float eps = 1e-5f;

  float s = 0.f;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x)
      s += a[k];
  }
  red[threadIdx.x] = s; blk_sum(red); s = red[0];
  const float mean = s / (float)cols;

  float v = 0.f;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x) {
      const float d = a[k] - mean; v += d * d;
    }
  }
  red[threadIdx.x] = v; blk_sum(red); v = red[0];
  const float inv = rsqrtf(v / (float)cols + eps);

  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x)
      c[k] = (a[k] - mean) * inv * G[k] + B[k];
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long rows = (long)ROWS, cols = (long)COLS;
  float *A, *G, *B, *C;
  CUT_CHECK(cudaMallocManaged(&A, rows * cols * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&G, cols * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&B, cols * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, rows * cols * sizeof(float)));
  cut_fill(A, rows * cols, 12345u);
  cut_fill(G, cols, 222u);
  cut_fill(B, cols, 333u);
  for (long i = 0; i < rows * cols; ++i) C[i] = 0.f;
  const int threads = 256;
  k_layernorm<<<(unsigned)rows, threads, (size_t)threads * 4>>>(A, G, B, C, rows, cols);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, rows * cols * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A)); CUT_CHECK(cudaFree(G));
  CUT_CHECK(cudaFree(B)); CUT_CHECK(cudaFree(C));
  return 0;
}
