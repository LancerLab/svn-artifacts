// softmax.cu — row-wise softmax over the trailing axis (COLS).
// One block per row; block-reduction in shared memory; chunked row traversal
// driven by the LOOP knob (M4). y = exp(x-max)/sum(exp(x-max)).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__device__ void blk_reduce_max(float* s) {
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + off]);
    __syncthreads();
  }
}
__device__ void blk_reduce_sum(float* s) {
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
    __syncthreads();
  }
}

__global__ void k_softmax(const float* __restrict__ A, float* __restrict__ C,
                          long rows, long cols, int rt) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  cut_empty_probe();
  cut_zstride_probe();
  __shared__ float cut_pad_scratch[512];
  cut_pad_probe<PADEXT>(cut_pad_scratch);
  float* red = smem;                                  // blockDim.x floats
  const long r = blockIdx.x;
  const float* a = A + r * cols;
  float* c = C + r * cols;
  const int nchunks = (int)((cols + TILE - 1) / TILE);
  int nl = (LOOP == -1) ? ((rt >= 0) ? rt : nchunks) : LOOP;
  if (NBOUND) nl = -1 - nl;   // M4.6 negative bound
  if (REVB) nl = nl + 1;      // M1.7 reversed bound

  float m = -INFINITY;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * STEP * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x)
      m = fmaxf(m, a[k]);
  }
  red[threadIdx.x] = m;
  blk_reduce_max(red);
  m = red[0];

  float s = 0.f;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * STEP * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x)
      s += expf(a[k] - m);
  }
  red[threadIdx.x] = s;
  blk_reduce_sum(red);
  s = red[0];

  for (int it = 0; it < nl; ++it) {
    const long off = (long)it * STEP * TILE;
    for (long k = off + threadIdx.x; k < off + TILE && k < cols; k += blockDim.x)
      c[k] = expf(a[k] - m) / s;
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long rows = (long)ROWS, cols = (long)COLS;
  float *A, *C;
  CUT_CHECK(cudaMallocManaged(&A, rows * cols * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, rows * cols * sizeof(float)));
  cut_fill(A, rows * cols, 12345u);
  for (long i = 0; i < rows * cols; ++i) C[i] = 0.f;
  const int threads = 256;
  const size_t smem = (size_t)threads * 4;
  const int rt = cut_env_int("CUT_LOOP_RT", -1);
  const long nrow = (PBOUND >= 0) ? (long)PBOUND : rows;   // M4.2
  k_softmax<<<(unsigned)nrow, threads, smem>>>(A, C, rows, cols, rt);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, rows * cols * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(C));
  return 0;
}
