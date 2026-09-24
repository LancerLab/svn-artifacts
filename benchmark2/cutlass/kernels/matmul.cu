// matmul.cu — C[M,N] = A[M,K] @ B[K,N], tiled over (BM,BN) with a BK K-loop.
// LOOP controls the number of K-tiles consumed (M4); STEP controls K-tile
// stride. Base (LOOP=-1, STEP=1) consumes all K-tiles.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

#ifndef BM
#define BM 16
#endif
#ifndef BN
#define BN 16
#endif
#ifndef BK
#define BK 16
#endif

__global__ void k_matmul(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K, int rt) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  float* sA = smem;                 // BM*BK
  float* sB = smem + BM * BK;       // BK*BN
  cut_empty_probe();
  cut_zstride_probe();
  __shared__ float cut_pad_scratch[512];
  cut_pad_probe<PADEXT>(cut_pad_scratch);
  const int row = blockIdx.y * BM;
  const int col = blockIdx.x * BN;
  const int nk = (K + BK - 1) / BK;
  int nl = (LOOP == -1) ? ((rt >= 0) ? rt : nk) : LOOP;
  if (NBOUND) nl = -1 - nl;   // M4.6 negative bound
  if (REVB) nl = nl + 1;      // M1.7 reversed bound
  float acc = 0.f;
  for (int it = 0; it < nl; ++it) {
    int k0 = it * STEP * BK;
    if (k0 >= K) break;
    int kend = (k0 + BK < K) ? (k0 + BK) : K;
    for (int i = threadIdx.y; i < BM; i += blockDim.y)
      for (int k = threadIdx.x; k < BK; k += blockDim.x) {
        const int gr = row + i, gk = k0 + k;
        const long ai = cut_m2_read(
            cut_m3_read((long)gr * K + gk, (long)M * K), (long)M * K);
        sA[i * BK + k] = (gr < M && gk < kend) ? A[ai] : 0.f;
      }
    for (int k = threadIdx.y; k < BK; k += blockDim.y)
      for (int j = threadIdx.x; j < BN; j += blockDim.x) {
        const int gk = k0 + k, gc = col + j;
        const long bi = cut_m3_read((long)gk * N + gc, (long)K * N);
        sB[k * BN + j] = (gk < kend && gc < N) ? B[bi] : 0.f;
      }
    __syncthreads();
    for (int i = threadIdx.y; i < BM; i += blockDim.y)
      for (int j = threadIdx.x; j < BN; j += blockDim.x)
        for (int k = 0; k < BK; ++k) acc += sA[i * BK + k] * sB[k * BN + j];
    __syncthreads();
  }
  for (int i = threadIdx.y; i < BM; i += blockDim.y)
    for (int j = threadIdx.x; j < BN; j += blockDim.x) {
      const int gr = row + i, gc = col + j;
      if (gr < M && gc < N) C[gr * N + gc] = acc;
    }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const int M = (int)MM, N = (int)NN, K = (int)KK;
  float *A, *B, *C;
  CUT_CHECK(cudaMallocManaged(&A, (size_t)M * K * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&B, (size_t)K * N * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, (size_t)M * N * sizeof(float)));
  cut_fill(A, (size_t)M * K, 12345u);
  cut_fill(B, (size_t)K * N, 67890u);
  for (size_t i = 0; i < (size_t)M * N; ++i) C[i] = 0.f;
  dim3 threads(16, 16);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  if (PBOUND >= 0) grid.x = (unsigned)PBOUND;   // M4.2 parallelby bound
  const size_t smem = (size_t)(BM * BK + BK * BN) * 4;
  const int rt = cut_env_int("CUT_LOOP_RT", -1);
  k_matmul<<<grid, threads, smem>>>(A, B, C, M, N, K, rt);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, (size_t)M * N * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A)); CUT_CHECK(cudaFree(B)); CUT_CHECK(cudaFree(C));
  return 0;
}
