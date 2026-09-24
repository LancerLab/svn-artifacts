// batch_norm.cu — inference batch norm, NCHW, per-channel affine.
// y = (x-mean[c])/sqrt(var[c]+eps)*gamma[c]+beta[c]. LOOP controls tile trips.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_bn(const float* __restrict__ X, const float* __restrict__ G,
                     const float* __restrict__ Bt, const float* __restrict__ Mu,
                     const float* __restrict__ Va, float* __restrict__ Y,
                     long n, int C, long HW) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  const long stride = (long)TILE * (long)gridDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    const long off = (long)blockIdx.x * TILE + (long)it * STEP * stride;
    for (long i = off + threadIdx.x; i < off + TILE && i < n; i += blockDim.x) {
      const int c = (int)((i / HW) % C);
      const long xi = cut_m3_read(i, n);
      Y[i] = (X[xi] - Mu[c]) * rsqrtf(Va[c] + 1e-5f) * G[c] + Bt[c];
    }
  }
  (void)smem;
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const int N = (int)NN, C = (int)CC, H = (int)HH, Wd = (int)WW;
  const long HW = (long)H * Wd, n = (long)N * C * HW;
  float *X, *G, *Bt, *Mu, *Va, *Y;
  CUT_CHECK(cudaMallocManaged(&X, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&G, C * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Bt, C * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Mu, C * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Va, C * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Y, n * sizeof(float)));
  cut_fill(X, n, 12345u); cut_fill(G, C, 222u); cut_fill(Bt, C, 333u);
  cut_fill(Mu, C, 444u); cut_fill(Va, C, 555u);
  for (long i = 0; i < n; ++i) Y[i] = 0.f;
  const int threads = 256;
  const long grid = (n + TILE - 1) / TILE;
  k_bn<<<(unsigned)grid, threads, (size_t)TILE * 4>>>(X, G, Bt, Mu, Va, Y, n, C, HW);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, Y, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(X)); CUT_CHECK(cudaFree(G)); CUT_CHECK(cudaFree(Bt));
  CUT_CHECK(cudaFree(Mu)); CUT_CHECK(cudaFree(Va)); CUT_CHECK(cudaFree(Y));
  return 0;
}
