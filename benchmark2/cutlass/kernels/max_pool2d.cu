// max_pool2d.cu — 2-D max pooling, CHW, kw=kh=k, stride k.
// LOOP controls tile trips (M4). out(C, H/k, W/k).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_maxpool(const float* __restrict__ X, float* __restrict__ Y,
                          int C, int H, int Wd, int K) {
  const int OH = H / K, OW = Wd / K;
  const long total = (long)C * OH * OW;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    const int ow = idx % OW;
    const int oh = (idx / OW) % OH;
    const int c = idx / ((long)OW * OH);
    float m = -INFINITY;
    for (int it = 0; it < nl; ++it) {
      for (int kh = 0; kh < K; ++kh)
        for (int kw = 0; kw < K; ++kw)
          m = fmaxf(m, X[((long)c * H + oh * K + kh) * Wd + ow * K + kw]);
    }
    Y[idx] = m;
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const int C = (int)Ch, H = (int)HH, Wd = (int)WW, K = (int)KK;
  const long xn = (long)C * H * Wd, yn = (long)C * (H / K) * (Wd / K);
  float *X, *Y;
  CUT_CHECK(cudaMallocManaged(&X, xn * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Y, yn * sizeof(float)));
  cut_fill(X, xn, 12345u);
  for (long i = 0; i < yn; ++i) Y[i] = 0.f;
  const int threads = 256;
  const int grid = (int)((yn + threads - 1) / threads);
  k_maxpool<<<grid, threads>>>(X, Y, C, H, Wd, K);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, Y, yn * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(X)); CUT_CHECK(cudaFree(Y));
  return 0;
}
