// conv2d.cu — valid 2-D convolution, NCHW. out(B,CO,OH,OW).
// LOOP controls how many input-channel tiles are consumed (M4).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

#ifndef CB
#define CB 4
#endif

__global__ void k_conv2d(const float* __restrict__ X,
                         const float* __restrict__ W,
                         float* __restrict__ Y,
                         int B, int C, int H, int Wd, int CO, int KH, int KW) {
  const int OH = H - KH + 1, OW = Wd - KW + 1;
  const long total = (long)B * CO * OH * OW;
  const long xn = (long)B * C * H * Wd;
  const int ncb = (C + CB - 1) / CB;
  const int nl = (LOOP < 0) ? ncb : LOOP;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    const int ow = idx % OW;
    const int oh = (idx / OW) % OH;
    const int co = (idx / ((long)OW * OH)) % CO;
    const int b = idx / ((long)OW * OH * CO);
    float acc = 0.f;
    for (int it = 0; it < nl; ++it) {
      const int c0 = it * STEP * CB;
      for (int c = c0; c < c0 + CB && c < C; ++c)
        for (int kh = 0; kh < KH; ++kh)
          for (int kw = 0; kw < KW; ++kw) {
            const long xi = cut_m2_read(cut_m3_read(
                ((long)b * C + c) * H * Wd + (oh + kh) * Wd + (ow + kw), xn), xn);
            acc += X[xi] * W[((long)co * C + c) * KH * KW + kh * KW + kw];
          }
    }
    Y[idx] = acc;
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const int B = (int)BB, C = (int)CC, H = (int)HH, Wd = (int)WW,
            CO = (int)COO, KH = (int)KHh, KW = (int)KWw;
  const int OH = H - KH + 1, OW = Wd - KW + 1;
  const long xn = (long)B * C * H * Wd, wn = (long)CO * C * KH * KW;
  const long yn = (long)B * CO * OH * OW;
  float *X, *Wt, *Y;
  CUT_CHECK(cudaMallocManaged(&X, xn * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Wt, wn * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&Y, yn * sizeof(float)));
  cut_fill(X, xn, 12345u);
  cut_fill(Wt, wn, 67890u);
  for (long i = 0; i < yn; ++i) Y[i] = 0.f;
  const int threads = 256;
  const int grid = (int)((yn + threads - 1) / threads);
  k_conv2d<<<grid, threads>>>(X, Wt, Y, B, C, H, Wd, CO, KH, KW);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, Y, yn * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(X)); CUT_CHECK(cudaFree(Wt)); CUT_CHECK(cudaFree(Y));
  return 0;
}
