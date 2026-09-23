// embedding.cu — out[i, :] = table[idx[i], :], idx carried as float in [0, V).
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_embedding(const float* __restrict__ T, const float* __restrict__ I,
                            float* __restrict__ C, long V, long D, long nidx) {
  const long total = nidx * D;
  const long gsz = (long)gridDim.x * blockDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x +
                  (long)it * STEP * gsz;
         j < total; j += gsz) {
      const long i = j / D, d = j % D;
      long v = (long)(I[i] * (float)V);
      if (v >= V) v = V - 1;
      C[j] = T[v * D + d];
    }
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long V = (long)EM_V, D = (long)EM_D, nidx = (long)EM_N;
  const long total = nidx * D;
  float *T, *I, *C;
  CUT_CHECK(cudaMallocManaged(&T, V * D * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&I, nidx * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, total * sizeof(float)));
  cut_fill(T, V * D, 12345u);
  cut_fill(I, nidx, 67890u);
  for (long i = 0; i < total; ++i) C[i] = 0.f;
  const int threads = 256;
  const long grid = (total + threads - 1) / threads;
  k_embedding<<<(unsigned)grid, threads>>>(T, I, C, V, D, nidx);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, total * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(T));
  CUT_CHECK(cudaFree(I));
  CUT_CHECK(cudaFree(C));
  return 0;
}
