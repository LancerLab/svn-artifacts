// relu.cu — y[i] = max(a[i], 0), tiled gmem->smem->register->gmem.
#include <cute/tensor.hpp>
#include "common.cuh"
using namespace cute;

__global__ void k_relu(const float* __restrict__ A, float* __restrict__ C,
                       long n) {
  extern __shared__ __align__(SALIGN * 4) float smem[];
  CutM1RankProbe<M1DEF>::run();
  const long stride = (long)TILE * (long)gridDim.x;
  const int nl = (LOOP < 0) ? 1 : LOOP;
  for (int it = 0; it < nl; ++it) {
    long off = (long)blockIdx.x * TILE + (long)it * STEP * stride;
    if (M1DEF == 14)   // swapped tile coordinate: read the next block's tile
      off = ((long)blockIdx.x + 1) * TILE + (long)it * STEP * stride;
    const long cnt = (n - off < TILE) ? (n - off) : TILE;
    if (cnt <= 0) continue;
    auto gA = make_tensor(make_gmem_ptr(A + off), make_shape(cnt),
                          make_stride((M1DEF == 4) ? 2 : 1));   // M1.4 bad stride
    auto gC = make_tensor(make_gmem_ptr(C + off), make_shape(cnt),
                          make_stride(Int<1>{}));
    auto tA = make_tensor(make_smem_ptr(smem), make_shape(cnt),
                          make_stride(Int<1>{}));
    copy(gA, tA);
    for (long k = threadIdx.x; (M1DEF == 2) ? (k <= cnt) : (k < cnt);
         k += blockDim.x) {
      long r = k;
      if (M1DEF == 9)  r = k + 1;                  // M1.9 displaced base
      if (M1DEF == 12) r = (k + 1) % cnt;          // M1.12 wrong loop variable
      // M1.19: a narrow (16-bit) index carrier. The global index is truncated
      // to 16 bits; the low bits alias within the same 64K window that holds
      // the tile, so the element addressed is wrong but still legal.
      if (M1DEF == 19) r = (k + off) & 0xFFFF;
      if (M1DEF == 19 && r >= cnt) continue;   // skip aliases past the tile
      r = cut_m2_read(r, cnt);                 // M2 shape-compat defect
      float v = tA(r);
      if (M1DEF == 11)                             // M1.11 read-after-write overlap
        v = tA(r) + ((r + 1 < cnt) ? tA(r + 1) : 0.f);
      gC(k) = v > 0.f ? v : 0.f;
    }
  }
}

int main(int argc, char** argv) {
  const char* out = (argc > 1) ? argv[1] : "out.bin";
  const long n = (long)N_ELEM;
  float *A, *C;
  CUT_CHECK(cudaMallocManaged(&A, n * sizeof(float)));
  CUT_CHECK(cudaMallocManaged(&C, n * sizeof(float)));
  cut_fill(A, n, 12345u);
  for (long i = 0; i < n; ++i) C[i] = 0.f;
  const int threads = 256;
  const long grid = (n + (long)TILE - 1) / (long)TILE;
  const size_t smem = (size_t)((TILE > SMEM_ELT) ? TILE : SMEM_ELT) * 4;
  k_relu<<<(unsigned)grid, threads, smem>>>(A, C, n);
  CUT_CHECK(cudaGetLastError());
  CUT_CHECK(cudaDeviceSynchronize());
  if (cut_dump(out, C, n * sizeof(float))) return 3;
  CUT_CHECK(cudaFree(A));
  CUT_CHECK(cudaFree(C));
  return 0;
}
