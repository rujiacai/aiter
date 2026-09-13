// Does read bandwidth at LOW occupancy depend on waves, or on outstanding loads per wave?
// Sweeps unroll (memory-level parallelism) x block count (occupancy) at fixed 1 GB.
#include <hip/hip_runtime.h>
#include <cstdio>

typedef float f32x4 __attribute__((ext_vector_type(4)));

template <int UNROLL>
__global__ __launch_bounds__(256) void read_kernel(const f32x4* __restrict__ src,
                                                   float* __restrict__ sink,
                                                   size_t n4) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  f32x4 acc = {0.f, 0.f, 0.f, 0.f};
  size_t step = stride * UNROLL;
  // n4 is an exact multiple of step for the configs we launch -> no bounds check,
  // so all UNROLL loads issue back-to-back before the first s_waitcnt.
  for (size_t i = idx; i < n4; i += step) {
    f32x4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = src[i + (size_t)u * stride];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc += v[u];
  }
  if (acc.x == 1e30f) sink[idx] = acc.x;
}

static f32x4* g_d = nullptr;
static float* g_sink = nullptr;

template <int UNROLL>
double run(size_t bytes, int blocks, int iters) {
  size_t n4 = bytes / sizeof(f32x4);
  size_t step = (size_t)blocks * 256 * UNROLL;
  n4 = (n4 / step) * step;  // trim so the loop needs no bounds check
  size_t eff = n4 * sizeof(f32x4);

  dim3 grid(blocks), block(256);
  for (int i = 0; i < 3; ++i)
    hipLaunchKernelGGL((read_kernel<UNROLL>), grid, block, 0, 0, g_d, g_sink, n4);
  hipDeviceSynchronize();

  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int i = 0; i < iters; ++i)
    hipLaunchKernelGGL((read_kernel<UNROLL>), grid, block, 0, 0, g_d, g_sink, n4);
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  float ms = 0; hipEventElapsedTime(&ms, e0, e1);
  hipEventDestroy(e0); hipEventDestroy(e1);
  return (double)eff / ((double)ms / 1e3 / iters) / 1e9;
}

int main() {
  size_t bytes = 1024ull << 20;
  hipMalloc(&g_d, bytes);
  hipMalloc(&g_sink, 1 << 20);
  hipMemset(g_d, 1, bytes);

  int blocks[] = {160, 240, 320, 480, 640, 960};
  printf("1 GB streaming read, GB/s.  rows = blocks (WG of 256 thr); cols = unroll (loads in flight/thread)\n");
  printf("%-8s %-6s %8s %8s %8s %8s %8s %8s\n", "blocks", "WG/CU", "u1", "u2", "u4", "u8", "u16", "u32");
  for (int b : blocks) {
    printf("%-8d %-6.1f", b, b / 80.0);
    printf(" %8.0f", run<1>(bytes, b, 20));
    printf(" %8.0f", run<2>(bytes, b, 20));
    printf(" %8.0f", run<4>(bytes, b, 20));
    printf(" %8.0f", run<8>(bytes, b, 20));
    printf(" %8.0f", run<16>(bytes, b, 20));
    printf(" %8.0f", run<32>(bytes, b, 20));
    printf("\n");
    fflush(stdout);
  }
  return 0;
}
