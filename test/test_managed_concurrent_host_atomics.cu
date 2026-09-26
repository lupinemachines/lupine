// systemWideAtomics' managed-memory branch (cuda-samples v11.8/v12.4, which do
// not check concurrentManagedAccess) with a differently named kernel: the host
// updates a managed array right after a launch, with no sync, while the kernel
// updates it too. Both sides' updates must survive.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace {

constexpr unsigned int kBlocks = 64;
constexpr unsigned int kThreads = 256;
constexpr int kKernelValue = 7;

__global__ void accumulate(int *values) {
#if __CUDA_ARCH__ >= 600
  atomicAdd_system(&values[0], 1);
#else
  atomicAdd(&values[0], 1);
#endif
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    values[1] = kKernelValue;
  }
}

int check(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  return 1;
}

} // namespace

int main() {
  int *values = nullptr;
  if (check(cudaMallocManaged(&values, 4 * sizeof(int)), "cudaMallocManaged")) {
    return 1;
  }
  for (int i = 0; i < 4; ++i) {
    values[i] = 0;
  }

  accumulate<<<kBlocks, kThreads>>>(values);
  // Let the kernel finish first, as the sample's long host loop does.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  for (unsigned int i = 0; i < kBlocks * kThreads; ++i) {
    __sync_fetch_and_add(&values[0], 1);
  }
  values[2] = 5;
  if (check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  int expected = static_cast<int>(2 * kBlocks * kThreads);
  bool ok = values[0] == expected && values[1] == kKernelValue && values[2] == 5;
  std::printf("total=%d expected=%d kernel=%d host=%d\n", values[0],
              expected, values[1], values[2]);
  cudaFree(values);
  if (!ok) {
    std::fprintf(stderr, "FAIL: host or device managed updates were lost\n");
    return 1;
  }
  std::printf("PASS: concurrent host and device managed updates both land\n");
  return 0;
}
