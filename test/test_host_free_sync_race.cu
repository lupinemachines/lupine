// A stream synchronization snapshots all mapped host allocations so it can
// invalidate device-written mirrors. A concurrent cudaFreeHost must not let
// that synchronization use an allocation after its client views are unmapped.
#include <atomic>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>
#include <vector>

static constexpr int kGuardAllocations = 64;
static constexpr int kIterations = 1000;
static constexpr size_t kBytes = 4096;

static int fail(cudaError_t error, const char *operation) {
  if (error == cudaSuccess) {
    return 0;
  }
  printf("RESULT: ERROR %s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

int main() {
  int device_count = 0;
  if (fail(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
      device_count == 0) {
    return 2;
  }

  cudaStream_t stream = nullptr;
  if (fail(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return 2;
  }

  // Keep enough live mappings that a synchronize has meaningful work between
  // taking its snapshot and reaching the newly allocated tail entry.
  std::vector<void *> guards;
  guards.reserve(kGuardAllocations);
  for (int i = 0; i < kGuardAllocations; ++i) {
    void *host = nullptr;
    if (fail(cudaMallocHost(&host, kBytes), "guard cudaMallocHost")) {
      return 2;
    }
    guards.push_back(host);
  }

  std::atomic<bool> stop{false};
  std::atomic<cudaError_t> sync_error{cudaSuccess};
  std::thread synchronizer([&] {
    while (!stop.load(std::memory_order_acquire)) {
      cudaError_t error = cudaStreamSynchronize(stream);
      if (error != cudaSuccess) {
        sync_error.store(error, std::memory_order_release);
        break;
      }
    }
  });

  int result = 0;
  for (int i = 0; i < kIterations; ++i) {
    unsigned char *host = nullptr;
    if (fail(cudaMallocHost(reinterpret_cast<void **>(&host), kBytes),
             "racing cudaMallocHost")) {
      result = 2;
      break;
    }
    host[0] = static_cast<unsigned char>(i);
    if (fail(cudaFreeHost(host), "racing cudaFreeHost")) {
      result = 2;
      break;
    }
  }

  stop.store(true, std::memory_order_release);
  synchronizer.join();
  cudaError_t synchronize_result = sync_error.load(std::memory_order_acquire);
  if (synchronize_result != cudaSuccess) {
    fail(synchronize_result, "concurrent cudaStreamSynchronize");
    result = 2;
  }

  for (void *host : guards) {
    if (fail(cudaFreeHost(host), "guard cudaFreeHost")) {
      result = 2;
    }
  }
  if (fail(cudaStreamDestroy(stream), "cudaStreamDestroy")) {
    result = 2;
  }

  if (result == 0) {
    printf("RESULT: PASS\n");
  }
  return result;
}
