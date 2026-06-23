#include <cuda_runtime_api.h>

#include <cstdio>

__global__ void increment_kernel(int *value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *value += 1;
  }
}

static int check_cuda(cudaError_t status, const char *call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %d %s\n", call, static_cast<int>(status),
                 cudaGetErrorString(status));
    return 1;
  }
  return 0;
}

static cudaError_t prefetch_async(const void *ptr, size_t count, int device) {
#if CUDART_VERSION >= 12020
  cudaMemLocation location = {};
  location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost
                                            : cudaMemLocationTypeDevice;
  location.id = device == cudaCpuDeviceId ? 0 : device;
  return cudaMemPrefetchAsync(ptr, count, location, 0, nullptr);
#else
  return cudaMemPrefetchAsync(ptr, count, device, nullptr);
#endif
}

int main() {
  int count = 0;
  if (check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount") != 0) {
    return 1;
  }
  if (count < 1) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 2;
  }

  int *value = nullptr;
  if (check_cuda(cudaMallocManaged(&value, sizeof(*value)),
                 "cudaMallocManaged") != 0) {
    return 3;
  }
  *value = 41;

  if (check_cuda(prefetch_async(value, sizeof(*value), 0),
                 "cudaMemPrefetchAsync(device)") != 0) {
    cudaFree(value);
    return 4;
  }
  increment_kernel<<<1, 1>>>(value);
  if (check_cuda(cudaGetLastError(), "increment_kernel launch") != 0 ||
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") != 0) {
    cudaFree(value);
    return 5;
  }

  if (check_cuda(prefetch_async(value, sizeof(*value), cudaCpuDeviceId),
                 "cudaMemPrefetchAsync(cpu)") != 0 ||
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(cpu)") != 0) {
    cudaFree(value);
    return 6;
  }

  if (*value != 42) {
    std::fprintf(stderr, "unexpected value after prefetch: %d\n", *value);
    cudaFree(value);
    return 7;
  }
  std::printf("managed prefetch value=%d\n", *value);

  if (check_cuda(cudaFree(value), "cudaFree") != 0) {
    return 8;
  }
  return 0;
}
