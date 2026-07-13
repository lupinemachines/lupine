#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

__global__ void write_sparse(unsigned char *data, size_t offset,
                             unsigned char value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    data[offset] = value;
  }
}

__global__ void write_dense(unsigned char *data, size_t size,
                            unsigned char value) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; index < size; index += stride) {
    data[index] = value;
  }
}

static bool sync_and_report(size_t allocation_bytes, const char *pattern,
                            size_t dirty_payload_bytes) {
  auto started = std::chrono::steady_clock::now();
  cudaError_t result = cudaDeviceSynchronize();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - started)
                     .count();
  if (result != cudaSuccess) {
    std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                 cudaGetErrorString(result));
    return false;
  }
  std::printf("%zu,%s,%zu,%lld\n", allocation_bytes, pattern,
              dirty_payload_bytes, static_cast<long long>(elapsed));
  return true;
}

int main() {
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t sizes[] = {1ULL << 20, 16ULL << 20, 64ULL << 20, 256ULL << 20,
                          1ULL << 30};
  std::puts("allocation_bytes,pattern,dirty_payload_bytes,sync_latency_us");

  for (size_t size : sizes) {
    unsigned char *data = nullptr;
    cudaError_t result = cudaMallocManaged(&data, size);
    if (result != cudaSuccess) {
      std::fprintf(stderr, "cudaMallocManaged(%zu) failed: %s\n", size,
                   cudaGetErrorString(result));
      return 1;
    }

    result = cudaMemset(data, 0, size);
    if (result != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "initialization failed for %zu bytes\n", size);
      return 1;
    }

    write_sparse<<<1, 1>>>(data, size / 2, 0x5a);
    if (!sync_and_report(size, "sparse", page_size)) {
      return 1;
    }

    write_dense<<<256, 256>>>(data, size, 0xa5);
    if (!sync_and_report(size, "dense", size)) {
      return 1;
    }
    cudaFree(data);
  }
  return 0;
}
