#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <unistd.h>

__global__ void write_alias(unsigned char *alias, size_t offset,
                            unsigned char value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    alias[offset] = value;
  }
}

__global__ void add_alias(int *alias, int value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(alias, value);
  }
}

static bool check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(result));
  return false;
}

int main() {
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t allocation_size = 64 * 1024 * 1024;
  unsigned char *data = nullptr;
  if (!check(cudaMallocManaged(&data, allocation_size), "cudaMallocManaged")) {
    return 1;
  }

  std::memset(data, 0, allocation_size);
  if (!check(cudaDeviceSynchronize(), "initial cudaDeviceSynchronize")) {
    return 1;
  }

  unsigned char *offset_alias = data + 7 * page_size;
  write_alias<<<1, 1>>>(offset_alias, 13, 0x41);
  if (!check(cudaDeviceSynchronize(), "offset alias synchronize") ||
      offset_alias[13] != 0x41 || data[6 * page_size] != 0 ||
      data[8 * page_size] != 0) {
    std::fprintf(stderr, "offset alias result mismatch\n");
    return 1;
  }

  int *atomic_alias = reinterpret_cast<int *>(data + 11 * page_size);
  add_alias<<<1, 1>>>(atomic_alias, 9);
  if (!check(cudaDeviceSynchronize(), "atomic synchronize") ||
      *atomic_alias != 9) {
    std::fprintf(stderr, "atomic result mismatch\n");
    return 1;
  }

  cudaStream_t first = nullptr;
  cudaStream_t second = nullptr;
  if (!check(cudaStreamCreate(&first), "first cudaStreamCreate") ||
      !check(cudaStreamCreate(&second), "second cudaStreamCreate")) {
    return 1;
  }
  write_alias<<<1, 1, 0, first>>>(data + 13 * page_size, 3, 0x51);
  write_alias<<<1, 1, 0, second>>>(data + 17 * page_size, 5, 0x61);
  if (!check(cudaStreamSynchronize(first), "first stream synchronize") ||
      !check(cudaStreamSynchronize(second), "second stream synchronize") ||
      data[13 * page_size + 3] != 0x51 || data[17 * page_size + 5] != 0x61) {
    std::fprintf(stderr, "multi-stream result mismatch\n");
    return 1;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (!check(cudaStreamBeginCapture(first, cudaStreamCaptureModeGlobal),
             "cudaStreamBeginCapture")) {
    return 1;
  }
  write_alias<<<1, 1, 0, first>>>(data + 21 * page_size, 7, 0x71);
  if (!check(cudaStreamEndCapture(first, &graph), "cudaStreamEndCapture") ||
      !check(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
             "cudaGraphInstantiate") ||
      !check(cudaGraphLaunch(graph_exec, first), "cudaGraphLaunch") ||
      !check(cudaStreamSynchronize(first), "graph stream synchronize") ||
      data[21 * page_size + 7] != 0x71) {
    std::fprintf(stderr, "graph result mismatch\n");
    return 1;
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(second);
  cudaStreamDestroy(first);
  cudaFree(data);

  const size_t mapped_size = 8 * 1024 * 1024;
  unsigned char *mapped_host = nullptr;
  unsigned char *mapped_device = nullptr;
  if (!check(cudaHostAlloc(&mapped_host, mapped_size, cudaHostAllocMapped),
             "cudaHostAlloc") ||
      !check(cudaHostGetDevicePointer(&mapped_device, mapped_host, 0),
             "cudaHostGetDevicePointer")) {
    return 1;
  }
  std::memset(mapped_host, 0, mapped_size);
  if (!check(cudaDeviceSynchronize(), "mapped initialization synchronize")) {
    return 1;
  }
  write_alias<<<1, 1>>>(mapped_device + 5 * page_size, 19, 0x81);
  if (!check(cudaDeviceSynchronize(), "mapped sparse synchronize") ||
      mapped_host[5 * page_size + 19] != 0x81 ||
      mapped_host[4 * page_size] != 0 || mapped_host[6 * page_size] != 0) {
    std::fprintf(stderr, "mapped sparse result mismatch\n");
    return 1;
  }
  cudaFreeHost(mapped_host);

  std::puts("managed and mapped sparse device sync passed");
  return 0;
}
