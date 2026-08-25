// Verify that an async DtoH copy into pageable host memory has landed by the
// time the call returns. The driver is free to make such a copy synchronous
// and always does, staging it through pinned memory; only a page-locked
// destination is truly deferred. Libraries such as cuSPARSE read their scalar
// results this way and never synchronize, and the destination is often a stack
// variable that is gone by the next synchronize.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kElements = 4096;
constexpr size_t kBytes = kElements * sizeof(int);
constexpr int kUntouched = -1;

bool check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  return false;
}

__global__ void fill(int *out, int value, size_t elements) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    out[index] = value;
  }
}

// Reads back into a stack buffer the way cuSPARSE reads its result scalars: a
// deferred copy would either never arrive or land on a dead frame.
bool read_back_on_stack(const int *device, cudaStream_t stream, int value,
                        const char *label) {
  int host[kElements];
  std::fill(host, host + kElements, kUntouched);
  if (!check(
          cudaMemcpyAsync(host, device, kBytes, cudaMemcpyDeviceToHost, stream),
          label)) {
    return false;
  }
  if (!std::all_of(host, host + kElements,
                   [value](int seen) { return seen == value; })) {
    std::fprintf(stderr, "FAIL: %s did not deliver %d before returning\n",
                 label, value);
    return false;
  }
  return true;
}

} // namespace

int main() {
  int *device = nullptr;
  cudaStream_t stream = nullptr;
  if (!check(cudaMalloc(reinterpret_cast<void **>(&device), kBytes),
             "cudaMalloc") ||
      !check(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return 1;
  }

  const unsigned int blocks = (kElements + 255) / 256;
  fill<<<blocks, 256, 0, stream>>>(device, 7, kElements);
  if (!check(cudaGetLastError(), "launch fill on stream") ||
      !read_back_on_stack(device, stream, 7, "stream DtoH")) {
    return 1;
  }

  // The same guarantee holds on the legacy default stream, which is what a
  // library that never creates a stream of its own uses.
  fill<<<blocks, 256>>>(device, 11, kElements);
  if (!check(cudaGetLastError(), "launch fill on default stream") ||
      !read_back_on_stack(device, nullptr, 11, "default-stream DtoH")) {
    return 1;
  }

  // A heap destination is pageable too, and its bytes must be visible without
  // a synchronize just the same.
  std::vector<int> heap(kElements, kUntouched);
  fill<<<blocks, 256, 0, stream>>>(device, 13, kElements);
  if (!check(cudaGetLastError(), "launch fill for heap readback") ||
      !check(cudaMemcpyAsync(heap.data(), device, kBytes,
                             cudaMemcpyDeviceToHost, stream),
             "heap DtoH")) {
    return 1;
  }
  if (!std::all_of(heap.begin(), heap.end(),
                   [](int seen) { return seen == 13; })) {
    std::fprintf(stderr, "FAIL: heap DtoH did not deliver 13 before "
                         "returning\n");
    return 1;
  }

  // A page-locked range is tracked rounded out to whole pages, so a plain
  // allocation that merely shares a page with one is still pageable and must
  // not inherit its deferred copy.
  std::vector<int> registered(kElements, kUntouched);
  if (!check(
          cudaHostRegister(registered.data(), kBytes, cudaHostRegisterDefault),
          "cudaHostRegister")) {
    return 1;
  }
  std::vector<int> neighbour(kElements, kUntouched);
  fill<<<blocks, 256, 0, stream>>>(device, 17, kElements);
  if (!check(cudaGetLastError(), "launch fill for neighbour readback") ||
      !check(cudaMemcpyAsync(neighbour.data(), device, kBytes,
                             cudaMemcpyDeviceToHost, stream),
             "page-neighbour DtoH")) {
    return 1;
  }
  if (!std::all_of(neighbour.begin(), neighbour.end(),
                   [](int seen) { return seen == 17; })) {
    std::fprintf(stderr, "FAIL: an allocation sharing a page with a registered "
                         "one was treated as page-locked\n");
    return 1;
  }
  cudaHostUnregister(registered.data());

  cudaStreamDestroy(stream);
  cudaFree(device);
  std::printf("PASS: pageable async DtoH completes before returning\n");
  return 0;
}
