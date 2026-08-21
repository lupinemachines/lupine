// Verify that an event recorded in the legacy default stream returns deferred
// DtoH copies from every implicitly ordered blocking stream, but not from a
// non-blocking stream or from work submitted after the event.
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

bool all_equal(const int *values, int expected) {
  return std::all_of(values, values + kElements,
                     [expected](int value) { return value == expected; });
}

bool allocate_pair(int value, int **device, int **host) {
  std::vector<int> source(kElements, value);
  if (!check(cudaMalloc(reinterpret_cast<void **>(device), kBytes),
             "cudaMalloc") ||
      !check(cudaMemcpy(*device, source.data(), kBytes, cudaMemcpyHostToDevice),
             "cudaMemcpy HtoD") ||
      !check(cudaMallocHost(reinterpret_cast<void **>(host), kBytes),
             "cudaMallocHost")) {
    return false;
  }
  std::fill(*host, *host + kElements, kUntouched);
  return true;
}

} // namespace

int main() {
  int *first_device = nullptr;
  int *second_device = nullptr;
  int *later_device = nullptr;
  int *nonblocking_device = nullptr;
  int *first_host = nullptr;
  int *second_host = nullptr;
  int *later_host = nullptr;
  int *nonblocking_host = nullptr;
  cudaStream_t first_stream = nullptr;
  cudaStream_t second_stream = nullptr;
  cudaStream_t nonblocking_stream = nullptr;
  cudaEvent_t event = nullptr;

  if (!allocate_pair(7, &first_device, &first_host) ||
      !allocate_pair(11, &second_device, &second_host) ||
      !allocate_pair(13, &later_device, &later_host) ||
      !allocate_pair(17, &nonblocking_device, &nonblocking_host) ||
      !check(cudaStreamCreate(&first_stream), "create first stream") ||
      !check(cudaStreamCreate(&second_stream), "create second stream") ||
      !check(
          cudaStreamCreateWithFlags(&nonblocking_stream, cudaStreamNonBlocking),
          "create non-blocking stream") ||
      !check(cudaEventCreate(&event), "create event") ||
      !check(cudaMemcpyAsync(first_host, first_device, kBytes,
                             cudaMemcpyDeviceToHost, first_stream),
             "first DtoH") ||
      !check(cudaMemcpyAsync(second_host, second_device, kBytes,
                             cudaMemcpyDeviceToHost, second_stream),
             "second DtoH") ||
      !check(cudaMemcpyAsync(nonblocking_host, nonblocking_device, kBytes,
                             cudaMemcpyDeviceToHost, nonblocking_stream),
             "non-blocking DtoH") ||
      !check(cudaEventRecord(event, nullptr), "record default-stream event") ||
      !check(cudaMemcpyAsync(later_host, later_device, kBytes,
                             cudaMemcpyDeviceToHost, first_stream),
             "later DtoH") ||
      !check(cudaEventSynchronize(event), "synchronize event")) {
    return 1;
  }

  if (!all_equal(first_host, 7) || !all_equal(second_host, 11)) {
    std::fprintf(stderr,
                 "FAIL: default-stream event did not deliver blocking-stream "
                 "DtoH data\n");
    return 1;
  }
  if (!all_equal(later_host, kUntouched) ||
      !all_equal(nonblocking_host, kUntouched)) {
    std::fprintf(stderr,
                 "FAIL: default-stream event delivered unordered DtoH data\n");
    return 1;
  }

  if (!check(cudaStreamSynchronize(first_stream), "synchronize first stream") ||
      !check(cudaStreamSynchronize(nonblocking_stream),
             "synchronize non-blocking stream") ||
      !all_equal(later_host, 13) || !all_equal(nonblocking_host, 17)) {
    std::fprintf(stderr, "FAIL: deferred DtoH data was not delivered later\n");
    return 1;
  }

  cudaEventDestroy(event);
  cudaStreamDestroy(first_stream);
  cudaStreamDestroy(second_stream);
  cudaStreamDestroy(nonblocking_stream);
  cudaFreeHost(first_host);
  cudaFreeHost(second_host);
  cudaFreeHost(later_host);
  cudaFreeHost(nonblocking_host);
  cudaFree(first_device);
  cudaFree(second_device);
  cudaFree(later_device);
  cudaFree(nonblocking_device);

  std::printf("PASS: legacy default-stream event DtoH ordering\n");
  return 0;
}
