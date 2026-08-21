// Verify that event completion returns only deferred device-to-host copies
// ordered before that event. Copies on other streams, or after the event on
// the same stream, must remain untouched until their own synchronization.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kElements = 1 << 16;
constexpr int kUntouched = -1;
constexpr unsigned long long kDelayCycles = 1000000000ULL;

__global__ void delay_cycles(unsigned long long cycles) {
  unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

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

bool allocate_device_pattern(int **device, int value) {
  std::vector<int> source(kElements, value);
  return check(cudaMalloc(reinterpret_cast<void **>(device),
                          kElements * sizeof(int)),
               "cudaMalloc") &&
         check(cudaMemcpy(*device, source.data(), kElements * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy HtoD");
}

bool allocate_host(int **host) {
  if (!check(cudaHostAlloc(reinterpret_cast<void **>(host),
                           kElements * sizeof(int), cudaHostAllocDefault),
             "cudaHostAlloc")) {
    return false;
  }
  std::fill(*host, *host + kElements, kUntouched);
  return true;
}

bool test_other_stream_is_not_drained(unsigned long long delay) {
  int *ready_device = nullptr;
  int *delayed_device = nullptr;
  int *ready_host = nullptr;
  int *delayed_host = nullptr;
  cudaStream_t ready_stream = nullptr;
  cudaStream_t delayed_stream = nullptr;
  cudaEvent_t ready_event = nullptr;

  if (!allocate_device_pattern(&ready_device, 7) ||
      !allocate_device_pattern(&delayed_device, 9) ||
      !allocate_host(&ready_host) || !allocate_host(&delayed_host) ||
      !check(cudaStreamCreate(&ready_stream), "create ready stream") ||
      !check(cudaStreamCreate(&delayed_stream), "create delayed stream") ||
      !check(cudaEventCreate(&ready_event), "create ready event") ||
      !check(cudaMemcpyAsync(ready_host, ready_device, kElements * sizeof(int),
                             cudaMemcpyDeviceToHost, ready_stream),
             "ready DtoH") ||
      !check(cudaEventRecord(ready_event, ready_stream),
             "record ready event")) {
    return false;
  }

  delay_cycles<<<1, 1, 0, delayed_stream>>>(delay);
  if (!check(cudaGetLastError(), "launch delayed kernel") ||
      !check(cudaMemcpyAsync(delayed_host, delayed_device,
                             kElements * sizeof(int), cudaMemcpyDeviceToHost,
                             delayed_stream),
             "delayed DtoH") ||
      !check(cudaEventSynchronize(ready_event), "synchronize ready event")) {
    return false;
  }

  bool passed = all_equal(ready_host, 7) && all_equal(delayed_host, kUntouched);
  if (!passed) {
    std::fprintf(stderr,
                 "FAIL: event synchronization drained another stream\n");
  }
  if (!check(cudaStreamSynchronize(delayed_stream),
             "synchronize delayed stream") ||
      !all_equal(delayed_host, 9)) {
    std::fprintf(stderr,
                 "FAIL: delayed stream did not deliver its DtoH data\n");
    passed = false;
  }

  cudaEventDestroy(ready_event);
  cudaStreamDestroy(ready_stream);
  cudaStreamDestroy(delayed_stream);
  cudaFreeHost(ready_host);
  cudaFreeHost(delayed_host);
  cudaFree(ready_device);
  cudaFree(delayed_device);
  return passed;
}

bool test_later_same_stream_copy_is_not_drained(unsigned long long delay) {
  int *before_device = nullptr;
  int *after_device = nullptr;
  int *before_host = nullptr;
  int *after_host = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;

  if (!allocate_device_pattern(&before_device, 11) ||
      !allocate_device_pattern(&after_device, 13) ||
      !allocate_host(&before_host) || !allocate_host(&after_host) ||
      !check(cudaStreamCreate(&stream), "create same stream") ||
      !check(cudaEventCreate(&event), "create same-stream event") ||
      !check(cudaMemcpyAsync(before_host, before_device,
                             kElements * sizeof(int), cudaMemcpyDeviceToHost,
                             stream),
             "before-event DtoH") ||
      !check(cudaEventRecord(event, stream), "record same-stream event")) {
    return false;
  }

  delay_cycles<<<1, 1, 0, stream>>>(delay);
  if (!check(cudaGetLastError(), "launch same-stream delayed kernel") ||
      !check(cudaMemcpyAsync(after_host, after_device, kElements * sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "after-event DtoH") ||
      !check(cudaEventSynchronize(event), "synchronize same-stream event")) {
    return false;
  }

  bool passed = all_equal(before_host, 11) && all_equal(after_host, kUntouched);
  if (!passed) {
    std::fprintf(stderr,
                 "FAIL: event synchronization drained later stream work\n");
  }
  if (!check(cudaStreamSynchronize(stream), "synchronize same stream") ||
      !all_equal(after_host, 13)) {
    std::fprintf(stderr, "FAIL: later DtoH data was not delivered\n");
    passed = false;
  }

  cudaEventDestroy(event);
  cudaStreamDestroy(stream);
  cudaFreeHost(before_host);
  cudaFreeHost(after_host);
  cudaFree(before_device);
  cudaFree(after_device);
  return passed;
}

} // namespace

int main() {
  if (!test_other_stream_is_not_drained(kDelayCycles) ||
      !test_later_same_stream_copy_is_not_drained(kDelayCycles)) {
    return 1;
  }
  std::printf("PASS: event-scoped deferred DtoH copies\n");
  return 0;
}
