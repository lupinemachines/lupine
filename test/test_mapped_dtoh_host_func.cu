// An async copy into mapped host memory whose device pointer is in use, then a
// host function that reads it: on the same stream, and on another stream that
// waits for an event recorded after the copy. The host function follows the
// copy, so it must see the copied bytes even though the host read the buffer
// just before.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRounds = 20;

struct expectation {
  volatile unsigned int *mapped;
  unsigned int want;
  std::atomic<int> *failures;
};

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

void CUDART_CB read_mapped(void *data) {
  auto *expected = static_cast<expectation *>(data);
  unsigned int got = *expected->mapped;
  if (got != expected->want) {
    std::fprintf(stderr, "FAIL: host function read 0x%x want 0x%x\n", got,
                 expected->want);
    expected->failures->fetch_add(1);
  }
}

__global__ void store(unsigned int *destination, unsigned int value) {
  *destination = value;
}

} // namespace

int main() {
  unsigned int *mapped = nullptr;
  unsigned int *mapped_device = nullptr;
  unsigned int *device = nullptr;
  cudaStream_t first = nullptr;
  cudaStream_t second = nullptr;
  cudaEvent_t copied = nullptr;
  std::atomic<int> failures{0};
  check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
        "cudaHostAlloc");
  check(cudaHostGetDevicePointer(&mapped_device, mapped, 0),
        "cudaHostGetDevicePointer");
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), "stream");
  check(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), "stream");
  check(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "event");
  // The device has written through the mapped pointer, so the client's copy is
  // refreshed from then on.
  store<<<1, 1, 0, first>>>(mapped_device, 1);
  check(cudaStreamSynchronize(first), "first synchronize");

  expectation expectations[kRounds * 2];
  for (int round = 0; round < kRounds; ++round) {
    unsigned int value = 0x5000 + round;
    // The host has the buffer readable before the copy.
    if (*static_cast<volatile unsigned int *>(mapped) == value) {
      std::fprintf(stderr, "FAIL: round %d started with its own value\n",
                   round);
      return EXIT_FAILURE;
    }
    store<<<1, 1, 0, first>>>(device, value);
    check(cudaMemcpyAsync(mapped, device, sizeof(*mapped),
                          cudaMemcpyDeviceToHost, first),
          "DtoH into mapped memory");
    check(cudaEventRecord(copied, first), "cudaEventRecord");
    expectations[2 * round] = {mapped, value, &failures};
    check(cudaLaunchHostFunc(first, read_mapped, &expectations[2 * round]),
          "host function on the copy's stream");
    check(cudaStreamWaitEvent(second, copied, 0), "cudaStreamWaitEvent");
    expectations[2 * round + 1] = {mapped, value, &failures};
    check(cudaLaunchHostFunc(second, read_mapped, &expectations[2 * round + 1]),
          "host function behind the event");
    check(cudaStreamSynchronize(first), "synchronize first");
    check(cudaStreamSynchronize(second), "synchronize second");
  }
  check(cudaEventDestroy(copied), "cudaEventDestroy");
  check(cudaStreamDestroy(first), "cudaStreamDestroy");
  check(cudaStreamDestroy(second), "cudaStreamDestroy");
  check(cudaFree(device), "cudaFree");
  check(cudaFreeHost(mapped), "cudaFreeHost");
  if (failures.load() != 0) {
    std::fprintf(stderr, "FAIL: %d stale reads\n", failures.load());
    return EXIT_FAILURE;
  }
  std::printf("PASS: host functions see copies into mapped memory\n");
  return EXIT_SUCCESS;
}
