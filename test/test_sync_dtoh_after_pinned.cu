// An async copy into pinned memory on a blocking stream, then a synchronous
// copy on the legacy stream. The legacy stream waits for blocking streams, so
// once the synchronous copy returns the pinned bytes are there, with no other
// synchronize.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRounds = 50;
constexpr long long kSpinCycles = 2000000;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void slow_store(unsigned int *destination, unsigned int value) {
  long long start = clock64();
  while (clock64() - start < kSpinCycles) {
  }
  *destination = value;
}

} // namespace

int main() {
  unsigned int *pinned = nullptr;
  unsigned int *device = nullptr;
  unsigned int *other = nullptr;
  cudaStream_t blocking = nullptr;
  check(cudaMallocHost(&pinned, sizeof(*pinned)), "cudaMallocHost");
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaMalloc(&other, sizeof(*other)), "cudaMalloc");
  check(cudaStreamCreate(&blocking), "cudaStreamCreate");
  int failures = 0;
  for (int round = 0; round < kRounds; ++round) {
    unsigned int value = 0x7000 + round;
    slow_store<<<1, 1, 0, blocking>>>(device, value);
    check(cudaMemcpyAsync(pinned, device, sizeof(*pinned),
                          cudaMemcpyDeviceToHost, blocking),
          "pinned DtoH");
    unsigned int unrelated = 0;
    check(cudaMemcpy(&unrelated, other, sizeof(unrelated),
                     cudaMemcpyDeviceToHost),
          "synchronous DtoH");
    unsigned int got = *static_cast<volatile unsigned int *>(pinned);
    if (got != value) {
      std::fprintf(stderr, "FAIL: round %d: pinned 0x%x want 0x%x\n", round,
                   got, value);
      ++failures;
    }
  }
  check(cudaStreamDestroy(blocking), "cudaStreamDestroy");
  check(cudaFree(other), "cudaFree");
  check(cudaFree(device), "cudaFree");
  check(cudaFreeHost(pinned), "cudaFreeHost");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d stale pinned reads\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: a synchronous copy covers earlier pinned copies\n");
  return EXIT_SUCCESS;
}
