// A small async copy into pinned memory queued behind a large one, then the
// small buffer freed before anything synchronizes. Freeing pinned memory waits
// for the copies into it, so the program is valid; the large copy's bytes must
// still land, and the freed buffer must not be touched afterwards.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr size_t kLargeBytes = 64ull << 20;
constexpr int kRounds = 8;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

} // namespace

int main() {
  unsigned char *device = nullptr;
  unsigned char *large = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMalloc(&device, kLargeBytes), "cudaMalloc");
  check(cudaMallocHost(&large, kLargeBytes), "cudaMallocHost large");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreate");
  for (int round = 0; round < kRounds; ++round) {
    unsigned char value = static_cast<unsigned char>(0x30 + round);
    check(cudaMemsetAsync(device, value, kLargeBytes, stream),
          "cudaMemsetAsync");
    unsigned int *small = nullptr;
    check(cudaMallocHost(&small, sizeof(*small)), "cudaMallocHost small");
    check(cudaMemcpyAsync(large, device, kLargeBytes, cudaMemcpyDeviceToHost,
                          stream),
          "large DtoH");
    check(cudaMemcpyAsync(small, device, sizeof(*small), cudaMemcpyDeviceToHost,
                          stream),
          "small DtoH");
    check(cudaFreeHost(small), "cudaFreeHost while copies are queued");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    if (large[0] != value || large[kLargeBytes - 1] != value) {
      std::fprintf(stderr, "FAIL: round %d: large copy read 0x%x want 0x%x\n",
                   round, large[0], value);
      return EXIT_FAILURE;
    }
  }
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFreeHost(large), "cudaFreeHost");
  check(cudaFree(device), "cudaFree");
  std::printf("PASS: pinned buffers freed with copies queued\n");
  return EXIT_SUCCESS;
}
