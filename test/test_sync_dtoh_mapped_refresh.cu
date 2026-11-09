// A synchronous device-to-host copy waits for the legacy stream, and with it
// for every blocking and per-thread default stream; an async one into pageable
// memory waits for its own stream. Mapped host memory those streams' kernels
// wrote must read back current afterwards, as it does after any other
// synchronizing call.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRounds = 4;
constexpr long long kSpinCycles = 50000000;
const char *const kNames[] = {"linear", "2D", "async pageable"};

__global__ void slow_store(unsigned int *destination, unsigned int value) {
  long long start = clock64();
  while (clock64() - start < kSpinCycles) {
  }
  *destination = value;
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  unsigned int *mapped = nullptr;
  unsigned int *device = nullptr;
  cudaStream_t blocking = nullptr;
  check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
        "cudaHostAlloc");
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaStreamCreate(&blocking), "cudaStreamCreate");
  *mapped = 0;

  struct {
    const char *name;
    cudaStream_t stream;
  } const streams[] = {{"legacy", nullptr},
                       {"blocking", blocking},
                       {"per-thread", cudaStreamPerThread}};
  int failures = 0;
  unsigned int value = 0;
  for (int round = 0; round < kRounds; ++round) {
    for (const auto &stream : streams) {
      for (int kind = 0; kind < 3; ++kind) {
        ++value;
        slow_store<<<1, 1, 0, stream.stream>>>(mapped, value);
        check(cudaGetLastError(), "slow_store launch");
        unsigned int unrelated = 0;
        if (kind == 0) {
          check(cudaMemcpy(&unrelated, device, sizeof(unrelated),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy DtoH");
        } else if (kind == 1) {
          check(cudaMemcpy2D(&unrelated, sizeof(unrelated), device,
                             sizeof(unrelated), sizeof(unrelated), 1,
                             cudaMemcpyDeviceToHost),
                "cudaMemcpy2D DtoH");
        } else {
          check(cudaMemcpyAsync(&unrelated, device, sizeof(unrelated),
                                cudaMemcpyDeviceToHost, stream.stream),
                "cudaMemcpyAsync DtoH to pageable");
        }
        unsigned int got = *static_cast<volatile unsigned int *>(mapped);
        if (got != value) {
          std::fprintf(stderr,
                       "FAIL: round %d %s stream %s readback: mapped 0x%x "
                       "want 0x%x\n",
                       round, stream.name, kNames[kind], got, value);
          ++failures;
        }
      }
    }
  }

  check(cudaStreamDestroy(blocking), "cudaStreamDestroy");
  check(cudaFree(device), "cudaFree");
  check(cudaFreeHost(mapped), "cudaFreeHost");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d stale mapped reads\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: synchronous readbacks refresh mapped memory\n");
  return EXIT_SUCCESS;
}
