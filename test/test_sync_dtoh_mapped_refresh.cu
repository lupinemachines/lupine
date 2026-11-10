// A synchronous device-to-host copy waits for the legacy stream, and with it
// for every blocking and per-thread default stream; an async one into pageable
// memory waits for its own stream. Mapped host memory those streams' kernels
// wrote must read back current afterwards, as it does after any other
// synchronizing call. A readback into a registration whose device pointer is
// exposed must keep its own bytes through that refresh, over the whole
// registration, a subrange of it, or pitched rows.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

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

// Checks [0, bytes) of buffer against the bytes expected after each readback.
int compare(const char *name, const unsigned char *buffer,
            const unsigned char *expected, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) {
    if (buffer[i] != expected[i]) {
      std::fprintf(stderr, "FAIL: %s: byte %zu is 0x%x want 0x%x\n", name, i,
                   buffer[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

int registered_readbacks() {
  size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t bytes = page * 4;
  auto *buffer =
      static_cast<unsigned char *>(std::aligned_alloc(page, bytes));
  auto *expected = static_cast<unsigned char *>(std::malloc(bytes));
  unsigned char *source = nullptr;
  check(cudaMalloc(&source, bytes), "cudaMalloc source");
  std::memset(buffer, 0, bytes);
  std::memset(expected, 0, bytes);
  check(cudaHostRegister(buffer, bytes, cudaHostRegisterMapped),
        "cudaHostRegister");
  void *alias = nullptr;
  check(cudaHostGetDevicePointer(&alias, buffer, 0),
        "cudaHostGetDevicePointer");

  int failures = 0;
  // The whole registration.
  check(cudaMemset(source, 0x5a, bytes), "cudaMemset");
  check(cudaMemcpy(buffer, source, bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy whole registration");
  std::memset(expected, 0x5a, bytes);
  failures += compare("registered whole", buffer, expected, bytes);
  // A subrange of it.
  check(cudaMemset(source, 0x6b, bytes), "cudaMemset");
  check(cudaMemcpy(buffer + 100, source, 50, cudaMemcpyDeviceToHost),
        "cudaMemcpy registered subrange");
  std::memset(expected + 100, 0x6b, 50);
  failures += compare("registered subrange", buffer, expected, bytes);
  // Pitched rows, leaving the bytes between them alone.
  check(cudaMemset(source, 0x7c, bytes), "cudaMemset");
  check(cudaMemcpy2D(buffer + page, 256, source, 64, 64, 8,
                     cudaMemcpyDeviceToHost),
        "cudaMemcpy2D registered rows");
  for (size_t row = 0; row < 8; ++row) {
    std::memset(expected + page + row * 256, 0x7c, 64);
  }
  failures += compare("registered pitched", buffer, expected, bytes);
  // A later synchronize refreshes the registration again.
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  failures += compare("registered after synchronize", buffer, expected, bytes);

  check(cudaHostUnregister(buffer), "cudaHostUnregister");
  check(cudaFree(source), "cudaFree source");
  std::free(expected);
  std::free(buffer);
  return failures;
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
  failures += registered_readbacks();
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d stale mapped reads\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: synchronous readbacks refresh mapped memory\n");
  return EXIT_SUCCESS;
}
