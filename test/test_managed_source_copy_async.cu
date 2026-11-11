// An async copy out of managed memory into page-locked host memory returns
// without waiting for the stream, as the driver does: a host function ahead of
// it may be waiting on the caller. The copy still runs in stream order.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr auto kWaitLimit = std::chrono::seconds(5);

std::atomic<int> released{0};
std::atomic<int> timed_out{0};

__global__ void store(unsigned int *destination, unsigned int value) {
  *destination = value;
}

void CUDART_CB wait_for_caller(void *) {
  auto start = std::chrono::steady_clock::now();
  while (released.load() == 0) {
    if (std::chrono::steady_clock::now() - start > kWaitLimit) {
      timed_out.store(1);
      return;
    }
  }
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

int run(const char *name, unsigned int *destination, unsigned int *managed,
        cudaStream_t stream, unsigned int value) {
  released.store(0);
  timed_out.store(0);
  *destination = 0;
  store<<<1, 1, 0, stream>>>(managed, value);
  check(cudaGetLastError(), "store launch");
  check(cudaLaunchHostFunc(stream, wait_for_caller, nullptr),
        "cudaLaunchHostFunc");
  check(cudaMemcpyAsync(destination, managed, sizeof(*destination),
                        cudaMemcpyDefault, stream),
        "cudaMemcpyAsync from managed");
  // Only reached while the host function still waits if the copy returned.
  released.store(1);
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  int failures = 0;
  if (timed_out.load() != 0) {
    std::fprintf(stderr, "FAIL: %s: the copy waited for the stream\n", name);
    ++failures;
  }
  unsigned int got = *static_cast<volatile unsigned int *>(destination);
  if (got != value) {
    std::fprintf(stderr, "FAIL: %s: copied 0x%x want 0x%x\n", name, got, value);
    ++failures;
  }
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

  unsigned int *managed = nullptr;
  unsigned int *pinned = nullptr;
  unsigned int *mapped = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMallocManaged(&managed, sizeof(*managed)), "cudaMallocManaged");
  check(cudaMallocHost(&pinned, sizeof(*pinned)), "cudaMallocHost");
  check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
        "cudaHostAlloc");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  int failures = 0;
  for (unsigned int round = 0; round < 4; ++round) {
    failures +=
        run("pinned destination", pinned, managed, stream, 0x1000 + round);
    failures +=
        run("mapped destination", mapped, managed, stream, 0x2000 + round);
  }

  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFreeHost(mapped), "cudaFreeHost");
  check(cudaFreeHost(pinned), "cudaFreeHost");
  check(cudaFree(managed), "cudaFree");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d managed-source copies\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: copies out of managed memory return without waiting\n");
  return EXIT_SUCCESS;
}
