// cudaMemcpyDefault copies into managed memory are device-side writes in
// stream order. A kernel queued behind the copy must see the copied bytes and
// keep its own result: the copy must not reach the allocation afterwards.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kRounds = 8;

__global__ void add(unsigned int *value, unsigned int amount) {
  *value += amount;
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

  unsigned int *managed = nullptr;
  unsigned int *other = nullptr;
  unsigned int *device = nullptr;
  unsigned int *pinned = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMallocManaged(&managed, sizeof(*managed)), "cudaMallocManaged");
  check(cudaMallocManaged(&other, sizeof(*other)), "cudaMallocManaged");
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaMallocHost(&pinned, sizeof(*pinned)), "cudaMallocHost");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  unsigned int seven = 7;
  check(cudaMemcpy(device, &seven, sizeof(seven), cudaMemcpyHostToDevice),
        "cudaMemcpy");
  *managed = 0;
  *other = 7;
  *pinned = 7;
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  struct {
    const char *name;
    const unsigned int *source;
  } const sources[] = {
      {"device", device}, {"managed", other}, {"pinned", pinned}};
  int failures = 0;
  for (int round = 0; round < kRounds; ++round) {
    for (const auto &source : sources) {
      check(cudaMemcpyAsync(managed, source.source, sizeof(*managed),
                            cudaMemcpyDefault, stream),
            "cudaMemcpyAsync into managed");
      add<<<1, 1, 0, stream>>>(managed, 100);
      check(cudaGetLastError(), "add launch");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      add<<<1, 1, 0, stream>>>(managed, 1000);
      check(cudaGetLastError(), "add launch");
      unsigned int copied = 0;
      check(cudaMemcpyAsync(&copied, managed, sizeof(copied),
                            cudaMemcpyDefault, stream),
            "cudaMemcpyAsync from managed");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      unsigned int got = *static_cast<volatile unsigned int *>(managed);
      if (got != 1107 || copied != 1107) {
        std::fprintf(stderr,
                     "FAIL: round %d from %s: managed %u, read back %u, want "
                     "1107\n",
                     round, source.name, got, copied);
        ++failures;
      }
    }
  }

  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFreeHost(pinned), "cudaFreeHost");
  check(cudaFree(device), "cudaFree");
  check(cudaFree(other), "cudaFree");
  check(cudaFree(managed), "cudaFree");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d managed copies ran out of order\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: default copies into managed memory are stream-ordered\n");
  return EXIT_SUCCESS;
}
