// A large async copy into pinned memory left running when the program returns
// from main. Exiting must neither hang nor crash while the bytes are on their
// way.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main() {
  constexpr size_t kBytes = 64ull << 20;
  unsigned char *device = nullptr;
  unsigned char *host = nullptr;
  cudaStream_t stream = nullptr;
  if (cudaMalloc(&device, kBytes) != cudaSuccess ||
      cudaMallocHost(&host, kBytes) != cudaSuccess ||
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
          cudaSuccess) {
    std::fprintf(stderr, "FAIL: setup\n");
    return EXIT_FAILURE;
  }
  for (int i = 0; i < 4; ++i) {
    if (cudaMemcpyAsync(host, device, kBytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
      std::fprintf(stderr, "FAIL: cudaMemcpyAsync\n");
      return EXIT_FAILURE;
    }
  }
  std::printf("PASS: returning with copies in flight\n");
  return EXIT_SUCCESS;
}
