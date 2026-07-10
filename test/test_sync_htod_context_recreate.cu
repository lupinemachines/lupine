#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult result = (call);                                                   \
    if (result != CUDA_SUCCESS) {                                               \
      const char *message = nullptr;                                            \
      cuGetErrorString(result, &message);                                       \
      std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", #call,         \
                   __LINE__, message == nullptr ? "unknown" : message,         \
                   static_cast<int>(result));                                   \
      return 1;                                                                \
    }                                                                           \
  } while (0)

int main() {
  constexpr size_t small_bytes = 1024 * 1024 + 17;
  constexpr size_t large_bytes = 24 * 1024 * 1024 + 17;
  const size_t sizes[] = {small_bytes, large_bytes};
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  std::vector<unsigned char> source(large_bytes);
  std::vector<unsigned char> destination(large_bytes);
  for (int iteration = 0; iteration < 3; ++iteration) {
    CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
    CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
    CHECK(cuCtxCreate(&context, 0, device));
#endif
    CUdeviceptr remote = 0;
    CHECK(cuMemAlloc(&remote, large_bytes));
    std::fill(source.begin(), source.end(),
              static_cast<unsigned char>(0x31 + iteration));
    for (size_t bytes : sizes) {
      std::fill(destination.begin(), destination.end(), 0);
      CHECK(cuMemcpyHtoD(remote, source.data(), bytes));
      CHECK(cuMemcpyDtoH(destination.data(), remote, bytes));
      if (!std::equal(source.begin(), source.begin() + bytes,
                      destination.begin())) {
        std::fprintf(stderr, "context %d HtoD mismatch at size %zu\n",
                     iteration, bytes);
        return 1;
      }
    }
    CHECK(cuMemFree(remote));
    if (iteration == 1) {
      CHECK(cuCtxDetach(context));
    } else {
      CHECK(cuCtxDestroy(context));
    }
  }

  std::printf("PASS: synchronous HtoD staging follows context lifetime\n");
  return 0;
}
