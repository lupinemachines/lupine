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
    }                                                                          \
  } while (0)

int main() {
  constexpr size_t bytes = 8 * 1024 * 1024;
  constexpr int iterations = 16;

  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  CUdeviceptr remote = 0;
  CHECK(cuMemAlloc(&remote, bytes));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  std::vector<unsigned char> source(bytes);
  std::vector<unsigned char> destination(bytes);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::fill(source.begin(), source.end(),
              static_cast<unsigned char>(iteration + 1));
    CHECK(cuMemcpyHtoDAsync(remote, source.data(), bytes, stream));
    CHECK(cuStreamSynchronize(stream));
  }

  CHECK(cuMemcpyDtoH(destination.data(), remote, bytes));
  if (destination != source) {
    std::fprintf(stderr, "final asynchronous HtoD payload did not match\n");
    return 1;
  }

  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemFree(remote));
  CHECK(cuCtxDestroy(context));
  std::printf("PASS: asynchronous HtoD staging cleanup preserves data\n");
  return 0;
}
