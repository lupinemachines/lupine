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

namespace {

constexpr size_t kBytes = 8 * 1024 * 1024;
constexpr int kIterations = 16;

int run_uploads(const char *label, CUdeviceptr remote, CUstream stream,
                unsigned char *source) {
  std::vector<unsigned char> destination(kBytes);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    std::fill(source, source + kBytes,
              static_cast<unsigned char>(iteration + 1));
    CHECK(cuMemcpyHtoDAsync(remote, source, kBytes, stream));
    CHECK(cuStreamSynchronize(stream));
  }

  CHECK(cuMemcpyDtoH(destination.data(), remote, kBytes));
  if (!std::equal(destination.begin(), destination.end(), source)) {
    std::fprintf(stderr, "%s final asynchronous HtoD payload did not match\n",
                 label);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
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
  CHECK(cuMemAlloc(&remote, kBytes));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  std::vector<unsigned char> pageable(kBytes);
  if (run_uploads("pageable", remote, stream, pageable.data()) != 0) {
    return 1;
  }

  void *pinned = nullptr;
  CHECK(cuMemHostAlloc(&pinned, kBytes, CU_MEMHOSTALLOC_PORTABLE));
  if (run_uploads("pinned", remote, stream,
                  static_cast<unsigned char *>(pinned)) != 0) {
    return 1;
  }
  CHECK(cuMemFreeHost(pinned));

  CUdeviceptr managed = 0;
  CHECK(cuMemAllocManaged(&managed, kBytes, CU_MEM_ATTACH_GLOBAL));
  if (run_uploads("managed", remote, stream,
                  reinterpret_cast<unsigned char *>(managed)) != 0) {
    return 1;
  }
  CHECK(cuMemFree(managed));

  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemFree(remote));
  CHECK(cuCtxDestroy(context));
  std::printf("PASS: asynchronous pageable, pinned, and managed HtoD preserve "
              "data\n");
  return 0;
}
