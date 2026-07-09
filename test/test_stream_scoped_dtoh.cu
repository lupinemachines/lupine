#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "UNKNOWN";
}

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 result_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static bool all_equal(const std::vector<unsigned char> &bytes,
                      unsigned char value) {
  for (unsigned char byte : bytes) {
    if (byte != value) {
      return false;
    }
  }
  return true;
}

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

  constexpr size_t kBytes = 1 << 20;
  constexpr unsigned char kReadyValue = 0x31;
  constexpr unsigned char kBlockedValue = 0x7c;
  constexpr unsigned char kSentinel = 0xa5;

  CUstream ready_stream = nullptr;
  CUstream blocked_stream = nullptr;
  CUdeviceptr ready_device = 0;
  CUdeviceptr blocked_device = 0;
  std::vector<unsigned char> ready_host(kBytes, kSentinel);
  std::vector<unsigned char> blocked_host(kBytes, kSentinel);

  CHECK(cuStreamCreate(&ready_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&blocked_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuMemAlloc(&ready_device, kBytes));
  CHECK(cuMemAlloc(&blocked_device, kBytes));

  CHECK(cuMemsetD8Async(ready_device, kReadyValue, kBytes, ready_stream));
  CHECK(cuMemcpyDtoHAsync(ready_host.data(), ready_device, kBytes,
                          ready_stream));

  CHECK(cuMemsetD8Async(blocked_device, kBlockedValue, kBytes,
                        blocked_stream));
  CHECK(cuMemcpyDtoHAsync(blocked_host.data(), blocked_device, kBytes,
                          blocked_stream));

  CHECK(cuStreamSynchronize(ready_stream));
  if (!all_equal(ready_host, kReadyValue)) {
    std::fprintf(stderr, "ready stream copy did not complete correctly\n");
    return EXIT_FAILURE;
  }
  if (!all_equal(blocked_host, kSentinel)) {
    std::fprintf(stderr,
                 "blocked stream host buffer changed during unrelated stream "
                 "synchronize\n");
    return EXIT_FAILURE;
  }

  CHECK(cuStreamSynchronize(blocked_stream));
  if (!all_equal(blocked_host, kBlockedValue)) {
    std::fprintf(stderr, "blocked stream copy did not complete after sync\n");
    return EXIT_FAILURE;
  }

  CHECK(cuMemFree(ready_device));
  CHECK(cuMemFree(blocked_device));
  CHECK(cuStreamDestroy(ready_stream));
  CHECK(cuStreamDestroy(blocked_stream));
  CHECK(cuCtxDestroy(context));

  std::printf("PASS: deferred DtoH copies flush only synchronized streams\n");
  return EXIT_SUCCESS;
}
