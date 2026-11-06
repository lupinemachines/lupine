// An uncaptured async DtoH into registered memory is staged in server pinned
// memory until something collects it. A host function queued behind the copy
// collects it from inside a CUDA host callback, where the staging cannot be
// freed with a CUDA call. Repeat enough times that leaked staging would
// exhaust the server's pinned memory.
#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 name != nullptr ? name : "UNKNOWN", static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static std::atomic<int> callbacks{0};

static void CUDA_CB host_fn(void *) {
  callbacks.fetch_add(1, std::memory_order_relaxed);
}

int main(int argc, char **argv) {
  const size_t bytes = 64u << 20;
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 32;
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));

  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr device_buffer = 0;
  CHECK(cuMemAlloc(&device_buffer, bytes));
  auto *host = static_cast<unsigned int *>(std::malloc(bytes));
  CHECK(cuMemHostRegister(host, bytes, 0));

  for (int i = 0; i < iterations; ++i) {
    std::memset(host, 0, bytes);
    CHECK(cuMemsetD32Async(device_buffer, 0x1000u + i, bytes / 4, stream));
    CHECK(cuMemcpyDtoHAsync(host, device_buffer, bytes, stream));
    CHECK(cuLaunchHostFunc(stream, host_fn, nullptr));
    CHECK(cuStreamSynchronize(stream));
    for (size_t j = 0; j < bytes / 4; j += 4093) {
      if (host[j] != 0x1000u + i) {
        std::printf("FAIL: iteration %d word %zu is 0x%x\n", i, j, host[j]);
        return EXIT_FAILURE;
      }
    }
  }

  CHECK(cuMemHostUnregister(host));
  std::free(host);
  CHECK(cuMemFree(device_buffer));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuDevicePrimaryCtxRelease(device));
  if (callbacks.load() != iterations) {
    std::printf("FAIL: %d of %d host functions ran\n", callbacks.load(),
                iterations);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
