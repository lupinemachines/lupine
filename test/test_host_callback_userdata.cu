#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

struct CallbackState {
  std::atomic<int> count{0};
};

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

static void CUDA_CB host_callback(void *user_data) {
  auto *state = static_cast<CallbackState *>(user_data);
  state->count.fetch_add(1, std::memory_order_relaxed);
}

int main() {
  CHECK(cuInit(0));

  int device_count = 0;
  CHECK(cuDeviceGetCount(&device_count));
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CallbackState first;
  CallbackState second;
  CHECK(cuLaunchHostFunc(stream, host_callback, &first));
  CHECK(cuLaunchHostFunc(stream, host_callback, &second));
  CHECK(cuStreamSynchronize(stream));

  const int first_count = first.count.load(std::memory_order_relaxed);
  const int second_count = second.count.load(std::memory_order_relaxed);
  if (first_count != 1 || second_count != 1) {
    std::fprintf(stderr,
                 "callbacks with the same function used the wrong userData: "
                 "first=%d second=%d\n",
                 first_count, second_count);
    return EXIT_FAILURE;
  }

  CHECK(cuStreamDestroy(stream));
  CHECK(cuCtxDestroy(context));
  std::printf("PASS: host callbacks preserve per-registration userData\n");
  return EXIT_SUCCESS;
}
