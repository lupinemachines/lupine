#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char **argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
  const int warmup_iterations = argc > 2 ? std::atoi(argv[2]) : 100;
  if (iterations <= 0 || warmup_iterations < 0) {
    std::fprintf(stderr, "usage: %s [iterations>0] [warmup>=0]\n", argv[0]);
    return EXIT_FAILURE;
  }

  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  unsigned int flags = 0;
  for (int i = 0; i < warmup_iterations; ++i) {
    CHECK(cuCtxGetFlags(&flags));
  }

  std::vector<double> latency_us;
  latency_us.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    CHECK(cuCtxGetFlags(&flags));
    const auto end = std::chrono::steady_clock::now();
    latency_us.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  std::sort(latency_us.begin(), latency_us.end());
  const size_t median_index = latency_us.size() / 2;
  const size_t p95_index = std::min(
      latency_us.size() - 1, static_cast<size_t>(latency_us.size() * 0.95));
  std::printf("iterations=%d median_us=%.3f p95_us=%.3f flags=%u\n", iterations,
              latency_us[median_index], latency_us[p95_index], flags);

  CHECK(cuCtxDestroy(context));
  return EXIT_SUCCESS;
}
