#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static const char *error_name(cudaError_t result) {
  const char *name = cudaGetErrorName(result);
  return name == nullptr ? "unknown" : name;
}

static void check(cudaError_t result, const char *expr, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 error_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static bool worker_body(int value) {
  cudaStream_t stream = nullptr;
  int *device = nullptr;
  int host = 0;

  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaMalloc(&device, sizeof(*device)));
  CHECK(cudaMemcpyAsync(device, &value, sizeof(value), cudaMemcpyHostToDevice,
                        stream));
  CHECK(cudaMemcpyAsync(&host, device, sizeof(host), cudaMemcpyDeviceToHost,
                        stream));
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaFree(device));
  CHECK(cudaStreamDestroy(stream));

  if (host != value) {
    std::fprintf(stderr, "unexpected copied value: got %d want %d\n", host,
                 value);
    return false;
  }
  return true;
}

int main() {
  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(nullptr));

  constexpr int kThreadCount = 4;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([i, &failures]() {
      if (!worker_body(100 + i)) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (std::thread &thread : threads) {
    thread.join();
  }

  if (failures.load(std::memory_order_relaxed) != 0) {
    return 1;
  }
  std::printf("PASS: runtime default context is usable from worker threads\n");
  return 0;
}
