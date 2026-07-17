#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

__global__ static void delay_kernel(unsigned long long cycles) {
  unsigned long long start = clock64();
  while (clock64() - start < cycles) {
    asm volatile("");
  }
}

struct latency_sample {
  double idle_ms = 0.0;
  double long_ms = 0.0;
};

static latency_sample measure_lanes(int clock_khz, int delay_ms) {
  cudaStream_t long_stream = nullptr;
  cudaStream_t idle_stream = nullptr;
  CHECK(cudaStreamCreateWithFlags(&long_stream, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&idle_stream, cudaStreamNonBlocking));
  CHECK(cudaStreamSynchronize(long_stream));
  CHECK(cudaStreamSynchronize(idle_stream));

  unsigned long long cycles =
      static_cast<unsigned long long>(clock_khz) * delay_ms;
  delay_kernel<<<1, 1, 0, long_stream>>>(cycles);
  CHECK(cudaGetLastError());

  std::atomic<bool> long_sync_started{false};
  cudaError_t long_result = cudaSuccess;
  latency_sample sample;
  std::thread long_lane([&]() {
    long_result = cudaSetDevice(0);
    if (long_result != cudaSuccess) {
      long_sync_started.store(true, std::memory_order_release);
      return;
    }
    long_sync_started.store(true, std::memory_order_release);
    auto start = std::chrono::steady_clock::now();
    long_result = cudaStreamSynchronize(long_stream);
    sample.long_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  });

  while (!long_sync_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  auto idle_start = std::chrono::steady_clock::now();
  CHECK(cudaStreamSynchronize(idle_stream));
  sample.idle_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - idle_start)
                       .count();

  long_lane.join();
  CHECK(long_result);
  CHECK(cudaStreamDestroy(idle_stream));
  CHECK(cudaStreamDestroy(long_stream));
  return sample;
}

int main() {
  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(nullptr));

  int clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));

  latency_sample short_kernel = measure_lanes(clock_khz, 300);
  latency_sample long_kernel = measure_lanes(clock_khz, 600);

  std::printf("two-lane sync: kernel=300ms long=%.2fms idle=%.2fms\n",
              short_kernel.long_ms, short_kernel.idle_ms);
  std::printf("two-lane sync: kernel=600ms long=%.2fms idle=%.2fms\n",
              long_kernel.long_ms, long_kernel.idle_ms);

  constexpr double kIdleLatencyLimitMs = 100.0;
  if (short_kernel.idle_ms > kIdleLatencyLimitMs ||
      long_kernel.idle_ms > kIdleLatencyLimitMs) {
    std::fprintf(stderr,
                 "idle synchronization exceeded %.0fms; stdout capture may "
                 "still be serializing RPC lanes\n",
                 kIdleLatencyLimitMs);
    return EXIT_FAILURE;
  }

  std::printf(
      "PASS: idle synchronization stays bounded across kernel durations\n");
  return EXIT_SUCCESS;
}
