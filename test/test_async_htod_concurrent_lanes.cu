#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static void check(cudaError_t result, const char *expression, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expression, line,
                 cudaGetErrorString(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

int main() {
  constexpr int thread_count = 2;
  constexpr int iterations = 8;
  constexpr size_t bytes = 64 * 1024 * 1024;

  CHECK(cudaSetDevice(0));
  CHECK(cudaFree(nullptr));

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  for (int thread_index = 0; thread_index != thread_count; ++thread_index) {
    workers.emplace_back([&, thread_index] {
      CHECK(cudaSetDevice(0));
      cudaStream_t stream = nullptr;
      unsigned char *remote = nullptr;
      CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CHECK(cudaMalloc(&remote, bytes));
      std::vector<unsigned char> source(bytes);
      std::vector<unsigned char> destination(bytes);

      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int iteration = 0; iteration != iterations; ++iteration) {
        unsigned char pattern = static_cast<unsigned char>(
            1 + thread_index * iterations + iteration);
        std::fill(source.begin(), source.end(), pattern);
        CHECK(cudaMemcpyAsync(remote, source.data(), bytes,
                              cudaMemcpyHostToDevice, stream));
        std::fill(source.begin(), source.end(), 0xee);
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(destination.data(), remote, bytes,
                         cudaMemcpyDeviceToHost));
        if (std::find_if(destination.begin(), destination.end(),
                         [pattern](unsigned char value) {
                           return value != pattern;
                         }) != destination.end()) {
          failures.fetch_add(1, std::memory_order_relaxed);
          break;
        }
      }

      CHECK(cudaFree(remote));
      CHECK(cudaStreamDestroy(stream));
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  CHECK(cudaDeviceReset());
  if (failures.load(std::memory_order_relaxed) != 0) {
    std::fprintf(stderr, "concurrent async HtoD lane data mismatch\n");
    return EXIT_FAILURE;
  }
  std::printf("PASS: concurrent async HtoD RPC lanes preserve staging "
              "ownership\n");
  return EXIT_SUCCESS;
}
