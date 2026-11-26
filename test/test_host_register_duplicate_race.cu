// Two threads register the same range at once: one wins and the other is told
// the range is already registered. The loser frees the backing it allocated
// meanwhile, which must not wait on anything the registration holds, here
// with pinned device-to-host copies landing from another thread.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <unistd.h>

static constexpr int kRounds = 300;
static constexpr size_t kCopyBytes = 1 << 20;

static void check(cudaError_t result, const char *expr, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s\n", expr, line,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

int main() {
  size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t bytes = 4 * page;
  void *range = std::aligned_alloc(page, bytes);
  unsigned char *pinned = nullptr;
  unsigned char *device = nullptr;
  cudaStream_t stream = nullptr;
  CHECK(cudaHostAlloc(&pinned, kCopyBytes, cudaHostAllocDefault));
  CHECK(cudaMalloc(&device, kCopyBytes));
  CHECK(cudaMemset(device, 0x5a, kCopyBytes));
  CHECK(cudaStreamCreate(&stream));

  std::atomic<bool> stop{false};
  std::atomic<int> round{0};
  std::atomic<int> registered{0};
  std::atomic<int> refused{0};
  std::atomic<int> unexpected{0};
  std::atomic<int> finished{0};
  // Leaves each copy's landed bytes uncollected for a moment, as an
  // application that does other work between copies would.
  std::thread copier([&] {
    while (!stop.load(std::memory_order_acquire)) {
      CHECK(cudaMemcpyAsync(pinned, device, kCopyBytes, cudaMemcpyDeviceToHost,
                            stream));
      CHECK(cudaStreamSynchronize(stream));
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  auto racer = [&] {
    for (int r = 1; r <= kRounds; ++r) {
      while (round.load(std::memory_order_acquire) < r) {
        std::this_thread::yield();
      }
      cudaError_t result = cudaHostRegister(range, bytes, 0);
      if (result == cudaSuccess) {
        registered.fetch_add(1);
      } else if (result == cudaErrorHostMemoryAlreadyRegistered) {
        refused.fetch_add(1);
      } else {
        unexpected.fetch_add(1);
      }
      finished.fetch_add(1);
    }
  };
  std::thread first(racer);
  std::thread second(racer);

  int failures = 0;
  for (int r = 1; r <= kRounds && failures == 0; ++r) {
    int before = finished.load();
    round.store(r, std::memory_order_release);
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (finished.load() < before + 2) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::printf("FAIL: round %d: a registration did not return\n", r);
        std::fflush(stdout);
        _exit(EXIT_FAILURE);
      }
      std::this_thread::yield();
    }
    // Both may also run one after the other, in which case the second is
    // refused just the same.
    if (registered.load() != r || refused.load() != r ||
        unexpected.load() != 0) {
      std::printf("FAIL: round %d: %d registered, %d refused, %d other\n", r,
                  registered.load(), refused.load(), unexpected.load());
      failures = 1;
    }
    (void)cudaGetLastError();
    CHECK(cudaHostUnregister(range));
  }
  stop.store(true, std::memory_order_release);
  round.store(kRounds + 1, std::memory_order_release);
  if (failures != 0) {
    std::fflush(stdout);
    _exit(EXIT_FAILURE);
  }
  first.join();
  second.join();
  copier.join();
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(device));
  CHECK(cudaFreeHost(pinned));
  std::free(range);
  std::printf("PASS\n");
  return 0;
}
