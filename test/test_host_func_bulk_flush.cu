// A host function that dirties a large pinned range must be able to publish
// it while another thread's bulk device-to-host copy waits behind that host
// function on the same stream. The copy holds the bulk connections until the
// stream reaches it, so the callback's flush must not wait for them.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static constexpr size_t kDirtyBytes = 16u << 20;
static constexpr size_t kCopyBytes = 64u << 20;

static void check(cudaError_t result, const char *expr, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s\n", expr, line,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

struct CallbackState {
  unsigned char *host = nullptr;
  unsigned char value = 0;
  std::atomic<bool> ready{false};
};

static bool wait_for(const std::atomic<bool> &flag, int seconds) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

__global__ void touch(const unsigned char *mapped, unsigned char *out) {
  out[0] = mapped[0];
}

static void CUDART_CB fill(void *opaque) {
  auto *state = static_cast<CallbackState *>(opaque);
  if (!wait_for(state->ready, 60)) {
    return;
  }
  std::memset(state->host, state->value, kDirtyBytes);
}

int main() {
  CallbackState state;
  unsigned char *device_dirty = nullptr;
  unsigned char *device_copy = nullptr;
  cudaStream_t stream = nullptr;
  CHECK(cudaHostAlloc(&state.host, kDirtyBytes, cudaHostAllocMapped));
  CHECK(cudaMalloc(&device_dirty, kDirtyBytes));
  CHECK(cudaMalloc(&device_copy, kCopyBytes));
  CHECK(cudaMemset(device_copy, 0x5a, kCopyBytes));
  CHECK(cudaStreamCreate(&stream));
  std::memset(state.host, 0, kDirtyBytes);
  // Device code reaching the mapped range makes host writes to it tracked.
  unsigned char *mapped = nullptr;
  CHECK(cudaHostGetDevicePointer(&mapped, state.host, 0));
  touch<<<1, 1, 0, stream>>>(mapped, device_dirty);
  CHECK(cudaGetLastError());
  CHECK(cudaStreamSynchronize(stream));
  std::vector<unsigned char> readback(kCopyBytes);
  // Opens the bulk connections before the host function is queued.
  CHECK(cudaMemcpy(readback.data(), device_copy, kCopyBytes,
                   cudaMemcpyDeviceToHost));

  int failures = 0;
  for (int round = 0; round < 3; ++round) {
    state.value = static_cast<unsigned char>(round + 1);
    state.ready.store(false, std::memory_order_relaxed);
    CHECK(cudaLaunchHostFunc(stream, fill, &state));
    std::atomic<bool> copied{false};
    std::thread copier([&] {
      CHECK(cudaMemcpyAsync(readback.data(), device_copy, kCopyBytes,
                            cudaMemcpyDeviceToHost, stream));
      CHECK(cudaStreamSynchronize(stream));
      copied.store(true, std::memory_order_release);
    });
    // Let the copy take the bulk connections before the host function runs.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    state.ready.store(true, std::memory_order_release);
    if (!wait_for(copied, 60)) {
      std::fprintf(stderr,
                   "FAIL: round %d: the bulk copy behind a host function that "
                   "dirtied %zu bytes did not finish\n",
                   round, kDirtyBytes);
      std::fflush(stderr);
      std::_Exit(EXIT_FAILURE);
    }
    copier.join();
    CHECK(cudaMemcpy(device_dirty, state.host, kDirtyBytes,
                     cudaMemcpyHostToDevice));
    std::vector<unsigned char> dirty(kDirtyBytes);
    CHECK(cudaMemcpy(dirty.data(), device_dirty, kDirtyBytes,
                     cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kCopyBytes; i += 4096) {
      if (readback[i] != 0x5a) {
        std::fprintf(stderr, "FAIL: round %d copy read %d at %zu\n", round,
                     readback[i], i);
        ++failures;
        break;
      }
    }
    for (size_t i = 0; i < kDirtyBytes; i += 4096) {
      if (dirty[i] != state.value) {
        std::fprintf(stderr, "FAIL: round %d host write lost at %zu\n", round,
                     i);
        ++failures;
        break;
      }
    }
  }
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(device_copy));
  CHECK(cudaFree(device_dirty));
  CHECK(cudaFreeHost(state.host));
  if (failures != 0) {
    return EXIT_FAILURE;
  }
  std::printf("PASS: a host function's large flush does not wait for a bulk "
              "copy queued behind it\n");
  return EXIT_SUCCESS;
}
