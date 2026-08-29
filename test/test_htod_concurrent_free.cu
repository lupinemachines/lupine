#include <cuda.h>

// Regression test for the server-side pageable HtoD producer deadlock. This
// file is auto-discovered by test/run_custom_tests.sh.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr size_t kCopyBytes = 2048;
constexpr size_t kFreeBytes = 1024 * 1024;
constexpr int kCopyIterations = 2000;
constexpr int kFreeIterations = 2000;
constexpr int kFreeThreads = 3;

const char *error_name(CUresult result) {
  const char *name = nullptr;
  (void)cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

bool record(CUresult result, const char *operation, std::atomic<int> &failure) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  int expected = 0;
  if (failure.compare_exchange_strong(expected, static_cast<int>(result))) {
    std::fprintf(stderr, "%s failed: %s (%d)\n", operation, error_name(result),
                 static_cast<int>(result));
  }
  return false;
}

} // namespace

int main() {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuInit failed: %s (%d)\n", error_name(result),
                 static_cast<int>(result));
    return 1;
  }

  CUdevice device = 0;
  if ((result = cuDeviceGet(&device, 0)) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuDeviceGet failed: %s (%d)\n", error_name(result),
                 static_cast<int>(result));
    return 1;
  }

  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  result = cuCtxCreate(&context, nullptr, 0, device);
#else
  result = cuCtxCreate(&context, 0, device);
#endif
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuCtxCreate failed: %s (%d)\n", error_name(result),
                 static_cast<int>(result));
    return 1;
  }

  CUstream stream = nullptr;
  CUdeviceptr destination = 0;
  if ((result = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING)) !=
          CUDA_SUCCESS ||
      (result = cuMemAlloc(&destination, kCopyBytes)) != CUDA_SUCCESS) {
    std::fprintf(stderr, "setup failed: %s (%d)\n", error_name(result),
                 static_cast<int>(result));
    return 1;
  }

  std::vector<unsigned char> source(kCopyBytes, 0x5a);
  std::atomic<bool> start{false};
  std::atomic<bool> finished{false};
  std::atomic<int> failure{0};
  std::atomic<int> copy_progress{0};
  std::atomic<int> free_progress{0};

  std::thread watchdog([&] {
    for (int poll = 0; poll < 200; ++poll) {
      if (finished.load(std::memory_order_acquire)) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::fprintf(stderr,
                 "timed out racing pageable HtoD against unrelated frees "
                 "(copies=%d frees=%d)\n",
                 copy_progress.load(std::memory_order_relaxed),
                 free_progress.load(std::memory_order_relaxed));
    std::_Exit(124);
  });

  std::thread copier([&] {
    if (!record(cuCtxSetCurrent(context), "copy cuCtxSetCurrent", failure)) {
      return;
    }
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int iteration = 0; iteration < kCopyIterations &&
                            failure.load(std::memory_order_acquire) == 0;
         ++iteration) {
      if (!record(cuMemcpyHtoDAsync(destination, source.data(), source.size(),
                                    stream),
                  "cuMemcpyHtoDAsync", failure)) {
        return;
      }
      copy_progress.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::vector<std::thread> free_threads;
  free_threads.reserve(kFreeThreads);
  for (int thread = 0; thread < kFreeThreads; ++thread) {
    free_threads.emplace_back([&] {
      if (!record(cuCtxSetCurrent(context), "free cuCtxSetCurrent", failure)) {
        return;
      }
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kFreeIterations &&
                              failure.load(std::memory_order_acquire) == 0;
           ++iteration) {
        CUdeviceptr allocation = 0;
        if (!record(cuMemAlloc(&allocation, kFreeBytes), "cuMemAlloc",
                    failure) ||
            !record(cuMemFree(allocation), "cuMemFree", failure)) {
          return;
        }
        free_progress.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  copier.join();
  for (auto &thread : free_threads) {
    thread.join();
  }

  if (failure.load(std::memory_order_acquire) == 0) {
    (void)record(cuStreamSynchronize(stream), "cuStreamSynchronize", failure);
  }
  std::vector<unsigned char> copied(kCopyBytes);
  if (failure.load(std::memory_order_acquire) == 0 &&
      record(cuMemcpyDtoH(copied.data(), destination, copied.size()),
             "cuMemcpyDtoH", failure) &&
      copied != source) {
    std::fprintf(stderr, "HtoD result did not match the pageable source\n");
    failure.store(-1, std::memory_order_release);
  }
  finished.store(true, std::memory_order_release);
  watchdog.join();

  (void)cuMemFree(destination);
  (void)cuStreamDestroy(stream);
  (void)cuCtxDestroy(context);
  if (failure.load(std::memory_order_acquire) != 0) {
    return 1;
  }
  std::printf("PASS: pageable HtoD remains live during unrelated frees\n");
  return 0;
}
