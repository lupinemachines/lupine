// Stream callbacks under concurrency. Many threads pipeline
// pinned-alloc / async copies / cuStreamAddCallback / frees on one
// connection; the callback delivery must never block the server's CUDA
// callback thread on a client round trip, or the connection deadlocks and
// dies on the dead-peer timer. Also checks that cuStreamSynchronize still
// waits for the client-side callback to finish, since the server-side
// stream no longer does.
#include <cuda.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
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

struct CallbackSignal {
  std::mutex mutex;
  std::condition_variable cond;
  bool done = false;

  void signal() {
    std::lock_guard<std::mutex> lock(mutex);
    done = true;
    cond.notify_all();
  }

  bool wait_seconds(int seconds) {
    std::unique_lock<std::mutex> lock(mutex);
    return cond.wait_for(lock, std::chrono::seconds(seconds),
                         [this] { return done; });
  }
};

static void CUDA_CB signal_callback(CUstream, CUresult status,
                                    void *user_data) {
  check(status, "callback status", __LINE__);
  static_cast<CallbackSignal *>(user_data)->signal();
}

static std::atomic<bool> slow_callback_finished{false};

static void CUDA_CB slow_callback(CUstream, CUresult status, void *) {
  check(status, "slow callback status", __LINE__);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  slow_callback_finished.store(true, std::memory_order_release);
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

  // cuStreamSynchronize must not return before a pending stream callback has
  // finished running on this side of the connection.
  {
    CUstream stream = nullptr;
    CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CUdeviceptr d = 0;
    CHECK(cuMemAlloc(&d, 4096));
    CHECK(cuMemsetD8Async(d, 0, 4096, stream));
    CHECK(cuStreamAddCallback(stream, slow_callback, nullptr, 0));
    CHECK(cuStreamSynchronize(stream));
    if (!slow_callback_finished.load(std::memory_order_acquire)) {
      std::fprintf(stderr,
                   "FAIL: cuStreamSynchronize returned before the stream "
                   "callback finished\n");
      return EXIT_FAILURE;
    }
    CHECK(cuMemFree(d));
    CHECK(cuStreamDestroy(stream));
    std::printf("sync waits for pending stream callback\n");
  }

  constexpr int kThreads = 8;
  constexpr int kIterations = 4;
  constexpr size_t kElements = 100000;

  std::vector<std::thread> threads;
  std::atomic<bool> failed{false};
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      check(cuCtxSetCurrent(context), "cuCtxSetCurrent", __LINE__);
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        CUstream stream = nullptr;
        CUdeviceptr d_data = 0;
        int *h_data = nullptr;
        check(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate",
              __LINE__);
        check(cuMemAlloc(&d_data, kElements * sizeof(int)), "cuMemAlloc",
              __LINE__);
        check(cuMemHostAlloc(reinterpret_cast<void **>(&h_data),
                             kElements * sizeof(int), CU_MEMHOSTALLOC_PORTABLE),
              "cuMemHostAlloc", __LINE__);
        int base = t * 1000 + iteration;
        for (size_t i = 0; i < kElements; ++i) {
          h_data[i] = base + static_cast<int>(i);
        }
        check(cuMemcpyHtoDAsync(d_data, h_data, kElements * sizeof(int),
                                stream),
              "cuMemcpyHtoDAsync", __LINE__);
        check(cuMemcpyDtoHAsync(h_data, d_data, kElements * sizeof(int),
                                stream),
              "cuMemcpyDtoHAsync", __LINE__);
        CallbackSignal signal;
        check(cuStreamAddCallback(stream, signal_callback, &signal, 0),
              "cuStreamAddCallback", __LINE__);
        if (!signal.wait_seconds(60)) {
          std::fprintf(stderr,
                       "FAIL: stream callback t=%d iteration=%d never fired\n",
                       t, iteration);
          failed.store(true, std::memory_order_release);
          return;
        }
        for (size_t i = 0; i < kElements; i += 4096) {
          if (h_data[i] != base + static_cast<int>(i)) {
            std::fprintf(stderr, "FAIL: data mismatch t=%d iteration=%d i=%zu\n",
                         t, iteration, i);
            failed.store(true, std::memory_order_release);
            return;
          }
        }
        check(cuMemFree(d_data), "cuMemFree", __LINE__);
        check(cuMemFreeHost(h_data), "cuMemFreeHost", __LINE__);
        check(cuStreamDestroy(stream), "cuStreamDestroy", __LINE__);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  if (failed.load(std::memory_order_acquire)) {
    return EXIT_FAILURE;
  }

  std::printf("PASS: %d threads x %d callback iterations survived\n", kThreads,
              kIterations);
  return EXIT_SUCCESS;
}
