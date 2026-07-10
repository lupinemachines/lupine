#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

static bool expect_schedule(unsigned int expected) {
  for (int i = 0; i < 8; ++i) {
    unsigned int flags = 0;
    CUresult result = cuCtxGetFlags(&flags);
    if (result != CUDA_SUCCESS || (flags & CU_CTX_SCHED_MASK) != expected) {
      std::fprintf(stderr,
                   "unexpected context flags: result=%s (%d), got=%u, "
                   "expected=%u\n",
                   result_name(result), static_cast<int>(result),
                   flags & CU_CTX_SCHED_MASK, expected);
      return false;
    }
  }
  return true;
}

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  CUcontext spin_context = nullptr;
  CUcontext yield_context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&spin_context, nullptr, CU_CTX_SCHED_SPIN, device));
  CHECK(cuCtxCreate(&yield_context, nullptr, CU_CTX_SCHED_YIELD, device));
#else
  CHECK(cuCtxCreate(&spin_context, CU_CTX_SCHED_SPIN, device));
  CHECK(cuCtxCreate(&yield_context, CU_CTX_SCHED_YIELD, device));
#endif

  if (!expect_schedule(CU_CTX_SCHED_YIELD)) {
    return EXIT_FAILURE;
  }
  CHECK(cuCtxSetCurrent(spin_context));
  if (!expect_schedule(CU_CTX_SCHED_SPIN)) {
    return EXIT_FAILURE;
  }

  CHECK(cuCtxSetCurrent(yield_context));
  CHECK(cuCtxPushCurrent(spin_context));
  if (!expect_schedule(CU_CTX_SCHED_SPIN)) {
    return EXIT_FAILURE;
  }
  CUcontext popped = nullptr;
  CHECK(cuCtxPopCurrent(&popped));
  if (popped != spin_context || !expect_schedule(CU_CTX_SCHED_YIELD)) {
    std::fprintf(stderr,
                 "context push/pop did not restore the prior context\n");
    return EXIT_FAILURE;
  }

  CHECK(cuCtxSetCurrent(nullptr));
  std::atomic<int> failures{0};
  std::thread spin_worker([&]() {
    if (cuCtxSetCurrent(spin_context) != CUDA_SUCCESS ||
        !expect_schedule(CU_CTX_SCHED_SPIN)) {
      failures.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::thread yield_worker([&]() {
    if (cuCtxSetCurrent(yield_context) != CUDA_SUCCESS ||
        !expect_schedule(CU_CTX_SCHED_YIELD)) {
      failures.fetch_add(1, std::memory_order_relaxed);
    }
  });
  spin_worker.join();
  yield_worker.join();

  // A new lane starts without a server-side current context. It must apply the
  // process default hint once, then reuse it for subsequent default routing.
  CHECK(cuCtxSetCurrent(yield_context));
  CHECK(cuCtxSetCurrent(nullptr));
  for (int i = 0; i < 2; ++i) {
    std::thread default_worker([&]() {
      if (!expect_schedule(CU_CTX_SCHED_YIELD)) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
    default_worker.join();
  }

  if (failures.load(std::memory_order_relaxed) != 0) {
    return EXIT_FAILURE;
  }

  // Destroying any context invalidates the lane cache generation. The current
  // context must be re-established before another default-routed call.
  CHECK(cuCtxSetCurrent(yield_context));
  if (!expect_schedule(CU_CTX_SCHED_YIELD)) {
    return EXIT_FAILURE;
  }
  CHECK(cuCtxDestroy(spin_context));
  if (!expect_schedule(CU_CTX_SCHED_YIELD)) {
    return EXIT_FAILURE;
  }
  CHECK(cuCtxDestroy(yield_context));
  std::printf(
      "PASS: cached context routing is correct across lanes and switches\n");
  return EXIT_SUCCESS;
}
