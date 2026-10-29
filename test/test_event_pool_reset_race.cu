#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!check(cuInit(0), "cuInit") || !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device), "retain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }
  int stale = 0;
  for (int trial = 0; trial < 400; ++trial) {
    // The previous trial's reset emptied every pool and its check refilled
    // only its own flags, so the creator fetches a batch.
    unsigned int flags = CU_EVENT_DISABLE_TIMING;
    if (trial % 2 != 0) {
      flags |= CU_EVENT_BLOCKING_SYNC;
    }
    // Sweep the creator's batch round trip across the other thread's reset.
    // On a hit, the batch outlives its context and poisons the next pool.
    auto delay =
        std::chrono::microseconds(trial < 200 ? 5 * trial : 50 * (trial - 200));
    std::atomic<int> ready{0};
    std::thread creator([&] {
      cuCtxSetCurrent(context);
      ready.fetch_add(1);
      while (ready.load() < 2) {
      }
      auto until = std::chrono::steady_clock::now() + delay;
      while (std::chrono::steady_clock::now() < until) {
      }
      CUevent event = nullptr;
      (void)cuEventCreate(&event, flags);
    });
    ready.fetch_add(1);
    while (ready.load() < 2) {
    }
    if (!check(cuDevicePrimaryCtxReset(device), "reset")) {
      creator.join();
      return 1;
    }
    creator.join();
    CUcontext again = nullptr;
    CUevent event = nullptr;
    if (!check(cuDevicePrimaryCtxRetain(&again, device), "retain") ||
        !check(cuDevicePrimaryCtxRelease(device), "release") ||
        !check(cuCtxSetCurrent(again), "cuCtxSetCurrent") ||
        !check(cuEventCreate(&event, flags), "cuEventCreate")) {
      return 1;
    }
    context = again;
    if (cuEventRecord(event, nullptr) != CUDA_SUCCESS ||
        cuEventSynchronize(event) != CUDA_SUCCESS) {
      ++stale;
    }
    (void)cuEventDestroy(event);
  }
  if (stale != 0) {
    std::fprintf(stderr, "%d events from a reset context\n", stale);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
