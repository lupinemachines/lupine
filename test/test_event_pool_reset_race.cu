#include <cuda.h>

#include <algorithm>
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
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device), "retain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }
  // The race window is the creator's batch round trip, so step the creator's
  // start across the release in half round trips.
  auto rtt = std::chrono::steady_clock::duration::max();
  for (int i = 0; i < 5; ++i) {
    auto begin = std::chrono::steady_clock::now();
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize")) {
      return 1;
    }
    rtt = std::min(rtt, std::chrono::steady_clock::now() - begin);
  }
  int stale = 0;
  for (int trial = 0; trial < 24; ++trial) {
    // Drop the last reference while the creator's batch request is in
    // flight. On a hit the batch outlives its context, and the next retain
    // usually returns the same handle value, whose pool it poisons.
    auto delay = rtt * trial / 2;
    // The previous release emptied every pool and its check refilled only
    // its own flags, so the creator fetches a batch.
    unsigned int flags = CU_EVENT_DISABLE_TIMING;
    if (trial % 2 != 0) {
      flags |= CU_EVENT_BLOCKING_SYNC;
    }
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
    if (!check(cuDevicePrimaryCtxRelease(device), "release")) {
      creator.join();
      return 1;
    }
    creator.join();
    if (!check(cuDevicePrimaryCtxRetain(&context, device), "retain") ||
        !check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
      return 1;
    }
    // Use several pooled events: a poisoned one is freed memory on the
    // server, which does not reliably fail on a single use.
    CUevent events[8] = {};
    for (CUevent &event : events) {
      if (!check(cuEventCreate(&event, flags), "cuEventCreate")) {
        return 1;
      }
      if (cuEventRecord(event, nullptr) != CUDA_SUCCESS) {
        ++stale;
      }
    }
    for (CUevent event : events) {
      if (cuEventSynchronize(event) != CUDA_SUCCESS) {
        ++stale;
      }
      (void)cuEventDestroy(event);
    }
  }
  if (stale != 0) {
    std::fprintf(stderr, "%d events from a released context\n", stale);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
