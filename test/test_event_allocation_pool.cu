// Exercise batch boundaries, flag buckets, concurrent creators and context
// invalidation. Run natively as well as through the remote driver shim.
#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

static void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s: CUDA result %d\n", operation, int(result));
    std::exit(1);
  }
}

static void exercise(CUcontext context, unsigned flags, size_t count) {
  check(cuCtxSetCurrent(context), "set context");
  std::vector<CUevent> events(count, nullptr);
  std::set<CUevent> unique;
  for (auto &event : events) {
    check(cuEventCreate(&event, flags), "create event");
    if (event == nullptr || !unique.insert(event).second) {
      std::fprintf(stderr, "duplicate live event\n");
      std::exit(1);
    }
    check(cuEventQuery(event), "query unrecorded event");
    check(cuEventRecord(event, nullptr), "record event");
  }
  for (auto event : events) {
    check(cuEventSynchronize(event), "synchronize event");
  }
  if (!(flags & CU_EVENT_DISABLE_TIMING) && events.size() > 1) {
    float elapsed = -1;
    check(cuEventElapsedTime(&elapsed, events.front(), events.back()),
          "timing");
    if (elapsed < 0) {
      std::exit(1);
    }
  }
  for (auto event : events) {
    check(cuEventDestroy(event), "destroy event");
  }
}

int main() {
  check(cuInit(0), "init");
  CUdevice device = 0;
  check(cuDeviceGet(&device, 0), "device");
  CUcontext primary = nullptr;
  check(cuDevicePrimaryCtxRetain(&primary, device), "retain");
  check(cuCtxSetCurrent(primary), "set primary");
  if (std::getenv("LUPINE_TEST_EVENT_FIRST_OOM")) {
    CUevent event = reinterpret_cast<CUevent>(uintptr_t(0x1234));
    if (cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) !=
            CUDA_ERROR_OUT_OF_MEMORY ||
        event != reinterpret_cast<CUevent>(uintptr_t(0x1234))) {
      std::fprintf(stderr, "first allocation failure changed the output\n");
      return 1;
    }
  }
  for (unsigned flags :
       {0u, unsigned(CU_EVENT_BLOCKING_SYNC), unsigned(CU_EVENT_DISABLE_TIMING),
        unsigned(CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING)}) {
    exercise(primary, flags, 65);
  }
  CUevent invalid = reinterpret_cast<CUevent>(uintptr_t(0x1234));
  if (cuEventCreate(&invalid, 0x80000000u) != CUDA_ERROR_INVALID_VALUE ||
      invalid != reinterpret_cast<CUevent>(uintptr_t(0x1234))) {
    std::fprintf(stderr, "invalid flags changed output or error\n");
    return 1;
  }
  if (cuEventCreate(nullptr, 0) != CUDA_ERROR_INVALID_VALUE) {
    return 1;
  }
  CUevent ipc = nullptr;
  check(cuEventCreate(&ipc, CU_EVENT_INTERPROCESS | CU_EVENT_DISABLE_TIMING),
        "create IPC event");
  CUipcEventHandle ipc_handle;
  check(cuIpcGetEventHandle(&ipc_handle, ipc), "export IPC event");
  check(cuEventDestroy(ipc), "destroy IPC event");

  std::mutex mutex;
  std::set<CUevent> live;
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&] {
      check(cuCtxSetCurrent(primary), "thread context");
      for (int i = 0; i < 20; ++i) {
        CUevent event = nullptr;
        check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "thread create");
        std::lock_guard<std::mutex> lock(mutex);
        if (!live.insert(event).second) {
          std::abort();
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  for (auto event : live) {
    check(cuEventDestroy(event), "thread event destroy");
  }

  // A user-created context must have its own buckets and leave no stale
  // handles if the driver later reuses its context address.
  for (int iteration = 0; iteration < 3; ++iteration) {
    CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
    check(cuCtxCreate(&context, nullptr, 0, device), "create context");
#else
    check(cuCtxCreate(&context, 0, device), "create context");
#endif
    exercise(context, CU_EVENT_DISABLE_TIMING, 3);
    check(cuCtxDestroy(context), "destroy context");
  }
  check(cuCtxSetCurrent(nullptr), "clear context");
  check(cuDevicePrimaryCtxReset(device), "reset primary");
  check(cuDevicePrimaryCtxRetain(&primary, device), "retain reset primary");
  exercise(primary, CU_EVENT_DISABLE_TIMING, 3);
  check(cuCtxSetCurrent(nullptr), "clear reset context");
  check(cuDevicePrimaryCtxRelease(device), "release extra retain");
  check(cuDevicePrimaryCtxRelease(device), "release primary");
  check(cuDevicePrimaryCtxRetain(&primary, device), "retain released primary");
  exercise(primary, CU_EVENT_DISABLE_TIMING, 3);
  check(cuCtxSetCurrent(nullptr), "clear released context");
  check(cuDevicePrimaryCtxRelease(device), "final release");
  std::puts("test_event_allocation_pool: PASS");
}
