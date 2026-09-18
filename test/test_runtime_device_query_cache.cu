#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static void check_device(int expected) {
  int actual = -1;
  require(cudaGetDevice(&actual) == cudaSuccess && actual == expected,
          "cudaGetDevice returned a stale or incorrect device");

  for (int i = 0; i < 100; ++i) {
    require(cudaGetDevice(&actual) == cudaSuccess && actual == expected,
            "repeated cudaGetDevice returned the wrong device");
  }
}

// A device answer must not survive because unrelated requests keep refreshing
// it, nor go stale because they invalidate it.
static void unrelated_work() {
  void *buffer = nullptr;
  require(cudaMalloc(&buffer, 256) == cudaSuccess, "cudaMalloc failed");
  require(cudaMemset(buffer, 0, 256) == cudaSuccess, "cudaMemset failed");
  require(cudaFree(buffer) == cudaSuccess, "cudaFree failed");
}

static void check_device_under_traffic(int expected) {
  check_device(expected);
  unrelated_work();
  check_device(expected);
}

// The device a context belongs to, read through the driver's own tracking.
static int context_device(CUcontext context) {
  CUcontext previous = nullptr;
  require(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "cuCtxSetCurrent failed");
  CUdevice device = -1;
  require(cuCtxGetDevice(&device) == CUDA_SUCCESS, "cuCtxGetDevice failed");
  require(cuCtxSetCurrent(previous) == CUDA_SUCCESS, "restore context failed");
  return static_cast<int>(device);
}

int main() {
  int count = 0;
  require(cudaGetDeviceCount(&count) == cudaSuccess && count > 0,
          "cudaGetDeviceCount failed");
  require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  check_device_under_traffic(0);
  require(cudaGetDevice(nullptr) == cudaErrorInvalidValue,
          "cudaGetDevice accepted a null output");
  check_device_under_traffic(0);

  int selected = count - 1;
  require(cudaSetDevice(selected) == cudaSuccess, "cudaSetDevice failed");
  check_device_under_traffic(selected);
  require(cudaSetDevice(count) == cudaErrorInvalidDevice,
          "cudaSetDevice accepted an invalid ordinal");
  check_device_under_traffic(selected);

  // Alternating selections: a device answer kept one call too long shows up
  // here even when nothing else touches the lane in between.
  for (int i = 0; i < 20; ++i) {
    const int device = i % count;
    require(cudaSetDevice(device) == cudaSuccess, "alternating set failed");
    int actual = -1;
    require(cudaGetDevice(&actual) == cudaSuccess && actual == device,
            "alternating cudaSetDevice/cudaGetDevice disagreed");
  }

  // Device state belongs to the calling thread, even when the first thread
  // already has a cached answer for another server/device.
  require(cudaSetDevice(selected) == cudaSuccess, "cudaSetDevice failed");
  std::thread other([] {
    require(cudaSetDevice(0) == cudaSuccess, "thread cudaSetDevice failed");
    check_device_under_traffic(0);
  });
  other.join();
  check_device_under_traffic(selected);

  // Threads switching devices concurrently: one thread's selection must not
  // answer another's query.
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([i, count] {
      for (int round = 0; round < 25; ++round) {
        const int device = (i + round) % count;
        require(cudaSetDevice(device) == cudaSuccess,
                "concurrent cudaSetDevice failed");
        int actual = -1;
        require(cudaGetDevice(&actual) == cudaSuccess && actual == device,
                "concurrent cudaGetDevice returned another thread's device");
        unrelated_work();
        require(cudaGetDevice(&actual) == cudaSuccess && actual == device,
                "concurrent cudaGetDevice went stale");
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  check_device_under_traffic(selected);

  require(cudaSetDevice(0) == cudaSuccess, "restore device zero failed");
  CUcontext context = nullptr;
  require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr,
          "runtime did not initialize a driver context");
  require(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS, "detach context failed");
  check_device_under_traffic(0);
  // Selecting a device with no context current leaves nothing in the context
  // handle to notice the move by.
  require(cudaSetDevice(selected) == cudaSuccess,
          "cudaSetDevice with no current context failed");
  check_device_under_traffic(selected);
  require(cudaSetDevice(0) == cudaSuccess, "restore device zero failed");
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
  check_device_under_traffic(0);
  CUdevice driver_device = 0;
  require(cuDeviceGet(&driver_device, 0) == CUDA_SUCCESS, "cuDeviceGet failed");
  for (int i = 0; i < 3; ++i) {
    CUcontext temporary = nullptr;
    require(cuCtxCreate_v2(&temporary, 0, driver_device) == CUDA_SUCCESS,
            "create non-primary context failed");
    check_device_under_traffic(0);
    require(cuCtxDestroy(temporary) == CUDA_SUCCESS, "destroy context failed");
    require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
    check_device_under_traffic(0);
  }

  // cudaSetDevice binds the device's primary context even when that device is
  // already current, so a driver context set on the lane in between has to be
  // displaced rather than left in place.
  CUcontext non_primary = nullptr;
  require(cuCtxCreate_v2(&non_primary, 0, driver_device) == CUDA_SUCCESS,
          "create non-primary context failed");
  CUcontext current = nullptr;
  require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == non_primary,
          "cuCtxCreate did not make its context current");
  require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");
  require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != non_primary,
          "cudaSetDevice left a non-primary context current");
  check_device_under_traffic(0);
  require(cuCtxDestroy(non_primary) == CUDA_SUCCESS, "destroy context failed");
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
  check_device_under_traffic(0);

  // A driver context on another device answers device queries while it is
  // current, through set, push and pop alike.
  if (count > 1) {
    CUdevice other_device = 0;
    require(cuDeviceGet(&other_device, selected) == CUDA_SUCCESS,
            "cuDeviceGet failed");
    CUcontext other_context = nullptr;
    require(cuCtxCreate_v2(&other_context, 0, other_device) == CUDA_SUCCESS,
            "create context on second device failed");
    require(context_device(other_context) == selected,
            "context reported the wrong device");
    require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
    check_device_under_traffic(0);
    for (int i = 0; i < 5; ++i) {
      require(cuCtxPushCurrent(other_context) == CUDA_SUCCESS,
              "cuCtxPushCurrent failed");
      check_device_under_traffic(selected);
      require(cuCtxPopCurrent(nullptr) == CUDA_SUCCESS,
              "cuCtxPopCurrent failed");
      check_device_under_traffic(0);
      require(cuCtxSetCurrent(other_context) == CUDA_SUCCESS,
              "cuCtxSetCurrent failed");
      check_device_under_traffic(selected);
      require(cuCtxSetCurrent(context) == CUDA_SUCCESS,
              "cuCtxSetCurrent failed");
      check_device_under_traffic(0);
    }
    require(cuCtxDestroy(other_context) == CUDA_SUCCESS,
            "destroy context failed");
    require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
    check_device_under_traffic(0);

    // Select another device and read it back before any driver call makes the
    // shim notice the context that came with it, then hand device zero's
    // context back. The handle a query is keyed on is the same one on both
    // sides of the move, so only the binding itself says the answer changed.
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
    CUcontext zero_context = nullptr;
    require(cuCtxGetCurrent(&zero_context) == CUDA_SUCCESS &&
                zero_context != nullptr,
            "device zero has no current context");
    require(cudaSetDevice(selected) == cudaSuccess, "cudaSetDevice failed");
    int moved = -1;
    require(cudaGetDevice(&moved) == cudaSuccess && moved == selected,
            "cudaGetDevice did not follow cudaSetDevice");
    require(cuCtxSetCurrent(zero_context) == CUDA_SUCCESS,
            "restore device zero context failed");
    check_device_under_traffic(0);
  }

  require(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
  check_device(0);
  std::puts("runtime device queries preserve device, thread, and context state "
            "without repeated RPCs");
}
