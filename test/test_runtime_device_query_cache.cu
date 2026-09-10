#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#ifndef _WIN32
#include "../cuda_client_rpc.h"
#include <dlfcn.h>
#endif

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

#ifndef _WIN32
  auto epoch = reinterpret_cast<uint64_t (*)(conn_t *)>(
      dlsym(RTLD_DEFAULT, "lupine_rpc_thread_request_epoch"));
  auto connection = reinterpret_cast<conn_t *(*)(unsigned int)>(
      dlsym(RTLD_DEFAULT, "lupine_rpc_client_get_connection"));
  uint64_t before = epoch && connection ? epoch(connection(0)) : 0;
#endif
  for (int i = 0; i < 100; ++i) {
    require(cudaGetDevice(&actual) == cudaSuccess && actual == expected,
            "repeated cudaGetDevice returned the wrong device");
  }
#ifndef _WIN32
  if (before != 0) {
    require(epoch(connection(0)) == before,
            "repeated device queries introduced network round trips");
  }
#endif
}

int main() {
  int count = 0;
  require(cudaGetDeviceCount(&count) == cudaSuccess && count > 0,
          "cudaGetDeviceCount failed");
  require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  check_device(0);
  require(cudaGetDevice(nullptr) == cudaErrorInvalidValue,
          "cudaGetDevice accepted a null output");
  check_device(0);

  int selected = count - 1;
  require(cudaSetDevice(selected) == cudaSuccess, "cudaSetDevice failed");
  check_device(selected);
  require(cudaSetDevice(count) == cudaErrorInvalidDevice,
          "cudaSetDevice accepted an invalid ordinal");
  check_device(selected);

  // Device state belongs to the calling thread, even when the first thread
  // already has a cached answer for another server/device.
  std::thread other([] {
    require(cudaSetDevice(0) == cudaSuccess, "thread cudaSetDevice failed");
    check_device(0);
  });
  other.join();
  check_device(selected);

  require(cudaSetDevice(0) == cudaSuccess, "restore device zero failed");
  CUcontext context = nullptr;
  require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr,
          "runtime did not initialize a driver context");
  require(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS, "detach context failed");
  check_device(0);
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "restore context failed");
  check_device(0);
  require(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
  check_device(0);
  std::puts("runtime device queries preserve device, thread, and context state "
            "without repeated RPCs");
}
