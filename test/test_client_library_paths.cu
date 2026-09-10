#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits.h>

static bool library_path(void *symbol, char *path) {
  Dl_info info{};
  if (!dladdr(symbol, &info) || !info.dli_fname ||
      !realpath(info.dli_fname, path)) {
    std::fprintf(stderr, "cannot resolve loaded CUDA library\n");
    return false;
  }
  return true;
}

static bool check_cuda(cudaError_t status, const char *call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(status));
    return false;
  }
  return true;
}

int main() {
  char driver[PATH_MAX], runtime[PATH_MAX], expected[PATH_MAX];
  if (!library_path(reinterpret_cast<void *>(cuDriverGetVersion), driver) ||
      !library_path(reinterpret_cast<void *>(cudaRuntimeGetVersion), runtime)) {
    return 1;
  }
  std::printf("driver=%s\nruntime=%s\n", driver, runtime);
  std::fflush(stdout);

  const char *variant = std::getenv("LUPINE_TEST_VARIANT");
  if (variant) {
    const char *driver_shim = std::getenv("LUPINE_LIB");
    const char *runtime_shim = std::getenv("LUPINE_CUDART_LIB");
    if (!driver_shim || !realpath(driver_shim, expected) ||
        std::strcmp(driver, expected) != 0) {
      std::fprintf(stderr, "driver did not resolve to the selected shim\n");
      return 1;
    }
    if (!runtime_shim || !realpath(runtime_shim, expected)) {
      std::fprintf(stderr, "cannot resolve the expected runtime shim\n");
      return 1;
    }
    const bool shim_loaded = std::strcmp(runtime, expected) == 0;
    const bool driver_only = std::strcmp(variant, "driver-only") == 0;
    const bool driver_runtime = std::strcmp(variant, "driver-runtime") == 0;
    if ((!driver_only && !driver_runtime) || shim_loaded != driver_runtime) {
      std::fprintf(stderr, "incorrect runtime library for %s\n", variant);
      return 1;
    }
  }

  int driver_version = 0, runtime_version = 0, devices = 0;
  if (cuInit(0) != CUDA_SUCCESS ||
      cuDriverGetVersion(&driver_version) != CUDA_SUCCESS ||
      !check_cuda(cudaRuntimeGetVersion(&runtime_version),
                  "cudaRuntimeGetVersion") ||
      !check_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") ||
      devices < 1) {
    std::fprintf(stderr, "driver/runtime device query failed\n");
    return 1;
  }

  unsigned char host[64]{};
  void *device = nullptr;
  if (!check_cuda(cudaMalloc(&device, sizeof(host)), "cudaMalloc")) {
    return 1;
  }
  bool ok =
      check_cuda(cudaMemset(device, 0x2a, sizeof(host)), "cudaMemset") &&
      check_cuda(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost),
                 "cudaMemcpy");
  for (unsigned char value : host) {
    ok = ok && value == 0x2a;
  }

  // Driver-allocated pinned memory makes this a genuinely deferred copy and
  // also checks interoperability between driver allocation and runtime sync.
  void *pinned = nullptr;
  if (cuMemAllocHost(&pinned, sizeof(host)) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuMemAllocHost failed\n");
    return 1;
  }
  auto *async_host = static_cast<unsigned char *>(pinned);
  std::memset(async_host, 0, sizeof(host));
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate") ||
      !check_cuda(cudaMemcpyAsync(async_host, device, sizeof(host),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync") ||
      !check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
    return 1;
  }
  for (size_t i = 0; i < sizeof(host); ++i) {
    if (async_host[i] != 0x2a) {
      std::fprintf(
          stderr,
          "async DtoH after cudaStreamSynchronize: got 0x%x, want 0x2a\n",
          async_host[i]);
      return 1;
    }
  }
  ok = check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy") && ok;
  if (cuMemFreeHost(pinned) != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuMemFreeHost failed\n");
    ok = false;
  }
  ok = check_cuda(cudaFree(device), "cudaFree") && ok;
  if (!ok) {
    std::fprintf(stderr, "runtime GPU memory roundtrip failed\n");
    return 1;
  }
  std::printf("PASS: %s driver=%d runtime=%d devices=%d; sync and async GPU "
              "memory roundtrips\n",
              variant ? variant : "default", driver_version, runtime_version,
              devices);
  return 0;
}
