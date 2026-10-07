#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

__global__ void add_one(int *values) { values[threadIdx.x] += 1; }

// A round of ordinary work: allocate, launch, read back. A reset unloads the
// primary context's modules, so the launch is what proves the context that came
// back is usable and not just allocatable.
static void round_trip(const char *stage) {
  int *device_values = nullptr;
  cudaError_t status = cudaMalloc(&device_values, 4 * sizeof(int));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: cudaMalloc failed: %s\n", stage,
                 cudaGetErrorString(status));
    std::exit(1);
  }
  require(cudaMemset(device_values, 0, 4 * sizeof(int)) == cudaSuccess,
          "cudaMemset failed");
  add_one<<<1, 4>>>(device_values);
  require(cudaGetLastError() == cudaSuccess, "kernel launch failed");
  int results[4] = {};
  require(cudaMemcpy(results, device_values, sizeof(results),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          "cudaMemcpy to host failed");
  for (int i = 0; i < 4; ++i) {
    require(results[i] == 1, "kernel produced wrong results");
  }
  require(cudaFree(device_values) == cudaSuccess, "cudaFree failed");
}

int main() {
  int count = 0;
  require(cudaGetDeviceCount(&count) == cudaSuccess && count > 0,
          "cudaGetDeviceCount failed");

  // NVIDIA's libcudart keeps its per-device runtime state in the driver's
  // context-local storage, so a reset that leaves that state behind is one the
  // runtime never notices: it reuses the destroyed primary context and every
  // allocation afterwards fails with cudaErrorContextIsDestroyed.
  for (int round = 0; round < 3; ++round) {
    round_trip("before reset");
    require(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
    round_trip("after reset");
  }

  if (count > 1) {
    require(cudaSetDevice(1) == cudaSuccess, "cudaSetDevice(1) failed");
    round_trip("second device");
    require(cudaDeviceReset() == cudaSuccess,
            "cudaDeviceReset on the second device failed");
    round_trip("second device after reset");
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
    round_trip("first device after resetting the second");
    require(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
  }

  // The driver-level equivalent: the last release destroys the primary
  // context, and the next retain must hand back one that can allocate.
  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  CUdevice device = 0;
  require(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "cuDeviceGet failed");
  CUcontext context = nullptr;
  require(cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS,
          "cuDevicePrimaryCtxRetain failed");
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "cuCtxSetCurrent failed");
  require(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS,
          "cuDevicePrimaryCtxRelease failed");
  require(cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS,
          "cuDevicePrimaryCtxRetain after release failed");
  require(cuCtxSetCurrent(context) == CUDA_SUCCESS,
          "cuCtxSetCurrent after retain failed");
  CUdeviceptr pointer = 0;
  require(cuMemAlloc(&pointer, 1024) == CUDA_SUCCESS,
          "cuMemAlloc in the reretained primary context failed");
  require(cuMemFree(pointer) == CUDA_SUCCESS, "cuMemFree failed");
  require(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS,
          "final cuDevicePrimaryCtxRelease failed");

  std::puts("a device comes back after cudaDeviceReset");
  return 0;
}
