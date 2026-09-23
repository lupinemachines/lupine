// Launches through the driver's parameter-buffer form, where every argument
// arrives packed in one blob instead of an array of pointers.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_DRIVER(call)                                                     \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void fill(float *out, float value, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = value;
  }
}

static const int kCount = 256;

static int check(const float *device, float want) {
  std::vector<float> host(kCount, 0.0f);
  CHECK_CUDA(cudaMemcpy(host.data(), device, kCount * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < kCount; ++i) {
    if (host[i] != want) {
      fprintf(stderr, "element %d is %f, want %f\n", i, host[i], want);
      return 1;
    }
  }
  return 0;
}

int main() {
  CHECK_CUDA(cudaSetDevice(0));
  float *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, kCount * sizeof(float)));
  CHECK_CUDA(cudaMemset(device, 0, kCount * sizeof(float)));

  cudaFunction_t fill_function = nullptr;
  CHECK_CUDA(cudaGetFuncBySymbol(&fill_function, (const void *)fill));
  struct {
    float *out;
    float value;
    int count;
  } args = {device, 3.5f, kCount};
  size_t args_size = sizeof(args);
  void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &args,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                   CU_LAUNCH_PARAM_END};
  CHECK_DRIVER(cuLaunchKernel((CUfunction)fill_function, kCount / 64, 1, 1, 64,
                              1, 1, 0, nullptr, nullptr, extra));
  CHECK_CUDA(cudaDeviceSynchronize());
  if (check(device, 3.5f)) {
    return 1;
  }
  printf("cuLaunchKernel parameter buffer: passed\n");

  args.value = 7.25f;
  CUlaunchConfig config = {};
  config.gridDimX = kCount / 64;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 64;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.hStream = nullptr;
  CHECK_DRIVER(
      cuLaunchKernelEx(&config, (CUfunction)fill_function, nullptr, extra));
  CHECK_CUDA(cudaDeviceSynchronize());
  if (check(device, 7.25f)) {
    return 1;
  }
  printf("cuLaunchKernelEx parameter buffer: passed\n");

  CHECK_CUDA(cudaFree(device));
  printf("parameter buffer launches: all checks passed\n");
  return 0;
}
