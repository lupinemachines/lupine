#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdio>

static int check_cuda(cudaError_t status, const char *call) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %d %s\n", call, static_cast<int>(status),
                 cudaGetErrorString(status));
    return 1;
  }
  return 0;
}

static int check_cublas(cublasStatus_t status, const char *call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(status));
    return 1;
  }
  return 0;
}

int main() {
  int count = 0;
  if (check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount") != 0) {
    return 1;
  }
  std::printf("cudaGetDeviceCount count=%d\n", count);
  if (count < 1) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 2;
  }

  cublasHandle_t handle = nullptr;
  if (check_cublas(cublasCreate(&handle), "cublasCreate") != 0) {
    return 3;
  }
  std::printf("cublasCreate handle=%p\n", static_cast<void *>(handle));

  int version = 0;
  if (check_cublas(cublasGetVersion(handle, &version), "cublasGetVersion") !=
      0) {
    return 4;
  }
  std::printf("cublasGetVersion version=%d\n", version);

  if (check_cublas(cublasDestroy(handle), "cublasDestroy") != 0) {
    return 5;
  }
  return 0;
}
