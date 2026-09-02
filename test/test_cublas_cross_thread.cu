#include <cublas_v2.h>
#include <cuda.h>

#include <cstdio>
#include <thread>

static bool check_cuda(CUresult status, const char *call) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s failed: %d %s\n", call, static_cast<int>(status),
               name == nullptr ? "" : name);
  return false;
}

static bool check_cublas(cublasStatus_t status, const char *call) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(status));
  return false;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!check_cuda(cuInit(0), "cuInit") ||
      !check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check_cuda(cuDevicePrimaryCtxRetain(&context, device),
                  "cuDevicePrimaryCtxRetain") ||
      !check_cuda(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }

  float a = 3.0F;
  float b = 4.0F;
  float c = 0.0F;
  CUdeviceptr device_a = 0;
  CUdeviceptr device_b = 0;
  CUdeviceptr device_c = 0;
  if (!check_cuda(cuMemAlloc(&device_a, sizeof(a)), "cuMemAlloc A") ||
      !check_cuda(cuMemAlloc(&device_b, sizeof(b)), "cuMemAlloc B") ||
      !check_cuda(cuMemAlloc(&device_c, sizeof(c)), "cuMemAlloc C") ||
      !check_cuda(cuMemcpyHtoD(device_a, &a, sizeof(a)), "cuMemcpyHtoD A") ||
      !check_cuda(cuMemcpyHtoD(device_b, &b, sizeof(b)), "cuMemcpyHtoD B")) {
    return 2;
  }

  cublasHandle_t handle = nullptr;
  if (!check_cublas(cublasCreate(&handle), "cublasCreate")) {
    return 3;
  }

  cublasStatus_t thread_status = CUBLAS_STATUS_INTERNAL_ERROR;
  std::thread worker([&] {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    thread_status =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha,
                    reinterpret_cast<const float *>(device_a), 1,
                    reinterpret_cast<const float *>(device_b), 1, &beta,
                    reinterpret_cast<float *>(device_c), 1);
  });
  worker.join();

  if (!check_cublas(thread_status, "cross-thread cublasSgemm") ||
      !check_cuda(cuCtxSynchronize(), "cuCtxSynchronize") ||
      !check_cuda(cuMemcpyDtoH(&c, device_c, sizeof(c)), "cuMemcpyDtoH C")) {
    return 4;
  }
  if (c != 12.0F) {
    std::fprintf(stderr, "unexpected result: %f\n", c);
    return 5;
  }

  if (!check_cublas(cublasDestroy(handle), "cublasDestroy") ||
      !check_cuda(cuMemFree(device_a), "cuMemFree A") ||
      !check_cuda(cuMemFree(device_b), "cuMemFree B") ||
      !check_cuda(cuMemFree(device_c), "cuMemFree C") ||
      !check_cuda(cuDevicePrimaryCtxRelease(device),
                  "cuDevicePrimaryCtxRelease")) {
    return 6;
  }
  return 0;
}
