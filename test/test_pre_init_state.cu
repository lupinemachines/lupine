// Client-answered entry points must report CUDA_ERROR_NOT_INITIALIZED before
// cuInit. No device code here, so nothing initializes CUDA before main runs.
#include <cstdio>
#include <cuda.h>
#include <cudaProfiler.h>

static int failures = 0;

static void expect(const char *what, CUresult actual, CUresult expected) {
  if (actual != expected) {
    std::printf("FAIL: %s -> %d (expected %d)\n", what, static_cast<int>(actual),
                static_cast<int>(expected));
    ++failures;
  } else {
    std::printf("ok: %s -> %d\n", what, static_cast<int>(actual));
  }
}

int main() {
  CUcontext ctx = reinterpret_cast<CUcontext>(0xdeadbeefULL);
  CUdevice dev = 0;

  expect("cuCtxGetCurrent before cuInit", cuCtxGetCurrent(&ctx),
         CUDA_ERROR_NOT_INITIALIZED);
  if (ctx != reinterpret_cast<CUcontext>(0xdeadbeefULL)) {
    std::printf("FAIL: cuCtxGetCurrent wrote its out parameter before cuInit\n");
    ++failures;
  }
  expect("cuCtxGetDevice before cuInit", cuCtxGetDevice(&dev),
         CUDA_ERROR_NOT_INITIALIZED);
  expect("cuCtxSetCurrent before cuInit", cuCtxSetCurrent(nullptr),
         CUDA_ERROR_NOT_INITIALIZED);
  expect("cuCtxPushCurrent before cuInit", cuCtxPushCurrent(nullptr),
         CUDA_ERROR_NOT_INITIALIZED);
  expect("cuCtxPopCurrent before cuInit", cuCtxPopCurrent(&ctx),
         CUDA_ERROR_NOT_INITIALIZED);
  expect("cuProfilerStart before cuInit", cuProfilerStart(),
         CUDA_ERROR_NOT_INITIALIZED);
  expect("cuProfilerStop before cuInit", cuProfilerStop(),
         CUDA_ERROR_NOT_INITIALIZED);

  expect("cuInit", cuInit(0), CUDA_SUCCESS);

  expect("cuDeviceGet after cuInit", cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  expect("cuCtxGetCurrent after cuInit", cuCtxGetCurrent(&ctx), CUDA_SUCCESS);
  CUcontext primary = nullptr;
  expect("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&primary, dev),
         CUDA_SUCCESS);
  expect("cuCtxSetCurrent after cuInit", cuCtxSetCurrent(primary),
         CUDA_SUCCESS);
  expect("cuCtxGetDevice after cuInit", cuCtxGetDevice(&dev), CUDA_SUCCESS);
  expect("cuProfilerStart after cuInit", cuProfilerStart(), CUDA_SUCCESS);
  expect("cuProfilerStop after cuInit", cuProfilerStop(), CUDA_SUCCESS);
  expect("cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease(dev),
         CUDA_SUCCESS);

  std::printf(failures == 0 ? "PASSED\n" : "FAILED\n");
  return failures == 0 ? 0 : 1;
}
