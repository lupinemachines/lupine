// Regression test: the driver's free/release entry points accept a null
// argument the way the real driver does. Probed on an RTX 4090 (driver
// 590.48.01): cuMemFreeHost(NULL) and cuMemFree(0) succeed, while
// cuMemHostUnregister(NULL) reports CUDA_ERROR_INVALID_VALUE.
#include <cuda.h>
#include <stdio.h>

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static int failures = 0;

static void expect(const char *what, CUresult got, CUresult want) {
  if (got != want) {
    printf("RESULT: FAIL %s = %d %s (want %d)\n", what, (int)got, cn(got),
           (int)want);
    failures++;
  }
}

int main() {
  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }

  expect("cuMemFreeHost(NULL)", cuMemFreeHost(nullptr), CUDA_SUCCESS);
  expect("cuMemFree(0)", cuMemFree(0), CUDA_SUCCESS);
  expect("cuMemHostUnregister(NULL)", cuMemHostUnregister(nullptr),
         CUDA_ERROR_INVALID_VALUE);

  // A real pinned allocation still frees after the null calls.
  void *host = nullptr;
  CUresult r = cuMemHostAlloc(&host, 4096, 0);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: ERROR cuMemHostAlloc %s\n", cn(r));
    return 2;
  }
  expect("cuMemFreeHost(host)", cuMemFreeHost(host), CUDA_SUCCESS);

  cuDevicePrimaryCtxRelease(dev);
  if (failures != 0) {
    return 1;
  }
  printf("RESULT: OK\n");
  return 0;
}
