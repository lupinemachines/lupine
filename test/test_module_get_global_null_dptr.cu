// Regression test: a size-only cuModuleGetGlobal (NULL dptr) must still drain
// the device pointer from the response, or the next response misparses.
#include <cuda.h>
#include <stdio.h>

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static const char kGlobalPtx[] = ".version 6.4\n"
                                 ".target sm_52\n"
                                 ".address_size 64\n"
                                 "\n"
                                 ".visible .global .align 4 .u32 counter[4];\n";

int main() {
  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }

  CUmodule mod = nullptr;
  CUresult r = cuModuleLoadData(&mod, kGlobalPtx);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: ERROR cuModuleLoadData %s\n", cn(r));
    return 2;
  }

  size_t bytes = 0;
  r = cuModuleGetGlobal(nullptr, &bytes, mod, "counter");
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL size-only cuModuleGetGlobal %s\n", cn(r));
    return 1;
  }
  if (bytes != 16) {
    printf("RESULT: FAIL size-only bytes %zu (want 16)\n", bytes);
    return 1;
  }

  CUdeviceptr dptr = 0;
  size_t bytes2 = 0;
  r = cuModuleGetGlobal(&dptr, &bytes2, mod, "counter");
  if (r != CUDA_SUCCESS || dptr == 0 || bytes2 != 16) {
    printf("RESULT: FAIL follow-up cuModuleGetGlobal %s dptr=%llu bytes=%zu\n",
           cn(r), (unsigned long long)dptr, bytes2);
    return 1;
  }

  unsigned int value = 0xa5a5a5a5u;
  r = cuMemcpyHtoD(dptr, &value, sizeof(value));
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuMemcpyHtoD %s\n", cn(r));
    return 1;
  }
  unsigned int readback = 0;
  r = cuMemcpyDtoH(&readback, dptr, sizeof(readback));
  if (r != CUDA_SUCCESS || readback != value) {
    printf("RESULT: FAIL cuMemcpyDtoH %s readback=0x%x\n", cn(r), readback);
    return 1;
  }

  cuModuleUnload(mod);
  cuDevicePrimaryCtxRelease(dev);
  printf("RESULT: OK\n");
  return 0;
}
