// A failed call that leaves its optional out-parameters alone natively must
// leave them alone through lupine too, not fill them from the server.
#include <cuda.h>
#include <stdio.h>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000
int main() {
  printf("SKIP: cuLibraryGetGlobal requires CUDA 12.0 or newer\n");
  return 0;
}
#else

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
  CUlibrary lib = nullptr;
  CUresult r = cuLibraryLoadData(&lib, kGlobalPtx, nullptr, nullptr, 0,
                                 nullptr, nullptr, 0);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: ERROR cuLibraryLoadData %s\n", cn(r));
    return 2;
  }

  CUdeviceptr dptr = 0xDEAD;
  size_t bytes = 0xBEEF;
  r = cuLibraryGetGlobal(&dptr, &bytes, lib, "missing");
  if (r != CUDA_ERROR_NOT_FOUND || dptr != 0xDEAD || bytes != 0xBEEF) {
    printf("RESULT: FAIL cuLibraryGetGlobal %s dptr=0x%llx bytes=0x%zx\n",
           cn(r), (unsigned long long)dptr, bytes);
    return 1;
  }
  r = cuLibraryGetManaged(&dptr, &bytes, lib, "missing");
  if (r != CUDA_ERROR_NOT_FOUND || dptr != 0xDEAD || bytes != 0xBEEF) {
    printf("RESULT: FAIL cuLibraryGetManaged %s dptr=0x%llx bytes=0x%zx\n",
           cn(r), (unsigned long long)dptr, bytes);
    return 1;
  }

  r = cuLibraryGetGlobal(nullptr, &bytes, lib, "counter");
  if (r != CUDA_SUCCESS || bytes != 16) {
    printf("RESULT: FAIL size-only cuLibraryGetGlobal %s bytes=%zu\n", cn(r),
           bytes);
    return 1;
  }
  r = cuLibraryGetGlobal(&dptr, &bytes, lib, "counter");
  if (r != CUDA_SUCCESS || dptr == 0xDEAD || dptr == 0 || bytes != 16) {
    printf("RESULT: FAIL cuLibraryGetGlobal %s dptr=0x%llx bytes=%zu\n", cn(r),
           (unsigned long long)dptr, bytes);
    return 1;
  }

  cuLibraryUnload(lib);
  cuDevicePrimaryCtxRelease(dev);
  printf("RESULT: OK\n");
  return 0;
}
#endif
