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
  CUresult r = cuLibraryLoadData(&lib, kGlobalPtx, nullptr, nullptr, 0, nullptr,
                                 nullptr, 0);
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

  CUmodule mod = nullptr;
  r = cuLibraryGetModule(&mod, lib);
  if (r != CUDA_SUCCESS || mod == nullptr) {
    printf("RESULT: FAIL cuLibraryGetModule %s\n", cn(r));
    return 1;
  }
  r = cuModuleGetGlobal(nullptr, nullptr, mod, "counter");
  if (r != CUDA_ERROR_INVALID_VALUE) {
    printf("RESULT: FAIL cuModuleGetGlobal(null, null) %s\n", cn(r));
    return 1;
  }
  dptr = 0xDEAD;
  bytes = 0xBEEF;
  r = cuModuleGetGlobal(&dptr, &bytes, mod, "missing");
  if (r != CUDA_ERROR_NOT_FOUND || dptr != 0 || bytes != 0) {
    printf("RESULT: FAIL cuModuleGetGlobal(missing) %s dptr=0x%llx "
           "bytes=0x%zx\n",
           cn(r), (unsigned long long)dptr, bytes);
    return 1;
  }
  dptr = 0xDEAD;
  bytes = 0xBEEF;
  r = cuModuleGetGlobal(&dptr, &bytes, mod, nullptr);
  if (r != CUDA_ERROR_INVALID_VALUE || dptr != 0 || bytes != 0) {
    printf("RESULT: FAIL cuModuleGetGlobal(null name) %s dptr=0x%llx "
           "bytes=0x%zx\n",
           cn(r), (unsigned long long)dptr, bytes);
    return 1;
  }

  int min_grid = 12345;
  int block_size = 67890;
  r = cuOccupancyMaxPotentialBlockSize(&min_grid, &block_size, nullptr, nullptr,
                                       0, 0);
  if (r != CUDA_ERROR_INVALID_HANDLE || min_grid != 12345 ||
      block_size != 67890) {
    printf(
        "RESULT: FAIL cuOccupancyMaxPotentialBlockSize %s grid=%d block=%d\n",
        cn(r), min_grid, block_size);
    return 1;
  }
  r = cuOccupancyMaxPotentialBlockSizeWithFlags(&min_grid, &block_size, nullptr,
                                                nullptr, 0, 0, 0);
  if (r != CUDA_ERROR_INVALID_HANDLE || min_grid != 12345 ||
      block_size != 67890) {
    printf("RESULT: FAIL cuOccupancyMaxPotentialBlockSizeWithFlags %s grid=%d "
           "block=%d\n",
           cn(r), min_grid, block_size);
    return 1;
  }

  cuLibraryUnload(lib);
  cuDevicePrimaryCtxRelease(dev);
  printf("RESULT: OK\n");
  return 0;
}
#endif
