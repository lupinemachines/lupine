// An attribute enum the shim does not know must fail the way the real driver
// fails it: CUDA_ERROR_INVALID_VALUE, not CUDA_ERROR_NOT_SUPPORTED.

#include <cuda.h>

#include <cstdio>

static int check(CUresult status, const char *call) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "%s failed: %d (%s)\n", call, static_cast<int>(status),
                 name == nullptr ? "unknown" : name);
    return 1;
  }
  return 0;
}

static int expect_invalid_value(CUresult status, const char *call) {
  if (status != CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr, "%s returned %d, want CUDA_ERROR_INVALID_VALUE (%d)\n",
                 call, static_cast<int>(status),
                 static_cast<int>(CUDA_ERROR_INVALID_VALUE));
    return 1;
  }
  return 0;
}

int main() {
  int device_count = 0;
  if (check(cuInit(0), "cuInit") != 0 ||
      check(cuDeviceGetCount(&device_count), "cuDeviceGetCount") != 0) {
    return 1;
  }
  if (device_count < 1) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 2;
  }

  CUdevice device = 0;
  CUcontext context = nullptr;
  CUdeviceptr ptr = 0;
  CUmemoryPool pool = nullptr;
  if (check(cuDeviceGet(&device, 0), "cuDeviceGet") != 0 ||
      check(cuDevicePrimaryCtxRetain(&context, device),
            "cuDevicePrimaryCtxRetain") != 0 ||
      check(cuCtxSetCurrent(context), "cuCtxSetCurrent") != 0 ||
      check(cuMemAlloc(&ptr, 4096), "cuMemAlloc") != 0 ||
      check(cuDeviceGetMemPool(&pool, device), "cuDeviceGetMemPool") != 0) {
    return 1;
  }

  unsigned long long scratch = 0;
  CUpointer_attribute attributes[] = {static_cast<CUpointer_attribute>(9999)};
  void *data[] = {&scratch};
  if (expect_invalid_value(
          cuPointerGetAttribute(&scratch,
                                static_cast<CUpointer_attribute>(9999), ptr),
          "cuPointerGetAttribute(9999)") != 0 ||
      expect_invalid_value(cuPointerGetAttributes(1, attributes, data, ptr),
                           "cuPointerGetAttributes(9999)") != 0 ||
      expect_invalid_value(
          cuMemPoolGetAttribute(pool, static_cast<CUmemPool_attribute>(9999),
                                &scratch),
          "cuMemPoolGetAttribute(9999)") != 0 ||
      expect_invalid_value(
          cuMemPoolSetAttribute(pool, static_cast<CUmemPool_attribute>(9999),
                                &scratch),
          "cuMemPoolSetAttribute(9999)") != 0) {
    return 1;
  }

  int sync_memops = -1;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (check(cuPointerGetAttribute(&sync_memops,
                                  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, ptr),
            "cuPointerGetAttribute(SYNC_MEMOPS)") != 0 ||
      check(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo") != 0 ||
      check(cuMemFree(ptr), "cuMemFree") != 0 ||
      check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease") !=
          0) {
    return 1;
  }

  std::printf("unknown attribute enums report INVALID_VALUE, connection alive "
              "(total %zu bytes)\n",
              total_bytes);
  return 0;
}
