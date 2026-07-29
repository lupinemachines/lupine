// Exercises driver APIs that take a const pointer to a caller-owned struct or
// array. These used to send the client's pointer value over the wire, so the
// server dereferenced a client address and crashed the connection.
// Auto-discovered by test/run_custom_tests.sh via the test_*.cu glob.
#include <cuda.h>

#include <stdio.h>

static int check(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    printf("FAIL: %s -> %d (%s)\n", what, (int)result, name ? name : "?");
    return 1;
  }
  return 0;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (check(cuInit(0), "cuInit") || check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain") ||
      check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }

  // cuMemPoolCreate: const CUmemPoolProps *
  CUmemoryPool pool = nullptr;
  CUmemPoolProps props = {};
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;
  if (check(cuMemPoolCreate(&pool, &props), "cuMemPoolCreate")) {
    return 1;
  }
  printf("cuMemPoolCreate ok\n");

  // cuMemPoolSetAccess: const CUmemAccessDesc * plus a trailing count, so it
  // also covers the length-before-array ordering on the wire.
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (check(cuMemPoolSetAccess(pool, &access, 1), "cuMemPoolSetAccess")) {
    return 1;
  }
  printf("cuMemPoolSetAccess ok\n");

  // The connection must still be healthy: a crashed server would fail here.
  size_t free_bytes = 0, total_bytes = 0;
  if (check(cuMemGetInfo(&free_bytes, &total_bytes), "cuMemGetInfo after pool calls")) {
    return 1;
  }
  if (total_bytes == 0) {
    printf("FAIL: cuMemGetInfo reported no device memory\n");
    return 1;
  }

  if (check(cuMemPoolDestroy(pool), "cuMemPoolDestroy") ||
      check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }

  printf("PASS: const struct and array parameters survive the round trip\n");
  return 0;
}
