// Regression test verifying that cuMemFreeHost rejects cuMemHostRegister and
// managed allocations with CUDA_ERROR_INVALID_VALUE without retiring them.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

__device__ __managed__ unsigned int g_managed_value = 0x13572468U;

namespace {

bool expect_status(CUresult got, CUresult want, const char *what) {
  if (got == want) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: got %d, want %d\n", what,
               static_cast<int>(got), static_cast<int>(want));
  return false;
}

int test_device(int ordinal) {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (cuDeviceGet(&dev, ordinal) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: setup device %d\n", ordinal);
    return 1;
  }

  int failures = 0;

  // 1. cuMemFreeHost on a cuMemHostRegister buffer must fail with
  // CUDA_ERROR_INVALID_VALUE and leave the registration intact.
  void *reg_buf = aligned_alloc(4096, 4096);
  if (reg_buf == nullptr) {
    return 1;
  }
  std::memset(reg_buf, 0x2a, 4096);
  failures += !expect_status(
      cuMemHostRegister_v2(reg_buf, 4096, CU_MEMHOSTREGISTER_DEVICEMAP),
      CUDA_SUCCESS, "cuMemHostRegister_v2");
  failures += !expect_status(cuMemFreeHost(reg_buf), CUDA_ERROR_INVALID_VALUE,
                             "cuMemFreeHost(registered)");

  unsigned int flags = 0;
  failures += !expect_status(cuMemHostGetFlags(&flags, reg_buf), CUDA_SUCCESS,
                             "cuMemHostGetFlags after rejected cuMemFreeHost");
  if ((flags & CU_MEMHOSTALLOC_DEVICEMAP) == 0) {
    std::fprintf(stderr, "FAIL: expected DEVICEMAP flag on registered buffer\n");
    ++failures;
  }

  CUdeviceptr reg_dptr = 0;
  failures += !expect_status(cuMemHostGetDevicePointer_v2(&reg_dptr, reg_buf, 0),
                             CUDA_SUCCESS,
                             "cuMemHostGetDevicePointer_v2 after rejected free");
  failures += !expect_status(
      cuMemHostUnregister(static_cast<char *>(reg_buf) + 64),
      CUDA_ERROR_INVALID_VALUE, "cuMemHostUnregister(interior)");
  failures += !expect_status(cuMemHostUnregister(reg_buf), CUDA_SUCCESS,
                             "cuMemHostUnregister(registered)");
  failures += !expect_status(cuMemHostUnregister(reg_buf),
                             CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
                             "cuMemHostUnregister(already unregistered)");
  std::free(reg_buf);

  // 2. cuMemFreeHost on a cuMemAllocManaged allocation must fail with
  // CUDA_ERROR_INVALID_VALUE without unmapping or retiring the allocation.
  CUdeviceptr managed_ptr = 0;
  if (expect_status(
          cuMemAllocManaged(&managed_ptr, 4096, CU_MEM_ATTACH_GLOBAL),
          CUDA_SUCCESS, "cuMemAllocManaged")) {
    auto *host_view = reinterpret_cast<volatile unsigned int *>(managed_ptr);
    host_view[0] = 0xaabbccddU;
    failures += !expect_status(
        cuMemFreeHost(reinterpret_cast<void *>(managed_ptr)),
        CUDA_ERROR_INVALID_VALUE, "cuMemFreeHost(managed)");
    if (host_view[0] != 0xaabbccddU) {
      std::fprintf(stderr, "FAIL: managed host view corrupted after bad free\n");
      ++failures;
    }
    unsigned int readback = 0;
    failures += !expect_status(
        cuMemcpyDtoH_v2(&readback, managed_ptr, sizeof(readback)), CUDA_SUCCESS,
        "cuMemcpyDtoH_v2(managed after rejected cuMemFreeHost)");
    if (readback != 0xaabbccddU) {
      std::fprintf(stderr, "FAIL: managed DtoH got 0x%x, want 0xaabbccdd\n",
                   readback);
      ++failures;
    }
    failures += !expect_status(cuMemFree_v2(managed_ptr), CUDA_SUCCESS,
                               "cuMemFree_v2(managed)");
  } else {
    ++failures;
  }

  (void)cuDevicePrimaryCtxRelease(dev);
  return failures;
}

} // namespace

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: cuInit\n");
    return EXIT_FAILURE;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  int failures = 0;
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    failures += test_device(ordinal);
  }

  // 3. cuMemFreeHost on a __managed__ global must fail with
  // CUDA_ERROR_INVALID_VALUE.
  failures += !expect_status(cuMemFreeHost(&g_managed_value),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuMemFreeHost(__managed__ global)");
  if (g_managed_value != 0x13572468U) {
    std::fprintf(stderr, "FAIL: __managed__ global corrupted\n");
    ++failures;
  }

  if (failures != 0) {
    std::fprintf(stderr, "FAILED with %d error(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
