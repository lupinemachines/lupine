// Checks mapped host memory emulation against the driver's registration
// semantics: unaligned registration, overlapping re-registration, reported
// flags, unregister of allocated memory, and zero byte allocation.

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

static bool cu_ok(CUresult status, const char *call) {
  if (status == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s failed: %d (%s)\n", call, static_cast<int>(status),
               name == nullptr ? "unknown" : name);
  return false;
}

static bool cu_is(CUresult status, CUresult expected, const char *call) {
  if (status == expected) {
    return true;
  }
  std::fprintf(stderr, "%s returned %d, expected %d\n", call,
               static_cast<int>(status), static_cast<int>(expected));
  return false;
}

int main() {
  if (!cu_ok(cuInit(0), "cuInit")) {
    return 1;
  }

  int device_count = 0;
  if (!cu_ok(cuDeviceGetCount(&device_count), "cuDeviceGetCount")) {
    return 1;
  }
  if (device_count < 1) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 2;
  }

  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!cu_ok(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !cu_ok(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !cu_ok(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return 1;
  }

  long page_size_value = sysconf(_SC_PAGESIZE);
  size_t page_size =
      page_size_value > 0 ? static_cast<size_t>(page_size_value) : 4096;

  unsigned char *block = static_cast<unsigned char *>(
      std::aligned_alloc(page_size, page_size * 4));
  if (block == nullptr) {
    std::fprintf(stderr, "aligned_alloc failed\n");
    return 1;
  }
  // The mprotect-based mirror cannot track pages the registration does not own
  // outright, so an unaligned DEVICEMAP registration is rejected rather than
  // silently protecting the caller's neighbouring memory.
  unsigned char *unaligned = block + 64;
  if (!cu_is(cuMemHostRegister(unaligned, page_size * 2 + 17,
                               CU_MEMHOSTREGISTER_DEVICEMAP),
             CUDA_ERROR_INVALID_VALUE, "unaligned cuMemHostRegister")) {
    return 1;
  }

  unsigned char *aligned = block + page_size;
  if (!cu_ok(cuMemHostRegister(aligned, page_size, CU_MEMHOSTREGISTER_DEVICEMAP),
             "cuMemHostRegister")) {
    return 1;
  }
  if (!cu_is(cuMemHostRegister(aligned, page_size,
                               CU_MEMHOSTREGISTER_DEVICEMAP),
             CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
             "overlapping cuMemHostRegister")) {
    return 1;
  }

  CUdeviceptr mapped = 0;
  if (!cu_ok(cuMemHostGetDevicePointer(&mapped, aligned, 0),
             "cuMemHostGetDevicePointer")) {
    return 1;
  }
  if (mapped == 0) {
    std::fprintf(stderr, "cuMemHostGetDevicePointer returned a null pointer\n");
    return 1;
  }

  if (!cu_ok(cuMemHostUnregister(aligned), "cuMemHostUnregister")) {
    return 1;
  }
  std::free(block);

  void *pinned = nullptr;
  unsigned int flags = 0;
  if (!cu_ok(cuMemHostAlloc(&pinned, 4096, 0), "cuMemHostAlloc") ||
      !cu_ok(cuMemHostGetFlags(&flags, pinned), "cuMemHostGetFlags")) {
    return 1;
  }
  if ((flags & CU_MEMHOSTALLOC_DEVICEMAP) == 0) {
    std::fprintf(stderr,
                 "cuMemHostGetFlags reported 0x%x, expected DEVICEMAP\n",
                 flags);
    return 1;
  }

  if (!cu_is(cuMemHostUnregister(pinned), CUDA_ERROR_INVALID_VALUE,
             "cuMemHostUnregister of a cuMemHostAlloc pointer")) {
    return 1;
  }
  if (!cu_ok(cuMemFreeHost(pinned), "cuMemFreeHost")) {
    return 1;
  }

  void *empty = nullptr;
  if (!cu_ok(cuMemHostAlloc(&empty, 0, 0), "cuMemHostAlloc(0)") ||
      !cu_ok(cuMemFreeHost(empty), "cuMemFreeHost(0)") ||
      !cu_ok(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }

  std::printf("mapped host registration semantics OK\n");
  return 0;
}
