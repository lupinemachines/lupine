// Checks mapped host memory emulation against the driver's registration
// semantics: independent same-page registrations, unaligned registration,
// overlapping re-registration, RPC writes into registered memory, direct device
// access, reported flags, unregister of allocated memory, and zero byte
// allocation.

#include <cuda.h>
#include <cuda_runtime.h>

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

__global__ void increment_byte(unsigned char *value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++*value;
  }
}

__global__ void increment_registered_buffers(unsigned char *a, unsigned char *b,
                                             unsigned char *c,
                                             unsigned char *d) {
  ++*a;
  ++*b;
  ++*c;
  ++*d;
}

static bool check_same_page_registrations(unsigned char *block,
                                          size_t page_size,
                                          unsigned int flags) {
  // cuDSS hybrid execution registers several small, adjacent host buffers.
  // They share a page but have independent registrations and lifetimes.
  const size_t offsets[] = {64, 144, 192, 224};
  const size_t sizes[] = {80, 20, 20, 72};
  CUdeviceptr mapped[4] = {};
  for (size_t i = 0; i < page_size; ++i) {
    block[i] = 41;
  }
  for (int i = 0; i < 4; ++i) {
    if (!cu_ok(cuMemHostRegister(block + offsets[i], sizes[i], flags),
               "same-page cuMemHostRegister")) {
      return false;
    }
  }
  if (!cu_is(cuMemHostRegister(block + 64, 80, flags),
             CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
             "duplicate registration") ||
      !cu_is(cuMemHostRegister(block + 70, 20, flags),
             CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
             "overlap with preceding registration") ||
      !cu_is(cuMemHostRegister(block + 32, 40, flags),
             CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
             "overlap with following registration")) {
    return false;
  }

  // Removing the middle range must leave its neighbors registered, and the
  // freed range must be available for a new registration.
  if (!cu_ok(cuMemHostUnregister(block + offsets[1]),
             "unregister middle same-page buffer") ||
      !cu_ok(cuMemHostRegister(block + offsets[1], sizes[1], flags),
             "re-register middle same-page buffer")) {
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    unsigned int observed_flags = 0;
    if (!cu_ok(cuMemHostGetFlags(&observed_flags, block + offsets[i]),
               "same-page cuMemHostGetFlags")) {
      return false;
    }
    if ((observed_flags & flags) != flags) {
      std::fprintf(stderr, "same-page flags: got %u, expected bits %u\n",
                   observed_flags, flags);
      return false;
    }
    if ((flags & CU_MEMHOSTREGISTER_DEVICEMAP) != 0) {
      // Use an interior pointer to check that offsets are relative to each
      // registration, not the containing page or a neighboring buffer.
      if (!cu_ok(
              cuMemHostGetDevicePointer(&mapped[i], block + offsets[i] + 1, 0),
              "same-page cuMemHostGetDevicePointer")) {
        return false;
      }
    }
  }
  if ((flags & CU_MEMHOSTREGISTER_DEVICEMAP) != 0) {
    increment_registered_buffers<<<1, 1>>>(
        reinterpret_cast<unsigned char *>(mapped[0]),
        reinterpret_cast<unsigned char *>(mapped[1]),
        reinterpret_cast<unsigned char *>(mapped[2]),
        reinterpret_cast<unsigned char *>(mapped[3]));
  }
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return false;
  }
  for (size_t i = 0; i < page_size; ++i) {
    unsigned char expected = 41;
    if ((flags & CU_MEMHOSTREGISTER_DEVICEMAP) != 0) {
      for (size_t offset : offsets) {
        if (i == offset + 1) {
          expected = 42;
        }
      }
    }
    if (block[i] != expected) {
      std::fprintf(stderr, "same-page byte %zu: got %u, expected %u\n", i,
                   static_cast<unsigned int>(block[i]),
                   static_cast<unsigned int>(expected));
      return false;
    }
  }
  for (int i = 3; i >= 0; --i) {
    if (!cu_ok(cuMemHostUnregister(block + offsets[i]),
               "same-page cuMemHostUnregister")) {
      return false;
    }
  }
  return true;
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
  if (!check_same_page_registrations(block, page_size, 0) ||
      !check_same_page_registrations(block, page_size,
                                     CU_MEMHOSTREGISTER_DEVICEMAP)) {
    return 1;
  }
  // A sub-page registration (NPP registers a stack-resident label count this
  // way) shares its page with unrelated application bytes. The transfer may
  // only carry the registered range: writing the rest of the page back from
  // the server restores whatever the application had there at flush time.
  unsigned char *interior = block + 64;
  for (size_t i = 0; i < page_size; ++i) {
    block[i] = 0xA5;
  }
  if (!cu_ok(cuMemHostRegister(interior, 4,
                               CU_MEMHOSTREGISTER_PORTABLE |
                                   CU_MEMHOSTREGISTER_DEVICEMAP),
             "sub-page cuMemHostRegister")) {
    return 1;
  }
  CUdeviceptr interior_mapped = 0;
  if (!cu_ok(cuMemHostGetDevicePointer(&interior_mapped, interior, 0),
             "sub-page cuMemHostGetDevicePointer")) {
    return 1;
  }
  interior[0] = 41;
  increment_byte<<<1, 1>>>(reinterpret_cast<unsigned char *>(interior_mapped));
  for (size_t i = 0; i < page_size; ++i) {
    if (i < 64 || i >= 68) {
      block[i] = 0x5A;
    }
  }
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess || interior[0] != 42) {
    std::fprintf(stderr, "sub-page device access produced %u, expected 42\n",
                 static_cast<unsigned int>(interior[0]));
    return 1;
  }
  for (size_t i = 0; i < page_size; ++i) {
    if (i >= 64 && i < 68) {
      continue;
    }
    if (block[i] != 0x5A) {
      std::fprintf(stderr,
                   "sub-page registration clobbered page byte %zu: %u\n", i,
                   static_cast<unsigned int>(block[i]));
      return 1;
    }
  }
  if (!cu_ok(cuMemHostUnregister(interior), "sub-page cuMemHostUnregister")) {
    return 1;
  }

  unsigned char *unaligned = block + 64;
  if (!cu_ok(cuMemHostRegister(unaligned, page_size * 2 + 17,
                               CU_MEMHOSTREGISTER_PORTABLE |
                                   CU_MEMHOSTREGISTER_DEVICEMAP),
             "unaligned cuMemHostRegister")) {
    return 1;
  }
  if (!cu_is(cuMemHostRegister(block + page_size, page_size,
                               CU_MEMHOSTREGISTER_DEVICEMAP),
             CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
             "overlapping unaligned cuMemHostRegister")) {
    return 1;
  }

  CUdeviceptr unaligned_mapped = 0;
  if (!cu_ok(cuMemHostGetDevicePointer(&unaligned_mapped, unaligned, 0),
             "unaligned cuMemHostGetDevicePointer")) {
    return 1;
  }
  unaligned[0] = 41;
  increment_byte<<<1, 1>>>(reinterpret_cast<unsigned char *>(unaligned_mapped));
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess || unaligned[0] != 42) {
    std::fprintf(stderr,
                 "direct unaligned device access produced %u, expected 42\n",
                 static_cast<unsigned int>(unaligned[0]));
    return 1;
  }
  if (!cu_is(cuMemHostUnregister(unaligned + page_size),
             CUDA_ERROR_INVALID_VALUE,
             "interior unaligned cuMemHostUnregister")) {
    return 1;
  }
  if (!cu_ok(cuMemHostUnregister(unaligned), "unaligned cuMemHostUnregister")) {
    return 1;
  }

  unsigned char *aligned = block + page_size;
  if (!cu_ok(
          cuMemHostRegister(aligned, page_size, CU_MEMHOSTREGISTER_DEVICEMAP),
          "cuMemHostRegister")) {
    return 1;
  }
  if (!cu_is(
          cuMemHostRegister(aligned, page_size, CU_MEMHOSTREGISTER_DEVICEMAP),
          CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
          "overlapping cuMemHostRegister")) {
    return 1;
  }

  CUdeviceptr scratch = 0;
  unsigned char expected = 37;
  if (!cu_ok(cuMemAlloc(&scratch, 1), "cuMemAlloc") ||
      !cu_ok(cuMemcpyHtoD(scratch, &expected, 1), "cuMemcpyHtoD") ||
      !cu_ok(cuMemcpyDtoH(aligned, scratch, 1),
             "cuMemcpyDtoH into registered memory") ||
      aligned[0] != expected) {
    std::fprintf(stderr, "RPC write into registered memory was not visible\n");
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
  aligned[0] = 41;
  increment_byte<<<1, 1>>>(reinterpret_cast<unsigned char *>(mapped));
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess || aligned[0] != 42) {
    std::fprintf(stderr, "direct device access produced %u, expected 42\n",
                 static_cast<unsigned int>(aligned[0]));
    return 1;
  }

  if (!cu_ok(cuMemFree(scratch), "cuMemFree") ||
      !cu_ok(cuMemHostUnregister(aligned), "cuMemHostUnregister")) {
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
    std::fprintf(
        stderr, "cuMemHostGetFlags reported 0x%x, expected DEVICEMAP\n", flags);
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
