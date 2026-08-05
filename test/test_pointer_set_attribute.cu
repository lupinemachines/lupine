// Exercises cuPointerSetAttribute over the lupine wire.
//
// cuPointerSetAttribute takes a *pointer to* the value being set. A remoting
// layer that forwards the pointer value instead of the pointed-to bytes makes
// the server dereference an address from the client's address space, which
// kills the server process and takes the connection with it. This test sets
// CU_POINTER_ATTRIBUTE_SYNC_MEMOPS (the only settable pointer attribute) and
// then issues further driver calls to prove the connection survived.

#include <cuda.h>

#include <cstdio>
#include <cstring>

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

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (check((expr), #expr) != 0) {                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  CHECK(cuInit(0));

  // The check() helper above reports failures via cuGetErrorName, so confirm
  // the strings really come from a driver: names are enum spellings, strings
  // are descriptions rather than the same spelling, and codes no driver knows
  // report CUDA_ERROR_INVALID_VALUE with a NULL out-parameter.
  const char *name = nullptr;
  const char *desc = nullptr;
  CHECK(cuGetErrorName(CUDA_ERROR_DEVICE_UNAVAILABLE, &name));
  CHECK(cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &desc));
  if (std::strcmp(name, "CUDA_ERROR_DEVICE_UNAVAILABLE") != 0 ||
      std::strcmp(desc, "invalid argument") != 0) {
    std::fprintf(stderr, "error lookups report name=%s string=%s\n", name,
                 desc);
    return 1;
  }
  const char *unknown = name;
  if (cuGetErrorName(static_cast<CUresult>(12345), &unknown) !=
          CUDA_ERROR_INVALID_VALUE ||
      unknown != nullptr) {
    std::fprintf(stderr, "unknown error code was not rejected\n");
    return 1;
  }

  int device_count = 0;
  CHECK(cuDeviceGetCount(&device_count));
  if (device_count < 1) {
    std::fprintf(stderr, "no CUDA devices visible\n");
    return 2;
  }

  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  // cuCtxCreate is remapped to a 4-argument cuCtxCreate_v4 on newer CUDA, so
  // retain the primary context instead to stay signature-stable.
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));

  CUdeviceptr ptr = 0;
  CHECK(cuMemAlloc(&ptr, 4096));

  // The payload lives on the client's stack. If its address rather than its
  // contents reaches the server, the server dereferences a client stack
  // address and dies.
  int sync_memops = 1;
  CHECK(cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              ptr));

  // SYNC_MEMOPS is readable, so assert the value the driver now reports rather
  // than trusting the status code. A set that transmitted the wrong bytes can
  // still return CUDA_SUCCESS, so the status alone does not prove the payload
  // crossed the wire intact.
  int observed = -1;
  CHECK(cuPointerGetAttribute(&observed, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              ptr));
  if (observed != 1) {
    std::fprintf(stderr, "SYNC_MEMOPS read back as %d after setting 1\n",
                 observed);
    return 1;
  }

  // Prove the connection is still alive after the set.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
  if (total_bytes == 0) {
    std::fprintf(stderr, "cuMemGetInfo reported zero total memory\n");
    return 1;
  }

  // Round-trip real data too, so a dead-but-not-yet-noticed connection fails.
  unsigned int host_value = 0xabcd1234u;
  unsigned int read_back = 0;
  CHECK(cuMemcpyHtoD(ptr, &host_value, sizeof(host_value)));
  CHECK(cuMemcpyDtoH(&read_back, ptr, sizeof(read_back)));
  if (read_back != host_value) {
    std::fprintf(stderr, "memcpy round trip mismatch: got 0x%x want 0x%x\n",
                 read_back, host_value);
    return 1;
  }

  // Unsetting the attribute must work too, and must be observable. Setting a
  // different value and seeing it reflected rules out a stale or constant
  // readback masking a broken payload.
  sync_memops = 0;
  CHECK(cuPointerSetAttribute(&sync_memops, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              ptr));
  observed = -1;
  CHECK(cuPointerGetAttribute(&observed, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              ptr));
  if (observed != 0) {
    std::fprintf(stderr, "SYNC_MEMOPS read back as %d after setting 0\n",
                 observed);
    return 1;
  }
  CHECK(cuMemGetInfo(&free_bytes, &total_bytes));

  // An attribute that cannot be set must fail cleanly rather than send garbage
  // over the wire, and must leave the connection usable.
  CUresult bad = cuPointerSetAttribute(&sync_memops,
                                       CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
  if (bad == CUDA_SUCCESS) {
    std::fprintf(stderr, "cuPointerSetAttribute unexpectedly succeeded for a "
                         "non-settable attribute\n");
    return 1;
  }
  if (bad == CUDA_ERROR_DEVICE_UNAVAILABLE) {
    std::fprintf(stderr, "cuPointerSetAttribute killed the connection for a "
                         "non-settable attribute\n");
    return 1;
  }
  CHECK(cuMemGetInfo(&free_bytes, &total_bytes));

  CHECK(cuMemFree(ptr));
  CHECK(cuDevicePrimaryCtxRelease(device));

  std::printf("cuPointerSetAttribute ok, connection alive (total %zu bytes)\n",
              total_bytes);
  return 0;
}
