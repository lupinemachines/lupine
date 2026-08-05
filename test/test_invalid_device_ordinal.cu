// An out-of-range CUdevice ordinal must be rejected with
// CUDA_ERROR_INVALID_DEVICE, not CUDA_ERROR_DEVICE_UNAVAILABLE.

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

static int expect_invalid_device(CUresult status, const char *call) {
  if (status == CUDA_ERROR_INVALID_DEVICE) {
    return 0;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s returned %d (%s), want CUDA_ERROR_INVALID_DEVICE\n",
               call, static_cast<int>(status),
               name == nullptr ? "unknown" : name);
  return 1;
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
  if (check(cuDeviceGet(&device, 0), "cuDeviceGet") != 0) {
    return 1;
  }

  const CUdevice bogus = static_cast<CUdevice>(device_count + 99);
  CUdevice unused = 0;
  char name[64] = {0};
  int value = 0;
  size_t bytes = 0;
  CUcontext bogus_context = nullptr;
  unsigned int flags = 0;
  int active = 0;
  int can_access = -1;
  if (expect_invalid_device(cuDeviceGet(&unused, device_count + 99),
                            "cuDeviceGet(bogus)") != 0 ||
      expect_invalid_device(cuDeviceGetName(name, sizeof(name), bogus),
                            "cuDeviceGetName(bogus)") != 0 ||
      expect_invalid_device(
          cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, bogus),
          "cuDeviceGetAttribute(bogus)") != 0 ||
      expect_invalid_device(cuDeviceTotalMem(&bytes, bogus),
                            "cuDeviceTotalMem(bogus)") != 0 ||
      expect_invalid_device(cuDevicePrimaryCtxRetain(&bogus_context, bogus),
                            "cuDevicePrimaryCtxRetain(bogus)") != 0 ||
      expect_invalid_device(cuDevicePrimaryCtxGetState(bogus, &flags, &active),
                            "cuDevicePrimaryCtxGetState(bogus)") != 0 ||
      expect_invalid_device(cuDeviceCanAccessPeer(&can_access, device, bogus),
                            "cuDeviceCanAccessPeer(bogus)") != 0) {
    return 1;
  }

  CUcontext context = nullptr;
  CUdeviceptr ptr = 0;
  unsigned int host_value = 0xfeedface;
  unsigned int read_back = 0;
  if (check(cuDeviceGetName(name, sizeof(name), device), "cuDeviceGetName") !=
          0 ||
      check(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device),
            "cuDeviceGetAttribute") != 0 ||
      check(cuDevicePrimaryCtxRetain(&context, device),
            "cuDevicePrimaryCtxRetain") != 0 ||
      check(cuCtxSetCurrent(context), "cuCtxSetCurrent") != 0 ||
      check(cuMemAlloc(&ptr, sizeof(unsigned int)), "cuMemAlloc") != 0 ||
      check(cuMemcpyHtoD(ptr, &host_value, sizeof(host_value)),
            "cuMemcpyHtoD") != 0 ||
      check(cuMemcpyDtoH(&read_back, ptr, sizeof(read_back)), "cuMemcpyDtoH") !=
          0) {
    return 1;
  }
  if (read_back != host_value) {
    std::fprintf(stderr, "memcpy round trip mismatch: got 0x%x want 0x%x\n",
                 read_back, host_value);
    return 1;
  }
  if (check(cuMemFree(ptr), "cuMemFree") != 0 ||
      check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease") !=
          0) {
    return 1;
  }

  std::printf("invalid device ordinals rejected, connection alive (%s)\n",
              name);
  return 0;
}
