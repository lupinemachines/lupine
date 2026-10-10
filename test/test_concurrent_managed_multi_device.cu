#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>

namespace {

constexpr int kDevices = 2;

int check(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), cudaGetErrorString(status));
  return 1;
}

int check(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) {
    return 0;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), name == nullptr ? "unknown" : name);
  return 1;
}

int expect(bool condition, const char *message) {
  if (condition) {
    return 0;
  }
  std::fprintf(stderr, "FAIL: %s\n", message);
  return 1;
}

} // namespace

int main() {
  int device_count = 0;
  if (check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") != 0) {
    return 1;
  }
  if (device_count < kDevices) {
    std::printf("SKIP: need two CUDA devices, found %d\n", device_count);
    return 0;
  }

  for (int ordinal = 0; ordinal < kDevices; ++ordinal) {
    cudaDeviceProp properties = {};
    int runtime_attribute = 0;
    CUdevice device = 0;
    int driver_attribute = 0;
    if (check(cudaGetDeviceProperties(&properties, ordinal),
              "cudaGetDeviceProperties") != 0 ||
        check(cudaDeviceGetAttribute(&runtime_attribute,
                                     cudaDevAttrConcurrentManagedAccess,
                                     ordinal),
              "cudaDeviceGetAttribute") != 0 ||
        check(cuDeviceGet(&device, ordinal), "cuDeviceGet") != 0 ||
        check(cuDeviceGetAttribute(
                  &driver_attribute,
                  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device),
              "cuDeviceGetAttribute") != 0 ||
        expect(properties.concurrentManagedAccess == 1,
               "device properties hid concurrent managed access") != 0 ||
        expect(runtime_attribute == 1,
               "runtime attribute hid concurrent managed access") != 0 ||
        expect(driver_attribute == 1,
               "driver attribute hid concurrent managed access") != 0) {
      return 2;
    }
  }

  std::puts("two devices advertise concurrent managed access through one "
            "server");
  return 0;
}
