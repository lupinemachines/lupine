// cuDeviceGetP2PAttribute must validate in the driver's order: null value
// pointer, then devices, then attribute. Every case here uses one device.

#include <cuda.h>

#include <cstdio>

int main() {
  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuInit failed: %d\n", static_cast<int>(status));
    return 1;
  }

  int device_count = 0;
  status = cuDeviceGetCount(&device_count);
  if (status != CUDA_SUCCESS || device_count < 1) {
    std::fprintf(stderr, "no CUDA devices visible (status %d, count %d)\n",
                 static_cast<int>(status), device_count);
    return 2;
  }

  CUdevice device = 0;
  status = cuDeviceGet(&device, 0);
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "cuDeviceGet failed: %d\n", static_cast<int>(status));
    return 1;
  }

  int value = 0;
  status = cuDeviceGetP2PAttribute(
      &value, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, device, device);
  if (status != CUDA_ERROR_INVALID_DEVICE) {
    std::fprintf(stderr, "same-device query returned %d, want %d\n",
                 static_cast<int>(status),
                 static_cast<int>(CUDA_ERROR_INVALID_DEVICE));
    return 1;
  }

  // Devices are checked before the attribute, so a nonsense enum still reports
  // INVALID_DEVICE.
  status = cuDeviceGetP2PAttribute(
      &value, static_cast<CUdevice_P2PAttribute>(9999), device, device);
  if (status != CUDA_ERROR_INVALID_DEVICE) {
    std::fprintf(stderr,
                 "same-device query with an unknown attribute returned %d, "
                 "want %d\n",
                 static_cast<int>(status),
                 static_cast<int>(CUDA_ERROR_INVALID_DEVICE));
    return 1;
  }

  // The null value pointer is the one check that outranks the devices.
  status = cuDeviceGetP2PAttribute(
      nullptr, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, device, device);
  if (status != CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr, "null value pointer returned %d, want %d\n",
                 static_cast<int>(status),
                 static_cast<int>(CUDA_ERROR_INVALID_VALUE));
    return 1;
  }

  std::printf("cuDeviceGetP2PAttribute validation order ok\n");
  return 0;
}
