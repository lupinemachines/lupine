#include "hip_compat.h"

#include <cstdio>
#include <cstring>

extern "C" hipError_t hipInit(unsigned int) { return hipSuccess; }

extern "C" hipError_t hipGetDeviceCount(int *count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = 2;
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGet(int *device, int ordinal) {
  if (device == nullptr) {
    return hipErrorInvalidValue;
  }
  if (ordinal < 0 || ordinal >= 2) {
    return hipErrorInvalidDevice;
  }
  *device = ordinal;
  return hipSuccess;
}

extern "C" hipError_t
hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 *properties, int device) {
  if (properties == nullptr || device < 0 || device >= 2) {
    return hipErrorInvalidValue;
  }
  *properties = {};
  std::snprintf(properties->name, sizeof(properties->name), "fake-amd-%d",
                device);
  std::snprintf(properties->gcnArchName, sizeof(properties->gcnArchName),
                "gfx-fake-%d", device);
  properties->totalGlobalMem =
      static_cast<size_t>(8 + device) * 1024 * 1024 * 1024;
  properties->multiProcessorCount = 60 + device;
  properties->asicRevision = 100 + device;
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGetName(char *name, int len, int device) {
  if (name == nullptr || len <= 0 || device < 0 || device >= 2) {
    return hipErrorInvalidValue;
  }
  std::snprintf(name, static_cast<size_t>(len), "fake-amd-%d", device);
  return hipSuccess;
}

extern "C" hipError_t hipDeviceTotalMem(size_t *bytes, int device) {
  if (bytes == nullptr || device < 0 || device >= 2) {
    return hipErrorInvalidValue;
  }
  *bytes = static_cast<size_t>(8 + device) * 1024 * 1024 * 1024;
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGetAttribute(int *value, hipDeviceAttribute_t,
                                            int device) {
  if (value == nullptr || device < 0 || device >= 2) {
    return hipErrorInvalidValue;
  }
  *value = 60 + device;
  return hipSuccess;
}

extern "C" hipError_t hipDriverGetVersion(int *version) {
  if (version == nullptr) {
    return hipErrorInvalidValue;
  }
  *version = 70000000;
  return hipSuccess;
}

extern "C" hipError_t hipRuntimeGetVersion(int *version) {
  if (version == nullptr) {
    return hipErrorInvalidValue;
  }
  *version = 70000000;
  return hipSuccess;
}
