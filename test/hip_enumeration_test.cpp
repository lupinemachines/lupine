#include "hip_compat.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

template <typename Fn> Fn symbol(void *library, const char *name) {
  Fn result = reinterpret_cast<Fn>(dlsym(library, name));
  require(result != nullptr, name);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  require(argc == 2, "usage: hip_enumeration_test LIBAMDHIP64");
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  require(library != nullptr, dlerror());

  auto init = symbol<hipError_t (*)(unsigned int)>(library, "hipInit");
  auto get_count = symbol<hipError_t (*)(int *)>(library, "hipGetDeviceCount");
  auto get_device = symbol<hipError_t (*)(int *, int)>(library, "hipDeviceGet");
  auto get_properties = symbol<hipError_t (*)(hipDeviceProp_tR0600 *, int)>(
      library, "hipGetDevicePropertiesR0600");
  auto get_name =
      symbol<hipError_t (*)(char *, int, int)>(library, "hipDeviceGetName");
  auto total_mem =
      symbol<hipError_t (*)(size_t *, int)>(library, "hipDeviceTotalMem");
  auto get_attribute = symbol<hipError_t (*)(int *, hipDeviceAttribute_t, int)>(
      library, "hipDeviceGetAttribute");
  auto driver_version =
      symbol<hipError_t (*)(int *)>(library, "hipDriverGetVersion");
  auto runtime_version =
      symbol<hipError_t (*)(int *)>(library, "hipRuntimeGetVersion");

  require(init(0) == hipSuccess, "hipInit failed");
  int count = 0;
  require(get_count(&count) == hipSuccess && count == 2,
          "HIP device count changed");
  int device = -1;
  require(get_device(&device, 1) == hipSuccess && device == 1,
          "HIP virtual device mapping changed");
  require(get_device(&device, count) == hipErrorInvalidDevice,
          "HIP invalid ordinal behavior changed");

  hipDeviceProp_tR0600 properties = {};
  require(get_properties(&properties, 1) == hipSuccess,
          "HIP properties failed");
  require(std::strcmp(properties.name, "fake-amd-1") == 0 &&
              properties.hdpMemFlushCntl == nullptr &&
              properties.hdpRegFlushCntl == nullptr,
          "HIP properties payload changed");

  char name[32] = {};
  require(get_name(name, sizeof(name), 1) == hipSuccess &&
              std::strcmp(name, "fake-amd-1") == 0,
          "HIP device name failed");
  require(get_name(name, -1, 1) == hipErrorInvalidValue,
          "HIP negative name length was accepted");

  size_t bytes = 0;
  require(total_mem(&bytes, 1) == hipSuccess &&
              bytes == static_cast<size_t>(9) * 1024 * 1024 * 1024,
          "HIP total memory failed");
  int attribute = 0;
  require(get_attribute(&attribute, static_cast<hipDeviceAttribute_t>(0), 1) ==
                  hipSuccess &&
              attribute == 61,
          "HIP device attribute failed");
  int version = 0;
  require(driver_version(&version) == hipSuccess && version == 70000000,
          "HIP driver version failed");
  require(runtime_version(&version) == hipSuccess && version == 70000000,
          "HIP runtime version failed");

  require(dlclose(library) == 0, "HIP shim close failed");
  std::cout << "hip_enumeration_test: PASS" << std::endl;
  return 0;
}
