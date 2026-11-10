// NVML string and null-pointer validation regression test:
//  - nvmlDeviceGetName(dev, nullptr, len) and nvmlEventSetCreate(nullptr) must
//    return NVML_ERROR_INVALID_ARGUMENT without crashing or leaking handles
//    (and NVML_ERROR_UNINITIALIZED before nvmlInit).
//  - Zero-length non-null string buffers passed to nvmlSystemGetDriverVersion,
//    nvmlSystemGetNVMLVersion, nvmlDeviceGetName, and nvmlDeviceGetUUID must
//    return NVML_ERROR_INSUFFICIENT_SIZE (matching native NVML), while null
//    string buffers return NVML_ERROR_INVALID_ARGUMENT even when length == 0.
#include <nvml.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

namespace {

template <typename Fn> Fn load_sym(void *h, const char *name) {
  return reinterpret_cast<Fn>(dlsym(h, name));
}

bool expect_rc(nvmlReturn_t got, nvmlReturn_t want, const char *what) {
  if (got == want) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: got %d, want %d\n", what,
               static_cast<int>(got), static_cast<int>(want));
  return false;
}

} // namespace

int main() {
  void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    h = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (h == nullptr) {
    std::printf("SKIP: libnvidia-ml not found\n");
    return EXIT_SUCCESS;
  }

  auto fn_init = load_sym<decltype(&nvmlInit_v2)>(h, "nvmlInit_v2");
  auto fn_shutdown = load_sym<decltype(&nvmlShutdown)>(h, "nvmlShutdown");
  auto fn_drv_ver = load_sym<decltype(&nvmlSystemGetDriverVersion)>(
      h, "nvmlSystemGetDriverVersion");
  auto fn_nvml_ver = load_sym<decltype(&nvmlSystemGetNVMLVersion)>(
      h, "nvmlSystemGetNVMLVersion");
  auto fn_cnt =
      load_sym<decltype(&nvmlDeviceGetCount_v2)>(h, "nvmlDeviceGetCount_v2");
  auto fn_by_idx = load_sym<decltype(&nvmlDeviceGetHandleByIndex_v2)>(
      h, "nvmlDeviceGetHandleByIndex_v2");
  auto fn_name = load_sym<decltype(&nvmlDeviceGetName)>(h, "nvmlDeviceGetName");
  auto fn_uuid = load_sym<decltype(&nvmlDeviceGetUUID)>(h, "nvmlDeviceGetUUID");
  auto fn_ev_create =
      load_sym<decltype(&nvmlEventSetCreate)>(h, "nvmlEventSetCreate");

  int failures = 0;

  // Pre-init checks
  failures += !expect_rc(fn_name(nullptr, nullptr, 64),
                         NVML_ERROR_UNINITIALIZED, "preinit nvmlDeviceGetName");
  failures += !expect_rc(fn_ev_create(nullptr), NVML_ERROR_UNINITIALIZED,
                         "preinit nvmlEventSetCreate");

  if (!expect_rc(fn_init(), NVML_SUCCESS, "nvmlInit_v2")) {
    dlclose(h);
    return EXIT_FAILURE;
  }

  unsigned int count = 0;
  if (fn_cnt(&count) != NVML_SUCCESS || count == 0) {
    fn_shutdown();
    dlclose(h);
    std::printf("SKIP: no NVML devices found\n");
    return EXIT_SUCCESS;
  }

  nvmlDevice_t dev = nullptr;
  if (!expect_rc(fn_by_idx(0, &dev), NVML_SUCCESS,
                 "nvmlDeviceGetHandleByIndex_v2")) {
    fn_shutdown();
    dlclose(h);
    return EXIT_FAILURE;
  }

  char buf[128] = {};
  failures += !expect_rc(fn_drv_ver(nullptr, 64), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlSystemGetDriverVersion(nullptr, 64)");
  failures += !expect_rc(fn_drv_ver(nullptr, 0), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlSystemGetDriverVersion(nullptr, 0)");
  failures += !expect_rc(fn_drv_ver(buf, 0), NVML_ERROR_INSUFFICIENT_SIZE,
                         "nvmlSystemGetDriverVersion(buf, 0)");

  failures += !expect_rc(fn_nvml_ver(nullptr, 64), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlSystemGetNVMLVersion(nullptr, 64)");
  failures += !expect_rc(fn_nvml_ver(nullptr, 0), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlSystemGetNVMLVersion(nullptr, 0)");
  failures += !expect_rc(fn_nvml_ver(buf, 0), NVML_ERROR_INSUFFICIENT_SIZE,
                         "nvmlSystemGetNVMLVersion(buf, 0)");

  failures += !expect_rc(fn_name(dev, nullptr, 64), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlDeviceGetName(dev, nullptr, 64)");
  failures += !expect_rc(fn_name(dev, nullptr, 0), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlDeviceGetName(dev, nullptr, 0)");
  failures += !expect_rc(fn_name(dev, buf, 0), NVML_ERROR_INSUFFICIENT_SIZE,
                         "nvmlDeviceGetName(dev, buf, 0)");
  failures += !expect_rc(fn_name(dev, buf, sizeof(buf)), NVML_SUCCESS,
                         "nvmlDeviceGetName(dev, buf, 128)");

  failures += !expect_rc(fn_uuid(dev, nullptr, 64), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlDeviceGetUUID(dev, nullptr, 64)");
  failures += !expect_rc(fn_uuid(dev, nullptr, 0), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlDeviceGetUUID(dev, nullptr, 0)");
  failures += !expect_rc(fn_uuid(dev, buf, 0), NVML_ERROR_INSUFFICIENT_SIZE,
                         "nvmlDeviceGetUUID(dev, buf, 0)");
  failures += !expect_rc(fn_uuid(dev, buf, sizeof(buf)), NVML_SUCCESS,
                         "nvmlDeviceGetUUID(dev, buf, 128)");

  failures += !expect_rc(fn_ev_create(nullptr), NVML_ERROR_INVALID_ARGUMENT,
                         "nvmlEventSetCreate(nullptr)");

  failures += !expect_rc(fn_shutdown(), NVML_SUCCESS, "nvmlShutdown");
  dlclose(h);

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d NVML validation checks failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: NVML string and null-pointer validation matches native\n");
  return EXIT_SUCCESS;
}
