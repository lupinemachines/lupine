// Drives the NVML surface NCCL resolves with dlsym and calls while building a
// communicator. NCCL calls the P2P status matrix through a function pointer it
// never null-checks, so a symbol missing from the shim is a segfault inside
// ncclCommInitRank rather than an error NCCL reports.
#include <cuda_runtime.h>
#include <nvml.h>

#include <cstdio>
#include <dlfcn.h>
#include <vector>

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NVML(call)                                                       \
  do {                                                                         \
    nvmlReturn_t status = (call);                                              \
    if (status != NVML_SUCCESS) {                                              \
      fprintf(stderr, "%s failed: %d\n", #call, status);                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// The shim may legitimately not have the hardware for these, but it must have
// wired the call up: a missing entry point is what NCCL cannot survive.
#define CHECK_WIRED(call)                                                      \
  do {                                                                         \
    nvmlReturn_t status = (call);                                              \
    if (status == NVML_ERROR_FUNCTION_NOT_FOUND ||                             \
        status == NVML_ERROR_UNINITIALIZED) {                                  \
      fprintf(stderr, "%s not wired up: %d\n", #call, status);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void *library = nullptr;
static int missing = 0;

template <typename Fn> static Fn resolve(const char *name) {
  Fn fn = reinterpret_cast<Fn>(dlsym(library, name));
  if (fn == nullptr) {
    fprintf(stderr, "libnvidia-ml.so.1 is missing %s\n", name);
    missing++;
  }
  return fn;
}

int main() {
  library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (library == nullptr) {
    fprintf(stderr, "dlopen(libnvidia-ml.so.1) failed: %s\n", dlerror());
    return 1;
  }

  auto init = resolve<nvmlReturn_t (*)()>("nvmlInit");
  auto init_v2 = resolve<nvmlReturn_t (*)()>("nvmlInit_v2");
  auto shutdown = resolve<nvmlReturn_t (*)()>("nvmlShutdown");
  auto get_count = resolve<nvmlReturn_t (*)(unsigned int *)>("nvmlDeviceGetCount");
  auto get_count_v2 =
      resolve<nvmlReturn_t (*)(unsigned int *)>("nvmlDeviceGetCount_v2");
  auto by_index = resolve<nvmlReturn_t (*)(unsigned int, nvmlDevice_t *)>(
      "nvmlDeviceGetHandleByIndex");
  auto by_bus_id = resolve<nvmlReturn_t (*)(const char *, nvmlDevice_t *)>(
      "nvmlDeviceGetHandleByPciBusId");
  auto get_index = resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(
      "nvmlDeviceGetIndex");
  auto get_name = resolve<nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int)>(
      "nvmlDeviceGetName");
  auto error_string = resolve<const char *(*)(nvmlReturn_t)>("nvmlErrorString");
  auto memory_errors =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t,
                               nvmlEccCounterType_t, nvmlMemoryLocation_t,
                               unsigned long long *)>(
          "nvmlDeviceGetMemoryErrorCounter");
  auto compute_capability =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, int *, int *)>(
          "nvmlDeviceGetCudaComputeCapability");
  auto p2p_status = resolve<nvmlReturn_t (*)(
      nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t *)>(
      "nvmlDeviceGetP2PStatus");
  auto nvlink_state =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlEnableState_t *)>(
          "nvmlDeviceGetNvLinkState");
  auto nvlink_capability =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int,
                               nvmlNvLinkCapability_t, unsigned int *)>(
          "nvmlDeviceGetNvLinkCapability");
  auto nvlink_remote_pci =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlPciInfo_t *)>(
          "nvmlDeviceGetNvLinkRemotePciInfo");
  auto nvlink_remote_type =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int,
                               nvmlIntNvLinkDeviceType_t *)>(
          "nvmlDeviceGetNvLinkRemoteDeviceType");
  auto field_values =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, int, nvmlFieldValue_t *)>(
          "nvmlDeviceGetFieldValues");
  auto pci_info = resolve<nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *)>(
      "nvmlDeviceGetPciInfo_v3");
  auto pcie_max_speed =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(
          "nvmlDeviceGetPcieLinkMaxSpeed");
  auto pcie_generation = resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(
      "nvmlDeviceGetCurrPcieLinkGeneration");
  auto pcie_width = resolve<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(
      "nvmlDeviceGetCurrPcieLinkWidth");
#if defined(nvmlPciInfoExt_v1)
  auto pci_info_ext = resolve<nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfoExt_t *)>(
      "nvmlDeviceGetPciInfoExt");
#endif
#if defined(nvmlGpuFabricInfo_v2)
  auto fabric_info =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuFabricInfoV_t *)>(
          "nvmlDeviceGetGpuFabricInfoV");
#endif
#if defined(nvmlPlatformInfo_v1)
  auto platform_info =
      resolve<nvmlReturn_t (*)(nvmlDevice_t, nvmlPlatformInfo_t *)>(
          "nvmlDeviceGetPlatformInfo");
#endif
#if CUDA_VERSION >= 12040
  auto conf_compute_state =
      resolve<nvmlReturn_t (*)(nvmlConfComputeSystemState_t *)>(
          "nvmlSystemGetConfComputeState");
#endif
#if defined(nvmlSystemConfComputeSettings_v1)
  auto conf_compute_settings =
      resolve<nvmlReturn_t (*)(nvmlSystemConfComputeSettings_t *)>(
          "nvmlSystemGetConfComputeSettings");
#endif
  EXPECT(missing == 0);
  printf("symbols: passed\n");

  CHECK_NVML(init_v2());
  CHECK_NVML(init());
  unsigned int count = 0;
  CHECK_NVML(get_count_v2(&count));
  unsigned int legacy_count = 0;
  CHECK_NVML(get_count(&legacy_count));
  EXPECT(count == legacy_count);
  EXPECT(count > 0);

  std::vector<nvmlDevice_t> devices(count);
  for (unsigned int i = 0; i < count; ++i) {
    CHECK_NVML(by_index(i, &devices[i]));
    unsigned int index = count;
    CHECK_NVML(get_index(devices[i], &index));
    EXPECT(index == i);
    int major = 0;
    int minor = 0;
    CHECK_NVML(compute_capability(devices[i], &major, &minor));
    EXPECT(major > 0);
    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    CHECK_NVML(get_name(devices[i], name, sizeof(name)));
    nvmlPciInfo_t pci = {};
    CHECK_NVML(pci_info(devices[i], &pci));
    nvmlDevice_t by_bus = nullptr;
    CHECK_NVML(by_bus_id(pci.busId, &by_bus));
    EXPECT(by_bus == devices[i]);
  }
  printf("devices: passed\n");

  // What ncclNvmlEnsureInitialized() walks, and where it used to crash.
  for (unsigned int a = 0; a < count; ++a) {
    for (unsigned int b = 0; b < count; ++b) {
      nvmlGpuP2PStatus_t read_status = NVML_P2P_STATUS_UNKNOWN;
      nvmlGpuP2PStatus_t write_status = NVML_P2P_STATUS_UNKNOWN;
      CHECK_NVML(p2p_status(devices[a], devices[b], NVML_P2P_CAPS_INDEX_READ,
                            &read_status));
      CHECK_NVML(p2p_status(devices[a], devices[b], NVML_P2P_CAPS_INDEX_WRITE,
                            &write_status));
      EXPECT(read_status != NVML_P2P_STATUS_UNKNOWN);
      EXPECT(write_status != NVML_P2P_STATUS_UNKNOWN);
    }
  }
  printf("p2p status: passed\n");

  nvmlDevice_t device = devices[0];
  nvmlEnableState_t link_state = NVML_FEATURE_DISABLED;
  CHECK_WIRED(nvlink_state(device, 0, &link_state));
  unsigned int capability = 0;
  CHECK_WIRED(nvlink_capability(device, 0, NVML_NVLINK_CAP_P2P_SUPPORTED,
                                &capability));
  nvmlPciInfo_t remote = {};
  CHECK_WIRED(nvlink_remote_pci(device, 0, &remote));
  nvmlIntNvLinkDeviceType_t remote_type = NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
  CHECK_WIRED(nvlink_remote_type(device, 0, &remote_type));
  unsigned int max_speed = 0;
  CHECK_WIRED(pcie_max_speed(device, &max_speed));
  unsigned int generation = 0;
  CHECK_WIRED(pcie_generation(device, &generation));
  unsigned int width = 0;
  CHECK_WIRED(pcie_width(device, &width));
  unsigned long long errors = 0;
  CHECK_WIRED(memory_errors(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                            NVML_VOLATILE_ECC, NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                            &errors));
  nvmlFieldValue_t value = {};
  value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
  CHECK_WIRED(field_values(device, 1, &value));
  EXPECT(value.fieldId == NVML_FI_DEV_NVLINK_LINK_COUNT);
#if defined(nvmlPciInfoExt_v1)
  nvmlPciInfoExt_t pci_ext = {};
  pci_ext.version = nvmlPciInfoExt_v1;
  CHECK_WIRED(pci_info_ext(device, &pci_ext));
#endif
#if defined(nvmlGpuFabricInfo_v2)
  nvmlGpuFabricInfoV_t fabric = {};
  fabric.version = nvmlGpuFabricInfo_v2;
  CHECK_WIRED(fabric_info(device, &fabric));
#endif
#if defined(nvmlPlatformInfo_v1)
  nvmlPlatformInfo_t platform = {};
  platform.version = nvmlPlatformInfo_v1;
  CHECK_WIRED(platform_info(device, &platform));
#endif
#if CUDA_VERSION >= 12040
  nvmlConfComputeSystemState_t conf_state = {};
  CHECK_WIRED(conf_compute_state(&conf_state));
#endif
#if defined(nvmlSystemConfComputeSettings_v1)
  nvmlSystemConfComputeSettings_t conf_settings = {};
  conf_settings.version = nvmlSystemConfComputeSettings_v1;
  CHECK_WIRED(conf_compute_settings(&conf_settings));
#endif
  printf("topology queries: passed\n");

  EXPECT(error_string(NVML_ERROR_NOT_SUPPORTED) != nullptr);
  CHECK_NVML(shutdown());
  CHECK_NVML(shutdown());
  printf("nvml nccl init surface: all checks passed\n");
  return 0;
}
