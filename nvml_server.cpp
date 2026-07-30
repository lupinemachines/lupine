#include "nvml_server.h"

#include <cuda.h>
#include <nvml.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "codegen/gen_api.h"

// CUDA <= 12.6 ships NVML API 12, which does not define the versioned
// temperature struct. The host driver exports the symbol on newer drivers; this
// local definition preserves the ABI when building against older CUDA images.
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12080) ||                        \
    (defined(NVML_API_VERSION) && NVML_API_VERSION >= 13)
using lupine_nvmlTemperature_t = nvmlTemperature_t;
#else
typedef struct {
  unsigned int version;
  nvmlTemperatureSensors_t sensorType;
  int temperature;
} lupine_nvmlTemperature_t;
#endif

namespace {

nvmlReturn_t function_not_found() { return NVML_ERROR_FUNCTION_NOT_FOUND; }

void *nvml_library() {
#ifdef _WIN32
  static HMODULE lib = LoadLibraryA("nvml.dll");
  return lib;
#else
  static void *lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
  return lib;
#endif
}

template <typename Fn> Fn nvml_symbol(const char *name) {
  void *lib = nvml_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// Fixed wire size for the name/UUID strings in the nvmlInit device
// enumeration payload, independent of the NVML headers either side was built
// against.
constexpr unsigned int kEnumerateIdentifierLength = 96;

struct enumerated_device {
  nvmlDevice_t handle = nullptr;
  nvmlReturn_t name_status = NVML_ERROR_UNKNOWN;
  char name[kEnumerateIdentifierLength] = {};
  nvmlReturn_t uuid_status = NVML_ERROR_UNKNOWN;
  char uuid[kEnumerateIdentifierLength] = {};
  nvmlReturn_t pci_status = NVML_ERROR_UNKNOWN;
  nvmlPciInfo_t pci = {};
};

nvmlReturn_t enumerate_devices(std::vector<enumerated_device> *devices) {
  using count_fn_t = nvmlReturn_t (*)(unsigned int *);
  using handle_fn_t = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
  using string_fn_t = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
  using pci_fn_t = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *);
  count_fn_t count_fn = nvml_symbol<count_fn_t>("nvmlDeviceGetCount_v2");
  handle_fn_t handle_fn =
      nvml_symbol<handle_fn_t>("nvmlDeviceGetHandleByIndex_v2");
  if (count_fn == nullptr || handle_fn == nullptr) {
    return function_not_found();
  }
  unsigned int count = 0;
  nvmlReturn_t result = count_fn(&count);
  if (result != NVML_SUCCESS) {
    return result;
  }
  string_fn_t name_fn = nvml_symbol<string_fn_t>("nvmlDeviceGetName");
  string_fn_t uuid_fn = nvml_symbol<string_fn_t>("nvmlDeviceGetUUID");
  pci_fn_t pci_fn = nvml_symbol<pci_fn_t>("nvmlDeviceGetPciInfo_v3");
  for (unsigned int i = 0; i < count; ++i) {
    enumerated_device device;
    result = handle_fn(i, &device.handle);
    if (result != NVML_SUCCESS) {
      return result;
    }
    device.name_status =
        name_fn == nullptr
            ? function_not_found()
            : name_fn(device.handle, device.name, sizeof(device.name));
    device.uuid_status =
        uuid_fn == nullptr
            ? function_not_found()
            : uuid_fn(device.handle, device.uuid, sizeof(device.uuid));
    device.pci_status = pci_fn == nullptr ? function_not_found()
                                          : pci_fn(device.handle, &device.pci);
    devices->push_back(device);
  }
  return NVML_SUCCESS;
}

// The enumeration state must outlive rpc_write_end because rpc_write only
// queues pointers, so the caller owns it for the duration of the response.
struct enumeration_payload {
  nvmlReturn_t result = NVML_ERROR_UNKNOWN;
  unsigned int count = 0;
  std::vector<enumerated_device> devices;
};

void build_enumeration(enumeration_payload *payload) {
  payload->result = enumerate_devices(&payload->devices);
  if (payload->result != NVML_SUCCESS) {
    payload->devices.clear();
  }
  payload->count = static_cast<unsigned int>(payload->devices.size());
}

int write_enumeration(conn_t *conn, const enumeration_payload &payload) {
  if (rpc_write(conn, &payload.result, sizeof(payload.result)) < 0 ||
      rpc_write(conn, &payload.count, sizeof(payload.count)) < 0) {
    return -1;
  }
  for (const enumerated_device &device : payload.devices) {
    if (rpc_write(conn, &device.handle, sizeof(device.handle)) < 0 ||
        rpc_write(conn, &device.name_status, sizeof(device.name_status)) < 0 ||
        rpc_write(conn, device.name, sizeof(device.name)) < 0 ||
        rpc_write(conn, &device.uuid_status, sizeof(device.uuid_status)) < 0 ||
        rpc_write(conn, device.uuid, sizeof(device.uuid)) < 0 ||
        rpc_write(conn, &device.pci_status, sizeof(device.pci_status)) < 0 ||
        rpc_write(conn, &device.pci, sizeof(device.pci)) < 0) {
      return -1;
    }
  }
  return 0;
}

int handle_processes(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  unsigned int requested_count = 0;
  int has_infos = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &requested_count, sizeof(requested_count)) < 0 ||
      rpc_read(conn, &has_infos, sizeof(has_infos)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  unsigned int returned_count = requested_count;
  std::vector<nvmlProcessInfo_t> infos;
  if (has_infos && requested_count != 0) {
    infos.resize(requested_count);
  }

  using Fn =
      nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr
          ? function_not_found()
          : fn(device, &returned_count, infos.empty() ? nullptr : infos.data());
  unsigned int copied_count =
      has_infos ? std::min<unsigned int>(returned_count, requested_count) : 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_count, sizeof(returned_count)) < 0 ||
      rpc_write(conn, &copied_count, sizeof(copied_count)) < 0 ||
      (copied_count != 0 &&
       rpc_write(conn, infos.data(), copied_count * sizeof(infos[0])) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

#include "codegen/gen_nvml_server.inc"

// The init responses carry the device enumeration and identity payload so a
// fresh client learns every device handle, name, UUID, and PCI info without
// paying one network round trip per query.
int handle_nvmlInit_v2(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvmlReturn_t (*)();
  fn_t fn = nvml_symbol<fn_t>("nvmlInit_v2");
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn();
  enumeration_payload payload;
  if (result == NVML_SUCCESS) {
    build_enumeration(&payload);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == NVML_SUCCESS && write_enumeration(conn, payload) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvmlInitWithFlags(conn_t *conn) {
  unsigned int flags = 0;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvmlReturn_t (*)(unsigned int);
  fn_t fn = nvml_symbol<fn_t>("nvmlInitWithFlags");
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(flags);
  enumeration_payload payload;
  if (result == NVML_SUCCESS) {
    build_enumeration(&payload);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == NVML_SUCCESS && write_enumeration(conn, payload) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvmlDeviceGetComputeRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses");
}

int handle_nvmlDeviceGetComputeRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses_v2");
}

int handle_nvmlDeviceGetGraphicsRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetGraphicsRunningProcesses");
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetGraphicsRunningProcesses_v2");
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetMPSComputeRunningProcesses");
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetMPSComputeRunningProcesses_v2");
}
