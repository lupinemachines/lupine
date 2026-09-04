#include "lupine_platform.h"

#include <cuda.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "lupine_log.h"
#include "rpc.h"
#include "transport.h"

// CUDA <= 12.6 ships NVML API 12, which does not define the versioned
// temperature struct. Keep the wrapper ABI-compatible with newer nvidia-smi.
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

struct lupine_nvml_remote_device {
  unsigned int conn_index = 0;
  unsigned int remote_index = 0;
  nvmlDevice_t remote_device = nullptr;
  std::string server_label;
};

std::vector<lupine_nvml_remote_device> devices;
bool devices_ready = false;

// Real NVML reference counts init/shutdown; this shim connects lazily, so
// without a counter it could never report UNINITIALIZED.
std::atomic<int> init_refcount{0};

thread_local bool init_gate_bypass = false;

bool nvml_initialized() {
  return init_gate_bypass || init_refcount.load(std::memory_order_acquire) > 0;
}

struct uninitialized_entry_point {
  uninitialized_entry_point() { init_gate_bypass = true; }
  ~uninitialized_entry_point() { init_gate_bypass = false; }
};

nvmlReturn_t rpc_error() {
  return nvml_initialized() ? NVML_ERROR_UNKNOWN : NVML_ERROR_UNINITIALIZED;
}

void *nvml_transport_dispatch(void *argument) {
  auto *connection = static_cast<conn_t *>(argument);
  while (!connection->closed) {
    int op = rpc_dispatch(connection, 1);
    if (op < 0 || connection->closed) {
      break;
    }
    if (rpc_read_end(connection) < 0) {
      break;
    }
  }
  return nullptr;
}

const lupine_client_transport_config &nvml_transport_config() {
  static const auto config = [] {
    lupine_client_transport_config config;
    config.dial_policy = lupine_client_dial_policy::single_attempt;
    config.strict_endpoints = true;
    config.log_missing_server = true;
    config.dispatch = nvml_transport_dispatch;
    return config;
  }();
  return config;
}

void nvml_retire_thread_lane(uint64_t lane_id) {
  lupine_client_transport_retire_lane(lane_id);
}

pthread_once_t nvml_rpc_lifecycle_once = PTHREAD_ONCE_INIT;

void nvml_install_rpc_lifecycle_hooks() {
  const rpc_lifecycle_hooks hooks = {nullptr, nvml_retire_thread_lane, nullptr};
  if (rpc_set_lifecycle_hooks(&hooks) < 0) {
    LUPINE_LOG_ERROR("Failed to install NVML RPC lifecycle hooks");
  }
}

int open_connection() {
  if (pthread_once(&nvml_rpc_lifecycle_once,
                   nvml_install_rpc_lifecycle_hooks) != 0) {
    return -1;
  }
  return lupine_client_transport_open(nvml_transport_config());
}

conn_t *connection(unsigned int index = 0) {
  if (!nvml_initialized() || open_connection() < 0) {
    return nullptr;
  }
  return lupine_client_transport_connection(index);
}

void close_connections() {
  lupine_client_transport_close();
  devices_ready = false;
  devices.clear();
}

lupine_nvml_remote_device *mapped_device(nvmlDevice_t device) {
  if (device == nullptr || devices.empty()) {
    return nullptr;
  }
  uintptr_t begin = reinterpret_cast<uintptr_t>(devices.data());
  uintptr_t end = begin + devices.size() * sizeof(lupine_nvml_remote_device);
  uintptr_t value = reinterpret_cast<uintptr_t>(device);
  if (value < begin || value >= end ||
      (value - begin) % sizeof(lupine_nvml_remote_device) != 0) {
    return nullptr;
  }
  return &devices[(value - begin) / sizeof(lupine_nvml_remote_device)];
}

nvmlReturn_t call_no_args_on(conn_t *c, int op) {
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_uint_out_on(conn_t *c, int op, unsigned int *value) {
  nvmlReturn_t result = rpc_error();
  unsigned int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

nvmlReturn_t call_device_from_index_on(conn_t *c, int op, unsigned int index,
                                       nvmlDevice_t *device) {
  nvmlReturn_t result = rpc_error();
  nvmlDevice_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &index, sizeof(index)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

nvmlReturn_t ensure_devices() {
  if (!nvml_initialized() || open_connection() < 0) {
    return rpc_error();
  }
  if (devices_ready) {
    return NVML_SUCCESS;
  }

  devices.clear();
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *conn = lupine_client_transport_connection(i);
    unsigned int count = 0;
    nvmlReturn_t result =
        call_uint_out_on(conn, RPC_nvmlDeviceGetCount_v2, &count);
    if (result != NVML_SUCCESS) {
      devices.clear();
      return result;
    }
    for (unsigned int ordinal = 0; ordinal < count; ++ordinal) {
      nvmlDevice_t remote = nullptr;
      result = call_device_from_index_on(
          conn, RPC_nvmlDeviceGetHandleByIndex_v2, ordinal, &remote);
      if (result != NVML_SUCCESS) {
        devices.clear();
        return result;
      }
      const lupine_client_endpoint *endpoint =
          lupine_client_transport_endpoint(i);
      devices.push_back(lupine_nvml_remote_device{
          i, ordinal, remote, endpoint == nullptr ? "" : endpoint->label});
    }
  }
  devices_ready = true;
  return NVML_SUCCESS;
}

template <typename Lookup>
nvmlReturn_t lookup_device_on_all_connections(nvmlDevice_t *device,
                                              Lookup &&lookup) {
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }

  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }

  nvmlReturn_t first_error = NVML_ERROR_NOT_FOUND;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    nvmlDevice_t remote = nullptr;
    result = lookup(lupine_client_transport_connection(i), &remote);
    if (result != NVML_SUCCESS) {
      if (first_error == NVML_ERROR_NOT_FOUND &&
          result != NVML_ERROR_NOT_FOUND) {
        first_error = result;
      }
      continue;
    }

    auto mapped = std::find_if(
        devices.begin(), devices.end(), [&](const auto &candidate) {
          return candidate.conn_index == i && candidate.remote_device == remote;
        });
    if (mapped == devices.end()) {
      // The server returned a handle that was not part of the device table
      // built from nvmlDeviceGetCount/GetHandleByIndex. Never expose that raw
      // process-local pointer to the caller.
      if (first_error == NVML_ERROR_NOT_FOUND) {
        first_error = rpc_error();
      }
      continue;
    }

    *device = reinterpret_cast<nvmlDevice_t>(&*mapped);
    return NVML_SUCCESS;
  }
  return first_error;
}

conn_t *connection_for_device(nvmlDevice_t *device) {
  if (device == nullptr || ensure_devices() != NVML_SUCCESS) {
    return nullptr;
  }
  if (devices.empty()) {
    return nullptr;
  }
  auto *mapped = mapped_device(*device);
  if (mapped == nullptr) {
    return connection();
  }
  *device = mapped->remote_device;
  return connection(mapped->conn_index);
}

nvmlReturn_t call_no_args(int op) { return call_no_args_on(connection(), op); }

nvmlReturn_t call_device_string(int op, nvmlDevice_t device, char *value,
                                unsigned int length) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &length, sizeof(length)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      (length != 0 && rpc_read(c, value, length) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_processes(int op, nvmlDevice_t device,
                            unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  unsigned int requested_count = infoCount == nullptr ? 0 : *infoCount;
  int has_infos = infos == nullptr ? 0 : 1;
  unsigned int returned_count = 0;
  unsigned int copied_count = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &requested_count, sizeof(requested_count)) < 0 ||
      rpc_write(c, &has_infos, sizeof(has_infos)) < 0 ||
      rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &returned_count, sizeof(returned_count)) < 0 ||
      rpc_read(c, &copied_count, sizeof(copied_count)) < 0 ||
      (copied_count != 0 &&
       rpc_read(c, infos, copied_count * sizeof(*infos)) < 0) ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (infoCount != nullptr) {
    *infoCount = returned_count;
  }
  return result;
}

nvmlReturn_t call_event_set_create(nvmlEventSet_t *set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventSet_t temp = nullptr;
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetCreate) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (set != nullptr) {
    *set = temp;
  }
  return result;
}

nvmlReturn_t call_event_set_free(nvmlEventSet_t set) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetFree) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

nvmlReturn_t call_event_set_wait(nvmlEventSet_t set, nvmlEventData_t *data,
                                 unsigned int timeoutms) {
  conn_t *c = connection();
  nvmlReturn_t result = rpc_error();
  nvmlEventData_t temp = {};
  if (c == nullptr || rpc_write_start_request(c, RPC_nvmlEventSetWait_v2) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 ||
      rpc_write(c, &timeoutms, sizeof(timeoutms)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (data != nullptr) {
    *data = temp;
  }
  return result;
}

nvmlReturn_t call_device_register_events(nvmlDevice_t device,
                                         unsigned long long eventTypes,
                                         nvmlEventSet_t set) {
  conn_t *c = connection_for_device(&device);
  nvmlReturn_t result = rpc_error();
  if (c == nullptr ||
      rpc_write_start_request(c, RPC_nvmlDeviceRegisterEvents) < 0 ||
      rpc_write(c, &device, sizeof(device)) < 0 ||
      rpc_write(c, &eventTypes, sizeof(eventTypes)) < 0 ||
      rpc_write(c, &set, sizeof(set)) < 0 || rpc_wait_for_response(c) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  return result;
}

} // namespace

#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif
#ifdef nvmlDeviceGetHandleByPciBusId
#undef nvmlDeviceGetHandleByPciBusId
#endif
#ifdef nvmlDeviceGetPciInfo
#undef nvmlDeviceGetPciInfo
#endif
#ifdef nvmlDeviceGetComputeRunningProcesses
#undef nvmlDeviceGetComputeRunningProcesses
#endif
#ifdef nvmlDeviceGetGraphicsRunningProcesses
#undef nvmlDeviceGetGraphicsRunningProcesses
#endif
#ifdef nvmlDeviceGetMPSComputeRunningProcesses
#undef nvmlDeviceGetMPSComputeRunningProcesses
#endif
#ifdef nvmlEventSetWait
#undef nvmlEventSetWait
#endif

// Real NVML answers this one without nvmlInit, so rename the generated entry
// point and re-export it below with the gate lifted.
#define nvmlSystemGetNVMLVersion lupine_nvmlSystemGetNVMLVersion_gated

#include "codegen/gen_nvml_client.inc"

#undef nvmlSystemGetNVMLVersion

extern "C" nvmlReturn_t nvmlSystemGetNVMLVersion(char *version,
                                                 unsigned int length) {
  uninitialized_entry_point guard;
  return lupine_nvmlSystemGetNVMLVersion_gated(version, length);
}

extern "C" nvmlReturn_t nvmlInit_v2(void) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (open_connection() < 0) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return NVML_ERROR_UNKNOWN;
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    nvmlReturn_t result =
        call_no_args_on(lupine_client_transport_connection(i), RPC_nvmlInit_v2);
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (open_connection() < 0) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return NVML_ERROR_UNKNOWN;
  }
  nvmlReturn_t first_error = NVML_SUCCESS;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *c = lupine_client_transport_connection(i);
    nvmlReturn_t result = rpc_error();
    if (rpc_write_start_request(c, RPC_nvmlInitWithFlags) < 0 ||
        rpc_write(c, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(c) < 0 ||
        rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
      result = rpc_error();
    }
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" nvmlReturn_t nvmlShutdown(void) {
  // Claim a reference up front so an unmatched shutdown fails and concurrent
  // callers cannot tear the connections down twice.
  int previous = init_refcount.load(std::memory_order_acquire);
  do {
    if (previous <= 0) {
      return NVML_ERROR_UNINITIALIZED;
    }
  } while (!init_refcount.compare_exchange_weak(previous, previous - 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire));
  bool last = previous == 1;

  if (lupine_client_transport_size() == 0) {
    return NVML_SUCCESS;
  }
  unsigned int count = lupine_client_transport_size();

  // The server refcounts too, so forward every shutdown, not just the last.
  nvmlReturn_t first_error = NVML_SUCCESS;
  for (unsigned int i = 0; i < count; ++i) {
    nvmlReturn_t result = call_no_args_on(lupine_client_transport_connection(i),
                                          RPC_nvmlShutdown);
    if (result != NVML_SUCCESS && first_error == NVML_SUCCESS) {
      first_error = result;
    }
  }
  if (last) {
    close_connections();
  }
  return first_error;
}

extern "C" const char *nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
  case NVML_SUCCESS:
    return "Success";
  case NVML_ERROR_UNINITIALIZED:
    return "Uninitialized";
  case NVML_ERROR_INVALID_ARGUMENT:
    return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED:
    return "Not Supported";
  case NVML_ERROR_NO_PERMISSION:
    return "Insufficient Permissions";
  case NVML_ERROR_ALREADY_INITIALIZED:
    return "Already Initialized";
  case NVML_ERROR_NOT_FOUND:
    return "Not Found";
  case NVML_ERROR_INSUFFICIENT_SIZE:
    return "Insufficient Size";
  case NVML_ERROR_INSUFFICIENT_POWER:
    return "Insufficient External Power";
  case NVML_ERROR_DRIVER_NOT_LOADED:
    return "Driver Not Loaded";
  case NVML_ERROR_TIMEOUT:
    return "Timeout";
  case NVML_ERROR_IRQ_ISSUE:
    return "IRQ Issue";
  case NVML_ERROR_LIBRARY_NOT_FOUND:
    return "NVML Shared Library Not Found";
  case NVML_ERROR_FUNCTION_NOT_FOUND:
    return "Function Not Found";
  case NVML_ERROR_CORRUPTED_INFOROM:
    return "Corrupted InfoROM";
  case NVML_ERROR_GPU_IS_LOST:
    return "GPU is lost";
  case NVML_ERROR_RESET_REQUIRED:
    return "GPU requires reset";
  case NVML_ERROR_OPERATING_SYSTEM:
    return "The operating system has blocked the request";
  case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
    return "RM has detected an NVML/RM version mismatch";
  case NVML_ERROR_IN_USE:
    return "GPU is currently in use";
  case NVML_ERROR_MEMORY:
    return "Insufficient memory";
  case NVML_ERROR_NO_DATA:
    return "No data";
  case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
    return "VGPU ECC not supported";
  case NVML_ERROR_INSUFFICIENT_RESOURCES:
    return "Insufficient resources";
  default:
    return "Unknown Error";
  }
}

extern "C" nvmlReturn_t nvmlInternalGetExportTable(const void **ppExportTable,
                                                   const void *exportTableId) {
  if (ppExportTable == nullptr || exportTableId == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  // The one table ID nvidia-smi asks for; this shim proxies none of the
  // internal entry points it holds, so the table stays empty.
  static const unsigned char known_export_table_id[16] = {
      0xc4, 0xfe, 0x3e, 0x6c, 0xc9, 0x8f, 0x6c, 0x4e,
      0xa3, 0x27, 0xee, 0x69, 0x6e, 0x12, 0xf7, 0xc4};
  if (memcmp(exportTableId, known_export_table_id,
             sizeof(known_export_table_id)) != 0) {
    *ppExportTable = nullptr;
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  static void *empty_table[512] = {};
  *ppExportTable = empty_table;
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
  return call_event_set_create(set);
}

extern "C" nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
  return call_event_set_free(set);
}

extern "C" nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set,
                                            nvmlEventData_t *data,
                                            unsigned int timeoutms) {
  return call_event_set_wait(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlEventSetWait(nvmlEventSet_t set,
                                         nvmlEventData_t *data,
                                         unsigned int timeoutms) {
  return nvmlEventSetWait_v2(set, data, timeoutms);
}

extern "C" nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device,
                                                 unsigned long long eventTypes,
                                                 nvmlEventSet_t set) {
  return call_device_register_events(device, eventTypes, set);
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (deviceCount == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *deviceCount = static_cast<unsigned int>(devices.size());
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
  return nvmlDeviceGetCount_v2(deviceCount);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                                      nvmlDevice_t *device) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (device == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  if (index >= devices.size()) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *device = reinterpret_cast<nvmlDevice_t>(&devices[index]);
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index,
                                                   nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByIndex_v2(index, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                                      nvmlDevice_t *device) {
  return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                                          unsigned int length) {
  nvmlReturn_t result =
      call_device_string(RPC_nvmlDeviceGetName, device, name, length);
  if (result != NVML_SUCCESS || name == nullptr || length == 0) {
    return result;
  }

  auto *mapped = mapped_device(device);
  if (mapped == nullptr || mapped->server_label.empty()) {
    return result;
  }

  size_t used = strnlen(name, length);
  if (used >= length) {
    name[length - 1] = '\0';
    used = length - 1;
  }
  if (used + 1 < length) {
    snprintf(name + used, length - used, " (via lupine %s)",
             mapped->server_label.c_str());
  }
  return result;
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device,
                                           unsigned int *index) {
  nvmlReturn_t result = ensure_devices();
  if (result != NVML_SUCCESS) {
    return result;
  }
  if (devices.empty() || index == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  // Deliberate divergence: the fan-out index is fabricated client-side, so it
  // can only be recovered from a handle this shim minted.
  auto *mapped = mapped_device(device);
  if (mapped == nullptr) {
    return NVML_ERROR_INVALID_ARGUMENT;
  }
  *index = static_cast<unsigned int>(mapped - devices.data());
  return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device,
                                                nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device,
                                             nvmlPciInfo_t *pci) {
  return nvmlDeviceGetPciInfo_v3(device, pci);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetComputeRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetGraphicsRunningProcesses_v2, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses, device,
                        infoCount, infos);
}

extern "C" nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos) {
  return call_processes(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v2, device,
                        infoCount, infos);
}
