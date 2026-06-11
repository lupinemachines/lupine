#include "nvml_server.h"

#include <cuda.h>
#include <nvml.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstring>
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

template <typename Fn> nvmlReturn_t call0(const char *name) {
  Fn fn = nvml_symbol<Fn>(name);
  if (fn == nullptr) {
    return function_not_found();
  }
  return fn();
}

int write_return(conn_t *conn, int request_id, nvmlReturn_t result) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_no_args(conn_t *conn, nvmlReturn_t (*call)()) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  return write_return(conn, request_id, call());
}

int handle_uint_out(conn_t *conn, const char *name) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  unsigned int value = 0;
  using Fn = nvmlReturn_t (*)(unsigned int *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(&value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_device_from_uint(conn_t *conn, const char *name) {
  unsigned int index = 0;
  if (rpc_read(conn, &index, sizeof(index)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  nvmlDevice_t device = nullptr;
  using Fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(index, &device);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &device, sizeof(device)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_device_string(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  unsigned int length = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<char> buffer(std::max(1u, length), '\0');
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, buffer.data(), length);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (length != 0 && rpc_write(conn, buffer.data(), length) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename T> int handle_device_value(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  if (rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  T value = {};
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, T *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename T> int handle_device_two_args_zero(conn_t *conn) {
  nvmlDevice_t device = nullptr;
  nvmlMemoryErrorType_t error_type = {};
  nvmlEccCounterType_t counter_type = {};
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &error_type, sizeof(error_type)) < 0 ||
      rpc_read(conn, &counter_type, sizeof(counter_type)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  T value = {};
  nvmlReturn_t result = NVML_SUCCESS;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename T> int handle_device_three_args_zero(conn_t *conn) {
  nvmlDevice_t device = nullptr;
  nvmlMemoryErrorType_t error_type = {};
  nvmlEccCounterType_t counter_type = {};
  nvmlMemoryLocation_t location_type = {};
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &error_type, sizeof(error_type)) < 0 ||
      rpc_read(conn, &counter_type, sizeof(counter_type)) < 0 ||
      rpc_read(conn, &location_type, sizeof(location_type)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  T value = {};
  nvmlReturn_t result = NVML_SUCCESS;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
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

int handle_event_set_create(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  nvmlEventSet_t set = nullptr;
  using Fn = nvmlReturn_t (*)(nvmlEventSet_t *);
  Fn fn = nvml_symbol<Fn>("nvmlEventSetCreate");
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(&set);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &set, sizeof(set)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_event_set_free(conn_t *conn) {
  nvmlEventSet_t set = nullptr;
  if (rpc_read(conn, &set, sizeof(set)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  using Fn = nvmlReturn_t (*)(nvmlEventSet_t);
  Fn fn = nvml_symbol<Fn>("nvmlEventSetFree");
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(set);
  return write_return(conn, request_id, result);
}

int handle_event_set_wait(conn_t *conn) {
  nvmlEventSet_t set = nullptr;
  unsigned int timeoutms = 0;
  if (rpc_read(conn, &set, sizeof(set)) < 0 ||
      rpc_read(conn, &timeoutms, sizeof(timeoutms)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  nvmlEventData_t data = {};
  using Fn = nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventData_t *, unsigned int);
  Fn fn = nvml_symbol<Fn>("nvmlEventSetWait_v2");
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(set, &data, timeoutms);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &data, sizeof(data)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_device_register_events(conn_t *conn) {
  nvmlDevice_t device = nullptr;
  unsigned long long event_types = 0;
  nvmlEventSet_t set = nullptr;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &event_types, sizeof(event_types)) < 0 ||
      rpc_read(conn, &set, sizeof(set)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  using Fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, nvmlEventSet_t);
  Fn fn = nvml_symbol<Fn>("nvmlDeviceRegisterEvents");
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, event_types, set);
  return write_return(conn, request_id, result);
}

} // namespace

#include "codegen/gen_nvml_server.inc"

int handle_nvmlInit_v2(conn_t *conn) {
  using Fn = nvmlReturn_t (*)();
  return handle_no_args(conn, []() { return call0<Fn>("nvmlInit_v2"); });
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
  using Fn = nvmlReturn_t (*)(unsigned int);
  Fn fn = nvml_symbol<Fn>("nvmlInitWithFlags");
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(flags);
  return write_return(conn, request_id, result);
}

int handle_nvmlShutdown(conn_t *conn) {
  using Fn = nvmlReturn_t (*)();
  return handle_no_args(conn, []() { return call0<Fn>("nvmlShutdown"); });
}

int handle_nvmlDeviceGetCount_v2(conn_t *conn) {
  return handle_uint_out(conn, "nvmlDeviceGetCount_v2");
}

int handle_nvmlDeviceGetHandleByIndex_v2(conn_t *conn) {
  return handle_device_from_uint(conn, "nvmlDeviceGetHandleByIndex_v2");
}

int handle_nvmlDeviceGetName(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetName");
}

int handle_nvmlDeviceGetIndex(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceGetIndex");
}

int handle_nvmlDeviceGetTotalEccErrors(conn_t *conn) {
  return handle_device_two_args_zero<unsigned long long>(conn);
}

int handle_nvmlDeviceGetDetailedEccErrors(conn_t *conn) {
  return handle_device_two_args_zero<nvmlEccErrorCounts_t>(conn);
}

int handle_nvmlDeviceGetMemoryErrorCounter(conn_t *conn) {
  return handle_device_three_args_zero<unsigned long long>(conn);
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

int handle_nvmlEventSetCreate(conn_t *conn) {
  return handle_event_set_create(conn);
}

int handle_nvmlEventSetFree(conn_t *conn) {
  return handle_event_set_free(conn);
}

int handle_nvmlEventSetWait_v2(conn_t *conn) {
  return handle_event_set_wait(conn);
}

int handle_nvmlDeviceRegisterEvents(conn_t *conn) {
  return handle_device_register_events(conn);
}
