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

int handle_string_no_device(conn_t *conn, const char *name) {
  unsigned int length = 0;
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<char> buffer(std::max(1u, length), '\0');
  using Fn = nvmlReturn_t (*)(char *, unsigned int);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(buffer.data(), length);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      (length != 0 && rpc_write(conn, buffer.data(), length) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_int_out(conn_t *conn, const char *name) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  int value = 0;
  using Fn = nvmlReturn_t (*)(int *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result = fn == nullptr ? function_not_found() : fn(&value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
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

int handle_device_from_string(conn_t *conn, const char *name) {
  unsigned int length = 0;
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  std::vector<char> value(std::max(1u, length), '\0');
  if (length != 0 && rpc_read(conn, value.data(), length) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  nvmlDevice_t device = nullptr;
  using Fn = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(value.data(), &device);

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

template <typename T>
int handle_device_struct(conn_t *conn, const char *name, T initial = {}) {
  nvmlDevice_t device = nullptr;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &initial, sizeof(initial)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  using Fn = nvmlReturn_t (*)(nvmlDevice_t, T *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, &initial);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &initial, sizeof(initial)) < 0 ||
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

template <typename Arg, typename Out>
int handle_device_arg_value(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  Arg arg = {};
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &arg, sizeof(arg)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  Out value = {};
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, Arg, Out *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, arg, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename A, typename B, typename Out>
int handle_device_two_args_value(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  A first = {};
  B second = {};
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &first, sizeof(first)) < 0 ||
      rpc_read(conn, &second, sizeof(second)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  Out value = {};
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, A, B, Out *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, first, second, &value);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename A, typename B, typename C, typename Out>
int handle_device_three_args_value(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  A first = {};
  B second = {};
  C third = {};
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &first, sizeof(first)) < 0 ||
      rpc_read(conn, &second, sizeof(second)) < 0 ||
      rpc_read(conn, &third, sizeof(third)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  Out value = {};
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, A, B, C, Out *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result = fn == nullptr
                            ? function_not_found()
                            : fn(device, first, second, third, &value);

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

template <typename A, typename B>
int handle_device_two_values(conn_t *conn, const char *name) {
  nvmlDevice_t device = nullptr;
  if (rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  A first = {};
  B second = {};
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, A *, B *);
  Fn fn = nvml_symbol<Fn>(name);
  nvmlReturn_t result =
      fn == nullptr ? function_not_found() : fn(device, &first, &second);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &first, sizeof(first)) < 0 ||
      rpc_write(conn, &second, sizeof(second)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

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

int handle_nvmlSystemGetDriverVersion(conn_t *conn) {
  return handle_string_no_device(conn, "nvmlSystemGetDriverVersion");
}

int handle_nvmlSystemGetNVMLVersion(conn_t *conn) {
  return handle_string_no_device(conn, "nvmlSystemGetNVMLVersion");
}

int handle_nvmlSystemGetCudaDriverVersion(conn_t *conn) {
  return handle_int_out(conn, "nvmlSystemGetCudaDriverVersion");
}

int handle_nvmlSystemGetCudaDriverVersion_v2(conn_t *conn) {
  return handle_int_out(conn, "nvmlSystemGetCudaDriverVersion_v2");
}

int handle_nvmlDeviceGetCount_v2(conn_t *conn) {
  return handle_uint_out(conn, "nvmlDeviceGetCount_v2");
}

int handle_nvmlDeviceGetHandleByIndex_v2(conn_t *conn) {
  return handle_device_from_uint(conn, "nvmlDeviceGetHandleByIndex_v2");
}

int handle_nvmlDeviceGetHandleByUUID(conn_t *conn) {
  return handle_device_from_string(conn, "nvmlDeviceGetHandleByUUID");
}

int handle_nvmlDeviceGetHandleByPciBusId_v2(conn_t *conn) {
  return handle_device_from_string(conn, "nvmlDeviceGetHandleByPciBusId_v2");
}

int handle_nvmlDeviceGetName(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetName");
}

int handle_nvmlDeviceGetUUID(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetUUID");
}

int handle_nvmlDeviceGetIndex(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceGetIndex");
}

int handle_nvmlDeviceGetMinorNumber(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceGetMinorNumber");
}

int handle_nvmlDeviceGetPciInfo_v3(conn_t *conn) {
  return handle_device_struct<nvmlPciInfo_t>(conn, "nvmlDeviceGetPciInfo_v3");
}

int handle_nvmlDeviceGetMemoryInfo(conn_t *conn) {
  return handle_device_struct<nvmlMemory_t>(conn, "nvmlDeviceGetMemoryInfo");
}

int handle_nvmlDeviceGetUtilizationRates(conn_t *conn) {
  return handle_device_struct<nvmlUtilization_t>(
      conn, "nvmlDeviceGetUtilizationRates");
}

int handle_nvmlDeviceGetTemperature(conn_t *conn) {
  return handle_device_arg_value<nvmlTemperatureSensors_t, unsigned int>(
      conn, "nvmlDeviceGetTemperature");
}

int handle_nvmlDeviceGetPowerUsage(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceGetPowerUsage");
}

int handle_nvmlDeviceGetPowerManagementLimit(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetPowerManagementLimit");
}

int handle_nvmlDeviceGetClockInfo(conn_t *conn) {
  return handle_device_arg_value<nvmlClockType_t, unsigned int>(
      conn, "nvmlDeviceGetClockInfo");
}

int handle_nvmlDeviceGetMaxClockInfo(conn_t *conn) {
  return handle_device_arg_value<nvmlClockType_t, unsigned int>(
      conn, "nvmlDeviceGetMaxClockInfo");
}

int handle_nvmlDeviceGetPerformanceState(conn_t *conn) {
  return handle_device_value<nvmlPstates_t>(conn,
                                            "nvmlDeviceGetPerformanceState");
}

int handle_nvmlDeviceGetComputeMode(conn_t *conn) {
  return handle_device_value<nvmlComputeMode_t>(conn,
                                                "nvmlDeviceGetComputeMode");
}

int handle_nvmlDeviceGetPersistenceMode(conn_t *conn) {
  return handle_device_value<nvmlEnableState_t>(conn,
                                                "nvmlDeviceGetPersistenceMode");
}

int handle_nvmlDeviceGetFanSpeed(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceGetFanSpeed");
}

int handle_nvmlDeviceGetBrand(conn_t *conn) {
  return handle_device_value<nvmlBrandType_t>(conn, "nvmlDeviceGetBrand");
}

int handle_nvmlDeviceGetVbiosVersion(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetVbiosVersion");
}

int handle_nvmlDeviceGetSerial(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetSerial");
}

int handle_nvmlDeviceGetBoardPartNumber(conn_t *conn) {
  return handle_device_string(conn, "nvmlDeviceGetBoardPartNumber");
}

int handle_nvmlDeviceGetDisplayMode(conn_t *conn) {
  return handle_device_value<nvmlEnableState_t>(conn,
                                                "nvmlDeviceGetDisplayMode");
}

int handle_nvmlDeviceGetDisplayActive(conn_t *conn) {
  return handle_device_value<nvmlEnableState_t>(conn,
                                                "nvmlDeviceGetDisplayActive");
}

int handle_nvmlDeviceGetCurrPcieLinkGeneration(conn_t *conn) {
  return handle_device_value<unsigned int>(
      conn, "nvmlDeviceGetCurrPcieLinkGeneration");
}

int handle_nvmlDeviceGetCurrPcieLinkWidth(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetCurrPcieLinkWidth");
}

int handle_nvmlDeviceGetMaxPcieLinkGeneration(conn_t *conn) {
  return handle_device_value<unsigned int>(
      conn, "nvmlDeviceGetMaxPcieLinkGeneration");
}

int handle_nvmlDeviceGetMaxPcieLinkWidth(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetMaxPcieLinkWidth");
}

int handle_nvmlDeviceGetPcieThroughput(conn_t *conn) {
  return handle_device_arg_value<nvmlPcieUtilCounter_t, unsigned int>(
      conn, "nvmlDeviceGetPcieThroughput");
}

int handle_nvmlDeviceGetPcieReplayCounter(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetPcieReplayCounter");
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

int handle_nvmlDeviceGetMaxMigDeviceCount(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetMaxMigDeviceCount");
}

int handle_nvmlDeviceGetTotalEccErrors(conn_t *conn) {
  return handle_device_two_args_value<nvmlMemoryErrorType_t,
                                      nvmlEccCounterType_t, unsigned long long>(
      conn, "nvmlDeviceGetTotalEccErrors");
}

int handle_nvmlDeviceGetDetailedEccErrors(conn_t *conn) {
  return handle_device_two_args_value<
      nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t>(
      conn, "nvmlDeviceGetDetailedEccErrors");
}

int handle_nvmlDeviceGetMemoryErrorCounter(conn_t *conn) {
  return handle_device_three_args_value<
      nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t,
      unsigned long long>(conn, "nvmlDeviceGetMemoryErrorCounter");
}

int handle_nvmlDeviceGetEccMode(conn_t *conn) {
  return handle_device_two_values<nvmlEnableState_t, nvmlEnableState_t>(
      conn, "nvmlDeviceGetEccMode");
}

int handle_nvmlDeviceGetTemperatureV(conn_t *conn) {
  return handle_device_struct<lupine_nvmlTemperature_t>(
      conn, "nvmlDeviceGetTemperatureV");
}

int handle_nvmlDeviceGetEnforcedPowerLimit(conn_t *conn) {
  return handle_device_value<unsigned int>(conn,
                                           "nvmlDeviceGetEnforcedPowerLimit");
}

int handle_nvmlDeviceGetMemoryInfo_v2(conn_t *conn) {
  return handle_device_struct<nvmlMemory_v2_t>(conn,
                                               "nvmlDeviceGetMemoryInfo_v2");
}

int handle_nvmlDeviceGetMigMode(conn_t *conn) {
  return handle_device_two_values<unsigned int, unsigned int>(
      conn, "nvmlDeviceGetMigMode");
}

int handle_nvmlDeviceGetVirtualizationMode(conn_t *conn) {
  return handle_device_value<nvmlGpuVirtualizationMode_t>(
      conn, "nvmlDeviceGetVirtualizationMode");
}

int handle_nvmlDeviceIsMigDeviceHandle(conn_t *conn) {
  return handle_device_value<unsigned int>(conn, "nvmlDeviceIsMigDeviceHandle");
}

int handle_nvmlDeviceGetNvLinkRemoteDeviceType(conn_t *conn) {
  return handle_device_arg_value<unsigned int, nvmlIntNvLinkDeviceType_t>(
      conn, "nvmlDeviceGetNvLinkRemoteDeviceType");
}

int handle_nvmlDeviceGetNvLinkRemotePciInfo_v2(conn_t *conn) {
  return handle_device_arg_value<unsigned int, nvmlPciInfo_t>(
      conn, "nvmlDeviceGetNvLinkRemotePciInfo_v2");
}
