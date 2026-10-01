#include "nvml_server.h"

#include <cuda.h>
#include <nvml.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "nvml_runtime.h"

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

template <typename Fn> Fn nvml_symbol(const char *name) {
  return lupine_nvml_symbol<Fn>(name);
}

int handle_processes(conn_t *conn, const char *name, size_t info_size) {
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
  std::vector<char> infos;
  if (has_infos && requested_count != 0) {
    infos.resize(requested_count * info_size);
  }

  using Fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, void *);
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
      rpc_write(conn, infos.data(), copied_count * info_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

#include "codegen/gen_nvml_server.inc"

int handle_nvmlDeviceGetComputeRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses",
                          sizeof(nvmlProcessInfo_v1_t));
}

int handle_nvmlDeviceGetComputeRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses_v2",
                          sizeof(nvmlProcessInfo_t));
}

int handle_nvmlDeviceGetComputeRunningProcesses_v3(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses_v3",
                          sizeof(nvmlProcessInfo_t));
}

int handle_nvmlDeviceGetGraphicsRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetGraphicsRunningProcesses",
                          sizeof(nvmlProcessInfo_v1_t));
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetGraphicsRunningProcesses_v2",
                          sizeof(nvmlProcessInfo_t));
}

int handle_nvmlDeviceGetGraphicsRunningProcesses_v3(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetGraphicsRunningProcesses_v3",
                          sizeof(nvmlProcessInfo_t));
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetMPSComputeRunningProcesses",
                          sizeof(nvmlProcessInfo_v1_t));
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetMPSComputeRunningProcesses_v2",
                          sizeof(nvmlProcessInfo_t));
}

int handle_nvmlDeviceGetMPSComputeRunningProcesses_v3(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetMPSComputeRunningProcesses_v3",
                          sizeof(nvmlProcessInfo_t));
}
