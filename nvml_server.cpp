#include "nvml_server.h"

#include <cuda.h>
#include <nvml.h>

#include <algorithm>
#include <cstdlib>
#include <new>
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
      rpc_write(conn, infos.data(), copied_count * sizeof(*infos.data())) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

#include "codegen/gen_nvml_server.inc"

int handle_nvmlDeviceGetComputeRunningProcesses(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses");
}

int handle_nvmlDeviceGetComputeRunningProcesses_v2(conn_t *conn) {
  return handle_processes(conn, "nvmlDeviceGetComputeRunningProcesses_v2");
}

int handle_nvmlDeviceGetProcessUtilization(conn_t *conn) {
  nvmlDevice_t device = nullptr;
  unsigned int capacity = 0;
  int has_samples = 0;
  unsigned long long timestamp = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &capacity, sizeof(capacity)) < 0 ||
      rpc_read(conn, &has_samples, sizeof(has_samples)) < 0 ||
      rpc_read(conn, &timestamp, sizeof(timestamp)) < 0) {
    return -1;
  }
  const int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  // Bound per-request storage (32 MiB), including requests from older or
  // non-Lupine callers. A larger caller capacity still works for any result
  // fitting this bound. Never fabricate samples, PIDs, or utilization values.
  constexpr unsigned int max_samples = 1U << 20;
  unsigned int returned_count =
      has_samples ? std::min(capacity, max_samples) : capacity;
  std::vector<nvmlProcessUtilizationSample_t> samples;
  nvmlReturn_t result = NVML_ERROR_INVALID_ARGUMENT;
  using Fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessUtilizationSample_t *,
                              unsigned int *, unsigned long long);
  Fn fn = nvml_symbol<Fn>("nvmlDeviceGetProcessUtilization");
  if (has_samples == 0 || has_samples == 1) {
    if (fn == nullptr) {
      result = function_not_found();
    } else {
      try {
        // Non-null at capacity zero is distinct from a null count-only query.
        if (has_samples) {
          samples.resize(std::max(1U, returned_count));
        }
        result = fn(device, has_samples ? samples.data() : nullptr,
                    &returned_count, timestamp);
      } catch (const std::bad_alloc &) {
        result = NVML_ERROR_MEMORY;
      }
    }
  }
  if (result == NVML_ERROR_INSUFFICIENT_SIZE && has_samples &&
      capacity > max_samples && returned_count > max_samples &&
      returned_count <= capacity) {
    result = NVML_ERROR_MEMORY;
  }
  if (result == NVML_SUCCESS &&
      (returned_count > capacity || returned_count > samples.size() ||
       (!has_samples && returned_count != 0))) {
    result = NVML_ERROR_UNKNOWN;
  }
  const unsigned int copied_count = result == NVML_SUCCESS ? returned_count : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &returned_count, sizeof(returned_count)) < 0 ||
      rpc_write(conn, &copied_count, sizeof(copied_count)) < 0 ||
      (copied_count != 0 &&
       rpc_write(conn, samples.data(),
                 size_t{copied_count} * sizeof(samples[0])) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
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
