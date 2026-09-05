#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "codegen/gen_rpc_ids.h"
#include "lupine_log.h"
#include "rpc.h"
#include "transport.h"

namespace {

struct lupine_hip_remote_device {
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
};

std::mutex devices_mutex;
std::vector<lupine_hip_remote_device> devices;
bool devices_ready = false;
uint64_t devices_generation = 0;
std::atomic<uint64_t> connection_generation{1};

hipError_t rpc_error() { return hipErrorUnknown; }

void *hip_transport_dispatch(void *argument) {
  auto *connection = static_cast<conn_t *>(argument);
  while (!connection->closed) {
    int op = rpc_dispatch(connection, 1);
    if (op < 0 || connection->closed || rpc_read_end(connection) < 0) {
      break;
    }
  }
  return nullptr;
}

// Teardown can hold the transport mutex while invoking this callback. An
// atomic generation invalidates the cache without taking devices_mutex in the
// reverse order; the next device-scoped call rebuilds the ordinal table.
void hip_connection_closed(conn_t *) {
  connection_generation.fetch_add(1, std::memory_order_acq_rel);
}

struct hip_transport_shutdown {
  ~hip_transport_shutdown() { lupine_client_transport_close(); }
};

const lupine_client_transport_config &hip_transport_config() {
  static hip_transport_shutdown shutdown;
  static const auto config = [] {
    lupine_client_transport_config config;
    config.dial_policy = lupine_client_dial_policy::single_attempt;
    config.strict_endpoints = true;
    config.log_missing_server = true;
    config.dispatch = hip_transport_dispatch;
    config.connection_closed = hip_connection_closed;
    return config;
  }();
  (void)shutdown;
  return config;
}

void hip_retire_thread_lane(uint64_t lane_id) {
  lupine_client_transport_retire_lane(lane_id);
}

pthread_once_t hip_rpc_lifecycle_once = PTHREAD_ONCE_INIT;

void install_rpc_lifecycle_hooks() {
  const rpc_lifecycle_hooks hooks = {hip_connection_closed,
                                     hip_retire_thread_lane, nullptr};
  if (rpc_set_lifecycle_hooks(&hooks) < 0) {
    LUPINE_LOG_ERROR("Failed to install HIP RPC lifecycle hooks");
  }
}

int open_connections() {
  if (pthread_once(&hip_rpc_lifecycle_once, install_rpc_lifecycle_hooks) != 0) {
    return -1;
  }
  return lupine_client_transport_open(hip_transport_config());
}

conn_t *connection(unsigned int index = 0) {
  if (open_connections() < 0) {
    return nullptr;
  }
  return lupine_client_transport_connection(index);
}

hipError_t call_int_out_on(conn_t *conn, int op, int *value) {
  hipError_t result = rpc_error();
  int remote_value = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_value, sizeof(remote_value)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = remote_value;
  }
  return result;
}

hipError_t call_device_get_on(conn_t *conn, int ordinal, int *device) {
  hipError_t result = rpc_error();
  int remote_device = 0;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_hipDeviceGet) < 0 ||
      rpc_write(conn, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = remote_device;
  }
  return result;
}

// devices_mutex must be held by the caller. Keeping discovery under this lock
// makes the virtual ordinal table immutable while another thread is routing a
// device-scoped request.
hipError_t ensure_devices_locked() {
  if (open_connections() < 0) {
    return rpc_error();
  }
  uint64_t generation = connection_generation.load(std::memory_order_acquire);
  if (devices_ready && devices_generation == generation) {
    return hipSuccess;
  }

  std::vector<lupine_hip_remote_device> discovered;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *conn = lupine_client_transport_connection(i);
    int count = 0;
    hipError_t result = call_int_out_on(conn, RPC_hipGetDeviceCount, &count);
    if (result != hipSuccess) {
      return result;
    }
    if (count < 0) {
      return rpc_error();
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      int remote_device = 0;
      result = call_device_get_on(conn, ordinal, &remote_device);
      if (result != hipSuccess) {
        return result;
      }
      discovered.push_back({i, remote_device});
    }
  }
  if (connection_generation.load(std::memory_order_acquire) != generation) {
    return rpc_error();
  }
  devices = std::move(discovered);
  devices_generation = generation;
  devices_ready = true;
  return hipSuccess;
}

conn_t *connection_for_device(int *device) {
  if (device == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(devices_mutex);
  if (ensure_devices_locked() != hipSuccess || devices.empty()) {
    return nullptr;
  }
  int virtual_ordinal = *device;
  if (virtual_ordinal < 0 ||
      virtual_ordinal >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  lupine_hip_remote_device mapped = devices[virtual_ordinal];
  *device = mapped.remote_ordinal;
  return lupine_client_transport_connection(mapped.conn_index);
}

} // namespace

#include "codegen/gen_hip_client.inc"

extern "C" hipError_t hipInit(unsigned int flags) {
  std::lock_guard<std::mutex> lock(devices_mutex);
  if (open_connections() < 0) {
    return record(rpc_error());
  }

  hipError_t first_error = hipSuccess;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *conn = lupine_client_transport_connection(i);
    hipError_t result = rpc_error();
    if (rpc_write_start_request(conn, RPC_hipInit) < 0 ||
        rpc_write(conn, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      result = rpc_error();
    }
    if (result != hipSuccess && first_error == hipSuccess) {
      first_error = result;
    }
  }

  devices_ready = false;
  devices.clear();
  return record(first_error);
}

extern "C" hipError_t hipGetDeviceCount(int *count) {
  if (count == nullptr) {
    return record(hipErrorInvalidValue);
  }
  std::lock_guard<std::mutex> lock(devices_mutex);
  hipError_t result = ensure_devices_locked();
  if (result != hipSuccess) {
    return record(result);
  }
  *count = static_cast<int>(devices.size());
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGet(int *device, int ordinal) {
  if (device == nullptr) {
    return record(hipErrorInvalidValue);
  }
  std::lock_guard<std::mutex> lock(devices_mutex);
  hipError_t result = ensure_devices_locked();
  if (result != hipSuccess) {
    return record(result);
  }
  if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
    return record(hipErrorInvalidDevice);
  }
  *device = ordinal;
  return hipSuccess;
}

extern "C" hipError_t hipGetLastError() {
  const hipError_t error = local_error;
  local_error = hipSuccess;
  return error;
}

extern "C" hipError_t hipExtGetLastError() { return hipGetLastError(); }

extern "C" hipError_t hipPeekAtLastError() { return local_error; }
