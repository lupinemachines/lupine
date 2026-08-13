#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hip_compat.h"

#include "codegen/gen_api.h"
#include "lupine_client_transport.h"
#include "lupine_log.h"
#include "rpc.h"

namespace {

// HIP exposes devices as plain int ordinals (unlike NVML's opaque
// nvmlDevice_t handles), so the virtual device table just remembers which
// connection a virtual ordinal lives on and what the remote ordinal is.
struct lupine_hip_remote_device {
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
};

std::vector<lupine_hip_remote_device> devices;
bool devices_ready = false;

// hipInit is optional in the HIP runtime (the runtime auto-initializes on
// first use), so unlike the NVML shim we do not gate calls on an init
// reference count. We still forward hipInit to the server and keep a count so
// a future hipDeviceReset-style teardown has somewhere to hook.
std::atomic<int> init_refcount{0};

hipError_t rpc_error() { return hipErrorUnknown; }

// Lazily brings up the shared transport pool (idempotent) and returns the pool
// slot for `index`, or nullptr if the pool is not up or the index is out of
// range. The transport machinery -- LUPINE_SERVER parse, connect, TLS, http2
// init, dispatch thread, close/join/destroy -- lives in lupine_client_transport;
// hip has no API-specific connect/close hooks, so it passes the minimal
// dispatch thread and null hooks.
conn_t *connection(unsigned int index = 0) {
  if (nconns == 0) {
    static const lupine_transport_hooks hooks = {
        lupine_transport_dispatch_thread, nullptr, nullptr};
    if (lupine_transport_open(&hooks) < 0) {
      return nullptr;
    }
  }
  if (index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
}

void close_connections() {
  // hip has no per-conn close hook, so pass nullptr and let the common transport
  // do the plain shutdown + lupine_socket_close + read_cond broadcast, then join
  // and destroy. Clear the HIP device table afterwards -- it caches remote
  // ordinals tied to the connections that just went away.
  lupine_transport_close_all(nullptr);

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    devices_ready = false;
    devices.clear();
    pthread_mutex_unlock(&conn_mutex);
  }
}

// hipGetDeviceCount/hipDeviceGet are @disabled client (the client fabricates
// the virtual device table itself), so the codegen emits no lupine_rpc_
// helpers for them. These round-trip the server directly to enumerate the
// remote device table.
hipError_t call_int_out_on(conn_t *c, int op, int *value) {
  hipError_t result = rpc_error();
  int temp = 0;
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

hipError_t call_device_get_on(conn_t *c, int op, int ordinal, int *device) {
  hipError_t result = rpc_error();
  int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

hipError_t ensure_devices() {
  if (connection() == nullptr) {
    return rpc_error();
  }
  if (devices_ready) {
    return hipSuccess;
  }

  devices.clear();
  for (int i = 0; i < nconns; ++i) {
    int count = 0;
    hipError_t result =
        call_int_out_on(&conns[i], RPC_hipGetDeviceCount, &count);
    if (result != hipSuccess) {
      devices.clear();
      return result;
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      int remote = 0;
      result =
          call_device_get_on(&conns[i], RPC_hipDeviceGet, ordinal, &remote);
      if (result != hipSuccess) {
        devices.clear();
        return result;
      }
      devices.push_back(
          lupine_hip_remote_device{static_cast<unsigned int>(i), remote});
    }
  }
  devices_ready = true;
  return hipSuccess;
}

// Rewrites a virtual device ordinal (index into the device table) into the
// remote ordinal the chosen connection understands, and returns that
// connection. Mirrors nvml_client's connection_for_device, but with int
// ordinals instead of opaque nvmlDevice_t handles.
conn_t *connection_for_device(int *device) {
  if (device == nullptr || ensure_devices() != hipSuccess) {
    return nullptr;
  }
  if (devices.empty()) {
    return nullptr;
  }
  int virt = *device;
  if (virt < 0 || virt >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  const auto &mapped = devices[virt];
  *device = mapped.remote_ordinal;
  return connection(mapped.conn_index);
}

} // namespace

#include "codegen/gen_hip_client.inc"

extern "C" hipError_t hipInit(unsigned int flags) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (connection() == nullptr) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return hipErrorUnknown;
  }
  hipError_t first_error = hipSuccess;
  for (int i = 0; i < nconns; ++i) {
    conn_t *c = &conns[i];
    hipError_t result = rpc_error();
    if (rpc_write_start_request(c, RPC_hipInit) < 0 ||
        rpc_write(c, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(c) < 0 ||
        rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
      result = rpc_error();
    }
    if (result != hipSuccess && first_error == hipSuccess) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" hipError_t hipGetDeviceCount(int *count) {
  hipError_t result = ensure_devices();
  if (result != hipSuccess) {
    return result;
  }
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = static_cast<int>(devices.size());
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGet(int *device, int ordinal) {
  hipError_t result = ensure_devices();
  if (result != hipSuccess) {
    return result;
  }
  if (device == nullptr) {
    return hipErrorInvalidValue;
  }
  if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
    return hipErrorInvalidDevice;
  }
  // The virtual ordinal is the table index; hipDeviceGet just validates it.
  *device = ordinal;
  return hipSuccess;
}