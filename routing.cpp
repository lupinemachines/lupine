#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "cache.h"
#include "client_routing.h"
#include "codegen/gen_api.h"

extern int rpc_open();
extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);

struct lupine_owner {
  bool local = false;
  unsigned int conn_index = 0;
};

struct lupine_deviceptr_allocation_record {
  CUdeviceptr base = 0;
  size_t size = 0;
  lupine_owner owner;
};

struct lupine_device_entry {
  bool local = false;
  int local_ordinal = -1;
  CUdevice local_device = -1;
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
  CUdevice remote_device = 0;
};

static std::mutex &lupine_routing_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::vector<lupine_device_entry> &lupine_device_table() {
  static auto *devices = new std::vector<lupine_device_entry>();
  return *devices;
}

static bool &lupine_device_table_ready() {
  static bool ready = false;
  return ready;
}

static std::unordered_map<CUcontext, lupine_owner> &lupine_context_owners() {
  static auto *owners = new std::unordered_map<CUcontext, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUmodule, lupine_owner> &lupine_module_owners() {
  static auto *owners = new std::unordered_map<CUmodule, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUlibrary, lupine_owner> &lupine_library_owners() {
  static auto *owners = new std::unordered_map<CUlibrary, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUfunction, lupine_owner> &lupine_function_owners() {
  static auto *owners = new std::unordered_map<CUfunction, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUstream, lupine_owner> &lupine_stream_owners() {
  static auto *owners = new std::unordered_map<CUstream, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUevent, lupine_owner> &lupine_event_owners() {
  static auto *owners = new std::unordered_map<CUevent, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUmemoryPool, lupine_owner> &
lupine_memory_pool_owners() {
  static auto *owners = new std::unordered_map<CUmemoryPool, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUgraph, lupine_owner> &lupine_graph_owners() {
  static auto *owners = new std::unordered_map<CUgraph, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUgraphNode, lupine_owner> &
lupine_graph_node_owners() {
  static auto *owners = new std::unordered_map<CUgraphNode, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUgraphExec, lupine_owner> &
lupine_graph_exec_owners() {
  static auto *owners = new std::unordered_map<CUgraphExec, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUdeviceptr, lupine_owner> &
lupine_deviceptr_owners() {
  static auto *owners = new std::unordered_map<CUdeviceptr, lupine_owner>();
  return *owners;
}

static std::unordered_map<CUdeviceptr, lupine_deviceptr_allocation_record> &
lupine_deviceptr_allocations() {
  static auto *allocations =
      new std::unordered_map<CUdeviceptr, lupine_deviceptr_allocation_record>();
  return *allocations;
}

static int lupine_conn_index(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  int index = conn->logical_index;
  return index >= 0 && index < rpc_size() ? index : -1;
}

conn_t *lupine_thread_conn_by_index(unsigned int index) {
  // The Linux server forks one child process per accepted connection. CUDA
  // object handles, including primary-context handles used by libcudart, are
  // only valid inside that child process. Keep all client host threads on the
  // same connection and mirror thread-local current context with
  // cuCtxSetCurrent.
  return rpc_client_get_connection(index);
}

extern "C" bool lupine_route_is_local(lupine_route route) {
  return route.kind == LUPINE_ROUTE_LOCAL;
}

extern "C" conn_t *lupine_route_remote_conn(lupine_route route) {
  return route.kind == LUPINE_ROUTE_REMOTE ? route.conn : nullptr;
}

extern "C" bool lupine_local_cuda_symbol_if_routed(lupine_route route,
                                                   const char *symbol,
                                                   void **symbol_out) {
  if (!lupine_route_is_local(route)) {
    return false;
  }
  if (symbol_out != nullptr) {
    *symbol_out = lupine_real_cuda_symbol(symbol);
  }
  return true;
}

lupine_route lupine_remote_route_for_conn(conn_t *conn) {
  if (conn == nullptr) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }
  return lupine_route{LUPINE_ROUTE_REMOTE, conn};
}

static lupine_route lupine_local_route() {
  return lupine_route{LUPINE_ROUTE_LOCAL, nullptr};
}

static lupine_route lupine_route_for_owner(const lupine_owner &owner) {
  if (owner.local) {
    return lupine_local_route();
  }
  return lupine_remote_route_for_conn(
      lupine_thread_conn_by_index(owner.conn_index));
}

int lupine_route_identity(lupine_route route) {
  if (route.kind == LUPINE_ROUTE_LOCAL) {
    return -1;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    int index = lupine_conn_index(route.conn);
    return index >= 0 ? index : -2;
  }
  return -2;
}

extern "C" bool lupine_routes_share_server(lupine_route first,
                                           lupine_route second) {
  return lupine_route_identity(first) == lupine_route_identity(second);
}

extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first,
                                              CUdeviceptr second) {
  return lupine_routes_share_server(lupine_route_for_deviceptr(first),
                                    lupine_route_for_deviceptr(second));
}

lupine_route lupine_route_from_identity(int route_id) {
  if (route_id == -1) {
    return lupine_local_route();
  }
  if (route_id >= 0) {
    return lupine_remote_route_for_conn(
        lupine_thread_conn_by_index(static_cast<unsigned int>(route_id)));
  }
  return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
}

int lupine_known_deviceptr_route_id(CUdeviceptr ptr) {
  if (ptr == 0) {
    return -2;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto it = lupine_deviceptr_owners().find(ptr);
  if (it != lupine_deviceptr_owners().end()) {
    return it->second.local ? -1 : static_cast<int>(it->second.conn_index);
  }
  for (const auto &entry : lupine_deviceptr_allocations()) {
    const auto &allocation = entry.second;
    if (allocation.base == 0 || allocation.size == 0 || ptr < allocation.base) {
      continue;
    }
    uint64_t offset = static_cast<uint64_t>(ptr - allocation.base);
    if (offset < allocation.size) {
      return allocation.owner.local
                 ? -1
                 : static_cast<int>(allocation.owner.conn_index);
    }
  }
  return -2;
}

lupine_route lupine_route_from_known_kernel_deviceptr_args(
    const unsigned char *packed, size_t packed_size,
    const lupine_kernel_param_layout &layout, lupine_route fallback) {
  int route_id = -2;
  for (uint32_t i = 0; i < layout.count; ++i) {
    if (layout.sizes[i] != sizeof(CUdeviceptr) ||
        layout.offsets[i] + sizeof(CUdeviceptr) > packed_size) {
      continue;
    }
    CUdeviceptr ptr = 0;
    memcpy(&ptr, packed + layout.offsets[i], sizeof(ptr));
    int ptr_route_id = lupine_known_deviceptr_route_id(ptr);
    if (ptr_route_id == -2) {
      continue;
    }
    if (route_id == -2) {
      route_id = ptr_route_id;
    } else if (route_id != ptr_route_id) {
      return fallback;
    }
  }
  if (route_id == -2 || route_id == lupine_route_identity(fallback)) {
    return fallback;
  }
  lupine_route route = lupine_route_from_identity(route_id);
  return route.kind == LUPINE_ROUTE_INVALID ? fallback : route;
}

static CUresult lupine_remote_cuDeviceGetCount(conn_t *conn, int *count) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || count == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetCount) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, count, sizeof(*count)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult lupine_remote_cuDeviceGet(conn_t *conn, CUdevice *device,
                                          int ordinal) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || device == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGet) < 0 ||
      rpc_write(conn, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, device, sizeof(*device)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult lupine_ensure_device_table() {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (lupine_device_table_ready()) {
    return CUDA_SUCCESS;
  }

  auto &devices = lupine_device_table();
  devices.clear();

  using cuDeviceGetCount_fn = CUresult (*)(int *);
  using cuDeviceGet_fn = CUresult (*)(CUdevice *, int);
  auto local_count_fn =
      lupine_real_cuda_fn<cuDeviceGetCount_fn>("cuDeviceGetCount");
  auto local_get_fn = lupine_real_cuda_fn<cuDeviceGet_fn>("cuDeviceGet");
  if (local_count_fn != nullptr && local_get_fn != nullptr) {
    int local_count = 0;
    if (local_count_fn(&local_count) == CUDA_SUCCESS && local_count > 0) {
      for (int ordinal = 0; ordinal < local_count; ++ordinal) {
        CUdevice local_device = 0;
        if (local_get_fn(&local_device, ordinal) == CUDA_SUCCESS) {
          lupine_device_entry entry;
          entry.local = true;
          entry.local_ordinal = ordinal;
          entry.local_device = local_device;
          devices.push_back(entry);
        }
      }
    }
  }

  if (rpc_open() == 0) {
    int connection_count = rpc_size();
    for (int i = 0; i < connection_count; ++i) {
      conn_t *conn = rpc_client_get_connection(static_cast<unsigned int>(i));
      int remote_count = 0;
      CUresult result = lupine_remote_cuDeviceGetCount(conn, &remote_count);
      if (result != CUDA_SUCCESS) {
        devices.clear();
        return result;
      }
      for (int ordinal = 0; ordinal < remote_count; ++ordinal) {
        CUdevice remote_device = 0;
        result = lupine_remote_cuDeviceGet(conn, &remote_device, ordinal);
        if (result != CUDA_SUCCESS) {
          devices.clear();
          return result;
        }
        lupine_device_entry entry;
        entry.local = false;
        entry.conn_index = static_cast<unsigned int>(i);
        entry.remote_ordinal = ordinal;
        entry.remote_device = remote_device;
        devices.push_back(entry);
      }
    }
  }

  if (devices.empty()) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_device_table_ready() = true;
  return CUDA_SUCCESS;
}

CUresult lupine_virtual_device_count(int *count) {
  if (count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  *count = static_cast<int>(lupine_device_table().size());
  return CUDA_SUCCESS;
}

CUresult lupine_virtual_device_for_ordinal(CUdevice *device, int ordinal) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (ordinal < 0 ||
      ordinal >= static_cast<int>(lupine_device_table().size())) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = static_cast<CUdevice>(ordinal);
  return CUDA_SUCCESS;
}

static CUdevice lupine_virtual_device_for_route(lupine_route route,
                                                CUdevice route_device) {
  int conn_index = -1;
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    conn_index = lupine_conn_index(route.conn);
    if (conn_index < 0) {
      return -1;
    }
  } else if (route.kind != LUPINE_ROUTE_LOCAL) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto &devices = lupine_device_table();
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto &candidate = devices[i];
    bool matches =
        route.kind == LUPINE_ROUTE_LOCAL
            ? candidate.local && candidate.local_device == route_device
            : !candidate.local &&
                  candidate.conn_index ==
                      static_cast<unsigned int>(conn_index) &&
                  candidate.remote_device == route_device;
    if (matches) {
      return static_cast<CUdevice>(i);
    }
  }
  return -1;
}

extern "C" CUresult
lupine_lookup_device_on_all_routes_impl(CUdevice *device, void *context,
                                        lupine_device_lookup_callback lookup) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = lupine_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  bool has_local_device = false;
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    has_local_device =
        std::any_of(lupine_device_table().begin(), lupine_device_table().end(),
                    [](const auto &candidate) { return candidate.local; });
  }

  CUresult first_error = CUDA_ERROR_INVALID_DEVICE;
  int connection_count = rpc_size();
  for (int i = has_local_device ? -1 : 0; i < connection_count; ++i) {
    lupine_route route =
        i < 0 ? lupine_local_route()
              : lupine_remote_route_for_conn(rpc_client_get_connection(
                    static_cast<unsigned int>(i)));
    CUdevice route_device = 0;
    result = lookup(context, route, &route_device);
    if (result != CUDA_SUCCESS) {
      if (first_error == CUDA_ERROR_INVALID_DEVICE &&
          result != CUDA_ERROR_INVALID_DEVICE) {
        first_error = result;
      }
      continue;
    }

    CUdevice virtual_device =
        lupine_virtual_device_for_route(route, route_device);
    if (virtual_device < 0) {
      if (first_error == CUDA_ERROR_INVALID_DEVICE) {
        first_error = CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      continue;
    }

    *device = virtual_device;
    return CUDA_SUCCESS;
  }
  return first_error;
}

extern "C" conn_t *lupine_rpc_conn_for_device(CUdevice *device) {
  if (device == nullptr || lupine_ensure_device_table() != CUDA_SUCCESS) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  int local = static_cast<int>(*device);
  auto &devices = lupine_device_table();
  if (local < 0 || local >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  const lupine_device_entry &mapped = devices[local];
  if (mapped.local) {
    *device = mapped.local_device;
    return nullptr;
  }
  *device = mapped.remote_device;
  return lupine_thread_conn_by_index(mapped.conn_index);
}

extern "C" lupine_route lupine_route_for_device(CUdevice *device) {
  if (device == nullptr || lupine_ensure_device_table() != CUDA_SUCCESS) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }

  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  int local = static_cast<int>(*device);
  auto &devices = lupine_device_table();
  if (local < 0 || local >= static_cast<int>(devices.size())) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }
  const lupine_device_entry &mapped = devices[local];
  if (mapped.local) {
    *device = mapped.local_device;
    return lupine_local_route();
  }
  *device = mapped.remote_device;
  return lupine_remote_route_for_conn(
      lupine_thread_conn_by_index(mapped.conn_index));
}

extern "C" bool lupine_translate_device_for_conn(conn_t *conn,
                                                 CUdevice *device) {
  if (conn == nullptr || device == nullptr ||
      lupine_ensure_device_table() != CUDA_SUCCESS) {
    return false;
  }

  int conn_index = lupine_conn_index(conn);
  if (conn_index < 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  int local = static_cast<int>(*device);
  auto &devices = lupine_device_table();
  if (local < 0 || local >= static_cast<int>(devices.size())) {
    return false;
  }
  const lupine_device_entry &mapped = devices[local];
  if (mapped.local ||
      mapped.conn_index != static_cast<unsigned int>(conn_index)) {
    return false;
  }
  *device = mapped.remote_device;
  return true;
}

extern "C" CUdevice lupine_local_device_for_remote(conn_t *conn,
                                                   CUdevice remote_device) {
  if (conn == nullptr || lupine_ensure_device_table() != CUDA_SUCCESS) {
    return -1;
  }
  return lupine_virtual_device_for_route(lupine_remote_route_for_conn(conn),
                                         remote_device);
}

extern "C" void lupine_note_context_owner(CUcontext ctx, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (ctx == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_context_owners()[ctx] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_module_owner(CUmodule module, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (module == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_module_owners()[module] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_library_owner(CUlibrary library, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (library == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_library_owners()[library] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_function_owner(CUfunction function, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (function == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_function_owners()[function] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_stream_owner(CUstream stream, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (stream == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_stream_owners()[stream] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_event_owner(CUevent event, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (event == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_event_owners()[event] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_memory_pool_owner(CUmemoryPool pool, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (pool == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_memory_pool_owners()[pool] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_graph_owner(CUgraph graph, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (graph == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_owners()[graph] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_graph_node_owner(CUgraphNode node, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (node == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_node_owners()[node] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_graph_exec_owner(CUgraphExec exec, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (exec == nullptr || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_exec_owners()[exec] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

extern "C" void lupine_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (ptr == 0 || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_deviceptr_owners()[ptr] =
      lupine_owner{false, static_cast<unsigned int>(index)};
}

static void lupine_note_deviceptr_allocation_owner_locked(CUdeviceptr ptr,
                                                          size_t size,
                                                          lupine_owner owner) {
  lupine_deviceptr_owners()[ptr] = owner;
  if (size == 0) {
    lupine_deviceptr_allocations().erase(ptr);
    return;
  }
  lupine_deviceptr_allocations()[ptr] =
      lupine_deviceptr_allocation_record{ptr, size, owner};
}

extern "C" void lupine_note_deviceptr_allocation(CUdeviceptr ptr, size_t size,
                                                 conn_t *conn) {
  int index = lupine_conn_index(conn);
  if (ptr == 0 || index < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_note_deviceptr_allocation_owner_locked(
      ptr, size, lupine_owner{false, static_cast<unsigned int>(index)});
}

extern "C" void lupine_note_context_owner_route(CUcontext ctx,
                                                lupine_route route) {
  if (ctx == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_context_owner(ctx, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_context_owners()[ctx] = lupine_owner{true, 0};
}

extern "C" void lupine_note_module_owner_route(CUmodule module,
                                               lupine_route route) {
  if (module == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_module_owner(module, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_module_owners()[module] = lupine_owner{true, 0};
}

extern "C" void lupine_note_library_owner_route(CUlibrary library,
                                                lupine_route route) {
  if (library == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_library_owner(library, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_library_owners()[library] = lupine_owner{true, 0};
}

extern "C" void lupine_note_function_owner_route(CUfunction function,
                                                 lupine_route route) {
  if (function == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_function_owner(function, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_function_owners()[function] = lupine_owner{true, 0};
}

extern "C" void lupine_note_stream_owner_route(CUstream stream,
                                               lupine_route route) {
  if (stream == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_stream_owner(stream, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_stream_owners()[stream] = lupine_owner{true, 0};
}

extern "C" void lupine_note_event_owner_route(CUevent event,
                                              lupine_route route) {
  if (event == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_event_owner(event, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_event_owners()[event] = lupine_owner{true, 0};
}

extern "C" void lupine_note_memory_pool_owner_route(CUmemoryPool pool,
                                                    lupine_route route) {
  if (pool == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_memory_pool_owner(pool, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_memory_pool_owners()[pool] = lupine_owner{true, 0};
}

extern "C" void lupine_note_graph_owner_route(CUgraph graph,
                                              lupine_route route) {
  if (graph == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_graph_owner(graph, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_owners()[graph] = lupine_owner{true, 0};
}

extern "C" void lupine_note_graph_node_owner_route(CUgraphNode node,
                                                   lupine_route route) {
  if (node == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_graph_node_owner(node, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_node_owners()[node] = lupine_owner{true, 0};
}

extern "C" void lupine_note_graph_exec_owner_route(CUgraphExec exec,
                                                   lupine_route route) {
  if (exec == nullptr || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_graph_exec_owner(exec, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_graph_exec_owners()[exec] = lupine_owner{true, 0};
}

extern "C" void lupine_note_deviceptr_owner_route(CUdeviceptr ptr,
                                                  lupine_route route) {
  if (ptr == 0 || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_deviceptr_owner(ptr, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_deviceptr_owners()[ptr] = lupine_owner{true, 0};
}

extern "C" void lupine_note_deviceptr_allocation_route(CUdeviceptr ptr,
                                                       size_t size,
                                                       lupine_route route) {
  if (ptr == 0 || route.kind == LUPINE_ROUTE_INVALID) {
    return;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    lupine_note_deviceptr_allocation(ptr, size, route.conn);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_note_deviceptr_allocation_owner_locked(ptr, size,
                                                lupine_owner{true, 0});
}

extern "C" void lupine_forget_deviceptr_owner(CUdeviceptr ptr) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_deviceptr_owners().erase(ptr);
  lupine_deviceptr_allocations().erase(ptr);
}

static lupine_route lupine_route_for_context_if_known(CUcontext ctx) {
  if (ctx == nullptr) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }

  lupine_owner owner;
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_context_owners().find(ctx);
    if (it == lupine_context_owners().end()) {
      return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
    }
    owner = it->second;
  }
  return lupine_route_for_owner(owner);
}

extern "C" lupine_route lupine_route_for_context(CUcontext ctx) {
  if (ctx == nullptr) {
    return lupine_route_for_default();
  }
  lupine_route route = lupine_route_for_context_if_known(ctx);
  if (route.kind != LUPINE_ROUTE_INVALID) {
    return route;
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_module(CUmodule module) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_module_owners().find(module);
    if (it != lupine_module_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_library(CUlibrary library) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_library_owners().find(library);
    if (it != lupine_library_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_function(CUfunction function) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_function_owners().find(function);
    if (it != lupine_function_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_stream(CUstream stream) {
  if (stream == nullptr) {
    return lupine_route_for_default();
  }
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_stream_owners().find(stream);
    if (it != lupine_stream_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_event(CUevent event) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_event_owners().find(event);
    if (it != lupine_event_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_memory_pool(CUmemoryPool pool) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_memory_pool_owners().find(pool);
    if (it != lupine_memory_pool_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_graph(CUgraph graph) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_graph_owners().find(graph);
    if (it != lupine_graph_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_graph_node(CUgraphNode node) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_graph_node_owners().find(node);
    if (it != lupine_graph_node_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_graph_exec(CUgraphExec exec) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_graph_exec_owners().find(exec);
    if (it != lupine_graph_exec_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  return lupine_route_for_default();
}

extern "C" lupine_route lupine_route_for_deviceptr(CUdeviceptr ptr) {
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_deviceptr_owners().find(ptr);
    if (it != lupine_deviceptr_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
    for (const auto &entry : lupine_deviceptr_allocations()) {
      const auto &allocation = entry.second;
      if (allocation.base == 0 || allocation.size == 0 ||
          ptr < allocation.base) {
        continue;
      }
      uint64_t offset = static_cast<uint64_t>(ptr - allocation.base);
      if (offset < allocation.size) {
        return lupine_route_for_owner(allocation.owner);
      }
    }
  }
  return lupine_route_for_default();
}

CUresult lupine_set_current_context_on_route(lupine_route route,
                                             CUcontext ctx) {
  if (route.kind == LUPINE_ROUTE_INVALID) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  uint64_t epoch = lupine_lane_context_cache_epoch();
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUcontext);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxSetCurrent");
    result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(ctx);
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuCtxSetCurrent) < 0 ||
        rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  lupine_lane_context_cache_update(lupine_route_identity(route), ctx, epoch,
                                   result == CUDA_SUCCESS);
  return result;
}

extern "C" lupine_route lupine_route_for_current_context() {
  return lupine_route_for_context(lupine_current_context_hint());
}

static lupine_route lupine_route_for_default_context_hint(CUcontext ctx) {
  if (ctx == nullptr) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }

  lupine_route route = lupine_route_for_context_if_known(ctx);
  if (route.kind == LUPINE_ROUTE_INVALID) {
    return route;
  }

  if (!lupine_lane_context_cache_matches(lupine_route_identity(route), ctx)) {
    CUresult result = lupine_set_current_context_on_route(route, ctx);
    if (result != CUDA_SUCCESS) {
      return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
    }
  }
  lupine_accept_current_context_hint(ctx);
  return route;
}

extern "C" lupine_route lupine_route_for_default() {
  CUcontext current_hint = lupine_current_context_hint();
  if (current_hint != nullptr) {
    lupine_route route = lupine_route_for_default_context_hint(current_hint);
    if (route.kind != LUPINE_ROUTE_INVALID) {
      return route;
    }
  }
  CUcontext default_hint = lupine_default_context_hint_value();
  if (default_hint != nullptr) {
    lupine_route route = lupine_route_for_default_context_hint(default_hint);
    if (route.kind != LUPINE_ROUTE_INVALID) {
      return route;
    }
  }
  CUcontext global_hint = lupine_global_default_context_hint_value();
  if (global_hint != nullptr) {
    lupine_route route = lupine_route_for_default_context_hint(global_hint);
    if (route.kind != LUPINE_ROUTE_INVALID) {
      return route;
    }
  }
  conn_t *conn = lupine_thread_conn_by_index(0);
  if (conn != nullptr) {
    return lupine_remote_route_for_conn(conn);
  }
  return lupine_local_cuda_available()
             ? lupine_local_route()
             : lupine_route{LUPINE_ROUTE_INVALID, nullptr};
}

extern "C" conn_t *lupine_rpc_conn_for_current_context() {
  return lupine_route_remote_conn(lupine_route_for_current_context());
}

extern "C" conn_t *lupine_rpc_conn_for_context(CUcontext ctx) {
  return lupine_route_remote_conn(lupine_route_for_context(ctx));
}

extern "C" conn_t *lupine_rpc_conn_for_module(CUmodule module) {
  return lupine_route_remote_conn(lupine_route_for_module(module));
}

extern "C" conn_t *lupine_rpc_conn_for_function(CUfunction function) {
  return lupine_route_remote_conn(lupine_route_for_function(function));
}

extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream stream) {
  return lupine_route_remote_conn(lupine_route_for_stream(stream));
}

extern "C" conn_t *lupine_rpc_conn_for_event(CUevent event) {
  return lupine_route_remote_conn(lupine_route_for_event(event));
}

extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr) {
  return lupine_route_remote_conn(lupine_route_for_deviceptr(ptr));
}
