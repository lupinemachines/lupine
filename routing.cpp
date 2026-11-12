#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "cache.h"
#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"
#include "events.h"
#include "visible_devices.h"

extern int rpc_open();
extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);

struct lupine_device_entry {
  bool local = false;
  CUdevice local_device = -1;
  unsigned int conn_index = 0;
  CUdevice remote_device = 0;
};

struct lupine_owner_record {
  int route_id = -2;
  bool is_green_context = false;
  CUcontext pointer_context = nullptr;
};

static std::mutex &lupine_routing_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::vector<lupine_device_entry> &lupine_device_table() {
  static auto *devices = new std::vector<lupine_device_entry>();
  return *devices;
}

static std::atomic<bool> &lupine_device_table_ready() {
  static std::atomic<bool> ready{false};
  return ready;
}

template <typename Handle>
static std::unordered_map<Handle, lupine_owner_record> &lupine_owners() {
  static auto *owners = new std::unordered_map<Handle, lupine_owner_record>();
  return *owners;
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

lupine_route lupine_remote_route_for_conn(conn_t *conn) {
  if (conn == nullptr) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }
  return lupine_route{LUPINE_ROUTE_REMOTE, conn};
}

static lupine_route lupine_local_route() {
  return lupine_route{LUPINE_ROUTE_LOCAL, nullptr};
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

static CUresult lupine_remote_device_query(conn_t *conn, int op,
                                           const void *arg, size_t arg_size,
                                           void *out, size_t out_size) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (out == nullptr || lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, op, RPC_SCOPE_NONE) < 0 ||
      (arg_size != 0 && rpc_write(conn, arg, arg_size) < 0) ||
      rpc_wait_for_response(conn) < 0 || rpc_read(conn, out, out_size) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static std::string lupine_uuid_hex(const CUuuid &uuid) {
  static const char *digits = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (int i = 0; i < 16; ++i) {
    const unsigned char byte = static_cast<unsigned char>(uuid.bytes[i]);
    out.push_back(digits[(byte >> 4) & 0xf]);
    out.push_back(digits[byte & 0xf]);
  }
  return out;
}

// Narrows the table to what CUDA_VISIBLE_DEVICES leaves visible, in the order
// it lists. The table is the ordinal space, so dropping an entry here is what
// renumbers the rest and puts a device beyond a client's reach.
static void
lupine_apply_visible_devices(std::vector<lupine_device_entry> &devices) {
  const char *spec = getenv("CUDA_VISIBLE_DEVICES");
  if (spec == nullptr) {
    return;
  }
  const std::vector<int> visible = lupine_select_visible_devices(
      spec, static_cast<int>(devices.size()), [&devices](int index) {
        const lupine_device_entry &entry = devices[static_cast<size_t>(index)];
        CUuuid uuid = {};
        const CUresult result =
            entry.local ? lupine_call_real_cuda_fn("cuDeviceGetUuid_v2", &uuid,
                                                   entry.local_device)
                        : lupine_remote_device_query(
                              rpc_client_get_connection(entry.conn_index),
                              RPC_cuDeviceGetUuid_v2, &entry.remote_device,
                              sizeof(entry.remote_device), &uuid, 16);
        // A device that will not say what it is simply matches no UUID.
        return result == CUDA_SUCCESS ? lupine_uuid_hex(uuid) : std::string();
      });

  std::vector<lupine_device_entry> selected;
  selected.reserve(visible.size());
  for (const int index : visible) {
    selected.push_back(devices[static_cast<size_t>(index)]);
  }
  devices.swap(selected);
}

static CUresult lupine_ensure_device_table() {
  if (lupine_device_table_ready().load(std::memory_order_acquire)) {
    return CUDA_SUCCESS;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (lupine_device_table_ready().load(std::memory_order_relaxed)) {
    return CUDA_SUCCESS;
  }

  auto &devices = lupine_device_table();
  devices.clear();

  if (lupine_local_cuda_available()) {
    int local_count = 0;
    if (lupine_call_real_cuda_fn("cuDeviceGetCount", &local_count) ==
        CUDA_SUCCESS) {
      for (int ordinal = 0; ordinal < local_count; ++ordinal) {
        CUdevice local_device = 0;
        if (lupine_call_real_cuda_fn("cuDeviceGet", &local_device, ordinal) ==
            CUDA_SUCCESS) {
          devices.push_back({true, local_device, 0, 0});
        }
      }
    }
  }

  if (rpc_open() == 0) {
    int connection_count = rpc_size();
    for (int i = 0; i < connection_count; ++i) {
      conn_t *conn = rpc_client_get_connection(static_cast<unsigned int>(i));
      int remote_count = 0;
      CUresult result =
          lupine_remote_device_query(conn, RPC_cuDeviceGetCount, nullptr, 0,
                                     &remote_count, sizeof(remote_count));
      if (result != CUDA_SUCCESS) {
        devices.clear();
        return result;
      }
      for (int ordinal = 0; ordinal < remote_count; ++ordinal) {
        CUdevice remote_device = 0;
        result = lupine_remote_device_query(conn, RPC_cuDeviceGet, &ordinal,
                                            sizeof(ordinal), &remote_device,
                                            sizeof(remote_device));
        if (result != CUDA_SUCCESS) {
          devices.clear();
          return result;
        }
        devices.push_back(
            {false, -1, static_cast<unsigned int>(i), remote_device});
      }
    }
  }

  if (devices.empty()) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  // Selecting nothing out of a fleet that is there is an answer, not a
  // failure: the driver reports no devices rather than refusing the call.
  lupine_apply_visible_devices(devices);
  lupine_device_table_ready().store(true, std::memory_order_release);
  return CUDA_SUCCESS;
}

extern "C" bool
lupine_connection_supports_concurrent_managed_access(conn_t *conn) {
  // Identity mappings belong to one server connection. Managed pointers can
  // be shared only when every visible device routes through their owner.
  if (conn == nullptr || conn->va_size == 0 ||
      lupine_ensure_device_table() != CUDA_SUCCESS) {
    return false;
  }

  int index = lupine_conn_index(conn);
  if (index < 0) {
    return false;
  }

  const auto &devices = lupine_device_table();
  return !devices.empty() &&
         std::all_of(
             devices.begin(), devices.end(), [index](const auto &device) {
               return !device.local &&
                      device.conn_index == static_cast<unsigned int>(index);
             });
}

CUresult lupine_virtual_device_count(int *count) {
  if (count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }
  *count = static_cast<int>(lupine_device_table().size());
  return CUDA_SUCCESS;
}

bool lupine_devices_span_routes() {
  if (lupine_ensure_device_table() != CUDA_SUCCESS) {
    return false;
  }
  const auto &devices = lupine_device_table();
  return std::any_of(devices.begin(), devices.end(), [&](const auto &entry) {
    return entry.local != devices.front().local ||
           (!entry.local && entry.conn_index != devices.front().conn_index);
  });
}

static const lupine_device_entry *lupine_lookup_device_entry(int ordinal) {
  const auto &devices = lupine_device_table();
  return ordinal >= 0 && ordinal < static_cast<int>(devices.size())
             ? &devices[static_cast<size_t>(ordinal)]
             : nullptr;
}

CUresult lupine_virtual_device_for_ordinal(CUdevice *device, int ordinal) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_ensure_device_table();
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (lupine_lookup_device_entry(ordinal) == nullptr) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  *device = static_cast<CUdevice>(ordinal);
  return CUDA_SUCCESS;
}

CUdevice lupine_virtual_device_for_route(lupine_route route,
                                         CUdevice route_device) {
  int conn_index =
      route.kind == LUPINE_ROUTE_REMOTE ? lupine_conn_index(route.conn) : -1;
  if (lupine_ensure_device_table() != CUDA_SUCCESS ||
      (route.kind != LUPINE_ROUTE_LOCAL && conn_index < 0)) {
    return -1;
  }
  const auto &devices = lupine_device_table();
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto &c = devices[i];
    if (route.kind == LUPINE_ROUTE_LOCAL
            ? (c.local && c.local_device == route_device)
            : (!c.local &&
               c.conn_index == static_cast<unsigned int>(conn_index) &&
               c.remote_device == route_device)) {
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

  const bool has_local_device =
      std::any_of(lupine_device_table().begin(), lupine_device_table().end(),
                  [](const auto &candidate) { return candidate.local; });
  CUresult first_error = CUDA_ERROR_INVALID_DEVICE;
  int connection_count = rpc_size();
  for (int i = has_local_device ? -1 : 0; i < connection_count; ++i) {
    lupine_route route =
        i < 0 ? lupine_local_route()
              : lupine_remote_route_for_conn(
                    rpc_client_get_connection(static_cast<unsigned int>(i)));
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
    if (virtual_device >= 0) {
      *device = virtual_device;
      return CUDA_SUCCESS;
    }
  }
  return first_error;
}

extern "C" conn_t *lupine_rpc_conn_for_device(CUdevice *device) {
  return lupine_route_remote_conn(lupine_route_for_device(device));
}

extern "C" lupine_route lupine_route_for_device(CUdevice *device) {
  if (device == nullptr || lupine_ensure_device_table() != CUDA_SUCCESS) {
    return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
  }
  const lupine_device_entry *mapped =
      lupine_lookup_device_entry(static_cast<int>(*device));
  if (mapped == nullptr) {
    return lupine_route{LUPINE_ROUTE_UNKNOWN_DEVICE, nullptr};
  }
  *device = mapped->local ? mapped->local_device : mapped->remote_device;
  return mapped->local ? lupine_local_route()
                       : lupine_remote_route_for_conn(
                             lupine_thread_conn_by_index(mapped->conn_index));
}

extern "C" bool lupine_translate_device_for_conn(conn_t *conn,
                                                 CUdevice *device) {
  int conn_index = lupine_conn_index(conn);
  if (conn_index < 0 || device == nullptr ||
      lupine_ensure_device_table() != CUDA_SUCCESS) {
    return false;
  }
  const lupine_device_entry *mapped =
      lupine_lookup_device_entry(static_cast<int>(*device));
  if (mapped == nullptr || mapped->local ||
      mapped->conn_index != static_cast<unsigned int>(conn_index)) {
    return false;
  }
  *device = mapped->remote_device;
  return true;
}

extern "C" CUdevice lupine_local_device_for_remote(conn_t *conn,
                                                   CUdevice remote_device) {
  return conn == nullptr
             ? -1
             : lupine_virtual_device_for_route(
                   lupine_remote_route_for_conn(conn), remote_device);
}

struct alignas(64) lupine_routing_fast_state {
  std::atomic<uint64_t> handle_epoch{1};
  std::atomic<uint64_t> devptr_epoch{1};
  std::atomic<uint64_t> blocking_epoch{1};
  std::atomic<uint64_t> green_context_count{0};
  std::atomic<uint64_t> devptr_count{0};
  std::atomic<uint64_t> devptr_min{UINT64_MAX};
  std::atomic<uint64_t> devptr_last{0};
  std::atomic<bool> sentinel_stream_owned{false};
};
static_assert(sizeof(lupine_routing_fast_state) == 64,
              "fast routing state must fit in one cache line");

static lupine_routing_fast_state lupine_fast_state;

struct lupine_tls_handle_slot {
  uint64_t epoch;
  uintptr_t bits;
  int route_id;
  uint8_t tag;
  bool is_green;
};

struct lupine_tls_stream_blocking_slot {
  uint64_t epoch;
  uintptr_t key;
  bool blocking;
};

struct lupine_tls_alloc_slot {
  CUdeviceptr base;
  size_t size;
  CUcontext context;
  int route_id;
};

struct lupine_tls_gap_slot {
  CUdeviceptr base;
  CUdeviceptr last;
};

struct alignas(64) lupine_tls_routing_cache {
  uint64_t devptr_epoch;
  uint8_t alloc_hand;
  uint8_t gap_hand;
  std::array<lupine_tls_alloc_slot, 4> allocs;
  std::array<lupine_tls_gap_slot, 2> gaps;
  std::array<lupine_tls_stream_blocking_slot, 16> blocking_streams;
  std::array<lupine_tls_handle_slot, 32> handles;
};

static thread_local lupine_tls_routing_cache lupine_tls_cache = {};

static inline size_t lupine_splitmix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<size_t>(x ^ (x >> 31));
}

struct lupine_devptr_range_table {
  std::map<CUdeviceptr, size_t> ranges;
  size_t overlaps = 0;
};

static lupine_devptr_range_table &lupine_devptr_ranges() {
  static auto *table = new lupine_devptr_range_table();
  return *table;
}

static bool lupine_erase_devptr_range_locked(CUdeviceptr base) {
  auto &t = lupine_devptr_ranges();
  auto it = t.ranges.find(base);
  if (it == t.ranges.end()) {
    return false;
  }
  auto ov = [&](auto a, auto b) {
    return a != t.ranges.end() && b != t.ranges.end() &&
           b->first - a->first < a->second;
  };
  auto prev = it == t.ranges.begin() ? t.ranges.end() : std::prev(it);
  auto next = std::next(it);
  t.overlaps = t.overlaps - ov(prev, it) - ov(it, next) + ov(prev, next);
  t.ranges.erase(it);
  return true;
}

static void lupine_insert_devptr_range_locked(CUdeviceptr base, size_t size) {
  lupine_erase_devptr_range_locked(base);
  if (base == 0 || size == 0) {
    return;
  }
  auto &t = lupine_devptr_ranges();
  auto ov = [&](auto a, auto b) {
    return a != t.ranges.end() && b != t.ranges.end() &&
           b->first - a->first < a->second;
  };
  auto next = t.ranges.lower_bound(base);
  auto prev = next == t.ranges.begin() ? t.ranges.end() : std::prev(next);
  auto ins = t.ranges.emplace_hint(next, base, size);
  t.overlaps = t.overlaps - ov(prev, next) + ov(prev, ins) + ov(ins, next);
}

static uint64_t lupine_devptr_last_byte(CUdeviceptr base, size_t size) {
  uint64_t b = static_cast<uint64_t>(base);
  return size <= 1 ? b
                   : (b <= UINT64_MAX - (size - 1) ? b + size - 1 : UINT64_MAX);
}

static void lupine_sync_devptr_metadata_locked(uint64_t lo, uint64_t hi) {
  const auto &owners = lupine_owners<CUdeviceptr>();
  const auto &t = lupine_devptr_ranges();
  uint64_t count = owners.size() + t.ranges.size();
  if (count == 0) {
    lupine_fast_state.devptr_epoch.fetch_add(1, std::memory_order_acq_rel);
    lupine_fast_state.devptr_count.store(0, std::memory_order_release);
    lupine_fast_state.devptr_min.store(UINT64_MAX, std::memory_order_release);
    lupine_fast_state.devptr_last.store(0, std::memory_order_release);
    return;
  }
  if (owners.empty() && t.overlaps == 0) {
    auto back = std::prev(t.ranges.end());
    lupine_fast_state.devptr_min.store(t.ranges.begin()->first,
                                       std::memory_order_release);
    lupine_fast_state.devptr_last.store(
        lupine_devptr_last_byte(back->first, back->second),
        std::memory_order_release);
  } else if (lo != 0) {
    if (lo < lupine_fast_state.devptr_min.load(std::memory_order_relaxed)) {
      lupine_fast_state.devptr_min.store(lo, std::memory_order_release);
    }
    if (hi > lupine_fast_state.devptr_last.load(std::memory_order_relaxed)) {
      lupine_fast_state.devptr_last.store(hi, std::memory_order_release);
    }
  }
  lupine_fast_state.devptr_epoch.fetch_add(1, std::memory_order_acq_rel);
  lupine_fast_state.devptr_count.store(count, std::memory_order_release);
}

static void lupine_note_deviceptr_allocation_owner_locked(CUdeviceptr ptr,
                                                          size_t size,
                                                          int route_id,
                                                          CUcontext context) {
  if (size == 0) {
    auto &owner = lupine_owners<CUdeviceptr>()[ptr];
    owner.route_id = route_id;
    owner.pointer_context = context;
  } else {
    lupine_owners<CUdeviceptr>().erase(ptr);
  }
  lupine_insert_devptr_range_locked(ptr, size);
  lupine_deviceptr_allocation_cache_insert(ptr, size, route_id, context);
  lupine_sync_devptr_metadata_locked(ptr, lupine_devptr_last_byte(ptr, size));
}

#define LUPINE_FOR_EACH_OWNED_HANDLE(X)                                        \
  X(context, CUcontext, 1)                                                     \
  X(module, CUmodule, 2)                                                       \
  X(library, CUlibrary, 3)                                                     \
  X(function, CUfunction, 4)                                                   \
  X(stream, CUstream, 5)                                                       \
  X(event, CUevent, 6)                                                         \
  X(memory_pool, CUmemoryPool, 7)                                              \
  X(graph, CUgraph, 8)                                                         \
  X(graph_node, CUgraphNode, 9)                                                \
  X(graph_exec, CUgraphExec, 10)

template <typename Handle> static constexpr uint8_t lupine_handle_tag() {
  return 0;
}
#define LUPINE_DEFINE_HANDLE_TAG(name, type, id)                               \
  template <> constexpr uint8_t lupine_handle_tag<type>() { return id; }
LUPINE_FOR_EACH_OWNED_HANDLE(LUPINE_DEFINE_HANDLE_TAG)
#undef LUPINE_DEFINE_HANDLE_TAG

template <typename Handle>
static const lupine_tls_handle_slot *lupine_lookup_handle_owner(Handle handle) {
  constexpr uint8_t tag = lupine_handle_tag<Handle>();
  static_assert(tag != 0, "untagged handle type");
  uintptr_t bits = reinterpret_cast<uintptr_t>(handle);
  if (bits == 0) {
    return nullptr;
  }
  if constexpr (std::is_same_v<Handle, CUstream>) {
    if (bits <= LUPINE_STREAM_FETCH_SYNC_ALL_BITS &&
        !lupine_fast_state.sentinel_stream_owned.load(
            std::memory_order_acquire)) {
      return nullptr;
    }
  }
  uint64_t epoch =
      lupine_fast_state.handle_epoch.load(std::memory_order_acquire);
  size_t index = lupine_splitmix64(static_cast<uint64_t>(bits) ^
                                   (static_cast<uint64_t>(tag) << 56)) &
                 (lupine_tls_cache.handles.size() - 1);
  auto &slot = lupine_tls_cache.handles[index];
  if (slot.epoch == epoch && slot.bits == bits && slot.tag == tag) {
    return &slot;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  epoch = lupine_fast_state.handle_epoch.load(std::memory_order_relaxed);
  auto &owners = lupine_owners<Handle>();
  auto it = owners.find(handle);
  if (it == owners.end()) {
    return nullptr;
  }
  slot = {epoch, bits, it->second.route_id, tag, it->second.is_green_context};
  return &slot;
}

template <typename Handle>
static void lupine_note_owner_route(Handle handle, lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (handle == Handle{} || route_id < -1) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if constexpr (std::is_same_v<Handle, CUstream>) {
    if (reinterpret_cast<uintptr_t>(handle) <=
        LUPINE_STREAM_FETCH_SYNC_ALL_BITS) {
      lupine_fast_state.sentinel_stream_owned.store(true,
                                                    std::memory_order_release);
    }
  }
  auto [it, inserted] = lupine_owners<Handle>().try_emplace(handle);
  if constexpr (std::is_same_v<Handle, CUdeviceptr>) {
    it->second.route_id = route_id;
    it->second.pointer_context = lupine_current_context_hint();
    lupine_sync_devptr_metadata_locked(handle, handle);
  } else {
    if (!inserted && it->second.route_id != route_id) {
      lupine_fast_state.handle_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    it->second.route_id = route_id;
  }
}

extern "C" void lupine_note_module_owner(CUmodule module, conn_t *conn) {
  lupine_note_owner_route(module, lupine_remote_route_for_conn(conn));
}

extern "C" void lupine_note_library_owner(CUlibrary library, conn_t *conn) {
  lupine_note_owner_route(library, lupine_remote_route_for_conn(conn));
}

extern "C" void lupine_note_deviceptr_owner_route(CUdeviceptr ptr,
                                                  lupine_route route) {
  lupine_note_owner_route(ptr, route);
}

extern "C" void lupine_note_deviceptr_allocation_route(CUdeviceptr ptr,
                                                       size_t size,
                                                       lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (ptr == 0 || route_id < -1) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_note_deviceptr_allocation_owner_locked(ptr, size, route_id,
                                                lupine_current_context_hint());
}

extern "C" void lupine_note_deviceptr_allocation(CUdeviceptr ptr, size_t size,
                                                 conn_t *conn) {
  lupine_note_deviceptr_allocation_route(ptr, size,
                                         lupine_remote_route_for_conn(conn));
}

static bool lupine_note_disjoint_deviceptr_allocation(CUdeviceptr ptr,
                                                      size_t size,
                                                      lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (ptr == 0 || route_id < -1) {
    return true;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (lupine_deviceptr_allocation_cache_overlaps_other_route(ptr, size,
                                                             route_id)) {
    return false;
  }
  lupine_note_deviceptr_allocation_owner_locked(ptr, size, route_id,
                                                lupine_current_context_hint());
  return true;
}

static thread_local bool lupine_allocating_again = false;

extern "C" CUresult lupine_record_disjoint_allocation_impl(
    CUdeviceptr *ptr, size_t size, lupine_route route, void *context,
    lupine_disjoint_allocation_callback again) {
  // The repeated allocation re-enters here; this loop records its result.
  if (lupine_allocating_again) {
    return CUDA_SUCCESS;
  }
  CUresult result = CUDA_SUCCESS;
  while (result == CUDA_SUCCESS &&
         !lupine_note_disjoint_deviceptr_allocation(*ptr, size, route)) {
    lupine_allocating_again = true;
    result = again(context);
    lupine_allocating_again = false;
  }
  return result;
}

extern "C" void lupine_forget_deviceptr_owner(CUdeviceptr ptr) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  bool erased_owner = lupine_owners<CUdeviceptr>().erase(ptr) != 0;
  bool erased_range = lupine_erase_devptr_range_locked(ptr);
  lupine_deviceptr_allocation_cache_erase(ptr);
  if (erased_owner || erased_range) {
    lupine_sync_devptr_metadata_locked(0, 0);
  }
}

extern "C" void lupine_forget_context_owner(CUcontext ctx) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto &owners = lupine_owners<CUcontext>();
  auto it = owners.find(ctx);
  if (it != owners.end()) {
    if (it->second.is_green_context) {
      lupine_fast_state.green_context_count.fetch_sub(
          1, std::memory_order_acq_rel);
    }
    owners.erase(it);
    lupine_fast_state.handle_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
}

extern "C" void lupine_mark_context_green(CUcontext ctx) {
  if (ctx == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto [it, inserted] = lupine_owners<CUcontext>().try_emplace(ctx);
  if (!it->second.is_green_context) {
    it->second.is_green_context = true;
    if (!inserted) {
      lupine_fast_state.handle_epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    lupine_fast_state.green_context_count.fetch_add(1,
                                                    std::memory_order_acq_rel);
  }
}

extern "C" bool lupine_context_is_green(CUcontext ctx) {
  if (ctx == nullptr || lupine_fast_state.green_context_count.load(
                            std::memory_order_acquire) == 0) {
    return false;
  }
  const auto *slot = lupine_lookup_handle_owner(ctx);
  return slot != nullptr && slot->is_green;
}

namespace {

// Streams the legacy stream waits for: blocking streams this client created,
// and each live thread's per-thread default stream.
std::mutex &lupine_blocking_streams_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::vector<uintptr_t> &lupine_blocking_streams() {
  static auto *streams = new std::vector<uintptr_t>();
  return *streams;
}

std::atomic<size_t> lupine_blocking_stream_count{0};

void lupine_set_stream_blocking(uintptr_t key, bool blocking) {
  if (!blocking &&
      lupine_blocking_stream_count.load(std::memory_order_acquire) == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_blocking_streams_mutex());
  auto &streams = lupine_blocking_streams();
  auto it = std::find(streams.begin(), streams.end(), key);
  if (blocking && it == streams.end()) {
    streams.push_back(key);
    lupine_fast_state.blocking_epoch.fetch_add(1, std::memory_order_acq_rel);
    lupine_blocking_stream_count.store(streams.size(),
                                       std::memory_order_release);
  } else if (!blocking && it != streams.end()) {
    streams.erase(it);
    lupine_fast_state.blocking_epoch.fetch_add(1, std::memory_order_acq_rel);
    lupine_blocking_stream_count.store(streams.size(),
                                       std::memory_order_release);
  }
}

bool lupine_is_stream_blocking(uintptr_t key) {
  uint64_t epoch =
      lupine_fast_state.blocking_epoch.load(std::memory_order_acquire);
  size_t index =
      lupine_splitmix64(key) & (lupine_tls_cache.blocking_streams.size() - 1);
  auto &slot = lupine_tls_cache.blocking_streams[index];
  if (slot.epoch == epoch && slot.key == key) {
    return slot.blocking;
  }
  std::lock_guard<std::mutex> lock(lupine_blocking_streams_mutex());
  epoch = lupine_fast_state.blocking_epoch.load(std::memory_order_relaxed);
  const auto &streams = lupine_blocking_streams();
  bool blocking =
      std::find(streams.begin(), streams.end(), key) != streams.end();
  slot = {epoch, key, blocking};
  return blocking;
}

// Stops treating `key` as blocking, keeping the legacy stream behind what it
// already sent on `conns`. Retires `key`'s ticket on each connection before
// removing `key` from `lupine_blocking_streams` so no legacy request can miss
// it, and `key` is erased from `scope_last` immediately so a recycled handle
// starts clean. Returns whether it was blocking.
bool lupine_retire_blocking_stream(uintptr_t key, conn_t *const *conns,
                                   size_t count) {
  if (lupine_blocking_stream_count.load(std::memory_order_acquire) == 0) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_blocking_streams_mutex());
    auto &streams = lupine_blocking_streams();
    if (std::find(streams.begin(), streams.end(), key) == streams.end()) {
      return false;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    rpc_scope_retire(conns[i], key);
  }
  std::lock_guard<std::mutex> lock(lupine_blocking_streams_mutex());
  auto &streams = lupine_blocking_streams();
  auto it = std::find(streams.begin(), streams.end(), key);
  if (it != streams.end()) {
    streams.erase(it);
    lupine_fast_state.blocking_epoch.fetch_add(1, std::memory_order_acq_rel);
    lupine_blocking_stream_count.store(streams.size(),
                                       std::memory_order_release);
  }
  return true;
}

// The per-thread default stream's scope: one per thread, blocking while the
// thread lives.
struct lupine_thread_stream {
  std::vector<conn_t *> conns; // connections this thread's stream was used on
  lupine_thread_stream() { lupine_set_stream_blocking(key(), true); }
  ~lupine_thread_stream() {
    lupine_retire_blocking_stream(key(), conns.data(), conns.size());
  }
  // Unique per thread: a later thread can reuse this one's TLS address while
  // its key is still retired. The top bit keeps it apart from any handle.
  const uintptr_t id = (uintptr_t{1} << (sizeof(uintptr_t) * 8 - 1)) |
                       next_id.fetch_add(1, std::memory_order_relaxed);
  uintptr_t key() const { return id; }
  static std::atomic<uintptr_t> next_id;
};
std::atomic<uintptr_t> lupine_thread_stream::next_id{1};

const uintptr_t kLegacyStreamScope =
    reinterpret_cast<uintptr_t>(CU_STREAM_LEGACY);

lupine_thread_stream &lupine_this_thread_stream() {
  static thread_local lupine_thread_stream thread_stream;
  return thread_stream;
}

} // namespace

extern "C" uintptr_t lupine_stream_scope(CUstream stream) {
  uintptr_t bits = reinterpret_cast<uintptr_t>(stream);
  if (bits == LUPINE_STREAM_FETCH_READY_BITS) {
    return RPC_SCOPE_LANE;
  }
  if (bits == LUPINE_STREAM_FETCH_SYNC_ALL_BITS) {
    return RPC_SCOPE_ALL;
  }
  if (stream == nullptr || stream == CU_STREAM_LEGACY) {
    return kLegacyStreamScope;
  }
  return stream == CU_STREAM_PER_THREAD ? lupine_this_thread_stream().key()
                                        : bits;
}

extern "C" void lupine_note_stream_blocking(CUstream stream, bool blocking) {
  lupine_set_stream_blocking(reinterpret_cast<uintptr_t>(stream), blocking);
}

extern "C" int lupine_rpc_start_on_stream(conn_t *conn, int op, CUstream stream,
                                          unsigned flags) {
  uintptr_t key = lupine_stream_scope(stream);
  if (key == RPC_SCOPE_LANE) {
    return rpc_write_start_request_on(conn, op, RPC_SCOPE_LANE, flags);
  }
  if (key == RPC_SCOPE_ALL) {
    return rpc_write_start_request(conn, op);
  }
  if (key == kLegacyStreamScope) {
    if (rpc_write_start_request_on(conn, op, kLegacyStreamScope, flags) < 0) {
      return -1;
    }
    if (lupine_blocking_stream_count.load(std::memory_order_acquire) != 0) {
      std::lock_guard<std::mutex> lock(lupine_blocking_streams_mutex());
      for (uintptr_t blocking_key : lupine_blocking_streams()) {
        if (rpc_request_after(conn, blocking_key) < 0) {
          return -1;
        }
      }
    }
    // The connection keeps the retired stream tickets until a legacy request
    // that publishes a ticket has waited for them.
    return rpc_request_after_retired(conn);
  }
  bool blocking = stream == CU_STREAM_PER_THREAD;
  if (blocking) {
    auto &thread_stream = lupine_this_thread_stream();
    if (std::find(thread_stream.conns.begin(), thread_stream.conns.end(),
                  conn) == thread_stream.conns.end()) {
      thread_stream.conns.push_back(conn);
    }
  } else if (lupine_blocking_stream_count.load(std::memory_order_acquire) !=
             0) {
    blocking = lupine_is_stream_blocking(key);
  }
  if (rpc_write_start_request_on(conn, op, key, flags) < 0) {
    return -1;
  }
  return blocking ? rpc_request_after(conn, kLegacyStreamScope) : 0;
}

extern "C" int lupine_rpc_request_on_each(conn_t *conn,
                                          const void *const *handles,
                                          unsigned int count) {
  for (unsigned int i = 0; handles != nullptr && i < count; ++i) {
    if (rpc_request_on(conn, reinterpret_cast<uintptr_t>(handles[i])) < 0) {
      return -1;
    }
  }
  return 0;
}

extern "C" void lupine_forget_stream_owner(CUstream stream) {
  lupine_retire_stale_fetch_stream(stream);
  conn_t *conn = lupine_route_remote_conn(lupine_route_for_stream(stream));
  uintptr_t key = reinterpret_cast<uintptr_t>(stream);
  // A destroyed stream's scope only matters while the legacy stream still
  // waits for it.
  if (!lupine_retire_blocking_stream(key, &conn, conn != nullptr) &&
      conn != nullptr) {
    rpc_scope_forget(conn, key);
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (lupine_owners<CUstream>().erase(stream) != 0) {
    lupine_fast_state.handle_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
}

template <typename Handle>
static lupine_route lupine_route_for_known_owner(Handle handle) {
  const auto *slot = lupine_lookup_handle_owner(handle);
  return slot == nullptr ? lupine_route{}
                         : lupine_route_from_identity(slot->route_id);
}

template <typename Handle>
static lupine_route lupine_route_for_owner_or_default(Handle handle) {
  lupine_route route = lupine_route_for_known_owner(handle);
  return route.kind == LUPINE_ROUTE_INVALID ? lupine_route_for_default()
                                            : route;
}

template <typename Handle>
static void lupine_forget_scoped_owner(Handle handle) {
  if (conn_t *conn =
          lupine_route_remote_conn(lupine_route_for_owner_or_default(handle))) {
    rpc_scope_forget(conn, reinterpret_cast<uintptr_t>(handle));
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  if (lupine_owners<Handle>().erase(handle) != 0) {
    lupine_fast_state.handle_epoch.fetch_add(1, std::memory_order_acq_rel);
  }
}

extern "C" void lupine_forget_event_owner(CUevent event) {
  lupine_forget_scoped_owner(event);
  lupine_event_forget(event);
}

extern "C" void lupine_forget_graph_exec_owner(CUgraphExec exec) {
  lupine_forget_scoped_owner(exec);
}

#define LUPINE_DEFINE_OWNER_ROUTE(name, type, id)                              \
  extern "C" void lupine_note_##name##_owner_route(type h, lupine_route r) {   \
    lupine_note_owner_route(h, r);                                             \
  }                                                                            \
  extern "C" lupine_route lupine_route_for_##name(type h) {                    \
    return lupine_route_for_owner_or_default(h);                               \
  }

LUPINE_FOR_EACH_OWNED_HANDLE(LUPINE_DEFINE_OWNER_ROUTE)
#undef LUPINE_DEFINE_OWNER_ROUTE
#undef LUPINE_FOR_EACH_OWNED_HANDLE

// Unlike lupine_route_for_stream, does not fall back to the default route.
extern "C" lupine_route lupine_route_for_known_stream(CUstream stream) {
  return lupine_route_for_known_owner(stream);
}

static inline bool lupine_devptr_outside_bounds(CUdeviceptr ptr) {
  uint64_t p = static_cast<uint64_t>(ptr);
  return p == 0 ||
         lupine_fast_state.devptr_count.load(std::memory_order_acquire) == 0 ||
         p < lupine_fast_state.devptr_min.load(std::memory_order_acquire) ||
         p > lupine_fast_state.devptr_last.load(std::memory_order_acquire);
}

static inline bool
lupine_tls_lookup_devptr(const lupine_tls_routing_cache &tls, CUdeviceptr ptr,
                         uint64_t epoch,
                         const lupine_tls_alloc_slot **slot_out) {
  if (tls.devptr_epoch != epoch) {
    return false;
  }
  for (const auto &slot : tls.allocs) {
    if (ptr >= slot.base && ptr - slot.base < slot.size) {
      *slot_out = &slot;
      return true;
    }
  }
  for (const auto &gap : tls.gaps) {
    if (ptr >= gap.base && ptr <= gap.last) {
      return true;
    }
  }
  return false;
}

static inline bool
lupine_try_tls_devptr(CUdeviceptr ptr, const lupine_tls_alloc_slot **slot_out) {
  *slot_out = nullptr;
  return lupine_devptr_outside_bounds(ptr) ||
         lupine_tls_lookup_devptr(
             lupine_tls_cache, ptr,
             lupine_fast_state.devptr_epoch.load(std::memory_order_acquire),
             slot_out);
}

static void lupine_tls_populate_devptr_locked(
    CUdeviceptr ptr, const lupine_deviceptr_allocation_record *alloc) {
  const auto &table = lupine_devptr_ranges();
  if (ptr == 0 || table.overlaps != 0 ||
      !lupine_owners<CUdeviceptr>().empty()) {
    return;
  }
  auto &tls = lupine_tls_cache;
  uint64_t epoch =
      lupine_fast_state.devptr_epoch.load(std::memory_order_relaxed);
  if (tls.devptr_epoch != epoch) {
    tls.allocs = {};
    tls.gaps = {};
    tls.devptr_epoch = epoch;
  }
  const lupine_tls_alloc_slot *existing = nullptr;
  if (lupine_tls_lookup_devptr(tls, ptr, epoch, &existing)) {
    return;
  }
  auto next = table.ranges.upper_bound(ptr);
  if (alloc != nullptr) {
    auto it = std::prev(next);
    tls.allocs[tls.alloc_hand++ & 3] = {it->first, alloc->size, alloc->context,
                                        alloc->route_id};
  } else {
    CUdeviceptr lo = next == table.ranges.begin()
                         ? 1
                         : std::prev(next)->first + std::prev(next)->second;
    CUdeviceptr hi = next == table.ranges.end() ? UINT64_MAX : next->first - 1;
    tls.gaps[tls.gap_hand++ & 1] = {lo, hi};
  }
}

static bool lupine_lookup_deviceptr_route_locked(
    CUdeviceptr ptr, int *route_id,
    const lupine_deviceptr_allocation_record **alloc_out = nullptr) {
  auto &owners = lupine_owners<CUdeviceptr>();
  auto it = owners.find(ptr);
  if (it != owners.end() && alloc_out == nullptr) {
    if (route_id != nullptr) {
      *route_id = it->second.route_id;
    }
    return true;
  }
  const auto *alloc = lupine_deviceptr_allocation_cache_lookup(ptr);
  if (alloc_out != nullptr) {
    *alloc_out = alloc;
  }
  if (it != owners.end() || alloc != nullptr) {
    if (route_id != nullptr) {
      *route_id = it != owners.end() ? it->second.route_id : alloc->route_id;
    }
    return true;
  }
  return false;
}

static inline bool lupine_alloc_slot_covers(const lupine_tls_alloc_slot *slot,
                                            CUdeviceptr ptr, size_t bytes) {
  return slot != nullptr && bytes != 0 && ptr <= UINT64_MAX - (bytes - 1) &&
         bytes - 1 < slot->size - (ptr - slot->base);
}

extern "C" bool lupine_deviceptr_is_tracked(CUdeviceptr ptr) {
  const lupine_tls_alloc_slot *slot = nullptr;
  if (lupine_try_tls_devptr(ptr, &slot)) {
    return slot != nullptr;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  const lupine_deviceptr_allocation_record *alloc = nullptr;
  bool tracked = lupine_lookup_deviceptr_route_locked(ptr, nullptr, &alloc);
  lupine_tls_populate_devptr_locked(ptr, alloc);
  return tracked;
}

extern "C" bool lupine_deviceptr_allocation_covers(CUdeviceptr ptr,
                                                   size_t bytes, int route_id) {
  if (bytes == 0) {
    return false;
  }
  const lupine_tls_alloc_slot *slot = nullptr;
  if (lupine_try_tls_devptr(ptr, &slot) &&
      (slot == nullptr || ptr <= UINT64_MAX - (bytes - 1))) {
    return slot != nullptr && slot->route_id == route_id &&
           lupine_alloc_slot_covers(slot, ptr, bytes);
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  // An allocation is one interval, so holding both endpoints holds the range.
  const auto *allocation = lupine_deviceptr_allocation_cache_lookup(ptr);
  lupine_tls_populate_devptr_locked(ptr, allocation);
  return allocation != nullptr && allocation->route_id == route_id &&
         allocation ==
             lupine_deviceptr_allocation_cache_lookup(ptr + bytes - 1);
}

extern "C" lupine_route lupine_route_for_deviceptr(CUdeviceptr ptr) {
  const lupine_tls_alloc_slot *slot = nullptr;
  if (lupine_try_tls_devptr(ptr, &slot)) {
    return slot != nullptr ? lupine_route_from_identity(slot->route_id)
                           : lupine_route_for_default();
  }
  int route_id = -2;
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    const lupine_deviceptr_allocation_record *alloc = nullptr;
    bool found = lupine_lookup_deviceptr_route_locked(ptr, &route_id, &alloc);
    lupine_tls_populate_devptr_locked(ptr, alloc);
    if (found) {
      return lupine_route_from_identity(route_id);
    }
  }
  return lupine_route_for_default();
}

lupine_htod_routing_info
lupine_query_htod_routing(CUdeviceptr dst, const void *src, size_t bytes) {
  lupine_htod_routing_info info;
  CUdeviceptr src_ptr = reinterpret_cast<CUdeviceptr>(src);
  const lupine_tls_alloc_slot *dst_slot = nullptr, *src_slot = nullptr;
  if (lupine_try_tls_devptr(dst, &dst_slot) &&
      (bytes == 0 || src == nullptr ||
       lupine_try_tls_devptr(src_ptr, &src_slot))) {
    info.dst_route = dst_slot != nullptr
                         ? lupine_route_from_identity(dst_slot->route_id)
                         : lupine_route_for_default();
    info.dst_allocation_covers = lupine_alloc_slot_covers(dst_slot, dst, bytes);
    info.src_deviceptr_tracked = src_slot != nullptr;
    return info;
  }

  int dst_route_id = -2;
  bool dst_known = false;
  {
    const lupine_deviceptr_allocation_record *dst_alloc = nullptr;
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    dst_known =
        lupine_lookup_deviceptr_route_locked(dst, &dst_route_id, &dst_alloc);
    lupine_tls_populate_devptr_locked(dst, dst_alloc);
    if (bytes != 0 && dst_known && dst_alloc != nullptr &&
        dst_alloc->route_id == dst_route_id &&
        dst <= UINT64_MAX - (bytes - 1) &&
        dst_alloc ==
            lupine_deviceptr_allocation_cache_lookup(dst + bytes - 1)) {
      info.dst_allocation_covers = true;
    }
    if (bytes != 0 && src != nullptr) {
      const lupine_deviceptr_allocation_record *src_alloc = nullptr;
      info.src_deviceptr_tracked =
          lupine_lookup_deviceptr_route_locked(src_ptr, nullptr, &src_alloc);
      lupine_tls_populate_devptr_locked(src_ptr, src_alloc);
    }
  }
  info.dst_route = dst_known ? lupine_route_from_identity(dst_route_id)
                             : lupine_route_for_default();
  return info;
}

extern "C" CUcontext lupine_context_for_deviceptr(CUdeviceptr ptr) {
  const lupine_tls_alloc_slot *slot = nullptr;
  if (lupine_try_tls_devptr(ptr, &slot)) {
    return slot != nullptr ? slot->context : nullptr;
  }
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto owner = lupine_owners<CUdeviceptr>().find(ptr);
  if (owner != lupine_owners<CUdeviceptr>().end() &&
      owner->second.pointer_context != nullptr) {
    return owner->second.pointer_context;
  }
  const auto *allocation = lupine_deviceptr_allocation_cache_lookup(ptr);
  lupine_tls_populate_devptr_locked(ptr, allocation);
  return allocation != nullptr ? allocation->context : nullptr;
}

CUresult lupine_set_current_context_on_route(lupine_route route,
                                             CUcontext ctx) {
  if (route.kind == LUPINE_ROUTE_INVALID) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  // The server lane bound to this thread already has ctx current; setting it
  // again is idempotent, so skip the round trip.
  if (!lupine_route_is_local(route) &&
      lupine_lane_context_cache_matches(lupine_route_identity(route), ctx)) {
    return CUDA_SUCCESS;
  }
  uint64_t epoch = lupine_lane_context_cache_epoch();
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    result = lupine_call_real_cuda_fn("cuCtxSetCurrent", ctx);
  } else if (ctx != nullptr && lupine_routes_share_server(
                                   lupine_route_for_known_owner(ctx), route)) {
    // A live context this client created on this route cannot be refused, so
    // the lane needs no answer: its later requests run after the switch. An
    // unknown or destroyed one still waits, so its error is returned here.
    conn_t *conn = lupine_route_remote_conn(route);
    result = lupine_prepare_rpc(conn) < 0 ||
                     rpc_write_start_request_on(conn, RPC_cuCtxSetCurrent,
                                                RPC_SCOPE_LANE) < 0 ||
                     rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
                     rpc_write_end(conn) < 0
                 ? CUDA_ERROR_DEVICE_UNAVAILABLE
                 : CUDA_SUCCESS;
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request_on(conn, RPC_cuCtxSetCurrent, RPC_SCOPE_LANE) <
            0 ||
        rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  // The lane just moved to another context, and with it to that context's
  // device and to that context's server. Device answers cached against the old
  // binding are stale even when the set failed and left the lane somewhere
  // unknown.
  lupine_note_device_binding_changed();
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

  lupine_route route = lupine_route_for_known_owner(ctx);
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
  for (CUcontext hint :
       {lupine_current_context_hint(), lupine_default_context_hint_value(),
        lupine_global_default_context_hint_value()}) {
    lupine_route route = lupine_route_for_default_context_hint(hint);
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

extern "C" conn_t *lupine_rpc_conn_for_event(CUevent event) {
  return lupine_route_remote_conn(lupine_route_for_event(event));
}

extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr) {
  return lupine_route_remote_conn(lupine_route_for_deviceptr(ptr));
}
