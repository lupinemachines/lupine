#include "cuda_client.h"
#include "device_stdout.h"
#include "fatbin_filter.h"
#include "lupine_platform.h"
#include "macos_library_profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_occupancy.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <new>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <dlfcn.h>
#if defined(__linux__)
#include <link.h>
#endif
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#if !defined(__APPLE__) && !defined(__GLIBC__)
#error "Lupine CUDA client requires glibc, macOS, or Windows"
#endif
#endif

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include <cstdlib>
#include <cstring>

#include "cache.h"
#include "checkpoint.h"
#include "client_routing.h"
#include "codegen/gen_cuda_client.h"
#include "codegen/gen_rpc_ids.h"
#include "cuda_client_memcpy.h"
#include "cuda_client_rpc.h"
#include "cuda_profiler_compat.h"
#include "events.h"
#include "ipc.h"
#include "log_callbacks.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "rpc.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"
#include "transport.h"
// Reuse Zstd's bundled xxHash to preserve existing profile-cache keys.
#include "third_party/zstd/lib/common/xxhash.h"

void *rpc_client_dispatch_thread(void *arg);

extern "C" void lupine_invalidate_function_caches();
static void lupine_reset_conn_cache(conn_t *conn);

static void lupine_cuda_transport_connection_changed(conn_t *conn) {
  lupine_reset_conn_cache(conn);
  lupine_invalidate_current_context_cache();
}

static const lupine_client_transport_config &lupine_cuda_transport_config() {
  static const auto config = [] {
    lupine_client_transport_config config;
    config.dispatch = rpc_client_dispatch_thread;
    config.connection_opened = lupine_cuda_transport_connection_changed;
    config.connection_closed = lupine_cuda_transport_connection_changed;
    config.connection_kind = "cuda";
    config.w_offset = LUPINE_HOST_ALLOCATION_W_OFFSET;
    config.checkpoint_begin = lupine_checkpoint_wait_for_captures;
    config.checkpoint_end = lupine_checkpoint_resume_captures;
    return config;
  }();
  return config;
}

void rpc_destroy_thread_lane(uint64_t lane_id) {
  lupine_client_transport_retire_lane(lane_id);
}

static void lupine_discard_pending_log_callbacks(conn_t *conn);
static void lupine_complete_pending_log_callbacks(conn_t *conn,
                                                  int32_t stream_id);

static void lupine_rpc_connection_closed(conn_t *conn) {
  lupine_reset_conn_cache(conn);
  lupine_invalidate_current_context_cache();
  lupine_discard_pending_log_callbacks(conn);
}

static pthread_once_t lupine_rpc_lifecycle_once = PTHREAD_ONCE_INIT;

static void lupine_install_rpc_lifecycle_hooks() {
  const rpc_lifecycle_hooks hooks = {
      lupine_rpc_connection_closed, rpc_destroy_thread_lane,
      lupine_complete_pending_log_callbacks, lupine_host_range_is_protected};
  if (rpc_set_lifecycle_hooks(&hooks) < 0) {
    LUPINE_LOG_ERROR("Failed to install CUDA RPC lifecycle hooks");
  }
}

static CUresult lupine_remote_cuInit(conn_t *conn, unsigned int flags);

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;
struct lupine_library_image_record {
  uint32_t kind = 0;
  const void *code = nullptr;
  std::vector<unsigned char> image;
  std::unordered_map<int, CUlibrary> libraries_by_route;
};

struct lupine_library_kernel_record {
  CUlibrary library = nullptr;
  std::string name;
  std::unordered_map<int, CUkernel> kernels_by_route;
};

struct lupine_module_function_record {
  CUmodule module = nullptr;
  std::string name;
};

static CUresult lupine_warm_func_param_info(CUfunction function);
static CUresult lupine_warm_kernel_param_info(CUkernel kernel);

extern int rpc_size();
extern int rpc_open();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);

static bool lupine_env_flag(const char *name, bool default_value) {
  const char *value = getenv(name);
  return value != nullptr ? strcmp(value, "0") != 0 : default_value;
}

static bool lupine_stub_missing_enabled() {
  static bool enabled = lupine_env_flag("LUPINE_STUB_MISSING", true);
  return enabled;
}

static bool lupine_symbol_looks_like_driver_api(const char *symbol) {
  return symbol != nullptr && symbol[0] == 'c' && symbol[1] == 'u' &&
         symbol[2] >= 'A' && symbol[2] <= 'Z';
}

using lupine_dlsym_fn = void *(*)(void *, const char *);

#if defined(_WIN32)
static void *lupine_real_dlsym(void *handle, const char *name) {
  if (handle == nullptr || name == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<void *>(
      GetProcAddress(static_cast<HMODULE>(handle), name));
}
#elif defined(__GLIBC__)
static const char *lupine_dlsym_glibc_version() {
#if defined(__x86_64__)
  return "GLIBC_2.2.5";
#elif defined(__aarch64__)
  return "GLIBC_2.17";
#else
  return nullptr;
#endif
}

static void *lupine_real_dlsym(void *handle, const char *name) {
  static const lupine_dlsym_fn real_dlsym = [] {
    const char *version = lupine_dlsym_glibc_version();
    return version == nullptr ? nullptr
                              : reinterpret_cast<lupine_dlsym_fn>(
                                    dlvsym(RTLD_NEXT, "dlsym", version));
  }();
  return real_dlsym != nullptr ? real_dlsym(handle, name) : nullptr;
}
#else
static void *lupine_real_dlsym(void *handle, const char *name) {
  return ::dlsym(handle, name);
}
#endif

extern "C" CUresult lupine_unsupported_driver_api() {
  LUPINE_LOG_ERROR("LUPINE unsupported generic Driver API called");
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_missing_driver_api_called(const char *symbol) {
  LUPINE_LOG_ERROR("LUPINE missing Driver API called: " << symbol);
  return CUDA_ERROR_NOT_SUPPORTED;
}

template <typename T, typename Tag> static T &lupine_leaked(Tag) {
  static T *instance = new T();
  return *instance;
}

template <typename Map> struct lupine_lazy_cache {
  std::atomic<Map *> ptr{nullptr};
  std::atomic<size_t> active_inserts{0};
  std::atomic<bool> dirty{false};
  Map *try_get() const { return ptr.load(std::memory_order_acquire); }
  bool empty_fast() const {
    return !dirty.load(std::memory_order_acquire) &&
           active_inserts.load(std::memory_order_acquire) == 0 &&
           !dirty.load(std::memory_order_acquire);
  }
  Map &ensure() {
    if (Map *p = try_get())
      return *p;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    Map *p = ptr.load(std::memory_order_relaxed);
    if (!p)
      ptr.store(p = new Map(), std::memory_order_release);
    return *p;
  }
  template <typename K, typename V> bool find(const K &k, V &v) const {
    Map *p = try_get();
    return p != nullptr && p->find(k, v);
  }
  template <typename K, typename V> void insert_or_assign(K &&k, V &&v) {
    active_inserts.fetch_add(1, std::memory_order_acq_rel);
    dirty.store(true, std::memory_order_release);
    ensure().insert_or_assign(std::forward<K>(k), std::forward<V>(v));
    dirty.store(true, std::memory_order_release);
    active_inserts.fetch_sub(1, std::memory_order_release);
  }
  template <typename K> bool erase(const K &k) {
    Map *p = try_get();
    return p != nullptr && p->erase(k);
  }
  void clear() {
    if (Map *p = try_get(); p && !empty_fast()) {
      dirty.store(false, std::memory_order_release);
      p->clear();
      if (active_inserts.load(std::memory_order_acquire) != 0)
        dirty.store(true, std::memory_order_release);
    }
  }
};

template <typename K, typename V> struct lupine_tiny_cache {
  std::atomic<bool> nonempty{false};
  std::mutex mu;
  std::unordered_map<K, V> map;
  bool find(const K &k, V &v) {
    if (!nonempty.load(std::memory_order_acquire))
      return false;
    std::lock_guard<std::mutex> lock(mu);
    auto it = map.find(k);
    return it != map.end() && (v = it->second, true);
  }
  void insert_or_assign(const K &k, V v) {
    std::lock_guard<std::mutex> lock(mu);
    map[k] = std::move(v);
    nonempty.store(true, std::memory_order_release);
  }
  void clear() {
    if (!nonempty.load(std::memory_order_acquire))
      return;
    std::lock_guard<std::mutex> lock(mu);
    map.clear();
    nonempty.store(false, std::memory_order_release);
  }
};

template <typename First, typename... Rest>
static size_t lupine_hash_combine(const First &first, const Rest &...rest) {
  size_t hash = std::hash<First>{}(first);
  ((hash ^= std::hash<Rest>{}(rest) + 0x9e3779b9 + (hash << 6) + (hash >> 2)),
   ...);
  return hash;
}

template <typename... Args>
static bool lupine_read_all(conn_t *conn, Args &...args) {
  return ((rpc_read(conn, &args, sizeof(args)) >= 0) && ...);
}

template <typename... Args>
static bool lupine_write_lvals(conn_t *conn, Args &...args) {
  return ((rpc_write(conn, &args, sizeof(args)) >= 0) && ...);
}

template <typename... W, typename... R>
static bool lupine_rpc_call_impl(conn_t *conn, int op, bool unscoped,
                                 std::tuple<W &...> w, std::tuple<R &...> r) {
  return lupine_prepare_rpc(conn) >= 0 &&
         (unscoped ? rpc_write_start_request_on(conn, op, RPC_SCOPE_NONE)
                   : rpc_write_start_request(conn, op)) >= 0 &&
         std::apply([conn](W &...x) { return lupine_write_lvals(conn, x...); },
                    w) &&
         rpc_wait_for_response(conn) >= 0 &&
         std::apply([conn](R &...y) { return lupine_read_all(conn, y...); },
                    r) &&
         rpc_read_end(conn) >= 0;
}

template <typename WTuple, typename RTuple>
static bool lupine_rpc_call(conn_t *conn, int op, WTuple w, RTuple r) {
  return lupine_rpc_call_impl(conn, op, false, w, r);
}

template <typename WTuple, typename RTuple>
static bool lupine_rpc_call_unscoped(conn_t *conn, int op, WTuple w, RTuple r) {
  return lupine_rpc_call_impl(conn, op, true, w, r);
}

static std::unordered_map<std::string, std::vector<uint64_t>> &
lupine_private_export_hashes() {
  static std::unordered_map<std::string, std::vector<uint64_t>> hashes;
  return hashes;
}

struct lupine_private_jit_cache_object {
  uint32_t type = 0;
  std::vector<unsigned char> data;
};

using lupine_private_jit_cache =
    std::unordered_map<void *,
                       std::unique_ptr<lupine_private_jit_cache_object>>;

static std::mutex &lupine_private_jit_cache_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static lupine_private_jit_cache &lupine_private_jit_cache_objects() {
  return lupine_leaked<lupine_private_jit_cache>([] {});
}

static std::unordered_map<CUfunction, CUfunction> &lupine_private_node_map() {
  static std::unordered_map<CUfunction, CUfunction> mappings;
  return mappings;
}

struct lupine_kernel_function_key {
  int route_id = -2;
  CUcontext context = nullptr;
  CUkernel kernel = nullptr;

  bool operator==(const lupine_kernel_function_key &other) const {
    return route_id == other.route_id && context == other.context &&
           kernel == other.kernel;
  }
};

struct lupine_occupancy_key {
  int route_id = -2;
  CUfunction function = nullptr;
  int block_size = 0;
  size_t dynamic_smem_size = 0;
  unsigned int flags = 0;
  bool with_flags = false;

  bool operator==(const lupine_occupancy_key &other) const {
    return route_id == other.route_id && function == other.function &&
           block_size == other.block_size &&
           dynamic_smem_size == other.dynamic_smem_size &&
           flags == other.flags && with_flags == other.with_flags;
  }
};

enum class lupine_attribute_cache { function, function_set, kernel };

#if CUDA_VERSION >= 12000
// Cache successful remote encodings by every input, including the route.
// Serialize insertion and eviction to keep the bound strict across threads.
static std::mutex lupine_tensor_map_mutex;
static std::unordered_map<std::string, CUtensorMap> lupine_tensor_maps;
static constexpr size_t kLupineTensorMapCacheLimit = 4096;
static uint64_t lupine_tensor_map_generation = 0;
#endif

struct lupine_param_info_key {
  uintptr_t handle = 0;
  size_t index = 0;
  bool kernel = false;
  bool operator==(const lupine_param_info_key &o) const {
    return handle == o.handle && index == o.index && kernel == o.kernel;
  }
};

struct lupine_param_info_value {
  size_t offset = 0;
  size_t size = 0;
};
static_assert(sizeof(lupine_param_info_value) == 16);

struct lupine_param_layout {
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
};

struct lupine_param_layout_key {
  uintptr_t handle = 0;
  bool kernel = false;
  bool operator==(const lupine_param_layout_key &o) const {
    return handle == o.handle && kernel == o.kernel;
  }
};

struct lupine_key_hash {
  size_t operator()(const lupine_kernel_attribute_key &k) const {
    return lupine_hash_combine(k.route_id,
                               reinterpret_cast<uintptr_t>(k.kernel),
                               k.attribute, k.device);
  }
  size_t operator()(const lupine_function_attribute_key &k) const {
    return lupine_hash_combine(
        k.route_id, reinterpret_cast<uintptr_t>(k.function), k.attribute);
  }
  size_t operator()(const lupine_kernel_function_key &k) const {
    return lupine_hash_combine(k.route_id,
                               reinterpret_cast<uintptr_t>(k.context),
                               reinterpret_cast<uintptr_t>(k.kernel));
  }
  size_t operator()(const lupine_occupancy_key &k) const {
    return lupine_hash_combine(
        k.route_id, reinterpret_cast<uintptr_t>(k.function), k.block_size,
        k.dynamic_smem_size, k.flags, k.with_flags);
  }
  size_t operator()(const lupine_param_info_key &k) const {
    return lupine_hash_combine(k.handle, k.index, k.kernel);
  }
  size_t operator()(const lupine_param_layout_key &k) const {
    return lupine_hash_combine(k.handle, k.kernel);
  }
};

template <lupine_attribute_cache Kind> static auto &get_cache() {
  using key = std::conditional_t<Kind == lupine_attribute_cache::kernel,
                                 lupine_kernel_attribute_key,
                                 lupine_function_attribute_key>;
  static lupine_lazy_cache<libcuckoo::cuckoohash_map<key, int, lupine_key_hash>>
      cache;
  return cache;
}

static auto &lupine_library_images() {
  return lupine_leaked<
      std::unordered_map<CUlibrary, lupine_library_image_record>>([] {});
}

struct lupine_library_enumeration {
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
  std::vector<CUkernel> kernels;
};

static auto &lupine_library_enumerations() {
  return lupine_leaked<
      std::unordered_map<CUlibrary, lupine_library_enumeration>>([] {});
}

static auto &lupine_library_kernels() {
  return lupine_leaked<
      std::unordered_map<CUkernel, lupine_library_kernel_record>>([] {});
}

static auto &lupine_module_functions() {
  return lupine_leaked<
      std::unordered_map<CUfunction, lupine_module_function_record>>([] {});
}

static auto &lupine_function_kernels() {
  return lupine_leaked<std::unordered_map<CUfunction, CUkernel>>([] {});
}

static auto &lupine_library_functions() {
  return lupine_leaked<std::unordered_map<CUfunction, CUlibrary>>([] {});
}

static auto &lupine_library_modules() {
  return lupine_leaked<std::unordered_map<CUmodule, CUlibrary>>([] {});
}

static constexpr size_t kLupineMaxConns = 16;
static constexpr size_t kLupineMaxCachedDevices = 128;
static constexpr size_t kLupineMaxDevAttrs = 256;
static constexpr uint64_t kLupineCachedBit = uint64_t{1} << 32;

template <typename T>
static bool lupine_atomic_u32_load(const std::atomic<uint64_t> &slot, T *out) {
  uint64_t raw = slot.load(std::memory_order_acquire);
  if (raw & kLupineCachedBit)
    *out = static_cast<T>(static_cast<uint32_t>(raw));
  return (raw & kLupineCachedBit) != 0;
}

template <typename T>
static void lupine_atomic_u32_store(std::atomic<uint64_t> &slot, T val) {
  slot.store(kLupineCachedBit | static_cast<uint32_t>(val),
             std::memory_order_release);
}

struct lupine_device_snapshot_info {
  char name[256] = {};
  CUuuid uuid = {};
  uint64_t total_mem = 0;
};

// Unlike `capabilities` vs the virtual device table, `capabilities` includes
// devices hidden by the client's CUDA_VISIBLE_DEVICES: a CUlibrary is available
// to all contexts on its server.
struct lupine_conn_cache_slot {
  std::atomic<conn_t *> conn{nullptr};
  std::atomic<bool> initialized{false};
  std::atomic<bool> snapshot_attempted{false};
  std::atomic<uint64_t> driver_version{0};
  std::atomic<uint64_t> loading_mode{0};
  std::shared_ptr<const std::vector<unsigned>> capabilities;
};

static std::mutex &lupine_conn_cache_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static lupine_conn_cache_slot *lupine_conn_caches() {
  static auto *slots = new lupine_conn_cache_slot[kLupineMaxConns]();
  return slots;
}

static lupine_conn_cache_slot *lupine_conn_slot(conn_t *conn) {
  if (!conn || static_cast<unsigned>(conn->logical_index) >= kLupineMaxConns)
    return nullptr;
  auto &slot = lupine_conn_caches()[conn->logical_index];
  if (slot.conn.load(std::memory_order_acquire) == conn)
    return &slot;
  std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
  if (slot.conn.load(std::memory_order_relaxed) != conn) {
    slot.initialized.store(false, std::memory_order_relaxed);
    slot.snapshot_attempted.store(false, std::memory_order_relaxed);
    slot.driver_version.store(0, std::memory_order_relaxed);
    slot.loading_mode.store(0, std::memory_order_relaxed);
    slot.capabilities.reset();
    slot.conn.store(conn, std::memory_order_release);
  }
  return &slot;
}

static void lupine_reset_conn_cache(conn_t *conn) {
  if (!conn || static_cast<unsigned>(conn->logical_index) >= kLupineMaxConns)
    return;
  std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
  if (auto &slot = lupine_conn_caches()[conn->logical_index];
      slot.conn.load(std::memory_order_relaxed) == conn) {
    slot.conn.store(nullptr, std::memory_order_release);
    slot.capabilities.reset();
  }
}

struct lupine_cached_device_info {
  std::atomic<uint64_t> attrs[kLupineMaxDevAttrs]{};
  std::atomic<bool> has_snapshot{false};
  lupine_device_snapshot_info snapshot{};
};

static std::atomic<lupine_cached_device_info *>
    lupine_device_caches[kLupineMaxCachedDevices]{};

static lupine_cached_device_info *lupine_ensure_device_cache(int dev) {
  if (static_cast<unsigned>(dev) >= kLupineMaxCachedDevices)
    return nullptr;
  if (auto *p = lupine_device_caches[dev].load(std::memory_order_acquire))
    return p;
  std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
  auto *p = lupine_device_caches[dev].load(std::memory_order_relaxed);
  if (!p)
    lupine_device_caches[dev].store(p = new lupine_cached_device_info(),
                                    std::memory_order_release);
  return p;
}

static bool lupine_dev_attr_get(int dev, int attrib, int *out) {
  if (static_cast<unsigned>(dev) >= kLupineMaxCachedDevices ||
      static_cast<unsigned>(attrib) >= kLupineMaxDevAttrs)
    return false;
  auto *p = lupine_device_caches[dev].load(std::memory_order_acquire);
  return p != nullptr && lupine_atomic_u32_load(p->attrs[attrib], out);
}

// PyTorch's pin-memory path polls cuDevicePrimaryCtxGetState continuously
// (~42k calls in a 100-step makemore run); over a remote connection every
// poll is a blocked round trip that also occupies the connection's write
// path, so training RPCs queue behind it. Primary-context state on the
// per-connection server process can only change through this client's own
// RPCs, so the state is cached per device and invalidated by the manual
// wrappers of the four mutators (retain/release/reset/set-flags).
static std::atomic<uint64_t>
    lupine_primary_ctx_states[kLupineMaxCachedDevices]{};

void lupine_invalidate_primary_ctx_state(CUdevice dev) {
  if (static_cast<unsigned>(dev) < kLupineMaxCachedDevices)
    lupine_primary_ctx_states[dev].store(0, std::memory_order_release);
}

bool lupine_primary_ctx_state_cache_lookup(conn_t *conn, CUdevice dev,
                                           unsigned int *flags, int *active) {
  if (conn == nullptr || static_cast<unsigned>(dev) >= kLupineMaxCachedDevices)
    return false;
  uint64_t raw = lupine_primary_ctx_states[dev].load(std::memory_order_acquire);
  if (static_cast<int64_t>(raw) >= 0)
    return false;
  *flags = static_cast<unsigned int>(raw);
  *active = static_cast<int>(static_cast<int64_t>(raw << 1) >> 33);
  return true;
}

void lupine_primary_ctx_state_cache_insert(conn_t *conn, CUdevice dev,
                                           unsigned int flags, int active) {
  if (conn != nullptr && static_cast<unsigned>(dev) < kLupineMaxCachedDevices) {
    uint64_t raw =
        (uint64_t{1} << 63) |
        ((static_cast<uint64_t>(static_cast<uint32_t>(active)) & 0x7fffffffULL)
         << 32) |
        flags;
    lupine_primary_ctx_states[dev].store(raw, std::memory_order_release);
  }
}

static auto &lupine_kernel_function_cache() {
  static lupine_lazy_cache<libcuckoo::cuckoohash_map<
      lupine_kernel_function_key, CUfunction, lupine_key_hash>>
      cache;
  return cache;
}

static auto &lupine_kernel_function_keys() {
  return lupine_leaked<
      std::unordered_map<CUkernel, std::vector<lupine_kernel_function_key>>>(
      [] {});
}

static auto &lupine_occupancy_cache() {
  static lupine_lazy_cache<
      libcuckoo::cuckoohash_map<lupine_occupancy_key, int, lupine_key_hash>>
      cache;
  return cache;
}

template <typename T, size_t InlineCount> struct lupine_inline_buf {
  T inline_storage[InlineCount];
  std::unique_ptr<T[]> heap_storage;
  T *alloc(size_t count) {
    if (count <= InlineCount) {
      return inline_storage;
    }
    heap_storage.reset(new (std::nothrow) T[count]);
    return heap_storage.get();
  }
};

template <typename T, size_t InlineCount> struct lupine_inline_vec {
  T inline_storage[InlineCount];
  std::vector<T> overflow;
  size_t count = 0;
  const T *data() const {
    return overflow.empty() ? inline_storage : overflow.data();
  }
  bool push(const T &value) {
    if (overflow.empty() && count < InlineCount) {
      inline_storage[count++] = value;
      return true;
    }
    try {
      if (overflow.empty()) {
        overflow.reserve(InlineCount * 2);
        overflow.assign(inline_storage, inline_storage + count);
      }
      overflow.push_back(value);
      ++count;
      return true;
    } catch (const std::bad_alloc &) {
      return false;
    }
  }
};

struct lupine_wire_attribute_record {
  CUresult result = CUDA_ERROR_UNKNOWN;
  int attribute = 0;
  int value = 0;
};
static_assert(sizeof(lupine_wire_attribute_record) == 12);

// Snapshot responses carry the same (status, attribute, value) records for
// functions and kernels. They populate queried values, never last-set values.
template <typename Key>
static int lupine_read_attributes(conn_t *conn, Key key) {
  uint32_t attribute_count = 0;
  if (rpc_read_buffer(conn, &attribute_count, sizeof(attribute_count)) < 0 ||
      attribute_count > 64 * 1024) {
    return -1;
  }
  if (attribute_count == 0) {
    return 0;
  }
  lupine_inline_buf<lupine_wire_attribute_record, 64> buf;
  lupine_wire_attribute_record *records = buf.alloc(attribute_count);
  if (!records ||
      rpc_read_buffer(conn, records, attribute_count * sizeof(*records)) < 0) {
    return -1;
  }
  for (uint32_t i = 0; i < attribute_count; ++i) {
    if (records[i].result == CUDA_SUCCESS) {
      key.attribute = records[i].attribute;
      lupine_attribute_set(key, records[i].value);
    }
  }
  return 0;
}

static auto &lupine_param_info_cache() {
  static lupine_lazy_cache<libcuckoo::cuckoohash_map<
      lupine_param_info_key, lupine_param_info_value, lupine_key_hash>>
      cache;
  return cache;
}

static auto &lupine_param_layout_cache() {
  static lupine_lazy_cache<libcuckoo::cuckoohash_map<
      lupine_param_layout_key, std::shared_ptr<const lupine_param_layout>,
      lupine_key_hash>>
      cache;
  return cache;
}

static std::shared_ptr<const lupine_param_layout>
lupine_make_param_layout(const lupine_param_info_value *params, size_t count) {
  auto layout = std::make_shared<lupine_param_layout>();
  layout->sizes.resize(count);
  layout->offsets.resize(count);
  for (size_t i = 0; i < count; ++i) {
    layout->offsets[i] = params[i].offset;
    layout->sizes[i] = params[i].size;
  }
  return layout;
}

template <typename Handle> struct lupine_named_handle_view {
  Handle handle = nullptr;
  std::string_view name;
};

template <typename Handle> struct lupine_named_handle_key {
  Handle handle = nullptr;
  std::string name;

  bool operator==(const lupine_named_handle_key &other) const {
    return handle == other.handle && name == other.name;
  }
  bool operator==(const lupine_named_handle_view<Handle> &other) const {
    return handle == other.handle && name == other.name;
  }
  friend bool operator==(const lupine_named_handle_view<Handle> &lhs,
                         const lupine_named_handle_key &rhs) {
    return lhs.handle == rhs.handle && lhs.name == rhs.name;
  }
};

template <typename Handle> struct lupine_named_handle_hash {
  size_t operator()(const lupine_named_handle_key<Handle> &key) const {
    return reinterpret_cast<uintptr_t>(key.handle) ^
           std::hash<std::string_view>()(key.name);
  }
  size_t operator()(const lupine_named_handle_view<Handle> &key) const {
    return reinterpret_cast<uintptr_t>(key.handle) ^
           std::hash<std::string_view>()(key.name);
  }
};

template <typename Handle, typename Value>
using lupine_named_handle_map =
    libcuckoo::cuckoohash_map<lupine_named_handle_key<Handle>, Value,
                              lupine_named_handle_hash<Handle>,
                              std::equal_to<>>;

// Filled from library-load responses; serves cuLibraryGetKernel without a
// round trip.
static auto &lupine_library_kernel_names() {
  static lupine_lazy_cache<lupine_named_handle_map<CUlibrary, CUkernel>> cache;
  return cache;
}

// Filled from module-load responses; serves cuModuleGetFunction without a
// round trip. A loaded module's function set is fixed until it unloads.
// Unloading a module erases its entries and its functions' cached layouts and
// attributes (lupine_release_module_retained_strings); destroying the context
// that holds it drops the whole cache. Those are the only ways the server frees
// a module handle, so a recycled one never resolves against the functions of
// the module that held it before.
static auto &lupine_module_function_names() {
  static lupine_lazy_cache<lupine_named_handle_map<CUmodule, CUfunction>> cache;
  return cache;
}

static std::mutex &lupine_library_kernel_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static auto &lupine_private_6e16_nodes() {
  // cuFFT keeps these opaque nodes for the lifetime of its plans. Keep every
  // client-side proxy at a stable address until process exit as well; a small
  // ring aliases live nodes once a sufficiently large or multi-GPU plan asks
  // the driver to enumerate more functions.
  return lupine_leaked<std::vector<std::unique_ptr<unsigned char[]>>>([] {});
}

static std::mutex &lupine_private_6e16_node_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static unsigned char *lupine_allocate_private_6e16_node() {
  auto node = std::unique_ptr<unsigned char[]>(
      new (std::nothrow) unsigned char[0x500]());
  if (node == nullptr) {
    return nullptr;
  }
  unsigned char *address = node.get();
  std::lock_guard<std::mutex> lock(lupine_private_6e16_node_mutex());
  try {
    lupine_private_6e16_nodes().push_back(std::move(node));
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
  return address;
}

static std::atomic<size_t> lupine_library_kernel_count{0};
static std::atomic<size_t> lupine_private_node_count{0};

static std::mutex &lupine_private_node_mutex() {
  static std::mutex mutex;
  return mutex;
}

static void *lupine_local_libcuda_handle() {
  static std::once_flag once;
  static void *handle = nullptr;
  std::call_once(once, []() {
#ifdef __linux__
    lupine_native_cuda_call_guard native_call;
#endif
    const char *override_path = getenv("LUPINE_REAL_LIBCUDA");
#if defined(_WIN32)
    if (override_path != nullptr && override_path[0] != '\0') {
      handle = reinterpret_cast<void *>(LoadLibraryA(override_path));
      if (handle != nullptr) {
        return;
      }
    }
    char system_directory[MAX_PATH + 1] = {};
    UINT length = GetSystemDirectoryA(system_directory, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
      return;
    }
    std::string path(system_directory, length);
    path += "\\nvcuda.dll";
    handle = reinterpret_cast<void *>(LoadLibraryA(path.c_str()));
#else
    const char *paths[] = {
        override_path,
#if !defined(__APPLE__)
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
#endif
        nullptr,
    };
    for (const char *path : paths) {
      if (path == nullptr || path[0] == '\0') {
        continue;
      }
      handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        return;
      }
    }
#endif
  });
  return handle;
}

bool lupine_local_cuda_available() {
  return lupine_local_libcuda_handle() != nullptr;
}

extern "C" void *lupine_real_cuda_symbol(const char *name) {
  void *handle = lupine_local_libcuda_handle();
  if (handle == nullptr || name == nullptr) {
    return nullptr;
  }
  return lupine_real_dlsym(handle, name);
}

// Client-answered entry points must fail with NOT_INITIALIZED until cuInit;
// forwarded ones get the server's own state.
static std::atomic<bool> lupine_cuda_initialized{false};

bool lupine_cuda_is_initialized() {
  return lupine_cuda_initialized.load(std::memory_order_acquire);
}

// Every library calls cuInit; the server stays initialized, so one success
// per connection answers the rest. The driver still rejects non-zero flags.
static CUresult lupine_remote_cuInit(conn_t *conn, unsigned int flags) {
  auto *slot = lupine_conn_slot(conn);
  if (flags == 0 && slot && slot->initialized.load(std::memory_order_acquire))
    return CUDA_SUCCESS;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (!lupine_rpc_call(conn, RPC_cuInit, std::tie(flags), std::tie(result)))
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (result == CUDA_SUCCESS && slot)
    slot->initialized.store(true, std::memory_order_release);
  return result;
}

// The driver version and module loading mode are fixed for the life of the
// server process, so one reply per connection answers every library's query.
bool lupine_driver_version_cache_lookup(conn_t *conn, int *driverVersion) {
  auto *s = lupine_conn_slot(conn);
  return s != nullptr &&
         lupine_atomic_u32_load(s->driver_version, driverVersion);
}

void lupine_driver_version_cache_insert(conn_t *conn, int driverVersion) {
  if (auto *s = lupine_conn_slot(conn))
    lupine_atomic_u32_store(s->driver_version, driverVersion);
}

bool lupine_loading_mode_cache_lookup(conn_t *conn, CUmoduleLoadingMode *mode) {
  auto *s = lupine_conn_slot(conn);
  return s != nullptr && lupine_atomic_u32_load(s->loading_mode, mode);
}

void lupine_loading_mode_cache_insert(conn_t *conn, CUmoduleLoadingMode mode) {
  if (auto *s = lupine_conn_slot(conn))
    lupine_atomic_u32_store(s->loading_mode, mode);
}

static std::mutex &lupine_local_cuinit_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

extern "C" CUresult cuInit(unsigned int flags) {
  CUresult first_error = CUDA_SUCCESS;
  bool initialized_any = false;
  if (lupine_local_cuda_available()) {
    // lupine_apply_visible_devices filters the combined local + remote device
    // table once. Hide CUDA_VISIBLE_DEVICES while initializing the local
    // driver so it does not pre-filter or reject local devices beforehand.
    CUresult result = CUDA_ERROR_UNKNOWN;
    {
      std::lock_guard<std::mutex> lock(lupine_local_cuinit_mutex());
      const char *raw_visible = getenv("CUDA_VISIBLE_DEVICES");
      const bool had_visible = raw_visible != nullptr;
      std::string saved_visible = had_visible ? raw_visible : "";
      if (had_visible) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "");
#else
        unsetenv("CUDA_VISIBLE_DEVICES");
#endif
      }
      result = lupine_call_real_cuda_fn("cuInit", flags);
      if (had_visible) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", saved_visible.c_str());
#else
        setenv("CUDA_VISIBLE_DEVICES", saved_visible.c_str(), 1);
#endif
      }
    }
    if (result != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
      first_error = result;
    } else if (result == CUDA_SUCCESS) {
      initialized_any = true;
    }
  }
  if (rpc_open() == 0) {
    for (int i = 0; i < rpc_size(); ++i) {
      CUresult result =
          lupine_remote_cuInit(rpc_client_get_connection(i), flags);
      if (result != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
        first_error = result;
      } else if (result == CUDA_SUCCESS) {
        initialized_any = true;
      }
    }
  }
  if (initialized_any) {
    lupine_cuda_initialized.store(true, std::memory_order_release);
    return CUDA_SUCCESS;
  }
  if (first_error != CUDA_SUCCESS) {
    return first_error;
  }
  return getenv("LUPINE_SERVER") != nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                            : CUDA_ERROR_NO_DEVICE;
}

extern "C" CUresult cuDeviceGetCount(int *count) {
  return lupine_virtual_device_count(count);
}

extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  return lupine_virtual_device_for_ordinal(device, ordinal);
}

static void lupine_record_library_image(CUlibrary library, lupine_route route,
                                        uint32_t kind,
                                        const unsigned char *image,
                                        size_t image_size, const void *code) {
  int route_id = lupine_route_identity(route);
  if (library == nullptr || image == nullptr || image_size == 0 ||
      route_id == -2 || !lupine_devices_span_routes()) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  lupine_library_images()[library] = {
      kind,
      code,
      std::vector<unsigned char>(image, image + image_size),
      {{route_id, library}}};
}

static void lupine_cache_library_kernel(CUkernel kernel, CUlibrary library,
                                        std::string name, lupine_route route) {
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto &kernels = lupine_library_kernels();
  auto &record = kernels[kernel];
  record.library = library;
  record.name = std::move(name);
  record.kernels_by_route[lupine_route_identity(route)] = kernel;
  lupine_library_kernel_count.store(kernels.size(), std::memory_order_release);
}

bool lupine_try_cached_library_kernel(CUlibrary library, const char *name,
                                      CUkernel *kernel) {
  return lupine_library_kernel_names().find(
      lupine_named_handle_view<CUlibrary>{library, name}, *kernel);
}

extern "C" CUresult lupine_record_library_kernel(CUkernel kernel,
                                                 CUlibrary library,
                                                 const char *name,
                                                 lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (kernel == nullptr || library == nullptr || name == nullptr ||
      route_id == -2) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_note_function_owner_route(reinterpret_cast<CUfunction>(kernel), route);
  if (!lupine_route_is_local(route)) {
    CUresult result = lupine_warm_kernel_param_info(kernel);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  lupine_cache_library_kernel(kernel, library, name, route);
  return CUDA_SUCCESS;
}

static void lupine_cache_kernel_function(int route_id, CUcontext context,
                                         CUkernel kernel, CUfunction function) {
  lupine_kernel_function_key key{route_id, context, kernel};
  lupine_kernel_function_cache().insert_or_assign(key, function);
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto &keys = lupine_kernel_function_keys()[kernel];
  if (std::find(keys.begin(), keys.end(), key) == keys.end())
    keys.push_back(key);
}

static void lupine_record_library_function(CUfunction function,
                                           CUkernel kernel) {
  if (function == nullptr || kernel == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  lupine_function_kernels()[function] = kernel;
  auto kernel_record = lupine_library_kernels().find(kernel);
  if (kernel_record != lupine_library_kernels().end() &&
      kernel_record->second.library != nullptr) {
    lupine_library_functions()[function] = kernel_record->second.library;
  }
}

extern "C" void lupine_record_library_module(CUmodule module,
                                             CUlibrary library) {
  if (module == nullptr || library == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  lupine_library_modules()[module] = library;
}

extern "C" CUresult lupine_record_module_function(CUfunction function,
                                                  CUmodule module,
                                                  const char *name,
                                                  lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (function == nullptr || module == nullptr || name == nullptr ||
      route_id == -2) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_note_function_owner_route(function, route);
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &record = lupine_module_functions()[function];
    record.module = module;
    if (record.name.empty()) {
      record.name = name;
    }
  }
  return CUDA_SUCCESS;
}

static void lupine_cache_param_layout(
    uintptr_t handle, bool is_kernel,
    const std::shared_ptr<const lupine_param_layout> &layout) {
  lupine_param_layout_cache().insert_or_assign(
      lupine_param_layout_key{handle, is_kernel}, layout);
}

static void lupine_prefetch_function_param_layout(CUfunction function,
                                                  lupine_route route) {
  if (lupine_route_is_local(route)) {
    return;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(
          conn, LUPINE_RPC_lupineFunctionParamLayoutSnapshot) < 0 ||
      rpc_write(conn, &function, sizeof(function)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return;
  }

  uintptr_t handle = reinterpret_cast<uintptr_t>(function);
  lupine_inline_vec<lupine_param_info_value, 64> params;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  for (;;) {
    size_t offset = 0, size = 0;
    if (rpc_read_buffer(conn, &offset, sizeof(offset)) < 0 ||
        rpc_read_buffer(conn, &size, sizeof(size)) < 0 ||
        rpc_read_buffer(conn, &result, sizeof(result)) < 0) {
      return;
    }
    if (result != CUDA_SUCCESS) {
      break;
    }
    if (!params.push({offset, size})) {
      return;
    }
  }
  if (rpc_read_end(conn) < 0) {
    return;
  }
  if (result == CUDA_ERROR_INVALID_VALUE) {
    lupine_cache_param_layout(
        handle, false, lupine_make_param_layout(params.data(), params.count));
  } else {
    for (size_t i = 0; i < params.count; ++i) {
      lupine_param_info_cache().insert_or_assign(
          lupine_param_info_key{handle, i, false}, params.data()[i]);
    }
  }
}

// Reads the function snapshot a module-load response carries into the caches
// cuModuleGetFunction, cuFuncGetParamInfo and cuFuncGetAttribute read, so none
// of the three costs a round trip. A response that carries no functions (a
// failed load, or a server whose driver cannot enumerate a module) leaves the
// caches alone and every lookup falls back to its own request.
static constexpr uint32_t LUPINE_PARAM_COUNT_UNKNOWN = UINT32_MAX;

static int lupine_read_callable_wire_record(
    conn_t *conn, int route_id, CUkernel *kernel_out, CUfunction *function_out,
    std::string *name_out,
    std::shared_ptr<const lupine_param_layout> *layout_out,
    uint32_t *param_count_out) {
  uint32_t name_length = 0;
  if (rpc_read(conn, &name_length, sizeof(name_length)) < 0 ||
      name_length == 0 || name_length > 64 * 1024) {
    return -1;
  }
  lupine_inline_buf<char, 1024> name_buf;
  char *name = name_buf.alloc(name_length);
  if (!name || rpc_read(conn, name, name_length) < 0 ||
      name[name_length - 1] != '\0') {
    return -1;
  }
  if (kernel_out != nullptr &&
      (rpc_read(conn, kernel_out, sizeof(*kernel_out)) < 0 ||
       *kernel_out == nullptr)) {
    return -1;
  }
  uint32_t param_count = 0;
  if (rpc_read(conn, function_out, sizeof(*function_out)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0) {
    return -1;
  }
  const bool params_known = param_count != LUPINE_PARAM_COUNT_UNKNOWN;
  const uint32_t wire_params = params_known ? param_count : 0;
  if (wire_params > 64 * 1024) {
    return -1;
  }
  lupine_inline_buf<lupine_param_info_value, 64> params_buf;
  lupine_param_info_value *params_ptr = params_buf.alloc(wire_params);
  if (!params_ptr ||
      (wire_params != 0 &&
       rpc_read(conn, params_ptr, wire_params * sizeof(*params_ptr)) < 0) ||
      lupine_read_attributes(conn, lupine_function_attribute_key{
                                       route_id, *function_out, 0}) < 0) {
    return -1;
  }
  *name_out = std::string(name, name_length - 1);
  *layout_out = params_known ? lupine_make_param_layout(params_ptr, wire_params)
                             : nullptr;
  *param_count_out = param_count;
  return 0;
}

static int lupine_read_module_functions(conn_t *conn, CUmodule module,
                                        lupine_route route) {
  uint32_t count = 0;
  if (rpc_read(conn, &count, sizeof(count)) < 0 || count > 1024 * 1024) {
    return -1;
  }
  int route_id = lupine_route_identity(route);
  for (uint32_t i = 0; i < count; ++i) {
    CUfunction function = nullptr;
    std::string name;
    std::shared_ptr<const lupine_param_layout> layout;
    uint32_t param_count = 0;
    if (lupine_read_callable_wire_record(conn, route_id, nullptr, &function,
                                         &name, &layout, &param_count) < 0) {
      return -1;
    }
    if (param_count != LUPINE_PARAM_COUNT_UNKNOWN) {
      lupine_cache_param_layout(reinterpret_cast<uintptr_t>(function), false,
                                layout);
    }
    lupine_note_function_owner_route(function, route);
    {
      std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
      lupine_module_functions()[function] = {module, name};
    }
    lupine_module_function_names().insert_or_assign(
        lupine_named_handle_key<CUmodule>{module, std::move(name)}, function);
  }
  return 0;
}

static void lupine_prefill_function_attribute_snapshot(CUfunction function,
                                                       lupine_route route,
                                                       conn_t *conn) {
  if (function != nullptr && lupine_prepare_rpc(conn) >= 0 &&
      rpc_write_start_request(
          conn, LUPINE_RPC_lupineFunctionAttributeSnapshot) >= 0 &&
      rpc_write(conn, &function, sizeof(function)) >= 0 &&
      rpc_wait_for_response(conn) >= 0 &&
      lupine_read_attributes(
          conn, lupine_function_attribute_key{lupine_route_identity(route),
                                              function, 0}) >= 0)
    (void)rpc_read_end(conn);
}

bool lupine_try_cached_module_function(CUmodule module, const char *name,
                                       lupine_route route,
                                       CUfunction *function) {
  CUfunction cached = nullptr;
  if (!lupine_module_function_names().find(
          lupine_named_handle_view<CUmodule>{module, name}, cached)) {
    return false;
  }
  *function = cached;
  (void)lupine_record_module_function(cached, module, name, route);
  return true;
}

void lupine_finish_module_get_function(CUfunction function, CUmodule module,
                                       const char *name, lupine_route route) {
  (void)lupine_record_module_function(function, module, name, route);
  lupine_prefetch_function_param_layout(function, route);
  if (!lupine_route_is_local(route)) {
    lupine_prefill_function_attribute_snapshot(function, route,
                                               lupine_route_remote_conn(route));
  }
}

static int lupine_read_library(conn_t *conn, CUlibrary library);

static CUresult lupine_load_recorded_library_on_route(CUlibrary source_library,
                                                      lupine_route route,
                                                      CUlibrary *library);

static CUresult lupine_mmap_readonly_file(const char *path, void **mapping,
                                          size_t *size) {
  int fd = open(path, O_RDONLY | O_BINARY);
  if (fd < 0) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  // FILE_NOT_FOUND is reserved for open() failing.
  lupine_file_stat st = {};
  if (fstat(fd, &st) < 0) {
    close(fd);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (st.st_size <= 0) {
    close(fd);
    return CUDA_ERROR_INVALID_IMAGE;
  }
  *size = static_cast<size_t>(st.st_size);
  *mapping = mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return *mapping == MAP_FAILED ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
}

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  if (module == nullptr || fname == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    CUresult result = lupine_call_real_cuda_fn("cuModuleLoad", module, fname);
    if (result == CUDA_SUCCESS && module != nullptr) {
      lupine_note_module_owner_route(*module, route);
    }
    return result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  void *mapping = MAP_FAILED;
  size_t mapped_size = 0;
  CUresult result = lupine_mmap_readonly_file(fname, &mapping, &mapped_size);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  bool failed = lupine_prepare_rpc(conn) < 0 ||
                rpc_write_start_request(conn, RPC_cuModuleLoad) < 0 ||
                rpc_write(conn, &mapped_size, sizeof(mapped_size)) < 0 ||
                rpc_write(conn, mapping, mapped_size) < 0 ||
                rpc_wait_for_response(conn) < 0 ||
                rpc_read(conn, module, sizeof(CUmodule)) < 0 ||
                rpc_read(conn, &result, sizeof(result)) < 0 ||
                lupine_read_module_functions(conn, *module, route) < 0 ||
                rpc_read_end(conn) < 0;
  munmap(mapping, mapped_size);
  if (failed) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result == CUDA_SUCCESS && module != nullptr) {
    lupine_note_module_owner_route(*module, route);
  }
  return result;
}

static CUresult lupine_load_recorded_library_on_route(CUlibrary source_library,
                                                      lupine_route route,
                                                      CUlibrary *library) {
  if (library == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *library = nullptr;
  int route_id = lupine_route_identity(route);
  if (source_library == nullptr || route_id == -2) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_library_image_record record;
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_library_images().find(source_library);
    if (it == lupine_library_images().end()) {
      // Without a recorded image, the library only exists on its loading route.
      if (lupine_route_identity(lupine_route_for_library(source_library)) ==
          route_id) {
        *library = source_library;
        return CUDA_SUCCESS;
      }
      return CUDA_ERROR_NOT_FOUND;
    }
    auto cached = it->second.libraries_by_route.find(route_id);
    if (cached != it->second.libraries_by_route.end()) {
      *library = cached->second;
      return CUDA_SUCCESS;
    }
    record = it->second;
  }

  CUlibrary loaded = nullptr;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    if (record.code == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    result = lupine_call_real_cuda_fn("cuLibraryLoadData", &loaded, record.code,
                                      static_cast<CUjit_option *>(nullptr),
                                      static_cast<void **>(nullptr), 0u,
                                      static_cast<CUlibraryOption *>(nullptr),
                                      static_cast<void **>(nullptr), 0u);
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    size_t image_size = record.image.size();
    unsigned int zero_options = 0;
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
        rpc_write(conn, &record.kind, sizeof(record.kind)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write(conn, record.image.data(), image_size) < 0 ||
        rpc_write(conn, &zero_options, sizeof(zero_options)) < 0 ||
        rpc_write(conn, &zero_options, sizeof(zero_options)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &loaded, sizeof(loaded)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && lupine_read_library(conn, loaded) < 0) ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }

  if (result != CUDA_SUCCESS || loaded == nullptr) {
    return result;
  }

  lupine_note_library_owner_route(loaded, route);
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_library_images()[source_library].libraries_by_route[route_id] =
        loaded;
    auto &loaded_record = lupine_library_images()[loaded];
    loaded_record.kind = record.kind;
    loaded_record.code = record.code;
    loaded_record.image = std::move(record.image);
    loaded_record.libraries_by_route[route_id] = loaded;
  }
  *library = loaded;
  return CUDA_SUCCESS;
}

static CUresult
lupine_resolve_library_kernel_for_route(CUfunction function, lupine_route route,
                                        CUfunction *resolved,
                                        bool *is_kernel = nullptr) {
  if (is_kernel != nullptr) {
    *is_kernel = false;
  }
  if (resolved == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resolved = function;
  int route_id = lupine_route_identity(route);
  if (function == nullptr ||
      lupine_library_kernel_count.load(std::memory_order_acquire) == 0) {
    return CUDA_SUCCESS;
  }

  lupine_library_kernel_record record;
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it =
        lupine_library_kernels().find(reinterpret_cast<CUkernel>(function));
    if (it == lupine_library_kernels().end()) {
      return CUDA_SUCCESS;
    }
    if (is_kernel != nullptr) {
      *is_kernel = true;
    }
    if (route_id == -2) {
      return CUDA_SUCCESS;
    }
    auto cached = it->second.kernels_by_route.find(route_id);
    if (cached != it->second.kernels_by_route.end()) {
      *resolved = reinterpret_cast<CUfunction>(cached->second);
      return CUDA_SUCCESS;
    }
    record = it->second;
  }

  CUlibrary library = nullptr;
  CUresult result =
      lupine_load_recorded_library_on_route(record.library, route, &library);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  CUkernel kernel = nullptr;
  result = cuLibraryGetKernel(&kernel, library, record.name.c_str());
  if (result != CUDA_SUCCESS || kernel == nullptr) {
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &kernels = lupine_library_kernels();
    kernels[reinterpret_cast<CUkernel>(function)].kernels_by_route[route_id] =
        kernel;
    lupine_library_kernel_count.store(kernels.size(),
                                      std::memory_order_release);
  }
  *resolved = reinterpret_cast<CUfunction>(kernel);
  return CUDA_SUCCESS;
}

// A library's modules and globals belong to a context, so they come from the
// library as loaded on the current context's connection.
extern "C" CUresult lupine_resolve_library_for_route(CUlibrary *library,
                                                     lupine_route route) {
  return lupine_load_recorded_library_on_route(*library, route, library);
}

// Library kernels are context-independent, so a device-scoped kernel call
// must name the kernel as loaded on that device's connection.
extern "C" CUresult lupine_resolve_kernel_for_route(CUkernel *kernel,
                                                    lupine_route route) {
  CUfunction resolved = nullptr;
  CUresult result = lupine_resolve_library_kernel_for_route(
      reinterpret_cast<CUfunction>(*kernel), route, &resolved);
  *kernel = reinterpret_cast<CUkernel>(resolved);
  return result;
}

static CUfunction lupine_translate_private_function(CUfunction function) {
  if (lupine_private_node_count.load(std::memory_order_acquire) == 0) {
    return function;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    auto it = lupine_private_node_map().find(function);
    if (it != lupine_private_node_map().end()) {
      return it->second;
    }
  }
  return function;
}

static CUresult lupine_get_remote_private_module_node(CUcontext context,
                                                      CUmodule module,
                                                      CUfunction *server_node,
                                                      uint64_t *server_owner);

extern "C" CUfunction
lupine_translate_private_function_for_rpc(CUfunction function) {
  return lupine_translate_private_function(function);
}

static bool lupine_device_attribute_is_virtualized(conn_t *conn,
                                                   CUdevice_attribute attrib) {
  switch (attrib) {
  case CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM:
  case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
  case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
  case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST:
    return true;
  case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
    return !lupine_connection_supports_concurrent_managed_access(conn);
  default:
    return false;
  }
}

// One round trip that pulls every immutable per-device value the server can
// enumerate and prefills the client caches, so a metadata scan (for example
// torch reading ~110 attributes for every device at init) costs one RTT
// instead of one per query. Attempted once per connection; on any failure the
// per-call RPC paths still work, so this is purely best-effort.
static void lupine_prefill_device_snapshot(conn_t *conn) {
  auto *slot = lupine_conn_slot(conn);
  if (slot == nullptr ||
      slot->snapshot_attempted.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, LUPINE_RPC_lupineDeviceSnapshot,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0) {
    return;
  }
  if (result != CUDA_SUCCESS) {
    rpc_read_end(conn);
    return;
  }
  uint32_t device_count = 0;
  if (rpc_read(conn, &device_count, sizeof(device_count)) < 0) {
    return;
  }
  std::vector<int32_t> pairs;
  std::vector<unsigned> capabilities;
  for (uint32_t ordinal = 0; ordinal < device_count; ++ordinal) {
    lupine_device_snapshot_info info;
    uint32_t pair_count = 0;
    if (rpc_read(conn, info.name, sizeof(info.name)) < 0 ||
        rpc_read(conn, &info.uuid, sizeof(info.uuid)) < 0 ||
        rpc_read(conn, &info.total_mem, sizeof(info.total_mem)) < 0 ||
        rpc_read(conn, &pair_count, sizeof(pair_count)) < 0) {
      return;
    }
    pairs.resize(static_cast<size_t>(pair_count) * 2);
    if (pair_count != 0 &&
        rpc_read(conn, pairs.data(), pairs.size() * sizeof(int32_t)) < 0) {
      return;
    }
    int major = 0;
    int minor = -1;
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
      if (pairs[pair * 2] == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) {
        major = pairs[pair * 2 + 1];
      } else if (pairs[pair * 2] ==
                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) {
        minor = pairs[pair * 2 + 1];
      }
    }
    capabilities.push_back(major > 0 && major <= 999 && minor >= 0 && minor <= 9
                               ? static_cast<unsigned>(major * 10 + minor)
                               : 0);
    CUdevice local_dev =
        lupine_local_device_for_remote(conn, static_cast<CUdevice>(ordinal));
    auto *dev_cache = lupine_ensure_device_cache(local_dev);
    if (dev_cache == nullptr) {
      continue;
    }
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
      int attrib = static_cast<int>(pairs[pair * 2]);
      int value = static_cast<int>(pairs[pair * 2 + 1]);
      if (lupine_device_attribute_is_virtualized(
              conn, static_cast<CUdevice_attribute>(attrib))) {
        value = 0;
      }
      if (static_cast<unsigned>(attrib) < kLupineMaxDevAttrs)
        lupine_atomic_u32_store(dev_cache->attrs[attrib], value);
    }
    info.name[sizeof(info.name) - 1] = '\0';
    std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
    dev_cache->snapshot = info;
    dev_cache->has_snapshot.store(true, std::memory_order_release);
  }
  if (rpc_read_end(conn) >= 0) {
    auto caps =
        std::make_shared<const std::vector<unsigned>>(std::move(capabilities));
    std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
    if (slot->conn.load(std::memory_order_relaxed) == conn)
      slot->capabilities = std::move(caps);
  }
}

bool lupine_device_attribute_cache_lookup(conn_t *conn, CUdevice dev,
                                          CUdevice_attribute attrib, int *pi) {
  if (conn == nullptr)
    return false;
  if (lupine_dev_attr_get(dev, attrib, pi))
    return true;
  lupine_prefill_device_snapshot(conn);
  return lupine_dev_attr_get(dev, attrib, pi);
}

void lupine_device_attribute_cache_insert(conn_t *conn, CUdevice dev,
                                          CUdevice_attribute attrib, int *pi) {
  if (conn == nullptr)
    return;
  if (lupine_device_attribute_is_virtualized(conn, attrib))
    *pi = 0;
  if (static_cast<unsigned>(attrib) < kLupineMaxDevAttrs)
    if (auto *p = lupine_ensure_device_cache(dev))
      lupine_atomic_u32_store(p->attrs[attrib], *pi);
}

template <typename Fn>
static bool lupine_device_snapshot_with(conn_t *conn, CUdevice dev, Fn &&fn) {
  if (conn == nullptr || static_cast<unsigned>(dev) >= kLupineMaxCachedDevices)
    return false;
  lupine_prefill_device_snapshot(conn);
  auto *p = lupine_device_caches[dev].load(std::memory_order_acquire);
  if (p == nullptr || !p->has_snapshot.load(std::memory_order_acquire))
    return false;
  lupine_device_snapshot_info info;
  {
    std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
    info = p->snapshot;
  }
  fn(info);
  return true;
}

bool lupine_device_snapshot_lookup_name(conn_t *conn, CUdevice dev, char *name,
                                        int len) {
  return len > 0 && lupine_device_snapshot_with(conn, dev, [&](const auto &i) {
           snprintf(name, static_cast<size_t>(len), "%s", i.name);
         });
}

bool lupine_device_snapshot_lookup_uuid(conn_t *conn, CUdevice dev,
                                        CUuuid *uuid) {
  return lupine_device_snapshot_with(conn, dev,
                                     [&](const auto &i) { *uuid = i.uuid; });
}

bool lupine_device_snapshot_lookup_total_mem(conn_t *conn, CUdevice dev,
                                             size_t *bytes) {
  return lupine_device_snapshot_with(conn, dev, [&](const auto &i) {
    *bytes = static_cast<size_t>(i.total_mem);
  });
}

static void lupine_stream_pool_init(lupine_route route, CUdevice dev,
                                    CUcontext ctx);
static void lupine_stream_pool_discard(int route_id, CUdevice dev,
                                       CUcontext ctx);
static void lupine_event_pool_discard(int route_id, CUdevice remote_device,
                                      CUcontext ctx);
static void lupine_forget_context_local_storage(CUcontext ctx);

// One forwarded retain keeps a device's primary context alive for every
// retain this client holds, so later retains are counted here and only the
// release that drops the count to zero goes through. The driver keeps its own
// count across Reset but deactivates the context, so Reset releases the
// forwarded retain and the next retain goes through again.
struct lupine_primary_ctx_retain {
  CUcontext context = nullptr;
  int count = 0;
  bool forwarded = false;
};

static std::mutex &lupine_primary_ctx_retain_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static std::unordered_map<int, lupine_primary_ctx_retain> &
lupine_primary_ctx_retains() {
  return lupine_leaked<std::unordered_map<int, lupine_primary_ctx_retain>>(
      [] {});
}

static CUresult lupine_remote_primary_ctx_release(conn_t *conn,
                                                  CUdevice remote_dev) {
  CUresult return_value;
  if (!lupine_rpc_call(conn, RPC_cuDevicePrimaryCtxRelease_v2,
                       std::tie(remote_dev), std::tie(return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_invalidate_current_context_cache();
  }
  return return_value;
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    lupine_invalidate_primary_ctx_state(dev);
    return_value =
        lupine_call_real_cuda_fn("cuDevicePrimaryCtxRetain", pctx, remote_dev);
    if (return_value == CUDA_SUCCESS && pctx != nullptr) {
      lupine_note_context_owner_route(*pctx, route);
    }
    return return_value;
  }
  std::lock_guard<std::mutex> lock(lupine_primary_ctx_retain_mutex());
  auto &retains = lupine_primary_ctx_retains();
  auto retained = retains.find(static_cast<int>(dev));
  if (retained == retains.end() || !retained->second.forwarded) {
    lupine_invalidate_primary_ctx_state(dev);
    conn_t *conn = lupine_route_remote_conn(route);
    CUcontext context = nullptr;
    if (!lupine_rpc_call_unscoped(conn, RPC_cuDevicePrimaryCtxRetain,
                                  std::tie(remote_dev),
                                  std::tie(context, return_value))) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
    lupine_note_context_owner_route(context, route);
    lupine_stream_pool_init(route, dev, context);
    retained =
        retains.emplace(static_cast<int>(dev), lupine_primary_ctx_retain{})
            .first;
    retained->second.context = context;
    retained->second.forwarded = true;
  }
  ++retained->second.count;
  if (pctx != nullptr) {
    *pctx = retained->second.context;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    lupine_invalidate_primary_ctx_state(dev);
    CUresult return_value =
        lupine_call_real_cuda_fn("cuDevicePrimaryCtxRelease_v2", remote_dev);
    if (return_value == CUDA_SUCCESS) {
      lupine_invalidate_current_context_cache();
    }
    return return_value;
  }
  std::lock_guard<std::mutex> lock(lupine_primary_ctx_retain_mutex());
  auto &retains = lupine_primary_ctx_retains();
  auto retained = retains.find(static_cast<int>(dev));
  CUcontext context = nullptr;
  if (retained != retains.end()) {
    if (--retained->second.count > 0) {
      return CUDA_SUCCESS;
    }
    bool forwarded = retained->second.forwarded;
    context = retained->second.context;
    retains.erase(retained);
    if (!forwarded) {
      return CUDA_SUCCESS;
    }
  }
  lupine_invalidate_primary_ctx_state(dev);
  lupine_forget_context_local_storage(context);
  CUresult result = lupine_remote_primary_ctx_release(
      lupine_route_remote_conn(route), remote_dev);
  // After the server teardown: a batch requested before it is discarded too.
  lupine_stream_pool_discard(lupine_route_identity(route), dev, nullptr);
  lupine_event_pool_discard(lupine_route_identity(route), remote_dev, nullptr);
  return result;
}

extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  lupine_invalidate_primary_ctx_state(dev);
  // A reset unloads the primary context's modules; see
  // lupine_forget_destroyed_context.
  lupine_invalidate_function_caches();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn("cuDevicePrimaryCtxReset_v2", remote_dev);
    if (return_value == CUDA_SUCCESS) {
      lupine_invalidate_current_context_cache();
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::lock_guard<std::mutex> lock(lupine_primary_ctx_retain_mutex());
  if (!lupine_rpc_call(conn, RPC_cuDevicePrimaryCtxReset_v2,
                       std::tie(remote_dev), std::tie(return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_stream_pool_discard(lupine_route_identity(route), dev, nullptr);
  lupine_event_pool_discard(lupine_route_identity(route), remote_dev, nullptr);
  if (return_value == CUDA_SUCCESS) {
    lupine_invalidate_current_context_cache();
    auto &retains = lupine_primary_ctx_retains();
    auto retained = retains.find(static_cast<int>(dev));
    if (retained != retains.end() && retained->second.forwarded) {
      retained->second.forwarded = false;
      lupine_forget_context_local_storage(retained->second.context);
      (void)lupine_remote_primary_ctx_release(conn, remote_dev);
    }
  }
  return return_value;
}

static CUresult lupine_cuGetParamInfo_cached(uintptr_t handle,
                                             size_t param_index,
                                             size_t *param_offset,
                                             size_t *param_size, bool kernel) {
  if (param_offset == nullptr || param_size == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::shared_ptr<const lupine_param_layout> layout;
  if (lupine_param_layout_cache().find(lupine_param_layout_key{handle, kernel},
                                       layout) &&
      layout != nullptr) {
    if (param_index >= layout->sizes.size()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    *param_offset = layout->offsets[param_index];
    *param_size = layout->sizes[param_index];
    return CUDA_SUCCESS;
  }

  lupine_param_info_key key{handle, param_index, kernel};
  lupine_param_info_value cached;
  if (lupine_param_info_cache().find(key, cached)) {
    *param_offset = cached.offset;
    *param_size = cached.size;
    return CUDA_SUCCESS;
  }

  CUfunction function = reinterpret_cast<CUfunction>(handle);
  lupine_route route = lupine_route_for_function(function);
  if (lupine_route_is_local(route)) {
    if (kernel) {
      return lupine_call_real_cuda_fn<CUDA_ERROR_NOT_SUPPORTED>(
          "cuKernelGetParamInfo", reinterpret_cast<CUkernel>(handle),
          param_index, param_offset, param_size);
    }
    return lupine_call_real_cuda_fn<CUDA_ERROR_NOT_SUPPORTED>(
        "cuFuncGetParamInfo", function, param_index, param_offset, param_size);
  }

  CUcontext current_ctx = lupine_current_context_hint();
  if (kernel && current_ctx != nullptr &&
      lupine_routes_share_server(lupine_route_for_context(current_ctx),
                                 route)) {
    (void)lupine_set_current_context_on_route(route, current_ctx);
  }

  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  size_t offset = 0;
  size_t size = 0;
  conn_t *conn = lupine_route_remote_conn(route);
  int rpc = kernel ? RPC_cuKernelGetParamInfo : RPC_cuFuncGetParamInfo;
  if (!lupine_rpc_call_unscoped(conn, rpc, std::tie(handle, param_index),
                                std::tie(offset, size, result))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result == CUDA_SUCCESS) {
    lupine_param_info_cache().insert_or_assign(
        key, lupine_param_info_value{offset, size});
    *param_offset = offset;
    *param_size = size;
  }
  return result;
}

extern "C" CUresult cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex,
                                         size_t *paramOffset,
                                         size_t *paramSize) {
  return lupine_cuGetParamInfo_cached(reinterpret_cast<uintptr_t>(kernel),
                                      paramIndex, paramOffset, paramSize, true);
}

extern "C" CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex,
                                       size_t *paramOffset, size_t *paramSize) {
  return lupine_cuGetParamInfo_cached(reinterpret_cast<uintptr_t>(func),
                                      paramIndex, paramOffset, paramSize,
                                      false);
}

CUresult lupine_prepare_kernel_get_function(CUfunction *pFunc, CUkernel *kernel,
                                            lupine_route *route,
                                            CUcontext *current_context) {
  if (pFunc == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult context_status = cuCtxGetCurrent(current_context);
  if (context_status != CUDA_SUCCESS) {
    return context_status;
  }
  if (route->kind == LUPINE_ROUTE_INVALID) {
    *route = lupine_route_for_function(reinterpret_cast<CUfunction>(*kernel));
  }
  CUfunction route_kernel_function = reinterpret_cast<CUfunction>(*kernel);
  CUresult resolve_status = lupine_resolve_library_kernel_for_route(
      route_kernel_function, *route, &route_kernel_function);
  if (resolve_status != CUDA_SUCCESS) {
    return resolve_status;
  }
  *kernel = reinterpret_cast<CUkernel>(route_kernel_function);
  if (!lupine_route_is_local(*route) &&
      lupine_kernel_function_cache().find(
          lupine_kernel_function_key{lupine_route_identity(*route),
                                     *current_context, *kernel},
          *pFunc)) {
    lupine_record_library_function(*pFunc, *kernel);
    return CUDA_SUCCESS;
  }
  return CUDA_ERROR_NOT_READY;
}

CUresult lupine_finish_kernel_get_function(CUresult return_value,
                                           lupine_route route,
                                           CUcontext current_context,
                                           CUkernel kernel, CUfunction function,
                                           CUfunction *pFunc) {
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  lupine_note_function_owner_route(function, route);
  if (!lupine_route_is_local(route)) {
    return_value = lupine_warm_func_param_info(function);
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
    lupine_cache_kernel_function(lupine_route_identity(route), current_context,
                                 kernel, function);
  }
  *pFunc = function;
  lupine_record_library_function(function, kernel);
  return CUDA_SUCCESS;
}

// A function is one kernel instantiated in one context, so a set on it moves
// exactly one kernel-attribute entry, and a function of a plain module moves
// none; without either pairing the whole cache is the only safe target.
static void lupine_kernel_attribute_cache_erase_for_function(
    int route_id, CUfunction function, int attrib) {
  CUkernel kernel = nullptr;
  int device = -1;
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_function_kernels().find(function);
    if (it != lupine_function_kernels().end()) {
      kernel = it->second;
    } else {
      auto record = lupine_module_functions().find(function);
      if (record != lupine_module_functions().end() &&
          lupine_library_modules().count(record->second.module) == 0) {
        return;
      }
    }
  }
  if (kernel == nullptr ||
      !lupine_attribute_get(
          lupine_function_attribute_key{route_id, function,
                                        LUPINE_FUNC_ATTRIBUTE_DEVICE},
          device)) {
    get_cache<lupine_attribute_cache::kernel>().clear();
    return;
  }
  get_cache<lupine_attribute_cache::kernel>().erase(
      lupine_kernel_attribute_key{route_id, kernel, attrib, device});
}

static std::atomic<bool> lupine_extended_func_attr{false};
static std::atomic<bool> lupine_extended_kernel_attr{false};
static std::atomic<uint64_t> lupine_kernel_attr_devices{0};

template <typename Map, typename Match>
static void lupine_cache_erase_if(lupine_lazy_cache<Map> &lazy, Match match) {
  Map *cache = lazy.try_get();
  if (cache == nullptr || lazy.empty_fast() || cache->empty()) {
    return;
  }
  auto table = cache->lock_table();
  for (auto it = table.begin(); it != table.end();) {
    it = match(it->first, it->second) ? table.erase(it) : std::next(it);
  }
}

// Drops the occupancy answers that depend on one kernel, whichever handle the
// caller queried it through. Occupancy of every other kernel stays valid.
static void lupine_occupancy_cache_erase_function(int route_id,
                                                  CUfunction function) {
  std::vector<CUfunction> handles = {function};
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &pairs = lupine_function_kernels();
    auto paired = pairs.find(function);
    if (paired != pairs.end()) {
      handles.push_back(reinterpret_cast<CUfunction>(paired->second));
    }
    for (const auto &[other, kernel] : pairs) {
      if (reinterpret_cast<CUfunction>(kernel) == function) {
        handles.push_back(other);
      }
    }
  }
  lupine_cache_erase_if(lupine_occupancy_cache(), [&](const auto &key,
                                                      const auto &) {
    return key.route_id == route_id && std::find(handles.begin(), handles.end(),
                                                 key.function) != handles.end();
  });
}

bool lupine_attribute_get(const lupine_function_attribute_key &key, int &value,
                          lupine_attribute_source source) {
  auto &cache = source == lupine_attribute_source::set
                    ? get_cache<lupine_attribute_cache::function_set>()
                    : get_cache<lupine_attribute_cache::function>();
  return cache.find(key, value);
}

bool lupine_attribute_get(const lupine_kernel_attribute_key &key, int &value) {
  return get_cache<lupine_attribute_cache::kernel>().find(key, value);
}

void lupine_attribute_set(const lupine_function_attribute_key &key, int value,
                          lupine_attribute_source source) {
  if (key.attribute < LUPINE_FUNC_ATTRIBUTE_DEVICE ||
      key.attribute >= CU_FUNC_ATTRIBUTE_MAX) {
    lupine_extended_func_attr.store(true, std::memory_order_relaxed);
  }
  if (source == lupine_attribute_source::query) {
    get_cache<lupine_attribute_cache::function>().insert_or_assign(key, value);
    return;
  }
  lupine_kernel_attribute_cache_erase_for_function(key.route_id, key.function,
                                                   key.attribute);
  get_cache<lupine_attribute_cache::function>().erase(key);
  lupine_occupancy_cache_erase_function(key.route_id, key.function);
  get_cache<lupine_attribute_cache::function_set>().insert_or_assign(key,
                                                                     value);
}

void lupine_attribute_set(const lupine_kernel_attribute_key &key, int value,
                          lupine_attribute_source source) {
  if (static_cast<unsigned>(key.attribute) >= CU_FUNC_ATTRIBUTE_MAX ||
      static_cast<unsigned>(key.device) >= 64) {
    lupine_extended_kernel_attr.store(true, std::memory_order_relaxed);
  } else if (uint64_t bit = uint64_t{1} << key.device;
             !(lupine_kernel_attr_devices.load(std::memory_order_relaxed) &
               bit)) {
    lupine_kernel_attr_devices.fetch_or(bit, std::memory_order_release);
  }
  auto &cache = get_cache<lupine_attribute_cache::kernel>();
  int cached = 0;
  if (source == lupine_attribute_source::query) {
    cache.insert_or_assign(key, value);
    return;
  }
  if (cache.find(key, cached) && cached == value) {
    return;
  }
  cache.insert_or_assign(key, value);
  lupine_occupancy_cache_erase_function(
      key.route_id, reinterpret_cast<CUfunction>(key.kernel));
  // A kernel set changes functions without a function-level override.
  // Conservatively drop queries of this attribute and last-set memos; retain
  // unrelated snapshot inputs. The next query resolves override precedence.
  lupine_cache_erase_if(get_cache<lupine_attribute_cache::function>(),
                        [&](const auto &other, const auto &) {
                          return other.route_id == key.route_id &&
                                 other.attribute == key.attribute;
                        });
  get_cache<lupine_attribute_cache::function_set>().clear();
}

// Occupancy is a pure function of the kernel's attributes and the device's
// limits, both already cached client-side, so NVIDIA's header-only calculator
// answers a miss without a round trip. Only inputs verified against the driver
// are modelled: sm_7x/8x with the default cache config (any cache-config setter
// disables the path for the process) and flags the calculator ignores there.
// Everything else goes to the driver.
static std::atomic<bool> lupine_local_occupancy_enabled{true};

extern "C" void lupine_disable_local_occupancy() {
  lupine_local_occupancy_enabled.store(false);
}

static bool lupine_occupancy_local(lupine_route route, CUfunction function,
                                   int blockSize, size_t dynamicSMemSize,
                                   unsigned int flags, int *numBlocks) {
  if (!lupine_local_occupancy_enabled.load() ||
      (flags & ~CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE) != 0) {
    return false;
  }
  static const int kFunctionInputs[] = {
      LUPINE_FUNC_ATTRIBUTE_DEVICE,
      CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_FUNC_ATTRIBUTE_NUM_REGS,
      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT};
  static const CUdevice_attribute kDeviceInputs[] = {
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
      CU_DEVICE_ATTRIBUTE_WARP_SIZE,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
      CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK};
  int route_id = lupine_route_identity(route);
  conn_t *conn = lupine_route_remote_conn(route);
  int f[sizeof(kFunctionInputs) / sizeof(kFunctionInputs[0])];
  CUdevice device = -1;
  // A CUfunction carries its context's device in the snapshot; a CUkernel
  // handle (cudart's lazy-loading path) is evaluated on the current context.
  if (lupine_attribute_get(
          lupine_function_attribute_key{route_id, function, kFunctionInputs[0]},
          f[0])) {
    device = lupine_local_device_for_remote(conn, f[0]);
    for (size_t i = 1; i < sizeof(f) / sizeof(f[0]); ++i) {
      if (!lupine_attribute_get(
              lupine_function_attribute_key{route_id, function,
                                            kFunctionInputs[i]},
              f[i])) {
        return false;
      }
    }
  } else {
    CUdevice remote_device = -1;
    if (!lupine_current_context_device_cache_lookup(
            lupine_current_context_hint(), &device) ||
        (remote_device = device,
         !lupine_translate_device_for_conn(conn, &remote_device))) {
      return false;
    }
    for (size_t i = 1; i < sizeof(f) / sizeof(f[0]); ++i) {
      if (!lupine_attribute_get(
              lupine_kernel_attribute_key{
                  route_id, reinterpret_cast<CUkernel>(function),
                  kFunctionInputs[i], static_cast<int>(remote_device)},
              f[i])) {
        return false;
      }
    }
  }
  if (device < 0) {
    return false;
  }
  lupine_prefill_device_snapshot(conn);
  int d[sizeof(kDeviceInputs) / sizeof(kDeviceInputs[0])];
  for (size_t i = 0; i < sizeof(d) / sizeof(d[0]); ++i) {
    if (!lupine_dev_attr_get(device, kDeviceInputs[i], &d[i])) {
      return false;
    }
  }
  if (d[0] != 7 && d[0] != 8) {
    return false;
  }
  cudaOccDeviceProp prop;
  prop.computeMajor = d[0];
  prop.computeMinor = d[1];
  prop.maxThreadsPerBlock = d[2];
  prop.maxThreadsPerMultiprocessor = d[3];
  prop.regsPerBlock = d[4];
  prop.regsPerMultiprocessor = d[5];
  prop.warpSize = d[6];
  prop.sharedMemPerBlock = static_cast<size_t>(d[7]);
  prop.sharedMemPerMultiprocessor = static_cast<size_t>(d[8]);
  prop.numSms = d[9];
  prop.sharedMemPerBlockOptin = static_cast<size_t>(d[10]);
  prop.reservedSharedMemPerBlock = static_cast<size_t>(d[11]);
  cudaOccFuncAttributes attr;
  attr.maxThreadsPerBlock = f[1];
  attr.numRegs = f[2];
  attr.sharedSizeBytes = static_cast<size_t>(f[3]);
  attr.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
  attr.maxDynamicSharedSizeBytes = static_cast<size_t>(f[4]);
  attr.numBlockBarriers = 1;
  cudaOccDeviceState state;
  state.carveoutConfig = f[5];
  cudaOccResult result;
  if (cudaOccMaxActiveBlocksPerMultiprocessor(&result, &prop, &attr, &state,
                                              blockSize, dynamicSMemSize) !=
      CUDA_OCC_SUCCESS) {
    return false;
  }
  *numBlocks = result.activeBlocksPerMultiprocessor;
  return true;
}

bool lupine_try_cached_occupancy(lupine_route route, CUfunction func,
                                 int blockSize, size_t dynamicSMemSize,
                                 unsigned int flags, bool with_flags,
                                 int *numBlocks) {
  lupine_occupancy_key key{lupine_route_identity(route),
                           func,
                           blockSize,
                           dynamicSMemSize,
                           flags,
                           with_flags};
  if (lupine_occupancy_cache().find(key, *numBlocks)) {
    return true;
  }
  if (lupine_occupancy_local(route, func, blockSize, dynamicSMemSize, flags,
                             numBlocks)) {
    lupine_occupancy_cache().insert_or_assign(key, *numBlocks);
    return true;
  }
  return false;
}

void lupine_store_cached_occupancy(lupine_route route, CUfunction func,
                                   int blockSize, size_t dynamicSMemSize,
                                   unsigned int flags, bool with_flags,
                                   int numBlocks) {
  lupine_occupancy_cache().insert_or_assign(
      lupine_occupancy_key{lupine_route_identity(route), func, blockSize,
                           dynamicSMemSize, flags, with_flags},
      numBlocks);
}

static CUresult lupine_get_remote_private_module_node(CUcontext context,
                                                      CUmodule module,
                                                      CUfunction *server_node,
                                                      uint64_t *server_owner) {
  if (server_node == nullptr || server_owner == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *server_node = nullptr;
  *server_owner = 0;
  lupine_route route = context != nullptr ? lupine_route_for_context(context)
                                          : lupine_route_for_module(module);
  int route_id = lupine_route_identity(route);
  if (module != nullptr && route_id != -2 &&
      lupine_route_identity(lupine_route_for_module(module)) != route_id) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (!lupine_rpc_call_unscoped(
          conn, LUPINE_RPC_cuPrivateGetModuleNode, std::tie(context, module),
          std::tie(*server_node, *server_owner, result))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result == CUDA_SUCCESS && *server_node != nullptr) {
    lupine_note_function_owner_route(*server_node, route);
    result = lupine_warm_func_param_info(*server_node);
  }
  return result;
}

static uint64_t lupine_private_export_slot_hash(const char *table_name,
                                                int slot) {
  if (table_name == nullptr || slot < 0) {
    return 0;
  }
  auto &hashes = lupine_private_export_hashes();
  auto it = hashes.find(table_name);
  if (it == hashes.end() || static_cast<size_t>(slot) >= it->second.size()) {
    return 0;
  }
  return it->second[slot];
}

extern "C" CUresult
lupine_private_export_slot_called(int slot, const char *table_name,
                                  uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                  uint64_t arg3, uint64_t arg4, uint64_t arg5) {
  static uint64_t lupine_private_2131_fake_object[64] = {};

  LUPINE_TRACE_LOG("LUPINE private export table called: "
                   << (table_name != nullptr ? table_name : "(null)") << "["
                   << slot << "]" << std::hex << std::showbase << " hash="
                   << lupine_private_export_slot_hash(table_name, slot)
                   << " args=" << arg0 << "," << arg1 << "," << arg2 << ","
                   << arg3 << "," << arg4 << "," << arg5 << std::dec
                   << std::noshowbase);

  if (table_name != nullptr &&
      strcmp(table_name, "da9151d33ae6cc41a5c04f26d533e328") == 0) {
#if !defined(_WIN32)
    // This process-local emulation is based on the cuFFT JIT-callback
    // consumer. cuSPARSE/nvJitLink consumers use the same UUID with a
    // different object contract; preserve the generic NOT_SUPPORTED fallback
    // unless the cuFFT JIT-callback API is loaded in this process.
    if (lupine_real_dlsym(RTLD_DEFAULT, "cufftXtSetJITCallback") == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
#endif
    if (slot == 1) {
      if (arg0 == 0 || arg2 == 0 || arg3 == 0) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      auto object = std::unique_ptr<lupine_private_jit_cache_object>(
          new (std::nothrow) lupine_private_jit_cache_object());
      if (object == nullptr) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      object->type = static_cast<uint32_t>(arg1);
      try {
        const auto *data = reinterpret_cast<const unsigned char *>(arg2);
        object->data.assign(data, data + static_cast<uint32_t>(arg3));
      } catch (const std::bad_alloc &) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      if (object->type == 1) {
        static constexpr unsigned char elf_magic[] = {0x7f, 'E', 'L', 'F'};
        auto cubin = std::search(object->data.begin(), object->data.end(),
                                 std::begin(elf_magic), std::end(elf_magic));
        if (cubin == object->data.end()) {
          return CUDA_ERROR_INVALID_IMAGE;
        }
        object->data.erase(object->data.begin(), cubin);
      }
      void *handle = object.get();
      try {
        std::lock_guard<std::mutex> lock(lupine_private_jit_cache_mutex());
        lupine_private_jit_cache_objects().emplace(handle, std::move(object));
      } catch (const std::bad_alloc &) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      *reinterpret_cast<void **>(arg0) = handle;
      return CUDA_SUCCESS;
    }

    if (slot == 3 || slot == 4) {
      if (arg0 == 0 || arg2 == 0 || arg3 == 0) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      std::lock_guard<std::mutex> lock(lupine_private_jit_cache_mutex());
      auto it = lupine_private_jit_cache_objects().find(
          reinterpret_cast<void *>(arg0));
      if (it == lupine_private_jit_cache_objects().end()) {
        return CUDA_ERROR_INVALID_HANDLE;
      }
      // Type 2 is a lookup by LTO input. An in-process implementation has no
      // persistent driver cache, so report a miss and let cuFFT run nvJitLink
      // locally. It then stores the linked cubin as a type-1 object.
      if (it->second->type != 1) {
        return CUDA_ERROR_NOT_FOUND;
      }
      const auto &bytes = it->second->data;
      if (slot == 3) {
        auto cubin_size = static_cast<uint32_t>(bytes.size());
        void *cubin = malloc(cubin_size);
        if (cubin == nullptr) {
          return CUDA_ERROR_OUT_OF_MEMORY;
        }
        memcpy(cubin, bytes.data(), cubin_size);
        *reinterpret_cast<void **>(arg2) = cubin;
        *reinterpret_cast<uint32_t *>(arg3) = cubin_size;
        return CUDA_SUCCESS;
      }
      if (arg3 < bytes.size()) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      memcpy(reinterpret_cast<void *>(arg2), bytes.data(), bytes.size());
      return CUDA_SUCCESS;
    }

    if (slot == 2) {
      std::lock_guard<std::mutex> lock(lupine_private_jit_cache_mutex());
      return lupine_private_jit_cache_objects().erase(
                 reinterpret_cast<void *>(arg0)) == 0
                 ? CUDA_ERROR_INVALID_HANDLE
                 : CUDA_SUCCESS;
    }

    if (slot == 6) {
      if (arg0 == 0) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      free(reinterpret_cast<void *>(arg0));
      return CUDA_SUCCESS;
    }
  }

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0) {
    if (slot == 4) {
      if (arg1 == 0) {
        return CUDA_ERROR_UNKNOWN;
      }
      uint64_t output = arg0 == 0 ? 0 : 1;
      *reinterpret_cast<uint64_t *>(arg1) = output;
      LUPINE_TRACE_LOG("LUPINE private 2131[4] output=" << output);
      return CUDA_SUCCESS;
    }
    if (slot == 51) {
      if (arg2 == 0) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      if (arg1 == UINT64_MAX) {
        return static_cast<CUresult>(400);
      }
      *reinterpret_cast<uint64_t *>(arg2) =
          reinterpret_cast<uint64_t>(&lupine_private_2131_fake_object[0]);
      return CUDA_SUCCESS;
    }
    if (slot == 39) {
      if (arg1 == 0 || arg2 == 0) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      *reinterpret_cast<uint64_t *>(arg2) = 0;
      return CUDA_SUCCESS;
    }
  }

  if (table_name != nullptr &&
      strcmp(table_name, "6e163fbeb958444d835ce182aff1991e") == 0 &&
      slot == 7) {
    CUfunction server_node = nullptr;
    uint64_t server_owner = 0;
    CUresult node_result = lupine_get_remote_private_module_node(
        reinterpret_cast<CUcontext>(arg0), reinterpret_cast<CUmodule>(arg1),
        &server_node, &server_owner);
    if (node_result != CUDA_SUCCESS || server_node == nullptr) {
      LUPINE_TRACE_LOG("LUPINE private 6e16[7] remote node failed result="
                       << static_cast<int>(node_result)
                       << " module=" << reinterpret_cast<void *>(arg1));
      return node_result == CUDA_SUCCESS ? CUDA_ERROR_NOT_FOUND : node_result;
    }

    unsigned char *client_node = lupine_allocate_private_6e16_node();
    if (client_node == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *reinterpret_cast<uint64_t *>(client_node + 0x0) = 0x100000001ULL;
    *reinterpret_cast<uint64_t *>(client_node + 0x8) = server_owner;
    *reinterpret_cast<uint64_t *>(client_node + 0x10) = 0xc;
    *reinterpret_cast<uint64_t *>(client_node + 0x18) = 1;
    *reinterpret_cast<uint64_t *>(client_node + 0x28) = 0x80;
    *reinterpret_cast<uint32_t *>(client_node + 0x370) = 2;
    *reinterpret_cast<uint64_t *>(client_node + 0x480) = 0;
    {
      std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
      auto &nodes = lupine_private_node_map();
      nodes[reinterpret_cast<CUfunction>(client_node)] = server_node;
      lupine_private_node_count.store(nodes.size(), std::memory_order_release);
    }
    lupine_note_function_owner_route(reinterpret_cast<CUfunction>(client_node),
                                     lupine_route_for_function(server_node));
    LUPINE_TRACE_LOG("LUPINE private 6e16[7] mapped client_node="
                     << static_cast<void *>(client_node)
                     << " server_node=" << server_node << " server_owner="
                     << reinterpret_cast<void *>(server_owner));
    if (arg2 != 0) {
      using lupine_private_iterator_callback =
          void (*)(void *, void *, uint64_t);
      auto callback = reinterpret_cast<lupine_private_iterator_callback>(arg2);
      callback(reinterpret_cast<void *>(arg3), client_node,
               *reinterpret_cast<uint64_t *>(client_node + 8));
    }
    return static_cast<CUresult>(1);
  }

  return CUDA_ERROR_NOT_SUPPORTED;
}

static bool lupine_stub_private_exports_enabled() {
  static bool enabled = lupine_env_flag("LUPINE_STUB_PRIVATE_EXPORTS", false);
  return enabled;
}

static bool lupine_remote_private_exports_enabled() {
  static bool enabled = lupine_env_flag("LUPINE_REMOTE_PRIVATE_EXPORTS", true);
  return enabled;
}

#if defined(__aarch64__)
// Materializes a 64-bit constant into Xd with a fixed four-word
// movz/movk/movk/movk sequence. The length is deliberately constant (never
// elided for zero halfwords) so emitted stubs have a fixed size and are
// trivially auditable against a disassembler.
static void lupine_a64_emit_mov_imm64(uint32_t *out, unsigned reg,
                                      uint64_t value) {
  // movz Xd, #imm16
  out[0] = 0xd2800000u | (static_cast<uint32_t>(value & 0xffffu) << 5) | reg;
  // movk Xd, #imm16, lsl #16
  out[1] =
      0xf2a00000u | (static_cast<uint32_t>((value >> 16) & 0xffffu) << 5) | reg;
  // movk Xd, #imm16, lsl #32
  out[2] =
      0xf2c00000u | (static_cast<uint32_t>((value >> 32) & 0xffffu) << 5) | reg;
  // movk Xd, #imm16, lsl #48
  out[3] =
      0xf2e00000u | (static_cast<uint32_t>((value >> 48) & 0xffffu) << 5) | reg;
}
#endif

#if defined(__x86_64__) || defined(__aarch64__)
static void *lupine_publish_exec_stub(const void *bytes, size_t size,
                                      bool flush_icache) {
  unsigned char *code =
      static_cast<unsigned char *>(mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }
  memcpy(code, bytes, size);
  if (flush_icache) {
    __builtin___clear_cache(reinterpret_cast<char *>(code),
                            reinterpret_cast<char *>(code + size));
  }
  // Publish immutable executable code only after construction. macOS rejects
  // anonymous mappings that request write and execute permissions together.
  if (mprotect(code, size, PROT_READ | PROT_EXEC) != 0) {
    LUPINE_LOG_ERROR(
        "Failed to protect private export trampoline: " << strerror(errno));
    munmap(code, size);
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }
  return code;
}
#endif

static void *lupine_make_private_export_stub(int slot, const char *table_name) {
#if defined(_WIN32)
  (void)slot;
  (void)table_name;
  // NVIDIA's private export tables are undocumented and use a different
  // calling convention on Windows. Public CUDA Driver API entry points remain
  // fully available; return the standard unsupported stub for private slots.
  return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
#elif defined(__x86_64__)
  unsigned char code[52];
  void *handler = reinterpret_cast<void *>(&lupine_private_export_slot_called);
  size_t offset = 0;
  const unsigned char prologue[] = {
      0x48, 0x83, 0xec, 0x08, // sub rsp, 8
      0x41, 0x51,             // push r9
      0x41, 0x50,             // push r8
      0x49, 0x89, 0xc9,       // mov r9, rcx
      0x49, 0x89, 0xd0,       // mov r8, rdx
      0x48, 0x89, 0xf1,       // mov rcx, rsi
      0x48, 0x89, 0xfa,       // mov rdx, rdi
      0xbf                    // mov edi, imm32
  };
  memcpy(code + offset, prologue, sizeof(prologue));
  offset += sizeof(prologue);
  memcpy(code + offset, &slot, sizeof(slot));
  offset += sizeof(slot);
  code[offset++] = 0x48;
  code[offset++] = 0xbe;
  memcpy(code + offset, &table_name, sizeof(table_name));
  offset += sizeof(table_name);
  code[offset++] = 0x48;
  code[offset++] = 0xb8;
  memcpy(code + offset, &handler, sizeof(handler));
  offset += sizeof(handler);
  const unsigned char epilogue[] = {
      0xff, 0xd0,             // call rax
      0x48, 0x83, 0xc4, 0x18, // add rsp, 24
      0xc3                    // ret
  };
  memcpy(code + offset, epilogue, sizeof(epilogue));
  return lupine_publish_exec_stub(code, sizeof(code), false);
#elif defined(__aarch64__)
  // AAPCS64 mirror of the x86-64 thunk above: the six incoming integer/pointer
  // arguments are shifted up two slots and the (slot, table_name) pair is
  // injected in front, then we tail-branch to the handler. The handler takes
  // eight integer arguments, all of which fit in x0-x7, so no stack shuffling
  // (and therefore no frame at all) is required and x30 stays untouched.
  //
  //   stub(a0..a5) -> lupine_private_export_slot_called(slot, table_name,
  //                                                     a0..a5)
  void *handler = reinterpret_cast<void *>(&lupine_private_export_slot_called);
  uint32_t words[17];
  size_t n = 0;
  // Shift x0-x5 up into x2-x7. This MUST run highest-destination first so a
  // source register is never clobbered before it is read.
  // `mov Xd, Xm` is `orr Xd, xzr, Xm` = 0xaa0003e0 | (Rm << 16) | Rd.
  for (unsigned r = 6; r-- > 0;) {
    words[n++] = 0xaa0003e0u | (r << 16) | (r + 2);
  }
  // w0 = slot. `int` is 32-bit, and writing w0 zeroes the upper half of x0,
  // which matches the x86 `mov edi, imm32`.
  uint32_t slot_bits = static_cast<uint32_t>(slot);
  words[n++] = 0x52800000u | ((slot_bits & 0xffffu) << 5);
  words[n++] = 0x72a00000u | (((slot_bits >> 16) & 0xffffu) << 5);
  lupine_a64_emit_mov_imm64(&words[n], 1,
                            reinterpret_cast<uint64_t>(table_name));
  n += 4;
  // x16 = handler (x16/IP0 is the intra-procedure-call scratch register, so it
  // is free to clobber across this tail branch).
  lupine_a64_emit_mov_imm64(&words[n], 16, reinterpret_cast<uint64_t>(handler));
  n += 4;
  words[n++] = 0xd61f0200u; // br x16
  // Instruction-cache maintenance is mandatory on AArch64: the writes above
  // land in the data cache and are not otherwise visible to the instruction
  // fetcher.
  return lupine_publish_exec_stub(words, sizeof(words), true);
#else
  (void)slot;
  (void)table_name;
  return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
#endif
}

static const void *lupine_make_private_export_table(
    const char *table_name, size_t byte_size,
    const std::vector<uint64_t> &code_hashes = {}) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<void *>> tables;
  std::lock_guard<std::mutex> lock(mutex);
  auto existing = tables.find(table_name);
  if (existing != tables.end()) {
    return existing->second.data();
  }

  size_t entries = byte_size / sizeof(void *);
  std::vector<void *> table(entries, nullptr);
  if (!table.empty()) {
    table[0] = reinterpret_cast<void *>(byte_size);
  }
  char *stable_table_name = strdup(table_name);
  for (size_t i = 1; i < entries; ++i) {
    table[i] =
        lupine_make_private_export_stub(static_cast<int>(i), stable_table_name);
  }
  if (!code_hashes.empty()) {
    lupine_private_export_hashes()[table_name] = code_hashes;
  }
  auto inserted = tables.emplace(table_name, std::move(table));
  return inserted.first->second.data();
}

static std::string lupine_uuid_hex(const CUuuid *uuid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  if (uuid == nullptr) {
    return out;
  }
  out.reserve(32);
  for (unsigned char byte : uuid->bytes) {
    out.push_back(kHex[byte >> 4]);
    out.push_back(kHex[byte & 0xf]);
  }
  return out;
}

static const void *lupine_remote_private_export_table(const CUuuid *uuid) {
  if (uuid == nullptr || !lupine_remote_private_exports_enabled()) {
    return nullptr;
  }
  // The server answers from its own driver, so the reply, refusals included,
  // is fixed per table id for the life of the process.
  auto &answered =
      lupine_leaked<lupine_tiny_cache<std::string, const void *>>([] {});
  std::string uuid_hex = lupine_uuid_hex(uuid);
  const void *table = nullptr;
  if (answered.find(uuid_hex, table)) {
    return table;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult result = CUDA_ERROR_UNKNOWN;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[LUPINE_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, LUPINE_RPC_cuGetExportTableMetadata,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, uuid->bytes, sizeof(uuid->bytes)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      !lupine_read_all(conn, result, byte_size, slot_count, trusted)) {
    return nullptr;
  }

  if (slot_count > LUPINE_PRIVATE_EXPORT_MAX_SLOTS) {
    rpc_read_end(conn);
    return nullptr;
  }
  size_t hash_bytes = static_cast<size_t>(slot_count) * sizeof(uint64_t);
  if (hash_bytes != 0 && rpc_read(conn, hashes, hash_bytes) < 0) {
    rpc_read_end(conn);
    return nullptr;
  }
  if (rpc_read_end(conn) < 0) {
    return nullptr;
  }

  if (result != CUDA_SUCCESS || trusted == 0 || byte_size == 0 ||
      byte_size % sizeof(void *) != 0 || slot_count == 0 ||
      slot_count != byte_size / sizeof(void *) ||
      slot_count > LUPINE_PRIVATE_EXPORT_MAX_SLOTS) {
    answered.insert_or_assign(uuid_hex, nullptr);
    return nullptr;
  }

  std::vector<uint64_t> code_hashes(hashes, hashes + slot_count);
  LUPINE_TRACE_LOG("LUPINE remote cuGetExportTable metadata uuid="
                   << uuid_hex << " bytes=" << byte_size
                   << " slots=" << slot_count);
  table = lupine_make_private_export_table(
      uuid_hex.c_str(), static_cast<size_t>(byte_size), code_hashes);
  answered.insert_or_assign(uuid_hex, table);
  return table;
}

static const void *lupine_private_export_table_from_env(const CUuuid *uuid) {
  if (uuid == nullptr) {
    return nullptr;
  }
  static std::unordered_map<std::string, size_t> configured_tables = [] {
    std::unordered_map<std::string, size_t> tables;
    const char *raw = getenv("LUPINE_PRIVATE_EXPORT_TABLES");
    if (raw == nullptr || raw[0] == '\0') {
      return tables;
    }

    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
      size_t colon = item.find(':');
      if (colon == std::string::npos) {
        continue;
      }
      std::string uuid_hex = item.substr(0, colon);
      if (uuid_hex.size() != 32) {
        continue;
      }
      char *end = nullptr;
      unsigned long long byte_size =
          strtoull(item.c_str() + colon + 1, &end, 0);
      if (end == item.c_str() + colon + 1 || byte_size == 0 ||
          byte_size % sizeof(void *) != 0) {
        continue;
      }
      tables[uuid_hex] = static_cast<size_t>(byte_size);
    }
    return tables;
  }();

  std::string uuid_hex = lupine_uuid_hex(uuid);
  auto it = configured_tables.find(uuid_hex);
  if (it == configured_tables.end()) {
    return nullptr;
  }
  return lupine_make_private_export_table(uuid_hex.c_str(), it->second);
}

// The driver owns the authoritative spelling of every result code, so instead
// of mirroring its tables the shim asks whoever holds a real driver: the local
// driver on a local route, otherwise the server. That is byte-exact even for
// codes newer than this client's headers.
//
// CUDA promises the returned pointer has static lifetime, so answers are
// interned in a never-erased map and each code costs at most one round trip per
// process. unordered_map nodes are stable, so c_str() stays valid.
static CUresult lupine_error_string_lookup(int rpc_op, const char *symbol,
                                           bool want_name, CUresult error,
                                           const char **pStr) {
  if (pStr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = nullptr;

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn(symbol, error, pStr);
  }

  // The only hardcoded string: this is the shim's own most common error, and
  // its occurrence means the forward path below may already be gone.
  if (error == CUDA_ERROR_DEVICE_UNAVAILABLE) {
    *pStr = want_name ? "CUDA_ERROR_DEVICE_UNAVAILABLE"
                      : "CUDA-capable device(s) is/are busy or unavailable";
    return CUDA_SUCCESS;
  }

  static auto *mutex = new std::mutex();
  static auto *interned = new std::unordered_map<int, std::string>[2];
  std::unordered_map<int, std::string> &strings = interned[want_name];
  {
    std::lock_guard<std::mutex> lock(*mutex);
    auto found = strings.find(error);
    if (found != strings.end()) {
      *pStr = found->second.c_str();
      return CUDA_SUCCESS;
    }
  }

  // A length past any real driver string means a desynced stream.
  constexpr std::size_t kMaxLength = 4096;
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  std::size_t length = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, rpc_op, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &error, sizeof(error)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  std::lock_guard<std::mutex> lock(*mutex);
  *pStr = strings.emplace(error, std::move(text)).first->second.c_str();
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr) {
  return lupine_error_string_lookup(RPC_cuGetErrorName, "cuGetErrorName", true,
                                    error, pStr);
}

extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr) {
  return lupine_error_string_lookup(RPC_cuGetErrorString, "cuGetErrorString",
                                    false, error, pStr);
}

extern "C" CUresult cuProfilerInitialize(const char *, const char *,
                                         CUoutput_mode) {
  return lupine_cuda_is_initialized() ? CUDA_SUCCESS
                                      : CUDA_ERROR_NOT_INITIALIZED;
}

extern "C" CUresult cuProfilerStart(void) {
  return lupine_cuda_is_initialized() ? CUDA_SUCCESS
                                      : CUDA_ERROR_NOT_INITIALIZED;
}

extern "C" CUresult cuProfilerStop(void) {
  return lupine_cuda_is_initialized() ? CUDA_SUCCESS
                                      : CUDA_ERROR_NOT_INITIALIZED;
}

// torch's c10 stream pool creates 32 streams at each of up to 4 priorities on
// its first side-stream use, one round trip each. When a primary context is
// first retained on a remote route the client instead has the server create
// that whole pool under it with one lupineStreamPoolInit round trip, so
// stream requests with a matching (flags, priority) are answered locally;
// any other request is created singly. A stream belongs to the context
// current when it was created, so pools are keyed by (route, context) and
// dropped when that context goes away; pooled streams never handed out are
// freed with the server context.
static constexpr uint32_t kLupineStreamPoolMax = 4 * 32;

struct lupine_stream_pool {
  CUdevice dev;
  std::map<std::pair<unsigned int, int>, std::vector<CUstream>> streams;
};

static std::mutex &lupine_stream_pool_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static std::map<std::pair<int, CUcontext>, lupine_stream_pool> &
lupine_stream_pools() {
  return lupine_leaked<std::map<std::pair<int, CUcontext>, lupine_stream_pool>>(
      [] {});
}

// Bumped by each discard, under the pool mutex. A batch created across a
// discard may have died with its context, whose handle value may already be
// reused, so it is dropped. Accepted: a batch for an unrelated context is
// dropped too, and its unclaimed handles live until that context goes.
static uint64_t lupine_stream_pool_generation;

// The handles arrive in creation order (stream-major, priority-minor); they
// are pushed in reverse so hand-out pops them in that order.
static void lupine_stream_pool_init(lupine_route route, CUdevice dev,
                                    CUcontext ctx) {
  auto key = std::make_pair(lupine_route_identity(route), ctx);
  uint64_t generation;
  {
    std::lock_guard<std::mutex> lock(lupine_stream_pool_mutex());
    auto &pools = lupine_stream_pools();
    if (pools.find(key) != pools.end()) {
      return;
    }
    generation = lupine_stream_pool_generation;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint32_t created = 0;
  uint32_t priorities = 0;
  int least = 0;
  CUstream streams[kLupineStreamPoolMax];
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, LUPINE_RPC_lupineStreamPoolInit,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &created, sizeof(created)) < 0 ||
      rpc_read(conn, &priorities, sizeof(priorities)) < 0 ||
      rpc_read(conn, &least, sizeof(least)) < 0 ||
      created > kLupineStreamPoolMax || (created != 0 && priorities == 0) ||
      rpc_read(conn, streams, created * sizeof(*streams)) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_stream_pool_mutex());
  if (generation != lupine_stream_pool_generation) {
    return;
  }
  auto &pools = lupine_stream_pools();
  auto &pool = pools[key];
  pool.dev = dev;
  for (uint32_t i = created; i-- > 0;) {
    int priority = least - static_cast<int>(i % priorities);
    pool.streams[{CU_STREAM_NON_BLOCKING, priority}].push_back(streams[i]);
  }
}

static void lupine_stream_pool_discard(int route_id, CUdevice dev,
                                       CUcontext ctx) {
  std::lock_guard<std::mutex> lock(lupine_stream_pool_mutex());
  ++lupine_stream_pool_generation;
  auto &pools = lupine_stream_pools();
  for (auto it = pools.begin(); it != pools.end();) {
    bool matches = ctx != nullptr
                       ? it->first.second == ctx
                       : it->first.first == route_id && it->second.dev == dev;
    it = matches ? pools.erase(it) : std::next(it);
  }
}

static CUresult lupine_stream_create_remote(lupine_route route,
                                            CUstream *phStream,
                                            unsigned int flags, int priority) {
  if (phStream == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcontext ctx = lupine_current_context_hint();
  if (ctx != nullptr) {
    CUstream pooled = nullptr;
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(lupine_stream_pool_mutex());
      auto &pools = lupine_stream_pools();
      auto pool = pools.find({lupine_route_identity(route), ctx});
      if (pool != pools.end()) {
        auto bucket = pool->second.streams.find({flags, priority});
        if (bucket != pool->second.streams.end() && !bucket->second.empty()) {
          pooled = bucket->second.back();
          bucket->second.pop_back();
          found = true;
        }
      }
    }
    if (found) {
      *phStream = pooled;
      lupine_note_stream_owner_route(pooled, route);
      lupine_note_stream_blocking(pooled,
                                  (flags & CU_STREAM_NON_BLOCKING) == 0);
      return CUDA_SUCCESS;
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (!lupine_rpc_call_unscoped(conn, RPC_cuStreamCreateWithPriority,
                                std::tie(*phStream, flags, priority),
                                std::tie(*phStream, return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && phStream != nullptr) {
    lupine_note_stream_owner_route(*phStream, route);
    lupine_note_stream_blocking(*phStream,
                                (flags & CU_STREAM_NON_BLOCKING) == 0);
  }
  return return_value;
}

static CUresult lupine_stream_create_impl(CUstream *phStream,
                                          unsigned int flags, int priority,
                                          bool with_priority) {
  lupine_route route = lupine_route_for_current_context();
  if (!lupine_route_is_local(route)) {
    return lupine_stream_create_remote(route, phStream, flags, priority);
  }
  CUresult return_value =
      with_priority
          ? lupine_call_real_cuda_fn("cuStreamCreateWithPriority", phStream,
                                     flags, priority)
          : lupine_call_real_cuda_fn("cuStreamCreate", phStream, flags);
  if (return_value == CUDA_SUCCESS && phStream != nullptr) {
    lupine_note_stream_owner_route(*phStream, route);
  }
  return return_value;
}

extern "C" CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  return lupine_stream_create_impl(phStream, Flags, 0, false);
}

extern "C" CUresult cuStreamCreateWithPriority(CUstream *phStream,
                                               unsigned int flags,
                                               int priority) {
  return lupine_stream_create_impl(phStream, flags, priority, true);
}

CUresult cuStreamDestroy_v2(CUstream hStream);
// Allocate ordinary events on first use, in bounded batches per route, context
// and flags. IPC events retain the driver's one-at-a-time path. Handed-out
// events are never recycled: destroy still reaches CUDA, so an old handle can
// never refer to a cached replacement event. Unclaimed events die with context.
static constexpr uint32_t kLupineEventBatchSize = 32;

struct lupine_event_pool {
  CUdevice remote_device = -1;
  std::vector<CUevent> events;
};

static std::mutex &lupine_event_pool_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static std::map<std::tuple<int, CUcontext, unsigned>, lupine_event_pool> &
lupine_event_pools() {
  return lupine_leaked<
      std::map<std::tuple<int, CUcontext, unsigned>, lupine_event_pool>>([] {});
}

// Bumped by each discard, under the pool mutex, like the stream pool's.
static uint64_t lupine_event_pool_generation;

static void lupine_event_pool_discard(int route_id, CUdevice remote_device,
                                      CUcontext ctx) {
  std::lock_guard<std::mutex> lock(lupine_event_pool_mutex());
  ++lupine_event_pool_generation;
  auto &pools = lupine_event_pools();
  for (auto it = pools.begin(); it != pools.end();) {
    const bool matches =
        std::get<0>(it->first) == route_id &&
        (ctx != nullptr ? std::get<1>(it->first) == ctx
                        : it->second.remote_device == remote_device);
    it = matches ? pools.erase(it) : std::next(it);
  }
}

extern "C" CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    CUresult result = lupine_call_real_cuda_fn("cuEventCreate", phEvent, Flags);
    if (result == CUDA_SUCCESS && phEvent != nullptr) {
      lupine_note_event_owner_route(*phEvent, route);
    }
    return result;
  }
  if (phEvent == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUcontext context = lupine_current_context_hint();
  constexpr unsigned pool_flags =
      CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING;
  if (context != nullptr && (Flags & ~pool_flags) == 0) {
    const auto key =
        std::make_tuple(lupine_route_identity(route), context, Flags);
    uint64_t generation;
    {
      std::lock_guard<std::mutex> lock(lupine_event_pool_mutex());
      generation = lupine_event_pool_generation;
      auto &pools = lupine_event_pools();
      auto it = pools.find(key);
      if (it != pools.end() && !it->second.events.empty()) {
        *phEvent = it->second.events.back();
        it->second.events.pop_back();
        lupine_note_event_owner_route(*phEvent, route);
        return CUDA_SUCCESS;
      }
    }
    CUevent events[kLupineEventBatchSize];
    CUdevice device = -1;
    uint32_t count = 0;
    CUresult result = CUDA_ERROR_UNKNOWN;
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request_on(conn, LUPINE_RPC_lupineEventCreateBatch,
                                   RPC_SCOPE_NONE) < 0 ||
        rpc_write(conn, &context, sizeof(context)) < 0 ||
        rpc_write(conn, &Flags, sizeof(Flags)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 ||
        rpc_read(conn, &device, sizeof(device)) < 0 ||
        rpc_read(conn, &count, sizeof(count)) < 0 ||
        count > kLupineEventBatchSize ||
        rpc_read(conn, events, count * sizeof(*events)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (count == 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    *phEvent = events[0];
    {
      std::lock_guard<std::mutex> lock(lupine_event_pool_mutex());
      if (generation == lupine_event_pool_generation) {
        auto &pool = lupine_event_pools()[key];
        pool.remote_device = device;
        // Preserve creation order when popping the next handle.
        pool.events.insert(pool.events.end(),
                           std::reverse_iterator<CUevent *>(events + count),
                           std::reverse_iterator<CUevent *>(events + 1));
      }
    }
    lupine_note_event_owner_route(*phEvent, route);
    return CUDA_SUCCESS;
  }

  // Preserve the ordinary RPC for IPC, unknown flags and an unknown current
  // context, including its unchanged output value on a driver error.
  CUresult result;
  if (!lupine_rpc_call_unscoped(conn, RPC_cuEventCreate,
                                std::tie(*phEvent, Flags),
                                std::tie(*phEvent, result))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result == CUDA_SUCCESS) {
    lupine_note_event_owner_route(*phEvent, route);
  }
  return result;
}

CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
// Defined with the capture wrappers below; nonzero only while this client holds
// a stream capture open, where a pageable copy must stay a graph node.
extern std::atomic<int> lupine_active_stream_captures;

static thread_local CUcontext lupine_current_context = nullptr;
static thread_local CUcontext lupine_default_context_hint = nullptr;
static std::atomic<CUcontext> lupine_global_default_context_hint{nullptr};
static thread_local auto *lupine_context_stack = new std::vector<CUcontext>();

extern "C" CUcontext lupine_current_context_hint() {
  return lupine_current_context;
}

CUcontext lupine_default_context_hint_value() {
  return lupine_default_context_hint;
}

CUcontext lupine_global_default_context_hint_value() {
  return lupine_global_default_context_hint.load(std::memory_order_relaxed);
}

extern "C" void lupine_forget_destroyed_context(CUcontext ctx) {
  if (ctx == nullptr) {
    return;
  }
  // Destroying a context unloads the modules it holds, so every module and
  // function handle cached against it is free for the server to hand out
  // again. This is the other end of the module-unload invalidation. The
  // context handle itself can come back on any lane, so the device answers
  // cached against it go too.
  lupine_invalidate_function_caches();
  lupine_invalidate_current_context_cache();
  lupine_event_pool_discard(
      lupine_route_identity(lupine_route_for_context(ctx)), -1, ctx);
  lupine_forget_context_owner(ctx);
  lupine_forget_context_local_storage(ctx);
  lupine_stream_pool_discard(-1, -1, ctx);
  lupine_context_stack->erase(std::remove(lupine_context_stack->begin(),
                                          lupine_context_stack->end(), ctx),
                              lupine_context_stack->end());
  if (lupine_current_context == ctx) {
    if (!lupine_context_stack->empty()) {
      lupine_current_context = lupine_context_stack->back();
      lupine_context_stack->pop_back();
    } else {
      lupine_current_context = nullptr;
    }
  }
  if (lupine_default_context_hint == ctx) {
    lupine_default_context_hint = lupine_current_context;
  }
  CUcontext global_hint =
      lupine_global_default_context_hint.load(std::memory_order_relaxed);
  if (global_hint == ctx) {
    lupine_global_default_context_hint.store(lupine_current_context,
                                             std::memory_order_relaxed);
  }
}

// CUDA IPC shareable handles of type POSIX fd cannot cross the wire as raw
// fd numbers. The server child exports the real fd, parks it with the parent
// broker under a random token, and the client hands the application a local
// proxy fd wrapping that token (see ipc.h). Import reverses the exchange.

template <typename Handle>
static CUresult
lupine_export_shareable_handle(lupine_route route, int rpc_id, uint32_t kind,
                               void *shareableHandle, Handle handle,
                               CUmemAllocationHandleType handleType,
                               unsigned long long flags) {
  CUresult return_value;
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_ipc_token token = {};
  if (!lupine_rpc_call_unscoped(conn, rpc_id,
                                std::tie(handle, handleType, flags),
                                std::tie(token, return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && shareableHandle != nullptr) {
    int proxy_fd = -1;
    if (lupine_ipc_create_proxy_fd(kind, &token, &proxy_fd) < 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    *reinterpret_cast<int *>(shareableHandle) = proxy_fd;
  }
  return return_value;
}

CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                      CUmemGenericAllocationHandle handle,
                                      CUmemAllocationHandleType handleType,
                                      unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemExportToShareableHandle",
                                    shareableHandle, handle, handleType, flags);
  }
  return lupine_export_shareable_handle(route, RPC_cuMemExportToShareableHandle,
                                        LUPINE_IPC_FD_KIND_VMM_ALLOCATION,
                                        shareableHandle, handle, handleType,
                                        flags);
}

template <typename Handle>
static CUresult lupine_import_shareable_handle(
    lupine_route route, uint32_t rpc_op, uint32_t expected_kind, Handle *out,
    void *os_handle, CUmemAllocationHandleType handle_type,
    const unsigned long long *flags) {
  uint32_t kind = 0;
  lupine_ipc_token token = {};
  int proxy_fd = static_cast<int>(reinterpret_cast<uintptr_t>(os_handle));
  if (lupine_ipc_read_proxy_fd(proxy_fd, &kind, &token) < 0 ||
      kind != expected_kind) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, rpc_op, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &handle_type, sizeof(handle_type)) < 0 ||
      (flags != nullptr && rpc_write(conn, flags, sizeof(*flags)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, out, sizeof(*out)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

CUresult
cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                               void *osHandle,
                               CUmemAllocationHandleType shHandleType) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemImportFromShareableHandle", handle,
                                    osHandle, shHandleType);
  }
  return lupine_import_shareable_handle(
      route, RPC_cuMemImportFromShareableHandle,
      LUPINE_IPC_FD_KIND_VMM_ALLOCATION, handle, osHandle, shHandleType,
      nullptr);
}

CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool,
                                          CUmemAllocationHandleType handleType,
                                          unsigned long long flags) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemPoolExportToShareableHandle",
                                    handle_out, pool, handleType, flags);
  }
  return lupine_export_shareable_handle(
      route, RPC_cuMemPoolExportToShareableHandle,
      LUPINE_IPC_FD_KIND_MEMORY_POOL, handle_out, pool, handleType, flags);
}

CUresult
cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle,
                                   CUmemAllocationHandleType handleType,
                                   unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value =
      lupine_route_is_local(route)
          ? lupine_call_real_cuda_fn("cuMemPoolImportFromShareableHandle",
                                     pool_out, handle, handleType, flags)
          : lupine_import_shareable_handle(
                route, RPC_cuMemPoolImportFromShareableHandle,
                LUPINE_IPC_FD_KIND_MEMORY_POOL, pool_out, handle, handleType,
                &flags);
  if (return_value == CUDA_SUCCESS && pool_out != nullptr) {
    lupine_note_memory_pool_owner_route(*pool_out, route);
  }
  return return_value;
}

void lupine_accept_current_context_hint(CUcontext ctx) {
  if (lupine_current_context != ctx) {
    lupine_current_context = ctx;
  }
}

static CUresult lupine_set_remote_current_context(CUcontext ctx) {
  lupine_route route =
      lupine_route_for_context(ctx != nullptr ? ctx : lupine_current_context);
  return lupine_set_current_context_on_route(route, ctx);
}

static void lupine_publish_current_context(CUcontext ctx) {
  lupine_current_context = ctx;
  if (ctx != nullptr) {
    lupine_default_context_hint = ctx;
    lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
  }
}

extern "C" void lupine_note_ctx_create_route(CUcontext ctx,
                                             lupine_route route) {
  lupine_note_device_binding_changed();
  lupine_note_context_owner_route(ctx, route);
  lupine_lane_context_cache_store(lupine_route_identity(route), ctx);
  if (lupine_current_context != nullptr) {
    lupine_context_stack->push_back(lupine_current_context);
  }
  lupine_publish_current_context(ctx);
}

extern "C" void lupine_note_ctx_create(CUcontext ctx, conn_t *conn) {
  lupine_note_ctx_create_route(ctx, lupine_remote_route_for_conn(conn));
}

extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    if (lupine_current_context != nullptr) {
      lupine_context_stack->push_back(lupine_current_context);
    }
    lupine_publish_current_context(ctx);
  }
  return result;
}

extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (lupine_current_context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  CUcontext popped = lupine_current_context;
  CUcontext previous =
      lupine_context_stack->empty() ? nullptr : lupine_context_stack->back();
  CUresult result = lupine_set_remote_current_context(previous);
  if (result == CUDA_SUCCESS) {
    if (!lupine_context_stack->empty()) {
      lupine_context_stack->pop_back();
    }
    lupine_publish_current_context(previous);
    if (pctx != nullptr) {
      *pctx = popped;
    }
  }
  return result;
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (ctx == nullptr) {
    return lupine_current_context == nullptr ? CUDA_SUCCESS
                                             : cuCtxPopCurrent_v2(nullptr);
  }
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    lupine_publish_current_context(ctx);
  }
  return result;
}

extern "C" CUresult cuCtxGetCurrent(CUcontext *pctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pctx = lupine_current_context;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuCtxGetDevice(CUdevice *device) {
  // Unlike cuCtxGetCurrent, the driver validates the argument before the init
  // state here.
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuCtxGetDevice", device);
  }
  if (lupine_current_context_device_cache_lookup(lupine_current_context,
                                                 device)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUdevice remote_device = 0;
  CUresult return_value;
  if (!lupine_rpc_call_unscoped(conn, RPC_cuCtxGetDevice, std::tie(),
                                std::tie(remote_device, return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    *device = lupine_local_device_for_remote(conn, remote_device);
    lupine_current_context_device_cache_insert(lupine_current_context, *device);
  }
  return return_value;
}

extern "C" CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                            CUmem_range_attribute *attributes,
                                            size_t numAttributes,
                                            CUdeviceptr devPtr, size_t count) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemRangeGetAttributes", data, dataSizes,
                                    attributes, numAttributes, devPtr, count);
  }

  if (!data || !dataSizes || !attributes || !numAttributes || !count)
    return CUDA_ERROR_INVALID_VALUE;
  for (size_t i = 0; i < numAttributes; ++i) {
    if (!data[i] ||
        !lupine_mem_range_attribute_size_valid(attributes[i], dataSizes[i]))
      return CUDA_ERROR_INVALID_VALUE;
  }

  for (size_t i = 0; i < numAttributes; ++i) {
    CUresult result = cuMemRangeGetAttribute(data[i], dataSizes[i],
                                             attributes[i], devPtr, count);
    if (result != CUDA_SUCCESS)
      return result;
  }
  return CUDA_SUCCESS;
}

// The CUDA linker requires the caller's option arrays to stay valid for the
// lifetime of the CUlinkState, so cuLinkComplete writes its outputs straight
// back into the arrays cuLinkCreate was given.
struct lupine_jit_client_state {
  unsigned int num_options = 0;
  CUjit_option *options = nullptr;
  void **option_values = nullptr;
  std::vector<unsigned char> cubin;
};

static std::mutex &lupine_jit_client_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::unordered_map<CUlinkState, lupine_jit_client_state> &
lupine_jit_client_states() {
  static std::unordered_map<CUlinkState, lupine_jit_client_state> states;
  return states;
}

// JIT option write-back. Log buffers arrive as the bytes the driver wrote and
// need their terminator restored here, because the caller's buffer is
// uninitialized on entry and is read back as a C string. Every other option
// value is rewritten in place by the driver (CU_JIT_WALL_TIME and the log
// sizes report through the option array itself), so its word replaces the one
// that was sent.
static int lupine_read_jit_outputs(conn_t *conn, unsigned int numOptions,
                                   const CUjit_option *options,
                                   void **optionValues) {
  size_t log_sizes[2] = {0, 0};
  for (unsigned int i = 0; i < numOptions; ++i) {
    if (options[i] == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES) {
      log_sizes[0] = reinterpret_cast<uintptr_t>(optionValues[i]);
    } else if (options[i] == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES) {
      log_sizes[1] = reinterpret_cast<uintptr_t>(optionValues[i]);
    }
  }
  for (unsigned int i = 0; i < numOptions; ++i) {
    const bool is_info_log = options[i] == CU_JIT_INFO_LOG_BUFFER;
    const bool is_log = is_info_log || options[i] == CU_JIT_ERROR_LOG_BUFFER;
    size_t length = 0;
    if (rpc_read(conn, &length, sizeof(length)) < 0 ||
        (length != 0 &&
         rpc_read(conn, is_log ? optionValues[i] : &optionValues[i], length) <
             0)) {
      return -1;
    }
    if (is_log) {
      size_t log_size = log_sizes[is_info_log ? 0 : 1];
      if (log_size != 0 && optionValues[i] != nullptr) {
        auto *log = static_cast<char *>(optionValues[i]);
        log[length < log_size ? length : log_size - 1] = '\0';
      }
    }
  }
  return 0;
}

// numOptions is taken by non-const lvalue reference because rpc_write queues
// &numOptions without copying until rpc_write_end flushes the request.
static int lupine_write_jit_options(conn_t *conn, unsigned int &numOptions,
                                    const CUjit_option *options,
                                    void *const *optionValues) {
  return rpc_write(conn, &numOptions, sizeof(numOptions)) < 0 ||
                 rpc_write(conn, options, numOptions * sizeof(*options)) < 0 ||
                 rpc_write(conn, optionValues,
                           numOptions * sizeof(*optionValues)) < 0
             ? -1
             : 0;
}

extern "C" CUresult cuLinkCreate_v2(unsigned int numOptions,
                                    CUjit_option *options, void **optionValues,
                                    CUlinkState *stateOut) {
  if (stateOut == nullptr ||
      (numOptions != 0 && (options == nullptr || optionValues == nullptr))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuLinkCreate_v2", numOptions, options,
                                    optionValues, stateOut);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuLinkCreate_v2) < 0 ||
      lupine_write_jit_options(conn, numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, stateOut, sizeof(*stateOut)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
    auto &jit_state = lupine_jit_client_states()[*stateOut];
    jit_state.num_options = numOptions;
    jit_state.options = options;
    jit_state.option_values = optionValues;
  }
  return return_value;
}

static CUresult
lupine_remote_link_add_data(conn_t *conn, CUlinkState state,
                            CUjitInputType type, const void *data, size_t size,
                            const char *name, unsigned int numOptions,
                            CUjit_option *options, void **optionValues) {
  CUresult return_value;
  size_t name_len = name == nullptr ? 0 : strlen(name) + 1;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuLinkAddData_v2) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      rpc_write(conn, data, size) < 0 ||
      rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
      rpc_write(conn, name, name_len) < 0 ||
      lupine_write_jit_options(conn, numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_jit_outputs(conn, numOptions, options, optionValues) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type,
                                     void *data, size_t size, const char *name,
                                     unsigned int numOptions,
                                     CUjit_option *options,
                                     void **optionValues) {
  if ((size != 0 && data == nullptr) ||
      (numOptions != 0 && (options == nullptr || optionValues == nullptr))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuLinkAddData_v2", state, type, data, size,
                                    name, numOptions, options, optionValues);
  }
  return lupine_remote_link_add_data(lupine_route_remote_conn(route), state,
                                     type, data, size, name, numOptions,
                                     options, optionValues);
}

extern "C" CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                                     const char *path, unsigned int numOptions,
                                     CUjit_option *options,
                                     void **optionValues) {
  if (path == nullptr ||
      (numOptions != 0 && (options == nullptr || optionValues == nullptr))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuLinkAddFile_v2", state, type, path,
                                    numOptions, options, optionValues);
  }
  void *file_mapping = MAP_FAILED;
  size_t mapped_file_size = 0;
  CUresult return_value =
      lupine_mmap_readonly_file(path, &file_mapping, &mapped_file_size);
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  return_value = lupine_remote_link_add_data(
      lupine_route_remote_conn(route), state, type, file_mapping,
      mapped_file_size, path, numOptions, options, optionValues);
  munmap(file_mapping, mapped_file_size);
  return return_value;
}

extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut,
                                   size_t *sizeOut) {
  if (cubinOut == nullptr || sizeOut == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuLinkComplete", state, cubinOut, sizeOut);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t cubin_size = 0;
  unsigned int num_options = 0;
  CUjit_option *options = nullptr;
  void **option_values = nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuLinkComplete) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &cubin_size, sizeof(cubin_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  LUPINE_TRACE_LOG("LUPINE cuLinkComplete state="
                   << reinterpret_cast<void *>(state)
                   << " cubin_size=" << cubin_size);

  {
    std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
    auto &jit_state = lupine_jit_client_states()[state];
    jit_state.cubin.resize(cubin_size);
    if (cubin_size != 0 &&
        rpc_read(conn, jit_state.cubin.data(), cubin_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    num_options = jit_state.num_options;
    options = jit_state.options;
    option_values = jit_state.option_values;
    *cubinOut = jit_state.cubin.empty() ? nullptr : jit_state.cubin.data();
    *sizeOut = jit_state.cubin.size();
  }

  if (lupine_read_jit_outputs(conn, num_options, options, option_values) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLinkDestroy(CUlinkState state) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuLinkDestroy", state);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (!lupine_rpc_call(conn, RPC_cuLinkDestroy, std::tie(state),
                       std::tie(return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
  lupine_jit_client_states().erase(state);
  return return_value;
}

extern "C" int lupine_forward_remote_stdout(conn_t *conn) {
  return lupine_read_captured_stdout(conn);
}

extern "C" CUresult cuEventQuery(CUevent hEvent) {
  lupine_route route = lupine_route_for_event(hEvent);
  if (lupine_route_is_local(route)) {
    CUresult return_value = lupine_call_real_cuda_fn("cuEventQuery", hEvent);
    return return_value == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                        : return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  std::shared_lock<std::shared_mutex> event_lifecycle_lock(
      lupine_event_lifecycle_mutex());
  CUevent events[kLupineEventQueryBatch + 1] = {hEvent};
  uint64_t records[kLupineEventQueryBatch + 1];
  uint64_t tickets[kLupineEventQueryBatch + 1];
  CUresult results[kLupineEventQueryBatch + 1];
  if (!lupine_event_query_needed(hEvent, &records[0])) {
    return lupine_sync_mapped_device_to_host();
  }
  // Other pending events on this server ride along to warm the cache.
  uint32_t count =
      lupine_event_collect_query_batch(hEvent, conn, events + 1, records + 1);
  // A query checks the events' records instead of waiting for them: a record
  // another thread sent that has not been submitted answers not ready. Each
  // event that rides along carries its own record's ticket, so one still
  // being submitted does not make the others not ready.
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuEventQuery,
                                 reinterpret_cast<uintptr_t>(hEvent),
                                 RPC_REQUEST_SOFT) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  for (uint32_t i = 1; i <= count; ++i) {
    tickets[i] = rpc_request_pending_ticket(
        conn, reinterpret_cast<uintptr_t>(events[i]));
  }
  if (rpc_write(conn, &hEvent, sizeof(hEvent)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, events + 1, count * sizeof(*events)) < 0 ||
      rpc_write(conn, tickets + 1, count * sizeof(*tickets)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, results, (count + 1) * sizeof(*results)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_event_cache_completions(events, records, results, count + 1);
  return results[0] == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                    : results[0];
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                   CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    CUresult result =
        lupine_call_real_cuda_fn("cuCtxCreate_v2", pctx, flags, dev);
    if (result == CUDA_SUCCESS) {
      lupine_note_ctx_create_route(*pctx, route);
    }
    return result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (!lupine_rpc_call(conn, RPC_cuCtxCreate_v2, std::tie(flags, dev),
                       std::tie(*pctx, return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_ctx_create(*pctx, conn);
  }
  return return_value;
}

#if CUDA_VERSION >= 12050
extern "C" CUresult cuCtxCreate_v4(CUcontext *pctx,
                                   CUctxCreateParams *ctxCreateParams,
                                   unsigned int flags, CUdevice dev) {
  if (ctxCreateParams != nullptr &&
      (ctxCreateParams->execAffinityParams != nullptr ||
       ctxCreateParams->numExecAffinityParams != 0 ||
       ctxCreateParams->cigParams != nullptr)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuCtxCreate_v2(pctx, flags, dev);
}
#endif

// Bounded prefix scan, so a binary image cannot walk off the caller's buffer.
static bool lupine_image_looks_like_text(const char *image) {
  constexpr size_t kProbeBytes = 64;
  for (size_t i = 0; i < kProbeBytes; ++i) {
    unsigned char byte = static_cast<unsigned char>(image[i]);
    if (byte == '\0') {
      return i != 0;
    }
    if (byte < 0x20 && byte != '\t' && byte != '\n' && byte != '\v' &&
        byte != '\f' && byte != '\r') {
      return false;
    }
    if (byte > 0x7e) {
      return false;
    }
  }
  return true;
}

static bool lupine_locate_module_image(const void *image, uint32_t *kind,
                                       const unsigned char **data,
                                       size_t *size) {
  if (image == nullptr || kind == nullptr || data == nullptr ||
      size == nullptr) {
    return false;
  }

  const auto *wrapper = reinterpret_cast<const lupine_fatbin_wrapper *>(image);
  const void *fatbin = image;
  if (wrapper->magic == LUPINE_FATBINC_MAGIC &&
      (wrapper->version == 1 || wrapper->version == 2) &&
      wrapper->data != nullptr) {
    fatbin = wrapper->data;
    *kind = wrapper->version == 2 ? LUPINE_MODULE_IMAGE_FATBINC_V2
                                  : LUPINE_MODULE_IMAGE_FATBINC_V1;
  } else {
    *kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
  }

  const auto *header = reinterpret_cast<const lupine_fatbin_header *>(fatbin);
  if (header->magic != LUPINE_FATBIN_MAGIC || header->header_size == 0) {
    const auto *elf = static_cast<const unsigned char *>(image);
    if (std::memcmp(elf, ELFMAG, SELFMAG) == 0 && elf[EI_CLASS] == ELFCLASS64) {
      const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
      size_t image_size = sizeof(Elf64_Ehdr);
      auto bump = [&](size_t end) { image_size = std::max(image_size, end); };
      if (ehdr->e_phoff != 0 && ehdr->e_phentsize == sizeof(Elf64_Phdr)) {
        bump(ehdr->e_phoff + ehdr->e_phnum * sizeof(Elf64_Phdr));
        const auto *phdrs =
            reinterpret_cast<const Elf64_Phdr *>(elf + ehdr->e_phoff);
        for (int i = 0; i < ehdr->e_phnum; ++i)
          bump(phdrs[i].p_offset + phdrs[i].p_filesz);
      }
      if (ehdr->e_shoff != 0 && ehdr->e_shentsize == sizeof(Elf64_Shdr)) {
        bump(ehdr->e_shoff + ehdr->e_shnum * sizeof(Elf64_Shdr));
        const auto *shdrs =
            reinterpret_cast<const Elf64_Shdr *>(elf + ehdr->e_shoff);
        for (int i = 0; i < ehdr->e_shnum; ++i)
          if (shdrs[i].sh_type != SHT_NOBITS)
            bump(shdrs[i].sh_offset + shdrs[i].sh_size);
      }
      *kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
      *data = elf;
      *size = image_size;
      return true;
    }

    // Non-fatbin, non-ELF images are forwarded verbatim for the server's
    // driver to judge. Non-text bytes carry no length, so they cannot ship.
    const char *ptx = static_cast<const char *>(image);
    if (!lupine_image_looks_like_text(ptx)) {
      return false;
    }
    *kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
    *data = reinterpret_cast<const unsigned char *>(ptx);
    *size = std::strlen(ptx) + 1;
    return true;
  }
  *data = static_cast<const unsigned char *>(fatbin);
  *size = static_cast<size_t>(header->header_size) +
          static_cast<size_t>(header->files_size);
  return true;
}

// Whether any member entry of a packed fat binary could serve a device of this
// compute capability, so a load that cannot be served never reaches the wire:
// cuSPARSELt 0.6 offers 2494 fat binaries at init, of which 2220 and 149 MB
// carry nothing an Ada part can run.
//
// The test is deliberately wider than the driver's own selection, because
// skipping a load the driver would have taken is a correctness bug while
// forwarding one it would have refused only costs a round trip:
//   - a cubin is binary compatible forward within a major revision and only
//     within it, so its arch must share the device's major and must not exceed
//     the device's minor. The driver is stricter: it also refuses the
//     Tegra-only capabilities (8.7, 8.8) on a desktop part, and honours the
//     arch-conditional ('a') and family ('f') flags that pin an image to one
//     capability or family.
//   - PTX is JIT compiled for the device, which works for any target at or
//     below the device's capability, across major revisions too.
// Anything the walk cannot account for exactly — a container or entry version
// this does not know, an entry kind that is neither cubin nor PTX, sizes that
// do not add up to the packed image — forwards.
static bool lupine_fatbin_may_serve_device(const unsigned char *image,
                                           size_t size, unsigned device_cc) {
  const auto *header = reinterpret_cast<const lupine_fatbin_header *>(image);
  if (size < sizeof(lupine_fatbin_header) ||
      header->magic != LUPINE_FATBIN_MAGIC ||
      header->version != LUPINE_FATBIN_VERSION ||
      header->header_size < sizeof(lupine_fatbin_header) ||
      static_cast<uint64_t>(header->header_size) + header->files_size != size ||
      header->files_size == 0) {
    return true;
  }
  for (size_t offset = header->header_size; offset < size;) {
    size_t left = size - offset;
    if (left < sizeof(lupine_fatbin_entry)) {
      return true;
    }
    const auto *entry =
        reinterpret_cast<const lupine_fatbin_entry *>(image + offset);
    if (entry->version != LUPINE_FATBIN_ENTRY_VERSION ||
        entry->header_size < sizeof(lupine_fatbin_entry) ||
        entry->header_size > left ||
        entry->payload_size > left - entry->header_size || entry->arch == 0) {
      return true;
    }
    if (entry->kind == LUPINE_FATBIN_ENTRY_CUBIN) {
      if (entry->arch / 10 == device_cc / 10 && entry->arch <= device_cc) {
        return true;
      }
    } else if (entry->kind == LUPINE_FATBIN_ENTRY_PTX) {
      if (entry->arch <= device_cc) {
        return true;
      }
    } else {
      return true;
    }
    offset += entry->header_size + entry->payload_size;
  }
  return false;
}

// The bound device's compute capability, both lookups served from the caches
// the device snapshot prefills, so the fat binary test above costs no round
// trip of its own.
static bool lupine_current_device_compute_capability(unsigned *cc) {
  CUdevice device = 0;
  int major = 0;
  int minor = 0;
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device) != CUDA_SUCCESS) {
    return false;
  }
  *cc = static_cast<unsigned>(major) * 10 + static_cast<unsigned>(minor);
  return true;
}

static CUresult lupine_module_load_data_impl(CUmodule *module,
                                             const void *image, bool with_jit,
                                             unsigned int numOptions,
                                             CUjit_option *options,
                                             void **optionValues) {
  if (module == nullptr || image == nullptr ||
      (with_jit && numOptions != 0 &&
       (options == nullptr || optionValues == nullptr))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  uint32_t kind = 0;
  const unsigned char *image_data = nullptr;
  size_t image_size = 0;
  if (!lupine_locate_module_image(image, &kind, &image_data, &image_size)) {
    return CUDA_ERROR_INVALID_IMAGE;
  }
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    CUresult result =
        with_jit ? lupine_call_real_cuda_fn("cuModuleLoadDataEx", module, image,
                                            numOptions, options, optionValues)
                 : lupine_call_real_cuda_fn("cuModuleLoadData", module, image);
    if (result == CUDA_SUCCESS) {
      lupine_note_module_owner_route(*module, route);
    }
    return result;
  }
  unsigned device_cc = 0;
  if (!with_jit && lupine_current_device_compute_capability(&device_cc) &&
      !lupine_fatbin_may_serve_device(image_data, image_size, device_cc)) {
    return CUDA_ERROR_NO_BINARY_FOR_GPU;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, with_jit ? RPC_cuModuleLoadDataEx
                                             : RPC_cuModuleLoadData) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write(conn, image_data, image_size) < 0 ||
      (with_jit &&
       lupine_write_jit_options(conn, numOptions, options, optionValues) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (with_jit &&
       lupine_read_jit_outputs(conn, numOptions, options, optionValues) < 0) ||
      rpc_read(conn, module, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      lupine_read_module_functions(conn, *module, route) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_module_owner(*module, conn);
  }
  return return_value;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  return lupine_module_load_data_impl(module, image, false, 0, nullptr,
                                      nullptr);
}

// A fat binary container is one of the image shapes cuModuleLoadData already
// packs, and the kind tag the request carries picks the driver entry point the
// server loads it with.
extern "C" CUresult cuModuleLoadFatBinary(CUmodule *module,
                                          const void *fatCubin) {
  return cuModuleLoadData(module, fatCubin);
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                       unsigned int numOptions,
                                       CUjit_option *options,
                                       void **optionValues) {
  return lupine_module_load_data_impl(module, image, true, numOptions, options,
                                      optionValues);
}

// Field-for-field inverse of lupine_write_library. Populate the caches as
// records arrive, without a packed buffer or intermediate table. Only local
// cache operations are allowed here: the load response is still being read.
static int lupine_read_library(conn_t *conn, CUlibrary library) {
  CUresult result = CUDA_ERROR_UNKNOWN;
  CUdevice device = -1;
  uint32_t count = 0;
  if (rpc_read(conn, &result, sizeof(result)) < 0 ||
      rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 || count > 1024 * 1024) {
    return -1;
  }
  lupine_library_enumeration enumeration{result, {}};
  if (result != CUDA_SUCCESS) {
    if (count != 0) {
      return -1;
    }
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_library_enumerations()[library] = std::move(enumeration);
    return 0;
  }
  enumeration.kernels.reserve(count);
  lupine_route route = lupine_remote_route_for_conn(conn);
  int route_id = lupine_route_identity(route);
  CUcontext context = nullptr;
  (void)cuCtxGetCurrent(&context);
  for (uint32_t i = 0; i < count; ++i) {
    CUkernel kernel = nullptr;
    CUfunction function = nullptr;
    std::string name;
    std::shared_ptr<const lupine_param_layout> layout;
    uint32_t param_count = 0;
    if (lupine_read_callable_wire_record(conn, route_id, &kernel, &function,
                                         &name, &layout, &param_count) < 0 ||
        lupine_read_attributes(conn, lupine_kernel_attribute_key{
                                         route_id, kernel, 0, device}) < 0) {
      return -1;
    }
    enumeration.kernels.push_back(kernel);
    const bool params_known = param_count != LUPINE_PARAM_COUNT_UNKNOWN;
    if (function != nullptr && device >= 0) {
      lupine_attribute_set(
          lupine_function_attribute_key{route_id, function,
                                        LUPINE_FUNC_ATTRIBUTE_DEVICE},
          device);
    }
    if (params_known) {
      lupine_cache_param_layout(reinterpret_cast<uintptr_t>(kernel), true,
                                layout);
    }
    if (function != nullptr) {
      lupine_note_function_owner_route(function, route);
      if (params_known) {
        lupine_cache_param_layout(reinterpret_cast<uintptr_t>(function), false,
                                  layout);
      }
      lupine_cache_kernel_function(route_id, context, kernel, function);
    }
    lupine_library_kernel_names().insert_or_assign(
        lupine_named_handle_key<CUlibrary>{library, name}, kernel);
    lupine_note_function_owner_route(reinterpret_cast<CUfunction>(kernel),
                                     route);
    lupine_cache_library_kernel(kernel, library, std::move(name), route);
    if (function != nullptr) {
      lupine_record_library_function(function, kernel);
    }
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  lupine_library_enumerations()[library] = std::move(enumeration);
  return 0;
}

// A profiled library image: the build-id/UUID of the mapped object it is
// embedded in, its file offset there, and the library options. `record` is the
// profile form: kind, packed size, id length, id, offset, then the options with
// an absent value array encoded as each option's complement, which is also how
// they go on the wire. Claims compare the whole record, so a pointer-valued
// option that differs between runs misses. An image outside any mapped object
// has an empty id and is never profiled.
struct lupine_library_reference {
  std::string id;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint32_t kind = 0;
  std::string record;
};

static void lupine_library_reference_record(lupine_library_reference *ref,
                                            CUlibraryOption *libraryOptions,
                                            void **libraryOptionValues,
                                            unsigned int numLibraryOptions) {
  std::string &out = ref->record;
  auto append = [&](const void *data, size_t size) {
    if (size != 0) {
      out.append(static_cast<const char *>(data), size);
    }
  };
  auto id_size = static_cast<uint8_t>(ref->id.size());
  out.clear();
  append(&ref->kind, sizeof(ref->kind));
  append(&ref->size, sizeof(ref->size));
  append(&id_size, sizeof(id_size));
  append(ref->id.data(), id_size);
  append(&ref->offset, sizeof(ref->offset));
  append(&numLibraryOptions, sizeof(numLibraryOptions));
  append(libraryOptions, numLibraryOptions * sizeof(*libraryOptions));
  for (unsigned int i = 0; i < numLibraryOptions; ++i) {
    uintptr_t value = libraryOptionValues != nullptr
                          ? reinterpret_cast<uintptr_t>(libraryOptionValues[i])
                          : ~static_cast<uintptr_t>(libraryOptions[i]);
    append(&value, sizeof(value));
  }
}

// A profiled record goes on the wire only after its shape checks out: a
// truncated option tail would desynchronize the lane.
static bool lupine_library_reference_decode(const std::string &record,
                                            lupine_library_reference *ref) {
  if (record.size() < 21) {
    return false;
  }
  auto id_size = static_cast<uint8_t>(record[12]);
  std::memcpy(&ref->kind, record.data(), sizeof(ref->kind));
  std::memcpy(&ref->size, record.data() + 4, sizeof(ref->size));
  size_t at = 21u + id_size;
  unsigned int count = 0;
  if (record.size() < at + sizeof(count)) {
    return false;
  }
  std::memcpy(&count, record.data() + at, sizeof(count));
  at += sizeof(count) + static_cast<size_t>(count) * 12u;
  if (id_size == 0 || ref->size < sizeof(lupine_fatbin_header) ||
      at != record.size()) {
    return false;
  }
  ref->id.assign(record, 13, id_size);
  std::memcpy(&ref->offset, record.data() + 13 + id_size, sizeof(ref->offset));
  ref->record = record;
  return true;
}

// Locating a fatbin among the objects mapped into this process, in either
// direction: from its address to the containing object's build-id and file
// offset, or from a profiled build-id and offset back to its bytes.
static const unsigned char *lupine_validated_fatbin(const unsigned char *data,
                                                    uint64_t size) {
  const auto *header = reinterpret_cast<const lupine_fatbin_header *>(data);
  return data != nullptr && header->magic == LUPINE_FATBIN_MAGIC &&
                 header->header_size <= size &&
                 header->files_size == size - header->header_size
             ? data
             : nullptr;
}

#if defined(__linux__)
struct lupine_fatbin_locator {
  lupine_library_reference *ref;
  const unsigned char *data;
};

// The NT_GNU_BUILD_ID note of the object's PT_NOTE segments; note headers are
// padded to the segment's alignment, 4 or 8.
static std::string lupine_object_build_id(const dl_phdr_info *info) {
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) &ph = info->dlpi_phdr[i];
    const auto *notes =
        reinterpret_cast<const unsigned char *>(info->dlpi_addr + ph.p_vaddr);
    const size_t pad = ph.p_align == 8 ? 8 : 4;
    for (size_t at = 0;
         ph.p_type == PT_NOTE && at + sizeof(ElfW(Nhdr)) <= ph.p_filesz;) {
      const auto *note = reinterpret_cast<const ElfW(Nhdr) *>(notes + at);
      size_t name = at + sizeof(*note);
      size_t desc = name + ((note->n_namesz + pad - 1) & ~(pad - 1));
      if (desc + note->n_descsz > ph.p_filesz) {
        break;
      }
      if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz == 4 &&
          std::memcmp(notes + name, "GNU", 4) == 0) {
        return std::string(reinterpret_cast<const char *>(notes + desc),
                           note->n_descsz);
      }
      at = desc + ((note->n_descsz + pad - 1) & ~(pad - 1));
    }
  }
  return {};
}

static int lupine_locate_fatbin_in_object(dl_phdr_info *info, size_t,
                                          void *arg) {
  auto *locator = static_cast<lupine_fatbin_locator *>(arg);
  lupine_library_reference &ref = *locator->ref;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr) &ph = info->dlpi_phdr[i];
    uintptr_t start = info->dlpi_addr + ph.p_vaddr;
    // Unsigned: an address or offset before the segment wraps past p_filesz.
    uint64_t at = ref.id.empty()
                      ? reinterpret_cast<uintptr_t>(locator->data) - start
                      : ref.offset - ph.p_offset;
    if (ph.p_type != PT_LOAD || at >= ph.p_filesz ||
        ref.size > ph.p_filesz - at) {
      continue;
    }
    if (ref.id.empty()) {
      ref.id = lupine_object_build_id(info);
      ref.offset = ph.p_offset + at;
      return 1;
    }
    if (lupine_object_build_id(info) != ref.id) {
      return 0;
    }
    locator->data = reinterpret_cast<const unsigned char *>(start + at);
    return 1;
  }
  return 0;
}

static bool lupine_locate_fatbin(const void *data,
                                 lupine_library_reference *ref) {
  lupine_fatbin_locator locator{ref, static_cast<const unsigned char *>(data)};
  ref->id.clear();
  return dl_iterate_phdr(lupine_locate_fatbin_in_object, &locator) != 0 &&
         !ref->id.empty();
}

static const unsigned char *
lupine_mapped_fatbin(lupine_library_reference *ref) {
  lupine_fatbin_locator locator{ref, nullptr};
  return dl_iterate_phdr(lupine_locate_fatbin_in_object, &locator) != 0
             ? lupine_validated_fatbin(locator.data, ref->size)
             : nullptr;
}
#elif defined(__APPLE__)
static bool lupine_locate_fatbin(const void *data,
                                 lupine_library_reference *ref) {
  ref->id.clear();
  return lupine_macos_profile::image(static_cast<const unsigned char *>(data),
                                     ref->size, ref->id,
                                     ref->offset) != nullptr;
}

static const unsigned char *
lupine_mapped_fatbin(lupine_library_reference *ref) {
  return lupine_validated_fatbin(
      lupine_macos_profile::image(nullptr, ref->size, ref->id, ref->offset),
      ref->size);
}
#else
static bool lupine_locate_fatbin(const void *, lupine_library_reference *) {
  return false;
}

static const unsigned char *lupine_mapped_fatbin(lupine_library_reference *) {
  return nullptr;
}
#endif

// Both the length and filtered header belong to the RPC's caller, not to the
// serializer. rpc_write borrows their storage and the caller's image spans
// through response completion.
struct lupine_library_wire_image {
  const unsigned char *original;
  uint64_t size;
  lupine_filtered_fatbin filtered;
};
static_assert(sizeof(uint64_t) == sizeof(size_t));

static lupine_library_wire_image
lupine_prepare_library_image(conn_t *conn, const lupine_library_reference &ref,
                             const unsigned char *image, bool filter = true) {
  lupine_library_wire_image wire{image, ref.size, {}};
  std::shared_ptr<const std::vector<unsigned>> devices;
  if (filter) {
    if (auto *s = lupine_conn_slot(conn)) {
      std::lock_guard<std::mutex> lock(lupine_conn_cache_mutex());
      devices = s->capabilities;
    }
  }
  if (devices != nullptr) {
    wire.filtered = lupine_filter_fatbin_spans(image, ref.size, *devices);
    if (wire.filtered.valid) {
      wire.size = wire.filtered.size;
    }
  }
  return wire;
}

static int lupine_write_library_image(conn_t *conn,
                                      const lupine_library_reference &ref,
                                      const lupine_library_wire_image &image) {
  if (rpc_write(conn, &ref.kind, sizeof(ref.kind)) < 0 ||
      rpc_write(conn, &image.size, sizeof(image.size)) < 0) {
    return -1;
  }
  if (!image.filtered.valid) {
    return rpc_write(conn, image.original, static_cast<size_t>(image.size)) < 0
               ? -1
               : 0;
  }
  if (rpc_write(conn, &image.filtered.header, sizeof(image.filtered.header)) <
      0) {
    return -1;
  }
  const lupine_fatbin_span *spans = image.filtered.spans();
  for (size_t i = 0; i < image.filtered.span_count; ++i) {
    if (rpc_write(conn, image.original + spans[i].offset,
                  static_cast<size_t>(spans[i].size)) < 0) {
      return -1;
    }
  }
  return 0;
}

static int lupine_write_library_options(conn_t *conn,
                                        const lupine_library_reference &ref) {
  size_t at = 21 + ref.id.size();
  return rpc_write(conn, ref.record.data() + at, ref.record.size() - at);
}

static int lupine_read_library_load_response(
    conn_t *conn, unsigned int numJitOptions, const CUjit_option *jitOptions,
    void **jitOptionsValues, CUlibrary *library, CUresult *result) {
  if (rpc_read(conn, library, sizeof(*library)) < 0 ||
      lupine_read_jit_outputs(conn, numJitOptions, jitOptions,
                              jitOptionsValues) < 0 ||
      rpc_read(conn, result, sizeof(*result)) < 0 ||
      (*result == CUDA_SUCCESS && lupine_read_library(conn, *library) < 0)) {
    return -1;
  }
  return 0;
}

// One batch per process, for the route of the first remote load: the images
// the previous run of this command line loaded, re-read from this process's
// own mapping of their objects and requested in one round trip. Entries are
// handed out once, so every handle cudart receives is distinct and
// cuLibraryUnload keeps its native meaning.
struct lupine_library_batch_entry {
  lupine_library_reference ref;
  const unsigned char *image = nullptr;
  CUlibrary library = nullptr;
  CUresult result = CUDA_ERROR_UNKNOWN;
};

struct lupine_library_batch_state {
  std::mutex mutex;
  std::condition_variable finished_cv;
  bool started = false;
  bool finished = true;
  int route_id = -2;
  std::vector<lupine_library_batch_entry> entries;
  std::filesystem::path profile;
  std::string lines;
};

static lupine_library_batch_state &lupine_library_batch() {
  return lupine_leaked<lupine_library_batch_state>([] {});
}

// Profiles are keyed by LUPINE_LIBRARY_PROFILE or the full command line, so
// `python train.py` and `python eval.py` keep separate load orders.
static std::filesystem::path lupine_library_profile_path() {
  const char *name = std::getenv("LUPINE_LIBRARY_PROFILE");
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  const char *home = std::getenv("HOME");
  std::string key = name != nullptr ? name : "";
  if (name == nullptr) {
#if defined(__APPLE__)
    std::string argv = lupine_macos_profile::command_line();
#else
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    std::string argv((std::istreambuf_iterator<char>(cmdline)),
                     std::istreambuf_iterator<char>());
#endif
    char hex[17];
    std::snprintf(
        hex, sizeof(hex), "%016llx",
        static_cast<unsigned long long>(XXH64(argv.data(), argv.size(), 0)));
    key = argv.empty() ? "" : hex;
  }
  std::filesystem::path dir = xdg != nullptr && *xdg != '\0'
                                  ? std::filesystem::path(xdg)
                              : home != nullptr && *home != '\0'
                                  ? std::filesystem::path(home) / ".cache"
                                  : std::filesystem::path();
  return dir.empty() || key.empty() ? std::filesystem::path()
                                    : dir / "lupine" / "profiles" / key;
}

// Through a temp file, so a concurrent reader never sees a partial profile.
static void lupine_write_file_atomically(const std::filesystem::path &path,
                                         const std::string &data) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::filesystem::path temp = path;
  temp += "." +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".tmp";
  std::ofstream out(temp, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();
  if (out) {
    std::filesystem::rename(temp, path, ec);
  }
  std::filesystem::remove(temp, ec);
}

// Runs on its own lane with the app's context current: the server loads under
// it, and lupine_read_library keys its caches by the reading thread's context.
static void lupine_library_batch_main(conn_t *conn, CUcontext context) {
  auto &state = lupine_library_batch();
  lupine_current_context = context;
  std::vector<lupine_library_batch_entry *> sent;
  for (auto &entry : state.entries) {
    if ((entry.image = lupine_mapped_fatbin(&entry.ref)) != nullptr) {
      sent.push_back(&entry);
    }
  }
  // Finish selection before opening the request. The vector and its byte
  // buffers stay alive until every batch response has been consumed.
  std::vector<lupine_library_wire_image> images;
  images.reserve(sent.size());
  for (auto *entry : sent) {
    images.push_back(
        lupine_prepare_library_image(conn, entry->ref, entry->image));
  }
  auto count = static_cast<uint32_t>(sent.size());
  bool ok =
      count != 0 && lupine_prepare_rpc(conn) >= 0 &&
      rpc_write_start_request(conn, LUPINE_RPC_lupineLibraryLoadBatch) >= 0 &&
      rpc_write(conn, &context, sizeof(context)) >= 0 &&
      rpc_write(conn, &count, sizeof(count)) >= 0;
  for (size_t i = 0; i < sent.size(); ++i) {
    ok = ok && lupine_write_library_image(conn, sent[i]->ref, images[i]) >= 0 &&
         lupine_write_library_options(conn, sent[i]->ref) >= 0;
  }
  ok = ok && rpc_wait_for_response(conn) >= 0;
  uint32_t loaded = 0;
  for (auto *entry : sent) {
    if (!ok || lupine_read_library_load_response(conn, 0, nullptr, nullptr,
                                                 &entry->library,
                                                 &entry->result) < 0) {
      ok = false;
      entry->result = CUDA_ERROR_UNKNOWN;
    }
    loaded += entry->result == CUDA_SUCCESS;
  }
  ok = ok && rpc_read_end(conn) >= 0;
  LUPINE_TRACE_LOG("LUPINE library batch: "
                   << state.entries.size() << " profiled, " << count
                   << " sent, " << loaded << " loaded, ok=" << ok);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.finished = true;
  state.finished_cv.notify_all();
}

static void lupine_library_batch_start(lupine_route route) {
  auto &state = lupine_library_batch();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.started) {
    return;
  }
  state.started = true;
  state.route_id = lupine_route_identity(route);
  state.profile = lupine_library_profile_path();
  std::ifstream in(state.profile, std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  for (size_t at = 0; at + 4 <= data.size() && state.entries.size() < 4096;) {
    uint32_t size = 0;
    std::memcpy(&size, &data[at], sizeof(size));
    at += sizeof(size);
    lupine_library_batch_entry entry;
    if (size > data.size() - at ||
        !lupine_library_reference_decode(data.substr(at, size), &entry.ref)) {
      break;
    }
    at += size;
    state.entries.push_back(std::move(entry));
  }
  if (!state.entries.empty()) {
    state.finished = false;
    std::thread(lupine_library_batch_main, lupine_route_remote_conn(route),
                lupine_current_context)
        .detach();
  }
}

static bool lupine_library_batch_claim(lupine_route route,
                                       const lupine_library_reference &ref,
                                       CUlibrary *library) {
  auto &state = lupine_library_batch();
  std::unique_lock<std::mutex> lock(state.mutex);
  if (lupine_route_identity(route) != state.route_id) {
    return false;
  }
  state.finished_cv.wait(lock, [&] { return state.finished; });
  auto it = std::find_if(
      state.entries.begin(), state.entries.end(),
      [&](const auto &entry) { return entry.ref.record == ref.record; });
  if (it == state.entries.end() || it->result != CUDA_SUCCESS) {
    return false;
  }
  *library = it->library;
  state.entries.erase(it);
  return true;
}

static void lupine_library_profile_record(lupine_route route,
                                          const lupine_library_reference &ref) {
  auto &state = lupine_library_batch();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.profile.empty() || lupine_route_identity(route) != state.route_id) {
    return;
  }
  auto size = static_cast<uint32_t>(ref.record.size());
  state.lines.append(reinterpret_cast<const char *>(&size), sizeof(size));
  state.lines += ref.record;
  lupine_write_file_atomically(state.profile, state.lines);
}

extern "C" CUresult
cuLibraryLoadData(CUlibrary *library, const void *code,
                  CUjit_option *jitOptions, void **jitOptionsValues,
                  unsigned int numJitOptions, CUlibraryOption *libraryOptions,
                  void **libraryOptionValues, unsigned int numLibraryOptions) {
  if (library == nullptr || code == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((numJitOptions != 0 &&
       (jitOptions == nullptr || jitOptionsValues == nullptr)) ||
      (numLibraryOptions != 0 && libraryOptions == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uint32_t kind = 0;
  const unsigned char *image_data = nullptr;
  size_t image_size = 0;
  if (!lupine_locate_module_image(code, &kind, &image_data, &image_size)) {
    const auto *wrapper = reinterpret_cast<const lupine_fatbin_wrapper *>(code);
    LUPINE_TRACE_LOG("LUPINE cuLibraryLoadData could not pack image"
                     << " magic=0x" << std::hex << wrapper->magic
                     << " version=0x" << wrapper->version << std::dec);
    return CUDA_ERROR_INVALID_IMAGE;
  }

  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    CUresult result = lupine_call_real_cuda_fn(
        "cuLibraryLoadData", library, code, jitOptions, jitOptionsValues,
        numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
    if (result == CUDA_SUCCESS) {
      lupine_note_library_owner_route(*library, route);
      lupine_record_library_image(*library, route, kind, image_data, image_size,
                                  code);
    }
    return result;
  }

  // Loads carrying JIT options are never profiled: their log buffers and
  // wall-time outputs belong to the call that passes them.
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_library_reference ref;
  ref.kind = kind;
  ref.size = image_size;
  bool profiled =
      numJitOptions == 0 && ref.size >= sizeof(lupine_fatbin_header) &&
      reinterpret_cast<const lupine_fatbin_header *>(image_data)->magic ==
          LUPINE_FATBIN_MAGIC &&
      lupine_locate_fatbin(image_data, &ref);
  lupine_library_reference_record(&ref, libraryOptions, libraryOptionValues,
                                  numLibraryOptions);
  lupine_library_batch_start(route);
  CUresult return_value = CUDA_SUCCESS;
  const bool claimed =
      profiled && lupine_library_batch_claim(route, ref, library);
  // JIT options can request a target other than a server device, so preserve
  // those images verbatim. Profiles and cross-route replays retain originals.
  auto wire = lupine_prepare_library_image(conn, ref, image_data,
                                           !claimed && numJitOptions == 0);
  if (!claimed && (lupine_prepare_rpc(conn) < 0 ||
                   rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
                   lupine_write_library_image(conn, ref, wire) < 0 ||
                   lupine_write_jit_options(conn, numJitOptions, jitOptions,
                                            jitOptionsValues) < 0 ||
                   lupine_write_library_options(conn, ref) < 0 ||
                   rpc_wait_for_response(conn) < 0 ||
                   lupine_read_library_load_response(
                       conn, numJitOptions, jitOptions, jitOptionsValues,
                       library, &return_value) < 0 ||
                   rpc_read_end(conn) < 0)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_library_owner(*library, conn);
    lupine_record_library_image(*library, lupine_remote_route_for_conn(conn),
                                kind, image_data, image_size, code);
    if (profiled) {
      lupine_library_profile_record(route, ref);
    }
  }
  return return_value;
}

// The file is mapped read-only on the client and loaded as an image, like
// cuModuleLoad.
extern "C" CUresult cuLibraryLoadFromFile(
    CUlibrary *library, const char *fileName, CUjit_option *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    CUlibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  if (library == nullptr || fileName == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  void *mapping = MAP_FAILED;
  size_t mapped_size = 0;
  CUresult result = lupine_mmap_readonly_file(fileName, &mapping, &mapped_size);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  const auto *bytes = static_cast<const unsigned char *>(mapping);
  const bool is_binary =
      (mapped_size >= sizeof(lupine_fatbin_header) &&
       reinterpret_cast<const lupine_fatbin_header *>(bytes)->magic ==
           LUPINE_FATBIN_MAGIC) ||
      (mapped_size >= SELFMAG && std::memcmp(bytes, ELFMAG, SELFMAG) == 0);
  static const size_t page_size = []() -> size_t {
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<size_t>(p) : 4096;
  }();
#ifdef _WIN32
  const bool needs_nul = !is_binary && bytes[mapped_size - 1] != '\0';
#else
  const bool needs_nul = !is_binary && (mapped_size % page_size == 0) &&
                         bytes[mapped_size - 1] != '\0';
#endif
  std::unique_ptr<char[]> nul_copy;
  const void *code = mapping;
  if (needs_nul) {
    nul_copy.reset(new (std::nothrow) char[mapped_size + 1]);
    if (!nul_copy) {
      munmap(mapping, mapped_size);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(nul_copy.get(), mapping, mapped_size);
    nul_copy[mapped_size] = '\0';
    code = nul_copy.get();
  }
  result = cuLibraryLoadData(library, code, jitOptions, jitOptionsValues,
                             numJitOptions, libraryOptions, libraryOptionValues,
                             numLibraryOptions);
  if (result == CUDA_SUCCESS) {
    // mapping dies with this call; route replays load the recorded image.
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_library_images().find(*library);
    if (it != lupine_library_images().end()) {
      it->second.code = it->second.image.data();
    }
  }
  munmap(mapping, mapped_size);
  return result;
}

template <typename Fn>
static CUresult lupine_with_library_enumeration(CUlibrary library, Fn &&fn) {
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto it = lupine_library_enumerations().find(library);
  if (it == lupine_library_enumerations().end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (it->second.result == CUDA_SUCCESS) {
    fn(it->second.kernels);
  }
  return it->second.result;
}

extern "C" CUresult cuLibraryGetKernelCount(unsigned int *count,
                                            CUlibrary library) {
  if (count == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_route_is_local(lupine_route_for_library(library))) {
    return lupine_call_real_cuda_fn("cuLibraryGetKernelCount", count, library);
  }
  return lupine_with_library_enumeration(library, [&](const auto &kernels) {
    *count = static_cast<unsigned int>(kernels.size());
  });
}

extern "C" CUresult cuLibraryEnumerateKernels(CUkernel *kernels,
                                              unsigned int numKernels,
                                              CUlibrary library) {
  if (kernels == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_route_is_local(lupine_route_for_library(library))) {
    return lupine_call_real_cuda_fn("cuLibraryEnumerateKernels", kernels,
                                    numKernels, library);
  }
  return lupine_with_library_enumeration(library, [&](const auto &src) {
    std::copy_n(src.begin(), std::min<size_t>(numKernels, src.size()), kernels);
  });
}

template <typename Handle,
          CUresult (*Query)(Handle, size_t, size_t *, size_t *), bool IsKernel>
static CUresult lupine_get_param_layout(
    Handle handle, std::shared_ptr<const lupine_param_layout> *out = nullptr) {
  lupine_param_layout_key key{reinterpret_cast<uintptr_t>(handle), IsKernel};
  std::shared_ptr<const lupine_param_layout> cached;
  if (lupine_param_layout_cache().find(key, cached) && cached != nullptr) {
    if (out != nullptr) {
      *out = std::move(cached);
    }
    return CUDA_SUCCESS;
  }

  auto built = std::make_shared<lupine_param_layout>();
  for (uint32_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = Query(handle, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      std::shared_ptr<const lupine_param_layout> layout = std::move(built);
      lupine_param_layout_cache().insert_or_assign(key, layout);
      if (out != nullptr) {
        *out = std::move(layout);
      }
      return CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    built->offsets.push_back(offset);
    built->sizes.push_back(size);
  }
}

static CUresult lupine_get_func_param_layout(
    CUfunction function,
    std::shared_ptr<const lupine_param_layout> *out = nullptr) {
  return lupine_get_param_layout<CUfunction, cuFuncGetParamInfo, false>(
      function, out);
}

static CUresult lupine_get_kernel_param_layout(
    CUkernel kernel,
    std::shared_ptr<const lupine_param_layout> *out = nullptr) {
  return lupine_get_param_layout<CUkernel, cuKernelGetParamInfo, true>(kernel,
                                                                       out);
}

template <typename Params>
static CUresult lupine_kernel_node_param_layout(
    const Params &params, std::shared_ptr<const lupine_param_layout> *out) {
  if (params.extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (params.func != nullptr) {
    return lupine_get_func_param_layout(params.func, out);
  }
#if CUDA_VERSION >= 12000
  if (params.kern != nullptr) {
    return lupine_get_kernel_param_layout(params.kern, out);
  }
#endif
  return CUDA_ERROR_INVALID_HANDLE;
}

static int lupine_write_kernel_params(conn_t *conn, void *const *params,
                                      const std::vector<size_t> &sizes,
                                      uint32_t *param_count) {
  *param_count = static_cast<uint32_t>(sizes.size());
  if (rpc_write(conn, param_count, sizeof(*param_count)) < 0 ||
      rpc_write(conn, sizes.data(), sizes.size() * sizeof(*sizes.data())) < 0) {
    return -1;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (rpc_write(conn, params[i], sizes[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

// The parameter-buffer launch form packs the arguments into one blob, but the
// wire and pointer translation take them one by one, so point each argument at
// its offset in the caller's blob.
static void **
lupine_params_from_param_buffer(void **extra,
                                const std::vector<size_t> &offsets,
                                std::vector<void *> *params) {
  char *buffer = nullptr;
  for (size_t i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
    if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buffer = static_cast<char *>(extra[i + 1]);
    }
  }
  if (buffer == nullptr) {
    return nullptr;
  }
  for (size_t offset : offsets) {
    params->push_back(buffer + offset);
  }
  return params->data();
}

static CUresult
lupine_prepare_kernel_params(lupine_route route, void *const *kernel_params,
                             const std::vector<size_t> &sizes,
                             std::vector<CUdeviceptr> *managed = nullptr) {
  if (!sizes.empty() && kernel_params == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (kernel_params[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  uintptr_t min_addr = 0;
  uintptr_t max_addr = 0;
  if (!lupine_tracked_host_bounds(&min_addr, &max_addr)) {
    return CUDA_SUCCESS;
  }
  bool portable = false;
  uintptr_t candidates[128];
  size_t candidate_count = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    // Pointers also travel inside by-value structs (spans, iterators, tensor
    // descriptors), at pointer-aligned offsets within the parameter's bytes.
    const auto *bytes = static_cast<const unsigned char *>(kernel_params[i]);
    for (size_t offset = 0; offset + sizeof(CUdeviceptr) <= sizes[i];
         offset += alignof(CUdeviceptr)) {
      CUdeviceptr value = 0;
      memcpy(&value, bytes + offset, sizeof(value));
      uintptr_t addr = static_cast<uintptr_t>(value);
      if (addr != 0 && addr >= min_addr && addr < max_addr) {
        candidates[candidate_count++] = addr;
        if (candidate_count == std::size(candidates)) {
          lupine_prepare_mapped_host_pointers(candidates, candidate_count,
                                              &portable, managed);
          candidate_count = 0;
        }
      }
    }
  }
  if (candidate_count != 0) {
    lupine_prepare_mapped_host_pointers(candidates, candidate_count, &portable,
                                        managed);
  }
  return lupine_prepare_portable_host_allocations(route, portable);
}

static CUresult lupine_warm_func_param_info(CUfunction function) {
  return lupine_get_func_param_layout(function, nullptr);
}

static CUresult lupine_warm_kernel_param_info(CUkernel kernel) {
  return lupine_get_kernel_param_layout(kernel, nullptr);
}

extern "C" void lupine_invalidate_function_caches() {
#if CUDA_VERSION >= 12000
  {
    std::lock_guard<std::mutex> lock(lupine_tensor_map_mutex);
    lupine_tensor_maps.clear();
    ++lupine_tensor_map_generation;
  }
#endif
  lupine_module_function_names().clear();
  lupine_param_info_cache().clear();
  lupine_param_layout_cache().clear();
  if (lupine_kernel_function_cache().try_get() != nullptr) {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_kernel_function_keys().clear();
  }
  lupine_kernel_function_cache().clear();
  lupine_occupancy_cache().clear();
  get_cache<lupine_attribute_cache::kernel>().clear();
  get_cache<lupine_attribute_cache::function>().clear();
  get_cache<lupine_attribute_cache::function_set>().clear();
}

static CUresult lupine_resolve_launch_function_for_route(
    CUfunction requested_function, lupine_route route,
    CUfunction *launch_function, lupine_route *function_route,
    bool *is_kernel) {
  CUfunction resolved = nullptr;
  CUresult result = lupine_resolve_library_kernel_for_route(
      requested_function, route, &resolved, is_kernel);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  resolved = lupine_translate_private_function(resolved);
  lupine_route resolved_route = lupine_route_for_function(resolved);
  if (!lupine_routes_share_server(resolved_route, route)) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *launch_function = resolved;
  *function_route = resolved_route;
  return CUDA_SUCCESS;
}

struct lupine_prepared_launch {
  lupine_route launch_route{};
  lupine_route route{};
  std::shared_ptr<const lupine_param_layout> param_layout;
  std::vector<void *> buffer_params;
  std::vector<CUdeviceptr> managed;
  uint32_t param_count = 0;
};

static CUresult lupine_prepare_kernel_launch(CUfunction *f, CUstream hStream,
                                             void ***kernelParams, void **extra,
                                             lupine_prepared_launch *prepared) {
  CUfunction requested_function = *f;
  prepared->launch_route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                              : lupine_route_for_default();
  bool kernel_handle = false;
  CUresult status = lupine_resolve_launch_function_for_route(
      requested_function, prepared->launch_route, f, &prepared->route,
      &kernel_handle);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  status = kernel_handle
               ? lupine_get_kernel_param_layout(reinterpret_cast<CUkernel>(*f),
                                                &prepared->param_layout)
               : lupine_get_func_param_layout(*f, &prepared->param_layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  if (extra != nullptr) {
    prepared->buffer_params.clear();
    *kernelParams = lupine_params_from_param_buffer(
        extra, prepared->param_layout->offsets, &prepared->buffer_params);
  }
  prepared->managed.clear();
  return lupine_prepare_kernel_params(prepared->route, *kernelParams,
                                      prepared->param_layout->sizes,
                                      &prepared->managed);
}

static CUresult lupine_launch_grid_kernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra, bool cooperative) {
  thread_local lupine_prepared_launch prepared;
  CUresult status = lupine_prepare_kernel_launch(&f, hStream, &kernelParams,
                                                 extra, &prepared);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (!cooperative) {
    LUPINE_TRACE_LOG(
        "LUPINE cuLaunchKernel f="
        << f << " stream=" << hStream
        << " launch_route=" << lupine_route_identity(prepared.launch_route)
        << " function_route=" << lupine_route_identity(prepared.route)
        << " grid=(" << gridDimX << "," << gridDimY << "," << gridDimZ
        << ") block=(" << blockDimX << "," << blockDimY << "," << blockDimZ
        << ")");
  }
  if (lupine_route_is_local(prepared.route)) {
    return cooperative
               ? lupine_call_real_cuda_fn("cuLaunchCooperativeKernel", f,
                                          gridDimX, gridDimY, gridDimZ,
                                          blockDimX, blockDimY, blockDimZ,
                                          sharedMemBytes, hStream, kernelParams)
               : lupine_call_real_cuda_fn("cuLaunchKernel", f, gridDimX,
                                          gridDimY, gridDimZ, blockDimX,
                                          blockDimY, blockDimZ, sharedMemBytes,
                                          hStream, kernelParams, nullptr);
  }
  conn_t *conn = lupine_route_remote_conn(prepared.route);
  int rpc_id = cooperative ? RPC_cuLaunchCooperativeKernel : RPC_cuLaunchKernel;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, rpc_id, hStream) < 0 ||
      !lupine_write_lvals(conn, f, gridDimX, gridDimY, gridDimZ, blockDimX,
                          blockDimY, blockDimZ, sharedMemBytes, hStream) ||
      lupine_write_kernel_params(conn, kernelParams,
                                 prepared.param_layout->sizes,
                                 &prepared.param_count) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  for (CUdeviceptr pointer : prepared.managed)
    lupine_invalidate_launched_managed(pointer, hStream);
  return CUDA_SUCCESS;
}

extern "C" CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
               unsigned int gridDimZ, unsigned int blockDimX,
               unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra) {
  return lupine_launch_grid_kernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                   blockDimY, blockDimZ, sharedMemBytes,
                                   hStream, kernelParams, extra, false);
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                     void **kernelParams, void **extra) {
  if (config == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
#if CUDA_VERSION < 11080
  return CUDA_ERROR_NOT_SUPPORTED;
#else
  thread_local lupine_prepared_launch prepared;
  CUresult status = lupine_prepare_kernel_launch(
      &f, config->hStream, &kernelParams, extra, &prepared);
  if (status != CUDA_SUCCESS)
    return status;
  if (lupine_route_is_local(prepared.route))
    return lupine_call_real_cuda_fn<CUDA_ERROR_NOT_SUPPORTED>(
        "cuLaunchKernelEx", config, f, kernelParams, nullptr);
  conn_t *conn = lupine_route_remote_conn(prepared.route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuLaunchKernelEx, config->hStream) <
          0 ||
      rpc_write(conn, config, sizeof(*config)) < 0 ||
      rpc_write(conn, config->attrs,
                config->numAttrs * sizeof(*config->attrs)) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      lupine_write_kernel_params(conn, kernelParams,
                                 prepared.param_layout->sizes,
                                 &prepared.param_count) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  for (CUdeviceptr pointer : prepared.managed)
    lupine_invalidate_launched_managed(pointer, config->hStream);
  return CUDA_SUCCESS;
#endif
}

extern "C" CUresult
cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                          unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY,
                          unsigned int blockDimZ, unsigned int sharedMemBytes,
                          CUstream hStream, void **kernelParams) {
  return lupine_launch_grid_kernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                   blockDimY, blockDimZ, sharedMemBytes,
                                   hStream, kernelParams, nullptr, true);
}

static CUresult
lupine_validate_graph_dependencies(const CUgraphNode *dependencies,
                                   size_t numDependencies) {
  if (numDependencies != 0 && dependencies == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

// Returned graph parameter arrays belong to their node, independent of the
// caller's output struct. Repeated queries reuse storage until parameters
// change.
struct lupine_deep_cache_slot {
  size_t slot = 0;
  std::vector<unsigned char> bytes;
};
static std::mutex g_deep_cache_mutex;
static std::unordered_map<CUgraphNode, std::vector<lupine_deep_cache_slot>>
    g_deep_cache;
// Release paths acquire the handle-association mutex and this mutex together;
// no path acquires them separately in the opposite order.
static std::mutex g_retained_string_mutex;
static std::unordered_map<const void *, std::vector<char *>> g_retained_strings;

static void lupine_release_retained_strings_locked(const void *handle) {
  auto retained = g_retained_strings.find(handle);
  if (retained == g_retained_strings.end()) {
    return;
  }
  for (char *string : retained->second) {
    free(string);
  }
  g_retained_strings.erase(retained);
}

extern "C" const char *lupine_retain_returned_string(const void *handle,
                                                     const char *data,
                                                     size_t size) {
  if (handle == nullptr || (data == nullptr && size != 0)) {
    return nullptr;
  }

  auto *stored = static_cast<char *>(malloc(size + 1));
  if (stored == nullptr) {
    return nullptr;
  }
  if (size != 0) {
    memcpy(stored, data, size);
  }
  stored[size] = '\0';

  std::lock_guard<std::mutex> guard(g_retained_string_mutex);
  try {
    g_retained_strings[handle].push_back(stored);
  } catch (const std::bad_alloc &) {
    free(stored);
    return nullptr;
  }
  return stored;
}

template <typename Map, typename Pred, typename OnErase>
static void lupine_erase_retained_entries_locked(Map &map, Pred &&pred,
                                                 OnErase &&on_erase) {
  for (auto it = map.begin(); it != map.end();) {
    if (!pred(*it)) {
      ++it;
      continue;
    }
    on_erase(*it);
    lupine_release_retained_strings_locked(
        reinterpret_cast<const void *>(it->first));
    it = map.erase(it);
  }
}

static void lupine_erase_unloaded_callable_caches(
    const std::unordered_set<CUfunction> &unloaded_functions,
    const std::unordered_set<CUkernel> *unloaded_kernels,
    bool erase_function_set) {
  if (unloaded_functions.empty() &&
      (!unloaded_kernels || unloaded_kernels->empty()))
    return;
  auto has_fn = [&](CUfunction fn) {
    return unloaded_functions.count(fn) != 0;
  };
  auto has_kern = [&](CUkernel k) {
    return unloaded_kernels && unloaded_kernels->count(k) != 0;
  };
  lupine_cache_erase_if(
      lupine_param_info_cache(), [&](const auto &key, const auto &) {
        return key.kernel ? has_kern(reinterpret_cast<CUkernel>(key.handle))
                          : has_fn(reinterpret_cast<CUfunction>(key.handle));
      });
  for (CUfunction fn : unloaded_functions)
    lupine_param_layout_cache().erase(
        lupine_param_layout_key{reinterpret_cast<uintptr_t>(fn), false});
  if (unloaded_kernels) {
    for (CUkernel kern : *unloaded_kernels)
      lupine_param_layout_cache().erase(
          lupine_param_layout_key{reinterpret_cast<uintptr_t>(kern), true});
    if (auto *ka = get_cache<lupine_attribute_cache::kernel>().try_get();
        ka && !unloaded_kernels->empty()) {
      bool direct =
          !lupine_extended_kernel_attr.load(std::memory_order_relaxed);
      uint64_t devs =
          lupine_kernel_attr_devices.load(std::memory_order_acquire);
      for (CUkernel kern : *unloaded_kernels) {
        if (!direct)
          break;
        int rid = lupine_route_identity(
            lupine_route_for_function(reinterpret_cast<CUfunction>(kern)));
        if (rid == -2) {
          direct = false;
          break;
        }
        for (uint64_t m = devs, d = 0; m != 0; m >>= 1, ++d)
          if (m & 1)
            for (int a = 0; a < CU_FUNC_ATTRIBUTE_MAX; ++a)
              ka->erase(lupine_kernel_attribute_key{rid, kern, a,
                                                    static_cast<int>(d)});
      }
      if (!direct)
        lupine_cache_erase_if(
            get_cache<lupine_attribute_cache::kernel>(),
            [&](const auto &k, const auto &) { return has_kern(k.kernel); });
    }
  }
  if (auto *fn_attr = get_cache<lupine_attribute_cache::function>().try_get();
      fn_attr && !unloaded_functions.empty()) {
    bool direct = !lupine_extended_func_attr.load(std::memory_order_relaxed);
    for (CUfunction fn : unloaded_functions) {
      if (!direct)
        break;
      int rid = lupine_route_identity(lupine_route_for_function(fn));
      if (rid == -2) {
        direct = false;
        break;
      }
      for (int a = LUPINE_FUNC_ATTRIBUTE_DEVICE; a < CU_FUNC_ATTRIBUTE_MAX; ++a)
        fn_attr->erase(lupine_function_attribute_key{rid, fn, a});
    }
    if (!direct)
      lupine_cache_erase_if(
          get_cache<lupine_attribute_cache::function>(),
          [&](const auto &key, const auto &) { return has_fn(key.function); });
  }
  if (erase_function_set)
    lupine_cache_erase_if(
        get_cache<lupine_attribute_cache::function_set>(),
        [&](const auto &key, const auto &) { return has_fn(key.function); });
  lupine_cache_erase_if(
      lupine_occupancy_cache(), [&](const auto &key, const auto &) {
        return has_fn(key.function) ||
               has_kern(reinterpret_cast<CUkernel>(key.function));
      });
}

extern "C" void lupine_release_module_retained_strings(CUmodule module) {
  std::unordered_set<CUfunction> unloaded;
  {
    std::scoped_lock lock(lupine_library_kernel_mutex(),
                          g_retained_string_mutex);
    lupine_release_retained_strings_locked(
        reinterpret_cast<const void *>(module));
    lupine_erase_retained_entries_locked(
        lupine_module_functions(),
        [&](const auto &e) { return e.second.module == module; },
        [&](const auto &e) {
          unloaded.insert(e.first);
          if (!e.second.name.empty()) {
            lupine_module_function_names().erase(
                lupine_named_handle_view<CUmodule>{module, e.second.name});
          }
        });
    lupine_library_modules().erase(module);
  }
  lupine_erase_unloaded_callable_caches(unloaded, nullptr, true);
}

extern "C" void lupine_release_library_retained_strings(CUlibrary library) {
  // The library's functions are freed and their handles may be reused.
  get_cache<lupine_attribute_cache::function_set>().clear();
  std::unordered_set<CUkernel> unloaded_kernels;
  std::unordered_set<CUfunction> unloaded_functions;
  {
    std::scoped_lock lock(lupine_library_kernel_mutex(),
                          g_retained_string_mutex);
    lupine_library_images().erase(library);
    lupine_library_enumerations().erase(library);
    lupine_release_retained_strings_locked(
        reinterpret_cast<const void *>(library));
    auto &kernels = lupine_library_kernels();
    lupine_erase_retained_entries_locked(
        kernels, [&](const auto &e) { return e.second.library == library; },
        [&](const auto &e) {
          unloaded_kernels.insert(e.first);
          if (!e.second.name.empty()) {
            lupine_library_kernel_names().erase(
                lupine_named_handle_view<CUlibrary>{library, e.second.name});
          }
          for (const auto &route_entry : e.second.kernels_by_route) {
            unloaded_kernels.insert(route_entry.second);
          }
        });
    lupine_library_kernel_count.store(kernels.size(),
                                      std::memory_order_release);
    auto *kf_cache = lupine_kernel_function_cache().try_get();
    auto &kf_keys = lupine_kernel_function_keys();
    for (CUkernel kern : unloaded_kernels) {
      if (auto it = kf_keys.find(kern); it != kf_keys.end()) {
        if (kf_cache != nullptr)
          for (const auto &k : it->second)
            kf_cache->erase(k);
        kf_keys.erase(it);
      }
    }
    lupine_erase_retained_entries_locked(
        lupine_module_functions(),
        [&](const auto &e) {
          auto mod = lupine_library_modules().find(e.second.module);
          return mod != lupine_library_modules().end() &&
                 mod->second == library;
        },
        [&](const auto &e) {
          unloaded_functions.insert(e.first);
          if (!e.second.name.empty()) {
            lupine_module_function_names().erase(
                lupine_named_handle_view<CUmodule>{e.second.module,
                                                   e.second.name});
          }
        });
    lupine_erase_retained_entries_locked(
        lupine_library_functions(),
        [&](const auto &e) { return e.second == library; },
        [&](const auto &e) { unloaded_functions.insert(e.first); });
    auto &func_kernels = lupine_function_kernels();
    for (auto it = func_kernels.begin(); it != func_kernels.end();) {
      if (unloaded_functions.count(it->first) == 0 &&
          unloaded_kernels.count(it->second) == 0) {
        ++it;
        continue;
      }
      unloaded_functions.insert(it->first);
      it = func_kernels.erase(it);
    }
    lupine_erase_retained_entries_locked(
        lupine_library_modules(),
        [&](const auto &e) { return e.second == library; },
        [](const auto &) {});
  }
  lupine_erase_unloaded_callable_caches(unloaded_functions, &unloaded_kernels,
                                        false);
}

static void *lupine_deep_node_cache_get_locked(CUgraphNode node, size_t slot,
                                               size_t bytes) {
  auto &slots = g_deep_cache[node];
  for (auto &entry : slots) {
    if (entry.slot == slot) {
      // Repeated queries must reuse storage, including queries into different
      // output structs. Resizing is safe when a setter changed the parameters.
      entry.bytes.resize(bytes);
      return entry.bytes.data();
    }
  }
  lupine_deep_cache_slot added;
  added.slot = slot;
  added.bytes.resize(bytes);
  void *ptr = added.bytes.data();
  slots.push_back(std::move(added));
  return ptr;
}

extern "C" void *lupine_deep_node_cache_get(CUgraphNode node, size_t slot,
                                            size_t bytes) {
  std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
  try {
    return lupine_deep_node_cache_get_locked(node, slot, bytes);
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

extern "C" void lupine_deep_node_cache_reset(CUgraphNode node) {
  std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
  g_deep_cache.erase(node);
}

std::vector<CUgraphNode> lupine_deep_cache_graph_nodes(CUgraph graph) {
  {
    std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
    if (g_deep_cache.empty())
      return {};
  }
  // Enumerate before destruction: this also covers allocation nodes created
  // by stream capture, for which the client never observed an AddNode call.
  size_t count = 0;
  if (cuGraphGetNodes(graph, nullptr, &count) != CUDA_SUCCESS)
    return {};
  std::vector<CUgraphNode> nodes(count);
  if (count != 0 &&
      cuGraphGetNodes(graph, nodes.data(), &count) != CUDA_SUCCESS)
    return {};
  nodes.resize(count);
  return nodes;
}

static CUresult lupine_queue_graph_dependencies(conn_t *conn,
                                                const CUgraphNode *dependencies,
                                                const size_t *numDependencies) {
  if (numDependencies == nullptr ||
      rpc_write(conn, numDependencies, sizeof(*numDependencies)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (*numDependencies == 0) {
    return CUDA_SUCCESS;
  }
  if (rpc_write(conn, dependencies, *numDependencies * sizeof(CUgraphNode)) <
      0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

static CUkernel lupine_client_kernel_for_remote(CUkernel remote) {
  if (remote == nullptr ||
      lupine_library_kernel_count.load(std::memory_order_acquire) == 0) {
    return remote;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  for (const auto &entry : lupine_library_kernels()) {
    if (entry.first == remote) {
      continue;
    }
    for (const auto &route_entry : entry.second.kernels_by_route) {
      if (route_entry.second == remote) {
        return entry.first;
      }
    }
  }
  return remote;
}

static CUfunction lupine_client_function_for_remote(CUfunction remote) {
  if (remote == nullptr) {
    return nullptr;
  }
  CUkernel mapped =
      lupine_client_kernel_for_remote(reinterpret_cast<CUkernel>(remote));
  if (reinterpret_cast<CUfunction>(mapped) != remote) {
    return reinterpret_cast<CUfunction>(mapped);
  }
  if (lupine_private_node_count.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    for (const auto &entry : lupine_private_node_map()) {
      if (entry.first != remote && entry.second == remote) {
        return entry.first;
      }
    }
  }
  return remote;
}

extern "C" CUresult
cuGraphKernelNodeGetParams_v2(CUgraphNode hNode,
                              CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_graph_node(hNode);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuGraphKernelNodeGetParams_v2", hNode,
                                    nodeParams);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  CUresult param_info_result = CUDA_SUCCESS;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphKernelNodeGetParams_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read_buffer(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read(conn, nodeParams, sizeof(*nodeParams)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  nodeParams->kernelParams = nullptr;
  nodeParams->extra = nullptr;

  static thread_local std::vector<unsigned char> staging_data;
  staging_data.clear();
  lupine_inline_vec<size_t, 64> offsets;
  size_t data_bytes = 0;
  if (return_value == CUDA_SUCCESS) {
    constexpr size_t kAlign = alignof(std::max_align_t);
    for (;;) {
      size_t param_size = 0;
      CUresult param_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
      if (rpc_read_buffer(conn, &param_size, sizeof(param_size)) < 0 ||
          rpc_read_buffer(conn, &param_result, sizeof(param_result)) < 0) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      if (param_result != CUDA_SUCCESS) {
        param_info_result = param_result == CUDA_ERROR_INVALID_VALUE
                                ? CUDA_SUCCESS
                                : param_result;
        break;
      }
      size_t aligned_offset = (data_bytes + kAlign - 1) & ~(kAlign - 1);
      size_t alloc_size = std::max<size_t>(param_size, 1);
      if (aligned_offset > SIZE_MAX - alloc_size) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      data_bytes = aligned_offset + alloc_size;
      try {
        staging_data.resize(data_bytes);
      } catch (const std::bad_alloc &) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      if (param_size != 0 &&
          rpc_read(conn, staging_data.data() + aligned_offset, param_size) <
              0) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      if (!offsets.push(aligned_offset)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  if (rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  if (param_info_result != CUDA_SUCCESS) {
    return param_info_result;
  }

  if (nodeParams->func != nullptr) {
    nodeParams->func = lupine_client_function_for_remote(nodeParams->func);
  }
#if CUDA_VERSION >= 12000
  if (nodeParams->kern != nullptr) {
    nodeParams->kern = lupine_client_kernel_for_remote(nodeParams->kern);
  }
#endif

  static constexpr size_t kDeepSlotPtrs = 0x1000;
  static constexpr size_t kDeepSlotData = 0x1001;
  void **params_out = nullptr;
  if (offsets.count != 0) {
    std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
    try {
      params_out = static_cast<void **>(lupine_deep_node_cache_get_locked(
          hNode, kDeepSlotPtrs, offsets.count * sizeof(void *)));
      auto *base = static_cast<unsigned char *>(
          lupine_deep_node_cache_get_locked(hNode, kDeepSlotData, data_bytes));
      std::memcpy(base, staging_data.data(), data_bytes);
      for (size_t i = 0; i < offsets.count; ++i) {
        params_out[i] = base + offsets.data()[i];
      }
    } catch (const std::bad_alloc &) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  nodeParams->kernelParams = params_out;
  return return_value;
}

template <typename KernelNodeParams>
static CUresult lupine_prepare_kernel_node_params(
    lupine_route route, const KernelNodeParams &node_params,
    std::shared_ptr<const lupine_param_layout> *param_layout) {
  CUresult result = lupine_kernel_node_param_layout(node_params, param_layout);
  if (result == CUDA_SUCCESS) {
    result = lupine_prepare_kernel_params(route, node_params.kernelParams,
                                          (*param_layout)->sizes);
  }
  return result;
}

static CUresult lupine_graph_kernel_node_set_params(
    lupine_route route, CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_KERNEL_NODE_PARAMS *nodeParams, bool exec_node) {
  std::shared_ptr<const lupine_param_layout> param_layout;
  CUresult return_value =
      lupine_prepare_kernel_node_params(route, *nodeParams, &param_layout);
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint32_t param_count = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(
          conn, exec_node ? RPC_cuGraphExecKernelNodeSetParams_v2
                          : RPC_cuGraphKernelNodeSetParams_v2) < 0 ||
      (exec_node && rpc_write(conn, &hGraphExec, sizeof(hGraphExec)) < 0) ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      lupine_write_kernel_params(conn, nodeParams->kernelParams,
                                 param_layout->sizes, &param_count) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuGraphKernelNodeSetParams_v2(CUgraphNode hNode,
                              const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_graph_node(hNode);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuGraphKernelNodeSetParams_v2", hNode,
                                    nodeParams);
  }
  return lupine_graph_kernel_node_set_params(route, nullptr, hNode, nodeParams,
                                             false);
}

static CUresult lupine_finish_graph_add_node(conn_t *conn, lupine_route route,
                                             CUgraphNode *phGraphNode) {
  CUresult res = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &res, sizeof(res)) < 0 || rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (res == CUDA_SUCCESS)
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  return res;
}

extern "C" CUresult
cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                        const CUgraphNode *dependencies, size_t numDependencies,
                        const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (!phGraphNode || !nodeParams)
    return CUDA_ERROR_INVALID_VALUE;
  if (CUresult s =
          lupine_validate_graph_dependencies(dependencies, numDependencies))
    return s;
  lupine_route route = lupine_route_for_graph(hGraph);
  if (lupine_route_is_local(route)) {
    CUresult r =
        lupine_call_real_cuda_fn("cuGraphAddKernelNode_v2", phGraphNode, hGraph,
                                 dependencies, numDependencies, nodeParams);
    if (r == CUDA_SUCCESS)
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    return r;
  }
  std::shared_ptr<const lupine_param_layout> param_layout;
  if (CUresult r =
          lupine_prepare_kernel_node_params(route, *nodeParams, &param_layout))
    return r;
  conn_t *conn = lupine_route_remote_conn(route);
  uint32_t param_count = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGraphAddKernelNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      lupine_write_kernel_params(conn, nodeParams->kernelParams,
                                 param_layout->sizes, &param_count) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return lupine_finish_graph_add_node(conn, route, phGraphNode);
}

extern "C" CUresult
cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
  if (!phGraphNode || !copyParams)
    return CUDA_ERROR_INVALID_VALUE;
  if (CUresult s =
          lupine_validate_graph_dependencies(dependencies, numDependencies))
    return s;

  // A host source travels as its copied rows alone.
  size_t host_src_bytes = 0;
  size_t src_slice = copyParams->srcHeight * copyParams->srcPitch;
  const char *src_rows =
      (const char *)copyParams->srcHost + copyParams->srcZ * src_slice +
      copyParams->srcY * copyParams->srcPitch + copyParams->srcXInBytes;
  if (copyParams->srcMemoryType == CU_MEMORYTYPE_HOST) {
    host_src_bytes =
        copyParams->WidthInBytes * copyParams->Height * copyParams->Depth;
    if (host_src_bytes != 0 && copyParams->srcHost == nullptr)
      return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  if (lupine_route_is_local(route)) {
    CUresult r = lupine_call_real_cuda_fn("cuGraphAddMemcpyNode", phGraphNode,
                                          hGraph, dependencies, numDependencies,
                                          copyParams, ctx);
    if (r == CUDA_SUCCESS)
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    return r;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGraphAddMemcpyNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, copyParams, sizeof(*copyParams)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0 ||
      (host_src_bytes != 0 &&
       rpc_write_pitched(conn, src_rows, copyParams->WidthInBytes,
                         copyParams->Height, copyParams->srcPitch,
                         copyParams->Depth, src_slice) < 0))
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return lupine_finish_graph_add_node(conn, route, phGraphNode);
}

extern "C" CUresult
cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuGraphExecKernelNodeSetParams_v2",
                                    hGraphExec, hNode, nodeParams);
  }
  return lupine_graph_kernel_node_set_params(route, hGraphExec, hNode,
                                             nodeParams, true);
}

extern "C" CUresult cuGraphAddNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      const CUgraphEdgeData *dependencyData,
                                      size_t numDependencies,
                                      CUgraphNodeParams *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    local_result = lupine_call_real_cuda_fn(
        "cuGraphAddNode_v2", phGraphNode, hGraph, dependencies, dependencyData,
        numDependencies, nodeParams);
    if (local_result == CUDA_SUCCESS) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
      if (nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL &&
          nodeParams->conditional.phGraph_out != nullptr) {
        for (unsigned int i = 0; i < nodeParams->conditional.size; ++i) {
          lupine_note_graph_owner_route(nodeParams->conditional.phGraph_out[i],
                                        route);
        }
      }
    }
    return local_result;
  }

  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

#if CUDA_VERSION >= 12020
  // Reuse the type-specific wrappers for variants with nested host pointers
  // or callbacks. They already provide the process-boundary serialization
  // that a shallow CUgraphNodeParams copy cannot.
  switch (nodeParams->type) {
  case CU_GRAPH_NODE_TYPE_MEMCPY:
    return cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies,
                                numDependencies, &nodeParams->memcpy.copyParams,
                                nodeParams->memcpy.copyCtx);
  case CU_GRAPH_NODE_TYPE_MEMSET: {
    CUDA_MEMSET_NODE_PARAMS params{};
    params.dst = nodeParams->memset.dst;
    params.pitch = nodeParams->memset.pitch;
    params.value = nodeParams->memset.value;
    params.elementSize = nodeParams->memset.elementSize;
    params.width = nodeParams->memset.width;
    params.height = nodeParams->memset.height;
#if CUDA_VERSION >= 13020
    CUcontext ctx = nodeParams->memset.ctx;
#else
    CUcontext ctx = nullptr;
#endif
    return cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies,
                                numDependencies, &params, ctx);
  }
  case CU_GRAPH_NODE_TYPE_HOST: {
    CUDA_HOST_NODE_PARAMS params{};
    params.fn = nodeParams->host.fn;
    params.userData = nodeParams->host.userData;
    return cuGraphAddHostNode(phGraphNode, hGraph, dependencies,
                              numDependencies, &params);
  }
  default:
    break;
  }
#endif

  if (nodeParams->type != CU_GRAPH_NODE_TYPE_KERNEL &&
      nodeParams->type != CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  bool kernel = nodeParams->type == CU_GRAPH_NODE_TYPE_KERNEL;
  std::shared_ptr<const lupine_param_layout> param_layout;
  CUresult return_value = kernel ? lupine_prepare_kernel_node_params(
                                       route, nodeParams->kernel, &param_layout)
                                 : CUDA_SUCCESS;
  if (return_value != CUDA_SUCCESS) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  unsigned int child_count = kernel ? 0 : nodeParams->conditional.size;
  uint32_t param_count = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGraphAddNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (kernel &&
       lupine_write_kernel_params(conn, nodeParams->kernel.kernelParams,
                                  param_layout->sizes, &param_count) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  CUgraph *client_graphs = nullptr;
  if (return_value == CUDA_SUCCESS && child_count != 0) {
    client_graphs = static_cast<CUgraph *>(lupine_deep_node_cache_get(
        *phGraphNode, 0x1002, child_count * sizeof(CUgraph)));
    if (client_graphs == nullptr) {
      (void)rpc_read_end(conn);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (rpc_read(conn, client_graphs, child_count * sizeof(CUgraph)) < 0) {
      lupine_deep_node_cache_reset(*phGraphNode);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    nodeParams->conditional.phGraph_out = client_graphs;
  }
  if (rpc_read_end(conn) < 0) {
    if (client_graphs != nullptr) {
      lupine_deep_node_cache_reset(*phGraphNode);
    }
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
    for (unsigned int i = 0; i < child_count; ++i) {
      lupine_note_graph_owner_route(client_graphs[i], route);
    }
  }
  return return_value;
}

#ifdef cuGraphAddNode
#undef cuGraphAddNode
#endif
#if CUDA_VERSION >= 13000
extern "C" CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   const CUgraphEdgeData *dependencyData,
                                   size_t numDependencies,
                                   CUgraphNodeParams *nodeParams) {
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData,
                           numDependencies, nodeParams);
}
#else
extern "C" CUresult cuGraphAddNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   size_t numDependencies,
                                   CUgraphNodeParams *nodeParams) {
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, nullptr,
                           numDependencies, nodeParams);
}
#endif

// CUDA 12.3 remaps these source names to generated edge-data v2 wrappers.
// Keep the literal legacy ABI symbols for older binaries.
#ifdef cuGraphGetEdges
#undef cuGraphGetEdges
#endif
extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                    CUgraphNode *to, size_t *numEdges) {
  return cuGraphGetEdges_v2(hGraph, from, to, nullptr, numEdges);
}

#ifdef cuGraphNodeGetDependencies
#undef cuGraphNodeGetDependencies
#endif
#ifdef cuGraphNodeGetDependentNodes
#undef cuGraphNodeGetDependentNodes
#endif
extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode,
                                               CUgraphNode *dependencies,
                                               size_t *numDependencies) {
  return cuGraphNodeGetDependencies_v2(hNode, dependencies, nullptr,
                                       numDependencies);
}

extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                                 CUgraphNode *dependentNodes,
                                                 size_t *numDependentNodes) {
  return cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, nullptr,
                                         numDependentNodes);
}

#ifdef cuGraphAddDependencies
#undef cuGraphAddDependencies
#endif
#ifdef cuGraphRemoveDependencies
#undef cuGraphRemoveDependencies
#endif
extern "C" CUresult cuGraphAddDependencies(CUgraph hGraph,
                                           const CUgraphNode *from,
                                           const CUgraphNode *to,
                                           size_t numDependencies) {
  return cuGraphAddDependencies_v2(hGraph, from, to, nullptr, numDependencies);
}

extern "C" CUresult cuGraphRemoveDependencies(CUgraph hGraph,
                                              const CUgraphNode *from,
                                              const CUgraphNode *to,
                                              size_t numDependencies) {
  return cuGraphRemoveDependencies_v2(hGraph, from, to, nullptr,
                                      numDependencies);
}

#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
extern "C" CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                       unsigned long long flags) {
  return cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
}

static pending_log_callbacks &lupine_pending_logs() {
  return lupine_leaked<pending_log_callbacks>([] {});
}

static void lupine_complete_pending_log_callbacks(conn_t *conn,
                                                  int32_t stream) {
  lupine_pending_logs().complete(conn, stream);
}

static void lupine_discard_pending_log_callbacks(conn_t *conn) {
  lupine_pending_logs().discard(conn);
}

#if CUDA_VERSION >= 12090
static std::mutex &lupine_logs_callback_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

struct lupine_logs_callback_route {
  lupine_route route;
  CUlogsCallbackHandle server_callback = nullptr;
};

static std::unordered_map<CUlogsCallbackHandle, lupine_logs_callback_route> &
lupine_logs_callback_routes() {
  return lupine_leaked<
      std::unordered_map<CUlogsCallbackHandle, lupine_logs_callback_route>>(
      [] {});
}

extern "C" CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc,
                                           void *userData,
                                           CUlogsCallbackHandle *callback_out) {
  if (callbackFunc == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto *client_callback =
      callback_out == nullptr ? nullptr : new (std::nothrow) unsigned char;
  if (callback_out != nullptr && client_callback == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  lupine_route route = lupine_route_for_default();
  CUlogsCallbackHandle server_callback = nullptr;
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        "cuLogsRegisterCallback", callbackFunc, userData,
        callback_out == nullptr ? nullptr : &server_callback);
  } else if (!lupine_rpc_call(lupine_route_remote_conn(route),
                              RPC_cuLogsRegisterCallback,
                              std::tie(callbackFunc, userData),
                              std::tie(server_callback, return_value))) {
    delete client_callback;
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS && callback_out != nullptr) {
    auto callback = reinterpret_cast<CUlogsCallbackHandle>(client_callback);
    std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
    lupine_logs_callback_routes()[callback] = {route, server_callback};
    *callback_out = callback;
  } else {
    delete client_callback;
  }
  return return_value;
}

extern "C" CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback) {
  lupine_route route = lupine_route_for_default();
  CUlogsCallbackHandle server_callback = callback;
  {
    std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
    auto it = lupine_logs_callback_routes().find(callback);
    if (it != lupine_logs_callback_routes().end()) {
      route = it->second.route;
      server_callback = it->second.server_callback;
    }
  }

  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn("cuLogsUnregisterCallback", server_callback);
  } else if (!lupine_rpc_call(
                 lupine_route_remote_conn(route), RPC_cuLogsUnregisterCallback,
                 std::tie(server_callback), std::tie(return_value))) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS) {
    bool owned_callback = false;
    {
      std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
      owned_callback = lupine_logs_callback_routes().erase(callback) != 0;
    }
    if (owned_callback) {
      delete reinterpret_cast<unsigned char *>(callback);
    }
  }
  return return_value;
}
#endif

// Include admitted capture starts, not just completed BeginCapture calls: a
// different thread may query after the server starts capture but before the
// initiating thread receives its reply.
std::atomic<int> lupine_active_stream_captures{0};

static bool lupine_stream_handle_is_known(CUstream hStream) {
  if (hStream == nullptr || hStream == CU_STREAM_LEGACY ||
      hStream == CU_STREAM_PER_THREAD) {
    return true;
  }
  return lupine_route_for_known_stream(hStream).kind != LUPINE_ROUTE_INVALID;
}

// A stream's capture id and graph are fixed from the moment it enters a
// capture (BeginCapture, or joining through an event wait) until that capture
// ends, including after it rejoins the origin; only the dependency set moves.
// The first ACTIVE reply per stream therefore answers later status/id/graph
// queries until any capture ends. An invalidated capture keeps answering
// ACTIVE here until EndCapture reports it. Entries carry the epoch current
// when their query was issued, so a reply that lands after a concurrent
// EndCapture cannot describe the next capture.
struct lupine_stream_capture_entry {
  uint64_t epoch;
  cuuint64_t id;
  CUgraph graph;
};
static std::atomic<uint64_t> lupine_stream_capture_epoch{0};

static auto &lupine_stream_capture_cache() {
  return lupine_leaked<
      lupine_tiny_cache<CUstream, lupine_stream_capture_entry>>([] {});
}

static CUresult lupine_cuStreamGetCaptureInfo(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  static thread_local std::vector<CUgraphNode> capture_dependencies;
  static thread_local std::vector<CUgraphEdgeData> capture_edge_data;
  lupine_route route = stream != nullptr ? lupine_route_for_stream(stream)
                                         : lupine_route_for_default();

#if CUDA_VERSION >= 12000
  {
    CUresult local_result;
    if (lupine_route_is_local(route)) {
      local_result = lupine_call_real_cuda_fn(
          "cuStreamGetCaptureInfo_v3", stream, captureStatus_out, id_out,
          graph_out, dependencies_out, edgeData_out, numDependencies_out);
      if (local_result == CUDA_SUCCESS) {
        if (graph_out != nullptr) {
          lupine_note_graph_owner_route(*graph_out, route);
        }
        if (dependencies_out != nullptr && *dependencies_out != nullptr &&
            numDependencies_out != nullptr) {
          for (size_t i = 0; i < *numDependencies_out; ++i) {
            lupine_note_graph_node_owner_route((*dependencies_out)[i], route);
          }
        }
      }
      return local_result;
    }
  }
#else
  {
    if (lupine_route_is_local(route)) {
      return lupine_call_real_cuda_fn("cuStreamGetCaptureInfo", stream,
                                      captureStatus_out, id_out);
    }
  }
#endif

  // Like cuStreamIsCapturing, this query needs no server state when no capture
  // is outstanding. Optional outputs are only defined during active capture.
  // Keep unknown streams and uninitialized/detached contexts on the RPC path.
  if (captureStatus_out != nullptr && lupine_cuda_is_initialized() &&
      lupine_current_context != nullptr &&
      lupine_active_stream_captures.load() == 0 &&
      lupine_stream_handle_is_known(stream)) {
    *captureStatus_out = CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
  }

  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  uint64_t epoch = lupine_stream_capture_epoch.load();
  lupine_stream_capture_entry cached;
  if (dependencies_out == nullptr && edgeData_out == nullptr &&
      numDependencies_out == nullptr &&
      lupine_stream_capture_cache().find(stream, cached) &&
      cached.epoch == epoch) {
    if (captureStatus_out != nullptr) {
      *captureStatus_out = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    }
    if (id_out != nullptr) {
      *id_out = cached.id;
    }
    if (graph_out != nullptr) {
      *graph_out = cached.graph;
    }
    return CUDA_SUCCESS;
  }
  size_t dependency_count = 0;
  bool has_edge_data = false;
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, LUPINE_RPC_cuStreamGetCaptureInfo_v3,
                                 stream) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      !lupine_read_all(conn, status, id, graph, dependency_count,
                       has_edge_data)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  capture_dependencies.resize(dependency_count);
  if (dependency_count != 0 &&
      rpc_read(conn, capture_dependencies.data(),
               dependency_count * sizeof(CUgraphNode)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  capture_edge_data.clear();
  if (has_edge_data) {
    capture_edge_data.resize(dependency_count);
    if (dependency_count != 0 &&
        rpc_read(conn, capture_edge_data.data(),
                 dependency_count * sizeof(CUgraphEdgeData)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (captureStatus_out != nullptr) {
    *captureStatus_out = status;
  }
  if (id_out != nullptr) {
    *id_out = id;
  }
  if (graph_out != nullptr) {
    *graph_out = graph;
  }
  if (dependencies_out != nullptr) {
    *dependencies_out =
        capture_dependencies.empty() ? nullptr : capture_dependencies.data();
  }
  if (edgeData_out != nullptr) {
    *edgeData_out =
        capture_edge_data.empty() ? nullptr : capture_edge_data.data();
  }
  if (numDependencies_out != nullptr) {
    *numDependencies_out = dependency_count;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_owner_route(graph, route);
    for (CUgraphNode dependency : capture_dependencies) {
      lupine_note_graph_node_owner_route(dependency, route);
    }
    if (status == CU_STREAM_CAPTURE_STATUS_ACTIVE && stream != nullptr &&
        stream != CU_STREAM_LEGACY && stream != CU_STREAM_PER_THREAD) {
      lupine_stream_capture_cache().insert_or_assign(
          stream, lupine_stream_capture_entry{epoch, id, graph});
    }
  }
  return return_value;
}

extern "C" void lupine_stream_capture_begin() {
  lupine_checkpoint::capture_begin();
  lupine_active_stream_captures.fetch_add(1);
}

extern "C" void lupine_stream_capture_begin_complete(bool started) {
  lupine_checkpoint::capture_begin_complete(started);
  if (!started) {
    lupine_active_stream_captures.fetch_sub(1);
  }
}

extern "C" CUresult lupine_complete_stream_end_capture(CUresult result) {
  // CUDA_SUCCESS ends a valid capture. An invalidated or unjoined capture also
  // leaves capture mode when EndCapture reports the terminal error. Errors
  // such as WRONG_THREAD and UNMATCHED leave the tracked capture untouched.
  if (result == CUDA_SUCCESS ||
      result == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED ||
      result == CUDA_ERROR_STREAM_CAPTURE_UNJOINED) {
    lupine_checkpoint::capture_end();
    lupine_stream_capture_epoch.fetch_add(1);
    lupine_stream_capture_cache().clear();
    lupine_active_stream_captures.fetch_sub(1);
  }
  return result;
}

// Each connection serves this client thread on one lane thread, in order, so
// the swap only has to be queued ahead of the thread's next request, and the
// previous mode is known here: every thread starts GLOBAL on both sides.
// Every connection's lane gets it since any of them may serve that request.
extern "C" CUresult
cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  static thread_local CUstreamCaptureMode thread_mode =
      CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUstreamCaptureMode requested = *mode;
  if (lupine_local_cuda_available()) {
    CUresult result =
        lupine_call_real_cuda_fn("cuThreadExchangeStreamCaptureMode", mode);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  } else {
    *mode = thread_mode;
  }
  thread_mode = requested;
  if (rpc_open() != 0) {
    return CUDA_SUCCESS;
  }
  for (int i = 0; i < rpc_size(); ++i) {
    conn_t *conn = rpc_client_get_connection(i);
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuThreadExchangeStreamCaptureMode) <
            0 ||
        rpc_write(conn, &requested, sizeof(requested)) < 0 ||
        rpc_write_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  return CUDA_SUCCESS;
}

bool lupine_try_cached_stream_is_capturing(
    CUstream hStream, CUstreamCaptureStatus *captureStatus) {
  // Libraries poll capture state on the launch path. While this client has no
  // capture outstanding, a stream it knows cannot be capturing, and a stream
  // seen capturing stays so until a capture ends, so answer both without a
  // round trip. Unknown handles go to the server's driver instead.
  if (lupine_active_stream_captures.load() == 0 &&
      lupine_stream_handle_is_known(hStream)) {
    *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
    return true;
  }
  lupine_stream_capture_entry cached;
  if (lupine_stream_capture_cache().find(hStream, cached) &&
      cached.epoch == lupine_stream_capture_epoch.load()) {
    *captureStatus = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    return true;
  }
  return false;
}

extern "C" CUresult cuStreamGetCaptureInfo_v3(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  return lupine_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                       graph_out, dependencies_out,
                                       edgeData_out, numDependencies_out);
}

extern "C" CUresult cuStreamGetCaptureInfo_v2(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, size_t *numDependencies_out) {
  return lupine_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                       graph_out, dependencies_out, nullptr,
                                       numDependencies_out);
}

#ifdef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo
#endif
extern "C" CUresult
cuStreamGetCaptureInfo(CUstream stream,
                       CUstreamCaptureStatus *captureStatus_out,
                       cuuint64_t *id_out) {
  return lupine_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                       nullptr, nullptr, nullptr, nullptr);
}

extern "C" CUresult
cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph,
                            const CUgraphNode *dependencies,
                            const CUgraphEdgeData *dependencyData,
                            size_t numDependencies, CUstreamCaptureMode mode) {
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  lupine_materialize_host_allocations();
  lupine_capture_begin_guard capture_guard;
  lupine_route route =
      hGraph != nullptr ? lupine_route_for_graph(hGraph)
                        : (hStream != nullptr ? lupine_route_for_stream(hStream)
                                              : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return capture_guard.complete(lupine_call_real_cuda_fn(
        "cuStreamBeginCaptureToGraph", hStream, hGraph, dependencies,
        dependencyData, numDependencies, mode));
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_cuStreamBeginCaptureToGraph) <
          0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &mode, sizeof(mode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return capture_guard.complete(CUDA_ERROR_DEVICE_UNAVAILABLE);
  }
  return capture_guard.complete(return_value);
}

#ifdef cuStreamUpdateCaptureDependencies
#undef cuStreamUpdateCaptureDependencies
#endif
extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                      CUgraphNode *dependencies,
                                                      size_t numDependencies,
                                                      unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}

extern "C" CUresult cuStreamUpdateCaptureDependencies_ptsz(
    CUstream hStream, CUgraphNode *dependencies, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}

static bool lupine_uuid_equals(const CUuuid *uuid, const unsigned char *bytes) {
  return uuid != nullptr && memcmp(uuid->bytes, bytes, 16) == 0;
}

static CUresult lupine_cudart_load_cubin_ext(CUmodule *module,
                                             const void *fatbin, void *arg3,
                                             void *arg4, unsigned int arg5,
                                             const char *tag) {
  if (arg3 != nullptr || arg4 != nullptr || arg5 != 0) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuModuleLoadData(module, fatbin);
  LUPINE_TRACE_LOG("LUPINE cudart "
                   << tag << " result=" << static_cast<int>(result)
                   << " module=" << (module == nullptr ? nullptr : *module));
  return result;
}

extern "C" CUresult lupine_cudart_get_module_from_cubin(CUmodule *module,
                                                        const void *fatbin) {
  return lupine_cudart_load_cubin_ext(module, fatbin, nullptr, nullptr, 0,
                                      "get_module_from_cubin");
}

extern "C" CUresult lupine_cudart_get_primary_context(CUcontext *ctx,
                                                      CUdevice dev) {
  CUresult result = cuDevicePrimaryCtxRetain(ctx, dev);
  LUPINE_TRACE_LOG("LUPINE cudart get_primary_context dev="
                   << dev << " result=" << static_cast<int>(result)
                   << " ctx=" << (ctx == nullptr ? nullptr : *ctx));
  return result;
}

extern "C" CUresult
lupine_cudart_get_module_from_cubin_ext1(CUmodule *module, const void *fatbin,
                                         void *arg3, void *arg4,
                                         unsigned int arg5) {
  return lupine_cudart_load_cubin_ext(module, fatbin, arg3, arg4, arg5,
                                      "get_module_from_cubin_ext1");
}

extern "C" void lupine_cudart_noop_size_arg(size_t) {}
extern "C" uintptr_t lupine_dark_return_zero() { return 0; }
extern "C" uintptr_t lupine_dark_return_zero_1(const void *) { return 0; }
extern "C" uintptr_t lupine_dark_return_zero_2(const void *, const void *) {
  return 0;
}

extern "C" CUresult
lupine_cudart_get_module_from_cubin_ext2(const void *fatbin, CUmodule *module,
                                         void *arg3, void *arg4,
                                         unsigned int arg5) {
  return lupine_cudart_load_cubin_ext(module, fatbin, arg3, arg4, arg5,
                                      "get_module_from_cubin_ext2");
}

extern "C" CUresult lupine_cudart_load_compilers() { return CUDA_SUCCESS; }

template <size_t N>
static void lupine_dark_get_unknown_buffer(void **ptr, size_t *size) {
  static unsigned char buffer[N] = {};
  if (ptr != nullptr) {
    *ptr = buffer;
  }
  if (size != nullptr) {
    *size = sizeof(buffer);
  }
}

extern "C" void lupine_dark_get_unknown_buffer1(void **ptr, size_t *size) {
  lupine_dark_get_unknown_buffer<1024>(ptr, size);
}

extern "C" void lupine_dark_get_unknown_buffer2(void **ptr, size_t *size) {
  lupine_dark_get_unknown_buffer<14>(ptr, size);
}

using lupine_context_storage_dtor_t = void (*)(CUcontext context, void *key,
                                               void *value);

struct lupine_context_storage_value {
  void *value;
  lupine_context_storage_dtor_t dtor;
};

struct lupine_ctx_storage_t {
  std::atomic<size_t> count{0};
  std::mutex mu;
  std::unordered_map<CUcontext,
                     std::unordered_map<void *, lupine_context_storage_value>>
      map;
};

static auto &lupine_context_storage() {
  return lupine_leaked<lupine_ctx_storage_t>([] {});
}

// A destroyed context takes its context-local storage with it, and the server
// is free to hand the same handle back for the next context. NVIDIA's libcudart
// keeps its per-device runtime state here and re-initializes the device when
// the lookup misses, so an entry that outlives its context is the difference
// between a reset device coming back and staying broken.
static void lupine_forget_context_local_storage(CUcontext ctx) {
  auto &cs = lupine_context_storage();
  if (ctx == nullptr || cs.count.load(std::memory_order_acquire) == 0)
    return;
  std::lock_guard<std::mutex> lock(cs.mu);
  if (cs.map.erase(ctx) != 0)
    cs.count.store(cs.map.size(), std::memory_order_release);
}

static CUresult lupine_normalize_context(CUcontext *ctx) {
  if (ctx == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  // (CUcontext)-1 also means the calling thread's current context.
  if (*ctx != nullptr &&
      *ctx != reinterpret_cast<CUcontext>(~static_cast<uintptr_t>(0)))
    return CUDA_SUCCESS;
  *ctx = nullptr;
  return cuCtxGetCurrent(ctx);
}

static CUresult lupine_activate_context_local_storage_context(CUcontext ctx) {
  CUresult result = cuCtxSetCurrent(ctx);
  LUPINE_TRACE_LOG("LUPINE context storage activate ctx="
                   << ctx << " result=" << static_cast<int>(result));
  return result;
}

extern "C" CUresult
lupine_context_local_storage_put(CUcontext ctx, void *key, void *value,
                                 lupine_context_storage_dtor_t dtor) {
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS)
    return result;
  result = lupine_activate_context_local_storage_context(ctx);
  if (result != CUDA_SUCCESS)
    return result;
  {
    auto &cs = lupine_context_storage();
    std::lock_guard<std::mutex> lock(cs.mu);
    cs.map[ctx][key] = lupine_context_storage_value{value, dtor};
    cs.count.store(cs.map.size(), std::memory_order_release);
  }
  LUPINE_TRACE_LOG("LUPINE context storage put ctx=" << ctx << " key=" << key
                                                     << " value=" << value);
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_context_local_storage_delete(CUcontext ctx,
                                                        void *key) {
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS)
    return result;
  {
    auto &cs = lupine_context_storage();
    std::lock_guard<std::mutex> lock(cs.mu);
    auto it = cs.map.find(ctx);
    if (it == cs.map.end() || it->second.erase(key) == 0)
      return CUDA_ERROR_INVALID_HANDLE;
    if (it->second.empty()) {
      cs.map.erase(it);
      cs.count.store(cs.map.size(), std::memory_order_release);
    }
  }
  // The private context-local-storage destructor ABI is not stable enough to
  // invoke here; some CUDA libraries call explicit delete during their own
  // teardown and free the value themselves.
  LUPINE_TRACE_LOG("LUPINE context storage delete ctx=" << ctx
                                                        << " key=" << key);
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_context_local_storage_get(void **value,
                                                     CUcontext ctx, void *key) {
  if (value == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS)
    return result;
  bool found = false;
  if (auto &cs = lupine_context_storage();
      cs.count.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> lock(cs.mu);
    if (auto ctx_it = cs.map.find(ctx); ctx_it != cs.map.end())
      if (auto key_it = ctx_it->second.find(key);
          key_it != ctx_it->second.end())
        *value = key_it->second.value, found = true;
  }
  if (!found) {
    LUPINE_TRACE_LOG(
        "LUPINE context storage get missing key ctx=" << ctx << " key=" << key);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  LUPINE_TRACE_LOG("LUPINE context storage get ctx=" << ctx << " key=" << key
                                                     << " value=" << *value);
  return lupine_activate_context_local_storage_context(ctx);
}

extern "C" CUresult lupine_ctx_create_bypass(CUcontext *, unsigned int,
                                             CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_heap_alloc(const void **, size_t, size_t) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_heap_free(const void *, size_t *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_device_get_attribute_ext(CUdevice dev,
                                                    unsigned int attribute, int,
                                                    size_t (*result)[2]) {
  if (result == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int value = 0;
  CUresult status = cuDeviceGetAttribute(
      &value, static_cast<CUdevice_attribute>(attribute), dev);
  if (status == CUDA_SUCCESS) {
    (*result)[0] = static_cast<size_t>(value);
    (*result)[1] = 0;
  }
  return status;
}

extern "C" CUresult lupine_device_get_something(unsigned char *result,
                                                CUdevice) {
  if (result != nullptr) {
    *result = 0;
  }
  return CUDA_SUCCESS;
}

struct lupine_integrity_pass3_input {
  uint32_t driver_version;
  uint32_t version;
  uint32_t current_process;
  uint32_t current_thread;
  const void *cudart_table;
  const void *integrity_check_table;
  const void *fn_address;
  uint64_t unix_seconds;
};

struct lupine_integrity_device_hash_info {
  CUuuid guid;
  int32_t pci_domain;
  int32_t pci_bus;
  int32_t pci_device;
};

static void lupine_integrity_single_pass(unsigned char state[66],
                                         unsigned char byte) {
  static constexpr unsigned char MIXING_TABLE[256] = {
      0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1,
      0xEC, 0xF0, 0x06, 0x13, 0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C,
      0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA, 0x1E, 0x9B, 0x57, 0x3C,
      0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12,
      0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E,
      0xBB, 0x2F, 0xEE, 0x7A, 0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F,
      0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21, 0x80, 0x7F, 0x5D, 0x9A,
      0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03,
      0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7, 0x5E, 0x92, 0x2A,
      0xAC, 0x56, 0xAA, 0xC6, 0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6,
      0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1, 0x45, 0x9D, 0x70, 0x59,
      0x64, 0x71, 0x87, 0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02,
      0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69,
      0x34, 0x40, 0x7E, 0x0F, 0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A,
      0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26, 0x2C, 0x53, 0x0D, 0x6E,
      0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52,
      0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08,
      0x0C, 0xBD, 0xB1, 0x4A, 0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D,
      0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39, 0xF2, 0xEF, 0xB7, 0x0E,
      0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
      0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33,
      0x9F, 0x11, 0x83, 0x14};

  unsigned char index = state[0x40];
  state[index + 0x10] = byte;
  unsigned char next_index = (index + 1) & 0xf;
  state[index + 0x20] = state[index] ^ byte;
  unsigned char mixed = MIXING_TABLE[(byte ^ state[0x41]) & 0xff];
  unsigned char old = state[index + 0x30];
  state[index + 0x30] = mixed ^ old;
  state[0x41] = mixed ^ old;
  state[0x40] = next_index;
  if (next_index != 0) {
    return;
  }

  unsigned char rolling = 0x29;
  unsigned char round = 0;
  while (true) {
    rolling ^= state[0];
    state[0] = rolling;
    for (size_t i = 1; i < 0x30; ++i) {
      rolling = state[i] ^ MIXING_TABLE[rolling];
      state[i] = rolling;
    }
    rolling = static_cast<unsigned char>(rolling + round);
    round = static_cast<unsigned char>(round + 1);
    if (round == 0x12) {
      break;
    }
    rolling = MIXING_TABLE[rolling];
  }
}

static void lupine_integrity_hash_pass(unsigned char state[66],
                                       const void *data, size_t size,
                                       unsigned char xor_mask) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < size; ++i) {
    lupine_integrity_single_pass(state, bytes[i] ^ xor_mask);
  }
}

static void lupine_integrity_zero_result(unsigned char state[66]) {
  memset(state, 0, 16);
  memset(state + 48, 0, 18);
}

static void lupine_integrity_pass5(unsigned char state[66],
                                   uint64_t result[2]) {
  unsigned char temp = static_cast<unsigned char>(16 - state[64]);
  for (unsigned char i = 0; i < temp; ++i) {
    lupine_integrity_single_pass(state, temp);
  }
  for (size_t i = 0x30; i < 0x40; ++i) {
    lupine_integrity_single_pass(state, state[i]);
  }
  memcpy(&result[0], state, sizeof(uint64_t));
  memcpy(&result[1], state + sizeof(uint64_t), sizeof(uint64_t));
}

static CUresult lupine_get_device_hash_info(
    std::vector<lupine_integrity_device_hash_info> *devices) {
  if (devices == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int count = 0;
  CUresult status = cuDeviceGetCount(&count);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  devices->clear();
  devices->reserve(count);
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = 0;
    lupine_integrity_device_hash_info info = {};
    for (CUresult s :
         {cuDeviceGet(&device, ordinal), cuDeviceGetUuid_v2(&info.guid, device),
          cuDeviceGetAttribute(&info.pci_domain,
                               CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device),
          cuDeviceGetAttribute(&info.pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                               device),
          cuDeviceGetAttribute(&info.pci_device,
                               CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device)}) {
      if (s != CUDA_SUCCESS) {
        return s;
      }
    }
    devices->push_back(info);
  }
  return CUDA_SUCCESS;
}

static const void *lupine_get_cudart_export_table();
static const void *lupine_get_integrity_check_table();

extern "C" CUresult lupine_integrity_check(unsigned int version,
                                           uint64_t unix_seconds,
                                           uint64_t (*result)[2]) {
  if (result == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (version % 10 == 0) {
    (*result)[0] = 0x3341181c03cb675cULL;
    (*result)[1] = 0x8ed383aa1f4cd1e8ULL;
    return CUDA_SUCCESS;
  }
  if (version % 10 == 1) {
    (*result)[0] = 0x1841181c03cb675cULL;
    (*result)[1] = 0x8ed383aa1f4cd1e8ULL;
    return CUDA_SUCCESS;
  }

  int driver_version = 0;
  CUresult status = cuDriverGetVersion(&driver_version);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  std::vector<lupine_integrity_device_hash_info> devices;
  status = lupine_get_device_hash_info(&devices);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  static constexpr unsigned char pass1_result[16] = {
      0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
      0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F};

  unsigned char state[66] = {};
  lupine_integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x36);

#ifdef _WIN32
  uint32_t process_id = static_cast<uint32_t>(lupine_process_id());
  uint32_t thread_id = static_cast<uint32_t>(lupine_thread_id());
#else
  uint32_t process_id = static_cast<uint32_t>(getpid());
  uint32_t thread_id =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self()));
#ifdef __APPLE__
  // A translated Linux runtime can carry its own pthread descriptor. The
  // integrity exchange must use the caller's thread identity on both sides.
  Dl_info caller_info = {};
  if (dladdr(__builtin_return_address(0), &caller_info) != 0 &&
      caller_info.dli_fname != nullptr) {
    void *caller = dlopen(caller_info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    if (caller != nullptr) {
      using caller_pthread_self_t = uintptr_t (*)();
      auto caller_pthread_self = reinterpret_cast<caller_pthread_self_t>(
          lupine_real_dlsym(caller, "pthread_self"));
      if (caller_pthread_self != nullptr) {
        thread_id = static_cast<uint32_t>(caller_pthread_self());
      }
      dlclose(caller);
    }
  }
#endif
#endif
  LUPINE_TRACE_LOG("LUPINE integrity version="
                   << version << " seconds=" << unix_seconds
                   << " pid=" << process_id << " thread=" << thread_id);
  lupine_integrity_pass3_input input = {
      static_cast<uint32_t>(driver_version),
      version,
      process_id,
      thread_id,
      lupine_get_cudart_export_table(),
      lupine_get_integrity_check_table(),
      reinterpret_cast<const void *>(&lupine_integrity_check),
      unix_seconds,
  };
  lupine_integrity_hash_pass(state, &input, sizeof(input), 0);
  for (const auto &device : devices) {
    lupine_integrity_hash_pass(state, &device, sizeof(device), 0);
  }

  uint64_t pass5_result[2] = {};
  lupine_integrity_pass5(state, pass5_result);
  lupine_integrity_zero_result(state);
  lupine_integrity_hash_pass(state, pass1_result, sizeof(pass1_result), 0x5c);
  lupine_integrity_hash_pass(state, pass5_result, sizeof(pass5_result), 0);
  lupine_integrity_pass5(state, *result);
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_context_check_unsupported(CUcontext, unsigned int *,
                                                     const void **) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_context_check(CUcontext ctx, unsigned int *result1,
                                         const void **result2) {
  if (result1 != nullptr) {
    *result1 = lupine_context_is_green(ctx) ? 1 : 0;
  }
  (void)result2;
  return CUDA_SUCCESS;
}

extern "C" unsigned int lupine_context_check_fn3() { return 0; }

static const void *lupine_get_cudart_export_table() {
  static const void *table[13] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_cudart_get_module_from_cubin),
      reinterpret_cast<const void *>(&lupine_cudart_get_primary_context),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<const void *>(&lupine_cudart_get_module_from_cubin_ext1),
      reinterpret_cast<const void *>(&lupine_cudart_noop_size_arg),
      reinterpret_cast<const void *>(&lupine_cudart_get_module_from_cubin_ext2),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<const void *>(&lupine_cudart_load_compilers),
  };
  return table;
}

static const void *lupine_get_integrity_check_table() {
  static const void *table[3] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_integrity_check),
      nullptr,
  };
  return table;
}

extern "C" CUresult cuGetExportTable(const void **ppExportTable,
                                     const CUuuid *pExportTableId) {
  static const void *kToolsTls[4] = {
      reinterpret_cast<const void *>(sizeof(kToolsTls)),
      reinterpret_cast<const void *>(&lupine_dark_return_zero),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_1),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_1)};
  static const void *kToolsCbHooks[7] = {
      reinterpret_cast<const void *>(sizeof(kToolsCbHooks)),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_get_unknown_buffer1),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_get_unknown_buffer2)};
  static const void *kCtxStorage[4] = {
      reinterpret_cast<const void *>(&lupine_context_local_storage_put),
      reinterpret_cast<const void *>(&lupine_context_local_storage_delete),
      reinterpret_cast<const void *>(&lupine_context_local_storage_get),
      nullptr};
  static const void *kCtxBypass[2] = {
      reinterpret_cast<const void *>(sizeof(kCtxBypass)),
      reinterpret_cast<const void *>(&lupine_ctx_create_bypass)};
  static const void *kHeapAccess[3] = {
      reinterpret_cast<const void *>(sizeof(kHeapAccess)),
      reinterpret_cast<const void *>(&lupine_heap_alloc),
      reinterpret_cast<const void *>(&lupine_heap_free)};
  static const auto kDevExtRt = [] {
    std::array<const void *, 26> t{};
    t[0] = reinterpret_cast<const void *>(sizeof(t));
    t[5] = reinterpret_cast<const void *>(&lupine_device_get_attribute_ext);
    t[13] = reinterpret_cast<const void *>(&lupine_device_get_something);
    return t;
  }();
  static const auto kCtxChecks = [] {
    std::array<const void *, 17> t{};
    t[0] = reinterpret_cast<const void *>(sizeof(t));
    t[1] = reinterpret_cast<const void *>(&lupine_context_check_unsupported);
    t[2] = reinterpret_cast<const void *>(&lupine_context_check);
    t[3] = reinterpret_cast<const void *>(&lupine_context_check_fn3);
    return t;
  }();
  struct export_entry {
    unsigned char uuid[16];
    const void *table;
  };
  const export_entry kKnownTables[] = {
      {{0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a, 0x89, 0x87, 0xd9, 0x39,
        0x12, 0xfd, 0x9d, 0xf9},
       lupine_get_cudart_export_table()},
      {{0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47, 0x82, 0x98, 0xf6, 0xe7,
        0x8a, 0x3a, 0xec, 0xdc},
       kToolsTls},
      {{0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74, 0x93, 0xf2, 0x08, 0x00,
        0x20, 0x0c, 0x0a, 0x66},
       kToolsCbHooks},
      {{0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11, 0xa8, 0xc3, 0x68, 0xf3,
        0x55, 0xd8, 0x95, 0x93},
       kCtxStorage},
      {{0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a, 0x89, 0xa7, 0xd0, 0xdf,
        0x10, 0xe7, 0x72, 0x86},
       kCtxBypass},
      {{0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a, 0xac, 0xc5, 0x1d, 0x29,
        0xce, 0xa6, 0x31, 0xae},
       kHeapAccess},
      {{0xb1, 0x05, 0x41, 0xe1, 0xf7, 0xc7, 0xc7, 0x4a, 0x9f, 0x64, 0xf2, 0x23,
        0xbe, 0x99, 0xf1, 0xe2},
       kDevExtRt.data()},
      {{0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b, 0x8d, 0x34, 0xba, 0x12,
        0x3c, 0x66, 0xe1, 0xf2},
       lupine_get_integrity_check_table()},
      {{0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43, 0x92, 0xf6, 0xbb, 0xd5,
        0x00, 0x6d, 0xfa, 0x7e},
       kCtxChecks.data()},
  };

  if (pExportTableId != nullptr) {
    LUPINE_TRACE_LOG("LUPINE cuGetExportTable requested UUID: "
                     << lupine_uuid_hex(pExportTableId));
  }
  if (ppExportTable != nullptr) {
    *ppExportTable = nullptr;
  }
  if (ppExportTable == nullptr || pExportTableId == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (const auto &entry : kKnownTables) {
    if (lupine_uuid_equals(pExportTableId, entry.uuid)) {
      *ppExportTable = entry.table;
      return CUDA_SUCCESS;
    }
  }
  if (const void *private_table =
          lupine_remote_private_export_table(pExportTableId)) {
    *ppExportTable = private_table;
    return CUDA_SUCCESS;
  }
  if (lupine_stub_private_exports_enabled()) {
    if (const void *private_table =
            lupine_private_export_table_from_env(pExportTableId)) {
      *ppExportTable = private_table;
      return CUDA_SUCCESS;
    }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

static void *lupine_make_missing_stub(const char *symbol) {
  if (symbol == nullptr) {
    return nullptr;
  }

#if defined(__x86_64__) || defined(__aarch64__) || defined(_M_X64)
  static std::unordered_map<std::string, void *> stubs;
  auto existing = stubs.find(symbol);
  if (existing != stubs.end()) {
    return existing->second;
  }

#if defined(__x86_64__) || defined(_M_X64)
  constexpr size_t stub_size = 22;
#else
  constexpr size_t stub_words = 9;
  constexpr size_t stub_size = stub_words * sizeof(uint32_t);
#endif
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }

  char *stable_symbol = strdup(symbol);
  void *handler = reinterpret_cast<void *>(&lupine_missing_driver_api_called);
#if defined(_WIN32) || defined(__x86_64__)
  // Microsoft x64 passes the first argument in rcx (0xb9), SysV x86-64 in rdi
  // (0xbf).
  code[0] = 0x48;
#if defined(_WIN32)
  code[1] = 0xb9;
#else
  code[1] = 0xbf;
#endif
  memcpy(code + 2, &stable_symbol, sizeof(stable_symbol));
  code[10] = 0x48;
  code[11] = 0xb8;
  memcpy(code + 12, &handler, sizeof(handler));
  code[20] = 0xff;
  code[21] = 0xe0;
#else
  // AAPCS64 mirror of the x86-64 stub above: overwrite the first argument
  // register with the symbol name and tail-branch to the handler, discarding
  // whatever the caller passed.
  //
  //   stub(...) -> lupine_missing_driver_api_called(stable_symbol)
  uint32_t words[stub_words];
  // x0 = stable_symbol
  lupine_a64_emit_mov_imm64(&words[0], 0,
                            reinterpret_cast<uint64_t>(stable_symbol));
  // x16 = handler (IP0 scratch, free to clobber across the tail branch)
  lupine_a64_emit_mov_imm64(&words[4], 16, reinterpret_cast<uint64_t>(handler));
  words[8] = 0xd61f0200u; // br x16
  memcpy(code, words, stub_size);
  // Mandatory on AArch64: make the freshly written words visible to the
  // instruction fetcher.
  __builtin___clear_cache(reinterpret_cast<char *>(code),
                          reinterpret_cast<char *>(code + stub_size));
#endif
  stubs[symbol] = code;
  return code;
#else
  return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
#endif
}

#define LUPINE_DEFINE_UNSUPPORTED_STUB(name)                                   \
  extern "C" CUresult lupine_unsupported_##name() {                            \
    LUPINE_LOG_ERROR("LUPINE unsupported Driver API called: " #name);          \
    return CUDA_ERROR_NOT_SUPPORTED;                                           \
  }

LUPINE_DEFINE_UNSUPPORTED_STUB(cuUserObjectCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceRegisterAsyncNotification)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceUnregisterAsyncNotification)
#if CUDA_VERSION < 12090
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsRegisterCallback)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsUnregisterCallback)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsCurrent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToFile)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToMemory)
#endif
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetHostAtomicCapabilities)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetP2PAtomicCapabilities)

bool lupine_try_cached_kernel_library(CUkernel kernel, CUlibrary *pLib) {
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto it = lupine_library_kernels().find(kernel);
  if (it != lupine_library_kernels().end() && it->second.library != nullptr) {
    *pLib = it->second.library;
    return true;
  }
  return false;
}

static void *lupine_get_unsupported_stub(const char *symbol) {
  // clang-format off
  static const std::unordered_map<std::string, void *> stubs = {
#define LUPINE_STUB_ENTRY(name)                                                \
  { #name, (void *)&lupine_unsupported_##name }
      LUPINE_STUB_ENTRY(cuUserObjectCreate),
      LUPINE_STUB_ENTRY(cuDeviceRegisterAsyncNotification),
      LUPINE_STUB_ENTRY(cuDeviceUnregisterAsyncNotification),
#if CUDA_VERSION < 12090
      LUPINE_STUB_ENTRY(cuLogsRegisterCallback),
      LUPINE_STUB_ENTRY(cuLogsUnregisterCallback),
      LUPINE_STUB_ENTRY(cuLogsCurrent),
      LUPINE_STUB_ENTRY(cuLogsDumpToFile),
      LUPINE_STUB_ENTRY(cuLogsDumpToMemory),
#endif
      LUPINE_STUB_ENTRY(cuDeviceGetHostAtomicCapabilities),
      LUPINE_STUB_ENTRY(cuDeviceGetP2PAtomicCapabilities),
#undef LUPINE_STUB_ENTRY
  };
  // clang-format on
  if (symbol == nullptr) {
    return nullptr;
  }
  auto it = stubs.find(symbol);
  return it == stubs.end() ? nullptr : it->second;
}

void rpc_close(conn_t *conn) { lupine_client_transport_close_connection(conn); }

static void lupine_rpc_shutdown() { lupine_client_transport_close(); }

#ifdef _WIN32
static void lupine_rpc_destructor() { lupine_rpc_shutdown(); }
static const int lupine_rpc_destructor_registration =
    std::atexit(lupine_rpc_destructor);
#else
__attribute__((destructor)) static void lupine_rpc_destructor() {
  lupine_rpc_shutdown();
}
#endif

// Acknowledging a callback lets queued device work resume, so the host writes
// it made are applied on the server first, even when no application CUDA call
// follows. The requests that sends belong to the callback.
static int lupine_write_void_response(conn_t *conn, int request_id) {
  void *res = nullptr;
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &res, sizeof(res)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

static int lupine_answer_callback(conn_t *conn, int request_id) {
  rpc_begin_callback_reply(conn);
  int flushed =
      lupine_prepare_rpc(conn) < 0 || lupine_settle_host_flushes(conn) < 0 ? -1
                                                                           : 0;
  rpc_end_callback_reply();
  return flushed < 0 ? -1 : lupine_write_void_response(conn, request_id);
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  lupine_start_dtoh_receiver(conn);
  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == LUPINE_SIDE_EFFECT_HOST_FUNCTION) {
      // Device-to-host copies the callback follows land first.
      CUhostFn callback = nullptr;
      void *user_data = nullptr;
      if (lupine_read_dtoh_watermark(conn) < 0 ||
          !lupine_read_all(conn, callback, user_data)) {
        LUPINE_LOG_ERROR("Failed to read host callback request.");
        goto close_connection;
      }
      int request_id = rpc_read_end(conn);
      if (request_id < 0 || callback == nullptr) {
        LUPINE_LOG_ERROR("Invalid host callback request.");
        goto close_connection;
      }
      lupine_in_host_callback = true;
      (void)lupine_sync_mapped_device_to_host();
      callback(user_data);
      lupine_in_host_callback = false;
      if (lupine_answer_callback(conn, request_id) < 0) {
        LUPINE_LOG_ERROR("Failed to answer host callback.");
        goto close_connection;
      }
    } else if (op == LUPINE_SIDE_EFFECT_STREAM_CALLBACK) {
      CUstream stream = nullptr;
      CUresult status = CUDA_ERROR_UNKNOWN;
      CUstreamCallback callback = nullptr;
      void *user_data = nullptr;
      if (lupine_read_dtoh_watermark(conn) < 0 ||
          !lupine_read_all(conn, stream, status, callback, user_data)) {
        LUPINE_LOG_ERROR("Failed to read stream callback request.");
        break;
      }
      int request_id = rpc_read_end(conn);
      if (request_id < 0)
        break;
      if (callback != nullptr) {
        lupine_in_host_callback = true;
        (void)lupine_sync_mapped_device_to_host();
        callback(stream, status, user_data);
        lupine_in_host_callback = false;
      }
      if (lupine_answer_callback(conn, request_id) < 0) {
        LUPINE_LOG_ERROR("Failed to answer stream callback.");
        break;
      }
    } else if (op == LUPINE_SIDE_EFFECT_LOG_CALLBACK) {
#if CUDA_VERSION >= 12090
      CUlogLevel level = CU_LOG_LEVEL_ERROR;
      int32_t origin_stream_id = -1;
      size_t length = 0;
      CUlogsCallback callback = nullptr;
      void *user_data = nullptr;
      if (!lupine_read_all(conn, level, origin_stream_id, length)) {
        LUPINE_LOG_ERROR("Failed to read log callback metadata.");
        break;
      }

      char inline_msg[512];
      std::vector<char> heap_msg;
      char *msg_buf = inline_msg;
      if (length >= sizeof(inline_msg)) {
        try {
          if (length == SIZE_MAX) {
            throw std::bad_alloc();
          }
          heap_msg.resize(length + 1);
        } catch (...) {
          LUPINE_LOG_ERROR("Failed to allocate log callback message.");
          break;
        }
        msg_buf = heap_msg.data();
      }
      if ((length != 0 && rpc_read(conn, msg_buf, length) < 0) ||
          rpc_read(conn, &callback, sizeof(callback)) < 0 ||
          rpc_read(conn, &user_data, sizeof(user_data)) < 0) {
        LUPINE_LOG_ERROR("Failed to read log callback request.");
        break;
      }
      msg_buf[length] = '\0';

      int32_t callback_stream_id = rpc_current_http2_stream(conn);
      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        break;
      }

      // CUDA invokes log callbacks inline. Preserve that caller-thread
      // behavior for application RPC lanes because clients such as PyTorch
      // keep callback state in thread-local storage. Unbound callbacks retain
      // the dispatch-thread fallback.
      if ((origin_stream_id < 0 || origin_stream_id == callback_stream_id) &&
          callback != nullptr) {
        callback(user_data, level, msg_buf, length);
      } else if (callback != nullptr) {
        if (heap_msg.empty()) {
          try {
            heap_msg.assign(inline_msg, inline_msg + length + 1);
          } catch (...) {
            LUPINE_LOG_ERROR("Failed to allocate log callback message.");
            break;
          }
        }
        if (!lupine_pending_logs().enqueue(
                conn, origin_stream_id,
                [callback, user_data, level, length,
                 message = std::move(heap_msg)]() mutable {
                  callback(user_data, level, message.data(), length);
                })) {
          LUPINE_LOG_ERROR("Failed to queue log callback.");
          break;
        }
      }

      if (lupine_write_void_response(conn, request_id) < 0) {
        LUPINE_LOG_ERROR("Failed to acknowledge log callback.");
        break;
      }
#else
      LUPINE_LOG_ERROR("Received unsupported log callback request.");
      break;
#endif
    } else if (op == LUPINE_SIDE_EFFECT_READ_HOST_MEMORY) {
      struct host_read {
        const unsigned char *source = nullptr;
        size_t width = 0;
        size_t rows = 0;
        size_t row_stride = 0;
        size_t slices = 0;
        size_t slice_stride = 0;
        size_t logical_offset = 0;
        size_t bytes = 0;
      };
      host_read read;
      const void *data = nullptr;
      // Device-to-host copies the stream ran before this read land first.
      if (lupine_read_dtoh_watermark(conn) < 0 ||
          !lupine_read_all(conn, data, read.width, read.rows, read.row_stride,
                           read.slices, read.slice_stride, read.logical_offset,
                           read.bytes)) {
        LUPINE_LOG_ERROR("Failed to read host-memory side-effect request.");
        goto close_connection;
      }
      size_t span = read.width + (read.rows - 1) * read.row_stride +
                    (read.slices - 1) * read.slice_stride;
      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        LUPINE_LOG_ERROR("Invalid host-memory side-effect request.");
        break;
      }

      lupine_in_host_callback = true;
      int device_source = lupine_write_cross_route_device_source(
          conn, request_id,
          reinterpret_cast<CUdeviceptr>(data) + read.logical_offset,
          read.bytes);
      if (device_source <= 0) {
        lupine_in_host_callback = false;
      }
      if (device_source < 0) {
        LUPINE_LOG_ERROR("Failed to proxy cross-route device memory.");
        break;
      }
      if (device_source == 0) {
        continue;
      }

      read.source = static_cast<const unsigned char *>(
          lupine_mapped_host_read_source(data, span));
      lupine_in_host_callback = false;

      if (rpc_write_start_response(conn, request_id) < 0) {
        LUPINE_LOG_ERROR("Failed to return host-memory side effect.");
        break;
      }
      size_t offset = read.logical_offset;
      size_t remaining = read.bytes;
      while (remaining != 0) {
        size_t row_index = offset / read.width;
        size_t x = offset - row_index * read.width;
        size_t slice = row_index / read.rows;
        size_t row = row_index - slice * read.rows;
        size_t chunk = std::min(remaining, read.width - x);
        const void *source =
            read.source + slice * read.slice_stride + row * read.row_stride + x;
        if (rpc_write(conn, source, chunk) < 0) {
          LUPINE_LOG_ERROR("Failed to return host-memory side effect.");
          goto close_connection;
        }
        offset += chunk;
        remaining -= chunk;
      }
      if (rpc_write_end(conn) < 0) {
        LUPINE_LOG_ERROR("Failed to return host-memory side effect.");
        break;
      }
    } else if (op < 0 || conn->closed) {
      break;
    }
  }

  if (!conn->closed) {
    LUPINE_LOG_ERROR("Exiting dispatch thread due to an error.");
  }
close_connection:
  rpc_close(conn);
  lupine_client_transport_shutdown_bulk(conn);
  // The receiver's reads fail with the transport; the connection is not torn
  // down until this thread is joined.
  lupine_join_dtoh_receiver(conn);
  return nullptr;
}

int rpc_open() {
  if (pthread_once(&lupine_rpc_lifecycle_once,
                   lupine_install_rpc_lifecycle_hooks) != 0) {
    return -1;
  }
  return lupine_client_transport_open(lupine_cuda_transport_config());
}

conn_t *rpc_client_get_connection(unsigned int index) {
  if (rpc_open() < 0) {
    return nullptr;
  }
  return lupine_client_transport_connection(index);
}

int rpc_size() { return static_cast<int>(lupine_client_transport_size()); }

#if CUDA_VERSION >= 12000
bool lupine_tensor_map_cache_lookup(
    lupine_route route, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill, std::string *key, uint64_t *generation,
    CUtensorMap *tensorMap) {
  const size_t rank_bytes_u64 = tensorRank * sizeof(cuuint64_t);
  const size_t stride_bytes = (tensorRank - 1) * sizeof(cuuint64_t);
  const size_t rank_bytes_u32 = tensorRank * sizeof(cuuint32_t);
  auto append = [key](const auto &value) {
    key->append(reinterpret_cast<const char *>(&value), sizeof(value));
  };
  append(lupine_route_identity(route));
  append(tensorDataType);
  append(tensorRank);
  append(globalAddress);
  key->append(reinterpret_cast<const char *>(globalDim), rank_bytes_u64);
  if (stride_bytes != 0)
    key->append(reinterpret_cast<const char *>(globalStrides), stride_bytes);
  key->append(reinterpret_cast<const char *>(boxDim), rank_bytes_u32);
  key->append(reinterpret_cast<const char *>(elementStrides), rank_bytes_u32);
  append(interleave);
  append(swizzle);
  append(l2Promotion);
  append(oobFill);

  std::lock_guard<std::mutex> lock(lupine_tensor_map_mutex);
  auto cached = lupine_tensor_maps.find(*key);
  if (cached != lupine_tensor_maps.end()) {
    *tensorMap = cached->second;
    return true;
  }
  *generation = lupine_tensor_map_generation;
  return false;
}

void lupine_tensor_map_cache_insert(std::string key, uint64_t generation,
                                    const CUtensorMap &tensorMap) {
  std::lock_guard<std::mutex> lock(lupine_tensor_map_mutex);
  // Do not repopulate a cache cleared while this RPC was in flight.
  if (generation != lupine_tensor_map_generation)
    return;
  if (lupine_tensor_maps.size() >= kLupineTensorMapCacheLimit)
    lupine_tensor_maps.clear();
  lupine_tensor_maps.insert_or_assign(std::move(key), tensorMap);
}
#endif

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags);

static const std::unordered_map<std::string, void *> &
lupine_manual_function_map();

static void *lupine_lookup_shim_symbol(const char *symbol) {
  if (void *fn = get_function_pointer(symbol)) {
    return fn;
  }
  const auto &manual_map = lupine_manual_function_map();
  if (auto it = manual_map.find(symbol); it != manual_map.end()) {
    return it->second;
  }
  return lupine_get_unsupported_stub(symbol);
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  LUPINE_TRACE_LOG("cuGetProcAddress getting symbol: " << symbol);
  // Most wrappers route purely by symbol name. A few CUDA APIs changed ABI
  // without changing the name and must also use the requested API version.
  (void)flags;

  auto finish = [&](void *fn) {
    *pfn = fn;
    if (symbolStatus != nullptr) {
      *symbolStatus = fn != nullptr ? CU_GET_PROC_ADDRESS_SUCCESS
                                    : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return CUDA_SUCCESS;
  };
  auto match = [&](const char *name) {
    return symbol != nullptr && strcmp(symbol, name) == 0;
  };

  if (strcmp(symbol, "cuStreamGetCaptureInfo") == 0) {
    return finish(
        cudaVersion >= 12030
            ? reinterpret_cast<void *>(&cuStreamGetCaptureInfo_v3)
            : (cudaVersion >= 11030
                   ? reinterpret_cast<void *>(&cuStreamGetCaptureInfo_v2)
                   : reinterpret_cast<void *>(&cuStreamGetCaptureInfo)));
  }

  struct graph_query_entry {
    const char *name;
    void *legacy;
    void *edge_data;
  };
  static const graph_query_entry kGraphQueries[] = {
      {"cuGraphGetEdges", reinterpret_cast<void *>(&cuGraphGetEdges),
       reinterpret_cast<void *>(&cuGraphGetEdges_v2)},
      {"cuGraphNodeGetDependencies",
       reinterpret_cast<void *>(&cuGraphNodeGetDependencies),
       reinterpret_cast<void *>(&cuGraphNodeGetDependencies_v2)},
      {"cuGraphNodeGetDependentNodes",
       reinterpret_cast<void *>(&cuGraphNodeGetDependentNodes),
       reinterpret_cast<void *>(&cuGraphNodeGetDependentNodes_v2)},
      {"cuGraphAddDependencies",
       reinterpret_cast<void *>(&cuGraphAddDependencies),
       reinterpret_cast<void *>(&cuGraphAddDependencies_v2)},
      {"cuGraphRemoveDependencies",
       reinterpret_cast<void *>(&cuGraphRemoveDependencies),
       reinterpret_cast<void *>(&cuGraphRemoveDependencies_v2)},
      {"cuStreamUpdateCaptureDependencies",
       reinterpret_cast<void *>(&cuStreamUpdateCaptureDependencies),
       reinterpret_cast<void *>(&cuStreamUpdateCaptureDependencies_v2)},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       reinterpret_cast<void *>(&cuStreamUpdateCaptureDependencies_ptsz),
       reinterpret_cast<void *>(&cuStreamUpdateCaptureDependencies_v2)},
  };
  for (const auto &entry : kGraphQueries) {
    if (strcmp(symbol, entry.name) == 0) {
      return finish(cudaVersion >= 12030 ? entry.edge_data : entry.legacy);
    }
  }

#if CUDA_VERSION >= 13000
  if (cudaVersion >= 13000 && match("cuCtxGetDevice")) {
    return finish(reinterpret_cast<void *>(&cuCtxGetDevice_v2));
  }
  if (cudaVersion >= 13000 && match("cuCtxSynchronize")) {
    return finish(reinterpret_cast<void *>(&cuCtxSynchronize_v2));
  }
#endif
  if (cudaVersion < 12000 && match("cuGraphInstantiate")) {
    return finish(get_function_pointer("cuGraphInstantiate_v2"));
  }
  if (match("cuGraphExecUpdate")) {
    return finish(get_function_pointer(
        cudaVersion < 12000 ? "cuGraphExecUpdate" : "cuGraphExecUpdate_v2"));
  }
#if CUDA_VERSION >= 12020
  if (cudaVersion >= 12020 &&
      (match("cuMemPrefetchAsync") || match("cuMemPrefetchAsync_ptsz"))) {
    return finish(reinterpret_cast<void *>(&cuMemPrefetchAsync_v2));
  }
  if (cudaVersion >= 12020 && match("cuMemAdvise")) {
    return finish(reinterpret_cast<void *>(&cuMemAdvise_v2));
  }
#endif

  if (void *it = lupine_lookup_shim_symbol(symbol)) {
    LUPINE_TRACE_LOG("cuGetProcAddress: Mapped symbol '"
                     << symbol << "' to function: " << it);
    return finish(it);
  }
  if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    return finish(reinterpret_cast<void *>(&cuGetProcAddress_v2));
  }
  if (strcmp(symbol, "cuGetProcAddress") == 0) {
    return finish(cudaVersion >= 12000
                      ? reinterpret_cast<void *>(&cuGetProcAddress_v2)
                      : reinterpret_cast<void *>(&cuGetProcAddress));
  }

  LUPINE_TRACE_LOG("cuGetProcAddress: Symbol '"
                   << symbol << "' not found in cudaFunctionMap.");

  // fall back to the real loader before creating a local missing-symbol stub
#if defined(_WIN32)
  if (void *fn = lupine_real_cuda_symbol(symbol)) {
    return finish(fn);
  }
  void *libCudaHandle = lupine_local_libcuda_handle();
#else
  if (void *fn = lupine_real_dlsym(RTLD_DEFAULT, symbol)) {
    return finish(fn);
  }
#if defined(__APPLE__)
  void *libCudaHandle = dlopen("libcuda.dylib", RTLD_NOW | RTLD_GLOBAL);
#else
  void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
#endif
#endif
  if (libCudaHandle != nullptr) {
    if (void *fn = lupine_real_dlsym(libCudaHandle, symbol)) {
      return finish(fn);
    }
    LUPINE_TRACE_LOG("Error: Could not resolve symbol '" << symbol
                                                         << "' using dlsym.");
  }
  return finish((lupine_stub_missing_enabled() &&
                 lupine_symbol_looks_like_driver_api(symbol))
                    ? lupine_make_missing_stub(symbol)
                    : nullptr);
}

extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags) {
  // The v1 entry point predates symbolStatus, so the driver reports a missing
  // symbol as CUDA_ERROR_NOT_FOUND and leaves *pfn untouched, where v2 reports
  // CUDA_SUCCESS with CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND. Verified against
  // driver 590.48.01 (CUDA 13.1).
  void *resolved = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult result =
      cuGetProcAddress_v2(symbol, &resolved, cudaVersion, flags, &status);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND) {
    return CUDA_ERROR_NOT_FOUND;
  }
  *pfn = resolved;
  return CUDA_SUCCESS;
}

#if defined(__GLIBC__)
void *dlsym(void *handle, const char *name) noexcept {
  LUPINE_TRACE_LOG("dlsym: " << name);
  if (!lupine_symbol_looks_like_driver_api(name)) {
    return lupine_real_dlsym(handle, name);
  }
  /** proc address function calls are basically dlsym; we should handle this
   * differently at the top level. */
  if (strcmp(name, "cuGetProcAddress_v2") == 0) {
    return reinterpret_cast<void *>(&cuGetProcAddress_v2);
  }
  if (strcmp(name, "cuGetProcAddress") == 0) {
    return reinterpret_cast<void *>(&cuGetProcAddress);
  }
  if (void *func = lupine_lookup_shim_symbol(name)) {
    return func;
  }
  return lupine_stub_missing_enabled() ? lupine_make_missing_stub(name)
                                       : lupine_real_dlsym(handle, name);
}
#endif

extern "C" CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount);

static const std::unordered_map<std::string, void *> &
lupine_manual_function_map() {
  static const std::unordered_map<std::string, void *> manual_function_map = {
#if CUDA_VERSION >= 12050
      {"cuCtxCreate", (void *)cuCtxCreate_v4},
      {"cuCtxCreate_v4", (void *)cuCtxCreate_v4},
#else
      {"cuCtxCreate", (void *)cuCtxCreate_v2},
#endif
      {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuMemHostRegister", (void *)cuMemHostRegister},
      {"cuMemHostRegister_v2", (void *)cuMemHostRegister_v2},
      {"cuMemHostUnregister", (void *)cuMemHostUnregister},
      {"cuMemcpy_ptds", (void *)cuMemcpy_ptds},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoadFatBinary", (void *)cuModuleLoadFatBinary},
      {"cuLibraryLoadFromFile", (void *)cuLibraryLoadFromFile},
      {"cuLibraryGetKernelCount", (void *)cuLibraryGetKernelCount},
      {"cuLibraryEnumerateKernels", (void *)cuLibraryEnumerateKernels},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
#if CUDA_VERSION < 12030
      {"cuGraphGetEdges", (void *)cuGraphGetEdges},
      {"cuGraphNodeGetDependencies", (void *)cuGraphNodeGetDependencies},
      {"cuGraphNodeGetDependentNodes", (void *)cuGraphNodeGetDependentNodes},
      {"cuGraphAddDependencies", (void *)cuGraphAddDependencies},
      {"cuGraphRemoveDependencies", (void *)cuGraphRemoveDependencies},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
#endif
      {"cuGraphAddNode", (void *)cuGraphAddNode},
      {"cuGraphAddNode_v2", (void *)cuGraphAddNode_v2},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
  };
  return manual_function_map;
}
