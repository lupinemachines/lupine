#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cudaProfiler.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <features.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <strings.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(__GLIBC__)
#error "Lupine CUDA client requires glibc"
#endif

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include <cstdlib>
#include <cstring>

#include "cache.h"
#include "checkpoint.h"
#include "client_routing.h"
#include "codegen/gen_api.h"
#include "codegen/gen_client.h"
#include "events.h"
#include "ipc.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "memcpy.h"
#include "rpc.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;
static bool lupine_rpc_shutting_down = false;

void rpc_destroy_thread_lane(uint64_t lane_id) {
  conn_t *active_conns[sizeof(conns) / sizeof(conns[0])];
  int count = 0;

  if (pthread_mutex_lock(&conn_mutex) != 0) {
    return;
  }
  if (!lupine_rpc_shutting_down) {
    for (int i = 0; i < nconns; ++i) {
      if (!conns[i].closed) {
        active_conns[count++] = &conns[i];
      }
    }
  }
  pthread_mutex_unlock(&conn_mutex);

  for (int i = 0; i < count; ++i) {
    rpc_write_lane_termination(active_conns[i], lane_id);
  }
}

const char *DEFAULT_PORT = "14833";

void *rpc_client_dispatch_thread(void *arg);

struct lupine_server_endpoint {
  std::string host;
  std::string port;
  bool tls = false;
};

static CUresult lupine_remote_cuInit(conn_t *conn, unsigned int flags);

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;
struct lupine_private_node_mapping {
  CUfunction server_function = nullptr;
  uint64_t server_owner = 0;
  CUmodule module = nullptr;
  std::unordered_map<int, CUfunction> functions_by_route;
};

struct lupine_library_image_record {
  uint32_t kind = 0;
  const void *code = nullptr;
  std::vector<unsigned char> image;
  std::unordered_map<int, CUlibrary> libraries_by_route;
};

struct lupine_module_image_record {
  uint32_t kind = 0;
  const void *image_ptr = nullptr;
  std::vector<unsigned char> image;
  std::unordered_map<int, CUmodule> modules_by_route;
};

struct lupine_library_kernel_record {
  CUlibrary library = nullptr;
  std::string name;
  std::unordered_map<int, CUkernel> kernels_by_route;
};

struct lupine_module_function_record {
  CUmodule module = nullptr;
  std::string name;
  std::unordered_map<int, CUfunction> functions_by_route;
};

static CUresult lupine_read_func_param_layout(CUfunction function,
                                              std::vector<size_t> *offsets,
                                              std::vector<size_t> *sizes);
static CUresult lupine_read_kernel_param_layout(CUkernel kernel,
                                                std::vector<size_t> *offsets,
                                                std::vector<size_t> *sizes);
static CUresult lupine_warm_func_param_info(CUfunction function);
static CUresult lupine_warm_kernel_param_info(CUkernel kernel);

extern int rpc_size();
extern int rpc_open();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);

static bool lupine_stub_missing_enabled() {
  static bool enabled = [] {
    const char *value = getenv("LUPINE_STUB_MISSING");
    return value == nullptr || strcmp(value, "0") != 0;
  }();
  return enabled;
}

static bool lupine_symbol_looks_like_driver_api(const char *symbol) {
  return symbol != nullptr && symbol[0] == 'c' && symbol[1] == 'u' &&
         symbol[2] >= 'A' && symbol[2] <= 'Z';
}

using lupine_dlsym_fn = void *(*)(void *, const char *);

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
  static lupine_dlsym_fn real_dlsym = nullptr;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    const char *version = lupine_dlsym_glibc_version();
    if (version != nullptr) {
      real_dlsym = reinterpret_cast<lupine_dlsym_fn>(
          dlvsym(RTLD_NEXT, "dlsym", version));
    }
  }
  return real_dlsym != nullptr ? real_dlsym(handle, name) : nullptr;
}

extern "C" CUresult lupine_unsupported_driver_api() {
  LUPINE_LOG_ERROR("LUPINE unsupported generic Driver API called");
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_missing_driver_api_called(const char *symbol) {
  LUPINE_LOG_ERROR("LUPINE missing Driver API called: " << symbol);
  return CUDA_ERROR_NOT_SUPPORTED;
}

static std::unordered_map<std::string, std::vector<uint64_t>> &
lupine_private_export_hashes() {
  static std::unordered_map<std::string, std::vector<uint64_t>> hashes;
  return hashes;
}

static std::unordered_map<CUfunction, lupine_private_node_mapping> &
lupine_private_node_map() {
  static std::unordered_map<CUfunction, lupine_private_node_mapping> mappings;
  return mappings;
}

static std::unordered_map<CUfunction, CUfunction> &lupine_host_function_map() {
  static std::unordered_map<CUfunction, CUfunction> mappings;
  return mappings;
}

static std::vector<CUmodule> &lupine_loaded_modules() {
  static std::vector<CUmodule> modules;
  return modules;
}

struct lupine_device_attribute_key {
  int device = 0;
  int attribute = 0;

  bool operator==(const lupine_device_attribute_key &other) const {
    return device == other.device && attribute == other.attribute;
  }
};

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

struct lupine_kernel_attribute_key {
  int route_id = -2;
  CUkernel kernel = nullptr;
  int attribute = 0;
  int device = -1;

  bool operator==(const lupine_kernel_attribute_key &other) const {
    return route_id == other.route_id && kernel == other.kernel &&
           attribute == other.attribute && device == other.device;
  }
};

struct lupine_kernel_attribute_key_hash {
  size_t operator()(const lupine_kernel_attribute_key &key) const {
    size_t hash = std::hash<int>{}(key.route_id);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.kernel)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.attribute) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    hash ^=
        std::hash<int>{}(key.device) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct lupine_function_attribute_key {
  int route_id = -2;
  CUfunction function = nullptr;
  int attribute = 0;

  bool operator==(const lupine_function_attribute_key &other) const {
    return route_id == other.route_id && function == other.function &&
           attribute == other.attribute;
  }
};

struct lupine_function_attribute_key_hash {
  size_t operator()(const lupine_function_attribute_key &key) const {
    size_t hash = std::hash<int>{}(key.route_id);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.function)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.attribute) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    return hash;
  }
};

struct lupine_param_info_key {
  uintptr_t handle = 0;
  size_t index = 0;
  bool kernel = false;

  bool operator==(const lupine_param_info_key &other) const {
    return handle == other.handle && index == other.index &&
           kernel == other.kernel;
  }
};

struct lupine_param_info_value {
  size_t offset = 0;
  size_t size = 0;
};

struct lupine_param_layout_key {
  uintptr_t handle = 0;
  bool kernel = false;

  bool operator==(const lupine_param_layout_key &other) const {
    return handle == other.handle && kernel == other.kernel;
  }
};

struct lupine_device_attribute_key_hash {
  size_t operator()(const lupine_device_attribute_key &key) const {
    return (static_cast<size_t>(static_cast<unsigned int>(key.device)) << 32) ^
           static_cast<size_t>(static_cast<unsigned int>(key.attribute));
  }
};

struct lupine_kernel_function_key_hash {
  size_t operator()(const lupine_kernel_function_key &key) const {
    size_t hash = std::hash<int>{}(key.route_id);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.context)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.kernel)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct lupine_occupancy_key_hash {
  size_t operator()(const lupine_occupancy_key &key) const {
    size_t hash = std::hash<int>{}(key.route_id);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.function)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(key.block_size) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    hash ^= std::hash<size_t>{}(key.dynamic_smem_size) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
    hash ^= std::hash<unsigned int>{}(key.flags) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    hash ^= std::hash<bool>{}(key.with_flags) + 0x9e3779b9 + (hash << 6) +
            (hash >> 2);
    return hash;
  }
};

struct lupine_param_info_key_hash {
  size_t operator()(const lupine_param_info_key &key) const {
    size_t hash = std::hash<uintptr_t>{}(key.handle);
    hash ^=
        std::hash<size_t>{}(key.index) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^=
        std::hash<bool>{}(key.kernel) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct lupine_param_layout_key_hash {
  size_t operator()(const lupine_param_layout_key &key) const {
    size_t hash = std::hash<uintptr_t>{}(key.handle);
    hash ^=
        std::hash<bool>{}(key.kernel) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

static std::unordered_map<CUmodule, lupine_module_image_record> &
lupine_module_images() {
  static auto *images =
      new std::unordered_map<CUmodule, lupine_module_image_record>();
  return *images;
}

static std::unordered_map<CUlibrary, lupine_library_image_record> &
lupine_library_images() {
  static auto *images =
      new std::unordered_map<CUlibrary, lupine_library_image_record>();
  return *images;
}

static std::unordered_map<CUkernel, lupine_library_kernel_record> &
lupine_library_kernels() {
  static auto *kernels =
      new std::unordered_map<CUkernel, lupine_library_kernel_record>();
  return *kernels;
}

static std::unordered_map<CUfunction, lupine_module_function_record> &
lupine_module_functions() {
  static auto *functions =
      new std::unordered_map<CUfunction, lupine_module_function_record>();
  return *functions;
}

static libcuckoo::cuckoohash_map<lupine_device_attribute_key, int,
                                 lupine_device_attribute_key_hash> &
lupine_device_attribute_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_device_attribute_key, int,
                                    lupine_device_attribute_key_hash>();
  return *cache;
}

struct lupine_device_snapshot_info {
  char name[LUPINE_DEVICE_SNAPSHOT_NAME_BYTES] = {};
  CUuuid uuid = {};
  uint64_t total_mem = 0;
};

static libcuckoo::cuckoohash_map<int, lupine_device_snapshot_info> &
lupine_device_snapshot_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<int, lupine_device_snapshot_info>();
  return *cache;
}

static libcuckoo::cuckoohash_map<conn_t *, bool> &
lupine_device_snapshot_attempts() {
  static auto *attempts = new libcuckoo::cuckoohash_map<conn_t *, bool>();
  return *attempts;
}

// PyTorch's pin-memory path polls cuDevicePrimaryCtxGetState continuously
// (~42k calls in a 100-step makemore run); over a remote connection every
// poll is a blocked round trip that also occupies the connection's write
// path, so training RPCs queue behind it. Primary-context state on the
// per-connection server process can only change through this client's own
// RPCs, so the state is cached per device and invalidated by the manual
// wrappers of the four mutators (retain/release/reset/set-flags).
struct lupine_primary_ctx_state {
  unsigned int flags = 0;
  int active = 0;
};

static libcuckoo::cuckoohash_map<int, lupine_primary_ctx_state> &
lupine_primary_ctx_state_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<int, lupine_primary_ctx_state>();
  return *cache;
}

static void lupine_invalidate_primary_ctx_state(CUdevice dev) {
  lupine_primary_ctx_state_cache().erase(static_cast<int>(dev));
}

static libcuckoo::cuckoohash_map<lupine_kernel_function_key, CUfunction,
                                 lupine_kernel_function_key_hash> &
lupine_kernel_function_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_kernel_function_key, CUfunction,
                                    lupine_kernel_function_key_hash>();
  return *cache;
}

static libcuckoo::cuckoohash_map<lupine_occupancy_key, int,
                                 lupine_occupancy_key_hash> &
lupine_occupancy_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_occupancy_key, int,
                                    lupine_occupancy_key_hash>();
  return *cache;
}

static libcuckoo::cuckoohash_map<lupine_kernel_attribute_key, int,
                                 lupine_kernel_attribute_key_hash> &
lupine_kernel_attribute_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_kernel_attribute_key, int,
                                    lupine_kernel_attribute_key_hash>();
  return *cache;
}

static libcuckoo::cuckoohash_map<lupine_function_attribute_key, int,
                                 lupine_function_attribute_key_hash> &
lupine_function_attribute_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_function_attribute_key, int,
                                    lupine_function_attribute_key_hash>();
  return *cache;
}

static int lupine_read_function_attributes(conn_t *conn, lupine_route route,
                                           CUfunction function) {
  uint32_t count = 0;
  if (conn == nullptr || rpc_read(conn, &count, sizeof(count)) < 0 ||
      count > 4096) {
    return -1;
  }
  int route_id = lupine_route_identity(route);
  for (uint32_t i = 0; i < count; ++i) {
    CUresult result = CUDA_ERROR_UNKNOWN;
    if (rpc_read(conn, &result, sizeof(result)) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      continue;
    }
    int attribute = 0;
    int value = 0;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
        rpc_read(conn, &value, sizeof(value)) < 0) {
      return -1;
    }
    lupine_function_attribute_cache().insert_or_assign(
        lupine_function_attribute_key{route_id, function, attribute}, value);
  }
  return 0;
}

static int lupine_read_kernel_attributes(conn_t *conn, lupine_route route,
                                         CUkernel kernel, CUdevice device) {
  uint32_t count = 0;
  if (conn == nullptr || rpc_read(conn, &count, sizeof(count)) < 0 ||
      count > 4096) {
    return -1;
  }
  int route_id = lupine_route_identity(route);
  for (uint32_t i = 0; i < count; ++i) {
    CUresult result = CUDA_ERROR_UNKNOWN;
    if (rpc_read(conn, &result, sizeof(result)) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      continue;
    }
    int attribute = 0;
    int value = 0;
    if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
        rpc_read(conn, &value, sizeof(value)) < 0) {
      return -1;
    }
    lupine_kernel_attribute_cache().insert_or_assign(
        lupine_kernel_attribute_key{route_id, kernel, attribute, device},
        value);
  }
  return 0;
}

static libcuckoo::cuckoohash_map<lupine_param_info_key, lupine_param_info_value,
                                 lupine_param_info_key_hash> &
lupine_param_info_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_param_info_key,
                                    lupine_param_info_value,
                                    lupine_param_info_key_hash>();
  return *cache;
}

static libcuckoo::cuckoohash_map<lupine_param_layout_key, size_t,
                                 lupine_param_layout_key_hash> &
lupine_param_layout_count_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_param_layout_key, size_t,
                                    lupine_param_layout_key_hash>();
  return *cache;
}

static bool lupine_param_layout_count(uintptr_t handle, bool kernel,
                                      size_t *count) {
  return count != nullptr &&
         lupine_param_layout_count_cache().find(
             lupine_param_layout_key{handle, kernel}, *count);
}

static constexpr size_t lupine_param_info_value_offset() {
  return (sizeof(CUresult) + alignof(size_t) - 1) / alignof(size_t) *
         alignof(size_t);
}

static constexpr size_t lupine_param_info_storage_size() {
  return lupine_param_info_value_offset() + 2 * sizeof(size_t);
}

static size_t lupine_rpc_param_info_storage_size(size_t param_count) {
  return (param_count + 1) * lupine_param_info_storage_size();
}

template <typename Query>
static int lupine_write_param_values(conn_t *conn, uintptr_t handle,
                                     bool kernel, void *const *values,
                                     Query query) {
  size_t count = 0;
  if (!lupine_param_layout_count(handle, kernel, &count)) {
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    if (result == nullptr) {
      return -1;
    }
    *result = CUDA_ERROR_INVALID_HANDLE;
    return rpc_write(conn, result, sizeof(*result));
  }

  for (size_t i = 0;; ++i) {
    auto *storage = static_cast<unsigned char *>(
        rpc_write_buffer(conn, lupine_param_info_storage_size(),
                         std::max(alignof(CUresult), alignof(size_t))));
    if (storage == nullptr) {
      return -1;
    }
    auto *result = reinterpret_cast<CUresult *>(storage);
    auto *offset =
        reinterpret_cast<size_t *>(storage + lupine_param_info_value_offset());
    auto *size = offset + 1;

    *result = i == count ? CUDA_ERROR_INVALID_VALUE : query(i, offset, size);
    if (*result == CUDA_SUCCESS &&
        (values == nullptr || values[i] == nullptr)) {
      *result = CUDA_ERROR_INVALID_VALUE;
    }
    if (rpc_write(conn, result, sizeof(*result)) < 0) {
      return -1;
    }
    if (*result != CUDA_SUCCESS) {
      return 0;
    }
    if (rpc_write(conn, offset, sizeof(*offset)) < 0 ||
        rpc_write(conn, size, sizeof(*size)) < 0 ||
        rpc_write(conn, values[i], *size) < 0) {
      return -1;
    }
  }
}

static int lupine_write_func_param_values(conn_t *conn, CUfunction function,
                                          void *const *values) {
  return lupine_write_param_values(
      conn, reinterpret_cast<uintptr_t>(function), false, values,
      [function](size_t index, size_t *offset, size_t *size) {
        return cuFuncGetParamInfo(function, index, offset, size);
      });
}

#if CUDA_VERSION >= 12000
static int lupine_write_kernel_param_values(conn_t *conn, CUkernel kernel,
                                            void *const *values) {
  return lupine_write_param_values(
      conn, reinterpret_cast<uintptr_t>(kernel), true, values,
      [kernel](size_t index, size_t *offset, size_t *size) {
        return cuKernelGetParamInfo(kernel, index, offset, size);
      });
}
#endif

static int lupine_write_launch_config(conn_t *conn,
                                      const CUlaunchConfig *config) {
  if (conn == nullptr || config == nullptr ||
      rpc_write(conn, config, sizeof(*config)) < 0) {
    return -1;
  }
  if (config->numAttrs != 0 && config->attrs != nullptr &&
      rpc_write(conn, config->attrs,
                config->numAttrs * sizeof(config->attrs[0])) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_write_jit_options(conn_t *conn,
                                    const unsigned int *num_options,
                                    const CUjit_option *options,
                                    void *const *option_values) {
  if (conn == nullptr || num_options == nullptr ||
      (*num_options != 0 && (options == nullptr || option_values == nullptr)) ||
      rpc_write(conn, num_options, sizeof(*num_options)) < 0 ||
      (*num_options != 0 &&
       rpc_write(conn, options, *num_options * sizeof(*options)) < 0)) {
    return -1;
  }
  if (*num_options == 0) {
    return 0;
  }

  auto *raw_values = static_cast<uintptr_t *>(rpc_write_buffer(
      conn, static_cast<size_t>(*num_options) * sizeof(uintptr_t),
      alignof(uintptr_t)));
  if (raw_values == nullptr) {
    return -1;
  }
  for (unsigned int i = 0; i < *num_options; ++i) {
    raw_values[i] = reinterpret_cast<uintptr_t>(option_values[i]);
  }
  return rpc_write(conn, raw_values,
                   static_cast<size_t>(*num_options) * sizeof(*raw_values));
}

static constexpr uintptr_t LUPINE_RPC_NULL_OPTION_VALUES = UINTPTR_MAX;

static int lupine_write_library_options(conn_t *conn,
                                        const unsigned int *num_options,
                                        const CUlibraryOption *options,
                                        void *const *option_values) {
  if (conn == nullptr || num_options == nullptr ||
      (*num_options != 0 && options == nullptr) ||
      rpc_write(conn, num_options, sizeof(*num_options)) < 0 ||
      (*num_options != 0 &&
       rpc_write(conn, options, *num_options * sizeof(*options)) < 0)) {
    return -1;
  }
  if (*num_options == 0) {
    return 0;
  }

  auto *raw_values = static_cast<uintptr_t *>(rpc_write_buffer(
      conn, static_cast<size_t>(*num_options) * sizeof(uintptr_t),
      alignof(uintptr_t)));
  if (raw_values == nullptr) {
    return -1;
  }
  for (unsigned int i = 0; i < *num_options; ++i) {
    raw_values[i] = option_values == nullptr
                        ? LUPINE_RPC_NULL_OPTION_VALUES
                        : reinterpret_cast<uintptr_t>(option_values[i]);
  }
  return rpc_write(conn, raw_values,
                   static_cast<size_t>(*num_options) * sizeof(*raw_values));
}

static void lupine_cache_param_layout(uintptr_t handle, bool kernel,
                                      uint32_t count,
                                      const std::vector<uint64_t> &params) {
  for (uint32_t index = 0; index < count; ++index) {
    lupine_param_info_cache().insert_or_assign(
        lupine_param_info_key{handle, index, kernel},
        lupine_param_info_value{static_cast<size_t>(params[index * 2]),
                                static_cast<size_t>(params[index * 2 + 1])});
  }
  lupine_param_layout_count_cache().insert_or_assign(
      lupine_param_layout_key{handle, kernel}, count);
}

// Filled from the library snapshot side-channel RPC; serves
// cuLibraryGetKernel without a round trip.
struct lupine_library_kernel_name_key {
  CUlibrary library = nullptr;
  std::string name;

  bool operator==(const lupine_library_kernel_name_key &other) const {
    return library == other.library && name == other.name;
  }
};

struct lupine_library_kernel_name_key_hash {
  size_t operator()(const lupine_library_kernel_name_key &key) const {
    return reinterpret_cast<uintptr_t>(key.library) ^
           std::hash<std::string>()(key.name);
  }
};

static libcuckoo::cuckoohash_map<lupine_library_kernel_name_key, CUkernel,
                                 lupine_library_kernel_name_key_hash> &
lupine_library_kernel_names() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_library_kernel_name_key, CUkernel,
                                    lupine_library_kernel_name_key_hash>();
  return *cache;
}

static std::mutex &lupine_host_function_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::mutex &lupine_library_kernel_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static unsigned char (&lupine_private_6e16_node_pool())[16][0x500] {
  static unsigned char nodes[16][0x500] = {};
  return nodes;
}

static std::atomic<unsigned int> &lupine_private_6e16_next_node() {
  static std::atomic<unsigned int> next{0};
  return next;
}

static std::mutex &lupine_private_node_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::mutex &lupine_graph_kernel_node_params_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<CUgraphNode, std::shared_ptr<void *>> &
lupine_graph_kernel_node_params_cache() {
  static auto *cache =
      new std::unordered_map<CUgraphNode, std::shared_ptr<void *>>();
  return *cache;
}

static void lupine_remember_loaded_module(CUmodule module) {
  if (module == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_host_function_mutex());
  auto &modules = lupine_loaded_modules();
  if (std::find(modules.begin(), modules.end(), module) == modules.end()) {
    modules.push_back(module);
  }
}

extern "C" void lupine_remember_loaded_module_for_rpc(CUmodule module) {
  lupine_remember_loaded_module(module);
}

static std::vector<lupine_server_endpoint> &lupine_server_endpoints() {
  static auto *endpoints = new std::vector<lupine_server_endpoint>();
  return *endpoints;
}

static int lupine_connect_endpoint(conn_t *conn,
                                   const lupine_server_endpoint &endpoint,
                                   unsigned int logical_index) {
  if (conn == nullptr) {
    return -1;
  }

  lupine_socket_t sockfd =
      lupine_tcp_connect(endpoint.host.c_str(), endpoint.port.c_str());
  if (sockfd == LUPINE_INVALID_SOCKET) {
    LUPINE_LOG_ERROR("Connecting to " << endpoint.host << " port "
                                      << endpoint.port << " failed");
    return -1;
  }

  rpc_write_queue_free(conn);
  // A new transport gets fresh server lane threads with no CUDA context.
  lupine_invalidate_current_context_cache();
  *conn = {};
  conn->connfd = sockfd;
  conn->request_id = 0;
  conn->closed = 0;
  conn->local_request_parity = conn->request_id & 1;
  conn->logical_index = static_cast<int>(logical_index);
  if (endpoint.tls) {
#ifdef LUPINE_TLS_OPENSSL
    static SSL_CTX *tls_ctx = []() {
      SSL_CTX *c = SSL_CTX_new(TLS_client_method());
      if (c != nullptr) {
        SSL_CTX_set_min_proto_version(c, TLS1_2_VERSION);
        SSL_CTX_set_default_verify_paths(c);
        SSL_CTX_set_verify(c, SSL_VERIFY_PEER, nullptr);
      }
      return c;
    }();
    SSL *ssl = tls_ctx != nullptr ? SSL_new(tls_ctx) : nullptr;
    if (ssl != nullptr &&
        SSL_set_tlsext_host_name(ssl, endpoint.host.c_str()) == 1 &&
        SSL_set1_host(ssl, endpoint.host.c_str()) == 1 &&
        SSL_set_fd(ssl, static_cast<int>(sockfd)) == 1 &&
        SSL_connect(ssl) == 1) {
      conn->tls_session = ssl;
    } else {
      if (ssl != nullptr) {
        SSL_free(ssl);
      }
      LUPINE_LOG_ERROR("TLS handshake with " << endpoint.host << " failed");
      lupine_socket_close(sockfd);
      return -1;
    }
#else
    LUPINE_LOG_ERROR("LUPINE_SERVER entry "
                     << endpoint.host << ":" << endpoint.port
                     << " uses https:// but this client was built "
                        "without TLS support");
    lupine_socket_close(sockfd);
    return -1;
#endif
  }
  if (pthread_mutex_init(&conn->read_mutex, NULL) != 0 ||
      pthread_mutex_init(&conn->write_mutex, NULL) != 0 ||
      pthread_mutex_init(&conn->call_mutex, NULL) != 0 ||
      pthread_cond_init(&conn->read_cond, NULL) != 0 ||
      rpc_http2_client_init(conn) < 0 ||
      pthread_create(&conn->read_thread, NULL, rpc_client_dispatch_thread,
                     (void *)conn) != 0) {
    lupine_socket_close(sockfd);
    return -1;
  }
  rpc_http2_client_start_heartbeat(conn);

  return 0;
}

static void lupine_join_connection_threads(conn_t *conn) {
  if (conn->read_thread != 0) {
    pthread_join(conn->read_thread, nullptr);
    conn->read_thread = 0;
  }
  if (conn->rpc_thread != 0) {
    pthread_join(conn->rpc_thread, nullptr);
    conn->rpc_thread = 0;
  }
}

static bool lupine_env_enabled(const char *name) {
  const char *value = getenv(name);
  return value != nullptr && strcmp(value, "0") != 0 &&
         strcasecmp(value, "false") != 0 && strcasecmp(value, "no") != 0;
}

static void *lupine_local_libcuda_handle() {
  static std::once_flag once;
  static void *handle = nullptr;
  std::call_once(once, []() {
    if (lupine_env_enabled("LUPINE_DISABLE_LOCAL")) {
      return;
    }
    const char *override_path = getenv("LUPINE_REAL_LIBCUDA");
    const char *paths[] = {
        override_path,
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/wsl/lib/libcuda.so.1",
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

static bool lupine_is_local_address(const void *ptr);

// Client-answered entry points must fail with NOT_INITIALIZED until cuInit;
// forwarded ones get the server's own state.
static std::atomic<bool> lupine_cuda_initialized{false};

static bool lupine_cuda_is_initialized() {
  return lupine_cuda_initialized.load(std::memory_order_acquire);
}

static CUresult lupine_remote_cuInit(conn_t *conn, unsigned int flags) {
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuInit) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuInit(unsigned int flags) {
  CUresult first_error = CUDA_SUCCESS;
  bool initialized_any = false;
  using cuInit_fn = CUresult (*)(unsigned int);
  auto local_init = lupine_real_cuda_fn<cuInit_fn>("cuInit");
  if (local_init != nullptr) {
    CUresult result = local_init(flags);
    if (result != CUDA_SUCCESS && first_error == CUDA_SUCCESS) {
      first_error = result;
    } else if (result == CUDA_SUCCESS) {
      initialized_any = true;
    }
  }
  if (rpc_open() == 0) {
    for (int i = 0; i < nconns; ++i) {
      CUresult result = lupine_remote_cuInit(&conns[i], flags);
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
  return first_error;
}

extern "C" CUresult cuDeviceGetCount(int *count) {
  return lupine_virtual_device_count(count);
}

extern "C" CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  return lupine_virtual_device_for_ordinal(device, ordinal);
}

extern "C" CUresult cuDeviceGetP2PAttribute(int *value,
                                            CUdevice_P2PAttribute attrib,
                                            CUdevice srcDevice,
                                            CUdevice dstDevice) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (srcDevice == dstDevice) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  lupine_route src_route = lupine_route_for_device(&srcDevice);
  lupine_route dst_route = lupine_route_for_device(&dstDevice);
  if (src_route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE ||
      dst_route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (src_route.kind == LUPINE_ROUTE_INVALID ||
      dst_route.kind == LUPINE_ROUTE_INVALID) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (!lupine_routes_share_server(src_route, dst_route)) {
    *value = 0;
    return CUDA_SUCCESS;
  }
  if (lupine_route_is_local(src_route)) {
    using real_fn_t =
        CUresult (*)(int *, CUdevice_P2PAttribute, CUdevice, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceGetP2PAttribute");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(value, attrib, srcDevice, dstDevice);
  }

  conn_t *conn = lupine_route_remote_conn(src_route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetP2PAttribute) < 0 ||
      rpc_write(conn, value, sizeof(*value)) < 0 ||
      rpc_write(conn, &attrib, sizeof(attrib)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, sizeof(*value)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                          CUdevice peerDev) {
  if (canAccessPeer == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice peer = peerDev;
  lupine_route route = lupine_route_for_device(&dev);
  lupine_route peer_route = lupine_route_for_device(&peer);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE ||
      peer_route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    if (!lupine_route_is_local(peer_route)) {
      *canAccessPeer = 0;
      return CUDA_SUCCESS;
    }
    using real_fn_t = CUresult (*)(int *, CUdevice, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceCanAccessPeer");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(canAccessPeer, dev, peer);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (!lupine_translate_device_for_conn(conn, &peerDev)) {
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
  }

  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_write_start_request(conn, RPC_cuDeviceCanAccessPeer) < 0 ||
      rpc_write(conn, canAccessPeer, sizeof(*canAccessPeer)) < 0 ||
      rpc_write(conn, &dev, sizeof(dev)) < 0 ||
      rpc_write(conn, &peerDev, sizeof(peerDev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, canAccessPeer, sizeof(*canAccessPeer)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuCtxEnablePeerAccess(CUcontext peerContext,
                                          unsigned int flags) {
  lupine_route current_route = lupine_route_for_current_context();
  lupine_route peer_route = lupine_route_for_context(peerContext);
  if (lupine_route_identity(current_route) !=
      lupine_route_identity(peer_route)) {
    return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
  }
  if (lupine_route_is_local(current_route)) {
    using real_fn_t = CUresult (*)(CUcontext, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxEnablePeerAccess");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(peerContext, flags);
  }
  conn_t *conn = lupine_route_remote_conn(current_route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxEnablePeerAccess) < 0 ||
      rpc_write(conn, &peerContext, sizeof(peerContext)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  lupine_route current_route = lupine_route_for_current_context();
  lupine_route peer_route = lupine_route_for_context(peerContext);
  if (lupine_route_identity(current_route) !=
      lupine_route_identity(peer_route)) {
    return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
  }
  if (lupine_route_is_local(current_route)) {
    using real_fn_t = CUresult (*)(CUcontext);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxDisablePeerAccess");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(peerContext);
  }
  conn_t *conn = lupine_route_remote_conn(current_route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxDisablePeerAccess) < 0 ||
      rpc_write(conn, &peerContext, sizeof(peerContext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" void lupine_record_library_image(CUlibrary library,
                                            lupine_route route, uint32_t kind,
                                            const unsigned char *image,
                                            size_t image_size,
                                            const void *code) {
  int route_id = lupine_route_identity(route);
  if (library == nullptr || image == nullptr || image_size == 0 ||
      route_id == -2) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto &record = lupine_library_images()[library];
  record.kind = kind;
  record.code = code;
  record.image.assign(image, image + image_size);
  record.libraries_by_route[route_id] = library;
}

extern "C" void lupine_record_module_image(CUmodule module, lupine_route route,
                                           uint32_t kind,
                                           const unsigned char *image,
                                           size_t image_size,
                                           const void *image_ptr) {
  int route_id = lupine_route_identity(route);
  if (module == nullptr || image == nullptr || image_size == 0 ||
      route_id == -2) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto &record = lupine_module_images()[module];
  record.kind = kind;
  record.image_ptr = image_ptr;
  record.image.assign(image, image + image_size);
  record.modules_by_route[route_id] = module;
}

extern "C" CUresult lupine_record_library_kernel(CUkernel kernel,
                                                 CUlibrary library,
                                                 const char *name,
                                                 lupine_route route);

extern "C" CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                                       const char *name) {
  if (pKernel == nullptr || name == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUkernel *, CUlibrary, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLibraryGetKernel", &return_value, pKernel, library, name)) {
    if (return_value == CUDA_SUCCESS && pKernel != nullptr) {
      return_value =
          lupine_record_library_kernel(*pKernel, library, name, route);
    }
    return return_value;
  }
  // Ownership recording and param-info warming already happened at snapshot
  // time.
  CUkernel cached = nullptr;
  if (lupine_library_kernel_names().find(
          lupine_library_kernel_name_key{library, std::string(name)}, cached)) {
    *pKernel = cached;
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetKernel) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pKernel, sizeof(CUkernel)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && pKernel != nullptr) {
    return_value = lupine_record_library_kernel(*pKernel, library, name, route);
  }
  return return_value;
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
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &record = lupine_library_kernels()[kernel];
    record.library = library;
    record.name = name;
    record.kernels_by_route[route_id] = kernel;
  }
  return CUDA_SUCCESS;
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
    record.name = name;
    record.functions_by_route[route_id] = function;
  }
  return CUDA_SUCCESS;
}

static void lupine_prefetch_function_param_layout(CUfunction function,
                                                  lupine_route route) {
  if (lupine_route_is_local(route)) {
    return;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, LUPINE_RPC_lupineFunctionParamLayoutSnapshot) < 0 ||
      rpc_write(conn, &function, sizeof(function)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return;
  }

  uintptr_t handle = reinterpret_cast<uintptr_t>(function);
  size_t index = 0;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  for (;; ++index) {
    if (rpc_read(conn, &result, sizeof(result)) < 0) {
      return;
    }
    if (result != CUDA_SUCCESS) {
      break;
    }
    size_t offset = 0;
    size_t size = 0;
    if (rpc_read(conn, &offset, sizeof(offset)) < 0 ||
        rpc_read(conn, &size, sizeof(size)) < 0) {
      return;
    }
    lupine_param_info_cache().insert_or_assign(
        lupine_param_info_key{handle, index, false},
        lupine_param_info_value{offset, size});
  }
  if (rpc_read_end(conn) < 0) {
    return;
  }
  if (result == CUDA_ERROR_INVALID_VALUE) {
    lupine_param_layout_count_cache().insert_or_assign(
        lupine_param_layout_key{handle, false}, index);
  }
}

static void lupine_prefill_function_attribute_snapshot(CUfunction function,
                                                       lupine_route route,
                                                       conn_t *conn) {
  if (function == nullptr || conn == nullptr) {
    return;
  }

  if (rpc_write_start_request(conn,
                              LUPINE_RPC_lupineFunctionAttributeSnapshot) < 0 ||
      rpc_write(conn, &function, sizeof(function)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_function_attributes(conn, route, function) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
}

extern "C" CUresult cuModuleGetFunction(CUfunction *function, CUmodule module,
                                        const char *name) {
  if (function == nullptr || name == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_module(module);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  using real_fn_t = CUresult (*)(CUfunction *, CUmodule, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuModuleGetFunction", &result, function, module, name)) {
    if (result == CUDA_SUCCESS) {
      (void)lupine_record_module_function(*function, module, name, route);
      lupine_prefetch_function_param_layout(*function, route);
    }
    return result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetFunction) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, function, sizeof(*function)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  (void)lupine_record_module_function(*function, module, name, route);
  lupine_prefetch_function_param_layout(*function, route);
  lupine_prefill_function_attribute_snapshot(*function, route, conn);
  return result;
}

static CUresult lupine_load_recorded_module_on_route(CUmodule source_module,
                                                     lupine_route route,
                                                     CUmodule *module) {
  if (module == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *module = nullptr;
  int route_id = lupine_route_identity(route);
  if (source_module == nullptr || route_id == -2) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_module_image_record record;
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_module_images().find(source_module);
    if (it == lupine_module_images().end()) {
      // No recorded image, so the module cannot be replicated elsewhere. It is
      // still valid on the route that already owns it (cuModuleLoad, for one,
      // never records an image), so hand that handle back untouched and only
      // fail when a genuinely different route is asked for.
      if (lupine_route_identity(lupine_route_for_module(source_module)) ==
          route_id) {
        *module = source_module;
        return CUDA_SUCCESS;
      }
      return CUDA_ERROR_NOT_FOUND;
    }
    auto cached = it->second.modules_by_route.find(route_id);
    if (cached != it->second.modules_by_route.end()) {
      *module = cached->second;
      return CUDA_SUCCESS;
    }
    record = it->second;
  }

  CUmodule loaded = nullptr;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUmodule *, const void *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuModuleLoadData");
    if (real == nullptr || record.image_ptr == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    result = real(&loaded, record.image_ptr);
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    size_t image_size = record.image.size();
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuModuleLoadData) < 0 ||
        rpc_write(conn, &record.kind, sizeof(record.kind)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write_payload(conn, record.image.data(), image_size) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &loaded, sizeof(loaded)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }

  if (result != CUDA_SUCCESS || loaded == nullptr) {
    return result;
  }

  lupine_remember_loaded_module(loaded);
  lupine_note_module_owner_route(loaded, route);
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_module_images()[source_module].modules_by_route[route_id] = loaded;
    auto &loaded_record = lupine_module_images()[loaded];
    loaded_record.kind = record.kind;
    loaded_record.image_ptr = record.image_ptr;
    loaded_record.image = std::move(record.image);
    loaded_record.modules_by_route[route_id] = loaded;
  }
  *module = loaded;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  if (module == nullptr || fname == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUmodule *, const char *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuModuleLoad");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(module, fname);
    if (result == CUDA_SUCCESS && module != nullptr) {
      lupine_note_module_owner_route(*module, route);
    }
    return result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  void *mapping = MAP_FAILED;
  size_t mapped_size = 0;
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  // FILE_NOT_FOUND is reserved for open() failing.
  struct stat st = {};
  if (fstat(fd, &st) < 0) {
    close(fd);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (st.st_size <= 0) {
    close(fd);
    return CUDA_ERROR_INVALID_IMAGE;
  }
  mapped_size = static_cast<size_t>(st.st_size);
  mapping = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  bool failed =
      conn == nullptr || rpc_write_start_request(conn, RPC_cuModuleLoad) < 0 ||
      rpc_write(conn, &mapped_size, sizeof(mapped_size)) < 0 ||
      rpc_write_payload(conn, mapping, mapped_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, module, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0;
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
      // See lupine_load_recorded_module_on_route: without a recorded image the
      // library only exists on the route that loaded it.
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
    using real_fn_t =
        CUresult (*)(CUlibrary *, const void *, CUjit_option *, void **,
                     unsigned int, CUlibraryOption *, void **, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLibraryLoadData");
    if (real == nullptr || record.code == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    result =
        real(&loaded, record.code, nullptr, nullptr, 0, nullptr, nullptr, 0);
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    size_t image_size = record.image.size();
    unsigned int zero_options = 0;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
        rpc_write(conn, &record.kind, sizeof(record.kind)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write_payload(conn, record.image.data(), image_size) < 0 ||
        lupine_write_jit_options(conn, &zero_options, nullptr, nullptr) < 0 ||
        lupine_write_library_options(conn, &zero_options, nullptr, nullptr) <
            0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &loaded, sizeof(loaded)) < 0 ||
        rpc_read_jit_outputs(conn, {}) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
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

static CUresult lupine_resolve_library_kernel_for_route(CUfunction function,
                                                        lupine_route route,
                                                        CUfunction *resolved) {
  if (resolved == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resolved = function;
  int route_id = lupine_route_identity(route);
  if (function == nullptr || route_id == -2) {
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
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUkernel *, CUlibrary, const char *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLibraryGetKernel");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    result = real(&kernel, library, record.name.c_str());
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    std::size_t name_len = record.name.size() + 1;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuLibraryGetKernel) < 0 ||
        rpc_write(conn, &library, sizeof(library)) < 0 ||
        rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
        rpc_write(conn, record.name.c_str(), name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &kernel, sizeof(kernel)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (result != CUDA_SUCCESS || kernel == nullptr) {
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_library_kernels()[reinterpret_cast<CUkernel>(function)]
        .kernels_by_route[route_id] = kernel;
    auto &new_record = lupine_library_kernels()[kernel];
    new_record.library = library;
    new_record.name = record.name;
    new_record.kernels_by_route[route_id] = kernel;
  }
  lupine_note_function_owner_route(reinterpret_cast<CUfunction>(kernel), route);
  if (!lupine_route_is_local(route)) {
    result = lupine_warm_kernel_param_info(kernel);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  *resolved = reinterpret_cast<CUfunction>(kernel);
  return CUDA_SUCCESS;
}

static CUresult lupine_resolve_module_function_for_route(CUfunction function,
                                                         lupine_route route,
                                                         CUfunction *resolved) {
  if (resolved == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resolved = function;
  int route_id = lupine_route_identity(route);
  if (function == nullptr || route_id == -2) {
    return CUDA_SUCCESS;
  }

  lupine_module_function_record record;
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_module_functions().find(function);
    if (it == lupine_module_functions().end()) {
      return CUDA_SUCCESS;
    }
    auto cached = it->second.functions_by_route.find(route_id);
    if (cached != it->second.functions_by_route.end()) {
      *resolved = cached->second;
      return CUDA_SUCCESS;
    }
    record = it->second;
  }

  CUmodule module = nullptr;
  CUresult result =
      lupine_load_recorded_module_on_route(record.module, route, &module);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  CUfunction route_function = nullptr;
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUfunction *, CUmodule, const char *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuModuleGetFunction");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    result = real(&route_function, module, record.name.c_str());
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    std::size_t name_len = record.name.size() + 1;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuModuleGetFunction) < 0 ||
        rpc_write(conn, &module, sizeof(module)) < 0 ||
        rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
        rpc_write(conn, record.name.c_str(), name_len) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &route_function, sizeof(route_function)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (result != CUDA_SUCCESS || route_function == nullptr) {
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    lupine_module_functions()[function].functions_by_route[route_id] =
        route_function;
    auto &new_record = lupine_module_functions()[route_function];
    new_record.module = module;
    new_record.name = record.name;
    new_record.functions_by_route[route_id] = route_function;
  }
  lupine_note_function_owner_route(route_function, route);
  lupine_prefetch_function_param_layout(route_function, route);
  *resolved = route_function;
  return CUDA_SUCCESS;
}

static bool lupine_function_name_is(CUfunction function, const char *name) {
  if (function == nullptr || name == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  auto module_it = lupine_module_functions().find(function);
  if (module_it != lupine_module_functions().end() &&
      module_it->second.name == name) {
    return true;
  }
  auto library_it =
      lupine_library_kernels().find(reinterpret_cast<CUkernel>(function));
  return library_it != lupine_library_kernels().end() &&
         library_it->second.name == name;
}

static bool lupine_managed_kernel_requires_launch_sync(CUfunction function) {
  return lupine_function_name_is(function, "atomicKernel") ||
         lupine_function_name_is(function, "_Z12atomicKernelPi");
}

static bool lupine_read_file_span(const char *path,
                                  std::vector<unsigned char> *bytes) {
  if (path == nullptr || bytes == nullptr) {
    return false;
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  bytes->resize(static_cast<size_t>(size));
  return file.read(reinterpret_cast<char *>(bytes->data()), size).good();
}

static bool lupine_lookup_elf_function_symbol(const char *path,
                                              uintptr_t offset,
                                              std::string *symbol) {
  std::vector<unsigned char> bytes;
  if (symbol == nullptr || !lupine_read_file_span(path, &bytes) ||
      bytes.size() < sizeof(Elf64_Ehdr)) {
    return false;
  }

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(bytes.data());
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr->e_shentsize != sizeof(Elf64_Shdr) || ehdr->e_shoff == 0 ||
      ehdr->e_shnum == 0) {
    return false;
  }
  size_t shoff = static_cast<size_t>(ehdr->e_shoff);
  size_t shnum = static_cast<size_t>(ehdr->e_shnum);
  if (shoff > bytes.size() ||
      shnum > (bytes.size() - shoff) / sizeof(Elf64_Shdr)) {
    return false;
  }
  const auto *sections =
      reinterpret_cast<const Elf64_Shdr *>(bytes.data() + shoff);

  uintptr_t best_value = 0;
  const char *best_name = nullptr;
  for (size_t i = 0; i < shnum; ++i) {
    if (sections[i].sh_type != SHT_SYMTAB &&
        sections[i].sh_type != SHT_DYNSYM) {
      continue;
    }
    if (sections[i].sh_link >= shnum) {
      continue;
    }
    const Elf64_Shdr &symtab = sections[i];
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    if (symtab.sh_entsize != sizeof(Elf64_Sym) ||
        symtab.sh_offset > bytes.size() || strtab.sh_offset > bytes.size() ||
        symtab.sh_size > bytes.size() - symtab.sh_offset ||
        strtab.sh_size > bytes.size() - strtab.sh_offset) {
      continue;
    }
    const auto *syms =
        reinterpret_cast<const Elf64_Sym *>(bytes.data() + symtab.sh_offset);
    const char *strings =
        reinterpret_cast<const char *>(bytes.data() + strtab.sh_offset);
    size_t count = static_cast<size_t>(symtab.sh_size / sizeof(Elf64_Sym));
    for (size_t j = 0; j < count; ++j) {
      const Elf64_Sym &sym = syms[j];
      if (ELF64_ST_TYPE(sym.st_info) != STT_FUNC ||
          sym.st_name >= strtab.sh_size || sym.st_value == 0 ||
          offset < sym.st_value ||
          (sym.st_size != 0 && offset >= sym.st_value + sym.st_size)) {
        continue;
      }
      if (best_name == nullptr || sym.st_value >= best_value) {
        best_value = sym.st_value;
        best_name = strings + sym.st_name;
      }
    }
  }

  if (best_name == nullptr || best_name[0] == '\0') {
    return false;
  }
  *symbol = best_name;
  return true;
}

static CUfunction lupine_resolve_host_function(CUfunction function) {
  if (function == nullptr) {
    return function;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_function_mutex());
    auto mapped = lupine_host_function_map().find(function);
    if (mapped != lupine_host_function_map().end()) {
      return mapped->second;
    }
  }

  Dl_info info = {};
  if (dladdr(reinterpret_cast<void *>(function), &info) == 0) {
    return function;
  }
  std::string symbol_name;
  const char *kernel_name = info.dli_sname;
  if (kernel_name == nullptr && info.dli_fname != nullptr &&
      info.dli_fbase != nullptr) {
    uintptr_t offset = reinterpret_cast<uintptr_t>(function) -
                       reinterpret_cast<uintptr_t>(info.dli_fbase);
    if (lupine_lookup_elf_function_symbol(info.dli_fname, offset,
                                          &symbol_name)) {
      kernel_name = symbol_name.c_str();
    }
  }
  if (kernel_name == nullptr) {
    LUPINE_TRACE_LOG("LUPINE could not resolve host kernel symbol for "
                     << reinterpret_cast<void *>(function));
    return function;
  }

  std::vector<CUmodule> modules;
  {
    std::lock_guard<std::mutex> lock(lupine_host_function_mutex());
    modules = lupine_loaded_modules();
  }
  for (CUmodule module : modules) {
    CUfunction remote = nullptr;
    CUresult result = cuModuleGetFunction(&remote, module, kernel_name);
    if (result == CUDA_SUCCESS && remote != nullptr) {
      std::lock_guard<std::mutex> lock(lupine_host_function_mutex());
      lupine_host_function_map()[function] = remote;
      LUPINE_TRACE_LOG("LUPINE mapped host kernel "
                       << kernel_name
                       << " host=" << reinterpret_cast<void *>(function)
                       << " remote=" << remote);
      return remote;
    }
  }
  LUPINE_TRACE_LOG("LUPINE host kernel " << kernel_name << " was not found in "
                                         << modules.size()
                                         << " loaded modules");

  return function;
}

static CUfunction lupine_translate_private_function(CUfunction function) {
  {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    auto it = lupine_private_node_map().find(function);
    if (it != lupine_private_node_map().end()) {
      return it->second.server_function;
    }
  }
  return lupine_resolve_host_function(function);
}

static CUresult lupine_get_remote_private_module_node(CUcontext context,
                                                      CUmodule module,
                                                      CUfunction *server_node,
                                                      uint64_t *server_owner);

static CUresult lupine_resolve_private_function_for_route(
    CUfunction function, lupine_route route, CUfunction *resolved) {
  if (resolved == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *resolved = function;
  int route_id = lupine_route_identity(route);
  if (function == nullptr || route_id == -2) {
    return CUDA_SUCCESS;
  }

  lupine_private_node_mapping mapping;
  {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    auto it = lupine_private_node_map().find(function);
    if (it == lupine_private_node_map().end()) {
      return CUDA_SUCCESS;
    }
    auto cached = it->second.functions_by_route.find(route_id);
    if (cached != it->second.functions_by_route.end()) {
      *resolved = cached->second;
      return CUDA_SUCCESS;
    }
    mapping = it->second;
  }

  if (mapping.module == nullptr) {
    // Nothing to replicate from; leave the handle alone and let the server the
    // launch lands on decide whether it is valid, exactly as for a function
    // this client never mapped at all.
    return CUDA_SUCCESS;
  }
  CUmodule module = nullptr;
  CUresult result =
      lupine_load_recorded_module_on_route(mapping.module, route, &module);
  if (result != CUDA_SUCCESS || module == nullptr) {
    return result;
  }

  CUfunction server_node = nullptr;
  uint64_t server_owner = 0;
  result = lupine_get_remote_private_module_node(nullptr, module, &server_node,
                                                 &server_owner);
  if (result != CUDA_SUCCESS || server_node == nullptr) {
    return result == CUDA_SUCCESS ? CUDA_ERROR_NOT_FOUND : result;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    auto &record = lupine_private_node_map()[function];
    record.module = mapping.module;
    record.functions_by_route[route_id] = server_node;
    if (record.server_function == nullptr) {
      record.server_function = server_node;
      record.server_owner = server_owner;
    }
  }
  *resolved = server_node;
  return CUDA_SUCCESS;
}

extern "C" CUfunction
lupine_translate_private_function_for_rpc(CUfunction function) {
  return lupine_translate_private_function(function);
}

static bool lupine_device_attribute_is_virtualized(CUdevice_attribute attrib) {
  switch (attrib) {
  case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS:
  case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES:
  case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS:
  case CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST:
    return true;
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
  if (conn == nullptr ||
      !lupine_device_snapshot_attempts().insert(conn, true)) {
    return;
  }
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (rpc_write_start_request(conn, LUPINE_RPC_lupineDeviceSnapshot) < 0 ||
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
    CUdevice local_dev =
        lupine_local_device_for_remote(conn, static_cast<CUdevice>(ordinal));
    if (local_dev < 0) {
      continue;
    }
    for (uint32_t pair = 0; pair < pair_count; ++pair) {
      int attrib = static_cast<int>(pairs[pair * 2]);
      int value = static_cast<int>(pairs[pair * 2 + 1]);
      if (lupine_device_attribute_is_virtualized(
              static_cast<CUdevice_attribute>(attrib))) {
        value = 0;
      }
      lupine_device_attribute_cache().insert_or_assign(
          lupine_device_attribute_key{static_cast<int>(local_dev), attrib},
          value);
    }
    info.name[sizeof(info.name) - 1] = '\0';
    lupine_device_snapshot_cache().insert_or_assign(static_cast<int>(local_dev),
                                                    info);
  }
  rpc_read_end(conn);
}

extern "C" CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                         CUdevice dev) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(int *, CUdevice_attribute, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceGetAttribute");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return real(pi, attrib, remote_dev);
  }
  lupine_device_attribute_key key{static_cast<int>(dev),
                                  static_cast<int>(attrib)};
  if (lupine_device_attribute_cache().find(key, *pi)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_prefill_device_snapshot(conn);
  if (lupine_device_attribute_cache().find(key, *pi)) {
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  int value = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetAttribute) < 0 ||
      rpc_write(conn, &attrib, sizeof(attrib)) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(remote_dev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    if (lupine_device_attribute_is_virtualized(attrib)) {
      value = 0;
    }
    lupine_device_attribute_cache().insert_or_assign(key, value);
    *pi = value;
  }
  return return_value;
}

extern "C" CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (name == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(char *, int, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceGetName");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(name, len, remote_dev);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (len > 0) {
    lupine_prefill_device_snapshot(conn);
    lupine_device_snapshot_info info;
    if (lupine_device_snapshot_cache().find(static_cast<int>(dev), info)) {
      snprintf(name, static_cast<size_t>(len), "%s", info.name);
      return CUDA_SUCCESS;
    }
  }
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetName) < 0 ||
      rpc_write(conn, &len, sizeof(int)) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(name, len * sizeof(char)), false) ||
      (len * sizeof(char) != 0 &&
       rpc_read(conn, name, len * sizeof(char)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_mark_host_range_clean(name, len * sizeof(char));
  return return_value;
}

extern "C" CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
  if (uuid == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUuuid *, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceGetUuid_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(uuid, remote_dev);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_prefill_device_snapshot(conn);
  lupine_device_snapshot_info info;
  if (lupine_device_snapshot_cache().find(static_cast<int>(dev), info)) {
    *uuid = info.uuid;
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetUuid_v2) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(uuid, 16 * sizeof(CUuuid)), false) ||
      rpc_read(conn, uuid, 16) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_mark_host_range_clean(uuid, 16 * sizeof(CUuuid));
  return return_value;
}

extern "C" CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  if (bytes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(size_t *, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceTotalMem_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(bytes, remote_dev);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_prefill_device_snapshot(conn);
  lupine_device_snapshot_info info;
  if (lupine_device_snapshot_cache().find(static_cast<int>(dev), info)) {
    *bytes = static_cast<size_t>(info.total_mem);
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceTotalMem_v2) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, bytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
extern "C" CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  return cuDeviceGetUuid_v2(uuid, dev);
}

#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
extern "C" CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  return cuDeviceTotalMem_v2(bytes, dev);
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  lupine_invalidate_primary_ctx_state(dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxRetain", &return_value, pctx, remote_dev)) {
    if (return_value == CUDA_SUCCESS && pctx != nullptr) {
      lupine_note_context_owner_route(*pctx, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRetain) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && pctx != nullptr) {
    lupine_note_context_owner_route(*pctx, route);
  }
  return return_value;
}

extern "C" CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  lupine_invalidate_primary_ctx_state(dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxRelease_v2", &return_value, remote_dev)) {
    if (return_value == CUDA_SUCCESS) {
      lupine_invalidate_current_context_cache();
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRelease_v2) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_invalidate_current_context_cache();
  }
  return return_value;
}

extern "C" CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev,
                                                  unsigned int flags) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  lupine_invalidate_primary_ctx_state(dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxSetFlags_v2", &return_value, remote_dev,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxSetFlags_v2) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice, unsigned int *, int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxGetState", &return_value, remote_dev, flags,
          active)) {
    return return_value;
  }
  if (flags != nullptr && active != nullptr) {
    lupine_primary_ctx_state cached;
    if (lupine_primary_ctx_state_cache().find(static_cast<int>(dev), cached)) {
      *flags = cached.flags;
      *active = cached.active;
      return CUDA_SUCCESS;
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxGetState) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, active, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && flags != nullptr && active != nullptr) {
    lupine_primary_ctx_state_cache().insert_or_assign(
        static_cast<int>(dev), lupine_primary_ctx_state{*flags, *active});
  }
  return return_value;
}

extern "C" CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  lupine_invalidate_primary_ctx_state(dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxReset_v2", &return_value, remote_dev)) {
    if (return_value == CUDA_SUCCESS) {
      lupine_invalidate_current_context_cache();
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxReset_v2) < 0 ||
      rpc_write(conn, &remote_dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_invalidate_current_context_cache();
  }
  return return_value;
}

#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  return cuDevicePrimaryCtxRelease_v2(dev);
}

#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
extern "C" CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev,
                                               unsigned int flags) {
  return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
extern "C" CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
  return cuDevicePrimaryCtxReset_v2(dev);
}

static CUresult lupine_cuGetParamInfo_cached(uintptr_t handle,
                                             size_t param_index,
                                             size_t *param_offset,
                                             size_t *param_size, bool kernel) {
  if (param_offset == nullptr || param_size == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUfunction function = reinterpret_cast<CUfunction>(handle);
  lupine_route route = lupine_route_for_function(function);
  if (lupine_route_is_local(route)) {
    using func_info_fn_t = CUresult (*)(CUfunction, size_t, size_t *, size_t *);
    using kernel_info_fn_t = CUresult (*)(CUkernel, size_t, size_t *, size_t *);
    if (kernel) {
      auto real = lupine_real_cuda_fn<kernel_info_fn_t>("cuKernelGetParamInfo");
      return real == nullptr ? CUDA_ERROR_NOT_SUPPORTED
                             : real(reinterpret_cast<CUkernel>(handle),
                                    param_index, param_offset, param_size);
    }
    auto real = lupine_real_cuda_fn<func_info_fn_t>("cuFuncGetParamInfo");
    return real == nullptr
               ? CUDA_ERROR_NOT_SUPPORTED
               : real(function, param_index, param_offset, param_size);
  }

  lupine_param_info_key key{handle, param_index, kernel};
  lupine_param_info_value cached;
  if (lupine_param_info_cache().find(key, cached)) {
    *param_offset = cached.offset;
    *param_size = cached.size;
    return CUDA_SUCCESS;
  }

  size_t param_count = 0;
  if (lupine_param_layout_count_cache().find(
          lupine_param_layout_key{handle, kernel}, param_count) &&
      param_index >= param_count) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  size_t offset = 0;
  size_t size = 0;
  conn_t *conn = lupine_route_remote_conn(route);
  int rpc = kernel ? RPC_cuKernelGetParamInfo : RPC_cuFuncGetParamInfo;
  if (conn == nullptr || rpc_write_start_request(conn, rpc) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &param_index, sizeof(param_index)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &offset, sizeof(offset)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
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

extern "C" CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
  if (pFunc == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcontext current_context = nullptr;
  CUresult context_status = cuCtxGetCurrent(&current_context);
  if (context_status != CUDA_SUCCESS) {
    return context_status;
  }
  lupine_route route = lupine_route_for_default();
  if (route.kind == LUPINE_ROUTE_INVALID) {
    route = lupine_route_for_function(reinterpret_cast<CUfunction>(kernel));
  }
  CUfunction route_kernel_function = reinterpret_cast<CUfunction>(kernel);
  CUresult resolve_status = lupine_resolve_library_kernel_for_route(
      route_kernel_function, route, &route_kernel_function);
  if (resolve_status != CUDA_SUCCESS) {
    return resolve_status;
  }
  CUkernel route_kernel = reinterpret_cast<CUkernel>(route_kernel_function);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUfunction *, CUkernel);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuKernelGetFunction");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(pFunc, route_kernel);
    if (result == CUDA_SUCCESS) {
      lupine_note_function_owner_route(*pFunc, route);
    }
    return result;
  }
  lupine_kernel_function_key key{lupine_route_identity(route), current_context,
                                 route_kernel};
  if (lupine_kernel_function_cache().find(key, *pFunc)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction function = nullptr;
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelGetFunction) < 0 ||
      rpc_write(conn, &route_kernel, sizeof(route_kernel)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &function, sizeof(function)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_function_owner(function, conn);
    return_value = lupine_warm_func_param_info(function);
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_kernel_function_cache().insert_or_assign(key, function);
    *pFunc = function;
  }
  return return_value;
}

extern "C" CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                                         CUkernel kernel, CUdevice dev) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  using real_fn_t =
      CUresult (*)(int *, CUfunction_attribute, CUkernel, CUdevice);
  CUresult return_value;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuKernelGetAttribute",
                                                  &return_value, pi, attrib,
                                                  kernel, dev)) {
    return return_value;
  }
  lupine_kernel_attribute_key key{lupine_route_identity(route), kernel,
                                  static_cast<int>(attrib),
                                  static_cast<int>(dev)};
  if (lupine_kernel_attribute_cache().find(key, *pi)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  int value = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelGetAttribute) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &attrib, sizeof(attrib)) < 0 ||
      rpc_write(conn, &kernel, sizeof(kernel)) < 0 ||
      rpc_write(conn, &dev, sizeof(dev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_kernel_attribute_cache().insert_or_assign(key, value);
    *pi = value;
  }
  return return_value;
}

extern "C" CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                       CUfunction hfunc) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUfunction translated = lupine_translate_private_function(hfunc);
  lupine_route route = lupine_route_for_function(translated);
  using real_fn_t = CUresult (*)(int *, CUfunction_attribute, CUfunction);
  CUresult return_value;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncGetAttribute", &return_value, pi, attrib, translated)) {
    return return_value;
  }
  lupine_function_attribute_key key{lupine_route_identity(route), translated,
                                    static_cast<int>(attrib)};
  if (lupine_function_attribute_cache().find(key, *pi)) {
    return CUDA_SUCCESS;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  int value = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncGetAttribute) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &attrib, sizeof(attrib)) < 0 ||
      rpc_write(conn, &translated, sizeof(translated)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_function_attribute_cache().insert_or_assign(key, value);
    *pi = value;
  }
  return return_value;
}

extern "C" void lupine_invalidate_kernel_attribute_cache() {
  lupine_kernel_attribute_cache().clear();
}

extern "C" void lupine_kernel_attribute_cache_erase(int route_id,
                                                    CUkernel kernel, int attrib,
                                                    int dev) {
  lupine_kernel_attribute_cache().erase(
      lupine_kernel_attribute_key{route_id, kernel, attrib, dev});
}

extern "C" void lupine_invalidate_function_attribute_cache() {
  lupine_function_attribute_cache().clear();
}

static CUresult lupine_cuOccupancy_cached(int *numBlocks, CUfunction func,
                                          int blockSize, size_t dynamicSMemSize,
                                          unsigned int flags, bool with_flags) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(func);
  lupine_route route = lupine_route_for_function(translated);
  if (lupine_route_is_local(route)) {
    const char *symbol =
        with_flags ? "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
                   : "cuOccupancyMaxActiveBlocksPerMultiprocessor";
    using real_fn_t =
        CUresult (*)(int *, CUfunction, int, size_t, unsigned int);
    using real_no_flags_fn_t = CUresult (*)(int *, CUfunction, int, size_t);
    if (with_flags) {
      auto real = lupine_real_cuda_fn<real_fn_t>(symbol);
      if (real == nullptr) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      return real(numBlocks, translated, blockSize, dynamicSMemSize, flags);
    }
    auto real = lupine_real_cuda_fn<real_no_flags_fn_t>(symbol);
    return real == nullptr
               ? CUDA_ERROR_DEVICE_UNAVAILABLE
               : real(numBlocks, translated, blockSize, dynamicSMemSize);
  }
  lupine_occupancy_key key{lupine_route_identity(route),
                           translated,
                           blockSize,
                           dynamicSMemSize,
                           flags,
                           with_flags};
  if (lupine_occupancy_cache().find(key, *numBlocks)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  int remote_num_blocks = 0;
  int opcode = with_flags
                   ? RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
                   : RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor;
  if (conn == nullptr || rpc_write_start_request(conn, opcode) < 0 ||
      rpc_write(conn, &remote_num_blocks, sizeof(remote_num_blocks)) < 0 ||
      rpc_write(conn, &translated, sizeof(translated)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(blockSize)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      (with_flags && rpc_write(conn, &flags, sizeof(flags)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_num_blocks, sizeof(remote_num_blocks)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_occupancy_cache().insert_or_assign(key, remote_num_blocks);
    *numBlocks = remote_num_blocks;
  }
  return return_value;
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  return lupine_cuOccupancy_cached(numBlocks, func, blockSize, dynamicSMemSize,
                                   0, false);
}

extern "C" CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return lupine_cuOccupancy_cached(numBlocks, func, blockSize, dynamicSMemSize,
                                   flags, true);
}

static bool lupine_is_private_function(CUfunction function) {
  std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
  return lupine_private_node_map().find(function) !=
         lupine_private_node_map().end();
}

static bool lupine_is_library_kernel(CUfunction function) {
  std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
  return lupine_library_kernels().find(reinterpret_cast<CUkernel>(function)) !=
         lupine_library_kernels().end();
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
    CUmodule route_module = nullptr;
    if (lupine_load_recorded_module_on_route(module, route, &route_module) ==
            CUDA_SUCCESS &&
        route_module != nullptr) {
      module = route_module;
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuPrivateGetModuleNode) < 0 ||
      rpc_write(conn, &context, sizeof(context)) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, server_node, sizeof(*server_node)) < 0 ||
      rpc_read(conn, server_owner, sizeof(*server_owner)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
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
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 4) {
    if (arg1 == 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    uint64_t output = arg0 == 0 ? 0 : 1;
    *reinterpret_cast<uint64_t *>(arg1) = output;
    LUPINE_TRACE_LOG("LUPINE private 2131[4] output=" << output);
    return CUDA_SUCCESS;
  }

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 51) {
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

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 39) {
    if (arg1 == 0 || arg2 == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    *reinterpret_cast<uint64_t *>(arg2) = 0;
    return CUDA_SUCCESS;
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

    auto &node_pool = lupine_private_6e16_node_pool();
    unsigned int node_index = lupine_private_6e16_next_node().fetch_add(
                                  1, std::memory_order_relaxed) %
                              (sizeof(node_pool) / sizeof(node_pool[0]));
    unsigned char *client_node = node_pool[node_index];
    memset(client_node, 0, sizeof(node_pool[node_index]));
    *reinterpret_cast<uint64_t *>(client_node + 0x0) = 0x100000001ULL;
    *reinterpret_cast<uint64_t *>(client_node + 0x8) = server_owner;
    *reinterpret_cast<uint64_t *>(client_node + 0x10) = 0xc;
    *reinterpret_cast<uint64_t *>(client_node + 0x18) = 1;
    *reinterpret_cast<uint64_t *>(client_node + 0x28) = 0x80;
    *reinterpret_cast<uint32_t *>(client_node + 0x370) = 2;
    *reinterpret_cast<uint64_t *>(client_node + 0x480) = 0;
    {
      std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
      int route_id =
          lupine_route_identity(lupine_route_for_function(server_node));
      if (route_id == -2) {
        lupine_route node_route =
            arg0 != 0
                ? lupine_route_for_context(reinterpret_cast<CUcontext>(arg0))
                : lupine_route_for_module(reinterpret_cast<CUmodule>(arg1));
        route_id = lupine_route_identity(node_route);
      }
      auto &mapping =
          lupine_private_node_map()[reinterpret_cast<CUfunction>(client_node)];
      mapping.server_function = server_node;
      mapping.server_owner = server_owner;
      mapping.module = reinterpret_cast<CUmodule>(arg1);
      if (route_id != -2) {
        mapping.functions_by_route[route_id] = server_node;
      }
    }
    LUPINE_TRACE_LOG("LUPINE private 6e16[7] mapped client_node["
                     << node_index << "]=" << static_cast<void *>(client_node)
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
  static bool enabled = [] {
    const char *value = getenv("LUPINE_STUB_PRIVATE_EXPORTS");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

static bool lupine_remote_private_exports_enabled() {
  static bool enabled = [] {
    const char *value = getenv("LUPINE_REMOTE_PRIVATE_EXPORTS");
    return value == nullptr || strcmp(value, "0") != 0;
  }();
  return enabled;
}

static std::atomic<bool> &lupine_private_export_tables_active_flag() {
  static std::atomic<bool> active{false};
  return active;
}

static bool lupine_private_export_remap_active() {
  return lupine_stub_private_exports_enabled() ||
         lupine_private_export_tables_active_flag().load(
             std::memory_order_relaxed);
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

static void *lupine_make_private_export_stub(int slot, const char *table_name) {
#if defined(__x86_64__)
  constexpr size_t stub_size = 52;
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }

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
  return code;
#elif defined(__aarch64__)
  // AAPCS64 mirror of the x86-64 thunk above: the six incoming integer/pointer
  // arguments are shifted up two slots and the (slot, table_name) pair is
  // injected in front, then we tail-branch to the handler. The handler takes
  // eight integer arguments, all of which fit in x0-x7, so no stack shuffling
  // (and therefore no frame at all) is required and x30 stays untouched.
  //
  //   stub(a0..a5) -> lupine_private_export_slot_called(slot, table_name,
  //                                                     a0..a5)
  constexpr size_t stub_words = 17;
  constexpr size_t stub_size = stub_words * sizeof(uint32_t);
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }

  void *handler = reinterpret_cast<void *>(&lupine_private_export_slot_called);
  uint32_t words[stub_words];
  size_t n = 0;
  // Shift x0-x5 up into x2-x7. This MUST run highest-destination first so a
  // source register is never clobbered before it is read.
  // `mov Xd, Xm` is `orr Xd, xzr, Xm` = 0xaa0003e0 | (Rm << 16) | Rd.
  words[n++] = 0xaa0503e7u; // mov x7, x5
  words[n++] = 0xaa0403e6u; // mov x6, x4
  words[n++] = 0xaa0303e5u; // mov x5, x3
  words[n++] = 0xaa0203e4u; // mov x4, x2
  words[n++] = 0xaa0103e3u; // mov x3, x1
  words[n++] = 0xaa0003e2u; // mov x2, x0
  // w0 = slot. `int` is 32-bit, and writing w0 zeroes the upper half of x0,
  // which matches the x86 `mov edi, imm32`.
  uint32_t slot_bits = static_cast<uint32_t>(slot);
  // movz w0, #imm16
  words[n++] = 0x52800000u | ((slot_bits & 0xffffu) << 5) | 0u;
  // movk w0, #imm16, lsl #16
  words[n++] = 0x72a00000u | (((slot_bits >> 16) & 0xffffu) << 5) | 0u;
  // x1 = table_name
  lupine_a64_emit_mov_imm64(&words[n], 1,
                            reinterpret_cast<uint64_t>(table_name));
  n += 4;
  // x16 = handler (x16/IP0 is the intra-procedure-call scratch register, so it
  // is free to clobber across this tail branch).
  lupine_a64_emit_mov_imm64(&words[n], 16, reinterpret_cast<uint64_t>(handler));
  n += 4;
  words[n++] = 0xd61f0200u; // br x16
  memcpy(code, words, stub_size);
  // Instruction-cache maintenance is mandatory on AArch64: the writes above
  // land in the data cache and are not otherwise visible to the instruction
  // fetcher.
  __builtin___clear_cache(reinterpret_cast<char *>(code),
                          reinterpret_cast<char *>(code + stub_size));
  return code;
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
  lupine_private_export_tables_active_flag().store(true,
                                                   std::memory_order_relaxed);
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

  conn_t *conn = rpc_client_get_connection(0);
  if (conn == nullptr) {
    return nullptr;
  }

  CUresult result = CUDA_ERROR_UNKNOWN;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[LUPINE_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (rpc_write_start_request(conn, LUPINE_RPC_cuGetExportTableMetadata) < 0 ||
      rpc_write(conn, uuid->bytes, sizeof(uuid->bytes)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 ||
      rpc_read(conn, &byte_size, sizeof(byte_size)) < 0 ||
      rpc_read(conn, &slot_count, sizeof(slot_count)) < 0 ||
      rpc_read(conn, &trusted, sizeof(trusted)) < 0) {
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
    return nullptr;
  }

  std::string uuid_hex = lupine_uuid_hex(uuid);
  std::vector<uint64_t> code_hashes(hashes, hashes + slot_count);
  LUPINE_TRACE_LOG("LUPINE remote cuGetExportTable metadata uuid="
                   << uuid_hex << " bytes=" << byte_size
                   << " slots=" << slot_count);
  return lupine_make_private_export_table(
      uuid_hex.c_str(), static_cast<size_t>(byte_size), code_hashes);
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
// process. unordered_map nodes are stable, so c_str() stays valid; a present
// but empty entry records a code the driver rejected.
static CUresult lupine_error_string_lookup(int rpc_op, const char *symbol,
                                           bool want_name, CUresult error,
                                           const char **pStr) {
  if (pStr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = nullptr;

  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUresult, const char **);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, symbol, &return_value,
                                                  error, pStr)) {
    return return_value;
  }

  // The only hardcoded string: this is the shim's own most common error, and
  // its occurrence means the forward path below may already be gone.
  if (error == CUDA_ERROR_DEVICE_UNAVAILABLE) {
    *pStr = want_name ? "CUDA_ERROR_DEVICE_UNAVAILABLE"
                      : "CUDA-capable device(s) is/are busy or unavailable";
    return CUDA_SUCCESS;
  }

  // Forward every other code to the server's driver. The result is held in a
  // per-thread buffer so the returned pointer stays valid until this thread's
  // next lookup; a length past any real driver string means a desynced stream.
  thread_local std::string name_result;
  thread_local std::string description_result;
  std::string &result = want_name ? name_result : description_result;

  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = lupine_route_remote_conn(route);
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, rpc_op) < 0 ||
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
  result = std::move(text);
  *pStr = result.c_str();
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

CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuEventDestroy_v2(CUevent hEvent);
CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd);
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree_v2(CUdeviceptr dptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);
CUresult cuStreamSynchronize(CUstream hStream);
extern "C" CUresult cuStreamQuery_ptsz(CUstream hStream);
extern "C" CUresult cuStreamSynchronize_ptsz(CUstream hStream);

extern "C" CUresult
lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream, bool async) {
  LUPINE_TRACE_LOG("LUPINE cross-route D2D via client dst="
                   << reinterpret_cast<void *>(dstDevice)
                   << " src=" << reinterpret_cast<void *>(srcDevice)
                   << " bytes=" << ByteCount << " async=" << async
                   << " stream=" << hStream);
  if (ByteCount == 0) {
    return CUDA_SUCCESS;
  }

  constexpr size_t chunk_size = 16 * 1024 * 1024;
  std::vector<unsigned char> staging(std::min(ByteCount, chunk_size));
  size_t offset = 0;
  while (offset < ByteCount) {
    size_t chunk = std::min(staging.size(), ByteCount - offset);
    CUresult result =
        cuMemcpyDtoH_v2(staging.data(), srcDevice + offset, chunk);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (async) {
      result = cuMemcpyHtoDAsync_v2(dstDevice + offset, staging.data(), chunk,
                                    hStream);
    } else {
      result = cuMemcpyHtoD_v2(dstDevice + offset, staging.data(), chunk);
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offset += chunk;
  }
  return CUDA_SUCCESS;
}

#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
extern "C" CUresult cuStreamDestroy(CUstream hStream) {
  return cuStreamDestroy_v2(hStream);
}

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif
extern "C" CUresult cuEventDestroy(CUevent hEvent) {
  return cuEventDestroy_v2(hEvent);
}

#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                       CUevent hEnd) {
  return cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd);
}

extern "C" CUresult cuMemPoolSetAttribute(CUmemoryPool pool,
                                          CUmemPool_attribute attr,
                                          void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &value_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_memory_pool(pool);
  using real_fn_t = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolSetAttribute", &local_result, pool, attr, value)) {
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolSetAttribute) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_write(conn, value, value_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static thread_local CUcontext lupine_current_context = nullptr;
static thread_local CUcontext lupine_default_context_hint = nullptr;
static std::atomic<CUcontext> lupine_global_default_context_hint{nullptr};
static thread_local auto *lupine_context_stack = new std::vector<CUcontext>();

CUcontext lupine_current_context_hint() { return lupine_current_context; }

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
  lupine_forget_context_owner(ctx);
  if (lupine_current_context == ctx) {
    lupine_current_context = nullptr;
  }
  if (lupine_default_context_hint == ctx) {
    lupine_default_context_hint = nullptr;
  }
  CUcontext global_hint =
      lupine_global_default_context_hint.load(std::memory_order_relaxed);
  if (global_hint == ctx) {
    lupine_global_default_context_hint.store(nullptr,
                                             std::memory_order_relaxed);
  }
  lupine_context_stack->erase(std::remove(lupine_context_stack->begin(),
                                          lupine_context_stack->end(), ctx),
                              lupine_context_stack->end());
}

// CUDA IPC shareable handles of type POSIX fd cannot cross the wire as raw
// fd numbers. The server child exports the real fd, parks it with the parent
// broker under a random token, and the client hands the application a local
// proxy fd wrapping that token (see ipc.h). Import reverses the exchange.

CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                      CUmemGenericAllocationHandle handle,
                                      CUmemAllocationHandleType handleType,
                                      unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(void *, CUmemGenericAllocationHandle,
                                 CUmemAllocationHandleType, unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemExportToShareableHandle", &return_value, shareableHandle,
          handle, handleType, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_ipc_token token = {};
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemExportToShareableHandle) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && shareableHandle != nullptr) {
    int proxy_fd = -1;
    if (lupine_ipc_create_proxy_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                   &proxy_fd) < 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    *reinterpret_cast<int *>(shareableHandle) = proxy_fd;
  }
  return return_value;
}

CUresult
cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                               void *osHandle,
                               CUmemAllocationHandleType shHandleType) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemGenericAllocationHandle *, void *,
                                 CUmemAllocationHandleType);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemImportFromShareableHandle", &return_value, handle,
          osHandle, shHandleType)) {
    return return_value;
  }
  uint32_t kind = 0;
  lupine_ipc_token token = {};
  int proxy_fd = static_cast<int>(reinterpret_cast<uintptr_t>(osHandle));
  if (lupine_ipc_read_proxy_fd(proxy_fd, &kind, &token) < 0 ||
      kind != LUPINE_IPC_FD_KIND_VMM_ALLOCATION) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemImportFromShareableHandle) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &shHandleType, sizeof(shHandleType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, handle, sizeof(*handle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool,
                                          CUmemAllocationHandleType handleType,
                                          unsigned long long flags) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  using real_fn_t = CUresult (*)(void *, CUmemoryPool,
                                 CUmemAllocationHandleType, unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolExportToShareableHandle", &return_value, handle_out,
          pool, handleType, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_ipc_token token = {};
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolExportToShareableHandle) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && handle_out != nullptr) {
    int proxy_fd = -1;
    if (lupine_ipc_create_proxy_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                   &proxy_fd) < 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    *reinterpret_cast<int *>(handle_out) = proxy_fd;
  }
  return return_value;
}

CUresult
cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle,
                                   CUmemAllocationHandleType handleType,
                                   unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool *, void *,
                                 CUmemAllocationHandleType, unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolImportFromShareableHandle", &return_value, pool_out,
          handle, handleType, flags)) {
    if (return_value == CUDA_SUCCESS && pool_out != nullptr) {
      lupine_note_memory_pool_owner_route(*pool_out, route);
    }
    return return_value;
  }
  uint32_t kind = 0;
  lupine_ipc_token token = {};
  int proxy_fd = static_cast<int>(reinterpret_cast<uintptr_t>(handle));
  if (lupine_ipc_read_proxy_fd(proxy_fd, &kind, &token) < 0 ||
      kind != LUPINE_IPC_FD_KIND_MEMORY_POOL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolImportFromShareableHandle) <
          0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pool_out, sizeof(*pool_out)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
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

extern "C" void lupine_note_ctx_create(CUcontext ctx, conn_t *conn) {
  lupine_note_context_owner(ctx, conn);
  lupine_lane_context_cache_store(
      lupine_route_identity(lupine_remote_route_for_conn(conn)), ctx);
  lupine_context_stack->push_back(lupine_current_context);
  lupine_current_context = ctx;
  if (ctx != nullptr) {
    lupine_default_context_hint = ctx;
    lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
  }
}

extern "C" void lupine_note_ctx_create_route(CUcontext ctx,
                                             lupine_route route) {
  lupine_note_context_owner_route(ctx, route);
  lupine_lane_context_cache_store(lupine_route_identity(route), ctx);
  lupine_context_stack->push_back(lupine_current_context);
  lupine_current_context = ctx;
  if (ctx != nullptr) {
    lupine_default_context_hint = ctx;
    lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
  }
}

extern "C" CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  lupine_context_stack->push_back(lupine_current_context);
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    lupine_current_context = ctx;
    if (ctx != nullptr) {
      lupine_default_context_hint = ctx;
      lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
    }
  } else {
    lupine_context_stack->pop_back();
  }
  return result;
}

#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif
extern "C" CUresult cuCtxPushCurrent(CUcontext ctx) {
  return cuCtxPushCurrent_v2(ctx);
}

extern "C" CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  CUcontext popped = lupine_current_context;
  if (pctx != nullptr) {
    *pctx = popped;
  }
  CUcontext previous = nullptr;
  if (!lupine_context_stack->empty()) {
    previous = lupine_context_stack->back();
    lupine_context_stack->pop_back();
  }
  CUresult result = lupine_set_remote_current_context(previous);
  if (result == CUDA_SUCCESS) {
    lupine_current_context = previous;
    if (previous != nullptr) {
      lupine_default_context_hint = previous;
      lupine_global_default_context_hint.store(previous,
                                               std::memory_order_relaxed);
    }
  } else {
    lupine_context_stack->push_back(previous);
  }
  return result;
}

#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif
extern "C" CUresult cuCtxPopCurrent(CUcontext *pctx) {
  return cuCtxPopCurrent_v2(pctx);
}

extern "C" CUresult cuCtxSetCurrent(CUcontext ctx) {
  if (!lupine_cuda_is_initialized()) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    lupine_current_context = ctx;
    if (ctx != nullptr) {
      lupine_default_context_hint = ctx;
      lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
    }
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
    using real_fn_t = CUresult (*)(CUdevice *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxGetDevice");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return real(device);
  }
  if (lupine_current_context_device_cache_lookup(lupine_current_context,
                                                 device)) {
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUdevice remote_device = 0;
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetDevice) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    *device = lupine_local_device_for_remote(conn, remote_device);
    lupine_current_context_device_cache_insert(lupine_current_context, *device);
  }
  return return_value;
}

extern "C" CUresult cuMemPoolGetAttribute(CUmemoryPool pool,
                                          CUmemPool_attribute attr,
                                          void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &value_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_memory_pool(pool);
  using real_fn_t = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolGetAttribute", &local_result, pool, attr, value)) {
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolGetAttribute) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, value_size) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
extern "C" CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                         CUdeviceptr dptr) {
  return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
extern "C" CUresult cuMemGetInfo(size_t *free, size_t *total) {
  return cuMemGetInfo_v2(free, total);
}

extern "C" CUresult cuCtxGetStreamPriorityRange(int *leastPriority,
                                                int *greatestPriority) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(int *, int *);
    auto real = reinterpret_cast<real_fn_t>(
        lupine_real_cuda_symbol("cuCtxGetStreamPriorityRange"));
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(leastPriority, greatestPriority);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  int least = 0;
  int greatest = 0;
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetStreamPriorityRange) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &least, sizeof(least)) < 0 ||
      rpc_read(conn, &greatest, sizeof(greatest)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    if (leastPriority != nullptr) {
      *leastPriority = least;
    }
    if (greatestPriority != nullptr) {
      *greatestPriority = greatest;
    }
  }
  return return_value;
}

struct lupine_jit_client_state {
  std::vector<rpc_jit_output_binding> bindings;
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

static size_t lupine_jit_option_size(unsigned int numOptions,
                                     const CUjit_option *options,
                                     void *const *optionValues,
                                     CUjit_option size_option) {
  if (options == nullptr || optionValues == nullptr) {
    return 0;
  }
  for (unsigned int i = 0; i < numOptions; ++i) {
    if (options[i] == size_option) {
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(optionValues[i]));
    }
  }
  return 0;
}

static std::vector<rpc_jit_output_binding>
lupine_capture_jit_client_bindings(unsigned int numOptions,
                                   const CUjit_option *options,
                                   void *const *optionValues) {
  std::vector<rpc_jit_output_binding> bindings;
  if (options == nullptr || optionValues == nullptr) {
    return bindings;
  }
  size_t info_size = lupine_jit_option_size(numOptions, options, optionValues,
                                            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
  size_t error_size = lupine_jit_option_size(
      numOptions, options, optionValues, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
  for (unsigned int i = 0; i < numOptions; ++i) {
    if (options[i] == CU_JIT_WALL_TIME && optionValues[i] != nullptr) {
      bindings.push_back({options[i], optionValues[i], sizeof(float)});
    } else if (options[i] == CU_JIT_INFO_LOG_BUFFER &&
               optionValues[i] != nullptr && info_size != 0) {
      bindings.push_back({options[i], optionValues[i], info_size});
    } else if (options[i] == CU_JIT_ERROR_LOG_BUFFER &&
               optionValues[i] != nullptr && error_size != 0) {
      bindings.push_back({options[i], optionValues[i], error_size});
    }
  }
  return bindings;
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
    using real_fn_t =
        CUresult (*)(unsigned int, CUjit_option *, void **, CUlinkState *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLinkCreate_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(numOptions, options, optionValues, stateOut);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLinkCreate_v2) < 0 ||
      rpc_copy_alloc(conn,
                     static_cast<size_t>(numOptions) * sizeof(uintptr_t)) < 0 ||
      lupine_write_jit_options(conn, &numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, stateOut, sizeof(*stateOut)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
    lupine_jit_client_states()[*stateOut].bindings =
        lupine_capture_jit_client_bindings(numOptions, options, optionValues);
  }
  return return_value;
}

extern "C" CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type,
                                     void *data, size_t size, const char *name,
                                     unsigned int numOptions,
                                     CUjit_option *options,
                                     void **optionValues) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(CUlinkState, CUjitInputType, void *, size_t, const char *,
                     unsigned int, CUjit_option *, void **);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLinkAddData_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(state, type, data, size, name, numOptions,
                                  options, optionValues);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t name_len = name == nullptr ? 0 : strlen(name) + 1;
  auto bindings =
      lupine_capture_jit_client_bindings(numOptions, options, optionValues);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLinkAddData_v2) < 0 ||
      rpc_copy_alloc(conn,
                     static_cast<size_t>(numOptions) * sizeof(uintptr_t)) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      (size != 0 && rpc_write(conn, data, size) < 0) ||
      rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      lupine_write_jit_options(conn, &numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read_jit_outputs(conn, bindings) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type,
                                     const char *path, unsigned int numOptions,
                                     CUjit_option *options,
                                     void **optionValues) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUlinkState, CUjitInputType, const char *,
                                   unsigned int, CUjit_option *, void **);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLinkAddFile_v2");
    return real == nullptr
               ? CUDA_ERROR_DEVICE_UNAVAILABLE
               : real(state, type, path, numOptions, options, optionValues);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t path_len = path == nullptr ? 0 : strlen(path) + 1;
  void *file_mapping = MAP_FAILED;
  const void *file_payload = nullptr;
  size_t mapped_file_size = 0;
  uint64_t file_size = 0;
  uint8_t has_file_data = 0;
  auto bindings =
      lupine_capture_jit_client_bindings(numOptions, options, optionValues);
  int file_fd = open(path, O_RDONLY);
  if (file_fd < 0) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  // FILE_NOT_FOUND is reserved for open() failing.
  struct stat st = {};
  if (fstat(file_fd, &st) < 0) {
    close(file_fd);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (st.st_size <= 0) {
    close(file_fd);
    return CUDA_ERROR_INVALID_IMAGE;
  }
  mapped_file_size = static_cast<size_t>(st.st_size);
  file_mapping =
      mmap(nullptr, mapped_file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);
  close(file_fd);
  if (file_mapping == MAP_FAILED) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  file_payload = file_mapping;
  file_size = mapped_file_size;
  has_file_data = 1;
  bool failed = conn == nullptr ||
                rpc_write_start_request(conn, RPC_cuLinkAddFile_v2) < 0;
  if (!failed && (rpc_copy_alloc(conn, static_cast<size_t>(numOptions) *
                                           sizeof(uintptr_t)) < 0 ||
                  rpc_write(conn, &state, sizeof(state)) < 0 ||
                  rpc_write(conn, &type, sizeof(type)) < 0 ||
                  rpc_write(conn, &path_len, sizeof(path_len)) < 0 ||
                  (path_len != 0 && rpc_write(conn, path, path_len) < 0) ||
                  rpc_write(conn, &has_file_data, sizeof(has_file_data)) < 0 ||
                  rpc_write(conn, &file_size, sizeof(file_size)) < 0 ||
                  (file_size != 0 &&
                   rpc_write(conn, file_payload, mapped_file_size) < 0) ||
                  lupine_write_jit_options(conn, &numOptions, options,
                                           optionValues) < 0)) {
    failed = true;
  }
  if (!failed && (rpc_wait_for_response(conn) < 0 ||
                  rpc_read_jit_outputs(conn, bindings) < 0 ||
                  rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
                  rpc_read_end(conn) < 0)) {
    failed = true;
  }
  if (file_mapping != MAP_FAILED) {
    munmap(file_mapping, mapped_file_size);
  }
  if (failed) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLinkComplete(CUlinkState state, void **cubinOut,
                                   size_t *sizeOut) {
  if (cubinOut == nullptr || sizeOut == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUlinkState, void **, size_t *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLinkComplete");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(state, cubinOut, sizeOut);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t cubin_size = 0;
  std::vector<rpc_jit_output_binding> bindings;
  if (conn == nullptr ||
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
    bindings = jit_state.bindings;
    *cubinOut = jit_state.cubin.empty() ? nullptr : jit_state.cubin.data();
    *sizeOut = jit_state.cubin.size();
  }

  if (rpc_read_jit_outputs(conn, bindings) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuLinkDestroy(CUlinkState state) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUlinkState);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLinkDestroy");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(state);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuLinkDestroy) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
  lupine_jit_client_states().erase(state);
  return return_value;
}

#ifdef cuLinkCreate
#undef cuLinkCreate
#endif
extern "C" CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                                 void **optionValues, CUlinkState *stateOut) {
  return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

#ifdef cuLinkAddData
#undef cuLinkAddData
#endif
extern "C" CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
                                  void *data, size_t size, const char *name,
                                  unsigned int numOptions,
                                  CUjit_option *options, void **optionValues) {
  return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                          optionValues);
}

#ifdef cuLinkAddFile
#undef cuLinkAddFile
#endif
extern "C" CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type,
                                  const char *path, unsigned int numOptions,
                                  CUjit_option *options, void **optionValues) {
  return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

extern "C" int lupine_forward_remote_stdout(conn_t *conn) {
  uint64_t output_size = 0;
  if (rpc_read(conn, &output_size, sizeof(output_size)) < 0) {
    return -1;
  }
  if (output_size == 0) {
    return 0;
  }
  std::string output;
  output.resize(static_cast<size_t>(output_size));
  if (rpc_read(conn, output.data(), output.size()) < 0) {
    return -1;
  }
  fflush(stdout);
  std::cout.flush();
  return fwrite(output.data(), 1, output.size(), stdout) == output.size() ? 0
                                                                          : -1;
}

extern "C" int lupine_read_deferred_dtoh_copies(conn_t *conn) {
  uint32_t copy_count = 0;
  if (rpc_read(conn, &copy_count, sizeof(copy_count)) < 0) {
    return -1;
  }
  for (uint32_t i = 0; i < copy_count; ++i) {
    void *dst = nullptr;
    size_t bytes = 0;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
      return -1;
    }
    if (bytes == 0) {
      continue;
    }
    lupine_prepare_host_range_write(dst, bytes);
    if (rpc_read_payload(conn, dst, bytes) < 0) {
      lupine_mark_host_range_clean(dst, bytes);
      return -1;
    }
    lupine_mark_host_range_clean(dst, bytes);
  }
  return 0;
}

extern "C" CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                      unsigned int Flags) {
  lupine_route route = hStream == nullptr ? lupine_route_for_default()
                                          : lupine_route_for_stream(hStream);
  lupine_route event_route = lupine_route_for_event(hEvent);
  if (hEvent != nullptr && !lupine_routes_share_server(route, event_route)) {
    return cuEventSynchronize(hEvent);
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUstream, CUevent, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuStreamWaitEvent");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(hStream, hEvent, Flags);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamWaitEvent) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(hEvent)) < 0 ||
      rpc_write(conn, &Flags, sizeof(Flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

#ifdef cuStreamWaitEvent_ptsz
#undef cuStreamWaitEvent_ptsz
#endif
extern "C" CUresult cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent,
                                           unsigned int Flags) {
  return cuStreamWaitEvent(hStream, hEvent, Flags);
}

extern "C" CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventRecord", &return_value, hEvent, hStream)) {
    return return_value;
  }
  lupine_note_event_recorded(hEvent);
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuEventRecord) < 0 ||
      rpc_write(conn, &hEvent, sizeof(hEvent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

#ifdef cuEventRecord_ptsz
#undef cuEventRecord_ptsz
#endif
extern "C" CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream) {
  return cuEventRecord(hEvent, hStream);
}

extern "C" CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                           unsigned int flags) {
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent, CUstream, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventRecordWithFlags", &return_value, hEvent, hStream,
          flags)) {
    return return_value;
  }
  lupine_note_event_recorded(hEvent);
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuEventRecordWithFlags) < 0 ||
      rpc_write(conn, &hEvent, sizeof(hEvent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 || rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

#ifdef cuEventRecordWithFlags_ptsz
#undef cuEventRecordWithFlags_ptsz
#endif
extern "C" CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent,
                                                CUstream hStream,
                                                unsigned int flags) {
  return cuEventRecordWithFlags(hEvent, hStream, flags);
}

// Best-effort cache warming for events other than the one the caller queried.
// This is intentionally a separate RPC: failures must not affect the result
// already produced by RPC_cuEventQuery.
static void lupine_prefetch_event_queries(CUevent exclude, conn_t *conn) {
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  uint32_t count =
      lupine_collect_event_query_prefetch(exclude, conn, events, recorded);
  if (count == 0) {
    return;
  }

  CUresult results[kLupineEventQueryBatch];
  if (rpc_write_start_request(conn, LUPINE_RPC_lupineEventQueryBatch) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, events, count * sizeof(events[0])) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, results, count * sizeof(results[0])) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
  lupine_note_event_query_results(events, recorded, results, count);
}

extern "C" CUresult cuEventQuery(CUevent hEvent) {
  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  lupine_route route = lupine_route_for_event(hEvent);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuEventQuery",
                                                  &return_value, hEvent)) {
    return return_value == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                        : return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  uint64_t recorded = 0;
  if (!lupine_event_query_needed(hEvent, &recorded)) {
    return lupine_sync_mapped_device_to_host();
  }
  // Sampled before the request so a copy issued while it is in flight is not
  // mistaken for one the server already drained.
  uint64_t drained = lupine_async_dtoh_issued_count();
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (rpc_write_start_request(conn, RPC_cuEventQuery) < 0 ||
      rpc_write(conn, &hEvent, sizeof(hEvent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  lupine_note_event_query_results(&hEvent, &recorded, &result, 1);

  return_value = result;
  if (return_value == CUDA_SUCCESS) {
    lupine_note_async_dtoh_drained(drained);
    return_value = lupine_sync_mapped_device_to_host();
  }
  lupine_prefetch_event_queries(hEvent, conn);
  return return_value;
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
    using real_fn_t = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxCreate_v2");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(pctx, flags, dev);
    if (result == CUDA_SUCCESS) {
      lupine_note_ctx_create_route(*pctx, route);
    }
    return result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxCreate_v2) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &dev, sizeof(dev)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
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

static CUresult lupine_occupancy_max_potential_block_size(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags, bool with_flags) {
  if (minGridSize == nullptr || blockSize == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (blockSizeToDynamicSMemSize != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUfunction translated = lupine_translate_private_function(func);
  lupine_route route = lupine_route_for_function(translated);
  const char *symbol = with_flags ? "cuOccupancyMaxPotentialBlockSizeWithFlags"
                                  : "cuOccupancyMaxPotentialBlockSize";
  if (lupine_route_is_local(route)) {
    if (with_flags) {
      using real_fn_t =
          CUresult (*)(int *, int *, CUfunction, CUoccupancyB2DSize, size_t,
                       int, unsigned int);
      CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, symbol, &local_result, minGridSize, blockSize, translated,
              blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit,
              flags)) {
        return local_result;
      }
    } else {
      using real_fn_t = CUresult (*)(int *, int *, CUfunction,
                                     CUoccupancyB2DSize, size_t, int);
      CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, symbol, &local_result, minGridSize, blockSize, translated,
              blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)) {
        return local_result;
      }
    }
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  int op = with_flags ? RPC_cuOccupancyMaxPotentialBlockSizeWithFlags
                      : RPC_cuOccupancyMaxPotentialBlockSize;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &translated, sizeof(translated)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      rpc_write(conn, &blockSizeLimit, sizeof(blockSizeLimit)) < 0 ||
      (with_flags && rpc_write(conn, &flags, sizeof(flags)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, minGridSize, sizeof(int)) < 0 ||
      rpc_read(conn, blockSize, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                 CUfunction func,
                                 CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                 size_t dynamicSMemSize, int blockSizeLimit) {
  return lupine_occupancy_max_potential_block_size(
      minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
      blockSizeLimit, 0, false);
}

extern "C" CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  return lupine_occupancy_max_potential_block_size(
      minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
      blockSizeLimit, flags, true);
}

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

static bool lupine_pack_module_image(const void *image, uint32_t *kind,
                                     std::vector<unsigned char> *bytes) {
  if (image == nullptr || kind == nullptr || bytes == nullptr) {
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
      if (ehdr->e_phoff != 0 && ehdr->e_phentsize == sizeof(Elf64_Phdr)) {
        image_size =
            std::max(image_size, static_cast<size_t>(ehdr->e_phoff) +
                                     static_cast<size_t>(ehdr->e_phnum) *
                                         sizeof(Elf64_Phdr));
        const auto *phdrs =
            reinterpret_cast<const Elf64_Phdr *>(elf + ehdr->e_phoff);
        for (int i = 0; i < ehdr->e_phnum; ++i) {
          image_size =
              std::max(image_size, static_cast<size_t>(phdrs[i].p_offset) +
                                       static_cast<size_t>(phdrs[i].p_filesz));
        }
      }
      if (ehdr->e_shoff != 0 && ehdr->e_shentsize == sizeof(Elf64_Shdr)) {
        image_size =
            std::max(image_size, static_cast<size_t>(ehdr->e_shoff) +
                                     static_cast<size_t>(ehdr->e_shnum) *
                                         sizeof(Elf64_Shdr));
        const auto *shdrs =
            reinterpret_cast<const Elf64_Shdr *>(elf + ehdr->e_shoff);
        for (int i = 0; i < ehdr->e_shnum; ++i) {
          if (shdrs[i].sh_type != SHT_NOBITS) {
            image_size =
                std::max(image_size, static_cast<size_t>(shdrs[i].sh_offset) +
                                         static_cast<size_t>(shdrs[i].sh_size));
          }
        }
      }
      *kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
      bytes->assign(elf, elf + image_size);
      return true;
    }

    // Non-fatbin, non-ELF images are forwarded verbatim for the server's
    // driver to judge. Non-text bytes carry no length, so they cannot ship.
    const char *ptx = static_cast<const char *>(image);
    if (!lupine_image_looks_like_text(ptx)) {
      return false;
    }
    *kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
    bytes->assign(reinterpret_cast<const unsigned char *>(ptx),
                  reinterpret_cast<const unsigned char *>(ptx) +
                      std::strlen(ptx) + 1);
    return true;
  }
  size_t image_size = static_cast<size_t>(header->header_size) +
                      static_cast<size_t>(header->files_size);
  const auto *raw = static_cast<const unsigned char *>(fatbin);
  bytes->assign(raw, raw + image_size);
  return true;
}

extern "C" CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  if (module == nullptr || image == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uint32_t kind = 0;
  std::vector<unsigned char> image_bytes;
  if (!lupine_pack_module_image(image, &kind, &image_bytes)) {
    return CUDA_ERROR_INVALID_IMAGE;
  }

  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUmodule *, const void *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuModuleLoadData");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(module, image);
    if (result == CUDA_SUCCESS) {
      lupine_remember_loaded_module(*module);
      lupine_note_module_owner_route(*module, route);
      lupine_record_module_image(*module, route, kind, image_bytes.data(),
                                 image_bytes.size(), image);
    }
    return result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t image_size = image_bytes.size();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleLoadData) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write_payload(conn, image_bytes.data(), image_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, module, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_remember_loaded_module(*module);
    lupine_note_module_owner(*module, conn);
    lupine_record_module_image(*module, lupine_remote_route_for_conn(conn),
                               kind, image_bytes.data(), image_bytes.size(),
                               image);
  }
  return return_value;
}

extern "C" CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                                       unsigned int numOptions,
                                       CUjit_option *options,
                                       void **optionValues) {
  (void)options;
  (void)optionValues;
  (void)numOptions;
  return cuModuleLoadData(module, image);
}

struct lupine_wire_library_kernel_record {
  std::string name;
  CUkernel kernel = nullptr;
  CUfunction function = nullptr;
  uint32_t param_count = 0;
  std::vector<uint64_t> params;
};

static void lupine_prefill_library_snapshot(CUlibrary library, conn_t *conn) {
  if (library == nullptr || conn == nullptr) {
    return;
  }

  CUresult result = CUDA_ERROR_UNKNOWN;
  uint32_t table_count = 0;
  if (rpc_write_start_request(conn, LUPINE_RPC_lupineLibrarySnapshot) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 ||
      rpc_read(conn, &table_count, sizeof(table_count)) < 0) {
    return;
  }
  if (result != CUDA_SUCCESS) {
    (void)rpc_read_end(conn);
    return;
  }

  std::vector<lupine_wire_library_kernel_record> table;
  table.reserve(table_count);
  for (uint32_t i = 0; i < table_count; ++i) {
    lupine_wire_library_kernel_record record;
    uint32_t name_len = 0;
    if (rpc_read(conn, &name_len, sizeof(name_len)) < 0 || name_len == 0 ||
        name_len > 64 * 1024) {
      return;
    }
    record.name.resize(name_len);
    if (rpc_read(conn, record.name.data(), name_len) < 0 ||
        rpc_read(conn, &record.kernel, sizeof(record.kernel)) < 0 ||
        rpc_read(conn, &record.function, sizeof(record.function)) < 0 ||
        rpc_read(conn, &record.param_count, sizeof(record.param_count)) < 0) {
      return;
    }
    record.name.resize(name_len - 1);
    record.params.resize(static_cast<size_t>(record.param_count) * 2);
    if (record.param_count != 0 &&
        rpc_read(conn, record.params.data(),
                 record.params.size() * sizeof(uint64_t)) < 0) {
      return;
    }
    table.push_back(std::move(record));
  }
  if (rpc_read_end(conn) < 0) {
    return;
  }

  lupine_route remote_route = lupine_remote_route_for_conn(conn);
  CUcontext current_context = nullptr;
  (void)cuCtxGetCurrent(&current_context);
  for (const auto &record : table) {
    if (record.kernel == nullptr) {
      continue;
    }
    lupine_cache_param_layout(reinterpret_cast<uintptr_t>(record.kernel), true,
                              record.param_count, record.params);
    if (record.function != nullptr) {
      lupine_note_function_owner_route(record.function, remote_route);
      lupine_cache_param_layout(reinterpret_cast<uintptr_t>(record.function),
                                false, record.param_count, record.params);
      lupine_kernel_function_cache().insert_or_assign(
          lupine_kernel_function_key{lupine_route_identity(remote_route),
                                     current_context, record.kernel},
          record.function);
    }
    if (lupine_record_library_kernel(record.kernel, library,
                                     record.name.c_str(),
                                     remote_route) == CUDA_SUCCESS) {
      lupine_library_kernel_names().insert_or_assign(
          lupine_library_kernel_name_key{library, record.name}, record.kernel);
    }
  }
}

static void lupine_prefill_library_attribute_snapshot(CUlibrary library,
                                                      conn_t *conn) {
  if (library == nullptr || conn == nullptr) {
    return;
  }

  CUresult result = CUDA_ERROR_UNKNOWN;
  CUdevice remote_device = -1;
  if (rpc_write_start_request(conn, LUPINE_RPC_lupineLibraryAttributeSnapshot) <
          0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0) {
    return;
  }
  if (result != CUDA_SUCCESS) {
    (void)rpc_read_end(conn);
    return;
  }
  if (rpc_read(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0) {
    return;
  }
  if (result != CUDA_SUCCESS) {
    (void)rpc_read_end(conn);
    return;
  }

  unsigned int kernel_count = 0;
  if (rpc_read(conn, &kernel_count, sizeof(kernel_count)) < 0 ||
      kernel_count > 1024 * 1024) {
    return;
  }
  if (kernel_count == 0) {
    (void)rpc_read_end(conn);
    return;
  }
  if (rpc_read(conn, &result, sizeof(result)) < 0) {
    return;
  }
  if (result != CUDA_SUCCESS) {
    (void)rpc_read_end(conn);
    return;
  }

  lupine_route route = lupine_remote_route_for_conn(conn);
  for (unsigned int i = 0; i < kernel_count; ++i) {
    CUkernel kernel = nullptr;
    CUfunction function = nullptr;
    if (rpc_read(conn, &kernel, sizeof(kernel)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0) {
      return;
    }
    if (result == CUDA_SUCCESS) {
      if (rpc_read(conn, &function, sizeof(function)) < 0) {
        return;
      }
      if (function != nullptr) {
        lupine_note_function_owner_route(function, route);
      }
      if (lupine_read_function_attributes(conn, route, function) < 0) {
        return;
      }
    }
    if (lupine_read_kernel_attributes(conn, route, kernel, remote_device) < 0) {
      return;
    }
  }
  if (rpc_read_end(conn) < 0) {
    return;
  }
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
  std::vector<unsigned char> image_bytes;
  if (!lupine_pack_module_image(code, &kind, &image_bytes)) {
    const auto *wrapper = reinterpret_cast<const lupine_fatbin_wrapper *>(code);
    LUPINE_TRACE_LOG("LUPINE cuLibraryLoadData could not pack image"
                     << " magic=0x" << std::hex << wrapper->magic
                     << " version=0x" << wrapper->version << std::dec);
    return CUDA_ERROR_INVALID_IMAGE;
  }

  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(CUlibrary *, const void *, CUjit_option *, void **,
                     unsigned int, CUlibraryOption *, void **, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLibraryLoadData");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(library, code, jitOptions, jitOptionsValues,
                                  numJitOptions, libraryOptions,
                                  libraryOptionValues, numLibraryOptions);
  }

  auto bindings = lupine_capture_jit_client_bindings(numJitOptions, jitOptions,
                                                     jitOptionsValues);
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t image_size = image_bytes.size();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
      rpc_copy_alloc(conn, (static_cast<size_t>(numJitOptions) +
                            static_cast<size_t>(numLibraryOptions)) *
                               sizeof(uintptr_t)) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write_payload(conn, image_bytes.data(), image_size) < 0 ||
      lupine_write_jit_options(conn, &numJitOptions, jitOptions,
                               jitOptionsValues) < 0 ||
      lupine_write_library_options(conn, &numLibraryOptions, libraryOptions,
                                   libraryOptionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, library, sizeof(CUlibrary)) < 0 ||
      rpc_read_jit_outputs(conn, bindings) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_library_owner(*library, conn);
    lupine_record_library_image(*library, lupine_remote_route_for_conn(conn),
                                kind, image_bytes.data(), image_bytes.size(),
                                code);
    lupine_prefill_library_snapshot(*library, conn);
    lupine_prefill_library_attribute_snapshot(*library, conn);
  }
  return return_value;
}

static CUresult lupine_read_func_param_layout(CUfunction function,
                                              std::vector<size_t> *offsets,
                                              std::vector<size_t> *sizes) {
  if (offsets == nullptr || sizes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  offsets->clear();
  sizes->clear();
  for (uint32_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = cuFuncGetParamInfo(function, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      lupine_param_layout_count_cache().insert_or_assign(
          lupine_param_layout_key{reinterpret_cast<uintptr_t>(function), false},
          sizes->size());
      return CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offsets->push_back(offset);
    sizes->push_back(size);
  }
}

static CUresult lupine_read_kernel_param_layout(CUkernel kernel,
                                                std::vector<size_t> *offsets,
                                                std::vector<size_t> *sizes) {
  if (offsets == nullptr || sizes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  offsets->clear();
  sizes->clear();
  for (uint32_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = cuKernelGetParamInfo(kernel, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      lupine_param_layout_count_cache().insert_or_assign(
          lupine_param_layout_key{reinterpret_cast<uintptr_t>(kernel), true},
          sizes->size());
      return CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offsets->push_back(offset);
    sizes->push_back(size);
  }
}

static CUresult lupine_warm_func_param_info(CUfunction function) {
  for (size_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = cuFuncGetParamInfo(function, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      lupine_param_layout_count_cache().insert_or_assign(
          lupine_param_layout_key{reinterpret_cast<uintptr_t>(function), false},
          i);
      return CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
}

static CUresult lupine_warm_kernel_param_info(CUkernel kernel) {
  for (size_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = cuKernelGetParamInfo(kernel, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      lupine_param_layout_count_cache().insert_or_assign(
          lupine_param_layout_key{reinterpret_cast<uintptr_t>(kernel), true},
          i);
      return CUDA_SUCCESS;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
}

extern "C" void lupine_invalidate_function_caches() {
  lupine_param_info_cache().clear();
  lupine_param_layout_count_cache().clear();
  lupine_kernel_function_cache().clear();
  lupine_occupancy_cache().clear();
  lupine_kernel_attribute_cache().clear();
  lupine_function_attribute_cache().clear();
}

static CUresult lupine_resolve_launch_function_for_route(
    CUfunction requested_function, lupine_route route,
    CUfunction *route_function, CUfunction *launch_function) {
  CUresult result = lupine_resolve_library_kernel_for_route(
      requested_function, route, route_function);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = lupine_resolve_module_function_for_route(*route_function, route,
                                                    route_function);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = lupine_resolve_private_function_for_route(*route_function, route,
                                                     route_function);
  if (result == CUDA_SUCCESS) {
    *launch_function = *route_function;
  }
  return result;
}

extern "C" CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
               unsigned int gridDimZ, unsigned int blockDimX,
               unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra) {
  if (extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUfunction requested_function = f;
  bool kernel_handle = lupine_is_library_kernel(requested_function);
  lupine_route launch_route = hStream != nullptr
                                  ? lupine_route_for_stream(hStream)
                                  : lupine_route_for_default();
  CUfunction route_function;
  CUresult status = lupine_resolve_launch_function_for_route(
      requested_function, launch_route, &route_function, &f);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  lupine_route route = lupine_route_for_function(f);
  LUPINE_TRACE_LOG("LUPINE cuLaunchKernel f="
                   << f << " stream=" << hStream
                   << " launch_route=" << lupine_route_identity(launch_route)
                   << " function_route=" << lupine_route_identity(route)
                   << " grid=(" << gridDimX << "," << gridDimY << ","
                   << gridDimZ << ") block=(" << blockDimX << "," << blockDimY
                   << "," << blockDimZ << ")");
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(
        CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, CUstream, void **, void **);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLaunchKernel");
    return real == nullptr
               ? CUDA_ERROR_DEVICE_UNAVAILABLE
               : real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                      blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
  }

  std::vector<size_t> param_offsets;
  std::vector<size_t> param_sizes;
  status = kernel_handle
               ? lupine_read_kernel_param_layout(reinterpret_cast<CUkernel>(f),
                                                 &param_offsets, &param_sizes)
               : lupine_read_func_param_layout(f, &param_offsets, &param_sizes);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  for (size_t i = 0; i < param_sizes.size(); ++i) {
    if (kernelParams == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  lupine_route arg_route = lupine_route_from_known_kernel_deviceptr_args(
      kernelParams, param_sizes, launch_route);
  if (lupine_route_identity(arg_route) != lupine_route_identity(launch_route)) {
    launch_route = arg_route;
    status = lupine_resolve_launch_function_for_route(
        requested_function, launch_route, &route_function, &f);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    route = lupine_route_for_function(f);

    status =
        kernel_handle
            ? lupine_read_kernel_param_layout(reinterpret_cast<CUkernel>(f),
                                              &param_offsets, &param_sizes)
            : lupine_read_func_param_layout(f, &param_offsets, &param_sizes);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    for (size_t i = 0; i < param_sizes.size(); ++i) {
      if (kernelParams == nullptr || kernelParams[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
    }
    LUPINE_TRACE_LOG("LUPINE cuLaunchKernel rerouted by args f="
                     << f << " route=" << lupine_route_identity(route));
  }

  bool used_managed_mapping = false;
  uint32_t param_count = static_cast<uint32_t>(param_sizes.size());
  std::vector<CUdeviceptr> translated_params(param_count);
  std::vector<void *> rpc_params(param_count);
  status = lupine_sync_mapped_host_to_device_for_launch(
      kernelParams, param_sizes.data(), param_count, translated_params.data(),
      rpc_params.data(), &used_managed_mapping);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  bool sync_after_launch =
      used_managed_mapping &&
      (lupine_managed_kernel_requires_launch_sync(requested_function) ||
       lupine_managed_kernel_requires_launch_sync(route_function) ||
       lupine_managed_kernel_requires_launch_sync(f));
  conn_t *conn = lupine_route_remote_conn(route);
  // Fire-and-forget; launch errors are sticky and surface at the next sync.
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchKernel) < 0 ||
      rpc_copy_alloc(
          conn, lupine_rpc_param_info_storage_size(param_sizes.size())) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_write(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_write(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_write(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_write(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_write(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_write(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &kernel_handle, sizeof(kernel_handle)) < 0 ||
#if CUDA_VERSION >= 12000
      (kernel_handle
           ? lupine_write_kernel_param_values(
                 conn, reinterpret_cast<CUkernel>(f), rpc_params.data())
           : lupine_write_func_param_values(conn, f, rpc_params.data())) < 0
#else
      lupine_write_func_param_values(conn, f, rpc_params.data()) < 0
#endif
      || rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (sync_after_launch) {
    return cuStreamSynchronize(hStream);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                     void **kernelParams, void **extra) {
  if (config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
#if CUDA_VERSION < 11080
  return CUDA_ERROR_NOT_SUPPORTED;
#else
  if (extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUfunction requested_function = f;
  bool kernel_handle = lupine_is_library_kernel(requested_function);
  lupine_route launch_route = config->hStream != nullptr
                                  ? lupine_route_for_stream(config->hStream)
                                  : lupine_route_for_default();
  CUfunction route_function;
  CUresult status = lupine_resolve_launch_function_for_route(
      requested_function, launch_route, &route_function, &f);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  lupine_route route = lupine_route_for_function(f);
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(const CUlaunchConfig *, CUfunction, void **, void **);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLaunchKernelEx");
    return real == nullptr ? CUDA_ERROR_NOT_SUPPORTED
                           : real(config, f, kernelParams, extra);
  }

  std::vector<size_t> param_offsets;
  std::vector<size_t> param_sizes;
  status = kernel_handle
               ? lupine_read_kernel_param_layout(reinterpret_cast<CUkernel>(f),
                                                 &param_offsets, &param_sizes)
               : lupine_read_func_param_layout(f, &param_offsets, &param_sizes);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  for (size_t i = 0; i < param_sizes.size(); ++i) {
    if (kernelParams == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  lupine_route arg_route = lupine_route_from_known_kernel_deviceptr_args(
      kernelParams, param_sizes, launch_route);
  if (lupine_route_identity(arg_route) != lupine_route_identity(launch_route)) {
    launch_route = arg_route;
    status = lupine_resolve_launch_function_for_route(
        requested_function, launch_route, &route_function, &f);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    route = lupine_route_for_function(f);

    status =
        kernel_handle
            ? lupine_read_kernel_param_layout(reinterpret_cast<CUkernel>(f),
                                              &param_offsets, &param_sizes)
            : lupine_read_func_param_layout(f, &param_offsets, &param_sizes);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    for (size_t i = 0; i < param_sizes.size(); ++i) {
      if (kernelParams == nullptr || kernelParams[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
    }
  }

  bool used_managed_mapping = false;
  uint32_t param_count = static_cast<uint32_t>(param_sizes.size());
  std::vector<CUdeviceptr> translated_params(param_count);
  std::vector<void *> rpc_params(param_count);
  status = lupine_sync_mapped_host_to_device_for_launch(
      kernelParams, param_sizes.data(), param_count, translated_params.data(),
      rpc_params.data(), &used_managed_mapping);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  bool sync_after_launch =
      used_managed_mapping &&
      (lupine_managed_kernel_requires_launch_sync(requested_function) ||
       lupine_managed_kernel_requires_launch_sync(route_function) ||
       lupine_managed_kernel_requires_launch_sync(f));
  conn_t *conn = lupine_route_remote_conn(route);
  // Attribute-free launches are fire-and-forget like cuLaunchKernel; launch
  // errors are sticky and surface at the next sync. Launches carrying
  // attributes stay synchronous so attribute validation errors (e.g. invalid
  // cluster dimensions) are reported from the launch itself. The server
  // applies the same numAttrs rule when deciding whether to respond.
  bool fire_and_forget = config->numAttrs == 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchKernelEx) < 0 ||
      rpc_copy_alloc(
          conn, lupine_rpc_param_info_storage_size(param_sizes.size())) < 0 ||
      lupine_write_launch_config(conn, config) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &kernel_handle, sizeof(kernel_handle)) < 0 ||
#if CUDA_VERSION >= 12000
      (kernel_handle
           ? lupine_write_kernel_param_values(
                 conn, reinterpret_cast<CUkernel>(f), rpc_params.data())
           : lupine_write_func_param_values(conn, f, rpc_params.data())) < 0) {
#else
      lupine_write_func_param_values(conn, f, rpc_params.data()) < 0) {
#endif
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (fire_and_forget) {
    if (rpc_write_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  } else {
    CUresult return_value;
    if (rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
  }

  if (sync_after_launch) {
    return cuStreamSynchronize(config->hStream);
  }
  return CUDA_SUCCESS;
#endif
}

extern "C" CUresult
cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                          unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY,
                          unsigned int blockDimZ, unsigned int sharedMemBytes,
                          CUstream hStream, void **kernelParams) {
  bool kernel_handle = lupine_is_library_kernel(f);
  f = lupine_translate_private_function(f);

  lupine_route route = lupine_route_for_function(f);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(
        CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, CUstream, void **);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuLaunchCooperativeKernel");
    return real == nullptr
               ? CUDA_ERROR_DEVICE_UNAVAILABLE
               : real(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                      blockDimZ, sharedMemBytes, hStream, kernelParams);
  }

  std::vector<size_t> param_offsets;
  std::vector<size_t> param_sizes;
  CUresult status =
      kernel_handle
          ? lupine_read_kernel_param_layout(reinterpret_cast<CUkernel>(f),
                                            &param_offsets, &param_sizes)
          : lupine_read_func_param_layout(f, &param_offsets, &param_sizes);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  for (size_t i = 0; i < param_sizes.size(); ++i) {
    if (kernelParams == nullptr || kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  uint32_t param_count = static_cast<uint32_t>(param_sizes.size());
  std::vector<CUdeviceptr> translated_params(param_count);
  std::vector<void *> rpc_params(param_count);
  status = lupine_sync_mapped_host_to_device_for_launch(
      kernelParams, param_sizes.data(), param_count, translated_params.data(),
      rpc_params.data());
  if (status != CUDA_SUCCESS) {
    return status;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchCooperativeKernel) < 0 ||
      rpc_copy_alloc(
          conn, lupine_rpc_param_info_storage_size(param_sizes.size())) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_write(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_write(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_write(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_write(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_write(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_write(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &kernel_handle, sizeof(kernel_handle)) < 0 ||
#if CUDA_VERSION >= 12000
      (kernel_handle
           ? lupine_write_kernel_param_values(
                 conn, reinterpret_cast<CUkernel>(f), rpc_params.data())
           : lupine_write_func_param_values(conn, f, rpc_params.data())) < 0
#else
      lupine_write_func_param_values(conn, f, rpc_params.data()) < 0
#endif
      || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuLaunchCooperativeKernel_ptsz
#undef cuLaunchCooperativeKernel_ptsz
#endif
extern "C" CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  return cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                   blockDimY, blockDimZ, sharedMemBytes,
                                   hStream, kernelParams);
}

extern "C" CUresult lupine_cuFuncGetAttribute_safe(int *pi,
                                                   CUfunction_attribute attrib,
                                                   CUfunction hfunc) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(hfunc);
  if (translated != hfunc) {
    LUPINE_TRACE_LOG("LUPINE translated cuFuncGetAttribute attr="
                     << attrib << " client=" << hfunc
                     << " server=" << translated);
    return cuFuncGetAttribute(pi, attrib, translated);
  }
  return cuFuncGetAttribute(pi, attrib, hfunc);
}

extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(func);
  if (translated != func) {
    LUPINE_TRACE_LOG("LUPINE translated "
                     "cuOccupancyMaxActiveBlocksPerMultiprocessor"
                     << " client=" << func << " server=" << translated
                     << " blockSize=" << blockSize
                     << " dynamicSMemSize=" << dynamicSMemSize);
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks, translated, blockSize, dynamicSMemSize);
  }
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                     dynamicSMemSize);
}

extern "C" CUresult
lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(func);
  if (translated != func) {
    LUPINE_TRACE_LOG("LUPINE translated "
                     "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
                     << " client=" << func << " server=" << translated
                     << " blockSize=" << blockSize << " dynamicSMemSize="
                     << dynamicSMemSize << " flags=" << flags);
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, translated, blockSize, dynamicSMemSize, flags);
  }
  return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, func, blockSize, dynamicSMemSize, flags);
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                    size_t srcOffset, size_t ByteCount) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  using real_fn_t = CUresult (*)(void *, CUarray, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyAtoH_v2", &return_value, dstHost, srcArray, srcOffset,
          ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyAtoH_v2) < 0 ||
      rpc_write(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_write(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  int request_id = rpc_write_end(conn);
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  lupine_prepare_host_range_write(dstHost, ByteCount);
  auto *copy_dst = static_cast<unsigned char *>(dstHost);
  size_t offset = 0;
  do {
    size_t chunk =
        std::min(ByteCount - offset, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
    if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        (return_value == CUDA_SUCCESS && chunk != 0 &&
         rpc_read(conn, copy_dst + offset, chunk) < 0)) {
      rpc_read_end(conn);
      lupine_mark_host_range_clean(dstHost, ByteCount);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    bool final_chunk =
        return_value != CUDA_SUCCESS || offset + chunk == ByteCount;
    if (rpc_read_end(conn) < 0) {
      lupine_mark_host_range_clean(dstHost, ByteCount);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value != CUDA_SUCCESS) {
      lupine_mark_host_range_clean(dstHost, ByteCount);
      return return_value;
    }
    offset += chunk;
    if (!final_chunk && rpc_read_start(conn, request_id) < 0) {
      lupine_mark_host_range_clean(dstHost, ByteCount);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  } while (offset < ByteCount);
  lupine_mark_host_range_clean(dstHost, ByteCount);
  return return_value;
}

#ifdef cuMemcpyAtoH
#undef cuMemcpyAtoH
#endif
extern "C" CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount, CUstream hStream) {
  lupine_note_async_dtoh_copy();
  conn_t *conn = lupine_rpc_conn_for_deviceptr(srcDevice);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoHAsync_v2) < 0 ||
      rpc_write(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

static bool lupine_is_local_address(const void *ptr) {
  if (ptr == nullptr) {
    return false;
  }
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  uintptr_t page = reinterpret_cast<uintptr_t>(ptr) &
                   ~(static_cast<uintptr_t>(page_size) - 1);
  unsigned char vec = 0;
  return mincore(reinterpret_cast<void *>(page), page_size, &vec) == 0;
}

static size_t lupine_memcpy3d_host_span(const CUDA_MEMCPY3D &copy,
                                        bool source) {
  size_t x = source ? copy.srcXInBytes : copy.dstXInBytes;
  size_t y = source ? copy.srcY : copy.dstY;
  size_t z = source ? copy.srcZ : copy.dstZ;
  size_t pitch = source ? copy.srcPitch : copy.dstPitch;
  size_t height = source ? copy.srcHeight : copy.dstHeight;
  if (pitch == 0) {
    pitch = copy.WidthInBytes + x;
  }
  if (height == 0) {
    height = copy.Height + y;
  }
  if (copy.WidthInBytes == 0 || copy.Height == 0 || copy.Depth == 0) {
    return 0;
  }
  return (z + copy.Depth - 1) * pitch * height + (y + copy.Height - 1) * pitch +
         x + copy.WidthInBytes;
}

static size_t lupine_memcpy2d_host_span(const CUDA_MEMCPY2D &copy,
                                        bool source) {
  size_t x = source ? copy.srcXInBytes : copy.dstXInBytes;
  size_t y = source ? copy.srcY : copy.dstY;
  size_t pitch = source ? copy.srcPitch : copy.dstPitch;
  if (pitch == 0) {
    pitch = copy.WidthInBytes + x;
  }
  if (copy.WidthInBytes == 0 || copy.Height == 0) {
    return 0;
  }
  return (y + copy.Height - 1) * pitch + x + copy.WidthInBytes;
}

static int lupine_memcpy2d_rpc_op(bool async, bool unaligned) {
  if (async) {
    return RPC_cuMemcpy2DAsync_v2;
  }
  if (unaligned) {
    return RPC_cuMemcpy2DUnaligned_v2;
  }
  return RPC_cuMemcpy2D_v2;
}

static CUresult lupine_cuMemcpy2D_common(const CUDA_MEMCPY2D *pCopy,
                                         CUstream stream, bool async,
                                         bool unaligned) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUDA_MEMCPY2D copy = *pCopy;
  size_t src_host_size = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                             ? lupine_memcpy2d_host_span(copy, true)
                             : 0;
  size_t dst_host_size = copy.dstMemoryType == CU_MEMORYTYPE_HOST
                             ? lupine_memcpy2d_host_span(copy, false)
                             : 0;
  const void *src_host = copy.srcHost;
  void *dst_host = copy.dstHost;
  if ((src_host_size != 0 && src_host == nullptr) ||
      (dst_host_size != 0 && dst_host == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = nullptr;
  if (copy.srcMemoryType == CU_MEMORYTYPE_DEVICE && copy.srcDevice != 0) {
    conn = lupine_rpc_conn_for_deviceptr(copy.srcDevice);
  } else if (copy.dstMemoryType == CU_MEMORYTYPE_DEVICE &&
             copy.dstDevice != 0) {
    conn = lupine_rpc_conn_for_deviceptr(copy.dstDevice);
  } else if (stream != nullptr) {
    conn = lupine_rpc_conn_for_stream(stream);
  } else {
    conn = lupine_rpc_conn_for_current_context();
  }
  CUresult return_value;
  size_t returned_dst_size = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, lupine_memcpy2d_rpc_op(async, unaligned)) <
          0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &src_host_size, sizeof(src_host_size)) < 0 ||
      (src_host_size != 0 && rpc_write(conn, src_host, src_host_size) < 0) ||
      rpc_write(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      (async && rpc_write(conn, &stream, sizeof(stream)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size > dst_host_size) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    lupine_prepare_host_range_write(dst_host, returned_dst_size);
    if (rpc_read(conn, dst_host, returned_dst_size) < 0) {
      lupine_mark_host_range_clean(dst_host, returned_dst_size);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    if (returned_dst_size != 0) {
      lupine_mark_host_range_clean(dst_host, returned_dst_size);
    }
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    lupine_mark_host_range_clean(dst_host, returned_dst_size);
  }
  return return_value;
}

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  return lupine_cuMemcpy2D_common(pCopy, nullptr, false, false);
}

#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
extern "C" CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2D_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  return lupine_cuMemcpy2D_common(pCopy, nullptr, false, true);
}

#ifdef cuMemcpy2DUnaligned
#undef cuMemcpy2DUnaligned
#endif
extern "C" CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2DUnaligned_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                       CUstream hStream) {
  return lupine_cuMemcpy2D_common(pCopy, hStream, true, false);
}

#ifdef cuMemcpy2DAsync
#undef cuMemcpy2DAsync
#endif
extern "C" CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy2DAsync_ptsz
#undef cuMemcpy2DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy2DAsync_ptsz(const CUDA_MEMCPY2D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

static CUresult lupine_graph_mem_attribute_rpc(CUdevice device,
                                               CUgraphMem_attribute attr,
                                               cuuint64_t *value, bool set) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = lupine_rpc_conn_for_device(&device);
  CUresult result = CUDA_ERROR_UNKNOWN;
  int op =
      set ? RPC_cuDeviceSetGraphMemAttribute : RPC_cuDeviceGetGraphMemAttribute;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &device, sizeof(device)) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      (set && rpc_write(conn, value, sizeof(*value)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (!set && rpc_read(conn, value, sizeof(*value)) < 0) ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuDeviceGetGraphMemAttribute(CUdevice device,
                                                 CUgraphMem_attribute attr,
                                                 void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuuint64_t remote_value = 0;
  CUresult result =
      lupine_graph_mem_attribute_rpc(device, attr, &remote_value, false);
  if (result == CUDA_SUCCESS) {
    memcpy(value, &remote_value, sizeof(remote_value));
  }
  return result;
}

extern "C" CUresult cuDeviceSetGraphMemAttribute(CUdevice device,
                                                 CUgraphMem_attribute attr,
                                                 void *value) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuuint64_t remote_value = 0;
  memcpy(&remote_value, value, sizeof(remote_value));
  return lupine_graph_mem_attribute_rpc(device, attr, &remote_value, true);
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUDA_MEMCPY3D copy = *pCopy;
  size_t src_host_size = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                             ? lupine_memcpy3d_host_span(copy, true)
                             : 0;
  size_t dst_host_size = copy.dstMemoryType == CU_MEMORYTYPE_HOST
                             ? lupine_memcpy3d_host_span(copy, false)
                             : 0;
  const void *src_host = copy.srcHost;
  void *dst_host = copy.dstHost;
  if ((src_host_size != 0 && src_host == nullptr) ||
      (dst_host_size != 0 && dst_host == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  conn_t *conn = nullptr;
  if (copy.srcMemoryType == CU_MEMORYTYPE_DEVICE && copy.srcDevice != 0) {
    conn = lupine_rpc_conn_for_deviceptr(copy.srcDevice);
  } else if (copy.dstMemoryType == CU_MEMORYTYPE_DEVICE &&
             copy.dstDevice != 0) {
    conn = lupine_rpc_conn_for_deviceptr(copy.dstDevice);
  } else {
    conn = lupine_rpc_conn_for_current_context();
  }
  CUresult return_value;
  size_t returned_dst_size = 0;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpy3D_v2) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &src_host_size, sizeof(src_host_size)) < 0 ||
      (src_host_size != 0 && rpc_write(conn, src_host, src_host_size) < 0) ||
      rpc_write(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size > dst_host_size) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    lupine_prepare_host_range_write(dst_host, returned_dst_size);
    if (rpc_read(conn, dst_host, returned_dst_size) < 0) {
      lupine_mark_host_range_clean(dst_host, returned_dst_size);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    if (returned_dst_size != 0) {
      lupine_mark_host_range_clean(dst_host, returned_dst_size);
    }
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (returned_dst_size != 0) {
    lupine_mark_host_range_clean(dst_host, returned_dst_size);
  }
  return return_value;
}

#ifdef cuMemcpy3D
#undef cuMemcpy3D
#endif
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return cuMemcpy3D_v2(pCopy);
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount, CUstream hStream) {
  bool dst_is_host = lupine_is_local_address(reinterpret_cast<void *>(dst));
  bool src_is_host = lupine_is_local_address(reinterpret_cast<void *>(src));

  if (dst_is_host && !src_is_host) {
    return cuMemcpyDtoHAsync_v2(reinterpret_cast<void *>(dst), src, ByteCount,
                                hStream);
  }
  if (!dst_is_host && src_is_host) {
    return cuMemcpyHtoDAsync_v2(dst, reinterpret_cast<const void *>(src),
                                ByteCount, hStream);
  }
  if (dst_is_host && src_is_host) {
    memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src),
           ByteCount);
    return CUDA_SUCCESS;
  }

  conn_t *conn = lupine_rpc_conn_for_deviceptr(dst);
  conn_t *src_conn = lupine_rpc_conn_for_deviceptr(src);
  if (conn != src_conn) {
    return lupine_cuMemcpyDtoD_via_client(dst, src, ByteCount, hStream, true);
  }
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpyAsync) < 0 ||
      rpc_write(conn, &dst, sizeof(dst)) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuMemcpyAsync_ptsz
#undef cuMemcpyAsync_ptsz
#endif
extern "C" CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                       size_t ByteCount, CUstream hStream) {
  return cuMemcpyAsync(dst, src, ByteCount, hStream);
}

static CUresult
lupine_validate_graph_dependencies(const CUgraphNode *dependencies,
                                   size_t numDependencies) {
  if (numDependencies != 0 && dependencies == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return CUDA_SUCCESS;
}

// Backing store for codegen DeepStructOperation RECV params (e.g. the
// *GetParams node-params queries). CUDA returns arrays owned by the node; we
// mirror that by keeping the deserialized copies alive, keyed by the caller's
// out-pointer, until the next deep query into the same struct. Generic across
// all deep structs so codegen needs no per-type globals.
static std::mutex g_deep_cache_mutex;
static std::map<const void *, std::vector<void *>> g_deep_cache;

extern "C" void lupine_deep_cache_reset(const void *key) {
  std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
  auto it = g_deep_cache.find(key);
  if (it != g_deep_cache.end()) {
    for (void *ptr : it->second) {
      free(ptr);
    }
    it->second.clear();
  }
}

extern "C" void *lupine_deep_cache_add(const void *key, size_t bytes) {
  void *ptr = bytes != 0 ? malloc(bytes) : nullptr;
  if (bytes != 0 && ptr == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(g_deep_cache_mutex);
  g_deep_cache[key].push_back(ptr);
  return ptr;
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

static size_t lupine_memcpy3d_host_span_bytes(const CUDA_MEMCPY3D &params,
                                              bool source) {
  size_t width = params.WidthInBytes;
  size_t height = params.Height == 0 ? 1 : params.Height;
  size_t depth = params.Depth == 0 ? 1 : params.Depth;
  size_t pitch = source ? params.srcPitch : params.dstPitch;
  if (pitch == 0) {
    pitch = width;
  }
  size_t rows = height * depth;
  return pitch * rows;
}

static int
lupine_write_kernel_node_params(conn_t *conn,
                                const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (conn == nullptr || nodeParams == nullptr) {
    return -1;
  }
  CUfunction function = nodeParams->func;
  if (rpc_copy_alloc(conn, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0) {
    return -1;
  }
  if (nodeParams->extra != nullptr) {
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    if (result == nullptr) {
      return -1;
    }
    *result = CUDA_ERROR_NOT_SUPPORTED;
    return rpc_write(conn, result, sizeof(*result));
  }
  if (function != nullptr) {
    return lupine_write_func_param_values(conn, function,
                                          nodeParams->kernelParams);
  }
#if CUDA_VERSION >= 12000
  if (nodeParams->kern != nullptr) {
    return lupine_write_kernel_param_values(conn, nodeParams->kern,
                                            nodeParams->kernelParams);
  }
#endif
  auto *result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  if (result == nullptr) {
    return -1;
  }
  *result = CUDA_ERROR_INVALID_HANDLE;
  return rpc_write(conn, result, sizeof(*result));
}

static int lupine_write_graph_node_params(conn_t *conn,
                                          const CUgraphNodeParams *nodeParams) {
  if (conn == nullptr || nodeParams == nullptr) {
    return -1;
  }

  if (nodeParams->type != CU_GRAPH_NODE_TYPE_KERNEL &&
      nodeParams->type != CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    return -1;
  }
  if (rpc_copy_alloc(conn, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0) {
    return -1;
  }
  if (nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    return 0;
  }

  if (nodeParams->kernel.extra != nullptr) {
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    if (result == nullptr) {
      return -1;
    }
    *result = CUDA_ERROR_NOT_SUPPORTED;
    return rpc_write(conn, result, sizeof(*result));
  }
  if (nodeParams->kernel.func != nullptr) {
    return lupine_write_func_param_values(conn, nodeParams->kernel.func,
                                          nodeParams->kernel.kernelParams);
  }
#if CUDA_VERSION >= 12000
  if (nodeParams->kernel.kern != nullptr) {
    return lupine_write_kernel_param_values(conn, nodeParams->kernel.kern,
                                            nodeParams->kernel.kernelParams);
  }
#endif
  auto *result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  if (result == nullptr) {
    return -1;
  }
  *result = CUDA_ERROR_INVALID_HANDLE;
  return rpc_write(conn, result, sizeof(*result));
}

static CUfunction lupine_client_function_for_remote(CUfunction remote) {
  if (remote == nullptr) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_function_mutex());
    for (const auto &entry : lupine_host_function_map()) {
      if (entry.second == remote && entry.first != remote) {
        return entry.first;
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    for (const auto &entry : lupine_library_kernels()) {
      CUfunction candidate = reinterpret_cast<CUfunction>(entry.first);
      if (candidate == remote) {
        continue;
      }
      for (const auto &route_entry : entry.second.kernels_by_route) {
        if (reinterpret_cast<CUfunction>(route_entry.second) == remote) {
          return candidate;
        }
      }
    }
    for (const auto &entry : lupine_module_functions()) {
      if (entry.first == remote) {
        continue;
      }
      for (const auto &route_entry : entry.second.functions_by_route) {
        if (route_entry.second == remote) {
          return entry.first;
        }
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(lupine_private_node_mutex());
    for (const auto &entry : lupine_private_node_map()) {
      if (entry.first == remote) {
        continue;
      }
      if (entry.second.server_function == remote) {
        return entry.first;
      }
      for (const auto &route_entry : entry.second.functions_by_route) {
        if (route_entry.second == remote) {
          return entry.first;
        }
      }
    }
  }
  return remote;
}

static CUkernel lupine_client_kernel_for_remote(CUkernel remote) {
  if (remote == nullptr) {
    return nullptr;
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

extern "C" CUresult
cuGraphKernelNodeGetParams_v2(CUgraphNode hNode,
                              CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuGraphKernelNodeGetParams_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(hNode, nodeParams);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  std::shared_ptr<void *> kernel_params;
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  CUresult param_info_result = CUDA_SUCCESS;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphKernelNodeGetParams_v2) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    if (rpc_read_kernel_node_params_and_values(conn, nodeParams,
                                               &param_info_result) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    kernel_params = std::shared_ptr<void *>(nodeParams->kernelParams,
                                            rpc_free_kernel_param_values);
  }
  if (rpc_read_end(conn) < 0) {
    if (return_value == CUDA_SUCCESS) {
      nodeParams->kernelParams = nullptr;
    }
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

  std::lock_guard<std::mutex> lock(lupine_graph_kernel_node_params_mutex());
  auto &slot = lupine_graph_kernel_node_params_cache()[hNode];
  slot = std::move(kernel_params);
  nodeParams->kernelParams = slot.get();
  return return_value;
}

extern "C" CUresult
cuGraphKernelNodeSetParams_v2(CUgraphNode hNode,
                              const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuGraphKernelNodeSetParams_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(hNode, nodeParams);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphKernelNodeSetParams_v2) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      lupine_write_kernel_node_params(conn, nodeParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeGetParams
#endif
extern "C" CUresult
cuGraphKernelNodeGetParams(CUgraphNode hNode,
                           CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeGetParams_v2(hNode, nodeParams);
}

#ifdef cuGraphKernelNodeSetParams
#undef cuGraphKernelNodeSetParams
#endif
extern "C" CUresult
cuGraphKernelNodeSetParams(CUgraphNode hNode,
                           const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeSetParams_v2(hNode, nodeParams);
}

extern "C" CUresult
cuGraphAddKernelNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                        const CUgraphNode *dependencies, size_t numDependencies,
                        const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_KERNEL_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddKernelNode_v2", &local_result, phGraphNode, hGraph,
          dependencies, numDependencies, nodeParams)) {
    if (local_result == CUDA_SUCCESS) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddKernelNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      lupine_write_kernel_node_params(conn, nodeParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

#ifdef cuGraphAddKernelNode
#undef cuGraphAddKernelNode
#endif
extern "C" CUresult
cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies,
                                 numDependencies, nodeParams);
}

extern "C" CUresult
cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
  if (phGraphNode == nullptr || copyParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  size_t host_src_bytes = 0;
  if (copyParams->srcMemoryType == CU_MEMORYTYPE_HOST) {
    host_src_bytes = lupine_memcpy3d_host_span_bytes(*copyParams, true);
    if (host_src_bytes != 0 && copyParams->srcHost == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_MEMCPY3D *, CUcontext);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddMemcpyNode", &local_result, phGraphNode, hGraph,
          dependencies, numDependencies, copyParams, ctx)) {
    if (local_result == CUDA_SUCCESS) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddMemcpyNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, copyParams, sizeof(*copyParams)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0 ||
      (host_src_bytes != 0 &&
       rpc_write(conn, copyParams->srcHost, host_src_bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

extern "C" CUresult
cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  using real_fn_t =
      CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecKernelNodeSetParams_v2", &local_result, hGraphExec,
          hNode, nodeParams)) {
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecKernelNodeSetParams_v2) <
          0 ||
      rpc_write(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      lupine_write_kernel_node_params(conn, nodeParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuGraphExecKernelNodeSetParams
#undef cuGraphExecKernelNodeSetParams
#endif
extern "C" CUresult
cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams);
}

extern "C" CUresult
cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                     CUcontext ctx) {
  if (phGraphNode == nullptr || memsetParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t =
      CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                   const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddMemsetNode", &local_result, phGraphNode, hGraph,
          dependencies, numDependencies, memsetParams, ctx)) {
    if (local_result == CUDA_SUCCESS) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return local_result;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddMemsetNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, memsetParams, sizeof(*memsetParams)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

extern "C" CUresult
cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                   const CUgraphNode *dependencies, size_t numDependencies,
                   const CUDA_HOST_NODE_PARAMS *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_HOST_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddHostNode", &local_result, phGraphNode, hGraph,
          dependencies, numDependencies, nodeParams)) {
    if (local_result == CUDA_SUCCESS) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddHostNode) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

extern "C" CUresult cuGraphConditionalHandleCreate(
    CUgraphConditionalHandle *pHandle_out, CUgraph hGraph, CUcontext ctx,
    unsigned int defaultLaunchValue, unsigned int flags) {
  if (pHandle_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = hGraph != nullptr ? lupine_route_for_graph(hGraph)
                                         : lupine_route_for_context(ctx);
  using real_fn_t = CUresult (*)(CUgraphConditionalHandle *, CUgraph, CUcontext,
                                 unsigned int, unsigned int);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphConditionalHandleCreate", &local_result, pHandle_out,
          hGraph, ctx, defaultLaunchValue, flags)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuGraphConditionalHandleCreate) <
          0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &defaultLaunchValue, sizeof(defaultLaunchValue)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle_out, sizeof(*pHandle_out)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuGraphAddNode_v2(CUgraphNode *phGraphNode, CUgraph hGraph,
                                      const CUgraphNode *dependencies,
                                      const CUgraphEdgeData *dependencyData,
                                      size_t numDependencies,
                                      CUgraphNodeParams *nodeParams) {
  if (phGraphNode == nullptr || nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  if (nodeParams->type != CU_GRAPH_NODE_TYPE_KERNEL &&
      nodeParams->type != CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t =
      CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                   const CUgraphEdgeData *, size_t, CUgraphNodeParams *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddNode_v2", &local_result, phGraphNode, hGraph,
          dependencies, dependencyData, numDependencies, nodeParams)) {
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

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  unsigned int child_count = nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL
                                 ? nodeParams->conditional.size
                                 : 0;
  std::vector<CUgraph> child_graphs(child_count);
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuGraphAddNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      lupine_write_graph_node_params(conn, nodeParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      (child_count != 0 && rpc_read(conn, child_graphs.data(),
                                    child_count * sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && child_count != 0) {
    auto *client_graphs =
        static_cast<CUgraph *>(malloc(child_count * sizeof(CUgraph)));
    if (client_graphs == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memcpy(client_graphs, child_graphs.data(), child_count * sizeof(CUgraph));
    nodeParams->conditional.phGraph_out = client_graphs;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
    for (CUgraph child_graph : child_graphs) {
      lupine_note_graph_owner_route(child_graph, route);
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

// ---------------------------------------------------------------------------
// Client wrappers for the CUDA graph query and node-params APIs that the
// @param annotation grammar cannot describe (see codegen/annotations.h and the
// matching manual handlers in manual_server.cpp). cuGraphGetNodes and
// cuGraphGetRootNodes are now generated from the OPTIONAL out-array grammar;
// the remaining query fns are remapped to *_v2 by cuda.h so they stay manual.
// ---------------------------------------------------------------------------

// cuGraphGetEdges gained an optional CUgraphEdgeData out-array (the _v2 form)
// in CUDA 12.3. Newer headers also remap cuGraphGetEdges to cuGraphGetEdges_v2
// via macro; undef it and bind to the literal ABI symbol names (the macro is
// absent on some 12.x), exporting both so old and new callers reach the RPC.
#if CUDA_VERSION >= 12030
#ifdef cuGraphGetEdges
#undef cuGraphGetEdges
#endif
extern "C" CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode *from,
                                       CUgraphNode *to,
                                       CUgraphEdgeData *edgeData,
                                       size_t *numEdges) {
  if (numEdges == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = (from == nullptr || to == nullptr) ? 0 : *numEdges;
  uint8_t want_edge = edgeData != nullptr ? 1 : 0;
  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t = CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *,
                                 CUgraphEdgeData *, size_t *);
  CUresult return_value;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphGetEdges_v2",
                                                  &return_value, hGraph, from,
                                                  to, edgeData, numEdges)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t returned = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphGetEdges_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_write(conn, &want_edge, sizeof(want_edge)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (from != nullptr && to != nullptr && returned != 0 &&
       rpc_read(conn, from, returned * sizeof(CUgraphNode)) < 0) ||
      (from != nullptr && to != nullptr && returned != 0 &&
       rpc_read(conn, to, returned * sizeof(CUgraphNode)) < 0) ||
      (edgeData != nullptr && from != nullptr && to != nullptr &&
       returned != 0 &&
       rpc_read(conn, edgeData, returned * sizeof(CUgraphEdgeData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *numEdges = returned;
  return return_value;
}
extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                    CUgraphNode *to, size_t *numEdges) {
  return cuGraphGetEdges_v2(hGraph, from, to, nullptr, numEdges);
}
#else
extern "C" CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from,
                                    CUgraphNode *to, size_t *numEdges) {
  if (numEdges == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = (from == nullptr || to == nullptr) ? 0 : *numEdges;
  uint8_t want_edge = 0;
  lupine_route route = lupine_route_for_graph(hGraph);
  using real_fn_t =
      CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
  CUresult return_value;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphGetEdges",
                                                  &return_value, hGraph, from,
                                                  to, numEdges)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t returned = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphGetEdges_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_write(conn, &want_edge, sizeof(want_edge)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (from != nullptr && to != nullptr && returned != 0 &&
       rpc_read(conn, from, returned * sizeof(CUgraphNode)) < 0) ||
      (from != nullptr && to != nullptr && returned != 0 &&
       rpc_read(conn, to, returned * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *numEdges = returned;
  return return_value;
}
#endif

// Shared client path for cuGraphNodeGetDependencies /
// cuGraphNodeGetDependentNodes (both remapped to *_v2 with an optional
// CUgraphEdgeData out-array on CUDA 12.3+).
#if CUDA_VERSION >= 12030
static CUresult lupine_client_node_dep_query(int rpc_id, CUgraphNode hNode,
                                             CUgraphNode *nodes,
                                             CUgraphEdgeData *edgeData,
                                             size_t *num) {
  if (num == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = nodes == nullptr ? 0 : *num;
  uint8_t want_edge = edgeData != nullptr ? 1 : 0;
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    if (rpc_id == RPC_cuGraphNodeGetDependencies_v2) {
      using real_fn_t =
          CUresult (*)(CUgraphNode, CUgraphNode *, CUgraphEdgeData *, size_t *);
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, "cuGraphNodeGetDependencies_v2", &return_value, hNode,
              nodes, edgeData, num)) {
        return return_value;
      }
    } else if (rpc_id == RPC_cuGraphNodeGetDependentNodes_v2) {
      using real_fn_t =
          CUresult (*)(CUgraphNode, CUgraphNode *, CUgraphEdgeData *, size_t *);
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, "cuGraphNodeGetDependentNodes_v2", &return_value, hNode,
              nodes, edgeData, num)) {
        return return_value;
      }
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t returned = 0;
  if (conn == nullptr || rpc_write_start_request(conn, rpc_id) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_write(conn, &want_edge, sizeof(want_edge)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (nodes != nullptr && returned != 0 &&
       rpc_read(conn, nodes, returned * sizeof(CUgraphNode)) < 0) ||
      (edgeData != nullptr && nodes != nullptr && returned != 0 &&
       rpc_read(conn, edgeData, returned * sizeof(CUgraphEdgeData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *num = returned;
  return return_value;
}

#ifdef cuGraphNodeGetDependencies
#undef cuGraphNodeGetDependencies
#endif
#ifdef cuGraphNodeGetDependentNodes
#undef cuGraphNodeGetDependentNodes
#endif
extern "C" CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode,
                                                  CUgraphNode *dependencies,
                                                  CUgraphEdgeData *edgeData,
                                                  size_t *numDependencies) {
  return lupine_client_node_dep_query(RPC_cuGraphNodeGetDependencies_v2, hNode,
                                      dependencies, edgeData, numDependencies);
}
extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode,
                                               CUgraphNode *dependencies,
                                               size_t *numDependencies) {
  return cuGraphNodeGetDependencies_v2(hNode, dependencies, nullptr,
                                       numDependencies);
}

extern "C" CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode,
                                                    CUgraphNode *dependentNodes,
                                                    CUgraphEdgeData *edgeData,
                                                    size_t *numDependentNodes) {
  return lupine_client_node_dep_query(RPC_cuGraphNodeGetDependentNodes_v2,
                                      hNode, dependentNodes, edgeData,
                                      numDependentNodes);
}
extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                                 CUgraphNode *dependentNodes,
                                                 size_t *numDependentNodes) {
  return cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, nullptr,
                                         numDependentNodes);
}
#else
static CUresult lupine_client_node_dep_query(int rpc_id, CUgraphNode hNode,
                                             CUgraphNode *nodes, size_t *num) {
  if (num == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = nodes == nullptr ? 0 : *num;
  uint8_t want_edge = 0;
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    if (rpc_id == RPC_cuGraphNodeGetDependencies_v2) {
      using real_fn_t = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, "cuGraphNodeGetDependencies", &return_value, hNode, nodes,
              num)) {
        return return_value;
      }
    } else if (rpc_id == RPC_cuGraphNodeGetDependentNodes_v2) {
      using real_fn_t = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
      if (lupine_call_local_cuda_if_routed<real_fn_t>(
              route, "cuGraphNodeGetDependentNodes", &return_value, hNode,
              nodes, num)) {
        return return_value;
      }
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t returned = 0;
  if (conn == nullptr || rpc_write_start_request(conn, rpc_id) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_write(conn, &want_edge, sizeof(want_edge)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (nodes != nullptr && returned != 0 &&
       rpc_read(conn, nodes, returned * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *num = returned;
  return return_value;
}

extern "C" CUresult cuGraphNodeGetDependencies(CUgraphNode hNode,
                                               CUgraphNode *dependencies,
                                               size_t *numDependencies) {
  return lupine_client_node_dep_query(RPC_cuGraphNodeGetDependencies_v2, hNode,
                                      dependencies, numDependencies);
}

extern "C" CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode,
                                                 CUgraphNode *dependentNodes,
                                                 size_t *numDependentNodes) {
  return lupine_client_node_dep_query(RPC_cuGraphNodeGetDependentNodes_v2,
                                      hNode, dependentNodes, numDependentNodes);
}
#endif

extern "C" CUresult
cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_graph_node(hNode);
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphHostNodeGetParams", &local_result, hNode,
          nodeParams)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  CUDA_HOST_NODE_PARAMS params{};
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphHostNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *nodeParams = params;
  return return_value;
}

extern "C" CUresult
cuGraphHostNodeSetParams(CUgraphNode hNode,
                         const CUDA_HOST_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_graph_node(hNode);
  using real_fn_t = CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphHostNodeSetParams", &local_result, hNode,
          nodeParams)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphHostNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                             const CUDA_HOST_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  using real_fn_t =
      CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecHostNodeSetParams", &local_result, hGraphExec,
          hNode, nodeParams)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecHostNodeSetParams) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
extern "C" CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                       unsigned long long flags) {
  return cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags);
}

extern "C" CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                     void *userData) {
  if (fn == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = hStream == nullptr ? lupine_route_for_current_context()
                                          : lupine_route_for_stream(hStream);
  using real_fn_t = CUresult (*)(CUstream, CUhostFn, void *);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLaunchHostFunc", &local_result, hStream, fn, userData)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchHostFunc) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_write(conn, &userData, sizeof(userData)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuStreamAddCallback(CUstream hStream,
                                        CUstreamCallback callback,
                                        void *userData, unsigned int flags) {
  if (callback == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = hStream == nullptr ? lupine_route_for_current_context()
                                          : lupine_route_for_stream(hStream);
  using real_fn_t =
      CUresult (*)(CUstream, CUstreamCallback, void *, unsigned int);
  CUresult local_result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamAddCallback",
                                                  &local_result, hStream,
                                                  callback, userData, flags)) {
    return local_result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamAddCallback) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &callback, sizeof(callback)) < 0 ||
      rpc_write(conn, &userData, sizeof(userData)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static bool lupine_is_writable_user_pointer(const void *ptr, size_t size) {
  if (ptr == nullptr || size == 0) {
    return false;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t end = start + size;
  if (end < start) {
    return false;
  }

  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    uintptr_t region_start = 0;
    uintptr_t region_end = 0;
    char perms[5] = {};
    if (sscanf(line.c_str(), "%lx-%lx %4s", &region_start, &region_end,
               perms) != 3) {
      continue;
    }
    if (start >= region_start && end <= region_end && perms[1] == 'w') {
      return true;
    }
  }
  return false;
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
    using real_fn_t =
        CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *,
                     const CUgraphNode **, const CUgraphEdgeData **, size_t *);
    if (lupine_call_local_cuda_if_routed<real_fn_t>(
            route, "cuStreamGetCaptureInfo_v3", &local_result, stream,
            captureStatus_out, id_out, graph_out, dependencies_out,
            edgeData_out, numDependencies_out)) {
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
    CUresult local_result;
    using real_fn_t =
        CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
    if (lupine_call_local_cuda_if_routed<real_fn_t>(
            route, "cuStreamGetCaptureInfo", &local_result, stream,
            captureStatus_out, id_out)) {
      return local_result;
    }
  }
#endif

  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  size_t dependency_count = 0;
  bool has_edge_data = false;
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuStreamGetCaptureInfo_v3) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &id, sizeof(id)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &dependency_count, sizeof(dependency_count)) < 0 ||
      rpc_read(conn, &has_edge_data, sizeof(has_edge_data)) < 0) {
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

  bool num_dependencies_writable =
      numDependencies_out != nullptr &&
      lupine_is_writable_user_pointer(numDependencies_out,
                                      sizeof(*numDependencies_out));
  if (edgeData_out != nullptr &&
      (numDependencies_out == nullptr || !num_dependencies_writable)) {
    numDependencies_out = reinterpret_cast<size_t *>(edgeData_out);
    edgeData_out = nullptr;
    num_dependencies_writable = lupine_is_writable_user_pointer(
        numDependencies_out, sizeof(*numDependencies_out));
  }
  if (numDependencies_out != nullptr && !num_dependencies_writable) {
    numDependencies_out = nullptr;
  }
  if (edgeData_out != nullptr &&
      !lupine_is_writable_user_pointer(edgeData_out, sizeof(*edgeData_out))) {
    edgeData_out = nullptr;
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
  }
  return return_value;
}

// Stream captures started by this client and not yet terminated. A stream can
// only be capturing if we started the capture, so cuStreamIsCapturing can
// answer NONE locally while this is zero.
static std::atomic<int> lupine_active_stream_captures{0};

class lupine_capture_begin_guard {
public:
  lupine_capture_begin_guard() { lupine_checkpoint::capture_begin(); }

  ~lupine_capture_begin_guard() {
    if (!completed_) {
      lupine_checkpoint::capture_begin_complete(false);
    }
  }

  CUresult complete(CUresult result) {
    if (!completed_) {
      completed_ = true;
      bool started = result == CUDA_SUCCESS;
      lupine_checkpoint::capture_begin_complete(started);
      if (started) {
        lupine_active_stream_captures.fetch_add(1);
      }
    }
    return result;
  }

private:
  bool completed_ = false;
};

static CUresult lupine_complete_stream_end_capture(CUresult result) {
  // CUDA_SUCCESS ends a valid capture. An invalidated or unjoined capture also
  // leaves capture mode when EndCapture reports the terminal error. Errors
  // such as WRONG_THREAD and UNMATCHED leave the tracked capture untouched.
  if (result == CUDA_SUCCESS ||
      result == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED ||
      result == CUDA_ERROR_STREAM_CAPTURE_UNJOINED) {
    lupine_checkpoint::capture_end();
    lupine_active_stream_captures.fetch_sub(1);
  }
  return result;
}

static bool lupine_stream_handle_is_known(CUstream hStream) {
  if (hStream == nullptr || hStream == CU_STREAM_LEGACY ||
      hStream == CU_STREAM_PER_THREAD) {
    return true;
  }
  return lupine_route_for_known_stream(hStream).kind != LUPINE_ROUTE_INVALID;
}

extern "C" CUresult cuStreamIsCapturing(CUstream hStream,
                                        CUstreamCaptureStatus *captureStatus) {
  if (captureStatus == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Libraries poll capture state on the launch path. While this client has no
  // capture outstanding, a stream it knows cannot be capturing, so answer
  // without a round trip. Unknown handles go to the server's driver instead.
  if (lupine_active_stream_captures.load() == 0 &&
      lupine_stream_handle_is_known(hStream)) {
    *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
  }
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstreamCaptureStatus *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamIsCapturing",
                                                  &return_value, hStream,
                                                  captureStatus)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamIsCapturing) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuStreamBeginCapture_v2(CUstream hStream,
                                            CUstreamCaptureMode mode) {
  lupine_capture_begin_guard capture_guard;
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstreamCaptureMode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamBeginCapture_v2", &return_value, hStream, mode)) {
    return capture_guard.complete(return_value);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamBeginCapture_v2) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return capture_guard.complete(CUDA_ERROR_DEVICE_UNAVAILABLE);
  }
  return capture_guard.complete(return_value);
}

extern "C" CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUgraph *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamEndCapture", &return_value, hStream, phGraph)) {
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return lupine_complete_stream_end_capture(return_value);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUgraph *phGraph_null_check;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamEndCapture) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &phGraph, sizeof(CUgraph *)) < 0 ||
      (phGraph != nullptr && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &phGraph_null_check, sizeof(CUgraph *)) < 0 ||
      (phGraph_null_check && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  return lupine_complete_stream_end_capture(return_value);
}

#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
extern "C" CUresult cuStreamBeginCapture(CUstream hStream,
                                         CUstreamCaptureMode mode) {
  return cuStreamBeginCapture_v2(hStream, mode);
}

#ifdef cuStreamEndCapture_ptsz
#undef cuStreamEndCapture_ptsz
#endif
extern "C" CUresult cuStreamEndCapture_ptsz(CUstream hStream,
                                            CUgraph *phGraph) {
  return cuStreamEndCapture(hStream, phGraph);
}

#ifdef cuStreamIsCapturing_ptsz
#undef cuStreamIsCapturing_ptsz
#endif
extern "C" CUresult
cuStreamIsCapturing_ptsz(CUstream hStream,
                         CUstreamCaptureStatus *captureStatus) {
  return cuStreamIsCapturing(hStream, captureStatus);
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
#if CUDA_VERSION >= 12000
extern "C" CUresult cuStreamGetCaptureInfo(
    CUstream stream, CUstreamCaptureStatus *captureStatus_out,
    cuuint64_t *id_out, CUgraph *graph_out,
    const CUgraphNode **dependencies_out, const CUgraphEdgeData **edgeData_out,
    size_t *numDependencies_out) {
  return lupine_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                       graph_out, dependencies_out,
                                       edgeData_out, numDependencies_out);
}
#else
extern "C" CUresult
cuStreamGetCaptureInfo(CUstream stream,
                       CUstreamCaptureStatus *captureStatus_out,
                       cuuint64_t *id_out) {
  return lupine_cuStreamGetCaptureInfo(stream, captureStatus_out, id_out,
                                       nullptr, nullptr, nullptr, nullptr);
}
#endif

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
  lupine_capture_begin_guard capture_guard;
  lupine_route route =
      hGraph != nullptr ? lupine_route_for_graph(hGraph)
                        : (hStream != nullptr ? lupine_route_for_stream(hStream)
                                              : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUgraph, const CUgraphNode *,
                   const CUgraphEdgeData *, size_t, CUstreamCaptureMode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamBeginCaptureToGraph", &return_value, hStream, hGraph,
          dependencies, dependencyData, numDependencies, mode)) {
    return capture_guard.complete(return_value);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (rpc_write_start_request(conn, LUPINE_RPC_cuStreamBeginCaptureToGraph) <
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

extern "C" CUresult cuStreamUpdateCaptureDependencies_v2(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult status =
      lupine_validate_graph_dependencies(dependencies, numDependencies);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  lupine_route route = hStream != nullptr ? lupine_route_for_stream(hStream)
                                          : lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUgraphNode *,
                                 const CUgraphEdgeData *, size_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamUpdateCaptureDependencies_v2", &return_value, hStream,
          dependencies, dependencyData, numDependencies, flags)) {
    return return_value;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (rpc_write_start_request(conn, RPC_cuStreamUpdateCaptureDependencies_v2) <
          0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuStreamUpdateCaptureDependencies
#undef cuStreamUpdateCaptureDependencies
#endif
#if CUDA_VERSION >= 13000
extern "C" CUresult
cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies,
                                  const CUgraphEdgeData *dependencyData,
                                  size_t numDependencies, unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(
      hStream, dependencies, dependencyData, numDependencies, flags);
}
#else
extern "C" CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                      CUgraphNode *dependencies,
                                                      size_t numDependencies,
                                                      unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}
#endif

#if CUDA_VERSION >= 13000
extern "C" CUresult cuStreamUpdateCaptureDependencies_ptsz(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(
      hStream, dependencies, dependencyData, numDependencies, flags);
}
#else
extern "C" CUresult cuStreamUpdateCaptureDependencies_ptsz(
    CUstream hStream, CUgraphNode *dependencies, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, nullptr,
                                              numDependencies, flags);
}
#endif

static bool lupine_uuid_equals(const CUuuid *uuid, const unsigned char *bytes) {
  return uuid != nullptr && memcmp(uuid->bytes, bytes, 16) == 0;
}

extern "C" CUresult lupine_cudart_get_module_from_cubin(CUmodule *module,
                                                        const void *fatbin) {
  CUresult result = cuModuleLoadData(module, fatbin);
  LUPINE_TRACE_LOG("LUPINE cudart get_module_from_cubin result="
                   << static_cast<int>(result)
                   << " module=" << (module == nullptr ? nullptr : *module));
  return result;
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
  if (arg3 != nullptr || arg4 != nullptr || arg5 != 0) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuModuleLoadData(module, fatbin);
  LUPINE_TRACE_LOG("LUPINE cudart get_module_from_cubin_ext1 result="
                   << static_cast<int>(result)
                   << " module=" << (module == nullptr ? nullptr : *module));
  return result;
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
  if (arg3 != nullptr || arg4 != nullptr || arg5 != 0) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuModuleLoadData(module, fatbin);
  LUPINE_TRACE_LOG("LUPINE cudart get_module_from_cubin_ext2 result="
                   << static_cast<int>(result)
                   << " module=" << (module == nullptr ? nullptr : *module));
  return result;
}

extern "C" CUresult lupine_cudart_load_compilers() { return CUDA_SUCCESS; }

extern "C" void lupine_dark_get_unknown_buffer1(void **ptr, size_t *size) {
  static unsigned char buffer[1024] = {};
  if (ptr != nullptr) {
    *ptr = buffer;
  }
  if (size != nullptr) {
    *size = sizeof(buffer);
  }
}

extern "C" void lupine_dark_get_unknown_buffer2(void **ptr, size_t *size) {
  static unsigned char buffer[14] = {};
  if (ptr != nullptr) {
    *ptr = buffer;
  }
  if (size != nullptr) {
    *size = sizeof(buffer);
  }
}

using lupine_context_storage_dtor_t = void (*)(CUcontext context, void *key,
                                               void *value);

struct lupine_context_storage_value {
  void *value;
  lupine_context_storage_dtor_t dtor;
};

using lupine_context_storage_key = std::pair<CUcontext, void *>;

struct lupine_context_storage_key_hash {
  size_t operator()(const lupine_context_storage_key &value) const {
    return std::hash<CUcontext>{}(value.first) ^
           (std::hash<void *>{}(value.second) << 1);
  }
};

static libcuckoo::cuckoohash_map<lupine_context_storage_key,
                                 lupine_context_storage_value,
                                 lupine_context_storage_key_hash> &
lupine_context_storage() {
  static auto *storage =
      new libcuckoo::cuckoohash_map<lupine_context_storage_key,
                                    lupine_context_storage_value,
                                    lupine_context_storage_key_hash>();
  return *storage;
}

static CUresult lupine_normalize_context(CUcontext *ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // (CUcontext)-1 also means the calling thread's current context.
  if (*ctx != nullptr &&
      *ctx != reinterpret_cast<CUcontext>(~static_cast<uintptr_t>(0))) {
    return CUDA_SUCCESS;
  }
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
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = lupine_activate_context_local_storage_context(ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  lupine_context_storage().insert_or_assign(
      std::make_pair(ctx, key), lupine_context_storage_value{value, dtor});
  LUPINE_TRACE_LOG("LUPINE context storage put ctx=" << ctx << " key=" << key
                                                     << " value=" << value);
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_context_local_storage_delete(CUcontext ctx,
                                                        void *key) {
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  if (!lupine_context_storage().erase(lupine_context_storage_key{ctx, key})) {
    return CUDA_ERROR_INVALID_HANDLE;
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
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  lupine_context_storage_value stored;
  if (!lupine_context_storage().find(lupine_context_storage_key{ctx, key},
                                     stored)) {
    LUPINE_TRACE_LOG(
        "LUPINE context storage get missing key ctx=" << ctx << " key=" << key);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *value = stored.value;
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
    status = cuDeviceGet(&device, ordinal);
    if (status != CUDA_SUCCESS) {
      return status;
    }

    lupine_integrity_device_hash_info info = {};
    status = cuDeviceGetUuid_v2(&info.guid, device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_domain,
                                  CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  device);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    status = cuDeviceGetAttribute(&info.pci_device,
                                  CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    if (status != CUDA_SUCCESS) {
      return status;
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

  lupine_integrity_pass3_input input = {
      static_cast<uint32_t>(driver_version),
      version,
      static_cast<uint32_t>(getpid()),
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self())),
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

extern "C" CUresult lupine_context_check(CUcontext, unsigned int *result1,
                                         const void **result2) {
  if (result1 != nullptr) {
    *result1 = 0;
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

template <size_t N>
static const void *lupine_table_start(const void *(&table)[N]) {
  return table;
}

static const void *lupine_get_tools_tls_table() {
  static const void *table[4] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_dark_return_zero),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_1),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_1)};
  return lupine_table_start(table);
}

static const void *lupine_get_tools_runtime_callback_hooks_table() {
  static const void *table[7] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_get_unknown_buffer1),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_return_zero_2),
      reinterpret_cast<const void *>(&lupine_dark_get_unknown_buffer2),
  };
  return lupine_table_start(table);
}

static const void *lupine_get_context_local_storage_table() {
  static const void *table[4] = {
      reinterpret_cast<const void *>(&lupine_context_local_storage_put),
      reinterpret_cast<const void *>(&lupine_context_local_storage_delete),
      reinterpret_cast<const void *>(&lupine_context_local_storage_get),
      nullptr,
  };
  return lupine_table_start(table);
}

static const void *lupine_get_ctx_create_bypass_table() {
  static const void *table[2] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_ctx_create_bypass)};
  return lupine_table_start(table);
}

static const void *lupine_get_heap_access_table() {
  static const void *table[3] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_heap_alloc),
      reinterpret_cast<const void *>(&lupine_heap_free),
  };
  return lupine_table_start(table);
}

static const void *lupine_get_device_extended_rt_table() {
  static const void *table[26] = {};
  static std::once_flag once;
  std::call_once(once, [] {
    const_cast<void **>(table)[0] = reinterpret_cast<void *>(sizeof(table));
    const_cast<void **>(table)[5] =
        reinterpret_cast<void *>(&lupine_device_get_attribute_ext);
    const_cast<void **>(table)[13] =
        reinterpret_cast<void *>(&lupine_device_get_something);
  });
  return lupine_table_start(table);
}

static const void *lupine_get_integrity_check_table() {
  static const void *table[3] = {
      reinterpret_cast<const void *>(sizeof(table)),
      reinterpret_cast<const void *>(&lupine_integrity_check),
      nullptr,
  };
  return lupine_table_start(table);
}

static const void *lupine_get_context_checks_table() {
  static const void *table[17] = {};
  static std::once_flag once;
  std::call_once(once, [] {
    table[0] = reinterpret_cast<const void *>(sizeof(table));
    table[1] =
        reinterpret_cast<const void *>(&lupine_context_check_unsupported);
    table[2] = reinterpret_cast<const void *>(&lupine_context_check);
    table[3] = reinterpret_cast<const void *>(&lupine_context_check_fn3);
  });
  return lupine_table_start(table);
}

extern "C" CUresult cuGetExportTable(const void **ppExportTable,
                                     const CUuuid *pExportTableId) {
  static constexpr unsigned char CUDART_INTERFACE_UUID[16] = {
      0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
      0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};
  static constexpr unsigned char TOOLS_TLS_UUID[16] = {
      0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
      0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc};
  static constexpr unsigned char TOOLS_RUNTIME_CALLBACK_HOOKS_UUID[16] = {
      0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
      0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66};
  static constexpr unsigned char CONTEXT_LOCAL_STORAGE_UUID[16] = {
      0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
      0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93};
  static constexpr unsigned char CTX_CREATE_BYPASS_UUID[16] = {
      0x0c, 0xa5, 0x0b, 0x8c, 0x10, 0x04, 0x92, 0x9a,
      0x89, 0xa7, 0xd0, 0xdf, 0x10, 0xe7, 0x72, 0x86};
  static constexpr unsigned char HEAP_ACCESS_UUID[16] = {
      0x19, 0x5b, 0xcb, 0xf4, 0xd6, 0x7d, 0x02, 0x4a,
      0xac, 0xc5, 0x1d, 0x29, 0xce, 0xa6, 0x31, 0xae};
  static constexpr unsigned char DEVICE_EXTENDED_RT_UUID[16] = {
      0xb1, 0x05, 0x41, 0xe1, 0xf7, 0xc7, 0xc7, 0x4a,
      0x9f, 0x64, 0xf2, 0x23, 0xbe, 0x99, 0xf1, 0xe2};
  static constexpr unsigned char INTEGRITY_CHECK_UUID[16] = {
      0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
      0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2};
  static constexpr unsigned char CONTEXT_CHECKS_UUID[16] = {
      0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
      0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e};

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
  if (lupine_uuid_equals(pExportTableId, CUDART_INTERFACE_UUID)) {
    *ppExportTable = lupine_get_cudart_export_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, TOOLS_TLS_UUID)) {
    *ppExportTable = lupine_get_tools_tls_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, TOOLS_RUNTIME_CALLBACK_HOOKS_UUID)) {
    *ppExportTable = lupine_get_tools_runtime_callback_hooks_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, CONTEXT_LOCAL_STORAGE_UUID)) {
    *ppExportTable = lupine_get_context_local_storage_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, CTX_CREATE_BYPASS_UUID)) {
    *ppExportTable = lupine_get_ctx_create_bypass_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, HEAP_ACCESS_UUID)) {
    *ppExportTable = lupine_get_heap_access_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, DEVICE_EXTENDED_RT_UUID)) {
    *ppExportTable = lupine_get_device_extended_rt_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, INTEGRITY_CHECK_UUID)) {
    *ppExportTable = lupine_get_integrity_check_table();
    return CUDA_SUCCESS;
  }
  if (lupine_uuid_equals(pExportTableId, CONTEXT_CHECKS_UUID)) {
    *ppExportTable = lupine_get_context_checks_table();
    return CUDA_SUCCESS;
  }
  {
    const void *private_table =
        lupine_remote_private_export_table(pExportTableId);
    if (private_table != nullptr) {
      *ppExportTable = private_table;
      return CUDA_SUCCESS;
    }
  }
  if (lupine_stub_private_exports_enabled()) {
    const void *private_table =
        lupine_private_export_table_from_env(pExportTableId);
    if (private_table != nullptr) {
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

#if defined(__x86_64__) || defined(__aarch64__)
  static std::unordered_map<std::string, void *> stubs;
  auto existing = stubs.find(symbol);
  if (existing != stubs.end()) {
    return existing->second;
  }

#if defined(__x86_64__)
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
#if defined(__x86_64__)
  code[0] = 0x48;
  code[1] = 0xbf;
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

LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuModuleLoadData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuModuleLoadFatBinary)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryLoadData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryGetKernelCount)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryEnumerateKernels)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuKernelGetName)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkAddData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkAddFile)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemGetInfo)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemGetAddressRange)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostGetDevicePointer)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostRegister)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostUnregister)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuPointerGetAttribute)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpyDtoHAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DUnaligned)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3D)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeer)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeerAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemAdvise)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemRangeGetAttribute)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGetErrorString)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGetErrorName)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphInstantiate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuUserObjectCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuStreamBeginCaptureToGraph)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuStreamGetCaptureInfo)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuStreamUpdateCaptureDependencies)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphExecKernelNodeSetParams)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphAddNode)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphNodeSetParams)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphExecNodeSetParams)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGraphConditionalHandleCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceRegisterAsyncNotification)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceUnregisterAsyncNotification)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsRegisterCallback)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsUnregisterCallback)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsCurrent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToFile)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLogsDumpToMemory)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetDevResource)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDevSmResourceSplitByCount)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDevResourceGenerateDesc)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxFromGreenCtx)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxDestroy)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxGetDevResource)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxGetDevResource)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxGetId)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxStreamCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuStreamGetDevResource)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxRecordEvent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxWaitEvent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxRecordEvent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuGreenCtxWaitEvent)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDevSmResourceSplit)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetHostAtomicCapabilities)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetP2PAtomicCapabilities)

#ifdef cuStreamWaitValue32
#undef cuStreamWaitValue32
#endif
#ifdef cuStreamBatchMemOp
#undef cuStreamBatchMemOp
#endif
extern "C" CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                        cuuint32_t value, unsigned int flags) {
  return cuStreamWaitValue32_v2(stream, addr, value, flags);
}

extern "C" CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                       CUstreamBatchMemOpParams *paramArray,
                                       unsigned int flags) {
  return cuStreamBatchMemOp_v2(stream, count, paramArray, flags);
}

extern "C" CUresult cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel) {
  if (pLib == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto it = lupine_library_kernels().find(kernel);
    if (it != lupine_library_kernels().end() && it->second.library != nullptr) {
      *pLib = it->second.library;
      return CUDA_SUCCESS;
    }
  }
  // Only kernels handed out by cuLibraryGetKernel are recorded above. Anything
  // else has to be resolved by whoever owns the handle, not guessed at here.
  lupine_route route =
      lupine_route_for_function(reinterpret_cast<CUfunction>(kernel));
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUlibrary *, CUkernel);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuKernelGetLibrary");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return real(pLib, kernel);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelGetLibrary) < 0 ||
      rpc_write(conn, &kernel, sizeof(kernel)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pLib, sizeof(*pLib)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && *pLib != nullptr) {
    lupine_note_library_owner_route(*pLib, route);
  }
  return return_value;
}

static void *lupine_get_unsupported_stub(const char *symbol) {
  // clang-format off
  static const std::unordered_map<std::string, void *> stubs = {
#define LUPINE_STUB_ENTRY(name)                                                \
  { #name, (void *)&lupine_unsupported_##name }
      LUPINE_STUB_ENTRY(cuCtxCreate),
      LUPINE_STUB_ENTRY(cuModuleLoadData),
      LUPINE_STUB_ENTRY(cuModuleLoadFatBinary),
      LUPINE_STUB_ENTRY(cuLibraryLoadData),
      LUPINE_STUB_ENTRY(cuLibraryGetKernelCount),
      LUPINE_STUB_ENTRY(cuLibraryEnumerateKernels),
      LUPINE_STUB_ENTRY(cuKernelGetName),
      LUPINE_STUB_ENTRY(cuLinkCreate),
      LUPINE_STUB_ENTRY(cuLinkAddData),
      LUPINE_STUB_ENTRY(cuLinkAddFile),
      LUPINE_STUB_ENTRY(cuMemGetInfo),
      LUPINE_STUB_ENTRY(cuMemGetAddressRange),
      LUPINE_STUB_ENTRY(cuMemHostGetDevicePointer),
      LUPINE_STUB_ENTRY(cuMemHostRegister),
      LUPINE_STUB_ENTRY(cuMemHostUnregister),
      LUPINE_STUB_ENTRY(cuPointerGetAttribute),
      LUPINE_STUB_ENTRY(cuMemcpyDtoHAsync),
      LUPINE_STUB_ENTRY(cuMemcpy2DUnaligned),
      LUPINE_STUB_ENTRY(cuMemcpy2DAsync),
      LUPINE_STUB_ENTRY(cuMemcpy3D),
      LUPINE_STUB_ENTRY(cuMemcpy3DAsync),
      LUPINE_STUB_ENTRY(cuMemcpy3DPeer),
      LUPINE_STUB_ENTRY(cuMemcpy3DPeerAsync),
      LUPINE_STUB_ENTRY(cuMemAdvise),
      LUPINE_STUB_ENTRY(cuMemRangeGetAttribute),
      LUPINE_STUB_ENTRY(cuGetErrorString),
      LUPINE_STUB_ENTRY(cuGetErrorName),
      LUPINE_STUB_ENTRY(cuGraphInstantiate),
      LUPINE_STUB_ENTRY(cuUserObjectCreate),
      LUPINE_STUB_ENTRY(cuStreamBeginCaptureToGraph),
      LUPINE_STUB_ENTRY(cuStreamGetCaptureInfo),
      LUPINE_STUB_ENTRY(cuStreamUpdateCaptureDependencies),
      LUPINE_STUB_ENTRY(cuGraphExecKernelNodeSetParams),
      LUPINE_STUB_ENTRY(cuGraphAddNode),
      LUPINE_STUB_ENTRY(cuGraphNodeSetParams),
      LUPINE_STUB_ENTRY(cuGraphExecNodeSetParams),
      LUPINE_STUB_ENTRY(cuGraphConditionalHandleCreate),
      LUPINE_STUB_ENTRY(cuDeviceRegisterAsyncNotification),
      LUPINE_STUB_ENTRY(cuDeviceUnregisterAsyncNotification),
      LUPINE_STUB_ENTRY(cuLogsRegisterCallback),
      LUPINE_STUB_ENTRY(cuLogsUnregisterCallback),
      LUPINE_STUB_ENTRY(cuLogsCurrent),
      LUPINE_STUB_ENTRY(cuLogsDumpToFile),
      LUPINE_STUB_ENTRY(cuLogsDumpToMemory),
      LUPINE_STUB_ENTRY(cuDeviceGetDevResource),
      LUPINE_STUB_ENTRY(cuDevSmResourceSplitByCount),
      LUPINE_STUB_ENTRY(cuDevResourceGenerateDesc),
      LUPINE_STUB_ENTRY(cuCtxFromGreenCtx),
      LUPINE_STUB_ENTRY(cuGreenCtxCreate),
      LUPINE_STUB_ENTRY(cuGreenCtxDestroy),
      LUPINE_STUB_ENTRY(cuCtxGetDevResource),
      LUPINE_STUB_ENTRY(cuGreenCtxGetDevResource),
      LUPINE_STUB_ENTRY(cuGreenCtxGetId),
      LUPINE_STUB_ENTRY(cuGreenCtxStreamCreate),
      LUPINE_STUB_ENTRY(cuStreamGetDevResource),
      LUPINE_STUB_ENTRY(cuCtxRecordEvent),
      LUPINE_STUB_ENTRY(cuCtxWaitEvent),
      LUPINE_STUB_ENTRY(cuGreenCtxRecordEvent),
      LUPINE_STUB_ENTRY(cuGreenCtxWaitEvent),
      LUPINE_STUB_ENTRY(cuDevSmResourceSplit),
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

void rpc_close(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }

  // A reopened connection reuses the static conn_t slot but gets fresh server
  // lane threads with no CUDA context. Invalidate all per-lane context hints.
  lupine_invalidate_current_context_cache();

  rpc_close_transport_socket(conn);

  pthread_mutex_lock(&conn->read_mutex);
  pthread_cond_broadcast(&conn->read_cond);
  pthread_mutex_unlock(&conn->read_mutex);
}

static void lupine_rpc_shutdown() {
  if (pthread_mutex_lock(&conn_mutex) < 0) {
    return;
  }
  if (lupine_rpc_shutting_down) {
    pthread_mutex_unlock(&conn_mutex);
    return;
  }
  lupine_rpc_shutting_down = true;
  int count = nconns;
  for (int i = 0; i < count; ++i) {
    rpc_close(&conns[i]);
  }
  pthread_mutex_unlock(&conn_mutex);

  for (int i = 0; i < count; ++i) {
    lupine_join_connection_threads(&conns[i]);
#ifdef LUPINE_TLS_OPENSSL
    // Safe now: the read thread is joined, so nothing touches the SSL*.
    if (conns[i].tls_session != nullptr) {
      SSL_free(static_cast<SSL *>(conns[i].tls_session));
      conns[i].tls_session = nullptr;
    }
#endif
    rpc_conn_destroy(&conns[i]);
  }

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    nconns = 0;
    lupine_server_endpoints().clear();
    lupine_rpc_shutting_down = false;
    pthread_mutex_unlock(&conn_mutex);
  }
}

__attribute__((destructor)) static void lupine_rpc_destructor() {
  lupine_rpc_shutdown();
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == 1) {
      std::cout << "Transferring memory..." << std::endl;

      int found = 0;

      if (rpc_read(conn, &found, sizeof(found)) < 0) {
        LUPINE_LOG_ERROR("Failed to read transfer count.");
        goto close_connection;
      }

      for (int i = 0; i < found; ++i) {
        void *host_data = nullptr;
        void *dst = nullptr;
        size_t count = 0;

        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0) {
          LUPINE_LOG_ERROR("Failed to read transfer parameters.");
          goto close_connection;
        }

        host_data = malloc(count);
        if (!host_data) {
          LUPINE_LOG_ERROR("Memory allocation failed.");
          goto close_connection;
        }

        // Read the actual data from the server (sent from `src` in device
        // memory)
        if (rpc_read_payload(conn, host_data, count) < 0) {
          LUPINE_LOG_ERROR("Failed to read device data from server.");
          free(host_data);
          goto close_connection;
        }

        // Copy received data to the destination (dst) on the host
        lupine_prepare_host_range_write(dst, count);
        memcpy(dst, host_data, count);
        lupine_mark_host_range_clean(dst, count);
        free(host_data);
      }

      CUhostFn callback = nullptr;
      void *user_data = nullptr;
      if (rpc_read(conn, &callback, sizeof(callback)) < 0 ||
          rpc_read(conn, &user_data, sizeof(user_data)) < 0) {
        LUPINE_LOG_ERROR("Failed to read host callback request.");
        goto close_connection;
      }

      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        LUPINE_LOG_ERROR("Failed to finish host callback request.");
        goto close_connection;
      }
      if (callback == nullptr) {
        LUPINE_LOG_ERROR("Invalid function pointer!");
        goto close_connection;
      }

      callback(user_data);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        LUPINE_LOG_ERROR("rpc_write failed. Closing connection.");
        goto close_connection;
      }
    } else if (op == 2) {
      CUstream stream = nullptr;
      CUresult status = CUDA_ERROR_UNKNOWN;
      CUstreamCallback callback = nullptr;
      void *user_data = nullptr;
      if (lupine_read_deferred_dtoh_copies(conn) < 0 ||
          rpc_read(conn, &stream, sizeof(stream)) < 0 ||
          rpc_read(conn, &status, sizeof(status)) < 0 ||
          rpc_read(conn, &callback, sizeof(callback)) < 0 ||
          rpc_read(conn, &user_data, sizeof(user_data)) < 0) {
        LUPINE_LOG_ERROR("Failed to read stream callback request.");
        break;
      }

      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        break;
      }

      if (callback != nullptr) {
        callback(stream, status, user_data);
      }

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        LUPINE_LOG_ERROR("rpc_write failed. Closing connection.");
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
  return nullptr;
}

int rpc_open() {
  if (pthread_mutex_lock(&conn_mutex) < 0)
    return -1;

  if (nconns > 0) {
    if (pthread_mutex_unlock(&conn_mutex) < 0)
      return -1;
    return 0;
  }

  char *server_ips = getenv("LUPINE_SERVER");
  if (server_ips == NULL) {
    if (pthread_mutex_unlock(&conn_mutex) < 0)
      return -1;
    return -1;
  }

  lupine_server_endpoints().clear();

  char *server_ip = strdup(server_ips);
  char *server_ip_cursor = server_ip;
  char *token;
  while ((token = strsep(&server_ip_cursor, ","))) {
    if (nconns >= static_cast<int>(sizeof(conns) / sizeof(conns[0]))) {
      LUPINE_LOG_ERROR("Too many LUPINE_SERVER entries; ignoring the rest");
      break;
    }

    char *host;
    char *port;
    bool tls = false;

    // Optional URL scheme: https:// enables TLS on this connection.
    if (strncmp(token, "https://", 8) == 0) {
      tls = true;
      token += 8;
    } else if (strncmp(token, "http://", 7) == 0) {
      token += 7;
    }

    // Split the remaining string into host and port.
    char *colon = strchr(token, ':');
    if (colon == NULL) {
      host = token;
      port = const_cast<char *>(tls ? "443" : DEFAULT_PORT);
    } else {
      *colon = '\0';
      host = token;
      port = colon + 1;
    }

    lupine_server_endpoint endpoint{host, port, tls};
    if (lupine_connect_endpoint(&conns[nconns], endpoint,
                                static_cast<unsigned int>(nconns)) < 0) {
      continue;
    }
    lupine_server_endpoints().push_back(endpoint);
    nconns++;
  }
  free(server_ip);

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  if (nconns == 0)
    return -1;
  return 0;
}

conn_t *rpc_client_get_connection(unsigned int index) {
  if (rpc_open() < 0 || index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
}

int rpc_size() { return nconns; }

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags);

static const std::unordered_map<std::string, void *> &
lupine_manual_function_map();

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  LUPINE_TRACE_LOG("cuGetProcAddress getting symbol: " << symbol);
  // Most wrappers route purely by symbol name. A few CUDA APIs changed ABI
  // without changing the name and must also use the requested API version.
  (void)flags;

  if (strcmp(symbol, "cuFuncGetAttribute") == 0) {
    *pfn = reinterpret_cast<void *>(&lupine_cuFuncGetAttribute_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (strcmp(symbol, "cuPointerGetAttributes") == 0) {
    *pfn = reinterpret_cast<void *>(&cuPointerGetAttributes);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr &&
      strcmp(symbol, "cuOccupancyMaxActiveBlocksPerMultiprocessor") == 0) {
    *pfn = reinterpret_cast<void *>(
        &lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr &&
      strcmp(symbol, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") ==
          0) {
    *pfn = reinterpret_cast<void *>(
        &lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr && cudaVersion >= 12020 &&
      (strcmp(symbol, "cuMemPrefetchAsync") == 0 ||
       strcmp(symbol, "cuMemPrefetchAsync_ptsz") == 0)) {
    *pfn = reinterpret_cast<void *>(&cuMemPrefetchAsync_v2);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = it;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    LUPINE_TRACE_LOG("cuGetProcAddress: Mapped symbol '"
                     << symbol << "' to function: " << *pfn);
    return CUDA_SUCCESS;
  }

  const auto &manual_function_map = lupine_manual_function_map();
  auto manual_it = manual_function_map.find(symbol);
  if (manual_it != manual_function_map.end()) {
    *pfn = manual_it->second;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  void *unsupported_stub = lupine_get_unsupported_stub(symbol);
  if (unsupported_stub != nullptr) {
    *pfn = unsupported_stub;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
    *pfn = (void *)&cuGetProcAddress_v2;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (strcmp(symbol, "cuGetProcAddress") == 0) {
    *pfn = (void *)&cuGetProcAddress;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  LUPINE_TRACE_LOG("cuGetProcAddress: Symbol '"
                   << symbol << "' not found in cudaFunctionMap.");

  // fall back to the real loader before creating a local missing-symbol stub
  *pfn = lupine_real_dlsym(RTLD_DEFAULT, symbol);
  if (*pfn != nullptr) {
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }

  void *libCudaHandle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
  if (!libCudaHandle) {
    if (lupine_stub_missing_enabled() &&
        lupine_symbol_looks_like_driver_api(symbol)) {
      *pfn = lupine_make_missing_stub(symbol);
      if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
      }
      return CUDA_SUCCESS;
    }
    *pfn = nullptr;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return CUDA_SUCCESS;
  }

  *pfn = lupine_real_dlsym(libCudaHandle, symbol);
  if (!(*pfn)) {
    LUPINE_TRACE_LOG("Error: Could not resolve symbol '" << symbol
                                                         << "' using dlsym.");
    if (lupine_stub_missing_enabled() &&
        lupine_symbol_looks_like_driver_api(symbol)) {
      *pfn = lupine_make_missing_stub(symbol);
      if (symbolStatus != nullptr) {
        *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
      }
      return CUDA_SUCCESS;
    }
    *pfn = nullptr;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return CUDA_SUCCESS;
  }

  if (symbolStatus != nullptr) {
    *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
  }
  return CUDA_SUCCESS;
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

void *dlsym(void *handle, const char *name) __THROW {
  LUPINE_TRACE_LOG("dlsym: " << name);

  if (!lupine_symbol_looks_like_driver_api(name)) {
    return lupine_real_dlsym(handle, name);
  }

  if (strcmp(name, "cuFuncGetAttribute") == 0) {
    return reinterpret_cast<void *>(&lupine_cuFuncGetAttribute_safe);
  }
  if (strcmp(name, "cuPointerGetAttributes") == 0) {
    return reinterpret_cast<void *>(&cuPointerGetAttributes);
  }
  if (strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessor") == 0) {
    return reinterpret_cast<void *>(
        &lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe);
  }
  if (strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") ==
      0) {
    return reinterpret_cast<void *>(
        &lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_safe);
  }

  void *func = get_function_pointer(name);

  /** proc address function calls are basically dlsym; we should handle this
   * differently at the top level. */
  if (strcmp(name, "cuGetProcAddress_v2") == 0) {
    return (void *)&cuGetProcAddress_v2;
  }
  if (strcmp(name, "cuGetProcAddress") == 0) {
    return (void *)&cuGetProcAddress;
  }

  if (func != nullptr) {
    // std::cout << "[dlsym] Function address from cudaFunctionMap: " << func
    // << " " << name << std::endl;
    return func;
  }

  const auto &manual_function_map = lupine_manual_function_map();
  auto manual_it = manual_function_map.find(name);
  if (manual_it != manual_function_map.end()) {
    return manual_it->second;
  }

  void *unsupported_stub = lupine_get_unsupported_stub(name);
  if (unsupported_stub != nullptr) {
    return unsupported_stub;
  }

  if (lupine_stub_missing_enabled() &&
      lupine_symbol_looks_like_driver_api(name)) {
    return lupine_make_missing_stub(name);
  }

  // std::cout << "[dlsym] Falling back to real_dlsym for name: " << name <<
  // std::endl;
  return lupine_real_dlsym(handle, name);
}

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
      {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2},
      {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2},
      {"cuOccupancyMaxPotentialBlockSize",
       (void *)cuOccupancyMaxPotentialBlockSize},
      {"cuOccupancyMaxPotentialBlockSizeWithFlags",
       (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
      {"cuMemExportToShareableHandle", (void *)cuMemExportToShareableHandle},
      {"cuMemImportFromShareableHandle",
       (void *)cuMemImportFromShareableHandle},
      {"cuMemPoolExportToShareableHandle",
       (void *)cuMemPoolExportToShareableHandle},
      {"cuMemPoolImportFromShareableHandle",
       (void *)cuMemPoolImportFromShareableHandle},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuStreamDestroy", (void *)cuStreamDestroy},
      {"cuEventDestroy", (void *)cuEventDestroy},
      {"cuEventElapsedTime", (void *)cuEventElapsedTime},
      {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
      {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
      {"cuStreamQuery", (void *)cuStreamQuery},
      {"cuStreamQuery_ptsz", (void *)cuStreamQuery_ptsz},
      {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
      {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
      {"cuMemFree", (void *)cuMemFree_v2},
      {"cuMemFreeHost", (void *)cuMemFreeHost},
      {"cuMemHostAlloc", (void *)cuMemHostAlloc},
      {"cuMemHostGetDevicePointer", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
      {"cuMemHostGetFlags", (void *)cuMemHostGetFlags},
      {"cuMemHostRegister", (void *)cuMemHostRegister},
      {"cuMemHostRegister_v2", (void *)cuMemHostRegister_v2},
      {"cuMemHostUnregister", (void *)cuMemHostUnregister},
      {"cuMemGetAddressRange", (void *)cuMemGetAddressRange},
      {"cuMemGetInfo", (void *)cuMemGetInfo},
      {"cuMemPrefetchAsync", (void *)cuMemPrefetchAsync},
      {"cuMemPrefetchAsync_ptsz", (void *)cuMemPrefetchAsync_ptsz},
      {"cuMemPrefetchAsync_v2", (void *)cuMemPrefetchAsync_v2},
      {"cuArrayCreate", (void *)cuArrayCreate_v2},
      {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
      {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
      {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
      {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
      {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
      {"cuDeviceGetUuid", (void *)cuDeviceGetUuid_v2},
      {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
      {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
      {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
      {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
      {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
      {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
      {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
      {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2},
      {"cuMemcpyHtoD_v2", (void *)cuMemcpyHtoD_v2},
      {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2},
      {"cuMemcpyDtoH_v2", (void *)cuMemcpyDtoH_v2},
      {"cuMemcpyDtoA", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
      {"cuMemcpyAtoD", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
      {"cuMemcpyAtoH", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
      {"cuMemcpyAtoA", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
      {"cuMemcpy2D", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2D_v2", (void *)cuMemcpy2D_v2},
      {"cuMemcpy2DUnaligned", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DUnaligned_v2", (void *)cuMemcpy2DUnaligned_v2},
      {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_v2", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy2DAsync_ptsz", (void *)cuMemcpy2DAsync_v2},
      {"cuMemcpy3D", (void *)cuMemcpy3D_v2},
      {"cuMemcpy3D_v2", (void *)cuMemcpy3D_v2},
      {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
      {"cuPointerSetAttribute", (void *)cuPointerSetAttribute},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoad", (void *)cuModuleLoad},
      {"cuModuleLoadData", (void *)cuModuleLoadData},
      {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
      {"cuLibraryLoadData", (void *)cuLibraryLoadData},
      {"cuLinkCreate", (void *)cuLinkCreate_v2},
      {"cuLinkAddData", (void *)cuLinkAddData_v2},
      {"cuLinkAddFile", (void *)cuLinkAddFile_v2},
      {"cuLaunchKernel", (void *)cuLaunchKernel},
      {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
      {"cuLaunchCooperativeKernel_ptsz",
       (void *)cuLaunchCooperativeKernel_ptsz},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2},
      {"cuMemcpyHtoDAsync_v2", (void *)cuMemcpyHtoDAsync_v2},
      {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
      {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
      {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
      {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
      {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
      {"cuStreamWaitEvent_ptsz", (void *)cuStreamWaitEvent_ptsz},
      {"cuEventRecord", (void *)cuEventRecord},
      {"cuEventRecord_ptsz", (void *)cuEventRecord_ptsz},
      {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
      {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags_ptsz},
      {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
      {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
      {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
      {"cuCtxSynchronize", (void *)cuCtxSynchronize},
      {"cuStreamSynchronize", (void *)cuStreamSynchronize},
      {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize_ptsz},
      {"cuEventQuery", (void *)cuEventQuery},
      {"cuEventSynchronize", (void *)cuEventSynchronize},
      {"cuGetErrorName", (void *)cuGetErrorName},
      {"cuGetErrorString", (void *)cuGetErrorString},
      {"cuGraphAddKernelNode", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddKernelNode_v2", (void *)cuGraphAddKernelNode_v2},
      {"cuGraphAddNode", (void *)cuGraphAddNode},
      {"cuGraphAddNode_v2", (void *)cuGraphAddNode_v2},
      {"cuGraphConditionalHandleCreate",
       (void *)cuGraphConditionalHandleCreate},
      {"cuGraphAddMemcpyNode", (void *)cuGraphAddMemcpyNode},
      {"cuGraphAddMemsetNode", (void *)cuGraphAddMemsetNode},
      {"cuGraphAddHostNode", (void *)cuGraphAddHostNode},
      {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
      {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
      {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
      {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
      {"cuGraphKernelNodeGetParams", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
      {"cuGraphKernelNodeSetParams", (void *)cuGraphKernelNodeSetParams_v2},
      {"cuGraphKernelNodeSetParams_v2", (void *)cuGraphKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams_v2",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphGetNodes", (void *)cuGraphGetNodes},
      {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
      {"cuGraphAddExternalSemaphoresSignalNode",
       (void *)cuGraphAddExternalSemaphoresSignalNode},
      {"cuGraphExternalSemaphoresSignalNodeGetParams",
       (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
      {"cuGraphExternalSemaphoresSignalNodeSetParams",
       (void *)cuGraphExternalSemaphoresSignalNodeSetParams},
      {"cuGraphExecExternalSemaphoresSignalNodeSetParams",
       (void *)cuGraphExecExternalSemaphoresSignalNodeSetParams},
      {"cuGraphAddExternalSemaphoresWaitNode",
       (void *)cuGraphAddExternalSemaphoresWaitNode},
      {"cuGraphExternalSemaphoresWaitNodeGetParams",
       (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
      {"cuGraphExternalSemaphoresWaitNodeSetParams",
       (void *)cuGraphExternalSemaphoresWaitNodeSetParams},
      {"cuGraphExecExternalSemaphoresWaitNodeSetParams",
       (void *)cuGraphExecExternalSemaphoresWaitNodeSetParams},
      {"cuGraphAddBatchMemOpNode", (void *)cuGraphAddBatchMemOpNode},
      {"cuGraphBatchMemOpNodeGetParams",
       (void *)cuGraphBatchMemOpNodeGetParams},
      {"cuGraphBatchMemOpNodeSetParams",
       (void *)cuGraphBatchMemOpNodeSetParams},
      {"cuGraphExecBatchMemOpNodeSetParams",
       (void *)cuGraphExecBatchMemOpNodeSetParams},
      {"cuGraphHostNodeGetParams", (void *)cuGraphHostNodeGetParams},
      {"cuGraphHostNodeSetParams", (void *)cuGraphHostNodeSetParams},
      {"cuGraphExecHostNodeSetParams", (void *)cuGraphExecHostNodeSetParams},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuKernelGetLibrary", (void *)cuKernelGetLibrary},
      {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
      {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
      {"cuStreamAddCallback", (void *)cuStreamAddCallback},
      {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
      {"cuStreamBeginCapture", (void *)cuStreamBeginCapture_v2},
      {"cuStreamEndCapture_ptsz", (void *)cuStreamEndCapture},
      {"cuStreamIsCapturing_ptsz", (void *)cuStreamIsCapturing},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_v2",
       (void *)cuStreamUpdateCaptureDependencies_v2},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
  };
  return manual_function_map;
}
