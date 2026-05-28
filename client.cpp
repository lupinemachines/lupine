#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cudaProfiler.h>
#include <dlfcn.h>
#include <elf.h>
#include <features.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
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
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if !defined(__GLIBC__)
#error "Lupine CUDA client requires glibc"
#endif

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include <cstdlib>
#include <cstring>

#include "codegen/gen_api.h"
#include "codegen/gen_client.h"
#include "codegen/ptx_fatbin.hpp"
#include "rpc.h"

pthread_mutex_t conn_mutex;
conn_t conns[16];
int nconns = 0;
static bool lupine_rpc_shutting_down = false;

const char *DEFAULT_PORT = "14833";

std::map<void *, void *> host_funcs;

void add_host_node(void *fn, void *udata);

static constexpr int LUPINE_RPC_cuFuncGetParamLayout = 1000001;
static constexpr int LUPINE_RPC_cuCtxCreate_v2 = 1000002;
static constexpr int LUPINE_RPC_cuMemPoolSetAttribute = 1000003;
static constexpr int LUPINE_RPC_cuMemPoolGetAttribute = 1000004;
static constexpr int LUPINE_RPC_cuLaunchHostFunc = 1000005;
static constexpr int LUPINE_RPC_cuPointerGetAttribute = 1000006;
static constexpr int LUPINE_RPC_cuGetExportTableMetadata = 1000007;
static constexpr int LUPINE_RPC_cuPrivateGetModuleNode = 1000008;
static constexpr int LUPINE_RPC_cuPointerGetAttributes = 1000009;
static constexpr int LUPINE_RPC_cuStreamAddCallback = 1000010;
static constexpr int LUPINE_RPC_cuGraphConditionalHandleCreate = 1000011;
static constexpr int LUPINE_RPC_cuGraphAddNode_v2 = 1000012;
static constexpr int LUPINE_RPC_cuStreamBeginCaptureToGraph = 1000013;
static constexpr int LUPINE_RPC_cuStreamUpdateCaptureDependencies_v2 = 1000014;
static constexpr int LUPINE_RPC_cuStreamGetCaptureInfo_v3 = 1000015;
static constexpr int LUPINE_RPC_cuDeviceGetGraphMemAttribute = 1000016;
static constexpr int LUPINE_RPC_cuDeviceSetGraphMemAttribute = 1000017;
static constexpr int LUPINE_RPC_cuLinkAddData_v2 = 1000018;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;

struct lupine_kernel_param_layout {
  uint32_t count = 0;
  size_t offsets[64] = {};
  size_t sizes[64] = {};
};

struct lupine_kernel_param_layout_key {
  int route_id = -2;
  CUfunction function = nullptr;

  bool operator==(const lupine_kernel_param_layout_key &other) const {
    return route_id == other.route_id && function == other.function;
  }
};

struct lupine_kernel_param_layout_key_hash {
  size_t operator()(const lupine_kernel_param_layout_key &key) const {
    size_t hash = std::hash<int>{}(key.route_id);
    hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.function)) +
            0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

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

static bool lupine_mem_pool_attribute_size(CUmemPool_attribute attr,
                                           size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
    *size = sizeof(int);
    return true;
  case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
  case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
    *size = sizeof(cuuint64_t);
    return true;
  default:
    return false;
  }
}

static bool lupine_pointer_attribute_size(CUpointer_attribute attr,
                                          size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_POINTER_ATTRIBUTE_CONTEXT:
    *size = sizeof(CUcontext);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    *size = sizeof(void *);
    return true;
  case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    *size = sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS);
    return true;
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  case CU_POINTER_ATTRIBUTE_IS_MANAGED:
  case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
  case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
  case CU_POINTER_ATTRIBUTE_MAPPED:
  case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
  case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
#if CUDA_VERSION >= 13000
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
#endif
    *size = sizeof(int);
    return true;
  case CU_POINTER_ATTRIBUTE_BUFFER_ID:
  case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
    *size = sizeof(unsigned long long);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
  case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
  case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
    *size = sizeof(size_t);
    return true;
  case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
    *size = sizeof(CUmemoryPool);
    return true;
  default:
    return false;
  }
}

extern int rpc_size();
extern int rpc_open();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);

static bool lupine_trace_enabled() {
  static bool enabled = [] {
    const char *value = getenv("LUPINE_TRACE");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

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
      real_dlsym =
          reinterpret_cast<lupine_dlsym_fn>(dlvsym(RTLD_NEXT, "dlsym", version));
    }
  }
  return real_dlsym != nullptr ? real_dlsym(handle, name) : nullptr;
}

extern "C" CUresult lupine_unsupported_driver_api() {
  std::cerr << "LUPINE unsupported generic Driver API called" << std::endl;
  return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult lupine_missing_driver_api_called(const char *symbol) {
  std::cerr << "LUPINE missing Driver API called: " << symbol << std::endl;
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

static std::unordered_map<lupine_kernel_param_layout_key,
                          lupine_kernel_param_layout,
                          lupine_kernel_param_layout_key_hash> &
lupine_kernel_param_layout_cache() {
  static std::unordered_map<lupine_kernel_param_layout_key,
                            lupine_kernel_param_layout,
                            lupine_kernel_param_layout_key_hash>
      cache;
  return cache;
}

static std::mutex &lupine_kernel_param_layout_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

static std::vector<CUmodule> &lupine_loaded_modules() {
  static std::vector<CUmodule> modules;
  return modules;
}

static constexpr int LUPINE_ROUTE_REMOTE = 0;
static constexpr int LUPINE_ROUTE_LOCAL = 1;
static constexpr int LUPINE_ROUTE_INVALID = 2;

struct lupine_route {
  int kind = LUPINE_ROUTE_INVALID;
  conn_t *conn = nullptr;
};

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

struct lupine_primary_context_state {
  bool valid = false;
  unsigned int flags = 0;
  int active = 0;
};

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

static std::unordered_map<CUmodule, lupine_module_image_record> &
lupine_module_images() {
  static auto *images =
      new std::unordered_map<CUmodule, lupine_module_image_record>();
  return *images;
}

static std::unordered_map<CUlibrary, lupine_owner> &lupine_library_owners() {
  static auto *owners = new std::unordered_map<CUlibrary, lupine_owner>();
  return *owners;
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

static std::unordered_map<int, lupine_primary_context_state> &
lupine_primary_context_states() {
  static auto *states =
      new std::unordered_map<int, lupine_primary_context_state>();
  return *states;
}

static std::unordered_map<lupine_device_attribute_key, int,
                          lupine_device_attribute_key_hash> &
lupine_device_attribute_cache() {
  static auto *cache =
      new std::unordered_map<lupine_device_attribute_key, int,
                             lupine_device_attribute_key_hash>();
  return *cache;
}

static std::mutex &lupine_device_attribute_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<lupine_kernel_function_key, CUfunction,
                          lupine_kernel_function_key_hash> &
lupine_kernel_function_cache() {
  static auto *cache =
      new std::unordered_map<lupine_kernel_function_key, CUfunction,
                             lupine_kernel_function_key_hash>();
  return *cache;
}

static std::mutex &lupine_kernel_function_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<lupine_occupancy_key, int,
                          lupine_occupancy_key_hash> &
lupine_occupancy_cache() {
  static auto *cache = new std::unordered_map<lupine_occupancy_key, int,
                                              lupine_occupancy_key_hash>();
  return *cache;
}

static std::mutex &lupine_occupancy_cache_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
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

static int lupine_conn_index(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  for (int i = 0; i < nconns; ++i) {
    if (&conns[i] == conn) {
      return i;
    }
  }
  return -1;
}

static conn_t *lupine_conn_by_index(unsigned int index) {
  if (rpc_open() < 0 || index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
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

extern "C" void *lupine_real_cuda_symbol(const char *name) {
  void *handle = lupine_local_libcuda_handle();
  if (handle == nullptr || name == nullptr) {
    return nullptr;
  }
  return lupine_real_dlsym(handle, name);
}

template <typename Fn> static Fn lupine_real_cuda_fn(const char *name) {
  return reinterpret_cast<Fn>(lupine_real_cuda_symbol(name));
}

extern "C" bool lupine_route_is_local(lupine_route route) {
  return route.kind == LUPINE_ROUTE_LOCAL;
}

extern "C" conn_t *lupine_route_remote_conn(lupine_route route) {
  return route.kind == LUPINE_ROUTE_REMOTE ? route.conn : nullptr;
}

static lupine_route lupine_remote_route_for_conn(conn_t *conn) {
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
  return lupine_remote_route_for_conn(lupine_conn_by_index(owner.conn_index));
}

static int lupine_route_identity(lupine_route route) {
  if (route.kind == LUPINE_ROUTE_LOCAL) {
    return -1;
  }
  if (route.kind == LUPINE_ROUTE_REMOTE) {
    return lupine_conn_index(route.conn);
  }
  return -2;
}

extern "C" bool lupine_routes_share_server(lupine_route first,
                                           lupine_route second) {
  return lupine_route_identity(first) == lupine_route_identity(second);
}

extern "C" lupine_route lupine_route_for_deviceptr(CUdeviceptr ptr);

extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first,
                                              CUdeviceptr second) {
  return lupine_routes_share_server(lupine_route_for_deviceptr(first),
                                    lupine_route_for_deviceptr(second));
}

static lupine_route lupine_route_from_identity(int route_id) {
  if (route_id == -1) {
    return lupine_local_route();
  }
  if (route_id >= 0) {
    return lupine_remote_route_for_conn(
        lupine_conn_by_index(static_cast<unsigned int>(route_id)));
  }
  return lupine_route{LUPINE_ROUTE_INVALID, nullptr};
}

static int lupine_known_deviceptr_route_id(CUdeviceptr ptr) {
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

extern "C" lupine_route lupine_route_for_default();
static bool lupine_is_local_address(const void *ptr);

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
    for (int i = 0; i < nconns; ++i) {
      int remote_count = 0;
      CUresult result =
          lupine_remote_cuDeviceGetCount(&conns[i], &remote_count);
      if (result != CUDA_SUCCESS) {
        devices.clear();
        return result;
      }
      for (int ordinal = 0; ordinal < remote_count; ++ordinal) {
        CUdevice remote_device = 0;
        result = lupine_remote_cuDeviceGet(&conns[i], &remote_device, ordinal);
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

extern "C" CUresult lupine_cuInit_multi(unsigned int flags) {
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
  return initialized_any ? CUDA_SUCCESS : first_error;
}

extern "C" CUresult lupine_cuDeviceGetCount_multi(int *count) {
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

extern "C" CUresult lupine_cuDeviceGet_multi(CUdevice *device, int ordinal) {
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
  return lupine_conn_by_index(mapped.conn_index);
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
  return lupine_remote_route_for_conn(lupine_conn_by_index(mapped.conn_index));
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
  int conn_index = lupine_conn_index(conn);
  if (conn_index < 0) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto &devices = lupine_device_table();
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i].conn_index == static_cast<unsigned int>(conn_index) &&
        !devices[i].local && devices[i].remote_device == remote_device) {
      return static_cast<CUdevice>(i);
    }
  }
  return -1;
}

extern "C" lupine_route lupine_route_for_current_context();
extern "C" lupine_route lupine_route_for_context(CUcontext ctx);

extern "C" CUresult lupine_cuDeviceCanAccessPeer_multi(int *canAccessPeer,
                                                       CUdevice dev,
                                                       CUdevice peerDev) {
  if (canAccessPeer == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_device(&dev);
  if (lupine_route_is_local(route)) {
    CUdevice peer = peerDev;
    lupine_route peer_route = lupine_route_for_device(&peer);
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
    return CUDA_ERROR_INVALID_DEVICE;
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

extern "C" CUresult lupine_cuCtxEnablePeerAccess_multi(CUcontext peerContext,
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

extern "C" CUresult lupine_cuCtxDisablePeerAccess_multi(CUcontext peerContext) {
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

extern "C" lupine_route lupine_route_for_context(CUcontext ctx) {
  if (ctx == nullptr) {
    return lupine_route_for_default();
  }
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_context_owners().find(ctx);
    if (it != lupine_context_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
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

extern "C" void lupine_record_library_kernel(CUkernel kernel, CUlibrary library,
                                             const char *name,
                                             lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (kernel == nullptr || library == nullptr || name == nullptr ||
      route_id == -2) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &record = lupine_library_kernels()[kernel];
    record.library = library;
    record.name = name;
    record.kernels_by_route[route_id] = kernel;
  }
  lupine_note_function_owner_route(reinterpret_cast<CUfunction>(kernel), route);
}

extern "C" void lupine_record_module_function(CUfunction function,
                                              CUmodule module, const char *name,
                                              lupine_route route) {
  int route_id = lupine_route_identity(route);
  if (function == nullptr || module == nullptr || name == nullptr ||
      route_id == -2) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_library_kernel_mutex());
    auto &record = lupine_module_functions()[function];
    record.module = module;
    record.name = name;
    record.functions_by_route[route_id] = function;
  }
  lupine_note_function_owner_route(function, route);
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
        rpc_write(conn, record.image.data(), image_size) < 0 ||
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
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
        rpc_write(conn, &record.kind, sizeof(record.kind)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write(conn, record.image.data(), image_size) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &loaded, sizeof(loaded)) < 0 ||
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
  *resolved = route_function;
  return CUDA_SUCCESS;
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
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE could not resolve host kernel symbol for "
                << reinterpret_cast<void *>(function) << std::endl;
    }
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
      if (lupine_trace_enabled()) {
        std::cerr << "LUPINE mapped host kernel " << kernel_name
                  << " host=" << reinterpret_cast<void *>(function)
                  << " remote=" << remote << std::endl;
      }
      return remote;
    }
  }
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE host kernel " << kernel_name << " was not found in "
              << modules.size() << " loaded modules" << std::endl;
  }

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
    return CUDA_ERROR_NOT_FOUND;
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

extern "C" CUresult lupine_cuCtxGetCurrent_virtual(CUcontext *pctx);

extern "C" CUresult
lupine_cuDeviceGetAttribute_cached(int *pi, CUdevice_attribute attrib,
                                   CUdevice dev) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_device_attribute_key key{static_cast<int>(dev),
                                  static_cast<int>(attrib)};
  {
    std::lock_guard<std::mutex> lock(lupine_device_attribute_cache_mutex());
    auto it = lupine_device_attribute_cache().find(key);
    if (it != lupine_device_attribute_cache().end()) {
      *pi = it->second;
      return CUDA_SUCCESS;
    }
  }

  CUdevice remote_dev = dev;
  lupine_route route = lupine_route_for_device(&remote_dev);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(int *, CUdevice_attribute, CUdevice);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDeviceGetAttribute");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(pi, attrib, remote_dev);
    if (result == CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(lupine_device_attribute_cache_mutex());
      lupine_device_attribute_cache()[key] = *pi;
    }
    return result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
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
    std::lock_guard<std::mutex> lock(lupine_device_attribute_cache_mutex());
    lupine_device_attribute_cache()[key] = value;
    *pi = value;
  }
  return return_value;
}

extern "C" CUresult lupine_cuKernelGetFunction_cached(CUfunction *pFunc,
                                                      CUkernel kernel) {
  if (pFunc == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcontext current_context = nullptr;
  CUresult context_status = lupine_cuCtxGetCurrent_virtual(&current_context);
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
  lupine_kernel_function_key key{lupine_route_identity(route), current_context,
                                 route_kernel};
  {
    std::lock_guard<std::mutex> lock(lupine_kernel_function_cache_mutex());
    auto it = lupine_kernel_function_cache().find(key);
    if (it != lupine_kernel_function_cache().end()) {
      *pFunc = it->second;
      return CUDA_SUCCESS;
    }
  }

  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUfunction *, CUkernel);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuKernelGetFunction");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(pFunc, route_kernel);
    if (result == CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(lupine_kernel_function_cache_mutex());
      lupine_kernel_function_cache()[key] = *pFunc;
      lupine_note_function_owner_route(*pFunc, route);
    }
    return result;
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
    std::lock_guard<std::mutex> lock(lupine_kernel_function_cache_mutex());
    lupine_kernel_function_cache()[key] = function;
    *pFunc = function;
    lupine_note_function_owner(function, conn);
  }
  return return_value;
}

static CUresult lupine_cuOccupancy_cached(int *numBlocks, CUfunction func,
                                          int blockSize, size_t dynamicSMemSize,
                                          unsigned int flags, bool with_flags) {
  if (numBlocks == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(func);
  lupine_route route = lupine_route_for_function(translated);
  lupine_occupancy_key key{lupine_route_identity(route),
                           translated,
                           blockSize,
                           dynamicSMemSize,
                           flags,
                           with_flags};
  {
    std::lock_guard<std::mutex> lock(lupine_occupancy_cache_mutex());
    auto it = lupine_occupancy_cache().find(key);
    if (it != lupine_occupancy_cache().end()) {
      *numBlocks = it->second;
      return CUDA_SUCCESS;
    }
  }

  if (lupine_route_is_local(route)) {
    const char *symbol =
        with_flags ? "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
                   : "cuOccupancyMaxActiveBlocksPerMultiprocessor";
    using real_fn_t =
        CUresult (*)(int *, CUfunction, int, size_t, unsigned int);
    using real_no_flags_fn_t = CUresult (*)(int *, CUfunction, int, size_t);
    CUresult result;
    if (with_flags) {
      auto real = lupine_real_cuda_fn<real_fn_t>(symbol);
      if (real == nullptr) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      result = real(numBlocks, translated, blockSize, dynamicSMemSize, flags);
    } else {
      auto real = lupine_real_cuda_fn<real_no_flags_fn_t>(symbol);
      if (real == nullptr) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      result = real(numBlocks, translated, blockSize, dynamicSMemSize);
    }
    if (result == CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(lupine_occupancy_cache_mutex());
      lupine_occupancy_cache()[key] = *numBlocks;
    }
    return result;
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
    std::lock_guard<std::mutex> lock(lupine_occupancy_cache_mutex());
    lupine_occupancy_cache()[key] = remote_num_blocks;
    *numBlocks = remote_num_blocks;
  }
  return return_value;
}

extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  return lupine_cuOccupancy_cached(numBlocks, func, blockSize, dynamicSMemSize,
                                   0, false);
}

extern "C" CUresult
lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(
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

  if (lupine_trace_enabled()) {
    fprintf(stderr,
            "LUPINE private export table called: %s[%d] hash=%#llx "
            "args=%#llx,%#llx,%#llx,%#llx,%#llx,%#llx\n",
            table_name, slot,
            static_cast<unsigned long long>(
                lupine_private_export_slot_hash(table_name, slot)),
            static_cast<unsigned long long>(arg0),
            static_cast<unsigned long long>(arg1),
            static_cast<unsigned long long>(arg2),
            static_cast<unsigned long long>(arg3),
            static_cast<unsigned long long>(arg4),
            static_cast<unsigned long long>(arg5));
  }

  if (table_name != nullptr &&
      strcmp(table_name, "21318c60971432488ca641ff7324c8f2") == 0 &&
      slot == 4) {
    if (arg1 == 0) {
      return CUDA_ERROR_UNKNOWN;
    }
    uint64_t output = arg0 == 0 ? 0 : 1;
    *reinterpret_cast<uint64_t *>(arg1) = output;
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE private 2131[4] output=" << output << std::endl;
    }
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
      if (lupine_trace_enabled()) {
        std::cerr << "LUPINE private 6e16[7] remote node failed result="
                  << static_cast<int>(node_result)
                  << " module=" << reinterpret_cast<void *>(arg1) << std::endl;
      }
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
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE private 6e16[7] mapped client_node[" << node_index
                << "]=" << static_cast<void *>(client_node)
                << " server_node=" << server_node
                << " server_owner=" << reinterpret_cast<void *>(server_owner)
                << std::endl;
    }
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
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE remote cuGetExportTable metadata uuid=" << uuid_hex
              << " bytes=" << byte_size << " slots=" << slot_count << std::endl;
  }
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

static const char *lupine_error_name(CUresult error) {
  switch (error) {
  case CUDA_SUCCESS:
    return "CUDA_SUCCESS";
  case CUDA_ERROR_INVALID_VALUE:
    return "CUDA_ERROR_INVALID_VALUE";
  case CUDA_ERROR_OUT_OF_MEMORY:
    return "CUDA_ERROR_OUT_OF_MEMORY";
  case CUDA_ERROR_NOT_INITIALIZED:
    return "CUDA_ERROR_NOT_INITIALIZED";
  case CUDA_ERROR_DEINITIALIZED:
    return "CUDA_ERROR_DEINITIALIZED";
  case CUDA_ERROR_NO_DEVICE:
    return "CUDA_ERROR_NO_DEVICE";
  case CUDA_ERROR_INVALID_DEVICE:
    return "CUDA_ERROR_INVALID_DEVICE";
  case CUDA_ERROR_INVALID_IMAGE:
    return "CUDA_ERROR_INVALID_IMAGE";
  case CUDA_ERROR_INVALID_CONTEXT:
    return "CUDA_ERROR_INVALID_CONTEXT";
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
  case CUDA_ERROR_MAP_FAILED:
    return "CUDA_ERROR_MAP_FAILED";
  case CUDA_ERROR_UNMAP_FAILED:
    return "CUDA_ERROR_UNMAP_FAILED";
  case CUDA_ERROR_ARRAY_IS_MAPPED:
    return "CUDA_ERROR_ARRAY_IS_MAPPED";
  case CUDA_ERROR_ALREADY_MAPPED:
    return "CUDA_ERROR_ALREADY_MAPPED";
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    return "CUDA_ERROR_NO_BINARY_FOR_GPU";
  case CUDA_ERROR_ALREADY_ACQUIRED:
    return "CUDA_ERROR_ALREADY_ACQUIRED";
  case CUDA_ERROR_NOT_MAPPED:
    return "CUDA_ERROR_NOT_MAPPED";
  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
    return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
    return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
  case CUDA_ERROR_ECC_UNCORRECTABLE:
    return "CUDA_ERROR_ECC_UNCORRECTABLE";
  case CUDA_ERROR_UNSUPPORTED_LIMIT:
    return "CUDA_ERROR_UNSUPPORTED_LIMIT";
  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
    return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
    return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
  case CUDA_ERROR_INVALID_PTX:
    return "CUDA_ERROR_INVALID_PTX";
  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
    return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
  case CUDA_ERROR_NVLINK_UNCORRECTABLE:
    return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
  case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
    return "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
  case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
    return "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
  case CUDA_ERROR_JIT_COMPILATION_DISABLED:
    return "CUDA_ERROR_JIT_COMPILATION_DISABLED";
  case CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY:
    return "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY";
  case CUDA_ERROR_INVALID_SOURCE:
    return "CUDA_ERROR_INVALID_SOURCE";
  case CUDA_ERROR_FILE_NOT_FOUND:
    return "CUDA_ERROR_FILE_NOT_FOUND";
  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
    return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
    return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
  case CUDA_ERROR_OPERATING_SYSTEM:
    return "CUDA_ERROR_OPERATING_SYSTEM";
  case CUDA_ERROR_INVALID_HANDLE:
    return "CUDA_ERROR_INVALID_HANDLE";
  case CUDA_ERROR_ILLEGAL_STATE:
    return "CUDA_ERROR_ILLEGAL_STATE";
  case CUDA_ERROR_NOT_FOUND:
    return "CUDA_ERROR_NOT_FOUND";
  case CUDA_ERROR_NOT_READY:
    return "CUDA_ERROR_NOT_READY";
  case CUDA_ERROR_ILLEGAL_ADDRESS:
    return "CUDA_ERROR_ILLEGAL_ADDRESS";
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
  case CUDA_ERROR_LAUNCH_TIMEOUT:
    return "CUDA_ERROR_LAUNCH_TIMEOUT";
  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
    return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
    return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
    return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
  case CUDA_ERROR_ASSERT:
    return "CUDA_ERROR_ASSERT";
  case CUDA_ERROR_TOO_MANY_PEERS:
    return "CUDA_ERROR_TOO_MANY_PEERS";
  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
    return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
  case CUDA_ERROR_HARDWARE_STACK_ERROR:
    return "CUDA_ERROR_HARDWARE_STACK_ERROR";
  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
    return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
  case CUDA_ERROR_MISALIGNED_ADDRESS:
    return "CUDA_ERROR_MISALIGNED_ADDRESS";
  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
    return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
  case CUDA_ERROR_INVALID_PC:
    return "CUDA_ERROR_INVALID_PC";
  case CUDA_ERROR_LAUNCH_FAILED:
    return "CUDA_ERROR_LAUNCH_FAILED";
  case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
    return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
  case CUDA_ERROR_NOT_PERMITTED:
    return "CUDA_ERROR_NOT_PERMITTED";
  case CUDA_ERROR_NOT_SUPPORTED:
    return "CUDA_ERROR_NOT_SUPPORTED";
  case CUDA_ERROR_SYSTEM_NOT_READY:
    return "CUDA_ERROR_SYSTEM_NOT_READY";
  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
    return "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
  case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
    return "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
  case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
  case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
    return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
  case CUDA_ERROR_STREAM_CAPTURE_MERGE:
    return "CUDA_ERROR_STREAM_CAPTURE_MERGE";
  case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
  case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
    return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
  case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
    return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
  case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
    return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
  case CUDA_ERROR_CAPTURED_EVENT:
    return "CUDA_ERROR_CAPTURED_EVENT";
  case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
    return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
  case CUDA_ERROR_TIMEOUT:
    return "CUDA_ERROR_TIMEOUT";
  case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
    return "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
  case CUDA_ERROR_EXTERNAL_DEVICE:
    return "CUDA_ERROR_EXTERNAL_DEVICE";
  case CUDA_ERROR_UNKNOWN:
    return "CUDA_ERROR_UNKNOWN";
  default:
    return nullptr;
  }
}

extern "C" CUresult cuGetErrorName(CUresult error, const char **pStr) {
  if (pStr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const char *name = lupine_error_name(error);
  if (name == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = name;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuGetErrorString(CUresult error, const char **pStr) {
  return cuGetErrorName(error, pStr);
}

extern "C" CUresult cuProfilerInitialize(const char *, const char *,
                                         CUoutput_mode) {
  return CUDA_SUCCESS;
}

extern "C" CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }

extern "C" CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }

CUresult cuStreamDestroy_v2(CUstream hStream);
CUresult cuEventDestroy_v2(CUevent hEvent);
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

extern "C" CUresult lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice,
                                                   CUdeviceptr srcDevice,
                                                   size_t ByteCount,
                                                   CUstream hStream,
                                                   bool async) {
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cross-route D2D via client dst="
              << reinterpret_cast<void *>(dstDevice)
              << " src=" << reinterpret_cast<void *>(srcDevice)
              << " bytes=" << ByteCount << " async=" << async
              << " stream=" << hStream << std::endl;
  }
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
      if (result == CUDA_SUCCESS) {
        result = cuStreamSynchronize(hStream);
      }
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
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuMemPoolSetAttribute) < 0 ||
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
static std::atomic<unsigned long long> lupine_context_cache_generation{0};
static thread_local bool lupine_ctx_get_device_cache_valid = false;
static thread_local unsigned long long lupine_ctx_get_device_cache_generation =
    0;
static thread_local CUcontext lupine_ctx_get_device_cache_context = nullptr;
static thread_local CUdevice lupine_ctx_get_device_cache_device = -1;

extern "C" void lupine_invalidate_current_context_cache() {
  lupine_context_cache_generation.fetch_add(1, std::memory_order_acq_rel);
  lupine_ctx_get_device_cache_valid = false;
}

static void lupine_cache_current_context_device(CUdevice device) {
  lupine_ctx_get_device_cache_generation =
      lupine_context_cache_generation.load(std::memory_order_acquire);
  lupine_ctx_get_device_cache_context = lupine_current_context;
  lupine_ctx_get_device_cache_device = device;
  lupine_ctx_get_device_cache_valid = true;
}

static bool lupine_cached_current_context_device(CUdevice *device) {
  unsigned long long generation =
      lupine_context_cache_generation.load(std::memory_order_acquire);
  if (device == nullptr || !lupine_ctx_get_device_cache_valid ||
      lupine_ctx_get_device_cache_generation != generation ||
      lupine_ctx_get_device_cache_context != lupine_current_context) {
    return false;
  }
  *device = lupine_ctx_get_device_cache_device;
  return true;
}

extern "C" lupine_route lupine_route_for_current_context() {
  return lupine_route_for_context(lupine_current_context);
}

extern "C" lupine_route lupine_route_for_default() {
  if (lupine_current_context != nullptr) {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_context_owners().find(lupine_current_context);
    if (it != lupine_context_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  if (lupine_default_context_hint != nullptr) {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_context_owners().find(lupine_default_context_hint);
    if (it != lupine_context_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  CUcontext global_hint =
      lupine_global_default_context_hint.load(std::memory_order_relaxed);
  if (global_hint != nullptr) {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_context_owners().find(global_hint);
    if (it != lupine_context_owners().end()) {
      return lupine_route_for_owner(it->second);
    }
  }
  conn_t *conn = lupine_conn_by_index(0);
  if (conn != nullptr) {
    return lupine_remote_route_for_conn(conn);
  }
  return lupine_local_libcuda_handle() != nullptr
             ? lupine_local_route()
             : lupine_route{LUPINE_ROUTE_INVALID, nullptr};
}

extern "C" conn_t *lupine_rpc_conn_for_current_context() {
  return lupine_route_remote_conn(lupine_route_for_current_context());
}

static CUresult lupine_set_remote_current_context(CUcontext ctx) {
  lupine_route route =
      lupine_route_for_context(ctx != nullptr ? ctx : lupine_current_context);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUcontext);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxSetCurrent");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return real(ctx);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxSetCurrent) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" void lupine_note_ctx_create(CUcontext ctx, conn_t *conn) {
  lupine_note_context_owner(ctx, conn);
  lupine_context_stack->push_back(lupine_current_context);
  lupine_current_context = ctx;
  if (ctx != nullptr) {
    lupine_default_context_hint = ctx;
    lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
  }
  lupine_invalidate_current_context_cache();
}

extern "C" void lupine_note_ctx_create_route(CUcontext ctx,
                                             lupine_route route) {
  lupine_note_context_owner_route(ctx, route);
  lupine_context_stack->push_back(lupine_current_context);
  lupine_current_context = ctx;
  if (ctx != nullptr) {
    lupine_default_context_hint = ctx;
    lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
  }
  lupine_invalidate_current_context_cache();
}

extern "C" CUresult lupine_cuCtxPushCurrent_virtual(CUcontext ctx) {
  lupine_context_stack->push_back(lupine_current_context);
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    lupine_current_context = ctx;
    if (ctx != nullptr) {
      lupine_default_context_hint = ctx;
      lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
    }
    lupine_invalidate_current_context_cache();
  } else {
    lupine_context_stack->pop_back();
  }
  return result;
}

extern "C" CUresult lupine_cuCtxPopCurrent_virtual(CUcontext *pctx) {
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
    lupine_invalidate_current_context_cache();
  } else {
    lupine_context_stack->push_back(previous);
  }
  return result;
}

extern "C" CUresult lupine_cuCtxSetCurrent_virtual(CUcontext ctx) {
  CUresult result = lupine_set_remote_current_context(ctx);
  if (result == CUDA_SUCCESS) {
    lupine_current_context = ctx;
    if (ctx != nullptr) {
      lupine_default_context_hint = ctx;
      lupine_global_default_context_hint.store(ctx, std::memory_order_relaxed);
    }
    lupine_invalidate_current_context_cache();
  }
  return result;
}

extern "C" CUresult lupine_cuCtxGetCurrent_virtual(CUcontext *pctx) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pctx = lupine_current_context;
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_cuCtxGetDevice_cached(CUdevice *device) {
  if (device == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_cached_current_context_device(device)) {
    return CUDA_SUCCESS;
  }

  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUdevice *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuCtxGetDevice");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(device);
    if (result == CUDA_SUCCESS) {
      lupine_cache_current_context_device(*device);
    }
    return result;
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
    lupine_cache_current_context_device(*device);
  }
  return return_value;
}

extern "C" void lupine_note_primary_context_active(CUdevice dev) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto it = lupine_primary_context_states().find(static_cast<int>(dev));
  if (it != lupine_primary_context_states().end() && it->second.valid) {
    it->second.active = 1;
  }
}

extern "C" void lupine_note_primary_context_flags(CUdevice dev,
                                                  unsigned int flags) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  auto &state = lupine_primary_context_states()[static_cast<int>(dev)];
  state.flags = flags;
  if (!state.valid) {
    state.active = 0;
    state.valid = true;
  }
}

extern "C" void lupine_invalidate_primary_context_state(CUdevice dev) {
  std::lock_guard<std::mutex> lock(lupine_routing_mutex());
  lupine_primary_context_states().erase(static_cast<int>(dev));
}

extern "C" CUresult
lupine_cuDevicePrimaryCtxGetState_cached(CUdevice dev, unsigned int *flags,
                                         int *active) {
  if (flags == nullptr || active == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int local_device = static_cast<int>(dev);
  {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    auto it = lupine_primary_context_states().find(local_device);
    if (it != lupine_primary_context_states().end() && it->second.valid) {
      *flags = it->second.flags;
      *active = it->second.active;
      return CUDA_SUCCESS;
    }
  }

  CUdevice remote_device = dev;
  lupine_route route = lupine_route_for_device(&remote_device);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUdevice, unsigned int *, int *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuDevicePrimaryCtxGetState");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUresult result = real(remote_device, flags, active);
    if (result == CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(lupine_routing_mutex());
      lupine_primary_context_states()[local_device] =
          lupine_primary_context_state{true, *flags, *active};
    }
    return result;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxGetState) < 0 ||
      rpc_write(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(*flags)) < 0 ||
      rpc_read(conn, active, sizeof(*active)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(lupine_routing_mutex());
    lupine_primary_context_states()[local_device] =
        lupine_primary_context_state{true, *flags, *active};
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
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuMemPoolGetAttribute) < 0 ||
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

struct lupine_host_allocation {
  size_t size = 0;
  size_t storage_size = 0;
  size_t page_size = 0;
  size_t page_count = 0;
  unsigned int flags = 0;
  bool owned = false;
  bool owned_mmap = false;
  bool managed = false;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool tracking_enabled = false;
  bool all_host_dirty = false;
  bool dirty_overflow = false;
  std::vector<uint64_t> host_dirty_bits;
  std::vector<uint32_t> host_dirty_log;
  uint32_t host_dirty_count = 0;
};

struct lupine_mapped_host_snapshot {
  void *host = nullptr;
  size_t size = 0;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool managed = false;
};

using lupine_host_allocation_map =
    std::map<void *, lupine_host_allocation, std::less<void *>>;

static std::mutex &lupine_host_allocation_mutex() {
  static std::mutex mutex;
  return mutex;
}

static lupine_host_allocation_map &lupine_host_allocations() {
  static lupine_host_allocation_map allocations;
  return allocations;
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p);

struct lupine_fault_entry {
  uintptr_t base = 0;
  uintptr_t end = 0;
  lupine_host_allocation *allocation = nullptr;
};

static constexpr size_t LUPINE_MAX_FAULT_ENTRIES = 1024;
static lupine_fault_entry lupine_fault_entries[LUPINE_MAX_FAULT_ENTRIES];
static volatile sig_atomic_t lupine_fault_entry_count = 0;
static struct sigaction lupine_previous_sigsegv_action;
static bool lupine_sigsegv_handler_installed = false;

static size_t lupine_page_size() {
  long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

static size_t lupine_round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static void lupine_call_previous_sigsegv(int sig, siginfo_t *info, void *uctx) {
  if (lupine_previous_sigsegv_action.sa_flags & SA_SIGINFO) {
    if (lupine_previous_sigsegv_action.sa_sigaction != nullptr) {
      lupine_previous_sigsegv_action.sa_sigaction(sig, info, uctx);
      return;
    }
  } else if (lupine_previous_sigsegv_action.sa_handler == SIG_IGN) {
    return;
  } else if (lupine_previous_sigsegv_action.sa_handler != SIG_DFL &&
             lupine_previous_sigsegv_action.sa_handler != nullptr) {
    lupine_previous_sigsegv_action.sa_handler(sig);
    return;
  }

  sigaction(sig, &lupine_previous_sigsegv_action, nullptr);
  raise(sig);
}

static void lupine_sigsegv_handler(int sig, siginfo_t *info, void *uctx) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(info->si_addr);
  sig_atomic_t count = lupine_fault_entry_count;
  for (sig_atomic_t i = 0; i < count; ++i) {
    const lupine_fault_entry &entry = lupine_fault_entries[i];
    if (addr < entry.base || addr >= entry.end || entry.allocation == nullptr) {
      continue;
    }

    lupine_host_allocation *allocation = entry.allocation;
    size_t page_size = allocation->page_size;
    size_t page_index = (addr - entry.base) / page_size;
    if (page_index >= allocation->page_count) {
      break;
    }

    if (!allocation->all_host_dirty) {
      size_t word = page_index / 64;
      uint64_t bit = 1ULL << (page_index % 64);
      uint64_t previous = __atomic_fetch_or(&allocation->host_dirty_bits[word],
                                            bit, __ATOMIC_RELAXED);
      if ((previous & bit) == 0) {
        uint32_t slot = __atomic_fetch_add(&allocation->host_dirty_count, 1,
                                           __ATOMIC_RELAXED);
        if (slot < allocation->page_count) {
          allocation->host_dirty_log[slot] = static_cast<uint32_t>(page_index);
        } else {
          allocation->dirty_overflow = true;
        }
      }
    }

    void *page = reinterpret_cast<void *>(entry.base + page_index * page_size);
    mprotect(page, page_size, PROT_READ | PROT_WRITE);
    return;
  }

  lupine_call_previous_sigsegv(sig, info, uctx);
}

static void lupine_install_sigsegv_handler() {
  if (lupine_sigsegv_handler_installed) {
    return;
  }
  struct sigaction action = {};
  action.sa_sigaction = lupine_sigsegv_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_SIGINFO | SA_NODEFER;
  if (sigaction(SIGSEGV, &action, &lupine_previous_sigsegv_action) == 0) {
    lupine_sigsegv_handler_installed = true;
  }
}

static bool lupine_add_fault_entry(void *base, size_t size,
                                   lupine_host_allocation *allocation) {
  if (lupine_fault_entry_count >=
      static_cast<sig_atomic_t>(LUPINE_MAX_FAULT_ENTRIES)) {
    return false;
  }
  sig_atomic_t index = lupine_fault_entry_count;
  lupine_fault_entries[index] = {
      reinterpret_cast<uintptr_t>(base),
      reinterpret_cast<uintptr_t>(base) + size,
      allocation,
  };
  lupine_fault_entry_count = index + 1;
  return true;
}

static void lupine_remove_fault_entry(void *base) {
  uintptr_t target = reinterpret_cast<uintptr_t>(base);
  sig_atomic_t count = lupine_fault_entry_count;
  for (sig_atomic_t i = 0; i < count; ++i) {
    if (lupine_fault_entries[i].base != target) {
      continue;
    }
    for (sig_atomic_t j = i + 1; j < count; ++j) {
      lupine_fault_entries[j - 1] = lupine_fault_entries[j];
    }
    lupine_fault_entry_count = count - 1;
    return;
  }
}

static bool lupine_host_flags_request_mapping(unsigned int flags) {
  return (flags & (CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTREGISTER_DEVICEMAP)) !=
         0;
}

static bool lupine_protect_host_range(void *host, size_t size, int prot) {
  if (host == nullptr || size == 0) {
    return true;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  size_t page_size = lupine_page_size();
  uintptr_t page_start = start & ~(static_cast<uintptr_t>(page_size) - 1);
  uintptr_t end = lupine_round_up(start + size, page_size);
  return mprotect(reinterpret_cast<void *>(page_start), end - page_start,
                  prot) == 0;
}

static bool
lupine_enable_dirty_tracking_locked(void *host,
                                    lupine_host_allocation *allocation) {
  if (host == nullptr || allocation == nullptr || allocation->device_ptr == 0) {
    return true;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(host);
  if ((base % allocation->page_size) != 0 ||
      (allocation->storage_size % allocation->page_size) != 0) {
    return true;
  }

  allocation->tracking_enabled = true;
  allocation->all_host_dirty = true;
  allocation->host_dirty_bits.assign((allocation->page_count + 63) / 64, 0);
  allocation->host_dirty_log.assign(allocation->page_count, 0);
  allocation->host_dirty_count = 0;
  allocation->dirty_overflow = false;

  lupine_install_sigsegv_handler();
  if (!lupine_add_fault_entry(host, allocation->storage_size, allocation)) {
    allocation->tracking_enabled = false;
    return false;
  }
  if (!lupine_protect_host_range(host, allocation->storage_size, PROT_READ)) {
    lupine_remove_fault_entry(host);
    allocation->tracking_enabled = false;
    return false;
  }
  return true;
}

static void lupine_disable_dirty_tracking(void *host,
                                          lupine_host_allocation &allocation) {
  if (!allocation.tracking_enabled) {
    return;
  }
  lupine_protect_host_range(host, allocation.storage_size,
                            PROT_READ | PROT_WRITE);
  lupine_remove_fault_entry(host);
  allocation.tracking_enabled = false;
}

static std::vector<lupine_mapped_host_snapshot> lupine_mapped_host_snapshots() {
  std::vector<lupine_mapped_host_snapshot> snapshots;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (const auto &entry : lupine_host_allocations()) {
    if (entry.second.device_ptr != 0) {
      snapshots.push_back({entry.first, entry.second.size,
                           entry.second.device_ptr, entry.second.device_dirty,
                           entry.second.managed});
    }
  }
  return snapshots;
}

static bool lupine_dirty_bit_is_set(const lupine_host_allocation &allocation,
                                    size_t page_index) {
  size_t word = page_index / 64;
  uint64_t bit = 1ULL << (page_index % 64);
  return (allocation.host_dirty_bits[word] & bit) != 0;
}

static void lupine_clear_dirty_bit(lupine_host_allocation &allocation,
                                   size_t page_index) {
  size_t word = page_index / 64;
  uint64_t bit = 1ULL << (page_index % 64);
  allocation.host_dirty_bits[word] &= ~bit;
}

static CUresult lupine_copy_host_range_to_device(void *host,
                                                 CUdeviceptr device_ptr,
                                                 size_t offset, size_t bytes) {
  return cuMemcpyHtoD_v2(device_ptr + offset,
                         static_cast<unsigned char *>(host) + offset, bytes);
}

static CUresult
lupine_sync_dirty_host_pages_to_device(void *host,
                                       lupine_host_allocation &allocation) {
  if (allocation.device_ptr == 0 || allocation.size == 0) {
    return CUDA_SUCCESS;
  }

  if (!allocation.tracking_enabled) {
    return lupine_copy_host_range_to_device(host, allocation.device_ptr, 0,
                                            allocation.size);
  }

  if (allocation.all_host_dirty) {
    CUresult result = lupine_copy_host_range_to_device(
        host, allocation.device_ptr, 0, allocation.size);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    lupine_protect_host_range(host, allocation.storage_size, PROT_READ);
    std::fill(allocation.host_dirty_bits.begin(),
              allocation.host_dirty_bits.end(), 0);
    allocation.host_dirty_count = 0;
    allocation.dirty_overflow = false;
    allocation.all_host_dirty = false;
    return CUDA_SUCCESS;
  }

  std::vector<uint32_t> pages;
  uint32_t dirty_count = allocation.host_dirty_count;
  if (dirty_count > allocation.page_count) {
    dirty_count = static_cast<uint32_t>(allocation.page_count);
    allocation.dirty_overflow = true;
  }

  if (allocation.dirty_overflow) {
    for (size_t page = 0; page < allocation.page_count; ++page) {
      if (lupine_dirty_bit_is_set(allocation, page)) {
        pages.push_back(static_cast<uint32_t>(page));
      }
    }
  } else {
    pages.assign(allocation.host_dirty_log.begin(),
                 allocation.host_dirty_log.begin() + dirty_count);
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    pages.erase(std::remove_if(pages.begin(), pages.end(),
                               [&](uint32_t page) {
                                 return !lupine_dirty_bit_is_set(allocation,
                                                                 page);
                               }),
                pages.end());
  }

  size_t i = 0;
  while (i < pages.size()) {
    size_t first = pages[i];
    size_t last = first;
    ++i;
    while (i < pages.size() && pages[i] == last + 1) {
      last = pages[i];
      ++i;
    }

    size_t offset = first * allocation.page_size;
    size_t bytes = std::min((last - first + 1) * allocation.page_size,
                            allocation.size - offset);
    CUresult result = lupine_copy_host_range_to_device(
        host, allocation.device_ptr, offset, bytes);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    lupine_protect_host_range(static_cast<unsigned char *>(host) + offset,
                              bytes, PROT_READ);
    for (size_t page = first; page <= last; ++page) {
      lupine_clear_dirty_bit(allocation, page);
    }
  }

  allocation.host_dirty_count = 0;
  allocation.dirty_overflow = false;
  return CUDA_SUCCESS;
}

extern "C" void lupine_mark_host_range_clean(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_host_allocations().end() || !it->second.tracking_enabled) {
    return;
  }

  auto &allocation = it->second;
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + allocation.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / allocation.page_size;
  size_t last_page = (end - 1 - base) / allocation.page_size;
  bool covers_all = first_page == 0 && end >= base + allocation.size;

  if (allocation.all_host_dirty && covers_all) {
    allocation.all_host_dirty = false;
    std::fill(allocation.host_dirty_bits.begin(),
              allocation.host_dirty_bits.end(), 0);
    allocation.host_dirty_count = 0;
    allocation.dirty_overflow = false;
  } else if (!allocation.all_host_dirty) {
    for (size_t page = first_page; page <= last_page; ++page) {
      lupine_clear_dirty_bit(allocation, page);
    }
  }

  uintptr_t protect_start = base + first_page * allocation.page_size;
  size_t protect_size = (last_page - first_page + 1) * allocation.page_size;
  lupine_protect_host_range(reinterpret_cast<void *>(protect_start),
                            protect_size, PROT_READ);
}

extern "C" void lupine_prepare_host_range_write(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_host_allocations().end() || !it->second.tracking_enabled) {
    return;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + it->second.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / it->second.page_size;
  size_t last_page = (end - 1 - base) / it->second.page_size;
  uintptr_t protect_start = base + first_page * it->second.page_size;
  size_t protect_size = (last_page - first_page + 1) * it->second.page_size;
  lupine_protect_host_range(reinterpret_cast<void *>(protect_start),
                            protect_size, PROT_READ | PROT_WRITE);
}

CUresult lupine_sync_mapped_host_to_device() {
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (auto &entry : lupine_host_allocations()) {
    CUresult result =
        lupine_sync_dirty_host_pages_to_device(entry.first, entry.second);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  return CUDA_SUCCESS;
}

static bool lupine_device_ptr_in_mapping(CUdeviceptr ptr,
                                         const lupine_mapped_host_snapshot &m) {
  return ptr >= m.device_ptr && ptr < m.device_ptr + m.size;
}

static bool lupine_host_ptr_in_mapping(CUdeviceptr ptr,
                                       const lupine_mapped_host_snapshot &m,
                                       CUdeviceptr *translated) {
  uintptr_t host = reinterpret_cast<uintptr_t>(m.host);
  if (ptr < host || ptr >= host + m.size) {
    return false;
  }
  if (translated != nullptr) {
    *translated = m.device_ptr + (ptr - host);
  }
  return true;
}

static bool lupine_translate_managed_host_ptr(CUdeviceptr ptr,
                                              CUdeviceptr *translated) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_host_allocations().end() || !it->second.managed) {
    return false;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t addr = reinterpret_cast<uintptr_t>(host);
  if (translated != nullptr) {
    *translated = it->second.device_ptr + (addr - base);
  }
  return true;
}

static void lupine_mark_mapped_device_dirty(void *host) {
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_host_allocations().find(host);
  if (it != lupine_host_allocations().end()) {
    it->second.device_dirty = true;
  }
}

CUresult lupine_sync_mapped_host_to_device_for_launch(unsigned char *packed,
                                                      const size_t *offsets,
                                                      const size_t *sizes,
                                                      uint32_t count) {
  if (packed == nullptr || offsets == nullptr || sizes == nullptr) {
    return count == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  }
  CUresult result = lupine_sync_mapped_host_to_device();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::vector<lupine_mapped_host_snapshot> snapshots =
      lupine_mapped_host_snapshots();
  for (const auto &mapping : snapshots) {
    for (uint32_t i = 0; i < count; ++i) {
      if (sizes[i] != sizeof(CUdeviceptr)) {
        continue;
      }
      CUdeviceptr arg = 0;
      memcpy(&arg, packed + offsets[i], sizeof(arg));
      CUdeviceptr translated = 0;
      if (mapping.managed &&
          lupine_host_ptr_in_mapping(arg, mapping, &translated)) {
        memcpy(packed + offsets[i], &translated, sizeof(translated));
        lupine_mark_mapped_device_dirty(mapping.host);
        break;
      }
      if (lupine_device_ptr_in_mapping(arg, mapping)) {
        lupine_mark_mapped_device_dirty(mapping.host);
        break;
      }
    }
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_sync_mapped_device_to_host() {
  for (const auto &mapping : lupine_mapped_host_snapshots()) {
    if (!mapping.device_dirty || mapping.size == 0) {
      continue;
    }
    CUresult result =
        cuMemcpyDtoH_v2(mapping.host, mapping.device_ptr, mapping.size);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_host_allocations().find(mapping.host);
    if (it != lupine_host_allocations().end()) {
      it->second.device_dirty = false;
    }
  }
  return CUDA_SUCCESS;
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p) {
  auto &allocations = lupine_host_allocations();
  auto exact = allocations.find(p);
  if (exact != allocations.end()) {
    return exact;
  }

  auto upper = allocations.upper_bound(p);
  if (upper == allocations.begin()) {
    return allocations.end();
  }

  auto it = std::prev(upper);
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  if (addr >= base && addr < base + it->second.size) {
    return it;
  }
  return allocations.end();
}

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize,
                                   unsigned int Flags) {
  if (pp == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr unsigned int supported_flags = CU_MEMHOSTALLOC_PORTABLE |
                                           CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  if (lupine_host_flags_request_mapping(Flags)) {
    CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  void *ptr = nullptr;
  size_t page_size = lupine_page_size();
  size_t storage_size = lupine_round_up(bytesize, page_size);
  bool owned_mmap = device_ptr != 0;
  if (owned_mmap) {
    ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      ptr = nullptr;
    }
  } else if (posix_memalign(&ptr, page_size, bytesize) != 0) {
    ptr = nullptr;
  }
  if (ptr == nullptr) {
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = Flags;
    allocation.owned = true;
    allocation.owned_mmap = owned_mmap;
    allocation.device_ptr = device_ptr;
    auto inserted =
        lupine_host_allocations().emplace(ptr, std::move(allocation));
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      lupine_host_allocations().erase(inserted.first);
      if (owned_mmap) {
        munmap(ptr, storage_size);
      } else {
        free(ptr);
      }
      if (device_ptr != 0) {
        cuMemFree_v2(device_ptr);
      }
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  *pp = ptr;
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE local cuMemHostAlloc ptr=" << ptr
              << " bytes=" << bytesize << " flags=" << Flags << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}

#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
extern "C" CUresult cuMemAllocHost(void **pp, size_t bytesize) {
  return cuMemAllocHost_v2(pp, bytesize);
}

extern "C" CUresult cuMemFreeHost(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  bool owned = false;
  bool owned_mmap = false;
  size_t storage_size = 0;
  CUdeviceptr device_ptr = 0;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_host_allocations();
    auto it = allocations.find(p);
    if (it == allocations.end()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    owned = it->second.owned;
    owned_mmap = it->second.owned_mmap;
    storage_size = it->second.storage_size;
    device_ptr = it->second.device_ptr;
    lupine_disable_dirty_tracking(p, it->second);
    allocations.erase(it);
  }
  if (owned) {
    if (owned_mmap) {
      munmap(p, storage_size);
    } else {
      free(p);
    }
  }
  if (device_ptr != 0) {
    return cuMemFree_v2(device_ptr);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p,
                                                 unsigned int Flags) {
  if (pdptr == nullptr || p == nullptr || Flags != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(p);
  if (it == lupine_host_allocations().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((it->second.flags & CU_MEMHOSTALLOC_DEVICEMAP) == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  *pdptr = it->second.device_ptr + (addr - base);
  return CUDA_SUCCESS;
}

#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
extern "C" CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                              unsigned int Flags) {
  return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (pFlags == nullptr || p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(p);
  if (it == lupine_host_allocations().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pFlags = it->second.flags;
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize,
                                         unsigned int Flags) {
  if (p == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr unsigned int supported_flags =
      CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP |
      CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_READ_ONLY;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  if (lupine_host_flags_request_mapping(Flags)) {
    CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  if (lupine_host_allocations().find(p) != lupine_host_allocations().end()) {
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  }
  size_t page_size = lupine_page_size();
  lupine_host_allocation allocation;
  allocation.size = bytesize;
  allocation.storage_size = bytesize;
  allocation.page_size = page_size;
  allocation.page_count = lupine_round_up(bytesize, page_size) / page_size;
  allocation.flags = Flags;
  allocation.owned = false;
  allocation.owned_mmap = false;
  allocation.device_ptr = device_ptr;
  auto inserted = lupine_host_allocations().emplace(p, std::move(allocation));
  if (!lupine_enable_dirty_tracking_locked(p, &inserted.first->second)) {
    lupine_host_allocations().erase(inserted.first);
    if (device_ptr != 0) {
      cuMemFree_v2(device_ptr);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
}

#ifdef cuMemHostRegister
#undef cuMemHostRegister
#endif
extern "C" CUresult cuMemHostRegister(void *p, size_t bytesize,
                                      unsigned int Flags) {
  return cuMemHostRegister_v2(p, bytesize, Flags);
}

extern "C" CUresult cuMemHostUnregister(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUdeviceptr device_ptr = 0;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_host_allocations();
    auto it = allocations.find(p);
    if (it == allocations.end() || it->second.owned) {
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    device_ptr = it->second.device_ptr;
    lupine_disable_dirty_tracking(p, it->second);
    allocations.erase(it);
  }
  if (device_ptr != 0) {
    return cuMemFree_v2(device_ptr);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_cuMemAllocManaged_safe(CUdeviceptr *dptr,
                                                  size_t bytesize,
                                                  unsigned int flags) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr device_ptr = 0;
  CUresult result = cuMemAlloc_v2(&device_ptr, bytesize);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  size_t page_size = lupine_page_size();
  size_t storage_size = lupine_round_up(bytesize, page_size);
  void *ptr = MAP_FAILED;
#ifdef MAP_FIXED_NOREPLACE
  if ((device_ptr % page_size) == 0) {
    ptr = mmap(reinterpret_cast<void *>(device_ptr), storage_size,
               PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  }
#endif
  if (ptr == MAP_FAILED) {
    ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if (ptr == MAP_FAILED) {
    cuMemFree_v2(device_ptr);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.managed = true;
    allocation.device_ptr = device_ptr;
    auto inserted =
        lupine_host_allocations().emplace(ptr, std::move(allocation));
    if (!inserted.second) {
      munmap(ptr, storage_size);
      cuMemFree_v2(device_ptr);
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      lupine_host_allocations().erase(inserted.first);
      munmap(ptr, storage_size);
      cuMemFree_v2(device_ptr);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  *dptr = reinterpret_cast<CUdeviceptr>(ptr);
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_cuMemFree_v2_safe(CUdeviceptr dptr) {
  void *host = reinterpret_cast<void *>(dptr);
  lupine_host_allocation allocation;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(host);
    if (it != lupine_host_allocations().end() &&
        reinterpret_cast<void *>(dptr) == it->first && it->second.managed) {
      allocation = std::move(it->second);
      lupine_disable_dirty_tracking(it->first, allocation);
      lupine_host_allocations().erase(it);
      found = true;
    }
  }
  if (!found) {
    lupine_route route = lupine_route_for_deviceptr(dptr);
    if (lupine_route_is_local(route)) {
      using real_fn_t = CUresult (*)(CUdeviceptr);
      auto real = lupine_real_cuda_fn<real_fn_t>("cuMemFree_v2");
      if (real == nullptr) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      CUresult result = real(dptr);
      if (result == CUDA_SUCCESS) {
        lupine_forget_deviceptr_owner(dptr);
      }
      return result;
    }
    conn_t *conn = lupine_route_remote_conn(route);
    CUresult return_value;
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_cuMemFree_v2) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(dptr);
    }
    return return_value;
  }

  CUresult result = CUDA_SUCCESS;
  if (allocation.device_ptr != 0) {
    result = cuMemFree_v2(allocation.device_ptr);
  }
  if (allocation.owned_mmap && host != nullptr) {
    munmap(host, allocation.storage_size);
  } else if (allocation.owned && host != nullptr) {
    free(host);
  }
  return result;
}

extern "C" CUresult cuPointerGetAttribute(void *data,
                                          CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
  if (data == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!lupine_pointer_attribute_size(attribute, &value_size)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  unsigned char value[64] = {};
  if (value_size > sizeof(value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_translate_managed_host_ptr(ptr, &query_ptr);
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(void *, CUpointer_attribute, CUdeviceptr);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuPointerGetAttribute");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(data, attribute, query_ptr);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuPointerGetAttribute) < 0 ||
      rpc_write(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_write(conn, &query_ptr, sizeof(query_ptr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, value_size) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    memcpy(data, value, value_size);
    if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
      void *host = reinterpret_cast<void *>(ptr);
      memcpy(data, &host, sizeof(host));
    } else if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
      int is_managed = 1;
      memcpy(data, &is_managed, sizeof(is_managed));
    }
  }
  return return_value;
}

extern "C" CUresult
lupine_cuPointerGetAttributes_safe(unsigned int numAttributes,
                                   CUpointer_attribute *attributes, void **data,
                                   CUdeviceptr ptr) {
  if (numAttributes != 0 && (attributes == nullptr || data == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::vector<size_t> value_sizes(numAttributes, 0);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    if (!lupine_pointer_attribute_size(attributes[i], &value_sizes[i]) ||
        value_sizes[i] > 64) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_translate_managed_host_ptr(ptr, &query_ptr);
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(unsigned int, CUpointer_attribute *, void **, CUdeviceptr);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuPointerGetAttributes");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(numAttributes, attributes, data, query_ptr);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (rpc_write_start_request(conn, LUPINE_RPC_cuPointerGetAttributes) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(numAttributes)) < 0 ||
      (numAttributes != 0 &&
       rpc_write(conn, attributes,
                 numAttributes * sizeof(CUpointer_attribute)) < 0) ||
      rpc_write(conn, &query_ptr, sizeof(query_ptr)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  std::vector<std::vector<unsigned char>> values(numAttributes);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    size_t remote_size = 0;
    if (rpc_read(conn, &remote_size, sizeof(remote_size)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (remote_size > 64) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    values[i].resize(remote_size);
    if (remote_size != 0 && rpc_read(conn, values[i].data(), remote_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }

  CUresult return_value;
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS) {
    for (unsigned int i = 0; i < numAttributes; ++i) {
      if (data[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      if (values[i].size() > value_sizes[i]) {
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      memcpy(data[i], values[i].data(), values[i].size());
      if (managed_alias && attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
        void *host = reinterpret_cast<void *>(ptr);
        memcpy(data[i], &host, sizeof(host));
      } else if (managed_alias &&
                 attributes[i] == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
        int is_managed = 1;
        memcpy(data[i], &is_managed, sizeof(is_managed));
      }
    }
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

struct lupine_jit_client_binding {
  CUjit_option option;
  void *dst;
  size_t size;
};

struct lupine_jit_client_state {
  std::vector<lupine_jit_client_binding> bindings;
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

static std::vector<uintptr_t>
lupine_pack_jit_option_values(unsigned int numOptions,
                              void *const *optionValues) {
  std::vector<uintptr_t> values(numOptions);
  for (unsigned int i = 0; i < numOptions; ++i) {
    values[i] = reinterpret_cast<uintptr_t>(
        optionValues == nullptr ? nullptr : optionValues[i]);
  }
  return values;
}

static std::vector<lupine_jit_client_binding>
lupine_capture_jit_client_bindings(unsigned int numOptions,
                                   const CUjit_option *options,
                                   void *const *optionValues) {
  std::vector<lupine_jit_client_binding> bindings;
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

static int lupine_write_jit_options(conn_t *conn, unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  auto packed_values = lupine_pack_jit_option_values(numOptions, optionValues);
  return rpc_write(conn, &numOptions, sizeof(numOptions)) < 0 ||
                 (numOptions != 0 &&
                  rpc_write(conn, options, numOptions * sizeof(CUjit_option)) <
                      0) ||
                 (numOptions != 0 &&
                  rpc_write(conn, packed_values.data(),
                            numOptions * sizeof(uintptr_t)) < 0)
             ? -1
             : 0;
}

static int lupine_apply_jit_outputs(conn_t *conn, CUlinkState state) {
  uint32_t output_count = 0;
  if (rpc_read(conn, &output_count, sizeof(output_count)) < 0) {
    return -1;
  }
  if (lupine_trace_enabled()) {
    fprintf(stderr, "LUPINE cuLink JIT outputs state=%p count=%u\n",
            reinterpret_cast<void *>(state), output_count);
  }
  if (output_count > 32) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
  auto state_it = lupine_jit_client_states().find(state);
  for (uint32_t i = 0; i < output_count; ++i) {
    CUjit_option option;
    size_t payload_size = 0;
    if (rpc_read(conn, &option, sizeof(option)) < 0 ||
        rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
      return -1;
    }
    if (lupine_trace_enabled()) {
      fprintf(stderr,
              "LUPINE cuLink JIT output state=%p index=%u option=%d size=%zu\n",
              reinterpret_cast<void *>(state), i, static_cast<int>(option),
              payload_size);
    }
    if (payload_size > (16ull << 20)) {
      return -1;
    }
    std::vector<unsigned char> payload(payload_size);
    if (payload_size != 0 && rpc_read(conn, payload.data(), payload_size) < 0) {
      return -1;
    }
    if (state_it == lupine_jit_client_states().end()) {
      continue;
    }
    for (const auto &binding : state_it->second.bindings) {
      if (binding.option == option && binding.dst != nullptr) {
        memcpy(binding.dst, payload.data(),
               std::min(binding.size, payload.size()));
        break;
      }
    }
  }
  return 0;
}

extern "C" CUresult lupine_cuLinkCreate_v2_safe(unsigned int numOptions,
                                                CUjit_option *options,
                                                void **optionValues,
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
      lupine_write_jit_options(conn, numOptions, options, optionValues) < 0 ||
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

extern "C" CUresult
lupine_cuLinkAddData_v2_safe(CUlinkState state, CUjitInputType type, void *data,
                             size_t size, const char *name,
                             unsigned int numOptions, CUjit_option *options,
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
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuLinkAddData_v2) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      (size != 0 && rpc_write(conn, data, size) < 0) ||
      rpc_write(conn, &name_len, sizeof(name_len)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      lupine_write_jit_options(conn, numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_apply_jit_outputs(conn, state) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
lupine_cuLinkAddFile_v2_safe(CUlinkState state, CUjitInputType type,
                             const char *path, unsigned int numOptions,
                             CUjit_option *options, void **optionValues) {
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
  int file_fd = open(path, O_RDONLY);
  if (file_fd < 0) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  struct stat st = {};
  if (fstat(file_fd, &st) < 0 || st.st_size <= 0) {
    close(file_fd);
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  mapped_file_size = static_cast<size_t>(st.st_size);
  file_mapping =
      mmap(nullptr, mapped_file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);
  close(file_fd);
  if (file_mapping == MAP_FAILED) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  file_payload = file_mapping;
  file_size = mapped_file_size;
  has_file_data = 1;
  bool failed =
      conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLinkAddFile_v2) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      rpc_write(conn, &path_len, sizeof(path_len)) < 0 ||
      (path_len != 0 && rpc_write(conn, path, path_len) < 0) ||
      rpc_write(conn, &has_file_data, sizeof(has_file_data)) < 0 ||
      rpc_write(conn, &file_size, sizeof(file_size)) < 0 ||
      (file_size != 0 && rpc_write(conn, file_payload, mapped_file_size) < 0) ||
      lupine_write_jit_options(conn, numOptions, options, optionValues) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_apply_jit_outputs(conn, state) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0;
  if (file_mapping != MAP_FAILED) {
    munmap(file_mapping, mapped_file_size);
  }
  if (failed) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult lupine_cuLinkComplete_safe(CUlinkState state,
                                               void **cubinOut,
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
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLinkComplete) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &cubin_size, sizeof(cubin_size)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (lupine_trace_enabled()) {
    fprintf(stderr, "LUPINE cuLinkComplete state=%p cubin_size=%zu\n",
            reinterpret_cast<void *>(state), cubin_size);
  }

  {
    std::lock_guard<std::mutex> lock(lupine_jit_client_mutex());
    auto &jit_state = lupine_jit_client_states()[state];
    if (cubin_size > (256ull << 20)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    jit_state.cubin.resize(cubin_size);
    if (cubin_size != 0 &&
        rpc_read(conn, jit_state.cubin.data(), cubin_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    *cubinOut = jit_state.cubin.empty() ? nullptr : jit_state.cubin.data();
    *sizeOut = jit_state.cubin.size();
  }

  if (lupine_apply_jit_outputs(conn, state) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult lupine_cuLinkDestroy_safe(CUlinkState state) {
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

extern "C" CUresult
lupine_cuArrayCreate_v2_safe(CUarray *pHandle,
                             const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  if (pHandle == nullptr || pAllocateArray == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuArrayCreate_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(pHandle, pAllocateArray);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuArrayCreate_v2) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(*pAllocateArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(*pHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult
lupine_cuArray3DCreate_v2_safe(CUarray *pHandle,
                               const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  if (pHandle == nullptr || pAllocateArray == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuArray3DCreate_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(pHandle, pAllocateArray);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuArray3DCreate_v2) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(*pAllocateArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(*pHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static int lupine_forward_remote_stdout(conn_t *conn) {
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
    if (rpc_read(conn, dst, bytes) < 0) {
      return -1;
    }
    lupine_mark_host_range_clean(dst, bytes);
  }
  return 0;
}

static CUresult lupine_rpc_noarg_driver_call(int op) {
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route)) {
    const char *symbol =
        op == RPC_cuCtxSynchronize ? "cuCtxSynchronize" : nullptr;
    if (symbol == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    using real_fn_t = CUresult (*)();
    auto real = lupine_real_cuda_fn<real_fn_t>(symbol);
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real();
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (op == RPC_cuCtxSynchronize &&
       lupine_read_deferred_dtoh_copies(conn) < 0) ||
      (op == RPC_cuCtxSynchronize && lupine_forward_remote_stdout(conn) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_rpc_stream_driver_call(int op, CUstream stream) {
  lupine_route route = stream == nullptr ? lupine_route_for_current_context()
                                         : lupine_route_for_stream(stream);
  if (lupine_route_is_local(route)) {
    const char *symbol = nullptr;
    if (op == RPC_cuStreamQuery) {
      symbol = "cuStreamQuery";
    } else if (op == RPC_cuStreamDestroy_v2) {
      symbol = "cuStreamDestroy_v2";
    }
    if (symbol == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    using real_fn_t = CUresult (*)(CUstream);
    auto real = lupine_real_cuda_fn<real_fn_t>(symbol);
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(stream);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_rpc_event_driver_call(int op, CUevent event) {
  lupine_route route = lupine_route_for_event(event);
  if (lupine_route_is_local(route)) {
    const char *symbol = nullptr;
    if (op == RPC_cuEventSynchronize) {
      symbol = "cuEventSynchronize";
    } else if (op == RPC_cuEventQuery) {
      symbol = "cuEventQuery";
    } else if (op == RPC_cuEventDestroy_v2) {
      symbol = "cuEventDestroy_v2";
    }
    if (symbol == nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    using real_fn_t = CUresult (*)(CUevent);
    auto real = lupine_real_cuda_fn<real_fn_t>(symbol);
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(event);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &event, sizeof(event)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      ((op == RPC_cuEventSynchronize || op == RPC_cuEventQuery) &&
       lupine_read_deferred_dtoh_copies(conn) < 0) ||
      (op == RPC_cuEventSynchronize &&
       lupine_forward_remote_stdout(conn) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuCtxSynchronize() {
  CUresult result = lupine_rpc_noarg_driver_call(RPC_cuCtxSynchronize);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return lupine_sync_mapped_device_to_host();
}

extern "C" CUresult cuStreamSynchronize(CUstream hStream) {
  lupine_route route = hStream == nullptr ? lupine_route_for_current_context()
                                          : lupine_route_for_stream(hStream);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUstream);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuStreamSynchronize");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(hStream);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_UNKNOWN;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamSynchronize) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return lupine_sync_mapped_device_to_host();
}

#ifdef cuStreamSynchronize_ptsz
#undef cuStreamSynchronize_ptsz
#endif
extern "C" CUresult cuStreamSynchronize_ptsz(CUstream hStream) {
  return cuStreamSynchronize(hStream);
}

extern "C" CUresult cuEventSynchronize(CUevent hEvent) {
  CUresult result =
      lupine_rpc_event_driver_call(RPC_cuEventSynchronize, hEvent);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return lupine_sync_mapped_device_to_host();
}

extern "C" CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                   CUdevice dev) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_device(&dev);
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
      rpc_write_start_request(conn, LUPINE_RPC_cuCtxCreate_v2) < 0 ||
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

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  int op = with_flags ? RPC_cuOccupancyMaxPotentialBlockSizeWithFlags
                      : RPC_cuOccupancyMaxPotentialBlockSize;
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
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

static bool lupine_pack_module_image(const void *image, uint32_t *kind,
                                     std::vector<unsigned char> *bytes) {
  if (image == nullptr || kind == nullptr || bytes == nullptr) {
    return false;
  }

  const auto *wrapper = reinterpret_cast<const lupine::fatbin_wrapper *>(image);
  const void *fatbin = image;
  if (wrapper->magic == lupine::kFatbinWrapperMagic &&
      (wrapper->version == 1 || wrapper->version == 2) &&
      wrapper->data != nullptr) {
    fatbin = wrapper->data;
    *kind = wrapper->version == 2 ? lupine::kModuleImageFatbinWrapperV2
                                  : lupine::kModuleImageFatbinWrapperV1;
  } else {
    *kind = lupine::kModuleImageFatbinRaw;
  }

  const auto *header = reinterpret_cast<const lupine::fatbin_header *>(fatbin);
  if (header->magic != lupine::kFatbinMagic || header->header_size == 0) {
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
      *kind = lupine::kModuleImageFatbinRaw;
      bytes->assign(elf, elf + image_size);
      return true;
    }

    const char *ptx = static_cast<const char *>(image);
    const char *ptx_start = ptx;
    while (*ptx_start == ' ' || *ptx_start == '\t' || *ptx_start == '\r' ||
           *ptx_start == '\n') {
      ++ptx_start;
    }
    if (std::strncmp(ptx_start, ".version", 8) == 0 ||
        std::strncmp(ptx_start, "//", 2) == 0) {
      *kind = lupine::kModuleImageFatbinRaw;
      bytes->assign(reinterpret_cast<const unsigned char *>(ptx),
                    reinterpret_cast<const unsigned char *>(ptx) +
                        std::strlen(ptx) + 1);
      return true;
    }

    return false;
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
    return CUDA_ERROR_NOT_SUPPORTED;
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
      rpc_write(conn, image_bytes.data(), image_size) < 0 ||
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

extern "C" CUresult
cuLibraryLoadData(CUlibrary *library, const void *code,
                  CUjit_option *jitOptions, void **jitOptionsValues,
                  unsigned int numJitOptions, CUlibraryOption *libraryOptions,
                  void **libraryOptionValues, unsigned int numLibraryOptions) {
  if (library == nullptr || code == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  bool can_ignore_library_options =
      numLibraryOptions == 1 && libraryOptions != nullptr &&
      (libraryOptions[0] == CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE ||
       libraryOptions[0] == CU_LIBRARY_BINARY_IS_PRESERVED);
  if (numJitOptions != 0 ||
      !(numLibraryOptions == 0 || can_ignore_library_options)) {
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE cuLibraryLoadData unsupported options: jit="
                << numJitOptions << " library=" << numLibraryOptions
                << " first_library_option="
                << (libraryOptions == nullptr ? -1 : (int)libraryOptions[0])
                << std::endl;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  uint32_t kind = 0;
  std::vector<unsigned char> image_bytes;
  if (!lupine_pack_module_image(code, &kind, &image_bytes)) {
    if (lupine_trace_enabled()) {
      const auto *wrapper =
          reinterpret_cast<const lupine::fatbin_wrapper *>(code);
      std::cerr << "LUPINE cuLibraryLoadData could not pack image"
                << " magic=0x" << std::hex << wrapper->magic << " version=0x"
                << wrapper->version << std::dec << std::endl;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = lupine_rpc_conn_for_current_context();
  CUresult return_value;
  size_t image_size = image_bytes.size();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryLoadData) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
      rpc_write(conn, image_bytes.data(), image_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    lupine_note_library_owner(*library, conn);
    lupine_record_library_image(*library, lupine_remote_route_for_conn(conn),
                                kind, image_bytes.data(), image_bytes.size(),
                                code);
  }
  return return_value;
}

static CUresult
lupine_fetch_kernel_param_layout(CUfunction f,
                                 lupine_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_rpc_conn_for_function(f);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuFuncGetParamLayout) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &layout->count, sizeof(layout->count)) < 0 ||
      rpc_read(conn, layout->offsets, sizeof(layout->offsets)) < 0 ||
      rpc_read(conn, layout->sizes, sizeof(layout->sizes)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" void lupine_invalidate_kernel_param_layout_cache() {
  {
    std::lock_guard<std::mutex> lock(lupine_kernel_param_layout_cache_mutex());
    lupine_kernel_param_layout_cache().clear();
  }
  {
    std::lock_guard<std::mutex> lock(lupine_kernel_function_cache_mutex());
    lupine_kernel_function_cache().clear();
  }
  {
    std::lock_guard<std::mutex> lock(lupine_occupancy_cache_mutex());
    lupine_occupancy_cache().clear();
  }
}

extern "C" CUresult
lupine_get_kernel_param_layout_cached(CUfunction f,
                                      lupine_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  f = lupine_translate_private_function(f);
  lupine_route route = lupine_route_for_function(f);
  lupine_kernel_param_layout_key key{lupine_route_identity(route), f};
  {
    std::lock_guard<std::mutex> lock(lupine_kernel_param_layout_cache_mutex());
    auto it = lupine_kernel_param_layout_cache().find(key);
    if (it != lupine_kernel_param_layout_cache().end()) {
      *layout = it->second;
      return CUDA_SUCCESS;
    }
  }

  lupine_kernel_param_layout fetched = {};
  CUresult result = lupine_fetch_kernel_param_layout(f, &fetched);
  if (result == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(lupine_kernel_param_layout_cache_mutex());
    lupine_kernel_param_layout_cache()[key] = fetched;
    *layout = fetched;
  }
  return result;
}

static lupine_route lupine_route_from_known_kernel_deviceptr_args(
    const std::vector<unsigned char> &packed,
    const lupine_kernel_param_layout &layout, lupine_route fallback) {
  int route_id = -2;
  for (uint32_t i = 0; i < layout.count; ++i) {
    if (layout.sizes[i] != sizeof(CUdeviceptr) ||
        layout.offsets[i] + sizeof(CUdeviceptr) > packed.size()) {
      continue;
    }
    CUdeviceptr ptr = 0;
    memcpy(&ptr, packed.data() + layout.offsets[i], sizeof(ptr));
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
  lupine_route launch_route = hStream != nullptr
                                  ? lupine_route_for_stream(hStream)
                                  : lupine_route_for_default();
  CUfunction route_function = f;
  CUresult resolve_status =
      lupine_resolve_library_kernel_for_route(f, launch_route, &route_function);
  if (resolve_status != CUDA_SUCCESS) {
    return resolve_status;
  }
  resolve_status = lupine_resolve_module_function_for_route(
      route_function, launch_route, &route_function);
  if (resolve_status != CUDA_SUCCESS) {
    return resolve_status;
  }
  f = route_function;
  resolve_status = lupine_resolve_private_function_for_route(f, launch_route,
                                                             &route_function);
  if (resolve_status != CUDA_SUCCESS) {
    return resolve_status;
  }
  f = lupine_translate_private_function(route_function);

  lupine_route route = lupine_route_for_function(f);
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cuLaunchKernel f=" << f << " stream=" << hStream
              << " launch_route=" << lupine_route_identity(launch_route)
              << " function_route=" << lupine_route_identity(route) << " grid=("
              << gridDimX << "," << gridDimY << "," << gridDimZ << ") block=("
              << blockDimX << "," << blockDimY << "," << blockDimZ << ")"
              << std::endl;
  }
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

  lupine_kernel_param_layout layout;
  CUresult status = lupine_get_kernel_param_layout_cached(f, &layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (layout.count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  size_t total_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    if (kernelParams == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size = std::max(total_size, layout.offsets[i] + layout.sizes[i]);
  }

  std::vector<unsigned char> packed(total_size);
  for (uint32_t i = 0; i < layout.count; ++i) {
    memcpy(packed.data() + layout.offsets[i], kernelParams[i], layout.sizes[i]);
  }

  lupine_route arg_route = lupine_route_from_known_kernel_deviceptr_args(
      packed, layout, launch_route);
  if (lupine_route_identity(arg_route) != lupine_route_identity(launch_route)) {
    launch_route = arg_route;
    route_function = requested_function;
    resolve_status = lupine_resolve_library_kernel_for_route(
        requested_function, launch_route, &route_function);
    if (resolve_status != CUDA_SUCCESS) {
      return resolve_status;
    }
    resolve_status = lupine_resolve_module_function_for_route(
        route_function, launch_route, &route_function);
    if (resolve_status != CUDA_SUCCESS) {
      return resolve_status;
    }
    resolve_status = lupine_resolve_private_function_for_route(
        route_function, launch_route, &route_function);
    if (resolve_status != CUDA_SUCCESS) {
      return resolve_status;
    }
    f = lupine_translate_private_function(route_function);
    route = lupine_route_for_function(f);

    status = lupine_get_kernel_param_layout_cached(f, &layout);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    if (layout.count > 64) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    total_size = 0;
    for (uint32_t i = 0; i < layout.count; ++i) {
      if (kernelParams == nullptr || kernelParams[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      total_size = std::max(total_size, layout.offsets[i] + layout.sizes[i]);
    }
    packed.assign(total_size, 0);
    for (uint32_t i = 0; i < layout.count; ++i) {
      memcpy(packed.data() + layout.offsets[i], kernelParams[i],
             layout.sizes[i]);
    }
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE cuLaunchKernel rerouted by args f=" << f
                << " route=" << lupine_route_identity(route) << std::endl;
    }
  }

  status = lupine_sync_mapped_host_to_device_for_launch(
      packed.data(), layout.offsets, layout.sizes, layout.count);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchKernel) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_write(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_write(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_write(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_write(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_write(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_write(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &total_size, sizeof(total_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (lupine_trace_enabled() && return_value != CUDA_SUCCESS) {
    std::cerr << "LUPINE cuLaunchKernel result="
              << static_cast<int>(return_value) << std::endl;
  }
  return return_value;
}

extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                     void **kernelParams, void **extra) {
  if (config == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (config->numAttrs != 0 && lupine_trace_enabled()) {
    std::cerr << "LUPINE cuLaunchKernelEx ignoring " << config->numAttrs
              << " launch attributes" << std::endl;
  }
  return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                        config->blockDimX, config->blockDimY, config->blockDimZ,
                        config->sharedMemBytes, config->hStream, kernelParams,
                        extra);
}

extern "C" CUresult lupine_cuLaunchCooperativeKernel_safe(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
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

  lupine_kernel_param_layout layout;
  CUresult status = lupine_get_kernel_param_layout_cached(f, &layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (layout.count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  size_t total_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    if (kernelParams == nullptr || kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size = std::max(total_size, layout.offsets[i] + layout.sizes[i]);
  }

  std::vector<unsigned char> packed(total_size);
  for (uint32_t i = 0; i < layout.count; ++i) {
    memcpy(packed.data() + layout.offsets[i], kernelParams[i], layout.sizes[i]);
  }

  status = lupine_sync_mapped_host_to_device_for_launch(
      packed.data(), layout.offsets, layout.sizes, layout.count);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchCooperativeKernel) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 ||
      rpc_write(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_write(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_write(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_write(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_write(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_write(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_write(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &total_size, sizeof(total_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult lupine_cuFuncGetAttribute_safe(int *pi,
                                                   CUfunction_attribute attrib,
                                                   CUfunction hfunc) {
  if (pi == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction translated = lupine_translate_private_function(hfunc);
  if (translated != hfunc) {
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE translated cuFuncGetAttribute attr=" << attrib
                << " client=" << hfunc << " server=" << translated << std::endl;
    }
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
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE translated "
                   "cuOccupancyMaxActiveBlocksPerMultiprocessor"
                << " client=" << func << " server=" << translated
                << " blockSize=" << blockSize
                << " dynamicSMemSize=" << dynamicSMemSize << std::endl;
    }
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
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE translated "
                   "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
                << " client=" << func << " server=" << translated
                << " blockSize=" << blockSize
                << " dynamicSMemSize=" << dynamicSMemSize << " flags=" << flags
                << std::endl;
    }
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, translated, blockSize, dynamicSMemSize, flags);
  }
  return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, func, blockSize, dynamicSMemSize, flags);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount, CUstream hStream) {
  conn_t *conn = lupine_rpc_conn_for_deviceptr(srcDevice);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoHAsync_v2) < 0 ||
      rpc_write(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
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
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && returned_dst_size != 0) {
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
  int op = set ? LUPINE_RPC_cuDeviceSetGraphMemAttribute
               : LUPINE_RPC_cuDeviceGetGraphMemAttribute;
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
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS && returned_dst_size != 0) {
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

static CUresult
lupine_pack_kernel_params(const CUDA_KERNEL_NODE_PARAMS *nodeParams,
                          lupine_kernel_param_layout *layout,
                          std::vector<unsigned char> *packed) {
  if (nodeParams == nullptr || layout == nullptr || packed == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (nodeParams->extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUfunction func = nodeParams->func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(nodeParams->kern);
  }
#endif
  CUresult status = lupine_get_kernel_param_layout_cached(func, layout);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (layout->count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (layout->count != 0 && nodeParams->kernelParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t total_size = 0;
  for (uint32_t i = 0; i < layout->count; ++i) {
    if (nodeParams->kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size = std::max(total_size, layout->offsets[i] + layout->sizes[i]);
  }
  packed->assign(total_size, 0);
  for (uint32_t i = 0; i < layout->count; ++i) {
    memcpy(packed->data() + layout->offsets[i], nodeParams->kernelParams[i],
           layout->sizes[i]);
  }
  return CUDA_SUCCESS;
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

  lupine_kernel_param_layout layout;
  std::vector<unsigned char> packed;
  status = lupine_pack_kernel_params(nodeParams, &layout, &packed);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  CUDA_KERNEL_NODE_PARAMS serial_params = *nodeParams;
  serial_params.func = lupine_translate_private_function(serial_params.func);
  serial_params.kernelParams = nullptr;
  serial_params.extra = nullptr;

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  size_t packed_size = packed.size();
  if (rpc_write_start_request(conn, RPC_cuGraphAddKernelNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(*phGraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
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

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddMemcpyNode) < 0 ||
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
  return return_value;
}

extern "C" CUresult lupine_cuGraphExecKernelNodeSetParams_v2_safe(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  if (nodeParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_kernel_param_layout layout;
  std::vector<unsigned char> packed;
  CUresult status = lupine_pack_kernel_params(nodeParams, &layout, &packed);
  if (status != CUDA_SUCCESS) {
    return status;
  }

  CUDA_KERNEL_NODE_PARAMS serial_params = *nodeParams;
  serial_params.func = lupine_translate_private_function(serial_params.func);
  serial_params.kernelParams = nullptr;
  serial_params.extra = nullptr;

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  size_t packed_size = packed.size();
  if (rpc_write_start_request(conn, RPC_cuGraphExecKernelNodeSetParams_v2) <
          0 ||
      rpc_write(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
      rpc_write(conn, packed.data(), packed.size()) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
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

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddMemsetNode) < 0 ||
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
  add_host_node(reinterpret_cast<void *>(nodeParams->fn), nodeParams->userData);

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuGraphAddHostNode) < 0 ||
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
  return return_value;
}

extern "C" CUresult cuGraphConditionalHandleCreate(
    CUgraphConditionalHandle *pHandle_out, CUgraph hGraph, CUcontext ctx,
    unsigned int defaultLaunchValue, unsigned int flags) {
  if (pHandle_out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuGraphConditionalHandleCreate) <
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

  uint32_t param_count = 0;
  size_t packed_size = 0;
  std::vector<unsigned char> packed;
  CUgraphNodeParams serial_params = *nodeParams;
  if (nodeParams->type == CU_GRAPH_NODE_TYPE_KERNEL) {
    lupine_kernel_param_layout layout;
    status = lupine_pack_kernel_params(
        reinterpret_cast<const CUDA_KERNEL_NODE_PARAMS *>(&nodeParams->kernel),
        &layout, &packed);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    param_count = layout.count;
    packed_size = packed.size();
    CUfunction translated =
        serial_params.kernel.func != nullptr
            ? lupine_translate_private_function(serial_params.kernel.func)
            : lupine_translate_private_function(
                  reinterpret_cast<CUfunction>(serial_params.kernel.kern));
    if (serial_params.kernel.func != nullptr) {
      serial_params.kernel.func = translated;
    } else {
      serial_params.kernel.kern = reinterpret_cast<CUkernel>(translated);
    }
    serial_params.kernel.kernelParams = nullptr;
    serial_params.kernel.extra = nullptr;
  } else if (nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    serial_params.conditional.phGraph_out = nullptr;
  } else {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  unsigned int child_count = nodeParams->type == CU_GRAPH_NODE_TYPE_CONDITIONAL
                                 ? nodeParams->conditional.size
                                 : 0;
  std::vector<CUgraph> child_graphs(child_count);
  if (rpc_write_start_request(conn, LUPINE_RPC_cuGraphAddNode_v2) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_queue_graph_dependencies(conn, dependencies, &numDependencies) !=
          CUDA_SUCCESS ||
      rpc_write(conn, &serial_params, sizeof(serial_params)) < 0 ||
      rpc_write(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_write(conn, &packed_size, sizeof(packed_size)) < 0 ||
      (packed_size != 0 && rpc_write(conn, packed.data(), packed_size) < 0) ||
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

extern "C" CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes,
                                    size_t *numNodes) {
  if (numNodes == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t requested = nodes == nullptr ? 0 : *numNodes;
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  size_t returned = 0;
  if (rpc_write_start_request(conn, RPC_cuGraphGetNodes) < 0 ||
      rpc_write(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_write(conn, &requested, sizeof(requested)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &returned, sizeof(returned)) < 0 ||
      (nodes != nullptr && returned != 0 &&
       rpc_read(conn, nodes, returned * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  *numNodes = returned;
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
  add_host_node(reinterpret_cast<void *>(fn), userData);

  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuLaunchHostFunc) < 0 ||
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
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(conn, LUPINE_RPC_cuStreamAddCallback) < 0 ||
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

  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  size_t dependency_count = 0;
  bool has_edge_data = false;
  conn_t *conn = rpc_client_get_connection(0);
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
  return return_value;
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
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
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
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
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
  conn_t *conn = rpc_client_get_connection(0);
  CUresult return_value;
  if (rpc_write_start_request(
          conn, LUPINE_RPC_cuStreamUpdateCaptureDependencies_v2) < 0 ||
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
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cudart get_module_from_cubin result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
  return result;
}

extern "C" CUresult lupine_cudart_get_primary_context(CUcontext *ctx,
                                                      CUdevice dev) {
  CUresult result = cuDevicePrimaryCtxRetain(ctx, dev);
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cudart get_primary_context dev=" << dev
              << " result=" << static_cast<int>(result)
              << " ctx=" << (ctx == nullptr ? nullptr : *ctx) << std::endl;
  }
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
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cudart get_module_from_cubin_ext1 result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
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
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE cudart get_module_from_cubin_ext2 result="
              << static_cast<int>(result)
              << " module=" << (module == nullptr ? nullptr : *module)
              << std::endl;
  }
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

static std::mutex &lupine_context_storage_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

static std::unordered_map<
    CUcontext, std::unordered_map<void *, lupine_context_storage_value>> &
lupine_context_storage() {
  static auto *storage = new std::unordered_map<
      CUcontext, std::unordered_map<void *, lupine_context_storage_value>>();
  return *storage;
}

static CUresult lupine_normalize_context(CUcontext *ctx) {
  if (ctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (*ctx != nullptr) {
    return CUDA_SUCCESS;
  }
  return cuCtxGetCurrent(ctx);
}

extern "C" CUresult
lupine_context_local_storage_put(CUcontext ctx, void *key, void *value,
                                 lupine_context_storage_dtor_t dtor) {
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(lupine_context_storage_mutex());
  lupine_context_storage()[ctx][key] = {value, dtor};
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE context storage put ctx=" << ctx << " key=" << key
              << " value=" << value << std::endl;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult lupine_context_local_storage_delete(CUcontext ctx,
                                                        void *key) {
  CUresult result = lupine_normalize_context(&ctx);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  std::lock_guard<std::mutex> lock(lupine_context_storage_mutex());
  auto ctx_it = lupine_context_storage().find(ctx);
  if (ctx_it == lupine_context_storage().end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  auto value_it = ctx_it->second.find(key);
  if (value_it == ctx_it->second.end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  // The private context-local-storage destructor ABI is not stable enough to
  // invoke here; some CUDA libraries call explicit delete during their own
  // teardown and free the value themselves.
  ctx_it->second.erase(value_it);
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE context storage delete ctx=" << ctx << " key=" << key
              << std::endl;
  }
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

  std::lock_guard<std::mutex> lock(lupine_context_storage_mutex());
  auto ctx_it = lupine_context_storage().find(ctx);
  if (ctx_it == lupine_context_storage().end()) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  auto value_it = ctx_it->second.find(key);
  if (value_it == ctx_it->second.end()) {
    if (lupine_trace_enabled()) {
      std::cerr << "LUPINE context storage get missing key ctx=" << ctx
                << " key=" << key << std::endl;
    }
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *value = value_it->second.value;
  if (lupine_trace_enabled()) {
    std::cerr << "LUPINE context storage get ctx=" << ctx << " key=" << key
              << " value=" << *value << std::endl;
  }
  return CUDA_SUCCESS;
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

extern "C" CUresult lupine_context_check(CUcontext, unsigned int *result1,
                                         const void **result2) {
  if (result1 != nullptr) {
    *result1 = 0;
  }
  if (result2 != nullptr) {
    *result2 = nullptr;
  }
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
  static const void *table[4] = {
      reinterpret_cast<const void *>(sizeof(table)),
      nullptr,
      reinterpret_cast<const void *>(&lupine_context_check),
      reinterpret_cast<const void *>(&lupine_context_check_fn3),
  };
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

  if (lupine_trace_enabled() && pExportTableId != nullptr) {
    std::cerr << "LUPINE cuGetExportTable requested UUID:";
    for (unsigned char byte : pExportTableId->bytes) {
      fprintf(stderr, " %02x", byte);
    }
    std::cerr << std::endl;
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

#if defined(__x86_64__)
  static std::unordered_map<std::string, void *> stubs;
  auto existing = stubs.find(symbol);
  if (existing != stubs.end()) {
    return existing->second;
  }

  constexpr size_t stub_size = 22;
  unsigned char *code = static_cast<unsigned char *>(
      mmap(nullptr, stub_size, PROT_READ | PROT_WRITE | PROT_EXEC,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (code == MAP_FAILED) {
    return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
  }

  char *stable_symbol = strdup(symbol);
  void *handler = reinterpret_cast<void *>(&lupine_missing_driver_api_called);
  code[0] = 0x48;
  code[1] = 0xbf;
  memcpy(code + 2, &stable_symbol, sizeof(stable_symbol));
  code[10] = 0x48;
  code[11] = 0xb8;
  memcpy(code + 12, &handler, sizeof(handler));
  code[20] = 0xff;
  code[21] = 0xe0;
  stubs[symbol] = code;
  return code;
#else
  return reinterpret_cast<void *>(&lupine_unsupported_driver_api);
#endif
}

#define LUPINE_DEFINE_UNSUPPORTED_STUB(name)                                   \
  extern "C" CUresult lupine_unsupported_##name() {                            \
    std::cerr << "LUPINE unsupported Driver API called: " #name << std::endl;  \
    return CUDA_ERROR_NOT_SUPPORTED;                                           \
  }

LUPINE_DEFINE_UNSUPPORTED_STUB(cuCtxCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuModuleLoadData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuModuleLoadFatBinary)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryLoadData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryGetKernelCount)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLibraryEnumerateKernels)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuKernelGetName)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuKernelGetParamInfo)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkCreate)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkAddData)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuLinkAddFile)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemGetInfo)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemGetAddressRange)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostGetDevicePointer)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostRegister)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemHostUnregister)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuPointerGetAttribute)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpyDtoH)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpyDtoHAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DUnaligned)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy2DAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3D)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeer)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemcpy3DPeerAsync)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemAdvise)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuMemPrefetchAsync)
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
LUPINE_DEFINE_UNSUPPORTED_STUB(cuKernelGetLibrary)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetHostAtomicCapabilities)
LUPINE_DEFINE_UNSUPPORTED_STUB(cuDeviceGetP2PAtomicCapabilities)

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
      LUPINE_STUB_ENTRY(cuKernelGetParamInfo),
      LUPINE_STUB_ENTRY(cuLinkCreate),
      LUPINE_STUB_ENTRY(cuLinkAddData),
      LUPINE_STUB_ENTRY(cuLinkAddFile),
      LUPINE_STUB_ENTRY(cuMemGetInfo),
      LUPINE_STUB_ENTRY(cuMemGetAddressRange),
      LUPINE_STUB_ENTRY(cuMemHostGetDevicePointer),
      LUPINE_STUB_ENTRY(cuMemHostRegister),
      LUPINE_STUB_ENTRY(cuMemHostUnregister),
      LUPINE_STUB_ENTRY(cuPointerGetAttribute),
      LUPINE_STUB_ENTRY(cuMemcpyDtoH),
      LUPINE_STUB_ENTRY(cuMemcpyDtoHAsync),
      LUPINE_STUB_ENTRY(cuMemcpy2DUnaligned),
      LUPINE_STUB_ENTRY(cuMemcpy2DAsync),
      LUPINE_STUB_ENTRY(cuMemcpy3D),
      LUPINE_STUB_ENTRY(cuMemcpy3DAsync),
      LUPINE_STUB_ENTRY(cuMemcpy3DPeer),
      LUPINE_STUB_ENTRY(cuMemcpy3DPeerAsync),
      LUPINE_STUB_ENTRY(cuMemAdvise),
      LUPINE_STUB_ENTRY(cuMemPrefetchAsync),
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
      LUPINE_STUB_ENTRY(cuKernelGetLibrary),
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

  if (!conn->closed) {
    conn->closed = 1;
    shutdown(conn->connfd, SHUT_RDWR);
    close(conn->connfd);
  }

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
    if (conns[i].read_thread != 0) {
      pthread_join(conns[i].read_thread, nullptr);
      conns[i].read_thread = 0;
    }
    if (conns[i].rpc_thread != 0) {
      pthread_join(conns[i].rpc_thread, nullptr);
      conns[i].rpc_thread = 0;
    }
  }

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    nconns = 0;
    lupine_rpc_shutting_down = false;
    pthread_mutex_unlock(&conn_mutex);
  }
}

__attribute__((destructor)) static void lupine_rpc_destructor() {
  lupine_rpc_shutdown();
}

typedef void (*func_t)(void *);

void add_host_node(void *fn, void *udata) { host_funcs[fn] = udata; }

void invoke_host_func(void *fn) {
  for (const auto &pair : host_funcs) {
    if (pair.first == fn) {
      func_t func = reinterpret_cast<func_t>(pair.first);
      std::cout << "Invoking function at: " << pair.first << std::endl;
      func(pair.second);
      return;
    }
  }
}

void *rpc_client_dispatch_thread(void *arg) {
  conn_t *conn = (conn_t *)arg;
  int op;

  while (true) {
    op = rpc_dispatch(conn, 1);

    if (op == 1) {
      std::cout << "Transferring memory..." << std::endl;

      int found = 0;

      rpc_read(conn, &found, sizeof(int));

      for (int i = 0; i < found; ++i) {
        void *host_data = nullptr;
        void *dst = nullptr;
        size_t count = 0;

        if (rpc_read(conn, &dst, sizeof(void *)) < 0 ||
            rpc_read(conn, &count, sizeof(size_t)) < 0) {
          std::cerr << "Failed to read transfer parameters." << std::endl;
          break;
        }

        host_data = malloc(count);
        if (!host_data) {
          std::cerr << "Memory allocation failed." << std::endl;
          break;
        }

        // Read the actual data from the server (sent from `src` in device
        // memory)
        if (rpc_read(conn, host_data, count) < 0) {
          std::cerr << "Failed to read device data from server." << std::endl;
          free(host_data);
          break;
        }

        // Copy received data to the destination (dst) on the host
        lupine_prepare_host_range_write(dst, count);
        memcpy(dst, host_data, count);
        lupine_mark_host_range_clean(dst, count);
        free(host_data);
      }

      void *temp_mem;
      if (rpc_read(conn, &temp_mem, sizeof(void *)) <= 0) {
        std::cerr << "rpc_read failed for mem. Closing connection."
                  << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      void *mem = temp_mem;

      if (mem == nullptr) {
        std::cerr << "Invalid function pointer!" << std::endl;
        continue;
      }

      invoke_host_func(mem);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
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
        std::cerr << "Failed to read stream callback request." << std::endl;
        break;
      }

      int request_id = rpc_read_end(conn);
      if (request_id < 0) {
        break;
      }

      callback(stream, status, user_data);

      void *res = nullptr;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &res, sizeof(void *)) < 0 ||
          rpc_write_end(conn) < 0) {
        std::cerr << "rpc_write failed. Closing connection." << std::endl;
        break;
      }
    } else if (op < 0 || conn->closed) {
      break;
    }
  }

  if (!conn->closed) {
    std::cerr << "Exiting dispatch thread due to an error." << std::endl;
  }
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

  char *server_ip = strdup(server_ips);
  char *token;
  while ((token = strsep(&server_ip, ","))) {
    char *host;
    char *port;

    // Split the string into IP address and port
    char *colon = strchr(token, ':');
    if (colon == NULL) {
      host = token;
      port = const_cast<char *>(DEFAULT_PORT);
    } else {
      *colon = '\0';
      host = token;
      port = colon + 1;
    }

    addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      std::cout << "getaddrinfo of " << host << " port " << port << " failed"
                << std::endl;
      continue;
    }

    int flag = 1;
    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1) {
      printf("socket creation failed...\n");
      continue;
    }

    int opts = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag,
                          sizeof(int));
    if (connect(sockfd, res->ai_addr, res->ai_addrlen) < 0) {
      std::cerr << "Connecting to " << host << " port " << port
                << " failed: " << strerror(errno) << std::endl;
      close(sockfd);
      continue;
    }

    conns[nconns] = {};
    conns[nconns].connfd = sockfd;
    conns[nconns].request_id = 0;
    conns[nconns].closed = 0;
    conns[nconns].local_request_parity = conns[nconns].request_id & 1;
    if (pthread_mutex_init(&conns[nconns].read_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].write_mutex, NULL) < 0 ||
        pthread_mutex_init(&conns[nconns].call_mutex, NULL) < 0 ||
        pthread_cond_init(&conns[nconns].read_cond, NULL) < 0 ||
        rpc_http2_client_init(&conns[nconns]) < 0) {
      close(sockfd);
      return -1;
    }

    pthread_create(&conns[nconns].read_thread, NULL, rpc_client_dispatch_thread,
                   (void *)&conns[nconns]);

    nconns++;
  }

  if (pthread_mutex_unlock(&conn_mutex) < 0)
    return -1;
  if (nconns == 0)
    return -1;
  return 0;
}

conn_t *rpc_client_get_connection(unsigned int index) {
  if (rpc_open() < 0)
    return nullptr;
  return &conns[index];
}

int rpc_size() { return nconns; }

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *symbol, void **pfn,
                                     int cudaVersion, cuuint64_t flags);

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  if (lupine_trace_enabled()) {
    std::cout << "cuGetProcAddress getting symbol: " << symbol << std::endl;
  }

  if (symbol != nullptr && strcmp(symbol, "cuFuncGetAttribute") == 0) {
    *pfn = reinterpret_cast<void *>(&lupine_cuFuncGetAttribute_safe);
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    return CUDA_SUCCESS;
  }
  if (symbol != nullptr && strcmp(symbol, "cuPointerGetAttributes") == 0) {
    *pfn = reinterpret_cast<void *>(&lupine_cuPointerGetAttributes_safe);
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

  auto it = get_function_pointer(symbol);
  if (it != nullptr) {
    *pfn = it;
    if (symbolStatus != nullptr) {
      *symbolStatus = CU_GET_PROC_ADDRESS_SUCCESS;
    }
    if (lupine_trace_enabled()) {
      std::cout << "cuGetProcAddress: Mapped symbol '" << symbol
                << "' to function: " << *pfn << std::endl;
    }
    return CUDA_SUCCESS;
  }

  static const std::unordered_map<std::string, void *> manual_function_map = {
#if CUDA_VERSION >= 12050
      {"cuCtxCreate", (void *)cuCtxCreate_v4},
      {"cuCtxCreate_v4", (void *)cuCtxCreate_v4},
#else
      {"cuCtxCreate", (void *)cuCtxCreate_v2},
#endif
      {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
      {"cuOccupancyMaxPotentialBlockSize",
       (void *)cuOccupancyMaxPotentialBlockSize},
      {"cuOccupancyMaxPotentialBlockSizeWithFlags",
       (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuStreamDestroy", (void *)cuStreamDestroy},
      {"cuEventDestroy", (void *)cuEventDestroy},
      {"cuEventElapsedTime", (void *)cuEventElapsedTime},
      {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
      {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
      {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
      {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
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
      {"cuArrayCreate", (void *)cuArrayCreate_v2},
      {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
      {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
      {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
      {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
      {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
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
      {"cuPointerGetAttributes", (void *)lupine_cuPointerGetAttributes_safe},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoadData", (void *)cuModuleLoadData},
      {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
      {"cuLibraryLoadData", (void *)cuLibraryLoadData},
      {"cuLaunchKernel", (void *)cuLaunchKernel},
      {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
      {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
      {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
      {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
      {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
      {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
      {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
      {"cuCtxSynchronize", (void *)cuCtxSynchronize},
      {"cuStreamSynchronize", (void *)cuStreamSynchronize},
      {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize_ptsz},
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
      {"cuGraphExecKernelNodeSetParams",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams_v2",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphGetNodes", (void *)cuGraphGetNodes},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
      {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
      {"cuStreamAddCallback", (void *)cuStreamAddCallback},
      {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_v2",
       (void *)cuStreamUpdateCaptureDependencies_v2},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
  };
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

  if (lupine_trace_enabled()) {
    std::cout << "cuGetProcAddress: Symbol '" << symbol
              << "' not found in cudaFunctionMap." << std::endl;
  }

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
    if (lupine_trace_enabled()) {
      std::cerr << "Error: Could not resolve symbol '" << symbol
                << "' using dlsym." << std::endl;
    }
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
  return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

void *dlsym(void *handle, const char *name) __THROW {
  if (lupine_trace_enabled()) {
    std::cout << "dlsym: " << name << std::endl;
  }

  if (!lupine_symbol_looks_like_driver_api(name)) {
    return lupine_real_dlsym(handle, name);
  }

  if (name != nullptr && strcmp(name, "cuFuncGetAttribute") == 0) {
    return reinterpret_cast<void *>(&lupine_cuFuncGetAttribute_safe);
  }
  if (name != nullptr && strcmp(name, "cuPointerGetAttributes") == 0) {
    return reinterpret_cast<void *>(&lupine_cuPointerGetAttributes_safe);
  }
  if (name != nullptr &&
      strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessor") == 0) {
    return reinterpret_cast<void *>(
        &lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_safe);
  }
  if (name != nullptr &&
      strcmp(name, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") ==
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

  static const std::unordered_map<std::string, void *> manual_function_map = {
#if CUDA_VERSION >= 12050
      {"cuCtxCreate", (void *)cuCtxCreate_v4},
      {"cuCtxCreate_v4", (void *)cuCtxCreate_v4},
#else
      {"cuCtxCreate", (void *)cuCtxCreate_v2},
#endif
      {"cuCtxCreate_v2", (void *)cuCtxCreate_v2},
      {"cuOccupancyMaxPotentialBlockSize",
       (void *)cuOccupancyMaxPotentialBlockSize},
      {"cuOccupancyMaxPotentialBlockSizeWithFlags",
       (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
      {"cuProfilerInitialize", (void *)cuProfilerInitialize},
      {"cuProfilerStart", (void *)cuProfilerStart},
      {"cuProfilerStop", (void *)cuProfilerStop},
      {"cuStreamDestroy", (void *)cuStreamDestroy},
      {"cuEventDestroy", (void *)cuEventDestroy},
      {"cuEventElapsedTime", (void *)cuEventElapsedTime},
      {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
      {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
      {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
      {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
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
      {"cuArrayCreate", (void *)cuArrayCreate_v2},
      {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
      {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
      {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
      {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
      {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
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
      {"cuPointerGetAttributes", (void *)lupine_cuPointerGetAttributes_safe},
      {"cuGetExportTable", (void *)cuGetExportTable},
      {"cuModuleLoadData", (void *)cuModuleLoadData},
      {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
      {"cuLibraryLoadData", (void *)cuLibraryLoadData},
      {"cuLaunchKernel", (void *)cuLaunchKernel},
      {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
      {"cuMemcpyAsync", (void *)cuMemcpyAsync},
      {"cuMemcpyAsync_ptsz", (void *)cuMemcpyAsync},
      {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
      {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
      {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
      {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
      {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
      {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
      {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
      {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo},
      {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
      {"cuStreamGetCaptureInfo_v3", (void *)cuStreamGetCaptureInfo_v3},
      {"cuCtxSynchronize", (void *)cuCtxSynchronize},
      {"cuStreamSynchronize", (void *)cuStreamSynchronize},
      {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize_ptsz},
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
      {"cuGraphExecKernelNodeSetParams",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphExecKernelNodeSetParams_v2",
       (void *)cuGraphExecKernelNodeSetParams_v2},
      {"cuGraphGetNodes", (void *)cuGraphGetNodes},
      {"cuGraphInstantiate", (void *)cuGraphInstantiate},
      {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
      {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
      {"cuStreamAddCallback", (void *)cuStreamAddCallback},
      {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
      {"cuStreamBeginCaptureToGraph", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamBeginCaptureToGraph_ptsz", (void *)cuStreamBeginCaptureToGraph},
      {"cuStreamUpdateCaptureDependencies",
       (void *)cuStreamUpdateCaptureDependencies},
      {"cuStreamUpdateCaptureDependencies_v2",
       (void *)cuStreamUpdateCaptureDependencies_v2},
      {"cuStreamUpdateCaptureDependencies_ptsz",
       (void *)cuStreamUpdateCaptureDependencies_ptsz},
  };
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
