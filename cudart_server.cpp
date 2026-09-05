// cuda.h supplies CUDA_VERSION, which the generated guards and the registry
// both test; the runtime header alone only defines CUDART_VERSION.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "lupine_fatbin.h"
#include "rpc.h"

namespace {

cudaError_t function_not_found() { return cudaErrorNotSupported; }

// The runtime this server was compiled against; the handlers marshal its
// struct layouts, so an older or newer runtime on the machine is not a match.
void *cudart_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUDART_VERSION / 1000);
#ifdef _WIN32
    const std::string name = "cudart64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcudart.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cudart_symbol(const char *name) {
  void *lib = cudart_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// Every call resolves its entry point by name at the moment it is made.
#define LUPINE_CUDART_CALL(name, not_found, ...)                               \
  ([&] {                                                                       \
    auto fn = cudart_symbol<decltype(&name)>(#name);                           \
    return fn == nullptr ? (not_found) : fn(__VA_ARGS__);                      \
  }())

int write_result(conn_t *conn, int request_id, cudaError_t result) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// cudaGetErrorName and cudaGetErrorString return static strings, so the
// response carries the bytes length-prefixed instead of a result code.
int handle_error_text(conn_t *conn, const char *symbol) {
  cudaError_t error;
  if (rpc_read(conn, &error, sizeof(error)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cudaError_t);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  const char *text = fn == nullptr ? nullptr : fn(error);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cudaGetErrorName(conn_t *conn) {
  return handle_error_text(conn, "cudaGetErrorName");
}

int handle_cudaGetErrorString(conn_t *conn) {
  return handle_error_text(conn, "cudaGetErrorString");
}

int handle_cudaCreateChannelDesc(conn_t *conn) {
  int x, y, z, w;
  enum cudaChannelFormatKind f;
  if (rpc_read(conn, &x, sizeof(x)) < 0 || rpc_read(conn, &y, sizeof(y)) < 0 ||
      rpc_read(conn, &z, sizeof(z)) < 0 || rpc_read(conn, &w, sizeof(w)) < 0 ||
      rpc_read(conn, &f, sizeof(f)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  struct cudaChannelFormatDesc desc = {};
  auto fn =
      cudart_symbol<decltype(&cudaCreateChannelDesc)>("cudaCreateChannelDesc");
  if (fn != nullptr) {
    desc = fn(x, y, z, w, f);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &desc, sizeof(desc)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#if CUDART_VERSION >= 12000
int handle_cudaFuncGetName(conn_t *conn) {
  const void *func = nullptr;
  if (rpc_read(conn, &func, sizeof(func)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  const char *name = nullptr;
  cudaError_t result =
      LUPINE_CUDART_CALL(cudaFuncGetName, function_not_found(), &name, func);
  uint32_t length = result == cudaSuccess && name != nullptr
                        ? static_cast<uint32_t>(strlen(name))
                        : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, name, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
#endif

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

namespace {

// The runtime keeps pointers into what nvcc's startup code hands it, for as
// long as the fatbin stays registered: the wrapper, the image, and the
// symbol names. So does this.
struct fatbin_registration {
  lupine_fatbin_wrapper wrapper = {};
  std::vector<unsigned char> image;
  std::deque<std::string> strings;
  std::deque<void *> managed_slots;
  std::deque<uint3> indices;
  std::deque<dim3> dimensions;
  std::deque<int> warp_sizes;
};

std::mutex &registry_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<void **, fatbin_registration *> &registrations() {
  static auto *map = new std::unordered_map<void **, fatbin_registration *>();
  return *map;
}

// registry_mutex() must be held.
fatbin_registration *registration_for(void **handle) {
  auto entry = registrations().find(handle);
  return entry == registrations().end() ? nullptr : entry->second;
}

// Keeps a string for the registration's lifetime and hands out its address.
char *retain(fatbin_registration *registration, const char *text) {
  if (registration == nullptr || text == nullptr) {
    return nullptr;
  }
  registration->strings.emplace_back(text);
  return &registration->strings.back()[0];
}

typedef void **(*register_fat_binary_fn)(void *);
typedef void (*register_fat_binary_end_fn)(void **);
typedef void (*unregister_fat_binary_fn)(void **);
typedef void (*register_function_fn)(void **, const char *, char *,
                                     const char *, int, uint3 *, uint3 *,
                                     dim3 *, dim3 *, int *);
typedef void (*register_var_fn)(void **, char *, char *, const char *, int,
                                size_t, int, int);
typedef void (*register_managed_var_fn)(void **, void **, char *, const char *,
                                        int, size_t, int, int);
typedef void (*register_texture_fn)(void **, const void *, const void **,
                                    const char *, int, int, int);
typedef void (*register_surface_fn)(void **, const void *, const void **,
                                    const char *, int, int);
typedef void (*register_host_var_fn)(void **, const char *, char *, size_t);

template <typename T> T *retain(std::deque<T> &values, const T *value) {
  if (value == nullptr) {
    return nullptr;
  }
  values.push_back(*value);
  return &values.back();
}

} // namespace

int handle___cudaRegisterFatBinary(conn_t *conn) {
  auto *registration = new fatbin_registration();
  uint64_t length = 0;
  if (rpc_read(conn, &registration->wrapper.version,
               sizeof(registration->wrapper.version)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0) {
    delete registration;
    return -1;
  }
  registration->image.resize(static_cast<size_t>(length));
  if (length != 0 && rpc_read(conn, registration->image.data(),
                              registration->image.size()) < 0) {
    delete registration;
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    delete registration;
    return -1;
  }
  registration->wrapper.magic = LUPINE_FATBINC_MAGIC;
  registration->wrapper.data = registration->image.data();
  auto fn = cudart_symbol<register_fat_binary_fn>("__cudaRegisterFatBinary");
  void **handle = fn == nullptr ? nullptr : fn(&registration->wrapper);
  if (handle == nullptr) {
    delete registration;
  } else {
    std::lock_guard<std::mutex> lock(registry_mutex());
    registrations()[handle] = registration;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

namespace {

// The generated handlers own transport. These adapters keep storage that the
// runtime retains after a registration RPC returns.
cudaError_t unregister_fat_binary(void **handle) {
  auto fn =
      cudart_symbol<unregister_fat_binary_fn>("__cudaUnregisterFatBinary");
  if (fn == nullptr) {
    return function_not_found();
  }
  fn(handle);
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = registrations().find(handle);
  if (entry != registrations().end()) {
    delete entry->second;
    registrations().erase(entry);
  }
  return cudaSuccess;
}

cudaError_t register_function(void **handle, const char *hostFun,
                              char *deviceFun, const char *deviceName,
                              int thread_limit, uint3 *tid, uint3 *bid,
                              dim3 *bDim, dim3 *gDim, int *wSize) {
  auto fn = cudart_symbol<register_function_fn>("__cudaRegisterFunction");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  fn(handle, hostFun, retain(registration, deviceFun),
     retain(registration, deviceName), thread_limit,
     retain(registration->indices, tid), retain(registration->indices, bid),
     retain(registration->dimensions, bDim),
     retain(registration->dimensions, gDim),
     retain(registration->warp_sizes, wSize));
  return cudaSuccess;
}

cudaError_t register_var(void **handle, char *hostVar, char *deviceAddress,
                         const char *deviceName, int ext, size_t size,
                         int constant, int global) {
  auto fn = cudart_symbol<register_var_fn>("__cudaRegisterVar");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  fn(handle, hostVar, retain(registration, deviceAddress),
     retain(registration, deviceName), ext, size, constant, global);
  return cudaSuccess;
}

// The runtime writes to this slot, so it must remain server-owned. This does
// not make managed globals directly host-addressable on the client.
cudaError_t register_managed_var(void **handle, void **hostVarPtrAddress,
                                 char *deviceAddress, const char *deviceName,
                                 int ext, size_t size, int constant,
                                 int global) {
  (void)hostVarPtrAddress;
  auto fn = cudart_symbol<register_managed_var_fn>("__cudaRegisterManagedVar");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  registration->managed_slots.push_back(nullptr);
  fn(handle, &registration->managed_slots.back(),
     retain(registration, deviceAddress), retain(registration, deviceName), ext,
     size, constant, global);
  return cudaSuccess;
}

cudaError_t register_texture(void **handle, const void *hostVar,
                             const void **deviceAddress, const char *deviceName,
                             int dim, int norm, int ext) {
  auto fn = cudart_symbol<register_texture_fn>("__cudaRegisterTexture");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  fn(handle, hostVar,
     reinterpret_cast<const void **>(
         retain(registration, reinterpret_cast<const char *>(deviceAddress))),
     retain(registration, deviceName), dim, norm, ext);
  return cudaSuccess;
}

cudaError_t register_surface(void **handle, const void *hostVar,
                             const void **deviceAddress, const char *deviceName,
                             int dim, int ext) {
  auto fn = cudart_symbol<register_surface_fn>("__cudaRegisterSurface");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  fn(handle, hostVar,
     reinterpret_cast<const void **>(
         retain(registration, reinterpret_cast<const char *>(deviceAddress))),
     retain(registration, deviceName), dim, ext);
  return cudaSuccess;
}

cudaError_t register_host_var(void **handle, const char *deviceName,
                              char *hostVar, size_t size) {
  auto fn = cudart_symbol<register_host_var_fn>("__cudaRegisterHostVar");
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto *registration = registration_for(handle);
  if (fn == nullptr) {
    return function_not_found();
  }
  if (registration == nullptr) {
    return cudaErrorInvalidValue;
  }
  fn(handle, retain(registration, deviceName), hostVar, size);
  return cudaSuccess;
}

} // namespace

// ---------------------------------------------------------------------------
// Kernel launches
// ---------------------------------------------------------------------------

namespace {

// The argument sizes of an entry point, in order; the client packs the
// values by them. Only a runtime that can report the layout can launch.
int handle_param_layout(conn_t *conn) {
  const void *func = nullptr;
  if (rpc_read(conn, &func, sizeof(func)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  std::vector<size_t> sizes;
  cudaError_t result = cudaSuccess;
  using fn_t = cudaError_t (*)(const void *, size_t, size_t *, size_t *);
  fn_t fn = cudart_symbol<fn_t>("cudaFuncGetParamInfo");
  if (fn == nullptr) {
    result = cudaErrorNotSupported;
  }
  for (size_t index = 0; fn != nullptr; ++index) {
    size_t offset = 0;
    size_t size = 0;
    cudaError_t status = fn(func, index, &offset, &size);
    if (status == cudaErrorInvalidValue) {
      break;
    }
    if (status != cudaSuccess) {
      result = status;
      break;
    }
    sizes.push_back(size);
  }
  const uint32_t count = static_cast<uint32_t>(sizes.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (count != 0 &&
       rpc_write(conn, sizes.data(), count * sizeof(size_t)) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

struct packed_params {
  std::vector<size_t> sizes;
  std::vector<unsigned char> storage;
  std::vector<void *> pointers;
};

int read_params(conn_t *conn, packed_params *params) {
  uint32_t count = 0;
  if (rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  params->sizes.assign(count, 0);
  if (count != 0 &&
      rpc_read(conn, params->sizes.data(), count * sizeof(size_t)) < 0) {
    return -1;
  }
  size_t total = 0;
  for (size_t size : params->sizes) {
    total += size;
  }
  params->storage.resize(total);
  params->pointers.assign(count, nullptr);
  unsigned char *cursor = params->storage.data();
  for (uint32_t i = 0; i < count; ++i) {
    if (params->sizes[i] != 0 && rpc_read(conn, cursor, params->sizes[i]) < 0) {
      return -1;
    }
    params->pointers[i] = cursor;
    cursor += params->sizes[i];
  }
  return 0;
}

struct launch_config {
  cudaLaunchConfig_t config = {};
  std::vector<cudaLaunchAttribute> attributes;
};

int read_launch_config(conn_t *conn, launch_config *launch) {
  uint32_t attribute_count = 0;
  if (rpc_read(conn, &launch->config.gridDim, sizeof(launch->config.gridDim)) <
          0 ||
      rpc_read(conn, &launch->config.blockDim,
               sizeof(launch->config.blockDim)) < 0 ||
      rpc_read(conn, &launch->config.dynamicSmemBytes,
               sizeof(launch->config.dynamicSmemBytes)) < 0 ||
      rpc_read(conn, &launch->config.stream, sizeof(launch->config.stream)) <
          0 ||
      rpc_read(conn, &attribute_count, sizeof(attribute_count)) < 0) {
    return -1;
  }
  launch->attributes.resize(attribute_count);
  if (attribute_count != 0 &&
      rpc_read(conn, launch->attributes.data(),
               attribute_count * sizeof(cudaLaunchAttribute)) < 0) {
    return -1;
  }
  launch->config.attrs = launch->attributes.data();
  launch->config.numAttrs = attribute_count;
  return 0;
}

// cudaLaunchKernel, cudaLaunchCooperativeKernel and __cudaLaunchKernel take
// the same arguments once the entry point is a pointer-sized value.
template <typename Entry> int handle_launch(conn_t *conn, const char *symbol) {
  uint64_t async_sequence = 0;
  Entry entry;
  dim3 gridDim, blockDim;
  size_t sharedMem = 0;
  cudaStream_t stream = nullptr;
  packed_params params;
  if (rpc_read(conn, &async_sequence, sizeof(async_sequence)) < 0 ||
      rpc_read(conn, &entry, sizeof(entry)) < 0 ||
      rpc_read(conn, &gridDim, sizeof(gridDim)) < 0 ||
      rpc_read(conn, &blockDim, sizeof(blockDim)) < 0 ||
      rpc_read(conn, &sharedMem, sizeof(sharedMem)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      read_params(conn, &params) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t =
      cudaError_t (*)(Entry, dim3, dim3, void **, size_t, cudaStream_t);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  if (rpc_async_sequence_begin(conn, async_sequence) < 0) {
    return -1;
  }
  if (fn != nullptr) {
    fn(entry, gridDim, blockDim, params.pointers.data(), sharedMem, stream);
  }
  rpc_async_sequence_end(conn);
  return 0;
}

} // namespace

int handle_lupineCudartFuncParamLayout(conn_t *conn) {
  return handle_param_layout(conn);
}

int handle_cudaLaunchKernel(conn_t *conn) {
  return handle_launch<const void *>(conn, "cudaLaunchKernel");
}

int handle_cudaLaunchCooperativeKernel(conn_t *conn) {
  return handle_launch<const void *>(conn, "cudaLaunchCooperativeKernel");
}

#if CUDART_VERSION >= 13000
int handle___cudaLaunchKernel(conn_t *conn) {
  return handle_launch<cudaKernel_t>(conn, "__cudaLaunchKernel");
}
#endif

int handle_cudaLaunchKernelExC(conn_t *conn) {
  uint64_t async_sequence = 0;
  launch_config launch;
  const void *func = nullptr;
  packed_params params;
  if (rpc_read(conn, &async_sequence, sizeof(async_sequence)) < 0 ||
      read_launch_config(conn, &launch) < 0 ||
      rpc_read(conn, &func, sizeof(func)) < 0 ||
      read_params(conn, &params) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (rpc_async_sequence_begin(conn, async_sequence) < 0) {
    return -1;
  }
  LUPINE_CUDART_CALL(cudaLaunchKernelExC, function_not_found(), &launch.config,
                     func, params.pointers.data());
  rpc_async_sequence_end(conn);
  return 0;
}

namespace {

int handle_occupancy_for_config(conn_t *conn, const char *symbol) {
  const void *func = nullptr;
  launch_config launch;
  if (rpc_read(conn, &func, sizeof(func)) < 0 ||
      read_launch_config(conn, &launch) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  int value = 0;
  using fn_t = cudaError_t (*)(int *, const void *, const cudaLaunchConfig_t *);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  cudaError_t result =
      fn == nullptr ? function_not_found() : fn(&value, func, &launch.config);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cudaOccupancyMaxPotentialClusterSize(conn_t *conn) {
  return handle_occupancy_for_config(conn,
                                     "cudaOccupancyMaxPotentialClusterSize");
}

int handle_cudaOccupancyMaxActiveClusters(conn_t *conn) {
  return handle_occupancy_for_config(conn, "cudaOccupancyMaxActiveClusters");
}


#include "codegen/gen_cudart_server.inc"
