// cuda.h supplies CUDA_VERSION, which the generated guards and the registry
// both test; the runtime header alone only defines CUDART_VERSION.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "cuda_compat.h"
#include "cuda_server_memcpy.h"
#include "lupine_fatbin.h"
#include "rpc.h"

namespace {

cudaError_t function_not_found() { return cudaErrorNotSupported; }

// The runtime this server was compiled against; the handlers marshal its
// struct layouts, so an older or newer runtime on the machine is not a match.
void *cudart_library() {
  static void *library = []() -> void * {
    const int major = CUDART_VERSION / 1000;
#ifdef _WIN32
    const std::string suffix = major == 11 ? "110" : std::to_string(major);
    const std::string name = "cudart64_" + suffix + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string suffix = major == 11 ? "11.0" : std::to_string(major);
    const std::string name = "libcudart.so." + suffix;
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

namespace {

// Use the same reserved-address backing as the driver, but let the native
// runtime own its registration and lifetime.
const lupine_host_registration_ops runtime_host_registration = {
    [](void *pointer, size_t bytes, unsigned int flags) {
      return static_cast<CUresult>(LUPINE_CUDART_CALL(
          cudaHostRegister, function_not_found(), pointer, bytes, flags));
    },
    [](void *pointer) {
      return static_cast<CUresult>(LUPINE_CUDART_CALL(
          cudaHostUnregister, function_not_found(), pointer));
    },
    [](CUdeviceptr *device_pointer, void *pointer) {
      void *device = nullptr;
      cudaError_t result = LUPINE_CUDART_CALL(
          cudaHostGetDevicePointer, function_not_found(), &device, pointer, 0);
      *device_pointer = reinterpret_cast<CUdeviceptr>(device);
      return static_cast<CUresult>(result);
    }};

} // namespace

int handle_cudaMallocManaged(conn_t *conn) {
  size_t bytes = 0;
  unsigned int flags = 0;
  if (rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUdeviceptr pointer = 0;
  cudaError_t result = static_cast<cudaError_t>(lupine_server_allocate_managed(
      conn, &pointer, bytes, flags,
      [](CUdeviceptr *p, size_t size, unsigned int allocation_flags) {
        void *address = nullptr;
        cudaError_t status =
            LUPINE_CUDART_CALL(cudaMallocManaged, function_not_found(),
                               &address, size, allocation_flags);
        *p = reinterpret_cast<CUdeviceptr>(address);
        return static_cast<CUresult>(status);
      },
      [](CUdeviceptr p) {
        return static_cast<CUresult>(LUPINE_CUDART_CALL(
            cudaFree, function_not_found(), reinterpret_cast<void *>(p)));
      }));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pointer, sizeof(pointer)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaFree(conn_t *conn) {
  CUdeviceptr pointer = 0;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result = static_cast<cudaError_t>(
      lupine_server_free_device_allocation(pointer, [](CUdeviceptr p) {
        return static_cast<CUresult>(LUPINE_CUDART_CALL(
            cudaFree, function_not_found(), reinterpret_cast<void *>(p)));
      }));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaMallocHost(conn_t *conn) {
  size_t bytes = 0;
  if (rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  void *pointer = nullptr;
  cudaError_t result;
  if (conn->va_size == 0 || bytes == 0) {
    result = LUPINE_CUDART_CALL(cudaMallocHost, function_not_found(), &pointer,
                                bytes);
  } else {
    CUdeviceptr device_pointer = 0;
    result = static_cast<cudaError_t>(lupine_server_map_host_allocation(
        conn, &pointer, &device_pointer, bytes, 0, cudaHostRegisterMapped,
        runtime_host_registration));
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pointer, sizeof(pointer)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaHostAlloc(conn_t *conn) {
  size_t bytes = 0;
  unsigned int flags = 0;
  if (rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  void *pointer = nullptr;
  cudaError_t result;
  constexpr unsigned int valid_flags =
      cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;
  if (conn->va_size == 0 || bytes == 0 || (flags & ~valid_flags) != 0) {
    result = LUPINE_CUDART_CALL(cudaHostAlloc, function_not_found(), &pointer,
                                bytes, flags);
  } else {
    CUdeviceptr device_pointer = 0;
    result = static_cast<cudaError_t>(lupine_server_map_host_allocation(
        conn, &pointer, &device_pointer, bytes, flags,
        cudaHostRegisterMapped | (flags & cudaHostAllocPortable),
        runtime_host_registration));
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pointer, sizeof(pointer)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaHostRegister(conn_t *conn) {
  void *client_base = nullptr;
  size_t bytes = 0;
  unsigned int flags = 0;
  if (rpc_read(conn, &client_base, sizeof(client_base)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  void *pointer = nullptr;
  CUdeviceptr device_pointer = 0;
  cudaError_t result =
      static_cast<cudaError_t>(lupine_server_map_host_allocation(
          conn, &pointer, &device_pointer, bytes, flags,
          flags | cudaHostRegisterMapped, runtime_host_registration));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pointer, sizeof(pointer)) < 0 ||
      rpc_write(conn, &device_pointer, sizeof(device_pointer)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaHostUnregister(conn_t *conn) {
  void *pointer = nullptr;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result = static_cast<cudaError_t>(
      lupine_server_free_host_allocation(pointer, [](void *p) {
        return static_cast<CUresult>(
            LUPINE_CUDART_CALL(cudaHostUnregister, function_not_found(), p));
      }));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaFreeHost(conn_t *conn) {
  void *pointer = nullptr;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result = static_cast<cudaError_t>(
      lupine_server_free_host_allocation(pointer, [](void *p) {
        return static_cast<CUresult>(
            LUPINE_CUDART_CALL(cudaFreeHost, function_not_found(), p));
      }));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaHostGetFlags(conn_t *conn) {
  void *pointer = nullptr;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  unsigned int flags = 0;
  cudaError_t result = cudaSuccess;
  if (!lupine_server_host_allocation_flags(pointer, &flags)) {
    result = LUPINE_CUDART_CALL(cudaHostGetFlags, function_not_found(), &flags,
                                pointer);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
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

struct function_registration {
  const char *host_function = nullptr;
  std::string device_function;
  std::string device_name;
};

// NVIDIA keeps these names for later kernel lookup. List nodes keep their
// addresses stable as more functions are registered to the same fatbin.
struct fatbin_registration {
  lupine_fatbin_wrapper wrapper = {};
  std::vector<unsigned char> image;
  std::list<function_registration> functions;
};

std::mutex &registry_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<void **, fatbin_registration *> &registrations() {
  static auto *map = new std::unordered_map<void **, fatbin_registration *>();
  return *map;
}

typedef void **(*register_fat_binary_fn)(void *);
typedef void (*unregister_fat_binary_fn)(void **);

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
    lupine_note_device_stdout_image(registration->image.data(),
                                    registration->image.size());
    std::lock_guard<std::mutex> lock(registry_mutex());
    registrations()[handle] = registration;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle___cudaRegisterFunction(conn_t *conn) {
  void **handle = nullptr;
  function_registration names;
  size_t device_function_length = 0;
  size_t device_name_length = 0;
  int thread_limit = 0;
  uint3 *tid_present = nullptr, *bid_present = nullptr;
  dim3 *block_dim_present = nullptr, *grid_dim_present = nullptr;
  int *warp_size_present = nullptr;
  uint3 tid{}, bid{};
  dim3 block_dim, grid_dim;
  int warp_size = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &names.host_function, sizeof(names.host_function)) < 0 ||
      rpc_read(conn, &device_function_length, sizeof(device_function_length)) <
          0) {
    return -1;
  }
  names.device_function.resize(device_function_length);
  if (rpc_read(conn, names.device_function.data(), device_function_length) <
          0 ||
      rpc_read(conn, &device_name_length, sizeof(device_name_length)) < 0) {
    return -1;
  }
  names.device_name.resize(device_name_length);
  if (rpc_read(conn, names.device_name.data(), device_name_length) < 0 ||
      names.device_function.empty() || names.device_function.back() != '\0' ||
      names.device_name.empty() || names.device_name.back() != '\0' ||
      rpc_read(conn, &thread_limit, sizeof(thread_limit)) < 0 ||
      rpc_read(conn, &tid_present, sizeof(tid_present)) < 0 ||
      (tid_present != nullptr && rpc_read(conn, &tid, sizeof(tid)) < 0) ||
      rpc_read(conn, &bid_present, sizeof(bid_present)) < 0 ||
      (bid_present != nullptr && rpc_read(conn, &bid, sizeof(bid)) < 0) ||
      rpc_read(conn, &block_dim_present, sizeof(block_dim_present)) < 0 ||
      (block_dim_present != nullptr &&
       rpc_read(conn, &block_dim, sizeof(block_dim)) < 0) ||
      rpc_read(conn, &grid_dim_present, sizeof(grid_dim_present)) < 0 ||
      (grid_dim_present != nullptr &&
       rpc_read(conn, &grid_dim, sizeof(grid_dim)) < 0) ||
      rpc_read(conn, &warp_size_present, sizeof(warp_size_present)) < 0 ||
      (warp_size_present != nullptr &&
       rpc_read(conn, &warp_size, sizeof(warp_size)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = void (*)(void **, const char *, char *, const char *, int,
                        uint3 *, uint3 *, dim3 *, dim3 *, int *);
  auto fn = cudart_symbol<fn_t>("__cudaRegisterFunction");
  if (fn == nullptr) {
    return -1;
  }
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto entry = registrations().find(handle);
    if (entry == registrations().end()) {
      return -1;
    }
    entry->second->functions.push_back(std::move(names));
    auto &stored = entry->second->functions.back();
    fn(handle, stored.host_function, stored.device_function.data(),
       stored.device_name.c_str(), thread_limit,
       tid_present != nullptr ? &tid : nullptr,
       bid_present != nullptr ? &bid : nullptr,
       block_dim_present != nullptr ? &block_dim : nullptr,
       grid_dim_present != nullptr ? &grid_dim : nullptr,
       warp_size_present != nullptr ? &warp_size : nullptr);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle___cudaUnregisterFatBinary(conn_t *conn) {
  void **handle = nullptr;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn =
      cudart_symbol<unregister_fat_binary_fn>("__cudaUnregisterFatBinary");
  if (fn == nullptr) {
    return -1;
  }
  fn(handle);
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto entry = registrations().find(handle);
    if (entry != registrations().end()) {
      delete entry->second;
      registrations().erase(entry);
    }
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Completion and event bookkeeping shared with the driver copy paths
// ---------------------------------------------------------------------------

namespace {

int write_completion(conn_t *conn, int request_id, cudaError_t result,
                     std::vector<lupine_pending_dtoh_item> *pending,
                     const std::vector<lupine_graph_host_copy> &graph_copies,
                     const lupine_captured_stdout *capture) {
  const uint32_t count =
      static_cast<uint32_t>(pending->size() + graph_copies.size());
  bool failed = rpc_write_start_response(conn, request_id) < 0 ||
                rpc_copy_alloc(conn, 2 * sizeof(uint64_t)) < 0 ||
                rpc_write(conn, &count, sizeof(count)) < 0;
  for (const auto &copy : graph_copies) {
    if (failed) {
      break;
    }
    failed = rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
             rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
             rpc_write(conn, copy.server_src, copy.bytes) < 0;
  }
  failed =
      failed || lupine_write_pending_dtoh_copies(conn, *pending, false) < 0 ||
      (capture != nullptr &&
       lupine_write_captured_stdout(conn, *capture) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0;
  lupine_cleanup_pending_dtoh_copies(pending);
  return failed ? -1 : 0;
}

int handle_stream_completion(conn_t *conn, bool synchronize) {
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  if (synchronize) {
    lupine_start_stdout_capture(&capture);
  }
  cudaError_t result =
      synchronize
          ? LUPINE_CUDART_CALL(cudaStreamSynchronize, function_not_found(),
                               stream)
          : LUPINE_CUDART_CALL(cudaStreamQuery, function_not_found(), stream);
  if (synchronize) {
    lupine_finish_stdout_capture(&capture);
  }
  std::vector<lupine_pending_dtoh_item> pending;
  std::vector<lupine_graph_host_copy> graph_copies;
  if (result == cudaSuccess) {
    pending =
        lupine_detach_pending_dtoh_copies(conn, stream, stream == nullptr);
    graph_copies = lupine_take_stream_dtoh_copies(stream);
  }
  return write_completion(conn, request_id, result, &pending, graph_copies,
                          synchronize ? &capture : nullptr);
}

int handle_event_completion(conn_t *conn, bool synchronize) {
  cudaEvent_t event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  if (synchronize) {
    lupine_start_stdout_capture(&capture);
  }
  cudaError_t result =
      synchronize
          ? LUPINE_CUDART_CALL(cudaEventSynchronize, function_not_found(),
                               event)
          : LUPINE_CUDART_CALL(cudaEventQuery, function_not_found(), event);
  if (synchronize) {
    lupine_finish_stdout_capture(&capture);
  }
  std::vector<lupine_pending_dtoh_item> pending;
  if (result == cudaSuccess) {
    pending = lupine_detach_event_dtoh_copies(conn, event);
  }
  return write_completion(conn, request_id, result, &pending, {},
                          synchronize ? &capture : nullptr);
}

int handle_event_record(conn_t *conn, bool with_flags) {
  cudaEvent_t event = nullptr;
  cudaStream_t stream = nullptr;
  unsigned int flags = 0;
  if (rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result =
      with_flags
          ? LUPINE_CUDART_CALL(cudaEventRecordWithFlags, function_not_found(),
                               event, stream, flags)
          : LUPINE_CUDART_CALL(cudaEventRecord, function_not_found(), event,
                               stream);
  if (result == cudaSuccess) {
    lupine_record_event_capture_resources(event, stream);
    lupine_note_event_record(conn, event, stream);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cudaDeviceSynchronize(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  cudaError_t result =
      LUPINE_CUDART_CALL(cudaDeviceSynchronize, function_not_found());
  lupine_finish_stdout_capture(&capture);
  std::vector<lupine_pending_dtoh_item> pending;
  if (result == cudaSuccess) {
    pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  }
  return write_completion(conn, request_id, result, &pending, {}, &capture);
}

int handle_cudaStreamSynchronize(conn_t *conn) {
  return handle_stream_completion(conn, true);
}

int handle_cudaStreamQuery(conn_t *conn) {
  return handle_stream_completion(conn, false);
}

int handle_cudaEventSynchronize(conn_t *conn) {
  return handle_event_completion(conn, true);
}

int handle_cudaEventQuery(conn_t *conn) {
  return handle_event_completion(conn, false);
}

int handle_cudaEventRecord(conn_t *conn) {
  return handle_event_record(conn, false);
}

int handle_cudaEventRecordWithFlags(conn_t *conn) {
  return handle_event_record(conn, true);
}

int handle_cudaEventDestroy(conn_t *conn) {
  cudaEvent_t event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result =
      LUPINE_CUDART_CALL(cudaEventDestroy, function_not_found(), event);
  if (result == cudaSuccess) {
    lupine_forget_event_capture_resources(event);
    lupine_forget_event_dtoh_marker(conn, event);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaStreamBeginCapture(conn_t *conn) {
  cudaStream_t stream = nullptr;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *resources = lupine_begin_stream_capture_resources(stream);
  constexpr size_t scratch_size = 128ull * 1024ull * 1024ull;
  void *scratch = nullptr;
  cudaError_t result = LUPINE_CUDART_CALL(cudaMallocHost, function_not_found(),
                                          &scratch, scratch_size);
  if (result == cudaSuccess &&
      !lupine_graph_install_capture_scratch(resources, scratch, scratch_size)) {
    LUPINE_CUDART_CALL(cudaFreeHost, function_not_found(), scratch);
  }
  if (result == cudaSuccess) {
    result = static_cast<cudaError_t>(lupine_server_prepare_htod_capture(conn));
  }
  if (result == cudaSuccess) {
    result = LUPINE_CUDART_CALL(cudaStreamBeginCapture, function_not_found(),
                                stream, mode);
  }
  if (result != cudaSuccess) {
    lupine_discard_stream_capture_resources(resources);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaStreamEndCapture(conn_t *conn) {
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaGraph_t graph = nullptr;
  cudaError_t result = LUPINE_CUDART_CALL(cudaStreamEndCapture,
                                          function_not_found(), stream, &graph);
  lupine_finish_stream_capture_resources(stream, graph, result == cudaSuccess);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cudaLaunchHostFunc(conn_t *conn) {
  cudaStream_t stream = nullptr;
  cudaHostFn_t callback = nullptr;
  void *userData = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *data = new lupine_host_callback_data{
      conn, callback, userData, lupine_get_stream_resources(stream), stream};
  cudaError_t result = LUPINE_CUDART_CALL(
      cudaLaunchHostFunc, function_not_found(), stream,
      callback == nullptr ? nullptr : lupine_graph_host_callback, data);
  if (result != cudaSuccess) {
    delete data;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#if CUDART_VERSION >= 13000
int handle_cudaLaunchHostFunc_v2(conn_t *conn) {
  cudaStream_t stream = nullptr;
  cudaHostFn_t callback = nullptr;
  void *userData = nullptr;
  unsigned int syncMode = 0;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0 ||
      rpc_read(conn, &syncMode, sizeof(syncMode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *data = new lupine_host_callback_data{
      conn, callback, userData, lupine_get_stream_resources(stream), stream};
  cudaError_t result = LUPINE_CUDART_CALL(
      cudaLaunchHostFunc_v2, function_not_found(), stream,
      callback == nullptr ? nullptr : lupine_graph_host_callback, data,
      syncMode);
  if (result != cudaSuccess) {
    delete data;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
#endif

namespace {

void CUDART_CB runtime_stream_callback(cudaStream_t stream, cudaError_t status,
                                       void *userData) {
  lupine_stream_callback(stream, static_cast<CUresult>(status), userData);
}

} // namespace

int handle_cudaStreamAddCallback(conn_t *conn) {
  cudaStream_t stream = nullptr;
  cudaStreamCallback_t callback = nullptr;
  void *userData = nullptr;
  unsigned int flags = 0;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto *data = new lupine_stream_callback_data{
      conn, reinterpret_cast<CUstreamCallback>(callback), userData};
  cudaError_t result = LUPINE_CUDART_CALL(
      cudaStreamAddCallback, function_not_found(), stream,
      callback == nullptr ? nullptr : runtime_stream_callback, data, flags);
  if (result != cudaSuccess) {
    delete data;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Kernel launches
// ---------------------------------------------------------------------------

namespace {

// The client packs argument values in this order. Do not probe past the end
// with runtime APIs: their error would leak into the caller's cudaGetLastError.
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
  // CUDA 13 accepts both registered host entries and cudaKernel_t handles.
  // Identify host entries from their fatbin registration without probing a
  // runtime API with the wrong handle kind and changing its last-error state.
  CUkernel kernel = nullptr;
#if CUDART_VERSION >= 13000
  kernel = reinterpret_cast<CUkernel>(const_cast<void *>(func));
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    for (const auto &entry : registrations()) {
      for (const auto &registered_function : entry.second->functions) {
        if (registered_function.host_function == func) {
          kernel = nullptr;
          break;
        }
      }
      if (kernel == nullptr) {
        break;
      }
    }
  }
#endif
  cudaFunction_t function = nullptr;
  if (kernel == nullptr) {
    result = LUPINE_CUDART_CALL(cudaGetFuncBySymbol, function_not_found(),
                                &function, func);
  }
  // Driver metadata queries do not change the runtime's last-error state
  // when the terminating index is out of range. Launches still use CUDART.
  for (size_t index = 0; result == cudaSuccess; ++index) {
    size_t offset = 0;
    size_t size = 0;
    CUresult status =
        kernel != nullptr
            ? cuKernelGetParamInfo(kernel, index, &offset, &size)
            : cuFuncGetParamInfo(reinterpret_cast<CUfunction>(function), index,
                                 &offset, &size);
    if (status == CUDA_ERROR_INVALID_VALUE) {
      break;
    }
    if (status != CUDA_SUCCESS) {
      result = status == CUDA_ERROR_DEINITIALIZED
                   ? cudaErrorDeviceUninitialized
                   : static_cast<cudaError_t>(status);
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

#include "codegen/gen_cudart_server.inc"
