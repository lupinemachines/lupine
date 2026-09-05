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

// A length-prefixed byte string from the client, null-terminated here.
int read_bytes(conn_t *conn, std::string *text) {
  uint64_t length = 0;
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  text->assign(static_cast<size_t>(length), '\0');
  return length != 0 ? rpc_read(conn, &(*text)[0], text->size()) : 0;
}

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
// Copies
// ---------------------------------------------------------------------------

namespace {

// A host side is staged here: a host-to-device payload lands in a buffer the
// copy reads, and a device-to-host copy fills one the response carries. An
// asynchronous copy with a host side completes on its stream before the
// staging buffer goes away, which is also what makes the response correct.
struct copy_request {
  void *dst = nullptr;
  const void *src = nullptr;
  size_t count = 0;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cudaStream_t stream = nullptr;
  bool async = false;
};

int read_copy(conn_t *conn, bool async, copy_request *copy,
              std::vector<unsigned char> *staging) {
  copy->async = async;
  if (rpc_read(conn, &copy->dst, sizeof(copy->dst)) < 0 ||
      rpc_read(conn, &copy->src, sizeof(copy->src)) < 0 ||
      rpc_read(conn, &copy->count, sizeof(copy->count)) < 0 ||
      rpc_read(conn, &copy->kind, sizeof(copy->kind)) < 0 ||
      (async && rpc_read(conn, &copy->stream, sizeof(copy->stream)) < 0)) {
    return -1;
  }
  if (copy->kind == cudaMemcpyHostToDevice) {
    staging->resize(copy->count);
    if (copy->count != 0 &&
        rpc_read(conn, staging->data(), staging->size()) < 0) {
      return -1;
    }
    copy->src = staging->data();
  } else if (copy->kind == cudaMemcpyDeviceToHost) {
    staging->resize(copy->count);
    copy->dst = staging->data();
  }
  return 0;
}

cudaError_t run_copy(const copy_request &copy) {
  if (!copy.async) {
    return LUPINE_CUDART_CALL(cudaMemcpy, function_not_found(), copy.dst,
                              copy.src, copy.count, copy.kind);
  }
  cudaError_t result =
      LUPINE_CUDART_CALL(cudaMemcpyAsync, function_not_found(), copy.dst,
                         copy.src, copy.count, copy.kind, copy.stream);
  if (result == cudaSuccess && copy.kind != cudaMemcpyDeviceToDevice) {
    result = LUPINE_CUDART_CALL(cudaStreamSynchronize, function_not_found(),
                                copy.stream);
  }
  return result;
}

int handle_copy(conn_t *conn, bool async) {
  copy_request copy;
  std::vector<unsigned char> staging;
  if (read_copy(conn, async, &copy, &staging) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result = run_copy(copy);
  const bool respond_bytes =
      copy.kind == cudaMemcpyDeviceToHost && result == cudaSuccess;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (respond_bytes && copy.count != 0 &&
       rpc_write(conn, staging.data(), staging.size()) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// A pitched copy travels packed, so the staging buffer's pitch is its width.
int handle_copy_2d(conn_t *conn, bool async) {
  void *dst = nullptr;
  size_t dpitch = 0;
  const void *src = nullptr;
  size_t spitch = 0;
  size_t width = 0;
  size_t height = 0;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &dst, sizeof(dst)) < 0 ||
      rpc_read(conn, &dpitch, sizeof(dpitch)) < 0 ||
      rpc_read(conn, &src, sizeof(src)) < 0 ||
      rpc_read(conn, &spitch, sizeof(spitch)) < 0 ||
      rpc_read(conn, &width, sizeof(width)) < 0 ||
      rpc_read(conn, &height, sizeof(height)) < 0 ||
      rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  std::vector<unsigned char> staging;
  if (kind == cudaMemcpyHostToDevice) {
    staging.resize(width * height);
    if (!staging.empty() &&
        rpc_read(conn, staging.data(), staging.size()) < 0) {
      return -1;
    }
    src = staging.data();
    spitch = width;
  } else if (kind == cudaMemcpyDeviceToHost) {
    staging.resize(width * height);
    dst = staging.data();
    dpitch = width;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result;
  if (!async) {
    result = LUPINE_CUDART_CALL(cudaMemcpy2D, function_not_found(), dst, dpitch,
                                src, spitch, width, height, kind);
  } else {
    result =
        LUPINE_CUDART_CALL(cudaMemcpy2DAsync, function_not_found(), dst, dpitch,
                           src, spitch, width, height, kind, stream);
    if (result == cudaSuccess && kind != cudaMemcpyDeviceToDevice) {
      result = LUPINE_CUDART_CALL(cudaStreamSynchronize, function_not_found(),
                                  stream);
    }
  }
  const bool respond_bytes = kind == cudaMemcpyDeviceToHost &&
                             result == cudaSuccess && !staging.empty();
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (respond_bytes && rpc_write(conn, staging.data(), staging.size()) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_copy_symbol(conn_t *conn, bool async, bool to_symbol) {
  const void *symbol = nullptr;
  void *other = nullptr;
  size_t count = 0;
  size_t offset = 0;
  cudaMemcpyKind kind = cudaMemcpyDefault;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &symbol, sizeof(symbol)) < 0 ||
      rpc_read(conn, &other, sizeof(other)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      rpc_read(conn, &offset, sizeof(offset)) < 0 ||
      rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  const bool receive_bytes = to_symbol && kind == cudaMemcpyHostToDevice;
  const bool respond_bytes = !to_symbol && kind == cudaMemcpyDeviceToHost;
  std::vector<unsigned char> staging;
  if (receive_bytes || respond_bytes) {
    staging.resize(count);
    if (receive_bytes && count != 0 &&
        rpc_read(conn, staging.data(), staging.size()) < 0) {
      return -1;
    }
    other = staging.data();
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result;
  if (to_symbol && !async) {
    result = LUPINE_CUDART_CALL(cudaMemcpyToSymbol, function_not_found(),
                                symbol, other, count, offset, kind);
  } else if (to_symbol) {
    result = LUPINE_CUDART_CALL(cudaMemcpyToSymbolAsync, function_not_found(),
                                symbol, other, count, offset, kind, stream);
  } else if (!async) {
    result = LUPINE_CUDART_CALL(cudaMemcpyFromSymbol, function_not_found(),
                                other, symbol, count, offset, kind);
  } else {
    result = LUPINE_CUDART_CALL(cudaMemcpyFromSymbolAsync, function_not_found(),
                                other, symbol, count, offset, kind, stream);
  }
  if (async && result == cudaSuccess && kind != cudaMemcpyDeviceToDevice) {
    result =
        LUPINE_CUDART_CALL(cudaStreamSynchronize, function_not_found(), stream);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (respond_bytes && result == cudaSuccess && count != 0 &&
       rpc_write(conn, staging.data(), staging.size()) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cudaMemcpy(conn_t *conn) { return handle_copy(conn, false); }
int handle_cudaMemcpyAsync(conn_t *conn) { return handle_copy(conn, true); }
int handle_cudaMemcpy2D(conn_t *conn) { return handle_copy_2d(conn, false); }
int handle_cudaMemcpy2DAsync(conn_t *conn) {
  return handle_copy_2d(conn, true);
}
int handle_cudaMemcpyToSymbol(conn_t *conn) {
  return handle_copy_symbol(conn, false, true);
}
int handle_cudaMemcpyToSymbolAsync(conn_t *conn) {
  return handle_copy_symbol(conn, true, true);
}
int handle_cudaMemcpyFromSymbol(conn_t *conn) {
  return handle_copy_symbol(conn, false, false);
}
int handle_cudaMemcpyFromSymbolAsync(conn_t *conn) {
  return handle_copy_symbol(conn, true, false);
}

// ---------------------------------------------------------------------------
// Attributes whose width the attribute decides
// ---------------------------------------------------------------------------

namespace {

// The client says how wide its buffer is; the value is read or written whole.
template <typename Handle, typename Attr>
int handle_get_attribute(conn_t *conn, const char *symbol) {
  Handle handle;
  int attr = 0;
  size_t width = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &width, sizeof(width)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  std::vector<unsigned char> value(width);
  using fn_t = cudaError_t (*)(Handle, Attr, void *);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  cudaError_t result = fn == nullptr
                           ? function_not_found()
                           : fn(handle, static_cast<Attr>(attr), value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == cudaSuccess && width != 0 &&
       rpc_write(conn, value.data(), width) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename Handle, typename Attr>
int handle_set_attribute(conn_t *conn, const char *symbol) {
  Handle handle;
  int attr = 0;
  size_t width = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &width, sizeof(width)) < 0) {
    return -1;
  }
  std::vector<unsigned char> value(width);
  if (width != 0 && rpc_read(conn, value.data(), width) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cudaError_t (*)(Handle, Attr, void *);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  cudaError_t result = fn == nullptr
                           ? function_not_found()
                           : fn(handle, static_cast<Attr>(attr), value.data());
  return write_result(conn, request_id, result);
}

} // namespace

int handle_cudaMemPoolGetAttribute(conn_t *conn) {
  return handle_get_attribute<cudaMemPool_t, enum cudaMemPoolAttr>(
      conn, "cudaMemPoolGetAttribute");
}

int handle_cudaMemPoolSetAttribute(conn_t *conn) {
  return handle_set_attribute<cudaMemPool_t, enum cudaMemPoolAttr>(
      conn, "cudaMemPoolSetAttribute");
}

int handle_cudaDeviceGetGraphMemAttribute(conn_t *conn) {
  return handle_get_attribute<int, enum cudaGraphMemAttributeType>(
      conn, "cudaDeviceGetGraphMemAttribute");
}

int handle_cudaDeviceSetGraphMemAttribute(conn_t *conn) {
  return handle_set_attribute<int, enum cudaGraphMemAttributeType>(
      conn, "cudaDeviceSetGraphMemAttribute");
}

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
char *retain(fatbin_registration *registration, std::string text) {
  if (registration == nullptr) {
    return nullptr;
  }
  registration->strings.push_back(std::move(text));
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

int write_ack(conn_t *conn, int request_id) {
  int ack = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ack, sizeof(ack)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// An optional launch-bound value: a presence flag, then the value.
template <typename T> int read_optional(conn_t *conn, T *storage, T **value) {
  uint8_t present = 0;
  if (rpc_read(conn, &present, sizeof(present)) < 0) {
    return -1;
  }
  *value = present ? storage : nullptr;
  return present ? rpc_read(conn, storage, sizeof(*storage)) : 0;
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

int handle___cudaRegisterFatBinaryEnd(conn_t *conn) {
  void **handle = nullptr;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn =
      cudart_symbol<register_fat_binary_end_fn>("__cudaRegisterFatBinaryEnd");
  if (fn != nullptr) {
    fn(handle);
  }
  return write_ack(conn, request_id);
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
  if (fn != nullptr) {
    fn(handle);
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = registrations().find(handle);
  if (entry != registrations().end()) {
    delete entry->second;
    registrations().erase(entry);
  }
  return write_ack(conn, request_id);
}

int handle___cudaRegisterFunction(conn_t *conn) {
  void **handle = nullptr;
  const char *hostFun = nullptr;
  std::string deviceFun;
  std::string deviceName;
  int thread_limit = 0;
  uint3 tid_storage, bid_storage;
  dim3 bDim_storage, gDim_storage;
  int wSize_storage = 0;
  uint3 *tid, *bid;
  dim3 *bDim, *gDim;
  int *wSize;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &hostFun, sizeof(hostFun)) < 0 ||
      read_bytes(conn, &deviceFun) < 0 || read_bytes(conn, &deviceName) < 0 ||
      rpc_read(conn, &thread_limit, sizeof(thread_limit)) < 0 ||
      read_optional(conn, &tid_storage, &tid) < 0 ||
      read_optional(conn, &bid_storage, &bid) < 0 ||
      read_optional(conn, &bDim_storage, &bDim) < 0 ||
      read_optional(conn, &gDim_storage, &gDim) < 0 ||
      read_optional(conn, &wSize_storage, &wSize) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn = cudart_symbol<register_function_fn>("__cudaRegisterFunction");
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    fatbin_registration *registration = registration_for(handle);
    if (fn != nullptr && registration != nullptr) {
      fn(handle, hostFun, retain(registration, std::move(deviceFun)),
         retain(registration, std::move(deviceName)), thread_limit, tid, bid,
         bDim, gDim, wSize);
    }
  }
  return write_ack(conn, request_id);
}

int handle___cudaRegisterVar(conn_t *conn) {
  void **handle = nullptr;
  char *hostVar = nullptr;
  std::string deviceAddress;
  std::string deviceName;
  int ext = 0;
  size_t size = 0;
  int constant = 0;
  int global = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &hostVar, sizeof(hostVar)) < 0 ||
      read_bytes(conn, &deviceAddress) < 0 ||
      read_bytes(conn, &deviceName) < 0 ||
      rpc_read(conn, &ext, sizeof(ext)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0 ||
      rpc_read(conn, &constant, sizeof(constant)) < 0 ||
      rpc_read(conn, &global, sizeof(global)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn = cudart_symbol<register_var_fn>("__cudaRegisterVar");
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    fatbin_registration *registration = registration_for(handle);
    if (fn != nullptr && registration != nullptr) {
      fn(handle, hostVar, retain(registration, std::move(deviceAddress)),
         retain(registration, std::move(deviceName)), ext, size, constant,
         global);
    }
  }
  return write_ack(conn, request_id);
}

// The runtime writes the managed pointer into the slot it is given, so the
// slot lives here with the registration; the client keys the variable by the
// address of its own.
int handle___cudaRegisterManagedVar(conn_t *conn) {
  void **handle = nullptr;
  void **hostVarPtrAddress = nullptr;
  std::string deviceAddress;
  std::string deviceName;
  int ext = 0;
  size_t size = 0;
  int constant = 0;
  int global = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &hostVarPtrAddress, sizeof(hostVarPtrAddress)) < 0 ||
      read_bytes(conn, &deviceAddress) < 0 ||
      read_bytes(conn, &deviceName) < 0 ||
      rpc_read(conn, &ext, sizeof(ext)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0 ||
      rpc_read(conn, &constant, sizeof(constant)) < 0 ||
      rpc_read(conn, &global, sizeof(global)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn = cudart_symbol<register_managed_var_fn>("__cudaRegisterManagedVar");
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    fatbin_registration *registration = registration_for(handle);
    if (fn != nullptr && registration != nullptr) {
      registration->managed_slots.push_back(nullptr);
      fn(handle, &registration->managed_slots.back(),
         retain(registration, std::move(deviceAddress)),
         retain(registration, std::move(deviceName)), ext, size, constant,
         global);
    }
  }
  return write_ack(conn, request_id);
}

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
  Entry entry;
  dim3 gridDim, blockDim;
  size_t sharedMem = 0;
  cudaStream_t stream = nullptr;
  packed_params params;
  if (rpc_read(conn, &entry, sizeof(entry)) < 0 ||
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
  cudaError_t result = fn == nullptr
                           ? function_not_found()
                           : fn(entry, gridDim, blockDim,
                                params.pointers.data(), sharedMem, stream);
  return write_result(conn, request_id, result);
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
  launch_config launch;
  const void *func = nullptr;
  packed_params params;
  if (read_launch_config(conn, &launch) < 0 ||
      rpc_read(conn, &func, sizeof(func)) < 0 ||
      read_params(conn, &params) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaError_t result =
      LUPINE_CUDART_CALL(cudaLaunchKernelExC, function_not_found(),
                         &launch.config, func, params.pointers.data());
  return write_result(conn, request_id, result);
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

#if CUDART_VERSION >= 13000
int handle___cudaGetKernel(conn_t *conn) {
  const void *entry = nullptr;
  if (rpc_read(conn, &entry, sizeof(entry)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cudaKernel_t kernel = nullptr;
  using fn_t = cudaError_t (*)(cudaKernel_t *, const void *);
  fn_t fn = cudart_symbol<fn_t>("__cudaGetKernel");
  cudaError_t result =
      fn == nullptr ? function_not_found() : fn(&kernel, entry);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &kernel, sizeof(kernel)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
#endif

#include "codegen/gen_cudart_server.inc"
