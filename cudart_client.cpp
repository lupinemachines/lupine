// CUDA runtime API (libcudart) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcudart, sent on the
// CUDA driver shim's connections: the same server child owns the driver state
// and the runtime state, and one lane per client thread keeps the two APIs
// ordered. Most of the surface is generated. The calls in this file carry
// something the generated marshalling cannot: the host side of a copy, the
// image nvcc embedded in the caller, kernel arguments whose sizes only the
// server knows, or a value whose width the attribute decides. What cannot be
// forwarded at all -- memory or a function pointer inside the server, a
// callback into the client -- is a generated stub that returns
// cudaErrorNotSupported.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"
#include "lupine_fatbin.h"

namespace {

cudaError_t rpc_error() { return cudaErrorDevicesUnavailable; }

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------

// The virtual ordinal selects a server; that server's runtime owns device
// binding on the caller's lane.
thread_local int current_device = 0;

conn_t *connection_for_device(int *device) {
  return lupine_rpc_conn_for_device(device);
}

conn_t *connection() {
  int device = current_device;
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_stream(cudaStream_t stream) {
  return stream == nullptr ? connection() : lupine_rpc_conn_for_stream(stream);
}

conn_t *connection_for_event(cudaEvent_t event) {
  return event == nullptr ? connection() : lupine_rpc_conn_for_event(event);
}

void note_stream_owner(conn_t *conn, cudaStream_t stream) {
  lupine_rpc_note_stream_owner(conn, stream);
}

void note_event_owner(conn_t *conn, cudaEvent_t event) {
  lupine_rpc_note_event_owner(conn, event);
}

// Every server that holds a virtual device, once each.
std::vector<conn_t *> all_connections() {
  std::vector<conn_t *> connections;
  int count = 0;
  if (lupine_rpc_device_count(&count) < 0) {
    return connections;
  }
  for (int device = 0; device < count; ++device) {
    int remote_device = device;
    conn_t *conn = lupine_rpc_conn_for_device(&remote_device);
    if (conn != nullptr && std::find(connections.begin(), connections.end(),
                                     conn) == connections.end()) {
      connections.push_back(conn);
    }
  }
  return connections;
}

} // namespace

#include "codegen/gen_cudart_client.inc"

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  if (count == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  return lupine_rpc_device_count(count) < 0 ? record(rpc_error()) : cudaSuccess;
}

extern "C" cudaError_t cudaGetDevice(int *device) {
  if (device == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  *device = current_device;
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  int remote_device = device;
  conn_t *conn = connection_for_device(&remote_device);
  if (conn == nullptr) {
    return record(cudaErrorInvalidDevice);
  }
  cudaError_t result = lupine_rpc_cudaSetDevice(conn, remote_device);
  if (result != cudaSuccess) {
    return record(result);
  }
  current_device = device;
  // Keep driver calls mixed with runtime calls on the same context and route.
  CUcontext context = nullptr;
  if (cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS) {
    (void)cuCtxSetCurrent(context);
  }
  return cudaSuccess;
}

#if CUDART_VERSION >= 12000
extern "C" cudaError_t cudaInitDevice(int device, unsigned int deviceFlags,
                                      unsigned int flags) {
  int remote_device = device;
  conn_t *conn = connection_for_device(&remote_device);
  if (conn == nullptr) {
    return record(cudaErrorInvalidDevice);
  }
  return record(
      lupine_rpc_cudaInitDevice(conn, remote_device, deviceFlags, flags));
}
#endif

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
  if (len < 0 || (len != 0 && device_arr == nullptr)) {
    return record(cudaErrorInvalidValue);
  }
  std::vector<int> remote_devices(static_cast<size_t>(len));
  conn_t *conn = nullptr;
  for (int i = 0; i < len; ++i) {
    remote_devices[i] = device_arr[i];
    conn_t *device_conn = connection_for_device(&remote_devices[i]);
    if (device_conn == nullptr) {
      return record(cudaErrorInvalidDevice);
    }
    // The list is a single preference order, so it can only be forwarded to
    // one server; devices behind another connection are not expressible.
    if (conn != nullptr && device_conn != conn) {
      return record(cudaErrorNotSupported);
    }
    conn = device_conn;
  }
  if (conn == nullptr) {
    conn = connection();
  }
  return record(
      lupine_rpc_cudaSetValidDevices(conn, remote_devices.data(), len));
}

extern "C" cudaError_t cudaDeviceReset() {
  conn_t *conn = connection();
  cudaError_t result = lupine_rpc_cudaDeviceReset(conn);
  local_error = cudaSuccess;
  return record(result);
}

extern "C" cudaError_t cudaGetLastError() {
  const cudaError_t error = local_error;
  local_error = cudaSuccess;
  return error;
}

extern "C" cudaError_t cudaPeekAtLastError() { return local_error; }

namespace {

// The server's runtime owns the message strings, so they are fetched once and
// held per thread: the returned pointer stays valid until this thread's next
// lookup of the same kind. The shim's own unreachable-server code is answered
// locally, because its occurrence means the forward path may already be gone.
const char *error_text(int op, cudaError_t error, bool want_name) {
  if (error == cudaErrorDevicesUnavailable) {
    return want_name ? "cudaErrorDevicesUnavailable"
                     : "CUDA-capable device(s) is/are busy or unavailable";
  }
  thread_local std::string name_result;
  thread_local std::string description_result;
  std::string &result = want_name ? name_result : description_result;

  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = connection();
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &error, sizeof(error)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return want_name ? "cudaErrorUnknown" : "unknown error";
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return want_name ? "cudaErrorUnknown" : "unknown error";
  }
  result = std::move(text);
  return result.c_str();
}

} // namespace

extern "C" const char *cudaGetErrorName(cudaError_t error) {
  return error_text(RPC_cudaGetErrorName, error, true);
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
  return error_text(RPC_cudaGetErrorString, error, false);
}

extern "C" struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
                      enum cudaChannelFormatKind f) {
  struct cudaChannelFormatDesc desc = {};
  conn_t *conn = connection();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudaCreateChannelDesc) < 0 ||
      rpc_write(conn, &x, sizeof(x)) < 0 ||
      rpc_write(conn, &y, sizeof(y)) < 0 ||
      rpc_write(conn, &z, sizeof(z)) < 0 ||
      rpc_write(conn, &w, sizeof(w)) < 0 ||
      rpc_write(conn, &f, sizeof(f)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &desc, sizeof(desc)) < 0 || rpc_read_end(conn) < 0) {
    record(rpc_error());
  }
  return desc;
}

// ---------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------

namespace {

// The server can tell its own device memory from a pointer it has never seen,
// so a cudaMemcpyDefault copy asks it about each side to pick a direction.
bool server_memory(conn_t *conn, const void *pointer) {
  cudaPointerAttributes attributes = {};
  if (lupine_rpc_cudaPointerGetAttributes(conn, &attributes, pointer) !=
      cudaSuccess) {
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice ||
         attributes.type == cudaMemoryTypeManaged;
}

cudaMemcpyKind resolve_kind(conn_t *conn, const void *dst, const void *src,
                            cudaMemcpyKind kind) {
  if (kind != cudaMemcpyDefault) {
    return kind;
  }
  const bool dst_device = server_memory(conn, dst);
  const bool src_device = server_memory(conn, src);
  if (dst_device) {
    return src_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  }
  return src_device ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
}

// The host side of a copy, if there is one, travels as a payload after the
// parameters; the server answers with the result and, for a device-to-host
// copy, the bytes. Both ends are client memory only for host-to-host, which
// the server never sees.
cudaError_t copy(conn_t *conn, int op, void *dst, const void *src, size_t count,
                 cudaMemcpyKind kind, const cudaStream_t *stream) {
  if (conn == nullptr) {
    return record(rpc_error());
  }
  kind = resolve_kind(conn, dst, src, kind);
  if (kind == cudaMemcpyHostToHost) {
    memmove(dst, src, count);
    return cudaSuccess;
  }
  cudaError_t return_value = rpc_error();
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &dst, sizeof(dst)) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      (kind == cudaMemcpyHostToDevice && count != 0 &&
       rpc_write(conn, src, count) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      (kind == cudaMemcpyDeviceToHost && return_value == cudaSuccess &&
       count != 0 && rpc_read(conn, dst, count) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

// A pitched copy travels packed: `height` rows of `width` bytes.
cudaError_t copy_2d(conn_t *conn, int op, void *dst, size_t dpitch,
                    const void *src, size_t spitch, size_t width, size_t height,
                    cudaMemcpyKind kind, const cudaStream_t *stream) {
  if (conn == nullptr) {
    return record(rpc_error());
  }
  kind = resolve_kind(conn, dst, src, kind);
  if (kind == cudaMemcpyHostToHost) {
    for (size_t row = 0; row < height; ++row) {
      memmove(static_cast<char *>(dst) + row * dpitch,
              static_cast<const char *>(src) + row * spitch, width);
    }
    return cudaSuccess;
  }
  std::vector<unsigned char> packed;
  if (kind == cudaMemcpyHostToDevice) {
    packed.resize(width * height);
    for (size_t row = 0; row < height; ++row) {
      memcpy(packed.data() + row * width,
             static_cast<const char *>(src) + row * spitch, width);
    }
  }
  cudaError_t return_value = rpc_error();
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &dst, sizeof(dst)) < 0 ||
      rpc_write(conn, &dpitch, sizeof(dpitch)) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &spitch, sizeof(spitch)) < 0 ||
      rpc_write(conn, &width, sizeof(width)) < 0 ||
      rpc_write(conn, &height, sizeof(height)) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      (!packed.empty() && rpc_write(conn, packed.data(), packed.size()) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0) {
    return record(rpc_error());
  }
  if (kind == cudaMemcpyDeviceToHost && return_value == cudaSuccess &&
      width * height != 0) {
    packed.resize(width * height);
    if (rpc_read(conn, packed.data(), packed.size()) < 0) {
      return record(rpc_error());
    }
    for (size_t row = 0; row < height; ++row) {
      memcpy(static_cast<char *>(dst) + row * dpitch,
             packed.data() + row * width, width);
    }
  }
  return rpc_read_end(conn) < 0 ? rpc_error() : return_value;
}

// A symbol copy names the device side by the host entry point nvcc
// registered; only the other side can be client memory.
cudaError_t copy_symbol(conn_t *conn, int op, const void *symbol, void *other,
                        size_t count, size_t offset, cudaMemcpyKind kind,
                        const cudaStream_t *stream, bool to_symbol) {
  if (conn == nullptr) {
    return record(rpc_error());
  }
  if (kind == cudaMemcpyDefault) {
    const bool device = server_memory(conn, other);
    kind = device      ? cudaMemcpyDeviceToDevice
           : to_symbol ? cudaMemcpyHostToDevice
                       : cudaMemcpyDeviceToHost;
  }
  const bool send = to_symbol && kind == cudaMemcpyHostToDevice;
  const bool receive = !to_symbol && kind == cudaMemcpyDeviceToHost;
  cudaError_t return_value = rpc_error();
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &symbol, sizeof(symbol)) < 0 ||
      rpc_write(conn, &other, sizeof(other)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, &offset, sizeof(offset)) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      (send && count != 0 && rpc_write(conn, other, count) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      (receive && return_value == cudaSuccess && count != 0 &&
       rpc_read(conn, other, count) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

} // namespace

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  enum cudaMemcpyKind kind) {
  return copy(connection(), RPC_cudaMemcpy, dst, src, count, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                       enum cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  return copy(connection_for_stream(stream), RPC_cudaMemcpyAsync, dst, src,
              count, kind, &stream);
}

extern "C" cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                    size_t spitch, size_t width, size_t height,
                                    enum cudaMemcpyKind kind) {
  return copy_2d(connection(), RPC_cudaMemcpy2D, dst, dpitch, src, spitch,
                 width, height, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                         const void *src, size_t spitch,
                                         size_t width, size_t height,
                                         enum cudaMemcpyKind kind,
                                         cudaStream_t stream) {
  return copy_2d(connection_for_stream(stream), RPC_cudaMemcpy2DAsync, dst,
                 dpitch, src, spitch, width, height, kind, &stream);
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                          size_t count, size_t offset,
                                          enum cudaMemcpyKind kind) {
  return copy_symbol(connection(), RPC_cudaMemcpyToSymbol, symbol,
                     const_cast<void *>(src), count, offset, kind, nullptr,
                     true);
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                            size_t count, size_t offset,
                                            enum cudaMemcpyKind kind) {
  return copy_symbol(connection(), RPC_cudaMemcpyFromSymbol, symbol, dst, count,
                     offset, kind, nullptr, false);
}

extern "C" cudaError_t cudaMemcpyToSymbolAsync(const void *symbol,
                                               const void *src, size_t count,
                                               size_t offset,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  return copy_symbol(connection_for_stream(stream), RPC_cudaMemcpyToSymbolAsync,
                     symbol, const_cast<void *>(src), count, offset, kind,
                     &stream, true);
}

extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  return copy_symbol(connection_for_stream(stream),
                     RPC_cudaMemcpyFromSymbolAsync, symbol, dst, count, offset,
                     kind, &stream, false);
}

namespace {

// A peer copy names two virtual ordinals; both have to sit behind one server.
cudaError_t peer_connection(int *dstDevice, int *srcDevice, conn_t **conn) {
  conn_t *dst_conn = connection_for_device(dstDevice);
  conn_t *src_conn = connection_for_device(srcDevice);
  if (dst_conn == nullptr || src_conn == nullptr) {
    return cudaErrorInvalidDevice;
  }
  if (dst_conn != src_conn) {
    return cudaErrorNotSupported;
  }
  *conn = dst_conn;
  return cudaSuccess;
}

} // namespace

extern "C" cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                                      int srcDevice, size_t count) {
  conn_t *conn = nullptr;
  cudaError_t routed = peer_connection(&dstDevice, &srcDevice, &conn);
  if (routed != cudaSuccess) {
    return record(routed);
  }
  return record(
      lupine_rpc_cudaMemcpyPeer(conn, dst, dstDevice, src, srcDevice, count));
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                           const void *src, int srcDevice,
                                           size_t count, cudaStream_t stream) {
  conn_t *conn = nullptr;
  cudaError_t routed = peer_connection(&dstDevice, &srcDevice, &conn);
  if (routed != cudaSuccess) {
    return record(routed);
  }
  return record(lupine_rpc_cudaMemcpyPeerAsync(conn, dst, dstDevice, src,
                                               srcDevice, count, stream));
}

// ---------------------------------------------------------------------------
// Attributes whose width the attribute decides
// ---------------------------------------------------------------------------

namespace {

// The caller's buffer is as wide as the attribute says, so the width goes on
// the wire and the server reads or writes exactly that many bytes.
size_t mem_pool_attribute_width(enum cudaMemPoolAttr attr) {
  switch (attr) {
  case cudaMemPoolReuseFollowEventDependencies:
  case cudaMemPoolReuseAllowOpportunistic:
  case cudaMemPoolReuseAllowInternalDependencies:
    return sizeof(int);
  default:
    return sizeof(cuuint64_t);
  }
}

cudaError_t get_attribute(conn_t *conn, int op, const void *handle,
                          size_t handle_size, int attr, void *value,
                          size_t width) {
  if (value == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t return_value = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, handle, handle_size) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &width, sizeof(width)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      (return_value == cudaSuccess && rpc_read(conn, value, width) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

cudaError_t set_attribute(conn_t *conn, int op, const void *handle,
                          size_t handle_size, int attr, const void *value,
                          size_t width) {
  if (value == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t return_value = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, handle, handle_size) < 0 ||
      rpc_write(conn, &attr, sizeof(attr)) < 0 ||
      rpc_write(conn, &width, sizeof(width)) < 0 ||
      rpc_write(conn, value, width) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

} // namespace

extern "C" cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  return get_attribute(connection(), RPC_cudaMemPoolGetAttribute, &memPool,
                       sizeof(memPool), attr, value,
                       mem_pool_attribute_width(attr));
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  return set_attribute(connection(), RPC_cudaMemPoolSetAttribute, &memPool,
                       sizeof(memPool), attr, value,
                       mem_pool_attribute_width(attr));
}

extern "C" cudaError_t
cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  conn_t *conn = connection_for_device(&device);
  return get_attribute(conn, RPC_cudaDeviceGetGraphMemAttribute, &device,
                       sizeof(device), attr, value, sizeof(cuuint64_t));
}

extern "C" cudaError_t
cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  conn_t *conn = connection_for_device(&device);
  return set_attribute(conn, RPC_cudaDeviceSetGraphMemAttribute, &device,
                       sizeof(device), attr, value, sizeof(cuuint64_t));
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

namespace {

// nvcc registers each embedded image once per process; the runtime on every
// server has to see it, so a client handle stands for one server handle per
// connection and every later registration call fans out the same way.
struct fatbin_registration {
  std::vector<std::pair<conn_t *, void **>> handles;
};

std::mutex &registry_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<void **, fatbin_registration> &fatbins() {
  static auto *map = new std::unordered_map<void **, fatbin_registration>();
  return *map;
}

// The fatbin proper: an outer header followed by its member entries.
size_t fatbin_size(const void *image) {
  const auto *header = static_cast<const lupine_fatbin_header *>(image);
  if (header == nullptr || header->magic != LUPINE_FATBIN_MAGIC) {
    return 0;
  }
  return header->header_size + header->files_size;
}

// Sends the same registration to every server that holds the fatbin.
// Byte strings carry a uint64_t length followed by that many bytes. The length
// storage must live until rpc_wait_for_response sends the queued writes.
template <typename Write>
void register_on_each(void **fatCubinHandle, int op, Write write) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = fatbins().find(fatCubinHandle);
  if (entry == fatbins().end()) {
    return;
  }
  for (const auto &[conn, handle] : entry->second.handles) {
    int ack = 0;
    if (rpc_write_start_request(conn, op) < 0 ||
        rpc_write(conn, &handle, sizeof(handle)) < 0 || write(conn) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &ack, sizeof(ack)) < 0 || rpc_read_end(conn) < 0) {
      record(rpc_error());
    }
  }
}

struct call_configuration {
  dim3 grid;
  dim3 block;
  size_t shared;
  cudaStream_t stream;
};

thread_local std::vector<call_configuration> call_configurations;

} // namespace

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
  const auto *wrapper = static_cast<const lupine_fatbin_wrapper *>(fatCubin);
  uint32_t version = 0;
  const void *image = fatCubin;
  if (wrapper != nullptr && wrapper->magic == LUPINE_FATBINC_MAGIC) {
    version = wrapper->version;
    image = wrapper->data;
  }
  const uint64_t image_size = fatbin_size(image);
  if (image_size == 0) {
    record(cudaErrorInvalidKernelImage);
    return nullptr;
  }
  fatbin_registration registration;
  for (conn_t *conn : all_connections()) {
    void **handle = nullptr;
    if (rpc_write_start_request(conn, RPC___cudaRegisterFatBinary) < 0 ||
        rpc_write(conn, &version, sizeof(version)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write(conn, image, image_size) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &handle, sizeof(handle)) < 0 || rpc_read_end(conn) < 0 ||
        handle == nullptr) {
      record(rpc_error());
      continue;
    }
    registration.handles.emplace_back(conn, handle);
  }
  auto *client_handle = new void *[1];
  client_handle[0] = fatCubin;
  std::lock_guard<std::mutex> lock(registry_mutex());
  fatbins()[client_handle] = std::move(registration);
  return client_handle;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  register_on_each(fatCubinHandle, RPC___cudaRegisterFatBinaryEnd,
                   [](conn_t *) { return 0; });
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  register_on_each(fatCubinHandle, RPC___cudaUnregisterFatBinary,
                   [](conn_t *) { return 0; });
  std::lock_guard<std::mutex> lock(registry_mutex());
  if (fatbins().erase(fatCubinHandle) != 0) {
    delete[] fatCubinHandle;
  }
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
  (void)fatCubinHandle;
  return 1;
}

namespace {

// An optional launch-bound value: a presence flag, then the value.
template <typename T>
int write_optional(conn_t *conn, const uint8_t &present, const T *value) {
  if (rpc_write(conn, &present, sizeof(present)) < 0) {
    return -1;
  }
  return present ? rpc_write(conn, value, sizeof(*value)) : 0;
}

} // namespace

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize) {
  const uint64_t device_fun_len = deviceFun == nullptr ? 0 : strlen(deviceFun);
  const uint64_t device_name_len = deviceName == nullptr ? 0 : strlen(deviceName);
  const uint8_t present[5] = {tid != nullptr, bid != nullptr, bDim != nullptr,
                              gDim != nullptr, wSize != nullptr};
  register_on_each(fatCubinHandle, RPC___cudaRegisterFunction, [&](conn_t *c) {
    return rpc_write(c, &hostFun, sizeof(hostFun)) < 0 ||
                   rpc_write(c, &device_fun_len, sizeof(device_fun_len)) < 0 ||
                   rpc_write(c, deviceFun, device_fun_len) < 0 ||
                   rpc_write(c, &device_name_len, sizeof(device_name_len)) < 0 ||
                   rpc_write(c, deviceName, device_name_len) < 0 ||
                   rpc_write(c, &thread_limit, sizeof(thread_limit)) < 0 ||
                   write_optional(c, present[0], tid) < 0 ||
                   write_optional(c, present[1], bid) < 0 ||
                   write_optional(c, present[2], bDim) < 0 ||
                   write_optional(c, present[3], gDim) < 0 ||
                   write_optional(c, present[4], wSize) < 0
               ? -1
               : 0;
  });
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  const uint64_t device_address_len =
      deviceAddress == nullptr ? 0 : strlen(deviceAddress);
  const uint64_t device_name_len = deviceName == nullptr ? 0 : strlen(deviceName);
  register_on_each(fatCubinHandle, RPC___cudaRegisterVar, [&](conn_t *c) {
    return rpc_write(c, &hostVar, sizeof(hostVar)) < 0 ||
                   rpc_write(c, &device_address_len,
                             sizeof(device_address_len)) < 0 ||
                   rpc_write(c, deviceAddress, device_address_len) < 0 ||
                   rpc_write(c, &device_name_len, sizeof(device_name_len)) < 0 ||
                   rpc_write(c, deviceName, device_name_len) < 0 ||
                   rpc_write(c, &ext, sizeof(ext)) < 0 ||
                   rpc_write(c, &size, sizeof(size)) < 0 ||
                   rpc_write(c, &constant, sizeof(constant)) < 0 ||
                   rpc_write(c, &global, sizeof(global)) < 0
               ? -1
               : 0;
  });
}

// The runtime fills the managed pointer in on the server, so the client's
// slot stays empty: the variable is reachable by symbol, not by host access.
extern "C" void
__cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                         char *deviceAddress, const char *deviceName, int ext,
                         size_t size, int constant, int global) {
  const uint64_t device_address_len =
      deviceAddress == nullptr ? 0 : strlen(deviceAddress);
  const uint64_t device_name_len = deviceName == nullptr ? 0 : strlen(deviceName);
  register_on_each(
      fatCubinHandle, RPC___cudaRegisterManagedVar, [&](conn_t *c) {
        return rpc_write(c, &hostVarPtrAddress, sizeof(hostVarPtrAddress)) <
                           0 ||
                       rpc_write(c, &device_address_len,
                                 sizeof(device_address_len)) < 0 ||
                       rpc_write(c, deviceAddress, device_address_len) < 0 ||
                       rpc_write(c, &device_name_len, sizeof(device_name_len)) < 0 ||
                       rpc_write(c, deviceName, device_name_len) < 0 ||
                       rpc_write(c, &ext, sizeof(ext)) < 0 ||
                       rpc_write(c, &size, sizeof(size)) < 0 ||
                       rpc_write(c, &constant, sizeof(constant)) < 0 ||
                       rpc_write(c, &global, sizeof(global)) < 0
                   ? -1
                   : 0;
      });
}

// Texture and surface references were removed from the device API; only the
// objects are supported, and those carry no host-side registration.
extern "C" void __cudaRegisterTexture(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim, int norm,
                                      int ext) {
  (void)fatCubinHandle;
  (void)hostVar;
  (void)deviceAddress;
  (void)deviceName;
  (void)dim;
  (void)norm;
  (void)ext;
}

extern "C" void __cudaRegisterSurface(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim,
                                      int ext) {
  (void)fatCubinHandle;
  (void)hostVar;
  (void)deviceAddress;
  (void)deviceName;
  (void)dim;
  (void)ext;
}

// Host variables carry no device storage, so registering one only has to be
// accepted. The same is true of the unified-addressing table nvcc hands over.
extern "C" void __cudaRegisterHostVar(void **fatCubinHandle,
                                      const char *deviceName, char *hostVar,
                                      size_t size) {
  (void)fatCubinHandle;
  (void)deviceName;
  (void)hostVar;
  (void)size;
}

extern "C" void __cudaRegisterUnifiedTable(void *table) { (void)table; }

// The <<<>>> lowering pushes the configuration and pops it inside the host
// stub on the same thread, right before cudaLaunchKernel carries it over.
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                void *stream) {
  call_configurations.push_back(
      {gridDim, blockDim, sharedMem, static_cast<cudaStream_t>(stream)});
  return 0;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  void *stream) {
  if (call_configurations.empty()) {
    return cudaErrorInvalidConfiguration;
  }
  const call_configuration configuration = call_configurations.back();
  call_configurations.pop_back();
  if (gridDim != nullptr) {
    *gridDim = configuration.grid;
  }
  if (blockDim != nullptr) {
    *blockDim = configuration.block;
  }
  if (sharedMem != nullptr) {
    *sharedMem = configuration.shared;
  }
  if (stream != nullptr) {
    *static_cast<cudaStream_t *>(stream) = configuration.stream;
  }
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Kernel launches
// ---------------------------------------------------------------------------

namespace {

// Kernel arguments arrive as pointers to values of sizes only the loaded
// kernel knows, so the layout is fetched from the server once per entry point
// and the values travel packed in that order.
std::mutex &layouts_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<const void *, std::vector<size_t>> &layouts() {
  static auto *map =
      new std::unordered_map<const void *, std::vector<size_t>>();
  return *map;
}

cudaError_t param_sizes(conn_t *conn, const void *func,
                        std::vector<size_t> *sizes) {
  {
    std::lock_guard<std::mutex> lock(layouts_mutex());
    auto entry = layouts().find(func);
    if (entry != layouts().end()) {
      *sizes = entry->second;
      return cudaSuccess;
    }
  }
  cudaError_t return_value = rpc_error();
  uint32_t count = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_lupineCudartFuncParamLayout) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0) {
    return record(rpc_error());
  }
  sizes->assign(count, 0);
  if ((count != 0 &&
       rpc_read(conn, sizes->data(), count * sizeof(size_t)) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  if (return_value == cudaSuccess) {
    std::lock_guard<std::mutex> lock(layouts_mutex());
    layouts()[func] = *sizes;
  }
  return record(return_value);
}

int write_params(conn_t *conn, const uint32_t &count,
                 const std::vector<size_t> &sizes, void **args) {
  if (rpc_write(conn, &count, sizeof(count)) < 0 ||
      (count != 0 &&
       rpc_write(conn, sizes.data(), count * sizeof(size_t)) < 0)) {
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (rpc_write(conn, args[i], sizes[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

cudaError_t launch(conn_t *conn, int op, const void *func, dim3 gridDim,
                   dim3 blockDim, void **args, size_t sharedMem,
                   cudaStream_t stream) {
  std::vector<size_t> sizes;
  cudaError_t resolved = param_sizes(conn, func, &sizes);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  if (!sizes.empty() && args == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  const uint32_t count = static_cast<uint32_t>(sizes.size());
  cudaError_t return_value = rpc_error();
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      rpc_write(conn, &gridDim, sizeof(gridDim)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(blockDim)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(sharedMem)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      write_params(conn, count, sizes, args) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

// The launch attributes sit behind a pointer in the config, so they follow it
// as an array.
int write_launch_config(conn_t *conn, const cudaLaunchConfig_t &config,
                        const uint32_t &attribute_count) {
  return rpc_write(conn, &config.gridDim, sizeof(config.gridDim)) < 0 ||
                 rpc_write(conn, &config.blockDim, sizeof(config.blockDim)) <
                     0 ||
                 rpc_write(conn, &config.dynamicSmemBytes,
                           sizeof(config.dynamicSmemBytes)) < 0 ||
                 rpc_write(conn, &config.stream, sizeof(config.stream)) < 0 ||
                 rpc_write(conn, &attribute_count, sizeof(attribute_count)) <
                     0 ||
                 (attribute_count != 0 &&
                  rpc_write(conn, config.attrs,
                            attribute_count * sizeof(*config.attrs)) < 0)
             ? -1
             : 0;
}

cudaError_t occupancy_for_config(int op, int *result, const void *func,
                                 const cudaLaunchConfig_t *launchConfig) {
  if (result == nullptr || launchConfig == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  conn_t *conn = connection_for_stream(launchConfig->stream);
  const uint32_t attribute_count = launchConfig->numAttrs;
  cudaError_t return_value = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      write_launch_config(conn, *launchConfig, attribute_count) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, result, sizeof(*result)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

} // namespace

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream) {
  return launch(connection_for_stream(stream), RPC_cudaLaunchKernel, func,
                gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  return launch(connection_for_stream(stream), RPC_cudaLaunchCooperativeKernel,
                func, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                           const void *func, void **args) {
  if (config == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  conn_t *conn = connection_for_stream(config->stream);
  std::vector<size_t> sizes;
  cudaError_t resolved = param_sizes(conn, func, &sizes);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  if (!sizes.empty() && args == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  const uint32_t attribute_count = config->numAttrs;
  const uint32_t count = static_cast<uint32_t>(sizes.size());
  cudaError_t return_value = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaLaunchKernelExC) < 0 ||
      write_launch_config(conn, *config, attribute_count) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      write_params(conn, count, sizes, args) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

extern "C" cudaError_t
cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func,
                                     const cudaLaunchConfig_t *launchConfig) {
  return occupancy_for_config(RPC_cudaOccupancyMaxPotentialClusterSize,
                              clusterSize, func, launchConfig);
}

extern "C" cudaError_t
cudaOccupancyMaxActiveClusters(int *numClusters, const void *func,
                               const cudaLaunchConfig_t *launchConfig) {
  return occupancy_for_config(RPC_cudaOccupancyMaxActiveClusters, numClusters,
                              func, launchConfig);
}

#if CUDART_VERSION >= 13000
// nvcc's launch lowering resolves the host entry point once per launch site
// and passes the handle back on every launch. The handle is the server's, so
// it is only good for the device the resolving thread had current.
extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *kernel,
                                       const void *entryFuncAddr) {
  if (kernel == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  conn_t *conn = connection();
  cudaError_t return_value = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC___cudaGetKernel) < 0 ||
      rpc_write(conn, &entryFuncAddr, sizeof(entryFuncAddr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, kernel, sizeof(*kernel)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(return_value);
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                          dim3 blockDim, void **args,
                                          size_t sharedMem,
                                          cudaStream_t stream) {
  return launch(connection_for_stream(stream), RPC___cudaLaunchKernel, kernel,
                gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel,
                                               dim3 gridDim, dim3 blockDim,
                                               void **args, size_t sharedMem,
                                               cudaStream_t stream) {
  return __cudaLaunchKernel(kernel, gridDim, blockDim, args, sharedMem, stream);
}
#endif

#if CUDART_VERSION >= 12000
// The runtime owns the name string; a copy is held for the life of the
// process, so the pointer handed back stays valid as the API promises.
extern "C" cudaError_t cudaFuncGetName(const char **name, const void *func) {
  if (name == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  static auto *names = new std::unordered_map<const void *, std::string>();
  static auto *names_mutex = new std::mutex();
  {
    std::lock_guard<std::mutex> lock(*names_mutex);
    auto entry = names->find(func);
    if (entry != names->end()) {
      *name = entry->second.c_str();
      return cudaSuccess;
    }
  }
  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = connection();
  cudaError_t return_value = rpc_error();
  uint32_t length = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cudaFuncGetName) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return record(rpc_error());
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  if (return_value != cudaSuccess) {
    return record(return_value);
  }
  std::lock_guard<std::mutex> lock(*names_mutex);
  *name = names->emplace(func, std::move(text)).first->second.c_str();
  return cudaSuccess;
}
#endif
