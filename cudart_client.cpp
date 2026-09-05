// CUDA runtime API (libcudart) on the LUPINE client.
//
// Most calls are RPCs the server answers with its own libcudart, sent on the
// CUDA driver shim's connections: the same server child owns the driver state
// and the runtime state, and one lane per client thread keeps the two APIs
// ordered. Most of the surface is generated. This file handles the
// image nvcc embedded in the caller, kernel arguments whose sizes only the
// server knows, or a value whose width the attribute decides. What cannot be
// forwarded at all -- memory or a function pointer inside the server, a
// callback into the client -- is a generated stub that returns
// cudaErrorNotSupported.
// Copies and their completion paths call the driver shim, sharing its routing,
// staging and deferred host-copy handling.

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
  conn_t *conn = lupine_rpc_conn_for_device(&remote_device);
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
  conn_t *conn = lupine_rpc_conn_for_device(&remote_device);
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
    conn_t *device_conn = lupine_rpc_conn_for_device(&remote_devices[i]);
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
    int route_device = current_device;
    conn = lupine_rpc_conn_for_device(&route_device);
  }
  return record(
      lupine_rpc_cudaSetValidDevices(conn, remote_devices.data(), len));
}

extern "C" cudaError_t cudaDeviceReset() {
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
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
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
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
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
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

// Most driver and runtime error values agree. A deinitialized driver is not
// cudaErrorCudartUnloading (the runtime's historical value at the same number).
cudaError_t record(CUresult error) {
  return record(error == CUDA_ERROR_DEINITIALIZED
                    ? cudaErrorDeviceUninitialized
                    : static_cast<cudaError_t>(error));
}

cudaError_t copy_2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                    size_t width, size_t height, cudaMemcpyKind kind,
                    const cudaStream_t *stream) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  if (width > dpitch || width > spitch) {
    return record(cudaErrorInvalidPitchValue);
  }
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  // A depth-one volume accepts arbitrary device pitches; cuMemcpy2D can
  // reject pitches that did not come from cuMemAllocPitch.
  CUDA_MEMCPY3D copy = {};
  switch (kind) {
  case cudaMemcpyHostToHost:
    copy.srcMemoryType = copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    break;
  case cudaMemcpyHostToDevice:
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    break;
  case cudaMemcpyDeviceToHost:
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    break;
  case cudaMemcpyDeviceToDevice:
    copy.srcMemoryType = copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    break;
  case cudaMemcpyDefault:
    copy.srcMemoryType = copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
    break;
  }
  copy.srcHost = src;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(src);
  copy.srcPitch = spitch;
  copy.srcHeight = height;
  copy.dstHost = dst;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  copy.dstPitch = dpitch;
  copy.dstHeight = height;
  copy.WidthInBytes = width;
  copy.Height = height;
  copy.Depth = 1;
  return record(stream == nullptr ? cuMemcpy3D(&copy)
                                  : cuMemcpy3DAsync(&copy, *stream));
}

// The runtime resolves nvcc's host symbol key. Only the copy itself is mapped
// to the driver; retain the symbol bounds and owning route for interior
// pointers.
cudaError_t symbol_address(conn_t *conn, void **address, const void *symbol,
                           size_t offset, size_t count) {
  if (address == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  void *base = nullptr;
  size_t size = 0;
  cudaError_t result = lupine_rpc_cudaGetSymbolAddress(conn, &base, symbol);
  if (result == cudaSuccess) {
    result = lupine_rpc_cudaGetSymbolSize(conn, &size, symbol);
  }
  if (result != cudaSuccess) {
    return record(result);
  }
  if (offset > size || count > size - offset) {
    return record(cudaErrorInvalidValue);
  }
  lupine_rpc_note_allocation(conn, base, size);
  *address =
      reinterpret_cast<void *>(reinterpret_cast<CUdeviceptr>(base) + offset);
  return cudaSuccess;
}

cudaError_t copy_peer(void *dst, int dstDevice, const void *src, int srcDevice,
                      size_t count, const cudaStream_t *stream) {
  CUcontext dstContext = nullptr, srcContext = nullptr;
  CUresult result = cuDevicePrimaryCtxRetain(&dstContext, dstDevice);
  if (result != CUDA_SUCCESS) {
    return record(result);
  }
  result = cuDevicePrimaryCtxRetain(&srcContext, srcDevice);
  if (result == CUDA_SUCCESS) {
    result =
        stream == nullptr
            ? cuMemcpyPeer(reinterpret_cast<CUdeviceptr>(dst), dstContext,
                           reinterpret_cast<CUdeviceptr>(src), srcContext,
                           count)
            : cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dst), dstContext,
                                reinterpret_cast<CUdeviceptr>(src), srcContext,
                                count, *stream);
    (void)cuDevicePrimaryCtxRelease(srcDevice);
  }
  (void)cuDevicePrimaryCtxRelease(dstDevice);
  return record(result);
}

} // namespace

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  cudaMemcpyKind kind) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    std::memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return record(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count));
  case cudaMemcpyDeviceToHost:
    return record(cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count));
  case cudaMemcpyDeviceToDevice:
    return record(cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),
                               reinterpret_cast<CUdeviceptr>(src), count));
  case cudaMemcpyDefault:
    return record(cuMemcpy(reinterpret_cast<CUdeviceptr>(dst),
                           reinterpret_cast<CUdeviceptr>(src), count));
  }
  return record(cudaErrorInvalidMemcpyDirection);
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    std::memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return record(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src,
                                    count, stream));
  case cudaMemcpyDeviceToHost:
    return record(cuMemcpyDtoHAsync(dst, reinterpret_cast<CUdeviceptr>(src),
                                    count, stream));
  case cudaMemcpyDeviceToDevice:
    return record(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                    reinterpret_cast<CUdeviceptr>(src), count,
                                    stream));
  case cudaMemcpyDefault:
    return record(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                reinterpret_cast<CUdeviceptr>(src), count,
                                stream));
  }
  return record(cudaErrorInvalidMemcpyDirection);
}

extern "C" cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                    size_t spitch, size_t width, size_t height,
                                    cudaMemcpyKind kind) {
  return copy_2d(dst, dpitch, src, spitch, width, height, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                         const void *src, size_t spitch,
                                         size_t width, size_t height,
                                         cudaMemcpyKind kind,
                                         cudaStream_t stream) {
  return copy_2d(dst, dpitch, src, spitch, width, height, kind, &stream);
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
  return symbol_address(conn, devPtr, symbol, 0, 0);
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                          size_t count, size_t offset,
                                          cudaMemcpyKind kind) {
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice &&
      kind != cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
  void *address = nullptr;
  cudaError_t result = symbol_address(conn, &address, symbol, offset, count);
  if (result != cudaSuccess) {
    return result;
  }
  return cudaMemcpy(address, src, count, kind);
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                            size_t count, size_t offset,
                                            cudaMemcpyKind kind) {
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice &&
      kind != cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
  void *address = nullptr;
  cudaError_t result = symbol_address(conn, &address, symbol, offset, count);
  if (result != cudaSuccess) {
    return result;
  }
  return cudaMemcpy(dst, address, count, kind);
}

extern "C" cudaError_t cudaMemcpyToSymbolAsync(const void *symbol,
                                               const void *src, size_t count,
                                               size_t offset,
                                               cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice &&
      kind != cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  int route_device = current_device;
  conn_t *conn = stream == nullptr ? lupine_rpc_conn_for_device(&route_device)
                                   : lupine_rpc_conn_for_stream(stream);
  void *address = nullptr;
  cudaError_t result = symbol_address(conn, &address, symbol, offset, count);
  if (result != cudaSuccess) {
    return result;
  }
  return cudaMemcpyAsync(address, src, count, kind, stream);
}

extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                                 size_t count, size_t offset,
                                                 cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyDeviceToDevice &&
      kind != cudaMemcpyDefault) {
    return record(cudaErrorInvalidMemcpyDirection);
  }
  int route_device = current_device;
  conn_t *conn = stream == nullptr ? lupine_rpc_conn_for_device(&route_device)
                                   : lupine_rpc_conn_for_stream(stream);
  void *address = nullptr;
  cudaError_t result = symbol_address(conn, &address, symbol, offset, count);
  if (result != cudaSuccess) {
    return result;
  }
  return cudaMemcpyAsync(dst, address, count, kind, stream);
}

extern "C" cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                                      int srcDevice, size_t count) {
  return copy_peer(dst, dstDevice, src, srcDevice, count, nullptr);
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                           const void *src, int srcDevice,
                                           size_t count, cudaStream_t stream) {
  return copy_peer(dst, dstDevice, src, srcDevice, count, &stream);
}

// Driver copies may defer host results until synchronization. Use the same
// completion path to collect those bytes before returning to runtime callers.
extern "C" cudaError_t cudaDeviceSynchronize() {
  return record(cuCtxSynchronize());
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return record(cuStreamSynchronize(stream));
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return record(cuStreamQuery(stream));
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return record(cuEventSynchronize(event));
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  return record(cuEventQuery(event));
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return record(cuEventRecord(event, stream));
}

extern "C" cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,
                                                cudaStream_t stream,
                                                unsigned int flags) {
  return record(cuEventRecordWithFlags(event, stream, flags));
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return record(cuEventDestroy(event));
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
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
  return get_attribute(conn, RPC_cudaMemPoolGetAttribute, &memPool,
                       sizeof(memPool), attr, value,
                       mem_pool_attribute_width(attr));
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
  return set_attribute(conn, RPC_cudaMemPoolSetAttribute, &memPool,
                       sizeof(memPool), attr, value,
                       mem_pool_attribute_width(attr));
}

extern "C" cudaError_t
cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  conn_t *conn = lupine_rpc_conn_for_device(&device);
  return get_attribute(conn, RPC_cudaDeviceGetGraphMemAttribute, &device,
                       sizeof(device), attr, value, sizeof(cuuint64_t));
}

extern "C" cudaError_t
cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  conn_t *conn = lupine_rpc_conn_for_device(&device);
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

// Registration is replicated, but each request uses the generated marshaller.
template <typename Register, typename... Args>
void register_on_each(void **fatCubinHandle, Register call, Args... args) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = fatbins().find(fatCubinHandle);
  if (entry == fatbins().end()) {
    return;
  }
  for (const auto &[conn, handle] : entry->second.handles) {
    record(call(conn, handle, args...));
  }
}

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
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterFatBinaryEnd);
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaUnregisterFatBinary);
  std::lock_guard<std::mutex> lock(registry_mutex());
  if (fatbins().erase(fatCubinHandle) != 0) {
    delete[] fatCubinHandle;
  }
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
  (void)fatCubinHandle;
  return 1;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterFunction, hostFun,
                   deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim,
                   wSize);
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterVar, hostVar,
                   deviceAddress, deviceName, ext, size, constant, global);
}

extern "C" void
__cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                         char *deviceAddress, const char *deviceName, int ext,
                         size_t size, int constant, int global) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterManagedVar,
                   hostVarPtrAddress, deviceAddress, deviceName, ext, size,
                   constant, global);
}

extern "C" void __cudaRegisterTexture(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim, int norm,
                                      int ext) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterTexture, hostVar,
                   deviceAddress, deviceName, dim, norm, ext);
}

extern "C" void __cudaRegisterSurface(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim,
                                      int ext) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterSurface, hostVar,
                   deviceAddress, deviceName, dim, ext);
}

extern "C" void __cudaRegisterHostVar(void **fatCubinHandle,
                                      const char *deviceName, char *hostVar,
                                      size_t size) {
  register_on_each(fatCubinHandle, lupine_rpc___cudaRegisterHostVar, deviceName,
                   hostVar, size);
}

// The private table layout and pointer fixups are not described by the SDK.
// A client address is not a server handle, so this cannot be forwarded.
extern "C" void __cudaRegisterUnifiedTable(void *table) {
  (void)table;
  record(cudaErrorNotSupported);
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
  int route_device = current_device;
  conn_t *conn = launchConfig->stream == nullptr
                     ? lupine_rpc_conn_for_device(&route_device)
                     : lupine_rpc_conn_for_stream(launchConfig->stream);
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
  int route_device = current_device;
  conn_t *conn = stream == nullptr ? lupine_rpc_conn_for_device(&route_device)
                                   : lupine_rpc_conn_for_stream(stream);
  return launch(conn, RPC_cudaLaunchKernel, func, gridDim, blockDim, args,
                sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  int route_device = current_device;
  conn_t *conn = stream == nullptr ? lupine_rpc_conn_for_device(&route_device)
                                   : lupine_rpc_conn_for_stream(stream);
  return launch(conn, RPC_cudaLaunchCooperativeKernel, func, gridDim, blockDim,
                args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                           const void *func, void **args) {
  if (config == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  int route_device = current_device;
  conn_t *conn = config->stream == nullptr
                     ? lupine_rpc_conn_for_device(&route_device)
                     : lupine_rpc_conn_for_stream(config->stream);
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
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                          dim3 blockDim, void **args,
                                          size_t sharedMem,
                                          cudaStream_t stream) {
  int route_device = current_device;
  conn_t *conn = stream == nullptr ? lupine_rpc_conn_for_device(&route_device)
                                   : lupine_rpc_conn_for_stream(stream);
  return launch(conn, RPC___cudaLaunchKernel, kernel, gridDim, blockDim, args,
                sharedMem, stream);
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
  int route_device = current_device;
  conn_t *conn = lupine_rpc_conn_for_device(&route_device);
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
