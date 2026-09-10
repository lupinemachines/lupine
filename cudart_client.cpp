// CUDA runtime API (libcudart) on the LUPINE client.
//
// Most calls are RPCs the server answers with its own libcudart, sent on the
// CUDA driver shim's connections: the same server child owns the driver state
// and the runtime state, and one lane per client thread keeps the two APIs
// ordered. Most of the surface is generated. This file handles the
// image nvcc embedded in the caller, kernel arguments whose sizes only the
// server knows, or a value whose width the attribute decides.
// Complex copies call the driver shim, sharing its routing and staging.

#include <cuda.h>
#include <cuda_runtime_api.h>

// CUDA 12 headers rename this call to _v2. Export the original ABI too.
#ifdef cudaStreamGetCaptureInfo
#undef cudaStreamGetCaptureInfo
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_memcpy.h"
#include "cuda_client_rpc.h"
#include "ipc.h"
#include "lupine_fatbin.h"

namespace {

cudaError_t rpc_error() { return cudaErrorDevicesUnavailable; }

struct conditional_graphs {
  cudaGraph_t parent;
  std::vector<cudaGraph_t> children;
};

std::mutex &conditional_graphs_mutex() {
  static auto *mutex = new std::mutex;
  return *mutex;
}

std::unordered_map<cudaGraphNode_t, conditional_graphs> &
conditional_graph_cache() {
  static auto *cache =
      new std::unordered_map<cudaGraphNode_t, conditional_graphs>;
  return *cache;
}

// Most driver and runtime error values agree. A deinitialized driver is not
// cudaErrorCudartUnloading (the runtime's historical value at the same number).
cudaError_t runtime_error(CUresult error) {
  return error == CUDA_ERROR_DEINITIALIZED ? cudaErrorDeviceUninitialized
                                           : static_cast<cudaError_t>(error);
}

cudaError_t adopt_runtime_host_allocation(conn_t *conn, void **host,
                                          size_t bytes, unsigned int flags);

size_t mem_pool_attribute_width(enum cudaMemPoolAttr attr) {
  switch (attr) {
  case cudaMemPoolReuseFollowEventDependencies:
  case cudaMemPoolReuseAllowOpportunistic:
  case cudaMemPoolReuseAllowInternalDependencies:
#if CUDART_VERSION >= 13020
  case cudaMemPoolAttrLocationId:
  case cudaMemPoolAttrHwDecompressEnabled:
#endif
    return sizeof(int);
#if CUDART_VERSION >= 13020
  case cudaMemPoolAttrAllocationType:
    return sizeof(cudaMemAllocationType);
  case cudaMemPoolAttrExportHandleTypes:
    return sizeof(cudaMemAllocationHandleType);
  case cudaMemPoolAttrLocationType:
    return sizeof(cudaMemLocationType);
#endif
  default:
    return sizeof(cuuint64_t);
  }
}

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  if (lupine_prepare_rpc(conn) < 0) {
    return -1;
  }
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
int rpc_read_end(conn_t *conn) {
  int result = lupine_rpc_read_end(conn);
  if (result >= 0) {
    lupine_invalidate_runtime_context(conn);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------

// The selected server's runtime owns device binding on the caller's lane.
thread_local unsigned int current_connection_index = 0;

conn_t *connection() {
  return lupine_rpc_client_get_connection(current_connection_index);
}

conn_t *connection_for_device(int *device, cudaError_t *result = nullptr);

conn_t *connection_for_stream(cudaStream_t stream) {
  if (stream == nullptr || stream == cudaStreamLegacy ||
      stream == cudaStreamPerThread) {
    return connection();
  }
  return lupine_rpc_conn_for_stream(stream);
}

constexpr auto connection_for_event = lupine_rpc_conn_for_event;

// Registration must reach every configured server before device discovery.
std::vector<conn_t *> all_connections() {
  std::vector<conn_t *> connections;
  if (lupine_rpc_open() < 0) {
    return connections;
  }
  const int count = lupine_rpc_size();
  for (int index = 0; index < count; ++index) {
    conn_t *conn =
        lupine_rpc_client_get_connection(static_cast<unsigned int>(index));
    connections.push_back(conn);
  }
  return connections;
}

// nvcc registers each embedded image once per process; the runtime on every
// server has to see it, so a client handle stands for one server handle per
// connection and every later registration call fans out the same way.
struct fatbin_registration {
  std::vector<std::pair<unsigned int, void **>> handles;
};

// The opaque client handle owns immutable per-server handles. The caller owns
// its lifetime, just as it does the vendor's fatbin handle.
template <typename Call>
void broadcast_fatbin(void **fatCubinHandle, Call call) {
  auto *registration = reinterpret_cast<fatbin_registration *>(fatCubinHandle);
  if (registration == nullptr) {
    return;
  }
  for (const auto &[index, handle] : registration->handles) {
    call(lupine_rpc_client_get_connection(index), handle);
  }
}

void **fatbin_handle(conn_t *conn, void **fatCubinHandle) {
  auto *registration = reinterpret_cast<fatbin_registration *>(fatCubinHandle);
  if (registration != nullptr) {
    for (const auto &[index, handle] : registration->handles) {
      if (lupine_rpc_client_get_connection(index) == conn)
        return handle;
    }
  }
  return nullptr;
}

} // namespace

#include "codegen/gen_cudart_client.inc"

namespace {

conn_t *connection_for_device(int *device, cudaError_t *result) {
  int ordinal = *device;
  cudaError_t status = cudaErrorInvalidDevice;
  if (ordinal >= 0) {
    for (conn_t *conn : all_connections()) {
      int count = 0;
      status = lupine_rpc_cudaGetDeviceCount(conn, &count);
      if (status != cudaSuccess) {
        break;
      }
      if (ordinal < count) {
        *device = ordinal;
        if (result != nullptr) {
          *result = cudaSuccess;
        }
        return conn;
      }
      ordinal -= count;
      status = cudaErrorInvalidDevice;
    }
  }
  if (result != nullptr) {
    *result = status;
  }
  return nullptr;
}

cudaError_t adopt_runtime_host_allocation(conn_t *conn, void **host,
                                          size_t bytes, unsigned int flags) {
  void *remote_host = *host;
  void *device_ptr = nullptr;
  cudaError_t result =
      lupine_rpc_cudaHostGetDevicePointer(conn, &device_ptr, remote_host, 0);
  if (result == cudaSuccess) {
    result = runtime_error(lupine_adopt_host_allocation(
        conn, host, remote_host, reinterpret_cast<CUdeviceptr>(device_ptr),
        bytes, flags));
  }
  if (result != cudaSuccess) {
    lupine_rpc_cudaFreeHost(conn, remote_host);
    *host = nullptr;
  }
  return result;
}

} // namespace

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t bytes,
                                        unsigned int flags) {
  return runtime_error(lupine_register_host_allocation(
      connection(), ptr, bytes, flags,
      [](conn_t *conn, void *base, size_t size, unsigned int flags,
         void **server_host, CUdeviceptr *device_ptr) {
        cudaError_t result = rpc_error();
        if (rpc_write_start_request(conn, RPC_cudaHostRegister) < 0 ||
            rpc_write(conn, &base, sizeof(base)) < 0 ||
            rpc_write(conn, &size, sizeof(size)) < 0 ||
            rpc_write(conn, &flags, sizeof(flags)) < 0 ||
            rpc_wait_for_response(conn) < 0 ||
            rpc_read(conn, server_host, sizeof(*server_host)) < 0 ||
            rpc_read(conn, device_ptr, sizeof(*device_ptr)) < 0 ||
            rpc_read(conn, &result, sizeof(result)) < 0 ||
            rpc_read_end(conn) < 0) {
          return CUDA_ERROR_DEVICE_UNAVAILABLE;
        }
        return static_cast<CUresult>(result);
      },
      [](conn_t *conn, void *host) {
        return static_cast<CUresult>(lupine_rpc_cudaHostUnregister(conn, host));
      }));
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

extern "C" cudaError_t
cudaMemPoolExportToShareableHandle(void *handle_out, cudaMemPool_t pool,
                                   cudaMemAllocationHandleType handle_type,
                                   unsigned int flags) {
  if (handle_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  lupine_ipc_token token{};
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaMemPoolExportToShareableHandle) <
          0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
      rpc_write(conn, &handle_type, sizeof(handle_type)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == cudaSuccess &&
      lupine_ipc_create_proxy_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                 static_cast<int *>(handle_out)) < 0) {
    return cudaErrorUnknown;
  }
  return result;
}

extern "C" cudaError_t
cudaMemPoolImportFromShareableHandle(cudaMemPool_t *pool, void *handle,
                                     cudaMemAllocationHandleType handle_type,
                                     unsigned int flags) {
  if (pool == nullptr) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  lupine_ipc_token token{};
  uint32_t kind = 0;
  int fd = static_cast<int>(reinterpret_cast<uintptr_t>(handle));
  if (lupine_ipc_read_proxy_fd(fd, &kind, &token) < 0 ||
      kind != LUPINE_IPC_FD_KIND_MEMORY_POOL) {
    return cudaErrorInvalidValue;
  }
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaMemPoolImportFromShareableHandle) <
          0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &handle_type, sizeof(handle_type)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pool, sizeof(*pool)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  if (count == nullptr) {
    return cudaErrorInvalidValue;
  }
  const auto connections = all_connections();
  if (connections.empty()) {
    return rpc_error();
  }
  int total = 0;
  for (conn_t *conn : connections) {
    int remote_count = 0;
    cudaError_t result = lupine_rpc_cudaGetDeviceCount(conn, &remote_count);
    if (result != cudaSuccess) {
      return result;
    }
    total += remote_count;
  }
  *count = total;
  return cudaSuccess;
}

extern "C" cudaError_t cudaGetDevice(int *device) {
  if (device == nullptr) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  int remote_device = 0;
  cudaError_t result = lupine_rpc_cudaGetDevice(conn, &remote_device);
  if (result != cudaSuccess) {
    return result;
  }
  for (unsigned int index = 0; index < current_connection_index; ++index) {
    int count = 0;
    result = lupine_rpc_cudaGetDeviceCount(
        lupine_rpc_client_get_connection(index), &count);
    if (result != cudaSuccess) {
      return result;
    }
    remote_device += count;
  }
  *device = remote_device;
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  int remote_device = device;
  cudaError_t result = cudaSuccess;
  conn_t *conn = connection_for_device(&remote_device, &result);
  if (conn == nullptr) {
    return result;
  }
  result = lupine_rpc_cudaSetDevice(conn, remote_device);
  if (result != cudaSuccess) {
    return result;
  }
  const int count = lupine_rpc_size();
  for (int index = 0; index < count; ++index) {
    if (lupine_rpc_client_get_connection(static_cast<unsigned int>(index)) ==
        conn) {
      current_connection_index = static_cast<unsigned int>(index);
      break;
    }
  }
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
  if (len < 0 || (len != 0 && device_arr == nullptr)) {
    return cudaErrorInvalidValue;
  }
  std::vector<int> remote_devices(static_cast<size_t>(len));
  conn_t *conn = nullptr;
  for (int i = 0; i < len; ++i) {
    remote_devices[i] = device_arr[i];
    conn_t *device_conn = lupine_rpc_conn_for_device(&remote_devices[i]);
    if (device_conn == nullptr) {
      return cudaErrorInvalidDevice;
    }
    // The list is a single preference order, so it can only be forwarded to
    // one server; devices behind another connection are not expressible.
    if (conn != nullptr && device_conn != conn) {
      return cudaErrorNotSupported;
    }
    conn = device_conn;
  }
  if (conn == nullptr) {
    conn = connection();
  }
  return lupine_rpc_cudaSetValidDevices(conn, remote_devices.data(), len);
}

extern "C" cudaError_t cudaDeviceReset() {
  conn_t *conn = connection();
  return lupine_rpc_cudaDeviceReset(conn);
}

namespace {

cudaError_t read_completion(conn_t *conn, bool with_stdout) {
  cudaError_t result = rpc_error();
  if (rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      (with_stdout && lupine_forward_remote_stdout(conn) < 0) ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result == cudaSuccess
             ? runtime_error(lupine_sync_mapped_device_to_host())
             : result;
}

} // namespace

extern "C" cudaError_t cudaDeviceSynchronize() {
  conn_t *conn = connection();
  if (rpc_write_start_request(conn, RPC_cudaDeviceSynchronize) < 0) {
    return rpc_error();
  }
  return read_completion(conn, true);
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
  if (rpc_write_start_request(conn, RPC_cudaStreamSynchronize) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0) {
    return rpc_error();
  }
  return read_completion(conn, true);
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
  if (rpc_write_start_request(conn, RPC_cudaStreamQuery) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0) {
    return rpc_error();
  }
  return read_completion(conn, false);
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  conn_t *conn = connection_for_event(event);
  if (rpc_write_start_request(conn, RPC_cudaEventSynchronize) < 0 ||
      rpc_write(conn, &event, sizeof(event)) < 0) {
    return rpc_error();
  }
  return read_completion(conn, true);
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  std::shared_lock<std::shared_mutex> lock(lupine_event_lifecycle_mutex());
  conn_t *conn = connection_for_event(event);
  if (rpc_write_start_request(conn, RPC_cudaEventQuery) < 0 ||
      rpc_write(conn, &event, sizeof(event)) < 0) {
    return rpc_error();
  }
  return read_completion(conn, false);
}

namespace {

// The server's runtime owns the message strings, so they are fetched once and
// held per thread: the returned pointer stays valid until this thread's next
// lookup of the same kind.
const char *error_text(int op, cudaError_t error, bool want_name) {
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

#if CUDART_VERSION >= 12000
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,
                                            cudaGraph_t graph,
                                            unsigned long long flags) {
  if (exec == nullptr) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaGraphInstantiate) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, exec, sizeof(*exec)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}
#else
extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec,
                                            cudaGraph_t graph,
                                            cudaGraphNode_t *error_node,
                                            char *log, size_t log_size) {
  conn_t *conn = connection();
  cudaError_t result = rpc_error();
  bool has_error_node = error_node != nullptr;
  bool has_log = log != nullptr;
  bool log_has_data = false;
  if (exec == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rpc_write_start_request(conn, RPC_cudaGraphInstantiate) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &has_error_node, sizeof(has_error_node)) < 0 ||
      rpc_write(conn, &log_size, sizeof(log_size)) < 0 ||
      rpc_write(conn, &has_log, sizeof(has_log)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, exec, sizeof(*exec)) < 0 ||
      (has_error_node && rpc_read(conn, error_node, sizeof(*error_node)) < 0) ||
      rpc_read(conn, &log_has_data, sizeof(log_has_data)) < 0 ||
      (log_has_data && rpc_read(conn, log, log_size) < 0) ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t
cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph,
                    cudaGraphNode_t *error_node,
                    cudaGraphExecUpdateResult *update_result) {
  conn_t *conn = connection();
  cudaError_t result = rpc_error();
  bool has_error_node = error_node != nullptr;
  bool has_update_result = update_result != nullptr;
  if (rpc_write_start_request(conn, RPC_cudaGraphExecUpdate) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &has_error_node, sizeof(has_error_node)) < 0 ||
      rpc_write(conn, &has_update_result, sizeof(has_update_result)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (has_error_node && rpc_read(conn, error_node, sizeof(*error_node)) < 0) ||
      (has_update_result &&
       rpc_read(conn, update_result, sizeof(*update_result)) < 0) ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}
#endif

#if CUDART_VERSION < 13000
extern "C" cudaError_t
cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                    cudaGraphNode_t *dependencies, size_t count,
                                    unsigned int flags) {
  if (count != 0 && dependencies == nullptr) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection_for_stream(stream);
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaStreamUpdateCaptureDependencies) <
          0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, dependencies, count * sizeof(cudaGraphNode_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                                     cudaMemoryAdvise advice, int device) {
  conn_t *conn = connection();
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaMemAdvise) < 0 ||
      rpc_write(conn, &devPtr, sizeof(devPtr)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, &advice, sizeof(advice)) < 0 ||
      rpc_write(conn, &device, sizeof(device)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                            int dstDevice,
                                            cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_cudaMemPrefetchAsync) < 0 ||
      rpc_write(conn, &devPtr, sizeof(devPtr)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}
#endif

#if CUDART_VERSION >= 12000
extern "C" cudaError_t cudaStreamBeginCaptureToGraph(
    cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *edges, size_t count, cudaStreamCaptureMode mode) {
  if (count != 0 && dependencies == nullptr) {
    return cudaErrorInvalidValue;
  }
  // NVIDIA defines omitted edge data as a zero-initialized array.
  std::vector<cudaGraphEdgeData> defaults(edges == nullptr ? count : 0);
  return lupine_rpc_cudaStreamBeginCaptureToGraph(
      connection_for_stream(stream), stream, graph, dependencies,
      edges == nullptr ? defaults.data() : edges, count, mode);
}

#if CUDART_VERSION >= 13000
extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(
    cudaStream_t stream, cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *edges, size_t count, unsigned int flags) {
  if (count != 0 && dependencies == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::vector<cudaGraphEdgeData> defaults(edges == nullptr ? count : 0);
  return lupine_rpc_cudaStreamUpdateCaptureDependencies(
      connection_for_stream(stream), stream, dependencies,
      edges == nullptr ? defaults.data() : edges, count, flags);
}
#else
extern "C" cudaError_t cudaStreamUpdateCaptureDependencies_v2(
    cudaStream_t stream, cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *edges, size_t count, unsigned int flags) {
  if (count != 0 && dependencies == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::vector<cudaGraphEdgeData> defaults(edges == nullptr ? count : 0);
  return lupine_rpc_cudaStreamUpdateCaptureDependencies_v2(
      connection_for_stream(stream), stream, dependencies,
      edges == nullptr ? defaults.data() : edges, count, flags);
}
#endif
#endif

// ---------------------------------------------------------------------------
// Copies
// ---------------------------------------------------------------------------

namespace {

cudaError_t capture_info(int op, cudaStream_t stream,
                         cudaStreamCaptureStatus *status_out,
                         unsigned long long *id_out, cudaGraph_t *graph_out,
                         const cudaGraphNode_t **dependencies_out,
                         const CUgraphEdgeData **edges_out, size_t *count_out) {
  if (status_out == nullptr) {
    return cudaErrorInvalidValue;
  }
  static thread_local std::vector<cudaGraphNode_t> dependencies;
  static thread_local std::vector<CUgraphEdgeData> edges;
  conn_t *conn = connection_for_stream(stream);
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  unsigned long long id = 0;
  cudaGraph_t graph = nullptr;
  size_t count = 0;
  bool has_edges = false;
  cudaError_t result = rpc_error();
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &id, sizeof(id)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      rpc_read(conn, &has_edges, sizeof(has_edges)) < 0) {
    return rpc_error();
  }
  dependencies.resize(count);
  edges.resize(has_edges ? count : 0);
  if (rpc_read(conn, dependencies.data(), count * sizeof(cudaGraphNode_t)) <
          0 ||
      rpc_read(conn, edges.data(), edges.size() * sizeof(CUgraphEdgeData)) <
          0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == cudaSuccess) {
    *status_out = status;
    if (id_out != nullptr)
      *id_out = id;
    if (graph_out != nullptr)
      *graph_out = graph;
    if (count_out != nullptr)
      *count_out = count;
    if (dependencies_out != nullptr) {
      *dependencies_out = dependencies.empty() ? nullptr : dependencies.data();
    }
    if (edges_out != nullptr) {
      *edges_out = edges.empty() ? nullptr : edges.data();
    }
  }
  return result;
}

} // namespace

#if CUDART_VERSION < 13000
extern "C" cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                cudaStreamCaptureStatus *status,
                                                unsigned long long *id) {
  return capture_info(RPC_cudaStreamGetCaptureInfo, stream, status, id, nullptr,
                      nullptr, nullptr, nullptr);
}
#else
extern "C" cudaError_t
cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus *status,
                         unsigned long long *id, cudaGraph_t *graph,
                         const cudaGraphNode_t **dependencies,
                         const cudaGraphEdgeData **edges, size_t *count) {
  const CUgraphEdgeData *edge_data = nullptr;
  cudaError_t result =
      capture_info(RPC_cudaStreamGetCaptureInfo, stream, status, id, graph,
                   dependencies, &edge_data, count);
  if (result == cudaSuccess && edges != nullptr) {
    *edges = reinterpret_cast<const cudaGraphEdgeData *>(edge_data);
  }
  return result;
}
#endif

#if CUDART_VERSION < 13000
extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream, cudaStreamCaptureStatus *status,
    unsigned long long *id, cudaGraph_t *graph,
    const cudaGraphNode_t **dependencies, size_t *count) {
  return capture_info(RPC_cudaStreamGetCaptureInfo_v2, stream, status, id,
                      graph, dependencies, nullptr, count);
}
#if CUDART_VERSION >= 12000
extern "C" cudaError_t
cudaStreamGetCaptureInfo_v3(cudaStream_t stream,
                            cudaStreamCaptureStatus *status,
                            unsigned long long *id, cudaGraph_t *graph,
                            const cudaGraphNode_t **dependencies,
                            const cudaGraphEdgeData **edges, size_t *count) {
  const CUgraphEdgeData *edge_data = nullptr;
  cudaError_t result =
      capture_info(RPC_cudaStreamGetCaptureInfo_v3, stream, status, id, graph,
                   dependencies, &edge_data, count);
  if (result == cudaSuccess && edges != nullptr) {
    *edges = reinterpret_cast<const cudaGraphEdgeData *>(edge_data);
  }
  return result;
}
#endif
#endif

namespace {

cudaError_t copy_memory_types(cudaMemcpyKind kind, CUmemorytype *source,
                              CUmemorytype *destination) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return cudaErrorInvalidMemcpyDirection;
  }
  constexpr CUmemorytype types[][2] = {
      {CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_HOST},
      {CU_MEMORYTYPE_HOST, CU_MEMORYTYPE_DEVICE},
      {CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST},
      {CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_DEVICE},
      {CU_MEMORYTYPE_UNIFIED, CU_MEMORYTYPE_UNIFIED}};
  *source = types[kind][0];
  *destination = types[kind][1];
  return cudaSuccess;
}

cudaError_t copy_descriptor(const cudaMemcpy3DParms *params,
                            CUDA_MEMCPY3D *copy) {
  if (params == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t result = copy_memory_types(params->kind, &copy->srcMemoryType,
                                         &copy->dstMemoryType);
  if (result != cudaSuccess) {
    return result;
  }
  if ((params->srcArray != nullptr && params->srcPtr.ptr != nullptr) ||
      (params->dstArray != nullptr && params->dstPtr.ptr != nullptr)) {
    return cudaErrorInvalidValue;
  }
  size_t element_bytes = 0;
  for (cudaArray_t array : {params->srcArray, params->dstArray}) {
    if (array == nullptr) {
      continue;
    }
    cudaChannelFormatDesc format{};
    result = cudaGetChannelDesc(&format, array);
    if (result != cudaSuccess) {
      return result;
    }
    size_t bytes = (format.x + format.y + format.z + format.w) / 8;
    if (bytes == 0 || (element_bytes != 0 && bytes != element_bytes)) {
      return cudaErrorInvalidValue;
    }
    element_bytes = bytes;
  }
  if (element_bytes == 0) {
    element_bytes = 1;
  }
  copy->srcHost = params->srcPtr.ptr;
  copy->srcDevice = reinterpret_cast<CUdeviceptr>(params->srcPtr.ptr);
  copy->srcPitch = params->srcPtr.pitch;
  copy->srcHeight = params->srcPtr.ysize;
  copy->srcXInBytes = params->srcPos.x;
  copy->srcY = params->srcPos.y;
  copy->srcZ = params->srcPos.z;
  if (params->srcArray != nullptr) {
    copy->srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy->srcArray = reinterpret_cast<CUarray>(params->srcArray);
    copy->srcXInBytes *= element_bytes;
  }
  copy->dstHost = params->dstPtr.ptr;
  copy->dstDevice = reinterpret_cast<CUdeviceptr>(params->dstPtr.ptr);
  copy->dstPitch = params->dstPtr.pitch;
  copy->dstHeight = params->dstPtr.ysize;
  copy->dstXInBytes = params->dstPos.x;
  copy->dstY = params->dstPos.y;
  copy->dstZ = params->dstPos.z;
  if (params->dstArray != nullptr) {
    copy->dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy->dstArray = reinterpret_cast<CUarray>(params->dstArray);
    copy->dstXInBytes *= element_bytes;
  }
  copy->WidthInBytes = params->extent.width * element_bytes;
  copy->Height = params->extent.height;
  copy->Depth = params->extent.depth;
  return cudaSuccess;
}

cudaError_t copy_2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                    size_t width, size_t height, cudaMemcpyKind kind,
                    const cudaStream_t *stream) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return cudaErrorInvalidMemcpyDirection;
  }
  if (width > dpitch || width > spitch) {
    return cudaErrorInvalidPitchValue;
  }
  if (width == 0 || height == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return cudaErrorInvalidValue;
  }
  // A depth-one volume accepts arbitrary device pitches; cuMemcpy2D can
  // reject pitches that did not come from cuMemAllocPitch.
  CUDA_MEMCPY3D copy = {};
  copy_memory_types(kind, &copy.srcMemoryType, &copy.dstMemoryType);
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
  return runtime_error(stream == nullptr ? cuMemcpy3D(&copy)
                                         : cuMemcpy3DAsync(&copy, *stream));
}

// The runtime resolves nvcc's host symbol key. Only the copy itself is mapped
// to the driver; retain the symbol bounds and owning route for interior
// pointers.
cudaError_t symbol_address(conn_t *conn, void **address, const void *symbol,
                           size_t offset, size_t count) {
  if (address == nullptr) {
    return cudaErrorInvalidValue;
  }
  void *base = nullptr;
  size_t size = 0;
  cudaError_t result = lupine_rpc_cudaGetSymbolAddress(conn, &base, symbol);
  if (result == cudaSuccess) {
    result = lupine_rpc_cudaGetSymbolSize(conn, &size, symbol);
  }
  if (result != cudaSuccess) {
    return result;
  }
  if (offset > size || count > size - offset) {
    return cudaErrorInvalidValue;
  }
  lupine_note_deviceptr_allocation(reinterpret_cast<CUdeviceptr>(base), size,
                                   conn);
  *address =
      reinterpret_cast<void *>(reinterpret_cast<CUdeviceptr>(base) + offset);
  return cudaSuccess;
}

cudaError_t copy_peer(void *dst, int dstDevice, const void *src, int srcDevice,
                      size_t count, const cudaStream_t *stream) {
  CUcontext dstContext = nullptr, srcContext = nullptr;
  CUresult result = cuDevicePrimaryCtxRetain(&dstContext, dstDevice);
  if (result != CUDA_SUCCESS) {
    return runtime_error(result);
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
  return runtime_error(result);
}

} // namespace

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  cudaMemcpyKind kind) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return cudaErrorInvalidMemcpyDirection;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return cudaErrorInvalidValue;
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    std::memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return runtime_error(
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count));
  case cudaMemcpyDeviceToHost:
    return runtime_error(
        cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count));
  case cudaMemcpyDeviceToDevice:
    return runtime_error(cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),
                                      reinterpret_cast<CUdeviceptr>(src),
                                      count));
  case cudaMemcpyDefault:
    return runtime_error(cuMemcpy(reinterpret_cast<CUdeviceptr>(dst),
                                  reinterpret_cast<CUdeviceptr>(src), count));
  }
  return cudaErrorInvalidMemcpyDirection;
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  if (kind < cudaMemcpyHostToHost || kind > cudaMemcpyDefault) {
    return cudaErrorInvalidMemcpyDirection;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (dst == nullptr || src == nullptr) {
    return cudaErrorInvalidValue;
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    std::memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return runtime_error(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                           src, count, stream));
  case cudaMemcpyDeviceToHost:
    return runtime_error(cuMemcpyDtoHAsync(
        dst, reinterpret_cast<CUdeviceptr>(src), count, stream));
  case cudaMemcpyDeviceToDevice:
    return runtime_error(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst),
                                           reinterpret_cast<CUdeviceptr>(src),
                                           count, stream));
  case cudaMemcpyDefault:
    return runtime_error(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                       reinterpret_cast<CUdeviceptr>(src),
                                       count, stream));
  }
  return cudaErrorInvalidMemcpyDirection;
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

extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *params) {
  CUDA_MEMCPY3D copy{};
  cudaError_t result = copy_descriptor(params, &copy);
  return result == cudaSuccess ? runtime_error(cuMemcpy3D(&copy)) : result;
}

extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *params,
                                         cudaStream_t stream) {
  CUDA_MEMCPY3D copy{};
  cudaError_t result = copy_descriptor(params, &copy);
  return result == cudaSuccess ? runtime_error(cuMemcpy3DAsync(&copy, stream))
                               : result;
}

extern "C" cudaError_t
cudaGraphAddMemcpyNode(cudaGraphNode_t *node, cudaGraph_t graph,
                       const cudaGraphNode_t *dependencies,
                       size_t dependency_count,
                       const cudaMemcpy3DParms *params) {
  CUDA_MEMCPY3D copy{};
  cudaError_t result = copy_descriptor(params, &copy);
  if (result != cudaSuccess) {
    return result;
  }
  conn_t *conn = connection();
  CUcontext context = nullptr;
  CUresult status = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_write_start_request(conn, RPC_cuCtxGetCurrent) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (status != CUDA_SUCCESS) {
    return runtime_error(status);
  }
  return runtime_error(cuGraphAddMemcpyNode(node, graph, dependencies,
                                            dependency_count, &copy, context));
}

extern "C" cudaError_t
cudaGraphAddMemcpyNode1D(cudaGraphNode_t *node, cudaGraph_t graph,
                         const cudaGraphNode_t *dependencies,
                         size_t dependency_count, void *dst, const void *src,
                         size_t count, cudaMemcpyKind kind) {
  cudaMemcpy3DParms params{};
  params.srcPtr = {const_cast<void *>(src), count, count, 1};
  params.dstPtr = {dst, count, count, 1};
  params.extent = {count, 1, 1};
  params.kind = kind;
  return cudaGraphAddMemcpyNode(node, graph, dependencies, dependency_count,
                                &params);
}

namespace {

cudaError_t copy_array_2d(CUDA_MEMCPY3D *copy, size_t width, size_t height,
                          cudaMemcpyKind kind, const cudaStream_t *stream) {
  cudaError_t result =
      copy_memory_types(kind, &copy->srcMemoryType, &copy->dstMemoryType);
  if (result != cudaSuccess) {
    return result;
  }
  if (copy->srcArray != nullptr) {
    copy->srcMemoryType = CU_MEMORYTYPE_ARRAY;
  }
  if (copy->dstArray != nullptr) {
    copy->dstMemoryType = CU_MEMORYTYPE_ARRAY;
  }
  copy->WidthInBytes = width;
  copy->Height = height;
  copy->Depth = 1;
  copy->srcHeight = copy->dstHeight = height;
  return runtime_error(stream == nullptr ? cuMemcpy3D(copy)
                                         : cuMemcpy3DAsync(copy, *stream));
}

cudaError_t copy_array_linear(CUDA_MEMCPY3D copy, size_t bytes,
                              cudaMemcpyKind kind, const cudaStream_t *stream) {
  cudaArray_t array = reinterpret_cast<cudaArray_t>(
      copy.srcArray != nullptr ? copy.srcArray : copy.dstArray);
  cudaChannelFormatDesc format{};
  cudaExtent extent{};
  unsigned int flags = 0;
  cudaError_t result = cudaArrayGetInfo(&format, &extent, &flags, array);
  if (result != cudaSuccess) {
    return result;
  }
  const size_t row_bytes =
      extent.width * ((format.x + format.y + format.z + format.w) / 8);
  size_t &x = copy.srcArray != nullptr ? copy.srcXInBytes : copy.dstXInBytes;
  size_t &y = copy.srcArray != nullptr ? copy.srcY : copy.dstY;
  if (row_bytes == 0 || x >= row_bytes) {
    return cudaErrorInvalidValue;
  }
  while (bytes != 0) {
    size_t width = std::min(bytes, row_bytes - x);
    size_t rows = x == 0 && bytes >= row_bytes ? bytes / row_bytes : 1;
    copy.srcPitch = copy.dstPitch = width;
    result = copy_array_2d(&copy, width, rows, kind, stream);
    if (result != cudaSuccess) {
      return result;
    }
    size_t copied = width * rows;
    bytes -= copied;
    if (copy.srcArray != nullptr) {
      copy.dstHost = static_cast<unsigned char *>(copy.dstHost) + copied;
      copy.dstDevice += copied;
    } else {
      copy.srcHost = static_cast<const unsigned char *>(copy.srcHost) + copied;
      copy.srcDevice += copied;
    }
    x = 0;
    y += rows;
  }
  return cudaSuccess;
}

} // namespace

extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset,
                                           size_t hOffset, const void *src,
                                           size_t spitch, size_t width,
                                           size_t height, cudaMemcpyKind kind) {
  CUDA_MEMCPY3D copy{};
  copy.dstArray = reinterpret_cast<CUarray>(dst);
  copy.dstXInBytes = wOffset;
  copy.dstY = hOffset;
  copy.srcHost = src;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(src);
  copy.srcPitch = spitch;
  return copy_array_2d(&copy, width, height, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                                size_t hOffset, const void *src,
                                                size_t spitch, size_t width,
                                                size_t height,
                                                cudaMemcpyKind kind,
                                                cudaStream_t stream) {
  CUDA_MEMCPY3D copy{};
  copy.dstArray = reinterpret_cast<CUarray>(dst);
  copy.dstXInBytes = wOffset;
  copy.dstY = hOffset;
  copy.srcHost = src;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(src);
  copy.srcPitch = spitch;
  return copy_array_2d(&copy, width, height, kind, &stream);
}

extern "C" cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                             cudaArray_const_t src,
                                             size_t wOffset, size_t hOffset,
                                             size_t width, size_t height,
                                             cudaMemcpyKind kind) {
  CUDA_MEMCPY3D copy{};
  copy.srcArray = reinterpret_cast<CUarray>(const_cast<cudaArray_t>(src));
  copy.srcXInBytes = wOffset;
  copy.srcY = hOffset;
  copy.dstHost = dst;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  copy.dstPitch = dpitch;
  return copy_array_2d(&copy, width, height, kind, nullptr);
}

extern "C" cudaError_t
cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src,
                           size_t wOffset, size_t hOffset, size_t width,
                           size_t height, cudaMemcpyKind kind,
                           cudaStream_t stream) {
  CUDA_MEMCPY3D copy{};
  copy.srcArray = reinterpret_cast<CUarray>(const_cast<cudaArray_t>(src));
  copy.srcXInBytes = wOffset;
  copy.srcY = hOffset;
  copy.dstHost = dst;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  copy.dstPitch = dpitch;
  return copy_array_2d(&copy, width, height, kind, &stream);
}

extern "C" cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset,
                                         size_t hOffset, const void *src,
                                         size_t count, cudaMemcpyKind kind) {
  CUDA_MEMCPY3D copy{};
  copy.dstArray = reinterpret_cast<CUarray>(dst);
  copy.dstXInBytes = wOffset;
  copy.dstY = hOffset;
  copy.srcHost = src;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(src);
  return copy_array_linear(copy, count, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src,
                                           size_t wOffset, size_t hOffset,
                                           size_t count, cudaMemcpyKind kind) {
  CUDA_MEMCPY3D copy{};
  copy.srcArray = reinterpret_cast<CUarray>(const_cast<cudaArray_t>(src));
  copy.srcXInBytes = wOffset;
  copy.srcY = hOffset;
  copy.dstHost = dst;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  return copy_array_linear(copy, count, kind, nullptr);
}

extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset,
                                              size_t hOffset, const void *src,
                                              size_t count, cudaMemcpyKind kind,
                                              cudaStream_t stream) {
  CUDA_MEMCPY3D copy{};
  copy.dstArray = reinterpret_cast<CUarray>(dst);
  copy.dstXInBytes = wOffset;
  copy.dstY = hOffset;
  copy.srcHost = src;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(src);
  return copy_array_linear(copy, count, kind, &stream);
}

extern "C" cudaError_t
cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset,
                         size_t hOffset, size_t count, cudaMemcpyKind kind,
                         cudaStream_t stream) {
  CUDA_MEMCPY3D copy{};
  copy.srcArray = reinterpret_cast<CUarray>(const_cast<cudaArray_t>(src));
  copy.srcXInBytes = wOffset;
  copy.srcY = hOffset;
  copy.dstHost = dst;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(dst);
  return copy_array_linear(copy, count, kind, &stream);
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  conn_t *conn = connection();
  return symbol_address(conn, devPtr, symbol, 0, 0);
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                          size_t count, size_t offset,
                                          cudaMemcpyKind kind) {
  if (kind != cudaMemcpyHostToDevice && kind != cudaMemcpyDeviceToDevice &&
      kind != cudaMemcpyDefault) {
    return cudaErrorInvalidMemcpyDirection;
  }
  conn_t *conn = connection();
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
    return cudaErrorInvalidMemcpyDirection;
  }
  conn_t *conn = connection();
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
    return cudaErrorInvalidMemcpyDirection;
  }
  conn_t *conn = connection_for_stream(stream);
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
    return cudaErrorInvalidMemcpyDirection;
  }
  conn_t *conn = connection_for_stream(stream);
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

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

namespace {

// The fatbin proper: an outer header followed by its member entries.
size_t fatbin_size(const void *image) {
  const auto *header = static_cast<const lupine_fatbin_header *>(image);
  if (header == nullptr || header->magic != LUPINE_FATBIN_MAGIC) {
    return 0;
  }
  return header->header_size + header->files_size;
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
    return nullptr;
  }
  fatbin_registration registration;
  const auto connections = all_connections();
  for (unsigned int index = 0; index < connections.size(); ++index) {
    conn_t *conn = connections[index];
    void **handle = nullptr;
    if (rpc_write_start_request(conn, RPC___cudaRegisterFatBinary) < 0 ||
        rpc_write(conn, &version, sizeof(version)) < 0 ||
        rpc_write(conn, &image_size, sizeof(image_size)) < 0 ||
        rpc_write(conn, image, image_size) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &handle, sizeof(handle)) < 0 || rpc_read_end(conn) < 0 ||
        handle == nullptr) {
      continue;
    }
    registration.handles.emplace_back(index, handle);
  }
  if (registration.handles.empty()) {
    return nullptr;
  }
  return reinterpret_cast<void **>(
      new fatbin_registration(std::move(registration)));
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize) {
  const size_t deviceFun_len = std::strlen(deviceFun) + 1;
  const size_t deviceName_len = std::strlen(deviceName) + 1;
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    if (rpc_write_start_request(conn, RPC___cudaRegisterFunction) < 0 ||
        rpc_write(conn, &handle, sizeof(handle)) < 0 ||
        rpc_write(conn, &hostFun, sizeof(hostFun)) < 0 ||
        rpc_write(conn, &deviceFun_len, sizeof(deviceFun_len)) < 0 ||
        rpc_write(conn, deviceFun, deviceFun_len) < 0 ||
        rpc_write(conn, &deviceName_len, sizeof(deviceName_len)) < 0 ||
        rpc_write(conn, deviceName, deviceName_len) < 0 ||
        rpc_write(conn, &thread_limit, sizeof(thread_limit)) < 0 ||
        rpc_write(conn, &tid, sizeof(tid)) < 0 ||
        (tid != nullptr && rpc_write(conn, tid, sizeof(*tid)) < 0) ||
        rpc_write(conn, &bid, sizeof(bid)) < 0 ||
        (bid != nullptr && rpc_write(conn, bid, sizeof(*bid)) < 0) ||
        rpc_write(conn, &bDim, sizeof(bDim)) < 0 ||
        (bDim != nullptr && rpc_write(conn, bDim, sizeof(*bDim)) < 0) ||
        rpc_write(conn, &gDim, sizeof(gDim)) < 0 ||
        (gDim != nullptr && rpc_write(conn, gDim, sizeof(*gDim)) < 0) ||
        rpc_write(conn, &wSize, sizeof(wSize)) < 0 ||
        (wSize != nullptr && rpc_write(conn, wSize, sizeof(*wSize)) < 0) ||
        rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0) {
      return;
    }
  });
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  const size_t address_length = strlen(deviceAddress) + 1;
  const size_t name_length = strlen(deviceName) + 1;
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    if (rpc_write_start_request(conn, RPC___cudaRegisterVar) < 0 ||
        rpc_write(conn, &handle, sizeof(handle)) < 0 ||
        rpc_write(conn, &hostVar, sizeof(hostVar)) < 0 ||
        rpc_write(conn, &address_length, sizeof(address_length)) < 0 ||
        rpc_write(conn, deviceAddress, address_length) < 0 ||
        rpc_write(conn, &name_length, sizeof(name_length)) < 0 ||
        rpc_write(conn, deviceName, name_length) < 0 ||
        rpc_write(conn, &ext, sizeof(ext)) < 0 ||
        rpc_write(conn, &size, sizeof(size)) < 0 ||
        rpc_write(conn, &constant, sizeof(constant)) < 0 ||
        rpc_write(conn, &global, sizeof(global)) < 0 ||
        rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0) {
      return;
    }
  });
}

extern "C" void
__cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                         char *deviceAddress, const char *deviceName, int ext,
                         size_t size, int constant, int global) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterManagedVar(conn, handle, hostVarPtrAddress,
                                        deviceAddress, deviceName, ext, size,
                                        constant, global);
  });
}

extern "C" void __cudaRegisterTexture(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim, int norm,
                                      int ext) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterTexture(conn, handle, hostVar, deviceAddress,
                                     deviceName, dim, norm, ext);
  });
}

extern "C" void __cudaRegisterSurface(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim,
                                      int ext) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterSurface(conn, handle, hostVar, deviceAddress,
                                     deviceName, dim, ext);
  });
}

extern "C" void __cudaRegisterHostVar(void **fatCubinHandle,
                                      const char *deviceName, char *hostVar,
                                      size_t size) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterHostVar(conn, handle, deviceName, hostVar, size);
  });
}

extern "C" void __cudaRegisterUnifiedTable(void **fatCubinHandle,
                                           void *functionTable,
                                           size_t functionWindowSize,
                                           void *dataTable,
                                           size_t dataWindowSize) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterUnifiedTable(conn, handle, functionTable,
                                          functionWindowSize, dataTable,
                                          dataWindowSize);
  });
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    lupine_rpc___cudaRegisterFatBinaryEnd(conn, handle);
  });
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  broadcast_fatbin(fatCubinHandle, [&](conn_t *conn, void **handle) {
    if (rpc_write_start_request(conn, RPC___cudaUnregisterFatBinary) < 0 ||
        rpc_write(conn, &handle, sizeof(handle)) < 0 ||
        rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0) {
      return;
    }
  });
  delete reinterpret_cast<fatbin_registration *>(fatCubinHandle);
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
    return rpc_error();
  }
  sizes->assign(count, 0);
  if ((count != 0 &&
       rpc_read(conn, sizes->data(), count * sizeof(size_t)) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (return_value == cudaSuccess) {
    std::lock_guard<std::mutex> lock(layouts_mutex());
    layouts()[func] = *sizes;
  }
  return return_value;
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

cudaError_t graph_kernel_param_sizes(conn_t *conn,
                                     const cudaKernelNodeParams *params,
                                     std::vector<size_t> *sizes) {
  if (params == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (params->extra != nullptr) {
    return cudaErrorNotSupported;
  }
  cudaError_t result = param_sizes(conn, params->func, sizes);
  if (result != cudaSuccess) {
    return result;
  }
  if (!sizes->empty() && params->kernelParams == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (size_t index = 0; index < sizes->size(); ++index) {
    if (params->kernelParams[index] == nullptr) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t launch(conn_t *conn, int op, const void *func, dim3 gridDim,
                   dim3 blockDim, void **args, size_t sharedMem,
                   cudaStream_t stream) {
  std::vector<size_t> sizes;
  cudaError_t resolved = param_sizes(conn, func, &sizes);
  if (resolved != cudaSuccess) {
    return resolved;
  }
  if (!sizes.empty() && args == nullptr) {
    return cudaErrorInvalidValue;
  }
  const uint32_t count = static_cast<uint32_t>(sizes.size());
  for (uint32_t i = 0; i < count; ++i) {
    if (args[i] == nullptr)
      return cudaErrorInvalidValue;
  }
  lupine_mark_mapped_host_kernel_params(args, sizes.data(), count);
  uint64_t async_sequence = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_write_start_async_request(conn, op, &async_sequence) < 0 ||
      rpc_write(conn, &async_sequence, sizeof(async_sequence)) < 0 ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      rpc_write(conn, &gridDim, sizeof(gridDim)) < 0 ||
      rpc_write(conn, &blockDim, sizeof(blockDim)) < 0 ||
      rpc_write(conn, &sharedMem, sizeof(sharedMem)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      write_params(conn, count, sizes, args) < 0 ||
      lupine_rpc_write_end(conn) < 0) {
    return rpc_error();
  }
  lupine_invalidate_runtime_context(conn);
  return cudaSuccess;
}

} // namespace

extern "C" cudaError_t
cudaGraphAddKernelNode(cudaGraphNode_t *node, cudaGraph_t graph,
                       const cudaGraphNode_t *dependencies,
                       size_t dependency_count,
                       const cudaKernelNodeParams *params) {
  if (node == nullptr || params == nullptr ||
      (dependency_count != 0 && dependencies == nullptr)) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  std::vector<size_t> sizes;
  cudaError_t result = graph_kernel_param_sizes(conn, params, &sizes);
  if (result != cudaSuccess) {
    return result;
  }
  if (rpc_write_start_request(conn, RPC_cudaGraphAddKernelNode) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &dependency_count, sizeof(dependency_count)) < 0 ||
      rpc_write(conn, dependencies, dependency_count * sizeof(*dependencies)) <
          0 ||
      rpc_write(conn, params, sizeof(*params)) < 0 ||
      write_params(conn, static_cast<uint32_t>(sizes.size()), sizes,
                   params->kernelParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, node, sizeof(*node)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t
cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                             const cudaKernelNodeParams *params) {
  conn_t *conn = connection();
  std::vector<size_t> sizes;
  cudaError_t result = graph_kernel_param_sizes(conn, params, &sizes);
  if (result != cudaSuccess) {
    return result;
  }
  if (rpc_write_start_request(conn, RPC_cudaGraphKernelNodeSetParams) < 0 ||
      rpc_write(conn, &node, sizeof(node)) < 0 ||
      rpc_write(conn, params, sizeof(*params)) < 0 ||
      write_params(conn, static_cast<uint32_t>(sizes.size()), sizes,
                   params->kernelParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" cudaError_t
cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec, cudaGraphNode_t node,
                                 const cudaKernelNodeParams *params) {
  conn_t *conn = connection();
  std::vector<size_t> sizes;
  cudaError_t result = graph_kernel_param_sizes(conn, params, &sizes);
  if (result != cudaSuccess) {
    return result;
  }
  if (rpc_write_start_request(conn, RPC_cudaGraphExecKernelNodeSetParams) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &node, sizeof(node)) < 0 ||
      rpc_write(conn, params, sizeof(*params)) < 0 ||
      write_params(conn, static_cast<uint32_t>(sizes.size()), sizes,
                   params->kernelParams) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

#if CUDART_VERSION >= 12000
namespace {

cudaError_t graph_add_node(int op, cudaGraphNode_t *node, cudaGraph_t graph,
                           const cudaGraphNode_t *dependencies,
                           const cudaGraphEdgeData *edges,
                           size_t dependency_count,
                           cudaGraphNodeParams *params) {
  if (node == nullptr || params == nullptr ||
      (dependency_count != 0 && dependencies == nullptr)) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection();
  std::vector<size_t> sizes;
  cudaError_t result = cudaSuccess;
  if (params->type == cudaGraphNodeTypeKernel) {
    cudaKernelNodeParams kernel{};
    kernel.func = params->kernel.func;
    kernel.kernelParams = params->kernel.kernelParams;
    kernel.extra = params->kernel.extra;
    result = graph_kernel_param_sizes(conn, &kernel, &sizes);
    if (result != cudaSuccess) {
      return result;
    }
  } else {
    switch (params->type) {
    case cudaGraphNodeTypeConditional:
    case cudaGraphNodeTypeEmpty:
    case cudaGraphNodeTypeMemset:
    case cudaGraphNodeTypeGraph:
    case cudaGraphNodeTypeWaitEvent:
    case cudaGraphNodeTypeEventRecord:
    case cudaGraphNodeTypeMemFree:
      break;
    default:
      return cudaErrorNotSupported;
    }
  }
  bool has_edges = edges != nullptr;
  std::vector<cudaGraph_t> children(params->type == cudaGraphNodeTypeConditional
                                        ? params->conditional.size
                                        : 0);
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &dependency_count, sizeof(dependency_count)) < 0 ||
      rpc_write(conn, dependencies, dependency_count * sizeof(*dependencies)) <
          0 ||
      rpc_write(conn, &has_edges, sizeof(has_edges)) < 0 ||
      (has_edges &&
       rpc_write(conn, edges, dependency_count * sizeof(*edges)) < 0) ||
      rpc_write(conn, params, sizeof(*params)) < 0 ||
      (params->type == cudaGraphNodeTypeKernel &&
       write_params(conn, static_cast<uint32_t>(sizes.size()), sizes,
                    params->kernel.kernelParams) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, node, sizeof(*node)) < 0 ||
      rpc_read(conn, children.data(), children.size() * sizeof(cudaGraph_t)) <
          0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == cudaSuccess && !children.empty()) {
    std::lock_guard<std::mutex> lock(conditional_graphs_mutex());
    auto &entry = conditional_graph_cache()[*node];
    entry = {graph, std::move(children)};
    params->conditional.phGraph_out = entry.children.data();
  }
  return result;
}

} // namespace

#if CUDART_VERSION >= 13000
extern "C" cudaError_t cudaGraphAddNode(cudaGraphNode_t *node,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t *dependencies,
                                        const cudaGraphEdgeData *edges,
                                        size_t dependency_count,
                                        cudaGraphNodeParams *params) {
  return graph_add_node(RPC_cudaGraphAddNode, node, graph, dependencies, edges,
                        dependency_count, params);
}
#else
extern "C" cudaError_t cudaGraphAddNode(cudaGraphNode_t *node,
                                        cudaGraph_t graph,
                                        const cudaGraphNode_t *dependencies,
                                        size_t dependency_count,
                                        cudaGraphNodeParams *params) {
  return graph_add_node(RPC_cudaGraphAddNode, node, graph, dependencies,
                        nullptr, dependency_count, params);
}

extern "C" cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t *node,
                                           cudaGraph_t graph,
                                           const cudaGraphNode_t *dependencies,
                                           const cudaGraphEdgeData *edges,
                                           size_t dependency_count,
                                           cudaGraphNodeParams *params) {
  return graph_add_node(RPC_cudaGraphAddNode_v2, node, graph, dependencies,
                        edges, dependency_count, params);
}
#endif
#endif

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
  return launch(conn, RPC_cudaLaunchKernel, func, gridDim, blockDim, args,
                sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
  return launch(conn, RPC_cudaLaunchCooperativeKernel, func, gridDim, blockDim,
                args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                           const void *func, void **args) {
  if (config == nullptr ||
      (config->numAttrs != 0 && config->attrs == nullptr)) {
    return cudaErrorInvalidValue;
  }
  conn_t *conn = connection_for_stream(config->stream);
  std::vector<size_t> sizes;
  cudaError_t resolved = param_sizes(conn, func, &sizes);
  if (resolved != cudaSuccess) {
    return resolved;
  }
  if (!sizes.empty() && args == nullptr) {
    return cudaErrorInvalidValue;
  }
  const uint32_t attribute_count = config->numAttrs;
  const uint32_t count = static_cast<uint32_t>(sizes.size());
  for (uint32_t i = 0; i < count; ++i) {
    if (args[i] == nullptr)
      return cudaErrorInvalidValue;
  }
  lupine_mark_mapped_host_kernel_params(args, sizes.data(), count);
  uint64_t async_sequence = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_write_start_async_request(conn, RPC_cudaLaunchKernelExC,
                                           &async_sequence) < 0 ||
      rpc_write(conn, &async_sequence, sizeof(async_sequence)) < 0 ||
      rpc_write(conn, &config->gridDim, sizeof(config->gridDim)) < 0 ||
      rpc_write(conn, &config->blockDim, sizeof(config->blockDim)) < 0 ||
      rpc_write(conn, &config->dynamicSmemBytes,
                sizeof(config->dynamicSmemBytes)) < 0 ||
      rpc_write(conn, &config->stream, sizeof(config->stream)) < 0 ||
      rpc_write(conn, &attribute_count, sizeof(attribute_count)) < 0 ||
      (attribute_count != 0 &&
       rpc_write(conn, config->attrs,
                 attribute_count * sizeof(*config->attrs)) < 0) ||
      rpc_write(conn, &func, sizeof(func)) < 0 ||
      write_params(conn, count, sizes, args) < 0 ||
      lupine_rpc_write_end(conn) < 0) {
    return rpc_error();
  }
  lupine_invalidate_runtime_context(conn);
  return cudaSuccess;
}

#if CUDART_VERSION >= 13000
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                          dim3 blockDim, void **args,
                                          size_t sharedMem,
                                          cudaStream_t stream) {
  conn_t *conn = connection_for_stream(stream);
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
    return cudaErrorInvalidValue;
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
    return rpc_error();
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (return_value != cudaSuccess) {
    return return_value;
  }
  std::lock_guard<std::mutex> lock(*names_mutex);
  *name = names->emplace(func, std::move(text)).first->second.c_str();
  return cudaSuccess;
}
#endif
