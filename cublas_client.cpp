#include <cublas_v2.h>
#include <cuda.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "cublas_bridge.h"
#include "codegen/gen_rpc_ids.h"

namespace {

const lupine_cublas_rpc_api *rpc_api() {
  const auto *api = lupine_cublas_rpc_api_v1();
  return api != nullptr && api->version == LUPINE_CUBLAS_RPC_API_VERSION
             ? api
             : nullptr;
}

std::mutex &handle_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<cublasHandle_t, int> &handle_routes() {
  static auto *routes = new std::unordered_map<cublasHandle_t, int>();
  return *routes;
}

} // namespace

static int lupine_cublas_default_route() {
  const auto *api = rpc_api();
  return api == nullptr ? -2 : api->default_route();
}

static int lupine_cublas_route_for_stream(cudaStream_t stream) {
  const auto *api = rpc_api();
  return api == nullptr ? -2 : api->stream_route(stream);
}

static int lupine_cublas_route_for_handle(cublasHandle_t handle) {
  {
    std::lock_guard<std::mutex> lock(handle_mutex());
    const auto found = handle_routes().find(handle);
    if (found != handle_routes().end()) {
      return found->second;
    }
  }
  return lupine_cublas_default_route();
}

static bool lupine_cublas_stream_matches_route(int route_id,
                                               cudaStream_t stream) {
  return route_id == lupine_cublas_route_for_stream(stream);
}

static conn_t *lupine_cublas_connection(int route_id) {
  const auto *api = rpc_api();
  return api == nullptr ? nullptr : api->connection(route_id);
}

static void lupine_cublas_note_handle(cublasHandle_t handle, int route_id) {
  std::lock_guard<std::mutex> lock(handle_mutex());
  handle_routes()[handle] = route_id;
}

static void lupine_cublas_forget_handle(cublasHandle_t handle) {
  std::lock_guard<std::mutex> lock(handle_mutex());
  handle_routes().erase(handle);
}

static int lupine_prepare_rpc(conn_t *conn) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->prepare(conn);
}

static int rpc_write_start_request(conn_t *conn, int operation) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->write_start_request(conn, operation);
}

static int rpc_write(conn_t *conn, const void *data, std::size_t size) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->write(conn, data, size);
}

static int rpc_wait_for_response(conn_t *conn) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->wait_for_response(conn);
}

static int rpc_read(conn_t *conn, void *data, std::size_t size) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->read(conn, data, size);
}

static int rpc_read_end(conn_t *conn) {
  const auto *api = rpc_api();
  return api == nullptr ? -1 : api->read_end(conn);
}

#include "codegen/gen_cublas_client.inc"

extern "C" size_t CUBLASWINAPI cublasGetCudartVersion() {
  return CUDA_VERSION;
}

extern "C" const char *CUBLASWINAPI
cublasGetStatusName(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

extern "C" const char *CUBLASWINAPI
cublasGetStatusString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "success";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "the cuBLAS library was not initialized";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "resource allocation failed";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "an invalid value was supplied";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "the device architecture is unsupported";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "memory mapping failed";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "GPU execution failed";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "an internal cuBLAS error occurred";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "the requested operation is not supported";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "a licensing error occurred";
  default:
    return "unknown cuBLAS status";
  }
}

extern "C" void CUBLASWINAPI cublasXerbla(const char *, int) {}
