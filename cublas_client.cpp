#include <cublas_v2.h>
#include <cuda.h>

#include <cstddef>
#include <cstring>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_api.h"

namespace {

const lupine_cuda_client_api *cuda_api() {
  static const auto *api = [] {
    const auto *candidate = lupine_cuda_client_api_v1();
    return candidate != nullptr &&
                   candidate->version == LUPINE_CUDA_CLIENT_API_VERSION
               ? candidate
               : nullptr;
  }();
  return api;
}

} // namespace

static cublasStatus_t lupine_cublas_call_begin(int operation,
                                               const lupine_cuda_owner *owners,
                                               std::size_t owner_count,
                                               lupine_cuda_call **call) {
  const auto *api = cuda_api();
  if (api == nullptr) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  const int result = api->call_begin(operation, owners, owner_count, call);
  if (result == LUPINE_CUDA_CALL_INVALID_OWNER ||
      result == LUPINE_CUDA_CALL_OWNER_MISMATCH) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return result == LUPINE_CUDA_CALL_SUCCESS ? CUBLAS_STATUS_SUCCESS
                                            : CUBLAS_STATUS_NOT_INITIALIZED;
}

static int rpc_write(lupine_cuda_call *call, const void *data,
                     std::size_t size) {
  const auto *api = cuda_api();
  return api == nullptr ? -1 : api->call_write(call, data, size);
}

static int rpc_wait_for_response(lupine_cuda_call *call) {
  const auto *api = cuda_api();
  return api == nullptr ? -1 : api->call_invoke(call);
}

static int rpc_read(lupine_cuda_call *call, void *data, std::size_t size) {
  const auto *api = cuda_api();
  return api == nullptr ? -1 : api->call_read(call, data, size);
}

static int rpc_read_end(lupine_cuda_call *call) {
  const auto *api = cuda_api();
  return api == nullptr ? -1 : api->call_finish(call);
}

static int lupine_cublas_note_handle(lupine_cuda_call *call,
                                     cublasHandle_t handle) {
  const auto *api = cuda_api();
  return api == nullptr
             ? -1
             : api->record_owner(call, LUPINE_CUDA_OWNER_CUBLAS_HANDLE,
                                 reinterpret_cast<std::uintptr_t>(handle));
}

static void lupine_cublas_forget_handle(cublasHandle_t handle) {
  const auto *api = cuda_api();
  if (api != nullptr) {
    api->forget_owner(LUPINE_CUDA_OWNER_CUBLAS_HANDLE,
                      reinterpret_cast<std::uintptr_t>(handle));
  }
}

static void lupine_cublas_call_destroy(lupine_cuda_call *call) {
  const auto *api = cuda_api();
  if (api != nullptr) {
    api->call_destroy(call);
  }
}

#include "codegen/gen_cublas_client.inc"

extern "C" size_t CUBLASWINAPI cublasGetCudartVersion() { return CUDA_VERSION; }

extern "C" const char *CUBLASWINAPI cublasGetStatusName(cublasStatus_t status) {
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
