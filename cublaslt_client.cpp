// cuBLASLt (libcublasLt) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcublasLt, on the
// CUDA driver shim's connections like the cuBLAS shim. A descriptor is a
// fixed-size opaque struct that travels by value with each call, so the only
// state here is what the generated marshalling has to know before a call:
// which connection a handle lives on, and each descriptor's pointer mode and
// scale type, which decide whether alpha and beta go by value and how wide.
// The Create and Destroy pairs are local: Create allocates the caller's
// struct and runs the generated Init.

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "codegen/gen_rpc_ids.h"
#include "cublas_scalar.h"
#include "cuda_client_rpc.h"

namespace {

cublasStatus_t rpc_error() { return CUBLAS_STATUS_NOT_INITIALIZED; }

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
// Handles
// ---------------------------------------------------------------------------

std::mutex handles_mutex;

// A call without a handle goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

// The owner is recorded in the driver client, so a handle from cublasCreate
// routes here exactly as one from cublasLtCreate does.
conn_t *connection_for_handle(cublasLtHandle_t handle) {
  return lupine_rpc_conn_for_blas_handle(handle);
}

void note_handle_owner(conn_t *conn, cublasLtHandle_t handle) {
  lupine_note_blas_handle_owner(handle, conn);
}

void forget_handle(cublasLtHandle_t handle) {
  lupine_forget_blas_handle_owner(handle);
}

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

// The library fills the caller's descriptor, but its layout is private, so the
// two attributes alpha and beta depend on are mirrored by address as the
// descriptor is initialized and set. An address that was never initialized
// through the shim reads as the library's defaults.
struct descriptor_state {
  cudaDataType scale_type;
  cublasLtPointerMode_t pointer_mode;
};

std::mutex descriptors_mutex;
std::unordered_map<const void *, descriptor_state> descriptors;

descriptor_state descriptor(const void *desc) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(desc);
  return it == descriptors.end()
             ? descriptor_state{CUDA_R_32F, CUBLASLT_POINTER_MODE_HOST}
             : it->second;
}

void note_descriptor(const void *desc, cudaDataType scale_type) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors[desc] = {scale_type, CUBLASLT_POINTER_MODE_HOST};
}

void forget_descriptor(const void *desc) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors.erase(desc);
}

void note_attribute(const void *desc, bool scale_type, bool pointer_mode,
                    const void *buf, size_t size) {
  if ((!scale_type && !pointer_mode) || buf == nullptr ||
      size < sizeof(int32_t)) {
    return;
  }
  int32_t value;
  std::memcpy(&value, buf, sizeof(value));
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(desc);
  if (it == descriptors.end()) {
    it = descriptors
             .emplace(desc,
                      descriptor_state{CUDA_R_32F, CUBLASLT_POINTER_MODE_HOST})
             .first;
  }
  if (scale_type) {
    it->second.scale_type = static_cast<cudaDataType>(value);
  } else {
    it->second.pointer_mode = static_cast<cublasLtPointerMode_t>(value);
  }
}

void note_attribute(cublasLtMatmulDescOpaque_t *desc,
                    cublasLtMatmulDescAttributes_t attr, const void *buf,
                    size_t size) {
  note_attribute(desc, attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                 attr == CUBLASLT_MATMUL_DESC_POINTER_MODE, buf, size);
}

void note_attribute(cublasLtMatrixTransformDescOpaque_t *desc,
                    cublasLtMatrixTransformDescAttributes_t attr,
                    const void *buf, size_t size) {
  note_attribute(desc, attr == CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
                 attr == CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE, buf,
                 size);
}

#if CUBLAS_VERSION >= 130100
// A matmul descriptor refers to its emulation descriptor by address, which
// means nothing to the server. The server keeps a copy of each emulation
// descriptor attached to a matmul descriptor, the matmul descriptor holds the
// copy's address, and these maps translate between the two.
std::mutex emulation_copies_mutex;
std::unordered_map<cublasLtEmulationDesc_t, cublasLtEmulationDesc_t>
    emulation_copies;
std::unordered_map<cublasLtEmulationDesc_t, cublasLtEmulationDesc_t>
    emulation_originals;

cublasLtEmulationDesc_t
copy_emulation_descriptor(cublasLtEmulationDesc_t desc) {
  conn_t *conn = connection();
  cublasLtEmulationDesc_t copy = nullptr;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_lupineCublasLtEmulationDescCopy) < 0 ||
      rpc_write(conn, &desc, sizeof(desc)) < 0 ||
      rpc_write(conn, desc, sizeof(*desc)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0 || rpc_read_end(conn) < 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(emulation_copies_mutex);
  emulation_copies[desc] = copy;
  emulation_originals[copy] = desc;
  return copy;
}

// The attached copy follows later changes to the caller's descriptor, as the
// library would see them through the address.
void refresh_emulation_copy(cublasLtEmulationDesc_t desc) {
  {
    std::lock_guard<std::mutex> lock(emulation_copies_mutex);
    if (emulation_copies.count(desc) == 0) {
      return;
    }
  }
  copy_emulation_descriptor(desc);
}
#endif

// The value the server's library is given for a matmul descriptor attribute:
// an emulation descriptor becomes the server's copy, held in `copy`.
const void *server_attribute(cublasLtMatmulDescAttributes_t attr,
                             const void *buf, size_t size, void **copy) {
#if CUBLAS_VERSION >= 130100
  cublasLtEmulationDesc_t desc = nullptr;
  if (attr == CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR &&
      size == sizeof(desc)) {
    std::memcpy(&desc, buf, sizeof(desc));
    if (desc != nullptr) {
      *copy = copy_emulation_descriptor(desc);
      return copy;
    }
  }
#else
  (void)attr;
  (void)size;
  (void)copy;
#endif
  return buf;
}

// The reverse for a get: the caller reads back its own emulation descriptor.
void client_attribute(cublasLtMatmulDescAttributes_t attr, void *buf,
                      size_t size) {
#if CUBLAS_VERSION >= 130100
  cublasLtEmulationDesc_t copy = nullptr;
  if (attr != CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR ||
      size < sizeof(copy)) {
    return;
  }
  std::memcpy(&copy, buf, sizeof(copy));
  std::lock_guard<std::mutex> lock(emulation_copies_mutex);
  auto it = emulation_originals.find(copy);
  if (it != emulation_originals.end()) {
    std::memcpy(buf, &it->second, sizeof(copy));
  }
#else
  (void)attr;
  (void)buf;
  (void)size;
#endif
}

// The vector modes place alpha on the device and beta on the host or nowhere.
bool scalar_on_host(const void *desc, const char *name) {
  const cublasLtPointerMode_t mode = descriptor(desc).pointer_mode;
  return mode == CUBLASLT_POINTER_MODE_HOST ||
         (mode == CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST &&
          std::strcmp(name, "beta") == 0);
}

size_t scalar_width(const void *desc) {
  return data_type_width(descriptor(desc).scale_type);
}

} // namespace

#include "codegen/gen_cublaslt_client.inc"

// ---------------------------------------------------------------------------
// Descriptor lifetime
// ---------------------------------------------------------------------------

namespace {

// Create allocates the struct the caller would otherwise provide to Init, and
// Destroy frees it; neither reaches the server.
template <typename Opaque, typename Init>
cublasStatus_t create(Opaque **out, Init init) {
  if (out == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  Opaque *desc = new Opaque();
  cublasStatus_t status = init(desc);
  if (status != CUBLAS_STATUS_SUCCESS) {
    delete desc;
    desc = nullptr;
  }
  *out = desc;
  return status;
}

template <typename Opaque> cublasStatus_t destroy(Opaque *desc) {
  if (desc == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  forget_descriptor(desc);
  delete desc;
  return CUBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" cublasStatus_t
cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                         cublasComputeType_t computeType,
                         cudaDataType_t scaleType) {
  return create(matmulDesc, [&](cublasLtMatmulDescOpaque_t *desc) {
    return cublasLtMatmulDescInit_internal(desc, sizeof(*desc), computeType,
                                           scaleType);
  });
}

extern "C" cublasStatus_t
cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
  return destroy(matmulDesc);
}

extern "C" cublasStatus_t
cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, cudaDataType type,
                           uint64_t rows, uint64_t cols, int64_t ld) {
  return create(matLayout, [&](cublasLtMatrixLayoutOpaque_t *desc) {
    return cublasLtMatrixLayoutInit_internal(desc, sizeof(*desc), type, rows,
                                             cols, ld);
  });
}

extern "C" cublasStatus_t
cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
  return destroy(matLayout);
}

extern "C" cublasStatus_t
cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t *transformDesc,
                                  cudaDataType scaleType) {
  return create(transformDesc, [&](cublasLtMatrixTransformDescOpaque_t *desc) {
    return cublasLtMatrixTransformDescInit_internal(desc, sizeof(*desc),
                                                    scaleType);
  });
}

extern "C" cublasStatus_t cublasLtMatrixTransformDescDestroy(
    cublasLtMatrixTransformDesc_t transformDesc) {
  return destroy(transformDesc);
}

extern "C" cublasStatus_t
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
  return create(pref, [&](cublasLtMatmulPreferenceOpaque_t *desc) {
    return cublasLtMatmulPreferenceInit_internal(desc, sizeof(*desc));
  });
}

extern "C" cublasStatus_t
cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
  return destroy(pref);
}

#if CUBLAS_VERSION >= 130100
extern "C" cublasStatus_t
cublasLtEmulationDescCreate(cublasLtEmulationDesc_t *emulationDesc) {
  return create(emulationDesc, [&](cublasLtEmulationDescOpaque_t *desc) {
    return cublasLtEmulationDescInit_internal(desc, sizeof(*desc));
  });
}

extern "C" cublasStatus_t
cublasLtEmulationDescDestroy(cublasLtEmulationDesc_t emulationDesc) {
  return destroy(emulationDesc);
}
#endif

// ---------------------------------------------------------------------------
// Static strings
// ---------------------------------------------------------------------------

namespace {

// The server's library owns the status strings, so each is fetched once and
// kept for the life of the process.
const char *status_text(int op, cublasStatus_t status) {
  static std::mutex mutex;
  static std::unordered_map<long long, std::string> texts;
  const long long key = (static_cast<long long>(op) << 32) | status;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = texts.find(key);
    if (it != texts.end()) {
      return it->second.c_str();
    }
  }
  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = connection();
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return "<unknown>";
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return "<unknown>";
  }
  std::lock_guard<std::mutex> lock(mutex);
  return texts.emplace(key, std::move(text)).first->second.c_str();
}

} // namespace

extern "C" const char *cublasLtGetStatusName(cublasStatus_t status) {
  return status_text(RPC_cublasLtGetStatusName, status);
}

extern "C" const char *cublasLtGetStatusString(cublasStatus_t status) {
  return status_text(RPC_cublasLtGetStatusString, status);
}

namespace {
void log_callback(void *user_data, int level, const char *function,
                  const char *message, size_t) {
  reinterpret_cast<cublasLtLoggerCallback_t>(user_data)(level, function,
                                                        message);
}

void log_file(void *user_data, int level, const char *function,
              const char *message, size_t) {
  auto *file = static_cast<FILE *>(user_data);
  static const char *levels[] = {"Off",   "Error", "Trace",
                                 "Hints", "Info",  "Api"};
  const char *name = level >= 0 && level <= 5 ? levels[level] : "Unknown";
  std::fprintf(file, "[cublasLt][%s][%s] %s\n", name, function, message);
  std::fflush(file);
}

std::recursive_mutex log_files_mutex;
std::unordered_map<conn_t *, std::shared_ptr<FILE>> owned_log_files;

cublasStatus_t set_log_file(FILE *file, std::shared_ptr<FILE> owned) {
  library_log_target target;
  if (file != nullptr) {
    target = {log_file, file};
  }
  conn_t *conn = connection();
  cublasStatus_t status = rpc_error();
  // A callback can replace the file from inside rpc_read_end. Publish the new
  // ownership first and keep the previous file alive until all its logs drain.
  std::lock_guard<std::recursive_mutex> lock(log_files_mutex);
  auto previous = owned_log_files[conn];
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cublasLtLoggerSetFile) < 0 ||
      rpc_write(conn, &target.callback, sizeof(target.callback)) < 0 ||
      rpc_write(conn, &target.user_data, sizeof(target.user_data)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0) {
    return rpc_error();
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    owned_log_files[conn] = std::move(owned);
  }
  if (rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}
} // namespace

extern "C" cublasStatus_t
cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) {
  library_log_target target;
  if (callback != nullptr) {
    target = {log_callback, reinterpret_cast<void *>(callback)};
  }
  conn_t *conn = connection();
  cublasStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cublasLtLoggerSetCallback) < 0 ||
      rpc_write(conn, &target.callback, sizeof(target.callback)) < 0 ||
      rpc_write(conn, &target.user_data, sizeof(target.user_data)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" cublasStatus_t cublasLtLoggerSetFile(FILE *file) {
  return set_log_file(file, {});
}

extern "C" cublasStatus_t cublasLtLoggerOpenFile(const char *logFile) {
  if (logFile == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  FILE *file = std::fopen(logFile, "w");
  if (file == nullptr) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return set_log_file(file, std::shared_ptr<FILE>(file, std::fclose));
}
