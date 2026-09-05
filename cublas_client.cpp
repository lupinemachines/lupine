// cuBLAS (libcublas) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcublas, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps
// the three APIs ordered. Nearly all of the surface is generated. The calls
// in this file carry something the generated marshalling cannot: a host
// vector or matrix with a stride, a static string, or a result that is not a
// status. A callback into the client is a generated stub that returns
// CUBLAS_STATUS_NOT_SUPPORTED.

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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

// A handle is created on the runtime's current device and routes every later
// call to that connection. Its pointer mode is kept here because the generated
// marshalling decides before each call whether a scalar's bytes or its address
// travel.
struct handle_state {
  conn_t *conn;
  bool host_pointers;
};

std::mutex handles_mutex;
std::unordered_map<cublasHandle_t, handle_state> handles;

// A call without a handle goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(cublasHandle_t handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(handle);
  return it == handles.end() ? nullptr : it->second.conn;
}

conn_t *connection_for_stream(cudaStream_t stream) {
  return stream == nullptr ? connection() : lupine_rpc_conn_for_stream(stream);
}

void note_handle_owner(conn_t *conn, cublasHandle_t handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles[handle] = {conn, true};
}

void note_pointer_mode(cublasHandle_t handle, cublasPointerMode_t mode) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(handle);
  if (it != handles.end()) {
    it->second.host_pointers = mode == CUBLAS_POINTER_MODE_HOST;
  }
}

// A destroyed handle's address may come back from a later cublasCreate,
// which records it afresh.
void forget_handle(cublasHandle_t handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles.erase(handle);
}

// cuBLAS places alpha and beta together, so the scalar's name is unused.
bool scalar_on_host(cublasHandle_t handle, const char *) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(handle);
  return it == handles.end() || it->second.host_pointers;
}

// ---------------------------------------------------------------------------
// Scalar widths
// ---------------------------------------------------------------------------

// GemmEx scales in the compute type, complex when C is.
size_t compute_scalar_width(cublasComputeType_t compute, cudaDataType c_type) {
  switch (compute) {
  case CUBLAS_COMPUTE_16F:
  case CUBLAS_COMPUTE_16F_PEDANTIC:
    return sizeof(__half);
  case CUBLAS_COMPUTE_32I:
  case CUBLAS_COMPUTE_32I_PEDANTIC:
    return sizeof(int);
  case CUBLAS_COMPUTE_64F:
  case CUBLAS_COMPUTE_64F_PEDANTIC:
    return c_type == CUDA_C_64F ? sizeof(cuDoubleComplex) : sizeof(double);
  default:
    return c_type == CUDA_C_32F ? sizeof(cuComplex) : sizeof(float);
  }
}

} // namespace

#include "codegen/gen_cublas_client.inc"

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

extern "C" const char *cublasGetStatusName(cublasStatus_t status) {
  return status_text(RPC_cublasGetStatusName, status);
}

extern "C" const char *cublasGetStatusString(cublasStatus_t status) {
  return status_text(RPC_cublasGetStatusString, status);
}

extern "C" cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut,
                                                int logToStdErr,
                                                const char *logFileName) {
  conn_t *conn = connection();
  const uint8_t has_name = logFileName != nullptr;
  const uint32_t length =
      has_name ? static_cast<uint32_t>(strlen(logFileName)) : 0;
  cublasStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cublasLoggerConfigure) < 0 ||
      rpc_write(conn, &logIsOn, sizeof(logIsOn)) < 0 ||
      rpc_write(conn, &logToStdOut, sizeof(logToStdOut)) < 0 ||
      rpc_write(conn, &logToStdErr, sizeof(logToStdErr)) < 0 ||
      rpc_write(conn, &has_name, sizeof(has_name)) < 0 ||
      (has_name && rpc_write(conn, &length, sizeof(length)) < 0) ||
      (length != 0 && rpc_write(conn, logFileName, length) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(status);
}

extern "C" void cublasXerbla(const char *srName, int info) {
  fprintf(stderr,
          " ** On entry to %s parameter number %d had an illegal value\n",
          srName, info);
}

// ---------------------------------------------------------------------------
// Host vectors and matrices
// ---------------------------------------------------------------------------

namespace {

// A host vector or matrix travels packed: `n` elements of `elemSize` bytes, or
// `cols` columns of `rows * elemSize` bytes. The server spreads it to the
// device stride the caller asked for, and packs a fetched one the same way.
// The index type is the call's, so the `_64` entry points send 64-bit counts.
template <typename Index>
cublasStatus_t set_vector(int op, Index n, Index elemSize, const void *x,
                          Index incx, void *devicePtr, Index incy,
                          const cudaStream_t *stream) {
  if (n < 0 || elemSize <= 0 || incx <= 0 || incy <= 0) {
    return record(CUBLAS_STATUS_INVALID_VALUE);
  }
  const size_t width = static_cast<size_t>(elemSize);
  const size_t bytes = static_cast<size_t>(n) * width;
  std::vector<unsigned char> packed;
  const void *payload = x;
  if (incx != 1 && bytes != 0) {
    packed.resize(bytes);
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
      memcpy(packed.data() + i * width,
             static_cast<const unsigned char *>(x) + i * incx * width, width);
    }
    payload = packed.data();
  }
  conn_t *conn =
      stream != nullptr ? connection_for_stream(*stream) : connection();
  cublasStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &n, sizeof(n)) < 0 ||
      rpc_write(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_write(conn, &devicePtr, sizeof(devicePtr)) < 0 ||
      rpc_write(conn, &incy, sizeof(incy)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      (bytes != 0 && rpc_write(conn, payload, bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(status);
}

template <typename Index>
cublasStatus_t get_vector(int op, Index n, Index elemSize, const void *x,
                          Index incx, void *y, Index incy,
                          const cudaStream_t *stream) {
  if (n < 0 || elemSize <= 0 || incx <= 0 || incy <= 0) {
    return record(CUBLAS_STATUS_INVALID_VALUE);
  }
  const size_t width = static_cast<size_t>(elemSize);
  const size_t bytes = static_cast<size_t>(n) * width;
  std::vector<unsigned char> packed(incy != 1 ? bytes : 0);
  void *payload = incy != 1 ? packed.data() : y;
  conn_t *conn =
      stream != nullptr ? connection_for_stream(*stream) : connection();
  cublasStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &n, sizeof(n)) < 0 ||
      rpc_write(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_write(conn, &x, sizeof(x)) < 0 ||
      rpc_write(conn, &incx, sizeof(incx)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      (status == CUBLAS_STATUS_SUCCESS && bytes != 0 &&
       rpc_read(conn, payload, bytes) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  if (incy != 1 && status == CUBLAS_STATUS_SUCCESS) {
    for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
      memcpy(static_cast<unsigned char *>(y) + i * incy * width,
             packed.data() + i * width, width);
    }
  }
  return record(status);
}

template <typename Index>
cublasStatus_t set_matrix(int op, Index rows, Index cols, Index elemSize,
                          const void *A, Index lda, void *B, Index ldb,
                          const cudaStream_t *stream) {
  if (rows < 0 || cols < 0 || elemSize <= 0 || lda <= 0 || ldb <= 0) {
    return record(CUBLAS_STATUS_INVALID_VALUE);
  }
  const size_t column = static_cast<size_t>(rows) * elemSize;
  const size_t bytes = column * static_cast<size_t>(cols);
  std::vector<unsigned char> packed;
  const void *payload = A;
  if (lda != rows && bytes != 0) {
    packed.resize(bytes);
    for (size_t j = 0; j < static_cast<size_t>(cols); ++j) {
      memcpy(packed.data() + j * column,
             static_cast<const unsigned char *>(A) + j * lda * elemSize,
             column);
    }
    payload = packed.data();
  }
  conn_t *conn =
      stream != nullptr ? connection_for_stream(*stream) : connection();
  cublasStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &rows, sizeof(rows)) < 0 ||
      rpc_write(conn, &cols, sizeof(cols)) < 0 ||
      rpc_write(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_write(conn, &B, sizeof(B)) < 0 ||
      rpc_write(conn, &ldb, sizeof(ldb)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      (bytes != 0 && rpc_write(conn, payload, bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  return record(status);
}

template <typename Index>
cublasStatus_t get_matrix(int op, Index rows, Index cols, Index elemSize,
                          const void *A, Index lda, void *B, Index ldb,
                          const cudaStream_t *stream) {
  if (rows < 0 || cols < 0 || elemSize <= 0 || lda <= 0 || ldb <= 0) {
    return record(CUBLAS_STATUS_INVALID_VALUE);
  }
  const size_t column = static_cast<size_t>(rows) * elemSize;
  const size_t bytes = column * static_cast<size_t>(cols);
  std::vector<unsigned char> packed(ldb != rows ? bytes : 0);
  void *payload = ldb != rows ? packed.data() : B;
  conn_t *conn =
      stream != nullptr ? connection_for_stream(*stream) : connection();
  cublasStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &rows, sizeof(rows)) < 0 ||
      rpc_write(conn, &cols, sizeof(cols)) < 0 ||
      rpc_write(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_write(conn, &A, sizeof(A)) < 0 ||
      rpc_write(conn, &lda, sizeof(lda)) < 0 ||
      (stream != nullptr && rpc_write(conn, stream, sizeof(*stream)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      (status == CUBLAS_STATUS_SUCCESS && bytes != 0 &&
       rpc_read(conn, payload, bytes) < 0) ||
      rpc_read_end(conn) < 0) {
    return record(rpc_error());
  }
  if (ldb != rows && status == CUBLAS_STATUS_SUCCESS) {
    for (size_t j = 0; j < static_cast<size_t>(cols); ++j) {
      memcpy(static_cast<unsigned char *>(B) + j * ldb * elemSize,
             packed.data() + j * column, column);
    }
  }
  return record(status);
}

} // namespace

extern "C" cublasStatus_t cublasSetVector(int n, int elemSize, const void *x,
                                          int incx, void *devicePtr, int incy) {
  return set_vector<int>(RPC_cublasSetVector, n, elemSize, x, incx, devicePtr,
                         incy, nullptr);
}

extern "C" cublasStatus_t cublasGetVector(int n, int elemSize, const void *x,
                                          int incx, void *y, int incy) {
  return get_vector<int>(RPC_cublasGetVector, n, elemSize, x, incx, y, incy,
                         nullptr);
}

extern "C" cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                                          const void *A, int lda, void *B,
                                          int ldb) {
  return set_matrix<int>(RPC_cublasSetMatrix, rows, cols, elemSize, A, lda, B,
                         ldb, nullptr);
}

extern "C" cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
                                          const void *A, int lda, void *B,
                                          int ldb) {
  return get_matrix<int>(RPC_cublasGetMatrix, rows, cols, elemSize, A, lda, B,
                         ldb, nullptr);
}

extern "C" cublasStatus_t cublasSetVectorAsync(int n, int elemSize,
                                               const void *hostPtr, int incx,
                                               void *devicePtr, int incy,
                                               cudaStream_t stream) {
  return set_vector<int>(RPC_cublasSetVectorAsync, n, elemSize, hostPtr, incx,
                         devicePtr, incy, &stream);
}

extern "C" cublasStatus_t cublasGetVectorAsync(int n, int elemSize,
                                               const void *devicePtr, int incx,
                                               void *hostPtr, int incy,
                                               cudaStream_t stream) {
  return get_vector<int>(RPC_cublasGetVectorAsync, n, elemSize, devicePtr, incx,
                         hostPtr, incy, &stream);
}

extern "C" cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,
                                               const void *A, int lda, void *B,
                                               int ldb, cudaStream_t stream) {
  return set_matrix<int>(RPC_cublasSetMatrixAsync, rows, cols, elemSize, A, lda,
                         B, ldb, &stream);
}

extern "C" cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,
                                               const void *A, int lda, void *B,
                                               int ldb, cudaStream_t stream) {
  return get_matrix<int>(RPC_cublasGetMatrixAsync, rows, cols, elemSize, A, lda,
                         B, ldb, &stream);
}

#if CUBLAS_VERSION >= 120000
extern "C" cublasStatus_t cublasSetVector_64(int64_t n, int64_t elemSize,
                                             const void *x, int64_t incx,
                                             void *devicePtr, int64_t incy) {
  return set_vector<int64_t>(RPC_cublasSetVector_64, n, elemSize, x, incx,
                             devicePtr, incy, nullptr);
}

extern "C" cublasStatus_t cublasGetVector_64(int64_t n, int64_t elemSize,
                                             const void *x, int64_t incx,
                                             void *y, int64_t incy) {
  return get_vector<int64_t>(RPC_cublasGetVector_64, n, elemSize, x, incx, y,
                             incy, nullptr);
}

extern "C" cublasStatus_t cublasSetMatrix_64(int64_t rows, int64_t cols,
                                             int64_t elemSize, const void *A,
                                             int64_t lda, void *B,
                                             int64_t ldb) {
  return set_matrix<int64_t>(RPC_cublasSetMatrix_64, rows, cols, elemSize, A,
                             lda, B, ldb, nullptr);
}

extern "C" cublasStatus_t cublasGetMatrix_64(int64_t rows, int64_t cols,
                                             int64_t elemSize, const void *A,
                                             int64_t lda, void *B,
                                             int64_t ldb) {
  return get_matrix<int64_t>(RPC_cublasGetMatrix_64, rows, cols, elemSize, A,
                             lda, B, ldb, nullptr);
}

extern "C" cublasStatus_t cublasSetVectorAsync_64(int64_t n, int64_t elemSize,
                                                  const void *hostPtr,
                                                  int64_t incx, void *devicePtr,
                                                  int64_t incy,
                                                  cudaStream_t stream) {
  return set_vector<int64_t>(RPC_cublasSetVectorAsync_64, n, elemSize, hostPtr,
                             incx, devicePtr, incy, &stream);
}

extern "C" cublasStatus_t cublasGetVectorAsync_64(int64_t n, int64_t elemSize,
                                                  const void *devicePtr,
                                                  int64_t incx, void *hostPtr,
                                                  int64_t incy,
                                                  cudaStream_t stream) {
  return get_vector<int64_t>(RPC_cublasGetVectorAsync_64, n, elemSize,
                             devicePtr, incx, hostPtr, incy, &stream);
}

extern "C" cublasStatus_t cublasSetMatrixAsync_64(int64_t rows, int64_t cols,
                                                  int64_t elemSize,
                                                  const void *A, int64_t lda,
                                                  void *B, int64_t ldb,
                                                  cudaStream_t stream) {
  return set_matrix<int64_t>(RPC_cublasSetMatrixAsync_64, rows, cols, elemSize,
                             A, lda, B, ldb, &stream);
}

extern "C" cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols,
                                                  int64_t elemSize,
                                                  const void *A, int64_t lda,
                                                  void *B, int64_t ldb,
                                                  cudaStream_t stream) {
  return get_matrix<int64_t>(RPC_cublasGetMatrixAsync_64, rows, cols, elemSize,
                             A, lda, B, ldb, &stream);
}
#endif
