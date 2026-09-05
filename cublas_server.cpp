// cublas_v2.h defines the marker cublas_api.h wants before including it, and
// the runtime header supplies the stream type the copies take.
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

cublasStatus_t function_not_found() { return CUBLAS_STATUS_NOT_SUPPORTED; }

// The library this server was compiled against; the handlers marshal its
// enums and pointer-mode scalars, so another major on the machine is not a
// match.
void *cublas_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUBLAS_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cublas64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcublas.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cublas_symbol(const char *name) {
  void *lib = cublas_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

int write_status(conn_t *conn, int request_id, cublasStatus_t status) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// cublasGetStatusName and cublasGetStatusString return static strings, so the
// response carries the bytes length-prefixed instead of a status.
int handle_status_text(conn_t *conn, const char *symbol) {
  cublasStatus_t status;
  if (rpc_read(conn, &status, sizeof(status)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cublasStatus_t);
  fn_t fn = cublas_symbol<fn_t>(symbol);
  const char *text = fn == nullptr ? nullptr : fn(status);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cublasGetStatusName(conn_t *conn) {
  return handle_status_text(conn, "cublasGetStatusName");
}

int handle_cublasGetStatusString(conn_t *conn) {
  return handle_status_text(conn, "cublasGetStatusString");
}

int handle_cublasGetCudartVersion(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = size_t (*)(void);
  fn_t fn = cublas_symbol<fn_t>("cublasGetCudartVersion");
  size_t version = fn == nullptr ? 0 : fn();
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &version, sizeof(version)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cublasLoggerConfigure(conn_t *conn) {
  int logIsOn, logToStdOut, logToStdErr;
  uint8_t has_name;
  uint32_t length = 0;
  if (rpc_read(conn, &logIsOn, sizeof(logIsOn)) < 0 ||
      rpc_read(conn, &logToStdOut, sizeof(logToStdOut)) < 0 ||
      rpc_read(conn, &logToStdErr, sizeof(logToStdErr)) < 0 ||
      rpc_read(conn, &has_name, sizeof(has_name)) < 0 ||
      (has_name && rpc_read(conn, &length, sizeof(length)) < 0)) {
    return -1;
  }
  std::string name(length, '\0');
  if (length != 0 && rpc_read(conn, &name[0], length) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cublasStatus_t (*)(int, int, int, const char *);
  fn_t fn = cublas_symbol<fn_t>("cublasLoggerConfigure");
  cublasStatus_t status = fn == nullptr ? function_not_found()
                                        : fn(logIsOn, logToStdOut, logToStdErr,
                                             has_name ? name.c_str() : nullptr);
  return write_status(conn, request_id, status);
}

// ---------------------------------------------------------------------------
// Host vectors and matrices
// ---------------------------------------------------------------------------

namespace {

// The client packs a host vector or matrix to stride 1, or to a leading
// dimension of `rows`; the library spreads it to the device stride the caller
// asked for, and packs a fetched one the same way. An async copy from pageable
// memory is staged by the time the call returns, so the packed buffer never
// has to outlive the handler.
template <typename Index>
int handle_set_vector(conn_t *conn, const char *symbol, bool async) {
  Index n, elemSize, incy;
  void *devicePtr;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &n, sizeof(n)) < 0 ||
      rpc_read(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_read(conn, &devicePtr, sizeof(devicePtr)) < 0 ||
      rpc_read(conn, &incy, sizeof(incy)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  std::vector<unsigned char> packed(static_cast<size_t>(n) * elemSize);
  if (!packed.empty() && rpc_read(conn, packed.data(), packed.size()) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using sync_t =
      cublasStatus_t (*)(Index, Index, const void *, Index, void *, Index);
  using async_t = cublasStatus_t (*)(Index, Index, const void *, Index, void *,
                                     Index, cudaStream_t);
  cublasStatus_t status = function_not_found();
  if (async) {
    if (async_t fn = cublas_symbol<async_t>(symbol)) {
      status = fn(n, elemSize, packed.data(), 1, devicePtr, incy, stream);
    }
  } else if (sync_t fn = cublas_symbol<sync_t>(symbol)) {
    status = fn(n, elemSize, packed.data(), 1, devicePtr, incy);
  }
  return write_status(conn, request_id, status);
}

template <typename Index>
int handle_get_vector(conn_t *conn, const char *symbol, bool async) {
  Index n, elemSize, incx;
  const void *x;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &n, sizeof(n)) < 0 ||
      rpc_read(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_read(conn, &x, sizeof(x)) < 0 ||
      rpc_read(conn, &incx, sizeof(incx)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(static_cast<size_t>(n) * elemSize);
  using sync_t =
      cublasStatus_t (*)(Index, Index, const void *, Index, void *, Index);
  using async_t = cublasStatus_t (*)(Index, Index, const void *, Index, void *,
                                     Index, cudaStream_t);
  cublasStatus_t status = function_not_found();
  if (async) {
    if (async_t fn = cublas_symbol<async_t>(symbol)) {
      status = fn(n, elemSize, x, incx, packed.data(), 1, stream);
    }
  } else if (sync_t fn = cublas_symbol<sync_t>(symbol)) {
    status = fn(n, elemSize, x, incx, packed.data(), 1);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      (status == CUBLAS_STATUS_SUCCESS && !packed.empty() &&
       rpc_write(conn, packed.data(), packed.size()) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

template <typename Index>
int handle_set_matrix(conn_t *conn, const char *symbol, bool async) {
  Index rows, cols, elemSize, ldb;
  void *B;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &rows, sizeof(rows)) < 0 ||
      rpc_read(conn, &cols, sizeof(cols)) < 0 ||
      rpc_read(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_read(conn, &B, sizeof(B)) < 0 ||
      rpc_read(conn, &ldb, sizeof(ldb)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  std::vector<unsigned char> packed(static_cast<size_t>(rows) * cols *
                                    elemSize);
  if (!packed.empty() && rpc_read(conn, packed.data(), packed.size()) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  const Index lda = rows > 0 ? rows : 1;
  using sync_t = cublasStatus_t (*)(Index, Index, Index, const void *, Index,
                                    void *, Index);
  using async_t = cublasStatus_t (*)(Index, Index, Index, const void *, Index,
                                     void *, Index, cudaStream_t);
  cublasStatus_t status = function_not_found();
  if (async) {
    if (async_t fn = cublas_symbol<async_t>(symbol)) {
      status = fn(rows, cols, elemSize, packed.data(), lda, B, ldb, stream);
    }
  } else if (sync_t fn = cublas_symbol<sync_t>(symbol)) {
    status = fn(rows, cols, elemSize, packed.data(), lda, B, ldb);
  }
  return write_status(conn, request_id, status);
}

template <typename Index>
int handle_get_matrix(conn_t *conn, const char *symbol, bool async) {
  Index rows, cols, elemSize, lda;
  const void *A;
  cudaStream_t stream = nullptr;
  if (rpc_read(conn, &rows, sizeof(rows)) < 0 ||
      rpc_read(conn, &cols, sizeof(cols)) < 0 ||
      rpc_read(conn, &elemSize, sizeof(elemSize)) < 0 ||
      rpc_read(conn, &A, sizeof(A)) < 0 ||
      rpc_read(conn, &lda, sizeof(lda)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(static_cast<size_t>(rows) * cols *
                                    elemSize);
  const Index ldb = rows > 0 ? rows : 1;
  using sync_t = cublasStatus_t (*)(Index, Index, Index, const void *, Index,
                                    void *, Index);
  using async_t = cublasStatus_t (*)(Index, Index, Index, const void *, Index,
                                     void *, Index, cudaStream_t);
  cublasStatus_t status = function_not_found();
  if (async) {
    if (async_t fn = cublas_symbol<async_t>(symbol)) {
      status = fn(rows, cols, elemSize, A, lda, packed.data(), ldb, stream);
    }
  } else if (sync_t fn = cublas_symbol<sync_t>(symbol)) {
    status = fn(rows, cols, elemSize, A, lda, packed.data(), ldb);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      (status == CUBLAS_STATUS_SUCCESS && !packed.empty() &&
       rpc_write(conn, packed.data(), packed.size()) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cublasSetVector(conn_t *conn) {
  return handle_set_vector<int>(conn, "cublasSetVector", false);
}
int handle_cublasGetVector(conn_t *conn) {
  return handle_get_vector<int>(conn, "cublasGetVector", false);
}
int handle_cublasSetMatrix(conn_t *conn) {
  return handle_set_matrix<int>(conn, "cublasSetMatrix", false);
}
int handle_cublasGetMatrix(conn_t *conn) {
  return handle_get_matrix<int>(conn, "cublasGetMatrix", false);
}
int handle_cublasSetVectorAsync(conn_t *conn) {
  return handle_set_vector<int>(conn, "cublasSetVectorAsync", true);
}
int handle_cublasGetVectorAsync(conn_t *conn) {
  return handle_get_vector<int>(conn, "cublasGetVectorAsync", true);
}
int handle_cublasSetMatrixAsync(conn_t *conn) {
  return handle_set_matrix<int>(conn, "cublasSetMatrixAsync", true);
}
int handle_cublasGetMatrixAsync(conn_t *conn) {
  return handle_get_matrix<int>(conn, "cublasGetMatrixAsync", true);
}

#if CUBLAS_VERSION >= 120000
int handle_cublasSetVector_64(conn_t *conn) {
  return handle_set_vector<int64_t>(conn, "cublasSetVector_64", false);
}
int handle_cublasGetVector_64(conn_t *conn) {
  return handle_get_vector<int64_t>(conn, "cublasGetVector_64", false);
}
int handle_cublasSetMatrix_64(conn_t *conn) {
  return handle_set_matrix<int64_t>(conn, "cublasSetMatrix_64", false);
}
int handle_cublasGetMatrix_64(conn_t *conn) {
  return handle_get_matrix<int64_t>(conn, "cublasGetMatrix_64", false);
}
int handle_cublasSetVectorAsync_64(conn_t *conn) {
  return handle_set_vector<int64_t>(conn, "cublasSetVectorAsync_64", true);
}
int handle_cublasGetVectorAsync_64(conn_t *conn) {
  return handle_get_vector<int64_t>(conn, "cublasGetVectorAsync_64", true);
}
int handle_cublasSetMatrixAsync_64(conn_t *conn) {
  return handle_set_matrix<int64_t>(conn, "cublasSetMatrixAsync_64", true);
}
int handle_cublasGetMatrixAsync_64(conn_t *conn) {
  return handle_get_matrix<int64_t>(conn, "cublasGetMatrixAsync_64", true);
}
#endif

#include "codegen/gen_cublas_server.inc"
