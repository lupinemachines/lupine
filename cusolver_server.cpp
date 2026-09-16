#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "cusolver_compat.h"
#include "rpc.h"

namespace {

cusolverStatus_t function_not_found() { return CUSOLVER_STATUS_NOT_SUPPORTED; }

// The library this server was compiled against; the handlers marshal its
// enums and structure sizes, so another major on the machine is not a match.
void *cusolver_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUSOLVER_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cusolver64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcusolver.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cusolver_symbol(const char *name) {
  void *lib = cusolver_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

} // namespace

// The history is a (maxiters + 1) by 2 column-major array of doubles:
// iteration numbers, then residual norms.
int handle_cusolverDnIRSInfosGetResidualHistory(conn_t *conn) {
  cusolverDnIRSInfos_t infos;
  if (rpc_read(conn, &infos, sizeof(infos)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using history_t = cusolverStatus_t (*)(cusolverDnIRSInfos_t, void **);
  using max_iters_t =
      cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t *);
  history_t history_fn =
      cusolver_symbol<history_t>("cusolverDnIRSInfosGetResidualHistory");
  max_iters_t max_iters_fn =
      cusolver_symbol<max_iters_t>("cusolverDnIRSInfosGetMaxIters");
  void *history = nullptr;
  cusolver_int_t max_iters = 0;
  cusolverStatus_t status = history_fn == nullptr || max_iters_fn == nullptr
                                ? function_not_found()
                                : max_iters_fn(infos, &max_iters);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    status = history_fn(infos, &history);
  }
  const uint64_t count = status == CUSOLVER_STATUS_SUCCESS && history != nullptr
                             ? 2 * (static_cast<uint64_t>(max_iters) + 1)
                             : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (count != 0 && rpc_write(conn, history, count * sizeof(double)) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

namespace {

// The arrays are CSR: each row pointer has n + 1 entries and its last is the
// factor's nnz. Every array is sent before any is freed, since the response
// references them until it is written.
struct host_factor {
  int *row_ptr = nullptr;
  int *col_ind = nullptr;
  double *val = nullptr;
  uint64_t rows = 0;
  uint64_t nnz = 0;

  void measure(int n) {
    rows = row_ptr == nullptr ? 0 : static_cast<uint64_t>(n) + 1;
    nnz = row_ptr == nullptr ? 0 : static_cast<uint64_t>(row_ptr[n]);
  }

  int write(conn_t *conn) const {
    return rpc_write(conn, &rows, sizeof(rows)) < 0 ||
                   (rows != 0 &&
                    rpc_write(conn, row_ptr, rows * sizeof(int)) < 0) ||
                   rpc_write(conn, &nnz, sizeof(nnz)) < 0 ||
                   (nnz != 0 &&
                    rpc_write(conn, col_ind, nnz * sizeof(int)) < 0) ||
                   rpc_write(conn, &nnz, sizeof(nnz)) < 0 ||
                   (nnz != 0 && rpc_write(conn, val, nnz * sizeof(double)) < 0)
               ? -1
               : 0;
  }

  void release() {
    free(row_ptr);
    free(col_ind);
    free(val);
  }
};

} // namespace

int handle_cusolverRfExtractBundledFactorsHost(conn_t *conn) {
  cusolverRfHandle_t handle;
  int n = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &n, sizeof(n)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cusolverStatus_t (*)(cusolverRfHandle_t, int *, int **, int **,
                                    double **);
  fn_t fn = cusolver_symbol<fn_t>("cusolverRfExtractBundledFactorsHost");
  int nnz = 0;
  host_factor factor;
  const cusolverStatus_t status =
      fn == nullptr
          ? function_not_found()
          : fn(handle, &nnz, &factor.row_ptr, &factor.col_ind, &factor.val);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    factor.measure(n);
  }
  int result = rpc_write_start_response(conn, request_id) < 0 ||
                       rpc_write(conn, &status, sizeof(status)) < 0 ||
                       (status == CUSOLVER_STATUS_SUCCESS &&
                        (rpc_write(conn, &nnz, sizeof(nnz)) < 0 ||
                         factor.write(conn) < 0)) ||
                       rpc_write_end(conn) < 0
                   ? -1
                   : 0;
  factor.release();
  return result;
}

int handle_cusolverRfExtractSplitFactorsHost(conn_t *conn) {
  cusolverRfHandle_t handle;
  int n = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &n, sizeof(n)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t =
      cusolverStatus_t (*)(cusolverRfHandle_t, int *, int **, int **, double **,
                           int *, int **, int **, double **);
  fn_t fn = cusolver_symbol<fn_t>("cusolverRfExtractSplitFactorsHost");
  int nnz_l = 0;
  int nnz_u = 0;
  host_factor lower;
  host_factor upper;
  const cusolverStatus_t status =
      fn == nullptr
          ? function_not_found()
          : fn(handle, &nnz_l, &lower.row_ptr, &lower.col_ind, &lower.val,
               &nnz_u, &upper.row_ptr, &upper.col_ind, &upper.val);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    lower.measure(n);
    upper.measure(n);
  }
  int result = rpc_write_start_response(conn, request_id) < 0 ||
                       rpc_write(conn, &status, sizeof(status)) < 0 ||
                       (status == CUSOLVER_STATUS_SUCCESS &&
                        (rpc_write(conn, &nnz_l, sizeof(nnz_l)) < 0 ||
                         lower.write(conn) < 0 ||
                         rpc_write(conn, &nnz_u, sizeof(nnz_u)) < 0 ||
                         upper.write(conn) < 0)) ||
                       rpc_write_end(conn) < 0
                   ? -1
                   : 0;
  lower.release();
  upper.release();
  return result;
}

int handle_cusolverRfBatchSetupHost(conn_t *conn) {
  cusolverRfHandle_t handle;
  int batch_size = 0;
  int n = 0;
  int nnz_a = 0;
  int nnz_l = 0;
  int nnz_u = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &batch_size, sizeof(batch_size)) < 0 ||
      rpc_read(conn, &n, sizeof(n)) < 0 ||
      rpc_read(conn, &nnz_a, sizeof(nnz_a)) < 0 ||
      rpc_read(conn, &nnz_l, sizeof(nnz_l)) < 0 ||
      rpc_read(conn, &nnz_u, sizeof(nnz_u)) < 0) {
    return -1;
  }
  const size_t rows = static_cast<size_t>(n) + 1;
  std::vector<int> row_a(rows), col_a(nnz_a), row_l(rows), col_l(nnz_l),
      row_u(rows), col_u(nnz_u), p(n), q(n);
  std::vector<double> values(static_cast<size_t>(batch_size) * nnz_a),
      val_l(nnz_l), val_u(nnz_u);
  std::vector<double *> val_a(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    val_a[i] = values.data() + static_cast<size_t>(i) * nnz_a;
  }
  if (rpc_read(conn, row_a.data(), rows * sizeof(int)) < 0 ||
      rpc_read(conn, col_a.data(), col_a.size() * sizeof(int)) < 0 ||
      rpc_read(conn, values.data(), values.size() * sizeof(double)) < 0 ||
      rpc_read(conn, row_l.data(), rows * sizeof(int)) < 0 ||
      rpc_read(conn, col_l.data(), col_l.size() * sizeof(int)) < 0 ||
      rpc_read(conn, val_l.data(), val_l.size() * sizeof(double)) < 0 ||
      rpc_read(conn, row_u.data(), rows * sizeof(int)) < 0 ||
      rpc_read(conn, col_u.data(), col_u.size() * sizeof(int)) < 0 ||
      rpc_read(conn, val_u.data(), val_u.size() * sizeof(double)) < 0 ||
      rpc_read(conn, p.data(), p.size() * sizeof(int)) < 0 ||
      rpc_read(conn, q.data(), q.size() * sizeof(int)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cusolverStatus_t (*)(int, int, int, int *, int *, double **, int,
                                    int *, int *, double *, int, int *, int *,
                                    double *, int *, int *, cusolverRfHandle_t);
  fn_t fn = cusolver_symbol<fn_t>("cusolverRfBatchSetupHost");
  const cusolverStatus_t status =
      fn == nullptr ? function_not_found()
                    : fn(batch_size, n, nnz_a, row_a.data(), col_a.data(),
                         val_a.data(), nnz_l, row_l.data(), col_l.data(),
                         val_l.data(), nnz_u, row_u.data(), col_u.data(),
                         val_u.data(), p.data(), q.data(), handle);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#include "codegen/gen_cusolver_server.inc"
