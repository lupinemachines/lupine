// cuSOLVER (libcusolver) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcusolver, sent on
// the CUDA driver shim's connections: the server child that owns the driver
// and runtime state runs the library too, and one lane per client thread keeps
// the APIs ordered. Nearly all of the surface is generated. This file keeps
// the object table the generated routing consults, the sizes of host arrays
// that no parameter bounds, and the calls whose results are host memory the
// library allocates.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cublas_scalar.h"
#include "cuda_client_rpc.h"
#include "cusolver_compat.h"

namespace {

cusolverStatus_t rpc_error() { return CUSOLVER_STATUS_NOT_INITIALIZED; }

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
// Handles, parameter sets, infos and IRS objects
// ---------------------------------------------------------------------------

// Each is an address in one server's library, recorded against the connection
// that created it, with the sizes later calls on it need: a low-level host
// info's dimension, factor nnz and workspace from its analysis, nnz and
// buffer-info calls; an Rf handle's dimension and batch size from setup; the
// csrperm workspace a cusolverSp handle last reported; and the residual
// history copied for an IRS infos. Two servers can hand out the same address;
// the later creation wins.
struct object_state {
  conn_t *conn = nullptr;
  int n = 0;
  int nnz_l = 0;
  int nnz_u = 0;
  size_t workspace = 0;
  int batch_size = 1;
  std::vector<double> residual_history;
};

std::mutex objects_mutex;
std::unordered_map<const void *, object_state> objects;

// A call without one goes to the runtime's current device, which the runtime
// shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(const void *object) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(object);
  return it == objects.end() ? nullptr : it->second.conn;
}

void note_handle_owner(conn_t *conn, const void *object) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  objects[object] = object_state{};
  objects[object].conn = conn;
}

void forget_handle_owner(const void *object) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  objects.erase(object);
}

template <typename Fn> void update_object(const void *object, Fn fn) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(object);
  if (it != objects.end()) {
    fn(it->second);
  }
}

template <typename T, typename Fn> T read_object(const void *object, Fn fn) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(object);
  return it == objects.end() ? T{} : fn(it->second);
}

void note_sp_host_dims(const void *info, int n) {
  update_object(info, [&](object_state &state) { state.n = n; });
}

void note_sp_host_nnz(const void *info, int nnz_l, int nnz_u) {
  update_object(info, [&](object_state &state) {
    state.nnz_l = nnz_l;
    state.nnz_u = nnz_u;
  });
}

void note_sp_host_workspace(const void *info, size_t bytes) {
  update_object(info, [&](object_state &state) { state.workspace = bytes; });
}

int sp_host_n(const void *info) {
  return read_object<int>(info, [](const object_state &s) { return s.n; });
}

int sp_host_nnz_l(const void *info) {
  return read_object<int>(info, [](const object_state &s) { return s.nnz_l; });
}

int sp_host_nnz_u(const void *info) {
  return read_object<int>(info, [](const object_state &s) { return s.nnz_u; });
}

size_t sp_host_workspace(const void *info) {
  return read_object<size_t>(info,
                             [](const object_state &s) { return s.workspace; });
}

void note_csrperm_buffer(const void *handle, size_t bytes) {
  note_sp_host_workspace(handle, bytes);
}

size_t csrperm_buffer_bytes(const void *handle) {
  return sp_host_workspace(handle);
}

void note_rf_dims(const void *handle, int n, int batch_size) {
  update_object(handle, [&](object_state &state) {
    state.n = n;
    state.batch_size = batch_size;
  });
}

int rf_n(const void *handle) {
  return read_object<int>(handle, [](const object_state &s) { return s.n; });
}

int rf_batch_size(const void *handle) {
  return read_object<int>(handle,
                          [](const object_state &s) { return s.batch_size; });
}

} // namespace

#include "codegen/gen_cusolver_client.inc"

// ---------------------------------------------------------------------------
// Host memory the library allocates
// ---------------------------------------------------------------------------

// The server copies the history into the response; the copy stays valid until
// the infos is destroyed, as the library's own array does.
extern "C" cusolverStatus_t
cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos,
                                     void **residual_history) {
  conn_t *conn = connection_for_handle(infos);
  cusolverStatus_t status = rpc_error();
  uint64_t count = 0;
  std::vector<double> history;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cusolverDnIRSInfosGetResidualHistory) <
          0 ||
      rpc_write(conn, &infos, sizeof(infos)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0) {
    return rpc_error();
  }
  history.resize(count);
  if ((count != 0 &&
       rpc_read(conn, history.data(), count * sizeof(double)) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    return status;
  }
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(infos);
  if (it == objects.end()) {
    return rpc_error();
  }
  it->second.residual_history = std::move(history);
  *residual_history = it->second.residual_history.empty()
                          ? nullptr
                          : it->second.residual_history.data();
  return status;
}

namespace {

// The library allocates the factor arrays with malloc and the caller frees
// them. The server sends each one length-prefixed and frees its own; the
// client hands the caller malloc'd copies.
template <typename T> bool read_host_array(conn_t *conn, T **out) {
  uint64_t count = 0;
  if (rpc_read(conn, &count, sizeof(count)) < 0) {
    return false;
  }
  T *array = static_cast<T *>(malloc((count == 0 ? 1 : count) * sizeof(T)));
  if (array == nullptr ||
      (count != 0 && rpc_read(conn, array, count * sizeof(T)) < 0)) {
    free(array);
    return false;
  }
  *out = array;
  return true;
}

} // namespace

extern "C" cusolverStatus_t
cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t handle, int *h_nnzM,
                                    int **h_Mp, int **h_Mi, double **h_Mx) {
  conn_t *conn = connection_for_handle(handle);
  const int n = rf_n(handle);
  cusolverStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cusolverRfExtractBundledFactorsHost) <
          0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &n, sizeof(n)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0) {
    return rpc_error();
  }
  if (status == CUSOLVER_STATUS_SUCCESS &&
      (rpc_read(conn, h_nnzM, sizeof(*h_nnzM)) < 0 ||
       !read_host_array(conn, h_Mp) || !read_host_array(conn, h_Mi) ||
       !read_host_array(conn, h_Mx))) {
    return rpc_error();
  }
  return rpc_read_end(conn) < 0 ? rpc_error() : status;
}

extern "C" cusolverStatus_t cusolverRfExtractSplitFactorsHost(
    cusolverRfHandle_t handle, int *h_nnzL, int **h_csrRowPtrL,
    int **h_csrColIndL, double **h_csrValL, int *h_nnzU, int **h_csrRowPtrU,
    int **h_csrColIndU, double **h_csrValU) {
  conn_t *conn = connection_for_handle(handle);
  const int n = rf_n(handle);
  cusolverStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cusolverRfExtractSplitFactorsHost) <
          0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &n, sizeof(n)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0) {
    return rpc_error();
  }
  if (status == CUSOLVER_STATUS_SUCCESS &&
      (rpc_read(conn, h_nnzL, sizeof(*h_nnzL)) < 0 ||
       !read_host_array(conn, h_csrRowPtrL) ||
       !read_host_array(conn, h_csrColIndL) ||
       !read_host_array(conn, h_csrValL) ||
       rpc_read(conn, h_nnzU, sizeof(*h_nnzU)) < 0 ||
       !read_host_array(conn, h_csrRowPtrU) ||
       !read_host_array(conn, h_csrColIndU) ||
       !read_host_array(conn, h_csrValU))) {
    return rpc_error();
  }
  return rpc_read_end(conn) < 0 ? rpc_error() : status;
}

// Each batch entry's values travel back to back after the shared pattern.
extern "C" cusolverStatus_t cusolverRfBatchSetupHost(
    int batchSize, int n, int nnzA, int *h_csrRowPtrA, int *h_csrColIndA,
    double *h_csrValA_array[], int nnzL, int *h_csrRowPtrL, int *h_csrColIndL,
    double *h_csrValL, int nnzU, int *h_csrRowPtrU, int *h_csrColIndU,
    double *h_csrValU, int *h_P, int *h_Q, cusolverRfHandle_t handle) {
  conn_t *conn = connection_for_handle(handle);
  cusolverStatus_t status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cusolverRfBatchSetupHost) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &batchSize, sizeof(batchSize)) < 0 ||
      rpc_write(conn, &n, sizeof(n)) < 0 ||
      rpc_write(conn, &nnzA, sizeof(nnzA)) < 0 ||
      rpc_write(conn, &nnzL, sizeof(nnzL)) < 0 ||
      rpc_write(conn, &nnzU, sizeof(nnzU)) < 0 ||
      rpc_write(conn, h_csrRowPtrA, (n + 1) * sizeof(int)) < 0 ||
      rpc_write(conn, h_csrColIndA, nnzA * sizeof(int)) < 0) {
    return rpc_error();
  }
  for (int i = 0; i < batchSize; ++i) {
    if (rpc_write(conn, h_csrValA_array[i], nnzA * sizeof(double)) < 0) {
      return rpc_error();
    }
  }
  if (rpc_write(conn, h_csrRowPtrL, (n + 1) * sizeof(int)) < 0 ||
      rpc_write(conn, h_csrColIndL, nnzL * sizeof(int)) < 0 ||
      rpc_write(conn, h_csrValL, nnzL * sizeof(double)) < 0 ||
      rpc_write(conn, h_csrRowPtrU, (n + 1) * sizeof(int)) < 0 ||
      rpc_write(conn, h_csrColIndU, nnzU * sizeof(int)) < 0 ||
      rpc_write(conn, h_csrValU, nnzU * sizeof(double)) < 0 ||
      rpc_write(conn, h_P, n * sizeof(int)) < 0 ||
      rpc_write(conn, h_Q, n * sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (status == CUSOLVER_STATUS_SUCCESS) {
    note_rf_dims(handle, n, batchSize);
  }
  return status;
}
