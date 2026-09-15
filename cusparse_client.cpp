// cuSPARSE (libcusparse) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcusparse, sent on
// the CUDA driver shim's connections: the server child that owns the driver
// and runtime state runs the library too, and one lane per client thread keeps
// the three APIs ordered. Nearly all of the surface is generated. This file
// keeps the table the generated routing and scalar marshalling consult, and
// the calls that return a static string.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "codegen/gen_rpc_ids.h"
#include "cublas_scalar.h"
#include "cuda_client_rpc.h"
#include "cusparse_compat.h"

namespace {

cusparseStatus_t rpc_error() { return CUSPARSE_STATUS_NOT_INITIALIZED; }

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
// Handles, descriptors and infos
// ---------------------------------------------------------------------------

// Each is an address in one server's library, recorded against the connection
// that created it. The generated marshalling decides before a call whether a
// scalar's bytes or its address travel, so a handle keeps its pointer mode,
// and a descriptor keeps the value type that sizes cusparseAxpby's and
// cusparseRot's scalars. Two servers can hand out the same address; the later
// creation wins.
struct owner_state {
  conn_t *conn;
  bool host_pointers;
  cudaDataType value_type;
};

std::mutex owners_mutex;
std::unordered_map<const void *, owner_state> owners;

// A call without a handle goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(const void *object) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  auto it = owners.find(object);
  return it == owners.end() ? nullptr : it->second.conn;
}

void note_handle_owner(conn_t *conn, const void *object,
                       cudaDataType value_type = CUDA_C_64F) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  owners[object] = {conn, true, value_type};
}

void forget_handle_owner(const void *object) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  owners.erase(object);
}

void note_pointer_mode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  auto it = owners.find(handle);
  if (it != owners.end()) {
    it->second.host_pointers = mode == CUSPARSE_POINTER_MODE_HOST;
  }
}

// cuSPARSE places every scalar of a call together, so the name is unused.
bool scalar_on_host(cusparseHandle_t handle, const char *) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  auto it = owners.find(handle);
  return it == owners.end() || it->second.host_pointers;
}

size_t descriptor_value_width(const void *descriptor) {
  std::lock_guard<std::mutex> lock(owners_mutex);
  auto it = owners.find(descriptor);
  return data_type_width(it == owners.end() ? CUDA_C_64F
                                            : it->second.value_type);
}

} // namespace

#include "codegen/gen_cusparse_client.inc"

// ---------------------------------------------------------------------------
// Static strings
// ---------------------------------------------------------------------------

namespace {

// The server's library owns the status strings, so each is fetched once and
// kept for the life of the process.
const char *status_text(int op, cusparseStatus_t status) {
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
  conn_t *conn = connection();
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0) {
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

extern "C" const char *cusparseGetErrorName(cusparseStatus_t status) {
  return status_text(RPC_cusparseGetErrorName, status);
}

extern "C" const char *cusparseGetErrorString(cusparseStatus_t status) {
  return status_text(RPC_cusparseGetErrorString, status);
}
