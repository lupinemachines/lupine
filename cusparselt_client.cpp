// cuSPARSELt (libcusparseLt) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcusparseLt, sent on
// the CUDA driver shim's connections like the cuBLAS shims. The library links
// its objects to each other by address, so every object lives on the server
// and the caller's storage holds its address there. Nearly all of the surface
// is generated; this file keeps what the generated routing and scalar
// marshalling consult before a call, and the calls that return a static
// string.

#include <cuda_runtime_api.h>
#include <cusparseLt.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

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
// Objects
// ---------------------------------------------------------------------------

// The caller's storage for an object holds the object's address in the
// server's library, which is what travels with every call that names it.
const void *remote_object(const void *storage) {
  const void *remote = nullptr;
  std::memcpy(&remote, storage, sizeof(remote));
  return remote;
}

// Each object is recorded against the connection that created it. A plan also
// keeps the matmul descriptor it was built from, and a matmul descriptor keeps
// whether vector scaling moved alpha or beta to the device, because the
// generated marshalling decides before a call whether a scalar's bytes or its
// address travel. Two servers can hand out the same address; the later
// creation wins.
struct object_state {
  conn_t *conn = nullptr;
  const void *matmul_descriptor = nullptr;
  bool alpha_on_device = false;
  bool beta_on_device = false;
};

std::mutex objects_mutex;
std::unordered_map<const void *, object_state> objects;

// A call without an object goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(const void *storage) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(remote_object(storage));
  return it == objects.end() ? nullptr : it->second.conn;
}

void note_object(conn_t *conn, const void *storage) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  objects[remote_object(storage)] = object_state{conn};
}

void forget_object(const void *storage) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  objects.erase(remote_object(storage));
}

void note_plan_descriptor(const void *plan, const void *matmul_descriptor) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(remote_object(plan));
  if (it != objects.end()) {
    it->second.matmul_descriptor = remote_object(matmul_descriptor);
  }
}

void note_scaling(const void *matmul_descriptor,
                  cusparseLtMatmulDescAttribute_t attribute, const void *data,
                  size_t size) {
  const bool alpha = attribute == CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING;
  if ((!alpha && attribute != CUSPARSELT_MATMUL_BETA_VECTOR_SCALING) ||
      data == nullptr || size < sizeof(int)) {
    return;
  }
  int value = 0;
  std::memcpy(&value, data, sizeof(value));
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(remote_object(matmul_descriptor));
  if (it == objects.end()) {
    return;
  }
  if (alpha) {
    it->second.alpha_on_device = value != 0;
  } else {
    it->second.beta_on_device = value != 0;
  }
}

// alpha and beta are host scalars the width of the compute type's, which is
// four bytes for every compute type the library accepts, unless the plan's
// matmul descriptor scales by a device vector.
bool scalar_on_host(const void *plan, const char *name) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(remote_object(plan));
  if (it == objects.end()) {
    return true;
  }
  auto descriptor = objects.find(it->second.matmul_descriptor);
  if (descriptor == objects.end()) {
    return true;
  }
  return std::strcmp(name, "alpha") == 0 ? !descriptor->second.alpha_on_device
                                         : !descriptor->second.beta_on_device;
}

} // namespace

#include "codegen/gen_cusparselt_client.inc"

// ---------------------------------------------------------------------------
// Static strings
// ---------------------------------------------------------------------------

#if CUSPARSELT_VERSION >= 700
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

extern "C" const char *cusparseLtGetErrorName(cusparseStatus_t status) {
  return status_text(RPC_cusparseLtGetErrorName, status);
}

extern "C" const char *cusparseLtGetErrorString(cusparseStatus_t status) {
  return status_text(RPC_cusparseLtGetErrorString, status);
}
#endif
