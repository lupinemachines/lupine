// cuSOLVERMg (libcusolverMg) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcusolverMg on the
// CUDA driver shim's connections, like the cuSOLVER shim. This file keeps the
// object table the generated routing consults, the device count that sizes a
// distributed matrix's array of device pointers, and the grid creation that
// picks its server from the devices it names.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cublas_scalar.h"
#include "cuda_client_rpc.h"
#include "cusolver_compat.h"

namespace {

cusolverStatus_t rpc_error() { return CUSOLVER_STATUS_NOT_INITIALIZED; }

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

// A handle, grid or matrix descriptor is an address in one server's library,
// recorded against the connection that created it. A grid keeps its device
// count and a descriptor the count of the grid it was created on, since a
// distributed matrix is an array of one device pointer per grid device.
struct object_state {
  conn_t *conn = nullptr;
  int64_t devices = 0;
};

std::mutex objects_mutex;
std::unordered_map<const void *, object_state> objects;

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
  objects[object] = object_state{conn, 0};
}

void forget_handle_owner(const void *object) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  objects.erase(object);
}

void note_matrix_grid(const void *desc, const void *grid) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto from = objects.find(grid);
  auto to = objects.find(desc);
  if (from != objects.end() && to != objects.end()) {
    to->second.devices = from->second.devices;
  }
}

int64_t mg_devices(const void *desc) {
  std::lock_guard<std::mutex> lock(objects_mutex);
  auto it = objects.find(desc);
  return it == objects.end() ? 0 : it->second.devices;
}

// The caller names virtual ordinals; the server's library wants its own. A
// handle or grid cannot span servers, so each is rewritten as if on the first
// device's server.
template <typename T>
std::vector<T> server_ordinals(const T *ids, int64_t count) {
  std::vector<T> server(ids, ids + count);
  for (T &id : server) {
    int device = static_cast<int>(id);
    lupine_rpc_conn_for_device(&device);
    id = static_cast<T>(device);
  }
  return server;
}

} // namespace

#include "codegen/gen_cusolvermg_client.inc"

extern "C" cusolverStatus_t
cusolverMgCreateDeviceGrid(cudaLibMgGrid_t *grid, int32_t numRowDevices,
                           int32_t numColDevices, const int32_t deviceId[],
                           cusolverMgGridMapping_t mapping) {
  const int64_t count = static_cast<int64_t>(numRowDevices) * numColDevices;
  int first = count > 0 ? deviceId[0] : 0;
  conn_t *conn = lupine_rpc_conn_for_device(&first);
  std::vector<int32_t> server = server_ordinals(deviceId, count);
  cusolverStatus_t status = lupine_rpc_cusolverMgCreateDeviceGrid(
      conn, grid, numRowDevices, numColDevices, server.data(), mapping);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> lock(objects_mutex);
    objects[*grid] = object_state{conn, count};
  }
  return status;
}
