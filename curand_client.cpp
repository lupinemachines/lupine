// cuRAND (libcurand) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcurand, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps
// the three APIs ordered. Nearly all of the surface is generated. The calls
// in this file hand out the library's static Sobol tables.

#include <cuda_runtime_api.h>
#include <curand.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

curandStatus_t rpc_error() { return CURAND_STATUS_INITIALIZATION_FAILED; }

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
// Generators and distributions
// ---------------------------------------------------------------------------

// A generator or discrete distribution is created on the runtime's current
// device and routes every later call to that connection. Whether a generator
// is a host one is kept here because the generated marshalling decides before
// each call whether its output comes back in the response.
struct handle_state {
  conn_t *conn;
  bool host;
};

std::mutex handles_mutex;
std::unordered_map<void *, handle_state> handles;

// A call without a handle goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(void *handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(handle);
  return it == handles.end() ? nullptr : it->second.conn;
}

void note_handle_owner(conn_t *conn, void *handle, bool host = false) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles[handle] = {conn, host};
}

// A destroyed handle's address may come back from a later create, which
// records it afresh.
void forget_handle(void *handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles.erase(handle);
}

bool scalar_on_host(curandGenerator_t generator, const char *) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(generator);
  return it != handles.end() && it->second.host;
}

} // namespace

#include "codegen/gen_curand_client.inc"

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

namespace {

// The library hands out pointers to tables it keeps for the life of the
// process. Each table is fetched once and kept as long, so the pointer the
// caller gets stays valid the same way.
std::mutex tables_mutex;
std::map<std::pair<int, int>, std::vector<unsigned char>> tables;

curandStatus_t table(int op, curandDirectionVectorSet_t set, void **out) {
  std::lock_guard<std::mutex> lock(tables_mutex);
  auto it = tables.find({op, set});
  if (it == tables.end()) {
    conn_t *conn = connection();
    curandStatus_t status = rpc_error();
    uint64_t bytes = 0;
    if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
        rpc_write(conn, &set, sizeof(set)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &status, sizeof(status)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
      return rpc_error();
    }
    std::vector<unsigned char> data(bytes);
    if ((bytes != 0 && rpc_read(conn, data.data(), bytes) < 0) ||
        rpc_read_end(conn) < 0) {
      return rpc_error();
    }
    if (status != CURAND_STATUS_SUCCESS) {
      return status;
    }
    it = tables.emplace(std::make_pair(op, set), std::move(data)).first;
  }
  *out = it->second.data();
  return CURAND_STATUS_SUCCESS;
}

} // namespace

extern "C" curandStatus_t
curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[],
                            curandDirectionVectorSet_t set) {
  void *data = nullptr;
  curandStatus_t status = table(RPC_curandGetDirectionVectors32, set, &data);
  if (status == CURAND_STATUS_SUCCESS) {
    *vectors = static_cast<curandDirectionVectors32_t *>(data);
  }
  return status;
}

extern "C" curandStatus_t
curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[],
                            curandDirectionVectorSet_t set) {
  void *data = nullptr;
  curandStatus_t status = table(RPC_curandGetDirectionVectors64, set, &data);
  if (status == CURAND_STATUS_SUCCESS) {
    *vectors = static_cast<curandDirectionVectors64_t *>(data);
  }
  return status;
}

extern "C" curandStatus_t
curandGetScrambleConstants32(unsigned int **constants) {
  void *data = nullptr;
  curandStatus_t status = table(RPC_curandGetScrambleConstants32,
                                curandDirectionVectorSet_t{}, &data);
  if (status == CURAND_STATUS_SUCCESS) {
    *constants = static_cast<unsigned int *>(data);
  }
  return status;
}

extern "C" curandStatus_t
curandGetScrambleConstants64(unsigned long long **constants) {
  void *data = nullptr;
  curandStatus_t status = table(RPC_curandGetScrambleConstants64,
                                curandDirectionVectorSet_t{}, &data);
  if (status == CURAND_STATUS_SUCCESS) {
    *constants = static_cast<unsigned long long *>(data);
  }
  return status;
}
