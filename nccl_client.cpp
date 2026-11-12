// NCCL (libnccl) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libnccl, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps
// the APIs ordered. A lane is one server thread, which is what NCCL's
// thread-local groups need. Nearly all of the surface is generated. The calls
// in this file carry something the generated marshalling cannot: the group
// brackets, a device list, an allocation's owner, or a string the library
// keeps.

#include <cuda.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

// nccl_device.h declares the device API's host half; these opaque handles are
// all of that header the shim needs.
typedef struct ncclDevComm ncclDevComm_t;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;

static ncclResult_t lupine_rpc_ncclGroupStart(conn_t *conn, bool submit_async);

namespace {

struct group_state {
  int depth = 0;
  conn_t *conn = nullptr;
  ncclResult_t error = ncclSuccess;
};
thread_local group_state group;

// A call the group refused reports why; otherwise the connection failed.
ncclResult_t rpc_error() {
  return group.error != ncclSuccess ? group.error : ncclSystemError;
}

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
// rpc.h's scope for a request on no object.
constexpr uintptr_t RPC_SCOPE_NONE = 0;
int rpc_write_start_request_on(conn_t *conn, int op, uintptr_t key) {
  return lupine_rpc_write_start_request_on(conn, op, key);
}
int rpc_write_start_request_on_stream(conn_t *conn, int op, void *stream) {
  return lupine_rpc_write_start_request_on_stream(conn, op, stream);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_write_end(conn_t *conn) { return lupine_rpc_write_end(conn); }
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }
void rpc_shutdown_transport_socket(conn_t *conn) {
  lupine_rpc_shutdown_transport_socket(conn);
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

// A thread's group reaches one server. The server opens it when the first
// call inside it arrives and closes it at the outermost ncclGroupEnd, whose
// result covers every call in between. A group cannot also reach a second
// server: that server's ncclGroupEnd may wait for the first's, and one client
// thread cannot hold both requests open; ranks behind different servers use a
// thread each, as ranks in different processes do natively.
void record_group_error(ncclResult_t err) {
  if (group.depth != 0 && group.error == ncclSuccess) {
    group.error = err;
  }
}

conn_t *enter_group(conn_t *conn, ncclResult_t null_error = ncclSystemError) {
  if (group.depth == 0) {
    return conn;
  }
  if (conn == nullptr) {
    record_group_error(null_error);
    return nullptr;
  }
  if (group.conn == conn) {
    return conn;
  }
  if (group.conn != nullptr) {
    record_group_error(ncclInvalidUsage);
    return nullptr;
  }
  if (lupine_rpc_ncclGroupStart(conn, true) != ncclSuccess) {
    record_group_error(ncclSystemError);
    return nullptr;
  }
  group.conn = conn;
  return conn;
}

// NCCL only queues a call made inside a group, so nothing waits for its
// answer; outside a group the call may block on a peer and is a request.
bool submit_async(conn_t *conn) {
  return group.depth != 0 && group.conn == conn;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

std::mutex handles_mutex;
std::unordered_map<const void *, conn_t *> handles;

// The runtime's current device is the device of the driver context the
// program's runtime made current on this thread, so its server is the
// context's. Before any context exists it is device zero, as cudaGetDevice
// answers, and the driver shim's own default route must not stand in for it;
// the device table needs the driver initialized, which the library's first
// call does natively.
conn_t *device_connection() {
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr) {
    return lupine_rpc_conn_for_current_context();
  }
  if (cuInit(0) != CUDA_SUCCESS) {
    return nullptr;
  }
  int device = 0;
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection() { return enter_group(device_connection()); }

// A null communicator is the new rank of ncclCommGrow, which runs on the
// current device.
conn_t *handle_connection(const void *handle) {
  if (handle == nullptr) {
    return device_connection();
  }
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(handle);
  return it == handles.end() ? nullptr : it->second;
}

conn_t *connection_for_handle(const void *handle) {
  if (handle == nullptr) {
    return connection();
  }
  return enter_group(handle_connection(handle), ncclInvalidArgument);
}

void note_handle_owner(conn_t *conn, const void *handle) {
  if (handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles[handle] = conn;
}

// A destroyed handle's address may come back from a later create, which
// records it afresh.
void forget_handle(const void *handle) {
  if (handle == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles.erase(handle);
}

// The server's library allocated the range, so other shims route calls on
// it by this record.
void note_allocation(conn_t *conn, void *ptr, size_t size) {
  lupine_note_deviceptr_allocation(reinterpret_cast<unsigned long long>(ptr),
                                   size, conn);
}

bool scalar_on_host(ncclScalarResidence_t residence, const char *) {
  return residence == ncclScalarHostImmediate;
}

// By enumerator value, which NCCL keeps stable, so older headers that lack
// the newer types still compile.
size_t nccl_type_width(ncclDataType_t type) {
  switch (static_cast<int>(type)) {
  case 2: // ncclInt32
  case 3: // ncclUint32
  case 7: // ncclFloat32
    return 4;
  case 4: // ncclInt64
  case 5: // ncclUint64
  case 8: // ncclFloat64
    return 8;
  case 6: // ncclFloat16
  case 9: // ncclBfloat16
    return 2;
  default:
    return 1;
  }
}

} // namespace

#include "codegen/gen_nccl_client.inc"

// ---------------------------------------------------------------------------
// Hand-written calls
// ---------------------------------------------------------------------------

extern "C" ncclResult_t ncclGroupStart() {
  ++group.depth;
  return ncclSuccess;
}

template <typename Finish> static ncclResult_t finish_group(Finish &&finish) {
  if (group.depth == 0) {
    return ncclInvalidUsage;
  }
  if (--group.depth != 0) {
    return ncclSuccess;
  }
  const group_state ended = group;
  group = group_state{};
  if (ended.conn == nullptr) {
    return ended.error;
  }
  const ncclResult_t result = finish(ended.conn);
  return ended.error != ncclSuccess ? ended.error : result;
}

extern "C" ncclResult_t ncclGroupEnd() {
  return finish_group(lupine_rpc_ncclGroupEnd);
}

#if NCCL_VERSION_CODE >= 22203
extern "C" ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
  const ncclResult_t result = finish_group([simInfo](conn_t *conn) {
    return lupine_rpc_ncclGroupSimulateEnd(conn, simInfo);
  });
  return (simInfo == nullptr && result == ncclSuccess) ? ncclInvalidArgument
                                                       : result;
}
#endif

#if NCCL_VERSION_CODE >= 21903
extern "C" ncclResult_t ncclMemFree(void *ptr) {
  if (ptr == nullptr) {
    return ncclSuccess;
  }
  conn_t *conn =
      lupine_rpc_conn_for_deviceptr(reinterpret_cast<unsigned long long>(ptr));
  const ncclResult_t result = lupine_rpc_ncclMemFree(conn, ptr);
  if (result == ncclSuccess) {
    lupine_forget_deviceptr_owner(reinterpret_cast<unsigned long long>(ptr));
  }
  return result;
}
#endif

// The server's library takes its own device ordinals, and a clique it creates
// in one call lives in one process, so every device must be on one server.
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev,
                                        const int *devlist) {
  if (comm == nullptr || ndev < 1) {
    record_group_error(ncclInvalidArgument);
    return ncclInvalidArgument;
  }
  int stack_devices[16];
  std::vector<int> heap_devices;
  int *devices = stack_devices;
  if (static_cast<size_t>(ndev) >
      sizeof(stack_devices) / sizeof(stack_devices[0])) {
    heap_devices.resize(static_cast<size_t>(ndev));
    devices = heap_devices.data();
  }
  conn_t *conn = nullptr;
  for (int i = 0; i < ndev; ++i) {
    devices[i] = devlist != nullptr ? devlist[i] : i;
    conn_t *device_conn = lupine_rpc_conn_for_device(&devices[i]);
    if (device_conn == nullptr) {
      record_group_error(ncclInvalidArgument);
      return ncclInvalidArgument;
    }
    if (conn != nullptr && device_conn != conn) {
      record_group_error(ncclInvalidUsage);
      return ncclInvalidUsage;
    }
    conn = device_conn;
  }
  conn = enter_group(conn);
  const ncclResult_t result =
      lupine_rpc_ncclCommInitAll(conn, comm, ndev, devices);
  if (result == ncclSuccess) {
    for (int i = 0; i < ndev; ++i) {
      note_handle_owner(conn, comm[i]);
    }
  }
  return result;
}

namespace {

constexpr uint32_t kMaxText = 1U << 20;

// A response carries the call's status, then its strings as a count and, for
// each, its length and bytes; the count is zero after a failure.
ncclResult_t read_strings(conn_t *conn, std::vector<std::string> *out) {
  ncclResult_t status = ncclSystemError;
  uint32_t count = 0;
  if (rpc_wait_for_response(conn) < 0) {
    return ncclSystemError;
  }
  if (rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 || count > kMaxText) {
    rpc_shutdown_transport_socket(conn);
    (void)rpc_read_end(conn);
    return ncclSystemError;
  }
  out->assign(count, std::string());
  for (std::string &text : *out) {
    uint32_t length = 0;
    if (rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxText) {
      rpc_shutdown_transport_socket(conn);
      (void)rpc_read_end(conn);
      return ncclSystemError;
    }
    text.resize(length);
    if (length != 0 && rpc_read(conn, &text[0], length) < 0) {
      rpc_shutdown_transport_socket(conn);
      (void)rpc_read_end(conn);
      return ncclSystemError;
    }
  }
  return rpc_read_end(conn) < 0 ? ncclSystemError : status;
}

ncclResult_t finish_single_string(conn_t *conn, std::string *out) {
  ncclResult_t status = ncclSystemError;
  uint32_t count = 0;
  if (rpc_wait_for_response(conn) < 0) {
    return ncclSystemError;
  }
  if (rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      (status == ncclSuccess ? count != 1 : count != 0)) {
    rpc_shutdown_transport_socket(conn);
    (void)rpc_read_end(conn);
    return ncclSystemError;
  }
  if (count == 1) {
    uint32_t length = 0;
    if (rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxText) {
      rpc_shutdown_transport_socket(conn);
      (void)rpc_read_end(conn);
      return ncclSystemError;
    }
    out->resize(length);
    if (length != 0 && rpc_read(conn, &(*out)[0], length) < 0) {
      rpc_shutdown_transport_socket(conn);
      (void)rpc_read_end(conn);
      return ncclSystemError;
    }
  }
  return rpc_read_end(conn) < 0 ? ncclSystemError : status;
}

template <typename Arg>
ncclResult_t read_single_string(conn_t *conn, int op, const Arg &arg,
                                std::string *out) {
  if (conn == nullptr ||
      rpc_write_start_request_on(conn, op, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &arg, sizeof(arg)) < 0) {
    return ncclSystemError;
  }
  return finish_single_string(conn, out);
}

} // namespace

// The server's library owns the strings, so each is fetched once and kept for
// the life of the process.
extern "C" const char *ncclGetErrorString(ncclResult_t result) {
  static std::mutex mutex;
  static std::unordered_map<int, std::string> texts;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = texts.find(result);
    if (it != texts.end()) {
      return it->second.c_str();
    }
  }
  std::string text;
  if (read_single_string(device_connection(), RPC_ncclGetErrorString, result,
                         &text) != ncclSuccess) {
    return "unknown result code";
  }
  std::lock_guard<std::mutex> lock(mutex);
  return texts.emplace(result, std::move(text)).first->second.c_str();
}

// Valid until the thread's next call, which outlives the library's own
// guarantee of until the next error.
extern "C" const char *ncclGetLastError(ncclComm_t comm) {
  thread_local std::string text;
  text.clear();
  (void)read_single_string(handle_connection(comm), RPC_ncclGetLastError, comm,
                           &text);
  return text.c_str();
}

#if NCCL_VERSION_CODE >= 23007
// The parameter system keeps each string only until the thread's next call
// of the same kind; so does the client.
extern "C" ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char **out) {
  if (out == nullptr) {
    return ncclInvalidArgument;
  }
  thread_local std::string text;
  const ncclResult_t status =
      read_single_string(handle_connection(h), RPC_ncclParamGetStr, h, &text);
  if (status == ncclSuccess) {
    *out = text.c_str();
  }
  return status;
}

extern "C" ncclResult_t
ncclParamGetParameter(const char *key, const char **value, int *valueLen) {
  if (key == nullptr || value == nullptr || valueLen == nullptr) {
    return ncclInvalidArgument;
  }
  const size_t raw_length = std::strlen(key);
  if (raw_length >= kMaxText) {
    return ncclInvalidArgument;
  }
  thread_local std::string text;
  conn_t *conn = device_connection();
  const uint32_t key_length = static_cast<uint32_t>(raw_length + 1);
  if (conn == nullptr ||
      rpc_write_start_request_on(conn, RPC_ncclParamGetParameter,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &key_length, sizeof(key_length)) < 0 ||
      rpc_write(conn, key, key_length) < 0) {
    return ncclSystemError;
  }
  const ncclResult_t status = finish_single_string(conn, &text);
  if (status == ncclSuccess) {
    *value = text.c_str();
    *valueLen = static_cast<int>(text.size());
  }
  return status;
}

extern "C" ncclResult_t ncclParamGetAllParameterKeys(const char ***table,
                                                     int *tableLen) {
  if (table == nullptr || tableLen == nullptr) {
    return ncclInvalidArgument;
  }
  thread_local std::vector<std::string> keys;
  thread_local std::vector<const char *> pointers;
  conn_t *conn = device_connection();
  std::vector<std::string> strings;
  if (conn == nullptr ||
      rpc_write_start_request_on(conn, RPC_ncclParamGetAllParameterKeys,
                                 RPC_SCOPE_NONE) < 0) {
    return ncclSystemError;
  }
  const ncclResult_t status = read_strings(conn, &strings);
  if (status == ncclSuccess) {
    keys = std::move(strings);
    pointers.clear();
    for (const std::string &key : keys) {
      pointers.push_back(key.c_str());
    }
    *table = pointers.data();
    *tableLen = static_cast<int>(pointers.size());
  }
  return status;
}
#endif
