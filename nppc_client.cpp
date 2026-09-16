// NPP core (libnppc) on the LUPINE client.
//
// The generated calls forward like every NPP library's. This file adds the
// global stream the process set, which the stream queries follow to its
// server, and the copies of the version and device names the library returns
// by address.

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "npp_client.h"

#include "codegen/gen_nppc_client.inc"

namespace {

std::atomic<cudaStream_t> global_stream{nullptr};

conn_t *global_stream_connection() {
  return connection_for_stream(global_stream.load());
}

} // namespace

extern "C" NppStatus nppSetStream(cudaStream_t hStream) {
  NppStatus status =
      lupine_rpc_nppSetStream(connection_for_stream(hStream), hStream);
  if (status == NPP_SUCCESS) {
    global_stream.store(hStream);
  }
  return status;
}

extern "C" cudaStream_t nppGetStream(void) {
  return lupine_rpc_nppGetStream(global_stream_connection());
}

extern "C" unsigned int nppGetStreamNumSMs(void) {
  return lupine_rpc_nppGetStreamNumSMs(global_stream_connection());
}

extern "C" unsigned int nppGetStreamMaxThreadsPerSM(void) {
  return lupine_rpc_nppGetStreamMaxThreadsPerSM(global_stream_connection());
}

extern "C" NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext) {
  conn_t *conn = global_stream_connection();
  NppStatus status = lupine_rpc_nppGetStreamContext(conn, pNppStreamContext);
  if (status == NPP_SUCCESS) {
    pNppStreamContext->nCudaDeviceId =
        lupine_local_device_for_remote(conn, pNppStreamContext->nCudaDeviceId);
  }
  return status;
}

extern "C" const NppLibraryVersion *nppGetLibVersion(void) {
  static NppLibraryVersion version{};
  static std::once_flag fetched;
  std::call_once(fetched, [] {
    conn_t *conn = connection();
    if (conn == nullptr ||
        rpc_write_start_request(conn, RPC_nppGetLibVersion) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &version, sizeof(version)) < 0 ||
        rpc_read_end(conn) < 0) {
      version = NppLibraryVersion{};
    }
  });
  return &version;
}

extern "C" const char *nppGetGpuName(void) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  static std::mutex mutex;
  static std::unordered_map<int, std::string> names;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = names.find(device);
    if (it != names.end()) {
      return it->second.c_str();
    }
  }
  int server_device = device;
  conn_t *conn = lupine_rpc_conn_for_device(&server_device);
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_nppGetGpuName) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0) {
    return nullptr;
  }
  std::string name(length, '\0');
  if ((length != 0 && rpc_read(conn, &name[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex);
  return names.emplace(device, std::move(name)).first->second.c_str();
}
