// nvJitLink (libnvJitLink) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libnvJitLink, sent on
// the CUDA driver shim's connections, so a link runs on the machine that loads
// its output. The calls in this file carry string arrays or nullable names,
// read the files the caller names, or copy out outputs sized by an earlier
// call; the rest is generated.

#include <cuda_runtime_api.h>
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

nvJitLinkResult rpc_error() { return NVJITLINK_ERROR_INTERNAL; }

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
// Link handles
// ---------------------------------------------------------------------------

// A link handle is created on the runtime's current device and routes every
// later call to that connection.
std::mutex handles_mutex;
std::unordered_map<void *, conn_t *> handles;

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
  return it == handles.end() ? nullptr : it->second;
}

// rpc_write queues the caller's bytes until the request is sent, so strings
// are encoded into a buffer the calling function keeps alive until then. A
// string is a presence byte, then its length and bytes when present.
void encode_string(std::vector<char> *out, const char *text) {
  const uint8_t present = text != nullptr;
  out->push_back(static_cast<char>(present));
  if (!present) {
    return;
  }
  const uint64_t length = strlen(text);
  const size_t at = out->size();
  out->resize(at + sizeof(length) + length);
  memcpy(out->data() + at, &length, sizeof(length));
  memcpy(out->data() + at + sizeof(length), text, length);
}

nvJitLinkResult status_response(conn_t *conn) {
  nvJitLinkResult status = rpc_error();
  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

// Data is a presence byte, then its size, then its bytes when present, then
// the name as a string.
nvJitLinkResult add_data(conn_t *conn, nvJitLinkHandle handle,
                         nvJitLinkInputType inputType, const void *data,
                         size_t size, const char *name) {
  const int32_t input_type = inputType;
  const uint8_t present = data != nullptr;
  const uint64_t wire_size = size;
  std::vector<char> encoded_name;
  encode_string(&encoded_name, name);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvJitLinkAddData) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &input_type, sizeof(input_type)) < 0 ||
      rpc_write(conn, &present, sizeof(present)) < 0 ||
      rpc_write(conn, &wire_size, sizeof(wire_size)) < 0 ||
      (present && size != 0 && rpc_write(conn, data, size) < 0) ||
      rpc_write(conn, encoded_name.data(), encoded_name.size()) < 0) {
    return rpc_error();
  }
  return status_response(conn);
}

// An output a Get<X> call copies out: the caller's buffer holds the size its
// Get<X>Size call reported, and nothing is written after a failure.
nvJitLinkResult copy_output(int op, nvJitLinkHandle handle, void *out) {
  conn_t *conn = connection_for_handle(handle);
  nvJitLinkResult status = rpc_error();
  uint64_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_read(conn, out, length) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

} // namespace

#include "codegen/gen_nvjitlink_client.inc"

extern "C" nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle,
                                           uint32_t numOptions,
                                           const char **options) {
  conn_t *conn = connection();
  const uint8_t present = options != nullptr;
  std::vector<char> encoded;
  for (uint32_t i = 0; present && i < numOptions; ++i) {
    encode_string(&encoded, options[i]);
  }
  nvJitLinkHandle created = nullptr;
  nvJitLinkResult status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvJitLinkCreate) < 0 ||
      rpc_write(conn, &numOptions, sizeof(numOptions)) < 0 ||
      rpc_write(conn, &present, sizeof(present)) < 0 ||
      (!encoded.empty() &&
       rpc_write(conn, encoded.data(), encoded.size()) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &created, sizeof(created)) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (created != nullptr) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    handles[created] = conn;
  }
  *handle = created;
  return status;
}

// A destroyed handle's address may come back from a later create, which
// records it afresh.
extern "C" nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle) {
  conn_t *conn = connection_for_handle(*handle);
  const nvJitLinkHandle destroyed = *handle;
  const nvJitLinkResult status = lupine_rpc_nvJitLinkDestroy(conn, handle);
  if (status == NVJITLINK_SUCCESS) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    handles.erase(destroyed);
  }
  return status;
}

extern "C" nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle,
                                            nvJitLinkInputType inputType,
                                            const void *data, size_t size,
                                            const char *name) {
  return add_data(connection_for_handle(handle), handle, inputType, data, size,
                  name);
}

// The named file is on the client: its bytes are added under the same name,
// which links the same as the library reading the file itself. A file the
// client cannot open is left to the server, whose library reports it.
extern "C" nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle,
                                            nvJitLinkInputType inputType,
                                            const char *fileName) {
  conn_t *conn = connection_for_handle(handle);
  if (fileName != nullptr) {
    std::ifstream in(fileName, std::ios::binary);
    if (in) {
      const std::string contents((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
      if (!in.bad()) {
        return add_data(conn, handle, inputType, contents.data(),
                        contents.size(), fileName);
      }
    }
  }
  const int32_t input_type = inputType;
  std::vector<char> encoded;
  encode_string(&encoded, fileName);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvJitLinkAddFile) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &input_type, sizeof(input_type)) < 0 ||
      rpc_write(conn, encoded.data(), encoded.size()) < 0) {
    return rpc_error();
  }
  return status_response(conn);
}

extern "C" nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle,
                                                   void *cubin) {
  return copy_output(RPC_nvJitLinkGetLinkedCubin, handle, cubin);
}

extern "C" nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle,
                                                 char *ptx) {
  return copy_output(RPC_nvJitLinkGetLinkedPtx, handle, ptx);
}

#if LUPINE_NVJITLINK_HAS_LTOIR
extern "C" nvJitLinkResult nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle,
                                                   void *ltoir) {
  return copy_output(RPC_nvJitLinkGetLinkedLTOIR, handle, ltoir);
}
#endif

extern "C" nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle,
                                                char *log) {
  return copy_output(RPC_nvJitLinkGetErrorLog, handle, log);
}

extern "C" nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle,
                                               char *log) {
  return copy_output(RPC_nvJitLinkGetInfoLog, handle, log);
}

// ---------------------------------------------------------------------------
// Release-suffixed names
// ---------------------------------------------------------------------------

// The header's inline wrappers call each function under the name of its
// release (__nvJitLinkCreate_13_3), and a library answers to every release of
// its major up to its own. CMake passes the header's release.
#if LUPINE_NVJITLINK_MAJOR >= 13
using release_options = const char *const *;
#else
using release_options = const char **;
#endif
#define LUPINE_NVJITLINK_NAME(call, major, minor) __##call##_##major##_##minor
#define LUPINE_NVJITLINK_RELEASE_(major, minor)                                \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkCreate, major, minor)(nvJitLinkHandle * handle,                 \
                                     uint32_t numOptions,                      \
                                     release_options options) {                \
    return nvJitLinkCreate(handle, numOptions,                                 \
                           const_cast<const char **>(options));                \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkDestroy, major, minor)(nvJitLinkHandle * handle) {              \
    return nvJitLinkDestroy(handle);                                           \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(nvJitLinkAddData, major,    \
                                                   minor)(                     \
      nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data,  \
      size_t size, const char *name) {                                         \
    return nvJitLinkAddData(handle, inputType, data, size, name);              \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkAddFile, major, minor)(nvJitLinkHandle handle,                  \
                                      nvJitLinkInputType inputType,            \
                                      const char *fileName) {                  \
    return nvJitLinkAddFile(handle, inputType, fileName);                      \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkComplete, major, minor)(nvJitLinkHandle handle) {               \
    return nvJitLinkComplete(handle);                                          \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedCubinSize, major, minor)(nvJitLinkHandle handle,       \
                                                 size_t * size) {              \
    return nvJitLinkGetLinkedCubinSize(handle, size);                          \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedCubin, major, minor)(nvJitLinkHandle handle,           \
                                             void *cubin) {                    \
    return nvJitLinkGetLinkedCubin(handle, cubin);                             \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedPtxSize, major, minor)(nvJitLinkHandle handle,         \
                                               size_t * size) {                \
    return nvJitLinkGetLinkedPtxSize(handle, size);                            \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedPtx, major, minor)(nvJitLinkHandle handle,             \
                                           char *ptx) {                        \
    return nvJitLinkGetLinkedPtx(handle, ptx);                                 \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetErrorLogSize, major, minor)(nvJitLinkHandle handle,          \
                                              size_t * size) {                 \
    return nvJitLinkGetErrorLogSize(handle, size);                             \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetErrorLog, major, minor)(nvJitLinkHandle handle, char *log) { \
    return nvJitLinkGetErrorLog(handle, log);                                  \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetInfoLogSize, major, minor)(nvJitLinkHandle handle,           \
                                             size_t * size) {                  \
    return nvJitLinkGetInfoLogSize(handle, size);                              \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetInfoLog, major, minor)(nvJitLinkHandle handle, char *log) {  \
    return nvJitLinkGetInfoLog(handle, log);                                   \
  }                                                                            \
  LUPINE_NVJITLINK_LTOIR_RELEASE_(major, minor)
#if LUPINE_NVJITLINK_HAS_LTOIR
#define LUPINE_NVJITLINK_LTOIR_RELEASE_(major, minor)                          \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedLTOIRSize, major, minor)(nvJitLinkHandle handle,       \
                                                 size_t * size) {              \
    return nvJitLinkGetLinkedLTOIRSize(handle, size);                          \
  }                                                                            \
  extern "C" nvJitLinkResult LUPINE_NVJITLINK_NAME(                            \
      nvJitLinkGetLinkedLTOIR, major, minor)(nvJitLinkHandle handle,           \
                                             void *ltoir) {                    \
    return nvJitLinkGetLinkedLTOIR(handle, ltoir);                             \
  }
#else
#define LUPINE_NVJITLINK_LTOIR_RELEASE_(major, minor)
#endif
#define LUPINE_NVJITLINK_RELEASE(minor)                                        \
  LUPINE_NVJITLINK_RELEASE_(LUPINE_NVJITLINK_MAJOR, minor)

LUPINE_NVJITLINK_RELEASE(0)
#if LUPINE_NVJITLINK_MINOR >= 1
LUPINE_NVJITLINK_RELEASE(1)
#endif
#if LUPINE_NVJITLINK_MINOR >= 2
LUPINE_NVJITLINK_RELEASE(2)
#endif
#if LUPINE_NVJITLINK_MINOR >= 3
LUPINE_NVJITLINK_RELEASE(3)
#endif
#if LUPINE_NVJITLINK_MINOR >= 4
LUPINE_NVJITLINK_RELEASE(4)
#endif
#if LUPINE_NVJITLINK_MINOR >= 5
LUPINE_NVJITLINK_RELEASE(5)
#endif
#if LUPINE_NVJITLINK_MINOR >= 6
LUPINE_NVJITLINK_RELEASE(6)
#endif
#if LUPINE_NVJITLINK_MINOR >= 7
LUPINE_NVJITLINK_RELEASE(7)
#endif
#if LUPINE_NVJITLINK_MINOR >= 8
LUPINE_NVJITLINK_RELEASE(8)
#endif
#if LUPINE_NVJITLINK_MINOR >= 9
LUPINE_NVJITLINK_RELEASE(9)
#endif
