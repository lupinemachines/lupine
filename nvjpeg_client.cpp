// nvJPEG (libnvjpeg) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libnvjpeg, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps the
// APIs ordered. Nearly all of the surface is generated. The calls in this file
// are the batched decoders, which take arrays of host bitstreams.

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

nvjpegStatus_t rpc_error() { return NVJPEG_STATUS_EXECUTION_FAILED; }

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
// Handles
// ---------------------------------------------------------------------------

// Every nvJPEG object is created on the runtime's current device, or with a
// handle that already lives on one, and routes every later call to that
// connection. A decode state also keeps the batch size its batched decoder was
// initialized with, which sizes the arrays a batched decode sends.
struct handle_state {
  conn_t *conn;
  int batch_size;
};

std::mutex handles_mutex;
std::unordered_map<void *, handle_state> handles;

// A call without a handle goes to the runtime's current device.
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

void note_handle_owner(conn_t *conn, void *handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles[handle] = {conn, 0};
}

// A destroyed handle's address may come back from a later create, which
// records it afresh.
void forget_handle(void *handle) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  handles.erase(handle);
}

void note_batch_size(void *state, int batch_size) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(state);
  if (it != handles.end()) {
    it->second.batch_size = batch_size;
  }
}

int batch_size_of(void *state) {
  std::lock_guard<std::mutex> lock(handles_mutex);
  auto it = handles.find(state);
  return it == handles.end() ? 0 : it->second.batch_size;
}

} // namespace

#include "codegen/gen_nvjpeg_client.inc"

// ---------------------------------------------------------------------------
// Batched decoding
// ---------------------------------------------------------------------------

namespace {

// A batch travels as its size, the bitstream lengths, each bitstream's bytes,
// the output descriptions and, for the Ex form, one decode parameter set per
// image; the stream closes the request.
nvjpegStatus_t
decode_batched(int op, nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
               const unsigned char *const *data, const size_t *lengths,
               nvjpegImage_t *destinations, nvjpegDecodeParams_t *decode_params,
               cudaStream_t stream) {
  const int batch_size = batch_size_of(jpeg_handle);
  if (data == nullptr || lengths == nullptr || destinations == nullptr ||
      batch_size <= 0) {
    return NVJPEG_STATUS_INVALID_PARAMETER;
  }
  const uint64_t count = static_cast<uint64_t>(batch_size);
  conn_t *conn = connection_for_handle(handle);
  nvjpegStatus_t status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &jpeg_handle, sizeof(jpeg_handle)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, lengths, count * sizeof(*lengths)) < 0) {
    return rpc_error();
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (lengths[i] != 0 && rpc_write(conn, data[i], lengths[i]) < 0) {
      return rpc_error();
    }
  }
  if (rpc_write(conn, destinations, count * sizeof(*destinations)) < 0 ||
      (decode_params != nullptr &&
       rpc_write(conn, decode_params, count * sizeof(*decode_params)) < 0) ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

} // namespace

extern "C" nvjpegStatus_t
nvjpegDecodeBatched(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                    const unsigned char *const *data, const size_t *lengths,
                    nvjpegImage_t *destinations, cudaStream_t stream) {
  return decode_batched(RPC_nvjpegDecodeBatched, handle, jpeg_handle, data,
                        lengths, destinations, nullptr, stream);
}

extern "C" nvjpegStatus_t
nvjpegDecodeBatchedEx(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                      const unsigned char *const *data, const size_t *lengths,
                      nvjpegImage_t *destinations,
                      nvjpegDecodeParams_t *decode_params,
                      cudaStream_t stream) {
  if (decode_params == nullptr) {
    return NVJPEG_STATUS_INVALID_PARAMETER;
  }
  return decode_batched(RPC_nvjpegDecodeBatchedEx, handle, jpeg_handle, data,
                        lengths, destinations, decode_params, stream);
}
