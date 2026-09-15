#ifndef LUPINE_NPP_CLIENT_H
#define LUPINE_NPP_CLIENT_H

// What every NPP client library's generated calls share. Each library is its
// own shared object, so each gets its own copy.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"
#include "npp_compat.h"

namespace {

NppStatus rpc_error() { return NPP_ERROR; }

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

conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

// A stream belongs to the server that created it; the default stream is the
// current device's.
conn_t *connection_for_stream(cudaStream_t stream) {
  return stream == nullptr ? connection() : lupine_rpc_conn_for_stream(stream);
}

conn_t *connection_for_stream(const NppStreamContext &context) {
  return connection_for_stream(context.hStream);
}

// Memory nppiMalloc or nppsMalloc returned belongs to the server that
// allocated it.
conn_t *connection_for_handle(void *pointer) {
  return lupine_rpc_conn_for_deviceptr(
      reinterpret_cast<unsigned long long>(pointer));
}

#if NPP_VERSION >= 12002
// nppiFusedAbsDiff_Threshold_GTVal reads one element of the image's type for
// each channel it thresholds, the alpha channel excepted.
size_t npp_threshold_bytes(NppDataType type, NppiChannels channels) {
  size_t width = 0;
  switch (type) {
  case NPP_8U:
  case NPP_8S:
    width = 1;
    break;
  case NPP_16U:
  case NPP_16S:
  case NPP_16F:
    width = 2;
    break;
  case NPP_32U:
  case NPP_32S:
  case NPP_32F:
    width = 4;
    break;
  case NPP_64U:
  case NPP_64S:
  case NPP_64F:
    width = 8;
    break;
  }
  switch (channels) {
  case NPP_CH_1:
    return width;
  case NPP_CH_3:
  case NPP_CH_A4:
    return 3 * width;
  default:
    return 0;
  }
}
#endif

} // namespace

#endif
