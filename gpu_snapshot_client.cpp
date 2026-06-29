// Client-side snapshot API: send the save/restore ops as ordinary RPCs over the
// existing connection. No pre-HTTP2 bootstrap, no CRIU. lupine_snapshot_load()
// must be called right after connecting (before allocations) so the server
// worker restores the VMM arena before it is used.
#include <cstdint>
#include <cstring>

#include "gpu_snapshot.h"
#include "rpc.h"

extern conn_t *rpc_client_get_connection(unsigned int index);

namespace {

CUresult send_snapshot_rpc(int op, const char *id) {
  if (id == nullptr || id[0] == '\0') return CUDA_ERROR_INVALID_VALUE;
  conn_t *conn = rpc_client_get_connection(0);
  if (conn == nullptr) return CUDA_ERROR_DEVICE_UNAVAILABLE;
  uint32_t len = (uint32_t)strlen(id);
  CUresult result;
  if (rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &len, sizeof(len)) < 0 ||
      rpc_write(conn, id, len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return result;
}

} // namespace

extern "C" CUresult lupine_snapshot_save(const char *id) {
  return send_snapshot_rpc(LUPINE_RPC_gpu_snapshot_save, id);
}

extern "C" CUresult lupine_snapshot_load(const char *id) {
  return send_snapshot_rpc(LUPINE_RPC_gpu_snapshot_restore, id);
}
