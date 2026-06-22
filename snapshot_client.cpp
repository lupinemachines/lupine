#include "snapshot.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

extern conn_t *rpc_client_get_connection(unsigned int index);

namespace {

static const char kBootstrapMagic[] = "LUPSNAP1";
static constexpr uint32_t kSnapshotIdLen = LUPINE_SNAPSHOT_ID_HEX_BYTES;

conn_t *snapshot_conn() { return rpc_client_get_connection(0); }

bool valid_client_snapshot_id(const char *id) {
  if (id == nullptr || strlen(id) != LUPINE_SNAPSHOT_ID_HEX_BYTES) {
    return false;
  }
  for (size_t i = 0; i < LUPINE_SNAPSHOT_ID_HEX_BYTES; ++i) {
    char c = id[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

CUresult write_snapshot_id(conn_t *conn, const char *id) {
  if (!valid_client_snapshot_id(id)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rpc_write(conn, &kSnapshotIdLen, sizeof(kSnapshotIdLen)) < 0 ||
      rpc_write(conn, id, kSnapshotIdLen) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

int send_all(lupine_socket_t connfd, const void *data, size_t size) {
  const char *cursor = static_cast<const char *>(data);
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = lupine_socket_send(connfd, cursor + sent, size - sent);
    if (n < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (n <= 0) {
      return -1;
    }
    sent += static_cast<size_t>(n);
  }
  return 0;
}

} // namespace

extern "C" int lupine_snapshot_write_bootstrap(lupine_socket_t connfd) {
  const char *id = getenv("LUPINE_SNAPSHOT_ID");
  if (id == nullptr || id[0] == '\0') {
    return 0;
  }
  if (!valid_client_snapshot_id(id)) {
    return -1;
  }
  uint32_t len = LUPINE_SNAPSHOT_ID_HEX_BYTES;
  if (send_all(connfd, kBootstrapMagic, sizeof(kBootstrapMagic) - 1) < 0 ||
      send_all(connfd, &len, sizeof(len)) < 0 ||
      send_all(connfd, id, len) < 0) {
    return -1;
  }
  return 0;
}

extern "C" CUresult lupine_snapshot_save_and_exit(const char *id) {
  if (!valid_client_snapshot_id(id)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = snapshot_conn();
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_snapshot_save_and_exit) < 0 ||
      write_snapshot_id(conn, id) != CUDA_SUCCESS ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  conn->closed = 1;
  return result;
}
