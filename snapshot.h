#ifndef LUPINE_SNAPSHOT_H
#define LUPINE_SNAPSHOT_H

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "rpc.h"

// A snapshot id can be any non-empty string within this bound; the server
// hashes it into the on-disk directory name, so it never needs to be a valid
// path component itself.
#define LUPINE_SNAPSHOT_ID_MAX_BYTES 255
#define LUPINE_SNAPSHOT_ID_BUFFER_BYTES (LUPINE_SNAPSHOT_ID_MAX_BYTES + 1)

static inline int lupine_snapshot_id_valid(const char *id) {
  if (id == NULL) {
    return 0;
  }
  size_t n = strlen(id);
  return n > 0 && n <= LUPINE_SNAPSHOT_ID_MAX_BYTES;
}

static constexpr int LUPINE_RPC_snapshot_save_and_exit = 1000104;

int handle_lupine_snapshot_save_and_exit(conn_t *conn);
int lupine_snapshot_restore_for_connection(const char *id, lupine_socket_t connfd);
int lupine_snapshot_read_bootstrap(lupine_socket_t connfd,
                                   char id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]);

#ifdef __cplusplus
extern "C" {
#endif

CUresult lupine_snapshot_load(const char *id);
CUresult lupine_snapshot_save_and_exit(const char *id);
int lupine_snapshot_write_bootstrap(lupine_socket_t connfd);

#ifdef __cplusplus
}
#endif

#endif
