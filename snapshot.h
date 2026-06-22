#ifndef LUPINE_SNAPSHOT_H
#define LUPINE_SNAPSHOT_H

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#include "rpc.h"

#define LUPINE_SNAPSHOT_ID_HEX_BYTES 32
#define LUPINE_SNAPSHOT_ID_BUFFER_BYTES (LUPINE_SNAPSHOT_ID_HEX_BYTES + 1)

static constexpr int LUPINE_RPC_snapshot_create = 1000100;
static constexpr int LUPINE_RPC_snapshot_status = 1000101;
static constexpr int LUPINE_RPC_snapshot_load = 1000102;
static constexpr int LUPINE_RPC_snapshot_delete = 1000103;
static constexpr int LUPINE_RPC_snapshot_save_and_exit = 1000104;

enum lupine_snapshot_state {
  LUPINE_SNAPSHOT_STATE_UNKNOWN = 0,
  LUPINE_SNAPSHOT_STATE_CREATING = 1,
  LUPINE_SNAPSHOT_STATE_READY = 2,
  LUPINE_SNAPSHOT_STATE_FAILED = 3,
  LUPINE_SNAPSHOT_STATE_UNSUPPORTED = 4,
};

struct lupine_snapshot_info {
  int state;
  uint64_t bytes;
  int64_t created_unix_seconds;
};

int handle_lupine_snapshot_create(conn_t *conn);
int handle_lupine_snapshot_status(conn_t *conn);
int handle_lupine_snapshot_load(conn_t *conn);
int handle_lupine_snapshot_delete(conn_t *conn);
int handle_lupine_snapshot_save_and_exit(conn_t *conn);
int lupine_snapshot_restore_for_connection(const char *id, lupine_socket_t connfd);
int lupine_snapshot_read_bootstrap(lupine_socket_t connfd,
                                   char id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]);

#ifdef __cplusplus
extern "C" {
#endif

CUresult lupine_snapshot_create(char *id, size_t id_len, unsigned int flags);
CUresult lupine_snapshot_status(const char *id, lupine_snapshot_info *info);
CUresult lupine_snapshot_load(const char *id, unsigned int flags);
CUresult lupine_snapshot_delete(const char *id);
CUresult lupine_snapshot_save_and_exit(const char *id);
int lupine_snapshot_write_bootstrap(lupine_socket_t connfd);

#ifdef __cplusplus
}
#endif

#endif
