#ifndef LUPINE_SNAPSHOT_H
#define LUPINE_SNAPSHOT_H

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#include "rpc.h"

#define LUPINE_SNAPSHOT_ID_HEX_BYTES 32
#define LUPINE_SNAPSHOT_ID_BUFFER_BYTES (LUPINE_SNAPSHOT_ID_HEX_BYTES + 1)

static constexpr int LUPINE_RPC_snapshot_save_and_exit = 1000104;

int handle_lupine_snapshot_save_and_exit(conn_t *conn);
int lupine_snapshot_restore_for_connection(const char *id, lupine_socket_t connfd);
int lupine_snapshot_read_bootstrap(lupine_socket_t connfd,
                                   char id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]);

#ifdef __cplusplus
extern "C" {
#endif

CUresult lupine_snapshot_save_and_exit(const char *id);
int lupine_snapshot_write_bootstrap(lupine_socket_t connfd);

#ifdef __cplusplus
}
#endif

#endif
