#ifndef LUPINE_CUDA_CLIENT_RPC_H
#define LUPINE_CUDA_CLIENT_RPC_H

#include <stddef.h>
#include <stdint.h>

// C exports of the driver client's RPC functions for sibling shims. Each
// wrapper has the same arguments and return value as its rpc_* counterpart.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct conn_t conn_t;

int lupine_rpc_write_start_request(conn_t *conn, int op);
int lupine_rpc_write_start_async_request(conn_t *conn, int op,
                                         uint64_t *sequence);
int lupine_rpc_write(conn_t *conn, const void *data, size_t size);
int lupine_rpc_write_end(conn_t *conn);
int lupine_rpc_wait_for_response(conn_t *conn);
int lupine_rpc_read(conn_t *conn, void *data, size_t size);
int lupine_rpc_read_end(conn_t *conn);

// Existing driver routing and ownership functions, shared with sibling shims.
conn_t *lupine_rpc_conn_for_device(int *device);
conn_t *lupine_rpc_conn_for_current_context(void);
conn_t *lupine_rpc_conn_for_deviceptr(unsigned long long ptr);
int lupine_prepare_rpc(conn_t *conn);
int lupine_local_device_for_remote(conn_t *conn, int remote_device);
void lupine_note_deviceptr_allocation(unsigned long long ptr, size_t size,
                                      conn_t *conn);
void lupine_forget_deviceptr_owner(unsigned long long ptr);

#ifdef __cplusplus
}
#endif

#endif
