#ifndef LUPINE_CUDA_SERVER_H
#define LUPINE_CUDA_SERVER_H

#include "rpc.h"

bool lupine_server_initialize_connection(conn_t *conn);
void lupine_server_cleanup_connection(conn_t *conn);
void lupine_server_cleanup_identity_allocations(conn_t *conn);
// Fails every copy still waiting for chunks: a lost bulk connection can never
// deliver them.
void lupine_server_bulk_connection_lost();
// Waits until the device-to-host deliveries already numbered have been sent,
// so a handoff does not park them.
void lupine_dtoh_drain(conn_t *conn);
// Wakes every wait on the connection's deliveries once it is shut down.
void lupine_dtoh_wake(conn_t *conn);
void lupine_server_cleanup_log_callbacks(conn_t *conn);

#endif
