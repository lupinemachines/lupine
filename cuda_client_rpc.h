#ifndef LUPINE_CUDA_CLIENT_RPC_H
#define LUPINE_CUDA_CLIENT_RPC_H

#include <stddef.h>
#include <stdint.h>

// The CUDA driver shim exports these for sibling shims (libcudart) whose calls
// must run on its connections: the server child that owns the driver state
// also runs the runtime calls, and one lane per thread keeps both APIs
// ordered. conn_t stays opaque to the sibling; a null connection means the
// device is unknown or served locally.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct conn_t conn_t;

int lupine_rpc_device_count(int *count);
// Maps a virtual device ordinal to its connection and rewrites it to the
// server's ordinal.
conn_t *lupine_rpc_conn_for_device(int *device);
// The connection a handle was created on. Handles created through a sibling
// shim are registered with the note calls below, so both shims route by the
// same ownership table.
conn_t *lupine_rpc_conn_for_stream(struct CUstream_st *stream);
conn_t *lupine_rpc_conn_for_event(struct CUevent_st *event);
void lupine_rpc_note_stream_owner(conn_t *conn, struct CUstream_st *stream);
void lupine_rpc_note_event_owner(conn_t *conn, struct CUevent_st *event);

// Flushes client-side state the request must observe, then opens the request.
int lupine_rpc_write_start_request(conn_t *conn, int op);
int lupine_rpc_write(conn_t *conn, const void *data, size_t size);
int lupine_rpc_wait_for_response(conn_t *conn);
int lupine_rpc_read(conn_t *conn, void *data, size_t size);
int lupine_rpc_read_end(conn_t *conn);

// Exported by the CUDA runtime shim (libcudart) for a sibling whose objects
// are created on the runtime's current device (libcublas): the connection
// serving that device, with the device bound on this thread's lane.
conn_t *lupine_cudart_connection(void);

#ifdef __cplusplus
}
#endif

#endif
