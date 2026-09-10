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

int lupine_rpc_open(void);
int lupine_rpc_size(void);
uint64_t lupine_rpc_thread_request_epoch(conn_t *conn);
conn_t *lupine_rpc_client_get_connection(unsigned int index);
int lupine_rpc_write_start_request(conn_t *conn, int op);
int lupine_rpc_write_start_async_request(conn_t *conn, int op,
                                         uint64_t *sequence);
int lupine_rpc_write(conn_t *conn, const void *data, size_t size);
int lupine_rpc_write_end(conn_t *conn);
int lupine_rpc_wait_for_response(conn_t *conn);
int lupine_rpc_read(conn_t *conn, void *data, size_t size);
int lupine_rpc_read_end(conn_t *conn);
int lupine_read_deferred_dtoh_copies(conn_t *conn);
int lupine_forward_remote_stdout(conn_t *conn);

// Existing driver routing and ownership functions, shared with sibling shims.
struct CUstream_st;
struct CUevent_st;
conn_t *lupine_rpc_conn_for_device(int *device);
conn_t *lupine_rpc_conn_for_current_context(void);
conn_t *lupine_rpc_conn_for_stream(struct CUstream_st *stream);
conn_t *lupine_rpc_conn_for_event(struct CUevent_st *event);
conn_t *lupine_rpc_conn_for_deviceptr(unsigned long long ptr);
int lupine_prepare_rpc(conn_t *conn);
void lupine_invalidate_runtime_context(conn_t *conn);
void lupine_note_stream_owner(struct CUstream_st *stream, conn_t *conn);
void lupine_note_event_owner(struct CUevent_st *event, conn_t *conn);
void lupine_forget_stream_owner(struct CUstream_st *stream);
void lupine_forget_event_owner(struct CUevent_st *event);
int lupine_local_device_for_remote(conn_t *conn, int remote_device);
void lupine_note_deviceptr_allocation(unsigned long long ptr, size_t size,
                                      conn_t *conn);
void lupine_forget_deviceptr_owner(unsigned long long ptr);
void lupine_deep_cache_reset(const void *key);
void *lupine_deep_cache_add(const void *key, size_t bytes);

#ifdef __cplusplus
}

#include <shared_mutex>
void lupine_event_invalidate_completion(struct CUevent_st *event);
std::shared_mutex &lupine_event_lifecycle_mutex();
#endif

#endif
