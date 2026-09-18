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
// cuBLAS and cuBLASLt name one object: NVIDIA's libraries take a
// cublasHandle_t wherever a cublasLtHandle_t is expected. The two shims are
// separate objects with separate state, so the owner they must agree on is
// recorded here, in the one they both load.
conn_t *lupine_rpc_conn_for_blas_handle(void *handle);
void lupine_note_blas_handle_owner(void *handle, conn_t *conn);
void lupine_forget_blas_handle_owner(void *handle);
int lupine_local_device_for_remote(conn_t *conn, int remote_device);
void lupine_note_deviceptr_allocation(unsigned long long ptr, size_t size,
                                      conn_t *conn);
void lupine_forget_deviceptr_owner(unsigned long long ptr);
void lupine_deep_cache_reset(const void *key);
void *lupine_deep_cache_add(const void *key, size_t bytes);
// The device a server answers a query with belongs to the lane bound to the
// calling thread. This token moves whenever anything can have rebound that
// lane, so a device answer cached against it is good until it does.
uint64_t lupine_device_binding_epoch(void);
void lupine_note_device_binding_changed(void);
void lupine_invalidate_current_context_cache(void);
// What the driver shim believes is current on this thread's lane, without the
// round trip it would spend confirming that belief.
struct CUctx_st *lupine_current_context_hint(void);

#ifdef __cplusplus
}

// Only the client invokes this function and dereferences user_data. The
// server carries both as opaque tokens, just like CUDA's log callback bridge.
using library_log_callback = void (*)(void *user_data, int level,
                                      const char *function, const char *message,
                                      size_t length);
struct library_log_target {
  library_log_callback callback = nullptr;
  void *user_data = nullptr;
};

constexpr uint32_t LUPINE_MAX_LIBRARY_LOG_BYTES = 4 * 1024 * 1024;

#include <shared_mutex>
void lupine_event_invalidate_completion(struct CUevent_st *event);
std::shared_mutex &lupine_event_lifecycle_mutex();
#endif

#endif
