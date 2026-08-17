#ifndef RPC_H
#define RPC_H

#include "cuda_compat.h"
#include "lupine_platform.h"
#include <stdint.h>
#include <vector>

// Uncompressed block size for the optional LZ4 payload framing. The framed
// bytes are produced lazily, one block at a time, by the HTTP/2 transport
// (h2.cpp) and decoded by the rpc_read_payload helpers (compress.cpp).
#define LUPINE_COMPRESS_BLOCK_BYTES (4 * 1024 * 1024)

struct rpc_write_entry {
  struct iovec iov;
  // 0 = plain bytes, 1 = framed with per-block LZ4 attempts, 2 = framed but
  // every block is stored raw (the source is already compressed, so the LZ4
  // attempt would only waste CPU; the wire format is unchanged).
  unsigned char framed;
};

struct rpc_http2_read_stats {
  uint64_t direct_bytes;
  uint64_t staged_bytes;
  uint64_t staged_read_bytes;
  uint64_t staged_buffers;
  uint64_t peak_staged_bytes;
};

#define LUPINE_RPC_TERMINATE_LANE 0xFFFF

// The server's HTTP/2 receive window, and with it the ceiling on the pinned
// staging a client can hold there: fire-and-forget async HtoD payload bytes
// stay uncredited until the staging buffer they landed in retires.
#define LUPINE_FF_STAGING_WINDOW_BYTES (64ull * 1024 * 1024)

// Events the explicit lupineEventQueryBatch cache-warming RPC may carry. Its
// request is a count followed by that many handles, and its response is one
// CUresult per handle in the same order.
#define LUPINE_EVENT_QUERY_BATCH_MAX 16

// Wire layout for LUPINE_RPC_lupineDeviceSnapshot. The response is all or
// nothing: a non-success result carries no payload, otherwise every device
// record holds a fixed-size name buffer, uuid, total memory, and a
// count-prefixed list of (attribute, value) pairs.
#define LUPINE_DEVICE_SNAPSHOT_NAME_BYTES 256

// LUPINE_RPC_lupineLibrarySnapshot is a best-effort metadata side channel.
// The request contains one CUlibrary. The response contains a CUresult and a
// count-prefixed list of kernel names, handles, and parameter layouts.

typedef struct conn_t conn_t;

struct conn_t {
  lupine_socket_t connfd;

  int request_id;
  int read_id;
  int read_op;
  uint64_t read_lane_id;
  int write_id;
  int write_op;
  uint64_t write_lane_id;

  pthread_t read_thread;
  pthread_t rpc_thread;
  pthread_mutex_t read_mutex, write_mutex, call_mutex;
  pthread_cond_t read_cond;
  // Explicitly managed so conn_t remains trivially destructible. libcudart can
  // call back into the shim during process teardown, after C++ globals have
  // begun finalizing.
  rpc_write_entry *write_queue;
  int write_queue_count;
  int write_queue_capacity;
  unsigned char *write_copy_buffer;
  size_t write_copy_capacity;
  size_t write_copy_offset;
  int local_request_parity;
  int logical_index;
  int closed;
  void *http2;
  void *tls_session; // SSL* for https:// client connections; otherwise null.
};

extern int rpc_dispatch(conn_t *conn, int parity);
extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_drain(conn_t *conn, size_t size);
extern int rpc_read_end(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
// Reserves the request-owned storage used by subsequent rpc_write_buffer
// calls. The reservation must be made once before the first buffered write in
// an RPC and is released with the request.
extern int rpc_copy_alloc(conn_t *conn, const size_t size);
// Returns the next aligned request-owned span and advances the copy cursor. A
// later call may grow the arena, so callers must queue or consume the returned
// span before requesting another one.
extern void *rpc_write_buffer(conn_t *conn, size_t size, size_t alignment);
extern int rpc_write_iovecs(conn_t *conn, const struct iovec *iovecs,
                            size_t count);
extern int rpc_write_framed(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);
extern int rpc_write_lane_termination(conn_t *conn, uint64_t lane_id);
// Marks a connection closed and atomically takes ownership of its transport
// socket before aborting it. Safe after a transport error has already set
// conn->closed, and safe for concurrent/idempotent cleanup.
extern void rpc_close_transport_socket(conn_t *conn);
extern void rpc_write_queue_free(conn_t *conn);
extern void rpc_conn_destroy(conn_t *conn);

// lupine_tcp_connect resolves host:port and returns a connected socket with
// the standard transport options applied (TCP_NODELAY + keepalive; see
// lupine_socket_apply_transport_options). A server that is not reachable yet
// (e.g. still provisioning) is retried a few times with exponential backoff,
// and each attempt is bounded by a deadline so a packet-filtered port cannot
// stall the loop for minutes. Returns the socket, or LUPINE_INVALID_SOCKET on
// permanent failure.
extern lupine_socket_t lupine_tcp_connect(const char *host, const char *port);

extern int rpc_read_kernel_node_params(conn_t *conn,
                                       CUDA_KERNEL_NODE_PARAMS *node_params);
extern int rpc_read_kernel_node_params_and_values(
    conn_t *conn, CUDA_KERNEL_NODE_PARAMS *node_params, CUresult *result);
extern void rpc_free_kernel_param_values(void **values);
#ifdef LUPINE_RPC_SERVER
extern int rpc_read_func_param_values(conn_t *conn, void ***values,
                                      CUfunction function, CUresult *result);
#if CUDA_VERSION >= 12000
extern int rpc_read_kernel_param_values(conn_t *conn, void ***values,
                                        CUkernel kernel, CUresult *result);
#endif
extern int rpc_read_kernel_node_param_values(
    conn_t *conn, CUDA_KERNEL_NODE_PARAMS *node_params, CUresult *result);
#endif
extern int rpc_read_launch_config(conn_t *conn, CUlaunchConfig *config,
                                  std::vector<CUlaunchAttribute> *attributes);
struct rpc_jit_output_binding {
  CUjit_option option;
  void *dst;
  size_t size;
};
struct rpc_jit_server_state {
  std::vector<CUjit_option> options;
  std::vector<void *> option_values;
  float wall_time = 0.0f;
  std::vector<char> info_log;
  std::vector<char> error_log;
  bool capture_wall_time = false;
  bool capture_info_log = false;
  bool capture_error_log = false;
};
extern int rpc_read_jit_options(conn_t *conn,
                                std::vector<CUjit_option> *options,
                                std::vector<uintptr_t> *raw_values);
extern int rpc_read_jit_options(conn_t *conn, rpc_jit_server_state *state);
extern int rpc_read_library_options(conn_t *conn,
                                    std::vector<CUlibraryOption> *options,
                                    std::vector<uintptr_t> *raw_values,
                                    bool *has_option_values);
extern int
rpc_read_jit_outputs(conn_t *conn,
                     const std::vector<rpc_jit_output_binding> &bindings);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_writev(conn_t *conn, const rpc_write_entry *entries,
                            int entry_count);
extern int rpc_http2_client_init(conn_t *conn);
extern void rpc_http2_client_start_heartbeat(conn_t *conn);
// Sends HEAD / and returns the x-lupine-cuda-version response header, or
// nullptr when the request fails or the server does not advertise a version.
// The returned pointer remains valid until rpc_http2_destroy() or
// rpc_conn_destroy(); the probe connection must not be reused for RPC.
extern const char *rpc_http2_client_probe(conn_t *conn);
// Returns -1 on failure, 0 for an RPC connection, and a positive value when
// the HTTP layer has already handled the request.
extern int rpc_http2_server_init(conn_t *conn);
extern int rpc_http2_compress_lz4(conn_t *conn);
// Returns the x-lupine-session request header after the server has consumed
// the HTTP/2 request headers, or nullptr when no session was supplied.
extern const char *rpc_http2_session_id(conn_t *conn);
extern int rpc_http2_get_read_stats(conn_t *conn, rpc_http2_read_stats *stats);

// Keeps the client-to-server TCP path active while a synchronous RPC waits for
// its response. The heartbeat is transport-only: it emits HTTP/2 PING frames
// and does not add, combine, or otherwise change CUDA RPCs.
extern void rpc_http2_response_wait_begin(conn_t *conn);
extern void rpc_http2_response_wait_end(conn_t *conn);

// Server-side flow control for payloads that outlive the read that received
// them. Between hold_begin and hold_end the transport stops crediting received
// DATA bytes back to the peer; hold_end returns the byte count the caller now
// owns and must hand to rpc_http2_window_release once the buffer those bytes
// landed in is idle. Held bytes are capped, so a caller that never releases
// costs window but cannot close it.
extern void rpc_http2_window_hold_begin(conn_t *conn);
extern uint64_t rpc_http2_window_hold_end(conn_t *conn);
extern void rpc_http2_window_release(conn_t *conn, uint64_t bytes);

// Optional LZ4 framing for large memory transfer payloads (see compress.cpp).
extern int lupine_payload_framed(conn_t *conn, size_t total_size);
extern int rpc_write_payload(conn_t *conn, const void *data, size_t size);
extern int rpc_read_payload(conn_t *conn, void *data, size_t size);
extern int rpc_read_payload_part(conn_t *conn, int framed, void *data,
                                 size_t size);
extern int rpc_drain_payload(conn_t *conn, int framed, size_t size);

#endif
