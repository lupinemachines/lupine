#ifndef RPC_H
#define RPC_H

#include "lupine_platform.h"
#include <stdint.h>
#include <vector>

// Uncompressed block size for the optional LZ4 payload framing. The framed
// bytes are produced lazily, one block at a time, by the HTTP/2 transport
// (h2.cpp) and decoded by the rpc_read_payload helpers (compress.cpp).
#define LUPINE_COMPRESS_BLOCK_BYTES (4 * 1024 * 1024)

// References caller-owned bytes while an RPC is being serialized. Plain
// cursors use data/size directly. Framed cursors keep uncompressed bytes in
// source/source_size while HTTP/2 materializes one framed block at a time into
// data/size.
struct rpc_write_cursor {
  const unsigned char *data = nullptr;
  size_t size = 0;
  const unsigned char *source = nullptr;
  size_t source_size = 0;

  static rpc_write_cursor plain(const void *data, size_t size) {
    return {static_cast<const unsigned char *>(data), size, nullptr, 0};
  }

  static rpc_write_cursor framed(const void *data, size_t size) {
    return {nullptr, 0, static_cast<const unsigned char *>(data), size};
  }

  size_t remaining() const { return size + source_size; }
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
// staging a client can hold there: fire-and-forget device-bound payload bytes
// stay uncredited until the staging buffer they landed in retires.
#define LUPINE_FF_STAGING_WINDOW_BYTES (64ull * 1024 * 1024)

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
  std::vector<rpc_write_cursor> write_queue;
  unsigned char *write_copy_buffer;
  size_t write_copy_capacity;
  size_t write_copy_offset;
  int local_request_parity;
  int logical_index;
  int closed;
  void *http2;
  void *tls_session; // SSL* for https:// client connections; otherwise null.
};

// Backends install these hooks before opening connections. They let the
// transport report lifecycle changes without depending on backend state.
// Installation is process-local and succeeds at most once per RPC core.
struct rpc_lifecycle_hooks {
  void (*connection_closed)(conn_t *conn);
  void (*thread_lane_destroyed)(uint64_t lane_id);
};
extern int rpc_set_lifecycle_hooks(const rpc_lifecycle_hooks *hooks);

// Client-side memory-coherence hooks, registered once by the CUDA client so
// reads into tracked host mappings carry their own demand-fetch bookkeeping
// (make-writable before, mark-current after). The hooks run on every read
// and no-op for untracked memory; nested invocations are balanced, so the
// primitives call them unconditionally. The server never registers them.
// Write-side refreshes stay at the call sites: they can fetch, which needs
// the connection, so they must run before the request holds it.
struct rpc_hooks {
  void (*before_read_into)(void *data, size_t size);
  void (*after_read_into)(void *data, size_t size);
};
extern void rpc_set_hooks(const rpc_hooks *hooks);
extern void rpc_hooks_before_read(void *data, size_t size);
extern void rpc_hooks_after_read(void *data, size_t size);

extern int rpc_dispatch(conn_t *conn, int parity);
extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
// Reads a field emitted by rpc_write_buffer. Keeping buffered reads distinct
// makes request and response serializers exact field-for-field inverses.
static inline int rpc_read_buffer(conn_t *conn, void *data, size_t size) {
  return rpc_read(conn, data, size);
}
extern int rpc_drain(conn_t *conn, size_t size);
extern int rpc_read_end(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

// Owns connection validation for the request chain: a null conn (route with no
// remote server) or a closed conn fails here, so callers surface their
// unavailable-server result without per-call-site null checks.
extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
// A zero-size write is a successful no-op, including when data is null.
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
// Pitched transfers move `slices` * `rows` rows of `width` bytes; rows sit
// `row_stride` bytes apart and slices `slice_stride` bytes apart at this
// endpoint. The wire carries the rows back-to-back, identical to a contiguous
// transfer of width * rows * slices bytes. A flat 2D region is one slice.
extern int rpc_write_pitched(conn_t *conn, const void *data, size_t width,
                             size_t rows, size_t row_stride, size_t slices,
                             size_t slice_stride);
extern int rpc_read_pitched(conn_t *conn, void *data, size_t width,
                            size_t rows, size_t row_stride, size_t slices,
                            size_t slice_stride);
// Reserves the request-owned storage used by subsequent rpc_write_buffer
// calls. The reservation must be made once before the first buffered write in
// an RPC and is released with the request.
extern int rpc_copy_alloc(conn_t *conn, const size_t size);
// Returns and queues the next aligned span in the request-owned allocation.
// The complete allocation is fixed before serialization starts, so returned
// pointers remain valid until the request ends.
extern void *rpc_write_buffer(conn_t *conn, size_t size, size_t alignment);
extern int rpc_write_cursors(conn_t *conn, const rpc_write_cursor *cursors,
                             size_t count);
extern int rpc_write_framed(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);
extern int rpc_write_lane_termination(conn_t *conn, uint64_t lane_id);
// Marks a connection closed and atomically takes ownership of its transport
// socket before aborting it. Safe after a transport error has already set
// conn->closed, and safe for concurrent/idempotent cleanup.
extern void rpc_close_transport_socket(conn_t *conn);
extern int rpc_conn_init(conn_t *conn, lupine_socket_t connfd, int request_id);
extern void rpc_conn_destroy(conn_t *conn);

// lupine_tcp_connect resolves host:port and returns a connected socket with
// the standard transport options applied (TCP_NODELAY + keepalive; see
// lupine_socket_apply_transport_options). A server that is not reachable yet
// (e.g. still provisioning) is retried up to max_retries times with
// exponential backoff and bounded attempts. Passing zero preserves the
// client's single blocking attempt. Returns the socket, or
// LUPINE_INVALID_SOCKET on permanent failure.
extern lupine_socket_t lupine_tcp_connect(const char *host, const char *port,
                                          unsigned int max_retries = 5);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_write(conn_t *conn,
                           std::vector<rpc_write_cursor> &cursors);
extern int rpc_http2_client_init(conn_t *conn);
extern void rpc_http2_client_start_heartbeat(conn_t *conn);
extern void rpc_http2_destroy(conn_t *conn);
struct rpc_http2_server_metadata {
  const char *backend_version;
};
// Sends HEAD / and returns the backend-version response header, or nullptr
// when the request fails or the server does not advertise a version.
// The returned pointer remains valid until rpc_http2_destroy() or
// rpc_conn_destroy(); the probe connection must not be reused for RPC.
extern const char *rpc_http2_client_probe(conn_t *conn);
// Returns -1 on failure, 0 for an RPC connection, and a positive value when
// the HTTP layer has already handled the request.
extern int rpc_http2_server_init(conn_t *conn);
extern int
rpc_http2_server_init_with_metadata(conn_t *conn,
                                    const rpc_http2_server_metadata *metadata);
extern int rpc_http2_compress_lz4(conn_t *conn);
// Returns the x-lupine-session request header after the server has consumed
// the HTTP/2 request headers, or nullptr when no session was supplied.
extern const char *rpc_http2_session_id(conn_t *conn);
extern int rpc_http2_get_read_stats(conn_t *conn, rpc_http2_read_stats *stats);

// Keeps the client-to-server TCP path active while a synchronous RPC waits for
// its response. The heartbeat is transport-only: it emits HTTP/2 PING frames
// and does not add, combine, or otherwise change application RPCs.
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
