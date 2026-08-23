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

// Remote accelerator and pinned-host addresses live in this upper canonical
// VA band. Driver shims map their application and transport views at fixed
// offsets from that address, so the RPC core can translate without knowing
// which accelerator backend owns the allocation.
static constexpr uintptr_t LUPINE_MIRROR_SERVER_BASE = UINT64_C(0x700000000000);
static constexpr uintptr_t LUPINE_MIRROR_WINDOW_SIZE = UINT64_C(0x100000000000);
static constexpr intptr_t LUPINE_MIRROR_R_OFFSET = -INT64_C(0x500000000000);
static constexpr intptr_t LUPINE_MIRROR_W_OFFSET = -INT64_C(0x400000000000);

struct rpc_mirror_write {
  uintptr_t start;
  size_t size;
};

// The server's HTTP/2 receive window, and with it the ceiling on the pinned
// staging a client can hold there: fire-and-forget device-bound payload bytes
// stay uncredited until the staging buffer they landed in retires.
#define LUPINE_FF_STAGING_WINDOW_BYTES (64ull * 1024 * 1024)

typedef struct conn_t conn_t;

struct conn_t {
  lupine_socket_t connfd;

  int request_id;
  int write_id;
  int write_op;
  int32_t write_stream_id;

  pthread_t read_thread;
  pthread_mutex_t write_mutex, call_mutex;
  std::vector<rpc_write_cursor> write_queue;
  std::vector<rpc_mirror_write> mirror_writes;
  unsigned char *write_copy_buffer;
  size_t write_copy_capacity;
  size_t write_copy_offset;
  int local_request_parity;
  int logical_index;
  int closed;
  void *http2;
  void *tls_session; // SSL* for https:// client connections; otherwise null.
  uintptr_t va_base;
  size_t va_size;
  uintptr_t va_next;
  intptr_t w_offset;
};

// Backends install these hooks before opening connections. They let the
// transport report lifecycle changes without depending on backend state.
// Installation is process-local and succeeds at most once per RPC core.
struct rpc_lifecycle_hooks {
  void (*connection_closed)(conn_t *conn);
  void (*thread_lane_destroyed)(uint64_t lane_id);
};
extern int rpc_set_lifecycle_hooks(const rpc_lifecycle_hooks *hooks);

extern int rpc_dispatch(conn_t *conn, int parity);
// Binds the calling thread to a peer-created HTTP/2 lane. Server lane workers
// use this before rpc_dispatch; the stream id is the lane identity.
extern int rpc_bind_http2_stream(conn_t *conn, int32_t stream_id);
extern void rpc_unbind_http2_stream(conn_t *conn);
extern int32_t rpc_current_http2_stream(conn_t *conn);
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
extern int rpc_read_pitched(conn_t *conn, void *data, size_t width, size_t rows,
                            size_t row_stride, size_t slices,
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
// Signals transport readers to stop without releasing connection resources.
// Owners use this before joining workers that may be blocked on the transport.
extern void rpc_shutdown_transport_socket(conn_t *conn);
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

constexpr int LUPINE_RPC_HTTP2_STREAM_END = -2;
constexpr int LUPINE_RPC_HTTP2_VA_CONFLICT = -3;
extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_read_stream(conn_t *conn, int32_t stream_id, void *data,
                                 size_t size);
extern int rpc_http2_write(conn_t *conn,
                           std::vector<rpc_write_cursor> &cursors);
extern int rpc_http2_write_stream(conn_t *conn, int32_t stream_id,
                                  std::vector<rpc_write_cursor> &cursors);
extern int32_t rpc_http2_dispatch_stream(conn_t *conn);
extern int32_t rpc_http2_lane_stream(conn_t *conn, uint64_t lane_id);
extern int rpc_http2_end_stream(conn_t *conn, int32_t stream_id);
extern int32_t rpc_http2_accept_stream(conn_t *conn);
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
// landed in is idle. Credit stays tagged with its stream because HTTP/2 flow
// control is stream-specific. Held bytes are capped, so a caller that never
// releases costs window but cannot close it.
struct rpc_http2_window_credit {
  int32_t stream_id = -1;
  uint64_t bytes = 0;
};
extern void rpc_http2_window_hold_begin(conn_t *conn);
extern rpc_http2_window_credit rpc_http2_window_hold_end(conn_t *conn);
extern void rpc_http2_window_release(conn_t *conn,
                                     rpc_http2_window_credit credit);

// Optional LZ4 framing for large memory transfer payloads (see compress.cpp).
extern int lupine_payload_framed(conn_t *conn, size_t total_size);
extern int rpc_write_payload(conn_t *conn, const void *data, size_t size);
extern int rpc_read_payload(conn_t *conn, void *data, size_t size);
extern int rpc_read_payload_part(conn_t *conn, int framed, void *data,
                                 size_t size);
extern int rpc_drain_payload(conn_t *conn, int framed, size_t size);

#endif
