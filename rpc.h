#ifndef RPC_H
#define RPC_H

#include "lupine_platform.h"

// Uncompressed block size for the optional LZ4 payload framing. The framed
// bytes are produced lazily, one block at a time, by the HTTP/2 transport
// (h2.cpp) and decoded by the rpc_read_payload helpers (compress.cpp).
#define LUPINE_COMPRESS_BLOCK_BYTES (4 * 1024 * 1024)

typedef struct {
  lupine_socket_t connfd;

  int request_id;
  int read_id;
  int write_id;
  int write_op;

  pthread_t read_thread;
  pthread_t rpc_thread;
  pthread_mutex_t read_mutex, write_mutex, call_mutex;
  pthread_cond_t read_cond;
  struct iovec write_iov[128];
  // write_iov_framed[i] marks write_iov[i] as a payload the transport
  // LZ4-frames lazily as it streams to the socket (see compress.cpp).
  // 0 = plain bytes, 1 = framed with per-block LZ4 attempts, 2 = framed but
  // every block is stored raw (the source is already compressed, so the LZ4
  // attempt would only waste CPU; the wire format is unchanged).
  unsigned char write_iov_framed[128];
  int write_iov_count;
  int local_request_parity;
  int closed;
  void *http2;
} conn_t;

extern int rpc_dispatch(conn_t *conn, int parity);
extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_drain(conn_t *conn, size_t size);
extern int rpc_read_end(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_framed(conn_t *conn, const void *data, const size_t size,
                            const int no_compress);
extern int rpc_write_end(conn_t *conn);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_writev(conn_t *conn, struct iovec *iov,
                            const unsigned char *framed, int iov_count);
extern int rpc_http2_client_init(conn_t *conn);
extern int rpc_http2_server_init(conn_t *conn);
extern int rpc_http2_compress_lz4(conn_t *conn);

// Optional LZ4 framing for large memory transfer payloads (see compress.cpp).
extern int lupine_payload_framed(conn_t *conn, size_t total_size);
extern int lupine_payload_precompressed(const void *data, size_t size);
extern int rpc_write_payload(conn_t *conn, const void *data, size_t size);
extern int rpc_read_payload(conn_t *conn, void *data, size_t size);
extern int rpc_read_payload_part(conn_t *conn, int framed, void *data,
                                 size_t size);
extern int rpc_drain_payload(conn_t *conn, int framed, size_t size);

#endif
