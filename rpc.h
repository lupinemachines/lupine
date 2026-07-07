#ifndef RPC_H
#define RPC_H

#include "lupine_platform.h"
#include <cuda.h>
#include <stdint.h>
#include <vector>

// Uncompressed block size for optional LZ4 payload framing. RPC envelopes use
// the same block format as the HTTP/2 transport's direct framed writes, and
// rpc_read_payload helpers decode those blocks into caller buffers.
#define LUPINE_COMPRESS_BLOCK_BYTES (4 * 1024 * 1024)

struct rpc_write_entry {
  struct iovec iov;
  // 0 = plain bytes, 1 = framed with per-block LZ4 attempts, 2 = framed but
  // every block is stored raw (the source is already compressed, so the LZ4
  // attempt would only waste CPU; the wire format is unchanged).
  unsigned char framed;
};

#define LUPINE_RPC_RELEASE_LANE (-2)

struct rpc_frame {
  int request_id = 0;
  uint32_t lane_id = 0;
  int op = 0;
  std::vector<unsigned char> payload;
};

typedef struct {
  lupine_socket_t connfd;

  int request_id;
  int read_id;
  int write_id;
  int write_op;
  uint32_t write_lane_id;
  uint32_t next_lane_id;

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
  int local_request_parity;
  int logical_index;
  int closed;
  void *http2;
  void *rpc_state;
} conn_t;

extern int rpc_dispatch(conn_t *conn, int parity);
extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_drain(conn_t *conn, size_t size);
extern int rpc_read_end(conn_t *conn);
extern int rpc_read_end_host_copy_chunk(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_framed(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);
extern void rpc_write_queue_free(conn_t *conn);
extern int rpc_connection_state_init(conn_t *conn);
extern void rpc_connection_state_free(conn_t *conn);
extern int rpc_read_wire_frame(conn_t *conn, rpc_frame *frame);
extern int rpc_deliver_response_frame(conn_t *conn, rpc_frame &&frame);
extern int rpc_activate_frame(conn_t *conn, rpc_frame &&frame);
extern uint32_t rpc_active_lane_id();

extern int rpc_write_kernel_param_values(conn_t *conn, uint32_t count,
                                         const size_t *sizes,
                                         void *const *values);
extern int rpc_read_kernel_param_values(conn_t *conn, uint32_t count,
                                        const size_t *offsets,
                                        const size_t *sizes,
                                        size_t payload_size, void *storage,
                                        size_t storage_size, void **values);
extern int rpc_write_jit_options(conn_t *conn, const unsigned int *num_options,
                                 const CUjit_option *options,
                                 void *const *option_values,
                                 std::vector<uintptr_t> *raw_values);
extern int rpc_read_jit_options(conn_t *conn,
                                std::vector<CUjit_option> *options,
                                std::vector<uintptr_t> *raw_values);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_writev(conn_t *conn, const rpc_write_entry *entries,
                            int entry_count);
extern int rpc_http2_client_init(conn_t *conn);
extern int rpc_http2_server_init(conn_t *conn);
extern int rpc_http2_compress_lz4(conn_t *conn);

// Optional LZ4 framing for large memory transfer payloads (see compress.cpp).
extern int lupine_payload_framed(conn_t *conn, size_t total_size);
extern int rpc_write_payload(conn_t *conn, const void *data, size_t size);
extern int rpc_read_payload(conn_t *conn, void *data, size_t size);
extern int rpc_read_payload_part(conn_t *conn, int framed, void *data,
                                 size_t size);
extern int rpc_drain_payload(conn_t *conn, int framed, size_t size);

#endif
