#ifndef RPC_H
#define RPC_H

#include "cache.h"
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
extern int rpc_write_iovecs(conn_t *conn, const struct iovec *iovecs,
                            size_t count);
extern int rpc_write_framed(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);
extern int rpc_write_lane_termination(conn_t *conn, uint64_t lane_id);
extern void rpc_write_queue_free(conn_t *conn);
extern void rpc_conn_destroy(conn_t *conn);

extern int rpc_write_kernel_param_values(conn_t *conn, uint32_t count,
                                         const size_t *sizes,
                                         void *const *values);
extern int rpc_read_kernel_param_values(conn_t *conn, uint32_t count,
                                        const size_t *offsets,
                                        const size_t *sizes,
                                        size_t payload_size, void *storage,
                                        size_t storage_size, void **values);
extern int
rpc_write_kernel_param_layout(conn_t *conn,
                              const lupine_kernel_param_layout *layout);
extern int rpc_read_kernel_param_layout(conn_t *conn,
                                        lupine_kernel_param_layout *layout);
extern int rpc_write_launch_config(conn_t *conn, const CUlaunchConfig *config);
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
  uint32_t output_count = 0;
  CUjit_option output_options[3];
  size_t output_sizes[3];
  const void *output_data[3];
  bool capture_wall_time = false;
  bool capture_info_log = false;
  bool capture_error_log = false;
};
extern int rpc_write_jit_options(conn_t *conn, const unsigned int *num_options,
                                 const CUjit_option *options,
                                 void *const *option_values,
                                 std::vector<uintptr_t> *raw_values);
extern int rpc_read_jit_options(conn_t *conn,
                                std::vector<CUjit_option> *options,
                                std::vector<uintptr_t> *raw_values);
extern int rpc_read_jit_options(conn_t *conn, rpc_jit_server_state *state);
extern int rpc_write_library_options(conn_t *conn,
                                     const unsigned int *num_options,
                                     const CUlibraryOption *options,
                                     void *const *option_values,
                                     std::vector<uintptr_t> *raw_values);
extern int rpc_read_library_options(conn_t *conn,
                                    std::vector<CUlibraryOption> *options,
                                    std::vector<uintptr_t> *raw_values,
                                    bool *has_option_values);
extern int rpc_write_jit_outputs(conn_t *conn, const uint32_t *output_count,
                                 const CUjit_option *options,
                                 const size_t *sizes, const void *const *data);
extern int rpc_write_jit_outputs(conn_t *conn, rpc_jit_server_state *state);
extern int
rpc_read_jit_outputs(conn_t *conn,
                     const std::vector<rpc_jit_output_binding> &bindings);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_writev(conn_t *conn, const rpc_write_entry *entries,
                            int entry_count);
extern int rpc_http2_client_init(conn_t *conn);
extern int rpc_http2_server_init(conn_t *conn);
extern int rpc_http2_compress_lz4(conn_t *conn);
extern int rpc_http2_get_read_stats(conn_t *conn, rpc_http2_read_stats *stats);

// Optional LZ4 framing for large memory transfer payloads (see compress.cpp).
extern int lupine_payload_framed(conn_t *conn, size_t total_size);
extern int rpc_write_payload(conn_t *conn, const void *data, size_t size);
extern int rpc_read_payload(conn_t *conn, void *data, size_t size);
extern int rpc_read_payload_part(conn_t *conn, int framed, void *data,
                                 size_t size);
extern int rpc_drain_payload(conn_t *conn, int framed, size_t size);

#endif
