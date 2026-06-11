#ifndef RPC_H
#define RPC_H

#include "lupine_platform.h"

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
extern int rpc_write_end(conn_t *conn);

extern int rpc_http2_read(conn_t *conn, void *data, size_t size);
extern int rpc_http2_writev(conn_t *conn, struct iovec *iov, int iov_count);
extern int rpc_http2_client_init(conn_t *conn);
extern int rpc_http2_server_init(conn_t *conn);

#ifndef _WIN32
// Connects to host:port over TCP, initializes the HTTP/2 client session on
// conn and spawns dispatch_thread to read responses. If verbose is set,
// failures are logged. Returns:
//   0 on success
//  -1 if the TCP connection could not be established
//  -2 if connection-state initialization failed (the socket is closed)
//  -3 if dispatch_thread could not be spawned (the socket is left open)
extern int rpc_client_open_connection(const char *host, const char *port,
                                      conn_t *conn,
                                      void *(*dispatch_thread)(void *),
                                      bool verbose);
#endif

#endif
