#include <sys/socket.h>

#include "rpc.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>


void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return NULL;

  while (1) {
    while (conn->read_id != 0)
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

    // the read id is zero so it's our turn to read the next int which is the
    // request id of the next request.
    if (rpc_read(conn, &conn->read_id, sizeof(int)) < 0 || conn->read_id == 0 ||
        pthread_cond_broadcast(&conn->read_cond) < 0)
      break;
  }
  pthread_mutex_unlock(&conn->read_mutex);
  conn->rpc_thread = 0;
  return NULL;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a
// request that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on
// entry.
int rpc_dispatch(conn_t *conn, int parity) {
  if (conn->rpc_thread == 0 &&
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch, (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  int op;

  while (conn->read_id < 2 || conn->read_id % 2 != parity)
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

  if (rpc_read(conn, &op, sizeof(int)) < 0) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return op;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on entry.
int rpc_read_start(conn_t *conn, int write_id) {
  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return -1;

  // wait for the active read id to be the request id we are waiting for
  while (conn->read_id != write_id)
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) < 0)
      return -1;

  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  int bytes_read = recv(conn->connfd, data, size, MSG_WAITALL);
  if (bytes_read == -1) {
    printf("recv error: %s\n", strerror(errno));
  }
  return bytes_read;
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(conn_t *conn) {
  int read_id = conn->read_id;
  conn->read_id = 0;
  if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
      pthread_mutex_unlock(&conn->read_mutex) < 0)
    return -1;
  return read_id;
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0)
    return -1;
  return 0;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(conn_t *conn, const int op) {
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    fprintf(stderr, "rpc_write_start failed due to rpc_open() < 0 || conns[index].write_mutex lock\n");
#endif
    return -1;
  }

  conn->write_iov_count = 2;               // skip 2 for the header
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  return 0;
}
// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_response(conn_t *conn, const int read_id) {
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    fprintf(stderr, "rpc_write_start failed due to rpc_open() < 0 || conns[index].write_mutex lock\n");
#endif
    return -1;
  }

  conn->write_iov_count = 1; // skip 1 for the header
  conn->write_id = read_id;
  conn->write_op = -1;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov[conn->write_iov_count++] = (struct iovec){(void *)data, size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  conn->write_iov[0] = {&conn->write_id, sizeof(int)};
  if (conn->write_op != -1) {
    conn->write_iov[1] = {&conn->write_op, sizeof(unsigned int)};
  }

  // write the request to the server
  if (writev(conn->connfd, conn->write_iov, conn->write_iov_count) < 0 ||
      pthread_mutex_unlock(&conn->write_mutex) < 0)
    return -1;
  return conn->write_id;
}
