#include "rpc.h"
#include <algorithm>
#include <iostream>
#include <string.h>

void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return NULL;

  while (!conn->closed) {
    while (conn->read_id != 0 && !conn->closed)
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
    if (conn->closed)
      break;

    // the read id is zero so it's our turn to read the next int which is the
    // request id of the next request.
    int bytes = rpc_read(conn, &conn->read_id, sizeof(int));
    if (bytes <= 0 || conn->read_id == 0) {
      conn->closed = 1;
      pthread_cond_broadcast(&conn->read_cond);
      break;
    }
    if (pthread_cond_broadcast(&conn->read_cond) < 0)
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
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch,
                     (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  int op;

  while (!conn->closed && (conn->read_id < 2 || conn->read_id % 2 != parity))
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

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
  while (!conn->closed && conn->read_id != write_id)
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) < 0)
      return -1;

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  return rpc_http2_read(conn, data, size);
}

int rpc_drain(conn_t *conn, size_t size) {
  char buffer[64 * 1024];
  size_t offset = 0;
  while (offset < size) {
    size_t chunk = std::min(sizeof(buffer), size - offset);
    if (rpc_read(conn, buffer, chunk) < 0) {
      return -1;
    }
    offset += chunk;
  }
  return 0;
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(conn_t *conn) {
  int read_id = conn->read_id;
  bool completes_local_request =
      read_id >= 2 && (read_id % 2) == conn->local_request_parity;
  conn->read_id = 0;
  if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
      pthread_mutex_unlock(&conn->read_mutex) < 0)
    return -1;
  if (completes_local_request && pthread_mutex_unlock(&conn->call_mutex) < 0)
    return -1;
  return read_id;
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  return 0;
}

int rpc_async_enabled(conn_t *conn) { return conn->async_window > 0; }

// rpc_async_collect_one waits for the reply to a fire-and-forget request, reads
// its 4-byte CUresult, and releases the read slot WITHOUT touching call_mutex
// (the sender already released it). Returns the CUresult, or -1 on error.
static int rpc_async_collect_one(conn_t *conn, int write_id) {
  if (rpc_read_start(conn, write_id) < 0)
    return -1;
  int rv = 0;
  int ok = rpc_read(conn, &rv, sizeof(rv));
  conn->read_id = 0;
  pthread_cond_broadcast(&conn->read_cond);
  pthread_mutex_unlock(&conn->read_mutex);
  return ok < 0 ? -1 : rv;
}

// rpc_async_drain_locked collects every outstanding fire-and-forget reply in
// FIFO order, latching the first error. The caller must hold call_mutex.
static int rpc_async_drain_locked(conn_t *conn) {
  const int cap = (int)(sizeof(conn->async_ids) / sizeof(conn->async_ids[0]));
  while (conn->async_count > 0) {
    int id = conn->async_ids[conn->async_head];
    int rv = rpc_async_collect_one(conn, id);
    conn->async_head = (conn->async_head + 1) % cap;
    conn->async_count--;
    if (rv < 0) {
      conn->closed = 1;
      return -1;
    }
    if (rv != 0 && conn->async_deferred == 0)
      conn->async_deferred = rv;
  }
  return 0;
}

// rpc_async_post sends the current request as fire-and-forget. Mirrors
// rpc_wait_for_response: it owns the call_mutex on entry and releases it before
// returning. The reply is collected by a later rpc_async_drain_locked.
int rpc_async_post(conn_t *conn) {
  int write_id = rpc_write_end(conn);
  if (write_id < 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  const int cap = (int)(sizeof(conn->async_ids) / sizeof(conn->async_ids[0]));
  int idx = (conn->async_head + conn->async_count) % cap;
  conn->async_ids[idx] = write_id;
  conn->async_count++;
  int rc = 0;
  if (conn->async_count >= conn->async_window || conn->async_count >= cap - 1)
    rc = rpc_async_drain_locked(conn);
  pthread_mutex_unlock(&conn->call_mutex);
  return rc;
}

int rpc_async_take_deferred(conn_t *conn) {
  if (conn->async_window <= 0)
    return 0;
  if (pthread_mutex_lock(&conn->call_mutex) < 0)
    return 0;
  int e = conn->async_deferred;
  conn->async_deferred = 0;
  pthread_mutex_unlock(&conn->call_mutex);
  return e;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(conn_t *conn, const int op) {
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->call_mutex) < 0) {
    return -1;
  }
  if (conn->closed) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  // Before issuing any request other than another fire-and-forget launch,
  // collect the replies to outstanding launches so this request's response is
  // next in FIFO order on the wire. Consecutive launches are intentionally NOT
  // drained here so they can pipeline (rpc_async_post bounds the backlog).
  if (conn->async_window > 0 && conn->async_count > 0 && op != conn->async_op) {
    rpc_async_drain_locked(conn);
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    pthread_mutex_unlock(&conn->call_mutex);
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
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  conn->write_iov_count = 1; // skip 1 for the header
  conn->write_id = read_id;
  conn->write_op = -1;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov_framed[conn->write_iov_count] = 0;
  conn->write_iov[conn->write_iov_count++] = {(void *)data, size};
  return 0;
}

// rpc_write_framed queues a payload that the transport LZ4-frames lazily,
// one block at a time, as the bytes are streamed to the socket. The caller's
// buffer must stay valid until rpc_write_end() returns, exactly like
// rpc_write(). See compress.cpp for the framing format.
int rpc_write_framed(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov_framed[conn->write_iov_count] = 1;
  conn->write_iov[conn->write_iov_count++] = {(void *)data, size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  if (conn->closed) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_iov[0] = {&conn->write_id, sizeof(int)};
  conn->write_iov_framed[0] = 0;
  if (conn->write_op != -1) {
    conn->write_iov[1] = {&conn->write_op, sizeof(unsigned int)};
    conn->write_iov_framed[1] = 0;
  }
  int write_id = conn->write_id;
  int iov_count = conn->write_iov_count;
  struct iovec iov[128];
  unsigned char framed[128];
  memcpy(iov, conn->write_iov, sizeof(struct iovec) * iov_count);
  memcpy(framed, conn->write_iov_framed, iov_count);

  int result = rpc_http2_writev(conn, iov, framed, iov_count);
  pthread_mutex_unlock(&conn->write_mutex);
  return result == 0 ? write_id : -1;
}
