#include "rpc.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <netdb.h>
#include <netinet/tcp.h>
#endif

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
    int next_id = 0;
    int bytes = rpc_read(conn, &next_id, sizeof(int));
    if (bytes < 0) {
      // rpc_read marked the connection closed, woke all waiters and released
      // the read mutex.
      conn->rpc_thread = 0;
      return NULL;
    }
    if (bytes == 0 || next_id == 0) {
      conn->closed = 1;
      pthread_cond_broadcast(&conn->read_cond);
      break;
    }
    conn->read_id = next_id;
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
    // rpc_read marked the connection closed and released the read mutex.
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
  if (conn->closed)
    return -1;
  int result = rpc_http2_read(conn, data, size);
  if (result < 0) {
    // the transport failed at the socket level so no more data will ever
    // arrive. mark the connection dead, wake every thread waiting on it, and
    // release the read state so callers that skip rpc_read_end after a failed
    // read do not leave the connection locked. all callers hold read_mutex
    // for the duration of rpc_read, and conn->closed only transitions to 1
    // under read_mutex, so this cleanup runs at most once.
    int read_id = conn->read_id;
    bool completes_local_request =
        read_id >= 2 && (read_id % 2) == conn->local_request_parity;
    conn->read_id = 0;
    conn->closed = 1;
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
    if (completes_local_request)
      pthread_mutex_unlock(&conn->call_mutex);
  }
  return result;
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
  if (conn->write_op != -1) {
    conn->write_iov[1] = {&conn->write_op, sizeof(unsigned int)};
  }
  int write_id = conn->write_id;
  int iov_count = conn->write_iov_count;
  struct iovec iov[128];
  memcpy(iov, conn->write_iov, sizeof(struct iovec) * iov_count);

  int result = rpc_http2_writev(conn, iov, iov_count);
  pthread_mutex_unlock(&conn->write_mutex);
  return result == 0 ? write_id : -1;
}

static int _rpc_env_int(const char *name, int fallback) {
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return fallback;
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0)
    return fallback;
  return (int)parsed;
}

static void _rpc_sleep_ms(int ms) {
#ifdef _WIN32
  Sleep(ms);
#else
  while (ms > 0) {
    int chunk = std::min(ms, 1000);
    usleep((useconds_t)chunk * 1000);
    ms -= chunk;
  }
#endif
}

// rpc_client_connect resolves host:port and connects to it, retrying failed
// attempts with exponential backoff. a server may not be reachable yet, for
// example while a GPU is still being provisioned, so the number of retries
// and the initial backoff are configurable with LUPINE_CONNECT_RETRIES and
// LUPINE_CONNECT_BACKOFF_MS.
lupine_socket_t rpc_client_connect(const char *host, const char *port) {
  int retries = _rpc_env_int("LUPINE_CONNECT_RETRIES", 5);
  int backoff_ms = _rpc_env_int("LUPINE_CONNECT_BACKOFF_MS", 1000);

  for (int attempt = 0;; ++attempt) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) == 0) {
      lupine_socket_t sockfd =
          socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (sockfd != LUPINE_INVALID_SOCKET) {
        int flag = 1;
        setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (const char *)&flag,
                   sizeof(flag));
        if (connect(sockfd, res->ai_addr, (socklen_t)res->ai_addrlen) == 0) {
          freeaddrinfo(res);
          return sockfd;
        }
        lupine_socket_close(sockfd);
      }
      freeaddrinfo(res);
    }
    if (attempt >= retries)
      return LUPINE_INVALID_SOCKET;

    int delay_ms = backoff_ms;
    for (int i = 0; i < attempt && delay_ms < 30000; ++i)
      delay_ms *= 2;
    delay_ms = std::min(delay_ms, 30000);
    std::cerr << "Connecting to " << host << " port " << port
              << " failed, retrying in " << delay_ms << "ms ("
              << (retries - attempt) << " retries left)" << std::endl;
    _rpc_sleep_ms(delay_ms);
  }
}
