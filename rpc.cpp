#include "rpc.h"
#include "lupine_log.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string.h>
#include <thread>

#ifndef _WIN32
#include <netdb.h>
#endif

// lupine_tcp_connect resolves host:port and connects with a bounded retry
// policy (see rpc.h). It only resolves and dials; the caller owns TLS setup
// and the HTTP/2 session. A transiently unreachable server (e.g. still
// provisioning) is retried with exponential backoff, and each attempt is
// bounded by a deadline so a packet-filtered port cannot stall the loop for
// minutes (the kernel's SYN retransmit backoff).
lupine_socket_t lupine_tcp_connect(const char *host, const char *port) {
  // Hardcoded connect policy: a few retries with exponential backoff to ride
  // out a server that is still starting, each capped so a black-holed port is
  // detected quickly instead of blocking for the full SYN-retransmit window.
  constexpr int kMaxRetries = 5;
  constexpr int kInitialBackoffMs = 1000;
  constexpr int kMaxBackoffMs = 30000;
  constexpr int kConnectTimeoutMs = 10000;

  for (int attempt = 0;; ++attempt) {
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    int gai_status = getaddrinfo(host, port, &hints, &res);
    if (gai_status != 0 || res == nullptr) {
      LUPINE_LOG_ERROR("Resolving "
                       << host << " port " << port << " failed: "
                       << (gai_status != 0 ? gai_strerror(gai_status)
                                           : strerror(errno)));
    } else {
      for (addrinfo *ai = res; ai != nullptr; ai = ai->ai_next) {
        lupine_socket_t sockfd =
            socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sockfd == LUPINE_INVALID_SOCKET) {
          continue;
        }
        lupine_socket_apply_transport_options(sockfd);
        if (lupine_socket_connect_with_timeout(
                sockfd, ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen),
                kConnectTimeoutMs) == 0) {
          freeaddrinfo(res);
          return sockfd;
        }
        lupine_socket_close(sockfd);
      }
      freeaddrinfo(res);
    }

    if (attempt >= kMaxRetries) {
      return LUPINE_INVALID_SOCKET;
    }

    int delay_ms = kInitialBackoffMs;
    for (int i = 0; i < attempt && delay_ms < kMaxBackoffMs; ++i) {
      delay_ms *= 2;
    }
    if (delay_ms > kMaxBackoffMs) {
      delay_ms = kMaxBackoffMs;
    }
    LUPINE_LOG_ERROR("Connecting to " << host << " port " << port
                                      << " failed, retrying in " << delay_ms
                                      << "ms (" << (kMaxRetries - attempt)
                                      << " retries left)");
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
}

extern void rpc_http2_destroy(conn_t *conn);

void rpc_close_transport_socket(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }

#ifdef _WIN32
  conn->closed = 1;
  lupine_socket_t socket = conn->connfd;
  conn->connfd = LUPINE_INVALID_SOCKET;
#else
  __atomic_store_n(&conn->closed, 1, __ATOMIC_RELEASE);
  lupine_socket_t socket = __atomic_exchange_n(
      &conn->connfd, LUPINE_INVALID_SOCKET, __ATOMIC_ACQ_REL);
#endif
  if (socket == LUPINE_INVALID_SOCKET) {
    return;
  }
  struct linger abortive = {};
  abortive.l_onoff = 1;
  abortive.l_linger = 0;
#ifdef _WIN32
  (void)setsockopt(socket, SOL_SOCKET, SO_LINGER,
                   reinterpret_cast<const char *>(&abortive), sizeof(abortive));
#else
  (void)setsockopt(socket, SOL_SOCKET, SO_LINGER, &abortive, sizeof(abortive));
#endif
  // Wake a reader blocked in recv without sending a FIN. SHUT_RDWR would
  // gracefully close the write side before SO_LINGER can reset the peer.
#ifdef _WIN32
  (void)shutdown(socket, SD_RECEIVE);
#else
  (void)shutdown(socket, SHUT_RD);
#endif
  (void)lupine_socket_close(socket);
}

static int rpc_write_queue_reserve(conn_t *conn, int capacity) {
  if (conn == nullptr || capacity < 0) {
    return -1;
  }
  if (capacity <= conn->write_queue_capacity) {
    return 0;
  }

  int new_capacity =
      conn->write_queue_capacity > 0 ? conn->write_queue_capacity : 16;
  while (new_capacity < capacity) {
    if (new_capacity > INT_MAX / 2) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2;
  }

  void *next = realloc(conn->write_queue, static_cast<size_t>(new_capacity) *
                                              sizeof(*conn->write_queue));
  if (next == nullptr) {
    return -1;
  }
  conn->write_queue = static_cast<rpc_write_entry *>(next);
  conn->write_queue_capacity = new_capacity;
  return 0;
}

static void rpc_write_buffer_release(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  free(conn->write_copy_buffer);
  conn->write_copy_buffer = nullptr;
  conn->write_copy_capacity = 0;
  conn->write_copy_offset = 0;
}

static int rpc_write_queue_reset(conn_t *conn, int count) {
  rpc_write_buffer_release(conn);
  if (conn != nullptr) {
    conn->write_queue_count = 0;
  }
  if (rpc_write_queue_reserve(conn, count) < 0) {
    return -1;
  }
  conn->write_queue_count = count;
  for (int i = 0; i < count; ++i) {
    conn->write_queue[i] = {};
  }
  return 0;
}

static int rpc_write_queue_push(conn_t *conn, const void *data, size_t size,
                                unsigned char framed) {
  if (conn == nullptr || conn->write_queue_count == INT_MAX ||
      rpc_write_queue_reserve(conn, conn->write_queue_count + 1) < 0) {
    return -1;
  }
  conn->write_queue[conn->write_queue_count++] = {{(void *)data, size}, framed};
  return 0;
}

void rpc_write_queue_free(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  rpc_write_buffer_release(conn);
  free(conn->write_queue);
  conn->write_queue = nullptr;
  conn->write_queue_count = 0;
  conn->write_queue_capacity = 0;
}

void rpc_conn_destroy(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  rpc_close_transport_socket(conn);
  rpc_http2_destroy(conn);
  rpc_write_queue_free(conn);
  pthread_mutex_destroy(&conn->read_mutex);
  pthread_mutex_destroy(&conn->write_mutex);
  pthread_mutex_destroy(&conn->call_mutex);
  pthread_cond_destroy(&conn->read_cond);
}

int rpc_write_lane_termination(conn_t *conn, uint64_t lane_id) {
  if (conn == nullptr || conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->call_mutex) != 0) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) != 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  conn->request_id = conn->request_id + 2;
  conn->write_id = conn->request_id;
  conn->write_op = LUPINE_RPC_TERMINATE_LANE;
  int result = -1;
  if (rpc_write_queue_reset(conn, 3) == 0) {
    conn->write_queue[0] = {{&conn->write_id, sizeof(conn->write_id)}, 0};
    conn->write_queue[1] = {{&lane_id, sizeof(lane_id)}, 0};
    conn->write_queue[2] = {{&conn->write_op, sizeof(conn->write_op)}, 0};
    result = rpc_http2_writev(conn, conn->write_queue, conn->write_queue_count);
  }
  pthread_mutex_unlock(&conn->write_mutex);
  pthread_mutex_unlock(&conn->call_mutex);
  return result;
}

namespace {

using rpc_connection_closed_hook = void (*)(conn_t *);
using rpc_thread_lane_destroyed_hook = void (*)(uint64_t);

std::atomic<rpc_connection_closed_hook> connection_closed_hook{nullptr};
std::atomic<rpc_thread_lane_destroyed_hook> thread_lane_destroyed_hook{nullptr};
std::atomic_flag lifecycle_hooks_set = ATOMIC_FLAG_INIT;

} // namespace

int rpc_set_lifecycle_hooks(const rpc_lifecycle_hooks *hooks) {
  if (hooks == nullptr ||
      lifecycle_hooks_set.test_and_set(std::memory_order_acq_rel)) {
    return -1;
  }
  connection_closed_hook.store(hooks->connection_closed,
                               std::memory_order_release);
  thread_lane_destroyed_hook.store(hooks->thread_lane_destroyed,
                                   std::memory_order_release);
  return 0;
}

static void rpc_mark_connection_closed(conn_t *conn) {
  conn->closed = 1;
  auto hook = connection_closed_hook.load(std::memory_order_acquire);
  if (hook != nullptr) {
    hook(conn);
  }
}

namespace {

struct rpc_thread_lane {
  uint64_t id = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));

  ~rpc_thread_lane() {
    auto hook = thread_lane_destroyed_hook.load(std::memory_order_acquire);
    if (hook != nullptr) {
      hook(id);
    }
  }
};

static thread_local rpc_thread_lane rpc_tls_lane;

uint64_t rpc_thread_lane_id(conn_t *conn) {
  (void)conn;
  return rpc_tls_lane.id;
}

} // namespace

void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  while (!conn->closed) {
    if (pthread_mutex_lock(&conn->read_mutex) != 0) {
      break;
    }
    while (conn->read_id != 0 && !conn->closed) {
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
    }
    if (conn->closed) {
      pthread_mutex_unlock(&conn->read_mutex);
      break;
    }

    int request_id = 0;
    if (rpc_http2_read(conn, &request_id, sizeof(request_id)) !=
            sizeof(request_id) ||
        request_id == 0) {
      rpc_mark_connection_closed(conn);
      pthread_cond_broadcast(&conn->read_cond);
      pthread_mutex_unlock(&conn->read_mutex);
      break;
    }

    conn->read_id = request_id;
    if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
        pthread_mutex_unlock(&conn->read_mutex) < 0) {
      break;
    }
  }
  rpc_mark_connection_closed(conn);
  pthread_cond_broadcast(&conn->read_cond);
  conn->rpc_thread = 0;
  return NULL;
}

int rpc_dispatch(conn_t *conn, int parity) {
  if (conn->rpc_thread == 0 &&
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch,
                     (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  while (!conn->closed && (conn->read_id < 2 || conn->read_id % 2 != parity)) {
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
  }

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  if (rpc_http2_read(conn, &conn->read_lane_id, sizeof(conn->read_lane_id)) !=
          sizeof(conn->read_lane_id) ||
      rpc_http2_read(conn, &conn->read_op, sizeof(conn->read_op)) !=
          sizeof(conn->read_op)) {
    rpc_mark_connection_closed(conn);
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }
  pthread_mutex_unlock(&conn->read_mutex);
  return conn->read_op;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// Once this returns, the matching frame is reserved for the caller until
// rpc_read_end() releases it back to the dispatch thread.
int rpc_read_start(conn_t *conn, int write_id) {
  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return -1;

  while (!conn->closed && conn->read_id != write_id) {
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) != 0) {
      pthread_mutex_unlock(&conn->read_mutex);
      return -1;
    }
  }

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  if (rpc_http2_read(conn, &conn->read_lane_id, sizeof(conn->read_lane_id)) !=
          sizeof(conn->read_lane_id) ||
      rpc_http2_read(conn, &conn->read_op, sizeof(conn->read_op)) !=
          sizeof(conn->read_op) ||
      conn->read_op != -1) {
    rpc_mark_connection_closed(conn);
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }
  pthread_mutex_unlock(&conn->read_mutex);
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

int rpc_read_end(conn_t *conn) {
  if (pthread_mutex_lock(&conn->read_mutex) != 0) {
    return -1;
  }
  int read_id = conn->read_id;
  conn->read_id = 0;
  if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
      pthread_mutex_unlock(&conn->read_mutex) < 0) {
    return -1;
  }
  return read_id;
}

// Per-op RPC statistics, enabled by setting LUPINE_RPC_STATS to an output
// path. Records how many requests each op sends and how long callers spend
// blocked waiting for the response; dumped as TSV at process exit.
#include <chrono>
#include <unordered_map>

struct lupine_rpc_stat {
  uint64_t count = 0;
  uint64_t wait_ns = 0;
};
static std::unordered_map<int, lupine_rpc_stat> lupine_rpc_stats;
static pthread_mutex_t lupine_rpc_stats_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char *lupine_rpc_stats_path() {
  static const char *path = getenv("LUPINE_RPC_STATS");
  return path;
}

static void lupine_rpc_stats_dump() {
  FILE *f = fopen(lupine_rpc_stats_path(), "w");
  if (f == nullptr) {
    return;
  }
  pthread_mutex_lock(&lupine_rpc_stats_mutex);
  for (const auto &entry : lupine_rpc_stats) {
    fprintf(f, "%d\t%llu\t%llu\n", entry.first,
            (unsigned long long)entry.second.count,
            (unsigned long long)entry.second.wait_ns);
  }
  pthread_mutex_unlock(&lupine_rpc_stats_mutex);
  fclose(f);
}

static void lupine_rpc_stats_record(int op, uint64_t count_inc,
                                    uint64_t wait_ns) {
  pthread_mutex_lock(&lupine_rpc_stats_mutex);
  static bool registered = false;
  if (!registered) {
    registered = true;
    atexit(lupine_rpc_stats_dump);
  }
  lupine_rpc_stat &stat = lupine_rpc_stats[op];
  stat.count += count_inc;
  stat.wait_ns += wait_ns;
  pthread_mutex_unlock(&lupine_rpc_stats_mutex);
}

static uint64_t lupine_rpc_stats_now_ns() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
  int op = conn->write_op;
  uint64_t start =
      lupine_rpc_stats_path() != nullptr ? lupine_rpc_stats_now_ns() : 0;
  int write_id = rpc_write_end(conn);
  if (write_id < 0) {
    return -1;
  }
  rpc_http2_response_wait_begin(conn);
  int result = rpc_read_start(conn, write_id);
  rpc_http2_response_wait_end(conn);
  if (result < 0) {
    return -1;
  }
  if (lupine_rpc_stats_path() != nullptr) {
    lupine_rpc_stats_record(op, 0, lupine_rpc_stats_now_ns() - start);
  }
  return 0;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(conn_t *conn, const int op) {
  if (conn == nullptr || conn->closed) {
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
    std::cerr << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }

  if (rpc_write_queue_reset(conn, 3) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  conn->write_lane_id = rpc_thread_lane_id(conn);
  if (lupine_rpc_stats_path() != nullptr) {
    lupine_rpc_stats_record(op, 1, 0);
  }
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
    std::cerr << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  if (rpc_write_queue_reset(conn, 3) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_id = read_id;
  conn->write_op = -1;
  conn->write_lane_id = conn->read_lane_id;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  if (size == 0) {
    return 0;
  }
  return rpc_write_queue_push(conn, data, size, 0);
}

int rpc_copy_alloc(conn_t *conn, const size_t size) {
  if (conn == nullptr || conn->write_copy_buffer != nullptr ||
      conn->write_copy_capacity != 0 || conn->write_copy_offset != 0) {
    return -1;
  }
  if (size == 0) {
    return 0;
  }
  conn->write_copy_buffer = static_cast<unsigned char *>(malloc(size));
  if (conn->write_copy_buffer == nullptr) {
    // Continuing after request serialization runs out of memory would leave
    // client-visible backend state ambiguous, so fail the process immediately.
    std::abort();
  }
  conn->write_copy_capacity = size;
  return 0;
}

void *rpc_write_buffer(conn_t *conn, size_t size, size_t alignment) {
  if (conn == nullptr || conn->write_copy_buffer == nullptr) {
    return nullptr;
  }

  size_t offset =
      (conn->write_copy_offset + alignment - 1) / alignment * alignment;
  size_t required = offset + size;
  if (required > conn->write_copy_capacity) {
    return nullptr;
  }
  void *buffer = conn->write_copy_buffer + offset;
  conn->write_copy_offset = required;
  if (rpc_write(conn, buffer, size) < 0) {
    return nullptr;
  }
  return buffer;
}

int rpc_write_iovecs(conn_t *conn, const struct iovec *iovecs, size_t count) {
  if (count == 0) {
    return 0;
  }
  if (conn == nullptr || iovecs == nullptr ||
      count > static_cast<size_t>(INT_MAX - conn->write_queue_count) ||
      rpc_write_queue_reserve(conn, conn->write_queue_count +
                                        static_cast<int>(count)) < 0) {
    return -1;
  }

  for (size_t i = 0; i < count; ++i) {
    if (iovecs[i].iov_base == nullptr && iovecs[i].iov_len != 0) {
      return -1;
    }
    conn->write_queue[conn->write_queue_count++] = {iovecs[i], 0};
  }
  return 0;
}

// rpc_write_framed queues a payload that the transport LZ4-frames lazily,
// one block at a time, as the bytes are streamed to the socket. The caller's
// buffer must stay valid until rpc_write_end() returns, exactly like
// rpc_write(). See compress.cpp for the framing format.
int rpc_write_framed(conn_t *conn, const void *data, const size_t size) {
  return rpc_write_queue_push(conn, data, size, 1);
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  bool request = conn->write_op != -1;
  if (conn->closed) {
    rpc_write_buffer_release(conn);
    pthread_mutex_unlock(&conn->write_mutex);
    if (request) {
      pthread_mutex_unlock(&conn->call_mutex);
    }
    return -1;
  }
  int write_id = conn->write_id;
  int result = -1;
  if (conn->write_queue_count >= 3) {
    conn->write_queue[0] = {{&conn->write_id, sizeof(conn->write_id)}, 0};
    conn->write_queue[1] = {{&conn->write_lane_id, sizeof(conn->write_lane_id)},
                            0};
    conn->write_queue[2] = {{&conn->write_op, sizeof(conn->write_op)}, 0};
    result = rpc_http2_writev(conn, conn->write_queue, conn->write_queue_count);
  }
  rpc_write_buffer_release(conn);
  pthread_mutex_unlock(&conn->write_mutex);
  if (request) {
    pthread_mutex_unlock(&conn->call_mutex);
  }
  return result == 0 ? write_id : -1;
}
