#include "rpc.h"
#include "lupine_log.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <new>
#include <string.h>
#include <thread>

#ifndef _WIN32
#include <netdb.h>
#include <sys/mman.h>
#endif

#if defined(__linux__) && !defined(MAP_FIXED_NOREPLACE)
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace {

#if !defined(_WIN32) && !defined(__APPLE__)
void *lupine_va_reserve_exact(uintptr_t base, size_t size) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  flags |= MAP_FIXED_NOREPLACE;
  void *mapping =
      mmap(reinterpret_cast<void *>(base), size, PROT_NONE, flags, -1, 0);
  if (mapping == MAP_FAILED) {
    return nullptr;
  }
  if (mapping != reinterpret_cast<void *>(base)) {
    munmap(mapping, size);
    return nullptr;
  }
  return mapping;
}
#endif

void lupine_va_destroy(conn_t *conn) {
  if (conn == nullptr || conn->va_size == 0) {
    return;
  }
#if !defined(_WIN32) && !defined(__APPLE__)
  munmap(reinterpret_cast<void *>(conn->va_base + conn->w_offset),
         conn->va_size);
  munmap(reinterpret_cast<void *>(conn->va_base), conn->va_size);
#endif
  conn->va_base = 0;
  conn->va_size = 0;
  conn->va_next = 0;
}

} // namespace

lupine_va_window lupine_va_local_window(void) {
#if defined(_WIN32) || defined(__APPLE__)
  // No arena can be hosted here, so state none and let the peer fall back.
  return {};
#else
  return {LUPINE_VA_FIRST_BASE, LUPINE_VA_ARENA_SIZE * LUPINE_VA_ARENA_COUNT};
#endif
}

int lupine_va_reserve_client(conn_t *conn, const lupine_va_window &window,
                             unsigned int min_slot, unsigned int *slot) {
  const size_t arena = window.size / LUPINE_VA_ARENA_COUNT;
  if (conn == nullptr || slot == nullptr || conn->va_size != 0 || arena == 0) {
    return -1;
  }
#if defined(_WIN32) || defined(__APPLE__)
  (void)min_slot;
  return 1;
#else
  for (unsigned int candidate = min_slot; candidate < LUPINE_VA_ARENA_COUNT;
       ++candidate) {
    uintptr_t offset = static_cast<uintptr_t>(candidate) * arena;
    uintptr_t base = window.base + offset;
    uintptr_t write_base = LUPINE_VA_WRITE_BASE + offset;
    if (lupine_va_reserve_exact(base, arena) == nullptr) {
      continue;
    }
    if (lupine_va_reserve_exact(write_base, arena) == nullptr) {
      munmap(reinterpret_cast<void *>(base), arena);
      continue;
    }
    conn->va_base = base;
    conn->va_size = arena;
    conn->va_next = 0;
    conn->w_offset =
        static_cast<intptr_t>(write_base) - static_cast<intptr_t>(base);
    *slot = candidate;
    return 0;
  }
  return -1;
#endif
}

bool lupine_va_claim(conn_t *conn, size_t size, size_t alignment,
                     uintptr_t *claimed) {
  if (conn == nullptr || claimed == nullptr || conn->va_size == 0 ||
      size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      size > conn->va_size) {
    return false;
  }
  uintptr_t current = __atomic_load_n(&conn->va_next, __ATOMIC_RELAXED);
  for (;;) {
    uintptr_t next_claim = (current + alignment - 1) & ~(alignment - 1);
    if (next_claim < current || next_claim < conn->va_base ||
        next_claim - conn->va_base > conn->va_size - size) {
      return false;
    }
    if (__atomic_compare_exchange_n(&conn->va_next, &current, next_claim + size,
                                    true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
      *claimed = next_claim;
      return true;
    }
  }
}

int lupine_va_reserve_server(conn_t *conn, uintptr_t base, size_t size) {
  if (conn == nullptr || conn->va_size != 0) {
    return -1;
  }
#if defined(_WIN32) || defined(__APPLE__)
  return -1;
#else
  if (lupine_va_reserve_exact(base, size) == nullptr) {
    return -1;
  }
  // Keep the placeholder for the connection lifetime. Backends replace only
  // monotonically claimed portions, so MAP_FIXED never touches foreign VMAs.
  conn->va_base = base;
  conn->va_size = size;
  conn->va_next = base;
  return 0;
#endif
}

// lupine_tcp_connect only resolves and dials; the caller owns TLS setup and
// the HTTP/2 session. Retried connections use bounded attempts and exponential
// backoff; zero retries preserves the one-shot blocking policy.
lupine_socket_t lupine_tcp_connect(const char *host, const char *port,
                                   unsigned int max_retries) {
  constexpr int kInitialBackoffMs = 1000;
  constexpr int kMaxBackoffMs = 30000;
  constexpr int kConnectTimeoutMs = 10000;

  // Every client dial funnels through here, and on Windows no socket call --
  // getaddrinfo included -- works until Winsock has been started. The server
  // starts it from main; a client shim has no such entry point of its own.
  if (lupine_socket_init() < 0) {
    LUPINE_LOG_ERROR("Socket initialization failed");
    return LUPINE_INVALID_SOCKET;
  }

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
        int result = max_retries == 0
                         ? connect(sockfd, ai->ai_addr,
                                   static_cast<socklen_t>(ai->ai_addrlen))
                         : lupine_socket_connect_with_timeout(
                               sockfd, ai->ai_addr,
                               static_cast<socklen_t>(ai->ai_addrlen),
                               kConnectTimeoutMs);
        if (result == 0) {
          freeaddrinfo(res);
          return sockfd;
        }
        lupine_socket_close(sockfd);
        if (max_retries == 0) {
          break;
        }
      }
      freeaddrinfo(res);
    }

    if (static_cast<unsigned int>(attempt) >= max_retries) {
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
                                      << "ms (" << (max_retries - attempt)
                                      << " retries left)");
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
}

namespace {

void rpc_shutdown_socket(lupine_socket_t socket) {
#ifdef _WIN32
  (void)shutdown(socket, SD_RECEIVE);
#else
  (void)shutdown(socket, SHUT_RD);
#endif
}

} // namespace

void rpc_shutdown_transport_socket(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }

#ifdef _WIN32
  conn->closed = 1;
  const lupine_socket_t socket = conn->connfd;
#else
  __atomic_store_n(&conn->closed, 1, __ATOMIC_RELEASE);
  const lupine_socket_t socket =
      __atomic_load_n(&conn->connfd, __ATOMIC_ACQUIRE);
#endif
  if (socket != LUPINE_INVALID_SOCKET) {
    rpc_shutdown_socket(socket);
  }
}

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
  rpc_shutdown_socket(socket);
  rpc_http2_destroy(conn);
  (void)lupine_socket_close(socket);
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

static int rpc_write_queue_reset(conn_t *conn, size_t count) {
  rpc_write_buffer_release(conn);
  if (conn == nullptr) {
    return -1;
  }
  conn->write_queue.clear();
  try {
    conn->write_queue.resize(count);
  } catch (const std::bad_alloc &) {
    return -1;
  }
  return 0;
}

static int rpc_write_queue_push(conn_t *conn, rpc_write_cursor cursor) {
  if (conn == nullptr ||
      conn->write_queue.size() >= static_cast<size_t>(INT_MAX) ||
      conn->write_queue.size() == conn->write_queue.max_size()) {
    return -1;
  }
  try {
    conn->write_queue.push_back(cursor);
  } catch (const std::bad_alloc &) {
    return -1;
  }
  return 0;
}

int rpc_conn_init(conn_t *conn, lupine_socket_t connfd, int request_id) {
  *conn = {};
  conn->connfd = connfd;
  conn->request_id = request_id;
  conn->local_request_parity = request_id & 1;
  if (pthread_mutex_init(&conn->write_mutex, nullptr) != 0) {
    goto fail;
  }
  if (pthread_mutex_init(&conn->call_mutex, nullptr) != 0) {
    pthread_mutex_destroy(&conn->write_mutex);
    goto fail;
  }
  return 0;

fail:
  rpc_close_transport_socket(conn);
  *conn = {};
  conn->connfd = LUPINE_INVALID_SOCKET;
  return -1;
}

void rpc_conn_destroy(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  rpc_close_transport_socket(conn);
  rpc_http2_destroy(conn);
  lupine_va_destroy(conn);
  rpc_write_buffer_release(conn);
  std::vector<rpc_write_cursor>().swap(conn->write_queue);
  std::vector<rpc_host_allocation_write>().swap(conn->host_allocation_writes);
  pthread_mutex_destroy(&conn->write_mutex);
  pthread_mutex_destroy(&conn->call_mutex);
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

struct rpc_read_frame {
  int32_t stream_id = -1;
  int request_id = 0;
};

struct rpc_response_route {
  int request_id = 0;
  int32_t stream_id = -1;
};

struct rpc_thread_io {
  conn_t *bound_conn = nullptr;
  int32_t bound_stream = -1;
  conn_t *read_conn = nullptr;
  rpc_read_frame read;
  conn_t *response_conn = nullptr;
  rpc_response_route response;
  conn_t *held_call_lock = nullptr;
};

static thread_local rpc_thread_io rpc_tls_io;

void rpc_release_held_call_lock(conn_t *conn) {
  if (rpc_tls_io.held_call_lock == conn) {
    rpc_tls_io.held_call_lock = nullptr;
    pthread_mutex_unlock(&conn->call_mutex);
  }
}

} // namespace

int rpc_bind_http2_stream(conn_t *conn, int32_t stream_id) {
  if (conn == nullptr || stream_id < 0) {
    return -1;
  }
  rpc_tls_io.bound_conn = conn;
  rpc_tls_io.bound_stream = stream_id;
  return 0;
}

void rpc_unbind_http2_stream(conn_t *conn) {
  if (rpc_tls_io.bound_conn == conn) {
    rpc_tls_io.bound_conn = nullptr;
    rpc_tls_io.bound_stream = -1;
  }
}

int32_t rpc_current_http2_stream(conn_t *conn) {
  if (rpc_tls_io.read_conn == conn) {
    return rpc_tls_io.read.stream_id;
  }
  if (rpc_tls_io.bound_conn == conn) {
    return rpc_tls_io.bound_stream;
  }
  return rpc_http2_dispatch_stream(conn);
}

int rpc_dispatch(conn_t *conn, int parity) {
  if (conn == nullptr || conn->closed) {
    return -1;
  }
  int32_t stream_id = rpc_current_http2_stream(conn);
  if (stream_id < 0) {
    return -1;
  }
  int request_id = 0;
  int op = 0;
  int read_result =
      rpc_http2_read_stream(conn, stream_id, &request_id, sizeof(request_id));
  if (read_result != sizeof(request_id) || request_id < 2 ||
      request_id % 2 != parity) {
    if (read_result != LUPINE_RPC_HTTP2_STREAM_END) {
      rpc_mark_connection_closed(conn);
    }
    return -1;
  }
  read_result = rpc_http2_read_stream(conn, stream_id, &op, sizeof(op));
  if (read_result != sizeof(op)) {
    if (read_result != LUPINE_RPC_HTTP2_STREAM_END) {
      rpc_mark_connection_closed(conn);
    }
    return -1;
  }
  rpc_tls_io.read_conn = conn;
  rpc_tls_io.read = {stream_id, request_id};
  return op;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// Once this returns, the matching frame is reserved for the caller until
// rpc_read_end() completes it.
int rpc_read_start(conn_t *conn, int write_id) {
  if (rpc_tls_io.response_conn != conn ||
      rpc_tls_io.response.request_id != write_id) {
    rpc_release_held_call_lock(conn);
    return -1;
  }
  int32_t stream_id = rpc_tls_io.response.stream_id;
  int request_id = 0;
  int op = 0;
  if (rpc_http2_read_stream(conn, stream_id, &request_id, sizeof(request_id)) !=
          sizeof(request_id) ||
      request_id != write_id ||
      rpc_http2_read_stream(conn, stream_id, &op, sizeof(op)) != sizeof(op) ||
      op != -1) {
    rpc_mark_connection_closed(conn);
    rpc_release_held_call_lock(conn);
    return -1;
  }
  rpc_tls_io.read_conn = conn;
  rpc_tls_io.read = {stream_id, request_id};
  return 0;
}

static int rpc_read_into_context(conn_t *conn, void *data, size_t size,
                                 int (*read)(conn_t *, void *, size_t)) {
  void *destination = data;
  bool host_allocation = false;
  uintptr_t address = reinterpret_cast<uintptr_t>(data);
  if (conn->va_size != 0 && conn->w_offset != 0 &&
      lupine_va_contains(conn, address, size)) {
    destination = reinterpret_cast<void *>(address + conn->w_offset);
    host_allocation = true;
  } else if (conn->va_size == 0 && conn->w_offset != 0) {
    uintptr_t read_base =
        LUPINE_HOST_ALLOCATION_SERVER_BASE + LUPINE_HOST_ALLOCATION_R_OFFSET;
    if (address >= read_base && size <= LUPINE_HOST_ALLOCATION_WINDOW_SIZE &&
        address - read_base <= LUPINE_HOST_ALLOCATION_WINDOW_SIZE - size) {
      uintptr_t server_address = address - LUPINE_HOST_ALLOCATION_R_OFFSET;
      destination = reinterpret_cast<void *>(server_address + conn->w_offset);
      host_allocation = true;
    }
  }
  int result = read(conn, destination, size);
  if (result < 0 || !host_allocation) {
    return result;
  }

  size_t written = static_cast<size_t>(result);
  if (pthread_mutex_lock(&conn->write_mutex) != 0) {
    return -1;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(data);
  try {
    if (!conn->host_allocation_writes.empty() &&
        conn->host_allocation_writes.back().start +
                conn->host_allocation_writes.back().size ==
            start) {
      conn->host_allocation_writes.back().size += written;
    } else {
      conn->host_allocation_writes.push_back({start, written});
    }
  } catch (const std::bad_alloc &) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  if (written != 0) {
    __atomic_store_n(&conn->host_allocation_writes_pending, 1,
                     __ATOMIC_RELEASE);
  }
  if (pthread_mutex_unlock(&conn->write_mutex) != 0) {
    return -1;
  }
  if (written != 0) {
    long configured_page_size = sysconf(_SC_PAGESIZE);
    uintptr_t page_size = configured_page_size > 0
                              ? static_cast<uintptr_t>(configured_page_size)
                              : static_cast<uintptr_t>(4096);
    uintptr_t page_start = start & ~(page_size - 1);
    uintptr_t page_end = (start + written + page_size - 1) & ~(page_size - 1);
    if (mprotect(reinterpret_cast<void *>(page_start), page_end - page_start,
                 PROT_READ) < 0) {
      return -1;
    }
  }
  return result;
}

static int rpc_read_http2(conn_t *conn, void *data, size_t size) {
  int32_t stream_id = rpc_current_http2_stream(conn);
  return stream_id < 0 ? -1
                       : rpc_http2_read_stream(conn, stream_id, data, size);
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  return rpc_read_into_context(conn, data, size, rpc_read_http2);
}

int rpc_read_pitched(conn_t *conn, void *data, size_t width, size_t rows,
                     size_t row_stride, size_t slices, size_t slice_stride) {
  for (size_t z = 0; z < slices; ++z) {
    char *slice = (char *)data + z * slice_stride;
    for (size_t row = 0; row < rows; ++row) {
      if (rpc_read(conn, slice + row * row_stride, width) < 0) {
        return -1;
      }
    }
  }
  return 0;
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
  if (rpc_tls_io.read_conn == conn) {
    int read_id = rpc_tls_io.read.request_id;
    rpc_tls_io.read_conn = nullptr;
    rpc_tls_io.read = {};
    rpc_release_held_call_lock(conn);
    return read_id;
  }
  rpc_release_held_call_lock(conn);
  if (conn != nullptr && !conn->closed) {
    return -1;
  }
  return -1;
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

  if (rpc_write_queue_reset(conn, 2) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  conn->write_stream_id = rpc_http2_lane_stream(conn, rpc_tls_lane.id);
  if (conn->write_stream_id < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
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

  if (rpc_write_queue_reset(conn, 2) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_id = read_id;
  conn->write_op = -1;
  conn->write_stream_id = rpc_current_http2_stream(conn);
  if (conn->write_stream_id < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  if (size == 0) {
    return 0;
  }
  return rpc_write_queue_push(conn, rpc_write_cursor(data, size));
}

// Rows are queued, not copied, so the caller's buffer must stay valid until
// the surrounding rpc_write_end.
int rpc_write_pitched(conn_t *conn, const void *data, size_t width, size_t rows,
                      size_t row_stride, size_t slices, size_t slice_stride) {
  for (size_t z = 0; z < slices; ++z) {
    const char *slice = (const char *)data + z * slice_stride;
    for (size_t row = 0; row < rows; ++row) {
      if (rpc_write(conn, slice + row * row_stride, width) < 0) {
        return -1;
      }
    }
  }
  return 0;
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

int rpc_write_cursors(conn_t *conn, const rpc_write_cursor *cursors,
                      size_t count) {
  if (count == 0) {
    return 0;
  }
  if (conn == nullptr || cursors == nullptr ||
      count > static_cast<size_t>(INT_MAX) - conn->write_queue.size() ||
      count > conn->write_queue.max_size() - conn->write_queue.size()) {
    return -1;
  }

  try {
    conn->write_queue.reserve(conn->write_queue.size() + count);
  } catch (const std::bad_alloc &) {
    return -1;
  }
  for (size_t i = 0; i < count; ++i) {
    if (cursors[i].data == nullptr && cursors[i].size != 0) {
      return -1;
    }
    conn->write_queue.push_back(cursors[i]);
  }
  return 0;
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
  int32_t write_stream_id = conn->write_stream_id;
  int result = -1;
  if (conn->write_queue.size() >= 2) {
    conn->write_queue[0] =
        rpc_write_cursor(&conn->write_id, sizeof(conn->write_id));
    conn->write_queue[1] =
        rpc_write_cursor(&conn->write_op, sizeof(conn->write_op));
    result = rpc_http2_write_stream(conn, write_stream_id, conn->write_queue);
  }
  rpc_write_buffer_release(conn);
  pthread_mutex_unlock(&conn->write_mutex);
  if (request) {
    if (result == 0) {
      rpc_tls_io.response_conn = conn;
      rpc_tls_io.response = {write_id, write_stream_id};
    }
    // Servers originate RPCs only on the dedicated server-to-client stream.
    // Keep that stream's request/response exchange serialized so a caller can
    // read its response directly without an application-level dispatcher.
    if (result == 0 && conn->local_request_parity == 1) {
      rpc_tls_io.held_call_lock = conn;
    } else {
      pthread_mutex_unlock(&conn->call_mutex);
    }
  }
  return result == 0 ? write_id : -1;
}
