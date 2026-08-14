#include "rpc.h"
#include "lupine_log.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string.h>
#include <thread>
#include <vector>

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
    LUPINE_LOG_ERROR("Connecting to "
                     << host << " port " << port << " failed, retrying in "
                     << delay_ms << "ms (" << (kMaxRetries - attempt)
                     << " retries left)");
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }
}

extern void rpc_http2_destroy(conn_t *conn);

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
                                              sizeof(conn->write_queue[0]));
  if (next == nullptr) {
    return -1;
  }
  conn->write_queue = static_cast<rpc_write_entry *>(next);
  conn->write_queue_capacity = new_capacity;
  return 0;
}

static void rpc_write_queue_release_owned(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  for (int i = 0; i < conn->write_queue_count; ++i) {
    rpc_write_entry *entry = &conn->write_queue[i];
    if (entry->owned) {
      free(entry->iov.iov_base);
      *entry = {};
    }
  }
}

static int rpc_write_queue_reset(conn_t *conn, int count) {
  rpc_write_queue_release_owned(conn);
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
                                unsigned char framed, unsigned char owned) {
  if (conn == nullptr || conn->write_queue_count == INT_MAX ||
      rpc_write_queue_reserve(conn, conn->write_queue_count + 1) < 0) {
    return -1;
  }
  conn->write_queue[conn->write_queue_count++] = {
      {(void *)data, size}, framed, owned};
  return 0;
}

void rpc_write_queue_free(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  rpc_write_queue_release_owned(conn);
  free(conn->write_queue);
  conn->write_queue = nullptr;
  conn->write_queue_count = 0;
  conn->write_queue_capacity = 0;
}

void rpc_conn_destroy(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
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

#ifdef LUPINE_RPC_CLIENT
extern void rpc_destroy_thread_lane(uint64_t lane_id);
extern "C" void lupine_invalidate_current_context_cache();
#else
static void rpc_destroy_thread_lane(uint64_t lane_id) { (void)lane_id; }
#endif

static void rpc_mark_connection_closed(conn_t *conn) {
  conn->closed = 1;
#ifdef LUPINE_RPC_CLIENT
  lupine_invalidate_current_context_cache();
#endif
}

namespace {

struct rpc_thread_lane {
  uint64_t id = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));

  ~rpc_thread_lane() { rpc_destroy_thread_lane(id); }
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
  if (lupine_rpc_stats_path() != nullptr) {
    int op = conn->write_op;
    uint64_t start = lupine_rpc_stats_now_ns();
    int write_id = rpc_write_end(conn);
    if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
      return -1;
    }
    lupine_rpc_stats_record(op, 0, lupine_rpc_stats_now_ns() - start);
    return 0;
  }
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
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
  return rpc_write_queue_push(conn, data, size, 0, 0);
}

int rpc_write_copy(conn_t *conn, const void *data, const size_t size) {
  if (size == 0) {
    return rpc_write_queue_push(conn, data, size, 0, 0);
  }
  if (data == nullptr) {
    return -1;
  }
  void *copy = malloc(size);
  if (copy == nullptr) {
    return -1;
  }
  memcpy(copy, data, size);
  if (rpc_write_queue_push(conn, copy, size, 0, 1) < 0) {
    free(copy);
    return -1;
  }
  return 0;
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

// CUDA_KERNEL_NODE_PARAMS gained kern and ctx in CUDA 12. Keep the RPC payload
// fixed at the CUDA 12 layout so older and newer endpoints stay aligned.
static const uint32_t rpc_kernel_node_reserved = 0;
#if CUDA_VERSION < 12000
static const uint64_t rpc_kernel_node_null_handle = 0;
#endif

static_assert(sizeof(CUfunction) == sizeof(uint64_t),
              "CUDA function handles must fit the kernel node wire ABI");
static_assert(sizeof(unsigned int) == sizeof(uint32_t),
              "CUDA dimension fields must fit the kernel node wire ABI");
#if CUDA_VERSION >= 12000
static_assert(sizeof(CUkernel) == sizeof(uint64_t),
              "CUDA kernel handles must fit the kernel node wire ABI");
static_assert(sizeof(CUcontext) == sizeof(uint64_t),
              "CUDA context handles must fit the kernel node wire ABI");
#endif

int rpc_write_kernel_node_params(conn_t *conn,
                                 const CUDA_KERNEL_NODE_PARAMS *node_params) {
  if (conn == nullptr || node_params == nullptr) {
    return -1;
  }
  if (rpc_write_copy(conn, &node_params->func, sizeof(node_params->func)) < 0 ||
      rpc_write_copy(conn, &node_params->gridDimX,
                     sizeof(node_params->gridDimX)) < 0 ||
      rpc_write_copy(conn, &node_params->gridDimY,
                     sizeof(node_params->gridDimY)) < 0 ||
      rpc_write_copy(conn, &node_params->gridDimZ,
                     sizeof(node_params->gridDimZ)) < 0 ||
      rpc_write_copy(conn, &node_params->blockDimX,
                     sizeof(node_params->blockDimX)) < 0 ||
      rpc_write_copy(conn, &node_params->blockDimY,
                     sizeof(node_params->blockDimY)) < 0 ||
      rpc_write_copy(conn, &node_params->blockDimZ,
                     sizeof(node_params->blockDimZ)) < 0 ||
      rpc_write_copy(conn, &node_params->sharedMemBytes,
                     sizeof(node_params->sharedMemBytes)) < 0 ||
      rpc_write(conn, &rpc_kernel_node_reserved,
                sizeof(rpc_kernel_node_reserved)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12000
  if (rpc_write_copy(conn, &node_params->kern, sizeof(node_params->kern)) < 0 ||
      rpc_write_copy(conn, &node_params->ctx, sizeof(node_params->ctx)) < 0) {
    return -1;
  }
#else
  if (rpc_write(conn, &rpc_kernel_node_null_handle,
                sizeof(rpc_kernel_node_null_handle)) < 0 ||
      rpc_write(conn, &rpc_kernel_node_null_handle,
                sizeof(rpc_kernel_node_null_handle)) < 0) {
    return -1;
  }
#endif
  return 0;
}

int rpc_read_kernel_node_params(conn_t *conn,
                                CUDA_KERNEL_NODE_PARAMS *node_params) {
  if (conn == nullptr || node_params == nullptr) {
    return -1;
  }

  *node_params = {};
  if (rpc_read(conn, &node_params->func, sizeof(node_params->func)) < 0 ||
      rpc_read(conn, &node_params->gridDimX, sizeof(node_params->gridDimX)) <
          0 ||
      rpc_read(conn, &node_params->gridDimY, sizeof(node_params->gridDimY)) <
          0 ||
      rpc_read(conn, &node_params->gridDimZ, sizeof(node_params->gridDimZ)) <
          0 ||
      rpc_read(conn, &node_params->blockDimX, sizeof(node_params->blockDimX)) <
          0 ||
      rpc_read(conn, &node_params->blockDimY, sizeof(node_params->blockDimY)) <
          0 ||
      rpc_read(conn, &node_params->blockDimZ, sizeof(node_params->blockDimZ)) <
          0 ||
      rpc_read(conn, &node_params->sharedMemBytes,
               sizeof(node_params->sharedMemBytes)) < 0 ||
      rpc_drain(conn, sizeof(uint32_t)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12000
  if (rpc_read(conn, &node_params->kern, sizeof(node_params->kern)) < 0 ||
      rpc_read(conn, &node_params->ctx, sizeof(node_params->ctx)) < 0) {
    return -1;
  }
#else
  if (rpc_drain(conn, 2 * sizeof(uint64_t)) < 0) {
    return -1;
  }
#endif
  return 0;
}

int rpc_write_kernel_param_values(conn_t *conn, uint32_t count,
                                  const size_t *sizes, void *const *values) {
  if (conn == nullptr ||
      (count != 0 && (sizes == nullptr || values == nullptr))) {
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (values[i] == nullptr || rpc_write(conn, values[i], sizes[i]) < 0) {
      return -1;
    }
  }
  return 0;
}

int rpc_read_kernel_param_values(conn_t *conn, uint32_t count,
                                 const size_t *offsets, const size_t *sizes,
                                 size_t payload_size, void *storage,
                                 size_t storage_size, void **values) {
  if (conn == nullptr ||
      (count != 0 && (offsets == nullptr || sizes == nullptr ||
                      storage == nullptr || values == nullptr))) {
    return -1;
  }

  size_t expected_payload_size = 0;
  for (uint32_t i = 0; i < count; ++i) {
    expected_payload_size += sizes[i];
  }
  if (payload_size != expected_payload_size) {
    return -1;
  }

  auto *bytes = static_cast<unsigned char *>(storage);
  for (uint32_t i = 0; i < count; ++i) {
    if (offsets[i] + sizes[i] > storage_size) {
      return -1;
    }
    unsigned char *dst = bytes + offsets[i];
    if (sizes[i] != 0 && rpc_read(conn, dst, sizes[i]) < 0) {
      return -1;
    }
    values[i] = dst;
  }
  return 0;
}

#ifdef LUPINE_RPC_SERVER
static CUresult rpc_collect_kernel_param_layout(CUfunction function,
                                                bool allow_kernel_handle,
                                                std::vector<size_t> *offsets,
                                                std::vector<size_t> *sizes) {
  offsets->clear();
  sizes->clear();
  bool use_kernel_info = false;
  for (uint32_t i = 0;; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result =
        use_kernel_info
            ? cuKernelGetParamInfo(reinterpret_cast<CUkernel>(function), i,
                                   &offset, &size)
            : cuFuncGetParamInfo(function, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      return CUDA_SUCCESS;
    }
    if (allow_kernel_handle && i == 0 && result == CUDA_ERROR_INVALID_HANDLE) {
      use_kernel_info = true;
      result = cuKernelGetParamInfo(reinterpret_cast<CUkernel>(function), i,
                                    &offset, &size);
      if (result == CUDA_ERROR_INVALID_VALUE) {
        return CUDA_SUCCESS;
      }
    }
    if (result != CUDA_SUCCESS) {
      if (allow_kernel_handle && i != 0) {
        return CUDA_SUCCESS;
      }
      offsets->clear();
      sizes->clear();
      return result;
    }
    offsets->push_back(offset);
    sizes->push_back(size);
  }
}

static int rpc_write_param_layout(conn_t *conn,
                                  const std::vector<size_t> &offsets,
                                  const std::vector<size_t> &sizes) {
  if (conn == nullptr || offsets.size() != sizes.size() ||
      offsets.size() > UINT32_MAX) {
    return -1;
  }
  uint32_t count = static_cast<uint32_t>(offsets.size());
  size_t array_bytes = offsets.size() * sizeof(offsets[0]);
  std::vector<unsigned char> storage;
  try {
    storage.resize(sizeof(count) + 2 * array_bytes);
  } catch (...) {
    return -1;
  }
  unsigned char *dst = storage.data();
  memcpy(dst, &count, sizeof(count));
  if (array_bytes != 0) {
    memcpy(dst + sizeof(count), offsets.data(), array_bytes);
    memcpy(dst + sizeof(count) + array_bytes, sizes.data(), array_bytes);
  }
  return rpc_write_copy(conn, storage.data(), storage.size());
}

int rpc_write_function_param_layout(conn_t *conn, CUfunction function,
                                    CUresult *result) {
  if (conn == nullptr || result == nullptr) {
    return -1;
  }
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
  if (*result == CUDA_SUCCESS) {
    *result =
        rpc_collect_kernel_param_layout(function, false, &offsets, &sizes);
  }
  return rpc_write_param_layout(conn, offsets, sizes);
}

int rpc_read_kernel_param_values(conn_t *conn, CUfunction function,
                                 uint32_t expected_count, size_t payload_size,
                                 std::vector<unsigned char> *storage,
                                 std::vector<void *> *values,
                                 CUresult *result) {
  if (conn == nullptr || storage == nullptr || values == nullptr ||
      result == nullptr) {
    return -1;
  }
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
  *result = rpc_collect_kernel_param_layout(function, true, &offsets, &sizes);
  if (*result != CUDA_SUCCESS) {
    return 0;
  }
  if (sizes.size() != expected_count) {
    *result = CUDA_ERROR_INVALID_VALUE;
    return 0;
  }

  size_t storage_size = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    storage_size = std::max(storage_size, offsets[i] + sizes[i]);
  }
  storage->assign(storage_size, 0);
  values->resize(expected_count);
  if (rpc_read_kernel_param_values(conn, expected_count, offsets.data(),
                                   sizes.data(), payload_size, storage->data(),
                                   storage->size(), values->data()) < 0) {
    return -1;
  }
  return 0;
}

int rpc_read_kernel_node_param_values(conn_t *conn,
                                      CUDA_KERNEL_NODE_PARAMS *node_params,
                                      std::vector<unsigned char> *storage,
                                      std::vector<void *> *values,
                                      CUresult *result) {
  uint32_t param_count = 0;
  size_t payload_size = 0;
  if (conn == nullptr || node_params == nullptr || storage == nullptr ||
      values == nullptr || result == nullptr ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }

  CUfunction function = node_params->func;
#if CUDA_VERSION >= 12000
  if (function == nullptr) {
    function = reinterpret_cast<CUfunction>(node_params->kern);
  }
#endif
  *result = CUDA_SUCCESS;
  if (rpc_read_kernel_param_values(conn, function, param_count, payload_size,
                                   storage, values, result) < 0 ||
      (*result != CUDA_SUCCESS && rpc_drain(conn, payload_size) < 0)) {
    return -1;
  }
  if (*result == CUDA_SUCCESS) {
    node_params->kernelParams = values->empty() ? nullptr : values->data();
    node_params->extra = nullptr;
  }
  return 0;
}

int rpc_write_kernel_param_layout_and_values(
    conn_t *conn, CUfunction function, void *const *values,
    size_t *payload_size, CUresult *result) {
  if (conn == nullptr || payload_size == nullptr || result == nullptr) {
    return -1;
  }

  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
  std::vector<unsigned char> value_storage;
  *payload_size = 0;
  if (*result == CUDA_SUCCESS) {
    *result = rpc_collect_kernel_param_layout(function, true, &offsets, &sizes);
  }
  if (*result == CUDA_SUCCESS && !sizes.empty() && values == nullptr) {
    *result = CUDA_ERROR_INVALID_VALUE;
  }
  if (*result == CUDA_SUCCESS) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (values[i] == nullptr || SIZE_MAX - *payload_size < sizes[i]) {
        *result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      *payload_size += sizes[i];
    }
  }
  if (*result == CUDA_SUCCESS) {
    try {
      value_storage.resize(*payload_size);
    } catch (...) {
      *result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (*result == CUDA_SUCCESS) {
    size_t packed_offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] != 0) {
        memcpy(value_storage.data() + packed_offset, values[i], sizes[i]);
      }
      packed_offset += sizes[i];
    }
  } else {
    offsets.clear();
    sizes.clear();
    value_storage.clear();
    *payload_size = 0;
  }

  return rpc_write_param_layout(conn, offsets, sizes) < 0 ||
                 rpc_write(conn, payload_size, sizeof(*payload_size)) < 0 ||
                 rpc_write_copy(conn, value_storage.data(),
                                value_storage.size()) < 0
             ? -1
             : 0;
}
#endif

int rpc_read_kernel_param_layout(conn_t *conn, std::vector<size_t> *offsets,
                                 std::vector<size_t> *sizes) {
  uint32_t count = 0;
  if (conn == nullptr || offsets == nullptr || sizes == nullptr ||
      rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  offsets->resize(count);
  sizes->resize(count);
  if (rpc_read(conn, offsets->data(), offsets->size() * sizeof((*offsets)[0])) <
          0 ||
      rpc_read(conn, sizes->data(), sizes->size() * sizeof((*sizes)[0])) < 0) {
    return -1;
  }
  return 0;
}

int rpc_read_kernel_node_params_with_layout(
    conn_t *conn, CUDA_KERNEL_NODE_PARAMS *node_params,
    std::vector<unsigned char> *storage, std::vector<void *> *values) {
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
  size_t payload_size = 0;
  if (conn == nullptr || node_params == nullptr || storage == nullptr ||
      values == nullptr || rpc_read_kernel_node_params(conn, node_params) < 0 ||
      rpc_read_kernel_param_layout(conn, &offsets, &sizes) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }

  size_t storage_size = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (offsets[i] > SIZE_MAX - sizes[i]) {
      return -1;
    }
    storage_size = std::max(storage_size, offsets[i] + sizes[i]);
  }
  try {
    storage->assign(storage_size, 0);
    values->resize(sizes.size());
  } catch (...) {
    return -1;
  }
  if (rpc_read_kernel_param_values(conn, static_cast<uint32_t>(sizes.size()),
                                   offsets.data(), sizes.data(), payload_size,
                                   storage->data(), storage->size(),
                                   values->data()) < 0) {
    return -1;
  }
  node_params->kernelParams = values->empty() ? nullptr : values->data();
  node_params->extra = nullptr;
  return 0;
}

int rpc_write_launch_config(conn_t *conn, const CUlaunchConfig *config) {
  if (conn == nullptr || config == nullptr) {
    return -1;
  }
  if (rpc_write(conn, config, sizeof(*config)) < 0 ||
      (config->numAttrs != 0 && config->attrs != nullptr &&
       rpc_write(conn, config->attrs,
                 config->numAttrs * sizeof(config->attrs[0])) < 0)) {
    return -1;
  }
  return 0;
}

int rpc_read_launch_config(conn_t *conn, CUlaunchConfig *config,
                           std::vector<CUlaunchAttribute> *attributes) {
  if (conn == nullptr || config == nullptr || attributes == nullptr) {
    return -1;
  }
  if (rpc_read(conn, config, sizeof(*config)) < 0) {
    return -1;
  }
  if (config->numAttrs == 0 || config->attrs == nullptr) {
    attributes->clear();
    config->attrs = nullptr;
    return 0;
  }
  attributes->resize(config->numAttrs);
  if (rpc_read(conn, attributes->data(),
               config->numAttrs * sizeof((*attributes)[0])) < 0) {
    return -1;
  }
  config->attrs = attributes->data();
  return 0;
}

int rpc_write_jit_options(conn_t *conn, const unsigned int *num_options,
                          const CUjit_option *options,
                          void *const *option_values,
                          std::vector<uintptr_t> *raw_values) {
  if (conn == nullptr || num_options == nullptr || raw_values == nullptr ||
      (*num_options != 0 && (options == nullptr || option_values == nullptr))) {
    return -1;
  }

  raw_values->resize(*num_options);
  for (unsigned int i = 0; i < *num_options; ++i) {
    (*raw_values)[i] = reinterpret_cast<uintptr_t>(option_values[i]);
  }

  if (rpc_write(conn, num_options, sizeof(*num_options)) < 0 ||
      (*num_options != 0 &&
       rpc_write(conn, options, *num_options * sizeof(CUjit_option)) < 0)) {
    return -1;
  }
  for (unsigned int i = 0; i < *num_options; ++i) {
    if (rpc_write(conn, &(*raw_values)[i], sizeof((*raw_values)[i])) < 0) {
      return -1;
    }
  }
  return 0;
}

int rpc_read_jit_options(conn_t *conn, std::vector<CUjit_option> *options,
                         std::vector<uintptr_t> *raw_values) {
  if (conn == nullptr || options == nullptr || raw_values == nullptr) {
    return -1;
  }

  unsigned int num_options = 0;
  if (rpc_read(conn, &num_options, sizeof(num_options)) < 0) {
    return -1;
  }

  options->resize(num_options);
  raw_values->resize(num_options);
  if (num_options != 0 &&
      rpc_read(conn, options->data(), num_options * sizeof(CUjit_option)) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_options; ++i) {
    if (rpc_read(conn, &(*raw_values)[i], sizeof((*raw_values)[i])) < 0) {
      return -1;
    }
  }
  return 0;
}

static size_t rpc_find_jit_size_option(const std::vector<CUjit_option> &options,
                                       const std::vector<uintptr_t> &raw_values,
                                       CUjit_option option) {
  for (size_t i = 0; i < options.size(); ++i) {
    if (options[i] == option) {
      return static_cast<size_t>(raw_values[i]);
    }
  }
  return 0;
}

int rpc_read_jit_options(conn_t *conn, rpc_jit_server_state *state) {
  if (state == nullptr) {
    return -1;
  }

  std::vector<uintptr_t> raw_values;
  if (rpc_read_jit_options(conn, &state->options, &raw_values) < 0) {
    return -1;
  }

  size_t info_size = rpc_find_jit_size_option(
      state->options, raw_values, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
  size_t error_size = rpc_find_jit_size_option(
      state->options, raw_values, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
  state->option_values.resize(state->options.size());
  if (info_size != 0) {
    state->info_log.assign(info_size, '\0');
  }
  if (error_size != 0) {
    state->error_log.assign(error_size, '\0');
  }

  for (size_t i = 0; i < state->options.size(); ++i) {
    switch (state->options[i]) {
    case CU_JIT_WALL_TIME:
      state->capture_wall_time = true;
      state->option_values[i] = &state->wall_time;
      break;
    case CU_JIT_INFO_LOG_BUFFER:
      state->capture_info_log = !state->info_log.empty();
      state->option_values[i] =
          state->info_log.empty() ? nullptr : state->info_log.data();
      break;
    case CU_JIT_ERROR_LOG_BUFFER:
      state->capture_error_log = !state->error_log.empty();
      state->option_values[i] =
          state->error_log.empty() ? nullptr : state->error_log.data();
      break;
    default:
      state->option_values[i] = reinterpret_cast<void *>(raw_values[i]);
      break;
    }
  }
  return 0;
}

static constexpr uintptr_t LUPINE_RPC_NULL_OPTION_VALUES = UINTPTR_MAX;

int rpc_write_library_options(conn_t *conn, const unsigned int *num_options,
                              const CUlibraryOption *options,
                              void *const *option_values,
                              std::vector<uintptr_t> *raw_values) {
  if (conn == nullptr || num_options == nullptr || raw_values == nullptr ||
      (*num_options != 0 && options == nullptr)) {
    return -1;
  }

  raw_values->resize(*num_options);
  for (unsigned int i = 0; i < *num_options; ++i) {
    (*raw_values)[i] = option_values == nullptr
                           ? LUPINE_RPC_NULL_OPTION_VALUES
                           : reinterpret_cast<uintptr_t>(option_values[i]);
  }

  if (rpc_write(conn, num_options, sizeof(*num_options)) < 0 ||
      (*num_options != 0 &&
       rpc_write(conn, options, *num_options * sizeof(CUlibraryOption)) < 0)) {
    return -1;
  }
  for (unsigned int i = 0; i < *num_options; ++i) {
    if (rpc_write(conn, &(*raw_values)[i], sizeof((*raw_values)[i])) < 0) {
      return -1;
    }
  }
  return 0;
}

int rpc_read_library_options(conn_t *conn,
                             std::vector<CUlibraryOption> *options,
                             std::vector<uintptr_t> *raw_values,
                             bool *has_option_values) {
  if (conn == nullptr || options == nullptr || raw_values == nullptr ||
      has_option_values == nullptr) {
    return -1;
  }

  unsigned int num_options = 0;
  if (rpc_read(conn, &num_options, sizeof(num_options)) < 0) {
    return -1;
  }

  options->resize(num_options);
  raw_values->resize(num_options);
  *has_option_values = num_options == 0;
  if (num_options != 0 && rpc_read(conn, options->data(),
                                   num_options * sizeof(CUlibraryOption)) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_options; ++i) {
    if (rpc_read(conn, &(*raw_values)[i], sizeof((*raw_values)[i])) < 0) {
      return -1;
    }
    bool value_is_present = (*raw_values)[i] != LUPINE_RPC_NULL_OPTION_VALUES;
    if (i == 0) {
      *has_option_values = value_is_present;
    } else if (*has_option_values != value_is_present) {
      return -1;
    }
  }
  return 0;
}

int rpc_write_jit_outputs(conn_t *conn, const uint32_t *output_count,
                          const CUjit_option *options, const size_t *sizes,
                          const void *const *data) {
  if (conn == nullptr || output_count == nullptr ||
      (*output_count != 0 &&
       (options == nullptr || sizes == nullptr || data == nullptr))) {
    return -1;
  }
  if (rpc_write(conn, output_count, sizeof(*output_count)) < 0) {
    return -1;
  }
  for (uint32_t i = 0; i < *output_count; ++i) {
    if (rpc_write(conn, &options[i], sizeof(options[i])) < 0 ||
        rpc_write(conn, &sizes[i], sizeof(sizes[i])) < 0 ||
        (sizes[i] != 0 &&
         (data[i] == nullptr || rpc_write(conn, data[i], sizes[i]) < 0))) {
      return -1;
    }
  }
  return 0;
}

int rpc_write_jit_outputs(conn_t *conn, rpc_jit_server_state *state) {
  if (state == nullptr) {
    return -1;
  }

  state->output_count = 0;
  if (state->capture_wall_time) {
    state->output_options[state->output_count] = CU_JIT_WALL_TIME;
    state->output_sizes[state->output_count] = sizeof(state->wall_time);
    state->output_data[state->output_count] = &state->wall_time;
    ++state->output_count;
  }
  if (state->capture_info_log) {
    state->output_options[state->output_count] = CU_JIT_INFO_LOG_BUFFER;
    state->output_sizes[state->output_count] = state->info_log.size();
    state->output_data[state->output_count] = state->info_log.data();
    ++state->output_count;
  }
  if (state->capture_error_log) {
    state->output_options[state->output_count] = CU_JIT_ERROR_LOG_BUFFER;
    state->output_sizes[state->output_count] = state->error_log.size();
    state->output_data[state->output_count] = state->error_log.data();
    ++state->output_count;
  }
  return rpc_write_jit_outputs(conn, &state->output_count,
                               state->output_options, state->output_sizes,
                               state->output_data);
}

static const rpc_jit_output_binding *
rpc_find_jit_output_binding(const std::vector<rpc_jit_output_binding> &bindings,
                            CUjit_option option) {
  for (const auto &binding : bindings) {
    if (binding.option == option && binding.dst != nullptr) {
      return &binding;
    }
  }
  return nullptr;
}

int rpc_read_jit_outputs(conn_t *conn,
                         const std::vector<rpc_jit_output_binding> &bindings) {
  uint32_t output_count = 0;
  if (rpc_read(conn, &output_count, sizeof(output_count)) < 0) {
    return -1;
  }
  if (output_count > 32) {
    return -1;
  }
  for (uint32_t i = 0; i < output_count; ++i) {
    CUjit_option option;
    size_t payload_size = 0;
    if (rpc_read(conn, &option, sizeof(option)) < 0 ||
        rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
      return -1;
    }
    if (payload_size > (16ull << 20)) {
      return -1;
    }
    const auto *binding = rpc_find_jit_output_binding(bindings, option);
    size_t direct_size =
        binding == nullptr ? 0 : std::min(binding->size, payload_size);
    if (direct_size != 0 && rpc_read(conn, binding->dst, direct_size) < 0) {
      return -1;
    }
    if (rpc_drain(conn, payload_size - direct_size) < 0) {
      return -1;
    }
  }
  return 0;
}

// rpc_write_framed queues a payload that the transport LZ4-frames lazily,
// one block at a time, as the bytes are streamed to the socket. The caller's
// buffer must stay valid until rpc_write_end() returns, exactly like
// rpc_write(). See compress.cpp for the framing format.
int rpc_write_framed(conn_t *conn, const void *data, const size_t size) {
  return rpc_write_queue_push(conn, data, size, 1, 0);
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  bool request = conn->write_op != -1;
  if (conn->closed) {
    rpc_write_queue_release_owned(conn);
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
  rpc_write_queue_release_owned(conn);
  pthread_mutex_unlock(&conn->write_mutex);
  if (request) {
    pthread_mutex_unlock(&conn->call_mutex);
  }
  return result == 0 ? write_id : -1;
}
