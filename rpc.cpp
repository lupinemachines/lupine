#include "rpc.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string.h>

namespace {
struct rpc_profile_stat {
  uint64_t count = 0;
  uint64_t bytes = 0;
  uint64_t wait_ns = 0;
};

std::mutex &rpc_profile_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<int, rpc_profile_stat> &rpc_profile_stats() {
  static auto *stats = new std::unordered_map<int, rpc_profile_stat>();
  return *stats;
}

bool rpc_profile_enabled() {
  static bool enabled = [] {
    const char *value = getenv("LUPINE_RPC_PROFILE");
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
  }();
  return enabled;
}

void rpc_profile_dump() {
  if (!rpc_profile_enabled()) {
    return;
  }
  std::vector<std::pair<int, rpc_profile_stat>> rows;
  {
    std::lock_guard<std::mutex> lock(rpc_profile_mutex());
    rows.assign(rpc_profile_stats().begin(), rpc_profile_stats().end());
  }
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    return a.second.wait_ns > b.second.wait_ns;
  });
  std::cerr << "LUPINE_RPC_PROFILE_BEGIN\n";
  for (const auto &row : rows) {
    int op = row.first;
    const rpc_profile_stat &stat = row.second;
    double wait_ms = static_cast<double>(stat.wait_ns) / 1.0e6;
    double avg_wait_us =
        stat.count == 0 ? 0.0 : static_cast<double>(stat.wait_ns) / stat.count / 1.0e3;
    std::cerr << "LUPINE_RPC_PROFILE op=" << op << " count=" << stat.count
              << " bytes=" << stat.bytes << " wait_ms=" << wait_ms
              << " avg_wait_us=" << avg_wait_us << "\n";
  }
  std::cerr << "LUPINE_RPC_PROFILE_END\n";
}

void rpc_profile_add(int op, uint64_t bytes, uint64_t wait_ns) {
  if (!rpc_profile_enabled() || op < 0) {
    return;
  }
  static std::once_flag once;
  std::call_once(once, [] { atexit(rpc_profile_dump); });
  std::lock_guard<std::mutex> lock(rpc_profile_mutex());
  auto &stat = rpc_profile_stats()[op];
  stat.count += 1;
  stat.bytes += bytes;
  stat.wait_ns += wait_ns;
}

uint64_t rpc_pending_write_bytes(conn_t *conn) {
  uint64_t bytes = sizeof(int);
  if (conn->write_op != -1) {
    bytes += sizeof(unsigned int);
  }
  for (int i = conn->write_op == -1 ? 1 : 2; i < conn->write_iov_count; ++i) {
    bytes += conn->write_iov[i].iov_len;
  }
  return bytes;
}
} // namespace

extern "C" void lupine_rpc_profile_reset() {
  if (!rpc_profile_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(rpc_profile_mutex());
  rpc_profile_stats().clear();
}

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
  int op = conn->write_op;
  uint64_t bytes = rpc_pending_write_bytes(conn);
  auto start = std::chrono::steady_clock::now();
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  auto end = std::chrono::steady_clock::now();
  rpc_profile_add(
      op, bytes,
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
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
  conn->write_iov_framed[conn->write_iov_count] = 0;
  conn->write_iov[conn->write_iov_count++] = {(void *)data, size};
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

int rpc_write_jit_options(conn_t *conn, const unsigned int *num_options,
                          const CUjit_option *options,
                          void *const *option_values,
                          std::vector<uintptr_t> *raw_values) {
  if (conn == nullptr || num_options == nullptr || raw_values == nullptr ||
      (*num_options != 0 && (options == nullptr || option_values == nullptr))) {
    return -1;
  }

  if (static_cast<size_t>(conn->write_iov_count) + 2 + *num_options >
      sizeof(conn->write_iov) / sizeof(conn->write_iov[0])) {
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
