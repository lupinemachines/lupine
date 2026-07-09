#include "rpc.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string.h>
#include <thread>
#include <vector>

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

static int rpc_write_queue_reset(conn_t *conn, int count) {
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
  free(conn->write_queue);
  conn->write_queue = nullptr;
  conn->write_queue_count = 0;
  conn->write_queue_capacity = 0;
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
#else
static void rpc_destroy_thread_lane(uint64_t lane_id) { (void)lane_id; }
#endif

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
      conn->closed = 1;
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
  conn->closed = 1;
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

  while (!conn->closed &&
         (conn->read_id < 2 || conn->read_id % 2 != parity)) {
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
  }

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  if (rpc_http2_read(conn, &conn->read_lane_id, sizeof(conn->read_lane_id)) !=
          sizeof(conn->read_lane_id) ||
      rpc_http2_read(conn, &conn->read_op, sizeof(conn->read_op)) !=
          sizeof(conn->read_op) ||
      rpc_http2_read(conn, &conn->read_body_remaining,
                     sizeof(conn->read_body_remaining)) !=
          sizeof(conn->read_body_remaining)) {
    conn->closed = 1;
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }
  conn->body_tracking_active = 1;
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
      rpc_http2_read(conn, &conn->read_body_remaining,
                     sizeof(conn->read_body_remaining)) !=
              sizeof(conn->read_body_remaining) ||
      conn->read_op != -1) {
    conn->closed = 1;
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }
  conn->body_tracking_active = 1;
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
  conn->body_tracking_active = 0;
  // Drain any body bytes the handler did not consume so the stream stays
  // aligned for the next frame.
  if (conn->read_body_remaining > 0) {
    rpc_drain(conn, conn->read_body_remaining);
    conn->read_body_remaining = 0;
  }
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

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
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
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }

  if (rpc_write_queue_reset(conn, 4) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  conn->write_lane_id = rpc_thread_lane_id(conn);
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

  if (rpc_write_queue_reset(conn, 4) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_id = read_id;
  conn->write_op = -1;
  conn->write_lane_id = conn->read_lane_id;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  return rpc_write_queue_push(conn, data, size, 0);
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
    pthread_mutex_unlock(&conn->write_mutex);
    if (request) {
      pthread_mutex_unlock(&conn->call_mutex);
    }
    return -1;
  }
  int write_id = conn->write_id;
  int result = -1;
  if (conn->write_queue_count >= 4) {
    // Compute total payload size (entries [4..count-1]).
    size_t body_size = 0;
    for (int i = 4; i < conn->write_queue_count; ++i) {
      body_size += conn->write_queue[i].iov.iov_len;
    }
    conn->write_body_size = body_size;
    conn->write_queue[0] = {{&conn->write_id, sizeof(conn->write_id)}, 0};
    conn->write_queue[1] = {
        {&conn->write_lane_id, sizeof(conn->write_lane_id)}, 0};
    conn->write_queue[2] = {{&conn->write_op, sizeof(conn->write_op)}, 0};
    conn->write_queue[3] = {{&conn->write_body_size, sizeof(conn->write_body_size)}, 0};
    result = rpc_http2_writev(conn, conn->write_queue, conn->write_queue_count);
  }
  pthread_mutex_unlock(&conn->write_mutex);
  if (request) {
    pthread_mutex_unlock(&conn->call_mutex);
  }
  return result == 0 ? write_id : -1;
}
