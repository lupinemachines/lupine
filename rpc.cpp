#include "rpc.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <lz4.h>
#include <memory>
#include <string.h>
#include <unordered_map>

static int rpc_write_control_frame(conn_t *conn, uint32_t lane_id, int op);

namespace {

struct rpc_frame_header {
  int32_t request_id = 0;
  uint32_t lane_id = 0;
  int32_t op = 0;
  uint64_t payload_size = 0;
};

enum rpc_segment_kind : uint32_t {
  RPC_SEGMENT_RAW = 0,
  RPC_SEGMENT_LZ4 = 1,
};

struct rpc_segment_header {
  uint32_t kind = RPC_SEGMENT_RAW;
  uint32_t reserved = 0;
  uint64_t encoded_size = 0;
  uint64_t decoded_size = 0;
};

struct rpc_connection_state {
  std::deque<rpc_frame> remote_requests;
  std::unordered_map<int, std::deque<rpc_frame>> responses;
};

struct rpc_active_frame {
  conn_t *conn = nullptr;
  int request_id = 0;
  uint32_t lane_id = 0;
  int op = 0;
  std::vector<unsigned char> payload;
  size_t offset = 0;
  rpc_segment_header segment = {};
  uint64_t segment_remaining = 0;
};

static thread_local rpc_active_frame tls_active_frame;

struct rpc_thread_lane_entry {
  conn_t *conn = nullptr;
  uint32_t lane_id = 0;
};

struct rpc_thread_lanes {
  std::vector<rpc_thread_lane_entry> lanes;

  ~rpc_thread_lanes() {
    for (const auto &lane : lanes) {
      if (lane.conn != nullptr && !lane.conn->closed) {
        rpc_write_control_frame(lane.conn, lane.lane_id,
                                LUPINE_RPC_RELEASE_LANE);
      }
    }
  }
};

static thread_local rpc_thread_lanes tls_thread_lanes;

rpc_connection_state *rpc_state(conn_t *conn) {
  if (conn == nullptr) {
    return nullptr;
  }
  return static_cast<rpc_connection_state *>(conn->rpc_state);
}

uint32_t rpc_lane_for_client_thread(conn_t *conn) {
  for (const auto &lane : tls_thread_lanes.lanes) {
    if (lane.conn == conn) {
      return lane.lane_id;
    }
  }
  uint32_t lane_id = ++conn->next_lane_id;
  tls_thread_lanes.lanes.push_back({conn, lane_id});
  return lane_id;
}

uint32_t rpc_lane_for_outgoing_request(conn_t *conn) {
  if (conn == nullptr) {
    return 0;
  }
  if (conn->local_request_parity == 0) {
    return rpc_lane_for_client_thread(conn);
  }
  return tls_active_frame.conn == conn ? tls_active_frame.lane_id : 0;
}

void rpc_close_with_broadcast(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  conn->closed = 1;
  pthread_cond_broadcast(&conn->read_cond);
}

} // namespace

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

int rpc_connection_state_init(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  if (conn->rpc_state != nullptr) {
    return 0;
  }
  try {
    conn->rpc_state = new rpc_connection_state();
  } catch (...) {
    return -1;
  }
  return 0;
}

void rpc_connection_state_free(conn_t *conn) {
  if (conn == nullptr || conn->rpc_state == nullptr) {
    return;
  }
  delete static_cast<rpc_connection_state *>(conn->rpc_state);
  conn->rpc_state = nullptr;
}

static int rpc_lz4_encode_segment(const rpc_write_entry &entry,
                                  std::vector<unsigned char> *out) {
  if (out == nullptr) {
    return -1;
  }
  out->clear();
  const char *src = static_cast<const char *>(entry.iov.iov_base);
  size_t remaining = entry.iov.iov_len;
  while (remaining > 0) {
    size_t raw = std::min<size_t>(LUPINE_COMPRESS_BLOCK_BYTES, remaining);
    size_t token_offset = out->size();
    out->resize(out->size() + sizeof(uint32_t));

    uint32_t token = 0;
    size_t block_len = raw;
    if (entry.framed == 1) {
      size_t bound = static_cast<size_t>(
          LZ4_compressBound(static_cast<int>(LUPINE_COMPRESS_BLOCK_BYTES)));
      size_t data_offset = out->size();
      out->resize(data_offset + bound);
      int compressed = LZ4_compress_default(
          src, reinterpret_cast<char *>(out->data() + data_offset),
          static_cast<int>(raw), static_cast<int>(bound));
      if (compressed > 0 && static_cast<size_t>(compressed) < raw) {
        token = static_cast<uint32_t>(compressed);
        block_len = static_cast<size_t>(compressed);
        out->resize(data_offset + block_len);
      } else {
        memcpy(out->data() + data_offset, src, raw);
        out->resize(data_offset + raw);
      }
    } else {
      size_t data_offset = out->size();
      out->resize(data_offset + raw);
      memcpy(out->data() + data_offset, src, raw);
    }

    memcpy(out->data() + token_offset, &token, sizeof(token));
    src += raw;
    remaining -= raw;
  }
  return 0;
}

static int rpc_write_current_frame_locked(conn_t *conn) {
  struct rpc_segment_storage {
    rpc_segment_header header;
    std::vector<unsigned char> encoded;
    const void *raw = nullptr;
  };

  std::vector<rpc_segment_storage> segments;
  segments.reserve(static_cast<size_t>(conn->write_queue_count));

  uint64_t payload_size = 0;
  for (int i = 0; i < conn->write_queue_count; ++i) {
    const rpc_write_entry &entry = conn->write_queue[i];
    rpc_segment_storage segment;
    segment.header.kind = entry.framed ? RPC_SEGMENT_LZ4 : RPC_SEGMENT_RAW;
    segment.header.decoded_size = static_cast<uint64_t>(entry.iov.iov_len);
    if (entry.framed) {
      if (rpc_lz4_encode_segment(entry, &segment.encoded) < 0) {
        return -1;
      }
      segment.header.encoded_size =
          static_cast<uint64_t>(segment.encoded.size());
    } else {
      segment.raw = entry.iov.iov_base;
      segment.header.encoded_size = static_cast<uint64_t>(entry.iov.iov_len);
    }
    payload_size += sizeof(rpc_segment_header) + segment.header.encoded_size;
    segments.push_back(std::move(segment));
  }

  rpc_frame_header header;
  header.request_id = conn->write_id;
  header.lane_id = conn->write_lane_id;
  header.op = conn->write_op;
  header.payload_size = payload_size;

  std::vector<rpc_write_entry> entries;
  entries.reserve(1 + segments.size() * 2);
  entries.push_back({{&header, sizeof(header)}, 0});
  for (auto &segment : segments) {
    entries.push_back({{&segment.header, sizeof(segment.header)}, 0});
    if (segment.header.encoded_size == 0) {
      continue;
    }
    void *data = segment.header.kind == RPC_SEGMENT_LZ4
                     ? static_cast<void *>(segment.encoded.data())
                     : const_cast<void *>(segment.raw);
    entries.push_back(
        {{data, static_cast<size_t>(segment.header.encoded_size)}, 0});
  }

  return rpc_http2_writev(conn, entries.data(),
                          static_cast<int>(entries.size()));
}

static int rpc_write_control_frame(conn_t *conn, uint32_t lane_id, int op) {
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
  conn->write_op = op;
  conn->write_lane_id = lane_id;
  conn->write_queue_count = 0;
  int result = rpc_write_current_frame_locked(conn);
  pthread_mutex_unlock(&conn->write_mutex);
  pthread_mutex_unlock(&conn->call_mutex);
  return result;
}

int rpc_read_wire_frame(conn_t *conn, rpc_frame *frame) {
  if (conn == nullptr || frame == nullptr || conn->closed) {
    return -1;
  }

  rpc_frame_header header;
  if (rpc_http2_read(conn, &header, sizeof(header)) != sizeof(header) ||
      header.request_id == 0) {
    rpc_close_with_broadcast(conn);
    return -1;
  }

  frame->request_id = header.request_id;
  frame->lane_id = header.lane_id;
  frame->op = header.op;
  try {
    frame->payload.resize(static_cast<size_t>(header.payload_size));
  } catch (...) {
    rpc_close_with_broadcast(conn);
    return -1;
  }
  if (!frame->payload.empty() &&
      rpc_http2_read(conn, frame->payload.data(), frame->payload.size()) < 0) {
    rpc_close_with_broadcast(conn);
    return -1;
  }
  return 0;
}

int rpc_activate_frame(conn_t *conn, rpc_frame &&frame) {
  tls_active_frame.conn = conn;
  tls_active_frame.request_id = frame.request_id;
  tls_active_frame.lane_id = frame.lane_id;
  tls_active_frame.op = frame.op;
  tls_active_frame.payload = std::move(frame.payload);
  tls_active_frame.offset = 0;
  return 0;
}

uint32_t rpc_active_lane_id() { return tls_active_frame.lane_id; }

static int rpc_load_next_segment(rpc_active_frame *frame) {
  if (frame == nullptr) {
    return -1;
  }
  if (frame->segment_remaining != 0) {
    return 0;
  }
  if (frame->offset == frame->payload.size()) {
    return -1;
  }
  if (frame->offset + sizeof(rpc_segment_header) > frame->payload.size()) {
    return -1;
  }
  memcpy(&frame->segment, frame->payload.data() + frame->offset,
         sizeof(frame->segment));
  frame->offset += sizeof(frame->segment);
  if ((frame->segment.kind != RPC_SEGMENT_RAW &&
       frame->segment.kind != RPC_SEGMENT_LZ4) ||
      frame->offset + frame->segment.encoded_size > frame->payload.size()) {
    return -1;
  }
  frame->segment_remaining = frame->segment.encoded_size;
  if (frame->segment_remaining == 0 && frame->offset != frame->payload.size()) {
    return rpc_load_next_segment(frame);
  }
  return 0;
}

int rpc_deliver_response_frame(conn_t *conn, rpc_frame &&frame) {
  rpc_connection_state *state = rpc_state(conn);
  if (state == nullptr || pthread_mutex_lock(&conn->read_mutex) != 0) {
    return -1;
  }
  state->responses[frame.request_id].push_back(std::move(frame));
  int result = pthread_cond_broadcast(&conn->read_cond);
  pthread_mutex_unlock(&conn->read_mutex);
  return result == 0 ? 0 : -1;
}

void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  while (!conn->closed) {
    rpc_frame frame;
    if (rpc_read_wire_frame(conn, &frame) < 0) {
      break;
    }
    if (frame.op == -1) {
      if (rpc_deliver_response_frame(conn, std::move(frame)) < 0) {
        break;
      }
      continue;
    }
    if (pthread_mutex_lock(&conn->read_mutex) != 0) {
      rpc_close_with_broadcast(conn);
      break;
    }
    rpc_connection_state *state = rpc_state(conn);
    if (state == nullptr) {
      pthread_mutex_unlock(&conn->read_mutex);
      rpc_close_with_broadcast(conn);
      break;
    }
    state->remote_requests.push_back(std::move(frame));
    pthread_cond_broadcast(&conn->read_cond);
    pthread_mutex_unlock(&conn->read_mutex);
  }
  rpc_close_with_broadcast(conn);
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
  (void)parity;
  if (conn->rpc_thread == 0 &&
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch,
                     (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  rpc_connection_state *state = rpc_state(conn);
  if (state == nullptr) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  while (!conn->closed && state->remote_requests.empty())
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  rpc_frame frame = std::move(state->remote_requests.front());
  state->remote_requests.pop_front();
  int op = frame.op;
  pthread_mutex_unlock(&conn->read_mutex);
  rpc_activate_frame(conn, std::move(frame));

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

  rpc_connection_state *state = rpc_state(conn);
  if (state == nullptr) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  while (!conn->closed && state->responses[write_id].empty())
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) < 0)
      return -1;

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  rpc_frame frame = std::move(state->responses[write_id].front());
  state->responses[write_id].pop_front();
  if (state->responses[write_id].empty()) {
    state->responses.erase(write_id);
  }
  pthread_mutex_unlock(&conn->read_mutex);
  rpc_activate_frame(conn, std::move(frame));
  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  if (tls_active_frame.conn == conn) {
    auto *out = static_cast<unsigned char *>(data);
    size_t copied = 0;
    while (copied < size) {
      if (rpc_load_next_segment(&tls_active_frame) < 0 ||
          tls_active_frame.segment_remaining == 0) {
        return -1;
      }
      size_t chunk = std::min<size_t>(
          size - copied,
          static_cast<size_t>(tls_active_frame.segment_remaining));
      if (chunk != 0) {
        memcpy(out + copied,
               tls_active_frame.payload.data() + tls_active_frame.offset,
               chunk);
      }
      tls_active_frame.offset += chunk;
      tls_active_frame.segment_remaining -= chunk;
      copied += chunk;
    }
    return static_cast<int>(size);
  }
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
  if (tls_active_frame.conn != conn) {
    return -1;
  }
  int read_id = tls_active_frame.request_id;
  tls_active_frame = {};
  return read_id;
}

int rpc_read_end_host_copy_chunk(conn_t *conn) { return rpc_read_end(conn); }

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

  if (rpc_write_queue_reset(conn, 0) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  conn->write_lane_id = rpc_lane_for_outgoing_request(conn);
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

  if (rpc_write_queue_reset(conn, 0) < 0) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_id = read_id;
  conn->write_op = -1;
  conn->write_lane_id = rpc_active_lane_id();
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

// rpc_write_framed queues a payload segment using the LZ4 block format from
// compress.cpp. The caller's buffer must stay valid until rpc_write_end()
// returns, exactly like rpc_write().
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
  int result = rpc_write_current_frame_locked(conn);
  pthread_mutex_unlock(&conn->write_mutex);
  if (request) {
    pthread_mutex_unlock(&conn->call_mutex);
  }
  return result == 0 ? write_id : -1;
}
