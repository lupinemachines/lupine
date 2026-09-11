#include "client_bundle.h"
#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <deque>
#include <errno.h>
#include <lz4frame.h>
#include <nghttp2/nghttp2.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <sstream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// Responses land straight in caller buffers, so the client keeps an
// effectively unlimited receive window; the server's window is the staging
// budget it is willing to have pinned on a client's behalf.
constexpr uint32_t kH2ClientWindow = 0x7fffffffU;
constexpr uint32_t kH2ServerWindow =
    static_cast<uint32_t>(LUPINE_FF_STAGING_WINDOW_BYTES);
// Ceiling on uncredited bytes, leaving the reader window for the bytes it is
// blocked on.
constexpr uint64_t kH2MaxHeldBytes = LUPINE_FF_STAGING_WINDOW_BYTES / 2;
constexpr uint32_t kH2MaxFrame = (16 * 1024 * 1024) - 1;
constexpr size_t kH2FrameHeaderLen = 9;
// LZ4F_max4MB, the encoder's block size. Input short of a block stays inside
// LZ4F until a flush, and each compression step feeds exactly one block so a
// very compressible body cannot delay its first byte until all of it has been
// compressed.
constexpr size_t kH2Lz4BlockBytes = 4 * 1024 * 1024;
constexpr size_t kH2MaxDataFrameBytes = 1024 * 1024;
constexpr size_t kH2DecodeBufferBytes = 64 * 1024;
// Retained capacity for drained staging buffers, a few frames' worth.
constexpr size_t kH2StagingPoolBytes = 4 * 1024 * 1024;
// Output nghttp2 may produce ahead of the socket before request writers block.
constexpr size_t kH2OutboundLimitBytes = 8 * 1024 * 1024;
// How long the client write thread polls for more output before it parks.
// Parked, every message costs its producer a futex wake and the two then
// contend for session_mutex; polling covers the gap between back-to-back RPCs.
// The server hosts one write thread per connection and does not poll.
constexpr auto kH2WriterPoll = std::chrono::microseconds(200);
constexpr std::array<uint8_t, 8> kH2ShutdownPing = {'l', 'u', 'p', 'i',
                                                    'n', 'e', 0,   1};
// Linux restarts slow start after an idle period of one retransmission
// timeout. Keeping response waits below the usual 200 ms minimum RTO avoids
// collapsing the congestion window between bursts on high-latency links.
constexpr long kH2HeartbeatIntervalMs = 100;

struct h2_buffer {
  std::vector<unsigned char> data;
  size_t offset = 0;
};

struct h2_stream {
  std::deque<h2_buffer> local_out;
  unsigned char *read_destination = nullptr;
  size_t read_remaining = 0;
  bool closed = false;
  bool remote_end = false;
  bool response_received = false;
  bool response_sent = false;
  bool content_encoding_seen = false;
  bool lz4_encoded = false;
  bool encoder_started = false;
  bool encoder_finished = false;
  bool decoder_finished = false;
  // One DATA item carries the stream's whole body. It defers whenever
  // `encoded` runs dry and is resumed when more output lands.
  bool provider_submitted = false;
  bool provider_deferred = false;
  bool flush_queued = false;
  // Input LZ4F holds back for its next block; only a flush emits it.
  size_t buffered = 0;
  // Compressed output nghttp2 has not framed yet: valid up to encoded_size,
  // consumed up to encoded_offset. Capacity is retained across bursts.
  std::vector<unsigned char> encoded;
  size_t encoded_size = 0;
  size_t encoded_offset = 0;
  LZ4F_compressionContext_t encoder = nullptr;
  LZ4F_decompressionContext_t decoder = nullptr;
  int response_status = 0;
  std::string requested_va_base;
  std::string requested_va_size;
  bool window_hold = false;
  uint64_t window_hold_bytes = 0;
};

struct h2_transport {
  lupine_socket_t netfd = LUPINE_INVALID_SOCKET;
  void *tls = nullptr; // Borrowed SSL* (owned by conn_t).
  bool server = false;
  bool request_received = false;
  bool request_handled = false;
  int request_status = 0;
  int32_t dispatch_stream_id = -1;
  nghttp2_session *session = nullptr;
  std::unordered_map<int32_t, h2_stream> streams;
  std::deque<int32_t> incoming_streams;
  std::unordered_map<uint64_t, int32_t> local_lanes;
  rpc_http2_read_stats read_stats = {};
  uint64_t staged_bytes = 0;
  uint64_t window_held = 0;
  // Drained staging buffers, kept for their capacity. A saturated stream
  // stages one buffer per DATA frame, and reallocating each one costs more
  // than the copy into it. Capped because a reader that falls behind can stage
  // far more than it will ever need again.
  std::vector<h2_buffer> buffer_pool;
  size_t buffer_pool_bytes = 0;
  // Wire bytes nghttp2 has produced that the socket has not taken yet. Only
  // the write thread sends, so a producer never blocks in the socket and
  // everything queued while one send is in flight leaves in the next one.
  std::vector<unsigned char> outbound;
  // Streams whose encoder holds input short of a block. The write thread
  // flushes them before each send, so every message queued while one send is
  // in flight shares a block instead of paying a block and a flush each.
  std::vector<int32_t> flush_pending;
  // Bumped whenever outbound or flush_pending grows, so the write thread can
  // poll for work without touching session_mutex.
  std::atomic<uint64_t> output_generation{0};
  pthread_mutex_t session_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t session_progress = PTHREAD_COND_INITIALIZER;
  pthread_cond_t heartbeat_progress = PTHREAD_COND_INITIALIZER;
  pthread_cond_t outbound_progress = PTHREAD_COND_INITIALIZER;
  pthread_t read_thread = {};
  pthread_t heartbeat_thread = {};
  pthread_t write_thread = {};
  bool write_stop = false;
  bool write_busy = false;
  bool write_failed = false;
  int response_waiters = 0;
  bool transport_failed = false;
  bool shutdown_acknowledged = false;
  std::string peer_cuda_version;
  std::string peer_capabilities;
  std::string peer_client_etag;
  std::string client_etag;
  std::string client_platform;
  std::string expected_client_etag;
  const lupine_client_bundle_registry *client_bundles = nullptr;
  std::string server_version;
  std::string server_capabilities;
  std::string session_id;
  std::string peer_va_base;
  std::string peer_va_size;
  std::string local_va_base;
  std::string local_va_size;
  std::string peer_window_base;
  std::string peer_window_size;
  std::string local_window_base;
  std::string local_window_size;
  conn_t *conn = nullptr;
};

h2_stream &h2_get_stream(h2_transport *transport, int32_t stream_id) {
  return transport->streams.try_emplace(stream_id).first->second;
}

bool h2_retryable_handshake_rejection(const h2_transport *transport,
                                      const h2_stream &stream) {
  return !transport->server && stream.response_received &&
         stream.response_status == 409;
}

void h2_release_codecs(h2_stream &stream) {
  if (stream.encoder != nullptr) {
    LZ4F_freeCompressionContext(stream.encoder);
    stream.encoder = nullptr;
  }
  if (stream.decoder != nullptr) {
    LZ4F_freeDecompressionContext(stream.decoder);
    stream.decoder = nullptr;
  }
}

void receive_bytes(h2_transport *transport, int32_t stream_id,
                   const unsigned char *data, size_t len) {
  if (len == 0) {
    return;
  }
  h2_stream &stream = h2_get_stream(transport, stream_id);
  size_t direct = std::min(len, stream.read_remaining);
  if (direct != 0) {
    memcpy(stream.read_destination, data, direct);
    stream.read_destination += direct;
    stream.read_remaining -= direct;
    transport->read_stats.direct_bytes += direct;
    data += direct;
    len -= direct;
  }
  if (len == 0) {
    return;
  }
  h2_buffer buffer;
  if (!transport->buffer_pool.empty()) {
    buffer = std::move(transport->buffer_pool.back());
    transport->buffer_pool.pop_back();
    transport->buffer_pool_bytes -= buffer.data.capacity();
  }
  buffer.offset = 0;
  buffer.data.assign(data, data + len);
  stream.local_out.push_back(std::move(buffer));
  transport->read_stats.staged_bytes += len;
  ++transport->read_stats.staged_buffers;
  transport->staged_bytes += len;
  transport->read_stats.peak_staged_bytes = std::max(
      transport->read_stats.peak_staged_bytes, transport->staged_bytes);
}

// Callers hold session_mutex, as every nghttp2_session_send does.
void h2_queue_output(h2_transport *transport, const struct iovec *iov,
                     int iov_count) {
  for (int i = 0; i < iov_count; ++i) {
    const auto *data = static_cast<const unsigned char *>(iov[i].iov_base);
    transport->outbound.insert(transport->outbound.end(), data,
                               data + iov[i].iov_len);
  }
  transport->output_generation.fetch_add(1, std::memory_order_release);
  pthread_cond_broadcast(&transport->outbound_progress);
}

int h2_write_socket(h2_transport *transport, const unsigned char *data,
                    size_t size) {
  while (size > 0) {
    ssize_t n;
#ifdef LUPINE_TLS_OPENSSL
    if (transport->tls != nullptr) {
      SSL *ssl = static_cast<SSL *>(transport->tls);
      int want = static_cast<int>(std::min(size, static_cast<size_t>(INT_MAX)));
      int r;
      while ((r = SSL_write(ssl, data, want)) <= 0) {
        int err = SSL_get_error(ssl, r);
        if (err != SSL_ERROR_WANT_READ && err != SSL_ERROR_WANT_WRITE) {
          return -1;
        }
      }
      n = r;
    } else
#endif
    {
      struct iovec iov = {const_cast<unsigned char *>(data), size};
      n = lupine_socket_sendv(transport->netfd, &iov, 1);
      if (n < 0) {
        if (lupine_socket_error_is_intr()) {
          continue;
        }
        return -1;
      }
      if (n == 0) {
        return -1;
      }
    }
    data += static_cast<size_t>(n);
    size -= static_cast<size_t>(n);
  }
  return 0;
}

ssize_t h2_send_callback(nghttp2_session *, const uint8_t *data, size_t length,
                         int, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  struct iovec iov = {const_cast<uint8_t *>(data), length};
  h2_queue_output(transport, &iov, 1);
  return static_cast<ssize_t>(length);
}

LZ4F_preferences_t h2_lz4_preferences() {
  LZ4F_preferences_t preferences = {};
  preferences.frameInfo.blockSizeID = LZ4F_max4MB;
  preferences.frameInfo.blockMode = LZ4F_blockLinked;
  // Linked blocks retain compression history across each RPC message flush.
  return preferences;
}

ssize_t h2_data_source_read_callback(nghttp2_session *, int32_t stream_id,
                                     uint8_t *, size_t length,
                                     uint32_t *data_flags,
                                     nghttp2_data_source *, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  size_t available = stream.encoded_size - stream.encoded_offset;
  if (available == 0) {
    stream.provider_deferred = true;
    return NGHTTP2_ERR_DEFERRED;
  }
  size_t produced = std::min(available, length);
  *data_flags |= NGHTTP2_DATA_FLAG_NO_COPY;
  if (produced == available && stream.encoder_finished) {
    *data_flags |= NGHTTP2_DATA_FLAG_EOF;
  }
  return static_cast<ssize_t>(produced);
}

int h2_send_data_callback(nghttp2_session *, nghttp2_frame *frame,
                          const uint8_t *framehd, size_t length,
                          nghttp2_data_source *, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);

  std::array<struct iovec, 4> iov = {};
  int iov_count = 0;
  iov[iov_count++] = {const_cast<uint8_t *>(framehd), kH2FrameHeaderLen};

  unsigned char padlen = 0;
  if (frame->data.padlen > 0) {
    padlen = static_cast<unsigned char>(frame->data.padlen - 1);
    iov[iov_count++] = {&padlen, 1};
  }
  iov[iov_count++] = {stream.encoded.data() + stream.encoded_offset, length};

  unsigned char padding[256] = {};
  if (frame->data.padlen > 1) {
    iov[iov_count++] = {padding, frame->data.padlen - 1};
  }
  h2_queue_output(transport, iov.data(), iov_count);

  stream.encoded_offset += length;
  if (stream.encoded_offset == stream.encoded_size) {
    stream.encoded_size = 0;
    stream.encoded_offset = 0;
  }
  return 0;
}

ssize_t h2_data_source_read_length_callback(nghttp2_session *, uint8_t,
                                            int32_t stream_id,
                                            int32_t session_remote_window_size,
                                            int32_t stream_remote_window_size,
                                            uint32_t remote_max_frame_size,
                                            void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  int32_t window =
      std::min(session_remote_window_size, stream_remote_window_size);
  if (window <= 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  h2_stream &stream = h2_get_stream(transport, stream_id);
  size_t max_len =
      std::min<size_t>(kH2MaxDataFrameBytes, remote_max_frame_size);
  max_len = std::min(max_len, stream.encoded_size - stream.encoded_offset);
  max_len = std::min<size_t>(max_len, static_cast<size_t>(window));
  return static_cast<ssize_t>(std::max<size_t>(1, max_len));
}

int h2_on_data_chunk_recv_callback(nghttp2_session *session, uint8_t,
                                   int32_t stream_id, const uint8_t *data,
                                   size_t len, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  if (transport->server) {
    size_t held = 0;
    if (stream.window_hold && transport->window_held < kH2MaxHeldBytes) {
      held = std::min<size_t>(
          len, static_cast<size_t>(kH2MaxHeldBytes - transport->window_held));
      transport->window_held += held;
      stream.window_hold_bytes += held;
    }
    if (len > held &&
        nghttp2_session_consume(session, stream_id, len - held) != 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  if (!stream.lz4_encoded || stream.decoder_finished) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (stream.decoder == nullptr) {
    if (LZ4F_isError(
            LZ4F_createDecompressionContext(&stream.decoder, LZ4F_VERSION))) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }

  size_t offset = 0;
  std::array<unsigned char, kH2DecodeBufferBytes> output;
  for (;;) {
    size_t input = len - offset;
    size_t produced = output.size();
    size_t result = LZ4F_decompress(stream.decoder, output.data(), &produced,
                                    data + offset, &input, nullptr);
    if (LZ4F_isError(result)) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    receive_bytes(transport, stream_id, output.data(), produced);
    offset += input;
    if (result == 0) {
      if (offset != len) {
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      }
      stream.decoder_finished = true;
      break;
    }
    if (input == 0 && produced == 0) {
      if (offset != len) {
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      }
      break;
    }
    // LZ4F can consume the complete encoded block while retaining decoded
    // output internally. Drain it before waiting for the next DATA frame.
    if (offset == len && produced < output.size()) {
      break;
    }
  }
  pthread_cond_broadcast(&transport->session_progress);
  return 0;
}

nghttp2_nv h2_nv(const char *name, const char *value) {
  return {reinterpret_cast<uint8_t *>(const_cast<char *>(name)),
          reinterpret_cast<uint8_t *>(const_cast<char *>(value)), strlen(name),
          strlen(value),
          NGHTTP2_NV_FLAG_NO_COPY_NAME | NGHTTP2_NV_FLAG_NO_COPY_VALUE};
}

constexpr char kLupineCudaVersionHeader[] = "x-lupine-cuda-version";
constexpr char kLupineSessionHeader[] = "x-lupine-session";
constexpr char kLupineVaBaseHeader[] = "x-lupine-va-base";
constexpr char kLupineVaSizeHeader[] = "x-lupine-va-size";
constexpr char kLupineCapabilitiesHeader[] = "x-lupine-capabilities";
constexpr char kLupineClientEtagHeader[] = "x-lupine-client-etag";
constexpr char kLupineClientPlatformHeader[] = "x-lupine-client-platform";
constexpr char kLupineVaWindowBaseHeader[] = "x-lupine-va-window-base";
constexpr char kLupineVaWindowSizeHeader[] = "x-lupine-va-window-size";
constexpr char kContentEncodingHeader[] = "content-encoding";
constexpr char kLz4Encoding[] = "lz4";

std::string h2_hex(uint64_t value) {
  std::ostringstream result;
  result << std::hex << value;
  return result.str();
}

bool h2_parse_hex64(const std::string &value, uint64_t *parsed) {
  if (value.empty() || parsed == nullptr) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  unsigned long long number = strtoull(value.c_str(), &end, 16);
  if (errno != 0 || end == value.c_str() || *end != '\0') {
    return false;
  }
  *parsed = static_cast<uint64_t>(number);
  return true;
}

bool h2_parse_hex(const std::string &value, uintptr_t *parsed) {
  uint64_t number = 0;
  if (parsed == nullptr || !h2_parse_hex64(value, &number) ||
      number > UINTPTR_MAX) {
    return false;
  }
  *parsed = static_cast<uintptr_t>(number);
  return true;
}

bool lupine_h2_debug_enabled() {
  const char *debug = getenv("LUPINE_DEBUG");
  if (debug != nullptr && debug[0] != '\0' && strcmp(debug, "0") != 0) {
    return true;
  }
  return lupine_trace_stream() != nullptr;
}

int h2_submit_server_response(h2_transport *transport, int32_t stream_id,
                              int status, bool end_stream) {
  h2_stream &stream = h2_get_stream(transport, stream_id);
  const char *status_text = "400";
  if (status == 200) {
    status_text = "200";
  } else if (status == 409) {
    status_text = "409";
  } else if (status == 426) {
    status_text = "426";
  }
  std::vector<nghttp2_nv> headers = {h2_nv(":status", status_text)};
  if (!end_stream) {
    headers.push_back(h2_nv(kContentEncodingHeader, kLz4Encoding));
  }
  if (!transport->server_version.empty()) {
    headers.push_back(
        h2_nv(kLupineCudaVersionHeader, transport->server_version.c_str()));
  }
  if (!transport->expected_client_etag.empty()) {
    headers.push_back(h2_nv(kLupineClientEtagHeader,
                            transport->expected_client_etag.c_str()));
  }
  if (!transport->server_capabilities.empty()) {
    headers.push_back(h2_nv(kLupineCapabilitiesHeader,
                            transport->server_capabilities.c_str()));
  }
  // State the window on every response, probe included: the client has to know
  // it before it reserves anything, and no single constant fits both platforms.
  const lupine_va_window window = lupine_va_local_window();
  if (window.size != 0) {
    transport->local_window_base = h2_hex(window.base);
    transport->local_window_size = h2_hex(window.size);
    headers.push_back(
        h2_nv(kLupineVaWindowBaseHeader, transport->local_window_base.c_str()));
    headers.push_back(
        h2_nv(kLupineVaWindowSizeHeader, transport->local_window_size.c_str()));
  }
  if (status == 200 && transport->conn != nullptr &&
      transport->conn->va_size != 0) {
    transport->local_va_base = h2_hex(transport->conn->va_base);
    transport->local_va_size = h2_hex(transport->conn->va_size);
    headers.push_back(
        h2_nv(kLupineVaBaseHeader, transport->local_va_base.c_str()));
    headers.push_back(
        h2_nv(kLupineVaSizeHeader, transport->local_va_size.c_str()));
  }
  uint8_t flags = end_stream ? NGHTTP2_FLAG_END_STREAM : NGHTTP2_FLAG_NONE;
  if (nghttp2_submit_headers(transport->session, flags, stream_id, nullptr,
                             headers.data(), headers.size(), nullptr) != 0) {
    return -1;
  }
  stream.response_sent = true;
  return 0;
}

int h2_on_frame_recv_callback(nghttp2_session *, const nghttp2_frame *frame,
                              void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  if (transport->server && frame->hd.type == NGHTTP2_PING &&
      (frame->hd.flags & NGHTTP2_FLAG_ACK) != 0 &&
      memcmp(frame->ping.opaque_data, kH2ShutdownPing.data(),
             kH2ShutdownPing.size()) == 0) {
    transport->shutdown_acknowledged = true;
  }
  if (!transport->server && frame->hd.type == NGHTTP2_GOAWAY &&
      lupine_h2_debug_enabled()) {
    std::string debug;
    if (frame->goaway.opaque_data != nullptr &&
        frame->goaway.opaque_data_len > 0) {
      debug.assign(reinterpret_cast<const char *>(frame->goaway.opaque_data),
                   frame->goaway.opaque_data_len);
    }
    std::ostringstream message;
    message << "LUPINE remote server sent HTTP/2 GOAWAY"
            << " error_code=" << frame->goaway.error_code;
    if (!debug.empty()) {
      message << " debug=\"" << debug << "\"";
    }
    if (lupine_trace_stream() != nullptr) {
      LUPINE_TRACE_LOG(message.str());
    } else {
      LUPINE_LOG_DEBUG(message.str());
    }
  }
  if (transport->server && frame->hd.type == NGHTTP2_HEADERS &&
      frame->headers.cat == NGHTTP2_HCAT_REQUEST) {
    h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);
    transport->request_received = true;
    bool probe = (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0;
    int status = stream.lz4_encoded || probe ? 200 : 400;
    if (!probe && status == 200 &&
        (!transport->client_etag.empty() ||
         !transport->client_platform.empty())) {
      bool complete = !transport->client_etag.empty() &&
                      !transport->client_platform.empty();
      const lupine_client_bundle_payload *bundle =
          complete ? lupine_client_bundle_lookup(transport->client_bundles,
                                                 transport->client_platform)
                   : nullptr;
      if (bundle != nullptr) {
        transport->expected_client_etag = bundle->etag;
      }
      if (bundle == nullptr || transport->client_etag != bundle->etag) {
        status = 426;
      }
    }
    bool has_va_base = !stream.requested_va_base.empty();
    bool has_va_size = !stream.requested_va_size.empty();
    if (!probe && (has_va_base || has_va_size)) {
      uintptr_t base = 0;
      uintptr_t size = 0;
      if (!has_va_base || !has_va_size ||
          !h2_parse_hex(stream.requested_va_base, &base) ||
          !h2_parse_hex(stream.requested_va_size, &size) || size > SIZE_MAX) {
        status = 400;
      } else if (lupine_va_reserve_server(transport->conn, base,
                                          static_cast<size_t>(size)) < 0) {
        status = 409;
      }
    }
    bool dispatch =
        status == 200 && !probe && transport->dispatch_stream_id < 0;
    transport->request_handled = probe || (status != 200 && status != 409);
    transport->request_status = status;
    if (dispatch) {
      transport->dispatch_stream_id = frame->hd.stream_id;
    } else if (status == 200 && !probe) {
      transport->incoming_streams.push_back(frame->hd.stream_id);
    }
    if (!stream.response_sent &&
        h2_submit_server_response(transport, frame->hd.stream_id, status,
                                  probe || status != 200) < 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  } else if (!transport->server && frame->hd.type == NGHTTP2_HEADERS &&
             frame->headers.cat == NGHTTP2_HCAT_RESPONSE) {
    h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);
    if ((frame->hd.flags & NGHTTP2_FLAG_END_STREAM) == 0 &&
        !stream.lz4_encoded) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    stream.response_received = true;
  }
  if ((frame->hd.type == NGHTTP2_DATA || frame->hd.type == NGHTTP2_HEADERS) &&
      (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0) {
    h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);
    if (frame->hd.type == NGHTTP2_DATA && stream.lz4_encoded &&
        !stream.decoder_finished) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    stream.remote_end = true;
    bool retryable_rejection =
        h2_retryable_handshake_rejection(transport, stream);
    if (frame->hd.stream_id == transport->dispatch_stream_id &&
        !retryable_rejection) {
      transport->transport_failed = true;
    }
  }
  pthread_cond_broadcast(&transport->session_progress);
  return 0;
}

int h2_on_stream_close_callback(nghttp2_session *, int32_t stream_id, uint32_t,
                                void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  h2_release_codecs(stream);
  stream.closed = true;
  bool retryable_rejection =
      h2_retryable_handshake_rejection(transport, stream);
  if (stream_id == transport->dispatch_stream_id && !retryable_rejection) {
    transport->transport_failed = true;
  }
  pthread_cond_broadcast(&transport->session_progress);
  return 0;
}

int h2_on_header_callback(nghttp2_session *, const nghttp2_frame *frame,
                          const uint8_t *name, size_t namelen,
                          const uint8_t *value, size_t valuelen, uint8_t,
                          void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  if (frame->hd.type != NGHTTP2_HEADERS) {
    return 0;
  }
  h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);
  if (namelen == strlen(kContentEncodingHeader) &&
      memcmp(name, kContentEncodingHeader, namelen) == 0) {
    stream.lz4_encoded = !stream.content_encoding_seen &&
                         valuelen == strlen(kLz4Encoding) &&
                         memcmp(value, kLz4Encoding, valuelen) == 0;
    stream.content_encoding_seen = true;
    return 0;
  }
  if (transport->server) {
    if (frame->headers.cat == NGHTTP2_HCAT_REQUEST) {
      if (namelen == strlen(kLupineSessionHeader) &&
          memcmp(name, kLupineSessionHeader, namelen) == 0) {
        transport->session_id.assign(reinterpret_cast<const char *>(value),
                                     valuelen);
      } else if (namelen == strlen(kLupineClientEtagHeader) &&
                 memcmp(name, kLupineClientEtagHeader, namelen) == 0) {
        transport->client_etag.assign(reinterpret_cast<const char *>(value),
                                      valuelen);
      } else if (namelen == strlen(kLupineClientPlatformHeader) &&
                 memcmp(name, kLupineClientPlatformHeader, namelen) == 0) {
        transport->client_platform.assign(reinterpret_cast<const char *>(value),
                                          valuelen);
      } else if (namelen == strlen(kLupineVaBaseHeader) &&
                 memcmp(name, kLupineVaBaseHeader, namelen) == 0) {
        stream.requested_va_base.assign(reinterpret_cast<const char *>(value),
                                        valuelen);
      } else if (namelen == strlen(kLupineVaSizeHeader) &&
                 memcmp(name, kLupineVaSizeHeader, namelen) == 0) {
        stream.requested_va_size.assign(reinterpret_cast<const char *>(value),
                                        valuelen);
      }
    }
    return 0;
  }
  if (frame->headers.cat != NGHTTP2_HCAT_RESPONSE) {
    return 0;
  }
  if (namelen == strlen(kLupineCudaVersionHeader) &&
      memcmp(name, kLupineCudaVersionHeader, namelen) == 0) {
    transport->peer_cuda_version.assign(reinterpret_cast<const char *>(value),
                                        valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineClientEtagHeader) &&
      memcmp(name, kLupineClientEtagHeader, namelen) == 0) {
    transport->peer_client_etag.assign(reinterpret_cast<const char *>(value),
                                       valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineCapabilitiesHeader) &&
      memcmp(name, kLupineCapabilitiesHeader, namelen) == 0) {
    transport->peer_capabilities.assign(reinterpret_cast<const char *>(value),
                                        valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineVaWindowBaseHeader) &&
      memcmp(name, kLupineVaWindowBaseHeader, namelen) == 0) {
    transport->peer_window_base.assign(reinterpret_cast<const char *>(value),
                                       valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineVaWindowSizeHeader) &&
      memcmp(name, kLupineVaWindowSizeHeader, namelen) == 0) {
    transport->peer_window_size.assign(reinterpret_cast<const char *>(value),
                                       valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineVaBaseHeader) &&
      memcmp(name, kLupineVaBaseHeader, namelen) == 0) {
    transport->peer_va_base.assign(reinterpret_cast<const char *>(value),
                                   valuelen);
    return 0;
  }
  if (namelen == strlen(kLupineVaSizeHeader) &&
      memcmp(name, kLupineVaSizeHeader, namelen) == 0) {
    transport->peer_va_size.assign(reinterpret_cast<const char *>(value),
                                   valuelen);
    return 0;
  }
  if (namelen != 7 || memcmp(name, ":status", namelen) != 0) {
    return 0;
  }

  stream.response_status = 0;
  for (size_t i = 0; i < valuelen; ++i) {
    if (value[i] < '0' || value[i] > '9') {
      return 0;
    }
    stream.response_status =
        stream.response_status * 10 + static_cast<int>(value[i] - '0');
  }
  return 0;
}

int h2_flush_session_locked(h2_transport *transport) {
  int result = nghttp2_session_send(transport->session);
  return result == 0 ? 0 : -1;
}

unsigned char *h2_reserve_encoded(h2_stream &stream, size_t capacity) {
  size_t required = stream.encoded_size + capacity;
  if (stream.encoded.size() < required) {
    stream.encoded.resize(required);
  }
  return stream.encoded.data() + stream.encoded_size;
}

int h2_start_encoder_locked(h2_stream &stream) {
  if (LZ4F_isError(
          LZ4F_createCompressionContext(&stream.encoder, LZ4F_VERSION))) {
    return -1;
  }
  const LZ4F_preferences_t preferences = h2_lz4_preferences();
  unsigned char *destination = h2_reserve_encoded(stream, LZ4F_HEADER_SIZE_MAX);
  size_t header = LZ4F_compressBegin(stream.encoder, destination,
                                     LZ4F_HEADER_SIZE_MAX, &preferences);
  if (LZ4F_isError(header)) {
    return -1;
  }
  stream.encoded_size += header;
  stream.encoder_started = true;
  return 0;
}

// Only the blocks the input completes come out; the remainder waits inside
// LZ4F for a flush. LZ4F_compressBound assumes a block already nearly full,
// so it is consulted only when a block will complete; otherwise the update
// needs just the frame end it reserves.
int h2_encode_locked(h2_stream &stream, const unsigned char *data,
                     size_t input) {
  const LZ4F_preferences_t preferences = h2_lz4_preferences();
  size_t capacity = stream.buffered + input < kH2Lz4BlockBytes
                        ? 8
                        : LZ4F_compressBound(input, &preferences);
  unsigned char *destination = h2_reserve_encoded(stream, capacity);
  size_t encoded = LZ4F_compressUpdate(stream.encoder, destination, capacity,
                                       data, input, nullptr);
  if (LZ4F_isError(encoded)) {
    return -1;
  }
  stream.encoded_size += encoded;
  stream.buffered = (stream.buffered + input) % kH2Lz4BlockBytes;
  return 0;
}

// Emits the block LZ4F is holding, or the frame end.
int h2_encode_terminal_locked(h2_stream &stream, bool finish) {
  size_t capacity = stream.buffered + 16;
  unsigned char *destination = h2_reserve_encoded(stream, capacity);
  size_t terminal =
      finish ? LZ4F_compressEnd(stream.encoder, destination, capacity, nullptr)
             : LZ4F_flush(stream.encoder, destination, capacity, nullptr);
  if (LZ4F_isError(terminal)) {
    return -1;
  }
  stream.encoded_size += terminal;
  stream.buffered = 0;
  if (finish) {
    stream.encoder_finished = true;
  }
  return 0;
}

// Hands the stream's encoded bytes to nghttp2. The stream's DATA item is
// submitted on first use and resumed from its deferral after that; whatever
// the peer's window does not cover stays encoded until the read thread pumps
// the session after the next WINDOW_UPDATE.
int h2_pump_stream_locked(h2_transport *transport, int32_t stream_id,
                          h2_stream &stream) {
  if (stream.encoded_offset == stream.encoded_size) {
    return 0;
  }
  if (!stream.provider_submitted) {
    nghttp2_data_provider provider = {};
    provider.read_callback = h2_data_source_read_callback;
    if (nghttp2_submit_data(transport->session, NGHTTP2_FLAG_NONE, stream_id,
                            &provider) != 0) {
      return -1;
    }
    stream.provider_submitted = true;
  } else if (stream.provider_deferred) {
    if (nghttp2_session_resume_data(transport->session, stream_id) != 0) {
      return -1;
    }
    stream.provider_deferred = false;
  }
  return h2_flush_session_locked(transport);
}

// Compresses the cursors in order. Whole blocks reach nghttp2 here; the tail
// short of a block stays with the encoder for the write thread to flush,
// unless the message is awaited: then nothing is gained by deferring, and the
// flush would only drag the encoder and framing state onto the other core.
int h2_write_stream_locked(h2_transport *transport, int32_t stream_id,
                           std::vector<rpc_write_cursor> &cursors, bool flush) {
  h2_stream &stream = h2_get_stream(transport, stream_id);
  if (stream.closed || stream.encoder_finished ||
      (!stream.encoder_started && h2_start_encoder_locked(stream) < 0)) {
    return -1;
  }
  for (rpc_write_cursor &cursor : cursors) {
    for (;;) {
      if (cursor.size == 0) {
        if (cursor.refill == nullptr) {
          break;
        }
        pthread_mutex_unlock(&transport->session_mutex);
        int refill = cursor.refill(cursor.refill_context, &cursor);
        pthread_mutex_lock(&transport->session_mutex);
        if (transport->transport_failed || stream.closed || refill < 0 ||
            (refill == 0 && cursor.remaining() != 0) ||
            (refill > 0 &&
             (cursor.data == nullptr || cursor.remaining() == 0))) {
          return -1;
        }
        if (refill == 0) {
          cursor.refill = nullptr;
        }
        continue;
      }
      size_t input = std::min(cursor.size, kH2Lz4BlockBytes);
      size_t encoded_before = stream.encoded_size;
      if (h2_encode_locked(stream, cursor.data, input) < 0 ||
          (stream.encoded_size != encoded_before &&
           h2_pump_stream_locked(transport, stream_id, stream) < 0)) {
        return -1;
      }
      cursor.data += input;
      cursor.size -= input;
      while (!transport->transport_failed && !stream.closed &&
             stream.encoded_size - stream.encoded_offset >
                 kH2OutboundLimitBytes) {
        pthread_cond_wait(&transport->session_progress,
                          &transport->session_mutex);
      }
      while (!transport->write_failed &&
             transport->outbound.size() > kH2OutboundLimitBytes) {
        pthread_cond_wait(&transport->outbound_progress,
                          &transport->session_mutex);
      }
      if (transport->transport_failed || stream.closed) {
        return -1;
      }
    }
  }
  if (stream.buffered == 0) {
    return 0;
  }
  if (flush) {
    return h2_encode_terminal_locked(stream, false) < 0 ||
                   h2_pump_stream_locked(transport, stream_id, stream) < 0
               ? -1
               : 0;
  }
  if (!stream.flush_queued) {
    transport->flush_pending.push_back(stream_id);
    stream.flush_queued = true;
  }
  return 0;
}

int h2_flush_pending_locked(h2_transport *transport) {
  int result = 0;
  for (int32_t stream_id : transport->flush_pending) {
    h2_stream &stream = h2_get_stream(transport, stream_id);
    stream.flush_queued = false;
    if (stream.closed || stream.encoder_finished) {
      continue;
    }
    if (h2_encode_terminal_locked(stream, false) < 0 ||
        h2_pump_stream_locked(transport, stream_id, stream) < 0) {
      result = -1;
    }
  }
  transport->flush_pending.clear();
  return result;
}

void h2_await_output_locked(h2_transport *transport) {
  if (!transport->server) {
    uint64_t seen =
        transport->output_generation.load(std::memory_order_relaxed);
    pthread_mutex_unlock(&transport->session_mutex);
    auto deadline = std::chrono::steady_clock::now() + kH2WriterPoll;
    while (transport->output_generation.load(std::memory_order_acquire) ==
               seen &&
           std::chrono::steady_clock::now() < deadline) {
    }
    pthread_mutex_lock(&transport->session_mutex);
  }
  if (transport->outbound.empty() && transport->flush_pending.empty() &&
      !transport->write_stop) {
    pthread_cond_wait(&transport->outbound_progress, &transport->session_mutex);
  }
}

void *h2_write_main(void *arg) {
  auto *transport = static_cast<h2_transport *>(arg);
  std::vector<unsigned char> chunk;
  pthread_mutex_lock(&transport->session_mutex);
  // transport_failed alone does not stop the drain: a rejected handshake
  // sets it while the GOAWAY and shutdown PING are still queued.
  for (;;) {
    while (transport->outbound.empty() && transport->flush_pending.empty() &&
           !transport->write_stop) {
      h2_await_output_locked(transport);
    }
    if (transport->write_stop) {
      break;
    }
    int result = h2_flush_pending_locked(transport);
    if (result == 0 && !transport->outbound.empty()) {
      chunk.clear();
      chunk.swap(transport->outbound);
      transport->write_busy = true;
      pthread_mutex_unlock(&transport->session_mutex);
      result = h2_write_socket(transport, chunk.data(), chunk.size());
      pthread_mutex_lock(&transport->session_mutex);
      transport->write_busy = false;
    }
    if (result < 0) {
      transport->write_failed = true;
      transport->transport_failed = true;
      pthread_cond_broadcast(&transport->session_progress);
      break;
    }
    pthread_cond_broadcast(&transport->outbound_progress);
  }
  pthread_cond_broadcast(&transport->outbound_progress);
  pthread_mutex_unlock(&transport->session_mutex);
  return nullptr;
}

void *h2_heartbeat_main(void *arg) {
  auto *transport = static_cast<h2_transport *>(arg);
  pthread_mutex_lock(&transport->session_mutex);
  for (;;) {
    while (transport->response_waiters == 0) {
      pthread_cond_wait(&transport->heartbeat_progress,
                        &transport->session_mutex);
    }
    if (transport->response_waiters < 0) {
      break;
    }

    pthread_mutex_unlock(&transport->session_mutex);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kH2HeartbeatIntervalMs));
    pthread_mutex_lock(&transport->session_mutex);

    std::array<uint8_t, 8> opaque = {};
    if (nghttp2_submit_ping(transport->session, NGHTTP2_FLAG_NONE,
                            opaque.data()) != 0 ||
        h2_flush_session_locked(transport) < 0) {
      // The heartbeat is best-effort. The normal RPC read/write path owns
      // transport error reporting and the in-flight CUDA call's result.
      break;
    }
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return nullptr;
}

ssize_t h2_read_socket(h2_transport *transport, unsigned char *buffer,
                       size_t size) {
#ifdef LUPINE_TLS_OPENSSL
  if (transport->tls != nullptr) {
    SSL *ssl = static_cast<SSL *>(transport->tls);
    for (;;) {
      int r = SSL_read(ssl, buffer, static_cast<int>(size));
      if (r > 0) {
        return r;
      }
      int err = SSL_get_error(ssl, r);
      if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
        continue;
      }
      return -1;
    }
  }
#endif
  ssize_t n;
  do {
    n = lupine_socket_recv(transport->netfd, buffer, size);
  } while (n < 0 && lupine_socket_error_is_intr());
  return n;
}

void *h2_read_main(void *arg) {
  auto *transport = static_cast<h2_transport *>(arg);
  unsigned char buffer[64 * 1024];
  for (;;) {
    ssize_t received = h2_read_socket(transport, buffer, sizeof(buffer));
    pthread_mutex_lock(&transport->session_mutex);
    size_t offset = 0;
    while (received > 0 && offset < static_cast<size_t>(received)) {
      ssize_t consumed =
          nghttp2_session_mem_recv(transport->session, buffer + offset,
                                   static_cast<size_t>(received) - offset);
      if (consumed <= 0) {
        received = -1;
        break;
      }
      offset += static_cast<size_t>(consumed);
    }
    if (received <= 0 || h2_flush_session_locked(transport) < 0) {
      transport->transport_failed = true;
      pthread_cond_broadcast(&transport->session_progress);
      pthread_mutex_unlock(&transport->session_mutex);
      return nullptr;
    }
    // Wake stream readers, acceptors, and flow-controlled writers after the
    // callbacks have applied this batch of connection events.
    pthread_cond_broadcast(&transport->session_progress);
    pthread_mutex_unlock(&transport->session_mutex);
  }
}

int32_t h2_submit_client_handshake(h2_transport *transport, conn_t *conn,
                                   bool probe) {
  transport->peer_cuda_version.clear();
  transport->peer_capabilities.clear();
  transport->peer_client_etag.clear();
  transport->peer_va_base.clear();
  transport->peer_va_size.clear();
  transport->peer_window_base.clear();
  transport->peer_window_size.clear();

  std::vector<nghttp2_nv> headers = {
      h2_nv(":method", probe ? "HEAD" : "POST"),
      h2_nv(":scheme", "http"),
      h2_nv(":path", "/"),
      h2_nv(":authority", "lupine"),
  };
  if (!probe) {
    headers.push_back(h2_nv(kContentEncodingHeader, kLz4Encoding));
    const char *client_etag = getenv("LUPINE_CLIENT_ETAG");
    const char *client_platform = getenv("LUPINE_CLIENT_PLATFORM");
    const char *session_id = getenv("LUPINE_SESSION");
    transport->client_etag =
        client_etag == nullptr ? std::string() : client_etag;
    transport->client_platform =
        client_platform == nullptr ? std::string() : client_platform;
    transport->session_id = session_id == nullptr ? std::string() : session_id;
    if (!transport->client_etag.empty()) {
      headers.push_back(
          h2_nv(kLupineClientEtagHeader, transport->client_etag.c_str()));
    }
    if (!transport->client_platform.empty()) {
      headers.push_back(h2_nv(kLupineClientPlatformHeader,
                              transport->client_platform.c_str()));
    }
    if (!transport->session_id.empty()) {
      headers.push_back(
          h2_nv(kLupineSessionHeader, transport->session_id.c_str()));
    }
    if (conn->va_size != 0) {
      transport->local_va_base = h2_hex(conn->va_base);
      transport->local_va_size = h2_hex(conn->va_size);
      headers.push_back(
          h2_nv(kLupineVaBaseHeader, transport->local_va_base.c_str()));
      headers.push_back(
          h2_nv(kLupineVaSizeHeader, transport->local_va_size.c_str()));
    }
  }
  uint8_t flags = probe ? NGHTTP2_FLAG_END_STREAM : NGHTTP2_FLAG_NONE;
  int32_t stream_id =
      nghttp2_submit_headers(transport->session, flags, -1, nullptr,
                             headers.data(), headers.size(), nullptr);
  if (stream_id >= 0) {
    transport->dispatch_stream_id = stream_id;
    h2_get_stream(transport, stream_id);
  }
  return stream_id;
}

void h2_drain_output_locked(h2_transport *transport) {
  while ((!transport->outbound.empty() || !transport->flush_pending.empty() ||
          transport->write_busy) &&
         !transport->write_failed) {
    pthread_cond_wait(&transport->outbound_progress, &transport->session_mutex);
  }
}

void h2_stop_write_thread(h2_transport *transport) {
  pthread_mutex_lock(&transport->session_mutex);
  h2_drain_output_locked(transport);
  transport->write_stop = true;
  pthread_cond_broadcast(&transport->outbound_progress);
  pthread_mutex_unlock(&transport->session_mutex);
  pthread_join(transport->write_thread, nullptr);
  transport->write_thread = 0;
}

int h2_init_direct(conn_t *conn, bool server, bool probe,
                   const rpc_http2_server_metadata *metadata = nullptr) {
  auto *transport = new h2_transport();
  transport->netfd = conn->connfd;
  transport->conn = conn;
  transport->tls = conn->tls_session;
  transport->server = server;
  if (metadata != nullptr && metadata->backend_version != nullptr) {
    transport->server_version = metadata->backend_version;
  }
  if (metadata != nullptr && metadata->capabilities != 0) {
    transport->server_capabilities = h2_hex(metadata->capabilities);
  }
  if (metadata != nullptr) {
    transport->client_bundles = metadata->client_bundles;
  }

  nghttp2_session_callbacks *callbacks = nullptr;
  if (nghttp2_session_callbacks_new(&callbacks) != 0) {
    delete transport;
    return -1;
  }
  nghttp2_session_callbacks_set_send_callback(callbacks, h2_send_callback);
  nghttp2_session_callbacks_set_send_data_callback(callbacks,
                                                   h2_send_data_callback);
  nghttp2_session_callbacks_set_data_source_read_length_callback(
      callbacks, h2_data_source_read_length_callback);
  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
      callbacks, h2_on_data_chunk_recv_callback);
  nghttp2_session_callbacks_set_on_frame_recv_callback(
      callbacks, h2_on_frame_recv_callback);
  nghttp2_session_callbacks_set_on_stream_close_callback(
      callbacks, h2_on_stream_close_callback);
  nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                   h2_on_header_callback);

  // The server credits received DATA back by hand so a fire-and-forget payload
  // keeps its window charged for as long as its staging buffer lives.
  nghttp2_option *option = nullptr;
  if (server) {
    if (nghttp2_option_new(&option) != 0) {
      nghttp2_session_callbacks_del(callbacks);
      delete transport;
      return -1;
    }
    nghttp2_option_set_no_auto_window_update(option, 1);
  }
  int session_result =
      server ? nghttp2_session_server_new2(&transport->session, callbacks,
                                           transport, option)
             : nghttp2_session_client_new(&transport->session, callbacks,
                                          transport);
  nghttp2_session_callbacks_del(callbacks);
  if (option != nullptr) {
    nghttp2_option_del(option);
  }
  if (session_result != 0) {
    delete transport;
    return -1;
  }

  const uint32_t window = server ? kH2ServerWindow : kH2ClientWindow;
  nghttp2_settings_entry settings[] = {
      {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, window},
      {NGHTTP2_SETTINGS_MAX_FRAME_SIZE, kH2MaxFrame},
  };
  if (nghttp2_submit_settings(transport->session, NGHTTP2_FLAG_NONE, settings,
                              2) != 0 ||
      nghttp2_session_set_local_window_size(
          transport->session, NGHTTP2_FLAG_NONE, 0,
          static_cast<int32_t>(window)) != 0) {
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }

  if (!server) {
    if (h2_submit_client_handshake(transport, conn, probe) < 0) {
      nghttp2_session_del(transport->session);
      delete transport;
      return -1;
    }
  }

  conn->http2 = transport;
  if (pthread_create(&transport->write_thread, nullptr, h2_write_main,
                     transport) != 0) {
    conn->http2 = nullptr;
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }
  pthread_mutex_lock(&transport->session_mutex);
  int flush_result = h2_flush_session_locked(transport);
  pthread_mutex_unlock(&transport->session_mutex);
  if (flush_result < 0 || pthread_create(&transport->read_thread, nullptr,
                                         h2_read_main, transport) != 0) {
    conn->http2 = nullptr;
    h2_stop_write_thread(transport);
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }
  return 0;
}

} // namespace

int rpc_http2_read_stream(conn_t *conn, int32_t stream_id, void *data,
                          size_t size) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  auto *out = static_cast<unsigned char *>(data);
  size_t copied = 0;
  pthread_mutex_lock(&transport->session_mutex);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  while (!stream.local_out.empty() && copied < size) {
    h2_buffer &front = stream.local_out.front();
    size_t available = front.data.size() - front.offset;
    size_t chunk = std::min(available, size - copied);
    memcpy(out + copied, front.data.data() + front.offset, chunk);
    front.offset += chunk;
    copied += chunk;
    transport->read_stats.staged_read_bytes += chunk;
    transport->staged_bytes -= chunk;
    if (front.offset == front.data.size()) {
      if (transport->buffer_pool_bytes + front.data.capacity() <=
          kH2StagingPoolBytes) {
        transport->buffer_pool_bytes += front.data.capacity();
        transport->buffer_pool.push_back(std::move(front));
      }
      stream.local_out.pop_front();
    }
  }
  if (copied == size) {
    pthread_mutex_unlock(&transport->session_mutex);
    return static_cast<int>(size);
  }
  if (stream.read_destination != nullptr ||
      (stream.response_status != 0 && stream.response_status != 200)) {
    pthread_mutex_unlock(&transport->session_mutex);
    return -1;
  }
  if (transport->transport_failed || stream.closed || stream.remote_end) {
    int result = stream.remote_end ? LUPINE_RPC_HTTP2_STREAM_END : -1;
    pthread_mutex_unlock(&transport->session_mutex);
    return result;
  }

  stream.read_destination = out + copied;
  stream.read_remaining = size - copied;
  while (stream.read_remaining != 0 && !transport->transport_failed &&
         !stream.closed && !stream.remote_end &&
         (stream.response_status == 0 || stream.response_status == 200)) {
    pthread_cond_wait(&transport->session_progress, &transport->session_mutex);
  }
  bool complete = stream.read_remaining == 0;
  stream.read_destination = nullptr;
  stream.read_remaining = 0;
  int result = -1;
  if (complete) {
    result = static_cast<int>(size);
  } else if (stream.remote_end) {
    result = LUPINE_RPC_HTTP2_STREAM_END;
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int rpc_http2_read(conn_t *conn, void *data, size_t size) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  return rpc_http2_read_stream(conn, transport->dispatch_stream_id, data, size);
}

int rpc_http2_write_stream(conn_t *conn, int32_t stream_id,
                           std::vector<rpc_write_cursor> &cursors, bool flush) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  if (std::all_of(
          cursors.begin(), cursors.end(),
          [](const rpc_write_cursor &cursor) { return !cursor.pending(); })) {
    return 0;
  }

  pthread_mutex_lock(&transport->session_mutex);
  int result = h2_write_stream_locked(transport, stream_id, cursors, flush);
  pthread_mutex_unlock(&transport->session_mutex);
  // Signalled after the unlock so the write thread never wakes into a mutex
  // its producer still holds.
  transport->output_generation.fetch_add(1, std::memory_order_release);
  pthread_cond_broadcast(&transport->outbound_progress);
  return result;
}

int rpc_http2_write(conn_t *conn, std::vector<rpc_write_cursor> &cursors) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  return rpc_http2_write_stream(conn, transport->dispatch_stream_id, cursors,
                                false);
}

int32_t rpc_http2_dispatch_stream(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  int32_t stream_id = transport->dispatch_stream_id;
  pthread_mutex_unlock(&transport->session_mutex);
  return stream_id;
}

int32_t rpc_http2_lane_stream(conn_t *conn, uint64_t lane_id) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  if (transport->server) {
    int32_t stream_id = transport->dispatch_stream_id;
    pthread_mutex_unlock(&transport->session_mutex);
    return stream_id;
  }
  auto existing = transport->local_lanes.find(lane_id);
  if (existing != transport->local_lanes.end()) {
    int32_t stream_id = existing->second;
    pthread_mutex_unlock(&transport->session_mutex);
    return stream_id;
  }
  if (transport->transport_failed) {
    pthread_mutex_unlock(&transport->session_mutex);
    return -1;
  }

  std::array<nghttp2_nv, 5> headers = {
      h2_nv(":method", "POST"),
      h2_nv(":scheme", "http"),
      h2_nv(":path", "/"),
      h2_nv(":authority", "lupine"),
      h2_nv(kContentEncodingHeader, kLz4Encoding),
  };
  int32_t stream_id =
      nghttp2_submit_headers(transport->session, NGHTTP2_FLAG_NONE, -1, nullptr,
                             headers.data(), headers.size(), nullptr);
  if (stream_id >= 0) {
    h2_get_stream(transport, stream_id);
    transport->local_lanes.emplace(lane_id, stream_id);
    if (h2_flush_session_locked(transport) < 0) {
      stream_id = -1;
      transport->transport_failed = true;
    }
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return stream_id;
}

namespace {

int h2_end_stream_locked(h2_transport *transport, int32_t stream_id) {
  h2_stream &stream = h2_get_stream(transport, stream_id);
  if (stream.closed || stream.encoder_finished ||
      (!stream.encoder_started && h2_start_encoder_locked(stream) < 0) ||
      h2_encode_terminal_locked(stream, true) < 0) {
    return -1;
  }
  return h2_pump_stream_locked(transport, stream_id, stream);
}

} // namespace

int rpc_http2_flush(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  h2_drain_output_locked(transport);
  int result = transport->write_failed ? -1 : 0;
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int rpc_http2_end_stream(conn_t *conn, int32_t stream_id) {
  if (conn == nullptr || conn->http2 == nullptr || stream_id < 0) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  int result = h2_end_stream_locked(transport, stream_id);
  pthread_mutex_unlock(&transport->session_mutex);
  return result == 0 ? 0 : -1;
}

int rpc_write_lane_termination(conn_t *conn, uint64_t lane_id) {
  if (conn == nullptr || conn->closed || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  auto lane = transport->local_lanes.find(lane_id);
  if (lane == transport->local_lanes.end()) {
    pthread_mutex_unlock(&transport->session_mutex);
    return 0;
  }
  int32_t stream_id = lane->second;
  transport->local_lanes.erase(lane);
  int result = h2_end_stream_locked(transport, stream_id);
  pthread_mutex_unlock(&transport->session_mutex);
  return result == 0 ? 0 : -1;
}

int32_t rpc_http2_accept_stream(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  while (transport->incoming_streams.empty() && !transport->transport_failed) {
    pthread_cond_wait(&transport->session_progress, &transport->session_mutex);
  }
  int32_t stream_id = -1;
  if (!transport->incoming_streams.empty()) {
    stream_id = transport->incoming_streams.front();
    transport->incoming_streams.pop_front();
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return stream_id;
}

const char *rpc_http2_session_id(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return nullptr;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  return transport->session_id.empty() ? nullptr
                                       : transport->session_id.c_str();
}

int rpc_http2_get_read_stats(conn_t *conn, rpc_http2_read_stats *stats) {
  if (conn == nullptr || conn->http2 == nullptr || stats == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  *stats = transport->read_stats;
  pthread_mutex_unlock(&transport->session_mutex);
  return 0;
}

void rpc_http2_response_wait_begin(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  if (transport->server) {
    return;
  }
  pthread_mutex_lock(&transport->session_mutex);
  if (transport->heartbeat_thread != 0 && transport->response_waiters >= 0) {
    ++transport->response_waiters;
    pthread_cond_broadcast(&transport->heartbeat_progress);
  }
  pthread_mutex_unlock(&transport->session_mutex);
}

void rpc_http2_response_wait_end(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  if (transport->response_waiters > 0) {
    --transport->response_waiters;
  }
  pthread_mutex_unlock(&transport->session_mutex);
}

void rpc_http2_window_hold_begin(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  int32_t stream_id = rpc_current_http2_stream(conn);
  pthread_mutex_lock(&transport->session_mutex);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  stream.window_hold = true;
  stream.window_hold_bytes = 0;
  pthread_mutex_unlock(&transport->session_mutex);
}

rpc_http2_window_credit rpc_http2_window_hold_end(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return {};
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  int32_t stream_id = rpc_current_http2_stream(conn);
  pthread_mutex_lock(&transport->session_mutex);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  rpc_http2_window_credit credit{stream_id, stream.window_hold_bytes};
  stream.window_hold = false;
  stream.window_hold_bytes = 0;
  pthread_mutex_unlock(&transport->session_mutex);
  return credit;
}

// Whichever thread retires the staging emits the credit itself, under
// session_mutex alone: the transport never takes a staging lock, and nothing
// holding session_mutex waits on staging, so a reader starved of window is
// never queued behind the release that would feed it. kH2MaxHeldBytes closes
// the other half of the cycle -- staging that never retires cannot shut the
// window on the reads that would retire it.
void rpc_http2_window_release(conn_t *conn, rpc_http2_window_credit credit) {
  if (conn == nullptr || conn->http2 == nullptr || credit.bytes == 0 ||
      credit.stream_id < 0) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  transport->window_held -= std::min(credit.bytes, transport->window_held);
  if (nghttp2_session_consume(transport->session, credit.stream_id,
                              credit.bytes) == 0) {
    (void)h2_flush_session_locked(transport);
  }
  pthread_mutex_unlock(&transport->session_mutex);
}

int rpc_http2_client_init(conn_t *conn) {
  if (h2_init_direct(conn, false, false) < 0) {
    return -1;
  }
  if (conn->va_size == 0) {
    return 0;
  }
  return rpc_http2_client_await_ready(conn);
}

int rpc_http2_client_retry_handshake(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  int32_t previous_stream_id = transport->dispatch_stream_id;
  h2_stream &previous = h2_get_stream(transport, previous_stream_id);
  int result = -1;
  if (!transport->server && !transport->transport_failed &&
      h2_retryable_handshake_rejection(transport, previous)) {
    // The request side of the rejected stream was deliberately left open in
    // case it became the long-lived RPC stream. Retire only that stream, then
    // negotiate the next arena without replacing the HTTP/2 connection.
    if (nghttp2_submit_rst_stream(transport->session, NGHTTP2_FLAG_NONE,
                                  previous_stream_id, NGHTTP2_CANCEL) == 0 &&
        h2_submit_client_handshake(transport, conn, false) >= 0 &&
        h2_flush_session_locked(transport) == 0) {
      result = 0;
    } else {
      transport->transport_failed = true;
      pthread_cond_broadcast(&transport->session_progress);
    }
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return result == 0 ? rpc_http2_client_await_ready(conn) : -1;
}

// The server answers the request headers before any payload flows, so this
// costs one round trip and settles both the arena request and the build check
// on the connection that will carry the session. Kept separate from
// rpc_http2_client_init so a caller with no live peer can skip the wait.
int rpc_http2_client_await_ready(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  h2_stream &stream = h2_get_stream(transport, transport->dispatch_stream_id);
  while (!stream.response_received && !transport->transport_failed) {
    pthread_cond_wait(&transport->session_progress, &transport->session_mutex);
  }
  uintptr_t peer_base = 0;
  uintptr_t peer_size = 0;
  bool responded = stream.response_received;
  int status = stream.response_status;
  bool arena_granted = responded && status == 200 &&
                       h2_parse_hex(transport->peer_va_base, &peer_base) &&
                       h2_parse_hex(transport->peer_va_size, &peer_size) &&
                       peer_base == conn->va_base && peer_size == conn->va_size;
  std::string expected_client_etag = transport->peer_client_etag;
  pthread_mutex_unlock(&transport->session_mutex);

  // The server can move to a new bundle between discovery and connect. Do not
  // misreport that as an arena conflict; the launcher needs to refetch and
  // restart the process before any CUDA state exists.
  if (responded && status == 426) {
    LUPINE_LOG_ERROR("LUPINE client bundle is no longer current"
                     << (expected_client_etag.empty()
                             ? ""
                             : "; server expects " + expected_client_etag));
    return LUPINE_RPC_HTTP2_CLIENT_MISMATCH;
  }
  if (conn->va_size == 0) {
    return responded && status == 200 ? 0 : -1;
  }
  if (arena_granted) {
    return 0;
  }
  return responded && status == 409 ? LUPINE_RPC_HTTP2_VA_CONFLICT : -1;
}

void rpc_http2_client_start_heartbeat(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  if (transport->heartbeat_thread == 0 && transport->response_waiters >= 0) {
    pthread_t thread = {};
    if (pthread_create(&thread, nullptr, h2_heartbeat_main, transport) == 0) {
      transport->heartbeat_thread = thread;
    }
  }
  pthread_mutex_unlock(&transport->session_mutex);
}

const char *rpc_http2_client_probe(conn_t *conn) {
  if (h2_init_direct(conn, false, true) < 0) {
    return nullptr;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  h2_stream &stream = h2_get_stream(transport, transport->dispatch_stream_id);
  while (!stream.response_received && !transport->transport_failed) {
    pthread_cond_wait(&transport->session_progress, &transport->session_mutex);
  }
  const char *version = nullptr;
  if (stream.response_received && stream.response_status == 200 &&
      !transport->peer_cuda_version.empty()) {
    version = transport->peer_cuda_version.c_str();
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return version;
}

// False when the peer stated no window, in which case it hosts no arena at all
// and the caller should not reserve one.
bool rpc_http2_peer_va_window(conn_t *conn, lupine_va_window *window) {
  if (conn == nullptr || conn->http2 == nullptr || window == nullptr) {
    return false;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  uintptr_t base = 0;
  uintptr_t size = 0;
  bool parsed = h2_parse_hex(transport->peer_window_base, &base) &&
                h2_parse_hex(transport->peer_window_size, &size);
  pthread_mutex_unlock(&transport->session_mutex);
  if (!parsed || size == 0 || base > UINTPTR_MAX - size) {
    return false;
  }
  window->base = base;
  window->size = static_cast<size_t>(size);
  return true;
}

bool rpc_http2_peer_supports(conn_t *conn, uint64_t capabilities) {
  if (conn == nullptr || conn->http2 == nullptr || capabilities == 0) {
    return false;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  uint64_t advertised = 0;
  bool parsed = h2_parse_hex64(transport->peer_capabilities, &advertised);
  pthread_mutex_unlock(&transport->session_mutex);
  return parsed && (advertised & capabilities) == capabilities;
}

int rpc_http2_server_init(conn_t *conn) {
  return rpc_http2_server_init_with_metadata(conn, nullptr);
}

int rpc_http2_server_init_with_metadata(
    conn_t *conn, const rpc_http2_server_metadata *metadata) {
  if (h2_init_direct(conn, true, false, metadata) < 0) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  int result = -1;
  for (;;) {
    while (!transport->request_received && !transport->transport_failed) {
      pthread_cond_wait(&transport->session_progress,
                        &transport->session_mutex);
    }
    if (transport->transport_failed) {
      break;
    }
    if (transport->request_status == 409) {
      transport->request_received = false;
      continue;
    }
    result = transport->request_handled ? 1 : 0;
    break;
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int rpc_http2_server_graceful_shutdown(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return -1;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  int32_t last_stream_id =
      nghttp2_session_get_last_proc_stream_id(transport->session);
  int result =
      nghttp2_submit_goaway(transport->session, NGHTTP2_FLAG_NONE,
                            last_stream_id, NGHTTP2_NO_ERROR, nullptr, 0);
  if (result == 0) {
    result = nghttp2_submit_ping(transport->session, NGHTTP2_FLAG_NONE,
                                 kH2ShutdownPing.data());
  }
  if (result == 0) {
    result = h2_flush_session_locked(transport);
  }
  while (result == 0 && !transport->shutdown_acknowledged &&
         !transport->transport_failed) {
    if (pthread_cond_wait(&transport->session_progress,
                          &transport->session_mutex) != 0) {
      result = -1;
    }
  }
  if (!transport->shutdown_acknowledged) {
    result = -1;
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return result == 0 ? 0 : -1;
}

void rpc_http2_destroy(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  conn->http2 = nullptr;
#ifdef _WIN32
  (void)shutdown(transport->netfd, SD_RECEIVE);
#else
  (void)shutdown(transport->netfd, SHUT_RD);
#endif
  pthread_mutex_lock(&transport->session_mutex);
  transport->response_waiters = -1;
  transport->transport_failed = true;
  pthread_cond_broadcast(&transport->heartbeat_progress);
  pthread_cond_broadcast(&transport->session_progress);
  pthread_mutex_unlock(&transport->session_mutex);
  if (transport->heartbeat_thread != 0) {
    pthread_join(transport->heartbeat_thread, nullptr);
    transport->heartbeat_thread = 0;
  }
  if (transport->read_thread != 0) {
    pthread_join(transport->read_thread, nullptr);
    transport->read_thread = 0;
  }
  if (transport->write_thread != 0) {
    h2_stop_write_thread(transport);
  }
  if (transport->session != nullptr) {
    nghttp2_session_del(transport->session);
    transport->session = nullptr;
  }
  for (auto &[stream_id, stream] : transport->streams) {
    (void)stream_id;
    h2_release_codecs(stream);
  }
  pthread_cond_destroy(&transport->outbound_progress);
  pthread_cond_destroy(&transport->heartbeat_progress);
  pthread_cond_destroy(&transport->session_progress);
  pthread_mutex_destroy(&transport->session_mutex);
  delete transport;
}
