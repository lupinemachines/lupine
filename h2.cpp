#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <deque>
#include <errno.h>
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
#include <zlib.h>

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
constexpr size_t kH2InflateBufferBytes = 64 * 1024;
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
  bool gzip_encoded = false;
  bool deflater_initialized = false;
  bool deflater_finished = false;
  bool inflater_initialized = false;
  bool inflater_finished = false;
  z_stream deflater = {};
  z_stream inflater = {};
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
  int32_t dispatch_stream_id = -1;
  nghttp2_session *session = nullptr;
  std::unordered_map<int32_t, h2_stream> streams;
  std::deque<int32_t> incoming_streams;
  std::unordered_map<uint64_t, int32_t> local_lanes;
  rpc_http2_read_stats read_stats = {};
  uint64_t staged_bytes = 0;
  uint64_t window_held = 0;
  pthread_mutex_t session_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t session_progress = PTHREAD_COND_INITIALIZER;
  pthread_cond_t heartbeat_progress = PTHREAD_COND_INITIALIZER;
  pthread_t read_thread = {};
  pthread_t heartbeat_thread = {};
  int response_waiters = 0;
  bool transport_failed = false;
  std::string peer_cuda_version;
  std::string peer_wire_identity;
  std::string server_version;
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

void h2_release_codecs(h2_stream &stream) {
  if (stream.deflater_initialized) {
    deflateEnd(&stream.deflater);
    stream.deflater_initialized = false;
  }
  if (stream.inflater_initialized) {
    inflateEnd(&stream.inflater);
    stream.inflater_initialized = false;
  }
}

struct h2_write_source {
  std::vector<rpc_write_cursor> *cursors = nullptr;
  size_t index = 0;
  uint64_t progress = 0;
  bool finish_stream = false;
  bool complete = false;
};

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
  buffer.data.assign(data, data + len);
  stream.local_out.push_back(std::move(buffer));
  transport->read_stats.staged_bytes += len;
  ++transport->read_stats.staged_buffers;
  transport->staged_bytes += len;
  transport->read_stats.peak_staged_bytes = std::max(
      transport->read_stats.peak_staged_bytes, transport->staged_bytes);
}

int h2_write_all(h2_transport *transport, const uint8_t *data, size_t size) {
  while (size != 0) {
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
      struct iovec iov = {const_cast<uint8_t *>(data), size};
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
    data += n;
    size -= static_cast<size_t>(n);
  }
  return 0;
}

ssize_t h2_send_callback(nghttp2_session *, const uint8_t *data, size_t length,
                         int, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  return h2_write_all(transport, data, length) == 0
             ? static_cast<ssize_t>(length)
             : NGHTTP2_ERR_CALLBACK_FAILURE;
}

int h2_start_deflater(h2_stream &stream) {
  if (stream.deflater_finished) {
    return -1;
  }
  if (stream.deflater_initialized) {
    return 0;
  }
  if (deflateInit2(&stream.deflater, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                   MAX_WBITS + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
    return -1;
  }
  stream.deflater_initialized = true;
  return 0;
}

ssize_t h2_data_source_read_callback(nghttp2_session *, int32_t stream_id,
                                     uint8_t *buffer, size_t length,
                                     uint32_t *data_flags,
                                     nghttp2_data_source *source,
                                     void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  auto *write_source = static_cast<h2_write_source *>(source->ptr);
  h2_stream &stream = h2_get_stream(transport, stream_id);
  if (length == 0 || length > UINT_MAX || write_source->complete ||
      h2_start_deflater(stream) < 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  z_stream &deflater = stream.deflater;
  deflater.next_out = buffer;
  deflater.avail_out = static_cast<uInt>(length);
  while (deflater.avail_out != 0 && !write_source->complete) {
    rpc_write_cursor *cursor = nullptr;
    if (write_source->cursors != nullptr) {
      auto &cursors = *write_source->cursors;
      while (write_source->index < cursors.size() &&
             cursors[write_source->index].remaining() == 0) {
        ++write_source->index;
      }
      if (write_source->index < cursors.size()) {
        cursor = &cursors[write_source->index];
        deflater.next_in = const_cast<Bytef *>(cursor->data);
        deflater.avail_in = static_cast<uInt>(
            std::min(cursor->size, static_cast<size_t>(UINT_MAX)));
      }
    }

    const uInt input_before = deflater.avail_in;
    const uInt output_before = deflater.avail_out;
    int flush = cursor == nullptr
                    ? (write_source->finish_stream ? Z_FINISH : Z_SYNC_FLUSH)
                    : Z_NO_FLUSH;
    int result = deflate(&deflater, flush);
    const size_t consumed = input_before - deflater.avail_in;
    const size_t produced = output_before - deflater.avail_out;
    write_source->progress += consumed + produced;

    if (cursor != nullptr) {
      cursor->data += consumed;
      cursor->size -= consumed;
      if (cursor->size == 0) {
        ++write_source->index;
      }
    }

    if (result == Z_STREAM_END && write_source->finish_stream) {
      stream.deflater_finished = true;
      write_source->complete = true;
      *data_flags |= NGHTTP2_DATA_FLAG_EOF;
    } else if (result != Z_OK && result != Z_BUF_ERROR) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    } else if (cursor == nullptr && !write_source->finish_stream &&
               deflater.avail_out != 0) {
      write_source->complete = true;
      *data_flags |= NGHTTP2_DATA_FLAG_EOF | NGHTTP2_DATA_FLAG_NO_END_STREAM;
    } else if (consumed == 0 && produced == 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  return static_cast<ssize_t>(length - deflater.avail_out);
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
  if (!stream.gzip_encoded || stream.inflater_finished || len > UINT_MAX) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  if (!stream.inflater_initialized) {
    if (inflateInit2(&stream.inflater, MAX_WBITS + 16) != Z_OK) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    stream.inflater_initialized = true;
  }

  z_stream &inflater = stream.inflater;
  inflater.next_in = const_cast<Bytef *>(data);
  inflater.avail_in = static_cast<uInt>(len);
  std::array<unsigned char, kH2InflateBufferBytes> output;
  for (;;) {
    inflater.next_out = output.data();
    inflater.avail_out = static_cast<uInt>(output.size());
    const uInt input_before = inflater.avail_in;
    int result = inflate(&inflater, Z_NO_FLUSH);
    size_t produced = output.size() - inflater.avail_out;
    receive_bytes(transport, stream_id, output.data(), produced);
    if (result == Z_STREAM_END) {
      if (inflater.avail_in != 0) {
        return NGHTTP2_ERR_CALLBACK_FAILURE;
      }
      stream.inflater_finished = true;
      break;
    }
    if (result == Z_BUF_ERROR && inflater.avail_in == 0 && produced == 0) {
      break;
    }
    if (result != Z_OK ||
        (input_before == inflater.avail_in && produced == 0)) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    if (inflater.avail_in == 0 && inflater.avail_out != 0) {
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
constexpr char kLupineWireIdentityHeader[] = "x-lupine-wire-identity";
constexpr char kLupineVaWindowBaseHeader[] = "x-lupine-va-window-base";
constexpr char kLupineVaWindowSizeHeader[] = "x-lupine-va-window-size";
constexpr char kContentEncodingHeader[] = "content-encoding";
constexpr char kGzipEncoding[] = "gzip";

#ifndef LUPINE_WIRE_IDENTITY
#define LUPINE_WIRE_IDENTITY ""
#endif

std::string h2_hex(uintptr_t value) {
  std::ostringstream result;
  result << std::hex << value;
  return result.str();
}

bool h2_parse_hex(const std::string &value, uintptr_t *parsed) {
  if (value.empty() || parsed == nullptr) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  unsigned long long number = strtoull(value.c_str(), &end, 16);
  if (errno != 0 || end == value.c_str() || *end != '\0' ||
      number > UINTPTR_MAX) {
    return false;
  }
  *parsed = static_cast<uintptr_t>(number);
  return true;
}

// An empty identity on either side means that peer was built without git and
// cannot state what it is; treat that as unverifiable and let the connection
// proceed rather than refusing every source-tarball build.
bool h2_wire_identity_compatible(const std::string &local,
                                 const std::string &peer) {
  return local.empty() || peer.empty() || local == peer;
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
  }
  std::vector<nghttp2_nv> headers = {h2_nv(":status", status_text)};
  if (!end_stream) {
    headers.push_back(h2_nv(kContentEncodingHeader, kGzipEncoding));
  }
  if (!transport->server_version.empty()) {
    headers.push_back(
        h2_nv(kLupineCudaVersionHeader, transport->server_version.c_str()));
  }
  if (LUPINE_WIRE_IDENTITY[0] != '\0') {
    headers.push_back(h2_nv(kLupineWireIdentityHeader, LUPINE_WIRE_IDENTITY));
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
    int status = stream.gzip_encoded || probe ? 200 : 400;
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
    transport->request_handled = probe || status != 200;
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
        !stream.gzip_encoded) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    stream.response_received = true;
  }
  if ((frame->hd.type == NGHTTP2_DATA || frame->hd.type == NGHTTP2_HEADERS) &&
      (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) != 0) {
    h2_stream &stream = h2_get_stream(transport, frame->hd.stream_id);
    if (frame->hd.type == NGHTTP2_DATA && stream.gzip_encoded &&
        !stream.inflater_finished) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    stream.remote_end = true;
    if (frame->hd.stream_id == transport->dispatch_stream_id) {
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
  if (stream_id == transport->dispatch_stream_id) {
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
    stream.gzip_encoded = !stream.content_encoding_seen &&
                          valuelen == strlen(kGzipEncoding) &&
                          memcmp(value, kGzipEncoding, valuelen) == 0;
    stream.content_encoding_seen = true;
    return 0;
  }
  if (transport->server) {
    if (frame->headers.cat == NGHTTP2_HCAT_REQUEST) {
      if (namelen == strlen(kLupineSessionHeader) &&
          memcmp(name, kLupineSessionHeader, namelen) == 0) {
        transport->session_id.assign(reinterpret_cast<const char *>(value),
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
  if (namelen == strlen(kLupineWireIdentityHeader) &&
      memcmp(name, kLupineWireIdentityHeader, namelen) == 0) {
    transport->peer_wire_identity.assign(reinterpret_cast<const char *>(value),
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

int h2_send_source_locked(h2_transport *transport, int32_t stream_id,
                          h2_write_source *source) {
  nghttp2_data_provider provider = {};
  provider.source.ptr = source;
  provider.read_callback = h2_data_source_read_callback;
  int result = nghttp2_submit_data(transport->session, NGHTTP2_FLAG_NONE,
                                   stream_id, &provider);
  // nghttp2 retains provider.source.ptr until the provider reaches EOF. Keep
  // the stack-backed source alive while flow control pauses the stream, and
  // release the mutex while the read thread applies WINDOW_UPDATE frames.
  while (result == 0 && !source->complete) {
    uint64_t progress = source->progress;
    if (h2_flush_session_locked(transport) < 0) {
      result = -1;
      break;
    }
    if (source->complete) {
      break;
    }
    if (transport->transport_failed) {
      result = -1;
      break;
    }
    if (source->progress == progress &&
        pthread_cond_wait(&transport->session_progress,
                          &transport->session_mutex) != 0) {
      result = -1;
      break;
    }
  }
  return result == 0 ? 0 : -1;
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

  nghttp2_session_callbacks *callbacks = nullptr;
  if (nghttp2_session_callbacks_new(&callbacks) != 0) {
    delete transport;
    return -1;
  }
  nghttp2_session_callbacks_set_send_callback(callbacks, h2_send_callback);
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
    const char *session_id = getenv("LUPINE_SESSION");
    std::vector<nghttp2_nv> headers = {
        h2_nv(":method", probe ? "HEAD" : "POST"),
        h2_nv(":scheme", "http"),
        h2_nv(":path", "/"),
        h2_nv(":authority", "lupine"),
    };
    if (!probe) {
      headers.push_back(h2_nv(kContentEncodingHeader, kGzipEncoding));
      if (session_id != nullptr && session_id[0] != '\0') {
        headers.push_back(h2_nv(kLupineSessionHeader, session_id));
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
    if (stream_id < 0) {
      nghttp2_session_del(transport->session);
      delete transport;
      return -1;
    }
    transport->dispatch_stream_id = stream_id;
    h2_get_stream(transport, stream_id);
  }

  conn->http2 = transport;
  pthread_mutex_lock(&transport->session_mutex);
  int flush_result = h2_flush_session_locked(transport);
  pthread_mutex_unlock(&transport->session_mutex);
  if (flush_result < 0 || pthread_create(&transport->read_thread, nullptr,
                                         h2_read_main, transport) != 0) {
    conn->http2 = nullptr;
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }
  return 0;
}

} // namespace

const char *lupine_wire_identity(void) { return LUPINE_WIRE_IDENTITY; }

bool lupine_wire_identity_compatible(const char *local, const char *peer) {
  return h2_wire_identity_compatible(local == nullptr ? "" : local,
                                     peer == nullptr ? "" : peer);
}

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
                           std::vector<rpc_write_cursor> &cursors) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  if (std::all_of(cursors.begin(), cursors.end(),
                  [](const rpc_write_cursor &cursor) {
                    return cursor.remaining() == 0;
                  })) {
    return 0;
  }

  h2_write_source source;
  source.cursors = &cursors;
  pthread_mutex_lock(&transport->session_mutex);
  int result = h2_send_source_locked(transport, stream_id, &source);
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int rpc_http2_write(conn_t *conn, std::vector<rpc_write_cursor> &cursors) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  return rpc_http2_write_stream(conn, transport->dispatch_stream_id, cursors);
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
      h2_nv(kContentEncodingHeader, kGzipEncoding),
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
  h2_write_source source;
  source.finish_stream = true;
  return h2_send_source_locked(transport, stream_id, &source);
}

} // namespace

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
  std::string peer_identity = transport->peer_wire_identity;
  pthread_mutex_unlock(&transport->session_mutex);

  // Check the build before the arena verdict: a mismatch is fatal, so reporting
  // it as an arena conflict would send the caller off retrying other slots.
  if (!h2_wire_identity_compatible(LUPINE_WIRE_IDENTITY, peer_identity)) {
    LUPINE_LOG_ERROR("LUPINE server was built from "
                     << (peer_identity.empty() ? "an unstated revision"
                                               : peer_identity.c_str())
                     << ", this client from " << LUPINE_WIRE_IDENTITY
                     << "; rebuild both from the same tree");
    return LUPINE_RPC_HTTP2_IDENTITY_MISMATCH;
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

// Valid until the transport is destroyed. A server built without git advertises
// nothing, so an empty result means "unstated", not "mismatched".
const char *rpc_http2_peer_wire_identity(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return "";
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  pthread_mutex_lock(&transport->session_mutex);
  const char *identity = transport->peer_wire_identity.c_str();
  pthread_mutex_unlock(&transport->session_mutex);
  return identity;
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
  while (!transport->request_received && !transport->transport_failed) {
    pthread_cond_wait(&transport->session_progress, &transport->session_mutex);
  }
  int result = -1;
  if (transport->request_received) {
    result = transport->request_handled ? 1 : 0;
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
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
  if (transport->session != nullptr) {
    nghttp2_session_del(transport->session);
    transport->session = nullptr;
  }
  for (auto &[stream_id, stream] : transport->streams) {
    (void)stream_id;
    h2_release_codecs(stream);
  }
  pthread_cond_destroy(&transport->heartbeat_progress);
  pthread_cond_destroy(&transport->session_progress);
  pthread_mutex_destroy(&transport->session_mutex);
  delete transport;
}
