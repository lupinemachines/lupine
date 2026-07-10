#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <climits>
#include <deque>
#include <errno.h>
#include <lz4.h>
#include <nghttp2/nghttp2.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <sstream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kH2InitialWindow = 0x7fffffffU;
constexpr uint32_t kH2MaxFrame = (16 * 1024 * 1024) - 1;
constexpr size_t kH2FrameHeaderLen = 9;

struct h2_buffer {
  std::vector<unsigned char> data;
  size_t offset = 0;
};

struct h2_transport {
  lupine_socket_t netfd = LUPINE_INVALID_SOCKET;
  void *tls = nullptr; // Borrowed SSL* (owned by conn_t).
  bool server = false;
  bool response_sent = false;
  bool compress_lz4 = false;
  int32_t stream_id = 1;
  int response_status = 0;
  nghttp2_session *session = nullptr;
  std::deque<h2_buffer> local_out;
  // Reusable scratch holding the one LZ4-framed payload block currently in
  // flight (see h2_materialize_block). Writes are serialized per connection,
  // so a single buffer suffices and memory stays bounded by one block.
  std::vector<unsigned char> compress_scratch;
  pthread_mutex_t session_mutex = PTHREAD_MUTEX_INITIALIZER;
};

// h2_write_cursor either points at caller bytes that are sent verbatim
// (base/len) or, for LZ4-framed payloads, at uncompressed source bytes
// (src/src_len) that are compressed into the transport scratch one block at
// a time as nghttp2 pulls data. For a framed cursor, base/len cover only the
// currently materialized block.
struct h2_write_cursor {
  const unsigned char *base = nullptr;
  size_t len = 0;
  const char *src = nullptr;
  size_t src_len = 0;
  // For a framed cursor whose source is already compressed (e.g. a fatbin
  // with compressed members), skip the per-block LZ4 attempt and emit every
  // block with the raw fallback token. Same wire format, no wasted CPU.
};

struct h2_write_source {
  std::vector<h2_write_cursor> cursors;
  size_t index = 0;
  size_t pending_len = 0;

  size_t remaining() const {
    size_t total = 0;
    for (size_t i = index; i < cursors.size(); ++i) {
      total += cursors[i].len + cursors[i].src_len;
    }
    return total;
  }
};

void queue_bytes(std::deque<h2_buffer> &queue, const unsigned char *data,
                 size_t len) {
  if (len == 0) {
    return;
  }
  h2_buffer buffer;
  buffer.data.assign(data, data + len);
  queue.push_back(std::move(buffer));
}

// Maximum buffers per vectored send. Frames carry far fewer iovecs than this
// in practice, but cap defensively so a single sendmsg never exceeds the
// platform's IOV_MAX and fails with EMSGSIZE.
constexpr int kH2MaxSendIov = 512;

#ifdef LUPINE_TLS_OPENSSL
// Generated RPCs queue many 4- and 8-byte fields. Packing consecutive small
// fields into one bounded TLS record avoids an SSL_write per field while large
// payload spans continue to pass directly to OpenSSL.
constexpr size_t kH2TlsCoalesceCapacity = 4 * 1024;
constexpr size_t kH2TlsCoalesceFragmentMax = 256;

struct h2_tls_batch {
  int count = 0;
  size_t size = 0;
};

h2_tls_batch h2_plan_tls_batch(const struct iovec *iov, int iov_count) {
  h2_tls_batch batch;
  while (batch.count < iov_count) {
    size_t size = iov[batch.count].iov_len;
    if (size > kH2TlsCoalesceFragmentMax ||
        size > kH2TlsCoalesceCapacity - batch.size) {
      break;
    }
    batch.size += size;
    ++batch.count;
  }
  if (batch.count < 2) {
    return {};
  }
  return batch;
}
#endif

int h2_write_all(h2_transport *transport, const struct iovec *iov,
                 int iov_count) {
  std::vector<struct iovec> local(iov, iov + iov_count);
  struct iovec *cursor = local.data();
  int count = iov_count;
#ifdef LUPINE_TLS_OPENSSL
  std::array<unsigned char, kH2TlsCoalesceCapacity> tls_scratch;
#endif
  while (count > 0) {
    ssize_t n;
#ifdef LUPINE_TLS_OPENSSL
    if (transport->tls != nullptr) {
      SSL *ssl = static_cast<SSL *>(transport->tls);
      const void *data = cursor[0].iov_base;
      size_t size = cursor[0].iov_len;
      const h2_tls_batch batch = h2_plan_tls_batch(cursor, count);
      if (batch.count != 0) {
        size_t offset = 0;
        for (int i = 0; i < batch.count; ++i) {
          memcpy(tls_scratch.data() + offset, cursor[i].iov_base,
                 cursor[i].iov_len);
          offset += cursor[i].iov_len;
        }
        data = tls_scratch.data();
        size = batch.size;
      }
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
      // Send all currently pending buffers in one syscall instead of one send()
      // per buffer. Coalescing avoids emitting the 9-byte HTTP/2 frame header
      // as its own TCP segment and cuts syscall overhead on every frame.
      int batch = std::min(count, kH2MaxSendIov);
      n = lupine_socket_sendv(transport->netfd, cursor, batch);
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
    size_t written = static_cast<size_t>(n);
    while (count > 0 && written >= cursor[0].iov_len) {
      written -= cursor[0].iov_len;
      ++cursor;
      --count;
    }
    if (count > 0 && written != 0) {
      cursor[0].iov_base = static_cast<char *>(cursor[0].iov_base) + written;
      cursor[0].iov_len -= written;
    }
  }
  return 0;
}

ssize_t h2_send_callback(nghttp2_session *, const uint8_t *data, size_t length,
                         int, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  struct iovec iov = {const_cast<uint8_t *>(data), length};
  return h2_write_all(transport, &iov, 1) == 0 ? static_cast<ssize_t>(length)
                                               : NGHTTP2_ERR_CALLBACK_FAILURE;
}

// h2_materialize_block compresses the next payload block of a framed cursor
// into the transport's reusable scratch buffer as [uint32 token][bytes]
// (token == 0 means the block is stored raw; see compress.cpp for the wire
// format). Compressing lazily, one block per call, keeps memory bounded and
// lets early blocks reach the wire while later blocks are still being
// compressed. Only the cursor at write_source->index is ever materialized,
// and only after its previous block has been fully sent, so a single scratch
// buffer per connection is safe.
void h2_materialize_block(h2_transport *transport, h2_write_cursor &cursor) {
  size_t raw = std::min<size_t>(LUPINE_COMPRESS_BLOCK_BYTES, cursor.src_len);
  size_t bound =
      static_cast<size_t>(LZ4_compressBound(LUPINE_COMPRESS_BLOCK_BYTES));
  if (transport->compress_scratch.size() < sizeof(uint32_t) + bound) {
    transport->compress_scratch.resize(sizeof(uint32_t) + bound);
  }
  unsigned char *out = transport->compress_scratch.data();
  int compressed = LZ4_compress_default(
      cursor.src, reinterpret_cast<char *>(out + sizeof(uint32_t)),
      static_cast<int>(raw), static_cast<int>(bound));
  uint32_t token = 0;
  size_t block_len = raw;
  if (compressed > 0 && static_cast<size_t>(compressed) < raw) {
    token = static_cast<uint32_t>(compressed);
    block_len = static_cast<size_t>(compressed);
  } else {
    memcpy(out + sizeof(uint32_t), cursor.src, raw);
  }
  memcpy(out, &token, sizeof(token));
  cursor.base = out;
  cursor.len = sizeof(uint32_t) + block_len;
  cursor.src += raw;
  cursor.src_len -= raw;
}

ssize_t h2_data_source_read_callback(nghttp2_session *, int32_t, uint8_t *,
                                     size_t length, uint32_t *data_flags,
                                     nghttp2_data_source *source,
                                     void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  auto *write_source = static_cast<h2_write_source *>(source->ptr);
  auto &cursors = write_source->cursors;
  while (write_source->index < cursors.size() &&
         cursors[write_source->index].len == 0 &&
         cursors[write_source->index].src_len == 0) {
    ++write_source->index;
  }
  if (write_source->index == cursors.size()) {
    *data_flags |= NGHTTP2_DATA_FLAG_EOF | NGHTTP2_DATA_FLAG_NO_END_STREAM;
    write_source->pending_len = 0;
    return 0;
  }
  if (cursors[write_source->index].len == 0) {
    h2_materialize_block(transport, cursors[write_source->index]);
  }
  // Offer only materialized bytes; a framed cursor with unconsumed source
  // stops the scan because its next block does not exist yet.
  size_t available = 0;
  bool lazy_pending = false;
  for (size_t i = write_source->index; i < cursors.size(); ++i) {
    available += cursors[i].len;
    if (cursors[i].src_len > 0) {
      lazy_pending = true;
      break;
    }
  }
  size_t chunk = std::min(available, length);
  *data_flags |= NGHTTP2_DATA_FLAG_NO_COPY;
  if (chunk == available && !lazy_pending) {
    *data_flags |= NGHTTP2_DATA_FLAG_EOF | NGHTTP2_DATA_FLAG_NO_END_STREAM;
  }
  write_source->pending_len = chunk;
  return static_cast<ssize_t>(chunk);
}

ssize_t h2_data_source_read_length_callback(nghttp2_session *, uint8_t, int32_t,
                                            int32_t session_remote_window_size,
                                            int32_t stream_remote_window_size,
                                            uint32_t remote_max_frame_size,
                                            void *) {
  int32_t window =
      std::min(session_remote_window_size, stream_remote_window_size);
  if (window <= 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  size_t max_len = std::min<size_t>(kH2MaxFrame, remote_max_frame_size);
  max_len = std::min<size_t>(max_len, static_cast<size_t>(window));
  return static_cast<ssize_t>(std::max<size_t>(1, max_len));
}

int h2_send_data_callback(nghttp2_session *, nghttp2_frame *frame,
                          const uint8_t *framehd, size_t length,
                          nghttp2_data_source *source, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  auto *write_source = static_cast<h2_write_source *>(source->ptr);
  if (length != write_source->pending_len) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  std::vector<struct iovec> iov;
  iov.reserve(write_source->cursors.size() - write_source->index + 3);
  iov.push_back({const_cast<uint8_t *>(framehd), kH2FrameHeaderLen});

  unsigned char padlen = 0;
  if (frame->data.padlen > 0) {
    padlen = static_cast<unsigned char>(frame->data.padlen - 1);
    iov.push_back({&padlen, 1});
  }

  size_t remaining = length;
  size_t cursor_index = write_source->index;
  while (remaining > 0 && cursor_index < write_source->cursors.size()) {
    auto &cursor = write_source->cursors[cursor_index];
    size_t chunk = std::min(remaining, cursor.len);
    iov.push_back({const_cast<unsigned char *>(cursor.base), chunk});
    remaining -= chunk;
    ++cursor_index;
  }
  if (remaining != 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  unsigned char padding[256] = {};
  if (frame->data.padlen > 1) {
    iov.push_back({padding, frame->data.padlen - 1});
  }

  if (h2_write_all(transport, iov.data(), static_cast<int>(iov.size())) < 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  remaining = length;
  while (remaining > 0) {
    auto &cursor = write_source->cursors[write_source->index];
    size_t chunk = std::min(remaining, cursor.len);
    cursor.base += chunk;
    cursor.len -= chunk;
    remaining -= chunk;
    if (cursor.len == 0) {
      if (cursor.src_len != 0) {
        // Framed cursor: the sent block is done but more source remains; the
        // next block is materialized by the next read callback.
        break;
      }
      ++write_source->index;
    }
  }
  write_source->pending_len = 0;
  return 0;
}

int h2_on_data_chunk_recv_callback(nghttp2_session *, uint8_t, int32_t,
                                   const uint8_t *data, size_t len,
                                   void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  queue_bytes(transport->local_out, data, len);
  return 0;
}

nghttp2_nv h2_nv(const char *name, const char *value) {
  return {reinterpret_cast<uint8_t *>(const_cast<char *>(name)),
          reinterpret_cast<uint8_t *>(const_cast<char *>(value)), strlen(name),
          strlen(value),
          NGHTTP2_NV_FLAG_NO_COPY_NAME | NGHTTP2_NV_FLAG_NO_COPY_VALUE};
}

constexpr char kLupineCompressHeader[] = "x-lupine-compress";
constexpr char kLupineCompressLz4[] = "lz4";

bool lupine_h2_debug_enabled() {
  const char *debug = getenv("LUPINE_DEBUG");
  if (debug != nullptr && debug[0] != '\0' && strcmp(debug, "0") != 0) {
    return true;
  }
  return lupine_trace_stream() != nullptr;
}

int h2_submit_server_response(h2_transport *transport) {
  nghttp2_nv headers[] = {h2_nv(":status", "200")};
  if (nghttp2_submit_headers(transport->session, NGHTTP2_FLAG_NONE,
                             transport->stream_id, nullptr, headers, 1,
                             nullptr) != 0) {
    return -1;
  }
  transport->response_sent = true;
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
    transport->stream_id = frame->hd.stream_id;
    if (!transport->response_sent && h2_submit_server_response(transport) < 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
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
  if (transport->server) {
    if (frame->headers.cat == NGHTTP2_HCAT_REQUEST &&
        namelen == strlen(kLupineCompressHeader) &&
        memcmp(name, kLupineCompressHeader, namelen) == 0 &&
        valuelen == strlen(kLupineCompressLz4) &&
        memcmp(value, kLupineCompressLz4, valuelen) == 0) {
      transport->compress_lz4 = true;
    }
    return 0;
  }
  if (frame->headers.cat != NGHTTP2_HCAT_RESPONSE) {
    return 0;
  }
  if (namelen != 7 || memcmp(name, ":status", 7) != 0) {
    return 0;
  }

  transport->response_status = 0;
  for (size_t i = 0; i < valuelen; ++i) {
    if (value[i] < '0' || value[i] > '9') {
      return 0;
    }
    transport->response_status =
        transport->response_status * 10 + static_cast<int>(value[i] - '0');
  }
  return 0;
}

int h2_flush_session_locked(h2_transport *transport) {
  int result = nghttp2_session_send(transport->session);
  return result == 0 ? 0 : -1;
}

int h2_flush_session(h2_transport *transport) {
  pthread_mutex_lock(&transport->session_mutex);
  int result = h2_flush_session_locked(transport);
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int h2_read_from_net(h2_transport *transport) {
  unsigned char buffer[64 * 1024];
  ssize_t n = 0;
#ifdef LUPINE_TLS_OPENSSL
  if (transport->tls != nullptr) {
    SSL *ssl = static_cast<SSL *>(transport->tls);
    for (;;) {
      int r = SSL_read(ssl, buffer, sizeof(buffer));
      if (r > 0) {
        n = r;
        break;
      }
      int err = SSL_get_error(ssl, r);
      if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
        continue;
      }
      return -1;
    }
  } else
#endif
  {
    do {
      n = lupine_socket_recv(transport->netfd, buffer, sizeof(buffer));
    } while (n < 0 && lupine_socket_error_is_intr());
  }
  if (n <= 0) {
    return -1;
  }

  size_t offset = 0;
  pthread_mutex_lock(&transport->session_mutex);
  while (offset < static_cast<size_t>(n)) {
    ssize_t consumed = nghttp2_session_mem_recv(
        transport->session, buffer + offset, static_cast<size_t>(n) - offset);
    if (consumed <= 0) {
      pthread_mutex_unlock(&transport->session_mutex);
      return -1;
    }
    offset += static_cast<size_t>(consumed);
  }
  int result = h2_flush_session_locked(transport);
  pthread_mutex_unlock(&transport->session_mutex);
  return result;
}

int h2_init_direct(conn_t *conn, bool server) {
  auto *transport = new h2_transport();
  transport->netfd = conn->connfd;
  transport->tls = conn->tls_session;
  transport->server = server;

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
  nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                   h2_on_header_callback);

  int session_result = server ? nghttp2_session_server_new(&transport->session,
                                                           callbacks, transport)
                              : nghttp2_session_client_new(
                                    &transport->session, callbacks, transport);
  nghttp2_session_callbacks_del(callbacks);
  if (session_result != 0) {
    delete transport;
    return -1;
  }

  nghttp2_settings_entry settings[] = {
      {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, kH2InitialWindow},
      {NGHTTP2_SETTINGS_MAX_FRAME_SIZE, kH2MaxFrame},
  };
  if (nghttp2_submit_settings(transport->session, NGHTTP2_FLAG_NONE, settings,
                              2) != 0 ||
      nghttp2_session_set_local_window_size(
          transport->session, NGHTTP2_FLAG_NONE, 0,
          static_cast<int32_t>(kH2InitialWindow)) != 0) {
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }

  if (!server) {
    nghttp2_nv headers[] = {
        h2_nv(":method", "POST"),
        h2_nv(":scheme", "http"),
        h2_nv(":path", "/"),
        h2_nv(":authority", "lupine"),
        h2_nv(kLupineCompressHeader, kLupineCompressLz4),
    };
    transport->compress_lz4 = true;
    size_t header_count = 5;
    int32_t stream_id =
        nghttp2_submit_headers(transport->session, NGHTTP2_FLAG_NONE, -1,
                               nullptr, headers, header_count, nullptr);
    if (stream_id < 0) {
      nghttp2_session_del(transport->session);
      delete transport;
      return -1;
    }
    transport->stream_id = stream_id;
  }

  conn->http2 = transport;
  if (h2_flush_session(transport) < 0) {
    conn->http2 = nullptr;
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }
  return 0;
}

} // namespace

int rpc_http2_read(conn_t *conn, void *data, size_t size) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  auto *out = static_cast<unsigned char *>(data);
  size_t copied = 0;
  while (copied < size) {
    while (!transport->local_out.empty() && copied < size) {
      h2_buffer &front = transport->local_out.front();
      size_t available = front.data.size() - front.offset;
      size_t chunk = std::min(available, size - copied);
      memcpy(out + copied, front.data.data() + front.offset, chunk);
      front.offset += chunk;
      copied += chunk;
      if (front.offset == front.data.size()) {
        transport->local_out.pop_front();
      }
    }
    if (copied == size) {
      return static_cast<int>(size);
    }
    if ((transport->response_status != 0 &&
         transport->response_status != 200) ||
        h2_read_from_net(transport) < 0) {
      return -1;
    }
  }
  return static_cast<int>(size);
}

int rpc_http2_writev(conn_t *conn, const rpc_write_entry *entries,
                     int entry_count) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  h2_write_source source;
  source.cursors.reserve(entry_count);
  for (int i = 0; i < entry_count; ++i) {
    const rpc_write_entry &entry = entries[i];
    if (entry.iov.iov_len == 0) {
      continue;
    }
    h2_write_cursor cursor;
    if (entry.framed) {
      cursor.src = static_cast<const char *>(entry.iov.iov_base);
      cursor.src_len = entry.iov.iov_len;
    } else {
      cursor.base = static_cast<const unsigned char *>(entry.iov.iov_base);
      cursor.len = entry.iov.iov_len;
    }
    source.cursors.push_back(cursor);
  }
  if (source.cursors.empty()) {
    return 0;
  }

  nghttp2_data_provider provider = {};
  provider.source.ptr = &source;
  provider.read_callback = h2_data_source_read_callback;
  pthread_mutex_lock(&transport->session_mutex);
  if (nghttp2_submit_data(transport->session, NGHTTP2_FLAG_NONE,
                          transport->stream_id, &provider) != 0 ||
      h2_flush_session_locked(transport) < 0 || source.remaining() != 0) {
    pthread_mutex_unlock(&transport->session_mutex);
    return -1;
  }
  pthread_mutex_unlock(&transport->session_mutex);
  return 0;
}

int rpc_http2_compress_lz4(conn_t *conn) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  return transport != nullptr && transport->compress_lz4 ? 1 : 0;
}

int rpc_http2_client_init(conn_t *conn) { return h2_init_direct(conn, false); }

int rpc_http2_server_init(conn_t *conn) { return h2_init_direct(conn, true); }

void rpc_http2_destroy(conn_t *conn) {
  if (conn == nullptr || conn->http2 == nullptr) {
    return;
  }
  auto *transport = static_cast<h2_transport *>(conn->http2);
  conn->http2 = nullptr;
  if (transport->session != nullptr) {
    nghttp2_session_del(transport->session);
    transport->session = nullptr;
  }
  pthread_mutex_destroy(&transport->session_mutex);
  delete transport;
}
