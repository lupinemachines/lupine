#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <nghttp2/nghttp2.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern void *_rpc_read_id_dispatch(void *);

namespace {

struct h2_pair {
  conn_t client = {};
  conn_t server = {};

  h2_pair() {
    client.connfd = -1;
    server.connfd = -1;
  }

  ~h2_pair() {
    rpc_conn_destroy(&client);
    rpc_conn_destroy(&server);
    if (client.connfd >= 0) {
      close(client.connfd);
    }
    if (server.connfd >= 0) {
      close(server.connfd);
    }
  }
};

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

void init_rpc_read(conn_t *conn);
void init_rpc_write(conn_t *conn);
void exchange_settings(h2_pair *pair);

h2_pair make_pair() {
  int fds[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");

  h2_pair pair;
  pair.client.connfd = fds[0];
  pair.server.connfd = fds[1];
  init_rpc_read(&pair.client);
  init_rpc_write(&pair.client);
  require(pthread_mutex_init(&pair.client.call_mutex, nullptr) == 0,
          "client call mutex init failed");
  init_rpc_read(&pair.server);
  init_rpc_write(&pair.server);
  require(pthread_mutex_init(&pair.server.call_mutex, nullptr) == 0,
          "server call mutex init failed");
  require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  return pair;
}

void init_rpc_read(conn_t *conn) {
  require(pthread_mutex_init(&conn->read_mutex, nullptr) == 0,
          "read mutex init failed");
  require(pthread_cond_init(&conn->read_cond, nullptr) == 0,
          "read cond init failed");
}

void init_rpc_write(conn_t *conn) {
  require(pthread_mutex_init(&conn->write_mutex, nullptr) == 0,
          "write mutex init failed");
}

void read_rpc_prefix(conn_t *conn) {
  require(pthread_mutex_lock(&conn->read_mutex) == 0,
          "prefix read mutex lock failed");
  require(rpc_read(conn, &conn->read_id, sizeof(conn->read_id)) ==
              static_cast<int>(sizeof(conn->read_id)),
          "prefix request id read failed");
  require(pthread_cond_broadcast(&conn->read_cond) == 0,
          "prefix cond broadcast failed");
  require(pthread_mutex_unlock(&conn->read_mutex) == 0,
          "prefix read mutex unlock failed");
}

void write_all(conn_t *conn, const std::vector<std::string> &chunks) {
  std::vector<rpc_write_entry> entries;
  entries.reserve(chunks.size());
  for (const std::string &chunk : chunks) {
    entries.push_back({{const_cast<char *>(chunk.data()), chunk.size()}, 0});
  }
  require(rpc_http2_writev(conn, entries.data(),
                           static_cast<int>(entries.size())) == 0,
          "h2 write failed");
}

std::string read_string(conn_t *conn, size_t size) {
  std::string output(size, '\0');
  require(rpc_http2_read(conn, output.data(), output.size()) ==
              static_cast<int>(output.size()),
          "h2 read failed");
  return output;
}

rpc_http2_read_stats read_stats(conn_t *conn) {
  rpc_http2_read_stats stats = {};
  require(rpc_http2_get_read_stats(conn, &stats) == 0, "read stats failed");
  return stats;
}

bool raw_write_all(lupine_socket_t socket, const unsigned char *data,
                   size_t size) {
  while (size != 0) {
    struct iovec iov = {const_cast<unsigned char *>(data), size};
    ssize_t written = lupine_socket_sendv(socket, &iov, 1);
    if (written < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool raw_read_exact(lupine_socket_t socket, unsigned char *data, size_t size) {
  while (size != 0) {
    ssize_t received = lupine_socket_recv(socket, data, size);
    if (received < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    data += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

void test_client_to_server() {
  h2_pair pair = make_pair();
  std::string message = "hello over h2";
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, message.size()); });
  write_all(&pair.client, {message});
  reader.join();
  require(received == message, "client-to-server payload mismatch");
}

void test_server_to_client_after_request_headers() {
  h2_pair pair = make_pair();
  std::string request = "request";
  std::string response = "response";
  std::string received_request;
  std::thread reader(
      [&] { received_request = read_string(&pair.server, request.size()); });
  write_all(&pair.client, {request});
  reader.join();
  require(received_request == request, "server did not receive request");

  write_all(&pair.server, {response});
  require(read_string(&pair.client, response.size()) == response,
          "server-to-client payload mismatch");
}

void test_fragmented_iovec() {
  h2_pair pair = make_pair();
  std::vector<std::string> chunks = {"alpha", "", ":", "beta", ":gamma"};
  std::string received;
  std::thread reader([&] { received = read_string(&pair.server, 16); });
  write_all(&pair.client, chunks);
  reader.join();
  require(received == "alpha:beta:gamma", "fragmented payload mismatch");
}

void test_fragmented_frames_direct() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  const rpc_http2_read_stats before = read_stats(&pair.server);
  const std::string expected = "fragmented-data";
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, expected.size()); });
  write_all(&pair.client, {"fragment"});
  write_all(&pair.client, {"ed"});
  write_all(&pair.client, {"-data"});
  reader.join();
  require(received == expected, "fragmented frame mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes - before.direct_bytes == received.size(),
          "fragmented frames were not read directly");
  require(after.staged_bytes == before.staged_bytes,
          "fragmented frames unexpectedly staged bytes");
}

void test_partial_read_stages_only_overflow() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  std::string payload(4096, '\0');
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i & 0x7f);
  }
  std::string received(payload.size(), '\0');
  const rpc_http2_read_stats before = read_stats(&pair.server);
  std::thread reader([&] {
    require(rpc_http2_read(&pair.server, received.data(), 7) == 7,
            "partial prefix read failed");
    require(rpc_http2_read(&pair.server, received.data() + 7,
                           received.size() - 7) ==
                static_cast<int>(received.size() - 7),
            "partial suffix read failed");
  });
  write_all(&pair.client, {payload});
  reader.join();
  require(received == payload, "partial read payload mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes - before.direct_bytes == 7,
          "partial read direct byte count mismatch");
  require(after.staged_bytes - before.staged_bytes == payload.size() - 7,
          "partial read staged byte count mismatch");
  require(after.staged_read_bytes - before.staged_read_bytes ==
              payload.size() - 7,
          "partial read staged-copy count mismatch");
  require(after.staged_buffers - before.staged_buffers == 1,
          "partial read staging allocation count mismatch");
  require(after.peak_staged_bytes >= payload.size() - 7,
          "partial read peak staging mismatch");
}

void test_truncated_read_clears_direct_destination() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  std::vector<unsigned char> guarded(48, 0xa5);
  const std::string prefix = "truncated";
  int read_result = 0;
  std::thread reader([&] {
    read_result = rpc_http2_read(&pair.server, guarded.data() + 8, 32);
  });
  write_all(&pair.client, {prefix});
  require(shutdown(pair.client.connfd, SHUT_WR) == 0,
          "truncated writer shutdown failed");
  reader.join();
  require(read_result == -1, "truncated read unexpectedly succeeded");
  require(std::memcmp(guarded.data() + 8, prefix.data(), prefix.size()) == 0,
          "truncated read lost received prefix");
  for (size_t i = 0; i < guarded.size(); ++i) {
    if (i >= 8 && i < 8 + prefix.size()) {
      continue;
    }
    require(guarded[i] == 0xa5, "truncated read wrote outside prefix");
  }
}

void test_concurrent_response_lanes() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  constexpr int kFirstId = 200;
  constexpr int kSecondId = 202;
  std::vector<unsigned char> first(1024 * 1024);
  std::vector<unsigned char> second(1024 * 1024);
  for (size_t i = 0; i < first.size(); ++i) {
    first[i] = static_cast<unsigned char>(i & 0xff);
    second[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
  }
  std::vector<unsigned char> received_first(first.size());
  std::vector<unsigned char> received_second(second.size());

  pthread_t dispatcher = {};
  require(pthread_create(&dispatcher, nullptr, _rpc_read_id_dispatch,
                         &pair.client) == 0,
          "response dispatcher start failed");
  pair.client.rpc_thread = dispatcher;
  int first_result = -1;
  int second_result = -1;
  std::thread first_reader([&] {
    if (rpc_read_start(&pair.client, kFirstId) == 0 &&
        rpc_read(&pair.client, received_first.data(), received_first.size()) ==
            static_cast<int>(received_first.size())) {
      first_result = rpc_read_end(&pair.client);
    }
  });
  std::thread second_reader([&] {
    if (rpc_read_start(&pair.client, kSecondId) == 0 &&
        rpc_read(&pair.client, received_second.data(),
                 received_second.size()) ==
            static_cast<int>(received_second.size())) {
      second_result = rpc_read_end(&pair.client);
    }
  });

  const rpc_http2_read_stats before = read_stats(&pair.client);
  pair.server.read_lane_id = 2;
  require(rpc_write_start_response(&pair.server, kSecondId) == 0,
          "second response start failed");
  require(rpc_write(&pair.server, second.data(), second.size()) == 0,
          "second response payload failed");
  require(rpc_write_end(&pair.server) == kSecondId,
          "second response end failed");
  pair.server.read_lane_id = 1;
  require(rpc_write_start_response(&pair.server, kFirstId) == 0,
          "first response start failed");
  require(rpc_write(&pair.server, first.data(), first.size()) == 0,
          "first response payload failed");
  require(rpc_write_end(&pair.server) == kFirstId, "first response end failed");

  first_reader.join();
  second_reader.join();
  require(first_result == kFirstId && second_result == kSecondId,
          "concurrent response id mismatch");
  require(received_first == first && received_second == second,
          "concurrent response payload mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.client);
  require(after.direct_bytes > before.direct_bytes,
          "concurrent responses did not use direct receive");
  require(after.peak_staged_bytes <= 64 * 1024,
          "concurrent response read-ahead was not bounded");

  shutdown(pair.client.connfd, SHUT_RDWR);
  pthread_join(dispatcher, nullptr);
  pair.client.rpc_thread = 0;
}

void exchange_settings(h2_pair *pair) {
  std::string request = "x";
  std::string response = "y";
  std::string received_request;
  std::thread reader(
      [&] { received_request = read_string(&pair->server, request.size()); });
  write_all(&pair->client, {request});
  reader.join();
  require(received_request == request, "settings exchange request mismatch");

  write_all(&pair->server, {response});
  require(read_string(&pair->client, response.size()) == response,
          "settings exchange response mismatch");
}

void test_large_payload() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  std::string payload(2 * 1024 * 1024, '\0');
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>('a' + (i % 26));
  }

  size_t midpoint = payload.size() / 2;
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, payload.size()); });
  write_all(&pair.client,
            {payload.substr(0, midpoint), payload.substr(midpoint)});
  reader.join();
  require(received == payload, "large payload mismatch");
}

void test_payload_larger_than_flow_control_window() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  constexpr size_t payload_size =
      static_cast<size_t>(INT32_MAX) + 64 * 1024 + 1;

  void *payload = mmap(nullptr, payload_size, PROT_READ,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  require(payload != MAP_FAILED, "flow-control payload mmap failed");

  std::atomic<bool> read_failed{false};
  size_t received = 0;
  std::thread server_reader([&] {
    std::array<unsigned char, 64 * 1024> buffer = {};
    while (received < payload_size) {
      size_t chunk = std::min(buffer.size(), payload_size - received);
      if (rpc_http2_read(&pair.server, buffer.data(), chunk) !=
          static_cast<int>(chunk)) {
        read_failed = true;
        break;
      }
      if (!std::all_of(buffer.begin(), buffer.begin() + chunk,
                       [](unsigned char value) { return value == 0; })) {
        read_failed = true;
        break;
      }
      received += chunk;
    }
  });

  // Production connections always have an RPC dispatch thread reading control
  // frames. It does not receive application bytes here, but processing the
  // peer's WINDOW_UPDATE frames is what lets a large write make progress.
  std::thread client_control_reader([&] {
    unsigned char unused = 0;
    (void)rpc_http2_read(&pair.client, &unused, sizeof(unused));
  });

  rpc_write_entry entry = {{payload, payload_size}, 0};
  int write_result = rpc_http2_writev(&pair.client, &entry, 1);
  if (write_result != 0) {
    shutdown(pair.client.connfd, SHUT_RDWR);
    shutdown(pair.server.connfd, SHUT_RDWR);
  }
  server_reader.join();
  shutdown(pair.client.connfd, SHUT_RDWR);
  client_control_reader.join();
  munmap(payload, payload_size);

  require(write_result == 0, "flow-controlled write failed before completion");
  require(!read_failed, "flow-controlled read failed");
  require(received == payload_size, "flow-controlled payload was truncated");
}

void test_reset_wakes_flow_controlled_writer() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  // Flush the client's acknowledgement of the server's initial SETTINGS and
  // consume it through the server session before switching this test to raw
  // HTTP/2 control frames.
  write_all(&pair.client, {"m"});
  require(read_string(&pair.server, 1) == "m",
          "failed to drain initial SETTINGS acknowledgement");

  std::thread client_control_reader([&] {
    unsigned char unused = 0;
    (void)rpc_http2_read(&pair.client, &unused, sizeof(unused));
  });

  // Shrink the peer's stream window so the writer pauses after 64 KiB rather
  // than requiring another multi-gigabyte test payload.
  std::array<unsigned char, 15> settings = {};
  settings[2] = 6;
  settings[3] = NGHTTP2_SETTINGS;
  settings[10] = NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE;
  settings[13] = settings[14] = 0xff;
  require(raw_write_all(pair.server.connfd, settings.data(), settings.size()),
          "failed to send reduced-window SETTINGS");
  std::array<unsigned char, 9> settings_ack = {};
  require(raw_read_exact(pair.server.connfd, settings_ack.data(),
                         settings_ack.size()),
          "failed to read SETTINGS acknowledgement");
  require(settings_ack[0] == 0 && settings_ack[1] == 0 &&
              settings_ack[2] == 0 && settings_ack[3] == NGHTTP2_SETTINGS &&
              (settings_ack[4] & NGHTTP2_FLAG_ACK) != 0,
          "invalid SETTINGS acknowledgement");

  std::atomic<bool> reset_failed{false};
  std::thread server_reset([&] {
    std::array<unsigned char, 64 * 1024> buffer = {};
    size_t received = 0;
    bool reset_sent = false;
    for (;;) {
      ssize_t chunk =
          lupine_socket_recv(pair.server.connfd, buffer.data(), buffer.size());
      if (chunk < 0 && lupine_socket_error_is_intr()) {
        continue;
      }
      if (chunk <= 0) {
        break;
      }
      received += static_cast<size_t>(chunk);
      if (!reset_sent && received >= 32 * 1024) {
        std::array<unsigned char, 13> reset = {};
        reset[2] = 4;
        reset[3] = NGHTTP2_RST_STREAM;
        reset[8] = 1;
        reset[12] = NGHTTP2_CANCEL;
        reset_sent =
            raw_write_all(pair.server.connfd, reset.data(), reset.size());
        if (!reset_sent) {
          reset_failed = true;
          break;
        }
      }
    }
    if (!reset_sent) {
      reset_failed = true;
    }
  });

  std::string payload(128 * 1024, 'x');
  rpc_write_entry entry = {{payload.data(), payload.size()}, 0};
  int write_result = rpc_http2_writev(&pair.client, &entry, 1);
  shutdown(pair.client.connfd, SHUT_RDWR);
  shutdown(pair.server.connfd, SHUT_RDWR);
  server_reset.join();
  client_control_reader.join();

  require(write_result < 0,
          "reset flow-controlled write unexpectedly succeeded");
  require(!reset_failed, "failed to deliver RST_STREAM");
}

// Round-trips a multi-block LZ4-framed payload: the transport compresses it
// lazily block by block (h2.cpp) and rpc_read_payload_part decodes it with
// chunked, block-aligned reads (compress.cpp). The payload mixes
// compressible and random data so both compressed and raw block tokens are
// exercised, and plain iovecs surround the framed one as in a real message.
void test_framed_payload_round_trip() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  std::string prefix = "head";
  std::string suffix = "tail";
  std::vector<char> payload(2 * LUPINE_COMPRESS_BLOCK_BYTES + 123457);
  unsigned int seed = 42;
  for (size_t i = 0; i < payload.size() / 2; ++i) {
    payload[i] = static_cast<char>(i % 7);
  }
  for (size_t i = payload.size() / 2; i < payload.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    payload[i] = static_cast<char>(seed >> 24);
  }

  std::string received_prefix;
  std::string received_suffix;
  std::vector<char> received(payload.size());
  std::thread reader([&] {
    received_prefix = read_string(&pair.server, prefix.size());
    size_t first = LUPINE_COMPRESS_BLOCK_BYTES;
    require(rpc_read_payload_part(&pair.server, 1, received.data(), first) ==
                static_cast<int>(first),
            "framed read part 1 failed");
    require(rpc_read_payload_part(&pair.server, 1, received.data() + first,
                                  received.size() - first) ==
                static_cast<int>(received.size() - first),
            "framed read part 2 failed");
    received_suffix = read_string(&pair.server, suffix.size());
  });

  rpc_write_entry entries[3] = {
      {{const_cast<char *>(prefix.data()), prefix.size()}, 0},
      {{payload.data(), payload.size()}, 1},
      {{const_cast<char *>(suffix.data()), suffix.size()}, 0},
  };
  require(rpc_http2_writev(&pair.client, entries, 3) == 0,
          "framed write failed");
  reader.join();
  require(received_prefix == prefix, "framed prefix mismatch");
  require(received == payload, "framed payload mismatch");
  require(received_suffix == suffix, "framed suffix mismatch");
}

void test_rpc_write_queue_grows() {
  int value = 7;

  conn_t zero_length = {};
  require(rpc_write(&zero_length, &value, 0) == 0,
          "zero-length rpc_write failed");
  require(zero_length.write_queue_count == 1,
          "zero-length rpc_write did not consume one queue entry");
  require(zero_length.write_queue[0].iov.iov_len == 0,
          "zero-length rpc_write changed length");
  rpc_write_queue_free(&zero_length);

  h2_pair pair = make_pair();
  init_rpc_write(&pair.client);
  init_rpc_read(&pair.server);

  constexpr int kCount = 300;
  std::vector<int> values(kCount);
  for (int i = 0; i < kCount; ++i) {
    values[i] = i;
  }

  std::vector<int> received(kCount, -1);
  std::thread reader([&] {
    read_rpc_prefix(&pair.server);
    require(pair.server.read_id == 17, "large queue request id mismatch");
    require(rpc_read_start(&pair.server, 17) == 0,
            "large queue read start failed");
    for (int i = 0; i < kCount; ++i) {
      require(rpc_read(&pair.server, &received[i], sizeof(received[i])) ==
                  static_cast<int>(sizeof(received[i])),
              "large queue payload read failed");
    }
    require(rpc_read_end(&pair.server) == 17, "large queue read_end failed");
  });

  require(rpc_write_start_response(&pair.client, 17) == 0,
          "large queue response start failed");
  for (int i = 0; i < kCount; ++i) {
    require(rpc_write(&pair.client, &values[i], sizeof(values[i])) == 0,
            "large queue rpc_write failed");
  }
  require(pair.client.write_queue_count == kCount + 3,
          "large queue count mismatch");
  require(rpc_write_end(&pair.client) == 17, "large queue write_end failed");
  reader.join();
  require(received == values, "large queue payload mismatch");
}

void test_rpc_lz4_payload_round_trip() {
  h2_pair pair = make_pair();
  init_rpc_write(&pair.client);
  init_rpc_read(&pair.server);

  std::string prefix = "before";
  std::string suffix = "after";
  std::vector<char> payload(LUPINE_COMPRESS_BLOCK_BYTES + 128 * 1024);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i % 13);
  }

  std::string received_prefix(prefix.size(), '\0');
  std::string received_suffix(suffix.size(), '\0');
  std::vector<char> received(payload.size());
  const rpc_http2_read_stats before = read_stats(&pair.server);
  std::thread reader([&] {
    read_rpc_prefix(&pair.server);
    require(pair.server.read_id == 23, "lz4 payload request id mismatch");
    require(rpc_read_start(&pair.server, 23) == 0,
            "lz4 payload read start failed");
    require(rpc_read(&pair.server, received_prefix.data(),
                     received_prefix.size()) ==
                static_cast<int>(received_prefix.size()),
            "lz4 payload prefix read failed");
    require(rpc_read_payload(&pair.server, received.data(), received.size()) ==
                static_cast<int>(received.size()),
            "lz4 payload payload read failed");
    require(rpc_read(&pair.server, received_suffix.data(),
                     received_suffix.size()) ==
                static_cast<int>(received_suffix.size()),
            "lz4 payload suffix read failed");
    require(rpc_read_end(&pair.server) == 23, "lz4 payload read_end failed");
  });

  require(rpc_write_start_response(&pair.client, 23) == 0,
          "lz4 payload response start failed");
  require(rpc_write(&pair.client, prefix.data(), prefix.size()) == 0,
          "lz4 payload prefix write failed");
  require(rpc_write_payload(&pair.client, payload.data(), payload.size()) == 0,
          "lz4 payload payload write failed");
  require(rpc_write(&pair.client, suffix.data(), suffix.size()) == 0,
          "lz4 payload suffix write failed");
  require(rpc_write_end(&pair.client) == 23, "lz4 payload write_end failed");
  reader.join();

  require(received_prefix == prefix, "lz4 payload prefix mismatch");
  require(received == payload, "lz4 payload payload mismatch");
  require(received_suffix == suffix, "lz4 payload suffix mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes > before.direct_bytes,
          "lz4 payload did not use direct receive");
}

} // namespace

int main() {
  test_rpc_write_queue_grows();
  test_rpc_lz4_payload_round_trip();
  test_client_to_server();
  test_server_to_client_after_request_headers();
  test_fragmented_iovec();
  test_fragmented_frames_direct();
  test_partial_read_stages_only_overflow();
  test_truncated_read_clears_direct_destination();
  test_concurrent_response_lanes();
  test_large_payload();
  test_framed_payload_round_trip();
  test_payload_larger_than_flow_control_window();
  test_reset_wakes_flow_controlled_writer();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
