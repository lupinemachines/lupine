#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct h2_pair {
  conn_t client = {};
  conn_t server = {};

  ~h2_pair() {
    if (client.connfd >= 0) {
      close(client.connfd);
    }
    if (server.connfd >= 0) {
      close(server.connfd);
    }
    rpc_write_queue_free(&client);
    rpc_write_queue_free(&server);
    rpc_connection_state_free(&client);
    rpc_connection_state_free(&server);
  }
};

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

h2_pair make_pair() {
  int fds[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");

  h2_pair pair;
  pair.client.connfd = fds[0];
  pair.server.connfd = fds[1];
  require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  return pair;
}

void init_rpc_read(conn_t *conn) {
  require(pthread_mutex_init(&conn->read_mutex, nullptr) == 0,
          "read mutex init failed");
  require(pthread_cond_init(&conn->read_cond, nullptr) == 0,
          "read cond init failed");
  require(rpc_connection_state_init(conn) == 0, "rpc state init failed");
}

void init_rpc_write(conn_t *conn) {
  require(pthread_mutex_init(&conn->write_mutex, nullptr) == 0,
          "write mutex init failed");
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
    rpc_frame frame;
    require(rpc_read_wire_frame(&pair.server, &frame) == 0,
            "large queue frame read failed");
    require(frame.request_id == 17, "large queue request id mismatch");
    require(frame.op == -1, "large queue response op mismatch");
    require(rpc_deliver_response_frame(&pair.server, std::move(frame)) == 0,
            "large queue deliver failed");
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
  require(pair.client.write_queue_count == kCount + 1,
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
  std::thread reader([&] {
    rpc_frame frame;
    require(rpc_read_wire_frame(&pair.server, &frame) == 0,
            "lz4 payload frame read failed");
    require(frame.request_id == 23, "lz4 payload request id mismatch");
    require(frame.op == -1, "lz4 payload response op mismatch");
    require(rpc_deliver_response_frame(&pair.server, std::move(frame)) == 0,
            "lz4 payload deliver failed");
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
}

} // namespace

int main() {
  test_rpc_write_queue_grows();
  test_rpc_lz4_payload_round_trip();
  test_client_to_server();
  test_server_to_client_after_request_headers();
  test_fragmented_iovec();
  test_large_payload();
  test_framed_payload_round_trip();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
