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

void write_all(conn_t *conn, const std::vector<std::string> &chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const std::string &chunk : chunks) {
    iov.push_back({const_cast<char *>(chunk.data()), chunk.size()});
  }
  require(rpc_http2_writev(conn, iov.data(), nullptr,
                           static_cast<int>(iov.size())) == 0,
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

  struct iovec iov[3] = {
      {const_cast<char *>(prefix.data()), prefix.size()},
      {payload.data(), payload.size()},
      {const_cast<char *>(suffix.data()), suffix.size()},
  };
  unsigned char framed[3] = {0, 1, 0};
  require(rpc_http2_writev(&pair.client, iov, framed, 3) == 0,
          "framed write failed");
  reader.join();
  require(received_prefix == prefix, "framed prefix mismatch");
  require(received == payload, "framed payload mismatch");
  require(received_suffix == suffix, "framed suffix mismatch");
}

} // namespace

int main() {
  test_client_to_server();
  test_server_to_client_after_request_headers();
  test_fragmented_iovec();
  test_large_payload();
  test_framed_payload_round_trip();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
