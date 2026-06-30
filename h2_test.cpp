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

void init_conn(conn_t *conn, lupine_socket_t fd, int request_id) {
  *conn = {};
  conn->connfd = fd;
  conn->request_id = request_id;
  conn->local_request_parity = conn->request_id & 1;
  require(pthread_mutex_init(&conn->read_mutex, nullptr) == 0,
          "read mutex init failed");
  require(pthread_mutex_init(&conn->write_mutex, nullptr) == 0,
          "write mutex init failed");
  require(pthread_mutex_init(&conn->call_mutex, nullptr) == 0,
          "call mutex init failed");
  require(pthread_cond_init(&conn->read_cond, nullptr) == 0,
          "read cond init failed");
}

h2_pair make_pair() {
  int fds[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");

  h2_pair pair;
  init_conn(&pair.client, fds[0], 0);
  init_conn(&pair.server, fds[1], 1);
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

void test_truncated_rpc_response_releases_locks() {
  h2_pair pair = make_pair();
  constexpr int op = 12345;
  int client_dispatch_result = 0;

  std::thread client_dispatch([&] {
    while (!pair.client.closed) {
      int dispatch_op = rpc_dispatch(&pair.client, 1);
      if (dispatch_op < 0 || pair.client.closed) {
        break;
      }
      if (rpc_read_end(&pair.client) < 0) {
        client_dispatch_result = -1;
        break;
      }
    }
  });

  std::thread server([&] {
    int dispatch_op = rpc_dispatch(&pair.server, 0);
    require(dispatch_op == op, "server received wrong rpc op");
    int request_id = rpc_read_end(&pair.server);
    require(request_id > 0, "server request read_end failed");

    char partial_response = 0x5a;
    require(rpc_write_start_response(&pair.server, request_id) == 0,
            "server response start failed");
    require(rpc_write(&pair.server, &partial_response,
                      sizeof(partial_response)) == 0,
            "server truncated response write failed");
    require(rpc_write_end(&pair.server) == request_id,
            "server truncated response flush failed");
    shutdown(pair.server.connfd, SHUT_RDWR);
    close(pair.server.connfd);
    pair.server.connfd = -1;
  });

  require(rpc_write_start_request(&pair.client, op) == 0,
          "client request start failed");
  require(rpc_wait_for_response(&pair.client) == 0,
          "client did not receive response header");

  int response = 0;
  require(rpc_read(&pair.client, &response, sizeof(response)) < 0,
          "truncated response read unexpectedly succeeded");
  require(pair.client.closed, "client connection was not marked closed");
  require(pthread_mutex_trylock(&pair.client.call_mutex) == 0,
          "call mutex left locked after truncated response");
  pthread_mutex_unlock(&pair.client.call_mutex);
  require(rpc_write_start_request(&pair.client, op + 1) < 0,
          "new rpc request did not fail on closed connection");

  server.join();
  client_dispatch.join();
  if (pair.client.rpc_thread != 0) {
    pthread_join(pair.client.rpc_thread, nullptr);
    pair.client.rpc_thread = 0;
  }
  require(pthread_mutex_trylock(&pair.client.read_mutex) == 0,
          "read mutex left locked after truncated response");
  pthread_mutex_unlock(&pair.client.read_mutex);
  require(client_dispatch_result == 0, "client dispatch failed unexpectedly");
}

} // namespace

int main() {
  test_client_to_server();
  test_server_to_client_after_request_headers();
  test_fragmented_iovec();
  test_large_payload();
  test_framed_payload_round_trip();
  test_truncated_rpc_response_releases_locks();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
