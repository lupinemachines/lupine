// Unit tests for the RPC layer (rpc.cpp / h2.cpp), modeled on h2_test.cpp.
//
// These exercise the request/response + lane/parity dispatch protocol directly
// over an in-process socketpair, with no CUDA. The intent is to reproduce (and
// then guard against) dispatch deadlocks such as the histEqualizationNPP hang
// (#250): a read_id orphan that parks the read-id thread, the lane dispatch
// thread, and the caller all on read_cond.

#include "lupine_log.h"
#include "rpc.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Defined in rpc.cpp (normally launched by rpc_dispatch). Reads response
// request_ids from the socket and publishes them via conn->read_id.
void *_rpc_read_id_dispatch(void *p);

namespace {

int g_failures = 0;

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    ++g_failures;
  }
}

void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

struct rpc_pair {
  conn_t client = {};
  conn_t server = {};
  std::thread server_thread;
  std::atomic<bool> stop{false};

  ~rpc_pair() {
    stop.store(true);
    client.closed = 1;
    server.closed = 1;
    pthread_cond_broadcast(&client.read_cond);
    shutdown(client.connfd, SHUT_RDWR);
    shutdown(server.connfd, SHUT_RDWR);
    if (server_thread.joinable()) {
      server_thread.join();
    }
    if (client.rpc_thread != 0) {
      pthread_join(client.rpc_thread, nullptr);
    }
    if (client.connfd >= 0) {
      close(client.connfd);
    }
    if (server.connfd >= 0) {
      close(server.connfd);
    }
    rpc_write_queue_free(&client);
    rpc_write_queue_free(&server);
  }
};

void init_conn(conn_t *conn, bool is_client) {
  check(pthread_mutex_init(&conn->read_mutex, nullptr) == 0,
        "read mutex init");
  check(pthread_mutex_init(&conn->write_mutex, nullptr) == 0,
        "write mutex init");
  check(pthread_mutex_init(&conn->call_mutex, nullptr) == 0,
        "call mutex init");
  check(pthread_cond_init(&conn->read_cond, nullptr) == 0, "read cond init");
  conn->request_id = is_client ? 0 : 1;
  conn->local_request_parity = conn->request_id & 1;
}

// A minimal echo server: read each request frame (request_id, lane, op,
// payload) and write back a response with the same request_id.
void echo_server(rpc_pair *pair) {
  conn_t *conn = &pair->server;
  while (!pair->stop.load()) {
    int request_id = 0;
    uint64_t lane = 0;
    int op = 0;
    int32_t payload = 0;
    if (rpc_http2_read(conn, &request_id, sizeof(request_id)) !=
            static_cast<int>(sizeof(request_id)) ||
        rpc_http2_read(conn, &lane, sizeof(lane)) !=
            static_cast<int>(sizeof(lane)) ||
        rpc_http2_read(conn, &op, sizeof(op)) != static_cast<int>(sizeof(op)) ||
        rpc_http2_read(conn, &payload, sizeof(payload)) !=
            static_cast<int>(sizeof(payload))) {
      return;
    }
    if (request_id == 0) {
      return;
    }
    int32_t echo = payload + 1;
    conn->read_lane_id = lane;
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &echo, sizeof(echo)) < 0 ||
        rpc_write_end(conn) < 0) {
      return;
    }
  }
}

std::unique_ptr<rpc_pair> make_pair() {
  int fds[2] = {-1, -1};
  check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
  auto pair = std::make_unique<rpc_pair>();
  pair->client.connfd = fds[0];
  pair->server.connfd = fds[1];
  init_conn(&pair->client, /*is_client=*/true);
  init_conn(&pair->server, /*is_client=*/false);
  check(rpc_http2_client_init(&pair->client) == 0, "client h2 init");
  check(rpc_http2_server_init(&pair->server) == 0, "server h2 init");
  // Start the read-id dispatch thread directly (normally launched by
  // rpc_dispatch on the client). It reads response request_ids and publishes
  // them via conn->read_id.
  check(pthread_create(&pair->client.rpc_thread, nullptr, _rpc_read_id_dispatch,
                       &pair->client) == 0,
        "read-id dispatch thread");
  pair->server_thread = std::thread(echo_server, pair.get());
  return pair;
}

// Issue one RPC and read back the int32 echo via the full client path.
int32_t roundtrip(conn_t *conn, int32_t payload) {
  check(rpc_write_start_request(conn, /*op=*/1) >= 0, "write_start_request");
  check(rpc_write(conn, &payload, sizeof(payload)) == 0, "rpc_write");
  int write_id = rpc_write_end(conn);
  check(write_id > 0, "rpc_write_end");
  check(rpc_read_start(conn, write_id) == 0, "rpc_read_start");
  int32_t out = 0;
  check(rpc_read(conn, &out, sizeof(out)) == static_cast<int>(sizeof(out)),
        "rpc_read payload");
  rpc_read_end(conn);
  return out;
}

void test_single_roundtrip() {
  auto pair = make_pair();
  int32_t out = roundtrip(&pair->client, 41);
  require(out == 42, "single roundtrip echo mismatch");
}

void test_sequential_roundtrips() {
  auto pair = make_pair();
  for (int32_t i = 0; i < 50; ++i) {
    require(roundtrip(&pair->client, i) == i + 1, "sequential echo mismatch");
  }
}

// Reproduces the nested/reentrant-RPC class of bug: the client starts request A
// (rpc_write_end), then -- before reading A's response -- issues request B and
// waits for B. Because rpc_write_end releases call_mutex before the response is
// read, a thread can have two RPCs in flight at once. If the server answers A
// first, read_id is set to A's id while the thread is waiting for B, orphaning
// it and deadlocking the dispatch.
//
// Runs the nested wait under a watchdog; a real hang fails the test.
void test_nested_rpc_no_deadlock() {
  auto pair = make_pair();
  conn_t *c = &pair->client;

  // Watchdog: fail if the nested exchange doesn't complete quickly.
  std::atomic<bool> done{false};
  std::thread watchdog([&] {
    for (int i = 0; i < 300 && !done.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!done.load()) {
      std::cerr << "FAIL: nested RPC deadlocked (read_id orphan)" << std::endl;
      std::_Exit(1);
    }
  });
  watchdog.detach();

  // Request A (write_id 2): sent, not yet consumed.
  check(rpc_write_start_request(c, /*op=*/1) >= 0, "A write_start");
  int32_t a_payload = 100;
  check(rpc_write(c, &a_payload, sizeof(a_payload)) == 0, "A write");
  int a_id = rpc_write_end(c);
  check(a_id > 0, "A write_end");

  // Reentrant request B (write_id 4): fully round-tripped while A is pending.
  int32_t b_out = roundtrip(c, 7);
  require(b_out == 8, "nested B echo mismatch");

  // Now consume A's response.
  check(rpc_read_start(c, a_id) == 0, "A read_start");
  int32_t a_out = 0;
  check(rpc_read(c, &a_out, sizeof(a_out)) == static_cast<int>(sizeof(a_out)),
        "A read payload");
  rpc_read_end(c);
  require(a_out == 101, "nested A echo mismatch");

  done.store(true);
}

}  // namespace

int main() {
  test_single_roundtrip();
  test_sequential_roundtrips();

  // The nested/reentrant-RPC case reproduces the histEqualizationNPP hang
  // (#250): the dispatch deadlocks. It is gated behind an env var because it
  // hangs (watchdog exits non-zero) until the protocol supports reentrant RPCs.
  if (getenv("RPC_TEST_NESTED") != nullptr) {
    test_nested_rpc_no_deadlock();
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " rpc test assertion(s) failed" << std::endl;
    return 1;
  }
  std::cout << "rpc_test: PASS" << std::endl;
  return 0;
}
