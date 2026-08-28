// Transport coverage that runs on every platform the client ships for. The
// existing CPU tests are Unix-only because they pair sockets with socketpair(),
// so nothing exercised the Windows dial path until a server was in front of it:
// the client never started Winsock, and every connect failed at runtime while
// CI stayed green. Everything here goes over loopback TCP for that reason.
#include "lupine_log.h"
#include "rpc.h"

#include <chrono>
#include <future>
#include <stdio.h>
#include <string>
#include <thread>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  printf("%-5s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) {
    ++failures;
  }
}

// A listening loopback socket plus the port it landed on. Port 0 lets the OS
// pick, so concurrent CI jobs never collide.
struct listener {
  lupine_socket_t fd = LUPINE_INVALID_SOCKET;
  std::string port;
};

bool open_listener(listener *out) {
  if (lupine_socket_init() < 0) {
    return false;
  }
  lupine_socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == LUPINE_INVALID_SOCKET) {
    return false;
  }
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  socklen_t length = sizeof(address);
  if (bind(fd, reinterpret_cast<sockaddr *>(&address), length) != 0 ||
      listen(fd, 4) != 0 ||
      getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
    lupine_socket_close(fd);
    return false;
  }
  out->fd = fd;
  out->port = std::to_string(ntohs(address.sin_port));
  return true;
}

// The regression this file exists for: on Windows nothing had started Winsock
// by the time a client dialed, so the connect failed however healthy the peer.
void test_client_dial_starts_the_socket_layer() {
  listener server;
  if (!open_listener(&server)) {
    check(false, "loopback listener");
    return;
  }
  lupine_socket_t dialed =
      lupine_tcp_connect("127.0.0.1", server.port.c_str(), 0);
  check(dialed != LUPINE_INVALID_SOCKET,
        "lupine_tcp_connect reaches a loopback listener");
  // Only accept once the dial succeeded. Accepting unconditionally would block
  // forever on exactly the failure this test exists to report.
  if (dialed != LUPINE_INVALID_SOCKET) {
    lupine_socket_t accepted = accept(server.fd, nullptr, nullptr);
    check(accepted != LUPINE_INVALID_SOCKET,
          "listener accepts the dialed socket");
    if (accepted != LUPINE_INVALID_SOCKET) {
      lupine_socket_close(accepted);
    }
    lupine_socket_close(dialed);
  }
  lupine_socket_close(server.fd);
}

// A peer that hangs up mid-session must not leave the other side stuck in
// teardown. Bounded so a regression fails the run instead of hanging CI.
void test_teardown_completes_after_an_abrupt_peer_close() {
  listener server;
  if (!open_listener(&server)) {
    check(false, "loopback listener");
    return;
  }
  lupine_socket_t dialed =
      lupine_tcp_connect("127.0.0.1", server.port.c_str(), 0);
  if (dialed == LUPINE_INVALID_SOCKET) {
    check(false, "connected pair for teardown");
    lupine_socket_close(server.fd);
    return;
  }
  lupine_socket_t accepted = accept(server.fd, nullptr, nullptr);
  lupine_socket_close(server.fd);
  if (accepted == LUPINE_INVALID_SOCKET) {
    check(false, "connected pair for teardown");
    lupine_socket_close(dialed);
    return;
  }

  conn_t peer = {};
  if (rpc_conn_init(&peer, accepted, 1) < 0) {
    check(false, "peer conn init");
    return;
  }
  // Drop the dialing side without a graceful shutdown, the way a client that
  // refuses its peer mid-handshake does.
  lupine_socket_close(dialed);

  auto done = std::async(std::launch::async, [&peer] {
    rpc_conn_destroy(&peer);
    return true;
  });
  check(done.wait_for(std::chrono::seconds(20)) == std::future_status::ready,
        "conn teardown finishes after the peer hangs up");
  if (done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    (void)done.get();
  } else {
    // Leaked on purpose: joining a stuck teardown would hang the run.
    done.wait_for(std::chrono::seconds(0));
    std::exit(1);
  }
}

} // namespace

int main() {
  test_client_dial_starts_the_socket_layer();
  test_teardown_completes_after_an_abrupt_peer_close();
  printf("\ntransport_test: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
