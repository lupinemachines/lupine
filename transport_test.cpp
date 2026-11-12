// Transport coverage that runs on every platform the client ships for. The
// existing CPU tests are Unix-only because they pair sockets with socketpair(),
// so nothing exercised the Windows dial path until a server was in front of it:
// the client never started Winsock, and every connect failed at runtime while
// CI stayed green. Everything here goes over loopback TCP for that reason.
#include "lupine_log.h"
#include "process_handoff.h"
#include "rpc.h"
#include "test_platform.h"
#include "transport.h"

#include <atomic>
#include <chrono>
#include <future>
#include <stdio.h>
#include <string>
#include <thread>
#include <utility>

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

bool open_listener(listener *out, int backlog = 4) {
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
      listen(fd, backlog) != 0 ||
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

#ifdef __linux__
// A server host that vanishes after ordering a checkpoint reconnect leaves the
// client redialing a peer that never answers its SYN. Process exit closes the
// transport, which must interrupt that dial instead of waiting out the kernel's
// SYN retransmissions (about two minutes on Linux). Linux drops SYNs to a
// listener whose accept queue is full, which stands in for the vanished host.
void test_close_interrupts_a_checkpoint_redial() {
  listener server;
  if (!open_listener(&server, 0)) {
    check(false, "loopback listener");
    return;
  }
  setenv("LUPINE_SESSION", "lease", 1);
  const std::string checkpoint = "lease.1";
  lupine_client_transport_config config;
  config.dial_policy = lupine_client_dial_policy::single_attempt;
  std::string servers = "127.0.0.1:" + server.port;
  auto opened = std::async(std::launch::async, [&] {
    return lupine_client_transport_open(config, servers.c_str());
  });

  conn_t peer = {};
  lupine_socket_t accepted = accept(server.fd, nullptr, nullptr);
  bool ready = accepted != LUPINE_INVALID_SOCKET &&
               rpc_conn_init(&peer, accepted, 1) == 0 &&
               rpc_http2_server_init_with_metadata(&peer, nullptr) == 0 &&
               opened.get() == 0;
  char command;
  std::string received;
  int32_t stream = -1;
  int drained = 0;
  ready = ready && lupine_handoff_send_control(&peer, 'P', checkpoint) == 0 &&
          (stream = rpc_http2_accept_stream(&peer)) >= 0 &&
          rpc_bind_http2_stream(&peer, stream) == 0 &&
          rpc_dispatch(&peer, 0) == LUPINE_HANDOFF_DRAIN_REQUEST;
  int request = ready ? rpc_read_end(&peer) : -1;
  // The client then reports the drain to its supervisor, which a server does
  // not accept. Waiting for that refusal leaves this end's socket idle, so the
  // supervisor's remaining controls can be written to it directly.
  ready = request >= 0 && rpc_write_start_response(&peer, request) >= 0 &&
          rpc_write(&peer, &drained, sizeof(drained)) >= 0 &&
          rpc_write_end(&peer) >= 0 &&
          lupine_handoff_receive_control(&peer, &command, &received) < 0;
  // Fill the accept queue so the redial's SYN goes unanswered.
  lupine_socket_t filler = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<uint16_t>(std::stoi(server.port)));
  ready = ready && connect(filler, reinterpret_cast<sockaddr *>(&address),
                           sizeof(address)) == 0;
  for (char control : {'F', 'C'}) {
    std::string frame(9, '\0');
    frame[2] = static_cast<char>(checkpoint.size() + 1);
    frame[3] = static_cast<char>(LUPINE_HANDOFF_CONTROL_FRAME);
    frame += control;
    frame += checkpoint;
    ready = ready && send(accepted, frame.data(), frame.size(), 0) ==
                         static_cast<ssize_t>(frame.size());
  }
  check(ready, "checkpoint handoff reaches the reconnect");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  auto closed = std::async(std::launch::async, [] {
    lupine_client_transport_close();
    return true;
  });
  check(closed.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
        "closing the client interrupts a checkpoint redial");
  if (closed.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    // Leaked on purpose: joining a stuck close would hang the run.
    std::exit(1);
  }
  rpc_conn_destroy(&peer);
  lupine_socket_close(filler);
  lupine_socket_close(server.fd);
  unsetenv("LUPINE_SESSION");
}
#endif

struct mock_server {
  listener sock;
  rpc_http2_server_metadata meta = {};
  conn_t peer = {}, bulk[LUPINE_BULK_CONNECTIONS_MAX] = {};
  unsigned int nbulk = 0;
  bool inited = false;
  std::atomic<bool> accepted{false}, stop{false};
  std::thread worker;

  explicit mock_server(const char *tok = nullptr,
                       std::atomic<bool> *gate = nullptr) {
    meta.bulk_token = tok;
    if (!open_listener(&sock, 16))
      return;
    worker = std::thread([this, gate]() {
      lupine_socket_t fd = accept(sock.fd, nullptr, nullptr);
      if (fd == LUPINE_INVALID_SOCKET || rpc_conn_init(&peer, fd, 1) != 0)
        return;
      inited = accepted = true;
      while (gate && !*gate)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (rpc_http2_server_init_with_metadata(&peer, &meta) != 0 ||
          !meta.bulk_token)
        return;
      while (!stop && nbulk < LUPINE_BULK_CONNECTIONS_MAX) {
        if (!lupine_test_wait_readable(sock.fd, 10))
          continue;
        lupine_socket_t bfd = accept(sock.fd, nullptr, nullptr);
        char pre[LUPINE_BULK_PREAMBLE_BYTES];
        size_t n = 0;
        while (bfd != LUPINE_INVALID_SOCKET && n < sizeof(pre)) {
          ssize_t r = lupine_socket_recv(bfd, pre + n, sizeof(pre) - n);
          if (r <= 0)
            break;
          n += static_cast<size_t>(r);
        }
        if (bfd == LUPINE_INVALID_SOCKET)
          break;
        if (n != sizeof(pre))
          lupine_socket_close(bfd);
        else if (rpc_conn_init(&bulk[nbulk], bfd, 1) == 0)
          rpc_http2_server_init_with_metadata(&bulk[nbulk], &meta) == 0
              ? (void)++nbulk
              : rpc_conn_destroy(&bulk[nbulk]);
      }
    });
  }
  ~mock_server() {
    stop = true;
    if (worker.joinable())
      worker.join();
    while (nbulk > 0)
      rpc_conn_destroy(&bulk[--nbulk]);
    if (inited)
      rpc_conn_destroy(&peer);
    if (sock.fd != LUPINE_INVALID_SOCKET)
      lupine_socket_close(sock.fd);
  }
};

void test_fleet_publication_and_bulk_lanes() {
  lupine_client_transport_config cfg;
  cfg.dial_policy = lupine_client_dial_policy::single_attempt;
  cfg.strict_endpoints = true;
  auto dial = [&](const std::string &s) {
    return lupine_client_transport_open(cfg, s.c_str());
  };
  {
    std::atomic<bool> release{false};
    mock_server s0, s1(nullptr, &release);
    auto rc = std::async(std::launch::async, [&] {
      return dial("127.0.0.1:" + s0.sock.port + ",127.0.0.1:" + s1.sock.port);
    });
    while (!s1.accepted)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(lupine_client_transport_size() == 0 &&
              !lupine_client_transport_connection(0),
          "fleet size stays 0 while second endpoint handshake is gated");
    release = true;
    check(rc.get() == 0 && lupine_client_transport_size() == 2,
          "fleet size publishes atomically as 2 after all endpoints connect");
    lupine_client_transport_close();
  }
  listener dead;
  if (open_listener(&dead)) {
    lupine_socket_close(dead.fd);
    for (bool strict : {true, false}) {
      mock_server ok;
      cfg.strict_endpoints = strict;
      int rc = dial("127.0.0.1:" + ok.sock.port + ",127.0.0.1:" + dead.port);
      check(strict ? (rc < 0 && ok.accepted && !lupine_client_transport_size())
                   : (rc == 0 && lupine_client_transport_size() == 1),
            strict ? "strict open rolls back earlier connections on failure"
                   : "non-strict open retains healthy server when peer fails");
      lupine_client_transport_close();
    }
  }
  constexpr const char *kTok = "0123456789abcdef0123456789abcdef";
  constexpr unsigned int kMax = LUPINE_BULK_CONNECTIONS_MAX;
  cfg.strict_endpoints = true;
  bool env_ok = true;
  for (auto [env, want] :
       std::initializer_list<std::pair<const char *, unsigned int>>{
           {"-1", 0},
           {"0", 0},
           {"99", kMax},
           {"bogus", 0},
           {"999999999999999999999", kMax}}) {
    lupine_test_setenv("LUPINE_BULK_CONNECTIONS", env);
    mock_server srv(kTok);
    int rc = dial("127.0.0.1:" + srv.sock.port);
    auto *lanes = lupine_client_transport_bulk_lanes(
        lupine_client_transport_connection(0));
    env_ok =
        env_ok && rc == 0 && (!want ? !lanes : (lanes && lanes->count == want));
    lupine_client_transport_close();
  }
  check(env_ok,
        "LUPINE_BULK_CONNECTIONS parses negative, invalid, and overflow");

  lupine_test_setenv("LUPINE_BULK_CONNECTIONS", "2");
  mock_server srv(kTok);
  int rc = dial("127.0.0.1:" + srv.sock.port);
  conn_t *conn = lupine_client_transport_connection(0);
  std::thread reader([&] {
    for (int i = 0; i < 200; ++i)
      (void)lupine_client_transport_bulk_lanes(conn);
  });
  std::thread closer([&] { lupine_client_transport_shutdown_bulk(conn); });
  reader.join();
  closer.join();
  check(rc == 0 && !lupine_client_transport_bulk_lanes(conn),
        "concurrent bulk_lanes and shutdown_bulk settle cleanly");
  lupine_client_transport_close();
  lupine_test_unsetenv("LUPINE_BULK_CONNECTIONS");
}

} // namespace

int main() {
  test_client_dial_starts_the_socket_layer();
  test_teardown_completes_after_an_abrupt_peer_close();
#ifdef __linux__
  test_close_interrupts_a_checkpoint_redial();
#endif
  test_fleet_publication_and_bulk_lanes();
  printf("\ntransport_test: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
