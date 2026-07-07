// Robustness regression test: a malformed peer sends cuModuleLoad with
// image_size = SIZE_MAX. Before the fix the server computed
// std::vector<unsigned char>(image_size + 1) which wraps to 0, then attempted
// to read SIZE_MAX payload bytes and hung the connection forever. After the
// fix the server rejects the oversized request and closes the connection.
//
// This binary links the lupine RPC transport directly and speaks the wire
// protocol itself; it does not need a GPU, only a running lupine_driver_server
// reachable via LUPINE_SERVER (host:port, same env the client shim uses).
#include "codegen/gen_api.h"
#include "rpc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>

// Defined in rpc.cpp (C++ linkage): reads response ids and wakes readers.
extern void *_rpc_read_id_dispatch(void *p);

static bool ParseServer(std::string *host, std::string *port) {
  const char *env = getenv("LUPINE_SERVER");
  std::string server =
      (env != nullptr && env[0] != '\0') ? env : "127.0.0.1:14833";
  auto colon = server.rfind(':');
  if (colon == std::string::npos) {
    return false;
  }
  *host = server.substr(0, colon);
  *port = server.substr(colon + 1);
  return !host->empty() && !port->empty();
}

static int Connect(const std::string &host, const std::string &port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
    return -1;
  }
  int fd = -1;
  for (auto *p = res; p != nullptr; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
      break;
    }
    close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  return fd;
}

static void OnAlarm(int) {
  std::cerr
      << "RESULT: HANG (server did not close within timeout -> vulnerable)"
      << std::endl;
  _exit(1);
}

int main() {
  std::string host;
  std::string port;
  if (!ParseServer(&host, &port)) {
    std::cerr << "invalid LUPINE_SERVER" << std::endl;
    return 2;
  }

  int fd = Connect(host, port);
  if (fd < 0) {
    std::cerr << "connect to " << host << ":" << port << " failed" << std::endl;
    return 2;
  }

  conn_t conn{};
  conn.connfd = fd;
  conn.request_id = 0;
  conn.local_request_parity = 0;
  pthread_mutex_init(&conn.read_mutex, nullptr);
  pthread_mutex_init(&conn.write_mutex, nullptr);
  pthread_mutex_init(&conn.call_mutex, nullptr);
  pthread_cond_init(&conn.read_cond, nullptr);
  if (rpc_http2_client_init(&conn) < 0) {
    std::cerr << "http/2 init failed" << std::endl;
    return 2;
  }
  pthread_t disp;
  if (pthread_create(&disp, nullptr, _rpc_read_id_dispatch, &conn) != 0) {
    std::cerr << "dispatch thread spawn failed" << std::endl;
    return 2;
  }
  conn.rpc_thread = disp;

  if (rpc_write_start_request(&conn, RPC_cuModuleLoad) < 0) {
    std::cerr << "write_start_request failed" << std::endl;
    return 2;
  }
  size_t image_size = SIZE_MAX; // malicious: image_size + 1 wraps to 0
  if (rpc_write(&conn, &image_size, sizeof(image_size)) < 0) {
    std::cerr << "write failed" << std::endl;
    return 2;
  }
  int wid = rpc_write_end(&conn);
  if (wid < 0) {
    std::cerr << "write_end failed" << std::endl;
    return 2;
  }

  // The server must reject the request and close the connection promptly.
  // If it instead hangs (the pre-fix behavior), SIGALRM fires after 5s.
  signal(SIGALRM, OnAlarm);
  alarm(5);
  int rc = rpc_read_start(&conn, wid);
  alarm(0);

  if (rc < 0) {
    std::cout << "RESULT: PASS (server rejected malformed request)"
              << std::endl;
    return 0;
  }
  std::cerr << "RESULT: UNEXPECTED response, rc=" << rc << std::endl;
  return 3;
}
