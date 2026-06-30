#include "lupine_platform.h"

#include <arpa/inet.h>
#include <cstdlib>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << "\n";
    std::exit(1);
  }
}

int get_int_sockopt(int fd, int level, int optname) {
  int value = 0;
  socklen_t len = sizeof(value);
  require(getsockopt(fd, level, optname, &value, &len) == 0,
          "getsockopt failed");
  return value;
}

} // namespace

int main() {
  int server = socket(AF_INET, SOCK_STREAM, 0);
  require(server >= 0, "server socket failed");
  require(lupine_socket_set_reuseaddr(server) == 0, "reuseaddr failed");

  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  require(bind(server, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0,
          "bind failed");
  require(listen(server, 1) == 0, "listen failed");

  socklen_t addr_len = sizeof(addr);
  require(getsockname(server, reinterpret_cast<sockaddr *>(&addr), &addr_len) ==
              0,
          "getsockname failed");

  int client = socket(AF_INET, SOCK_STREAM, 0);
  require(client >= 0, "client socket failed");
  require(lupine_socket_configure_stream(client) == 0,
          "socket configure failed");
  require(connect(client, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ==
              0,
          "connect failed");

  int accepted = accept(server, nullptr, nullptr);
  require(accepted >= 0, "accept failed");
  require(lupine_socket_configure_stream(accepted) == 0,
          "accepted socket configure failed");

  require(get_int_sockopt(client, IPPROTO_TCP, TCP_NODELAY) != 0,
          "TCP_NODELAY was not enabled");
  require(get_int_sockopt(client, SOL_SOCKET, SO_KEEPALIVE) != 0,
          "SO_KEEPALIVE was not enabled");
  require(get_int_sockopt(accepted, IPPROTO_TCP, TCP_NODELAY) != 0,
          "accepted TCP_NODELAY was not enabled");
  require(get_int_sockopt(accepted, SOL_SOCKET, SO_KEEPALIVE) != 0,
          "accepted SO_KEEPALIVE was not enabled");

  close(accepted);
  close(client);
  close(server);
  std::cout << "socket_tuning_test: PASS\n";
  return 0;
}
