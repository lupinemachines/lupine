#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int lupine_test_rpc_open();
extern "C" int lupine_test_rpc_size();

namespace {

constexpr int kEndpointCount = 17;
constexpr int kExpectedConnectionCount = 16;

int fail(const std::string &message) {
  std::cerr << message << std::endl;
  return 1;
}

int listen_on_loopback(uint16_t *port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  int reuse = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
      listen(fd, kEndpointCount) < 0) {
    close(fd);
    return -1;
  }

  socklen_t addr_len = sizeof(addr);
  if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addr_len) < 0) {
    close(fd);
    return -1;
  }

  *port = ntohs(addr.sin_port);
  return fd;
}

} // namespace

int main() {
  std::vector<int> listeners;
  listeners.reserve(kEndpointCount);

  std::ostringstream servers;
  for (int i = 0; i < kEndpointCount; ++i) {
    uint16_t port = 0;
    int fd = listen_on_loopback(&port);
    if (fd < 0) {
      return fail(std::string("failed to create listener: ") +
                  std::strerror(errno));
    }
    listeners.push_back(fd);
    if (i != 0) {
      servers << ',';
    }
    servers << "127.0.0.1:" << port;
  }

  std::string server_list = servers.str();
  if (setenv("LUPINE_SERVER", server_list.c_str(), 1) != 0) {
    return fail("failed to set LUPINE_SERVER");
  }

  int open_result = lupine_test_rpc_open();
  int connection_count = lupine_test_rpc_size();

  for (int fd : listeners) {
    close(fd);
  }

  if (open_result != 0) {
    return fail("rpc_open failed for local listeners");
  }
  if (connection_count != kExpectedConnectionCount) {
    return fail("rpc_open opened " + std::to_string(connection_count) +
                " connections; expected " +
                std::to_string(kExpectedConnectionCount));
  }

  return 0;
}
