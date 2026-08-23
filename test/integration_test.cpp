#include "rpc.h"

#include <cstdlib>
#include <iostream>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

void test_address_space();

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

void test_unknown_cuda_version_metadata() {
  int sockets[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0,
          "metadata test socketpair failed");

  conn_t client = {};
  conn_t server = {};
  require(rpc_conn_init(&client, sockets[0], 0) == 0 &&
              rpc_conn_init(&server, sockets[1], 1) == 0,
          "metadata test RPC initialization failed");

  const char *cuda_version = nullptr;
  std::thread probe([&] { cuda_version = rpc_http2_client_probe(&client); });
  int server_result = rpc_http2_server_init(&server);
  probe.join();

  require(server_result == 1, "HEAD / was not handled as a metadata request");
  require(cuda_version == nullptr, "HEAD / advertised an unknown CUDA version");
  rpc_conn_destroy(&client);
  rpc_conn_destroy(&server);
  close(sockets[0]);
  close(sockets[1]);
}

} // namespace

int main() {
  test_address_space();
  test_unknown_cuda_version_metadata();
  std::cout << "integration_test: PASS" << std::endl;
  return 0;
}
