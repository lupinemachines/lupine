#include <cuda.h>

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

bool expect(CUresult actual, CUresult expected, const char *what) {
  if (actual == expected) {
    std::printf("PASS: %s returned %d\n", what, static_cast<int>(actual));
    return true;
  }
  std::fprintf(stderr, "FAIL: %s returned %d, expected %d\n", what,
               static_cast<int>(actual), static_cast<int>(expected));
  return false;
}

} // namespace

int main() {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) {
    std::perror("socket");
    return 1;
  }

  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  socklen_t address_size = sizeof(address);
  if (bind(listener, reinterpret_cast<sockaddr *>(&address), address_size) !=
          0 ||
      listen(listener, 1) != 0 ||
      getsockname(listener, reinterpret_cast<sockaddr *>(&address),
                  &address_size) != 0) {
    std::perror("loopback listener");
    close(listener);
    return 1;
  }

  std::thread peer([listener] {
    int connection = accept(listener, nullptr, nullptr);
    if (connection >= 0) {
      char request[64];
      (void)recv(connection, request, sizeof(request), 0);
      close(connection);
    }
    close(listener);
  });

  char server[64];
  std::snprintf(server, sizeof(server), "127.0.0.1:%u",
                static_cast<unsigned int>(ntohs(address.sin_port)));
  setenv("LUPINE_SERVER", server, 1);
  CUresult transport_failure = cuInit(0);
  peer.join();

  unsetenv("LUPINE_SERVER");
  CUresult no_backend = cuInit(0);

  bool ok = expect(transport_failure, CUDA_ERROR_DEVICE_UNAVAILABLE,
                   "cuInit after transport failure");
  ok &= expect(no_backend, CUDA_ERROR_NO_DEVICE,
               "cuInit without a configured backend");
  return ok ? 0 : 1;
}
