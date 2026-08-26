#ifndef LUPINE_TEST_PLATFORM_H
#define LUPINE_TEST_PLATFORM_H

// Shared scaffolding for tests that run on every platform the client ships for.
// socketpair() is the usual way to get a connected pair, but it is a Unix call,
// so tests built on it could only ever run on Linux and macOS. Loopback TCP is
// the portable equivalent and is what the transport uses in production anyway.
#include "lupine_platform.h"

#include <stdint.h>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#endif

// Fills a connected pair. Both ends are ordinary stream sockets, so callers
// close them with lupine_socket_close. Returns false with both ends left
// invalid if any step fails.
static inline bool lupine_test_connected_pair(lupine_socket_t pair[2]) {
  pair[0] = LUPINE_INVALID_SOCKET;
  pair[1] = LUPINE_INVALID_SOCKET;
  if (lupine_socket_init() < 0) {
    return false;
  }

  lupine_socket_t listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener == LUPINE_INVALID_SOCKET) {
    return false;
  }
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0; // Let the OS pick, so parallel tests never collide.
  socklen_t length = sizeof(address);
  if (bind(listener, reinterpret_cast<sockaddr *>(&address), length) != 0 ||
      listen(listener, 1) != 0 ||
      getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length) !=
          0) {
    lupine_socket_close(listener);
    return false;
  }

  lupine_socket_t client = socket(AF_INET, SOCK_STREAM, 0);
  if (client == LUPINE_INVALID_SOCKET) {
    lupine_socket_close(listener);
    return false;
  }
  if (connect(client, reinterpret_cast<sockaddr *>(&address), length) != 0) {
    lupine_socket_close(client);
    lupine_socket_close(listener);
    return false;
  }
  lupine_socket_t server = accept(listener, nullptr, nullptr);
  lupine_socket_close(listener);
  if (server == LUPINE_INVALID_SOCKET) {
    lupine_socket_close(client);
    return false;
  }

  // Match a production connection: the transport sets these on every socket it
  // opens, and without TCP_NODELAY a loopback pair coalesces the small frames
  // these tests exchange, which a socketpair never did.
  lupine_socket_apply_transport_options(client);
  lupine_socket_apply_transport_options(server);

  pair[0] = client;
  pair[1] = server;
  return true;
}

// Waits for readability. poll() is Unix-only; Winsock spells the same thing
// WSAPoll with an identical struct layout.
static inline bool lupine_test_wait_readable(lupine_socket_t socket,
                                             int timeout_ms) {
#ifdef _WIN32
  WSAPOLLFD descriptor = {};
  descriptor.fd = socket;
  descriptor.events = POLLRDNORM;
  return WSAPoll(&descriptor, 1, timeout_ms) > 0;
#else
  pollfd descriptor = {};
  descriptor.fd = socket;
  descriptor.events = POLLIN;
  return poll(&descriptor, 1, timeout_ms) > 0;
#endif
}

// Switches a socket to non-blocking. Unix goes through the file descriptor
// flags; Winsock has no fcntl and uses an ioctl instead.
static inline bool lupine_test_set_nonblocking(lupine_socket_t socket) {
#ifdef _WIN32
  u_long enabled = 1;
  return ioctlsocket(socket, FIONBIO, &enabled) == 0;
#else
  int flags = fcntl(socket, F_GETFL, 0);
  return flags >= 0 && fcntl(socket, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

// shutdown() direction constants and the "do not raise SIGPIPE" send flag are
// spelled differently, or not at all, on Winsock.
#ifdef _WIN32
#define LUPINE_TEST_SHUT_WR SD_SEND
#define LUPINE_TEST_SHUT_RDWR SD_BOTH
#define LUPINE_TEST_MSG_NOSIGNAL 0
#else
#define LUPINE_TEST_SHUT_WR SHUT_WR
#define LUPINE_TEST_SHUT_RDWR SHUT_RDWR
#define LUPINE_TEST_MSG_NOSIGNAL MSG_NOSIGNAL
#endif

// Winsock types the option value as const char*, POSIX as const void*.
static inline int lupine_test_setsockopt_int(lupine_socket_t socket, int level,
                                             int option, int value) {
  return setsockopt(socket, level, option,
                    reinterpret_cast<const char *>(&value), sizeof(value));
}

// The CRT has no setenv; _putenv_s takes the same pair and clears on empty.
static inline int lupine_test_setenv(const char *name, const char *value) {
#ifdef _WIN32
  return _putenv_s(name, value);
#else
  return setenv(name, value, 1);
#endif
}

static inline int lupine_test_unsetenv(const char *name) {
#ifdef _WIN32
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}

#endif // LUPINE_TEST_PLATFORM_H
