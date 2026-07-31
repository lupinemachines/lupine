#ifndef LUPINE_PLATFORM_H
#define LUPINE_PLATFORM_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <BaseTsd.h>
#include <algorithm>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <io.h>
#include <mutex>
#include <thread>
#include <vector>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <mstcpip.h>

using ssize_t = SSIZE_T;
using socklen_t = int;
using lupine_socket_t = SOCKET;

struct iovec {
  void *iov_base;
  size_t iov_len;
};

struct pthread_mutex_t {
  std::mutex mutex;
};

struct pthread_cond_t {
  std::condition_variable_any cond;
};

using pthread_t = std::thread *;

#define PTHREAD_MUTEX_INITIALIZER                                              \
  {}
#define PTHREAD_COND_INITIALIZER                                               \
  {}
#define LUPINE_INVALID_SOCKET INVALID_SOCKET
#define LUPINE_STDOUT_FD _fileno(stdout)

inline int pthread_mutex_init(pthread_mutex_t *, void *) { return 0; }
inline int pthread_mutex_destroy(pthread_mutex_t *) { return 0; }

inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
  mutex->mutex.lock();
  return 0;
}

inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
  mutex->mutex.unlock();
  return 0;
}

inline int pthread_cond_init(pthread_cond_t *, void *) { return 0; }
inline int pthread_cond_destroy(pthread_cond_t *) { return 0; }

inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
  std::unique_lock<std::mutex> lock(mutex->mutex, std::adopt_lock);
  cond->cond.wait(lock);
  lock.release();
  return 0;
}

inline int pthread_cond_broadcast(pthread_cond_t *cond) {
  cond->cond.notify_all();
  return 0;
}

inline int pthread_create(pthread_t *thread, void *, void *(*start)(void *),
                          void *arg) {
  try {
    *thread = new std::thread([start, arg]() { start(arg); });
  } catch (...) {
    return -1;
  }
  return 0;
}

inline int pthread_join(pthread_t thread, void **) {
  if (thread != nullptr) {
    thread->join();
    delete thread;
  }
  return 0;
}

inline int lupine_socket_init() {
  static int result = []() {
    WSADATA data;
    return WSAStartup(MAKEWORD(2, 2), &data) == 0 ? 0 : -1;
  }();
  return result;
}

inline bool lupine_socket_error_is_intr() {
  return WSAGetLastError() == WSAEINTR;
}

inline int lupine_socket_close(lupine_socket_t socket) {
  return closesocket(socket);
}

inline int lupine_socket_set_reuseaddr(lupine_socket_t socket) {
  const char enable = 1;
  return setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
}

inline ssize_t lupine_socket_recv(lupine_socket_t socket, void *data,
                                  size_t size) {
  int chunk = static_cast<int>(std::min<size_t>(size, INT_MAX));
  return recv(socket, static_cast<char *>(data), chunk, 0);
}

// Vectored send of up to `count` buffers in a single syscall. Returns the
// number of bytes accepted by the socket (which may be fewer than the total
// when the send buffer fills), or a negative value on error. Callers advance
// over the buffers and retry on a short write.
inline ssize_t lupine_socket_sendv(lupine_socket_t socket,
                                   const struct iovec *iov, int count) {
  std::vector<WSABUF> bufs(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    bufs[i].buf = static_cast<CHAR *>(iov[i].iov_base);
    bufs[i].len = static_cast<ULONG>(
        std::min<size_t>(iov[i].iov_len, static_cast<size_t>(ULONG_MAX)));
  }
  DWORD sent = 0;
  if (WSASend(socket, bufs.data(), static_cast<DWORD>(count), &sent, 0, nullptr,
              nullptr) != 0) {
    return -1;
  }
  return static_cast<ssize_t>(sent);
}

inline int lupine_fd_dup(int fd) { return _dup(fd); }
inline int lupine_fd_dup2(int source, int dest) { return _dup2(source, dest); }
inline int lupine_fd_close(int fd) { return _close(fd); }
inline ssize_t lupine_fd_read(int fd, void *data, size_t size) {
  return _read(fd, data,
               static_cast<unsigned int>(std::min<size_t>(size, UINT_MAX)));
}
inline long lupine_fd_seek(int fd, long offset, int origin) {
  return _lseek(fd, offset, origin);
}
inline int lupine_fd_fileno(FILE *file) { return _fileno(file); }
inline int lupine_fd_truncate(int fd, long length) {
  return _chsize(fd, length);
}

#else

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

using lupine_socket_t = int;

#define LUPINE_INVALID_SOCKET (-1)
#define LUPINE_STDOUT_FD STDOUT_FILENO

inline int lupine_socket_init() { return 0; }
inline bool lupine_socket_error_is_intr() { return errno == EINTR; }
inline int lupine_socket_close(lupine_socket_t socket) { return close(socket); }
inline int lupine_socket_set_reuseaddr(lupine_socket_t socket) {
  const int enable = 1;
  return setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
}
inline ssize_t lupine_socket_recv(lupine_socket_t socket, void *data,
                                  size_t size) {
  return recv(socket, data, size, 0);
}
// Vectored send of up to `count` buffers in a single syscall. Returns the
// number of bytes accepted by the socket (which may be fewer than the total
// when the send buffer fills), or a negative value on error. Callers advance
// over the buffers and retry on a short write.
inline ssize_t lupine_socket_sendv(lupine_socket_t socket,
                                   const struct iovec *iov, int count) {
  struct msghdr msg = {};
  msg.msg_iov = const_cast<struct iovec *>(iov);
  msg.msg_iovlen = static_cast<size_t>(count);
  return sendmsg(socket, &msg, MSG_NOSIGNAL);
}

inline int lupine_fd_dup(int fd) { return dup(fd); }
inline int lupine_fd_dup2(int source, int dest) { return dup2(source, dest); }
inline int lupine_fd_close(int fd) { return close(fd); }
inline ssize_t lupine_fd_read(int fd, void *data, size_t size) {
  return read(fd, data, size);
}
inline off_t lupine_fd_seek(int fd, off_t offset, int origin) {
  return lseek(fd, offset, origin);
}
inline int lupine_fd_fileno(FILE *file) { return fileno(file); }
// Truncates the open file description behind `fd` to exactly `length` bytes.
// Used to reset the reused device-printf capture file to empty.
inline int lupine_fd_truncate(int fd, off_t length) {
  return ftruncate(fd, length);
}

#endif

// lupine_socket_apply_transport_options sets the TCP options every lupine
// connection uses:
//
//   * TCP_NODELAY so small RPC frames are not delayed by Nagle.
//   * SO_KEEPALIVE with tuned probes, so a long-lived connection survives the
//     idle gaps in long-running workloads. Stateful middleboxes (NAT gateways,
//     cloud load balancers, conntrack entries, firewalls) silently reap idle
//     flows far sooner than the kernel's default 2-hour keepalive; a transient
//     blip then surfaces as a fatal RPC error. Keepalive probes are emitted
//     only while the connection is idle, so active transfers pay no latency.
//     With the defaults a dead peer is detected in ~105s instead of hanging on
//     the retransmit timer. Socket buffer sizing is left to the OS, which
//     auto-tunes on modern kernels.
//
// Returns 0 on success, -1 on an invalid descriptor.
inline int lupine_socket_apply_transport_options(lupine_socket_t fd) {
  if (fd == LUPINE_INVALID_SOCKET) {
    return -1;
  }

  // Seconds a connection may sit idle before the first keepalive probe, the
  // interval between probes, and how many unanswered probes declare the peer
  // dead. Chosen to keep NAT/load-balancer conntrack entries warm (60s is
  // shorter than every common middlebox idle timeout) while bounding dead-peer
  // detection to ~105s.
  constexpr int kKeepidleSec = 60;
  constexpr int kKeepintvlSec = 15;
  constexpr int kKeepcnt = 3;

  int enabled = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
             reinterpret_cast<const char *>(&enabled), sizeof(enabled));
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE,
             reinterpret_cast<const char *>(&enabled), sizeof(enabled));

  int keepidle = kKeepidleSec;
  int keepintvl = kKeepintvlSec;
  int keepcnt = kKeepcnt;
#ifdef _WIN32
  // SIO_KEEPALIVE_VALS sets the idle and probe intervals in one ioctl.
  tcp_keepalive ka;
  ka.onoff = 1;
  ka.keepalivetime = static_cast<ULONG>(keepidle) * 1000;
  ka.keepaliveinterval = static_cast<ULONG>(keepintvl) * 1000;
  DWORD bytes_returned = 0;
  WSAIoctl(fd, SIO_KEEPALIVE_VALS, &ka, sizeof(ka), nullptr, 0,
           &bytes_returned, nullptr, nullptr);
#ifdef TCP_KEEPCNT
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, reinterpret_cast<const char *>(&keepcnt),
             sizeof(keepcnt));
#endif
#else
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,
             reinterpret_cast<const char *>(&keepidle), sizeof(keepidle));
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL,
             reinterpret_cast<const char *>(&keepintvl), sizeof(keepintvl));
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,
             reinterpret_cast<const char *>(&keepcnt), sizeof(keepcnt));
#endif
  return 0;
}

// lupine_socket_connect_with_timeout connects `fd` to `addr`, waiting up to
// `timeout_ms` milliseconds. A non-positive timeout performs a plain blocking
// connect (the historical behavior). A bounded timeout prevents a
// packet-filtered port from blocking the connect-retry loop for minutes
// (the kernel's SYN retransmit backoff). Returns 0 on success, -1 on error
// or timeout.
inline int lupine_socket_connect_with_timeout(lupine_socket_t fd,
                                               const struct sockaddr *addr,
                                               socklen_t addrlen,
                                               int timeout_ms) {
  if (timeout_ms <= 0) {
    return connect(fd, addr, addrlen);
  }
#ifdef _WIN32
  u_long nonblocking = 1;
  if (ioctlsocket(fd, FIONBIO, &nonblocking) != 0) {
    return connect(fd, addr, addrlen);
  }
  int rc = connect(fd, addr, addrlen);
  if (rc == 0) {
    nonblocking = 0;
    ioctlsocket(fd, FIONBIO, &nonblocking);
    return 0;
  }
  if (WSAGetLastError() != WSAEWOULDBLOCK) {
    nonblocking = 0;
    ioctlsocket(fd, FIONBIO, &nonblocking);
    return -1;
  }
  fd_set write_fds;
  FD_ZERO(&write_fds);
  FD_SET(fd, &write_fds);
  struct timeval tv;
  tv.tv_sec = timeout_ms / 1000;
  tv.tv_usec = (timeout_ms % 1000) * 1000;
  rc = select(0, nullptr, &write_fds, nullptr, &tv);
  nonblocking = 0;
  ioctlsocket(fd, FIONBIO, &nonblocking);
  if (rc <= 0) {
    return -1;
  }
  int so_error = 0;
  int so_len = sizeof(so_error);
  if (getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&so_error),
                 &so_len) != 0 ||
      so_error != 0) {
    return -1;
  }
  return 0;
#else
  int saved_flags = fcntl(fd, F_GETFL, 0);
  if (saved_flags < 0 || fcntl(fd, F_SETFL, saved_flags | O_NONBLOCK) < 0) {
    return connect(fd, addr, addrlen);
  }
  int rc = connect(fd, addr, addrlen);
  if (rc == 0) {
    fcntl(fd, F_SETFL, saved_flags);
    return 0;
  }
  if (errno != EINPROGRESS) {
    fcntl(fd, F_SETFL, saved_flags);
    return -1;
  }
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLOUT;
  pfd.revents = 0;
  rc = poll(&pfd, 1, timeout_ms);
  fcntl(fd, F_SETFL, saved_flags); // restore blocking mode regardless
  if (rc <= 0) {
    return -1;
  }
  int so_error = 0;
  socklen_t so_len = sizeof(so_error);
  if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &so_len) != 0 ||
      so_error != 0) {
    return -1;
  }
  return 0;
#endif
}

#endif
