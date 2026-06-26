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

#else

#include <algorithm>
#include <cerrno>
#include <cstdio>
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

#endif

#endif
