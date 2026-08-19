#ifndef LUPINE_PLATFORM_H
#define LUPINE_PLATFORM_H

#if __has_include(<elf.h>)
#include <elf.h>
#else

// CUDA cubins are ELF images even when the client is built for a host
// platform that does not provide <elf.h> (notably macOS). This is the subset
// of the ELF64 ABI used by the module-image parser.
#include <cstdint>

using Elf64_Addr = std::uint64_t;
using Elf64_Off = std::uint64_t;
using Elf64_Half = std::uint16_t;
using Elf64_Word = std::uint32_t;
using Elf64_Xword = std::uint64_t;
using Elf64_Sxword = std::int64_t;

constexpr int EI_NIDENT = 16;

struct Elf64_Ehdr {
  unsigned char e_ident[EI_NIDENT];
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;
  Elf64_Half e_shstrndx;
};

struct Elf64_Phdr {
  Elf64_Word p_type;
  Elf64_Word p_flags;
  Elf64_Off p_offset;
  Elf64_Addr p_vaddr;
  Elf64_Addr p_paddr;
  Elf64_Xword p_filesz;
  Elf64_Xword p_memsz;
  Elf64_Xword p_align;
};

struct Elf64_Shdr {
  Elf64_Word sh_name;
  Elf64_Word sh_type;
  Elf64_Xword sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  Elf64_Xword sh_size;
  Elf64_Word sh_link;
  Elf64_Word sh_info;
  Elf64_Xword sh_addralign;
  Elf64_Xword sh_entsize;
};

struct Elf64_Sym {
  Elf64_Word st_name;
  unsigned char st_info;
  unsigned char st_other;
  Elf64_Half st_shndx;
  Elf64_Addr st_value;
  Elf64_Xword st_size;
};

#define ELFMAG "\177ELF"
#define SELFMAG 4
#define EI_CLASS 4
#define ELFCLASS64 2
#define SHT_SYMTAB 2
#define SHT_NOBITS 8
#define SHT_DYNSYM 11
#define STT_FUNC 2
#define ELF64_ST_TYPE(info) ((info) & 0x0f)

#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <BaseTsd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <io.h>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>
// clang-format off: Windows extension headers require winsock2.h first.
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <mstcpip.h>
#include <windows.h>
// clang-format on

using ssize_t = SSIZE_T;
using socklen_t = int;
using lupine_socket_t = SOCKET;

struct iovec {
  void *iov_base;
  size_t iov_len;
};

using pthread_mutex_t = SRWLOCK;
using pthread_cond_t = CONDITION_VARIABLE;
using pthread_t = HANDLE;
using pthread_once_t = INIT_ONCE;
using pid_t = int;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT
#define LUPINE_INVALID_SOCKET INVALID_SOCKET
#define LUPINE_STDOUT_FD _fileno(stdout)

inline int pthread_mutex_init(pthread_mutex_t *mutex, void *) {
  InitializeSRWLock(mutex);
  return 0;
}
inline int pthread_mutex_destroy(pthread_mutex_t *) { return 0; }

inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
  AcquireSRWLockExclusive(mutex);
  return 0;
}

inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
  ReleaseSRWLockExclusive(mutex);
  return 0;
}

inline int pthread_cond_init(pthread_cond_t *cond, void *) {
  InitializeConditionVariable(cond);
  return 0;
}
inline int pthread_cond_destroy(pthread_cond_t *) { return 0; }

inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
  return SleepConditionVariableSRW(cond, mutex, INFINITE, 0) ? 0 : -1;
}

inline int pthread_cond_broadcast(pthread_cond_t *cond) {
  WakeAllConditionVariable(cond);
  return 0;
}

struct lupine_windows_thread_start {
  void *(*start)(void *);
  void *arg;
};

inline DWORD WINAPI lupine_windows_thread_main(void *opaque) {
  auto *thread_start = static_cast<lupine_windows_thread_start *>(opaque);
  void *(*start)(void *) = thread_start->start;
  void *arg = thread_start->arg;
  delete thread_start;
  start(arg);
  return 0;
}

inline int pthread_create(pthread_t *thread, void *, void *(*start)(void *),
                          void *arg) {
  auto *thread_start =
      new (std::nothrow) lupine_windows_thread_start{start, arg};
  if (thread_start == nullptr) {
    return -1;
  }
  *thread = CreateThread(nullptr, 0, lupine_windows_thread_main, thread_start,
                         0, nullptr);
  if (*thread == nullptr) {
    delete thread_start;
    return -1;
  }
  return 0;
}

inline int pthread_join(pthread_t thread, void **) {
  if (thread != nullptr) {
    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0) {
      return -1;
    }
    CloseHandle(thread);
  }
  return 0;
}

inline BOOL CALLBACK lupine_windows_once_callback(PINIT_ONCE, PVOID parameter,
                                                  PVOID *) {
  reinterpret_cast<void (*)()>(parameter)();
  return TRUE;
}

inline int pthread_once(pthread_once_t *once, void (*init)()) {
  return InitOnceExecuteOnce(once, lupine_windows_once_callback,
                             reinterpret_cast<void *>(init), nullptr)
             ? 0
             : -1;
}

inline DWORD lupine_thread_id() { return GetCurrentThreadId(); }
inline DWORD lupine_process_id() { return GetCurrentProcessId(); }

inline char *lupine_strdup(const char *value) {
  return value == nullptr ? nullptr : _strdup(value);
}

inline int lupine_strcasecmp(const char *first, const char *second) {
  return _stricmp(first, second);
}

inline char *lupine_strsep(char **string, const char *delimiters) {
  if (string == nullptr || *string == nullptr) {
    return nullptr;
  }
  char *token = *string;
  char *cursor = token;
  while (*cursor != '\0') {
    if (std::strchr(delimiters, *cursor) != nullptr) {
      *cursor = '\0';
      *string = cursor + 1;
      return token;
    }
    ++cursor;
  }
  *string = nullptr;
  return token;
}

// Minimal POSIX virtual-memory compatibility for the client shim. Anonymous
// mappings use VirtualAlloc; file mappings use CreateFileMapping. munmap
// distinguishes the two through VirtualQuery so the call sites can retain
// their existing ownership model.
#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_FIXED_NOREPLACE 0x100000
#define MAP_FAILED reinterpret_cast<void *>(static_cast<intptr_t>(-1))

inline DWORD lupine_windows_page_protection(int prot) {
  if ((prot & PROT_EXEC) != 0) {
    return (prot & PROT_WRITE) != 0 ? PAGE_EXECUTE_READWRITE
                                    : PAGE_EXECUTE_READ;
  }
  if ((prot & PROT_WRITE) != 0) {
    return PAGE_READWRITE;
  }
  if ((prot & PROT_READ) != 0) {
    return PAGE_READONLY;
  }
  return PAGE_NOACCESS;
}

inline void *mmap(void *address, size_t length, int prot, int flags, int fd,
                  long long offset) {
  if (length == 0) {
    return MAP_FAILED;
  }
  if ((flags & MAP_ANONYMOUS) != 0) {
    void *mapping = VirtualAlloc(address, length, MEM_RESERVE | MEM_COMMIT,
                                 lupine_windows_page_protection(prot));
    return mapping == nullptr ? MAP_FAILED : mapping;
  }
  intptr_t os_handle = _get_osfhandle(fd);
  if (os_handle == -1) {
    return MAP_FAILED;
  }
  DWORD protect = (prot & PROT_WRITE) != 0 ? PAGE_READWRITE : PAGE_READONLY;
  HANDLE file_mapping = CreateFileMappingA(reinterpret_cast<HANDLE>(os_handle),
                                           nullptr, protect, 0, 0, nullptr);
  if (file_mapping == nullptr) {
    return MAP_FAILED;
  }
  DWORD access = (prot & PROT_WRITE) != 0 ? FILE_MAP_WRITE : FILE_MAP_READ;
  ULARGE_INTEGER view_offset = {};
  view_offset.QuadPart = static_cast<unsigned long long>(offset);
  void *mapping = MapViewOfFile(file_mapping, access, view_offset.HighPart,
                                view_offset.LowPart, length);
  CloseHandle(file_mapping);
  return mapping == nullptr ? MAP_FAILED : mapping;
}

inline int munmap(void *address, size_t) {
  if (address == nullptr || address == MAP_FAILED) {
    return -1;
  }
  MEMORY_BASIC_INFORMATION info = {};
  if (VirtualQuery(address, &info, sizeof(info)) == 0) {
    return -1;
  }
  if (info.Type == MEM_MAPPED || info.Type == MEM_IMAGE) {
    return UnmapViewOfFile(address) ? 0 : -1;
  }
  return VirtualFree(info.AllocationBase, 0, MEM_RELEASE) ? 0 : -1;
}

inline int mprotect(void *address, size_t length, int prot) {
  DWORD previous = 0;
  return VirtualProtect(address, length, lupine_windows_page_protection(prot),
                        &previous)
             ? 0
             : -1;
}

inline int mincore(void *address, size_t, unsigned char *residency) {
  MEMORY_BASIC_INFORMATION info = {};
  if (VirtualQuery(address, &info, sizeof(info)) == 0 ||
      info.State != MEM_COMMIT) {
    return -1;
  }
  if (residency != nullptr) {
    *residency = 1;
  }
  return 0;
}

// Windows access violations are the native equivalent of the SIGSEGV dirty
// tracking hook used on Unix. These definitions provide only the subset of
// sigaction/sigaltstack consumed by memcpy.cpp.
struct siginfo_t {
  void *si_addr;
};

using lupine_signal_handler_t = void (*)(int);
struct sigaction {
  void (*sa_sigaction)(int, siginfo_t *, void *);
  lupine_signal_handler_t sa_handler;
  int sa_flags;
  int sa_mask;
};

struct stack_t {
  void *ss_sp;
  size_t ss_size;
  int ss_flags;
};

#define SA_SIGINFO 0x1
#define SA_NODEFER 0x2
#define SA_ONSTACK 0x4

inline sigaction lupine_windows_sigsegv_action = {};
inline PVOID lupine_windows_exception_handler_handle = nullptr;

inline LONG CALLBACK
lupine_windows_exception_handler(EXCEPTION_POINTERS *exception) {
  if (exception == nullptr || exception->ExceptionRecord == nullptr ||
      exception->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
      lupine_windows_sigsegv_action.sa_sigaction == nullptr) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  siginfo_t info = {reinterpret_cast<void *>(
      exception->ExceptionRecord->ExceptionInformation[1])};
  lupine_windows_sigsegv_action.sa_sigaction(SIGSEGV, &info, exception);
  return EXCEPTION_CONTINUE_EXECUTION;
}

inline int sigaction(int signal, const struct sigaction *action,
                     struct sigaction *previous) {
  if (signal != SIGSEGV) {
    return -1;
  }
  if (previous != nullptr) {
    *previous = lupine_windows_sigsegv_action;
  }
  if (action != nullptr) {
    lupine_windows_sigsegv_action = *action;
    if (lupine_windows_exception_handler_handle == nullptr &&
        action->sa_sigaction != nullptr) {
      lupine_windows_exception_handler_handle =
          AddVectoredExceptionHandler(1, lupine_windows_exception_handler);
      if (lupine_windows_exception_handler_handle == nullptr) {
        return -1;
      }
    }
  }
  return 0;
}

inline int sigemptyset(int *set) {
  if (set != nullptr) {
    *set = 0;
  }
  return 0;
}

inline int sigaltstack(const stack_t *, stack_t *) { return 0; }

inline long sysconf(int) {
  SYSTEM_INFO info = {};
  GetSystemInfo(&info);
  return static_cast<long>(info.dwPageSize);
}

#define _SC_PAGESIZE 30

inline int sched_yield() {
  SwitchToThread();
  return 0;
}

#if defined(_MSC_VER) && !defined(__clang__)
// MSVC does not expose GCC's __atomic_* builtins. The client state fields are
// deliberately plain scalars because they are shared with signal/exception
// handlers; use the equivalent std::atomic operations at those call sites.
#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif

inline std::memory_order lupine_atomic_order(int order) {
  switch (order) {
  case __ATOMIC_RELAXED:
    return std::memory_order_relaxed;
  case __ATOMIC_ACQUIRE:
  case __ATOMIC_CONSUME:
    return std::memory_order_acquire;
  case __ATOMIC_RELEASE:
    return std::memory_order_release;
  case __ATOMIC_ACQ_REL:
    return std::memory_order_acq_rel;
  default:
    return std::memory_order_seq_cst;
  }
}

inline std::memory_order lupine_atomic_failure_order(int order) {
  if (order == __ATOMIC_ACQ_REL || order == __ATOMIC_ACQUIRE ||
      order == __ATOMIC_CONSUME) {
    return std::memory_order_acquire;
  }
  if (order == __ATOMIC_SEQ_CST) {
    return std::memory_order_seq_cst;
  }
  return std::memory_order_relaxed;
}

template <typename T> struct lupine_atomic_identity {
  using type = T;
};

template <typename T>
inline std::atomic<T> *lupine_atomic_pointer(volatile T *value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "atomic compatibility requires a scalar type");
  return reinterpret_cast<std::atomic<T> *>(const_cast<T *>(value));
}

template <typename T>
inline const std::atomic<T> *lupine_atomic_pointer(const volatile T *value) {
  static_assert(std::is_trivially_copyable<T>::value,
                "atomic compatibility requires a scalar type");
  return reinterpret_cast<const std::atomic<T> *>(const_cast<const T *>(value));
}

template <typename T>
inline T lupine_atomic_load_n(const volatile T *value, int order) {
  return lupine_atomic_pointer(value)->load(lupine_atomic_order(order));
}

template <typename T>
inline void
lupine_atomic_store_n(volatile T *value,
                      typename lupine_atomic_identity<T>::type desired,
                      int order) {
  lupine_atomic_pointer(value)->store(desired, lupine_atomic_order(order));
}

template <typename T>
inline T
lupine_atomic_exchange_n(volatile T *value,
                         typename lupine_atomic_identity<T>::type desired,
                         int order) {
  return lupine_atomic_pointer(value)->exchange(desired,
                                                lupine_atomic_order(order));
}

template <typename T>
inline T
lupine_atomic_add_fetch(volatile T *value,
                        typename lupine_atomic_identity<T>::type amount,
                        int order) {
  return lupine_atomic_pointer(value)->fetch_add(amount,
                                                 lupine_atomic_order(order)) +
         amount;
}

template <typename T>
inline T
lupine_atomic_sub_fetch(volatile T *value,
                        typename lupine_atomic_identity<T>::type amount,
                        int order) {
  return lupine_atomic_pointer(value)->fetch_sub(amount,
                                                 lupine_atomic_order(order)) -
         amount;
}

template <typename T>
inline bool lupine_atomic_compare_exchange_n(
    volatile T *value, T *expected,
    typename lupine_atomic_identity<T>::type desired, bool weak,
    int success_order, int failure_order) {
  if (weak) {
    return lupine_atomic_pointer(value)->compare_exchange_weak(
        *expected, desired, lupine_atomic_order(success_order),
        lupine_atomic_failure_order(failure_order));
  }
  return lupine_atomic_pointer(value)->compare_exchange_strong(
      *expected, desired, lupine_atomic_order(success_order),
      lupine_atomic_failure_order(failure_order));
}

#define __atomic_load_n lupine_atomic_load_n
#define __atomic_store_n lupine_atomic_store_n
#define __atomic_exchange_n lupine_atomic_exchange_n
#define __atomic_add_fetch lupine_atomic_add_fetch
#define __atomic_sub_fetch lupine_atomic_sub_fetch
#define __atomic_compare_exchange_n lupine_atomic_compare_exchange_n
#endif

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
#ifdef MSG_NOSIGNAL
  return sendmsg(socket, &msg, MSG_NOSIGNAL);
#else
  return sendmsg(socket, &msg, 0);
#endif
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
//   * TCP_USER_TIMEOUT, where available, applies the same dead-peer bound
//     while application data is unacknowledged. TCP keepalive does not run
//     while data is in flight, which otherwise leaves a disconnected client
//     on the much longer system retransmission timeout.
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
#ifdef SO_NOSIGPIPE
  // Darwin has no MSG_NOSIGNAL; apply the equivalent behavior to the socket.
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE,
             reinterpret_cast<const char *>(&enabled), sizeof(enabled));
#endif

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
  constexpr int kDeadPeerTimeoutMs =
      (kKeepidleSec + kKeepintvlSec * kKeepcnt) * 1000;
#if defined(__APPLE__)
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE,
#else
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,
#endif
             reinterpret_cast<const char *>(&keepidle), sizeof(keepidle));
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL,
             reinterpret_cast<const char *>(&keepintvl), sizeof(keepintvl));
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,
             reinterpret_cast<const char *>(&keepcnt), sizeof(keepcnt));
#ifdef TCP_USER_TIMEOUT
  int user_timeout = kDeadPeerTimeoutMs;
  setsockopt(fd, IPPROTO_TCP, TCP_USER_TIMEOUT, &user_timeout,
             sizeof(user_timeout));
#endif
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
