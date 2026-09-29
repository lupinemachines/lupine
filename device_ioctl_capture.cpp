#include "device_ioctl_capture.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <new>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <sys/user.h>
#elif defined(__aarch64__)
#include <asm/ptrace.h>
#include <elf.h>
#endif

namespace {
constexpr unsigned long rm_alloc = 0xc030462bul;
// PR_SET_PTRACER authorizes one child for the whole process.
std::mutex capture_mutex;
struct capture {
  std::atomic<int> state{0};
  int fd = -1;
  uint32_t client = 0, memory = 0;
};
struct registers {
  long number, result;
  unsigned long fd, command, argument;
};
bool read_registers(pid_t target, registers *output) {
#if defined(__x86_64__)
  user_regs_struct r{};
  if (ptrace(PTRACE_GETREGS, target, nullptr, &r) < 0)
    return false;
  *output = {static_cast<long>(r.orig_rax), static_cast<long>(r.rax), r.rdi,
             r.rsi, r.rdx};
#elif defined(__aarch64__)
  user_pt_regs r{};
  iovec io{&r, sizeof(r)};
  if (ptrace(PTRACE_GETREGSET, target, reinterpret_cast<void *>(NT_PRSTATUS),
             &io) < 0)
    return false;
  *output = {static_cast<long>(r.regs[8]), static_cast<long>(r.regs[0]),
             r.regs[0], r.regs[1], r.regs[2]};
#else
  (void)target;
  (void)output;
  return false;
#endif
  return true;
}
// After fork this child uses only stack storage and syscall wrappers. It never
// enters inherited C++/CUDA state or acquires an inherited userspace lock.
void trace(pid_t target, capture *shared) {
  while (shared->state.load(std::memory_order_acquire) == 0)
    sched_yield();
  if (ptrace(PTRACE_SEIZE, target, nullptr,
             reinterpret_cast<void *>(PTRACE_O_TRACESYSGOOD)) < 0) {
    shared->state.store(-1, std::memory_order_release);
    _exit(1);
  }
  int status = 0;
  if (ptrace(PTRACE_INTERRUPT, target, nullptr, nullptr) < 0 ||
      waitpid(target, &status, __WALL) < 0) {
    shared->state.store(-1, std::memory_order_release);
    (void)ptrace(PTRACE_DETACH, target, nullptr, nullptr);
    _exit(1);
  }
  shared->state.store(2, std::memory_order_release);
  bool entering = true, pending = false;
  unsigned long argument = 0, descriptor = 0;
  int signal = 0;
  for (;;) {
    if (ptrace(PTRACE_SYSCALL, target, nullptr,
               reinterpret_cast<void *>(static_cast<intptr_t>(signal))) < 0 ||
        waitpid(target, &status, __WALL) < 0 || !WIFSTOPPED(status))
      break;
    signal = 0;
    if (shared->state.load(std::memory_order_acquire) < 0)
      break;
    if (WSTOPSIG(status) != (SIGTRAP | 0x80)) {
      // Deliver signals on the next resume without losing the syscall stop
      // that follows them or changing entry/exit parity.
      if ((static_cast<unsigned int>(status) >> 16) == 0)
        signal = WSTOPSIG(status);
      continue;
    }
    registers call{};
    if (!read_registers(target, &call))
      break;
    if (entering && call.number == SYS_ioctl && call.command == rm_alloc) {
      pending = true;
      argument = call.argument;
      descriptor = call.fd;
    } else if (!entering && pending) {
      pending = false;
      unsigned char bytes[48]{};
      iovec local{bytes, sizeof(bytes)};
      iovec remote{reinterpret_cast<void *>(argument), sizeof(bytes)};
      uint32_t client = 0, memory = 0, rm_status = 1;
      if (call.result == 0 &&
          process_vm_readv(target, &local, 1, &remote, 1, 0) == sizeof(bytes)) {
        std::memcpy(&client, bytes, 4);
        std::memcpy(&memory, bytes + 8, 4);
        std::memcpy(&rm_status, bytes + 40, 4);
        if (client && memory && !rm_status) {
          shared->fd = static_cast<int>(descriptor);
          shared->client = client;
          shared->memory = memory;
          shared->state.store(3, std::memory_order_release);
          (void)ptrace(PTRACE_DETACH, target, nullptr, nullptr);
          _exit(0);
        }
      }
    }
    entering = !entering;
  }
  shared->state.store(-1, std::memory_order_release);
  (void)ptrace(PTRACE_DETACH, target, nullptr, nullptr);
  _exit(1);
}
bool wait_state(capture *shared, int state) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (shared->state.load(std::memory_order_acquire) == state) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    sched_yield();
  }
  return true;
}
} // namespace

int lupine_capture_rm_alloc(int (*operation)(void *), void *argument, int *fd,
                            uint32_t *client, uint32_t *memory) {
  std::lock_guard<std::mutex> lock(capture_mutex);
  void *mapping = mmap(nullptr, sizeof(capture), PROT_READ | PROT_WRITE,
                       MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  if (mapping == MAP_FAILED)
    return errno;
  auto *shared = new (mapping) capture;
  pid_t target = static_cast<pid_t>(syscall(SYS_gettid));
  pid_t child = fork();
  if (child == 0)
    trace(target, shared);
  int error = 0;
  if (child < 0)
    error = errno;
  else {
    if (prctl(PR_SET_PTRACER, child, 0, 0, 0) < 0)
      error = errno;
    else {
      shared->state.store(1, std::memory_order_release);
      if (!wait_state(shared, 1) ||
          shared->state.load(std::memory_order_acquire) != 2)
        error = ENOTSUP;
      else {
        int result = operation(argument);
        if (shared->state.load(std::memory_order_acquire) != 3) {
          shared->state.store(-2, std::memory_order_release);
          (void)syscall(SYS_getpid);
        }
        if (result != 0 || shared->state.load(std::memory_order_acquire) != 3)
          error = EIO;
        else {
          *fd = shared->fd;
          *client = shared->client;
          *memory = shared->memory;
        }
      }
    }
    if (error)
      (void)kill(child, SIGKILL);
    int status;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    (void)prctl(PR_SET_PTRACER, 0, 0, 0, 0);
  }
  munmap(mapping, sizeof(capture));
  return error;
}
