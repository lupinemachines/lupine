#include "device_ioctl_capture.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

static volatile sig_atomic_t delivered;
static void signal_handler(int) { ++delivered; }
static int allocation(void *argument) {
  raise(SIGUSR1);
  raise(SIGTRAP);
  unsigned char bytes[48]{};
  uint32_t client = 123, memory = 456;
  std::memcpy(bytes, &client, 4);
  std::memcpy(bytes + 8, &memory, 4);
  return static_cast<int>(
      syscall(SYS_ioctl, *static_cast<int *>(argument), 0xc030462bul, bytes));
}
static int failure(void *) { return 1; }
int main() {
  signal(SIGUSR1, signal_handler);
  signal(SIGTRAP, signal_handler);
  // A seccomp ERRNO action with errno zero emulates a successful kernel ioctl.
  // The real ptrace observer still sees its entry, exit, argument and result.
  sock_filter filter[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_ioctl, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, args[1])),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0xc030462bul, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)};
  sock_fprog program{sizeof(filter) / sizeof(filter[0]), filter};
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
      syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &program))
    return 77;
  int native = open("/dev/null", O_RDONLY), descriptor = -1;
  uint32_t client = 0, memory = 0;
  int result = lupine_capture_rm_alloc(allocation, &native, &descriptor,
                                       &client, &memory);
  if (result == ENOTSUP || result == EPERM)
    return 77;
  if (result || delivered != 2 || descriptor != native || client != 123 ||
      memory != 456) {
    std::fprintf(stderr, "capture failed: %d fd=%d client=%u memory=%u\n",
                 result, descriptor, client, memory);
    return 1;
  }
  if (lupine_capture_rm_alloc(failure, nullptr, &descriptor, &client,
                              &memory) != EIO)
    return 1;
  close(native);
  return 0;
}
