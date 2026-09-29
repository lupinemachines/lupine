#include "device_ioctl.h"
#include "device_ioctl_guard.h"
#include "device_ioctl_proxy.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int releases;
static int opens;
static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s (errno=%d)\n", message, errno);
    std::exit(1);
  }
}
void lupine_device_proxy_remote_close(int, uint64_t) { ++releases; }
int lupine_device_open(const char *path, int flags) {
  ++opens;
  int fd = lupine_device_proxy_create(
      7, 123, std::strcmp(path, "/dev/nvidia-uvm") == 0);
  if (fd >= 0 && !(flags & O_CLOEXEC))
    fcntl(fd, F_SETFD, 0);
  return fd;
}
int lupine_device_ioctl(int, uint32_t command, void *) {
  errno = command == lupine_uvm::map ? EACCES : ENOTTY;
  return -1;
}
int main() {
  setenv("LUPINE_SERVER", "test", 1);
  {
    lupine_native_cuda_call_guard native_call;
    int native = open("/dev/nvidia-uvm", O_RDWR);
    require(opens == 0, "native CUDA device opens bypass proxy hooks");
    if (native >= 0)
      close(native);
  }
  int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  require(fd >= 0 && (fcntl(fd, F_GETFD) & FD_CLOEXEC), "proxy open/CLOEXEC");
  pid_t child = fork();
  require(child >= 0, "fork for descriptor cleanup");
  if (child == 0) {
    int copy = dup(fd);
    _exit(copy >= 0 && close(copy) == 0 && close(fd) == 0 && releases == 0 ? 0
                                                                           : 1);
  }
  int child_status;
  require(waitpid(child, &child_status, 0) == child &&
              WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
          "fork child cleans fds without parent's RPC state");
  char path[64], target[256];
  std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  ssize_t size = readlink(path, target, sizeof(target) - 1);
  require(size > 0, "proxy descriptor discoverable");
  target[size] = 0;
  require(std::strstr(target, "/dev/nvidia-uvm"), "UVM name discoverable");
  int duplicate = dup(fd);
  int third = fcntl(fd, F_DUPFD_CLOEXEC, 0);
  require(duplicate >= 0 && third >= 0, "duplicates created");
  require(lupine_device_proxy_get(duplicate)->token == 123,
          "dup retains remote token");
  require(close(fd) == 0 && releases == 0, "close retains remote duplicate");
  require(ioctl(duplicate, 33, nullptr) == -1 && errno == EACCES,
          "proxy ioctl forwarded");
  require(ioctl(duplicate, 999) == -1 && errno == ENOTTY, "no-argument ioctl");
  require(mmap(nullptr, 4096, PROT_READ, MAP_SHARED, duplicate, 0) ==
                  MAP_FAILED &&
              errno == ENODEV,
          "device mmap explicitly unsupported");
  require(dup2(-1, third) == -1 && lupine_device_proxy_get(third),
          "failed dup2 preserves target");
  require(dup2(third, third) == third, "dup2 same descriptor");
  require(dup3(third, third, 0) == -1 && errno == EINVAL,
          "dup3 same descriptor");
  require(close(duplicate) == 0 && releases == 0,
          "fcntl duplicate still alive");
  int ordinary = open("/dev/null", O_RDONLY);
  require(ordinary >= 0, "ordinary open forwarded");
  require(dup2(ordinary, third) == third && releases == 1,
          "dup2 replacement releases proxy");
  require(!lupine_device_proxy_get(third),
          "replacement is ordinary descriptor");
  close(third);
  close(ordinary);
  int pipes[2];
  require(pipe(pipes) == 0 && write(pipes[1], "abc", 3) == 3, "ordinary pipe");
  int available = 0;
  require(ioctl(pipes[0], FIONREAD, &available) == 0 && available == 3,
          "ordinary ioctl preserved");
  close(pipes[0]);
  close(pipes[1]);
  void *mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  require(mapping != MAP_FAILED && munmap(mapping, 4096) == 0,
          "ordinary mmap preserved");
  char name[] = "/tmp/lupine-ioctl.XXXXXX";
  fd = mkstemp(name);
  require(fd >= 0, "ordinary creation");
  close(fd);
  unlink(name);
  return 0;
}
