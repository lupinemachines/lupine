#include "device_ioctl_guard.h"
#include "device_ioctl_proxy.h"

#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
template <typename Function> Function native_symbol(const char *name) {
  return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}

bool is_remote_device(const char *path) {
  if (lupine_native_cuda_call_depth)
    return false;
  const char *server = std::getenv("LUPINE_SERVER");
  return path && server && *server &&
         (std::strcmp(path, "/dev/nvidia-uvm") == 0 ||
          std::strcmp(path, "/dev/nvidiactl") == 0);
}
int open_device(int directory, const char *path, int flags, va_list arguments) {
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE)
    mode = va_arg(arguments, unsigned int);
  if (is_remote_device(path))
    return lupine_device_open(path, flags);
  static auto native =
      native_symbol<int (*)(int, const char *, int, ...)>("openat");
  return native(directory, path, flags, mode);
}
} // namespace

extern "C" int open(const char *path, int flags, ...) {
  va_list args;
  va_start(args, flags);
  int result = open_device(AT_FDCWD, path, flags, args);
  va_end(args);
  return result;
}
extern "C" int open64(const char *path, int flags, ...) {
  va_list args;
  va_start(args, flags);
  int result = open_device(AT_FDCWD, path, flags, args);
  va_end(args);
  return result;
}
extern "C" int openat(int directory, const char *path, int flags, ...) {
  va_list args;
  va_start(args, flags);
  int result = open_device(directory, path, flags, args);
  va_end(args);
  return result;
}
extern "C" int openat64(int directory, const char *path, int flags, ...) {
  va_list args;
  va_start(args, flags);
  int result = open_device(directory, path, flags, args);
  va_end(args);
  return result;
}
extern "C" int __open_2(const char *path, int flags) {
  if (is_remote_device(path))
    return lupine_device_open(path, flags);
  static auto native = native_symbol<int (*)(const char *, int)>("__open_2");
  return native(path, flags);
}
extern "C" int __open64_2(const char *path, int flags) {
  if (is_remote_device(path))
    return lupine_device_open(path, flags);
  static auto native = native_symbol<int (*)(const char *, int)>("__open64_2");
  return native(path, flags);
}
extern "C" int __openat_2(int dir, const char *path, int flags) {
  if (is_remote_device(path))
    return lupine_device_open(path, flags);
  static auto native =
      native_symbol<int (*)(int, const char *, int)>("__openat_2");
  return native(dir, path, flags);
}
extern "C" int __openat64_2(int dir, const char *path, int flags) {
  if (is_remote_device(path))
    return lupine_device_open(path, flags);
  static auto native =
      native_symbol<int (*)(int, const char *, int)>("__openat64_2");
  return native(dir, path, flags);
}

// ioctl/fcntl's optional argument occupies the third integer argument register
// on Linux amd64 and arm64. A fixed ABI entry preserves that register without
// reading a nonexistent C variadic argument for commands with no argument.
extern "C" int lupine_ioctl_entry(int fd, unsigned long command,
                                  uintptr_t argument) __asm__("ioctl");
extern "C" int lupine_ioctl_entry(int fd, unsigned long command,
                                  uintptr_t argument) {
  if (!lupine_device_proxy_get(fd)) {
    static auto native =
        native_symbol<int (*)(int, unsigned long, uintptr_t)>("ioctl");
    return native(fd, command, argument);
  }
  if (command > UINT32_MAX) {
    errno = ENOTTY;
    return -1;
  }
  return lupine_device_ioctl(fd, static_cast<uint32_t>(command),
                             reinterpret_cast<void *>(argument));
}
extern "C" int close(int fd) {
  if (lupine_device_proxy_get(fd))
    return lupine_device_proxy_close(fd);
  static auto native = native_symbol<int (*)(int)>("close");
  return native(fd);
}
extern "C" int dup(int fd) noexcept {
  return lupine_device_proxy_dup(fd, -1, 0, false);
}
extern "C" int dup2(int fd, int target) noexcept {
  return lupine_device_proxy_dup(fd, target, 0, true);
}
extern "C" int dup3(int fd, int target, int flags) noexcept {
  if (fd == target) {
    errno = EINVAL;
    return -1;
  }
  return lupine_device_proxy_dup(fd, target, flags, true);
}
extern "C" int lupine_fcntl_entry(int fd, int command,
                                  uintptr_t argument) __asm__("fcntl");
extern "C" int lupine_fcntl_entry(int fd, int command, uintptr_t argument) {
  if (command == F_DUPFD || command == F_DUPFD_CLOEXEC)
    return lupine_device_proxy_fcntl(fd, command, argument);
  static auto native = native_symbol<int (*)(int, int, uintptr_t)>("fcntl");
  return native(fd, command, argument);
}
extern "C" int lupine_fcntl64_entry(int fd, int command,
                                    uintptr_t argument) __asm__("fcntl64");
extern "C" int lupine_fcntl64_entry(int fd, int command, uintptr_t argument) {
  return lupine_fcntl_entry(fd, command, argument);
}
extern "C" void *mmap(void *address, size_t size, int protection, int flags,
                      int fd, off_t offset) noexcept {
  if (lupine_device_proxy_get(fd)) {
    errno = ENODEV;
    return MAP_FAILED;
  }
  static auto native =
      native_symbol<void *(*)(void *, size_t, int, int, int, off_t)>("mmap");
  return native(address, size, protection, flags, fd, offset);
}
extern "C" void *mmap64(void *address, size_t size, int protection, int flags,
                        int fd, off64_t offset) noexcept {
  return mmap(address, size, protection, flags, fd, offset);
}
