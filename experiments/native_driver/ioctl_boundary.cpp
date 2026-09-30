// GPU-test fixture only. Force NVOS02_PARAMETERS to cross a page boundary;
// this caught a real context-creation failure when the second page was skipped.
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" int boundary_ioctl(int fd, unsigned long command,
                              uintptr_t argument) __asm__("ioctl");
extern "C" int boundary_ioctl(int fd, unsigned long command,
                              uintptr_t argument) {
  static auto next = reinterpret_cast<int (*)(int, unsigned long, uintptr_t)>(
      dlsym(RTLD_NEXT, "ioctl"));
  if (command != 0xc0384627 || !argument)
    return next(fd, command, argument);
  constexpr size_t size = 56;
  void *memory = reinterpret_cast<void *>(
      syscall(SYS_mmap, nullptr, 8192, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (memory == MAP_FAILED)
    return -1;
  // Preserve all native resource IDs and nested pointers. Only the test's
  // ioctl argument buffer placement changes; both ends see that same address.
  auto *buffer = static_cast<unsigned char *>(memory) + 4096 - 24;
  std::memcpy(buffer, reinterpret_cast<void *>(argument), size);
  int result = next(fd, command, reinterpret_cast<uintptr_t>(buffer));
  int error = errno;
  std::memcpy(reinterpret_cast<void *>(argument), buffer, size);
  syscall(SYS_munmap, memory, 8192);
  errno = error;
  return result;
}
