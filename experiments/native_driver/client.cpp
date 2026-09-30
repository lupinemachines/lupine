#include "wire.h"
// Feasibility experiment, not a supported Lupine backend. Native libcuda owns
// CUDA bookkeeping in this process; this shim forwards Linux device operations.
// In particular, the MMIO signal handler below is research-only.
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <set>
#include <signal.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <vector>

static std::mutex mutex;
static std::set<int> descriptors;
static int peer = -1;
struct DeviceBlock {
  Range range;
  bool mmio, writable;
  std::vector<unsigned char> image;
  void *write_alias;
};
static std::vector<DeviceBlock> devices;
static thread_local bool transport = false;
static void install_mmio_trap();
static void retire_blocks(uint64_t begin, uint64_t end);
const char *native_driver_metadata_path(const char *path);
static bool proxy_fd(int fd) {
  std::lock_guard<std::mutex> lock(mutex);
  return descriptors.count(fd) != 0;
}
static bool device(const char *path) {
  return path && std::strncmp(path, "/dev/nvidia", 11) == 0;
}
static void connect_peer() {
  if (peer >= 0)
    return;
  peer = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(16130);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int nodelay = 1;
  setsockopt(peer, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  if (connect(peer, reinterpret_cast<sockaddr *>(&address), sizeof(address)) <
      0) {
    std::perror("connect");
    std::exit(2);
  }
}
static int device_open(const char *path, int flags) {
  std::lock_guard<std::mutex> lock(mutex);
  transport = true;
  connect_peer();
  int temporary = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  int fd = fcntl(temporary, F_DUPFD_CLOEXEC, 512);
  int error = errno;
  syscall(SYS_close, temporary);
  if (fd < 0) {
    transport = false;
    errno = error;
    return -1;
  }
  send_message(peer, {OPEN, uint32_t(std::strlen(path) + 1), uint64_t(fd),
                      uint64_t(flags)});
  transfer(peer, const_cast<char *>(path), std::strlen(path) + 1, true);
  Message response = receive_message(peer);
  if (response.count) {
    syscall(SYS_close, fd);
    fd = -1;
    errno = response.count;
  } else
    descriptors.insert(fd);
  transport = false;
  return fd;
}
static int open_path(const char *path, int flags, mode_t mode) {
  if (!transport && device(path))
    return device_open(path, flags);
  return syscall(SYS_openat, AT_FDCWD, native_driver_metadata_path(path), flags,
                 mode);
}
extern "C" int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list a;
    va_start(a, flags);
    mode = va_arg(a, unsigned);
    va_end(a);
  }
  return open_path(path, flags, mode);
}
extern "C" int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
    va_list a;
    va_start(a, flags);
    mode = va_arg(a, unsigned);
    va_end(a);
  }
  return open_path(path, flags, mode);
}
extern "C" int __open_2(const char *path, int flags) {
  return open_path(path, flags, 0);
}
extern "C" int __open64_2(const char *path, int flags) {
  return open_path(path, flags, 0);
}
extern "C" int close(int fd) {
  if (transport || !proxy_fd(fd))
    return syscall(SYS_close, fd);
  std::lock_guard<std::mutex> lock(mutex);
  transport = true;
  send_message(peer, {CLOSE, 0, uint64_t(fd), 0});
  receive_message(peer);
  descriptors.erase(fd);
  transport = false;
  return syscall(SYS_close, fd);
}
static std::vector<Range> client_ranges() {
  std::vector<Range> result;
  FILE *maps = fopen("/proc/self/maps", "r");
  if (!maps) {
    std::perror("client mappings");
    std::exit(2);
  }
  char line[1024];
  while (fgets(line, sizeof(line), maps)) {
    unsigned long begin, end;
    char access[5]{};
    if (sscanf(line, "%lx-%lx %4s", &begin, &end, access) == 3 &&
        access[0] == 'r' && begin < 0x800000000000ULL) {
      std::vector<Range> pieces{{begin, end}};
      for (const auto &block : devices)
        subtract_range(pieces, block.range);
      result.insert(result.end(), pieces.begin(), pieces.end());
    }
  }
  fclose(maps);
  return result;
}
extern "C" void *mmap(void *address, size_t length, int prot, int flags, int fd,
                      off_t offset) noexcept {
  if (transport || !proxy_fd(fd)) {
    if (!transport && (flags & MAP_FIXED))
      retire_blocks(reinterpret_cast<uint64_t>(address),
                    reinterpret_cast<uint64_t>(address) + length);
    return reinterpret_cast<void *>(
        syscall(SYS_mmap, address, length, prot, flags, fd, offset));
  }
  std::lock_guard<std::mutex> lock(mutex);
  transport = true;
  int backing = syscall(SYS_memfd_create, "native-gpu-block", 1);
  syscall(SYS_ftruncate, backing, length);
  int placement = flags & (MAP_FIXED | MAP_FIXED_NOREPLACE);
  void *p = reinterpret_cast<void *>(
      syscall(SYS_mmap, address, length, PROT_READ | PROT_WRITE,
              MAP_SHARED | placement, backing, 0));
  if (p == MAP_FAILED) {
    syscall(SYS_close, backing);
    transport = false;
    return p;
  }
  send_message(peer, {MAP, 0, reinterpret_cast<uint64_t>(p), length});
  Mapping info{uint32_t(fd), uint32_t(prot), uint32_t(flags), 0,
               uint64_t(offset)};
  transfer(peer, &info, sizeof(info), true);
  Message response = receive_message(peer);
  if (response.count) {
    syscall(SYS_munmap, p, length);
    syscall(SYS_close, backing);
    transport = false;
    errno = response.count;
    return MAP_FAILED;
  }
  Message bytes = receive_message(peer);
  transfer(peer, static_cast<char *>(p) + bytes.a, bytes.count, false);
  DeviceBlock block{
      {reinterpret_cast<uint64_t>(p), reinterpret_cast<uint64_t>(p) + length},
      length == 65536,
      (prot & PROT_WRITE) != 0,
      std::vector<unsigned char>(length),
      reinterpret_cast<void *>(syscall(SYS_mmap, nullptr, length,
                                       PROT_READ | PROT_WRITE, MAP_SHARED,
                                       backing, 0))};
  syscall(SYS_close, backing);
  std::memcpy(block.image.data(), p, length);
  devices.push_back(std::move(block));
  if (length == 65536)
    install_mmio_trap();
  syscall(SYS_mprotect, p, length, length == 65536 ? PROT_READ : prot);
  transport = false;
  return p;
}
extern "C" void *mmap64(void *address, size_t length, int prot, int flags,
                        int fd, off64_t offset) noexcept {
  return mmap(address, length, prot, flags, fd, offset);
}
static void retire_blocks(uint64_t begin, uint64_t end) {
  std::lock_guard<std::mutex> lock(mutex);
  transport = true;
  for (auto d = devices.begin(); d != devices.end();) {
    Range range = d->range;
    if (end <= range.begin || begin >= range.end) {
      ++d;
      continue;
    }
    if (begin > range.begin || end < range.end) {
      std::fprintf(stderr, "unsupported partial mapped-block retirement\n");
      std::_Exit(3);
    }
    send_message(peer, {UNMAP, 0, range.begin, range.end - range.begin});
    receive_message(peer);
    if (reinterpret_cast<uint64_t>(d->write_alias) != range.begin)
      syscall(SYS_munmap, d->write_alias, range.end - range.begin);
    d = devices.erase(d);
  }
  transport = false;
}
extern "C" int munmap(void *address, size_t length) noexcept {
  if (!transport)
    retire_blocks(reinterpret_cast<uint64_t>(address),
                  reinterpret_cast<uint64_t>(address) + length);
  return syscall(SYS_munmap, address, length);
}
extern "C" int mprotect(void *address, size_t length, int prot) noexcept {
  if (!transport && prot == PROT_NONE)
    retire_blocks(reinterpret_cast<uint64_t>(address),
                  reinterpret_cast<uint64_t>(address) + length);
  return syscall(SYS_mprotect, address, length, prot);
}
static void flush_blocks() {
  for (auto &block : devices) {
    if (!block.writable || block.mmio)
      continue;
    std::vector<unsigned char> snapshot(block.image.size());
    std::memcpy(snapshot.data(), block.write_alias, snapshot.size());
    changed_bytes(block.image.data(), snapshot.data(), snapshot.size(),
                  [&](size_t begin, size_t size) {
                    send_message(peer, {SYNC, uint32_t(size),
                                        block.range.begin + begin, 0});
                    transfer(peer, snapshot.data() + begin, size, true);
                    std::memcpy(block.image.data() + begin,
                                snapshot.data() + begin, size);
                  });
  }
}
static void apply_change(Message response) {
  std::vector<unsigned char> before(response.count), after(response.count);
  transfer(peer, before.data(), response.count, false);
  transfer(peer, after.data(), response.count, false);
  uint64_t destination = response.a;
  for (auto &block : devices)
    if (response.a >= block.range.begin &&
        response.a + response.count <= block.range.end)
      destination = reinterpret_cast<uint64_t>(block.write_alias) + response.a -
                    block.range.begin;
  // Do not copy the whole ioctl page: it can include unrelated, changing
  // client stack data next to the kernel's output fields.
  changed_bytes(before.data(), after.data(), response.count,
                [&](size_t begin, size_t size) {
                  iovec local{after.data() + begin, size},
                      remote{reinterpret_cast<void *>(destination + begin),
                             size};
                  if (process_vm_writev(getpid(), &local, 1, &remote, 1, 0) !=
                      ssize_t(size)) {
                    std::perror("write client range");
                    std::exit(2);
                  }
                });
  for (auto &block : devices)
    if (response.a >= block.range.begin &&
        response.a + response.count <= block.range.end)
      std::memcpy(block.image.data() + response.a - block.range.begin,
                  after.data(), response.count);
}
// Research-only x86 MMIO emulation. Do not treat this signal-handler transport
// as production-safe: it uses a mutex and supports only observed 32-bit MOVs.
static struct sigaction previous_segv {};
static void mmio_fault(int, siginfo_t *info, void *state) {
  auto *context = static_cast<ucontext_t *>(state);
  uint64_t address = reinterpret_cast<uint64_t>(info->si_addr);
  for (auto &block : devices) {
    if (!block.mmio || address != block.range.begin + 0x90)
      continue;
    auto *instruction =
        reinterpret_cast<unsigned char *>(context->uc_mcontext.gregs[REG_RIP]);
    size_t length = 0;
    unsigned rex = 0;
    if ((instruction[length] & 0xf0) == 0x40)
      rex = instruction[length++];
    if (instruction[length++] != 0x89 || (rex & 8))
      break;
    unsigned modrm = instruction[length++];
    unsigned mode = modrm >> 6, base = modrm & 7;
    unsigned source = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
    if (mode == 3)
      break;
    if (base == 4)
      base = instruction[length++] & 7;
    if (mode == 1)
      ++length;
    if (mode == 2 || (mode == 0 && base == 5))
      length += 4;
    static const int registers[] = {
        REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
        REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
    uint32_t value = context->uc_mcontext.gregs[registers[source]];
    std::lock_guard<std::mutex> lock(mutex);
    transport = true;
    std::memcpy(static_cast<char *>(block.write_alias) + 0x90, &value, 4);
    flush_blocks();
    send_message(peer, {SYNC, 4, address, 1});
    transfer(peer, &value, 4, true);
    send_message(peer, {POLL, 0, 0, 0});
    for (;;) {
      Message response = receive_message(peer);
      if (response.kind == CHANGE) {
        apply_change(response);
        continue;
      }
      if (response.kind != DONE || response.count)
        std::_Exit(3);
      break;
    }
    context->uc_mcontext.gregs[REG_RIP] += length;
    transport = false;
    return;
  }
  std::fprintf(
      stderr, "unhandled fault address=%p rip=%llx instruction=", info->si_addr,
      (unsigned long long)context->uc_mcontext.gregs[REG_RIP]);
  auto *code =
      reinterpret_cast<unsigned char *>(context->uc_mcontext.gregs[REG_RIP]);
  for (int i = 0; i < 12; ++i)
    std::fprintf(stderr, "%02x", code[i]);
  std::fprintf(stderr, "\n");
  std::_Exit(3);
}
static void install_mmio_trap() {
  static bool installed = false;
  if (installed)
    return;
  struct sigaction action {};
  action.sa_sigaction = mmio_fault;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, &previous_segv);
  installed = true;
}
extern "C" int proxy_ioctl(int fd, unsigned long command,
                           uintptr_t argument) __asm__("ioctl");
extern "C" int proxy_ioctl(int fd, unsigned long command, uintptr_t argument) {
  if (transport || !proxy_fd(fd))
    return syscall(SYS_ioctl, fd, command, argument);
  std::lock_guard<std::mutex> lock(mutex);
  transport = true;
  flush_blocks();
  auto ranges = client_ranges();
  send_message(peer, {IOCTL, uint32_t(ranges.size()), uint64_t(fd), command});
  transfer(peer, ranges.data(), ranges.size() * sizeof(Range), true);
  transfer(peer, &argument, sizeof(argument), true);
  for (;;) {
    Message response = receive_message(peer);
    if (response.kind == PAGE) {
      std::vector<unsigned char> bytes(response.count);
      iovec local{bytes.data(), bytes.size()},
          remote{reinterpret_cast<void *>(response.a), bytes.size()};
      if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) !=
          ssize_t(bytes.size())) {
        std::perror("read client page");
        std::exit(2);
      }
      transfer(peer, bytes.data(), bytes.size(), true);
      continue;
    }
    if (response.kind == REGISTER) {
      Range range{response.a, response.a + response.count};
      DeviceBlock block{range, false, true,
                        std::vector<unsigned char>(response.count),
                        reinterpret_cast<void *>(response.a)};
      std::memcpy(block.image.data(), block.write_alias, response.count);
      devices.push_back(std::move(block));
      continue;
    }
    if (response.kind == CHANGE) {
      apply_change(response);
      continue;
    }
    if (response.kind != DONE)
      std::exit(2);
    transport = false;
    if (response.count)
      errno = response.count;
    return int(response.a);
  }
}

extern "C" int poll(struct pollfd *fds, nfds_t count, int timeout) {
  if (transport)
    return syscall(SYS_poll, fds, count, timeout);
  bool remote = false;
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (nfds_t i = 0; i < count; ++i)
      if (descriptors.count(fds[i].fd))
        remote = true;
  }
  if (!remote)
    return syscall(SYS_poll, fds, count, timeout);
  auto started = std::chrono::steady_clock::now();
  for (;;) {
    int ready = 0;
    {
      std::lock_guard<std::mutex> lock(mutex);
      transport = true;
      std::vector<pollfd> native(fds, fds + count), local(fds, fds + count);
      for (nfds_t i = 0; i < count; ++i) {
        native[i].revents = local[i].revents = 0;
        if (descriptors.count(fds[i].fd))
          local[i].fd = -1;
        else
          native[i].fd = -1;
      }
      int result = syscall(SYS_poll, local.data(), count, 0);
      if (result < 0) {
        transport = false;
        return result;
      }
      flush_blocks();
      send_message(peer, {POLL, uint32_t(count), 0, 0});
      transfer(peer, native.data(), native.size() * sizeof(pollfd), true);
      for (;;) {
        Message response = receive_message(peer);
        if (response.kind == CHANGE) {
          apply_change(response);
          continue;
        }
        if (response.kind != DONE)
          std::exit(2);
        if (response.count) {
          transport = false;
          errno = response.count;
          return -1;
        }
        break;
      }
      transfer(peer, native.data(), native.size() * sizeof(pollfd), false);
      for (nfds_t i = 0; i < count; ++i) {
        fds[i].revents = native[i].revents | local[i].revents;
        if (fds[i].revents)
          ++ready;
      }
      transport = false;
    }
    if (ready)
      return ready;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - started)
                       .count();
    if (timeout >= 0 && elapsed >= timeout)
      return 0;
    usleep(10000);
  }
}
