#include "device_ioctl.h"
#include "rpc.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <sys/ioctl.h>
#include <unistd.h>
#include <unordered_map>

namespace {
struct req_hdr {
  uint64_t token;
  uint32_t command;
  uint32_t size;
  uint64_t rm_token;
};
static_assert(sizeof(req_hdr) == 24);

struct resp_hdr {
  int32_t result;
  int32_t error;
};
static_assert(sizeof(resp_hdr) == 8);

struct descriptor {
  int fd;
  bool uvm;
  ~descriptor() { close(fd); }
};
std::mutex descriptors_mutex;
std::unordered_map<uint64_t, std::shared_ptr<descriptor>> descriptors;
uint64_t next_token = 1;
std::atomic<int> cached_fd[2]{-1, -1};

uint64_t retain(int fd, bool uvm) {
  int duplicate = fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (duplicate < 0)
    return 0;
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  if (descriptors.size() >= 4096) {
    close(duplicate);
    errno = EMFILE;
    return 0;
  }
  uint64_t token = next_token++;
  descriptors.emplace(
      token, std::shared_ptr<descriptor>(new descriptor{duplicate, uvm}));
  return token;
}
std::shared_ptr<descriptor> lookup(uint64_t token) {
  auto found = descriptors.find(token);
  return found == descriptors.end() ? nullptr : found->second;
}
void release(uint64_t token) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors.erase(token);
}
bool is_target_fd(int fd, const char *expected) {
  char path[64];
  int written = std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(path))
    return false;
  char target[PATH_MAX];
  ssize_t size = readlink(path, target, sizeof(target) - 1);
  if (size < 0)
    return false;
  target[size] = 0;
  return std::strcmp(target, expected) == 0;
}
int find_native_fd(bool uvm) {
  const char *expected = uvm ? "/dev/nvidia-uvm" : "/dev/nvidiactl";
  int cached = cached_fd[uvm].load(std::memory_order_relaxed);
  if (cached >= 0 && is_target_fd(cached, expected))
    return cached;
  DIR *directory = opendir("/proc/self/fd");
  if (!directory)
    return -1;
  int result = -1;
  while (dirent *entry = readdir(directory)) {
    char *end;
    long value = strtol(entry->d_name, &end, 10);
    if (*end || value < 0 || value > INT_MAX)
      continue;
    if (is_target_fd(static_cast<int>(value), expected)) {
      result = static_cast<int>(value);
      break;
    }
  }
  closedir(directory);
  if (result >= 0)
    cached_fd[uvm].store(result, std::memory_order_relaxed);
  else
    errno = ENODEV;
  return result;
}
} // namespace

int handle_lupineDeviceOpen(conn_t *conn) {
  uint32_t kind;
  if (rpc_read(conn, &kind, sizeof(kind)) < 0)
    return -1;
  int request = rpc_read_end(conn);
  if (request < 0)
    return -1;
  uint64_t token = 0;
  int32_t error = EINVAL;
  if (kind <= 1) {
    int fd = find_native_fd(kind == 1);
    if (fd >= 0)
      token = retain(fd, kind == 1);
    error = token ? 0 : errno;
  }
  return rpc_write_start_response(conn, request) < 0 ||
                 rpc_write(conn, &token, sizeof(token)) < 0 ||
                 rpc_write(conn, &error, sizeof(error)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
int handle_lupineDeviceClose(conn_t *conn) {
  uint64_t token;
  if (rpc_read(conn, &token, sizeof(token)) < 0)
    return -1;
  int request = rpc_read_end(conn);
  if (request < 0)
    return -1;
  release(token);
  int32_t result = 0;
  return rpc_write_start_response(conn, request) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
int handle_lupineDeviceIoctl(conn_t *conn) {
  req_hdr req;
  std::array<unsigned char, lupine_uvm::map_size> buffer;
  if (rpc_read(conn, &req, sizeof(req)) < 0)
    return -1;
  auto layout = lupine_uvm::describe(req.command);
  if (!layout.size || req.size != layout.size)
    return -1;
  if (rpc_read(conn, buffer.data(), req.size) < 0)
    return -1;
  int request = rpc_read_end(conn);
  if (request < 0)
    return -1;
  std::shared_ptr<descriptor> fd, rm;
  {
    std::lock_guard<std::mutex> lock(descriptors_mutex);
    fd = lookup(req.token);
    if (fd && fd->uvm && req.command == lupine_uvm::map)
      rm = lookup(req.rm_token);
  }
  int32_t result = -1, error = EBADF;
  if (fd && fd->uvm) {
    bool valid = req.command != lupine_uvm::map || (rm && !rm->uvm);
    if (valid) {
      // RM handles belong to the server's native driver. Translate only the
      // proxy descriptor and leave handle validation to that driver.
      if (req.command == lupine_uvm::map)
        lupine_uvm::put(buffer.data(), lupine_uvm::rm_fd_offset,
                        static_cast<int32_t>(rm->fd));
      result = ioctl(fd->fd, req.command, buffer.data());
      error = result < 0 ? errno : 0;
    }
  }
  resp_hdr resp{result, error};
  return rpc_write_start_response(conn, request) < 0 ||
                 rpc_write(conn, &resp, sizeof(resp)) < 0 ||
                 rpc_write(conn, buffer.data(), req.size) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
