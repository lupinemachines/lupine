#include "device_ioctl_proxy.h"

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>

namespace {
std::atomic<size_t> active_proxies{0};

struct registry {
  pid_t process = getpid();
  std::mutex mutex;
  std::unordered_map<int, std::shared_ptr<lupine_device_proxy>> descriptors;
};
registry &proxies() {
  // Libc teardown can call close after other static destructors have run.
  static auto *state = new registry;
  return *state;
}
// A fork child cannot use the parent's RPC threads or inherited mutex.
// Treat inherited proxies as ordinary fds so pre-exec close/dup stay safe.
registry *active_registry(int fd = 0) {
  if (fd < 0 || active_proxies.load(std::memory_order_acquire) == 0)
    return nullptr;
  auto &state = proxies();
  return state.process == getpid() ? &state : nullptr;
}
void publish_count(const registry &state) {
  active_proxies.store(state.descriptors.size(), std::memory_order_release);
}
int raw_dup(int fd, int destination, int flags, bool fixed) {
  if (!fixed)
    return static_cast<int>(syscall(SYS_fcntl, fd, F_DUPFD, 0));
  if (fd == destination && flags == 0)
    return static_cast<int>(syscall(SYS_fcntl, fd, F_GETFD, 0)) < 0 ? -1 : fd;
  return static_cast<int>(syscall(SYS_dup3, fd, destination, flags));
}
} // namespace

int lupine_device_proxy_create(int route, uint64_t token, bool uvm) {
  if (proxies().process != getpid()) {
    errno = ENOTSUP;
    return -1;
  }
  int fd = static_cast<int>(syscall(SYS_memfd_create,
                                    uvm ? "/dev/nvidia-uvm" : "/dev/nvidiactl",
                                    MFD_CLOEXEC));
  if (fd < 0)
    return -1;
  auto proxy = std::shared_ptr<lupine_device_proxy>(
      new lupine_device_proxy{route, token, uvm},
      [](lupine_device_proxy *value) {
        int saved_errno = errno;
        lupine_device_proxy_remote_close(value->route, value->token);
        delete value;
        errno = saved_errno;
      });
  auto &state = proxies();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.descriptors[fd] = std::move(proxy);
  publish_count(state);
  return fd;
}

std::shared_ptr<lupine_device_proxy> lupine_device_proxy_get(int fd) {
  auto *state = active_registry(fd);
  if (!state)
    return nullptr;
  std::lock_guard<std::mutex> lock(state->mutex);
  auto found = state->descriptors.find(fd);
  return found == state->descriptors.end() ? nullptr : found->second;
}

bool lupine_device_proxy_try_close(int fd, int *result) {
  auto *state = active_registry(fd);
  if (!state)
    return false;
  std::shared_ptr<lupine_device_proxy> removed;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto found = state->descriptors.find(fd);
    if (found == state->descriptors.end())
      return false;
    // Linux releases the descriptor even when close reports EINTR/EIO.
    *result = static_cast<int>(syscall(SYS_close, fd));
    removed = std::move(found->second);
    state->descriptors.erase(found);
    publish_count(*state);
  }
  return true;
}

int lupine_device_proxy_close(int fd) {
  int result;
  return lupine_device_proxy_try_close(fd, &result)
             ? result
             : static_cast<int>(syscall(SYS_close, fd));
}

int lupine_device_proxy_dup(int fd, int destination, int flags, bool fixed) {
  auto *state = active_registry();
  if (!state)
    return raw_dup(fd, destination, flags, fixed);
  std::shared_ptr<lupine_device_proxy> replaced;
  int result;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    result = raw_dup(fd, destination, flags, fixed);
    if (result < 0 || result == fd)
      return result;
    auto source = state->descriptors.find(fd);
    auto owner = source == state->descriptors.end() ? nullptr : source->second;
    auto target = state->descriptors.find(result);
    if (target != state->descriptors.end()) {
      replaced = std::move(target->second);
      state->descriptors.erase(target);
    }
    if (owner)
      state->descriptors.emplace(result, std::move(owner));
    publish_count(*state);
  }
  return result;
}

int lupine_device_proxy_fcntl(int fd, int command, uintptr_t argument) {
  auto *state = active_registry(fd);
  if (!state)
    return static_cast<int>(syscall(SYS_fcntl, fd, command, argument));
  std::lock_guard<std::mutex> lock(state->mutex);
  int result = static_cast<int>(syscall(SYS_fcntl, fd, command, argument));
  if (result >= 0 && (command == F_DUPFD || command == F_DUPFD_CLOEXEC)) {
    auto found = state->descriptors.find(fd);
    if (found != state->descriptors.end()) {
      state->descriptors.emplace(result, found->second);
      publish_count(*state);
    }
  }
  return result;
}
