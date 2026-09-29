#include "device_ioctl.h"
#include "device_ioctl_capture.h"
#include "rpc.h"
#include "server_checkpoint.h"

#include <array>
#include <cerrno>
#include <climits>
#include <cuda.h>
#include <dirent.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <unordered_map>

namespace {
struct descriptor {
  int fd;
  bool uvm;
  uint32_t client, memory;
  ~descriptor() { close(fd); }
};
std::mutex descriptors_mutex;
std::unordered_map<uint64_t, std::shared_ptr<descriptor>> descriptors;
uint64_t next_token = 1;

uint64_t retain(int fd, bool uvm, uint32_t client = 0, uint32_t memory = 0) {
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
  descriptors.emplace(token, std::shared_ptr<descriptor>(new descriptor{
                                 duplicate, uvm, client, memory}));
  return token;
}
std::shared_ptr<descriptor> lookup(uint64_t token) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto found = descriptors.find(token);
  return found == descriptors.end() ? nullptr : found->second;
}
void release(uint64_t token) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors.erase(token);
}
int find_native_fd(bool uvm) {
  DIR *directory = opendir("/proc/self/fd");
  if (!directory)
    return -1;
  int result = -1;
  while (dirent *entry = readdir(directory)) {
    char *end;
    long value = strtol(entry->d_name, &end, 10);
    if (*end || value < 0 || value > INT_MAX)
      continue;
    std::string path = std::string("/proc/self/fd/") + entry->d_name;
    char target[PATH_MAX];
    ssize_t size = readlink(path.c_str(), target, sizeof(target) - 1);
    if (size < 0)
      continue;
    target[size] = 0;
    if (std::string(target) == (uvm ? "/dev/nvidia-uvm" : "/dev/nvidiactl")) {
      result = static_cast<int>(value);
      break;
    }
  }
  closedir(directory);
  if (result < 0)
    errno = ENODEV;
  return result;
}
struct allocation_operation {
  CUmemGenericAllocationHandle handle = 0;
  uint64_t size;
  int device;
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
};
int allocate(void *argument) {
  auto &allocation = *static_cast<allocation_operation *>(argument);
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = allocation.device;
  allocation.result =
      cuMemCreate(&allocation.handle, allocation.size, &prop, 0);
  return allocation.result;
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
  uint64_t token, rm_token;
  uint32_t command, size;
  std::array<unsigned char, lupine_uvm::map_size> buffer{};
  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &command, sizeof(command)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0 ||
      rpc_read(conn, &rm_token, sizeof(rm_token)) < 0)
    return -1;
  auto layout = lupine_uvm::describe(command);
  if (!layout.size || size != layout.size)
    return -1;
  if (rpc_read(conn, buffer.data(), size) < 0)
    return -1;
  int request = rpc_read_end(conn);
  if (request < 0)
    return -1;
  auto fd = lookup(token);
  auto rm = lookup(rm_token);
  int32_t result = -1, error = EBADF;
  if (fd && fd->uvm) {
    bool valid =
        command != lupine_uvm::map ||
        (rm && !rm->uvm && rm->client && rm->memory &&
         rm->client == lupine_uvm::get<uint32_t>(
                           buffer.data(), lupine_uvm::rm_client_offset) &&
         rm->memory == lupine_uvm::get<uint32_t>(buffer.data(),
                                                 lupine_uvm::rm_memory_offset));
    if (valid) {
      if (command == lupine_uvm::map)
        lupine_uvm::put(buffer.data(), lupine_uvm::rm_fd_offset,
                        static_cast<int32_t>(rm->fd));
      result = ioctl(fd->fd, command, buffer.data());
      error = result < 0 ? errno : 0;
    }
  }
  return rpc_write_start_response(conn, request) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write(conn, &error, sizeof(error)) < 0 ||
                 rpc_write(conn, buffer.data(), size) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
int handle_lupineUvmCreate(conn_t *conn) {
  uint64_t size;
  int32_t device;
  if (rpc_read(conn, &size, sizeof(size)) < 0 ||
      rpc_read(conn, &device, sizeof(device)) < 0)
    return -1;
  int request = rpc_read_end(conn);
  if (request < 0)
    return -1;
  allocation_operation allocation{0, size, device};
  uint64_t rm_token = 0, uvm_token = 0;
  uint32_t client = 0, memory = 0;
  int fd = -1;
  int error = EINVAL;
  if (size && size <= SIZE_MAX && device >= 0) {
    error = lupine_server_checkpoint_capture_rm(allocate, &allocation, &fd,
                                                &client, &memory);
    if (error == ENOSYS)
      error =
          lupine_capture_rm_alloc(allocate, &allocation, &fd, &client, &memory);
    if (!error && allocation.result == CUDA_SUCCESS) {
      int uvm = find_native_fd(true);
      if (uvm >= 0) {
        rm_token = retain(fd, false, client, memory);
        uvm_token = retain(uvm, true);
      }
      if (!rm_token || !uvm_token)
        error = errno;
    }
  }
  if (error) {
    if (allocation.handle)
      (void)cuMemRelease(allocation.handle);
    allocation.handle = 0;
    release(rm_token);
    release(uvm_token);
    rm_token = uvm_token = 0;
    if (allocation.result == CUDA_SUCCESS)
      allocation.result = CUDA_ERROR_NOT_SUPPORTED;
  }
  int32_t result = allocation.result;
  uint64_t handle = allocation.handle;
  return rpc_write_start_response(conn, request) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write(conn, &handle, sizeof(handle)) < 0 ||
                 rpc_write(conn, &rm_token, sizeof(rm_token)) < 0 ||
                 rpc_write(conn, &uvm_token, sizeof(uvm_token)) < 0 ||
                 rpc_write(conn, &client, sizeof(client)) < 0 ||
                 rpc_write(conn, &memory, sizeof(memory)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
