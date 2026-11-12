#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"
#include "device_ioctl.h"
#include "device_ioctl_proxy.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

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

bool copy_argument(void *destination, const void *source, size_t size,
                   bool write) {
  iovec local{write ? const_cast<void *>(source) : destination, size};
  iovec remote{write ? destination : const_cast<void *>(source), size};
  ssize_t copied = write ? process_vm_writev(getpid(), &local, 1, &remote, 1, 0)
                         : process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
  if (copied != static_cast<ssize_t>(size)) {
    errno = EFAULT;
    return false;
  }
  return true;
}
conn_t *connection(int route) {
  return lupine_route_remote_conn(lupine_route_from_identity(route));
}
bool finish(conn_t *conn) {
  if (rpc_wait_for_response(conn) < 0) {
    errno = EIO;
    return false;
  }
  return true;
}
} // namespace

void lupine_device_proxy_remote_close(int route, uint64_t token) {
  conn_t *conn = connection(route);
  int32_t result;
  if (!conn || conn->closed || lupine_prepare_rpc(conn) != 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineDeviceClose) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 || !finish(conn))
    return;
  (void)rpc_read(conn, &result, sizeof(result));
  (void)rpc_read_end(conn);
}

int lupine_device_open(const char *path, int flags) {
  if (flags & (O_CREAT | O_TRUNC | O_EXCL) ||
      (flags & O_TMPFILE) == O_TMPFILE) {
    errno = EINVAL;
    return -1;
  }
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_is_local(route))
    return static_cast<int>(syscall(SYS_openat, AT_FDCWD, path, flags, 0));
  conn_t *conn = lupine_route_remote_conn(route);
  uint32_t kind = std::strcmp(path, "/dev/nvidia-uvm") == 0 ? 1 : 0;
  uint64_t token = 0;
  int32_t error = 0;
  if (!conn || lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineDeviceOpen) < 0 ||
      rpc_write(conn, &kind, sizeof(kind)) < 0 || !finish(conn) ||
      rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &error, sizeof(error)) < 0 || rpc_read_end(conn) < 0) {
    errno = EIO;
    return -1;
  }
  if (error) {
    errno = error;
    return -1;
  }
  int fd = lupine_device_proxy_create(lupine_route_identity(route), token,
                                      kind == 1);
  if (fd < 0)
    lupine_device_proxy_remote_close(lupine_route_identity(route), token);
  if (fd >= 0 && !(flags & O_CLOEXEC))
    (void)syscall(SYS_fcntl, fd, F_SETFD, 0);
  return fd;
}

int lupine_device_ioctl(int fd, uint32_t command, void *argument) {
  auto proxy = lupine_device_proxy_get(fd);
  auto layout = lupine_uvm::describe(command);
  if (!proxy || !proxy->uvm || !layout.size) {
    errno = ENOTTY;
    return -1;
  }
  std::array<unsigned char, lupine_uvm::map_size> buffer;
  if (!copy_argument(buffer.data(), argument, layout.size, false))
    return -1;
  uint64_t rm_token = 0;
  int32_t original_rm_fd = -1;
  std::shared_ptr<lupine_device_proxy> rm;
  if (command == lupine_uvm::map) {
    original_rm_fd =
        lupine_uvm::get<int32_t>(buffer.data(), lupine_uvm::rm_fd_offset);
    rm = lupine_device_proxy_get(original_rm_fd);
    if (!rm || rm->uvm || rm->route != proxy->route) {
      errno = EBADF;
      return -1;
    }
    rm_token = rm->token;
  }
  conn_t *conn = connection(proxy->route);
  uint32_t size = static_cast<uint32_t>(layout.size);
  req_hdr req{proxy->token, command, size, rm_token};
  resp_hdr resp{-1, 0};
  if (!conn || lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineDeviceIoctl) < 0 ||
      rpc_write(conn, &req, sizeof(req)) < 0 ||
      rpc_write(conn, buffer.data(), size) < 0 || !finish(conn) ||
      rpc_read(conn, &resp, sizeof(resp)) < 0 ||
      rpc_read(conn, buffer.data(), size) < 0 || rpc_read_end(conn) < 0) {
    errno = EIO;
    return -1;
  }
  if (command == lupine_uvm::map)
    lupine_uvm::put(buffer.data(), lupine_uvm::rm_fd_offset, original_rm_fd);
  if (!copy_argument(argument, buffer.data(), size, true))
    return -1;
  if (resp.result == 0 &&
      lupine_uvm::get<uint32_t>(buffer.data(), layout.status) == 0) {
    auto address = lupine_uvm::get<uint64_t>(buffer.data(), 0);
    if (command == lupine_uvm::map || command == lupine_uvm::sparse)
      lupine_note_deviceptr_allocation_route(
          address, lupine_uvm::get<uint64_t>(buffer.data(), 8),
          lupine_route_from_identity(proxy->route));
    // An external unmap may remove only part of a range. Keep its routing
    // identity until UVM_FREE releases the reservation.
    if (command == lupine_uvm::free)
      lupine_forget_deviceptr_owner(address);
  }
  if (resp.result < 0)
    errno = resp.error;
  return resp.result;
}
