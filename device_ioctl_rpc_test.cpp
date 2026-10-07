#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"
#include "device_ioctl.h"
#include "device_ioctl_proxy.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

int handle_lupineDeviceOpen(conn_t *);
int handle_lupineDeviceClose(conn_t *);
int handle_lupineDeviceIoctl(conn_t *);

static conn_t connection{};
static int native_uvm, native_rm;
static std::atomic<int> native_calls, ioctl_error;
static std::atomic<uint32_t> driver_status;
static uint64_t tracked_address;

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s (errno=%d)\n", message, errno);
    std::exit(1);
  }
}

// Production HTTP/2 and RPC transport, with only the native GPU operations
// mocked. A single lane preserves the client's normal request lifecycle.
static void serve(conn_t *conn) {
  require(rpc_http2_server_init_with_metadata(conn, nullptr) == 0,
          "server h2 init");
  int32_t stream = rpc_http2_accept_stream(conn);
  require(stream >= 0 && rpc_bind_http2_stream(conn, stream) == 0,
          "server lane");
  for (;;) {
    int operation = rpc_dispatch(conn, 0);
    if (operation < 0)
      break;
    int result = -1;
    switch (operation) {
    case LUPINE_RPC_lupineDeviceOpen:
      result = handle_lupineDeviceOpen(conn);
      break;
    case LUPINE_RPC_lupineDeviceClose:
      result = handle_lupineDeviceClose(conn);
      break;
    case LUPINE_RPC_lupineDeviceIoctl:
      result = handle_lupineDeviceIoctl(conn);
      break;
    }
    require(result == 0, "server handler response");
  }
  rpc_unbind_http2_stream(conn);
}

extern "C" int lupine_prepare_rpc(conn_t *) { return 0; }
extern "C" lupine_route lupine_route_for_current_context() {
  return {LUPINE_ROUTE_REMOTE, &connection};
}
extern "C" bool lupine_route_is_local(lupine_route) { return false; }
extern "C" conn_t *lupine_route_remote_conn(lupine_route route) {
  return route.conn;
}
int lupine_route_identity(lupine_route) { return 2; }
lupine_route lupine_route_from_identity(int) {
  return lupine_route_for_current_context();
}
extern "C" void lupine_note_deviceptr_allocation_route(CUdeviceptr address,
                                                       size_t size,
                                                       lupine_route route) {
  require(size == 8192 && route.conn == &connection, "mapping route and size");
  tracked_address = address;
}
extern "C" void lupine_forget_deviceptr_owner(CUdeviceptr address) {
  require(address == tracked_address, "free removes mapped route");
  tracked_address = 0;
}

extern "C" ssize_t readlink(const char *path, char *buffer,
                            size_t size) noexcept {
  constexpr const char *prefix = "/proc/self/fd/";
  if (std::strncmp(path, prefix, std::strlen(prefix)) == 0) {
    int fd = std::atoi(path + std::strlen(prefix));
    const char *target = fd == native_uvm  ? "/dev/nvidia-uvm"
                         : fd == native_rm ? "/dev/nvidiactl"
                                           : nullptr;
    if (target) {
      size_t length = std::min(size, std::strlen(target));
      std::memcpy(buffer, target, length);
      return static_cast<ssize_t>(length);
    }
  }
  return syscall(SYS_readlinkat, AT_FDCWD, path, buffer, size);
}
extern "C" int ioctl(int fd, unsigned long command, ...) noexcept {
  va_list arguments;
  va_start(arguments, command);
  auto *buffer = va_arg(arguments, unsigned char *);
  va_end(arguments);
  require(fd != native_uvm && fcntl(fd, F_GETFD) >= 0,
          "server owns a duplicate native descriptor");
  ++native_calls;
  if (command == lupine_uvm::map) {
    int rm = lupine_uvm::get<int32_t>(buffer, lupine_uvm::rm_fd_offset);
    require(rm != native_rm && fcntl(rm, F_GETFD) >= 0,
            "nested descriptor translated");
    require(lupine_uvm::get<uint32_t>(buffer, lupine_uvm::rm_client_offset) ==
                    123 &&
                lupine_uvm::get<uint32_t>(buffer,
                                          lupine_uvm::rm_memory_offset) == 456,
            "RM handles forwarded to native driver");
  }
  lupine_uvm::put(buffer, lupine_uvm::describe(command).status,
                  driver_status.load());
  if (ioctl_error) {
    errno = ioctl_error;
    return -1;
  }
  return 0;
}

int main() {
  native_uvm = memfd_create("native-uvm", MFD_CLOEXEC);
  native_rm = memfd_create("native-rm", MFD_CLOEXEC);
  require(native_uvm >= 0 && native_rm >= 0, "native stand-ins");
  int sockets[2];
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socket pair");
  conn_t server_connection{};
  require(rpc_conn_init(&connection, sockets[0], 0) == 0 &&
              rpc_conn_init(&server_connection, sockets[1], 1) == 0,
          "RPC connections");
  std::thread server(serve, &server_connection);
  require(rpc_http2_client_init(&connection) == 0, "client h2 init");
  int uvm_fd = lupine_device_open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  int rm_fd = lupine_device_open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  require(uvm_fd >= 0 && rm_fd >= 0, "open initialized device descriptors");
  std::array<unsigned char, lupine_uvm::map_size> map{};
  lupine_uvm::put(map.data(), 0, uint64_t{0x100000000});
  lupine_uvm::put(map.data(), 8, uint64_t{8192});
  lupine_uvm::put(map.data(), lupine_uvm::rm_fd_offset, rm_fd);
  lupine_uvm::put(map.data(), lupine_uvm::rm_client_offset, uint32_t{123});
  lupine_uvm::put(map.data(), lupine_uvm::rm_memory_offset, uint32_t{456});
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == 0,
          "map RPC");
  require(tracked_address == 0x100000000 &&
              lupine_uvm::get<int32_t>(map.data(), lupine_uvm::rm_fd_offset) ==
                  rm_fd,
          "client descriptor and route preserved");
  driver_status = 42;
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == 0 &&
              lupine_uvm::get<uint32_t>(map.data(), 9260) == 42,
          "driver status copied back");
  driver_status = 0;
  ioctl_error = EACCES;
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == -1 &&
              errno == EACCES,
          "syscall errno copied back");
  ioctl_error = 0;
  int previous_calls = native_calls;
  require(lupine_device_ioctl(uvm_fd, 999, nullptr) == -1 && errno == ENOTTY,
          "unknown ioctl");
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map,
                              reinterpret_cast<void *>(1)) == -1 &&
              errno == EFAULT,
          "bad caller pointer");
  lupine_uvm::put(map.data(), lupine_uvm::rm_fd_offset, native_rm);
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == -1 &&
              errno == EBADF,
          "ordinary nested descriptor rejected");
  lupine_uvm::put(map.data(), lupine_uvm::rm_fd_offset, uvm_fd);
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == -1 &&
              errno == EBADF,
          "UVM nested descriptor rejected");
  require(native_calls == previous_calls,
          "invalid requests stay out of native driver");
  auto other = lupine_device_proxy_create(99, 1234, false);
  lupine_uvm::put(map.data(), lupine_uvm::rm_fd_offset, other);
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::map, map.data()) == -1 &&
              errno == EBADF,
          "cross-route RM descriptor rejected");
  lupine_device_proxy_close(other);
  std::array<unsigned char, 40> unmap{};
  lupine_uvm::put(unmap.data(), 0, tracked_address);
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::unmap, unmap.data()) == 0 &&
              tracked_address,
          "partial unmap retains routing");
  require(lupine_device_ioctl(uvm_fd, lupine_uvm::free, unmap.data()) == 0 &&
              !tracked_address,
          "free retires routing");
  lupine_device_proxy_close(rm_fd);
  lupine_device_proxy_close(uvm_fd);
  close(native_uvm);
  close(native_rm);
  rpc_close_transport_socket(&connection);
  server.join();
  rpc_conn_destroy(&connection);
  rpc_conn_destroy(&server_connection);
  close(server_connection.connfd);
  return 0;
}
