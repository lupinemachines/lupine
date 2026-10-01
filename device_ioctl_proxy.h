#pragma once

#include <cstdint>
#include <memory>

struct lupine_device_proxy {
  int route;
  uint64_t token;
  bool uvm;
};

int lupine_device_proxy_create(int route, uint64_t token, bool uvm);
std::shared_ptr<lupine_device_proxy> lupine_device_proxy_get(int fd);
// Hooks use the raw syscalls internally so RPC transport descriptors are never
// interposed recursively. Duplicates retain the same remote descriptor.
int lupine_device_proxy_close(int fd);
int lupine_device_proxy_dup(int fd, int destination, int flags, bool fixed);
int lupine_device_proxy_fcntl(int fd, int command, uintptr_t argument);
void lupine_device_proxy_remote_close(int route, uint64_t token);
int lupine_device_open(const char *path, int flags);
int lupine_device_ioctl(int fd, uint32_t command, void *argument);
