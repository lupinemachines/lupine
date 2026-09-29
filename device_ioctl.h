#pragma once

#include <stddef.h>
#include <stdint.h>

// Optional Linux extension, resolved with dlsym from the CUDA driver library.
// Select a context on the requested remote route before calling. Descriptors
// are owned local proxies: close RM after mapping, retain UVM for teardown.
// The returned CUDA handle remains owned by the caller, including when proxy
// creation fails after a successful native allocation.
typedef struct lupine_uvm_allocation_v1 {
  size_t struct_size;
  uint64_t handle;
  int32_t rm_fd;
  int32_t uvm_fd;
  uint32_t h_client;
  uint32_t h_memory;
} lupine_uvm_allocation_v1;

#ifdef __cplusplus
extern "C" {
#endif
int lupine_uvm_create_v1(size_t size, int device,
                         lupine_uvm_allocation_v1 *allocation);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cstring>

namespace lupine_uvm {
constexpr uint32_t map = 33;
constexpr uint32_t free = 34;
constexpr uint32_t unmap = 66;
constexpr uint32_t reserve = 73;
constexpr uint32_t sparse = 74;
constexpr size_t map_size = 9264;
constexpr size_t rm_fd_offset = 9248;
constexpr size_t rm_client_offset = 9252;
constexpr size_t rm_memory_offset = 9256;

struct schema {
  size_t size;
  size_t status;
};

inline schema describe(uint32_t command) {
  switch (command) {
  case map:
    return {map_size, map_size - 4};
  case free:
    return {16, 8};
  case reserve:
    return {24, 16};
  case unmap:
  case sparse:
    return {40, 32};
  default:
    return {0, 0};
  }
}

template <typename T> T get(const void *buffer, size_t offset) {
  T value;
  std::memcpy(&value, static_cast<const unsigned char *>(buffer) + offset,
              sizeof(value));
  return value;
}
template <typename T> void put(void *buffer, size_t offset, T value) {
  std::memcpy(static_cast<unsigned char *>(buffer) + offset, &value,
              sizeof(value));
}
} // namespace lupine_uvm
#endif
