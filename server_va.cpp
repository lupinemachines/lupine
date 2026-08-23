#include "server_va.h"

#include "rpc.h"

#include <cerrno>
#include <cstdint>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace {

static constexpr uintptr_t LUPINE_SERVER_VA_ALIGNMENT = UINT64_C(0x200000);

thread_local conn_t *lupine_server_va_connection = nullptr;

void *system_mmap(void *address, size_t size, int protection, int flags, int fd,
                  off_t offset) {
  return reinterpret_cast<void *>(
      syscall(SYS_mmap, address, size, protection, flags, fd, offset));
}

uintptr_t claim_address(conn_t *conn, size_t size, size_t alignment) {
  if (conn == nullptr || conn->va_size == 0 || size == 0 || alignment == 0 ||
      (alignment & (alignment - 1)) != 0 || size > conn->va_size) {
    return 0;
  }

  uintptr_t current = __atomic_load_n(&conn->va_next, __ATOMIC_RELAXED);
  for (;;) {
    uintptr_t address = (current + alignment - 1) & ~(alignment - 1);
    if (address < current || address < conn->va_base ||
        address - conn->va_base > conn->va_size - size) {
      return 0;
    }
    uintptr_t next = address + size;
    if (__atomic_compare_exchange_n(&conn->va_next, &current, next, true,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
      return address;
    }
  }
}

void *map_in_arena(conn_t *conn, size_t size, size_t alignment, int protection,
                   int flags, int fd, off_t offset) {
  uintptr_t address = claim_address(conn, size, alignment);
  if (address == 0) {
    errno = ENOMEM;
    return MAP_FAILED;
  }

  flags &= ~(MAP_FIXED | MAP_FIXED_NOREPLACE);
  return system_mmap(reinterpret_cast<void *>(address), size, protection,
                     flags | MAP_FIXED, fd, offset);
}

void *interposed_mmap(void *address, size_t size, int protection, int flags,
                      int fd, off_t offset) {
  conn_t *conn = lupine_server_va_connection;
  if (conn != nullptr && address == nullptr && fd == -1 &&
      protection == PROT_NONE && (flags & MAP_ANONYMOUS) != 0 &&
      (flags & MAP_STACK) == 0) {
    return map_in_arena(conn, size, LUPINE_SERVER_VA_ALIGNMENT, protection,
                        flags, fd, offset);
  }
  return system_mmap(address, size, protection, flags, fd, offset);
}

} // namespace

CUresult lupine_server_va_alloc_managed(conn_t *conn, CUdeviceptr *pointer,
                                        size_t size, unsigned int flags) {
  conn_t *previous = lupine_server_va_connection;
  lupine_server_va_connection = conn;
  CUresult result = cuMemAllocManaged(pointer, size, flags);
  lupine_server_va_connection = previous;
  return result;
}

void *lupine_server_va_map(conn_t *conn, size_t size, size_t alignment,
                           int protection, int flags) {
  return map_in_arena(conn, size, alignment, protection, flags | MAP_ANONYMOUS,
                      -1, 0);
}

extern "C" void *mmap(void *address, size_t size, int protection, int flags,
                      int fd, off_t offset) {
  return interposed_mmap(address, size, protection, flags, fd, offset);
}

extern "C" void *mmap64(void *address, size_t size, int protection, int flags,
                        int fd, off64_t offset) {
  static_assert(sizeof(off_t) == sizeof(off64_t));
  return interposed_mmap(address, size, protection, flags, fd,
                         static_cast<off_t>(offset));
}
