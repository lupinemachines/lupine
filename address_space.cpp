#include "address_space.h"

#include "rpc.h"

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__linux__) && !defined(MAP_FIXED_NOREPLACE)
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace {

bool checked_range(uintptr_t base, size_t size, uintptr_t address,
                   size_t bytes) {
  return bytes <= size && address >= base && address - base <= size - bytes;
}

bool candidate_range(uintptr_t base, size_t size) {
  if (size != LUPINE_VA_ARENA_SIZE || base % LUPINE_VA_ARENA_SIZE != 0 ||
      base < LUPINE_VA_FIRST_BASE) {
    return false;
  }
  uintptr_t offset = base - LUPINE_VA_FIRST_BASE;
  return offset / LUPINE_VA_ARENA_SIZE < LUPINE_VA_ARENA_COUNT &&
         offset % LUPINE_VA_ARENA_SIZE == 0;
}

#if !defined(_WIN32) && !defined(__APPLE__)
void *reserve_exact(uintptr_t base, size_t size) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  flags |= MAP_FIXED_NOREPLACE;
  void *mapping =
      mmap(reinterpret_cast<void *>(base), size, PROT_NONE, flags, -1, 0);
  if (mapping == MAP_FAILED) {
    return nullptr;
  }
  if (mapping != reinterpret_cast<void *>(base)) {
    munmap(mapping, size);
    return nullptr;
  }
  return mapping;
}

#endif

} // namespace

int lupine_va_reserve_client(conn_t *conn, unsigned int min_slot,
                             unsigned int *slot) {
  if (conn == nullptr || slot == nullptr || conn->va_size != 0) {
    return -1;
  }
#if defined(_WIN32) || defined(__APPLE__)
  (void)min_slot;
  return 1;
#else
  for (unsigned int candidate = min_slot; candidate < LUPINE_VA_ARENA_COUNT;
       ++candidate) {
    uintptr_t base = LUPINE_VA_FIRST_BASE +
                     static_cast<uintptr_t>(candidate) * LUPINE_VA_ARENA_SIZE;
    uintptr_t write_base = base + LUPINE_VA_WRITE_OFFSET;
    if (reserve_exact(base, LUPINE_VA_ARENA_SIZE) == nullptr) {
      continue;
    }
    if (reserve_exact(write_base, LUPINE_VA_ARENA_SIZE) == nullptr) {
      munmap(reinterpret_cast<void *>(base), LUPINE_VA_ARENA_SIZE);
      continue;
    }
    conn->va_base = base;
    conn->va_size = LUPINE_VA_ARENA_SIZE;
    conn->va_next = 0;
    conn->w_offset = LUPINE_VA_WRITE_OFFSET;
    conn->r_offset = 0;
    *slot = candidate;
    return 0;
  }
  return -1;
#endif
}

int lupine_va_reserve_server(conn_t *conn, uintptr_t base, size_t size) {
  if (conn == nullptr || conn->va_size != 0 || !candidate_range(base, size)) {
    return -1;
  }
#if defined(_WIN32) || defined(__APPLE__)
  return -1;
#else
  if (reserve_exact(base, size) == nullptr) {
    return -1;
  }
  // Keep the placeholder for the connection lifetime. Backends replace only
  // monotonically claimed portions, so MAP_FIXED never touches foreign VMAs.
  conn->va_base = base;
  conn->va_size = size;
  conn->va_next = base;
  return 0;
#endif
}

bool lupine_va_contains(const conn_t *conn, uintptr_t address, size_t size) {
  return conn != nullptr && conn->va_size != 0 &&
         checked_range(conn->va_base, conn->va_size, address, size);
}

void lupine_va_destroy(conn_t *conn) {
  if (conn == nullptr || conn->va_size == 0) {
    return;
  }
#if !defined(_WIN32) && !defined(__APPLE__)
  if (conn->w_offset == LUPINE_VA_WRITE_OFFSET) {
    munmap(reinterpret_cast<void *>(conn->va_base + LUPINE_VA_WRITE_OFFSET),
           conn->va_size);
  }
  munmap(reinterpret_cast<void *>(conn->va_base), conn->va_size);
#endif
  conn->va_base = 0;
  conn->va_size = 0;
  conn->va_next = 0;
}
