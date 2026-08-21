#include "address_space.h"

#include "rpc.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <new>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct lupine_va_space {
  uintptr_t base = 0;
  size_t size = 0;
  bool client = false;
  std::mutex mutex;
  std::map<uintptr_t, size_t> free_ranges;
};

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
#ifdef MAP_FIXED_NOREPLACE
  flags |= MAP_FIXED_NOREPLACE;
#endif
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

void insert_free_range(lupine_va_space *space, uintptr_t address, size_t size) {
  auto next = space->free_ranges.lower_bound(address);
  if (next != space->free_ranges.begin()) {
    auto previous = std::prev(next);
    if (previous->first + previous->second == address) {
      address = previous->first;
      size += previous->second;
      space->free_ranges.erase(previous);
    }
  }
  next = space->free_ranges.lower_bound(address);
  if (next != space->free_ranges.end() && address + size == next->first) {
    size += next->second;
    space->free_ranges.erase(next);
  }
  space->free_ranges.emplace(address, size);
}

} // namespace

int lupine_va_reserve_client(conn_t *conn, unsigned int min_slot,
                             unsigned int *slot) {
  if (conn == nullptr || slot == nullptr || conn->va_space != nullptr) {
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
    auto *space = new (std::nothrow) lupine_va_space;
    if (space == nullptr) {
      munmap(reinterpret_cast<void *>(write_base), LUPINE_VA_ARENA_SIZE);
      munmap(reinterpret_cast<void *>(base), LUPINE_VA_ARENA_SIZE);
      return -1;
    }
    space->base = base;
    space->size = LUPINE_VA_ARENA_SIZE;
    space->client = true;
    conn->va_base = base;
    conn->va_size = LUPINE_VA_ARENA_SIZE;
    conn->w_offset = LUPINE_VA_WRITE_OFFSET;
    conn->r_offset = 0;
    conn->va_space = space;
    *slot = candidate;
    return 0;
  }
  return -1;
#endif
}

int lupine_va_reserve_server(conn_t *conn, uintptr_t base, size_t size) {
  if (conn == nullptr || conn->va_space != nullptr ||
      !candidate_range(base, size)) {
    return -1;
  }
#if defined(_WIN32) || defined(__APPLE__)
  return -1;
#else
  if (reserve_exact(base, size) == nullptr) {
    return -1;
  }
  auto *space = new (std::nothrow) lupine_va_space;
  if (space == nullptr) {
    munmap(reinterpret_cast<void *>(base), size);
    return -1;
  }
  space->base = base;
  space->size = size;
  space->free_ranges.emplace(base, size);
  // The exact mmap above is the connection preflight. Keeping it would make
  // CUDA and HIP VMM treat every backend subarena hint as occupied, so the
  // server releases the placeholder and retains logical ownership here.
  // Each eventual host or accelerator mapping is still exact-checked.
  munmap(reinterpret_cast<void *>(base), size);
  conn->va_base = base;
  conn->va_size = size;
  conn->va_space = space;
  return 0;
#endif
}

uintptr_t lupine_va_allocate(conn_t *conn, size_t size, size_t alignment) {
  if (conn == nullptr || conn->va_space == nullptr || size == 0 ||
      alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return 0;
  }
  auto *space = static_cast<lupine_va_space *>(conn->va_space);
  if (space->client) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(space->mutex);
  for (auto it = space->free_ranges.begin(); it != space->free_ranges.end();
       ++it) {
    uintptr_t start = it->first;
    size_t available = it->second;
    uintptr_t aligned = (start + alignment - 1) & ~(alignment - 1);
    if (aligned < start || aligned - start > available ||
        size > available - (aligned - start)) {
      continue;
    }
    uintptr_t end = aligned + size;
    uintptr_t free_end = start + available;
    space->free_ranges.erase(it);
    if (aligned != start) {
      space->free_ranges.emplace(start, aligned - start);
    }
    if (end != free_end) {
      space->free_ranges.emplace(end, free_end - end);
    }
#if !defined(_WIN32) && !defined(__APPLE__)
    if (munmap(reinterpret_cast<void *>(aligned), size) == 0) {
      return aligned;
    }
#endif
    insert_free_range(space, aligned, size);
    return 0;
  }
  return 0;
}

bool lupine_va_release(conn_t *conn, uintptr_t address, size_t size) {
  if (conn == nullptr || conn->va_space == nullptr || size == 0) {
    return false;
  }
  auto *space = static_cast<lupine_va_space *>(conn->va_space);
  if (space->client ||
      !checked_range(space->base, space->size, address, size)) {
    return false;
  }
#if defined(_WIN32) || defined(__APPLE__)
  return false;
#else
  std::lock_guard<std::mutex> lock(space->mutex);
  insert_free_range(space, address, size);
  return true;
#endif
}

bool lupine_va_contains(const conn_t *conn, uintptr_t address, size_t size) {
  return conn != nullptr && conn->va_size != 0 &&
         checked_range(conn->va_base, conn->va_size, address, size);
}

void lupine_va_destroy(conn_t *conn) {
  if (conn == nullptr || conn->va_space == nullptr) {
    return;
  }
  auto *space = static_cast<lupine_va_space *>(conn->va_space);
#if !defined(_WIN32) && !defined(__APPLE__)
  if (space->client) {
    munmap(reinterpret_cast<void *>(space->base), space->size);
    munmap(reinterpret_cast<void *>(space->base + LUPINE_VA_WRITE_OFFSET),
           space->size);
  }
#endif
  delete space;
  conn->va_space = nullptr;
  conn->va_base = 0;
  conn->va_size = 0;
}
