#include "lupine_platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#ifndef _WIN32
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#if !defined(__APPLE__)
#include <sys/syscall.h>
#endif
#include <sys/uio.h>
#include <unistd.h>
#endif
#include <vector>

#include <cuda.h>

#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"
#include "cuda_client_memcpy.h"
#include "events.h"

// Defined with the capture wrappers in cuda_client.cpp; nonzero only while this
// client holds a stream capture open, where a pageable copy stays a graph node.
extern std::atomic<int> lupine_active_stream_captures;
#include "lupine_attr_sizes.h"
#include "lupine_log.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

static void lupine_pointer_attribute_cache_clear();
#include "rpc.h"

extern int rpc_size();
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                         size_t ByteCount);
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t ByteCount);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream);
extern "C" CUresult
lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream, bool async);

#ifdef CU_MEM_LOCATION_TYPE_HOST
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    CU_MEM_LOCATION_TYPE_HOST;
#else
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    static_cast<CUmemLocationType>(2);
#endif

struct lupine_host_allocation {
  size_t size = 0;
  size_t storage_size = 0;
  // Exact caller range; the tracked range above is rounded out to whole pages,
  // so Lupine only owns [data_offset, data_offset + user_size) of it.
  uintptr_t user_base = 0;
  size_t user_size = 0;
  size_t data_offset = 0;
  size_t page_size = 0;
  size_t page_count = 0;
  unsigned int flags = 0;
  bool owned = false;
  bool owned_mmap = false;
  void *io_alias = nullptr;
  bool managed = false;
  bool local_cuda = false;
  bool client_to_server_only = false;
  CUdeviceptr server_host_ptr = 0;
  CUdeviceptr device_ptr = 0;
  bool tracking_enabled = false;
  bool device_pointer_exposed = false;
  int fault_slot = -1;
  // 0 fresh, 1 invalidated (PROT_NONE), 2 fetching (handler-only 1->2),
  // 3 invalidating (normal-context only).
  volatile sig_atomic_t device_stale = 0;
  // Captured at invalidation; the handler must not take rpc_open()'s mutex.
  conn_t *stale_fetch_conn = nullptr;
  // Fetch owner; its own nested faults unprotect instead of self-waiting.
  volatile pid_t stale_fetch_tid = 0;
  // Per-chunk fetched flags; owner (state 2) sets, invalidator (3) clears.
  unsigned char *fresh_chunks = nullptr;
  size_t fresh_chunk_count = 0;
  size_t fetch_seq_next = 0;
  size_t fetch_readahead = 0;
  volatile sig_atomic_t full_dirty = 0;
  volatile sig_atomic_t retiring = 0;
  uint32_t pending_dirty_ranges = 0;
  uintptr_t host_base = 0;
  CUdeviceptr device_alloc_base = 0;
  int route_id = -2;
};

struct lupine_mapped_host_snapshot {
  void *host = nullptr;
  size_t data_offset = 0;
  size_t data_bytes = 0;
  CUdeviceptr device_ptr = 0;
  CUdeviceptr server_host_ptr = 0;
  bool host_allocation = false;
  bool device_pointer_exposed = false;
};

using lupine_host_allocation_map =
    std::map<void *, lupine_host_allocation, std::less<void *>>;

static std::mutex &lupine_host_allocation_mutex() {
  static std::mutex mutex;
  return mutex;
}

// CUDA libraries can issue teardown requests after function-local statics have
// started finalizing. Keep the registry alive for those late request flushes.
static lupine_host_allocation_map &lupine_mutable_host_allocations_locked() {
  static lupine_host_allocation_map &allocations =
      *new lupine_host_allocation_map();
  return allocations;
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p);

// Large managed allocations receive an aligned real base that the client can
// map directly while preserving base-pointer APIs such as stream attachment.
static constexpr size_t LUPINE_MANAGED_ALLOCATION_MIN_BYTES = 2 * 1024 * 1024;
static constexpr size_t LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES =
    sizeof(CUdeviceptr) + sizeof(size_t);

static constexpr size_t LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES = 16;
static constexpr size_t LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES = 1024;
static constexpr uint32_t LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES = 64 * 1024;
static constexpr int LUPINE_MAPPED_INVALIDATE_MAX_ATTEMPTS = 10000;
// Sequential faults double the window to the cap, then fetch to the end.
static constexpr size_t LUPINE_DEMAND_FETCH_CHUNK_BYTES = 64 * 1024;
static constexpr size_t LUPINE_DEMAND_FETCH_READAHEAD_CAP = 4 * 1024 * 1024;

struct lupine_dirty_host_range {
  lupine_host_allocation *allocation = nullptr;
  uintptr_t start = 0;
  uintptr_t end = 0;
};

struct lupine_dirty_host_range_queue {
  std::mutex mutex;
  lupine_dirty_host_range ranges[LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES];
  unsigned char ready[LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES];
  uint32_t next = 0;
  uint32_t start = 0;
  volatile sig_atomic_t full_dirty_pending = 0;
};

static lupine_dirty_host_range_queue
    lupine_dirty_host_range_queues[LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES];
// Keep dirty state visible until a flush completes. A single exchanged boolean
// would let another request start while the first request is still flushing.
static uint64_t lupine_dirty_host_epoch = 0;
static volatile sig_atomic_t lupine_device_work_pending = 0;
static std::atomic<uint64_t> lupine_flushed_host_epoch{0};

struct lupine_fault_entry {
  uintptr_t base = 0;
  uintptr_t end = 0;
  lupine_host_allocation *allocation = nullptr;
};

static constexpr size_t LUPINE_MAX_FAULT_ENTRIES = 1024;
static lupine_fault_entry lupine_fault_entries[LUPINE_MAX_FAULT_ENTRIES];
static volatile sig_atomic_t lupine_fault_entry_high_water = 0;
static volatile sig_atomic_t lupine_active_fault_handlers = 0;
static struct sigaction lupine_previous_sigsegv_action;
static bool lupine_sigsegv_handler_installed = false;

static size_t lupine_page_size() {
  long page_size = sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 4096;
}

#ifndef _WIN32
static int lupine_create_shared_memory_fd() {
#if !defined(__APPLE__) && defined(SYS_memfd_create)
  constexpr unsigned int kMemfdCloseOnExec = 0x0001U;
  int fd = static_cast<int>(
      syscall(SYS_memfd_create, "lupine-host-allocation", kMemfdCloseOnExec));
  if (fd >= 0) {
    return fd;
  }
#endif

  static std::atomic<unsigned long long> next_name{0};
  for (int attempt = 0; attempt < 16; ++attempt) {
    char name[64];
    snprintf(name, sizeof(name), "/lupine-%ld-%llu",
             static_cast<long>(getpid()),
             next_name.fetch_add(1, std::memory_order_relaxed));
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
      shm_unlink(name);
      return fd;
    }
  }
  return -1;
}
#endif

static bool lupine_server_range(const conn_t *conn, uintptr_t server,
                                size_t size) {
  if (conn != nullptr && conn->va_size != 0) {
    return lupine_va_contains(conn, server, size);
  }
  if (server < LUPINE_HOST_ALLOCATION_SERVER_BASE ||
      size > LUPINE_HOST_ALLOCATION_WINDOW_SIZE ||
      server - LUPINE_HOST_ALLOCATION_SERVER_BASE >
          LUPINE_HOST_ALLOCATION_WINDOW_SIZE - size) {
    return false;
  }
  return true;
}

#ifndef _WIN32
static bool lupine_reserve_host_allocation_window(uintptr_t base) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
#ifdef MAP_FIXED_NOREPLACE
  flags |= MAP_FIXED_NOREPLACE;
#endif
  void *mapping =
      mmap(reinterpret_cast<void *>(base), LUPINE_HOST_ALLOCATION_WINDOW_SIZE,
           PROT_NONE, flags, -1, 0);
  if (mapping == MAP_FAILED) {
    return false;
  }
  if (mapping != reinterpret_cast<void *>(base)) {
    munmap(mapping, LUPINE_HOST_ALLOCATION_WINDOW_SIZE);
    return false;
  }
  return true;
}

static bool lupine_reserve_host_allocation_windows() {
  static std::once_flag once;
  static bool reserved = false;
  std::call_once(once, [] {
    uintptr_t read_base =
        LUPINE_HOST_ALLOCATION_SERVER_BASE + LUPINE_HOST_ALLOCATION_R_OFFSET;
    uintptr_t write_base =
        LUPINE_HOST_ALLOCATION_SERVER_BASE + LUPINE_HOST_ALLOCATION_W_OFFSET;
    if (!lupine_reserve_host_allocation_window(read_base)) {
      return;
    }
    if (!lupine_reserve_host_allocation_window(write_base)) {
      munmap(reinterpret_cast<void *>(read_base),
             LUPINE_HOST_ALLOCATION_WINDOW_SIZE);
      return;
    }
    reserved = true;
  });
  return reserved;
}

static void lupine_restore_host_allocation_reservation(void *address,
                                                       size_t size) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  mmap(address, size, PROT_NONE, flags, -1, 0);
}
#endif

static bool lupine_create_shared_views(conn_t *conn, uintptr_t server,
                                       size_t size, void **host_view,
                                       void **io_alias) {
  *host_view = nullptr;
  *io_alias = nullptr;
  if (!lupine_server_range(conn, server, size)) {
    return false;
  }
  bool identity = conn != nullptr && conn->va_size != 0;
  uintptr_t host_address =
      identity ? server : server + LUPINE_HOST_ALLOCATION_R_OFFSET;
  uintptr_t alias_address = identity ? server + conn->w_offset
                                     : server + LUPINE_HOST_ALLOCATION_W_OFFSET;
#if defined(_WIN32)
  ULARGE_INTEGER mapping_size = {};
  mapping_size.QuadPart = size;
  HANDLE mapping =
      CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                         mapping_size.HighPart, mapping_size.LowPart, nullptr);
  if (mapping == nullptr) {
    return false;
  }
  void *host = MapViewOfFileEx(mapping, FILE_MAP_WRITE, 0, 0, size,
                               reinterpret_cast<void *>(host_address));
  void *alias = host == nullptr
                    ? nullptr
                    : MapViewOfFileEx(mapping, FILE_MAP_WRITE, 0, 0, size,
                                      reinterpret_cast<void *>(alias_address));
  CloseHandle(mapping);
  if (host == nullptr || alias == nullptr) {
    if (host != nullptr) {
      UnmapViewOfFile(host);
    }
    if (alias != nullptr) {
      UnmapViewOfFile(alias);
    }
    return false;
  }
#else
  if (!identity && !lupine_reserve_host_allocation_windows()) {
    return false;
  }
  int fd = lupine_create_shared_memory_fd();
  if (fd < 0 || ftruncate(fd, static_cast<off_t>(size)) != 0) {
    if (fd >= 0) {
      close(fd);
    }
    return false;
  }
  void *host = mmap(reinterpret_cast<void *>(host_address), size,
                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
  void *alias =
      host == MAP_FAILED
          ? MAP_FAILED
          : mmap(reinterpret_cast<void *>(alias_address), size,
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
  close(fd);
  if (host == MAP_FAILED || alias == MAP_FAILED) {
    if (host != MAP_FAILED) {
      lupine_restore_host_allocation_reservation(host, size);
    }
    if (alias != MAP_FAILED) {
      lupine_restore_host_allocation_reservation(alias, size);
    }
    return false;
  }
#endif
  *host_view = host;
  *io_alias = alias;
  return true;
}

static void lupine_release_shared_views(void *host, void *alias, size_t size) {
#if defined(_WIN32)
  if (host != nullptr) {
    UnmapViewOfFile(host);
  }
  if (alias != nullptr) {
    UnmapViewOfFile(alias);
  }
#else
  if (host != nullptr) {
    lupine_restore_host_allocation_reservation(host, size);
  }
  if (alias != nullptr) {
    lupine_restore_host_allocation_reservation(alias, size);
  }
#endif
}

static pid_t lupine_gettid() {
#if defined(_WIN32)
  return static_cast<pid_t>(lupine_thread_id());
#elif defined(__APPLE__)
  return static_cast<pid_t>(pthread_mach_thread_np(pthread_self()));
#else
  return static_cast<pid_t>(syscall(SYS_gettid));
#endif
}

static size_t lupine_round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static void lupine_call_previous_sigsegv(int sig, siginfo_t *info, void *uctx) {
  if (lupine_previous_sigsegv_action.sa_flags & SA_SIGINFO) {
    if (lupine_previous_sigsegv_action.sa_sigaction != nullptr) {
      lupine_previous_sigsegv_action.sa_sigaction(sig, info, uctx);
      return;
    }
  } else if (lupine_previous_sigsegv_action.sa_handler == SIG_IGN) {
    return;
  } else if (lupine_previous_sigsegv_action.sa_handler != SIG_DFL &&
             lupine_previous_sigsegv_action.sa_handler != nullptr) {
    lupine_previous_sigsegv_action.sa_handler(sig);
    return;
  }

  sigaction(sig, &lupine_previous_sigsegv_action, nullptr);
  raise(sig);
}

static bool lupine_demand_fetch_run(lupine_host_allocation *allocation,
                                    size_t fault_offset, size_t *out_start,
                                    size_t *out_end);
static void lupine_finish_demand_fetch(lupine_host_allocation *allocation,
                                       size_t span_start, size_t span_end);
static bool lupine_make_mapped_range_fresh(lupine_host_allocation *allocation,
                                           size_t start_offset,
                                           size_t end_offset);

static bool
lupine_reserve_dirty_host_range(lupine_dirty_host_range_queue &queue,
                                uint32_t *slot) {
  uint32_t next = __atomic_load_n(&queue.next, __ATOMIC_RELAXED);
  while (next < LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES &&
         !__atomic_compare_exchange_n(&queue.next, &next, next + 1, false,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
  if (next >= LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES) {
    return false;
  }
  *slot = next;
  return true;
}

static void lupine_queue_dirty_host_range(lupine_host_allocation *allocation,
                                          uintptr_t start, uintptr_t end) {
  int route_id = allocation->route_id;
  if (start >= end || route_id < 0 ||
      route_id >= static_cast<int>(LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES)) {
    return;
  }

  auto &queue = lupine_dirty_host_range_queues[route_id];
  uint32_t slot = 0;
  __atomic_add_fetch(&allocation->pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
  if (lupine_reserve_dirty_host_range(queue, &slot)) {
    queue.ranges[slot] = {allocation, start, end};
    __atomic_store_n(&queue.ready[slot], 1, __ATOMIC_RELEASE);
    __atomic_add_fetch(&lupine_dirty_host_epoch, 1, __ATOMIC_RELEASE);
    return;
  }

  __atomic_sub_fetch(&allocation->pending_dirty_ranges, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&allocation->full_dirty, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&queue.full_dirty_pending, 1, __ATOMIC_RELEASE);
  __atomic_add_fetch(&lupine_dirty_host_epoch, 1, __ATOMIC_RELEASE);
}

static void lupine_sigsegv_handler(int sig, siginfo_t *info, void *uctx) {
  __atomic_add_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_ACQUIRE);
  uintptr_t addr = reinterpret_cast<uintptr_t>(info->si_addr);
  sig_atomic_t count =
      __atomic_load_n(&lupine_fault_entry_high_water, __ATOMIC_ACQUIRE);
  for (sig_atomic_t i = 0; i < count; ++i) {
    const lupine_fault_entry &entry = lupine_fault_entries[i];
    lupine_host_allocation *allocation =
        __atomic_load_n(&entry.allocation, __ATOMIC_ACQUIRE);
    if (addr < entry.base || addr >= entry.end || allocation == nullptr) {
      continue;
    }

    size_t page_size = allocation->page_size;
    size_t page_index = (addr - entry.base) / page_size;
    if (page_index >= allocation->page_count) {
      break;
    }

    int route_id = allocation->route_id;
    if (route_id < 0 ||
        route_id >= static_cast<int>(LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES)) {
      __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
      lupine_call_previous_sigsegv(sig, info, uctx);
      return;
    }

    uintptr_t page = entry.base + page_index * page_size;
    if (__atomic_load_n(&allocation->retiring, __ATOMIC_ACQUIRE) != 0) {
      mprotect(reinterpret_cast<void *>(page), page_size,
               PROT_READ | PROT_WRITE);
      __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
      return;
    }

    sig_atomic_t stale =
        __atomic_load_n(&allocation->device_stale, __ATOMIC_ACQUIRE);
    size_t fault_offset = addr - entry.base;
    size_t fault_chunk = fault_offset / LUPINE_DEMAND_FETCH_CHUNK_BYTES;
    bool chunk_fresh = allocation->fresh_chunks != nullptr &&
                       fault_chunk < allocation->fresh_chunk_count &&
                       __atomic_load_n(&allocation->fresh_chunks[fault_chunk],
                                       __ATOMIC_ACQUIRE) != 0;
    if (stale != 0 && !chunk_fresh) {
      if (stale == 2 && __atomic_load_n(&allocation->stale_fetch_tid,
                                        __ATOMIC_ACQUIRE) == lupine_gettid()) {
        // Our own fetch faulted on a concurrently re-protected page.
        mprotect(reinterpret_cast<void *>(page), page_size,
                 PROT_READ | PROT_WRITE);
        __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
        return;
      }
      sig_atomic_t expected = 1;
      if (stale == 1 && __atomic_compare_exchange_n(
                            &allocation->device_stale, &expected, 2, false,
                            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        __atomic_store_n(&allocation->stale_fetch_tid, lupine_gettid(),
                         __ATOMIC_RELEASE);
        size_t span_start = 0;
        size_t span_end = 0;
        bool fetched = lupine_demand_fetch_run(allocation, fault_offset,
                                               &span_start, &span_end);
        if (fetched) {
          lupine_finish_demand_fetch(allocation, span_start, span_end);
        }
        __atomic_store_n(&allocation->stale_fetch_tid, 0, __ATOMIC_RELEASE);
        if (!fetched) {
          // Unrecoverable; crash rather than serve stale bytes.
          __atomic_store_n(&allocation->device_stale, 1, __ATOMIC_RELEASE);
          __atomic_sub_fetch(&lupine_active_fault_handlers, 1,
                             __ATOMIC_RELEASE);
          lupine_call_previous_sigsegv(sig, info, uctx);
          return;
        }
      } else {
        while (true) {
          sig_atomic_t state =
              __atomic_load_n(&allocation->device_stale, __ATOMIC_ACQUIRE);
          if (state != 2 && state != 3) {
            break;
          }
        }
      }
      __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
      return;
    }

    lupine_queue_dirty_host_range(allocation, page, page + page_size);

    mprotect(reinterpret_cast<void *>(page), page_size, PROT_READ | PROT_WRITE);
    __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
    return;
  }

  __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
  lupine_call_previous_sigsegv(sig, info, uctx);
}

// Handler RPC reads need real stack; the guard page turns overflow into a
// fault instead of silent TLS corruption.
struct lupine_signal_stack {
  void *mapping = MAP_FAILED;
  size_t mapping_size = 0;
  ~lupine_signal_stack() {
    if (mapping != MAP_FAILED) {
      munmap(mapping, mapping_size);
    }
  }
};

static void lupine_install_sigsegv_handler() {
  constexpr size_t kSignalStackBytes = 1024 * 1024;
  static thread_local lupine_signal_stack signal_stack;
  static thread_local bool signal_stack_installed = false;
  if (!signal_stack_installed) {
    size_t page_size = lupine_page_size();
    size_t mapping_size = kSignalStackBytes + page_size;
    void *mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping != MAP_FAILED && mprotect(mapping, page_size, PROT_NONE) == 0) {
      signal_stack.mapping = mapping;
      signal_stack.mapping_size = mapping_size;
      stack_t stack = {};
      stack.ss_sp = static_cast<unsigned char *>(mapping) + page_size;
      stack.ss_size = kSignalStackBytes;
      signal_stack_installed = sigaltstack(&stack, nullptr) == 0;
    } else if (mapping != MAP_FAILED) {
      munmap(mapping, mapping_size);
    }
  }
  if (lupine_sigsegv_handler_installed) {
    return;
  }
  struct sigaction action = {};
  action.sa_sigaction = lupine_sigsegv_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;
  if (sigaction(SIGSEGV, &action, &lupine_previous_sigsegv_action) == 0) {
    lupine_sigsegv_handler_installed = true;
  }
}

static int lupine_add_fault_entry(void *base, size_t size,
                                  lupine_host_allocation *allocation) {
  sig_atomic_t high_water =
      __atomic_load_n(&lupine_fault_entry_high_water, __ATOMIC_ACQUIRE);
  for (sig_atomic_t index = 0;
       index < static_cast<sig_atomic_t>(LUPINE_MAX_FAULT_ENTRIES); ++index) {
    if (__atomic_load_n(&lupine_fault_entries[index].allocation,
                        __ATOMIC_ACQUIRE) != nullptr) {
      continue;
    }
    lupine_fault_entries[index].base = reinterpret_cast<uintptr_t>(base);
    lupine_fault_entries[index].end = reinterpret_cast<uintptr_t>(base) + size;
    __atomic_store_n(&lupine_fault_entries[index].allocation, allocation,
                     __ATOMIC_RELEASE);
    if (index >= high_water) {
      __atomic_store_n(&lupine_fault_entry_high_water, index + 1,
                       __ATOMIC_RELEASE);
    }
    return index;
  }
  return -1;
}

static void lupine_retire_fault_entry(int slot) {
  if (slot < 0 || slot >= static_cast<int>(LUPINE_MAX_FAULT_ENTRIES)) {
    return;
  }
  __atomic_store_n(&lupine_fault_entries[slot].allocation, nullptr,
                   __ATOMIC_RELEASE);
}

// lupine_active_fault_handlers counts handlers on every allocation, not just
// the one being retired, and a handler leaving lupine_finish_demand_fetch
// takes lupine_host_allocation_mutex(). Waiting here while holding that mutex
// deadlocks the two against each other, so callers drain once they have let
// it go.
static void lupine_drain_fault_handlers() {
  while (__atomic_load_n(&lupine_active_fault_handlers, __ATOMIC_ACQUIRE) !=
         0) {
    sched_yield();
  }
}

static bool lupine_host_flags_request_mapping(unsigned int flags) {
  return (flags & (CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTREGISTER_DEVICEMAP)) !=
         0;
}

static bool lupine_protect_host_range(void *host, size_t size, int prot) {
  if (host == nullptr || size == 0) {
    return true;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  size_t page_size = lupine_page_size();
  uintptr_t page_start = start & ~(static_cast<uintptr_t>(page_size) - 1);
  uintptr_t end = lupine_round_up(start + size, page_size);
  return mprotect(reinterpret_cast<void *>(page_start), end - page_start,
                  prot) == 0;
}

static bool
lupine_enable_dirty_tracking_locked(void *host,
                                    lupine_host_allocation *allocation) {
  if (host == nullptr || allocation == nullptr ||
      allocation->server_host_ptr == 0 || allocation->local_cuda) {
    return true;
  }
  if (allocation->tracking_enabled) {
    return true;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(host);
  if ((base % allocation->page_size) != 0 ||
      (allocation->storage_size % allocation->page_size) != 0) {
    return true;
  }

  allocation->tracking_enabled = true;
  allocation->host_base = base;

  allocation->fresh_chunk_count =
      lupine_round_up(allocation->storage_size,
                      LUPINE_DEMAND_FETCH_CHUNK_BYTES) /
      LUPINE_DEMAND_FETCH_CHUNK_BYTES;
  allocation->fresh_chunks = static_cast<unsigned char *>(
      calloc(allocation->fresh_chunk_count, sizeof(unsigned char)));
  if (allocation->fresh_chunks == nullptr) {
    allocation->tracking_enabled = false;
    return false;
  }

  lupine_install_sigsegv_handler();
  allocation->fault_slot =
      lupine_add_fault_entry(host, allocation->storage_size, allocation);
  if (allocation->fault_slot < 0) {
    free(allocation->fresh_chunks);
    allocation->fresh_chunks = nullptr;
    allocation->tracking_enabled = false;
    return false;
  }
  if (!lupine_protect_host_range(host, allocation->storage_size, PROT_READ)) {
    lupine_retire_fault_entry(allocation->fault_slot);
    lupine_drain_fault_handlers();
    allocation->fault_slot = -1;
    free(allocation->fresh_chunks);
    allocation->fresh_chunks = nullptr;
    allocation->tracking_enabled = false;
    return false;
  }
  return true;
}

static void lupine_disable_dirty_tracking(void *host,
                                          lupine_host_allocation &allocation) {
  if (!allocation.tracking_enabled) {
    return;
  }
  lupine_protect_host_range(host, allocation.storage_size,
                            PROT_READ | PROT_WRITE);
  lupine_retire_fault_entry(allocation.fault_slot);
  allocation.fault_slot = -1;
  allocation.tracking_enabled = false;
}

static std::vector<lupine_mapped_host_snapshot> lupine_mapped_host_snapshots() {
  std::vector<lupine_mapped_host_snapshot> snapshots;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (const auto &entry : lupine_mutable_host_allocations_locked()) {
    if (entry.second.device_ptr != 0 && !entry.second.local_cuda &&
        !entry.second.client_to_server_only) {
      snapshots.push_back({entry.first, entry.second.data_offset,
                           entry.second.user_size, entry.second.device_ptr,
                           entry.second.server_host_ptr, entry.second.owned,
                           entry.second.device_pointer_exposed});
    }
  }
  return snapshots;
}

static CUresult lupine_collect_host_allocation_writes(conn_t *conn) {
  std::vector<rpc_host_allocation_write> writes;
  if (pthread_mutex_lock(&conn->write_mutex) != 0) {
    return CUDA_ERROR_UNKNOWN;
  }
  writes.swap(conn->host_allocation_writes);
  __atomic_store_n(&conn->host_allocation_writes_pending, 0, __ATOMIC_RELEASE);
  if (pthread_mutex_unlock(&conn->write_mutex) != 0) {
    return CUDA_ERROR_UNKNOWN;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (const auto &write : writes) {
    auto it = lupine_find_host_allocation_locked(
        reinterpret_cast<void *>(write.start));
    if (it == lupine_mutable_host_allocations_locked().end() ||
        !it->second.tracking_enabled) {
      continue;
    }
    auto &allocation = it->second;
    uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    size_t offset = write.start - base;
    size_t bytes = std::min(write.size, allocation.size - offset);
    if (bytes == 0) {
      continue;
    }
    uintptr_t end = write.start + bytes;
    lupine_queue_dirty_host_range(&allocation, write.start, end);

    if (__atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE) != 1) {
      continue;
    }
    if (allocation.fresh_chunks != nullptr) {
      size_t chunk_bytes = LUPINE_DEMAND_FETCH_CHUNK_BYTES;
      size_t first_chunk = lupine_round_up(offset, chunk_bytes) / chunk_bytes;
      size_t last_chunk = (end - base) / chunk_bytes;
      for (size_t index = first_chunk;
           index < last_chunk && index < allocation.fresh_chunk_count;
           ++index) {
        __atomic_store_n(&allocation.fresh_chunks[index], 1, __ATOMIC_RELEASE);
      }
    }
    size_t first_page = offset / allocation.page_size;
    size_t last_page = (end - 1 - base) / allocation.page_size;
    uintptr_t protect_start = base + first_page * allocation.page_size;
    size_t protect_size = (last_page - first_page + 1) * allocation.page_size;
    lupine_protect_host_range(reinterpret_cast<void *>(protect_start),
                              protect_size, PROT_READ);
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_flush_dirty_host_pages_to_route(size_t route_id) {
  conn_t *conn = lupine_route_remote_conn(
      lupine_route_from_identity(static_cast<int>(route_id)));
  if (conn == nullptr) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  CUresult collect_result = lupine_collect_host_allocation_writes(conn);
  if (collect_result != CUDA_SUCCESS) {
    return collect_result;
  }

  auto &queue = lupine_dirty_host_range_queues[route_id];
  std::lock_guard<std::mutex> route_lock(queue.mutex);

  uint32_t reserved = __atomic_load_n(&queue.next, __ATOMIC_ACQUIRE);
  uint32_t end = queue.start;
  while (end < reserved &&
         __atomic_load_n(&queue.ready[end], __ATOMIC_ACQUIRE) != 0) {
    ++end;
  }
  bool has_full_dirty =
      __atomic_load_n(&queue.full_dirty_pending, __ATOMIC_ACQUIRE) != 0;

  std::vector<lupine_dirty_host_range> ranges;
  ranges.reserve(end - queue.start);
  for (uint32_t slot = queue.start; slot < end; ++slot) {
    ranges.push_back(queue.ranges[slot]);
    __atomic_store_n(&queue.ready[slot], 0, __ATOMIC_RELEASE);
  }
  queue.start = end;
  uint32_t expected = end;
  if (__atomic_compare_exchange_n(&queue.next, &expected, 0, false,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    queue.start = 0;
  }

  std::vector<lupine_host_allocation *> stale_full_dirty_fetches;
  if (has_full_dirty && __atomic_exchange_n(&queue.full_dirty_pending, 0,
                                            __ATOMIC_ACQ_REL) != 0) {
    std::lock_guard<std::mutex> allocation_lock(lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      auto &allocation = entry.second;
      if (!allocation.tracking_enabled ||
          allocation.route_id != static_cast<int>(route_id) ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0 ||
          __atomic_exchange_n(&allocation.full_dirty, 0, __ATOMIC_ACQ_REL) ==
              0) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
      ranges.push_back({&allocation, allocation.host_base,
                        allocation.host_base + allocation.storage_size});
      if (__atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE) != 0) {
        stale_full_dirty_fetches.push_back(&allocation);
      }
    }
  }

  // Nothing observes writes to a registration, so the whole window goes over
  // before device work can read it.
  {
    std::lock_guard<std::mutex> allocation_lock(lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      auto &allocation = entry.second;
      if (allocation.tracking_enabled || !allocation.device_pointer_exposed ||
          allocation.server_host_ptr == 0 || allocation.local_cuda ||
          allocation.route_id != static_cast<int>(route_id) ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
      ranges.push_back({&allocation, allocation.host_base,
                        allocation.host_base + allocation.size});
    }
  }

  if (ranges.empty()) {
    return CUDA_SUCCESS;
  }

  // A full flush must not push an invalidated host allocation; fetch it
  // first. The
  // pushed ranges hold pending references, so retirement waits.
  for (lupine_host_allocation *allocation : stale_full_dirty_fetches) {
    lupine_make_mapped_range_fresh(allocation, 0, allocation->size);
  }

  auto release_ranges = [&](bool restore) {
    for (const auto &range : ranges) {
      if (restore && range.allocation->tracking_enabled &&
          __atomic_load_n(&range.allocation->retiring, __ATOMIC_ACQUIRE) == 0) {
        __atomic_store_n(&range.allocation->full_dirty, 1, __ATOMIC_RELEASE);
        __atomic_store_n(&queue.full_dirty_pending, 1, __ATOMIC_RELEASE);
      }
      __atomic_sub_fetch(&range.allocation->pending_dirty_ranges, 1,
                         __ATOMIC_RELEASE);
    }
  };

  std::sort(
      ranges.begin(), ranges.end(),
      [](const lupine_dirty_host_range &a, const lupine_dirty_host_range &b) {
        if (a.allocation != b.allocation) {
          return a.allocation->host_base < b.allocation->host_base;
        }
        if (a.start != b.start) {
          return a.start < b.start;
        }
        return a.end < b.end;
      });
  std::vector<lupine_dirty_host_range> merged;
  merged.reserve(ranges.size());
  for (const auto &range : ranges) {
    if (merged.empty() || merged.back().allocation != range.allocation ||
        merged.back().end < range.start) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }

  for (const auto &range : merged) {
    if (range.allocation->tracking_enabled &&
        !lupine_protect_host_range(reinterpret_cast<void *>(range.start),
                                   range.end - range.start, PROT_READ)) {
      release_ranges(true);
      return CUDA_ERROR_UNKNOWN;
    }
  }

  std::array<std::array<unsigned char, LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES>,
             LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES>
      headers;
  std::array<rpc_write_cursor, LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES * 2>
      cursors;

  auto send_batch = [&](uint32_t count) {
    if (count == 0) {
      return CUDA_SUCCESS;
    }
    // This is the flush performed by lupine_prepare_rpc(), so it must use
    // the raw transport primitive.
    if (rpc_write_start_request(conn, LUPINE_RPC_lupineManagedHostFlush) < 0 ||
        rpc_write(conn, &count, sizeof(count)) < 0 ||
        rpc_write_cursors(conn, cursors.data(), count * 2) < 0 ||
        rpc_write_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return CUDA_SUCCESS;
  };

  uint32_t count = 0;
  for (const auto &range : merged) {
    auto &allocation = *range.allocation;
    uintptr_t data_start = allocation.host_base + allocation.data_offset;
    uintptr_t start = std::max(range.start, data_start);
    uintptr_t end = std::min(range.end, data_start + allocation.user_size);
    if (end <= start) {
      continue;
    }
    size_t offset = start - allocation.host_base;
    size_t bytes = end - start;
    CUdeviceptr dst = allocation.server_host_ptr + offset;
    memcpy(headers[count].data(), &dst, sizeof(dst));
    memcpy(headers[count].data() + sizeof(dst), &bytes, sizeof(bytes));
    cursors[count * 2] = {headers[count].data(),
                          LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES};
    const void *source = reinterpret_cast<void *>(start);
    if (allocation.io_alias != nullptr) {
      source = static_cast<unsigned char *>(allocation.io_alias) + offset;
    }
    cursors[count * 2 + 1] = {source, static_cast<ssize_t>(bytes)};
    ++count;
    if (count == LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES) {
      CUresult result = send_batch(count);
      if (result != CUDA_SUCCESS) {
        release_ranges(true);
        return result;
      }
      count = 0;
    }
  }
  CUresult result = send_batch(count);
  if (result != CUDA_SUCCESS) {
    release_ranges(true);
    return result;
  }
  release_ranges(false);
  return CUDA_SUCCESS;
}

static CUresult lupine_flush_dirty_host_pages_to_server() {
  for (int route_id = 0; route_id < rpc_size(); ++route_id) {
    CUresult result =
        lupine_flush_dirty_host_pages_to_route(static_cast<size_t>(route_id));
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  return CUDA_SUCCESS;
}

extern "C" int lupine_prepare_rpc(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  __atomic_store_n(&lupine_device_work_pending, 1, __ATOMIC_RELEASE);
  if (__atomic_load_n(&conn->host_allocation_writes_pending,
                      __ATOMIC_ACQUIRE) != 0 &&
      lupine_collect_host_allocation_writes(conn) != CUDA_SUCCESS) {
    return -1;
  }

  uint64_t dirty_epoch =
      __atomic_load_n(&lupine_dirty_host_epoch, __ATOMIC_ACQUIRE);
  if (dirty_epoch !=
      lupine_flushed_host_epoch.load(std::memory_order_acquire)) {
    if (lupine_flush_dirty_host_pages_to_server() != CUDA_SUCCESS) {
      return -1;
    }
    lupine_flushed_host_epoch.store(dirty_epoch, std::memory_order_release);
  }
  return 0;
}

static CUresult
lupine_drain_retiring_dirty_ranges(lupine_host_allocation *allocation) {
  if (allocation == nullptr) {
    return CUDA_SUCCESS;
  }
  CUresult result = CUDA_SUCCESS;
  if (allocation->route_id >= 0 &&
      allocation->route_id <
          static_cast<int>(LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES)) {
    result = lupine_flush_dirty_host_pages_to_route(
        static_cast<size_t>(allocation->route_id));
    while (result == CUDA_SUCCESS &&
           __atomic_load_n(&allocation->pending_dirty_ranges,
                           __ATOMIC_ACQUIRE) != 0) {
    }
  }
  if (allocation->fresh_chunks != nullptr) {
    lupine_drain_fault_handlers();
    free(allocation->fresh_chunks);
    allocation->fresh_chunks = nullptr;
    allocation->fresh_chunk_count = 0;
  }
  return result;
}

// A managed allocation's client host alias is mapped at the server device VA
// (identity arenas are reserve-or-fail at connect), so alias addresses go on
// the wire unchanged; callers only need to know whether ptr is one.
extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  return it != lupine_mutable_host_allocations_locked().end() &&
         it->second.managed;
}

static bool lupine_translate_client_host_ptr_to_server(
    CUdeviceptr ptr, CUdeviceptr *translated,
    CUdeviceptr *server_base = nullptr, CUdeviceptr *device_base = nullptr) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      it->second.server_host_ptr == 0 || it->second.local_cuda) {
    return false;
  }

  uintptr_t client_base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t client_addr = reinterpret_cast<uintptr_t>(host);
  if (translated != nullptr) {
    *translated = it->second.server_host_ptr + (client_addr - client_base);
  }
  if (server_base != nullptr) {
    *server_base = it->second.server_host_ptr;
  }
  if (device_base != nullptr) {
    *device_base = it->second.device_ptr;
  }
  return true;
}

// Resolve a complete client host span to the matching server allocation. The
// range check matters for pitched copies: finding the first byte is not enough
// when the final row or slice extends beyond the tracked allocation.
static bool lupine_translate_client_host_range_to_server(
    const void *host, size_t size, int route_id, CUdeviceptr *translated,
    bool *managed = nullptr) {
  if (host == nullptr || translated == nullptr) {
    return false;
  }
  uintptr_t address = reinterpret_cast<uintptr_t>(host);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
  if (it == lupine_mutable_host_allocations_locked().end() ||
      it->second.server_host_ptr == 0 || it->second.local_cuda ||
      it->second.route_id != route_id) {
    return false;
  }
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  size_t offset = address - base;
  if (offset > it->second.size || size > it->second.size - offset) {
    return false;
  }
  *translated = it->second.server_host_ptr + offset;
  if (managed != nullptr) {
    *managed = it->second.managed;
  }
  return true;
}

static bool lupine_host_ptr_is_tracked(CUdeviceptr ptr) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  return lupine_find_host_allocation_locked(host) !=
         lupine_mutable_host_allocations_locked().end();
}

// Whether the driver would treat this address as page-locked, which is true of
// exactly the ranges the caller handed to cuMemHostAlloc, cuMemAllocHost,
// cuMemHostRegister or cuMemAllocManaged. Tracked ranges are rounded out to
// whole pages, so the exact caller range is what has to be tested: a plain
// malloc that merely shares a page with a registered one is still pageable, and
// treating it as page-locked would defer a copy the driver completes inline.
extern "C" bool lupine_host_ptr_is_page_locked(const void *host) {
  if (host == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return false;
  }
  uintptr_t address = reinterpret_cast<uintptr_t>(host);
  return address >= it->second.user_base &&
         address < it->second.user_base + it->second.user_size;
}

static bool lupine_managed_host_alias_base(CUdeviceptr ptr,
                                           CUdeviceptr *alias_base) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.managed) {
    return false;
  }
  *alias_base = it->second.device_ptr;
  return true;
}

static bool lupine_is_client_mapped_address(CUdeviceptr ptr) {
  if (ptr == 0) {
    return false;
  }
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  uintptr_t page =
      static_cast<uintptr_t>(ptr) & ~(static_cast<uintptr_t>(page_size) - 1);
#if defined(__APPLE__)
  char residency = 0;
#else
  unsigned char residency = 0;
#endif
  return mincore(reinterpret_cast<void *>(page), page_size, &residency) == 0;
}

extern "C" bool lupine_copy_pointer_is_host(CUdeviceptr ptr) {
  if (lupine_host_ptr_is_tracked(ptr)) {
    return true;
  }
  if (lupine_deviceptr_is_tracked(ptr)) {
    return false;
  }
  return lupine_is_client_mapped_address(ptr);
}

static lupine_copy_direction lupine_infer_copy_direction(CUdeviceptr dst,
                                                         CUdeviceptr src) {
  const bool dst_is_host = lupine_copy_pointer_is_host(dst);
  const bool src_is_host = lupine_copy_pointer_is_host(src);
  if (dst_is_host && src_is_host) {
    return lupine_copy_direction::host_to_host;
  }
  if (src_is_host) {
    return lupine_copy_direction::host_to_device;
  }
  if (dst_is_host) {
    return lupine_copy_direction::device_to_host;
  }
  return lupine_copy_direction::device_to_device;
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src,
                             size_t ByteCount) {
  if (ByteCount == 0) {
    return CUDA_SUCCESS;
  }

  switch (lupine_infer_copy_direction(dst, src)) {
  case lupine_copy_direction::host_to_host:
    memmove(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src),
            ByteCount);
    return CUDA_SUCCESS;
  case lupine_copy_direction::host_to_device:
    return cuMemcpyHtoD_v2(dst, reinterpret_cast<const void *>(src), ByteCount);
  case lupine_copy_direction::device_to_host:
    return cuMemcpyDtoH_v2(reinterpret_cast<void *>(dst), src, ByteCount);
  case lupine_copy_direction::device_to_device:
    return cuMemcpyDtoD_v2(dst, src, ByteCount);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

#ifdef cuMemcpy_ptds
#undef cuMemcpy_ptds
#endif
extern "C" CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount) {
  return cuMemcpy(dst, src, ByteCount);
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount, CUstream hStream) {
  switch (lupine_infer_copy_direction(dst, src)) {
  case lupine_copy_direction::host_to_host:
    memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src),
           ByteCount);
    return CUDA_SUCCESS;
  case lupine_copy_direction::host_to_device:
    return cuMemcpyHtoDAsync_v2(dst, reinterpret_cast<const void *>(src),
                                ByteCount, hStream);
  case lupine_copy_direction::device_to_host:
    return cuMemcpyDtoHAsync_v2(reinterpret_cast<void *>(dst), src, ByteCount,
                                hStream);
  case lupine_copy_direction::device_to_device:
    return cuMemcpyDtoDAsync_v2(dst, src, ByteCount, hStream);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

#ifdef cuMemcpyAsync_ptsz
#undef cuMemcpyAsync_ptsz
#endif
extern "C" CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                       size_t ByteCount, CUstream hStream) {
  return cuMemcpyAsync(dst, src, ByteCount, hStream);
}

// Wire-fetch [offset, offset + bytes) of the backing. Handler-safe: must
// not take lupine_host_allocation_mutex(); caller made the range writable.
// Callers work in tracked-page space, so the span can reach past the bytes the
// allocation owns; this is the only place server bytes land in client pages,
// so it
// clips here rather than trusting each caller to.
static bool lupine_fetch_stale_range(lupine_host_allocation *allocation,
                                     size_t offset, size_t bytes) {
  size_t end =
      std::min(offset + bytes, allocation->data_offset + allocation->user_size);
  offset = std::max(offset, allocation->data_offset);
  bytes = end - offset;

  conn_t *conn = allocation->stale_fetch_conn;
  auto *dst =
      allocation->io_alias != nullptr
          ? static_cast<unsigned char *>(allocation->io_alias) + offset
          : reinterpret_cast<unsigned char *>(allocation->host_base + offset);
  CUdeviceptr src = allocation->device_ptr + offset;
  if (allocation->host_base == 0 || allocation->device_ptr == 0) {
    return false;
  }
  // Demand fetch can run from the fault handler and must not re-enter the
  // normal CUDA request-start flush.
  CUstream fetch_stream = CU_STREAM_LEGACY;
  if (rpc_write_start_request(conn, RPC_cuMemcpyDtoH_v2) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &fetch_stream, sizeof(fetch_stream)) < 0) {
    return false;
  }
  int request_id = rpc_write_end(conn);
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return false;
  }
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  size_t read_offset = 0;
  do {
    size_t chunk =
        std::min(bytes - read_offset, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
    if (rpc_read(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && chunk != 0 &&
         rpc_read_payload(conn, dst + read_offset, chunk) < 0)) {
      rpc_read_end(conn);
      return false;
    }
    bool final_chunk = result != CUDA_SUCCESS || read_offset + chunk == bytes;
    if (rpc_read_end(conn) < 0) {
      return false;
    }
    if (result != CUDA_SUCCESS) {
      return false;
    }
    read_offset += chunk;
    if (!final_chunk && rpc_read_start(conn, request_id) < 0) {
      return false;
    }
  } while (read_offset < bytes);
  return true;
}

// Publish a finished fetch and drop the stale state once every chunk is fresh.
static void lupine_finish_demand_fetch(lupine_host_allocation *allocation,
                                       size_t span_start, size_t span_end) {
  bool all_fresh = true;
  for (size_t i = 0; i < allocation->fresh_chunk_count; ++i) {
    if (__atomic_load_n(&allocation->fresh_chunks[i], __ATOMIC_ACQUIRE) == 0) {
      all_fresh = false;
      break;
    }
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  if (span_end > span_start) {
    mprotect(reinterpret_cast<void *>(allocation->host_base + span_start),
             span_end - span_start, PROT_READ);
  }
  __atomic_store_n(&allocation->device_stale, all_fresh ? 0 : 1,
                   __ATOMIC_RELEASE);
}

// Fetch the stale run at fault_offset, widened by readahead and clipped at
// the first fresh chunk so a refetch cannot clobber tracked host writes.
static bool lupine_demand_fetch_run(lupine_host_allocation *allocation,
                                    size_t fault_offset, size_t *out_start,
                                    size_t *out_end) {
  size_t chunk_bytes = LUPINE_DEMAND_FETCH_CHUNK_BYTES;
  if (allocation->fresh_chunks == nullptr) {
    return false;
  }
  size_t start = fault_offset / chunk_bytes * chunk_bytes;
  size_t limit;
  if (start != 0 && start == allocation->fetch_seq_next &&
      allocation->fetch_readahead >= chunk_bytes) {
    if (allocation->fetch_readahead >= LUPINE_DEMAND_FETCH_READAHEAD_CAP) {
      limit = allocation->storage_size;
    } else {
      allocation->fetch_readahead = std::min(allocation->fetch_readahead * 2,
                                             LUPINE_DEMAND_FETCH_READAHEAD_CAP);
      limit = start + allocation->fetch_readahead;
    }
  } else {
    allocation->fetch_readahead = chunk_bytes;
    limit = start + chunk_bytes;
  }
  limit = std::min(limit, allocation->storage_size);

  size_t end = start;
  while (end < limit &&
         __atomic_load_n(&allocation->fresh_chunks[end / chunk_bytes],
                         __ATOMIC_ACQUIRE) == 0) {
    end += chunk_bytes;
  }
  end = std::min(end, allocation->storage_size);
  *out_start = start;
  *out_end = end;
  if (end <= start) {
    allocation->fetch_seq_next = start + chunk_bytes;
    return true;
  }

  if (allocation->io_alias == nullptr) {
    mprotect(reinterpret_cast<void *>(allocation->host_base + start),
             end - start, PROT_READ | PROT_WRITE);
  }
  if (!lupine_fetch_stale_range(allocation, start, end - start)) {
    return false;
  }
  for (size_t index = start / chunk_bytes;
       index < (end + chunk_bytes - 1) / chunk_bytes &&
       index < allocation->fresh_chunk_count;
       ++index) {
    __atomic_store_n(&allocation->fresh_chunks[index], 1, __ATOMIC_RELEASE);
  }
  allocation->fetch_seq_next = end;
  return true;
}

// Make a host allocation range fresh from normal context. Must not be called
// while
// holding the allocation mutex or any connection.
static bool lupine_make_mapped_range_fresh(lupine_host_allocation *allocation,
                                           size_t start_offset,
                                           size_t end_offset) {
  size_t chunk_bytes = LUPINE_DEMAND_FETCH_CHUNK_BYTES;
  for (;;) {
    sig_atomic_t state =
        __atomic_load_n(&allocation->device_stale, __ATOMIC_ACQUIRE);
    if (state == 0) {
      return true;
    }
    if (__atomic_load_n(&allocation->retiring, __ATOMIC_ACQUIRE) != 0) {
      return true;
    }
    if (state == 2 || state == 3) {
      sched_yield();
      continue;
    }
    sig_atomic_t expected = 1;
    if (__atomic_compare_exchange_n(&allocation->device_stale, &expected, 2,
                                    false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      break;
    }
  }
  __atomic_store_n(&allocation->stale_fetch_tid, lupine_gettid(),
                   __ATOMIC_RELEASE);
  bool ok = allocation->fresh_chunks != nullptr;
  size_t position = start_offset / chunk_bytes * chunk_bytes;
  size_t bound = std::min(lupine_round_up(end_offset, chunk_bytes),
                          allocation->storage_size);
  size_t span_start = position;
  size_t span_end = position;
  while (ok && position < bound) {
    if (__atomic_load_n(&allocation->fresh_chunks[position / chunk_bytes],
                        __ATOMIC_ACQUIRE) != 0) {
      position += chunk_bytes;
      continue;
    }
    size_t run_end = position;
    while (run_end < bound &&
           __atomic_load_n(&allocation->fresh_chunks[run_end / chunk_bytes],
                           __ATOMIC_ACQUIRE) == 0) {
      run_end += chunk_bytes;
    }
    run_end = std::min(run_end, allocation->storage_size);
    if (allocation->io_alias == nullptr) {
      mprotect(reinterpret_cast<void *>(allocation->host_base + position),
               run_end - position, PROT_READ | PROT_WRITE);
    }
    ok = lupine_fetch_stale_range(allocation, position, run_end - position);
    if (ok) {
      for (size_t index = position / chunk_bytes;
           index < (run_end + chunk_bytes - 1) / chunk_bytes &&
           index < allocation->fresh_chunk_count;
           ++index) {
        __atomic_store_n(&allocation->fresh_chunks[index], 1, __ATOMIC_RELEASE);
      }
      span_end = run_end;
    }
    position = run_end;
  }
  if (ok) {
    lupine_finish_demand_fetch(allocation, span_start, span_end);
  } else {
    __atomic_store_n(&allocation->device_stale, 1, __ATOMIC_RELEASE);
  }
  __atomic_store_n(&allocation->stale_fetch_tid, 0, __ATOMIC_RELEASE);
  return ok;
}

// Materialize stale data before the connection is held, then return the
// permanent W alias so later R invalidation cannot fault inside the RPC.
extern "C" const void *lupine_mapped_host_read_source(const void *host,
                                                      size_t size) {
  if (host == nullptr || size == 0) {
    return host;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  if (start > UINTPTR_MAX - size) {
    return host;
  }
  struct pending_fetch {
    lupine_host_allocation *allocation;
    size_t start_offset;
    size_t end_offset;
  };
  std::vector<pending_fetch> fetches;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      auto &allocation = entry.second;
      if (!allocation.tracking_enabled ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0 ||
          __atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE) == 0) {
        continue;
      }
      uintptr_t base = reinterpret_cast<uintptr_t>(entry.first);
      if (base + allocation.size <= start || base >= start + size) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
      size_t overlap_start = start > base ? start - base : 0;
      size_t overlap_end = std::min(start + size - base, allocation.size);
      fetches.push_back({&allocation, overlap_start, overlap_end});
    }
  }
  for (const auto &fetch : fetches) {
    lupine_make_mapped_range_fresh(fetch.allocation, fetch.start_offset,
                                   fetch.end_offset);
    __atomic_sub_fetch(&fetch.allocation->pending_dirty_ranges, 1,
                       __ATOMIC_RELEASE);
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
  if (it == lupine_mutable_host_allocations_locked().end() ||
      it->second.io_alias == nullptr) {
    return host;
  }
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  if (start + size > base + it->second.size) {
    return host;
  }
  return static_cast<unsigned char *>(it->second.io_alias) + (start - base);
}

// A write through the writable alias raises no fault, so nothing records it.
// Queue the span instead, so the next request flushes it back to the server's
// copy of the mapping the way a faulting write would have.
static void lupine_note_mapped_host_write(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.tracking_enabled) {
    return;
  }
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t end = std::min(start + size, base + it->second.size);
  lupine_queue_dirty_host_range(&it->second, start, end);
}

// Servicing a fault means a copy, and a copy inside a capture window
// invalidates the capture.
extern "C" void lupine_materialize_host_allocations() {
  std::vector<lupine_host_allocation *> stale;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      if (entry.second.tracking_enabled &&
          __atomic_load_n(&entry.second.device_stale, __ATOMIC_ACQUIRE) != 0) {
        stale.push_back(&entry.second);
      }
    }
  }
  for (lupine_host_allocation *allocation : stale) {
    lupine_make_mapped_range_fresh(allocation, 0, allocation->size);
  }
}

extern "C" CUresult lupine_sync_mapped_device_to_host() {
  if (__atomic_exchange_n(&lupine_device_work_pending, 0, __ATOMIC_ACQ_REL) ==
      0) {
    return CUDA_SUCCESS;
  }

  for (const auto &mapping : lupine_mapped_host_snapshots()) {
    // A host allocation is identity mapped and reachable through the pointer
    // the caller already holds; a registration's device address is unrelated,
    // so nothing reaches it until cuMemHostGetDevicePointer hands one out.
    if (mapping.data_bytes == 0 ||
        (!mapping.host_allocation && !mapping.device_pointer_exposed)) {
      continue;
    }
    bool invalidated = false;
    bool give_up = false;
    for (int attempt = 0; !invalidated && !give_up; ++attempt) {
      {
        std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
        auto it = lupine_mutable_host_allocations_locked().find(mapping.host);
        if (it == lupine_mutable_host_allocations_locked().end()) {
          break;
        }
        auto &allocation = it->second;
        conn_t *conn = lupine_route_remote_conn(
            lupine_route_from_identity(allocation.route_id));
        if (!allocation.tracking_enabled || conn == nullptr ||
            __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
          give_up = true;
          break;
        }
        sig_atomic_t previous =
            __atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE);
        if (previous != 2) {
          sig_atomic_t expected = previous;
          if (__atomic_compare_exchange_n(&allocation.device_stale, &expected,
                                          3, false, __ATOMIC_ACQ_REL,
                                          __ATOMIC_ACQUIRE)) {
            allocation.stale_fetch_conn = conn;
            if (allocation.fresh_chunks != nullptr) {
              memset(allocation.fresh_chunks, 0, allocation.fresh_chunk_count);
            }
            allocation.fetch_seq_next = 0;
            allocation.fetch_readahead = 0;
            if (lupine_protect_host_range(mapping.host, allocation.storage_size,
                                          PROT_NONE)) {
              __atomic_store_n(&allocation.device_stale, 1, __ATOMIC_RELEASE);
              invalidated = true;
            } else {
              __atomic_store_n(&allocation.device_stale, previous,
                               __ATOMIC_RELEASE);
              give_up = true;
            }
          }
        }
      }
      if (!invalidated && !give_up) {
        if (attempt >= LUPINE_MAPPED_INVALIDATE_MAX_ATTEMPTS) {
          give_up = true;
        } else {
          sched_yield();
        }
      }
    }
    if (invalidated) {
      continue;
    }
    CUresult result = cuMemcpyDtoH_v2(
        static_cast<unsigned char *>(mapping.host) + mapping.data_offset,
        mapping.device_ptr + mapping.data_offset, mapping.data_bytes);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  // Deferred DtoH payloads land through the writable alias. Apply those
  // writes after invalidation so their pages remain readable and a later
  // device-bound RPC can flush them back to the server copy.
  for (int route_id = 0; route_id < rpc_size(); ++route_id) {
    conn_t *conn = lupine_route_remote_conn(
        lupine_route_from_identity(static_cast<int>(route_id)));
    if (conn != nullptr) {
      CUresult result = lupine_collect_host_allocation_writes(conn);
      if (result != CUDA_SUCCESS) {
        return result;
      }
    }
  }
  return CUDA_SUCCESS;
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p) {
  auto &allocations = lupine_mutable_host_allocations_locked();
  auto exact = allocations.find(p);
  if (exact != allocations.end()) {
    return exact;
  }

  auto upper = allocations.upper_bound(p);
  if (upper == allocations.begin()) {
    return allocations.end();
  }

  auto it = std::prev(upper);
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  if (addr >= base && addr < base + it->second.size) {
    return it;
  }
  return allocations.end();
}

static void lupine_covering_pages(void *p, size_t bytesize, uintptr_t *base,
                                  size_t *size) {
  size_t page_size = lupine_page_size();
  uintptr_t start = reinterpret_cast<uintptr_t>(p);
  uintptr_t page_base = start & ~(static_cast<uintptr_t>(page_size) - 1);
  uintptr_t end = lupine_round_up(start + bytesize, page_size);
  *base = page_base;
  *size = end - page_base;
}

static bool lupine_host_pages_registered_locked(uintptr_t base, size_t size) {
  auto &allocations = lupine_mutable_host_allocations_locked();
  uintptr_t end = base + size;
  auto upper = allocations.upper_bound(reinterpret_cast<void *>(base));
  if (upper != allocations.begin()) {
    auto prev = std::prev(upper);
    uintptr_t prev_base = 0;
    size_t prev_size = 0;
    size_t prev_span = std::max(prev->second.size, prev->second.storage_size);
    lupine_covering_pages(prev->first, prev_span == 0 ? 1 : prev_span,
                          &prev_base, &prev_size);
    if (prev_base + prev_size > base) {
      return true;
    }
  }
  return upper != allocations.end() &&
         reinterpret_cast<uintptr_t>(upper->first) < end;
}

// The response carries the server-side device alias alongside the host
// pointer, so a mapped allocation costs one round trip instead of two. The
// alias is 0 when the allocation is not device-mapped or the server could not
// resolve one; callers fall back to querying it on first use.
static CUresult lupine_remote_cuMemHostAlloc(void **remote_host,
                                             CUdeviceptr *device_ptr,
                                             size_t bytesize,
                                             unsigned int flags,
                                             lupine_route route) {
  if (remote_host == nullptr || device_ptr == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *device_ptr = 0;
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemHostAlloc", remote_host, bytesize,
                                    flags);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  *remote_host = nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemHostAlloc) < 0 ||
      rpc_write(conn, remote_host, sizeof(*remote_host)) < 0 ||
      rpc_write(conn, &bytesize, sizeof(bytesize)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, remote_host, sizeof(*remote_host)) < 0 ||
      rpc_read(conn, device_ptr, sizeof(*device_ptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    *device_ptr = 0;
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_remote_cuMemFreeHost(void *remote_host,
                                            lupine_route route) {
  if (remote_host == nullptr) {
    return CUDA_SUCCESS;
  }
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemFreeHost", remote_host);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemFreeHost) < 0 ||
      rpc_write(conn, &remote_host, sizeof(remote_host)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_remote_cuMemHostGetDevicePointer(CUdeviceptr *device_ptr,
                                                        void *remote_host,
                                                        unsigned int flags,
                                                        lupine_route route) {
  if (device_ptr == nullptr || remote_host == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2", device_ptr,
                                    remote_host, flags);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  *device_ptr = 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemHostGetDevicePointer_v2) < 0 ||
      rpc_write(conn, device_ptr, sizeof(*device_ptr)) < 0 ||
      rpc_write(conn, &remote_host, sizeof(remote_host)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, device_ptr, sizeof(*device_ptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_remote_cuMemHostGetFlags(unsigned int *flags,
                                                void *remote_host,
                                                lupine_route route) {
  if (flags == nullptr || remote_host == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemHostGetFlags", flags, remote_host);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemHostGetFlags) < 0 ||
      rpc_write(conn, flags, sizeof(*flags)) < 0 ||
      rpc_write(conn, &remote_host, sizeof(remote_host)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(*flags)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult lupine_register_host(void *p, size_t bytesize,
                                     unsigned int flags,
                                     bool client_to_server_only);

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize,
                                   unsigned int Flags) {
  if (pp == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr unsigned int supported_flags = CU_MEMHOSTALLOC_PORTABLE |
                                           CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    CUresult result =
        lupine_call_real_cuda_fn("cuMemHostAlloc", pp, bytesize, Flags);
    if (result != CUDA_SUCCESS) {
      return result;
    }

    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = bytesize;
    allocation.page_size = lupine_page_size();
    allocation.page_count =
        lupine_round_up(bytesize, allocation.page_size) / allocation.page_size;
    allocation.user_base = reinterpret_cast<uintptr_t>(*pp);
    allocation.user_size = bytesize;
    allocation.flags = Flags;
    allocation.owned = true;
    allocation.local_cuda = true;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(*pp);
    allocation.route_id = lupine_route_identity(route);
    {
      std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
      if (!lupine_mutable_host_allocations_locked()
               .emplace(*pp, std::move(allocation))
               .second) {
        (void)lupine_call_real_cuda_fn("cuMemFreeHost", *pp);
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
    lupine_note_deviceptr_allocation_route(reinterpret_cast<CUdeviceptr>(*pp),
                                           bytesize, route);
    return CUDA_SUCCESS;
  }

  void *remote_host = nullptr;
  CUdeviceptr device_ptr = 0;
  if (bytesize != 0) {
    CUresult result =
        lupine_remote_cuMemHostAlloc(&remote_host, &device_ptr, bytesize,
                                     Flags | CU_MEMHOSTALLOC_DEVICEMAP, route);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  void *ptr = nullptr;
  void *io_alias = nullptr;
  size_t page_size = lupine_page_size();
  size_t storage_size =
      bytesize == 0 ? page_size : lupine_round_up(bytesize, page_size);
  bool mapped = false;
  if (bytesize == 0) {
    ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mapped = ptr != MAP_FAILED;
    if (!mapped) {
      ptr = nullptr;
    }
  } else {
    conn_t *conn = lupine_route_remote_conn(route);
    mapped = lupine_create_shared_views(
        conn, reinterpret_cast<uintptr_t>(remote_host), storage_size, &ptr,
        &io_alias);
  }
  if (!mapped) {
    lupine_remote_cuMemFreeHost(remote_host, route);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.user_base = reinterpret_cast<uintptr_t>(ptr);
    allocation.user_size = bytesize;
    allocation.flags = Flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.io_alias = io_alias;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(remote_host);
    allocation.device_ptr = device_ptr;
    allocation.route_id = lupine_route_identity(route);
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto inserted = allocations.emplace(ptr, std::move(allocation));
    if (!inserted.second) {
      if (io_alias != nullptr) {
        lupine_release_shared_views(ptr, io_alias, storage_size);
      } else {
        munmap(ptr, storage_size);
      }
      lupine_remote_cuMemFreeHost(remote_host, route);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      allocations.erase(inserted.first);
      if (io_alias != nullptr) {
        lupine_release_shared_views(ptr, io_alias, storage_size);
      } else {
        munmap(ptr, storage_size);
      }
      lupine_remote_cuMemFreeHost(remote_host, route);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (remote_host != nullptr) {
    lupine_note_deviceptr_allocation_route(
        reinterpret_cast<CUdeviceptr>(remote_host), bytesize, route);
    lupine_note_deviceptr_allocation_route(device_ptr, bytesize, route);
  }
  *pp = ptr;
  LUPINE_TRACE_LOG("LUPINE local cuMemHostAlloc ptr="
                   << ptr << " remote=" << remote_host << " bytes=" << bytesize
                   << " flags=" << Flags);
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}

#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
extern "C" CUresult cuMemAllocHost(void **pp, size_t bytesize) {
  return cuMemAllocHost_v2(pp, bytesize);
}

extern "C" CUresult cuMemFreeHost(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }

  bool owned = false;
  bool owned_mmap = false;
  bool local_cuda = false;
  void *io_alias = nullptr;
  size_t storage_size = 0;
  CUdeviceptr server_host_ptr = 0;
  CUdeviceptr device_ptr = 0;
  lupine_host_allocation *retiring_allocation = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = allocations.find(p);
    if (it == allocations.end()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    owned = it->second.owned;
    owned_mmap = it->second.owned_mmap;
    local_cuda = it->second.local_cuda;
    io_alias = it->second.io_alias;
    storage_size = it->second.storage_size;
    server_host_ptr = it->second.server_host_ptr;
    device_ptr = it->second.device_ptr;
    __atomic_store_n(&it->second.retiring, 1, __ATOMIC_RELEASE);
    lupine_pointer_attribute_cache_clear();
    lupine_disable_dirty_tracking(p, it->second);
    retiring_allocation = &it->second;
  }
  flush_result = lupine_drain_retiring_dirty_ranges(retiring_allocation);
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = allocations.find(p);
    if (it == allocations.end() || &it->second != retiring_allocation) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    allocations.erase(it);
  }
  if (local_cuda) {
    CUresult result = lupine_call_real_cuda_fn("cuMemFreeHost", p);
    if (result == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(reinterpret_cast<CUdeviceptr>(p));
      if (device_ptr != 0) {
        lupine_forget_deviceptr_owner(device_ptr);
      }
    }
    return result;
  }
  if (owned) {
    if (owned_mmap) {
      if (io_alias != nullptr) {
        lupine_release_shared_views(p, io_alias, storage_size);
      } else {
        munmap(p, storage_size);
      }
    } else {
      free(p);
    }
  }
  CUresult result = CUDA_SUCCESS;
  if (server_host_ptr != 0) {
    lupine_route route = lupine_route_for_deviceptr(server_host_ptr);
    result = lupine_remote_cuMemFreeHost(
        reinterpret_cast<void *>(server_host_ptr), route);
    if (result == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(server_host_ptr);
      if (device_ptr != 0) {
        lupine_forget_deviceptr_owner(device_ptr);
      }
    }
  }
  return result;
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p,
                                                 unsigned int Flags) {
  if (pdptr == nullptr || p == nullptr || Flags != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  void *allocation_host = nullptr;
  void *remote_host = nullptr;
  size_t allocation_size = 0;
  bool local_cuda = false;
  bool known_allocation = false;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(p);
    if (it != lupine_mutable_host_allocations_locked().end()) {
      known_allocation = true;
      if (it->second.device_ptr != 0) {
        if (!it->second.device_pointer_exposed) {
          it->second.device_pointer_exposed = true;
          __atomic_add_fetch(&lupine_dirty_host_epoch, 1, __ATOMIC_RELEASE);
        }
        uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        *pdptr = it->second.device_ptr + (addr - base);
        return CUDA_SUCCESS;
      }
      allocation_host = it->first;
      remote_host = reinterpret_cast<void *>(it->second.server_host_ptr);
      allocation_size = it->second.size;
      local_cuda = it->second.local_cuda;
    }
  }

  if (!known_allocation) {
    lupine_route route = lupine_route_for_default();
    if (lupine_route_is_local(route)) {
      return lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2", pdptr, p,
                                      Flags);
    }

    size_t page_size = lupine_page_size();
    uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~(page_size - 1);
    CUresult result =
        lupine_register_host(reinterpret_cast<void *>(page), page_size,
                             CU_MEMHOSTREGISTER_DEVICEMAP, true);
    if (result != CUDA_SUCCESS &&
        result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
      return result;
    }
    return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
  }

  CUdeviceptr new_device_ptr = 0;
  lupine_route route =
      lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(remote_host));
  CUresult alloc_result;
  if (local_cuda) {
    alloc_result =
        lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2",
                                 &new_device_ptr, allocation_host, Flags);
  } else {
    alloc_result = lupine_remote_cuMemHostGetDevicePointer(
        &new_device_ptr, remote_host, Flags, route);
  }
  if (alloc_result != CUDA_SUCCESS) {
    return alloc_result;
  }

  CUresult result = CUDA_SUCCESS;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(p);
    if (it == lupine_mutable_host_allocations_locked().end()) {
      result = CUDA_ERROR_INVALID_VALUE;
    } else {
      if (it->second.device_ptr == 0) {
        it->second.device_ptr = new_device_ptr;
        lupine_note_deviceptr_allocation_route(new_device_ptr, allocation_size,
                                               route);
        if (!lupine_enable_dirty_tracking_locked(allocation_host,
                                                 &it->second)) {
          it->second.device_ptr = 0;
          result = CUDA_ERROR_OUT_OF_MEMORY;
        }
      }
      if (result == CUDA_SUCCESS) {
        uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        *pdptr = it->second.device_ptr + (addr - base);
      }
    }
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return CUDA_SUCCESS;
}

#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
extern "C" CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                              unsigned int Flags) {
  return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

extern "C" CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (pFlags == nullptr || p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  void *remote_host = nullptr;
  unsigned int local_flags = 0;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(p);
    if (it == lupine_mutable_host_allocations_locked().end()) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    remote_host = reinterpret_cast<void *>(it->second.server_host_ptr);
    local_flags = it->second.flags;
  }

  if (remote_host != nullptr) {
    unsigned int flags = 0;
    lupine_route route =
        lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(remote_host));
    CUresult result =
        lupine_remote_cuMemHostGetFlags(&flags, remote_host, route);
    if (result == CUDA_SUCCESS) {
      *pFlags = flags;
    }
    return result;
  }
  *pFlags = local_flags;
  return CUDA_SUCCESS;
}

static CUresult lupine_register_host(void *p, size_t bytesize,
                                     unsigned int Flags,
                                     bool client_to_server_only) {
  if (p == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  constexpr unsigned int supported_flags =
      CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP |
      CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_READ_ONLY;
  if ((Flags & ~supported_flags) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  uintptr_t covering_base = 0;
  size_t covering_size = 0;
  lupine_covering_pages(p, bytesize, &covering_base, &covering_size);

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (lupine_host_pages_registered_locked(covering_base, covering_size)) {
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    }
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    CUresult result =
        lupine_call_real_cuda_fn("cuMemHostRegister_v2", p, bytesize, Flags);
    if (result != CUDA_SUCCESS) {
      return result;
    }

    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = bytesize;
    allocation.page_size = lupine_page_size();
    allocation.page_count =
        lupine_round_up(bytesize, allocation.page_size) / allocation.page_size;
    allocation.flags = Flags;
    allocation.user_base = reinterpret_cast<uintptr_t>(p);
    allocation.user_size = bytesize;
    allocation.local_cuda = true;
    allocation.client_to_server_only = client_to_server_only;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(p);
    allocation.route_id = lupine_route_identity(route);
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (!lupine_mutable_host_allocations_locked()
             .emplace(p, std::move(allocation))
             .second) {
      (void)lupine_call_real_cuda_fn("cuMemHostUnregister", p);
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    }
    return CUDA_SUCCESS;
  }

  size_t page_size = lupine_page_size();
  void *tracked = reinterpret_cast<void *>(covering_base);

  void *server_host = nullptr;
  CUdeviceptr device_ptr = 0;
  if (lupine_host_flags_request_mapping(Flags)) {
    unsigned int host_flags = CU_MEMHOSTALLOC_DEVICEMAP;
    if ((Flags & CU_MEMHOSTREGISTER_PORTABLE) != 0) {
      host_flags |= CU_MEMHOSTALLOC_PORTABLE;
    }
    CUresult result = lupine_remote_cuMemHostAlloc(
        &server_host, &device_ptr, covering_size, host_flags, route);
    if (result != CUDA_SUCCESS) {
      if (server_host != nullptr) {
        lupine_remote_cuMemFreeHost(server_host, route);
      }
      return result;
    }
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto &allocations = lupine_mutable_host_allocations_locked();
  if (lupine_host_pages_registered_locked(covering_base, covering_size)) {
    if (server_host != nullptr) {
      lupine_remote_cuMemFreeHost(server_host, route);
    }
    return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  }
  lupine_host_allocation allocation;
  allocation.size = covering_size;
  allocation.storage_size = covering_size;
  allocation.user_base = reinterpret_cast<uintptr_t>(p);
  allocation.user_size = bytesize;
  allocation.data_offset = reinterpret_cast<uintptr_t>(p) - covering_base;
  allocation.page_size = page_size;
  allocation.page_count = covering_size / page_size;
  allocation.flags = Flags;
  allocation.owned = false;
  allocation.owned_mmap = false;
  allocation.client_to_server_only = client_to_server_only;
  allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(server_host);
  allocation.device_ptr = device_ptr;
  allocation.host_base = covering_base;
  allocation.route_id = lupine_route_identity(route);
  allocations.emplace(tracked, std::move(allocation));
  if (server_host != nullptr) {
    lupine_note_deviceptr_allocation_route(
        reinterpret_cast<CUdeviceptr>(server_host), covering_size, route);
    lupine_note_deviceptr_allocation_route(device_ptr, covering_size, route);
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemHostRegister_v2(void *p, size_t bytesize,
                                         unsigned int Flags) {
  return lupine_register_host(p, bytesize, Flags, false);
}

#ifdef cuMemHostRegister
#undef cuMemHostRegister
#endif
extern "C" CUresult cuMemHostRegister(void *p, size_t bytesize,
                                      unsigned int Flags) {
  return cuMemHostRegister_v2(p, bytesize, Flags);
}

extern "C" CUresult cuMemHostUnregister(void *p) {
  if (p == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  bool local_cuda = false;
  CUdeviceptr server_host_ptr = 0;
  CUdeviceptr device_ptr = 0;
  void *tracked = nullptr;
  lupine_host_allocation *retiring_allocation = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = lupine_find_host_allocation_locked(p);
    if (it == allocations.end()) {
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    if (it->second.owned ||
        it->second.user_base != reinterpret_cast<uintptr_t>(p)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    tracked = it->first;
    local_cuda = it->second.local_cuda;
    server_host_ptr = it->second.server_host_ptr;
    device_ptr = it->second.device_ptr;
    __atomic_store_n(&it->second.retiring, 1, __ATOMIC_RELEASE);
    lupine_pointer_attribute_cache_clear();
    lupine_disable_dirty_tracking(tracked, it->second);
    retiring_allocation = &it->second;
  }
  flush_result = lupine_drain_retiring_dirty_ranges(retiring_allocation);
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = allocations.find(tracked);
    if (it == allocations.end() || &it->second != retiring_allocation) {
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    allocations.erase(it);
  }
  if (local_cuda) {
    CUresult result = lupine_call_real_cuda_fn("cuMemHostUnregister", p);
    if (result == CUDA_SUCCESS && device_ptr != 0) {
      lupine_forget_deviceptr_owner(device_ptr);
    }
    return result;
  }
  CUresult result = CUDA_SUCCESS;
  if (server_host_ptr != 0) {
    lupine_route route = lupine_route_for_deviceptr(server_host_ptr);
    result = lupine_remote_cuMemFreeHost(
        reinterpret_cast<void *>(server_host_ptr), route);
    if (result == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(server_host_ptr);
      if (device_ptr != 0) {
        lupine_forget_deviceptr_owner(device_ptr);
      }
    }
  }
  return result;
}

extern "C" CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                      unsigned int flags) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    CUresult result =
        lupine_call_real_cuda_fn("cuMemAllocManaged", dptr, bytesize, flags);
    if (result == CUDA_SUCCESS) {
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    }
    return result;
  }

  size_t page_size = lupine_page_size();
  size_t storage_size = lupine_round_up(bytesize, page_size);
  CUdeviceptr device_alloc_base = 0;
  size_t backing_size = std::max(bytesize, LUPINE_MANAGED_ALLOCATION_MIN_BYTES);
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemAllocManaged) < 0 ||
      rpc_write(conn, &device_alloc_base, sizeof(device_alloc_base)) < 0 ||
      rpc_write(conn, &backing_size, sizeof(backing_size)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &device_alloc_base, sizeof(device_alloc_base)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUdeviceptr device_ptr = device_alloc_base;
  lupine_note_deviceptr_allocation_route(device_alloc_base, backing_size,
                                         route);
  void *ptr = nullptr;
  void *io_alias = nullptr;
  if (!lupine_create_shared_views(conn, device_ptr, storage_size, &ptr,
                                  &io_alias)) {
    cuMemFree_v2(device_alloc_base);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.user_base = reinterpret_cast<uintptr_t>(ptr);
    allocation.user_size = bytesize;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.io_alias = io_alias;
    allocation.managed = true;
    allocation.server_host_ptr = device_ptr;
    allocation.device_ptr = device_ptr;
    allocation.device_alloc_base = device_alloc_base;
    allocation.route_id = lupine_route_identity(route);
    auto inserted = lupine_mutable_host_allocations_locked().emplace(
        ptr, std::move(allocation));
    if (!inserted.second) {
      lupine_release_shared_views(ptr, io_alias, storage_size);
      cuMemFree_v2(device_alloc_base);
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      lupine_mutable_host_allocations_locked().erase(inserted.first);
      lupine_release_shared_views(ptr, io_alias, storage_size);
      cuMemFree_v2(device_alloc_base);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }

  *dptr = reinterpret_cast<CUdeviceptr>(ptr);
  lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }

  void *host = reinterpret_cast<void *>(dptr);
  lupine_host_allocation allocation;
  lupine_host_allocation *retiring_allocation = nullptr;
  bool found = false;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(host);
    if (it != lupine_mutable_host_allocations_locked().end() &&
        reinterpret_cast<void *>(dptr) == it->first && it->second.managed) {
      __atomic_store_n(&it->second.retiring, 1, __ATOMIC_RELEASE);
      lupine_pointer_attribute_cache_clear();
      lupine_disable_dirty_tracking(it->first, it->second);
      retiring_allocation = &it->second;
      found = true;
    }
  }
  if (!found) {
    lupine_route route = lupine_route_for_deviceptr(dptr);
    if (lupine_route_is_local(route)) {
      CUresult result = lupine_call_real_cuda_fn("cuMemFree_v2", dptr);
      if (result == CUDA_SUCCESS) {
        lupine_forget_deviceptr_owner(dptr);
      }
      return result;
    }
    conn_t *conn = lupine_route_remote_conn(route);
    CUresult return_value;
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemFree_v2) < 0 ||
        rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(dptr);
    }
    return return_value;
  }

  flush_result = lupine_drain_retiring_dirty_ranges(retiring_allocation);
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_mutable_host_allocations_locked().find(host);
    if (it == lupine_mutable_host_allocations_locked().end() ||
        &it->second != retiring_allocation) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    allocation = std::move(it->second);
    lupine_mutable_host_allocations_locked().erase(it);
  }

  CUresult result = CUDA_SUCCESS;
  if (allocation.device_ptr != 0) {
    CUdeviceptr free_ptr = allocation.device_alloc_base != 0
                               ? allocation.device_alloc_base
                               : allocation.device_ptr;
    result = cuMemFree_v2(free_ptr);
  }
  lupine_forget_deviceptr_owner(reinterpret_cast<CUdeviceptr>(host));
  lupine_forget_deviceptr_owner(allocation.server_host_ptr);
  lupine_forget_deviceptr_owner(allocation.device_ptr);
  if (allocation.owned_mmap && host != nullptr) {
    if (allocation.io_alias != nullptr) {
      lupine_release_shared_views(host, allocation.io_alias,
                                  allocation.storage_size);
    } else {
      munmap(host, allocation.storage_size);
    }
  } else if (allocation.owned && host != nullptr) {
    free(host);
  }
  return result;
}

#ifdef cuMemFree
#undef cuMemFree
#endif
extern "C" CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

extern "C" CUresult cuPointerGetAttribute(void *data,
                                          CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
  if (data == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t value_size = 0;
  if (!lupine_pointer_attribute_size(attribute, &value_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  unsigned char value[64] = {};
  if (value_size > sizeof(value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_is_managed_host_alias(ptr);
  CUdeviceptr remote_host_base = 0;
  CUdeviceptr remote_device_base = 0;
  bool remote_host_alias = false;
  if (!managed_alias) {
    remote_host_alias = lupine_translate_client_host_ptr_to_server(
        ptr, &query_ptr, &remote_host_base, &remote_device_base);
  }
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuPointerGetAttribute", data, attribute,
                                    query_ptr);
  }
  // Identity-VA managed allocations use device VMM on the server, so the
  // server has no host alias to report. Return the client allocation instead.
  if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
    void *host = reinterpret_cast<void *>(ptr);
    memcpy(data, &host, sizeof(host));
    return CUDA_SUCCESS;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuPointerGetAttribute) < 0 ||
      rpc_write(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_write(conn, &query_ptr, sizeof(query_ptr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value, value_size) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (return_value == CUDA_SUCCESS) {
    memcpy(data, value, value_size);
    if ((managed_alias || remote_host_alias) &&
        attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
      void *host = reinterpret_cast<void *>(ptr);
      memcpy(data, &host, sizeof(host));
    } else if (remote_host_alias &&
               attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
      CUdeviceptr device_alias =
          remote_device_base + (query_ptr - remote_host_base);
      memcpy(data, &device_alias, sizeof(device_alias));
    } else if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
      int is_managed = 1;
      memcpy(data, &is_managed, sizeof(is_managed));
    }
  }
  return return_value;
}

// `value` points at the attribute payload, so the wire must carry the bytes it
// points to. Sending the pointer itself would make the server dereference an
// address from this process's address space.
extern "C" CUresult cuPointerSetAttribute(const void *value,
                                          CUpointer_attribute attribute,
                                          CUdeviceptr ptr) {
  if (value == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUdeviceptr target_ptr = ptr;
  if (!lupine_is_managed_host_alias(ptr)) {
    lupine_translate_client_host_ptr_to_server(ptr, &target_ptr);
  }
  lupine_route route = lupine_route_for_deviceptr(target_ptr);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuPointerSetAttribute", value, attribute,
                                    target_ptr);
  }

  size_t value_size = 0;
  if (!lupine_settable_pointer_attribute_size(attribute, &value_size)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  unsigned char payload[64] = {};
  if (value_size > sizeof(payload)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  memcpy(payload, value, value_size);

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuPointerSetAttribute) < 0 ||
      rpc_write(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_write(conn, &target_ptr, sizeof(target_ptr)) < 0 ||
      rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
      rpc_write(conn, payload, value_size) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

// Attribute values of a live mapped-host allocation are immutable, and
// PyTorch's pin-memory path queries the same pinned buffers on every batch,
// so the server's answers are cached per (pointer, attribute list). The cache
// is cleared whenever any mapped allocation retires, since a later allocation
// may reuse the address.
struct lupine_pointer_attribute_cache_key {
  CUdeviceptr ptr = 0;
  std::vector<CUpointer_attribute> attributes;

  bool operator==(const lupine_pointer_attribute_cache_key &other) const {
    return ptr == other.ptr && attributes == other.attributes;
  }
};

struct lupine_pointer_attribute_cache_key_hash {
  size_t operator()(const lupine_pointer_attribute_cache_key &key) const {
    size_t hash = std::hash<CUdeviceptr>()(key.ptr);
    for (CUpointer_attribute attribute : key.attributes) {
      hash = hash * 31 + static_cast<size_t>(attribute);
    }
    return hash;
  }
};

static libcuckoo::cuckoohash_map<lupine_pointer_attribute_cache_key,
                                 std::vector<std::vector<unsigned char>>,
                                 lupine_pointer_attribute_cache_key_hash> &
lupine_pointer_attribute_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<lupine_pointer_attribute_cache_key,
                                    std::vector<std::vector<unsigned char>>,
                                    lupine_pointer_attribute_cache_key_hash>();
  return *cache;
}

static void lupine_pointer_attribute_cache_clear() {
  lupine_pointer_attribute_cache().clear();
}

extern "C" CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                           CUpointer_attribute *attributes,
                                           void **data, CUdeviceptr ptr) {
  if (numAttributes != 0 && (attributes == nullptr || data == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::vector<size_t> value_sizes(numAttributes, 0);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    if (!lupine_pointer_attribute_size(attributes[i], &value_sizes[i])) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_is_managed_host_alias(ptr);
  CUdeviceptr remote_host_base = 0;
  CUdeviceptr remote_device_base = 0;
  bool remote_host_alias = false;
  if (!managed_alias) {
    remote_host_alias = lupine_translate_client_host_ptr_to_server(
        ptr, &query_ptr, &remote_host_base, &remote_device_base);
  }
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuPointerGetAttributes", numAttributes,
                                    attributes, data, query_ptr);
  }

  // A pointer in neither the host-allocation registry nor the device-pointer
  // tables is unregistered host memory, for which the driver's answer is
  // deterministic (probed on a native driver: CUDA_SUCCESS; both pointer
  // aliases echo the query; type/managed/mapped are zero; ordinal is -1;
  // context and buffer id are zero; the range attributes are left untouched).
  // PyTorch's pin_memory path asks this about every unpinned source tensor,
  // so answering locally removes a per-batch round trip. Attributes outside
  // the probed set fall through to the server.
  if (!managed_alias && !lupine_host_ptr_is_tracked(ptr) &&
      !lupine_deviceptr_is_tracked(ptr)) {
    bool replicable = true;
    for (unsigned int i = 0; replicable && i < numAttributes; ++i) {
      switch (attributes[i]) {
      case CU_POINTER_ATTRIBUTE_CONTEXT:
      case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
      case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
      case CU_POINTER_ATTRIBUTE_HOST_POINTER:
      case CU_POINTER_ATTRIBUTE_IS_MANAGED:
      case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
      case CU_POINTER_ATTRIBUTE_MAPPED:
      case CU_POINTER_ATTRIBUTE_BUFFER_ID:
      case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
      case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
        break;
      default:
        replicable = false;
        break;
      }
      if (replicable && data[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
    }
    if (replicable) {
      for (unsigned int i = 0; i < numAttributes; ++i) {
        switch (attributes[i]) {
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
          memcpy(data[i], &ptr, value_sizes[i]);
          break;
        }
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
          int ordinal = -1;
          memcpy(data[i], &ordinal, value_sizes[i]);
          break;
        }
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
          break;
        default:
          memset(data[i], 0, value_sizes[i]);
          break;
        }
      }
      return CUDA_SUCCESS;
    }
  }

  // Keyed by the owning allocation, not the query pointer: PyTorch's pinned
  // pool hands out tensors at varying offsets inside a few allocations, and
  // every attribute except the pointer aliases is offset-independent. The
  // aliases are recomputed for the queried pointer on a hit.
  lupine_pointer_attribute_cache_key cache_key;
  std::vector<std::vector<unsigned char>> cached_values;
  bool cacheable = false;
  if (managed_alias) {
    cacheable = lupine_managed_host_alias_base(ptr, &cache_key.ptr);
  } else if (remote_host_alias) {
    cache_key.ptr = remote_host_base;
    cacheable = true;
  }
  if (cacheable) {
    cache_key.attributes.assign(attributes, attributes + numAttributes);
    if (lupine_pointer_attribute_cache().find(cache_key, cached_values)) {
      for (unsigned int i = 0; i < numAttributes; ++i) {
        if (data[i] == nullptr || cached_values[i].size() != value_sizes[i]) {
          return CUDA_ERROR_INVALID_VALUE;
        }
        memcpy(data[i], cached_values[i].data(), cached_values[i].size());
        if ((managed_alias || remote_host_alias) &&
            attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
          void *host = reinterpret_cast<void *>(ptr);
          memcpy(data[i], &host, sizeof(host));
        } else if (managed_alias &&
                   attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
          memcpy(data[i], &query_ptr, sizeof(query_ptr));
        } else if (remote_host_alias &&
                   attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
          CUdeviceptr device_alias =
              remote_device_base + (query_ptr - remote_host_base);
          memcpy(data[i], &device_alias, sizeof(device_alias));
        } else if (managed_alias &&
                   attributes[i] == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
          int is_managed = 1;
          memcpy(data[i], &is_managed, sizeof(is_managed));
        }
      }
      return CUDA_SUCCESS;
    }
  }

  std::vector<CUpointer_attribute> request_attributes;
  const CUpointer_attribute *rpc_attributes = attributes;
  if (managed_alias) {
    request_attributes.assign(attributes, attributes + numAttributes);
    for (auto &attribute : request_attributes) {
      if (attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
        attribute = CU_POINTER_ATTRIBUTE_DEVICE_POINTER;
      }
    }
    rpc_attributes = request_attributes.data();
  }

  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuPointerGetAttributes) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(numAttributes)) < 0 ||
      rpc_write(conn, rpc_attributes,
                numAttributes * sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &query_ptr, sizeof(query_ptr)) < 0 ||
      rpc_wait_for_response(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  std::vector<std::vector<unsigned char>> values(numAttributes);
  for (unsigned int i = 0; i < numAttributes; ++i) {
    size_t remote_size = 0;
    if (rpc_read(conn, &remote_size, sizeof(remote_size)) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    values[i].resize(remote_size);
    if (remote_size != 0 && rpc_read(conn, values[i].data(), remote_size) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }

  CUresult return_value;
  if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS) {
    for (unsigned int i = 0; i < numAttributes; ++i) {
      if (data[i] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      // Version skew: the server's width does not fit the caller's buffer.
      if (values[i].size() != value_sizes[i]) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      memcpy(data[i], values[i].data(), values[i].size());
      if ((managed_alias || remote_host_alias) &&
          attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
        void *host = reinterpret_cast<void *>(ptr);
        memcpy(data[i], &host, sizeof(host));
      } else if (remote_host_alias &&
                 attributes[i] == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
        CUdeviceptr device_alias =
            remote_device_base + (query_ptr - remote_host_base);
        memcpy(data[i], &device_alias, sizeof(device_alias));
      } else if (managed_alias &&
                 attributes[i] == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
        int is_managed = 1;
        memcpy(data[i], &is_managed, sizeof(is_managed));
      }
    }
    if (cacheable) {
      lupine_pointer_attribute_cache().insert_or_assign(cache_key,
                                                        std::move(values));
    }
  }
  return return_value;
}

// The chunks are staged in a bounded pool on the server, so a copy of any size
// costs the same fixed staging there.
// Every cuMemcpy* entry point lowers to this. The descriptor is already
// widened to three dimensions, so one wire op and one server handler cover
// linear, pitched and array transfers alike. `blocking` says whether the caller
// must not see the call return until the bytes have landed: true for the
// synchronous entry points, and true for an asynchronous one whose host side is
// pageable, because the driver completes those before returning.
static CUresult lupine_memcpy_htoh(const CUDA_MEMCPY3D &copy) {
  const size_t run = copy.WidthInBytes;
  const size_t rows = copy.Height;
  const size_t slices = copy.Depth;
  const size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  const size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  // A UNIFIED descriptor carries the host address in the device field.
  const char *from = (copy.srcMemoryType == CU_MEMORYTYPE_HOST
                          ? static_cast<const char *>(copy.srcHost)
                          : reinterpret_cast<const char *>(copy.srcDevice)) +
                     copy.srcZ * src_slice_pitch + copy.srcY * copy.srcPitch +
                     copy.srcXInBytes;
  char *to = (copy.dstMemoryType == CU_MEMORYTYPE_HOST
                  ? static_cast<char *>(copy.dstHost)
                  : reinterpret_cast<char *>(copy.dstDevice)) +
             copy.dstZ * dst_slice_pitch + copy.dstY * copy.dstPitch +
             copy.dstXInBytes;
  for (size_t slice = 0; slice < slices; ++slice) {
    for (size_t row = 0; row < rows; ++row) {
      memmove(to + slice * dst_slice_pitch + row * copy.dstPitch,
              from + slice * src_slice_pitch + row * copy.srcPitch, run);
    }
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_memcpy_htod(CUDA_MEMCPY3D copy, CUstream stream,
                                   bool blocking) {
  const size_t run = copy.WidthInBytes;
  const size_t rows = copy.Height;
  const size_t slices = copy.Depth;
  const size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  // The server stages the payload and hands it to the driver as host memory,
  // so the wire descriptor names the host end that way however the caller
  // spelled it.
  // A UNIFIED descriptor carries the host address in the device field.
  const void *host = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                         ? copy.srcHost
                         : reinterpret_cast<const void *>(copy.srcDevice);
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = host;

  lupine_route route = copy.dstMemoryType == CU_MEMORYTYPE_ARRAY
                           ? lupine_route_for_stream(stream)
                           : lupine_route_for_deviceptr(copy.dstDevice);
  if (lupine_route_is_local(route)) {
    // The driver's own synchronous copy, not an asynchronous one plus a wait:
    // synchronizing the stream would also wait on work this copy has nothing
    // to do with.
    return blocking
               ? lupine_call_real_cuda_fn("cuMemcpy3D_v2", &copy)
               : lupine_call_real_cuda_fn("cuMemcpy3DAsync_v2", &copy, stream);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  // A mapped host allocation may be stale; reading it has to go through the
  // mirror the fault handler maintains. That happens before the request opens,
  // because servicing a fault is itself a request on this thread.
  size_t span =
      (slices - 1) * src_slice_pitch + (rows - 1) * copy.srcPitch + run;
  const char *source = static_cast<const char *>(host) +
                       copy.srcZ * src_slice_pitch + copy.srcY * copy.srcPitch +
                       copy.srcXInBytes;
  CUdeviceptr server_source = 0;
  const bool use_server_source = lupine_translate_client_host_range_to_server(
      source, span, lupine_route_identity(route), &server_source);
  const char *source_rows = source;
  if (use_server_source) {
    copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
    copy.srcDevice = server_source;
    copy.srcXInBytes = 0;
    copy.srcY = 0;
    copy.srcZ = 0;
  } else {
    source_rows =
        static_cast<const char *>(lupine_mapped_host_read_source(source, span));
  }
  const uint8_t wire_flags =
      (blocking ? LUPINE_MEMCPY_BLOCKING : 0) |
      (use_server_source ? LUPINE_MEMCPY_SERVER_SOURCE : 0);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineMemcpy) < 0 ||
      rpc_write(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      (!use_server_source &&
       rpc_write_pitched_payload(conn, source_rows, run, rows, copy.srcPitch,
                                 slices, src_slice_pitch) < 0)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (!blocking) {
    // Fire and forget, as the driver's is: the payload is on its way and an
    // execution failure surfaces from the caller's next synchronize rather
    // than from here. The server answers nothing, so nothing is left unread.
    return rpc_write_end(conn) < 0 ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                   : CUDA_SUCCESS;
  }
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

static CUresult lupine_memcpy_dtoh(CUDA_MEMCPY3D copy, CUstream stream,
                                   bool blocking) {
  const size_t run = copy.WidthInBytes;
  const size_t rows = copy.Height;
  const size_t slices = copy.Depth;
  const size_t total = run * rows * slices;
  const size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  // A UNIFIED descriptor carries the host address in the device field.
  void *host = copy.dstMemoryType == CU_MEMORYTYPE_HOST
                   ? copy.dstHost
                   : reinterpret_cast<void *>(copy.dstDevice);
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = host;

  lupine_route route = copy.srcMemoryType == CU_MEMORYTYPE_ARRAY
                           ? lupine_route_for_stream(stream)
                           : lupine_route_for_deviceptr(copy.srcDevice);
  if (lupine_route_is_local(route)) {
    // The driver's own synchronous copy, not an asynchronous one plus a wait:
    // synchronizing the stream would also wait on work this copy has nothing
    // to do with.
    return blocking
               ? lupine_call_real_cuda_fn("cuMemcpy3D_v2", &copy)
               : lupine_call_real_cuda_fn("cuMemcpy3DAsync_v2", &copy, stream);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  // The response lands in the caller's buffer while the read is open, so a
  // stale mapped allocation is settled first: faulting mid-read would hand
  // this thread's read state to the fault handler's own request.
  char *written_destination = static_cast<char *>(host) +
                              copy.dstZ * dst_slice_pitch +
                              copy.dstY * copy.dstPitch + copy.dstXInBytes;
  size_t written_span =
      (slices - 1) * dst_slice_pitch + (rows - 1) * copy.dstPitch + run;
  char *destination_rows = const_cast<char *>(static_cast<const char *>(
      lupine_mapped_host_read_source(written_destination, written_span)));
  // A tracked pinned allocation has a server-side mirror. Tell the server where
  // this packed response belongs so it can update that mirror while the bytes
  // are already in its staging ring. The client then does not have to upload
  // the same bytes again before its next request.
  CUdeviceptr mirrored_destination = 0;
  bool managed_destination = false;
  if (!lupine_translate_client_host_range_to_server(
          written_destination, written_span, lupine_route_identity(route),
          &mirrored_destination,
          &managed_destination) ||
      managed_destination) {
    mirrored_destination = 0;
  }
  const bool use_server_destination = mirrored_destination != 0;
  const uint8_t wire_flags =
      (blocking ? LUPINE_MEMCPY_BLOCKING : 0) |
      (use_server_destination ? LUPINE_MEMCPY_SERVER_DESTINATION : 0);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineMemcpy) < 0 ||
      rpc_write(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      (use_server_destination && rpc_write(conn, &mirrored_destination,
                                           sizeof(mirrored_destination)) < 0)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  // The server answers one chunk at a time and states each chunk's size, so
  // the two ends never have to agree on how it was split.
  int request_id = rpc_write_end(conn);
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  const bool contiguous_destination =
      copy.dstPitch == run && (slices == 1 || dst_slice_pitch == run * rows);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  std::vector<unsigned char> staging;
  size_t offset = 0;
  while (offset < total) {
    uint64_t wire_bytes = 0;
    if (rpc_read(conn, &result, sizeof(result)) < 0 ||
        rpc_read(conn, &wire_bytes, sizeof(wire_bytes)) < 0) {
      rpc_read_end(conn);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (wire_bytes > total - offset) {
      rpc_read_end(conn);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    const size_t bytes = static_cast<size_t>(wire_bytes);
    if (bytes != 0 && contiguous_destination) {
      // The rows are back to back, so the payload lands in the caller's buffer
      // directly. This is also the path the fault handler takes, where a heap
      // allocation would not be safe.
      if (rpc_read_payload_part(conn, destination_rows + offset, bytes) < 0) {
        rpc_read_end(conn);
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      offset += bytes;
    } else if (bytes != 0) {
      try {
        if (staging.size() < bytes) {
          staging.resize(bytes);
        }
      } catch (...) {
        rpc_read_end(conn);
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      if (rpc_read_payload_part(conn, staging.data(), bytes) < 0) {
        rpc_read_end(conn);
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      for (size_t filled = 0; filled < bytes;) {
        size_t position = offset + filled;
        size_t index = position / run;
        size_t column = position % run;
        size_t span = std::min(run - column, bytes - filled);
        memcpy(destination_rows + (index / rows) * dst_slice_pitch +
                   (index % rows) * copy.dstPitch + column,
               staging.data() + filled, span);
        filled += span;
      }
      offset += bytes;
    }
    bool finished = result != CUDA_SUCCESS || offset == total;
    if (rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (finished) {
      if (result == CUDA_SUCCESS && mirrored_destination == 0) {
        lupine_note_mapped_host_write(written_destination, written_span);
      }
      return result;
    }
    if (rpc_read_start(conn, request_id) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  }
  return result;
}

static CUresult lupine_memcpy_dtod(CUDA_MEMCPY3D copy, CUstream stream,
                                   bool blocking) {
  const size_t run = copy.WidthInBytes;
  const size_t rows = copy.Height;
  const size_t slices = copy.Depth;
  const size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  const size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  if (copy.srcMemoryType == CU_MEMORYTYPE_DEVICE &&
      copy.dstMemoryType == CU_MEMORYTYPE_DEVICE &&
      !lupine_deviceptrs_share_route(copy.dstDevice, copy.srcDevice)) {
    // The two ends live on different servers, so each run makes the trip
    // through the client.
    for (size_t slice = 0; slice < slices; ++slice) {
      for (size_t row = 0; row < rows; ++row) {
        CUresult result = lupine_cuMemcpyDtoD_via_client(
            copy.dstDevice + (copy.dstZ + slice) * dst_slice_pitch +
                (copy.dstY + row) * copy.dstPitch + copy.dstXInBytes,
            copy.srcDevice + (copy.srcZ + slice) * src_slice_pitch +
                (copy.srcY + row) * copy.srcPitch + copy.srcXInBytes,
            run, stream, !blocking);
        if (result != CUDA_SUCCESS) {
          return result;
        }
      }
    }
    return CUDA_SUCCESS;
  }

  lupine_route route = copy.srcMemoryType != CU_MEMORYTYPE_ARRAY
                           ? lupine_route_for_deviceptr(copy.srcDevice)
                           : (copy.dstMemoryType != CU_MEMORYTYPE_ARRAY
                                  ? lupine_route_for_deviceptr(copy.dstDevice)
                                  : lupine_route_for_stream(stream));
  if (lupine_route_is_local(route)) {
    // The driver's own synchronous copy, not an asynchronous one plus a wait:
    // synchronizing the stream would also wait on work this copy has nothing
    // to do with.
    return blocking
               ? lupine_call_real_cuda_fn("cuMemcpy3D_v2", &copy)
               : lupine_call_real_cuda_fn("cuMemcpy3DAsync_v2", &copy, stream);
  }

  // Nothing is staged, so the answer is just the result, and it is always read:
  // the server writes one whether or not the caller is waiting.
  conn_t *conn = lupine_route_remote_conn(route);
  const uint8_t wire_flags = blocking ? LUPINE_MEMCPY_BLOCKING : 0;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, LUPINE_RPC_lupineMemcpy) < 0 ||
      rpc_write(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

// Turn an eager synchronous DtoH into a server-local copy when the destination
// is managed. The existing mapped-allocation fault path will migrate the result
// to the client when it is observed.
static bool lupine_prepare_lazy_managed_destination(CUDA_MEMCPY3D &copy,
                                                    CUstream stream,
                                                    bool blocking,
                                                    bool host_source) {
  if (!blocking || host_source || copy.dstMemoryType != CU_MEMORYTYPE_HOST) {
    return false;
  }

  const size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  const size_t destination_offset = copy.dstZ * dst_slice_pitch +
                                    copy.dstY * copy.dstPitch +
                                    copy.dstXInBytes;
  const size_t destination_span = (copy.Depth - 1) * dst_slice_pitch +
                                  (copy.Height - 1) * copy.dstPitch +
                                  copy.WidthInBytes;
  auto *destination =
      static_cast<unsigned char *>(copy.dstHost) + destination_offset;
  CUdeviceptr translated = 0;
  bool managed = false;
  lupine_route route = copy.srcMemoryType == CU_MEMORYTYPE_ARRAY
                           ? lupine_route_for_stream(stream)
                           : lupine_route_for_deviceptr(copy.srcDevice);
  if (!lupine_translate_client_host_range_to_server(
          destination, destination_span, lupine_route_identity(route),
          &translated, &managed) ||
      !managed) {
    return false;
  }

  copy.dstMemoryType = CU_MEMORYTYPE_UNIFIED;
  copy.dstDevice = translated;
  copy.dstXInBytes = 0;
  copy.dstY = 0;
  copy.dstZ = 0;
  return true;
}

extern "C" CUresult lupine_memcpy(const CUDA_MEMCPY3D *request, CUstream stream,
                                  bool blocking) {
  CUDA_MEMCPY3D copy = *request;
  // A zero-width run carries no bytes. Missing row and slice dimensions are
  // represented internally as one so every path can use the same explicit 3D
  // geometry without supporting zero-sized rows or slices.
  if (copy.WidthInBytes == 0) {
    return CUDA_SUCCESS;
  }
  copy.Height = std::max<size_t>(copy.Height, 1);
  copy.Depth = std::max<size_t>(copy.Depth, 1);
  // A UNIFIED address is client-space: a managed alias is already the server's
  // address for that allocation and a plain device pointer is server-valid, but
  // an ordinary host pointer's bytes have to travel. An array is device memory
  // the client cannot address at all.
  const bool host_source = copy.srcMemoryType == CU_MEMORYTYPE_HOST ||
                           (copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED &&
                            !lupine_is_managed_host_alias(copy.srcDevice) &&
                            lupine_copy_pointer_is_host(copy.srcDevice));

  const bool lazy_managed_destination =
      lupine_prepare_lazy_managed_destination(copy, stream, blocking,
                                              host_source);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  CUresult result;
  if (host_source) {
    result = host_destination ? lupine_memcpy_htoh(copy)
                              : lupine_memcpy_htod(copy, stream, blocking);
  } else {
    result = host_destination ? lupine_memcpy_dtoh(copy, stream, blocking)
                              : lupine_memcpy_dtod(copy, stream, blocking);
  }
  if (result == CUDA_SUCCESS && lazy_managed_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                    size_t ByteCount) {
  // A linear copy is a single run: WidthInBytes carries the whole transfer and
  // Height and Depth are one, so the server's run-boundary rule never bites.
  CUDA_MEMCPY3D copy = {};
  copy.srcPitch = copy.dstPitch = ByteCount;
  copy.srcHeight = copy.dstHeight = 1;
  copy.WidthInBytes = ByteCount;
  copy.Height = 1;
  copy.Depth = 1;
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = srcDevice;
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = dstHost;
  return lupine_memcpy(&copy, CU_STREAM_LEGACY, true);
}

extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                    size_t ByteCount) {
  CUDA_MEMCPY3D copy = {};
  copy.srcPitch = copy.dstPitch = ByteCount;
  copy.srcHeight = copy.dstHeight = 1;
  copy.WidthInBytes = ByteCount;
  copy.Height = 1;
  copy.Depth = 1;
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = srcHost;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = dstDevice;
  return lupine_memcpy(&copy, CU_STREAM_LEGACY, true);
}

#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
extern "C" CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                         const void *srcHost, size_t ByteCount,
                                         CUstream hStream) {
  if (ByteCount != 0 && srcHost == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUDA_MEMCPY3D copy = {};
  copy.srcPitch = copy.dstPitch = ByteCount;
  copy.srcHeight = copy.dstHeight = 1;
  copy.WidthInBytes = ByteCount;
  copy.Height = 1;
  copy.Depth = 1;
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = srcHost;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = dstDevice;
  return lupine_memcpy(&copy, hStream, false);
}

#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

extern "C" CUresult
lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream, bool async) {
  LUPINE_TRACE_LOG("LUPINE cross-route D2D via client dst="
                   << reinterpret_cast<void *>(dstDevice)
                   << " src=" << reinterpret_cast<void *>(srcDevice)
                   << " bytes=" << ByteCount << " async=" << async
                   << " stream=" << hStream);
  if (ByteCount == 0) {
    return CUDA_SUCCESS;
  }

  constexpr size_t chunk_size = 16 * 1024 * 1024;
  std::vector<unsigned char> staging(std::min(ByteCount, chunk_size));
  size_t offset = 0;
  while (offset < ByteCount) {
    size_t chunk = std::min(staging.size(), ByteCount - offset);
    CUresult result =
        cuMemcpyDtoH_v2(staging.data(), srcDevice + offset, chunk);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (async) {
      result = cuMemcpyHtoDAsync_v2(dstDevice + offset, staging.data(), chunk,
                                    hStream);
    } else {
      result = cuMemcpyHtoD_v2(dstDevice + offset, staging.data(), chunk);
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offset += chunk;
  }
  return CUDA_SUCCESS;
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                    size_t srcOffset, size_t ByteCount) {
  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemcpyAtoH_v2", dstHost, srcArray,
                                    srcOffset, ByteCount);
  }
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemcpyAtoH_v2) < 0 ||
      rpc_write(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_write(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  int request_id = rpc_write_end(conn);
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  auto *copy_dst = static_cast<unsigned char *>(dstHost);
  size_t offset = 0;
  do {
    size_t chunk =
        std::min(ByteCount - offset, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
    if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        (return_value == CUDA_SUCCESS && chunk != 0 &&
         rpc_read(conn, copy_dst + offset, chunk) < 0)) {
      rpc_read_end(conn);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    bool final_chunk =
        return_value != CUDA_SUCCESS || offset + chunk == ByteCount;
    if (rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
    offset += chunk;
    if (!final_chunk && rpc_read_start(conn, request_id) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  } while (offset < ByteCount);
  return return_value;
}

#ifdef cuMemcpyAtoH
#undef cuMemcpyAtoH
#endif
extern "C" CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount, CUstream hStream) {
  // "API synchronization behavior" lets the driver make this copy synchronous
  // when the destination is pageable, and every driver does. Callers rely on it
  // instead of synchronizing: cuSPARSE reads its scalar results (nnz counts,
  // SpVV dot products) that way, and a deferred copy either never lands or
  // lands on a stack frame the caller has already left. The bytes come back
  // through the same bounded, chunked path the synchronous copy uses, issued on
  // this stream so they stay ordered behind its prior work. Capture is the
  // exception: there the copy only becomes a graph node and must not block.
  if (ByteCount != 0 && !lupine_host_ptr_is_page_locked(dstHost) &&
      lupine_active_stream_captures.load(std::memory_order_relaxed) == 0) {
    CUDA_MEMCPY3D copy = {};
    copy.srcPitch = copy.dstPitch = ByteCount;
    copy.srcHeight = copy.dstHeight = 1;
    copy.WidthInBytes = ByteCount;
    copy.Height = 1;
    copy.Depth = 1;
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = srcDevice;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = dstHost;
    return lupine_memcpy(&copy, hStream, true);
  }

  conn_t *conn = lupine_rpc_conn_for_deviceptr(srcDevice);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoHAsync_v2) < 0 ||
      rpc_write(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (ByteCount != 0) {
    lupine_event_stale_semaphore_add(1);
  }
  return CUDA_SUCCESS;
}

#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // A 2D copy is a 3D copy of a single slice. srcHeight and dstHeight carry
  // the rows the offset reaches past, which is what the driver validates
  // against.
  CUDA_MEMCPY3D copy = {};
  copy.srcXInBytes = pCopy->srcXInBytes;
  copy.srcY = pCopy->srcY;
  copy.srcMemoryType = pCopy->srcMemoryType;
  copy.srcHost = pCopy->srcHost;
  copy.srcDevice = pCopy->srcDevice;
  copy.srcArray = pCopy->srcArray;
  copy.srcPitch = pCopy->srcPitch;
  copy.srcHeight = pCopy->srcY + pCopy->Height;
  copy.dstXInBytes = pCopy->dstXInBytes;
  copy.dstY = pCopy->dstY;
  copy.dstMemoryType = pCopy->dstMemoryType;
  copy.dstHost = pCopy->dstHost;
  copy.dstDevice = pCopy->dstDevice;
  copy.dstArray = pCopy->dstArray;
  copy.dstPitch = pCopy->dstPitch;
  copy.dstHeight = pCopy->dstY + pCopy->Height;
  copy.WidthInBytes = pCopy->WidthInBytes;
  copy.Height = pCopy->Height;
  copy.Depth = 1;
  // The pitched entry points invalidate every mapped host allocation once the
  // device side has changed, which the linear ones have never done.
  CUresult result = lupine_memcpy(&copy, CU_STREAM_LEGACY, true);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  if (result == CUDA_SUCCESS && !host_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
extern "C" CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2D_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // A 2D copy is a 3D copy of a single slice. srcHeight and dstHeight carry
  // the rows the offset reaches past, which is what the driver validates
  // against.
  CUDA_MEMCPY3D copy = {};
  copy.srcXInBytes = pCopy->srcXInBytes;
  copy.srcY = pCopy->srcY;
  copy.srcMemoryType = pCopy->srcMemoryType;
  copy.srcHost = pCopy->srcHost;
  copy.srcDevice = pCopy->srcDevice;
  copy.srcArray = pCopy->srcArray;
  copy.srcPitch = pCopy->srcPitch;
  copy.srcHeight = pCopy->srcY + pCopy->Height;
  copy.dstXInBytes = pCopy->dstXInBytes;
  copy.dstY = pCopy->dstY;
  copy.dstMemoryType = pCopy->dstMemoryType;
  copy.dstHost = pCopy->dstHost;
  copy.dstDevice = pCopy->dstDevice;
  copy.dstArray = pCopy->dstArray;
  copy.dstPitch = pCopy->dstPitch;
  copy.dstHeight = pCopy->dstY + pCopy->Height;
  copy.WidthInBytes = pCopy->WidthInBytes;
  copy.Height = pCopy->Height;
  copy.Depth = 1;
  // The pitched entry points invalidate every mapped host allocation once the
  // device side has changed, which the linear ones have never done.
  CUresult result = lupine_memcpy(&copy, CU_STREAM_LEGACY, true);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  if (result == CUDA_SUCCESS && !host_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

#ifdef cuMemcpy2DUnaligned
#undef cuMemcpy2DUnaligned
#endif
extern "C" CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2DUnaligned_v2(pCopy);
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                       CUstream hStream) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // A 2D copy is a 3D copy of a single slice. srcHeight and dstHeight carry
  // the rows the offset reaches past, which is what the driver validates
  // against.
  CUDA_MEMCPY3D copy = {};
  copy.srcXInBytes = pCopy->srcXInBytes;
  copy.srcY = pCopy->srcY;
  copy.srcMemoryType = pCopy->srcMemoryType;
  copy.srcHost = pCopy->srcHost;
  copy.srcDevice = pCopy->srcDevice;
  copy.srcArray = pCopy->srcArray;
  copy.srcPitch = pCopy->srcPitch;
  copy.srcHeight = pCopy->srcY + pCopy->Height;
  copy.dstXInBytes = pCopy->dstXInBytes;
  copy.dstY = pCopy->dstY;
  copy.dstMemoryType = pCopy->dstMemoryType;
  copy.dstHost = pCopy->dstHost;
  copy.dstDevice = pCopy->dstDevice;
  copy.dstArray = pCopy->dstArray;
  copy.dstPitch = pCopy->dstPitch;
  copy.dstHeight = pCopy->dstY + pCopy->Height;
  copy.WidthInBytes = pCopy->WidthInBytes;
  copy.Height = pCopy->Height;
  copy.Depth = 1;
  // The pitched entry points invalidate every mapped host allocation once the
  // device side has changed, which the linear ones have never done.
  CUresult result = lupine_memcpy(&copy, hStream, false);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  if (result == CUDA_SUCCESS && !host_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

#ifdef cuMemcpy2DAsync
#undef cuMemcpy2DAsync
#endif
extern "C" CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy2DAsync_ptsz
#undef cuMemcpy2DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy2DAsync_ptsz(const CUDA_MEMCPY2D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUDA_MEMCPY3D copy = *pCopy;
  // The pitched entry points invalidate every mapped host allocation once the
  // device side has changed, which the linear ones have never done.
  CUresult result = lupine_memcpy(&copy, CU_STREAM_LEGACY, true);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  if (result == CUDA_SUCCESS && !host_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                       CUstream hStream) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUDA_MEMCPY3D copy = *pCopy;
  // The pitched entry points invalidate every mapped host allocation once the
  // device side has changed, which the linear ones have never done.
  CUresult result = lupine_memcpy(&copy, hStream, false);
  const bool host_destination =
      copy.dstMemoryType == CU_MEMORYTYPE_HOST ||
      (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
       !lupine_is_managed_host_alias(copy.dstDevice) &&
       lupine_copy_pointer_is_host(copy.dstDevice));
  if (result == CUDA_SUCCESS && !host_destination) {
    result = lupine_sync_mapped_device_to_host();
  }
  return result;
}

extern "C" CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Both context handles must live on one connection; the copy then behaves
  // like an ordinary one there. Peers on different connections are not
  // supported.
  lupine_route src_route = lupine_route_for_context(pCopy->srcContext);
  lupine_route dst_route = lupine_route_for_context(pCopy->dstContext);
  if (lupine_route_identity(src_route) != lupine_route_identity(dst_route) ||
      lupine_route_is_local(dst_route)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  conn_t *conn = lupine_route_remote_conn(dst_route);
  CUDA_MEMCPY3D_PEER copy = *pCopy;
  // UNIFIED addresses are client-space: a managed alias is already the
  // server's address for that allocation, a plain host pointer's bytes must
  // travel, and a device pointer is already server-valid.
  if (copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED) {
    if (!lupine_is_managed_host_alias(copy.srcDevice) &&
        lupine_copy_pointer_is_host(copy.srcDevice)) {
      copy.srcHost = (const void *)copy.srcDevice;
      copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    }
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED) {
    if (!lupine_is_managed_host_alias(copy.dstDevice) &&
        lupine_copy_pointer_is_host(copy.dstDevice)) {
      copy.dstHost = (void *)copy.dstDevice;
      copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    }
  }
  bool src_host = copy.srcMemoryType == CU_MEMORYTYPE_HOST;
  bool dst_host = copy.dstMemoryType == CU_MEMORYTYPE_HOST;
  lupine_copy_direction direction =
      src_host ? (dst_host ? lupine_copy_direction::host_to_host
                           : lupine_copy_direction::host_to_device)
               : (dst_host ? lupine_copy_direction::device_to_host
                           : lupine_copy_direction::device_to_device);
  size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  const char *src_base = (const char *)copy.srcHost +
                         copy.srcZ * src_slice_pitch +
                         copy.srcY * copy.srcPitch + copy.srcXInBytes;
  char *dst_base = (char *)copy.dstHost + copy.dstZ * dst_slice_pitch +
                   copy.dstY * copy.dstPitch + copy.dstXInBytes;
  CUresult return_value;
  switch (direction) {
  case lupine_copy_direction::host_to_host:
    for (size_t z = 0; z < copy.Depth; ++z) {
      for (size_t row = 0; row < copy.Height; ++row) {
        memmove(dst_base + z * dst_slice_pitch + row * copy.dstPitch,
                src_base + z * src_slice_pitch + row * copy.srcPitch,
                copy.WidthInBytes);
      }
    }
    return CUDA_SUCCESS;
  case lupine_copy_direction::host_to_device: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeer) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_write_pitched_payload(conn, src_base, copy.WidthInBytes,
                                  copy.Height, copy.srcPitch, copy.Depth,
                                  src_slice_pitch) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      return_value = lupine_sync_mapped_device_to_host();
    }
    return return_value;
  }
  case lupine_copy_direction::device_to_host: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeer) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        (return_value == CUDA_SUCCESS &&
         rpc_read_pitched_payload(conn, dst_base, copy.WidthInBytes,
                                  copy.Height, copy.dstPitch, copy.Depth,
                                  dst_slice_pitch) < 0) ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return return_value;
  }
  case lupine_copy_direction::device_to_device: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeer) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      return_value = lupine_sync_mapped_device_to_host();
    }
    return return_value;
  }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                        CUstream hStream) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Both context handles must live on one connection; the copy then behaves
  // like an ordinary one there. Peers on different connections are not
  // supported.
  lupine_route src_route = lupine_route_for_context(pCopy->srcContext);
  lupine_route dst_route = lupine_route_for_context(pCopy->dstContext);
  if (lupine_route_identity(src_route) != lupine_route_identity(dst_route) ||
      lupine_route_is_local(dst_route)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  conn_t *conn = lupine_route_remote_conn(dst_route);
  CUDA_MEMCPY3D_PEER copy = *pCopy;
  // UNIFIED addresses are client-space: a managed alias is already the
  // server's address for that allocation, a plain host pointer's bytes must
  // travel, and a device pointer is already server-valid.
  if (copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED) {
    if (!lupine_is_managed_host_alias(copy.srcDevice) &&
        lupine_copy_pointer_is_host(copy.srcDevice)) {
      copy.srcHost = (const void *)copy.srcDevice;
      copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    }
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED) {
    if (!lupine_is_managed_host_alias(copy.dstDevice) &&
        lupine_copy_pointer_is_host(copy.dstDevice)) {
      copy.dstHost = (void *)copy.dstDevice;
      copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    }
  }
  bool src_host = copy.srcMemoryType == CU_MEMORYTYPE_HOST;
  bool dst_host = copy.dstMemoryType == CU_MEMORYTYPE_HOST;
  lupine_copy_direction direction =
      src_host ? (dst_host ? lupine_copy_direction::host_to_host
                           : lupine_copy_direction::host_to_device)
               : (dst_host ? lupine_copy_direction::device_to_host
                           : lupine_copy_direction::device_to_device);
  size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  const char *src_base = (const char *)copy.srcHost +
                         copy.srcZ * src_slice_pitch +
                         copy.srcY * copy.srcPitch + copy.srcXInBytes;
  char *dst_base = (char *)copy.dstHost + copy.dstZ * dst_slice_pitch +
                   copy.dstY * copy.dstPitch + copy.dstXInBytes;
  CUresult return_value;
  switch (direction) {
  case lupine_copy_direction::host_to_host:
    for (size_t z = 0; z < copy.Depth; ++z) {
      for (size_t row = 0; row < copy.Height; ++row) {
        memmove(dst_base + z * dst_slice_pitch + row * copy.dstPitch,
                src_base + z * src_slice_pitch + row * copy.srcPitch,
                copy.WidthInBytes);
      }
    }
    return CUDA_SUCCESS;
  case lupine_copy_direction::host_to_device: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeerAsync) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_write_pitched_payload(conn, src_base, copy.WidthInBytes,
                                  copy.Height, copy.srcPitch, copy.Depth,
                                  src_slice_pitch) < 0 ||
        rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      return_value = lupine_sync_mapped_device_to_host();
    }
    return return_value;
  }
  case lupine_copy_direction::device_to_host: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeerAsync) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        (return_value == CUDA_SUCCESS &&
         rpc_read_pitched_payload(conn, dst_base, copy.WidthInBytes,
                                  copy.Height, copy.dstPitch, copy.Depth,
                                  dst_slice_pitch) < 0) ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return return_value;
  }
  case lupine_copy_direction::device_to_device: {
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request(conn, RPC_cuMemcpy3DPeerAsync) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value == CUDA_SUCCESS) {
      return_value = lupine_sync_mapped_device_to_host();
    }
    return return_value;
  }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

#ifdef cuMemcpy3D
#undef cuMemcpy3D
#endif
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return cuMemcpy3D_v2(pCopy);
}

#ifdef cuMemcpy3DAsync
#undef cuMemcpy3DAsync
#endif
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy3DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy3DAsync_ptsz
#undef cuMemcpy3DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy3DAsync_ptsz(const CUDA_MEMCPY3D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy3DAsync_v2(pCopy, hStream);
}
