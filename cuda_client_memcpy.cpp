#include "lupine_platform.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
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

#include "cache.h"
#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"
#include "cuda_client_memcpy.h"
#include "events.h"
#include "managed_advice.h"

// Defined with the capture wrappers in cuda_client.cpp; nonzero only while this
// client holds a stream capture open, where a pageable copy stays a graph node.
extern std::atomic<int> lupine_active_stream_captures;
#include "lupine_attr_sizes.h"
#include "lupine_log.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

static void lupine_pointer_attribute_cache_clear();
#include "rpc.h"
#include "transport.h"

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

// CUDA stream handles are process-local to their owning driver. A copy whose
// data lives on another route must fall back to the synchronous staging path.
static bool lupine_device_copy_uses_remote_callback(CUdeviceptr destination,
                                                    CUdeviceptr source);
static CUresult lupine_bulk_push(conn_t *conn, lupine_bulk_lanes *lanes,
                                 uint64_t destination, const void *source,
                                 size_t bytes, bool to_host);
static constexpr size_t LUPINE_BULK_COPY_MIN_BYTES = 8 * 1024 * 1024;

static bool lupine_stream_crosses_route(CUstream stream, lupine_route route) {
  if (stream == nullptr) {
    return false;
  }
  lupine_route stream_route = lupine_route_for_known_stream(stream);
  return stream_route.kind != LUPINE_ROUTE_INVALID &&
         !lupine_routes_share_server(route, stream_route);
}

static lupine_route lupine_route_for_stream_or_default(CUstream stream) {
  return stream != nullptr ? lupine_route_for_stream(stream)
                           : lupine_route_for_default();
}

static inline CUstream lupine_stream_fetch_ready() {
  return reinterpret_cast<CUstream>(LUPINE_STREAM_FETCH_READY_BITS);
}

static inline CUstream lupine_stream_fetch_sync_all() {
  return reinterpret_cast<CUstream>(LUPINE_STREAM_FETCH_SYNC_ALL_BITS);
}

struct lupine_host_allocation {
  size_t size = 0;
  size_t storage_size = 0;
  // Exact caller range; owned allocations may have page-rounded storage.
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
  bool managed_global = false;
  lupine_managed_advice advice;
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
  // Captured with it; the fetch binds it on whichever lane ends up faulting.
  CUcontext stale_fetch_context = nullptr;
  // Captured with it; a launch-time invalidation fetches behind the launch.
  CUstream stale_fetch_stream = lupine_stream_fetch_ready();
  // The requests the fetch must follow: the launch's stream, every request
  // when launches on different streams invalidated it, and none after a
  // synchronize, which already waited for the writes.
  uintptr_t stale_fetch_scope = RPC_SCOPE_LANE;
  pid_t stale_fetch_ptds_tid = 0;
  bool stale_rearm_pending = false;
  conn_t *stale_rearm_conn = nullptr;
  CUcontext stale_rearm_context = nullptr;
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
  std::vector<int> portable_routes;
  std::vector<int> dtoh_routes;
};

struct lupine_mapped_host_snapshot {
  void *host = nullptr;
  size_t data_offset = 0;
  size_t data_bytes = 0;
  CUdeviceptr device_ptr = 0;
  CUdeviceptr server_host_ptr = 0;
  bool managed = false;
  bool device_pointer_exposed = false;
};

using lupine_host_allocation_map =
    std::map<void *, lupine_host_allocation, std::less<void *>>;

template <typename T, typename Tag> static T &lupine_leaked(Tag) {
  static T *instance = new T();
  return *instance;
}

// The registry below has process lifetime; its locks must remain usable when
// CUDA library finalizers query or release allocations during dylib teardown.
static std::mutex &lupine_host_allocation_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

// Bumped under the host allocation mutex by every table insertion.
static uint64_t lupine_host_allocation_generation;
static std::atomic<size_t> lupine_tracked_host_count{0};
static std::atomic<uintptr_t> lupine_tracked_host_min{UINTPTR_MAX};
static std::atomic<uintptr_t> lupine_tracked_host_max{0};
static std::atomic<size_t> lupine_exposed_or_managed_host_count{0};

static std::mutex &lupine_portable_registration_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

// CUDA libraries can issue teardown requests after function-local statics have
// started finalizing. Keep the registry alive for those late request flushes.
static lupine_host_allocation_map &lupine_mutable_host_allocations_locked() {
  return lupine_leaked<lupine_host_allocation_map>([] {});
}

// The device-side bases (device_ptr, server_host_ptr) of every allocation
// whose base differs from its host key, so a kernel argument resolves in
// O(log N) whichever address space it names.
static std::map<CUdeviceptr, void *> &lupine_host_allocation_bases_locked() {
  return lupine_leaked<std::map<CUdeviceptr, void *>>([] {});
}

static std::map<uintptr_t, void *> &lupine_host_allocation_aliases_locked() {
  return lupine_leaked<std::map<uintptr_t, void *>>([] {});
}

static inline uintptr_t lupine_clamped_end(uintptr_t base, size_t span) {
  span = std::max<size_t>(span, 1);
  return span > UINTPTR_MAX - base ? UINTPTR_MAX : base + span;
}

static void lupine_expand_tracked_host_bounds_locked(uintptr_t base,
                                                     size_t span) {
  uintptr_t end = lupine_clamped_end(base, span);
  if (base < lupine_tracked_host_min.load(std::memory_order_relaxed)) {
    lupine_tracked_host_min.store(base, std::memory_order_release);
  }
  if (end > lupine_tracked_host_max.load(std::memory_order_relaxed)) {
    lupine_tracked_host_max.store(end, std::memory_order_release);
  }
}

static void lupine_note_host_allocation_inserted_locked(
    lupine_host_allocation_map::const_iterator it) {
  lupine_expand_tracked_host_bounds_locked(
      reinterpret_cast<uintptr_t>(it->first),
      std::max(it->second.size, it->second.storage_size));
  for (CUdeviceptr dev_base :
       {it->second.device_ptr, it->second.server_host_ptr}) {
    if (dev_base != 0) {
      lupine_expand_tracked_host_bounds_locked(static_cast<uintptr_t>(dev_base),
                                               it->second.size);
    }
  }
  if (it->second.managed || it->second.device_pointer_exposed) {
    lupine_exposed_or_managed_host_count.fetch_add(1,
                                                   std::memory_order_release);
  }
  lupine_tracked_host_count.fetch_add(1, std::memory_order_release);
}

static void lupine_recompute_host_allocation_summary_locked() {
  const auto &allocations = lupine_mutable_host_allocations_locked();
  const auto &bases = lupine_host_allocation_bases_locked();
  uintptr_t min_addr = UINTPTR_MAX;
  uintptr_t max_addr = 0;
  size_t exposed_or_managed = 0;
  auto include_range = [&](uintptr_t base, size_t span) {
    min_addr = std::min(min_addr, base);
    max_addr = std::max(max_addr, lupine_clamped_end(base, span));
  };
  for (const auto &entry : allocations) {
    include_range(reinterpret_cast<uintptr_t>(entry.first),
                  std::max(entry.second.size, entry.second.storage_size));
    for (CUdeviceptr dev_base :
         {entry.second.device_ptr, entry.second.server_host_ptr}) {
      if (dev_base != 0) {
        include_range(static_cast<uintptr_t>(dev_base), entry.second.size);
      }
    }
    exposed_or_managed +=
        (entry.second.managed || entry.second.device_pointer_exposed) ? 1 : 0;
  }
  for (const auto &base_entry : bases) {
    auto it = allocations.find(base_entry.second);
    include_range(static_cast<uintptr_t>(base_entry.first),
                  it != allocations.end() ? it->second.size : size_t{1});
  }
  lupine_tracked_host_min.store(min_addr, std::memory_order_release);
  lupine_tracked_host_max.store(max_addr, std::memory_order_release);
  lupine_exposed_or_managed_host_count.store(exposed_or_managed,
                                             std::memory_order_release);
  lupine_tracked_host_count.store(allocations.size() + bases.size(),
                                  std::memory_order_release);
}

bool lupine_tracked_host_bounds(uintptr_t *min_addr, uintptr_t *max_addr) {
  if (lupine_tracked_host_count.load(std::memory_order_acquire) == 0) {
    return false;
  }
  uintptr_t lo = lupine_tracked_host_min.load(std::memory_order_acquire);
  uintptr_t hi = lupine_tracked_host_max.load(std::memory_order_acquire);
  if (min_addr != nullptr) {
    *min_addr = lo;
  }
  if (max_addr != nullptr) {
    *max_addr = hi;
  }
  return lo < hi;
}

static inline bool lupine_addr_in_tracked_host_bounds(uintptr_t addr) {
  uintptr_t lo = 0;
  uintptr_t hi = 0;
  return lupine_tracked_host_bounds(&lo, &hi) && addr >= lo && addr < hi;
}

static inline bool lupine_range_in_tracked_host_bounds(uintptr_t start,
                                                       size_t size) {
  uintptr_t lo = 0;
  uintptr_t hi = 0;
  return size != 0 && start <= UINTPTR_MAX - size &&
         lupine_tracked_host_bounds(&lo, &hi) && start < hi &&
         start + size > lo;
}

static void
lupine_index_host_allocation_locked(lupine_host_allocation_map::iterator it,
                                    bool add) {
  for (CUdeviceptr base : {it->second.device_ptr, it->second.server_host_ptr}) {
    if (base == 0 || base == reinterpret_cast<CUdeviceptr>(it->first)) {
      continue;
    }
    if (add) {
      lupine_host_allocation_bases_locked()[base] = it->first;
      lupine_expand_tracked_host_bounds_locked(
          static_cast<uintptr_t>(base), std::max<size_t>(it->second.size, 1));
    } else {
      lupine_host_allocation_bases_locked().erase(base);
    }
  }
  uintptr_t alias = reinterpret_cast<uintptr_t>(it->second.io_alias);
  if (alias != 0) {
    if (add) {
      lupine_host_allocation_aliases_locked()[alias] = it->first;
    } else {
      lupine_host_allocation_aliases_locked().erase(alias);
    }
  }
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p);
static CUresult
lupine_adopt_host_allocation(conn_t *conn, void **host, void *remote_host,
                             CUdeviceptr device_ptr, size_t bytesize,
                             unsigned int flags, bool managed = false);
static CUresult
lupine_remote_cuMemHostAlloc(void **remote_host, CUdeviceptr *device_ptr,
                             size_t bytesize, unsigned int flags,
                             lupine_route route, void *requested = nullptr);
extern "C" int lupine_forward_remote_stdout(conn_t *conn);
static CUresult lupine_remote_cuMemFreeHost(void *remote_host,
                                            lupine_route route,
                                            bool drain_only = false);

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
// Dirty publications receive globally ordered tickets, and every RPC waits for
// the latest one. A host write is ordered before another thread's CUDA call by
// whatever synchronization the application uses between those threads, which
// the client cannot observe, so the publishing thread's own calls are not the
// only ones that must see it. The acknowledged global watermark lets a caller
// skip the flush when another thread already covered its ticket.
static uint64_t lupine_dirty_host_epoch = 0;
static volatile sig_atomic_t lupine_device_work_pending = 0;
thread_local bool lupine_in_host_callback = false;

int lupine_settle_host_flushes(conn_t *conn) {
  if (!__atomic_exchange_n(&conn->host_flushes_unsettled, 0,
                           __ATOMIC_ACQ_REL)) {
    return 0;
  }
  // Flushes go in order on their own lane, so an empty one is answered only
  // once every flush sent before it is applied.
  uint32_t count = 0;
  return rpc_write_start_request_on(conn, LUPINE_RPC_lupineManagedHostFlush,
                                    RPC_SCOPE_HOST_WRITES) < 0 ||
                 rpc_write(conn, &count, sizeof(count)) < 0 ||
                 rpc_wait_for_response(conn) < 0 || rpc_read_end(conn) < 0
             ? -1
             : 0;
}
static std::atomic<uint64_t> lupine_flushed_host_epoch{0};
static std::mutex lupine_host_flush_mutex;

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
#if defined(__APPLE__)
static struct sigaction lupine_previous_sigbus_action;
#endif
static bool lupine_sigsegv_handler_installed = false;

// RPC responses can land in either ordinary pinned I/O buffers or protected
// mapped memory. Consult the signal-safe registry without taking an allocation
// lock: a demand fetch can itself read an RPC response from a fault handler.
bool lupine_host_range_is_protected(uintptr_t start, size_t size) {
  sig_atomic_t count =
      __atomic_load_n(&lupine_fault_entry_high_water, __ATOMIC_ACQUIRE);
  for (sig_atomic_t i = 0; i < count; ++i) {
    const auto &entry = lupine_fault_entries[i];
    if (__atomic_load_n(&entry.allocation, __ATOMIC_ACQUIRE) == nullptr) {
      continue;
    }
    uintptr_t base = __atomic_load_n(&entry.base, __ATOMIC_RELAXED);
    uintptr_t end = __atomic_load_n(&entry.end, __ATOMIC_RELAXED);
    if (start >= base && start < end && size <= end - start) {
      return true;
    }
  }
  return false;
}

static size_t lupine_page_size() {
  static const size_t page_size = []() {
    long configured = sysconf(_SC_PAGESIZE);
    return configured > 0 ? static_cast<size_t>(configured) : 4096;
  }();
  return page_size;
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
#if defined(__APPLE__)
  // Without an identity arena, CUDA chooses the server address. macOS uses
  // independently mapped local views and the allocation registry translates
  // their pointers, so the server address need not lie in the legacy bands.
  return server != 0 && size <= UINTPTR_MAX - server;
#else
  if (server < LUPINE_HOST_ALLOCATION_SERVER_BASE ||
      size > LUPINE_HOST_ALLOCATION_WINDOW_SIZE ||
      server - LUPINE_HOST_ALLOCATION_SERVER_BASE >
          LUPINE_HOST_ALLOCATION_WINDOW_SIZE - size) {
    return false;
  }
  return true;
#endif
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
#if defined(__APPLE__)
  munmap(address, size);
#else
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
#ifdef MAP_NORESERVE
  flags |= MAP_NORESERVE;
#endif
  mmap(address, size, PROT_NONE, flags, -1, 0);
#endif
}
#endif

static bool lupine_create_shared_views(conn_t *conn, uintptr_t server,
                                       size_t size, void **host_view,
                                       void **io_alias) {
  *host_view = nullptr;
  *io_alias = nullptr;
  if (!lupine_server_range(conn, server, size)) {
    LUPINE_TRACE_LOG("LUPINE host views rejected server="
                     << reinterpret_cast<void *>(server) << " size=" << size
                     << " arena=" << (conn ? conn->va_size : 0));
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
#if !defined(__APPLE__)
  if (!identity && !lupine_reserve_host_allocation_windows()) {
    LUPINE_TRACE_LOG("LUPINE host views reservation failed errno=" << errno);
    return false;
  }
#endif
  int fd = lupine_create_shared_memory_fd();
  if (fd < 0 || ftruncate(fd, static_cast<off_t>(size)) != 0) {
    LUPINE_TRACE_LOG(
        "LUPINE host views storage failed fd=" << fd << " errno=" << errno);
    if (fd >= 0) {
      close(fd);
    }
    return false;
  }
#if defined(__APPLE__)
  void *host_hint = nullptr;
  void *alias_hint = nullptr;
  int shared_flags = MAP_SHARED;
#else
  void *host_hint = reinterpret_cast<void *>(host_address);
  void *alias_hint = reinterpret_cast<void *>(alias_address);
  int shared_flags = MAP_SHARED | MAP_FIXED;
#endif
  void *host =
      mmap(host_hint, size, PROT_READ | PROT_WRITE, shared_flags, fd, 0);
  void *alias =
      host == MAP_FAILED
          ? MAP_FAILED
          : mmap(alias_hint, size, PROT_READ | PROT_WRITE, shared_flags, fd, 0);
  close(fd);
  if (host == MAP_FAILED || alias == MAP_FAILED) {
    LUPINE_TRACE_LOG("LUPINE host views mapping failed host="
                     << host << " alias=" << alias << " errno=" << errno
                     << " server=" << reinterpret_cast<void *>(server)
                     << " size=" << size);
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
#if defined(__APPLE__)
  const auto &previous = sig == SIGBUS ? lupine_previous_sigbus_action
                                       : lupine_previous_sigsegv_action;
#else
  const auto &previous = lupine_previous_sigsegv_action;
#endif
  if (previous.sa_flags & SA_SIGINFO) {
    if (previous.sa_sigaction != nullptr) {
      previous.sa_sigaction(sig, info, uctx);
      return;
    }
  } else if (previous.sa_handler == SIG_IGN) {
    return;
  } else if (previous.sa_handler != SIG_DFL && previous.sa_handler != nullptr) {
    previous.sa_handler(sig);
    return;
  }

  sigaction(sig, &previous, nullptr);
  raise(sig);
}

static bool lupine_demand_fetch_run(lupine_host_allocation *allocation,
                                    size_t fault_offset, size_t *out_start,
                                    size_t *out_end);
static bool lupine_finish_demand_fetch(lupine_host_allocation *allocation,
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

static void lupine_require_dirty_host_flush() {
  __atomic_add_fetch(&lupine_dirty_host_epoch, 1, __ATOMIC_ACQ_REL);
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
    lupine_require_dirty_host_flush();
    return;
  }

  __atomic_sub_fetch(&allocation->pending_dirty_ranges, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&allocation->full_dirty, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&queue.full_dirty_pending, 1, __ATOMIC_RELEASE);
  lupine_require_dirty_host_flush();
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
      } else if (lupine_in_host_callback) {
        // Another client thread is parked mid-fetch behind this host callback.
        // Fetch the needed span directly on the non-blocking ready stream
        // without taking over device_stale from the parked caller.
        size_t span_start = 0;
        size_t span_end = 0;
        if (!lupine_demand_fetch_run(allocation, fault_offset, &span_start,
                                     &span_end)) {
          __atomic_sub_fetch(&lupine_active_fault_handlers, 1,
                             __ATOMIC_RELEASE);
          lupine_call_previous_sigsegv(sig, info, uctx);
          return;
        }
        if (span_end > span_start) {
          mprotect(reinterpret_cast<void *>(allocation->host_base + span_start),
                   span_end - span_start, PROT_READ);
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

    // Writable first, then published. The faulting store runs after this
    // returns, so a flush that takes the page has to re-protect it, and the
    // store faults again; one that runs before it is published never touches
    // the page, and a later flush sends the store. Published first, a flush in
    // between would send the old bytes and the page would turn writable after.
    mprotect(reinterpret_cast<void *>(page), page_size, PROT_READ | PROT_WRITE);
    lupine_queue_dirty_host_range(allocation, page, page + page_size);
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
#if defined(__APPLE__)
    // Darwin reports writes to protected shared mappings as SIGBUS.
    if (sigaction(SIGBUS, &action, &lupine_previous_sigbus_action) != 0) {
      sigaction(SIGSEGV, &lupine_previous_sigsegv_action, nullptr);
      return;
    }
#endif
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
    __atomic_store_n(&lupine_fault_entries[index].base,
                     reinterpret_cast<uintptr_t>(base), __ATOMIC_RELAXED);
    __atomic_store_n(&lupine_fault_entries[index].end,
                     reinterpret_cast<uintptr_t>(base) + size,
                     __ATOMIC_RELAXED);
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

static void
lupine_expose_host_device_pointer(void *host,
                                  lupine_host_allocation &allocation) {
  if (!allocation.device_pointer_exposed) {
    allocation.device_pointer_exposed = true;
    if (!allocation.managed) {
      lupine_exposed_or_managed_host_count.fetch_add(1,
                                                     std::memory_order_release);
    }
    // Ordinary pinned buffers stay writable for read(2)/fread until device
    // code can access them directly. Include writes made before tracking began.
    // If tracking cannot be installed, the untracked mapping uses full flushes
    // and eager readback at synchronization, like a host registration.
    if (allocation.owned && !allocation.managed && !allocation.local_cuda &&
        lupine_enable_dirty_tracking_locked(host, &allocation) &&
        allocation.tracking_enabled) {
      lupine_queue_dirty_host_range(&allocation, allocation.host_base,
                                    allocation.host_base + allocation.size);
    }
  } else if (allocation.tracking_enabled) {
    return;
  }
  lupine_require_dirty_host_flush();
}

static std::vector<lupine_mapped_host_snapshot> lupine_mapped_host_snapshots() {
  std::vector<lupine_mapped_host_snapshot> snapshots;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  // Managed memory can be reached through nested pointers, so synchronize it
  // conservatively. Pinned host memory remains host-authoritative until its
  // device mapping is exposed through a launch or an explicit pointer query.
  for (const auto &entry : lupine_mutable_host_allocations_locked()) {
    if (entry.second.device_ptr != 0 && !entry.second.local_cuda &&
        !entry.second.client_to_server_only && entry.second.user_size != 0 &&
        (entry.second.managed || entry.second.device_pointer_exposed)) {
      snapshots.push_back({entry.first, entry.second.data_offset,
                           entry.second.user_size, entry.second.device_ptr,
                           entry.second.server_host_ptr, entry.second.managed,
                           entry.second.device_pointer_exposed});
    }
  }
  return snapshots;
}

static CUdeviceptr
lupine_host_backing_for_route(const lupine_host_allocation &allocation,
                              int route_id) {
  if (allocation.route_id == route_id) {
    return allocation.server_host_ptr;
  }
  if (std::find(allocation.portable_routes.begin(),
                allocation.portable_routes.end(),
                route_id) != allocation.portable_routes.end()) {
    return allocation.device_ptr;
  }
  return 0;
}

static lupine_host_allocation_map::iterator
lupine_find_mapped_host_pointer_locked(CUdeviceptr pointer, size_t *offset) {
  auto &allocations = lupine_mutable_host_allocations_locked();
  auto upper = allocations.upper_bound(reinterpret_cast<void *>(pointer));
  if (upper != allocations.begin()) {
    auto it = std::prev(upper);
    *offset = pointer - reinterpret_cast<CUdeviceptr>(it->first);
    if (*offset < it->second.size) {
      return it;
    }
  }
  auto &bases = lupine_host_allocation_bases_locked();
  auto base = bases.upper_bound(pointer);
  if (base != bases.begin()) {
    --base;
    auto it = allocations.find(base->second);
    *offset = pointer - base->first;
    if (it != allocations.end() && *offset < it->second.size) {
      return it;
    }
  }
  return allocations.end();
}

static bool lupine_expose_mapped_host_pointer_locked(CUdeviceptr argument,
                                                     bool *portable,
                                                     bool accumulate = false) {
  size_t offset = 0;
  auto it = lupine_find_mapped_host_pointer_locked(argument, &offset);
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return false;
  }
  lupine_expose_host_device_pointer(it->first, it->second);
  if (portable != nullptr) {
    bool is_portable = !it->second.managed &&
                       (it->second.flags & CU_MEMHOSTALLOC_PORTABLE) != 0;
    *portable = accumulate ? (*portable || is_portable) : is_portable;
  }
  return it->second.managed;
}

bool lupine_prepare_mapped_host_pointer(CUdeviceptr argument, bool *portable) {
  if (argument == 0 ||
      !lupine_addr_in_tracked_host_bounds(static_cast<uintptr_t>(argument))) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  return lupine_expose_mapped_host_pointer_locked(argument, portable, false);
}

void lupine_prepare_mapped_host_pointers(const uintptr_t *candidates,
                                         size_t count, bool *portable,
                                         std::vector<CUdeviceptr> *managed) {
  if (candidates == nullptr || count == 0 ||
      lupine_tracked_host_count.load(std::memory_order_acquire) == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  if (lupine_mutable_host_allocations_locked().empty()) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    CUdeviceptr argument = static_cast<CUdeviceptr>(candidates[i]);
    if (argument != 0 &&
        lupine_expose_mapped_host_pointer_locked(argument, portable, true) &&
        managed != nullptr) {
      managed->push_back(argument);
    }
  }
}

// A copy or memset into a host allocation's device address writes it from the
// device, as a kernel given that address would, so it exposes the allocation
// the same way: the next synchronize then refreshes the client's view.
extern "C" void lupine_note_device_write(CUdeviceptr ptr) {
  if (ptr == 0 ||
      !lupine_addr_in_tracked_host_bounds(static_cast<uintptr_t>(ptr))) {
    return;
  }
  bool portable = false;
  lupine_prepare_mapped_host_pointer(ptr, &portable);
}

CUresult lupine_prepare_portable_host_allocations(lupine_route route,
                                                  bool expose_portable) {
  if (lupine_tracked_host_count.load(std::memory_order_acquire) == 0) {
    return CUDA_SUCCESS;
  }
  // Registrations stay until their allocation is freed, so a route that had
  // every portable allocation registered stays complete until an insertion.
  static thread_local std::vector<uint64_t> complete_generation;
  int route_id = lupine_route_identity(route);
  size_t slot = static_cast<size_t>(route_id + 2);
  if (!expose_portable && slot < complete_generation.size() &&
      complete_generation[slot] ==
          __atomic_load_n(&lupine_host_allocation_generation,
                          __ATOMIC_ACQUIRE)) {
    return CUDA_SUCCESS;
  }
  std::lock_guard<std::mutex> mapping_lock(
      lupine_portable_registration_mutex());
  std::unique_lock<std::mutex> lock(lupine_host_allocation_mutex());
  uint64_t generation = lupine_host_allocation_generation;
  // Register the backing ranges, rather than guessing which launch arguments
  // contain pointers. A portable allocation may be reached through another
  // allocation, without ever appearing as a kernel argument itself.
  for (auto &entry : lupine_mutable_host_allocations_locked()) {
    auto &allocation = entry.second;
    if (allocation.managed || allocation.size == 0 ||
        (allocation.flags & CU_MEMHOSTALLOC_PORTABLE) == 0 ||
        __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
      continue;
    }
    // Host-I/O buffers stay writable until device code can reach them. When
    // a launch exposes portable memory, include possible nested allocations.
    if (expose_portable) {
      lupine_expose_host_device_pointer(entry.first, allocation);
    }
    if (lupine_host_backing_for_route(allocation, route_id) != 0) {
      continue;
    }
    void *host = entry.first;
    if (allocation.local_cuda && allocation.device_ptr == 0) {
      lock.unlock();
      CUdeviceptr device_ptr = 0;
      CUresult result = lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2",
                                                 &device_ptr, host, 0);
      lock.lock();
      if (result != CUDA_SUCCESS) {
        return result;
      }
      allocation.device_ptr = device_ptr;
      lupine_expand_tracked_host_bounds_locked(
          static_cast<uintptr_t>(device_ptr),
          std::max<size_t>(allocation.size, 1));
    }
    void *requested = reinterpret_cast<void *>(allocation.device_ptr);
    size_t bytes = allocation.storage_size;
    if (requested == nullptr ||
        (lupine_route_is_local(route) && requested != host)) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    // Allocate the membership entry before creating resources on the peer.
    allocation.portable_routes.reserve(allocation.portable_routes.size() + 1);
    lock.unlock();
    CUdeviceptr device_ptr = 0;
    CUresult result;
    if (lupine_route_is_local(route)) {
      result = lupine_call_real_cuda_fn("cuMemHostRegister_v2", host, bytes,
                                        CU_MEMHOSTREGISTER_PORTABLE |
                                            CU_MEMHOSTREGISTER_DEVICEMAP);
      if (result == CUDA_SUCCESS) {
        result = lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2",
                                          &device_ptr, host, 0);
        if (result == CUDA_SUCCESS && device_ptr != allocation.device_ptr) {
          result = CUDA_ERROR_NOT_SUPPORTED;
        }
        if (result != CUDA_SUCCESS) {
          (void)lupine_call_real_cuda_fn("cuMemHostUnregister", host);
        }
      }
    } else {
      void *remote_host = nullptr;
      result = lupine_remote_cuMemHostAlloc(&remote_host, &device_ptr, bytes,
                                            CU_MEMHOSTALLOC_PORTABLE |
                                                CU_MEMHOSTALLOC_DEVICEMAP,
                                            route, requested);
    }
    lock.lock();
    if (result != CUDA_SUCCESS) {
      return result;
    }
    allocation.portable_routes.push_back(route_id);
    lupine_require_dirty_host_flush();
  }
  if (slot >= complete_generation.size()) {
    complete_generation.resize(slot + 1, UINT64_MAX);
  }
  complete_generation[slot] = generation;
  return CUDA_SUCCESS;
}

// Writable aliases are private to the transport. Only the pinned DtoH RPC
// returns one as a destination; ordinary CUDA calls return application
// pointers.
static lupine_host_allocation_map::iterator
lupine_find_io_alias_locked(uintptr_t address, size_t bytes, int route_id) {
  auto &aliases = lupine_host_allocation_aliases_locked();
  auto alias_it = aliases.upper_bound(address);
  if (alias_it == aliases.begin()) {
    return lupine_mutable_host_allocations_locked().end();
  }
  --alias_it;
  uintptr_t alias = alias_it->first;
  auto &allocations = lupine_mutable_host_allocations_locked();
  auto it = allocations.find(alias_it->second);
  if (it == allocations.end()) {
    return allocations.end();
  }
  const auto &allocation = it->second;
  if (allocation.route_id == route_id && address >= alias &&
      address - alias <= allocation.size &&
      bytes <= allocation.size - (address - alias)) {
    return it;
  }
  return allocations.end();
}

// The client half of pushed device-to-host delivery: one thread per
// connection reads deliveries off a lane of its own and applies them in order.
// It runs no application code and sends no request, so it keeps up while
// callers, and the dispatch thread inside a host callback, wait for it. It
// does take lupine_host_allocation_mutex() and conn->write_mutex, briefly and
// never while reading the transport, to record bytes that landed in pinned
// memory. Large deliveries come over a bulk connection, on a stream of its own
// so it never waits for a caller's bulk copy.
struct lupine_dtoh_receiver {
  std::mutex mutex;
  std::condition_variable applied_changed;
  std::atomic<uint64_t> applied{0};
  std::atomic<bool> stopped{false};
  std::thread thread;
  conn_t *bulk_conn = nullptr;
  int32_t bulk_stream = -1;
};

// The copy id of a delivery that carries no bytes: its copy wrote the server's
// copy of memory refreshed on access, which is marked stale instead.
static constexpr uint64_t kLupineDtohInvalidate = ~uint64_t{0};

static void lupine_mark_pushed_stale(conn_t *conn, void *host);

// Reads bulk copy `copy_id` into `data` over one bulk stream; every chunk
// carries its offset.
static bool lupine_bulk_read_stream(conn_t *bulk, int32_t stream_id,
                                    uint64_t copy_id, unsigned char *data) {
  // A bulk sub-request takes no ticket: the request it serves orders it.
  struct {
    int request_id;
    int op;
    uint64_t ticket;
    uint64_t copy_id;
  } request = {2, LUPINE_RPC_lupineBulkRead, ~uint64_t{0}, copy_id};
  rpc_write_cursor cursor(&request, sizeof(request));
  if (rpc_http2_write_stream(bulk, stream_id, &cursor, 1) < 0) {
    return false;
  }
  for (;;) {
    struct {
      int request_id;
      int op;
      uint64_t offset;
      uint64_t bytes;
    } frame;
    if (rpc_http2_read_stream(bulk, stream_id, &frame, sizeof(frame)) != 0 ||
        frame.op != -1 ||
        (frame.bytes != 0 &&
         rpc_http2_read_stream(bulk, stream_id, data + frame.offset,
                               frame.bytes) != 0)) {
      return false;
    }
    if (frame.bytes == 0) {
      return true;
    }
  }
}

// Lands one pushed span: from the push lane, or with a nonzero copy id from
// the bulk connection. Bytes for pinned memory arrive through its writable
// alias and are recorded so the view stays readable after an invalidation,
// while collection knows the server already has them.
static int lupine_read_pushed_span(conn_t *conn, void *destination,
                                   size_t bytes, bool pinned_destination,
                                   intptr_t alias_offset,
                                   lupine_dtoh_receiver *receiver,
                                   uint64_t copy_id) {
  int result = 0;
  void *foreign_alias =
      alias_offset != 0
          ? static_cast<void *>(static_cast<unsigned char *>(destination) +
                                alias_offset)
          : nullptr;
  if (copy_id == 0) {
    result = rpc_read(
        conn, foreign_alias != nullptr ? foreign_alias : destination, bytes);
    if (result == 0 && foreign_alias != nullptr) {
      result = rpc_note_host_allocation_write(conn, destination, bytes);
    }
  } else {
    void *alias = foreign_alias != nullptr
                      ? foreign_alias
                      : rpc_host_allocation_alias(conn, destination, bytes);
    result = !lupine_bulk_read_stream(
                 receiver->bulk_conn, receiver->bulk_stream, copy_id,
                 static_cast<unsigned char *>(alias != nullptr ? alias
                                                               : destination))
                 ? -1
             : alias != nullptr
                 ? rpc_note_host_allocation_write(conn, destination, bytes)
                 : 0;
  }
  if (result < 0 || !pinned_destination || bytes == 0) {
    return result;
  }
  if (pthread_mutex_lock(&conn->write_mutex) != 0) {
    return -1;
  }
  try {
    conn->host_allocation_writes.push_back(
        {reinterpret_cast<uintptr_t>(destination), bytes});
  } catch (const std::bad_alloc &) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  __atomic_store_n(&conn->host_allocation_writes_pending, 1, __ATOMIC_RELEASE);
  return pthread_mutex_unlock(&conn->write_mutex) == 0 ? result : -1;
}

// Applies one delivery: its rows, then its bytes in the client's own layout.
static int lupine_read_pushed_rows(conn_t *conn, lupine_dtoh_receiver *receiver,
                                   uint64_t copy_id) {
  lupine_host_rows rows;
  if (rpc_read(conn, &rows, sizeof(rows)) < 0) {
    return -1;
  }
  if (copy_id == kLupineDtohInvalidate) {
    lupine_mark_pushed_stale(conn, rows.dst);
    return 0;
  }
  if (rows.width == 0 || rows.height == 0 || rows.depth == 0) {
    return 0;
  }
  bool pinned_destination = false;
  intptr_t alias_offset = 0;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    pinned_destination =
        lupine_find_io_alias_locked(
            reinterpret_cast<uintptr_t>(rows.dst), rows.width,
            lupine_route_identity(lupine_remote_route_for_conn(conn))) !=
        lupine_mutable_host_allocations_locked().end();
    if (!pinned_destination &&
        !lupine_va_contains(conn, reinterpret_cast<uintptr_t>(rows.dst),
                            rows.width)) {
      auto it = lupine_find_host_allocation_locked(rows.dst);
      if (it != lupine_mutable_host_allocations_locked().end() &&
          it->second.io_alias != nullptr) {
        alias_offset = static_cast<unsigned char *>(it->second.io_alias) -
                       static_cast<unsigned char *>(it->first);
      }
    }
  }
  const bool contiguous =
      (rows.height == 1 && rows.depth == 1) ||
      (rows.depth == 1 && rows.pitch == rows.width) ||
      (rows.pitch == rows.width && rows.slice == rows.height * rows.pitch);
  if (contiguous) {
    return lupine_read_pushed_span(conn, rows.dst, rows.bytes(),
                                   pinned_destination, alias_offset, receiver,
                                   copy_id);
  }
  for (size_t index = 0; index < rows.height * rows.depth; ++index) {
    if (lupine_read_pushed_span(conn, rows.row(index), rows.width,
                                pinned_destination, alias_offset, receiver,
                                0) < 0) {
      return -1;
    }
  }
  return 0;
}

static lupine_dtoh_receiver *lupine_dtoh_receiver_for(conn_t *conn) {
  return rpc_dtoh_state<lupine_dtoh_receiver>(conn);
}

static void lupine_dtoh_receiver_stop(lupine_dtoh_receiver *receiver) {
  {
    std::lock_guard<std::mutex> lock(receiver->mutex);
    receiver->stopped.store(true, std::memory_order_release);
  }
  receiver->applied_changed.notify_all();
}

static void lupine_dtoh_receiver_main(conn_t *conn) {
  lupine_dtoh_receiver *receiver = lupine_dtoh_receiver_for(conn);
  uint64_t lane = std::hash<std::thread::id>{}(std::this_thread::get_id());
  if (lupine_bulk_lanes *lanes = lupine_client_transport_bulk_lanes(conn)) {
    receiver->bulk_stream = rpc_http2_lane_stream(lanes->conn[0], lane);
    if (receiver->bulk_stream >= 0) {
      receiver->bulk_conn = lanes->conn[0];
    }
  }
  // A request nobody waits for an answer to, that the server keeps as the
  // lane its push worker sends on.
  struct {
    int request_id;
    int op;
    uint64_t ticket;
    uint64_t bulk_reader;
  } request = {2, LUPINE_RPC_lupineDtohPush, ~uint64_t{0},
               receiver->bulk_conn != nullptr ? 1u : 0u};
  rpc_write_cursor cursors[] = {rpc_write_cursor(&request, sizeof(request))};
  int32_t stream_id = rpc_http2_lane_stream(conn, lane);
  if (stream_id >= 0 &&
      rpc_http2_write_stream(conn, stream_id, cursors, std::size(cursors)) ==
          0 &&
      rpc_bind_http2_stream(conn, stream_id) == 0) {
    uint64_t number = 0;
    uint64_t copy_id = 0;
    while (rpc_read(conn, &number, sizeof(number)) == 0 &&
           rpc_read(conn, &copy_id, sizeof(copy_id)) == 0 &&
           lupine_read_pushed_rows(conn, receiver, copy_id) == 0) {
      {
        std::lock_guard<std::mutex> lock(receiver->mutex);
        receiver->applied.store(number, std::memory_order_release);
      }
      receiver->applied_changed.notify_all();
    }
    rpc_unbind_http2_stream(conn);
  }
  lupine_dtoh_receiver_stop(receiver);
}

void lupine_start_dtoh_receiver(conn_t *conn) {
  lupine_dtoh_receiver *receiver = lupine_dtoh_receiver_for(conn);
  try {
    receiver->thread = std::thread(lupine_dtoh_receiver_main, conn);
  } catch (const std::system_error &) {
    LUPINE_LOG_ERROR("Starting the device-to-host receiver failed");
    lupine_dtoh_receiver_stop(receiver);
  }
}

void lupine_join_dtoh_receiver(conn_t *conn) {
  lupine_dtoh_receiver *receiver = lupine_dtoh_receiver_for(conn);
  if (receiver->thread.joinable()) {
    receiver->thread.join();
  }
}

static int lupine_wait_dtoh(conn_t *conn, uint64_t number) {
  lupine_dtoh_receiver *receiver = lupine_dtoh_receiver_for(conn);
  if (receiver->applied.load(std::memory_order_acquire) >= number) {
    return 0;
  }
  std::unique_lock<std::mutex> lock(receiver->mutex);
  receiver->applied_changed.wait(lock, [&] {
    return receiver->applied.load(std::memory_order_acquire) >= number ||
           receiver->stopped.load(std::memory_order_acquire);
  });
  return receiver->applied.load(std::memory_order_acquire) >= number ? 0 : -1;
}

extern "C" int lupine_read_dtoh_watermark(conn_t *conn) {
  uint64_t number = 0;
  return rpc_read(conn, &number, sizeof(number)) < 0
             ? -1
             : lupine_wait_dtoh(conn, number);
}

static CUresult lupine_collect_host_allocation_writes(conn_t *conn) {
  if (__atomic_load_n(&conn->host_allocation_writes_pending,
                      __ATOMIC_ACQUIRE) == 0) {
    return CUDA_SUCCESS;
  }
  std::vector<rpc_host_allocation_write> writes;
  if (pthread_mutex_lock(&conn->write_mutex) != 0) {
    return CUDA_ERROR_UNKNOWN;
  }
  writes.swap(conn->host_allocation_writes);
  __atomic_store_n(&conn->host_allocation_writes_pending, 0, __ATOMIC_RELEASE);
  if (pthread_mutex_unlock(&conn->write_mutex) != 0) {
    return CUDA_ERROR_UNKNOWN;
  }
  if (writes.empty()) {
    return CUDA_SUCCESS;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (const auto &write : writes) {
    uintptr_t start = write.start;
    bool pinned_destination = false;
    auto it =
        lupine_find_host_allocation_locked(reinterpret_cast<void *>(start));
    if (it == lupine_mutable_host_allocations_locked().end()) {
      it = lupine_find_io_alias_locked(
          start, write.size,
          lupine_route_identity(lupine_remote_route_for_conn(conn)));
      if (it != lupine_mutable_host_allocations_locked().end()) {
        start = reinterpret_cast<uintptr_t>(it->first) + start -
                reinterpret_cast<uintptr_t>(it->second.io_alias);
        pinned_destination = true;
      }
    }
    if (it == lupine_mutable_host_allocations_locked().end() ||
        !it->second.tracking_enabled) {
      continue;
    }
    auto &allocation = it->second;
    uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    size_t offset = start - base;
    size_t bytes = std::min(write.size, allocation.size - offset);
    if (bytes == 0) {
      continue;
    }
    uintptr_t end = start + bytes;
    if (!pinned_destination) {
      lupine_queue_dirty_host_range(&allocation, start, end);
    } else {
      uintptr_t page_start = start - start % allocation.page_size;
      uintptr_t page_end = lupine_round_up(end, allocation.page_size);
      if (!lupine_protect_host_range(reinterpret_cast<void *>(page_start),
                                     page_end - page_start, PROT_READ)) {
        return CUDA_ERROR_UNKNOWN;
      }
    }

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
  if (__atomic_load_n(&conn->host_allocation_writes_pending,
                      __ATOMIC_ACQUIRE) != 0) {
    CUresult collect_result = lupine_collect_host_allocation_writes(conn);
    if (collect_result != CUDA_SUCCESS) {
      return collect_result;
    }
  }

  auto &queue = lupine_dirty_host_range_queues[route_id];
  std::lock_guard<std::mutex> route_lock(queue.mutex);

  uint32_t reserved = __atomic_load_n(&queue.next, __ATOMIC_ACQUIRE);
  uint32_t end = queue.start;
  while (end < reserved) {
    // A signal handler reserves its lock-free slot before publishing it. A
    // later slot can therefore carry an epoch included in our completion
    // watermark while this earlier producer is still filling its range. Do
    // not advance the watermark past that hole.
    while (__atomic_load_n(&queue.ready[end], __ATOMIC_ACQUIRE) == 0) {
      sched_yield();
    }
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

  // Nothing observes writes to an untracked primary registration, so its
  // whole window goes over before device work can read it. Dirty-page tracking
  // is anchored at the primary mapping, so portable secondary mappings also
  // receive the allocation's current contents before work is submitted there.
  if (lupine_exposed_or_managed_host_count.load(std::memory_order_acquire) !=
      0) {
    std::lock_guard<std::mutex> allocation_lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    for (auto it = allocations.begin(); it != allocations.end(); ++it) {
      auto &allocation = it->second;
      if (!allocation.device_pointer_exposed ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
        continue;
      }
      if (lupine_host_backing_for_route(allocation,
                                        static_cast<int>(route_id)) == 0) {
        continue;
      }
      bool primary = allocation.route_id == static_cast<int>(route_id);
      if (primary && (allocation.tracking_enabled || allocation.local_cuda)) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
      ranges.push_back(
          {&allocation, allocation.host_base,
           allocation.host_base +
               (primary ? allocation.size : allocation.storage_size)});
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
  size_t merged_count = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (merged_count == 0 ||
        ranges[merged_count - 1].allocation != ranges[i].allocation ||
        ranges[merged_count - 1].end < ranges[i].start) {
      if (merged_count != i) {
        ranges[merged_count] = ranges[i];
      }
      ++merged_count;
    } else {
      ranges[merged_count - 1].end =
          std::max(ranges[merged_count - 1].end, ranges[i].end);
      __atomic_sub_fetch(&ranges[i].allocation->pending_dirty_ranges, 1,
                         __ATOMIC_RELEASE);
    }
  }
  ranges.resize(merged_count);

  for (const auto &range : ranges) {
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
    // the raw transport primitives. No answer is needed: the server applies
    // the bytes before releasing this request's ticket, and every later
    // request from any lane, such as a server-authoritative HtoD, waits for
    // it.
    if (rpc_write_start_request_on(conn, LUPINE_RPC_lupineManagedHostFlush,
                                   RPC_SCOPE_HOST_WRITES) < 0 ||
        rpc_write(conn, &count, sizeof(count)) < 0 ||
        rpc_write_cursors(conn, cursors.data(), count * 2) < 0 ||
        rpc_write_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    __atomic_store_n(&conn->host_flushes_unsettled, 1, __ATOMIC_RELEASE);
    return CUDA_SUCCESS;
  };

  // Copies this large fan their chunks out over the bulk connections instead of
  // travelling on the session connection; the batch ahead of them goes first
  // only so a batch never straddles one.
  uint32_t count = 0;
  for (const auto &range : ranges) {
    auto &allocation = *range.allocation;
    uintptr_t data_start = allocation.host_base + allocation.data_offset;
    uintptr_t start = std::max(range.start, data_start);
    uintptr_t end = std::min(range.end, data_start + allocation.user_size);
    if (end <= start) {
      continue;
    }
    size_t offset = start - allocation.host_base;
    size_t bytes = end - start;
    CUdeviceptr backing =
        lupine_host_backing_for_route(allocation, static_cast<int>(route_id));
    if (backing == 0) {
      release_ranges(true);
      return CUDA_ERROR_INVALID_VALUE;
    }
    CUdeviceptr dst = backing + offset;
    const void *source = reinterpret_cast<void *>(start);
    if (allocation.io_alias != nullptr) {
      source = static_cast<unsigned char *>(allocation.io_alias) + offset;
    }
    lupine_bulk_lanes *lanes = bytes >= LUPINE_BULK_COPY_MIN_BYTES
                                   ? lupine_client_transport_bulk_lanes(conn)
                                   : nullptr;
    if (lanes != nullptr) {
      CUresult result = send_batch(count);
      count = 0;
      if (result == CUDA_SUCCESS) {
        result = lupine_bulk_push(conn, lanes, dst, source, bytes, true);
      }
      if (result != CUDA_SUCCESS) {
        release_ranges(true);
        return result;
      }
      continue;
    }
    memcpy(headers[count].data(), &dst, sizeof(dst));
    memcpy(headers[count].data() + sizeof(dst), &bytes, sizeof(bytes));
    cursors[count * 2] = rpc_write_cursor(
        headers[count].data(), LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES);
    cursors[count * 2 + 1] = rpc_write_cursor(source, bytes);
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

static CUresult lupine_flush_all_dirty_host_routes_locked() {
  uint64_t flush_epoch =
      __atomic_load_n(&lupine_dirty_host_epoch, __ATOMIC_ACQUIRE);
  for (int route_id = 0; route_id < rpc_size(); ++route_id) {
    CUresult result =
        lupine_flush_dirty_host_pages_to_route(static_cast<size_t>(route_id));
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  lupine_flushed_host_epoch.store(flush_epoch, std::memory_order_release);
  return CUDA_SUCCESS;
}

static CUresult lupine_collect_all_pending_host_writes() {
  for (int route_id = 0; route_id < rpc_size(); ++route_id) {
    conn_t *conn =
        lupine_route_remote_conn(lupine_route_from_identity(route_id));
    if (conn != nullptr &&
        __atomic_load_n(&conn->host_allocation_writes_pending,
                        __ATOMIC_ACQUIRE) != 0) {
      CUresult result = lupine_collect_host_allocation_writes(conn);
      if (result != CUDA_SUCCESS) {
        return result;
      }
    }
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_flush_dirty_host_pages_to_server() {
  std::lock_guard<std::mutex> flush_lock(lupine_host_flush_mutex);
  return lupine_flush_all_dirty_host_routes_locked();
}

extern "C" int lupine_prepare_rpc(conn_t *conn) {
  if (conn == nullptr) {
    return -1;
  }
  __atomic_store_n(&lupine_device_work_pending, 1, __ATOMIC_RELEASE);
  if (lupine_collect_all_pending_host_writes() != CUDA_SUCCESS) {
    return -1;
  }

  uint64_t required_epoch =
      __atomic_load_n(&lupine_dirty_host_epoch, __ATOMIC_ACQUIRE);
  if (lupine_flushed_host_epoch.load(std::memory_order_acquire) <
      required_epoch) {
    std::lock_guard<std::mutex> flush_lock(lupine_host_flush_mutex);
    while (lupine_flushed_host_epoch.load(std::memory_order_acquire) <
           required_epoch) {
      if (lupine_flush_all_dirty_host_routes_locked() != CUDA_SUCCESS) {
        return -1;
      }
    }
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
    std::lock_guard<std::mutex> flush_lock(lupine_host_flush_mutex);
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
         (it->second.managed || it->second.managed_global);
}

extern "C" CUresult lupine_managed_mem_advise(CUdeviceptr ptr, size_t count,
                                              CUmem_advise advice,
                                              CUdevice device) {
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(reinterpret_cast<void *>(ptr));
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.managed)
    return CUDA_ERROR_INVALID_VALUE;
  auto &allocation = it->second;
  size_t offset = ptr - allocation.user_base;
  if (!count || offset >= allocation.user_size ||
      count > allocation.user_size - offset)
    return CUDA_ERROR_INVALID_VALUE;
  return allocation.advice.advise(offset, count, allocation.page_size, advice,
                                  device);
}

extern "C" bool lupine_managed_mem_range_attribute(
    void *data, size_t data_size, CUmem_range_attribute attribute,
    CUdeviceptr ptr, size_t count, CUresult *result) {
  if (!lupine_managed_advice::supports(attribute))
    return false;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(reinterpret_cast<void *>(ptr));
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.managed || it->second.local_cuda)
    return false;
  const auto &allocation = it->second;
  size_t offset = ptr - allocation.user_base;
  if (!count || offset >= allocation.user_size ||
      count > allocation.user_size - offset) {
    *result = CUDA_ERROR_INVALID_VALUE;
  } else {
    *result = allocation.advice.query(data, data_size, attribute, offset, count,
                                      allocation.page_size);
  }
  return true;
}

static bool lupine_translate_client_host_range_locked(CUdeviceptr *translated,
                                                      const void *host,
                                                      size_t size,
                                                      int route_id) {
  auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
  // Unprotected pinned buffers can change through read(2) without publishing
  // dirty pages. Copies must transfer their current client bytes.
  if (it == lupine_mutable_host_allocations_locked().end() ||
      it->second.server_host_ptr == 0 || it->second.local_cuda ||
      !it->second.tracking_enabled || it->second.route_id != route_id) {
    return false;
  }
  size_t offset = reinterpret_cast<uintptr_t>(host) -
                  reinterpret_cast<uintptr_t>(it->first);
  if (offset > it->second.size || size > it->second.size - offset) {
    return false;
  }
  *translated = it->second.server_host_ptr + offset;
  return true;
}

static bool lupine_translate_client_host_range_to_server(
    CUdeviceptr *translated, const void *host, size_t size, int route_id) {
  if (host == nullptr || translated == nullptr ||
      !lupine_addr_in_tracked_host_bounds(reinterpret_cast<uintptr_t>(host))) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  return lupine_translate_client_host_range_locked(translated, host, size,
                                                   route_id);
}

static inline bool
lupine_allocation_covers_user_range(const lupine_host_allocation &allocation,
                                    uintptr_t address, size_t bytes) {
  return address >= allocation.user_base &&
         address < allocation.user_base + allocation.user_size &&
         bytes <= allocation.user_size - (address - allocation.user_base);
}

static void lupine_note_dtoh_route_locked(lupine_host_allocation &allocation,
                                          conn_t *conn) {
  if (conn == nullptr || allocation.local_cuda ||
      __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
    return;
  }
  int route_id = lupine_route_identity(lupine_remote_route_for_conn(conn));
  if (route_id >= 0 && route_id != allocation.route_id &&
      std::find(allocation.dtoh_routes.begin(), allocation.dtoh_routes.end(),
                route_id) == allocation.dtoh_routes.end()) {
    allocation.dtoh_routes.push_back(route_id);
  }
}

static bool lupine_pinned_dtoh_destination(conn_t *conn, void *destination,
                                           size_t bytes, bool *page_locked,
                                           void **client_alias,
                                           void **server_host) {
  if (page_locked != nullptr) {
    *page_locked = false;
  }
  if (client_alias != nullptr) {
    *client_alias = nullptr;
  }
  if (server_host != nullptr) {
    *server_host = nullptr;
  }
  if (destination == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(destination);
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return false;
  }
  auto &allocation = it->second;
  uintptr_t address = reinterpret_cast<uintptr_t>(destination);
  bool is_page_locked =
      lupine_allocation_covers_user_range(allocation, address, bytes);
  if (page_locked != nullptr) {
    *page_locked = is_page_locked;
  }
  if (!is_page_locked || bytes == 0 || conn == nullptr) {
    return false;
  }
  lupine_note_dtoh_route_locked(allocation, conn);
  // Memory whose device pointer is exposed and tracked is refreshed from the
  // server's copy and fetched on access, so a copy into it writes only that
  // copy and sends no client address. Other page-locked memory gets the bytes
  // pushed, through its writable alias where it has one; exposed but untracked
  // memory is read back from the server's copy at a synchronize, so that copy
  // is named to be updated too.
  bool exposed = allocation.device_pointer_exposed;
  bool refreshed = exposed && allocation.tracking_enabled;
  if (allocation.local_cuda || allocation.server_host_ptr == 0 ||
      allocation.route_id !=
          lupine_route_identity(lupine_remote_route_for_conn(conn)) ||
      __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
    return false;
  }
  size_t offset = address - reinterpret_cast<uintptr_t>(it->first);
  if (client_alias != nullptr && !refreshed) {
    *client_alias =
        allocation.io_alias != nullptr
            ? static_cast<unsigned char *>(allocation.io_alias) + offset
            : destination;
  }
  if (server_host != nullptr && exposed) {
    *server_host =
        reinterpret_cast<void *>(allocation.server_host_ptr + offset);
  }
  return true;
}

static bool lupine_host_ptr_is_tracked(CUdeviceptr ptr) {
  if (ptr == 0 ||
      !lupine_addr_in_tracked_host_bounds(static_cast<uintptr_t>(ptr))) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  return lupine_find_host_allocation_locked(reinterpret_cast<void *>(ptr)) !=
         lupine_mutable_host_allocations_locked().end();
}

static bool lupine_host_range_is_page_locked(const void *host, size_t bytes,
                                             conn_t *dtoh_conn = nullptr) {
  if (host == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !lupine_allocation_covers_user_range(
          it->second, reinterpret_cast<uintptr_t>(host), bytes)) {
    return false;
  }
  if (bytes != 0) {
    lupine_note_dtoh_route_locked(it->second, dtoh_conn);
  }
  return true;
}

static bool lupine_is_client_mapped_address(CUdeviceptr ptr) {
  if (ptr == 0) {
    return false;
  }
  size_t page_size = lupine_page_size();
  uintptr_t page =
      static_cast<uintptr_t>(ptr) & ~(static_cast<uintptr_t>(page_size) - 1);
#if defined(__APPLE__)
  char residency = 0;
#else
  unsigned char residency = 0;
#endif
  return mincore(reinterpret_cast<void *>(page), page_size, &residency) == 0;
}

static bool lupine_copy_pointer_is_host(CUdeviceptr ptr) {
  if (lupine_host_ptr_is_tracked(ptr)) {
    return true;
  }
  if (lupine_deviceptr_is_tracked(ptr)) {
    return false;
  }
  return lupine_is_client_mapped_address(ptr);
}

// A managed alias is already the server's address for its allocation, so a
// copy into or out of it runs on the device side, in stream order, as in 2D and
// 3D copies. Treated as host memory, bytes copied into it would land in the
// client's view and reach the server only with a later dirty-page flush, on
// top of whatever the stream's subsequent work had written there; and a copy
// out of it into page-locked memory would block the caller until the stream
// drained, where the driver returns at once.
static uint8_t lupine_infer_copy_direction(CUdeviceptr dst, CUdeviceptr src) {
  enum class host_ptr_class { untracked, tracked_host, managed_alias };
  auto classify_locked = [](CUdeviceptr ptr) {
    auto it = lupine_find_host_allocation_locked(reinterpret_cast<void *>(ptr));
    if (it == lupine_mutable_host_allocations_locked().end()) {
      return host_ptr_class::untracked;
    }
    return (it->second.managed || it->second.managed_global)
               ? host_ptr_class::managed_alias
               : host_ptr_class::tracked_host;
  };
  host_ptr_class dst_class;
  host_ptr_class src_class;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    dst_class = classify_locked(dst);
    src_class = classify_locked(src);
  }
  auto is_host = [](CUdeviceptr ptr, host_ptr_class cls) {
    if (cls == host_ptr_class::managed_alias) {
      return false;
    }
    if (cls == host_ptr_class::tracked_host) {
      return true;
    }
    return !lupine_deviceptr_is_tracked(ptr) &&
           lupine_is_client_mapped_address(ptr);
  };
  const bool dst_is_host = is_host(dst, dst_class);
  const bool src_is_host = is_host(src, src_class);
  if (dst_is_host && src_is_host) {
    return LUPINE_COPY_DIRECTION_HTOH;
  }
  if (src_is_host) {
    return LUPINE_COPY_DIRECTION_HTOD;
  }
  if (dst_is_host) {
    return LUPINE_COPY_DIRECTION_DTOH;
  }
  return LUPINE_COPY_DIRECTION_DTOD;
}

static bool lupine_stream_is_capturing(CUstream hStream) {
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  return lupine_active_stream_captures.load(std::memory_order_relaxed) != 0 &&
         cuStreamIsCapturing(hStream, &capture_status) == CUDA_SUCCESS &&
         capture_status != CU_STREAM_CAPTURE_STATUS_NONE;
}

struct lupine_captured_htoh_copy {
  char *dst = nullptr;
  const char *src = nullptr;
  size_t width = 0;
  size_t height = 1;
  size_t depth = 1;
  size_t dst_pitch = 0;
  size_t src_pitch = 0;
  size_t dst_slice_pitch = 0;
  size_t src_slice_pitch = 0;
};

static void CUDA_CB lupine_run_captured_htoh_copy(void *user_data) {
  const auto *copy = static_cast<const lupine_captured_htoh_copy *>(user_data);
  if (copy == nullptr || copy->width == 0 || copy->height == 0 ||
      copy->depth == 0) {
    return;
  }
  (void)lupine_sync_mapped_device_to_host();
  if (copy->height == 1 && copy->depth == 1) {
    memmove(copy->dst, copy->src, copy->width);
    return;
  }
  for (size_t z = 0; z < copy->depth; ++z) {
    for (size_t row = 0; row < copy->height; ++row) {
      memmove(copy->dst + z * copy->dst_slice_pitch + row * copy->dst_pitch,
              copy->src + z * copy->src_slice_pitch + row * copy->src_pitch,
              copy->width);
    }
  }
}

static CUresult
lupine_capture_host_to_host(char *dst, const char *src, size_t width,
                            size_t height, size_t depth, size_t dst_pitch,
                            size_t src_pitch, size_t dst_slice_pitch,
                            size_t src_slice_pitch, CUstream hStream) {
  if (width != 0 && height != 0 && depth != 0 &&
      (dst == nullptr || src == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto *spec = new (std::nothrow) lupine_captured_htoh_copy{
      dst,       src,       width,           height,         depth,
      dst_pitch, src_pitch, dst_slice_pitch, src_slice_pitch};
  if (spec == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  CUresult result =
      cuLaunchHostFunc(hStream, lupine_run_captured_htoh_copy, spec);
  if (result != CUDA_SUCCESS) {
    delete spec;
  }
  return result;
}

// A copy between host pointers is still a copy on its stream: it waits for the
// work ahead of it, such as a host function filling a pinned source or a
// kernel writing a mapped one, and work behind it sees what it wrote. When
// either side is memory the client tracks, finish that work before copying
// here. Synchronizing refreshes mapped and managed views, and the stores below
// reach the server as dirty pages before any later request.
static CUresult lupine_order_host_to_host(const void *dst, const void *src,
                                          CUstream hStream) {
  if (!lupine_host_ptr_is_tracked(reinterpret_cast<CUdeviceptr>(dst)) &&
      !lupine_host_ptr_is_tracked(reinterpret_cast<CUdeviceptr>(src))) {
    return CUDA_SUCCESS;
  }
  if (lupine_stream_is_capturing(hStream)) {
    return CUDA_SUCCESS;
  }
  return cuStreamSynchronize(hStream);
}

static CUresult lupine_memcpy_host_to_host(void *dst, const void *src,
                                           size_t ByteCount, CUstream hStream,
                                           bool async) {
  if (async && lupine_stream_is_capturing(hStream)) {
    return lupine_capture_host_to_host(
        static_cast<char *>(dst), static_cast<const char *>(src), ByteCount, 1,
        1, ByteCount, ByteCount, ByteCount, ByteCount, hStream);
  }
  CUresult result = lupine_order_host_to_host(dst, src, hStream);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (ByteCount != 0) {
    if (dst == nullptr || src == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    memmove(dst, src, ByteCount);
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_memcpy(CUdeviceptr dst, CUdeviceptr src,
                              size_t ByteCount, CUstream order) {
  if (ByteCount == 0) {
    return CUDA_SUCCESS;
  }

  switch (lupine_infer_copy_direction(dst, src)) {
  case LUPINE_COPY_DIRECTION_HTOH:
    return lupine_memcpy_host_to_host(reinterpret_cast<void *>(dst),
                                      reinterpret_cast<const void *>(src),
                                      ByteCount, order, false);
  case LUPINE_COPY_DIRECTION_HTOD:
    return cuMemcpyHtoD_v2(dst, reinterpret_cast<const void *>(src), ByteCount);
  case LUPINE_COPY_DIRECTION_DTOH:
    return cuMemcpyDtoH_v2(reinterpret_cast<void *>(dst), src, ByteCount);
  case LUPINE_COPY_DIRECTION_DTOD:
    return cuMemcpyDtoD_v2(dst, src, ByteCount);
  }
  return CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src,
                             size_t ByteCount) {
  return lupine_memcpy(dst, src, ByteCount, nullptr);
}

#ifdef cuMemcpy_ptds
#undef cuMemcpy_ptds
#endif
extern "C" CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount) {
  return lupine_memcpy(dst, src, ByteCount, CU_STREAM_PER_THREAD);
}

extern "C" CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount, CUstream hStream) {
  switch (lupine_infer_copy_direction(dst, src)) {
  case LUPINE_COPY_DIRECTION_HTOH:
    return lupine_memcpy_host_to_host(reinterpret_cast<void *>(dst),
                                      reinterpret_cast<const void *>(src),
                                      ByteCount, hStream, true);
  case LUPINE_COPY_DIRECTION_HTOD:
    return cuMemcpyHtoDAsync_v2(dst, reinterpret_cast<const void *>(src),
                                ByteCount, hStream);
  case LUPINE_COPY_DIRECTION_DTOH:
    return cuMemcpyDtoHAsync_v2(reinterpret_cast<void *>(dst), src, ByteCount,
                                hStream);
  case LUPINE_COPY_DIRECTION_DTOD:
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

// Reads a device-to-host response sent as LUPINE_RPC_TRANSFER_CHUNK_BYTES
// chunks, each led by its CUresult; a failed chunk ends the response.
// Each chunk carries the deliveries numbered once it was copied. A readback
// waited for its stream, so it waits for those to land too; a demand fetch,
// which reads the server's copy for a fault, does not.
static CUresult lupine_read_dtoh_chunks(conn_t *conn, int request_id,
                                        unsigned char *dst, size_t bytes,
                                        bool readback = true) {
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  uint64_t delivered = 0;
  size_t offset = 0;
  do {
    size_t chunk =
        std::min(bytes - offset, (size_t)LUPINE_RPC_TRANSFER_CHUNK_BYTES);
    if (rpc_read(conn, &delivered, sizeof(delivered)) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && chunk != 0 &&
         rpc_read(conn, dst + offset, chunk) < 0)) {
      rpc_read_end(conn);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    bool final_chunk = result != CUDA_SUCCESS || offset + chunk == bytes;
    if (rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offset += chunk;
    if (!final_chunk && rpc_read_start(conn, request_id) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  } while (offset < bytes);
  return readback && lupine_wait_dtoh(conn, delivered) < 0
             ? CUDA_ERROR_DEVICE_UNAVAILABLE
             : result;
}

// Wire-fetch [offset, offset + bytes) of the backing. Handler-safe: must
// not take lupine_host_allocation_mutex(); caller made the range writable.
static CUresult lupine_rpc_memcpy_htoh_1d(conn_t *conn, uintptr_t scope,
                                          void *dst, CUdeviceptr server_src,
                                          size_t bytes) {
  uint8_t direction = LUPINE_COPY_DIRECTION_HTOH;
  CUDA_MEMCPY3D copy = {};
  copy.srcMemoryType = copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = reinterpret_cast<const void *>(server_src);
  copy.dstHost = dst;
  copy.srcPitch = copy.dstPitch = copy.WidthInBytes = bytes;
  copy.srcHeight = copy.dstHeight = copy.Height = copy.Depth = 1;
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (rpc_write_start_request_on(conn, RPC_cuMemcpy3D_v2, scope) < 0 ||
      rpc_write(conn, &direction, sizeof(direction)) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 ||
      (result == CUDA_SUCCESS && rpc_read(conn, dst, bytes) < 0) ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

// Callers work in tracked-page space, so the span can reach past the bytes the
// allocation owns; this is the only place server bytes land in client pages,
// so it clips here rather than trusting each caller to.
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
  uintptr_t fetch_scope =
      lupine_in_host_callback ? RPC_SCOPE_LANE : allocation->stale_fetch_scope;
  // Pinned backing is already CPU accessible on the server. A CUDA copy here
  // can deadlock when cuMemFree holds a driver lock while waiting for the host
  // callback that faulted. Read the completed bytes without entering CUDA.
  // Waiting for every request instead would hold the fetch behind any thread
  // flooding another stream with unanswered work.
  if (!allocation->managed) {
    return lupine_rpc_memcpy_htoh_1d(conn, fetch_scope, dst,
                                     allocation->server_host_ptr + offset,
                                     bytes) == CUDA_SUCCESS;
  }
  // Any thread can fault, so any lane can carry the fetch. A touch inside a
  // host-func callback faults on the RPC dispatch thread, whose lane has never
  // carried a CUDA call and so has no context current on the server to copy
  // under. Demand fetch runs from the fault handler and must not re-enter the
  // normal CUDA request-start flush, so the binding goes straight to the
  // transport rather than through the routing helper, and consults the same
  // per-lane cache so an already-bound lane still fetches in one round trip.
  CUcontext context = allocation->stale_fetch_context;
  if (context != nullptr &&
      !lupine_lane_context_cache_matches(allocation->route_id, context)) {
    uint64_t epoch = lupine_lane_context_cache_epoch();
    CUresult bound = CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (rpc_write_start_request_on(conn, RPC_cuCtxSetCurrent, RPC_SCOPE_LANE) <
            0 ||
        rpc_write(conn, &context, sizeof(context)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &bound, sizeof(bound)) < 0 || rpc_read_end(conn) < 0) {
      return false;
    }
    lupine_note_device_binding_changed();
    lupine_lane_context_cache_update(allocation->route_id, context, epoch,
                                     bound == CUDA_SUCCESS);
    if (bound != CUDA_SUCCESS) {
      return false;
    }
  }
  // The stream running a host-func callback stays blocked until the client
  // answers it, so a fetch queued behind a launch there would wait on the
  // callback that is waiting on it; the launch that callback follows is done.
  CUstream fetch_stream = allocation->stale_fetch_stream;
  if (lupine_in_host_callback) {
    fetch_stream = lupine_stream_fetch_ready();
  } else if (fetch_stream == CU_STREAM_PER_THREAD &&
             allocation->stale_fetch_ptds_tid != lupine_gettid()) {
    fetch_stream = lupine_stream_fetch_sync_all();
    fetch_scope = RPC_SCOPE_ALL;
  }
  if (rpc_write_start_request_on(conn, RPC_cuMemcpyDtoH_v2, fetch_scope) < 0 ||
      rpc_write(conn, &src, sizeof(src)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &fetch_stream, sizeof(fetch_stream)) < 0) {
    return false;
  }
  return lupine_read_dtoh_chunks(conn,
                                 (rpc_expect_answer(conn), rpc_write_end(conn)),
                                 dst, bytes, false) == CUDA_SUCCESS;
}

static bool lupine_arm_stale_fetch(void *host,
                                   lupine_host_allocation &allocation,
                                   conn_t *conn, CUcontext fetch_context,
                                   CUstream stream, uintptr_t scope,
                                   pid_t ptds_tid, sig_atomic_t previous);

// Publish a finished fetch and drop the stale state once every chunk is fresh.
// Returns true if a pushed copy invalidated the allocation mid-fetch and
// re-armed it stale, so normal-context callers know to re-fetch.
static bool lupine_finish_demand_fetch(lupine_host_allocation *allocation,
                                       size_t span_start, size_t span_end) {
  bool all_fresh = true;
  for (size_t i = 0; i < allocation->fresh_chunk_count; ++i) {
    if (__atomic_load_n(&allocation->fresh_chunks[i], __ATOMIC_ACQUIRE) == 0) {
      all_fresh = false;
      break;
    }
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  if (allocation->stale_rearm_pending) {
    allocation->stale_rearm_pending = false;
    bool keep_launch_scope =
        allocation->stale_fetch_stream != lupine_stream_fetch_ready();
    CUstream arm_stream = keep_launch_scope ? allocation->stale_fetch_stream
                                            : lupine_stream_fetch_ready();
    uintptr_t arm_scope =
        keep_launch_scope ? allocation->stale_fetch_scope : RPC_SCOPE_LANE;
    pid_t arm_ptds_tid =
        keep_launch_scope ? allocation->stale_fetch_ptds_tid : 0;
    (void)lupine_arm_stale_fetch(
        reinterpret_cast<void *>(allocation->host_base), *allocation,
        allocation->stale_rearm_conn, allocation->stale_rearm_context,
        arm_stream, arm_scope, arm_ptds_tid, 1);
    return true;
  }
  if (span_end > span_start) {
    mprotect(reinterpret_cast<void *>(allocation->host_base + span_start),
             span_end - span_start, PROT_READ);
  }
  __atomic_store_n(&allocation->device_stale, all_fresh ? 0 : 1,
                   __ATOMIC_RELEASE);
  return false;
}

static bool lupine_fetch_stale_span(lupine_host_allocation *allocation,
                                    size_t start, size_t end) {
  if (allocation->io_alias == nullptr) {
    mprotect(reinterpret_cast<void *>(allocation->host_base + start),
             end - start, PROT_READ | PROT_WRITE);
  }
  if (!lupine_fetch_stale_range(allocation, start, end - start)) {
    return false;
  }
  size_t chunk_bytes = LUPINE_DEMAND_FETCH_CHUNK_BYTES;
  for (size_t index = start / chunk_bytes;
       index < (end + chunk_bytes - 1) / chunk_bytes &&
       index < allocation->fresh_chunk_count;
       ++index) {
    __atomic_store_n(&allocation->fresh_chunks[index], 1, __ATOMIC_RELEASE);
  }
  return true;
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
  if (!lupine_fetch_stale_span(allocation, start, end)) {
    return false;
  }
  allocation->fetch_seq_next = end;
  return true;
}

// Make a host allocation range fresh from normal context. Must not be called
// while holding the allocation mutex or any connection.
static bool lupine_make_mapped_range_fresh(lupine_host_allocation *allocation,
                                           size_t start_offset,
                                           size_t end_offset) {
  size_t chunk_bytes = LUPINE_DEMAND_FETCH_CHUNK_BYTES;
  for (;;) {
    bool owns_stale = true;
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
        if (lupine_in_host_callback) {
          owns_stale = false;
          break;
        }
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
    if (owns_stale) {
      __atomic_store_n(&allocation->stale_fetch_tid, lupine_gettid(),
                       __ATOMIC_RELEASE);
    }
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
      ok = lupine_fetch_stale_span(allocation, position, run_end);
      if (ok) {
        span_end = run_end;
      }
      position = run_end;
    }
    bool rearmed = false;
    if (owns_stale) {
      if (ok) {
        rearmed = lupine_finish_demand_fetch(allocation, span_start, span_end);
      } else {
        __atomic_store_n(&allocation->device_stale, 1, __ATOMIC_RELEASE);
      }
      __atomic_store_n(&allocation->stale_fetch_tid, 0, __ATOMIC_RELEASE);
    } else if (ok && span_end > span_start) {
      std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
      mprotect(reinterpret_cast<void *>(allocation->host_base + span_start),
               span_end - span_start, PROT_READ);
    }
    if (!rearmed) {
      return ok;
    }
  }
}

// Materialize stale data before the connection is held, then return the
// permanent W alias so later R invalidation cannot fault inside the RPC.
extern "C" const void *lupine_mapped_host_read_source(const void *host,
                                                      size_t size) {
  if (host == nullptr || size == 0) {
    return host;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  if (start > UINTPTR_MAX - size ||
      !lupine_range_in_tracked_host_bounds(start, size)) {
    return host;
  }
  uintptr_t end = start + size;
  auto resolve_alias_locked = [&]() -> const void * {
    auto it = lupine_find_host_allocation_locked(const_cast<void *>(host));
    if (it == lupine_mutable_host_allocations_locked().end() ||
        it->second.io_alias == nullptr) {
      return host;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    if (end > base + it->second.size) {
      return host;
    }
    return static_cast<unsigned char *>(it->second.io_alias) + (start - base);
  };
  struct pending_fetch {
    lupine_host_allocation *allocation;
    size_t start_offset;
    size_t end_offset;
  };
  std::vector<pending_fetch> fetches;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = allocations.upper_bound(const_cast<void *>(host));
    if (it != allocations.begin()) {
      --it;
    }
    for (; it != allocations.end(); ++it) {
      uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      if (base >= end) {
        break;
      }
      auto &allocation = it->second;
      if (base + allocation.size <= start || !allocation.tracking_enabled ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0 ||
          __atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE) == 0) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1, __ATOMIC_ACQ_REL);
      size_t overlap_start = start > base ? start - base : 0;
      size_t overlap_end = std::min(end - base, allocation.size);
      fetches.push_back({&allocation, overlap_start, overlap_end});
    }
    if (fetches.empty()) {
      return resolve_alias_locked();
    }
  }
  for (const auto &fetch : fetches) {
    lupine_make_mapped_range_fresh(fetch.allocation, fetch.start_offset,
                                   fetch.end_offset);
    __atomic_sub_fetch(&fetch.allocation->pending_dirty_ranges, 1,
                       __ATOMIC_RELEASE);
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  return resolve_alias_locked();
}

// Servicing a fault means a copy, and a copy inside a capture window
// invalidates the capture.
extern "C" void lupine_materialize_host_allocations() {
  std::vector<lupine_host_allocation *> stale;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      if (entry.second.tracking_enabled &&
          __atomic_load_n(&entry.second.retiring, __ATOMIC_ACQUIRE) == 0 &&
          __atomic_load_n(&entry.second.device_stale, __ATOMIC_ACQUIRE) != 0) {
        __atomic_add_fetch(&entry.second.pending_dirty_ranges, 1,
                           __ATOMIC_ACQ_REL);
        stale.push_back(&entry.second);
      }
    }
  }
  for (lupine_host_allocation *allocation : stale) {
    lupine_make_mapped_range_fresh(allocation, 0, allocation->size);
    __atomic_sub_fetch(&allocation->pending_dirty_ranges, 1, __ATOMIC_RELEASE);
  }
}

// The lane that services a demand fetch needs a context current on the server,
// and the thread that faults may be one that never made a CUDA call of its own.
// Resolve the allocation's context here, where the routing tables are
// reachable, in the order lupine_route_for_default() resolves a route.
static CUcontext lupine_demand_fetch_context(CUdeviceptr device_ptr) {
  CUcontext context = lupine_context_for_deviceptr(device_ptr);
  if (context == nullptr) {
    context = lupine_current_context_hint();
  }
  if (context == nullptr) {
    context = lupine_default_context_hint_value();
  }
  if (context == nullptr) {
    context = lupine_global_default_context_hint_value();
  }
  return context;
}

static CUresult lupine_copy_dtoh_pageable(conn_t *conn, void *dstHost,
                                          CUdeviceptr srcDevice,
                                          size_t ByteCount, CUstream hStream);
static bool lupine_arm_stale_fetch(void *host,
                                   lupine_host_allocation &allocation,
                                   conn_t *conn, CUcontext fetch_context,
                                   CUstream stream, uintptr_t scope,
                                   pid_t ptds_tid, sig_atomic_t previous) {
  allocation.stale_fetch_conn = conn;
  allocation.stale_fetch_context = fetch_context;
  allocation.stale_fetch_stream = stream;
  allocation.stale_fetch_scope = scope;
  allocation.stale_fetch_ptds_tid = ptds_tid;
  if (allocation.fresh_chunks != nullptr) {
    memset(allocation.fresh_chunks, 0, allocation.fresh_chunk_count);
  }
  allocation.fetch_seq_next = 0;
  allocation.fetch_readahead = 0;
  bool protected_range =
      lupine_protect_host_range(host, allocation.storage_size, PROT_NONE);
  __atomic_store_n(&allocation.device_stale, protected_range ? 1 : previous,
                   __ATOMIC_RELEASE);
  return protected_range;
}

// Marks a tracked allocation stale, so its next touch fetches the server's
// copy. Returns 1 once marked, 0 while another thread is updating stale state,
// and -1 if its pages cannot be protected. Called with
// lupine_host_allocation_mutex() held.
static int lupine_mark_stale_locked(lupine_host_allocation &allocation,
                                    void *host, conn_t *conn,
                                    CUcontext fetch_context,
                                    bool defer_while_fetching = false) {
  sig_atomic_t previous =
      __atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE);
  if (previous == 2) {
    if (!defer_while_fetching) {
      return 0;
    }
    // Never block the push receiver behind a demand fetch whose stream may be
    // waiting on a host callback that in turn awaits this push watermark.
    allocation.stale_rearm_pending = true;
    allocation.stale_rearm_conn = conn;
    allocation.stale_rearm_context = fetch_context;
    if (allocation.io_alias != nullptr) {
      if (allocation.fresh_chunks != nullptr) {
        memset(allocation.fresh_chunks, 0, allocation.fresh_chunk_count);
      }
      (void)lupine_protect_host_range(host, allocation.storage_size, PROT_NONE);
    }
    return 1;
  }
  sig_atomic_t expected = previous;
  if (!__atomic_compare_exchange_n(&allocation.device_stale, &expected, 3,
                                   false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    return 0;
  }
  allocation.stale_rearm_pending = false;
  bool keep_launch_scope = previous == 1 && allocation.stale_fetch_stream !=
                                                lupine_stream_fetch_ready();
  CUstream arm_stream = keep_launch_scope ? allocation.stale_fetch_stream
                                          : lupine_stream_fetch_ready();
  uintptr_t arm_scope =
      keep_launch_scope ? allocation.stale_fetch_scope : RPC_SCOPE_LANE;
  pid_t arm_ptds_tid = keep_launch_scope ? allocation.stale_fetch_ptds_tid : 0;
  return lupine_arm_stale_fetch(host, allocation, conn, fetch_context,
                                arm_stream, arm_scope, arm_ptds_tid, previous)
             ? 1
             : -1;
}

// A pushed copy wrote the server's copy of tracked memory: the client's view
// of it goes stale behind the copy, as after a synchronize, so the next touch
// fetches the new bytes. Only tracked memory is refreshed this way.
static void lupine_mark_pushed_stale(conn_t *conn, void *host) {
  void *base = nullptr;
  CUdeviceptr device_ptr = 0;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(host);
    if (it == lupine_mutable_host_allocations_locked().end() ||
        !it->second.tracking_enabled) {
      return;
    }
    base = it->first;
    device_ptr = it->second.device_ptr;
  }
  // Resolved outside lupine_host_allocation_mutex(), as for a synchronize.
  CUcontext fetch_context = lupine_demand_fetch_context(device_ptr);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
      auto it = lupine_mutable_host_allocations_locked().find(base);
      if (it == lupine_mutable_host_allocations_locked().end() ||
          !it->second.tracking_enabled ||
          __atomic_load_n(&it->second.retiring, __ATOMIC_ACQUIRE) != 0 ||
          lupine_mark_stale_locked(it->second, base, conn, fetch_context,
                                   true) != 0) {
        return;
      }
    }
    sched_yield();
  }
}

extern "C" void lupine_retire_stale_fetch_stream(CUstream stream) {
  uintptr_t bits = reinterpret_cast<uintptr_t>(stream);
  if (stream == nullptr || stream == CU_STREAM_LEGACY ||
      stream == CU_STREAM_PER_THREAD ||
      bits == LUPINE_STREAM_FETCH_READY_BITS ||
      bits == LUPINE_STREAM_FETCH_SYNC_ALL_BITS) {
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (auto &entry : lupine_mutable_host_allocations_locked()) {
    auto &allocation = entry.second;
    if (allocation.stale_fetch_stream == stream) {
      allocation.stale_fetch_stream = lupine_stream_fetch_sync_all();
      allocation.stale_fetch_scope = RPC_SCOPE_ALL;
      allocation.stale_fetch_ptds_tid = 0;
    }
  }
}

extern "C" CUresult lupine_sync_mapped_device_to_host() {
  if (lupine_active_stream_captures.load(std::memory_order_relaxed) != 0) {
    return CUDA_SUCCESS;
  }
  if (!__atomic_exchange_n(&lupine_device_work_pending, 0, __ATOMIC_ACQ_REL)) {
    return CUDA_SUCCESS;
  }

  std::vector<lupine_mapped_host_snapshot> snapshots;
  if (lupine_exposed_or_managed_host_count.load(std::memory_order_acquire) !=
      0) {
    snapshots = lupine_mapped_host_snapshots();
  }
  for (const auto &mapping : snapshots) {
    // Resolved outside lupine_host_allocation_mutex(): the fault handler that
    // consumes it cannot reach the routing table.
    CUcontext fetch_context = lupine_demand_fetch_context(mapping.device_ptr);
    bool invalidated = false;
    bool skip = false;
    lupine_host_allocation *fallback_allocation = nullptr;
    for (int attempt = 0;
         !invalidated && !skip && fallback_allocation == nullptr; ++attempt) {
      {
        std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
        auto it = lupine_mutable_host_allocations_locked().find(mapping.host);
        if (it == lupine_mutable_host_allocations_locked().end()) {
          skip = true;
          break;
        }
        auto &allocation = it->second;
        auto hold_for_fallback = [&] {
          // cuMemFreeHost waits for this reference before unmapping either
          // client view, so the response can still land through the W alias.
          __atomic_add_fetch(&allocation.pending_dirty_ranges, 1,
                             __ATOMIC_ACQ_REL);
          fallback_allocation = &allocation;
        };
        conn_t *conn = lupine_route_remote_conn(
            lupine_route_from_identity(allocation.route_id));
        if (__atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
          skip = true;
          break;
        }
        if (!allocation.tracking_enabled || conn == nullptr) {
          hold_for_fallback();
          break;
        }
        int marked = lupine_mark_stale_locked(allocation, mapping.host, conn,
                                              fetch_context);
        if (marked > 0) {
          invalidated = true;
        } else if (marked < 0) {
          hold_for_fallback();
        }
        if (!invalidated && fallback_allocation == nullptr) {
          if (lupine_in_host_callback) {
            skip = true;
            break;
          }
          if (attempt >= LUPINE_MAPPED_INVALIDATE_MAX_ATTEMPTS) {
            hold_for_fallback();
          }
        }
      }
      if (!invalidated && !skip && fallback_allocation == nullptr) {
        sched_yield();
      }
    }
    if (invalidated || skip) {
      continue;
    }
    CUresult result = CUDA_SUCCESS;
    lupine_route fallback_route =
        lupine_route_from_identity(fallback_allocation->route_id);
    if (lupine_route_is_local(fallback_route)) {
      result = cuMemcpyDtoH_v2(
          static_cast<unsigned char *>(mapping.host) + mapping.data_offset,
          mapping.device_ptr + mapping.data_offset, mapping.data_bytes);
    } else if (!fallback_allocation->managed &&
               fallback_allocation->server_host_ptr != 0) {
      conn_t *conn = lupine_route_remote_conn(fallback_route);
      result =
          lupine_prepare_rpc(conn) < 0
              ? CUDA_ERROR_DEVICE_UNAVAILABLE
              : lupine_rpc_memcpy_htoh_1d(
                    conn, RPC_SCOPE_LANE,
                    static_cast<unsigned char *>(mapping.host) +
                        mapping.data_offset,
                    fallback_allocation->server_host_ptr + mapping.data_offset,
                    mapping.data_bytes);
    } else {
      if (fetch_context != nullptr &&
          lupine_current_context_hint() == nullptr) {
        (void)lupine_set_current_context_on_route(fallback_route,
                                                  fetch_context);
      }
      result = lupine_copy_dtoh_pageable(
          lupine_route_remote_conn(fallback_route),
          static_cast<unsigned char *>(mapping.host) + mapping.data_offset,
          mapping.device_ptr + mapping.data_offset, mapping.data_bytes,
          lupine_in_host_callback ? lupine_stream_fetch_ready()
                                  : CU_STREAM_LEGACY);
    }
    __atomic_sub_fetch(&fallback_allocation->pending_dirty_ranges, 1,
                       __ATOMIC_RELEASE);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  // Pushed DtoH payloads land through the writable alias. Apply those
  // writes after invalidation so their pages remain readable. Staging-backed
  // copies need a later flush; direct pinned copies are already current
  // remotely.
  return lupine_collect_all_pending_host_writes();
}

// A launch may write the managed memory it was passed. Until the next sync the
// host must not keep, and later flush back over the kernel's writes, bytes from
// before the launch, so its next touch fetches behind the launch instead.
extern "C" void lupine_invalidate_launched_managed(CUdeviceptr pointer,
                                                   CUstream stream) {
  if (lupine_active_stream_captures.load(std::memory_order_relaxed) != 0) {
    return;
  }
  CUcontext fetch_context = lupine_demand_fetch_context(pointer);
  uintptr_t call_scope = lupine_stream_scope(stream);
  pid_t call_ptds_tid = stream == CU_STREAM_PER_THREAD ? lupine_gettid() : 0;
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
      auto it =
          lupine_find_host_allocation_locked(reinterpret_cast<void *>(pointer));
      if (it == lupine_mutable_host_allocations_locked().end()) {
        return;
      }
      auto &allocation = it->second;
      conn_t *conn = lupine_route_remote_conn(
          lupine_route_from_identity(allocation.route_id));
      if (!allocation.tracking_enabled || conn == nullptr ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0) {
        return;
      }
      sig_atomic_t previous =
          __atomic_load_n(&allocation.device_stale, __ATOMIC_ACQUIRE);
      if (previous != 2 && previous != 3) {
        sig_atomic_t expected = previous;
        if (__atomic_compare_exchange_n(&allocation.device_stale, &expected, 3,
                                        false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE)) {
          CUstream arm_stream = stream;
          uintptr_t arm_scope = call_scope;
          pid_t arm_ptds_tid = call_ptds_tid;
          if (previous == 1 &&
              allocation.stale_fetch_stream != lupine_stream_fetch_ready() &&
              (allocation.stale_fetch_scope != call_scope ||
               (stream == CU_STREAM_PER_THREAD &&
                allocation.stale_fetch_ptds_tid != call_ptds_tid))) {
            arm_stream = lupine_stream_fetch_sync_all();
            arm_scope = RPC_SCOPE_ALL;
            arm_ptds_tid = 0;
          }
          (void)lupine_arm_stale_fetch(it->first, allocation, conn,
                                       fetch_context, arm_stream, arm_scope,
                                       arm_ptds_tid, previous);
          return;
        }
      }
      if (lupine_in_host_callback) {
        return;
      }
    }
    sched_yield();
  }
}

// The server loads modules with the identity arena active, so a __managed__
// global can get a client view at its own address. The driver packs globals
// from every module into shared pages, each its own allocation, so a view is
// a page run the server reads and writes with the CPU, as it does a host
// allocation's backing; a device copy could not span the neighbors. Kernels
// reach it without a launch param, so it is exposed from the start and
// fetched on first touch. A page an earlier view already covers is refetched
// instead; the lookup's request flushed its host writes. Accepted risk: host
// writes to such a page between a module load and the lookup flush stale bytes
// over the new global's initializer. Views outlive cuModuleUnload.
extern "C" void lupine_adopt_managed_global(lupine_route route,
                                            CUdeviceptr global, size_t bytes) {
  conn_t *conn = lupine_route_remote_conn(route);
  unsigned int managed = 0;
  if (conn == nullptr || bytes == 0 ||
      !lupine_va_contains(conn, global, bytes) ||
      cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            global) != CUDA_SUCCESS ||
      managed == 0) {
    return;
  }
  size_t page = lupine_page_size();
  CUdeviceptr start = global & ~static_cast<CUdeviceptr>(page - 1);
  CUdeviceptr end = lupine_round_up(global + bytes, page);
  if (lupine_host_ptr_is_tracked(start)) {
    lupine_invalidate_launched_managed(start, nullptr);
    start += page;
  }
  if (start < end && lupine_host_ptr_is_tracked(end - page)) {
    lupine_invalidate_launched_managed(end - page, nullptr);
    end -= page;
  }
  // Interior pages belong to this global alone, so a covered one means an
  // earlier lookup of it already made the view.
  void *host = nullptr;
  if (start >= end || lupine_host_ptr_is_tracked(start) ||
      lupine_adopt_host_allocation(conn, &host, reinterpret_cast<void *>(start),
                                   start, end - start, 0,
                                   false) != CUDA_SUCCESS) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocation = lupine_mutable_host_allocations_locked().at(host);
    allocation.managed_global = true;
    if (!lupine_enable_dirty_tracking_locked(host, &allocation) ||
        !allocation.tracking_enabled) {
      return;
    }
    if (!allocation.device_pointer_exposed) {
      allocation.device_pointer_exposed = true;
      if (!allocation.managed) {
        lupine_exposed_or_managed_host_count.fetch_add(
            1, std::memory_order_release);
      }
    }
  }
  lupine_invalidate_launched_managed(start, nullptr);
}

static lupine_host_allocation_map::iterator
lupine_find_host_allocation_locked(void *p) {
  auto &allocations = lupine_mutable_host_allocations_locked();
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

static bool lupine_host_range_registered_locked(uintptr_t base, size_t size) {
  auto &allocations = lupine_mutable_host_allocations_locked();
  uintptr_t end = base + size;
  auto upper = allocations.upper_bound(reinterpret_cast<void *>(base));
  if (upper != allocations.begin()) {
    auto prev = std::prev(upper);
    uintptr_t prev_base = reinterpret_cast<uintptr_t>(prev->first);
    // Owned allocations reserve their padded storage for page protection.
    size_t prev_size = std::max(prev->second.size, prev->second.storage_size);
    if (prev_base == base || prev_size > base - prev_base) {
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
static CUresult
lupine_remote_cuMemHostAlloc(void **remote_host, CUdeviceptr *device_ptr,
                             size_t bytesize, unsigned int flags,
                             lupine_route route, void *requested) {
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
  *remote_host = requested;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemHostAlloc, RPC_SCOPE_NONE) <
          0 ||
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
  if (return_value == CUDA_SUCCESS && requested != nullptr &&
      (*remote_host != requested ||
       *device_ptr != reinterpret_cast<CUdeviceptr>(requested))) {
    // An older peer may ignore the requested address. Fail before submitting
    // work with pointers that would refer to unrelated or unmapped memory.
    (void)lupine_remote_cuMemFreeHost(*remote_host, route);
    *remote_host = nullptr;
    *device_ptr = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return return_value;
}

static CUresult lupine_remote_cuMemFreeHost(void *remote_host,
                                            lupine_route route,
                                            bool drain_only) {
  if (remote_host == nullptr && !drain_only) {
    return CUDA_SUCCESS;
  }
  if (lupine_route_is_local(route)) {
    return drain_only ? CUDA_SUCCESS
                      : lupine_call_real_cuda_fn("cuMemFreeHost", remote_host);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemFreeHost) < 0 ||
      rpc_write(conn, &remote_host, sizeof(remote_host)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

static CUresult
lupine_release_portable_registrations(void *host, CUdeviceptr device_ptr,
                                      const std::vector<int> &routes) {
  CUresult result = CUDA_SUCCESS;
  for (int route_id : routes) {
    lupine_route route = lupine_route_from_identity(route_id);
    CUresult mapping_result =
        lupine_route_is_local(route)
            ? lupine_call_real_cuda_fn("cuMemHostUnregister", host)
            : lupine_remote_cuMemFreeHost(reinterpret_cast<void *>(device_ptr),
                                          route);
    if (result == CUDA_SUCCESS) {
      result = mapping_result;
    }
  }
  return result;
}

template <typename Out>
static CUresult
lupine_remote_host_query(int rpc_op, Out *out, void *remote_host,
                         const unsigned int *extra_flags, lupine_route route) {
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, rpc_op, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, out, sizeof(*out)) < 0 ||
      rpc_write(conn, &remote_host, sizeof(remote_host)) < 0 ||
      (extra_flags != nullptr &&
       rpc_write(conn, extra_flags, sizeof(*extra_flags)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, out, sizeof(*out)) < 0 ||
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
  *device_ptr = 0;
  return lupine_remote_host_query(RPC_cuMemHostGetDevicePointer_v2, device_ptr,
                                  remote_host, &flags, route);
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
  return lupine_remote_host_query(RPC_cuMemHostGetFlags, flags, remote_host,
                                  nullptr, route);
}

static CUresult lupine_register_host(void *p, size_t bytesize,
                                     unsigned int flags,
                                     bool client_to_server_only);

static void lupine_release_owned_host_storage(void *host, void *io_alias,
                                              size_t storage_size,
                                              bool owned_mmap = true) {
  if (host == nullptr) {
    return;
  }
  if (!owned_mmap) {
    free(host);
    return;
  }
  if (io_alias != nullptr) {
    lupine_release_shared_views(host, io_alias, storage_size);
  } else {
    munmap(host, storage_size);
  }
}

static lupine_host_allocation
lupine_make_host_allocation(void *host, size_t bytesize, size_t storage_size,
                            unsigned int flags, CUdeviceptr server_host_ptr,
                            CUdeviceptr device_ptr, lupine_route route) {
  lupine_host_allocation allocation;
  allocation.size = bytesize;
  allocation.storage_size = storage_size;
  allocation.page_size = lupine_page_size();
  allocation.page_count = lupine_round_up(storage_size, allocation.page_size) /
                          allocation.page_size;
  allocation.user_base = reinterpret_cast<uintptr_t>(host);
  allocation.host_base = reinterpret_cast<uintptr_t>(host);
  allocation.user_size = bytesize;
  allocation.flags = flags;
  allocation.server_host_ptr = server_host_ptr;
  allocation.device_ptr = device_ptr;
  allocation.route_id = lupine_route_identity(route);
  return allocation;
}

static CUresult lupine_adopt_host_allocation(conn_t *conn, void **host,
                                             void *remote_host,
                                             CUdeviceptr device_ptr,
                                             size_t bytesize,
                                             unsigned int flags, bool managed) {
  lupine_route route = lupine_remote_route_for_conn(conn);
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
    mapped = lupine_create_shared_views(
        conn, reinterpret_cast<uintptr_t>(remote_host), storage_size, &ptr,
        &io_alias);
  }
  if (!mapped) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation = lupine_make_host_allocation(
        ptr, bytesize, storage_size, flags,
        reinterpret_cast<CUdeviceptr>(remote_host), device_ptr, route);
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.io_alias = io_alias;
    allocation.managed = managed;
    allocation.device_alloc_base = managed ? device_ptr : 0;
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto inserted = allocations.emplace(ptr, std::move(allocation));
    __atomic_add_fetch(&lupine_host_allocation_generation, 1, __ATOMIC_RELEASE);
    if (!inserted.second) {
      lupine_release_owned_host_storage(ptr, io_alias, storage_size);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (managed &&
        !lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      allocations.erase(inserted.first);
      lupine_release_owned_host_storage(ptr, io_alias, storage_size);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    lupine_index_host_allocation_locked(inserted.first, true);
    lupine_note_host_allocation_inserted_locked(inserted.first);
  }
  if (remote_host != nullptr) {
    lupine_note_deviceptr_allocation_route(
        reinterpret_cast<CUdeviceptr>(remote_host), bytesize, route);
    lupine_note_deviceptr_allocation_route(device_ptr, bytesize, route);
  }
  *host = ptr;
  LUPINE_TRACE_LOG("LUPINE adopted host allocation ptr="
                   << ptr << " remote=" << remote_host << " bytes=" << bytesize
                   << " flags=" << flags);
  return CUDA_SUCCESS;
}

static bool lupine_insert_local_host_allocation(void *ptr, size_t bytesize,
                                                unsigned int flags,
                                                lupine_route route, bool owned,
                                                bool client_to_server_only) {
  lupine_host_allocation allocation =
      lupine_make_host_allocation(ptr, bytesize, bytesize, flags,
                                  reinterpret_cast<CUdeviceptr>(ptr), 0, route);
  allocation.owned = owned;
  allocation.local_cuda = true;
  allocation.client_to_server_only = client_to_server_only;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto inserted = lupine_mutable_host_allocations_locked().emplace(
      ptr, std::move(allocation));
  if (!inserted.second) {
    return false;
  }
  lupine_note_host_allocation_inserted_locked(inserted.first);
  __atomic_add_fetch(&lupine_host_allocation_generation, 1, __ATOMIC_RELEASE);
  return true;
}

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
    if (!lupine_insert_local_host_allocation(*pp, bytesize, Flags, route, true,
                                             false)) {
      (void)lupine_call_real_cuda_fn("cuMemFreeHost", *pp);
      return CUDA_ERROR_OUT_OF_MEMORY;
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
  CUresult result =
      lupine_adopt_host_allocation(lupine_route_remote_conn(route), pp,
                                   remote_host, device_ptr, bytesize, Flags);
  if (result != CUDA_SUCCESS) {
    lupine_remote_cuMemFreeHost(remote_host, route);
  }
  return result;
}

extern "C" CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
  return cuMemHostAlloc(pp, bytesize, 0);
}

using lupine_host_free_fn = CUresult (*)(conn_t *, void *);

static lupine_host_allocation *
lupine_begin_retiring_allocation_locked(void *host,
                                        lupine_host_allocation &allocation) {
  __atomic_store_n(&allocation.retiring, 1, __ATOMIC_RELEASE);
  lupine_pointer_attribute_cache_clear();
  lupine_disable_dirty_tracking(host, allocation);
  return &allocation;
}

static bool
lupine_extract_retired_allocation_locked(void *host,
                                         const lupine_host_allocation *expected,
                                         lupine_host_allocation *out) {
  auto &allocations = lupine_mutable_host_allocations_locked();
  auto it = allocations.find(host);
  if (it == allocations.end() || &it->second != expected) {
    return false;
  }
  lupine_index_host_allocation_locked(it, false);
  *out = std::move(it->second);
  allocations.erase(it);
  lupine_recompute_host_allocation_summary_locked();
  return true;
}

static CUresult
lupine_retire_and_release_host_allocation(void *p, bool owned_allocation,
                                          lupine_host_free_fn release) {
  // cuMemFreeHost has free(NULL) semantics, which the headers do not document.
  if (p == nullptr) {
    return owned_allocation ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  }

  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  std::unique_lock<std::mutex> mapping_lock(
      lupine_portable_registration_mutex());

  lupine_host_allocation *retiring_allocation = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = owned_allocation ? allocations.find(p)
                               : lupine_find_host_allocation_locked(p);
    if (owned_allocation) {
      if (it == allocations.end() || !it->second.owned || it->second.managed ||
          it->second.managed_global) {
        return CUDA_ERROR_INVALID_VALUE;
      }
    } else {
      if (it == allocations.end() || it->second.client_to_server_only) {
        return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
      }
      if (it->second.owned ||
          it->second.user_base != reinterpret_cast<uintptr_t>(p)) {
        return CUDA_ERROR_INVALID_VALUE;
      }
    }
    retiring_allocation =
        lupine_begin_retiring_allocation_locked(p, it->second);
  }
  flush_result = lupine_drain_retiring_dirty_ranges(retiring_allocation);
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  lupine_host_allocation allocation;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (!lupine_extract_retired_allocation_locked(p, retiring_allocation,
                                                  &allocation)) {
      return owned_allocation ? CUDA_ERROR_INVALID_VALUE
                              : CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
  }
  mapping_lock.unlock();
  CUresult result = lupine_release_portable_registrations(
      p, allocation.device_ptr, allocation.portable_routes);
  for (int route_id : allocation.dtoh_routes) {
    if (route_id == allocation.route_id ||
        std::find(allocation.portable_routes.begin(),
                  allocation.portable_routes.end(),
                  route_id) != allocation.portable_routes.end()) {
      continue;
    }
    CUresult drain_result = lupine_remote_cuMemFreeHost(
        nullptr, lupine_route_from_identity(route_id), true);
    if (result == CUDA_SUCCESS) {
      result = drain_result;
    }
  }
  if (allocation.local_cuda) {
    CUresult primary_result = release(nullptr, p);
    if (primary_result == CUDA_SUCCESS) {
      if (allocation.owned) {
        lupine_forget_deviceptr_owner(reinterpret_cast<CUdeviceptr>(p));
      }
      if (allocation.device_ptr != 0) {
        lupine_forget_deviceptr_owner(allocation.device_ptr);
      }
    }
    return result == CUDA_SUCCESS ? primary_result : result;
  }
  // The server's answer waits for the device-to-host deliveries it numbered
  // before freeing, so the views are still mapped when they land.
  if (allocation.server_host_ptr != 0) {
    lupine_route route = lupine_route_for_deviceptr(allocation.server_host_ptr);
    CUresult primary_result =
        release(lupine_route_remote_conn(route),
                reinterpret_cast<void *>(allocation.server_host_ptr));
    if (primary_result == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(allocation.server_host_ptr);
      if (allocation.device_ptr != 0) {
        lupine_forget_deviceptr_owner(allocation.device_ptr);
      }
    }
    if (result == CUDA_SUCCESS) {
      result = primary_result;
    }
  }
  if (allocation.owned) {
    lupine_release_owned_host_storage(
        p, allocation.io_alias, allocation.storage_size, allocation.owned_mmap);
  }
  return result;
}

extern "C" CUresult cuMemFreeHost(void *p) {
  return lupine_retire_and_release_host_allocation(
      p, true, [](conn_t *conn, void *remote_host) {
        lupine_route route = conn == nullptr
                                 ? lupine_route{LUPINE_ROUTE_LOCAL, nullptr}
                                 : lupine_remote_route_for_conn(conn);
        return lupine_remote_cuMemFreeHost(remote_host, route);
      });
}

extern "C" CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p,
                                                 unsigned int Flags) {
  if (pdptr == nullptr || p == nullptr || Flags != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto publish_device_ptr = [&](lupine_host_allocation_map::iterator it) {
    lupine_expose_host_device_pointer(it->first, it->second);
    *pdptr = it->second.device_ptr + (reinterpret_cast<uintptr_t>(p) -
                                      reinterpret_cast<uintptr_t>(it->first));
    return CUDA_SUCCESS;
  };
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
        return publish_device_ptr(it);
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
    // The query may be in an unregistered gap on a shared page. Retrying
    // that query would never find an allocation containing it.
    return result == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
               ? CUDA_ERROR_INVALID_VALUE
               : (result == CUDA_SUCCESS
                      ? cuMemHostGetDevicePointer_v2(pdptr, p, Flags)
                      : result);
  }

  CUdeviceptr new_device_ptr = 0;
  lupine_route route =
      lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(remote_host));
  CUresult alloc_result =
      local_cuda
          ? lupine_call_real_cuda_fn("cuMemHostGetDevicePointer_v2",
                                     &new_device_ptr, allocation_host, Flags)
          : lupine_remote_cuMemHostGetDevicePointer(&new_device_ptr,
                                                    remote_host, Flags, route);
  if (alloc_result != CUDA_SUCCESS) {
    return alloc_result;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(p);
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (it->second.device_ptr == 0) {
    it->second.device_ptr = new_device_ptr;
    lupine_note_deviceptr_allocation_route(new_device_ptr, allocation_size,
                                           route);
    if (!lupine_enable_dirty_tracking_locked(allocation_host, &it->second)) {
      it->second.device_ptr = 0;
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    lupine_index_host_allocation_locked(it, true);
  }
  return publish_device_ptr(it);
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

  // CUDA permits disjoint registrations within the same page. Keep each
  // caller's byte range independent, including its remote backing and lifetime.
  uintptr_t base = reinterpret_cast<uintptr_t>(p);
  if (bytesize > UINTPTR_MAX - base) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (lupine_host_range_registered_locked(base, bytesize)) {
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
    if (!lupine_insert_local_host_allocation(p, bytesize, Flags, route, false,
                                             client_to_server_only)) {
      (void)lupine_call_real_cuda_fn("cuMemHostUnregister", p);
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    }
    return CUDA_SUCCESS;
  }

  // UVA registrations can be queried for a GPU alias even without DEVICEMAP.
  // Keep the backing ready; expose it only when requested.
  unsigned int host_flags = CU_MEMHOSTALLOC_DEVICEMAP;
  if ((Flags & CU_MEMHOSTREGISTER_PORTABLE) != 0) {
    host_flags |= CU_MEMHOSTALLOC_PORTABLE;
  }
  void *server_host = nullptr;
  CUdeviceptr device_ptr = 0;
  CUresult result = lupine_remote_cuMemHostAlloc(&server_host, &device_ptr,
                                                 bytesize, host_flags, route);
  if (result != CUDA_SUCCESS) {
    if (server_host != nullptr) {
      (void)lupine_remote_cuMemFreeHost(server_host, route);
    }
    return result;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto &allocations = lupine_mutable_host_allocations_locked();
  if (lupine_host_range_registered_locked(base, bytesize)) {
    if (server_host != nullptr) {
      (void)lupine_remote_cuMemFreeHost(server_host, route);
    }
    return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  }
  lupine_host_allocation allocation = lupine_make_host_allocation(
      p, bytesize, bytesize, Flags, reinterpret_cast<CUdeviceptr>(server_host),
      device_ptr, route);
  allocation.client_to_server_only = client_to_server_only;
  auto inserted_it = allocations.emplace(p, std::move(allocation)).first;
  lupine_index_host_allocation_locked(inserted_it, true);
  lupine_note_host_allocation_inserted_locked(inserted_it);
  __atomic_add_fetch(&lupine_host_allocation_generation, 1, __ATOMIC_RELEASE);
  if (server_host != nullptr) {
    lupine_note_deviceptr_allocation_route(
        reinterpret_cast<CUdeviceptr>(server_host), bytesize, route);
    lupine_note_deviceptr_allocation_route(device_ptr, bytesize, route);
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
  return lupine_retire_and_release_host_allocation(
      p, false, [](conn_t *conn, void *host) {
        if (conn == nullptr) {
          return lupine_call_real_cuda_fn("cuMemHostUnregister", host);
        }
        return lupine_remote_cuMemFreeHost(host,
                                           lupine_remote_route_for_conn(conn));
      });
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

  CUdeviceptr device_alloc_base = 0;
  size_t backing_size = std::max(bytesize, LUPINE_MANAGED_ALLOCATION_MIN_BYTES);
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAllocManaged, RPC_SCOPE_NONE) <
          0 ||
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
  lupine_note_deviceptr_allocation_route(device_alloc_base, backing_size,
                                         route);
  void *ptr = nullptr;
  result = lupine_adopt_host_allocation(
      conn, &ptr, reinterpret_cast<void *>(device_alloc_base),
      device_alloc_base, bytesize, flags, true);
  if (result != CUDA_SUCCESS) {
    cuMemFree_v2(device_alloc_base);
    return result;
  }

  *dptr = reinterpret_cast<CUdeviceptr>(ptr);
  lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return CUDA_SUCCESS;
}

static CUresult lupine_release_backend_device_allocation(conn_t *conn,
                                                         CUdeviceptr ptr) {
  if (conn == nullptr) {
    return lupine_call_real_cuda_fn("cuMemFree_v2", ptr);
  }
  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMemFree_v2) < 0 ||
      rpc_write(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return result;
}

extern "C" CUresult cuMemFree_v2(CUdeviceptr dptr) {
  CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }

  void *host = reinterpret_cast<void *>(dptr);
  lupine_host_allocation allocation;
  lupine_host_allocation *retiring_allocation = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_find_host_allocation_locked(host);
    if (it != lupine_mutable_host_allocations_locked().end() &&
        host == it->first && it->second.managed) {
      retiring_allocation =
          lupine_begin_retiring_allocation_locked(it->first, it->second);
    }
  }
  if (retiring_allocation == nullptr) {
    lupine_route route = lupine_route_for_deviceptr(dptr);
    conn_t *conn = lupine_route_is_local(route)
                       ? nullptr
                       : lupine_route_remote_conn(route);
    CUresult result = lupine_release_backend_device_allocation(conn, dptr);
    if (result == CUDA_SUCCESS) {
      lupine_forget_deviceptr_owner(dptr);
    }
    return result;
  }

  flush_result = lupine_drain_retiring_dirty_ranges(retiring_allocation);
  if (flush_result != CUDA_SUCCESS) {
    return flush_result;
  }
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (!lupine_extract_retired_allocation_locked(host, retiring_allocation,
                                                  &allocation)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  CUresult result = CUDA_SUCCESS;
  if (allocation.device_ptr != 0) {
    CUdeviceptr free_ptr = allocation.device_alloc_base != 0
                               ? allocation.device_alloc_base
                               : allocation.device_ptr;
    result = lupine_release_backend_device_allocation(
        lupine_route_remote_conn(
            lupine_route_from_identity(allocation.route_id)),
        free_ptr);
  }
  lupine_forget_deviceptr_owner(reinterpret_cast<CUdeviceptr>(host));
  lupine_forget_deviceptr_owner(allocation.server_host_ptr);
  lupine_forget_deviceptr_owner(allocation.device_ptr);
  if (allocation.owned) {
    lupine_release_owned_host_storage(host, allocation.io_alias,
                                      allocation.storage_size,
                                      allocation.owned_mmap);
  }
  return result;
}

lupine_pointer_query_context lupine_resolve_pointer_query(CUdeviceptr ptr) {
  lupine_pointer_query_context ctx;
  ctx.orig_ptr = ptr;
  ctx.query_ptr = ptr;
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return ctx;
  }
  const auto &allocation = it->second;
  ctx.tracked_host = true;
  if (allocation.managed || allocation.managed_global) {
    ctx.managed_alias = true;
    ctx.managed_alias_base = allocation.device_ptr;
  } else if (allocation.server_host_ptr != 0 && !allocation.local_cuda) {
    ctx.remote_host_alias = true;
    uintptr_t client_base = reinterpret_cast<uintptr_t>(it->first);
    uintptr_t client_addr = reinterpret_cast<uintptr_t>(host);
    ctx.query_ptr = allocation.server_host_ptr + (client_addr - client_base);
    ctx.remote_host_base = allocation.server_host_ptr;
    ctx.remote_device_base = allocation.device_ptr;
  }
  if ((ctx.managed_alias || ctx.remote_host_alias) && !allocation.local_cuda &&
      !allocation.managed_global && !allocation.client_to_server_only) {
    uintptr_t address = reinterpret_cast<uintptr_t>(host);
    if (address >= allocation.user_base &&
        address < allocation.user_base + allocation.user_size) {
      ctx.has_client_range = true;
      ctx.client_range_start = static_cast<CUdeviceptr>(allocation.user_base);
      ctx.client_range_size = allocation.user_size;
    }
  }
  return ctx;
}

void lupine_fixup_pointer_attribute_value(
    const lupine_pointer_query_context &ctx, CUpointer_attribute attribute,
    void *data) {
  auto write_val = [data](const auto &v) { memcpy(data, &v, sizeof(v)); };
  if ((ctx.managed_alias || ctx.remote_host_alias) &&
      attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
    write_val(reinterpret_cast<void *>(ctx.orig_ptr));
  } else if (ctx.managed_alias &&
             attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
    write_val(ctx.query_ptr);
  } else if (ctx.remote_host_alias &&
             attribute == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
    write_val(ctx.remote_device_base + (ctx.query_ptr - ctx.remote_host_base));
  } else if (ctx.managed_alias &&
             attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
    write_val(int{1});
  } else if (ctx.has_client_range &&
             attribute == CU_POINTER_ATTRIBUTE_RANGE_START_ADDR) {
    write_val(ctx.client_range_start);
  } else if (ctx.has_client_range &&
             attribute == CU_POINTER_ATTRIBUTE_RANGE_SIZE) {
    write_val(ctx.client_range_size);
  }
}

extern "C" CUdeviceptr
lupine_resolve_pointer_attribute_target(CUdeviceptr ptr) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      it->second.managed || it->second.managed_global ||
      it->second.server_host_ptr == 0 || it->second.local_cuda) {
    return ptr;
  }
  uintptr_t client_base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t client_addr = reinterpret_cast<uintptr_t>(host);
  return it->second.server_host_ptr + (client_addr - client_base);
}

// Attribute values of a live mapped-host allocation are immutable, and
// PyTorch's pin-memory path queries the same pinned buffers on every batch,
// so the server's answers are cached per (pointer, attribute list). The cache
// is cleared whenever any mapped allocation retires, since a later allocation
// may reuse the address.
static uint64_t
lupine_pointer_attribute_cache_key(CUdeviceptr ptr,
                                   const CUpointer_attribute *attributes,
                                   unsigned int count) {
  uint64_t key = static_cast<uint64_t>(ptr);
  for (unsigned int i = 0; i < count; ++i) {
    key = (key ^ static_cast<uint64_t>(attributes[i])) * 1099511628211ULL;
  }
  return key ^ count;
}

static libcuckoo::cuckoohash_map<uint64_t, std::vector<unsigned char>> &
lupine_pointer_attribute_cache() {
  static auto *cache =
      new libcuckoo::cuckoohash_map<uint64_t, std::vector<unsigned char>>();
  return *cache;
}

static void lupine_pointer_attribute_cache_clear() {
  lupine_pointer_attribute_cache().clear();
}

extern "C" CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                           CUpointer_attribute *attributes,
                                           void **data, CUdeviceptr ptr) {
  if (numAttributes == 0 || attributes == nullptr || data == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t total_size = 0;
  for (unsigned int i = 0; i < numAttributes; ++i) {
    size_t value_size = 0;
    if (data[i] == nullptr ||
        !lupine_pointer_attribute_size(attributes[i], &value_size)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    total_size += value_size;
  }

  lupine_pointer_query_context query = lupine_resolve_pointer_query(ptr);
  lupine_route route = lupine_route_for_deviceptr(query.query_ptr);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuPointerGetAttributes", numAttributes,
                                    attributes, data, query.query_ptr);
  }

  // A pointer in neither the host-allocation registry nor the device-pointer
  // tables is unregistered host memory, for which the driver's answer is
  // deterministic (probed on a native driver: CUDA_SUCCESS; both pointer
  // aliases echo the query; type/managed/mapped are zero; ordinal is -1;
  // context and buffer id are zero; the range attributes are left untouched).
  // PyTorch's pin_memory path asks this about every unpinned source tensor,
  // so answering locally removes a per-batch round trip. Attributes outside
  // the probed set fall through to the server.
  if (!query.managed_alias && !query.tracked_host &&
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
    }
    if (replicable) {
      for (unsigned int i = 0; i < numAttributes; ++i) {
        size_t value_size = 0;
        lupine_pointer_attribute_size(attributes[i], &value_size);
        switch (attributes[i]) {
        case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
        case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
          memcpy(data[i], &ptr, value_size);
          break;
        }
        case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
          int ordinal = -1;
          memcpy(data[i], &ordinal, value_size);
          break;
        }
        case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
        case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
          break;
        default:
          memset(data[i], 0, value_size);
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
  CUdeviceptr cache_ptr =
      query.managed_alias ? query.managed_alias_base : query.remote_host_base;
  bool cacheable = query.managed_alias || query.remote_host_alias;
  uint64_t cache_key = cacheable ? lupine_pointer_attribute_cache_key(
                                       cache_ptr, attributes, numAttributes)
                                 : 0;
  bool cache_hit = false;
  if (cacheable) {
    lupine_pointer_attribute_cache().find_fn(
        cache_key, [&](const std::vector<unsigned char> &cached) {
          if (cached.size() != total_size) {
            return;
          }
          size_t offset = 0;
          for (unsigned int i = 0; i < numAttributes; ++i) {
            size_t value_size = 0;
            lupine_pointer_attribute_size(attributes[i], &value_size);
            memcpy(data[i], cached.data() + offset, value_size);
            lupine_fixup_pointer_attribute_value(query, attributes[i], data[i]);
            offset += value_size;
          }
          cache_hit = true;
        });
    if (cache_hit) {
      return CUDA_SUCCESS;
    }
  }

  CUpointer_attribute inline_attrs[16];
  std::unique_ptr<CUpointer_attribute[]> heap_attrs;
  const CUpointer_attribute *rpc_attributes = attributes;
  if (query.managed_alias) {
    CUpointer_attribute *rewritten = inline_attrs;
    if (numAttributes > std::size(inline_attrs)) {
      heap_attrs.reset(new (std::nothrow) CUpointer_attribute[numAttributes]);
      if (!heap_attrs) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      rewritten = heap_attrs.get();
    }
    for (unsigned int i = 0; i < numAttributes; ++i) {
      rewritten[i] = attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER
                         ? CU_POINTER_ATTRIBUTE_DEVICE_POINTER
                         : attributes[i];
    }
    rpc_attributes = rewritten;
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuPointerGetAttributes,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(numAttributes)) < 0 ||
      rpc_write(conn, rpc_attributes,
                numAttributes * sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &query.query_ptr, sizeof(query.query_ptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  if (return_value == CUDA_SUCCESS) {
    std::vector<unsigned char> cached;
    if (cacheable) {
      cached.resize(total_size);
    }
    size_t offset = 0;
    for (unsigned int i = 0; i < numAttributes; ++i) {
      size_t value_size = 0;
      lupine_pointer_attribute_size(attributes[i], &value_size);
      if (rpc_read(conn, data[i], value_size) < 0) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      if (cacheable) {
        memcpy(cached.data() + offset, data[i], value_size);
      }
      lupine_fixup_pointer_attribute_value(query, attributes[i], data[i]);
      offset += value_size;
    }
    if (cacheable) {
      lupine_pointer_attribute_cache().insert_or_assign(cache_key,
                                                        std::move(cached));
    }
  }
  if (rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

// Returns 1 when the address is not a cross-route device source, 0 after
// writing its complete response, and -1 on a transport or source-copy error.
extern "C" int lupine_write_cross_route_device_source(conn_t *destination_conn,
                                                      int request_id,
                                                      CUdeviceptr source,
                                                      size_t bytes) {
  if (destination_conn == nullptr || source == 0 ||
      !lupine_deviceptr_is_tracked(source) ||
      lupine_copy_pointer_is_host(source)) {
    return 1;
  }
  lupine_route source_route = lupine_route_for_deviceptr(source);
  lupine_route destination_route =
      lupine_remote_route_for_conn(destination_conn);
  if (lupine_route_is_local(source_route) ||
      lupine_routes_share_server(source_route, destination_route)) {
    return 1;
  }
  CUcontext source_context = lupine_context_for_deviceptr(source);
  if (source_context == nullptr) {
    return -1;
  }

  struct source_cursor_state {
    conn_t *conn = nullptr;
    int request_id = -1;
    size_t remaining = 0;
    size_t storage_size = 0;
    std::unique_ptr<unsigned char[]> storage;
  } source_cursor;
  source_cursor.conn = lupine_route_remote_conn(source_route);
  source_cursor.remaining = bytes;
  source_cursor.storage_size =
      std::min(bytes, static_cast<size_t>(LUPINE_RPC_TRANSFER_CHUNK_BYTES));
  if (source_cursor.storage_size != 0) {
    source_cursor.storage.reset(
        new (std::nothrow) unsigned char[source_cursor.storage_size]);
    if (!source_cursor.storage) {
      return -1;
    }
  }
  if (lupine_set_current_context_on_route(source_route, source_context) !=
      CUDA_SUCCESS) {
    return -1;
  }

  // Start the source read before opening the destination response builder.
  // Refill callbacks run while that builder owns the thread's RPC writer, so
  // calling cuMemcpyDtoH_v2 from a refill would try to open a second writer
  // and fail before reaching the source server. One streamed DtoH request lets
  // the callback consume source responses without nesting RPC writers.
  CUstream source_stream =
      lupine_in_host_callback ? lupine_stream_fetch_ready() : CU_STREAM_LEGACY;
  if (bytes != 0 &&
      (lupine_prepare_rpc(source_cursor.conn) < 0 ||
       lupine_rpc_start_on_stream(source_cursor.conn, RPC_cuMemcpyDtoH_v2,
                                  source_stream) < 0 ||
       rpc_write(source_cursor.conn, &source, sizeof(source)) < 0 ||
       rpc_write(source_cursor.conn, &bytes, sizeof(bytes)) < 0 ||
       rpc_write(source_cursor.conn, &source_stream, sizeof(source_stream)) <
           0 ||
       (rpc_expect_answer(source_cursor.conn),
        source_cursor.request_id = rpc_write_end(source_cursor.conn)) < 0)) {
    return -1;
  }

  auto refill_source = [](void *opaque, rpc_write_cursor *cursor) -> int {
    auto *source = static_cast<source_cursor_state *>(opaque);
    if (source == nullptr || cursor == nullptr) {
      return -1;
    }
    if (source->remaining == 0) {
      return 0;
    }

    size_t chunk = std::min(source->remaining, source->storage_size);
    CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (rpc_read_start(source->conn, source->request_id) < 0) {
      source->remaining = 0;
      LUPINE_LOG_ERROR("Cross-route DtoD source transport failed");
      return -1;
    }
    // The readback's delivery number is for the source connection's pushed
    // copies, which this copy between servers does not read.
    uint64_t delivered = 0;
    bool read_failed =
        rpc_read(source->conn, &delivered, sizeof(delivered)) < 0 ||
        rpc_read(source->conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_read(source->conn, source->storage.get(), chunk) < 0);
    if (rpc_read_end(source->conn) < 0 || read_failed) {
      source->remaining = 0;
      LUPINE_LOG_ERROR("Cross-route DtoD source transport failed");
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      source->remaining = 0;
      LUPINE_LOG_ERROR("Cross-route DtoD source read failed: " << result);
      return -1;
    }
    cursor->data = source->storage.get();
    cursor->size = chunk;
    source->remaining -= chunk;
    return 1;
  };
  rpc_write_cursor cursor(refill_source, &source_cursor);
  if (rpc_write_start_response(destination_conn, request_id) < 0 ||
      rpc_write_cursors(destination_conn, &cursor, 1) < 0 ||
      rpc_write_end(destination_conn) < 0) {
    rpc_write_cursor discard;
    while (source_cursor.remaining != 0 &&
           refill_source(&source_cursor, &discard) > 0) {
    }
    return -1;
  }
  return 0;
}

// The caller's lane requests the copy and holds it in FIFO order while each
// bulk connection asks the server for chunks; every chunk carries its offset,
// so arrival order across connections is irrelevant.
static CUresult lupine_bulk_pull(conn_t *conn, lupine_bulk_lanes *lanes,
                                 void *destination, CUdeviceptr source,
                                 size_t bytes, CUstream stream) {
  static std::atomic<uint64_t> next_copy_id{1};
  if (pthread_mutex_lock(&lanes->mutex) != 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  uint64_t copy_id = next_copy_id.fetch_add(1);
  uint32_t readers = lanes->count;
  int request_id = -1;
  if (lupine_rpc_start_on_stream(conn, LUPINE_RPC_lupineMemcpyDtoHBulk,
                                 stream) < 0 ||
      rpc_write(conn, &copy_id, sizeof(copy_id)) < 0 ||
      rpc_write(conn, &source, sizeof(source)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_write(conn, &readers, sizeof(readers)) < 0 ||
      (rpc_expect_answer(conn), request_id = rpc_write_end(conn)) < 0) {
    pthread_mutex_unlock(&lanes->mutex);
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  void *alias = rpc_host_allocation_alias(conn, destination, bytes);
  std::atomic<bool> received{true};
  auto *data =
      static_cast<unsigned char *>(alias != nullptr ? alias : destination);
  auto pull = [&](unsigned int lane) {
    if (!lupine_bulk_read_stream(lanes->conn[lane], lanes->stream[lane],
                                 copy_id, data)) {
      received = false;
    }
  };
  std::vector<std::thread> pullers;
  for (unsigned int lane = 1; lane < lanes->count; ++lane) {
    pullers.emplace_back(pull, lane);
  }
  pull(0);
  for (auto &puller : pullers) {
    puller.join();
  }
  if (!received) {
    lanes->failed = true;
  } else if (alias != nullptr &&
             rpc_note_host_allocation_write(conn, destination, bytes) < 0) {
    received = false;
  }
  pthread_mutex_unlock(&lanes->mutex);

  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  // The copy waited for its stream, so pushed copies that stream ran land
  // first.
  uint64_t delivered = 0;
  if (rpc_read_start(conn, request_id) < 0 ||
      rpc_read(conn, &delivered, sizeof(delivered)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0 || !received ||
      lupine_wait_dtoh(conn, delivered) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

// Copies into pageable client memory, ordered behind hStream's prior work.
// Chunks are staged in a bounded pool on the server, so a copy of any size
// costs the same fixed staging there.
static CUresult lupine_copy_dtoh_pageable(conn_t *conn, void *dstHost,
                                          CUdeviceptr srcDevice,
                                          size_t ByteCount, CUstream hStream) {
  if (lupine_prepare_rpc(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  uintptr_t stream_bits = reinterpret_cast<uintptr_t>(hStream);
  bool fetch_sentinel = stream_bits == LUPINE_STREAM_FETCH_READY_BITS ||
                        stream_bits == LUPINE_STREAM_FETCH_SYNC_ALL_BITS;
  lupine_bulk_lanes *lanes = !fetch_sentinel && !lupine_in_host_callback &&
                                     ByteCount >= LUPINE_BULK_COPY_MIN_BYTES
                                 ? lupine_client_transport_bulk_lanes(conn)
                                 : nullptr;
  if (lanes != nullptr) {
    return lupine_bulk_pull(conn, lanes, dstHost, srcDevice, ByteCount,
                            hStream);
  }
  if (lupine_rpc_start_on_stream(conn, RPC_cuMemcpyDtoH_v2, hStream) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return lupine_read_dtoh_chunks(
      conn, (rpc_expect_answer(conn), rpc_write_end(conn)),
      static_cast<unsigned char *>(dstHost), ByteCount);
}

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                    size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(srcDevice);
  if (lupine_route_is_local(route)) {
    return lupine_call_real_cuda_fn("cuMemcpyDtoH_v2", dstHost, srcDevice,
                                    ByteCount);
  }
  return lupine_copy_dtoh_pageable(lupine_route_remote_conn(route), dstHost,
                                   srcDevice, ByteCount, CU_STREAM_LEGACY);
}

// Bytes of the source that travel in the request: the whole copy for a
// pageable client source, nothing otherwise. A page-locked (tracked) source
// may still be written by work queued ahead of the copy, and CUDA reads it
// when the stream reaches the copy, so the server keeps pulling it at that
// point; a pageable source is staged at issue natively too, so sending it now
// is exact. Server-authoritative memory and device memory on another route
// never leave the client here either.
static uint64_t lupine_htod_pushed_bytes(bool is_server_authoritative,
                                         const void *source, size_t bytes) {
  if (is_server_authoritative || bytes == 0) {
    return 0;
  }
  uintptr_t start = reinterpret_cast<uintptr_t>(source);
  if (start > UINTPTR_MAX - bytes) {
    return 0;
  }
  if (lupine_range_in_tracked_host_bounds(start, bytes)) {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (lupine_host_range_registered_locked(start, bytes)) {
      return 0;
    }
  }
  CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(source);
  return !lupine_deviceptr_is_tracked(ptr) &&
                 lupine_is_client_mapped_address(ptr)
             ? bytes
             : 0;
}

struct lupine_htod_host_info {
  CUdeviceptr server_source = 0;
  bool is_server_authoritative = false;
  bool source_registered = false;
  bool dst_tracked_host = false;
};

static lupine_htod_host_info
lupine_query_htod_host_state(CUdeviceptr destination, const void *source,
                             size_t bytes, int route_id) {
  lupine_htod_host_info info;
  uintptr_t lo = 0, hi = 0;
  if (!lupine_tracked_host_bounds(&lo, &hi)) {
    return info;
  }
  uintptr_t dst_addr = static_cast<uintptr_t>(destination);
  bool dst_in_bounds = destination != 0 && dst_addr >= lo && dst_addr < hi;
  uintptr_t src_addr = reinterpret_cast<uintptr_t>(source);
  bool src_translate_candidate =
      source != nullptr && src_addr >= lo && src_addr < hi;
  bool src_reg_candidate = source != nullptr && bytes != 0 &&
                           src_addr <= UINTPTR_MAX - bytes && src_addr < hi &&
                           src_addr + bytes > lo;
  if (!dst_in_bounds && !src_translate_candidate && !src_reg_candidate) {
    return info;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  if (dst_in_bounds) {
    (void)lupine_expose_mapped_host_pointer_locked(destination, nullptr);
    info.dst_tracked_host = lupine_find_host_allocation_locked(
                                reinterpret_cast<void *>(destination)) !=
                            lupine_mutable_host_allocations_locked().end();
  }
  if (src_translate_candidate) {
    info.is_server_authoritative = lupine_translate_client_host_range_locked(
        &info.server_source, source, bytes, route_id);
  }
  if (!info.is_server_authoritative && src_reg_candidate) {
    info.source_registered =
        lupine_host_range_registered_locked(src_addr, bytes);
  }
  return info;
}

struct lupine_bulk_chunk_header {
  int request_id;
  int op;
  // No ticket: the copy RPC on the caller's lane orders the chunks.
  uint64_t ticket;
  uint64_t copy_id;
  uint64_t total;
  uint64_t offset;
  uint64_t bytes;
};

// The chunks only stage on the server, in whatever order the connections
// deliver them. The copy into `destination` (device memory, or with to_host
// the server's copy of a host allocation) is requested on the caller's lane
// once every chunk has been queued, so it stays FIFO behind the caller's
// earlier work on that lane and runs once the last chunk has landed. One
// thread per lane takes chunks from a shared counter, so a connection whose
// window has collapsed takes fewer of them instead of pacing the whole copy.
static CUresult lupine_bulk_push(conn_t *conn, lupine_bulk_lanes *lanes,
                                 uint64_t destination, const void *source,
                                 size_t bytes, bool to_host) {
  static std::atomic<uint64_t> next_copy_id{1};
  if (pthread_mutex_lock(&lanes->mutex) != 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  uint64_t copy_id = next_copy_id.fetch_add(1);
  size_t chunks = (bytes + LUPINE_RPC_TRANSFER_CHUNK_BYTES - 1) /
                  LUPINE_RPC_TRANSFER_CHUNK_BYTES;
  std::atomic<size_t> next_chunk{0};
  std::atomic<bool> queued{true};
  auto feed = [&](unsigned int lane) {
    const auto *data = static_cast<const unsigned char *>(source);
    for (size_t chunk = next_chunk.fetch_add(1); chunk < chunks && queued;
         chunk = next_chunk.fetch_add(1)) {
      size_t offset = chunk * LUPINE_RPC_TRANSFER_CHUNK_BYTES;
      size_t chunk_bytes = std::min(
          static_cast<size_t>(LUPINE_RPC_TRANSFER_CHUNK_BYTES), bytes - offset);
      lupine_bulk_chunk_header header = {2,
                                         LUPINE_RPC_lupineBulkChunk,
                                         ~uint64_t{0},
                                         copy_id,
                                         bytes,
                                         offset,
                                         chunk_bytes};
      rpc_write_cursor cursors[] = {{&header, sizeof(header)},
                                    {data + offset, chunk_bytes}};
      if (rpc_http2_write_stream(lanes->conn[lane], lanes->stream[lane],
                                 cursors, 2) < 0) {
        queued = false;
      }
    }
  };
  std::vector<std::thread> feeders;
  for (unsigned int lane = 1; lane < lanes->count; ++lane) {
    feeders.emplace_back(feed, lane);
  }
  feed(0);
  for (auto &feeder : feeders) {
    feeder.join();
  }
  if (!queued) {
    lanes->failed = true;
  }
  pthread_mutex_unlock(&lanes->mutex);

  uint8_t host = to_host ? 1 : 0;
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (!queued ||
      (to_host
           ? rpc_write_start_request_on(conn, LUPINE_RPC_lupineMemcpyHtoDBulk,
                                        RPC_SCOPE_HOST_WRITES)
           : lupine_rpc_start_on_stream(conn, LUPINE_RPC_lupineMemcpyHtoDBulk,
                                        nullptr)) < 0 ||
      rpc_write(conn, &copy_id, sizeof(copy_id)) < 0 ||
      rpc_write(conn, &destination, sizeof(destination)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &host, sizeof(host)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

struct lupine_htod_wire_state {
  lupine_htod_host_info host;
  bool is_server_authoritative = false;
  const void *wire_source = nullptr;
  uint64_t pushed_bytes = 0;
};

static lupine_htod_wire_state
lupine_prepare_htod_wire_state(const lupine_htod_routing_info &routing,
                               CUdeviceptr dstDevice, const void *srcHost,
                               size_t ByteCount) {
  lupine_htod_wire_state state;
  state.host = lupine_query_htod_host_state(
      dstDevice, srcHost, ByteCount, lupine_route_identity(routing.dst_route));
  state.is_server_authoritative = state.host.is_server_authoritative;
  state.wire_source =
      state.is_server_authoritative
          ? reinterpret_cast<const void *>(state.host.server_source)
          : srcHost;
  uintptr_t src_start = reinterpret_cast<uintptr_t>(srcHost);
  if (!state.is_server_authoritative && ByteCount != 0 &&
      src_start <= UINTPTR_MAX - ByteCount && !state.host.source_registered &&
      !routing.src_deviceptr_tracked &&
      lupine_is_client_mapped_address(reinterpret_cast<CUdeviceptr>(srcHost))) {
    state.pushed_bytes = ByteCount;
  }
  return state;
}

static bool lupine_send_htod_rpc(conn_t *conn, int rpc_op, CUdeviceptr &dst,
                                 const void *srcHost, size_t &bytes,
                                 const CUstream *stream,
                                 lupine_htod_wire_state &state,
                                 bool fire_and_forget, CUresult *out) {
  return lupine_prepare_rpc(conn) >= 0 &&
         lupine_rpc_start_on_stream(
             conn, rpc_op, stream != nullptr ? *stream : nullptr) >= 0 &&
         rpc_write(conn, &state.is_server_authoritative,
                   sizeof(state.is_server_authoritative)) >= 0 &&
         rpc_write(conn, &dst, sizeof(dst)) >= 0 &&
         rpc_write(conn, &bytes, sizeof(bytes)) >= 0 &&
         (stream == nullptr || rpc_write(conn, stream, sizeof(*stream)) >= 0) &&
         rpc_write(conn, &state.wire_source, sizeof(state.wire_source)) >= 0 &&
         rpc_write(conn, &state.pushed_bytes, sizeof(state.pushed_bytes)) >=
             0 &&
         (state.pushed_bytes == 0 ||
          rpc_write(conn, srcHost, state.pushed_bytes) >= 0) &&
         (fire_and_forget ? rpc_write_end(conn) >= 0
                          : rpc_wait_for_response(conn) >= 0 &&
                                rpc_read(conn, out, sizeof(*out)) >= 0 &&
                                rpc_read_end(conn) >= 0);
}

// A synchronous host-to-device copy this client can answer without the
// server's reply.
//
// What `cuMemcpyHtoD_v2` returning tells a correct caller is that the source
// buffer is free to reuse and that later CUDA work sees the bytes. It does not
// promise the transfer has landed: measured natively on driver 590.48, a 1 MiB
// pageable copy reports 24.8 GB/s and leaves 0.34 ms of transfer outstanding,
// because the driver stages it. It stages into a bounded ring and blocks when
// that overflows -- at 256 MiB and above the rate falls to the link's 2.7 GB/s
// -- and it never stages a page-locked source, which is why a pinned copy is
// fully blocking. Both rules are kept here: `lupine_htod_pushed_bytes` answers
// zero for any source overlapping a tracked host allocation, so only a pageable
// source is ever sent this way, and `rpc_write_end` has compressed or copied
// every byte into the stream's encoder before it returns, into a transport that
// blocks the caller once its outbound backlog passes 8 MiB rather than growing.
// A later request from any thread waits for this one's submission at the
// server (#919), as a later synchronous call does natively.
//
// The destination has to be one allocation this client tracks, on this route,
// with room for the copy, and never a host allocation's device alias, which the
// application could read back with no CUDA call at all. Everything else keeps
// the reply, so an argument error is still returned by the call that made it.
extern "C" CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                    size_t ByteCount) {
  lupine_htod_routing_info routing =
      lupine_query_htod_routing(dstDevice, srcHost, ByteCount);
  lupine_route route = routing.dst_route;
  CUresult return_value = CUDA_SUCCESS;
  if (lupine_route_is_local(route)) {
    lupine_note_device_write(dstDevice);
    return_value = lupine_call_real_cuda_fn("cuMemcpyHtoD_v2", dstDevice,
                                            srcHost, ByteCount);
    return return_value == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                        : return_value;
  }
  if (ByteCount != 0 && srcHost == nullptr) {
    lupine_note_device_write(dstDevice);
    return CUDA_ERROR_INVALID_VALUE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  lupine_htod_wire_state state =
      lupine_prepare_htod_wire_state(routing, dstDevice, srcHost, ByteCount);
  lupine_bulk_lanes *lanes = (state.pushed_bytes == ByteCount &&
                              ByteCount >= LUPINE_BULK_COPY_MIN_BYTES)
                                 ? lupine_client_transport_bulk_lanes(conn)
                                 : nullptr;
  // A striped copy still waits: its chunks travel on the bulk connections.
  // Otherwise the ticket of a copy sent without waiting keeps it in
  // submission order against work issued on other lanes, which is the
  // ordering the reply used to provide.
  bool fire_and_forget =
      lanes == nullptr && state.pushed_bytes == ByteCount && ByteCount != 0 &&
      lupine_active_stream_captures.load(std::memory_order_relaxed) == 0 &&
      routing.dst_allocation_covers && !state.host.dst_tracked_host;
  if (lanes != nullptr) {
    return_value = lupine_prepare_rpc(conn) < 0
                       ? CUDA_ERROR_DEVICE_UNAVAILABLE
                       : lupine_bulk_push(conn, lanes, dstDevice, srcHost,
                                          ByteCount, false);
  } else if (!lupine_send_htod_rpc(conn, RPC_cuMemcpyHtoD_v2, dstDevice,
                                   srcHost, ByteCount, nullptr, state,
                                   fire_and_forget, &return_value)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                      : return_value;
}

// An async host-to-device copy this client can issue without the server's
// reply.
//
// The reply only carries the enqueue result. Natively a page-locked source is
// read when the stream reaches the copy and the call returns at once; here the
// server copies a mirrored source from its own page-locked copy, and pulls a
// registered one from this client at that point through a callback the
// dispatch thread serves whatever the caller is doing. A pageable source of
// at most LUPINE_INLINE_HTOD_BYTES travels in the request, which has copied it
// before rpc_write_end returns, and the server hands it to the driver, which
// stages it before returning just as it does natively. Either way the server
// ends the copy's ticket once the copy is queued, not when the stream reaches
// it: a ticket held that long would make every later request from any thread
// wait behind work queued ahead of the copy, which may itself be waiting on
// one of those requests. A larger pageable source still waits, since its
// bytes are staged through a one-fragment ring the handler drains in stream
// order, and so does a device source on another route.
//
// The destination rules and the capture exclusion match the synchronous copy,
// so argument errors and a capture's failure still reach the call that made
// them; an enqueue failure past those checks reaches the client from its next
// synchronize, as for launches.
extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                         const void *srcHost, size_t ByteCount,
                                         CUstream hStream) {
  lupine_htod_routing_info routing =
      lupine_query_htod_routing(dstDevice, srcHost, ByteCount);
  lupine_route route = routing.dst_route;
  if (lupine_stream_crosses_route(hStream, route)) {
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
  }
  if (lupine_route_is_local(route)) {
    lupine_note_device_write(dstDevice);
    return lupine_call_real_cuda_fn("cuMemcpyHtoDAsync_v2", dstDevice, srcHost,
                                    ByteCount, hStream);
  }
  if (ByteCount != 0 && srcHost == nullptr) {
    lupine_note_device_write(dstDevice);
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_htod_wire_state state =
      lupine_prepare_htod_wire_state(routing, dstDevice, srcHost, ByteCount);
  bool inline_pageable =
      state.pushed_bytes == ByteCount && ByteCount <= LUPINE_INLINE_HTOD_BYTES;
  bool fire_and_forget =
      ByteCount != 0 &&
      (state.is_server_authoritative || inline_pageable ||
       state.host.source_registered) &&
      lupine_active_stream_captures.load(std::memory_order_relaxed) == 0 &&
      routing.dst_allocation_covers && !state.host.dst_tracked_host;
  CUresult return_value = CUDA_SUCCESS;
  return lupine_send_htod_rpc(lupine_route_remote_conn(route),
                              RPC_cuMemcpyHtoDAsync_v2, dstDevice, srcHost,
                              ByteCount, &hStream, state, fire_and_forget,
                              &return_value)
             ? return_value
             : CUDA_ERROR_DEVICE_UNAVAILABLE;
}

static bool lupine_device_copy_uses_remote_callback(CUdeviceptr destination,
                                                    CUdeviceptr source) {
  if (!lupine_deviceptr_is_tracked(source) ||
      lupine_copy_pointer_is_host(source)) {
    return false;
  }
  lupine_route destination_route = lupine_route_for_deviceptr(destination);
  lupine_route source_route = lupine_route_for_deviceptr(source);
  return !lupine_route_is_local(destination_route) &&
         !lupine_route_is_local(source_route) &&
         !lupine_routes_share_server(destination_route, source_route);
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

  if (lupine_device_copy_uses_remote_callback(dstDevice, srcDevice)) {
    const void *source = reinterpret_cast<const void *>(srcDevice);
    return async ? cuMemcpyHtoDAsync_v2(dstDevice, source, ByteCount, hStream)
                 : cuMemcpyHtoD_v2(dstDevice, source, ByteCount);
  }

  constexpr size_t chunk_size = 16 * 1024 * 1024;
  size_t staging_size = std::min(ByteCount, chunk_size);
  std::unique_ptr<unsigned char[]> staging(
      new (std::nothrow) unsigned char[staging_size]);
  if (!staging) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  size_t offset = 0;
  while (offset < ByteCount) {
    size_t chunk = std::min(staging_size, ByteCount - offset);
    CUresult result = cuMemcpyDtoH_v2(staging.get(), srcDevice + offset, chunk);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (async) {
      result = cuMemcpyHtoDAsync_v2(dstDevice + offset, staging.get(), chunk,
                                    hStream);
    } else {
      result = cuMemcpyHtoD_v2(dstDevice + offset, staging.get(), chunk);
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offset += chunk;
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_memcpy_htoa(CUarray dstArray, size_t dstOffset,
                                   const void *srcHost, size_t ByteCount,
                                   const CUstream *hStream) {
  lupine_route route = hStream != nullptr
                           ? lupine_route_for_stream_or_default(*hStream)
                           : lupine_route_for_default();
  if (hStream != nullptr && lupine_stream_crosses_route(*hStream, route)) {
    return lupine_memcpy_htoa(dstArray, dstOffset, srcHost, ByteCount, nullptr);
  }
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        hStream != nullptr
            ? lupine_call_real_cuda_fn("cuMemcpyHtoAAsync_v2", dstArray,
                                       dstOffset, srcHost, ByteCount, *hStream)
            : lupine_call_real_cuda_fn("cuMemcpyHtoA_v2", dstArray, dstOffset,
                                       srcHost, ByteCount);
    if (hStream == nullptr && return_value == CUDA_SUCCESS) {
      return_value = lupine_sync_mapped_device_to_host();
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUdeviceptr server_source = 0;
  bool is_server_authoritative =
      srcHost == nullptr ||
      lupine_translate_client_host_range_to_server(
          &server_source, srcHost, ByteCount, lupine_route_identity(route));
  const void *wire_source = is_server_authoritative
                                ? reinterpret_cast<const void *>(server_source)
                                : srcHost;
  uint64_t pushed_bytes =
      dstArray == nullptr ? 0
                          : lupine_htod_pushed_bytes(is_server_authoritative,
                                                     srcHost, ByteCount);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn,
                                 hStream != nullptr ? RPC_cuMemcpyHtoAAsync_v2
                                                    : RPC_cuMemcpyHtoA_v2,
                                 hStream != nullptr ? *hStream : nullptr) < 0 ||
      rpc_write(conn, &is_server_authoritative,
                sizeof(is_server_authoritative)) < 0 ||
      rpc_write(conn, &dstArray, sizeof(dstArray)) < 0 ||
      rpc_write(conn, &dstOffset, sizeof(dstOffset)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      (hStream != nullptr && rpc_write(conn, hStream, sizeof(*hStream)) < 0) ||
      rpc_write(conn, &wire_source, sizeof(wire_source)) < 0 ||
      rpc_write(conn, &pushed_bytes, sizeof(pushed_bytes)) < 0 ||
      (pushed_bytes != 0 && rpc_write(conn, srcHost, pushed_bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (hStream == nullptr && return_value == CUDA_SUCCESS) {
    return_value = lupine_sync_mapped_device_to_host();
  }
  return return_value;
}

extern "C" CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                    const void *srcHost, size_t ByteCount) {
  return lupine_memcpy_htoa(dstArray, dstOffset, srcHost, ByteCount, nullptr);
}

extern "C" CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                         const void *srcHost, size_t ByteCount,
                                         CUstream hStream) {
  return lupine_memcpy_htoa(dstArray, dstOffset, srcHost, ByteCount, &hStream);
}

static CUresult lupine_memcpy_atoh(void *dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   const CUstream *hStream) {
  lupine_route route = hStream != nullptr
                           ? lupine_route_for_stream_or_default(*hStream)
                           : lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    return hStream != nullptr
               ? lupine_call_real_cuda_fn("cuMemcpyAtoHAsync_v2", dstHost,
                                          srcArray, srcOffset, ByteCount,
                                          *hStream)
               : lupine_call_real_cuda_fn("cuMemcpyAtoH_v2", dstHost, srcArray,
                                          srcOffset, ByteCount);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn,
                                 hStream != nullptr ? RPC_cuMemcpyAtoHAsync_v2
                                                    : RPC_cuMemcpyAtoH_v2,
                                 hStream != nullptr ? *hStream : nullptr) < 0 ||
      rpc_write(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_write(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      (hStream != nullptr && rpc_write(conn, hStream, sizeof(*hStream)) < 0)) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return lupine_read_dtoh_chunks(
      conn, (rpc_expect_answer(conn), rpc_write_end(conn)),
      static_cast<unsigned char *>(dstHost), ByteCount);
}

extern "C" CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                    size_t srcOffset, size_t ByteCount) {
  return lupine_memcpy_atoh(dstHost, srcArray, srcOffset, ByteCount, nullptr);
}

extern "C" CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray,
                                         size_t srcOffset, size_t ByteCount,
                                         CUstream hStream) {
  return lupine_memcpy_atoh(dstHost, srcArray, srcOffset, ByteCount, &hStream);
}

// Another thread's capture on another stream leaves this one blocking.
static bool lupine_dtoh_blocks(const void *dstHost, size_t bytes,
                               CUstream hStream, conn_t *dtoh_conn = nullptr) {
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  return !lupine_host_range_is_page_locked(dstHost, bytes, dtoh_conn) &&
         (lupine_active_stream_captures.load(std::memory_order_relaxed) == 0 ||
          cuStreamIsCapturing(hStream, &capture_status) != CUDA_SUCCESS ||
          capture_status == CU_STREAM_CAPTURE_STATUS_NONE);
}

extern "C" CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                         size_t ByteCount, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(srcDevice);
  if (lupine_stream_crosses_route(hStream, route)) {
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
  }
  if (lupine_route_is_local(route)) {
    if (ByteCount != 0 && lupine_dtoh_blocks(dstHost, ByteCount, hStream)) {
      return lupine_call_real_cuda_fn("cuMemcpyDtoH_v2", dstHost, srcDevice,
                                      ByteCount);
    }
    return lupine_call_real_cuda_fn("cuMemcpyDtoHAsync_v2", dstHost, srcDevice,
                                    ByteCount, hStream);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  void *client_alias = nullptr;
  void *server_host = nullptr;
  bool page_locked = false;
  bool pinned_destination = lupine_pinned_dtoh_destination(
      conn, dstHost, ByteCount, &page_locked, &client_alias, &server_host);

  // "API synchronization behavior" lets the driver make this copy synchronous
  // when the destination is pageable, and every driver does. Callers rely on it
  // instead of synchronizing: cuSPARSE reads its scalar results (nnz counts,
  // SpVV dot products) that way, and a deferred copy either never lands or
  // lands on a stack frame the caller has already left. The bytes come back
  // through the same bounded, chunked path the synchronous copy uses, issued on
  // this stream so they stay ordered behind its prior work. Capture is the
  // exception: there the copy only becomes a graph node and must not block.
  if (ByteCount != 0 && !page_locked) {
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    if (lupine_active_stream_captures.load(std::memory_order_relaxed) == 0 ||
        cuStreamIsCapturing(hStream, &capture_status) != CUDA_SUCCESS ||
        capture_status == CU_STREAM_CAPTURE_STATUS_NONE) {
      return lupine_copy_dtoh_pageable(conn, dstHost, srcDevice, ByteCount,
                                       hStream);
    }
  }

  void *client_dst =
      pinned_destination && client_alias != nullptr ? client_alias : dstHost;
  uint8_t refresh = pinned_destination && client_alias == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyDtoHAsync_v2, hStream) < 0 ||
      rpc_write(conn, &client_dst, sizeof(client_dst)) < 0 ||
      rpc_write(conn, &server_host, sizeof(server_host)) < 0 ||
      rpc_write(conn, &refresh, sizeof(refresh)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

struct lupine_memcpy3d_api {
  int op;
  bool async;
  // Peer copies carry their contexts in the reserved context slots.
  bool peer;
  CUresult (*local)(const CUDA_MEMCPY3D &copy, CUstream stream);
  // Where an async copy goes when its stream is on another route.
  const lupine_memcpy3d_api *sync;
};

// Every 2D and 3D copy, promoted to the 3D descriptor.
static CUresult lupine_memcpy3d(CUDA_MEMCPY3D copy, CUstream hStream,
                                const lupine_memcpy3d_api &api) {
  // UNIFIED addresses are client-space: a managed alias is already the
  // server's address for that allocation, a plain host pointer's bytes must
  // travel, and a device pointer is already server-valid.
  if (copy.srcMemoryType == CU_MEMORYTYPE_UNIFIED &&
      !lupine_is_managed_host_alias(copy.srcDevice) &&
      lupine_copy_pointer_is_host(copy.srcDevice)) {
    copy.srcHost = (const void *)copy.srcDevice;
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_UNIFIED &&
      !lupine_is_managed_host_alias(copy.dstDevice) &&
      lupine_copy_pointer_is_host(copy.dstDevice)) {
    copy.dstHost = (void *)copy.dstDevice;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  }
  bool src_host = copy.srcMemoryType == CU_MEMORYTYPE_HOST;
  bool dst_host = copy.dstMemoryType == CU_MEMORYTYPE_HOST;
  uint8_t direction = src_host ? (dst_host ? LUPINE_COPY_DIRECTION_HTOH
                                           : LUPINE_COPY_DIRECTION_HTOD)
                               : (dst_host ? LUPINE_COPY_DIRECTION_DTOH
                                           : LUPINE_COPY_DIRECTION_DTOD);
  bool src_array = copy.srcMemoryType == CU_MEMORYTYPE_ARRAY;
  bool dst_array = copy.dstMemoryType == CU_MEMORYTYPE_ARRAY;
  size_t src_slice_pitch = copy.srcHeight * copy.srcPitch;
  size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  const char *src_base = (const char *)copy.srcHost +
                         copy.srcZ * src_slice_pitch +
                         copy.srcY * copy.srcPitch + copy.srcXInBytes;
  char *dst_base = (char *)copy.dstHost + copy.dstZ * dst_slice_pitch +
                   copy.dstY * copy.dstPitch + copy.dstXInBytes;
  CUresult return_value;
  if (direction == LUPINE_COPY_DIRECTION_HTOH) {
    if (api.async && lupine_stream_is_capturing(hStream)) {
      return lupine_capture_host_to_host(dst_base, src_base, copy.WidthInBytes,
                                         copy.Height, copy.Depth, copy.dstPitch,
                                         copy.srcPitch, dst_slice_pitch,
                                         src_slice_pitch, hStream);
    }
    // Ordered like a linear host-to-host copy.
    return_value = lupine_order_host_to_host(dst_base, src_base,
                                             api.async ? hStream : nullptr);
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
    for (size_t z = 0; z < copy.Depth; ++z) {
      for (size_t row = 0; row < copy.Height; ++row) {
        memmove(dst_base + z * dst_slice_pitch + row * copy.dstPitch,
                src_base + z * src_slice_pitch + row * copy.srcPitch,
                copy.WidthInBytes);
      }
    }
    return CUDA_SUCCESS;
  }

  lupine_route route;
  if (api.peer) {
    // Both context handles must live on one connection; the copy then behaves
    // like an ordinary one there. Peers on different connections are not
    // supported.
    route = lupine_route_for_context((CUcontext)copy.reserved1);
    if (lupine_route_identity(route) !=
            lupine_route_identity(
                lupine_route_for_context((CUcontext)copy.reserved0)) ||
        lupine_route_is_local(route)) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  } else {
    if (direction == LUPINE_COPY_DIRECTION_DTOD && !src_array && !dst_array &&
        !lupine_deviceptrs_share_route(copy.dstDevice, copy.srcDevice)) {
      // Different devices: the rows route through the client.
      for (size_t z = 0; z < copy.Depth; ++z) {
        for (size_t row = 0; row < copy.Height; ++row) {
          return_value = lupine_cuMemcpyDtoD_via_client(
              copy.dstDevice + (copy.dstZ + z) * dst_slice_pitch +
                  (copy.dstY + row) * copy.dstPitch + copy.dstXInBytes,
              copy.srcDevice + (copy.srcZ + z) * src_slice_pitch +
                  (copy.srcY + row) * copy.srcPitch + copy.srcXInBytes,
              copy.WidthInBytes, api.async ? hStream : nullptr, api.async);
          if (return_value != CUDA_SUCCESS) {
            return return_value;
          }
        }
      }
      return lupine_sync_mapped_device_to_host();
    }
    // A device pointer picks the route; array-only copies follow the stream.
    if (direction != LUPINE_COPY_DIRECTION_DTOH && !dst_array) {
      lupine_note_device_write(copy.dstDevice);
      route = lupine_route_for_deviceptr(copy.dstDevice);
    } else if (direction != LUPINE_COPY_DIRECTION_HTOD && !src_array) {
      route = lupine_route_for_deviceptr(copy.srcDevice);
    } else {
      route = api.async ? lupine_route_for_stream_or_default(hStream)
                        : lupine_route_for_current_context();
    }
    if (api.async && lupine_stream_crosses_route(hStream, route)) {
      return lupine_memcpy3d(copy, nullptr, *api.sync);
    }
    if (lupine_route_is_local(route)) {
      return api.local(copy, hStream);
    }
  }
  conn_t *conn = lupine_route_remote_conn(route);

  // Peer sources travel in the request; others are pulled when the stream
  // reaches the copy, or read on the server when it owns the bytes.
  bool push_source = direction == LUPINE_COPY_DIRECTION_HTOD && api.peer;
  bool pull_source = direction == LUPINE_COPY_DIRECTION_HTOD && !api.peer;
  bool is_server_authoritative = false;
  if (pull_source) {
    size_t source_span = copy.Depth == 0 || copy.Height == 0
                             ? 0
                             : (copy.Depth - 1) * src_slice_pitch +
                                   (copy.Height - 1) * copy.srcPitch +
                                   copy.WidthInBytes;
    CUdeviceptr server_source = 0;
    is_server_authoritative = lupine_translate_client_host_range_to_server(
        &server_source, src_base, source_span,
        lupine_route_identity(lupine_remote_route_for_conn(conn)));
    if (is_server_authoritative) {
      copy.srcMemoryType = CU_MEMORYTYPE_UNIFIED;
      copy.srcDevice = server_source;
      copy.srcHost = nullptr;
      copy.srcXInBytes = 0;
      copy.srcY = 0;
      copy.srcZ = 0;
    }
  }
  if (direction == LUPINE_COPY_DIRECTION_DTOH && api.async && !api.peer) {
    // Pushed like a linear async copy: the server stages the rows densely and
    // they land once the stream runs the copy, which a pageable destination
    // waits for right away.
    bool empty = copy.WidthInBytes == 0 || copy.Height == 0 || copy.Depth == 0;
    size_t dst_span = 0;
    bool span_valid = !empty;
    if (span_valid) {
      auto mul_ok = [](size_t a, size_t b, size_t *out) {
        if (b != 0 && a > ~size_t{0} / b) {
          return false;
        }
        *out = a * b;
        return true;
      };
      auto add_ok = [](size_t a, size_t b, size_t *out) {
        if (a > ~size_t{0} - b) {
          return false;
        }
        *out = a + b;
        return true;
      };
      size_t z_off = 0;
      size_t y_off = 0;
      span_valid = mul_ok(copy.Depth - 1, dst_slice_pitch, &z_off) &&
                   mul_ok(copy.Height - 1, copy.dstPitch, &y_off) &&
                   add_ok(z_off, y_off, &dst_span) &&
                   add_ok(dst_span, copy.WidthInBytes, &dst_span);
    }
    bool blocks =
        !empty &&
        (!span_valid || lupine_dtoh_blocks(dst_base, dst_span, hStream, conn));
    if (lupine_prepare_rpc(conn) < 0 ||
        lupine_rpc_start_on_stream(conn, api.op, hStream) < 0 ||
        rpc_write(conn, &direction, sizeof(direction)) < 0 ||
        rpc_write(conn, &copy, sizeof(copy)) < 0 ||
        rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
        rpc_write_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (!blocks) {
      return CUDA_SUCCESS;
    }
    if (lupine_prepare_rpc(conn) < 0 ||
        lupine_rpc_start_on_stream(conn, RPC_cuStreamSynchronize, hStream) <
            0 ||
        rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        lupine_read_dtoh_watermark(conn) < 0 ||
        lupine_forward_remote_stdout(conn) < 0 ||
        rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return return_value == CUDA_SUCCESS ? lupine_sync_mapped_device_to_host()
                                        : return_value;
  }
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, api.op, hStream) < 0 ||
      rpc_write(conn, &direction, sizeof(direction)) < 0 ||
      (pull_source && rpc_write(conn, &is_server_authoritative,
                                sizeof(is_server_authoritative)) < 0) ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 ||
      (push_source &&
       rpc_write_pitched(conn, src_base, copy.WidthInBytes, copy.Height,
                         copy.srcPitch, copy.Depth, src_slice_pitch) < 0) ||
      (api.async && rpc_write(conn, &hStream, sizeof(hStream)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (direction == LUPINE_COPY_DIRECTION_DTOH &&
       lupine_read_dtoh_watermark(conn) < 0) ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      (direction == LUPINE_COPY_DIRECTION_DTOH &&
       return_value == CUDA_SUCCESS &&
       rpc_read_pitched(conn, dst_base, copy.WidthInBytes, copy.Height,
                        copy.dstPitch, copy.Depth, dst_slice_pitch) < 0) ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  if (direction != LUPINE_COPY_DIRECTION_DTOH && return_value == CUDA_SUCCESS) {
    return_value = lupine_sync_mapped_device_to_host();
  }
  return return_value;
}

static const lupine_memcpy3d_api lupine_memcpy2d_api = {
    RPC_cuMemcpy2D_v2, false, false,
    [](const CUDA_MEMCPY3D &copy, CUstream) {
      CUDA_MEMCPY2D flat = lupine_memcpy2d_of(copy);
      return lupine_call_real_cuda_fn("cuMemcpy2D_v2", &flat);
    },
    nullptr};
static const lupine_memcpy3d_api lupine_memcpy2d_unaligned_api = {
    RPC_cuMemcpy2DUnaligned_v2, false, false,
    [](const CUDA_MEMCPY3D &copy, CUstream) {
      CUDA_MEMCPY2D flat = lupine_memcpy2d_of(copy);
      return lupine_call_real_cuda_fn("cuMemcpy2DUnaligned_v2", &flat);
    },
    nullptr};
static const lupine_memcpy3d_api lupine_memcpy2d_async_api = {
    RPC_cuMemcpy2DAsync_v2, true, false,
    [](const CUDA_MEMCPY3D &copy, CUstream s) {
      CUDA_MEMCPY2D flat = lupine_memcpy2d_of(copy);
      return lupine_call_real_cuda_fn("cuMemcpy2DAsync_v2", &flat, s);
    },
    &lupine_memcpy2d_api};
static const lupine_memcpy3d_api lupine_memcpy3d_sync_api = {
    RPC_cuMemcpy3D_v2, false, false,
    [](const CUDA_MEMCPY3D &c, CUstream) {
      return lupine_call_real_cuda_fn("cuMemcpy3D_v2", &c);
    },
    nullptr};
static const lupine_memcpy3d_api lupine_memcpy3d_async_api = {
    RPC_cuMemcpy3DAsync_v2, true, false,
    [](const CUDA_MEMCPY3D &c, CUstream s) {
      return lupine_call_real_cuda_fn("cuMemcpy3DAsync_v2", &c, s);
    },
    &lupine_memcpy3d_sync_api};
static const lupine_memcpy3d_api lupine_memcpy3d_peer_api = {
    RPC_cuMemcpy3DPeer, false, true, nullptr, nullptr};
static const lupine_memcpy3d_api lupine_memcpy3d_peer_async_api = {
    RPC_cuMemcpy3DPeerAsync, true, true, nullptr, nullptr};

template <typename Copy>
static CUresult lupine_memcpy3d_ptr(const Copy *pCopy, CUstream stream,
                                    const lupine_memcpy3d_api &api) {
  if (pCopy == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if constexpr (std::is_same_v<Copy, CUDA_MEMCPY3D>) {
    return lupine_memcpy3d(*pCopy, stream, api);
  } else {
    return lupine_memcpy3d(lupine_memcpy3d_of(*pCopy), stream, api);
  }
}

extern "C" CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  return lupine_memcpy3d_ptr(pCopy, nullptr, lupine_memcpy2d_api);
}

extern "C" CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  return lupine_memcpy3d_ptr(pCopy, nullptr, lupine_memcpy2d_unaligned_api);
}

extern "C" CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                       CUstream hStream) {
  return lupine_memcpy3d_ptr(pCopy, hStream, lupine_memcpy2d_async_api);
}

extern "C" CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  return lupine_memcpy3d_ptr(pCopy, nullptr, lupine_memcpy3d_sync_api);
}

extern "C" CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                       CUstream hStream) {
  return lupine_memcpy3d_ptr(pCopy, hStream, lupine_memcpy3d_async_api);
}

extern "C" CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  return lupine_memcpy3d_ptr(pCopy, nullptr, lupine_memcpy3d_peer_api);
}

extern "C" CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                        CUstream hStream) {
  return lupine_memcpy3d_ptr(pCopy, hStream, lupine_memcpy3d_peer_async_api);
}
