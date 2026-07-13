#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <signal.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#include <cuda.h>

#include "client_routing.h"
#include "codegen/gen_api.h"
#include "lupine_attr_sizes.h"
#include "lupine_log.h"
#include "memcpy.h"
#include "rpc.h"

extern int rpc_size();
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount);

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
  size_t page_size = 0;
  size_t page_count = 0;
  unsigned int flags = 0;
  bool owned = false;
  bool owned_mmap = false;
  bool managed = false;
  bool local_cuda = false;
  bool client_to_server_only = false;
  CUdeviceptr server_host_ptr = 0;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool tracking_enabled = false;
  int fault_slot = -1;
  volatile sig_atomic_t full_dirty = 0;
  volatile sig_atomic_t retiring = 0;
  uint32_t pending_dirty_ranges = 0;
  uintptr_t host_base = 0;
  CUdeviceptr device_alloc_base = 0;
  int route_id = -2;
};

struct lupine_mapped_host_snapshot {
  void *host = nullptr;
  size_t size = 0;
  CUdeviceptr device_ptr = 0;
  bool device_dirty = false;
  bool managed = false;
};

using lupine_host_allocation_map =
    std::map<void *, lupine_host_allocation, std::less<void *>>;

static std::mutex &lupine_host_allocation_mutex() {
  static std::mutex mutex;
  return mutex;
}

static lupine_host_allocation_map &lupine_mutable_host_allocations_locked() {
  static lupine_host_allocation_map allocations;
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

static bool lupine_reserve_dirty_host_range(
    lupine_dirty_host_range_queue &queue, uint32_t *slot) {
  uint32_t next = __atomic_load_n(&queue.next, __ATOMIC_RELAXED);
  while (next < LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES &&
         !__atomic_compare_exchange_n(&queue.next, &next, next + 1, false,
                                      __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
  }
  if (next >= LUPINE_MAX_MANAGED_HOST_DIRTY_RANGES) {
    return false;
  }
  *slot = next;
  return true;
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

    auto &queue = lupine_dirty_host_range_queues[route_id];
    uintptr_t page = entry.base + page_index * page_size;
    if (__atomic_load_n(&allocation->retiring, __ATOMIC_ACQUIRE) != 0) {
      mprotect(reinterpret_cast<void *>(page), page_size,
               PROT_READ | PROT_WRITE);
      __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
      return;
    }
    uint32_t slot = 0;
    __atomic_add_fetch(&allocation->pending_dirty_ranges, 1,
                       __ATOMIC_ACQ_REL);
    if (lupine_reserve_dirty_host_range(queue, &slot)) {
      queue.ranges[slot] = {allocation, page, page + page_size};
      __atomic_store_n(&queue.ready[slot], 1, __ATOMIC_RELEASE);
    } else {
      __atomic_sub_fetch(&allocation->pending_dirty_ranges, 1,
                         __ATOMIC_RELEASE);
      __atomic_store_n(&allocation->full_dirty, 1, __ATOMIC_RELEASE);
      __atomic_store_n(&queue.full_dirty_pending, 1, __ATOMIC_RELEASE);
    }

    mprotect(reinterpret_cast<void *>(page), page_size,
             PROT_READ | PROT_WRITE);
    __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
    return;
  }

  __atomic_sub_fetch(&lupine_active_fault_handlers, 1, __ATOMIC_RELEASE);
  lupine_call_previous_sigsegv(sig, info, uctx);
}

static void lupine_install_sigsegv_handler() {
  alignas(16) static thread_local unsigned char signal_stack[64 * 1024];
  static thread_local bool signal_stack_installed = false;
  if (!signal_stack_installed) {
    stack_t stack = {};
    stack.ss_sp = signal_stack;
    stack.ss_size = sizeof(signal_stack);
    signal_stack_installed = sigaltstack(&stack, nullptr) == 0;
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
    lupine_fault_entries[index].end =
        reinterpret_cast<uintptr_t>(base) + size;
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

static void lupine_remove_fault_entry(int slot) {
  if (slot < 0 || slot >= static_cast<int>(LUPINE_MAX_FAULT_ENTRIES)) {
    return;
  }
  __atomic_store_n(&lupine_fault_entries[slot].allocation, nullptr,
                   __ATOMIC_RELEASE);
  while (__atomic_load_n(&lupine_active_fault_handlers, __ATOMIC_ACQUIRE) !=
         0) {
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

  lupine_install_sigsegv_handler();
  allocation->fault_slot =
      lupine_add_fault_entry(host, allocation->storage_size, allocation);
  if (allocation->fault_slot < 0) {
    allocation->tracking_enabled = false;
    return false;
  }
  if (!lupine_protect_host_range(host, allocation->storage_size, PROT_READ)) {
    lupine_remove_fault_entry(allocation->fault_slot);
    allocation->fault_slot = -1;
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
  lupine_remove_fault_entry(allocation.fault_slot);
  allocation.fault_slot = -1;
  allocation.tracking_enabled = false;
}

static std::vector<lupine_mapped_host_snapshot> lupine_mapped_host_snapshots() {
  std::vector<lupine_mapped_host_snapshot> snapshots;
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  for (const auto &entry : lupine_mutable_host_allocations_locked()) {
    if (entry.second.device_ptr != 0 && !entry.second.local_cuda) {
      snapshots.push_back({entry.first, entry.second.size,
                           entry.second.device_ptr, entry.second.device_dirty,
                           entry.second.managed});
    }
  }
  return snapshots;
}

extern "C" void lupine_mark_host_range_clean(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.tracking_enabled) {
    return;
  }

  auto &allocation = it->second;
  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + allocation.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / allocation.page_size;
  size_t last_page = (end - 1 - base) / allocation.page_size;
  uintptr_t protect_start = base + first_page * allocation.page_size;
  size_t protect_size = (last_page - first_page + 1) * allocation.page_size;
  lupine_protect_host_range(reinterpret_cast<void *>(protect_start),
                            protect_size, PROT_READ);
}

extern "C" void lupine_prepare_host_range_write(void *host, size_t size) {
  if (host == nullptr || size == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.tracking_enabled) {
    return;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t start = reinterpret_cast<uintptr_t>(host);
  uintptr_t end = std::min(start + size, base + it->second.size);
  if (start >= end) {
    return;
  }

  size_t first_page = (start - base) / it->second.page_size;
  size_t last_page = (end - 1 - base) / it->second.page_size;
  uintptr_t protect_start = base + first_page * it->second.page_size;
  size_t protect_size = (last_page - first_page + 1) * it->second.page_size;
  lupine_protect_host_range(reinterpret_cast<void *>(protect_start),
                            protect_size, PROT_READ | PROT_WRITE);
}

static CUresult lupine_flush_dirty_host_pages_to_route(size_t route_id) {
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
  if (end == queue.start && !has_full_dirty) {
    return CUDA_SUCCESS;
  }

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

  if (has_full_dirty &&
      __atomic_exchange_n(&queue.full_dirty_pending, 0, __ATOMIC_ACQ_REL) !=
          0) {
    std::lock_guard<std::mutex> allocation_lock(
        lupine_host_allocation_mutex());
    for (auto &entry : lupine_mutable_host_allocations_locked()) {
      auto &allocation = entry.second;
      if (!allocation.tracking_enabled ||
          allocation.route_id != static_cast<int>(route_id) ||
          __atomic_load_n(&allocation.retiring, __ATOMIC_ACQUIRE) != 0 ||
          __atomic_exchange_n(&allocation.full_dirty, 0,
                              __ATOMIC_ACQ_REL) == 0) {
        continue;
      }
      __atomic_add_fetch(&allocation.pending_dirty_ranges, 1,
                         __ATOMIC_ACQ_REL);
      ranges.push_back({&allocation, allocation.host_base,
                        allocation.host_base + allocation.storage_size});
    }
  }

  if (ranges.empty()) {
    return CUDA_SUCCESS;
  }

  auto release_ranges = [&](bool restore) {
    for (const auto &range : ranges) {
      if (restore &&
          __atomic_load_n(&range.allocation->retiring, __ATOMIC_ACQUIRE) ==
              0) {
        __atomic_store_n(&range.allocation->full_dirty, 1, __ATOMIC_RELEASE);
        __atomic_store_n(&queue.full_dirty_pending, 1, __ATOMIC_RELEASE);
      }
      __atomic_sub_fetch(&range.allocation->pending_dirty_ranges, 1,
                         __ATOMIC_RELEASE);
    }
  };

  std::sort(ranges.begin(), ranges.end(),
            [](const lupine_dirty_host_range &a,
               const lupine_dirty_host_range &b) {
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
    if (merged.empty() ||
        merged.back().allocation != range.allocation ||
        merged.back().end < range.start) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }

  for (const auto &range : merged) {
    if (!lupine_protect_host_range(reinterpret_cast<void *>(range.start),
                                   range.end - range.start, PROT_READ)) {
      release_ranges(true);
      return CUDA_ERROR_UNKNOWN;
    }
  }

  conn_t *conn = lupine_route_remote_conn(
      lupine_route_from_identity(static_cast<int>(route_id)));
  if (conn == nullptr) {
    release_ranges(true);
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  std::array<std::array<unsigned char,
                        LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES>,
             LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES>
      headers;
  std::array<struct iovec, LUPINE_MANAGED_HOST_FLUSH_BATCH_RANGES * 2>
      iovecs;

  auto send_batch = [&](uint32_t count) {
    if (count == 0) {
      return CUDA_SUCCESS;
    }
    CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    if (rpc_write_start_request(conn, LUPINE_RPC_lupineManagedHostFlush) < 0 ||
        rpc_write(conn, &count, sizeof(count)) < 0 ||
        rpc_write_iovecs(conn, iovecs.data(), count * 2) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 ||
        rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return result;
  };

  uint32_t count = 0;
  for (const auto &range : merged) {
    auto &allocation = *range.allocation;
    size_t offset = range.start - allocation.host_base;
    if (offset >= allocation.size) {
      continue;
    }
    size_t bytes = std::min(range.end - range.start, allocation.size - offset);
    CUdeviceptr dst = allocation.server_host_ptr + offset;
    memcpy(headers[count].data(), &dst, sizeof(dst));
    memcpy(headers[count].data() + sizeof(dst), &bytes, sizeof(bytes));
    iovecs[count * 2] = {headers[count].data(),
                         LUPINE_MANAGED_HOST_FLUSH_HEADER_BYTES};
    iovecs[count * 2 + 1] = {reinterpret_cast<void *>(range.start), bytes};
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

extern "C" CUresult lupine_flush_dirty_host_pages_to_server() {
  for (int route_id = 0; route_id < rpc_size(); ++route_id) {
    CUresult result =
        lupine_flush_dirty_host_pages_to_route(static_cast<size_t>(route_id));
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_drain_retiring_dirty_ranges(
    lupine_host_allocation *allocation) {
  if (allocation == nullptr || allocation->route_id < 0 ||
      allocation->route_id >=
          static_cast<int>(LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES)) {
    return CUDA_SUCCESS;
  }
  CUresult result = lupine_flush_dirty_host_pages_to_route(
      static_cast<size_t>(allocation->route_id));
  if (result != CUDA_SUCCESS) {
    return result;
  }
  while (__atomic_load_n(&allocation->pending_dirty_ranges,
                         __ATOMIC_ACQUIRE) != 0) {
  }
  return CUDA_SUCCESS;
}

static bool lupine_device_ptr_in_mapping(CUdeviceptr ptr,
                                         const lupine_mapped_host_snapshot &m) {
  return ptr >= m.device_ptr && ptr < m.device_ptr + m.size;
}

static bool lupine_host_ptr_in_mapping(CUdeviceptr ptr,
                                       const lupine_mapped_host_snapshot &m,
                                       CUdeviceptr *translated) {
  uintptr_t host = reinterpret_cast<uintptr_t>(m.host);
  if (ptr < host || ptr >= host + m.size) {
    return false;
  }
  if (translated != nullptr) {
    *translated = m.device_ptr + (ptr - host);
  }
  return true;
}

extern "C" bool lupine_translate_managed_host_ptr(CUdeviceptr ptr,
                                                  CUdeviceptr *translated) {
  void *host = reinterpret_cast<void *>(ptr);
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it == lupine_mutable_host_allocations_locked().end() ||
      !it->second.managed) {
    return false;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
  uintptr_t addr = reinterpret_cast<uintptr_t>(host);
  if (translated != nullptr) {
    *translated = it->second.device_ptr + (addr - base);
  }
  return true;
}

static void lupine_mark_mapped_device_dirty(void *host) {
  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(host);
  if (it != lupine_mutable_host_allocations_locked().end() &&
      !it->second.client_to_server_only) {
    it->second.device_dirty = true;
  }
}

CUresult lupine_sync_mapped_host_to_device_for_launch(
    unsigned char *packed, const size_t *offsets, const size_t *sizes,
    uint32_t count, bool *used_managed_mapping) {
  if (packed == nullptr || offsets == nullptr || sizes == nullptr) {
    return count == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
  }
  if (used_managed_mapping != nullptr) {
    *used_managed_mapping = false;
  }
  std::vector<lupine_mapped_host_snapshot> snapshots =
      lupine_mapped_host_snapshots();
  bool used_managed = false;
  for (const auto &mapping : snapshots) {
    for (uint32_t i = 0; i < count; ++i) {
      if (sizes[i] != sizeof(CUdeviceptr)) {
        continue;
      }
      CUdeviceptr arg = 0;
      memcpy(&arg, packed + offsets[i], sizeof(arg));
      CUdeviceptr translated = 0;
      if (lupine_host_ptr_in_mapping(arg, mapping, &translated)) {
        memcpy(packed + offsets[i], &translated, sizeof(translated));
        lupine_mark_mapped_device_dirty(mapping.host);
        used_managed = used_managed || mapping.managed;
        break;
      }
      if (lupine_device_ptr_in_mapping(arg, mapping)) {
        lupine_mark_mapped_device_dirty(mapping.host);
        used_managed = used_managed || mapping.managed;
        break;
      }
    }
  }
  if (used_managed_mapping != nullptr) {
    *used_managed_mapping = used_managed;
  }
  return lupine_flush_dirty_host_pages_to_server();
}

extern "C" CUresult lupine_sync_mapped_device_to_host() {
  for (const auto &mapping : lupine_mapped_host_snapshots()) {
    if (!mapping.device_dirty || mapping.size == 0) {
      continue;
    }
    CUresult result =
        cuMemcpyDtoH_v2(mapping.host, mapping.device_ptr, mapping.size);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto it = lupine_mutable_host_allocations_locked().find(mapping.host);
    if (it != lupine_mutable_host_allocations_locked().end()) {
      it->second.device_dirty = false;
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

static CUresult lupine_remote_cuMemHostAlloc(void **remote_host,
                                             size_t bytesize,
                                             unsigned int flags,
                                             lupine_route route) {
  if (remote_host == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(void **, size_t, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostAlloc");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(remote_host, bytesize, flags);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  *remote_host = nullptr;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemHostAlloc) < 0 ||
      rpc_write(conn, remote_host, sizeof(*remote_host)) < 0 ||
      rpc_write(conn, &bytesize, sizeof(bytesize)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, remote_host, sizeof(*remote_host)) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
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
    using real_fn_t = CUresult (*)(void *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemFreeHost");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(remote_host);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemFreeHost) < 0 ||
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
    using real_fn_t = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostGetDevicePointer_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(device_ptr, remote_host, flags);
  }

  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  *device_ptr = 0;
  if (conn == nullptr ||
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

static CUresult lupine_register_host(void *p, size_t bytesize,
                                     unsigned int flags,
                                     bool client_to_server_only);

extern "C" CUresult cuMemHostAlloc(void **pp, size_t bytesize,
                                   unsigned int Flags) {
  if (pp == nullptr || bytesize == 0) {
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
    using real_fn_t = CUresult (*)(void **, size_t, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostAlloc");
    CUresult result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                      : real(pp, bytesize, Flags);
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
    allocation.owned = true;
    allocation.local_cuda = true;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(*pp);
    allocation.route_id = lupine_route_identity(route);
    {
      std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
      if (!lupine_mutable_host_allocations_locked()
               .emplace(*pp, std::move(allocation))
               .second) {
        using free_fn_t = CUresult (*)(void *);
        auto free_real = lupine_real_cuda_fn<free_fn_t>("cuMemFreeHost");
        if (free_real != nullptr) {
          free_real(*pp);
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
    lupine_note_deviceptr_allocation_route(reinterpret_cast<CUdeviceptr>(*pp),
                                           bytesize, route);
    return CUDA_SUCCESS;
  }

  void *remote_host = nullptr;
  CUresult result = lupine_remote_cuMemHostAlloc(
      &remote_host, bytesize, Flags | CU_MEMHOSTALLOC_DEVICEMAP, route);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUdeviceptr device_ptr = 0;
  result = lupine_remote_cuMemHostGetDevicePointer(&device_ptr, remote_host, 0,
                                                   route);
  if (result != CUDA_SUCCESS) {
    lupine_remote_cuMemFreeHost(remote_host, route);
    return result;
  }

  void *ptr = nullptr;
  size_t page_size = lupine_page_size();
  size_t storage_size = lupine_round_up(bytesize, page_size);
  ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    ptr = nullptr;
  }
  if (ptr == nullptr) {
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
    allocation.flags = Flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(remote_host);
    allocation.device_ptr = device_ptr;
    allocation.route_id = lupine_route_identity(route);
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto inserted = allocations.emplace(ptr, std::move(allocation));
    if (!inserted.second) {
      munmap(ptr, storage_size);
      lupine_remote_cuMemFreeHost(remote_host, route);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      allocations.erase(inserted.first);
      munmap(ptr, storage_size);
      lupine_remote_cuMemFreeHost(remote_host, route);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  lupine_note_deviceptr_allocation_route(
      reinterpret_cast<CUdeviceptr>(remote_host), bytesize, route);
  lupine_note_deviceptr_allocation_route(device_ptr, bytesize, route);
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
    storage_size = it->second.storage_size;
    server_host_ptr = it->second.server_host_ptr;
    device_ptr = it->second.device_ptr;
    __atomic_store_n(&it->second.retiring, 1, __ATOMIC_RELEASE);
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
    using real_fn_t = CUresult (*)(void *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemFreeHost");
    CUresult result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(p);
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
      munmap(p, storage_size);
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
      using real_fn_t = CUresult (*)(CUdeviceptr *, void *, unsigned int);
      auto real =
          lupine_real_cuda_fn<real_fn_t>("cuMemHostGetDevicePointer_v2");
      return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                             : real(pdptr, p, Flags);
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
    using real_fn_t = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostGetDevicePointer_v2");
    alloc_result = real == nullptr
                       ? CUDA_ERROR_DEVICE_UNAVAILABLE
                       : real(&new_device_ptr, allocation_host, Flags);
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

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto it = lupine_find_host_allocation_locked(p);
  if (it == lupine_mutable_host_allocations_locked().end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pFlags = it->second.flags;
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

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (lupine_mutable_host_allocations_locked().find(p) !=
        lupine_mutable_host_allocations_locked().end()) {
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    }
  }

  lupine_route route = lupine_route_for_default();
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(void *, size_t, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostRegister_v2");
    CUresult result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                      : real(p, bytesize, Flags);
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
    allocation.local_cuda = true;
    allocation.client_to_server_only = client_to_server_only;
    allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(p);
    allocation.route_id = lupine_route_identity(route);
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    if (!lupine_mutable_host_allocations_locked()
             .emplace(p, std::move(allocation))
             .second) {
      using unregister_fn_t = CUresult (*)(void *);
      auto unregister_real =
          lupine_real_cuda_fn<unregister_fn_t>("cuMemHostUnregister");
      if (unregister_real != nullptr) {
        unregister_real(p);
      }
      return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    }
    return CUDA_SUCCESS;
  }

  size_t page_size = lupine_page_size();
  if (lupine_host_flags_request_mapping(Flags) &&
      ((reinterpret_cast<uintptr_t>(p) % page_size) != 0 ||
       (bytesize % page_size) != 0)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  void *server_host = nullptr;
  CUdeviceptr device_ptr = 0;
  if (lupine_host_flags_request_mapping(Flags)) {
    unsigned int host_flags = CU_MEMHOSTALLOC_DEVICEMAP;
    if ((Flags & CU_MEMHOSTREGISTER_PORTABLE) != 0) {
      host_flags |= CU_MEMHOSTALLOC_PORTABLE;
    }
    CUresult result =
        lupine_remote_cuMemHostAlloc(&server_host, bytesize, host_flags, route);
    if (result == CUDA_SUCCESS) {
      result = lupine_remote_cuMemHostGetDevicePointer(&device_ptr, server_host,
                                                       0, route);
    }
    if (result != CUDA_SUCCESS) {
      if (server_host != nullptr) {
        lupine_remote_cuMemFreeHost(server_host, route);
      }
      return result;
    }
  }

  std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
  auto &allocations = lupine_mutable_host_allocations_locked();
  if (allocations.find(p) != allocations.end()) {
    if (server_host != nullptr) {
      lupine_remote_cuMemFreeHost(server_host, route);
    }
    return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
  }
  lupine_host_allocation allocation;
  allocation.size = bytesize;
  allocation.storage_size = bytesize;
  allocation.page_size = page_size;
  allocation.page_count = lupine_round_up(bytesize, page_size) / page_size;
  allocation.flags = Flags;
  allocation.owned = false;
  allocation.owned_mmap = false;
  allocation.client_to_server_only = client_to_server_only;
  allocation.server_host_ptr = reinterpret_cast<CUdeviceptr>(server_host);
  allocation.device_ptr = device_ptr;
  allocation.route_id = lupine_route_identity(route);
  auto inserted = allocations.emplace(p, std::move(allocation));
  if (!lupine_enable_dirty_tracking_locked(p, &inserted.first->second)) {
    allocations.erase(inserted.first);
    if (server_host != nullptr) {
      lupine_remote_cuMemFreeHost(server_host, route);
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (inserted.first->second.tracking_enabled) {
    int route_id = inserted.first->second.route_id;
    if (route_id < 0 ||
        route_id >= static_cast<int>(LUPINE_MAX_MANAGED_HOST_FLUSH_ROUTES)) {
      lupine_disable_dirty_tracking(p, inserted.first->second);
      allocations.erase(inserted.first);
      lupine_remote_cuMemFreeHost(server_host, route);
      return CUDA_ERROR_INVALID_VALUE;
    }
    __atomic_store_n(&inserted.first->second.full_dirty, 1, __ATOMIC_RELEASE);
    __atomic_store_n(
        &lupine_dirty_host_range_queues[route_id].full_dirty_pending, 1,
        __ATOMIC_RELEASE);
  }
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
  lupine_host_allocation *retiring_allocation = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    auto &allocations = lupine_mutable_host_allocations_locked();
    auto it = allocations.find(p);
    if (it == allocations.end() || it->second.owned) {
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    local_cuda = it->second.local_cuda;
    server_host_ptr = it->second.server_host_ptr;
    device_ptr = it->second.device_ptr;
    __atomic_store_n(&it->second.retiring, 1, __ATOMIC_RELEASE);
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
      return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    }
    allocations.erase(it);
  }
  if (local_cuda) {
    using real_fn_t = CUresult (*)(void *);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemHostUnregister");
    CUresult result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(p);
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
    using real_fn_t = CUresult (*)(CUdeviceptr *, size_t, unsigned int);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemAllocManaged");
    CUresult result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                      : real(dptr, bytesize, flags);
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
  if (conn == nullptr ||
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
  void *ptr = MAP_FAILED;
#ifdef MAP_FIXED_NOREPLACE
  ptr = mmap(reinterpret_cast<void *>(device_ptr), storage_size,
             PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
#endif
  if (ptr == MAP_FAILED) {
    ptr = mmap(nullptr, storage_size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if (ptr == MAP_FAILED) {
    cuMemFree_v2(device_alloc_base);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  {
    std::lock_guard<std::mutex> lock(lupine_host_allocation_mutex());
    lupine_host_allocation allocation;
    allocation.size = bytesize;
    allocation.storage_size = storage_size;
    allocation.page_size = page_size;
    allocation.page_count = storage_size / page_size;
    allocation.flags = flags;
    allocation.owned = true;
    allocation.owned_mmap = true;
    allocation.managed = true;
    allocation.server_host_ptr = device_ptr;
    allocation.device_ptr = device_ptr;
    allocation.device_alloc_base = device_alloc_base;
    allocation.route_id = lupine_route_identity(route);
    auto inserted = lupine_mutable_host_allocations_locked().emplace(
        ptr, std::move(allocation));
    if (!inserted.second) {
      munmap(ptr, storage_size);
      cuMemFree_v2(device_alloc_base);
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (!lupine_enable_dirty_tracking_locked(ptr, &inserted.first->second)) {
      lupine_mutable_host_allocations_locked().erase(inserted.first);
      munmap(ptr, storage_size);
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
      lupine_disable_dirty_tracking(it->first, it->second);
      retiring_allocation = &it->second;
      found = true;
    }
  }
  if (!found) {
    lupine_route route = lupine_route_for_deviceptr(dptr);
    if (lupine_route_is_local(route)) {
      using real_fn_t = CUresult (*)(CUdeviceptr);
      auto real = lupine_real_cuda_fn<real_fn_t>("cuMemFree_v2");
      if (real == nullptr) {
        return CUDA_ERROR_DEVICE_UNAVAILABLE;
      }
      CUresult result = real(dptr);
      if (result == CUDA_SUCCESS) {
        lupine_forget_deviceptr_owner(dptr);
      }
      return result;
    }
    conn_t *conn = lupine_route_remote_conn(route);
    CUresult return_value;
    if (conn == nullptr ||
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
    munmap(host, allocation.storage_size);
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
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  unsigned char value[64] = {};
  if (value_size > sizeof(value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_translate_managed_host_ptr(ptr, &query_ptr);
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(void *, CUpointer_attribute, CUdeviceptr);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuPointerGetAttribute");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(data, attribute, query_ptr);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  if (rpc_write_start_request(conn, RPC_cuPointerGetAttribute) < 0 ||
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
    if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
      void *host = reinterpret_cast<void *>(ptr);
      memcpy(data, &host, sizeof(host));
    } else if (managed_alias && attribute == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
      int is_managed = 1;
      memcpy(data, &is_managed, sizeof(is_managed));
    }
  }
  return return_value;
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
      return CUDA_ERROR_NOT_SUPPORTED;
    }
  }

  CUdeviceptr query_ptr = ptr;
  bool managed_alias = lupine_translate_managed_host_ptr(ptr, &query_ptr);
  lupine_route route = lupine_route_for_deviceptr(query_ptr);
  if (lupine_route_is_local(route)) {
    using real_fn_t =
        CUresult (*)(unsigned int, CUpointer_attribute *, void **, CUdeviceptr);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuPointerGetAttributes");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(numAttributes, attributes, data, query_ptr);
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (rpc_write_start_request(conn, RPC_cuPointerGetAttributes) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(numAttributes)) < 0 ||
      (numAttributes != 0 &&
       rpc_write(conn, attributes,
                 numAttributes * sizeof(CUpointer_attribute)) < 0) ||
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
      if (values[i].size() != value_sizes[i]) {
        return CUDA_ERROR_NOT_SUPPORTED;
      }
      memcpy(data[i], values[i].data(), values[i].size());
      if (managed_alias && attributes[i] == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
        void *host = reinterpret_cast<void *>(ptr);
        memcpy(data[i], &host, sizeof(host));
      } else if (managed_alias &&
                 attributes[i] == CU_POINTER_ATTRIBUTE_IS_MANAGED) {
        int is_managed = 1;
        memcpy(data[i], &is_managed, sizeof(is_managed));
      }
    }
  }
  return return_value;
}

static CUresult lupine_cuMemPrefetchAsync_location(CUdeviceptr devPtr,
                                                   size_t count,
                                                   CUmemLocation location,
                                                   unsigned int flags,
                                                   CUstream hStream) {
  CUdeviceptr translated = devPtr;
  if (lupine_translate_managed_host_ptr(devPtr, &translated)) {
    CUresult flush_result = lupine_flush_dirty_host_pages_to_server();
    if (flush_result != CUDA_SUCCESS) {
      return flush_result;
    }
  }

  CUmemLocation route_location = location;
  lupine_route route;
  if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
    CUdevice route_device = location.id;
    route = lupine_route_for_device(&route_device);
    route_location.id = route_device;
  } else {
    route = hStream != nullptr ? lupine_route_for_stream(hStream)
                               : lupine_route_for_deviceptr(translated);
  }
  if (hStream != nullptr) {
    lupine_route stream_route = lupine_route_for_stream(hStream);
    if (!lupine_routes_share_server(route, stream_route)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if (lupine_route_is_local(route)) {
#if CUDA_VERSION >= 12020
    using real_fn_t = CUresult (*)(CUdeviceptr, size_t, CUmemLocation,
                                   unsigned int, CUstream);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemPrefetchAsync_v2");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return real(translated, count, route_location, flags, hStream);
#else
    if (flags != 0 ||
        (route_location.type != CU_MEM_LOCATION_TYPE_DEVICE &&
         route_location.type != LUPINE_CU_MEM_LOCATION_TYPE_HOST)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    using real_fn_t = CUresult (*)(CUdeviceptr, size_t, CUdevice, CUstream);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemPrefetchAsync");
    if (real == nullptr) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    CUdevice dstDevice = route_location.type == CU_MEM_LOCATION_TYPE_DEVICE
                             ? route_location.id
                             : CU_DEVICE_CPU;
    return real(translated, count, dstDevice, hStream);
#endif
  }

  conn_t *conn = lupine_route_remote_conn(route);
  int location_type = static_cast<int>(route_location.type);
  CUresult return_value;
  if (conn == nullptr ||
      rpc_write_start_request(conn, LUPINE_RPC_cuMemPrefetchAsync) < 0 ||
      rpc_write(conn, &translated, sizeof(translated)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      rpc_write(conn, &location_type, sizeof(location_type)) < 0 ||
      rpc_write(conn, &route_location.id, sizeof(route_location.id)) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

extern "C" CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                       CUdevice dstDevice, CUstream hStream) {
  CUmemLocation location = {};
  location.type = dstDevice == CU_DEVICE_CPU ? LUPINE_CU_MEM_LOCATION_TYPE_HOST
                                             : CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = dstDevice == CU_DEVICE_CPU ? 0 : dstDevice;
  return lupine_cuMemPrefetchAsync_location(devPtr, count, location, 0,
                                            hStream);
}

extern "C" CUresult cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr, size_t count,
                                            CUdevice dstDevice,
                                            CUstream hStream) {
  return cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
}

extern "C" CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count,
                                          CUmemLocation location,
                                          unsigned int flags,
                                          CUstream hStream) {
  return lupine_cuMemPrefetchAsync_location(devPtr, count, location, flags,
                                            hStream);
}
