#include "cache.h"

#include <array>
#include <atomic>
#include <iterator>
#include <map>

namespace {

constexpr size_t kLaneContextCacheSlots = 32;

using deviceptr_allocations =
    std::map<CUdeviceptr, lupine_deviceptr_allocation_record>;

struct deviceptr_allocation_cache {
  deviceptr_allocations allocations;
  size_t overlapping_neighbors = 0;
};

deviceptr_allocation_cache &allocation_cache() {
  static auto *cache = new deviceptr_allocation_cache();
  return *cache;
}

using allocation_iterator = deviceptr_allocations::const_iterator;

allocation_iterator predecessor(const deviceptr_allocations &allocations,
                                allocation_iterator entry) {
  return entry == allocations.begin() ? allocations.end() : std::prev(entry);
}

bool overlaps(const deviceptr_allocations &allocations,
              allocation_iterator first, allocation_iterator second) {
  return first != allocations.end() && second != allocations.end() &&
         second->first - first->first < first->second.size;
}

deviceptr_allocations::iterator erase_entry(deviceptr_allocation_cache &cache,
                                            deviceptr_allocations::iterator it) {
  auto &allocations = cache.allocations;
  auto previous = predecessor(allocations, it);
  auto after = std::next(it);
  cache.overlapping_neighbors -= overlaps(allocations, previous, it);
  cache.overlapping_neighbors -= overlaps(allocations, it, after);
  cache.overlapping_neighbors += overlaps(allocations, previous, after);
  return allocations.erase(it);
}

struct current_context_device_cache_entry {
  uint64_t epoch = 0;
  CUcontext context = nullptr;
  CUdevice device = -1;
};

struct lane_context_cache_entry {
  uint64_t epoch = 0;
  int route_id = -2;
  CUcontext context = nullptr;
};

std::atomic<uint64_t> current_context_device_epoch{1};
thread_local current_context_device_cache_entry current_context_device;
std::atomic<uint64_t> lane_context_epoch{1};
thread_local std::array<lane_context_cache_entry, kLaneContextCacheSlots>
    lane_contexts;

lane_context_cache_entry *lane_context_cache_entry_for(int route_id) {
  if (route_id < -1) {
    return nullptr;
  }
  static_assert((kLaneContextCacheSlots & (kLaneContextCacheSlots - 1)) == 0,
                "lane context cache size must be a power of two");
  size_t slot = static_cast<size_t>(route_id) & (kLaneContextCacheSlots - 1);
  return &lane_contexts[slot];
}

} // namespace

void lupine_deviceptr_allocation_cache_insert(CUdeviceptr base, size_t size,
                                             int route_id, CUcontext context) {
  if (base == 0 || size == 0) {
    lupine_deviceptr_allocation_cache_erase(base);
    return;
  }
  auto &cache = allocation_cache();
  auto &allocations = cache.allocations;
  auto next = allocations.lower_bound(base);
  if (next != allocations.end() && next->first == base) {
    next = erase_entry(cache, next);
  }
  auto previous = predecessor(allocations, next);
  auto inserted = allocations.emplace_hint(
      next, base, lupine_deviceptr_allocation_record{size, route_id, context});
  cache.overlapping_neighbors -= overlaps(allocations, previous, next);
  cache.overlapping_neighbors += overlaps(allocations, previous, inserted);
  cache.overlapping_neighbors += overlaps(allocations, inserted, next);
}

void lupine_deviceptr_allocation_cache_erase(CUdeviceptr base) {
  auto &cache = allocation_cache();
  auto entry = cache.allocations.find(base);
  if (entry != cache.allocations.end()) {
    erase_entry(cache, entry);
  }
}

const lupine_deviceptr_allocation_record *
lupine_deviceptr_allocation_cache_lookup(CUdeviceptr ptr) {
  const auto &cache = allocation_cache();
  const auto &allocations = cache.allocations;
  auto entry = allocations.upper_bound(ptr);
  while (entry != allocations.begin()) {
    --entry;
    // Subtraction avoids wrapping the endpoint of high-address allocations.
    if (ptr - entry->first < entry->second.size) {
      return &entry->second;
    }
    if (cache.overlapping_neighbors == 0) {
      break;
    }
    // Disjoint allocations need only the predecessor lookup. Overlapping
    // records may hide the remainder of an enclosing allocation, so scan
    // backward and prefer the containing record with the greatest base.
  }
  return nullptr;
}

bool lupine_deviceptr_allocation_cache_overlaps_other_route(CUdeviceptr base,
                                                            size_t size,
                                                            int route_id) {
  const auto &cache = allocation_cache();
  auto entry = cache.allocations.lower_bound(base + size);
  while (entry != cache.allocations.begin()) {
    --entry;
    bool overlap =
        entry->first >= base || base - entry->first < entry->second.size;
    if (overlap && entry->second.route_id != route_id) {
      return true;
    }
    // Below base only the predecessor can reach it unless records overlap.
    if (entry->first < base && cache.overlapping_neighbors == 0) {
      break;
    }
  }
  return false;
}

bool lupine_current_context_device_cache_lookup(CUcontext context,
                                                CUdevice *device) {
  uint64_t current_epoch =
      current_context_device_epoch.load(std::memory_order_acquire);
  auto &entry = current_context_device;
  if (context == nullptr || device == nullptr || entry.epoch != current_epoch ||
      entry.context != context) {
    return false;
  }
  *device = entry.device;
  return true;
}

void lupine_current_context_device_cache_insert(CUcontext context,
                                                CUdevice device) {
  if (context == nullptr) {
    return;
  }
  current_context_device = {
      current_context_device_epoch.load(std::memory_order_acquire), context,
      device};
}

uint64_t lupine_lane_context_cache_epoch() {
  return lane_context_epoch.load(std::memory_order_acquire);
}

bool lupine_lane_context_cache_matches(int route_id, CUcontext context) {
  auto *entry = lane_context_cache_entry_for(route_id);
  return entry != nullptr && entry->route_id == route_id &&
         entry->context == context &&
         entry->epoch == lane_context_epoch.load(std::memory_order_acquire);
}

void lupine_lane_context_cache_update(int route_id, CUcontext context,
                                      uint64_t epoch, bool succeeded) {
  auto *entry = lane_context_cache_entry_for(route_id);
  if (entry == nullptr) {
    return;
  }
  if (!succeeded ||
      lane_context_epoch.load(std::memory_order_acquire) != epoch) {
    if (entry->route_id == route_id) {
      entry->epoch = 0;
    }
    return;
  }
  entry->route_id = route_id;
  entry->context = context;
  entry->epoch = epoch;
}

extern "C" void lupine_invalidate_current_context_cache() {
  current_context_device_epoch.fetch_add(1, std::memory_order_acq_rel);
  lane_context_epoch.fetch_add(1, std::memory_order_acq_rel);
}

extern "C" void lupine_note_device_binding_changed() {
  // The lane's driver context went with its device, so what this thread
  // believed was current there no longer holds.
  for (auto &entry : lane_contexts) {
    entry.epoch = 0;
  }
}
