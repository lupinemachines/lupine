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

std::atomic<uint64_t> &current_context_device_cache_epoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

current_context_device_cache_entry &current_context_device_cache() {
  static thread_local current_context_device_cache_entry cache;
  return cache;
}

std::atomic<uint64_t> &lane_context_cache_epoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

std::array<lane_context_cache_entry, kLaneContextCacheSlots> &
lane_context_cache() {
  static thread_local std::array<lane_context_cache_entry,
                                 kLaneContextCacheSlots>
      cache;
  return cache;
}

lane_context_cache_entry *lane_context_cache_entry_for(int route_id) {
  if (route_id < -1) {
    return nullptr;
  }
  static_assert((kLaneContextCacheSlots & (kLaneContextCacheSlots - 1)) == 0,
                "lane context cache size must be a power of two");
  size_t slot = static_cast<size_t>(route_id) & (kLaneContextCacheSlots - 1);
  return &lane_context_cache()[slot];
}

} // namespace

void lupine_deviceptr_allocation_cache_insert(CUdeviceptr base, size_t size,
                                             int route_id, CUcontext context) {
  lupine_deviceptr_allocation_cache_erase(base);
  if (base == 0 || size == 0) {
    return;
  }
  auto &cache = allocation_cache();
  auto &allocations = cache.allocations;
  auto next = allocations.lower_bound(base);
  auto previous = predecessor(allocations, next);
  auto inserted = allocations.emplace_hint(
      next, base, lupine_deviceptr_allocation_record{size, route_id, context});
  cache.overlapping_neighbors -= overlaps(allocations, previous, next);
  cache.overlapping_neighbors += overlaps(allocations, previous, inserted);
  cache.overlapping_neighbors += overlaps(allocations, inserted, next);
}

void lupine_deviceptr_allocation_cache_erase(CUdeviceptr base) {
  auto &cache = allocation_cache();
  auto &allocations = cache.allocations;
  auto entry = allocations.find(base);
  if (entry == allocations.end()) {
    return;
  }
  auto previous = predecessor(allocations, entry);
  auto next = std::next(entry);
  cache.overlapping_neighbors -= overlaps(allocations, previous, entry);
  cache.overlapping_neighbors -= overlaps(allocations, entry, next);
  cache.overlapping_neighbors += overlaps(allocations, previous, next);
  allocations.erase(entry);
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

bool lupine_current_context_device_cache_lookup(CUcontext context,
                                                CUdevice *device) {
  uint64_t current_epoch =
      current_context_device_cache_epoch().load(std::memory_order_acquire);
  auto &entry = current_context_device_cache();
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
  uint64_t epoch =
      current_context_device_cache_epoch().load(std::memory_order_acquire);
  auto &entry = current_context_device_cache();
  entry.context = context;
  entry.device = device;
  entry.epoch = epoch;
}

void lupine_current_context_device_cache_invalidate() {
  current_context_device_cache_epoch().fetch_add(1, std::memory_order_acq_rel);
}

uint64_t lupine_lane_context_cache_epoch() {
  return lane_context_cache_epoch().load(std::memory_order_acquire);
}

bool lupine_lane_context_cache_matches(int route_id, CUcontext context) {
  auto *entry = lane_context_cache_entry_for(route_id);
  return entry != nullptr && entry->route_id == route_id &&
         entry->context == context &&
         entry->epoch ==
             lane_context_cache_epoch().load(std::memory_order_acquire);
}

void lupine_lane_context_cache_update(int route_id, CUcontext context,
                                      uint64_t epoch, bool succeeded) {
  auto *entry = lane_context_cache_entry_for(route_id);
  if (entry == nullptr) {
    return;
  }
  if (!succeeded ||
      lane_context_cache_epoch().load(std::memory_order_acquire) != epoch) {
    if (entry->route_id == route_id) {
      entry->epoch = 0;
    }
    return;
  }
  entry->route_id = route_id;
  entry->context = context;
  entry->epoch = epoch;
}

void lupine_lane_context_cache_store(int route_id, CUcontext context) {
  lupine_lane_context_cache_update(route_id, context,
                                   lupine_lane_context_cache_epoch(), true);
}

extern "C" void lupine_invalidate_current_context_cache() {
  lupine_current_context_device_cache_invalidate();
  lane_context_cache_epoch().fetch_add(1, std::memory_order_acq_rel);
}

extern "C" void lupine_note_device_binding_changed() {
  // The lane's driver context went with its device, so what this thread
  // believed was current there no longer holds.
  for (auto &entry : lane_context_cache()) {
    entry.epoch = 0;
  }
}
