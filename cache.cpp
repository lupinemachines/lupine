#include "cache.h"

#include <array>
#include <atomic>

namespace {

constexpr size_t kLaneContextCacheSlots = 32;

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

bool lupine_current_context_device_cache_lookup(CUcontext context,
                                                CUdevice *device) {
  uint64_t current_epoch =
      current_context_device_cache_epoch().load(std::memory_order_acquire);
  auto &entry = current_context_device_cache();
  if (device == nullptr || entry.epoch != current_epoch ||
      entry.context != context) {
    return false;
  }
  *device = entry.device;
  return true;
}

void lupine_current_context_device_cache_insert(CUcontext context,
                                                CUdevice device) {
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
