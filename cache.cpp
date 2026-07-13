#include "cache.h"

#include <array>
#include <atomic>
#include <functional>

namespace {

constexpr size_t kKernelParamLayoutCacheSlots = 128;
constexpr size_t kLaneContextCacheSlots = 32;

struct kernel_param_layout_cache_key {
  int route_id = -2;
  CUfunction function = nullptr;

  bool operator==(const kernel_param_layout_cache_key &other) const {
    return route_id == other.route_id && function == other.function;
  }
};

struct kernel_param_layout_cache_entry {
  uint64_t epoch = 0;
  kernel_param_layout_cache_key key;
  lupine_kernel_param_layout layout;
};

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

std::atomic<uint64_t> &kernel_param_layout_cache_epoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

std::array<kernel_param_layout_cache_entry, kKernelParamLayoutCacheSlots> &
kernel_param_layout_cache() {
  static thread_local std::array<kernel_param_layout_cache_entry,
                                 kKernelParamLayoutCacheSlots>
      cache;
  return cache;
}

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

size_t
kernel_param_layout_cache_slot(const kernel_param_layout_cache_key &key) {
  size_t hash = std::hash<int>{}(key.route_id);
  hash ^= std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(key.function)) +
          0x9e3779b9 + (hash << 6) + (hash >> 2);
  static_assert(
      (kKernelParamLayoutCacheSlots & (kKernelParamLayoutCacheSlots - 1)) == 0,
      "kernel parameter layout cache size must be a power of two");
  return hash & (kKernelParamLayoutCacheSlots - 1);
}

} // namespace

bool lupine_kernel_param_layout_cache_lookup(int route_id, CUfunction function,
                                             lupine_kernel_param_layout *layout,
                                             uint64_t *epoch) {
  uint64_t current_epoch =
      kernel_param_layout_cache_epoch().load(std::memory_order_acquire);
  kernel_param_layout_cache_key key{route_id, function};
  auto &entry =
      kernel_param_layout_cache()[kernel_param_layout_cache_slot(key)];
  if (entry.epoch == current_epoch && entry.key == key) {
    *layout = entry.layout;
    return true;
  }
  *epoch = current_epoch;
  return false;
}

void lupine_kernel_param_layout_cache_insert(
    int route_id, CUfunction function, const lupine_kernel_param_layout &layout,
    uint64_t epoch) {
  if (kernel_param_layout_cache_epoch().load(std::memory_order_acquire) !=
      epoch) {
    return;
  }
  kernel_param_layout_cache_key key{route_id, function};
  auto &entry =
      kernel_param_layout_cache()[kernel_param_layout_cache_slot(key)];
  entry.key = key;
  entry.layout = layout;
  entry.epoch = epoch;
}

void lupine_kernel_param_layout_cache_invalidate() {
  kernel_param_layout_cache_epoch().fetch_add(1, std::memory_order_acq_rel);
}

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
