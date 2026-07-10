#include "cache.h"

#include <array>
#include <atomic>
#include <functional>

namespace {

constexpr size_t kKernelParamLayoutCacheSlots = 128;

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
