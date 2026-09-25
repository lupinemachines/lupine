#pragma once

#include <cstddef>
#include <iterator>
#include <map>

#include <cuda.h>

struct lupine_deviceptr_allocation_record {
  size_t size = 0;
  int route_id = -2;
  CUcontext context = nullptr;
};

// The caller holds the routing mutex. Disjoint allocations have logarithmic
// updates and lookups. Keep overlapping records too: an alias must not hide
// the remainder of an enclosing allocation after its own range ends.
class lupine_deviceptr_allocation_index {
public:
  void insert_or_assign(CUdeviceptr base, size_t size, int route_id,
                        CUcontext context) {
    erase(base);
    if (base == 0 || size == 0) {
      return;
    }
    auto next = allocations_.lower_bound(base);
    auto previous = predecessor(next);
    auto inserted = allocations_.emplace_hint(
        next, base,
        lupine_deviceptr_allocation_record{size, route_id, context});
    overlapping_neighbors_ -= overlaps(previous, next);
    overlapping_neighbors_ += overlaps(previous, inserted);
    overlapping_neighbors_ += overlaps(inserted, next);
  }

  void erase(CUdeviceptr base) {
    auto entry = allocations_.find(base);
    if (entry == allocations_.end()) {
      return;
    }
    auto previous = predecessor(entry);
    auto next = std::next(entry);
    overlapping_neighbors_ -= overlaps(previous, entry);
    overlapping_neighbors_ -= overlaps(entry, next);
    overlapping_neighbors_ += overlaps(previous, next);
    allocations_.erase(entry);
  }

  const lupine_deviceptr_allocation_record *find(CUdeviceptr ptr) const {
    auto entry = allocations_.upper_bound(ptr);
    while (entry != allocations_.begin()) {
      --entry;
      // Subtraction avoids wrapping the endpoint of high-address allocations.
      if (ptr - entry->first < entry->second.size) {
        return &entry->second;
      }
      if (overlapping_neighbors_ == 0) {
        break;
      }
      // Only overlapping allocations need the linear fallback. Prefer the
      // containing record with the greatest base when several cover ptr.
    }
    return nullptr;
  }

private:
  using allocations = std::map<CUdeviceptr, lupine_deviceptr_allocation_record>;
  using iterator = allocations::const_iterator;

  iterator predecessor(iterator entry) const {
    return entry == allocations_.begin() ? allocations_.end()
                                         : std::prev(entry);
  }

  bool overlaps(iterator first, iterator second) const {
    return first != allocations_.end() && second != allocations_.end() &&
           second->first - first->first < first->second.size;
  }

  allocations allocations_;
  size_t overlapping_neighbors_ = 0;
};
