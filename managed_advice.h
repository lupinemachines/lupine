#pragma once

#include <algorithm>
#include <cstring>
#include <cuda.h>
#include <iterator>
#include <limits>
#include <map>
#include <set>

// VMM-backed managed allocations cannot apply native UVM performance hints.
// Retain their observable advice independently, at CPU page granularity.
class lupine_managed_advice {
  struct state {
    bool read_mostly = false;
    int preferred = CU_DEVICE_INVALID;
    std::set<int> accessed_by;

    bool operator==(const state &other) const {
      return read_mostly == other.read_mostly && preferred == other.preferred &&
             accessed_by == other.accessed_by;
    }
  };
  std::map<size_t, state> ranges_;

  auto split(size_t page) {
    if (ranges_.empty())
      ranges_.emplace(0, state{});
    auto it = std::prev(ranges_.upper_bound(page));
    return ranges_.emplace(page, it->second).first;
  }

public:
  CUresult advise(size_t offset, size_t count, size_t page_size,
                  CUmem_advise advice, CUdevice device) {
    if (!count || !page_size ||
        count > std::numeric_limits<size_t>::max() - offset ||
        advice < CU_MEM_ADVISE_SET_READ_MOSTLY ||
        advice > CU_MEM_ADVISE_UNSET_ACCESSED_BY)
      return CUDA_ERROR_INVALID_VALUE;
    auto first = split(offset / page_size);
    auto last = split((offset + count - 1) / page_size + 1);
    for (auto it = first; it != last; ++it) {
      auto &value = it->second;
      switch (advice) {
      case CU_MEM_ADVISE_SET_READ_MOSTLY:
        value.read_mostly = true;
        break;
      case CU_MEM_ADVISE_UNSET_READ_MOSTLY:
        value.read_mostly = false;
        break;
      case CU_MEM_ADVISE_SET_PREFERRED_LOCATION:
        value.preferred = device;
        break;
      case CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION:
        value.preferred = CU_DEVICE_INVALID;
        break;
      case CU_MEM_ADVISE_SET_ACCESSED_BY:
        value.accessed_by.insert(device);
        break;
      case CU_MEM_ADVISE_UNSET_ACCESSED_BY:
        value.accessed_by.erase(device);
        break;
      }
    }
    for (auto it = std::next(ranges_.begin()); it != ranges_.end();) {
      if (std::prev(it)->second == it->second)
        it = ranges_.erase(it);
      else
        ++it;
    }
    return CUDA_SUCCESS;
  }

  static bool supports(CUmem_range_attribute attribute) {
    return attribute == CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY ||
           attribute == CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION ||
           attribute == CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY;
  }

  CUresult query(void *data, size_t data_size, CUmem_range_attribute attribute,
                 size_t offset, size_t count, size_t page_size) const {
    if (!count || !page_size ||
        count > std::numeric_limits<size_t>::max() - offset || !data ||
        !supports(attribute) || !data_size || data_size % sizeof(int) != 0 ||
        (attribute != CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY &&
         data_size != sizeof(int)))
      return CUDA_ERROR_INVALID_VALUE;
    state value;
    if (!ranges_.empty()) {
      auto it = std::prev(ranges_.upper_bound(offset / page_size));
      value = it->second;
      size_t last_page = (offset + count - 1) / page_size;
      while (++it != ranges_.end() && it->first <= last_page) {
        value.read_mostly &= it->second.read_mostly;
        if (value.preferred != it->second.preferred)
          value.preferred = CU_DEVICE_INVALID;
        for (auto device = value.accessed_by.begin();
             device != value.accessed_by.end();) {
          if (!it->second.accessed_by.count(*device))
            device = value.accessed_by.erase(device);
          else
            ++device;
        }
      }
    }
    if (attribute != CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY) {
      int result = attribute == CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
                       ? int(value.read_mostly)
                       : value.preferred;
      std::memcpy(data, &result, sizeof(result));
    } else {
      // List GPU ordinals before the CPU, then pad unused slots.
      auto *output = static_cast<int *>(data);
      size_t slots = data_size / sizeof(int);
      std::fill_n(output, slots, CU_DEVICE_INVALID);
      size_t copied = 0;
      for (int device : value.accessed_by) {
        if (device >= 0 && copied < slots)
          output[copied++] = device;
      }
      if (value.accessed_by.count(CU_DEVICE_CPU) && copied < slots)
        output[copied] = CU_DEVICE_CPU;
    }
    return CUDA_SUCCESS;
  }
};
