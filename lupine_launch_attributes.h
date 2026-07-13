#ifndef LUPINE_LAUNCH_ATTRIBUTES_H
#define LUPINE_LAUNCH_ATTRIBUTES_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda.h>

static constexpr uint32_t LUPINE_MAX_LAUNCH_ATTRIBUTES = 64;
static constexpr size_t LUPINE_LAUNCH_ATTRIBUTE_VALUE_BYTES = 64;

struct lupine_launch_attribute_wire {
  uint32_t id;
  unsigned char value[LUPINE_LAUNCH_ATTRIBUTE_VALUE_BYTES];
};

static inline size_t lupine_launch_attribute_value_size(uint32_t id) {
  switch (id) {
  case 0: // CU_LAUNCH_ATTRIBUTE_IGNORE
    return 0;
  case 1: // CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
    return sizeof(CUaccessPolicyWindow);
  case 2:  // CU_LAUNCH_ATTRIBUTE_COOPERATIVE
  case 3:  // CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY
  case 5:  // CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
  case 6:  // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
  case 8:  // CU_LAUNCH_ATTRIBUTE_PRIORITY
  case 10: // CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN
  case 14: // CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
  case 16: // CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING
    return sizeof(uint32_t);
  case 4:  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
  case 11: // CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION
    return 3 * sizeof(uint32_t);
  case 7:  // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
  case 12: // CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT
  case 13: // CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
    return sizeof(uint64_t) + 2 * sizeof(uint32_t);
  case 9: // CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
    return 2;
  default:
    return SIZE_MAX;
  }
}

#if CUDA_VERSION >= 11080
static inline CUresult
lupine_encode_launch_attribute(const CUlaunchAttribute &attribute,
                               lupine_launch_attribute_wire *wire) {
  if (wire == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const uint32_t id = static_cast<uint32_t>(attribute.id);
  const size_t value_size = lupine_launch_attribute_value_size(id);
  if (value_size == SIZE_MAX || value_size > sizeof(wire->value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  *wire = {};
  wire->id = id;
  memcpy(wire->value, attribute.value.pad, value_size);
  return CUDA_SUCCESS;
}

static inline CUresult
lupine_decode_launch_attribute(const lupine_launch_attribute_wire &wire,
                               CUlaunchAttribute *attribute) {
  if (attribute == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t value_size = lupine_launch_attribute_value_size(wire.id);
  if (value_size == SIZE_MAX || value_size > sizeof(wire.value)) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  *attribute = {};
  attribute->id = static_cast<CUlaunchAttributeID>(wire.id);
  memcpy(attribute->value.pad, wire.value, value_size);
  return CUDA_SUCCESS;
}
#endif

#endif
