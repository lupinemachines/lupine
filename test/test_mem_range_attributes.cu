// Regression coverage for the singular and plural managed-memory range
// attribute APIs. The ACCESSED_BY result is deliberately wider than one value
// so the RPC cannot accidentally pass by copying only a pointer-sized slot.

#include <cuda.h>

#include <cstdio>
#include <cstring>

namespace {

int check(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) {
    return 0;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), name == nullptr ? "unknown" : name);
  return 1;
}

struct RangeAttributes {
  int read_mostly = -99;
  int preferred_location = -99;
  int accessed_by[3] = {-99, -99, -99};
  int last_prefetch_location = -99;
};

} // namespace

int main() {
  int device_count = 0;
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (check(cuInit(0), "cuInit") != 0 ||
      check(cuDeviceGetCount(&device_count), "cuDeviceGetCount") != 0 ||
      device_count == 0 || check(cuDeviceGet(&device, 0), "cuDeviceGet") != 0 ||
      check(cuDevicePrimaryCtxRetain(&context, device),
            "cuDevicePrimaryCtxRetain") != 0 ||
      check(cuCtxSetCurrent(context), "cuCtxSetCurrent") != 0) {
    return 1;
  }

  constexpr size_t allocation_size = 4096;
  CUdeviceptr allocation = 0;
  if (check(cuMemAllocManaged(&allocation, allocation_size,
                              CU_MEM_ATTACH_GLOBAL),
            "cuMemAllocManaged") != 0) {
    cuDevicePrimaryCtxRelease(device);
    return 2;
  }

  CUmem_range_attribute attributes[] = {
      CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
      CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
      CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
      CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
  };
  constexpr size_t attribute_count = sizeof(attributes) / sizeof(attributes[0]);
  size_t data_sizes[] = {sizeof(int), sizeof(int), 3 * sizeof(int),
                         sizeof(int)};

  RangeAttributes individual;
  void *individual_data[] = {
      &individual.read_mostly,
      &individual.preferred_location,
      individual.accessed_by,
      &individual.last_prefetch_location,
  };
  for (size_t i = 0; i < attribute_count; ++i) {
    if (check(cuMemRangeGetAttribute(individual_data[i], data_sizes[i],
                                     attributes[i], allocation,
                                     allocation_size),
              "cuMemRangeGetAttribute") != 0) {
      cuMemFree(allocation);
      cuDevicePrimaryCtxRelease(device);
      return 3;
    }
  }

  RangeAttributes plural;
  void *plural_data[] = {
      &plural.read_mostly,
      &plural.preferred_location,
      plural.accessed_by,
      &plural.last_prefetch_location,
  };
  if (check(cuMemRangeGetAttributes(plural_data, data_sizes, attributes,
                                    attribute_count, allocation,
                                    allocation_size),
            "cuMemRangeGetAttributes") != 0) {
    cuMemFree(allocation);
    cuDevicePrimaryCtxRelease(device);
    return 4;
  }

  for (size_t i = 0; i < attribute_count; ++i) {
    if (std::memcmp(individual_data[i], plural_data[i], data_sizes[i]) != 0) {
      std::fprintf(stderr, "attribute %zu differs between singular and plural "
                           "queries\n",
                   i);
      cuMemFree(allocation);
      cuDevicePrimaryCtxRelease(device);
      return 5;
    }
  }

  if (check(cuMemFree(allocation), "cuMemFree") != 0 ||
      check(cuDevicePrimaryCtxRelease(device),
            "cuDevicePrimaryCtxRelease") != 0) {
    return 6;
  }

  std::printf("managed-memory range attributes match (%d, %d, [%d, %d, %d], "
              "%d)\n",
              individual.read_mostly, individual.preferred_location,
              individual.accessed_by[0], individual.accessed_by[1],
              individual.accessed_by[2], individual.last_prefetch_location);
  return 0;
}
