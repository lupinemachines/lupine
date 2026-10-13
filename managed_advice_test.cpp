#include "managed_advice.h"

#include <cassert>
#include <limits>

constexpr size_t page = 4096;
constexpr size_t bytes = 3 * page;

int main() {
  lupine_managed_advice advice;
  auto query = [&](CUmem_range_attribute attr, size_t offset = 0,
                   size_t count = bytes) {
    int value = 99;
    assert(advice.query(&value, sizeof(value), attr, offset, count, page) ==
           CUDA_SUCCESS);
    return value;
  };
  auto apply = [&](CUmem_advise hint, int device, size_t offset = 0,
                   size_t count = bytes) {
    assert(advice.advise(offset, count, page, hint, device) == CUDA_SUCCESS);
  };
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY) == 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION) == CU_DEVICE_INVALID);
  apply(CU_MEM_ADVISE_SET_READ_MOSTLY, 0);
  apply(CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU);
  apply(CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU);
  apply(CU_MEM_ADVISE_SET_ACCESSED_BY, 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY) == 1);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION) == CU_DEVICE_CPU);
  int devices[3] = {99, 99, 99};
  assert(advice.query(devices, sizeof(devices),
                      CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, 0, bytes,
                      page) == CUDA_SUCCESS);
  assert(devices[0] == 0 && devices[1] == CU_DEVICE_CPU &&
         devices[2] == CU_DEVICE_INVALID);
  apply(CU_MEM_ADVISE_UNSET_READ_MOSTLY, 0, page + 1, 1);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY) == 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0, page) == 1);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, page, page) == 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 2 * page, page) == 1);
  apply(CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 1, page, page);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION) == CU_DEVICE_INVALID);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, page, page) == 1);
  apply(CU_MEM_ADVISE_UNSET_ACCESSED_BY, 0, page, page);
  assert(advice.query(devices, sizeof(devices),
                      CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, 0, bytes,
                      page) == CUDA_SUCCESS);
  assert(devices[0] == CU_DEVICE_CPU && devices[1] == CU_DEVICE_INVALID);
  apply(CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION) == CU_DEVICE_INVALID);
  apply(CU_MEM_ADVISE_UNSET_READ_MOSTLY, 0);
  assert(query(CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY) == 0);
  int value;
  assert(advice.query(&value, 1, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0, bytes,
                      page) == CUDA_ERROR_INVALID_VALUE);
  assert(advice.query(nullptr, sizeof(value),
                      CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY, 0, bytes,
                      page) == CUDA_ERROR_INVALID_VALUE);
  assert(advice.advise(0, 0, page, CU_MEM_ADVISE_SET_READ_MOSTLY, 0) ==
         CUDA_ERROR_INVALID_VALUE);
  assert(advice.advise(1, std::numeric_limits<size_t>::max(), page,
                       CU_MEM_ADVISE_SET_READ_MOSTLY,
                       0) == CUDA_ERROR_INVALID_VALUE);
  assert(advice.query(&value, sizeof(value), CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                      0, bytes, 0) == CUDA_ERROR_INVALID_VALUE);
}
