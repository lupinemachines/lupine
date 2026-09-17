// Drives the virtual memory management calls a library makes when it frees
// memory it mapped itself: retaining the allocation handle from an address and
// asking for the range behind it, with either output left null.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_DRIVER(call)                                                     \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  CHECK_CUDA(cudaSetDevice(0));
  CUdevice device = 0;
  CHECK_DRIVER(cuDeviceGet(&device, 0));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  size_t granularity = 0;
  CHECK_DRIVER(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  EXPECT(granularity > 0);
  size_t size = granularity;

  CUmemGenericAllocationHandle handle = 0;
  CHECK_DRIVER(cuMemCreate(&handle, size, &prop, 0));
  CUdeviceptr pointer = 0;
  CHECK_DRIVER(cuMemAddressReserve(&pointer, size, 0, 0, 0));
  CHECK_DRIVER(cuMemMap(pointer, size, 0, handle, 0));
  CUmemAccessDesc access = {};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_DRIVER(cuMemSetAccess(pointer, size, &access, 1));
  CHECK_CUDA(cudaMemset((void *)pointer, 0, size));
  CHECK_CUDA(cudaDeviceSynchronize());

  CUmemGenericAllocationHandle retained = 0;
  CHECK_DRIVER(cuMemRetainAllocationHandle(&retained, (void *)pointer));
  EXPECT(retained != 0);

  size_t range = 0;
  CHECK_DRIVER(cuMemGetAddressRange(nullptr, &range, pointer));
  EXPECT(range == size);
  CUdeviceptr base = 0;
  CHECK_DRIVER(cuMemGetAddressRange(&base, nullptr, pointer + size / 2));
  EXPECT(base == pointer);
  CUdeviceptr both_base = 0;
  size_t both_size = 0;
  CHECK_DRIVER(cuMemGetAddressRange(&both_base, &both_size, pointer));
  EXPECT(both_base == pointer);
  EXPECT(both_size == size);
  printf("address range: passed\n");

  CHECK_DRIVER(cuMemRelease(retained));
  CHECK_DRIVER(cuMemUnmap(pointer, size));
  CHECK_DRIVER(cuMemRelease(handle));
  CHECK_DRIVER(cuMemAddressFree(pointer, size));
  printf("vmm address range: all checks passed\n");
  return 0;
}
