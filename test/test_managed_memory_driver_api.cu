// Direct driver-API coverage for managed allocations. The runtime API often
// resolves these functions through cuGetProcAddress, so exercising the exported
// legacy and v2 symbols separately catches ABI/version routing regressions.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>

namespace {

constexpr size_t kElements = 1u << 20;
constexpr size_t kBytes = kElements * sizeof(unsigned int);

using LegacyPrefetch = CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUdevice,
                                           CUstream);
using LegacyAdvise = CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUmem_advise,
                                         CUdevice);
using GetProcAddressV1 = CUresult(CUDAAPI *)(const char *, void **, int,
                                             cuuint64_t);
#if CUDA_VERSION >= 12020
using LocationPrefetch = CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUmemLocation,
                                             unsigned int, CUstream);
using LocationAdvise = CUresult(CUDAAPI *)(CUdeviceptr, size_t, CUmem_advise,
                                           CUmemLocation);
#endif

__global__ void xor_values(unsigned int *values, size_t count,
                           unsigned int mask) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] ^= mask;
  }
}

int check_driver(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) {
    return 0;
  }
  const char *name = nullptr;
  cuGetErrorName(status, &name);
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), name == nullptr ? "unknown" : name);
  return 1;
}

int check_runtime(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), cudaGetErrorString(status));
  return 1;
}

int expect(bool condition, const char *message) {
  if (condition) {
    return 0;
  }
  std::fprintf(stderr, "FAIL: %s\n", message);
  return 1;
}

template <typename Function> Function load_symbol(const char *name) {
  dlerror();
  void *symbol = dlsym(RTLD_DEFAULT, name);
  const char *error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    std::fprintf(stderr, "dlsym(%s) failed: %s\n", name,
                 error == nullptr ? "symbol not found" : error);
    return nullptr;
  }
  return reinterpret_cast<Function>(symbol);
}

} // namespace

int main() {
  if (check_driver(cuInit(0), "cuInit") != 0) {
    return 1;
  }

  int device_count = 0;
  CUdevice device = 0;
  if (check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount") != 0 ||
      expect(device_count > 0, "no CUDA devices visible") != 0 ||
      check_driver(cuDeviceGet(&device, 0), "cuDeviceGet") != 0 ||
      check_runtime(cudaSetDevice(0), "cudaSetDevice") != 0 ||
      check_runtime(cudaFree(nullptr), "cudaFree(nullptr)") != 0) {
    return 2;
  }

  CUdeviceptr allocation = 0;
  if (check_driver(cuMemAllocManaged(&allocation, kBytes, CU_MEM_ATTACH_GLOBAL),
                   "cuMemAllocManaged") != 0) {
    return 3;
  }
  auto *host = reinterpret_cast<unsigned int *>(allocation);

  int is_managed = 0;
  CUmemorytype memory_type = CU_MEMORYTYPE_HOST;
  void *host_alias = nullptr;
  CUdeviceptr device_alias = 0;
  if (check_driver(cuPointerGetAttribute(&is_managed,
                                         CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                         allocation),
                   "cuPointerGetAttribute(IS_MANAGED)") != 0 ||
      check_driver(cuPointerGetAttribute(&memory_type,
                                         CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         allocation),
                   "cuPointerGetAttribute(MEMORY_TYPE)") != 0 ||
      check_driver(cuPointerGetAttribute(&host_alias,
                                         CU_POINTER_ATTRIBUTE_HOST_POINTER,
                                         allocation),
                   "cuPointerGetAttribute(HOST_POINTER)") != 0 ||
      check_driver(cuPointerGetAttribute(&device_alias,
                                         CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                         allocation),
                   "cuPointerGetAttribute(DEVICE_POINTER)") != 0 ||
      expect(is_managed == 1, "driver allocation is not marked managed") != 0 ||
      expect(memory_type == CU_MEMORYTYPE_DEVICE,
             "driver allocation has the wrong memory type") != 0 ||
      expect(host_alias == host,
             "driver allocation has the wrong host alias") != 0 ||
      expect(device_alias != 0, "driver allocation has no device alias") != 0) {
    cuMemFree(allocation);
    return 4;
  }

  CUpointer_attribute queried_attributes[] = {
      CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
      CU_POINTER_ATTRIBUTE_HOST_POINTER};
  int queried_managed = 0;
  CUmemorytype queried_type = CU_MEMORYTYPE_HOST;
  void *queried_host = nullptr;
  void *attribute_values[] = {&queried_managed, &queried_type, &queried_host};
  if (check_driver(cuPointerGetAttributes(3, queried_attributes,
                                          attribute_values, allocation),
                   "cuPointerGetAttributes") != 0 ||
      expect(queried_managed == 1 && queried_type == CU_MEMORYTYPE_DEVICE &&
                 queried_host == host,
             "batched pointer attributes disagree with single queries") != 0) {
    cuMemFree(allocation);
    return 5;
  }

  for (size_t i = 0; i < kElements; ++i) {
    host[i] = static_cast<unsigned int>(i * 13u + 7u);
  }

  CUstream stream = nullptr;
  if (check_driver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                   "cuStreamCreate") != 0 ||
      check_driver(
          cuStreamAttachMemAsync(stream, allocation, 0, CU_MEM_ATTACH_SINGLE),
          "cuStreamAttachMemAsync") != 0) {
    cuMemFree(allocation);
    return 6;
  }

  LegacyPrefetch legacy_symbol =
      load_symbol<LegacyPrefetch>("cuMemPrefetchAsync");
  GetProcAddressV1 get_proc_address =
      load_symbol<GetProcAddressV1>("cuGetProcAddress");
  void *legacy_address = nullptr;
  if (legacy_symbol == nullptr || get_proc_address == nullptr ||
      check_driver(
          get_proc_address("cuMemPrefetchAsync", &legacy_address, 11080, 0),
          "cuGetProcAddress(cuMemPrefetchAsync, 11.8)") != 0 ||
      expect(legacy_address == reinterpret_cast<void *>(legacy_symbol),
             "cuGetProcAddress did not return the legacy prefetch ABI") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 7;
  }
  auto legacy = reinterpret_cast<LegacyPrefetch>(legacy_address);
  LegacyAdvise legacy_advise = load_symbol<LegacyAdvise>("cuMemAdvise");
  void *legacy_advise_address = nullptr;
  if (legacy_advise == nullptr ||
      check_driver(
          get_proc_address("cuMemAdvise", &legacy_advise_address, 11080, 0),
          "cuGetProcAddress(cuMemAdvise, 11.8)") != 0 ||
      expect(legacy_advise_address == reinterpret_cast<void *>(legacy_advise),
             "cuGetProcAddress did not return the legacy advise ABI") != 0 ||
      check_driver(
          reinterpret_cast<LegacyAdvise>(legacy_advise_address)(
              allocation, kBytes, CU_MEM_ADVISE_SET_READ_MOSTLY, CU_DEVICE_CPU),
          "cuMemAdvise(SET_READ_MOSTLY)") != 0 ||
      check_driver(reinterpret_cast<LegacyAdvise>(legacy_advise_address)(
                       allocation, kBytes, CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                       CU_DEVICE_CPU),
                   "cuMemAdvise(UNSET_READ_MOSTLY)") != 0 ||
      check_driver(legacy(allocation + 17 * sizeof(unsigned int),
                          kBytes - 31 * sizeof(unsigned int), device, stream),
                   "cuMemPrefetchAsync(device)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 7;
  }

  constexpr unsigned int kFirstMask = 0xa5a55a5au;
  xor_values<<<(kElements + 255) / 256, 256, 0,
               reinterpret_cast<cudaStream_t>(stream)>>>(host, kElements,
                                                         kFirstMask);
  if (check_runtime(cudaGetLastError(), "first xor_values launch") != 0 ||
      check_driver(legacy(allocation, kBytes, CU_DEVICE_CPU, stream),
                   "cuMemPrefetchAsync(host)") != 0 ||
      check_driver(cuStreamSynchronize(stream),
                   "cuStreamSynchronize(legacy)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 8;
  }

  for (size_t i = 0; i < kElements; ++i) {
    unsigned int expected =
        static_cast<unsigned int>(i * 13u + 7u) ^ kFirstMask;
    if (host[i] != expected) {
      std::fprintf(stderr,
                   "legacy prefetch mismatch at %zu: got %#x expected %#x\n", i,
                   host[i], expected);
      cuStreamDestroy(stream);
      cuMemFree(allocation);
      return 9;
    }
  }

#if CUDA_VERSION >= 12020
  LocationPrefetch location_prefetch =
      load_symbol<LocationPrefetch>("cuMemPrefetchAsync_v2");
  LocationAdvise location_advise =
      load_symbol<LocationAdvise>("cuMemAdvise_v2");
  void *location_address = nullptr;
  void *location_advise_address = nullptr;
  CUmemLocation gpu_location = {};
  gpu_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  gpu_location.id = device;
  if (location_prefetch == nullptr || location_advise == nullptr ||
      check_driver(
          get_proc_address("cuMemPrefetchAsync", &location_address, 12020, 0),
          "cuGetProcAddress(cuMemPrefetchAsync, 12.2)") != 0 ||
      expect(location_address == reinterpret_cast<void *>(location_prefetch),
             "cuGetProcAddress did not return the v2 prefetch ABI") != 0 ||
      check_driver(
          get_proc_address("cuMemAdvise", &location_advise_address, 12020, 0),
          "cuGetProcAddress(cuMemAdvise, 12.2)") != 0 ||
      expect(location_advise_address ==
                 reinterpret_cast<void *>(location_advise),
             "cuGetProcAddress did not return the v2 advise ABI") != 0 ||
      check_driver(reinterpret_cast<LocationAdvise>(location_advise_address)(
                       allocation, kBytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                       gpu_location),
                   "cuMemAdvise_v2(SET_PREFERRED_LOCATION)") != 0 ||
      check_driver(reinterpret_cast<LocationPrefetch>(location_address)(
                       allocation, kBytes, gpu_location, 0, stream),
                   "cuMemPrefetchAsync_v2(device)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 10;
  }

  constexpr unsigned int kSecondMask = 0x13579bdfu;
  xor_values<<<(kElements + 255) / 256, 256, 0,
               reinterpret_cast<cudaStream_t>(stream)>>>(host, kElements,
                                                         kSecondMask);
  CUmemLocation host_location = {};
  host_location.type = CU_MEM_LOCATION_TYPE_HOST;
  if (check_runtime(cudaGetLastError(), "second xor_values launch") != 0 ||
      check_driver(
          location_prefetch(allocation, kBytes, host_location, 0, stream),
          "cuMemPrefetchAsync_v2(host)") != 0 ||
      check_driver(cuStreamSynchronize(stream), "cuStreamSynchronize(v2)") !=
          0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 11;
  }

  for (size_t i = 0; i < kElements; ++i) {
    unsigned int expected =
        (static_cast<unsigned int>(i * 13u + 7u) ^ kFirstMask) ^ kSecondMask;
    if (host[i] != expected) {
      std::fprintf(stderr,
                   "v2 prefetch mismatch at %zu: got %#x expected %#x\n", i,
                   host[i], expected);
      cuStreamDestroy(stream);
      cuMemFree(allocation);
      return 12;
    }
  }
#endif

  constexpr size_t kSyncSetOffset = 200000;
  constexpr size_t kSyncSetElements = 1024;
  constexpr unsigned int kSyncSetValue = 0xc001d00du;
  if (check_driver(
          cuMemsetD32(allocation + kSyncSetOffset * sizeof(unsigned int),
                      kSyncSetValue, kSyncSetElements),
          "cuMemsetD32(managed)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 13;
  }
  for (size_t i = 0; i < kSyncSetElements; ++i) {
    if (host[kSyncSetOffset + i] != kSyncSetValue) {
      std::fprintf(stderr,
                   "synchronous managed memset mismatch at %zu: got %#x\n", i,
                   host[kSyncSetOffset + i]);
      cuStreamDestroy(stream);
      cuMemFree(allocation);
      return 14;
    }
  }

  constexpr size_t kSetOffset = 300000;
  constexpr size_t kSetElements = 2048;
  constexpr unsigned int kSetValue = 0xdecafbadu;
  if (check_driver(
          cuMemsetD32Async(allocation + kSetOffset * sizeof(unsigned int),
                           kSetValue, kSetElements, stream),
          "cuMemsetD32Async(managed)") != 0 ||
      check_driver(cuStreamSynchronize(stream),
                   "cuStreamSynchronize(memset)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 15;
  }
  for (size_t i = 0; i < kSetElements; ++i) {
    if (host[kSetOffset + i] != kSetValue) {
      std::fprintf(stderr, "managed memset mismatch at %zu: got %#x\n", i,
                   host[kSetOffset + i]);
      cuStreamDestroy(stream);
      cuMemFree(allocation);
      return 16;
    }
  }

  constexpr size_t kCopyOffset = 600000;
  constexpr size_t kCopyElements = 512;
  unsigned int copy_input[kCopyElements];
  unsigned int copy_output[kCopyElements] = {};
  for (size_t i = 0; i < kCopyElements; ++i) {
    copy_input[i] = static_cast<unsigned int>(i * 31u + 0x1234u);
  }
  if (check_driver(cuMemcpyHtoD(allocation + kCopyOffset * sizeof(unsigned int),
                                copy_input, sizeof(copy_input)),
                   "cuMemcpyHtoD(managed)") != 0 ||
      check_driver(cuMemcpyDtoH(copy_output,
                                allocation + kCopyOffset * sizeof(unsigned int),
                                sizeof(copy_output)),
                   "cuMemcpyDtoH(managed)") != 0) {
    cuStreamDestroy(stream);
    cuMemFree(allocation);
    return 17;
  }
  for (size_t i = 0; i < kCopyElements; ++i) {
    if (host[kCopyOffset + i] != copy_input[i] ||
        copy_output[i] != copy_input[i]) {
      std::fprintf(stderr,
                   "driver managed copy mismatch at %zu: managed=%#x host=%#x "
                   "expected=%#x\n",
                   i, host[kCopyOffset + i], copy_output[i], copy_input[i]);
      cuStreamDestroy(stream);
      cuMemFree(allocation);
      return 18;
    }
  }

  if (check_driver(cuStreamDestroy(stream), "cuStreamDestroy") != 0 ||
      check_driver(cuMemFree(allocation), "cuMemFree") != 0) {
    return 19;
  }

  std::printf("RESULT: PASS driver managed-memory APIs\n");
  return 0;
}
