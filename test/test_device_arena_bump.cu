// Run through run_custom_tests.sh. Set LUPINE_TEST_EXHAUST_DEVICE_ARENA=1
// to consume the full VA slot with repeatedly freed allocations, without
// retaining that much physical memory. LUPINE_DEVICE_ARENA=0 checks fallback.
#include <cuda.h>

#include "../rpc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(CUresult result, const char *call) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(result));
    std::exit(1);
  }
}

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static bool in_arena(CUdeviceptr pointer) {
  return pointer >= LUPINE_DEVICE_ARENA_BASE &&
         pointer - LUPINE_DEVICE_ARENA_BASE <
             LUPINE_DEVICE_ARENA_SLOT * LUPINE_VA_ARENA_COUNT;
}

static CUdeviceptr allocate(size_t bytes) {
  CUdeviceptr pointer = 0;
  check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
  return pointer;
}

static void round_trip(CUdeviceptr pointer, size_t bytes) {
  unsigned char input = 42;
  unsigned char output = 0;
  check(cuMemcpyHtoD(pointer + bytes - 1, &input, 1), "cuMemcpyHtoD last byte");
  check(cuMemcpyDtoH(&output, pointer + bytes - 1, 1),
        "cuMemcpyDtoH last byte");
  require(output == input, "allocation's last byte did not round trip");
}

int main() {
  check(cuInit(0), "cuInit");
  CUdevice device = 0;
  CUcontext context = nullptr;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  size_t granule = 0;
  check(cuMemGetAllocationGranularity(&granule, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
  const char *setting = std::getenv("LUPINE_DEVICE_ARENA");
  bool enabled = setting == nullptr || std::strcmp(setting, "0") != 0;

  size_t bytes = granule + 1;
  CUdeviceptr first = allocate(bytes);
  require(in_arena(first) == enabled, "unexpected allocation route");
  CUdeviceptr live = allocate(bytes);
  round_trip(first, bytes);
  check(cuMemFree(first), "cuMemFree first");
  check(cuCtxSynchronize(), "synchronize first free");
  CUdeviceptr next = allocate(bytes);
  if (enabled) {
    require(live == first + 2 * granule && next == live + 2 * granule,
            "allocation reused a freed address or rounded incorrectly");
  }
  round_trip(live, bytes);
  round_trip(next, bytes);

  // Host and managed allocations must still use their own coordinated range,
  // with client CPU mappings and server backing alongside device-only memory.
  void *host = nullptr;
  CUdeviceptr managed = 0;
  check(cuMemAllocHost(&host, sizeof(unsigned int)), "cuMemAllocHost");
  check(cuMemAllocManaged(&managed, sizeof(unsigned int), CU_MEM_ATTACH_GLOBAL),
        "cuMemAllocManaged");
  require(!in_arena(reinterpret_cast<CUdeviceptr>(host)) && !in_arena(managed),
          "host or managed allocation overlapped the device arena");
  *static_cast<unsigned int *>(host) = 123;
  check(cuMemcpyHtoD(live, host, sizeof(unsigned int)), "host to device arena");
  check(cuMemcpyDtoH(reinterpret_cast<void *>(managed), live,
                     sizeof(unsigned int)),
        "device arena to managed mirror");
  check(cuCtxSynchronize(), "synchronize managed mirror");
  require(*reinterpret_cast<unsigned int *>(managed) == 123,
          "host/device/managed round trip failed");
  check(cuMemFreeHost(host), "cuMemFreeHost");
  check(cuMemFree(managed), "cuMemFree managed");
  CUstream stream = nullptr;
  check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  check(cuMemFreeAsync(live, stream), "cuMemFreeAsync");
  check(cuStreamSynchronize(stream), "cuStreamSynchronize");
  check(cuStreamDestroy(stream), "cuStreamDestroy");
  check(cuMemFree(next), "cuMemFree next");
  check(cuCtxSynchronize(), "synchronize all frees");
  CUdeviceptr after_empty = allocate(granule);
  if (enabled) {
    require(after_empty == next + 2 * granule,
            "empty arena reset its virtual address counter");
  }
  check(cuMemFree(after_empty), "cuMemFree after_empty");

  CUdeviceptr small = allocate(512);
  require(!in_arena(small), "small allocation used the device arena");
  round_trip(small, 512);
  check(cuMemFree(small), "cuMemFree small");
  CUdeviceptr oversized = 0;
  require(cuMemAlloc(&oversized, std::numeric_limits<size_t>::max()) !=
              CUDA_SUCCESS,
          "oversized allocation wrapped and succeeded");
  CUdeviceptr after_overflow = allocate(granule);
  if (enabled) {
    require(after_overflow == after_empty + granule,
            "fallback or oversized request advanced the arena");
  }
  check(cuMemFree(after_overflow), "cuMemFree after_overflow");
  check(cuCtxSynchronize(), "synchronize fallback checks");

  if (enabled && std::getenv("LUPINE_TEST_EXHAUST_DEVICE_ARENA") != nullptr) {
    size_t available = 0;
    size_t total = 0;
    check(cuMemGetInfo(&available, &total), "cuMemGetInfo");
    size_t chunk = std::min(available / 4, size_t(8) << 30);
    chunk = chunk / granule * granule;
    require(chunk >= (size_t(256) << 20),
            "too little free memory for churn test");
    CUdeviceptr expected = after_overflow + granule;
    CUdeviceptr slot_end = first + LUPINE_DEVICE_ARENA_SLOT;
    size_t allocations = 0;
    for (; allocations <= LUPINE_DEVICE_ARENA_SLOT / chunk + 1; ++allocations) {
      CUdeviceptr pointer = allocate(chunk);
      if (!in_arena(pointer)) {
        require(chunk > slot_end - expected, "fell back before VA exhaustion");
        round_trip(pointer, chunk);
        check(cuMemFree(pointer), "cuMemFree exhausted fallback");
        break;
      }
      require(pointer == expected, "churn reused or skipped a virtual address");
      expected += chunk;
      check(cuMemFree(pointer), "cuMemFree churn");
      check(cuCtxSynchronize(), "synchronize churn free");
    }
    require(allocations <= LUPINE_DEVICE_ARENA_SLOT / chunk + 1,
            "arena did not fall back at exhaustion");
    // A smaller request may still fit in the tail. A second same-size request
    // must fall back again, rather than wrapping into earlier allocations.
    CUdeviceptr fallback = allocate(chunk);
    require(!in_arena(fallback), "exhausted arena wrapped on the next request");
    check(cuMemFree(fallback), "cuMemFree repeated fallback");
    std::printf("VA exhaustion fell back after %zu allocations of %zu bytes\n",
                allocations, chunk);
  }
  check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
  std::puts("device arena bump and allocation fallback checks passed");
  return 0;
}
