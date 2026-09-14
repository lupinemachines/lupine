// Exercises the generated HIP forwarding surface end to end: device queries,
// memory, copies, streams, events, graphs, arrays, textures and virtual memory.
// Host-only, so it builds with the host compiler against either the shim or
// the native runtime; LUPINE_HIP_SHIM=1 enables the checks that only hold for
// the shim (stubs report hipErrorNotSupported).
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t check_error = (call);                                           \
    if (check_error != hipSuccess) {                                           \
      fprintf(stderr, "%s:%d: %s returned %d\n", __FILE__, __LINE__, #call,    \
              static_cast<int>(check_error));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: expected %s\n", __FILE__, __LINE__, #condition); \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t kBytes = 1 << 20;

int check_bytes(void *device, unsigned char expected) {
  std::vector<unsigned char> host(kBytes,
                                  static_cast<unsigned char>(~expected));
  CHECK(hipMemcpyDtoH(host.data(), device, kBytes));
  for (size_t i = 0; i < kBytes; ++i) {
    EXPECT(host[i] == expected);
  }
  return 0;
}

int test_device() {
  int count = 0;
  CHECK(hipGetDeviceCount(&count));
  EXPECT(count > 0);
  CHECK(hipSetDevice(0));
  int device = -1;
  CHECK(hipGetDevice(&device));
  EXPECT(device == 0);

  int multiprocessors = 0;
  CHECK(hipDeviceGetAttribute(&multiprocessors,
                              hipDeviceAttributeMultiprocessorCount, 0));
  EXPECT(multiprocessors > 0);
  char name[256] = {};
  CHECK(hipDeviceGetName(name, sizeof(name), 0));
  EXPECT(name[0] != '\0');
  int major = 0, minor = 0;
  CHECK(hipDeviceComputeCapability(&major, &minor, 0));
  EXPECT(major > 0);
  hipUUID uuid = {};
  CHECK(hipDeviceGetUuid(&uuid, 0));

  char bus_id[64] = {};
  CHECK(hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), 0));
  EXPECT(bus_id[0] != '\0');
  int by_bus = -1;
  CHECK(hipDeviceGetByPCIBusId(&by_bus, bus_id));
  EXPECT(by_bus == 0);

  hipDeviceProp_t prop = {};
  CHECK(hipGetDeviceProperties(&prop, 0));
  EXPECT(strcmp(prop.name, name) == 0);
  int chosen = -1;
  CHECK(hipChooseDevice(&chosen, &prop));
  EXPECT(chosen == 0);

  int runtime_version = 0;
  CHECK(hipRuntimeGetVersion(&runtime_version));
  EXPECT(runtime_version > 0);
  size_t free_bytes = 0, total_bytes = 0;
  CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  EXPECT(total_bytes >= free_bytes && total_bytes > 0);
  size_t stack = 0;
  CHECK(hipDeviceGetLimit(&stack, hipLimitStackSize));
  int least = 0, greatest = 0;
  CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
  int can_access = -1;
  CHECK(hipDeviceCanAccessPeer(&can_access, 0, 0));
  EXPECT(can_access == 0);
  return 0;
}

int test_memory() {
  void *a = nullptr;
  void *b = nullptr;
  CHECK(hipMalloc(&a, kBytes));
  CHECK(hipMalloc(&b, kBytes));
  EXPECT(a != nullptr && b != nullptr && a != b);

  CHECK(hipMemset(a, 0x5a, kBytes));
  if (check_bytes(a, 0x5a))
    return 1;

  std::vector<unsigned char> pattern(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    pattern[i] = static_cast<unsigned char>(i * 7 + 3);
  }
  CHECK(hipMemcpyHtoD(a, pattern.data(), kBytes));
  CHECK(hipMemcpyDtoD(b, a, kBytes));
  std::vector<unsigned char> readback(kBytes);
  CHECK(hipMemcpyDtoH(readback.data(), b, kBytes));
  EXPECT(readback == pattern);

  CHECK(hipMemsetD32(a, 0x01020304, kBytes / 4));
  CHECK(hipMemcpyDtoH(readback.data(), a, kBytes));
  EXPECT(readback[0] == 0x04 && readback[1] == 0x03 && readback[2] == 0x02 &&
         readback[3] == 0x01);

  CHECK(hipMemcpyPeer(b, 0, a, 0, kBytes));
  CHECK(hipMemcpyDtoH(pattern.data(), b, kBytes));
  EXPECT(readback == pattern);

  hipDeviceptr_t base = nullptr;
  size_t range = 0;
  CHECK(hipMemGetAddressRange(&base, &range, static_cast<char *>(a) + 4096));
  EXPECT(base == a && range >= kBytes);
  size_t size = 0;
  CHECK(hipMemPtrGetInfo(a, &size));
  EXPECT(size >= kBytes);
  hipPointerAttribute_t attributes = {};
  CHECK(hipPointerGetAttributes(&attributes, a));
  EXPECT(attributes.type == hipMemoryTypeDevice);
  EXPECT(attributes.devicePointer == a);

  void *async = nullptr;
  CHECK(hipMallocAsync(&async, kBytes, nullptr));
  CHECK(hipMemsetAsync(async, 1, kBytes, nullptr));
  CHECK(hipFreeAsync(async, nullptr));
  CHECK(hipStreamSynchronize(nullptr));
  hipMemPool_t pool = nullptr;
  CHECK(hipDeviceGetDefaultMemPool(&pool, 0));
  CHECK(hipMemPoolTrimTo(pool, 0));

  CHECK(hipFree(a));
  CHECK(hipFree(b));
  return 0;
}

int test_streams_and_events() {
  void *a = nullptr;
  void *b = nullptr;
  CHECK(hipMalloc(&a, kBytes));
  CHECK(hipMalloc(&b, kBytes));
  hipStream_t stream = nullptr;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  unsigned int flags = 0;
  CHECK(hipStreamGetFlags(stream, &flags));
  EXPECT(flags == hipStreamNonBlocking);
  int priority = 1;
  CHECK(hipStreamGetPriority(stream, &priority));
  EXPECT(priority == 0);

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreateWithFlags(&stop, hipEventDefault));
  CHECK(hipEventRecord(start, stream));
  CHECK(hipMemsetAsync(a, 0x42, kBytes, stream));
  CHECK(hipMemcpyDtoDAsync(b, a, kBytes, stream));
  CHECK(hipMemsetD8Async(a, 0x24, kBytes, stream));
  CHECK(hipEventRecord(stop, stream));
  CHECK(hipEventSynchronize(stop));
  CHECK(hipEventQuery(stop));
  float ms = -1.0f;
  CHECK(hipEventElapsedTime(&ms, start, stop));
  EXPECT(ms >= 0.0f);
  CHECK(hipStreamWaitEvent(stream, stop, 0));
  CHECK(hipStreamQuery(stream));
  CHECK(hipStreamSynchronize(stream));
  if (check_bytes(b, 0x42))
    return 1;
  if (check_bytes(a, 0x24))
    return 1;

  CHECK(hipEventDestroy(start));
  CHECK(hipEventDestroy(stop));
  CHECK(hipStreamDestroy(stream));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  return 0;
}

int test_graphs() {
  void *a = nullptr;
  CHECK(hipMalloc(&a, kBytes));

  hipGraph_t graph = nullptr;
  CHECK(hipGraphCreate(&graph, 0));
  hipMemsetParams params = {};
  params.dst = a;
  params.elementSize = 1;
  params.height = 1;
  params.pitch = 0;
  params.value = 7;
  params.width = kBytes;
  hipGraphNode_t memset_node = nullptr;
  CHECK(hipGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &params));
  hipGraphNode_t empty_node = nullptr;
  CHECK(hipGraphAddEmptyNode(&empty_node, graph, &memset_node, 1));

  size_t node_count = 0;
  CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
  EXPECT(node_count == 2);
  std::vector<hipGraphNode_t> nodes(node_count);
  CHECK(hipGraphGetNodes(graph, nodes.data(), &node_count));
  EXPECT(node_count == 2);
  hipGraphNodeType type = hipGraphNodeTypeEmpty;
  CHECK(hipGraphNodeGetType(memset_node, &type));
  EXPECT(type == hipGraphNodeTypeMemset);
  size_t dependency_count = 0;
  CHECK(hipGraphNodeGetDependencies(empty_node, nullptr, &dependency_count));
  EXPECT(dependency_count == 1);
  hipMemsetParams stored = {};
  CHECK(hipGraphMemsetNodeGetParams(memset_node, &stored));
  EXPECT(stored.dst == a && stored.value == 7 && stored.width == kBytes);

  hipGraphExec_t exec = nullptr;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(exec, nullptr));
  CHECK(hipStreamSynchronize(nullptr));
  if (check_bytes(a, 7))
    return 1;
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphInstantiateWithFlags(&exec, graph, 0));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));

  hipStream_t stream = nullptr;
  CHECK(hipStreamCreate(&stream));
  CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  CHECK(hipMemsetAsync(a, 9, kBytes, stream));
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  CHECK(hipStreamIsCapturing(stream, &status));
  EXPECT(status == hipStreamCaptureStatusActive);
  hipGraph_t captured = nullptr;
  CHECK(hipStreamEndCapture(stream, &captured));
  CHECK(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(exec, stream));
  CHECK(hipStreamSynchronize(stream));
  if (check_bytes(a, 9))
    return 1;
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(captured));
  CHECK(hipStreamDestroy(stream));
  CHECK(hipFree(a));
  return 0;
}

int test_arrays_and_textures() {
  hipChannelFormatDesc desc = {};
  desc.x = 32;
  desc.f = hipChannelFormatKindFloat;
  hipArray_t array = nullptr;
  hipError_t supported = hipMallocArray(&array, &desc, 64, 0, 0);
  if (supported != hipSuccess) {
    // Image allocation is not available on every ROCm platform (WSL).
    printf("arrays: hipMallocArray returned %d here, skipped\n",
           static_cast<int>(supported));
    return 0;
  }
  float source[64];
  for (int i = 0; i < 64; ++i) {
    source[i] = static_cast<float>(i) * 0.5f;
  }
  CHECK(hipMemcpyHtoA(array, 0, source, sizeof(source)));
  float readback[64] = {};
  CHECK(hipMemcpyAtoH(readback, array, 0, sizeof(readback)));
  EXPECT(memcmp(readback, source, sizeof(source)) == 0);

  hipChannelFormatDesc stored = {};
  hipExtent extent = {};
  unsigned int flags = 1;
  CHECK(hipArrayGetInfo(&stored, &extent, &flags, array));
  EXPECT(stored.x == 32 && stored.f == hipChannelFormatKindFloat);
  EXPECT(extent.width == 64 && flags == 0);

  hipResourceDesc resource = {};
  resource.resType = hipResourceTypeArray;
  resource.res.array.array = array;
  hipTextureDesc texture = {};
  texture.readMode = hipReadModeElementType;
  hipTextureObject_t object = 0;
  CHECK(hipCreateTextureObject(&object, &resource, &texture, nullptr));
  hipResourceDesc queried = {};
  CHECK(hipGetTextureObjectResourceDesc(&queried, object));
  EXPECT(queried.resType == hipResourceTypeArray &&
         queried.res.array.array == array);
  CHECK(hipDestroyTextureObject(object));
  CHECK(hipFreeArray(array));
  return 0;
}

int test_virtual_memory() {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = 0;
  size_t granularity = 0;
  hipError_t supported = hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityMinimum);
  if (supported == hipErrorNotSupported) {
    printf("virtual memory management: not supported here, skipped\n");
    return 0;
  }
  CHECK(supported);
  EXPECT(granularity > 0);

  hipMemGenericAllocationHandle_t handle = 0;
  CHECK(hipMemCreate(&handle, granularity, &prop, 0));
  void *address = nullptr;
  CHECK(hipMemAddressReserve(&address, granularity, 0, nullptr, 0));
  CHECK(hipMemMap(address, granularity, 0, handle, 0));
  hipMemAccessDesc access = {};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  CHECK(hipMemSetAccess(address, granularity, &access, 1));
  CHECK(hipMemset(address, 3, granularity));
  std::vector<unsigned char> host(granularity);
  CHECK(hipMemcpyDtoH(host.data(), address, granularity));
  for (unsigned char byte : host) {
    EXPECT(byte == 3);
  }
  CHECK(hipMemUnmap(address, granularity));
  CHECK(hipMemRelease(handle));
  CHECK(hipMemAddressFree(address, granularity));
  return 0;
}

int test_errors() {
  int value = 0;
  EXPECT(hipDeviceGetAttribute(&value, static_cast<hipDeviceAttribute_t>(-1),
                               0) == hipErrorInvalidValue);
  EXPECT(hipPeekAtLastError() == hipErrorInvalidValue);
  EXPECT(hipGetLastError() == hipErrorInvalidValue);
  EXPECT(hipGetLastError() == hipSuccess);

  if (getenv("LUPINE_HIP_SHIM") != nullptr) {
    // Stubs answer on the client; the sticky error lives on the server.
    void *host = nullptr;
    EXPECT(hipHostMalloc(&host, 16, 0) == hipErrorNotSupported);
    EXPECT(hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToDevice) ==
           hipErrorNotSupported);
    EXPECT(hipGetLastError() == hipSuccess);
  }
  return 0;
}

} // namespace

int main() {
  struct {
    const char *name;
    int (*run)();
  } const tests[] = {
      {"device", test_device},
      {"memory", test_memory},
      {"streams and events", test_streams_and_events},
      {"graphs", test_graphs},
      {"arrays and textures", test_arrays_and_textures},
      {"virtual memory", test_virtual_memory},
      {"errors", test_errors},
  };
  for (const auto &test : tests) {
    if (test.run() != 0) {
      fprintf(stderr, "FAILED: %s\n", test.name);
      return 1;
    }
    printf("ok: %s\n", test.name);
  }
  int status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    fprintf(stderr, "hipDeviceSynchronize returned %d\n", status);
    return 1;
  }
  status = hipDeviceReset();
  if (status != hipSuccess) {
    fprintf(stderr, "hipDeviceReset returned %d\n", status);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
