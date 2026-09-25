// LUPINE_TEST_CONNECTIONS 2
// Driver counterpart of streamOrderedAllocation, extended with explicit pools
// and graph allocation properties for the second virtual device.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      std::fprintf(stderr, "%s failed: %d (line %d)\n", #call, result,         \
                   __LINE__);                                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s failed (line %d)\n", #condition, __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  REQUIRE(count >= 2);
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 1));
  CUcontext first = nullptr, second = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&first, 0));
  CHECK(cuDevicePrimaryCtxRetain(&second, device));
  CHECK(cuCtxSetCurrent(first));

  // Pool placement comes from its properties, independent of current context.
  CUmemPoolProps props = {};
  props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;
  CUmemoryPool pool = nullptr;
  CHECK(cuMemPoolCreate(&pool, &props));
  REQUIRE(props.location.id == device);
  CUcontext current = nullptr;
  CHECK(cuCtxGetCurrent(&current));
  REQUIRE(current == first);

  CUmemAccessDesc access = {};
  access.location = props.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK(cuMemPoolSetAccess(pool, &access, 1));
  REQUIRE(access.location.id == device);
  CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  CUmemLocation location = props.location;
  CHECK(cuMemPoolGetAccess(&flags, pool, &location));
  REQUIRE(flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  REQUIRE(location.id == device);

  // A descriptor for another server cannot name a peer in the pool's process.
  const char *servers = std::getenv("LUPINE_SERVER");
  if (servers != nullptr && std::strchr(servers, ',') != nullptr) {
    CUmemAccessDesc peer = access;
    peer.location.id = 0;
    REQUIRE(cuMemPoolSetAccess(pool, &peer, 1) == CUDA_ERROR_INVALID_VALUE);
    REQUIRE(cuMemPoolGetAccess(&flags, pool, &peer.location) ==
            CUDA_ERROR_INVALID_VALUE);
    REQUIRE(peer.location.id == 0);
  }
  CHECK(cuMemPoolGetAccess(&flags, pool, &location));

  CHECK(cuCtxSetCurrent(second));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr pointer = 0;
  CHECK(cuMemAllocFromPoolAsync(&pointer, 4096, pool, stream));
  CHECK(cuMemsetD32Async(pointer, 0x1234, 1, stream));
  CHECK(cuStreamSynchronize(stream));
  unsigned int value = 0;
  CHECK(cuMemcpyDtoH(&value, pointer, sizeof(value)));
  REQUIRE(value == 0x1234);
  CHECK(cuMemFreeAsync(pointer, stream));
  CHECK(cuStreamSynchronize(stream));

  CUgraph graph = nullptr;
  CUgraphNode node = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CUDA_MEM_ALLOC_NODE_PARAMS params = {};
  params.poolProps = props;
  params.accessDescs = &access;
  params.accessDescCount = 1;
  params.bytesize = 4096;
  CHECK(cuGraphAddMemAllocNode(&node, graph, nullptr, 0, &params));
  REQUIRE(params.dptr != 0 && params.accessDescs == &access);
  REQUIRE(params.poolProps.location.id == device &&
          access.location.id == device);

  CHECK(cuCtxSetCurrent(first));
  CUDA_MEM_ALLOC_NODE_PARAMS queried = {};
  const CUmemAccessDesc *saved = nullptr;
  for (int i = 0; i < 4; ++i) {
    CHECK(cuGraphMemAllocNodeGetParams(node, &queried));
    REQUIRE(queried.poolProps.location.id == device);
    REQUIRE(queried.accessDescCount == 1 && queried.accessDescs != nullptr);
    REQUIRE(queried.accessDescs[0].location.id == device);
    REQUIRE(queried.accessDescs[0].flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    if (saved != nullptr)
      REQUIRE(saved[0].location.id == device);
    saved = queried.accessDescs;
  }
  CHECK(cuCtxSetCurrent(second));
  CUgraphExec exec = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CHECK(cuGraphLaunch(exec, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemsetD32(params.dptr, 0x5678, 1));
  CHECK(cuMemcpyDtoH(&value, params.dptr, sizeof(value)));
  REQUIRE(value == 0x5678);
  CHECK(cuMemFree(params.dptr));
  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphDestroy(graph));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemPoolDestroy(pool));
  CHECK(cuDevicePrimaryCtxRelease(device));
  CHECK(cuDevicePrimaryCtxRelease(0));
  std::puts("memory pool and graph device routing: PASS");
}
