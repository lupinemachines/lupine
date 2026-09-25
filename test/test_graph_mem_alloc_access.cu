// Driver-only counterpart of NVIDIA graphMemoryNodes, with explicit access
// descriptors and GetParams lifetime checks. No GPU kernels/toolkit PTX needed.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult result = (call);                                                   \
    if (result != CUDA_SUCCESS) {                                              \
      std::fprintf(stderr, "%s failed: %d (line %d)\n", #call, result, __LINE__); \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s failed (line %d)\n", #condition, __LINE__);        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool matches(const CUmemAccessDesc *access, CUdevice device) {
  return access != nullptr &&
         access->location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
         access->location.id == device &&
         access->flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

static int test_alloc(CUdevice device, bool explicit_access) {
  CUgraph graph = nullptr;
  CUgraphNode node = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_MEM_ALLOC_NODE_PARAMS params = {};
  params.poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  params.poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  params.poolProps.location.id = device;
  params.bytesize = 4096;
  params.accessDescs = explicit_access ? &access : nullptr;
  params.accessDescCount = explicit_access ? 1 : 0;
  CHECK(cuGraphAddMemAllocNode(&node, graph, nullptr, 0, &params));
  REQUIRE(params.accessDescs == (explicit_access ? &access : nullptr));
  REQUIRE(params.accessDescCount == (explicit_access ? 1u : 0u));
  REQUIRE(matches(&access, device));
  REQUIRE(params.dptr != 0);

  // GetParams is output-only: it must ignore any previous count or pointer.
  CUDA_MEM_ALLOC_NODE_PARAMS queried;
  std::memset(&queried, 0xa5, sizeof(queried));
  CHECK(cuGraphMemAllocNodeGetParams(node, &queried));
  REQUIRE(queried.bytesize == params.bytesize && queried.dptr == params.dptr);
  REQUIRE(queried.poolProps.location.id == device);
  REQUIRE(queried.accessDescCount == params.accessDescCount);
  if (explicit_access)
    REQUIRE(matches(queried.accessDescs, device));
  const CUmemAccessDesc *saved = queried.accessDescs;

  // Reusing an output struct for another node must not invalidate saved arrays.
  CUgraph other_graph = nullptr;
  CUgraphNode other_node = nullptr;
  CHECK(cuGraphCreate(&other_graph, 0));
  CUDA_MEM_ALLOC_NODE_PARAMS other = params;
  other.accessDescs = nullptr;
  other.accessDescCount = 0;
  CHECK(cuGraphAddMemAllocNode(&other_node, other_graph, nullptr, 0, &other));
  CHECK(cuGraphMemAllocNodeGetParams(other_node, &queried));
  CHECK(cuGraphDestroy(other_graph));
  for (int i = 0; i < 16; ++i) {
    CHECK(cuGraphMemAllocNodeGetParams(node, &queried));
    if (explicit_access)
      REQUIRE(matches(saved, device) && matches(queried.accessDescs, device));
  }

  // Exercise the allocation, free it outside the graph, and replay.
  CUgraphExec exec = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  for (unsigned int expected = 1; expected <= 3; ++expected) {
    CHECK(cuGraphLaunch(exec, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemsetD32(params.dptr, expected, 1));
    unsigned int got = 0;
    CHECK(cuMemcpyDtoH(&got, params.dptr, sizeof(got)));
    REQUIRE(got == expected);
    CHECK(cuMemFree(params.dptr));
  }
  CHECK(cuStreamDestroy(stream));
  CHECK(cuGraphExecDestroy(exec));
  if (explicit_access)
    REQUIRE(matches(saved, device));
  CHECK(cuGraphDestroy(graph));
  return 0;
}

static int test_batch_params(CUcontext context) {
  CUdeviceptr value = 0;
  CHECK(cuMemAlloc(&value, sizeof(unsigned int)));
  CUgraph graph = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CUstreamBatchMemOpParams ops[2] = {};
  for (unsigned int i = 0; i < 2; ++i) {
    ops[i].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[i].writeValue.address = value;
    ops[i].writeValue.value = i + 1;
  }
  CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
  params.ctx = context;
  params.count = 1;
  params.paramArray = ops;
  CUgraphNode first = nullptr, second = nullptr;
  CHECK(cuGraphAddBatchMemOpNode(&first, graph, nullptr, 0, &params));
  params.paramArray = ops + 1;
  CHECK(cuGraphAddBatchMemOpNode(&second, graph, nullptr, 0, &params));

  CUDA_BATCH_MEM_OP_NODE_PARAMS queried = {};
  CHECK(cuGraphBatchMemOpNodeGetParams(first, &queried));
  const auto *saved = queried.paramArray;
  REQUIRE(saved != nullptr && saved[0].writeValue.value == 1);
  CHECK(cuGraphBatchMemOpNodeGetParams(second, &queried));
  REQUIRE(queried.paramArray != nullptr && queried.paramArray[0].writeValue.value == 2);
  REQUIRE(saved[0].writeValue.value == 1);
  CHECK(cuGraphDestroyNode(second));
  CHECK(cuGraphBatchMemOpNodeGetParams(first, &queried));
  REQUIRE(saved[0].writeValue.value == 1);

  // Parameter changes may replace the array, including a change in its size.
  params.count = 2;
  params.paramArray = ops;
  CHECK(cuGraphBatchMemOpNodeSetParams(first, &params));
  CHECK(cuGraphBatchMemOpNodeGetParams(first, &queried));
  REQUIRE(queried.count == 2 && queried.paramArray[1].writeValue.value == 2);
  CHECK(cuGraphDestroy(graph));
  CHECK(cuMemFree(value));
  return 0;
}

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));
  REQUIRE(test_alloc(device, false) == 0);
  REQUIRE(test_alloc(device, true) == 0);
  REQUIRE(test_batch_params(context) == 0);
  CHECK(cuDevicePrimaryCtxRelease(device));
  std::puts("graph parameter arrays: PASS");
}
