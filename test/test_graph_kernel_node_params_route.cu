// LUPINE_TEST_CONNECTIONS 2
// Reads and updates a kernel node of a graph built on the second virtual
// device while the first device's context is current.
#include <cuda.h>

#include <cstdio>

static const char kStorePtx[] = ".version 6.4\n"
                                ".target sm_52\n"
                                ".address_size 64\n"
                                ".visible .entry store(\n"
                                "  .param .u64 store_param_0,\n"
                                "  .param .u32 store_param_1\n"
                                ")\n"
                                "{\n"
                                "  .reg .b64 %rd<3>;\n"
                                "  .reg .b32 %r<2>;\n"
                                "  ld.param.u64 %rd1, [store_param_0];\n"
                                "  ld.param.u32 %r1, [store_param_1];\n"
                                "  cvta.to.global.u64 %rd2, %rd1;\n"
                                "  st.global.u32 [%rd2], %r1;\n"
                                "  ret;\n"
                                "}\n";

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      std::fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");       \
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
  CHECK(cuCtxSetCurrent(second));

  CUmodule module = nullptr;
  CUfunction store = nullptr;
  CUdeviceptr out = 0;
  CHECK(cuModuleLoadData(&module, kStorePtx));
  CHECK(cuModuleGetFunction(&store, module, "store"));
  CHECK(cuMemAlloc(&out, sizeof(unsigned)));

  unsigned value = 1;
  void *args[] = {&out, &value};
  CUDA_KERNEL_NODE_PARAMS params = {};
  params.func = store;
  params.gridDimX = params.gridDimY = params.gridDimZ = 1;
  params.blockDimX = params.blockDimY = params.blockDimZ = 1;
  params.kernelParams = args;
  CUgraph graph = nullptr;
  CUgraphNode node = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CHECK(cuGraphAddKernelNode(&node, graph, nullptr, 0, &params));

  CHECK(cuCtxSetCurrent(first));
  CUDA_KERNEL_NODE_PARAMS read = {};
  CHECK(cuGraphKernelNodeGetParams(node, &read));
  REQUIRE(read.func == store);
  REQUIRE(read.kernelParams != nullptr);
  REQUIRE(*static_cast<unsigned *>(read.kernelParams[1]) == 1);
  value = 2;
  CHECK(cuGraphKernelNodeSetParams(node, &params));

  CHECK(cuCtxSetCurrent(second));
  CUgraphExec exec = nullptr;
  unsigned got = 0;
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CHECK(cuGraphLaunch(exec, nullptr));
  CHECK(cuStreamSynchronize(nullptr));
  CHECK(cuMemcpyDtoH(&got, out, sizeof(got)));
  REQUIRE(got == 2);

  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphDestroy(graph));
  CHECK(cuMemFree(out));
  CHECK(cuModuleUnload(module));
  std::printf("graph kernel node params on the second device: passed\n");
  return 0;
}
