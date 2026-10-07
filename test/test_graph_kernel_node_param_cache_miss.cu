// Adds and updates graph kernel nodes right after an unrelated module unload,
// when the client holds no cached parameter layout for the kernel.
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
      std::fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static CUresult unload_other_module() {
  CUmodule other = nullptr;
  CUresult result = cuModuleLoadData(&other, kStorePtx);
  return result == CUDA_SUCCESS ? cuModuleUnload(other) : result;
}

static int run_and_check(CUgraphExec exec, CUdeviceptr out, unsigned want,
                         const char *what) {
  unsigned got = 0;
  CHECK(cuGraphLaunch(exec, nullptr));
  CHECK(cuStreamSynchronize(nullptr));
  CHECK(cuMemcpyDtoH(&got, out, sizeof(got)));
  if (got != want) {
    std::fprintf(stderr, "%s: stored %u, want %u\n", what, got, want);
    return 1;
  }
  std::printf("%s: passed\n", what);
  return 0;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction store = nullptr;
  CUdeviceptr out = 0;
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&device, 0));
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));
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
  CUgraphExec exec = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CHECK(unload_other_module());
  CHECK(cuGraphAddKernelNode(&node, graph, nullptr, 0, &params));
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  if (run_and_check(exec, out, 1, "cuGraphAddKernelNode")) {
    return 1;
  }

  value = 2;
  CHECK(unload_other_module());
  CHECK(cuGraphKernelNodeSetParams(node, &params));
  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  if (run_and_check(exec, out, 2, "cuGraphKernelNodeSetParams")) {
    return 1;
  }

  value = 3;
  CHECK(unload_other_module());
  CHECK(cuGraphExecKernelNodeSetParams(exec, node, &params));
  if (run_and_check(exec, out, 3, "cuGraphExecKernelNodeSetParams")) {
    return 1;
  }
  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphDestroy(graph));

#if CUDA_VERSION >= 12030
  value = 4;
  CUgraphNodeParams node_params = {};
  node_params.type = CU_GRAPH_NODE_TYPE_KERNEL;
  node_params.kernel.func = store;
  node_params.kernel.gridDimX = node_params.kernel.gridDimY =
      node_params.kernel.gridDimZ = 1;
  node_params.kernel.blockDimX = node_params.kernel.blockDimY =
      node_params.kernel.blockDimZ = 1;
  node_params.kernel.kernelParams = args;
  CHECK(cuGraphCreate(&graph, 0));
  CHECK(unload_other_module());
  CHECK(cuGraphAddNode_v2(&node, graph, nullptr, nullptr, 0, &node_params));
  CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
  if (run_and_check(exec, out, 4, "cuGraphAddNode_v2")) {
    return 1;
  }
  CHECK(cuGraphExecDestroy(exec));
  CHECK(cuGraphDestroy(graph));
#endif

  CHECK(cuMemFree(out));
  CHECK(cuModuleUnload(module));
  std::printf("graph kernel node param cache miss: all checks passed\n");
  return 0;
}
