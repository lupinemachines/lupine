// Regression coverage for kernel graph node params crossing the Lupine RPC
// boundary. cuGraphKernelNodeGetParams returns the server's node-owned
// kernelParams pointer as an opaque value; the param setters echo it back
// verbatim (memcpy'd structs included) without re-sending argument values,
// while the node add paths always marshal caller-built argument arrays.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

static const char *errstr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

#define DRV(call)                                                              \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call, errstr(_r),        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char *kPtx = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_kernel(
    .param .u64 out,
    .param .u32 value
)
{
    .reg .b64 %rd<2>;
    .reg .b32 %r<4>;

    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [value];
    ld.global.u32 %r2, [%rd1];
    add.u32 %r3, %r2, %r1;
    st.global.u32 [%rd1], %r3;
    ret;
}
)ptx";

static CUfunction node_function(const CUDA_KERNEL_NODE_PARAMS &params) {
  if (params.func != nullptr) {
    return params.func;
  }
#if CUDA_VERSION >= 12000
  return reinterpret_cast<CUfunction>(params.kern);
#else
  return nullptr;
#endif
}

static int read_device_u32(CUdeviceptr ptr) {
  unsigned int value = 0;
  CUresult r = cuMemcpyDtoH(&value, ptr, sizeof(value));
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "FATAL: cuMemcpyDtoH -> %s\n", errstr(r));
    std::exit(1);
  }
  return static_cast<int>(value);
}

static void reset_device_u32(CUdeviceptr ptr) {
  unsigned int value = 0;
  CUresult r = cuMemcpyHtoD(ptr, &value, sizeof(value));
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "FATAL: cuMemcpyHtoD -> %s\n", errstr(r));
    std::exit(1);
  }
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));

  CUmodule module;
  DRV(cuModuleLoadData(&module, kPtx));
  CUfunction func;
  DRV(cuModuleGetFunction(&func, module, "add_kernel"));

  CUdeviceptr out = 0;
  DRV(cuMemAlloc(&out, sizeof(unsigned int)));
  reset_device_u32(out);

  int initial = 3;
  void *kernel_params[] = {&out, &initial};
  CUDA_KERNEL_NODE_PARAMS params = {};
  params.func = func;
  params.gridDimX = 1;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.blockDimX = 1;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.kernelParams = kernel_params;

  CUgraph graph;
  DRV(cuGraphCreate(&graph, 0));
  CUgraphNode node;
  DRV(cuGraphAddKernelNode(&node, graph, nullptr, 0, &params));

  CUDA_KERNEL_NODE_PARAMS got = {};
  DRV(cuGraphKernelNodeGetParams(node, &got));
  CHECK(got.kernelParams != nullptr,
        "cuGraphKernelNodeGetParams returned a kernelParams token");
  CHECK(node_function(got) == func,
        "cuGraphKernelNodeGetParams round-tripped the function handle");

  DRV(cuGraphKernelNodeSetParams(node, &got));

  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CUstream stream;
  DRV(cuStreamCreate(&stream, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(read_device_u32(out) == initial,
        "token round trip through cuGraphKernelNodeSetParams kept arguments");

  CUDA_KERNEL_NODE_PARAMS fresh = {};
  DRV(cuGraphKernelNodeGetParams(node, &fresh));
  CUDA_KERNEL_NODE_PARAMS copied;
  memcpy(&copied, &fresh, sizeof(copied));
  reset_device_u32(out);
  DRV(cuGraphExecKernelNodeSetParams(exec, node, &copied));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(read_device_u32(out) == initial,
        "memcpy'd token struct resolved through the exec setter");

#if CUDA_VERSION >= 12060
  reset_device_u32(out);
  int generic_value = 13;
  void *generic_kernel_params[] = {&out, &generic_value};
  CUgraphNodeParams generic_params = {};
  generic_params.type = CU_GRAPH_NODE_TYPE_KERNEL;
  generic_params.kernel.func = func;
  generic_params.kernel.gridDimX = 1;
  generic_params.kernel.gridDimY = 1;
  generic_params.kernel.gridDimZ = 1;
  generic_params.kernel.blockDimX = 1;
  generic_params.kernel.blockDimY = 1;
  generic_params.kernel.blockDimZ = 1;
  generic_params.kernel.kernelParams = generic_kernel_params;

  CUgraph generic_graph;
  DRV(cuGraphCreate(&generic_graph, 0));
  CUgraphNode generic_node;
  DRV(cuGraphAddNode_v2(&generic_node, generic_graph, nullptr, nullptr, 0,
                        &generic_params));
  CUgraphExec generic_exec;
  DRV(cuGraphInstantiateWithFlags(&generic_exec, generic_graph, 0));
  DRV(cuGraphLaunch(generic_exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(read_device_u32(out) == generic_value,
        "cuGraphAddNode_v2 streamed kernel arguments");
  DRV(cuGraphExecDestroy(generic_exec));
  DRV(cuGraphDestroy(generic_graph));
#endif

  DRV(cuStreamDestroy(stream));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  DRV(cuMemFree(out));
  DRV(cuModuleUnload(module));
  DRV(cuDevicePrimaryCtxRelease(dev));
  return g_failures == 0 ? 0 : 1;
}
