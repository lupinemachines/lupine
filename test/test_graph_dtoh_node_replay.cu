// A graph device-to-host memcpy node stages its bytes in a server-side buffer
// that lupine ships to the client when the launch stream synchronizes. That
// staging buffer outlives the launch, so the delivery has to happen once:
// jacobiCudaGraphs reads its convergence sum through such a node every
// iteration, then overwrites the same host variable with a plain
// cuMemcpyDtoHAsync before one last synchronize. Replaying the node's stale
// bytes at that synchronize silently reverts the final result (issue #672).
//
// Pure driver API, no kernels (so it is arch-independent). Exits non-zero on
// the first failed assertion.
#include <cstdio>
#include <cstdlib>
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

static const unsigned int kGraphValue = 0x11112222u;
static const unsigned int kDirectValue = 0x33334444u;

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));

  CUstream stream;
  DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr value;
  DRV(cuMemAlloc(&value, sizeof(unsigned int)));

  // The host destination is a pageable stack variable, exactly as the sample
  // reads its sum: cuMemcpyDtoHAsync to pageable memory completes inline.
  unsigned int host = 0;

  CUgraph graph;
  DRV(cuGraphCreate(&graph, 0));

  CUDA_MEMSET_NODE_PARAMS memset_params = {};
  memset_params.dst = value;
  memset_params.pitch = 0;
  memset_params.value = kGraphValue;
  memset_params.elementSize = sizeof(unsigned int);
  memset_params.width = 1;
  memset_params.height = 1;
  CUgraphNode memset_node;
  DRV(cuGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &memset_params,
                           ctx));

  CUDA_MEMCPY3D copy_params = {};
  copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_params.srcDevice = value;
  copy_params.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy_params.dstHost = &host;
  copy_params.WidthInBytes = sizeof(unsigned int);
  copy_params.Height = 1;
  copy_params.Depth = 1;
  CUgraphNode copy_node;
  DRV(cuGraphAddMemcpyNode(&copy_node, graph, &memset_node, 1, &copy_params,
                           ctx));

  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(host == kGraphValue, "graph memcpy node delivers its bytes");

  // No launch since, so this synchronize owes the client nothing.
  host = 0;
  DRV(cuStreamSynchronize(stream));
  CHECK(host == 0, "idle synchronize does not replay the node's bytes");

  // What the sample does after it converges: overwrite the same host variable
  // from outside the graph, then synchronize one last time.
  DRV(cuMemsetD32Async(value, kDirectValue, 1, stream));
  DRV(cuMemcpyDtoHAsync(&host, value, sizeof(host), stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(host == kDirectValue, "direct copy survives the following synchronize");

  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  DRV(cuMemFree(value));
  DRV(cuStreamDestroy(stream));
  DRV(cuDevicePrimaryCtxRelease(dev));

  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
