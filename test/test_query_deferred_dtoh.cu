// A device-to-host copy into page-locked memory completes on the server and
// rides back to the client on a later response. A query that reports the copy
// complete has to bring its bytes along: polling cuStreamQuery or cuEventQuery
// until CUDA_SUCCESS and then reading the buffer is how torch's stream.query()
// and event.query() consume non-blocking copies.
#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <thread>

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

static const size_t kWords = 4096;

static bool all_equal(const unsigned int *host, unsigned int value) {
  for (size_t i = 0; i < kWords; ++i) {
    if (host[i] != value) {
      return false;
    }
  }
  return true;
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));

  CUstream stream, other;
  DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  DRV(cuStreamCreate(&other, CU_STREAM_NON_BLOCKING));
  CUdeviceptr dev_buf;
  DRV(cuMemAlloc(&dev_buf, kWords * sizeof(unsigned int)));
  unsigned int *host = nullptr;
  DRV(cuMemAllocHost(reinterpret_cast<void **>(&host),
                     kWords * sizeof(unsigned int)));

  // Stream query after an async copy into pinned memory.
  for (size_t i = 0; i < kWords; ++i) {
    host[i] = 0;
  }
  DRV(cuMemsetD32Async(dev_buf, 0x5a5a1234u, kWords, stream));
  DRV(cuMemcpyDtoHAsync(host, dev_buf, kWords * sizeof(unsigned int), stream));
  CUresult r;
  while ((r = cuStreamQuery(stream)) == CUDA_ERROR_NOT_READY) {
  }
  CHECK(r == CUDA_SUCCESS, "cuStreamQuery reaches CUDA_SUCCESS");
  CHECK(all_equal(host, 0x5a5a1234u),
        "cuStreamQuery success delivers the pinned DtoH bytes");
  DRV(cuStreamSynchronize(stream));

  // Event query after a graph whose memcpy node writes pinned memory. The
  // query of another event warms the cached completion of the graph's event.
  CUgraph graph;
  DRV(cuGraphCreate(&graph, 0));
  CUDA_MEMSET_NODE_PARAMS memset_params = {};
  memset_params.dst = dev_buf;
  memset_params.value = 0x77778888u;
  memset_params.elementSize = sizeof(unsigned int);
  memset_params.width = kWords;
  memset_params.height = 1;
  CUgraphNode memset_node;
  DRV(cuGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &memset_params,
                           ctx));
  CUDA_MEMCPY3D copy_params = {};
  copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_params.srcDevice = dev_buf;
  copy_params.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy_params.dstHost = host;
  copy_params.WidthInBytes = kWords * sizeof(unsigned int);
  copy_params.Height = 1;
  copy_params.Depth = 1;
  CUgraphNode copy_node;
  DRV(cuGraphAddMemcpyNode(&copy_node, graph, &memset_node, 1, &copy_params,
                           ctx));
  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));

  CUevent done, unrelated;
  DRV(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING));
  DRV(cuEventCreate(&unrelated, CU_EVENT_DISABLE_TIMING));
  for (size_t i = 0; i < kWords; ++i) {
    host[i] = 0;
  }
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuEventRecord(done, stream));
  DRV(cuEventRecord(unrelated, other));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  while ((r = cuEventQuery(unrelated)) == CUDA_ERROR_NOT_READY) {
  }
  CHECK(r == CUDA_SUCCESS, "cuEventQuery(unrelated) reaches CUDA_SUCCESS");
  while ((r = cuEventQuery(done)) == CUDA_ERROR_NOT_READY) {
  }
  CHECK(r == CUDA_SUCCESS, "cuEventQuery(done) reaches CUDA_SUCCESS");
  CHECK(all_equal(host, 0x77778888u),
        "cuEventQuery success delivers the graph's pinned DtoH bytes");
  DRV(cuStreamSynchronize(stream));

  DRV(cuEventDestroy(unrelated));
  DRV(cuEventDestroy(done));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  DRV(cuMemFreeHost(host));
  DRV(cuMemFree(dev_buf));
  DRV(cuStreamDestroy(other));
  DRV(cuStreamDestroy(stream));
  DRV(cuDevicePrimaryCtxRelease(dev));

  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
