// Captured device-to-host copies stage through server host memory that the
// graph replays into. Only captures that contain such a copy may pin host
// memory, and captures into an existing graph need the staging too. The
// staging is freed once no graph, exec or undelivered launch needs it.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

#define DRV(call)                                                              \
  do {                                                                         \
    CUresult r_ = (call);                                                      \
    if (r_ != CUDA_SUCCESS) {                                                  \
      std::fprintf(stderr, "FAIL: %s -> %d (line %d)\n", #call, (int)r_,       \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static void check(bool ok, const char *what) {
  std::fprintf(stderr, "%s: %s\n", ok ? "ok" : "FAIL", what);
  failures += ok ? 0 : 1;
}

static CUgraph capture_dtoh(CUstream stream, CUdeviceptr dev_buf,
                            unsigned int *host, size_t count,
                            unsigned int value) {
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemsetD32Async(dev_buf, value, count, stream));
  DRV(cuMemcpyDtoHAsync(host, dev_buf, count * sizeof(unsigned int), stream));
  CUgraph graph;
  DRV(cuStreamEndCapture(stream, &graph));
  return graph;
}

static CUgraphExec instantiate(CUgraph graph) {
  CUgraphExec exec;
#if CUDA_VERSION >= 12000
  DRV(cuGraphInstantiate(&exec, graph, 0));
#else
  DRV(cuGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
#endif
  return exec;
}

static bool filled(const unsigned int *host, size_t count,
                   unsigned int value) {
  return host[0] == value && host[count / 2] == value &&
         host[count - 1] == value;
}

static void launch_and_check(CUgraph graph, CUstream stream,
                             const unsigned int *host, unsigned int expected,
                             const char *what) {
  CUgraphExec exec = instantiate(graph);
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  check(host[0] == expected && host[1023] == expected, what);
  DRV(cuGraphExecDestroy(exec));
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));
  CUstream stream;
  DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  const size_t bytes = 1024 * sizeof(unsigned int);
  CUdeviceptr dev_buf;
  DRV(cuMemAlloc(&dev_buf, bytes));

  // 128 MiB each used to be pinned for the life of the server.
  for (int i = 0; i < 300; ++i) {
    DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    DRV(cuMemsetD32Async(dev_buf, i, 1024, stream));
    CUgraph graph;
    DRV(cuStreamEndCapture(stream, &graph));
    DRV(cuGraphDestroy(graph));
  }
  check(true, "300 captures");

  auto *pageable = static_cast<unsigned int *>(std::malloc(bytes));
  unsigned int *pinned = nullptr;
  DRV(cuMemAllocHost(reinterpret_cast<void **>(&pinned), bytes));

  std::memset(pageable, 0, bytes);
  std::memset(pinned, 0, bytes);
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemsetD32Async(dev_buf, 0x1234, 1024, stream));
  DRV(cuMemcpyDtoHAsync(pageable, dev_buf, bytes, stream));
  DRV(cuMemcpyDtoHAsync(pinned, dev_buf, bytes, stream));
  CUgraph graph;
  DRV(cuStreamEndCapture(stream, &graph));
  launch_and_check(graph, stream, pageable, 0x1234, "captured pageable DtoH");
  launch_and_check(graph, stream, pinned, 0x1234, "captured pinned DtoH");
  DRV(cuGraphDestroy(graph));

#if CUDA_VERSION >= 12030
  std::memset(pageable, 0, bytes);
  std::memset(pinned, 0, bytes);
  DRV(cuGraphCreate(&graph, 0));
  DRV(cuStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                  CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemsetD32Async(dev_buf, 0x5678, 1024, stream));
  DRV(cuMemcpyDtoHAsync(pageable, dev_buf, bytes, stream));
  DRV(cuMemcpyDtoHAsync(pinned, dev_buf, bytes, stream));
  DRV(cuStreamEndCapture(stream, &graph));
  launch_and_check(graph, stream, pageable, 0x5678,
                   "pageable DtoH captured into a graph");
  launch_and_check(graph, stream, pinned, 0x5678,
                   "pinned DtoH captured into a graph");
  DRV(cuGraphDestroy(graph));
#endif

  // The launch still owns the staging after its graph and exec are gone.
  const size_t big_count = 4 << 20;
  CUdeviceptr big_buf;
  DRV(cuMemAlloc(&big_buf, big_count * sizeof(unsigned int)));
  auto *big = static_cast<unsigned int *>(
      std::calloc(big_count, sizeof(unsigned int)));
  graph = capture_dtoh(stream, big_buf, big, big_count, 0x1111);
  CUgraphExec exec = instantiate(graph);
  DRV(cuGraphDestroy(graph));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuStreamSynchronize(stream));
  check(filled(big, big_count, 0x1111),
        "DtoH delivered after its graph and exec are destroyed");

  // A clone keeps the staging after the original graph is destroyed.
  graph = capture_dtoh(stream, big_buf, big, big_count, 0x2222);
  CUgraph clone;
  DRV(cuGraphClone(&clone, graph));
  DRV(cuGraphDestroy(graph));
  launch_and_check(clone, stream, big, 0x2222, "cloned DtoH graph");
  DRV(cuGraphDestroy(clone));

  // An updated exec copies into the new graph's staging, which must outlive
  // that graph.
  graph = capture_dtoh(stream, big_buf, big, big_count, 0x3333);
  exec = instantiate(graph);
  CUgraph update = capture_dtoh(stream, big_buf, big, big_count, 0x4444);
#if CUDA_VERSION >= 12000
  CUgraphExecUpdateResultInfo update_info = {};
  DRV(cuGraphExecUpdate(exec, update, &update_info));
#else
  CUgraphNode error_node = nullptr;
  CUgraphExecUpdateResult update_result;
  DRV(cuGraphExecUpdate(exec, update, &error_node, &update_result));
#endif
  DRV(cuGraphDestroy(graph));
  DRV(cuGraphDestroy(update));
  for (unsigned int i = 0; i < 2; ++i) {
    std::memset(big, 0, big_count * sizeof(unsigned int));
    DRV(cuGraphLaunch(exec, stream));
    DRV(cuStreamSynchronize(stream));
    check(filled(big, big_count, 0x4444), "updated exec DtoH");
  }
  DRV(cuGraphExecDestroy(exec));

  // 16 MiB of staging per capture used to stay pinned for the server's life.
  bool all = true;
  for (unsigned int i = 0; i < 200; ++i) {
    graph = capture_dtoh(stream, big_buf, big, big_count, i);
    exec = instantiate(graph);
    DRV(cuGraphLaunch(exec, stream));
    DRV(cuStreamSynchronize(stream));
    all = all && filled(big, big_count, i);
    DRV(cuGraphExecDestroy(exec));
    DRV(cuGraphDestroy(graph));
  }
  check(all, "200 captured 16 MiB DtoH graphs");

  std::free(big);
  DRV(cuMemFree(big_buf));
  DRV(cuMemFreeHost(pinned));
  std::free(pageable);
  DRV(cuMemFree(dev_buf));
  DRV(cuStreamDestroy(stream));
  DRV(cuDevicePrimaryCtxRelease(dev));
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::puts("PASS");
  return 0;
}
