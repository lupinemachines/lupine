// Captured device-to-host copies stage through server host memory that the
// graph replays into. Only captures that contain such a copy may pin host
// memory, and captures into an existing graph need the staging too.
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

static void launch_and_check(CUgraph graph, CUstream stream,
                             const unsigned int *host, unsigned int expected,
                             const char *what) {
  CUgraphExec exec;
#if CUDA_VERSION >= 12000
  DRV(cuGraphInstantiate(&exec, graph, 0));
#else
  DRV(cuGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
#endif
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
