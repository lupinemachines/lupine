#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult check_result = (call);                                            \
    if (check_result != CUDA_SUCCESS) {                                        \
      const char *message = nullptr;                                           \
      cuGetErrorString(check_result, &message);                                \
      std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", #call, __LINE__, \
                   message == nullptr ? "unknown" : message,                   \
                   static_cast<int>(check_result));                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static bool all_bytes_are(const std::vector<unsigned char> &bytes,
                          unsigned char expected) {
  return std::all_of(
      bytes.begin(), bytes.end(),
      [expected](unsigned char value) { return value == expected; });
}

static int check_captured_source(CUdeviceptr destination, CUstream stream,
                                 unsigned char *source, size_t bytes,
                                 const char *label) {
  std::vector<unsigned char> readback(bytes);
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(destination, source, bytes, stream));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph));
  CUgraphExec executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));

  for (unsigned char pattern :
       {static_cast<unsigned char>(0x33), static_cast<unsigned char>(0x44)}) {
    std::fill_n(source, bytes, pattern);
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
    if (!all_bytes_are(readback, pattern)) {
      std::fprintf(stderr, "captured %s HtoD used stale source data\n", label);
      return 1;
    }
  }

  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));
  return 0;
}

struct fill_request {
  unsigned char *source = nullptr;
  size_t bytes = 0;
  unsigned char value = 0;
};

static void CUDA_CB fill_source(void *opaque) {
  auto *request = static_cast<fill_request *>(opaque);
  std::fill_n(request->source, request->bytes, request->value);
}

static int check_captured_host_order(CUdeviceptr destination, CUstream stream,
                                     unsigned char *source, size_t bytes) {
  fill_request request{source, bytes, 0x55};
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuLaunchHostFunc(stream, fill_source, &request));
  CHECK(cuMemcpyHtoDAsync(destination, source, bytes, stream));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph));
  CUgraphExec executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));

  std::vector<unsigned char> readback(bytes);
  for (unsigned char pattern :
       {static_cast<unsigned char>(0x55), static_cast<unsigned char>(0x66)}) {
    request.value = pattern;
    std::fill_n(source, bytes, 0);
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
    if (!all_bytes_are(readback, pattern)) {
      std::fprintf(stderr,
                   "captured HtoD ran before its preceding host function\n");
      return 1;
    }
  }

  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));
  return 0;
}

int main() {
  // Five 8 MiB chunks force the two-slot server ring to wrap twice.
  constexpr size_t bytes = 40 * 1024 * 1024;
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  CUdeviceptr destination = 0;
  CHECK(cuMemAlloc(&destination, bytes));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  std::vector<unsigned char> source(bytes, 0x11);
  std::vector<unsigned char> readback(bytes);

  // Ordinary asynchronous HtoD also goes through the execution-time ring.
  CHECK(cuMemcpyHtoDAsync(destination, source.data(), bytes, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x11)) {
    std::fprintf(stderr, "pageable async HtoD produced stale data\n");
    return 1;
  }

  if (check_captured_source(destination, stream, source.data(), bytes,
                            "pageable") != 0) {
    return 1;
  }
  if (check_captured_host_order(destination, stream, source.data(),
                                4 * 1024 * 1024) != 0) {
    return 1;
  }

  constexpr size_t tracked_bytes = 4 * 1024 * 1024;
  unsigned char *pinned = nullptr;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&pinned), tracked_bytes,
                       CU_MEMHOSTALLOC_PORTABLE));
  if (check_captured_source(destination, stream, pinned, tracked_bytes,
                            "pinned") != 0) {
    return 1;
  }

  unsigned char *unified = nullptr;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&unified), tracked_bytes,
                       CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  if (check_captured_source(destination, stream, unified, tracked_bytes,
                            "unified") != 0) {
    return 1;
  }

  CUdeviceptr managed = 0;
  CHECK(cuMemAllocManaged(&managed, tracked_bytes, CU_MEM_ATTACH_GLOBAL));
  if (check_captured_source(destination, stream,
                            reinterpret_cast<unsigned char *>(managed),
                            tracked_bytes, "managed") != 0) {
    return 1;
  }

  CHECK(cuMemFree(managed));
  CHECK(cuMemFreeHost(unified));
  CHECK(cuMemFreeHost(pinned));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemFree(destination));
  CHECK(cuCtxDestroy(context));
  std::printf(
      "PASS: HtoD side effects preserve pageable and graph semantics\n");
  return 0;
}
