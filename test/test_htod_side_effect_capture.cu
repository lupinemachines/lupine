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

static int check_multiple_captured_copies(CUdeviceptr first_destination,
                                          CUdeviceptr second_destination,
                                          CUstream stream,
                                          unsigned char *first_source,
                                          unsigned char *second_source,
                                          size_t bytes) {
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(first_destination, first_source, bytes, stream));
  CHECK(cuMemcpyHtoDAsync(second_destination, second_source, bytes, stream));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph));
  CUgraphExec executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));
  CHECK(cuGraphLaunch(executable, stream));
  CHECK(cuStreamSynchronize(stream));

  std::vector<unsigned char> readback(bytes);
  CHECK(cuMemcpyDtoH(readback.data(), first_destination, bytes));
  if (!std::equal(readback.begin(), readback.end(), first_source)) {
    std::fprintf(stderr, "first copy in a captured HtoD sequence failed\n");
    return 1;
  }
  CHECK(cuMemcpyDtoH(readback.data(), second_destination, bytes));
  if (!std::equal(readback.begin(), readback.end(), second_source)) {
    std::fprintf(stderr, "second copy in a captured HtoD sequence failed\n");
    return 1;
  }

  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));
  return 0;
}

static int check_concurrent_captured_copies(
    CUdeviceptr first_destination, CUdeviceptr second_destination,
    CUstream first_stream, CUstream second_stream, unsigned char *first_source,
    unsigned char *second_source, size_t bytes) {
  CHECK(cuStreamBeginCapture(first_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(
      cuMemcpyHtoDAsync(first_destination, first_source, bytes, first_stream));
  CUgraph first_graph = nullptr;
  CHECK(cuStreamEndCapture(first_stream, &first_graph));
  CUgraphExec first_executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&first_executable, first_graph, 0));

  CHECK(cuStreamBeginCapture(second_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(second_destination, second_source, bytes,
                          second_stream));
  CUgraph second_graph = nullptr;
  CHECK(cuStreamEndCapture(second_stream, &second_graph));
  CUgraphExec second_executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&second_executable, second_graph, 0));

  CHECK(cuGraphLaunch(first_executable, first_stream));
  CHECK(cuGraphLaunch(second_executable, second_stream));
  CHECK(cuStreamSynchronize(first_stream));
  CHECK(cuStreamSynchronize(second_stream));

  std::vector<unsigned char> readback(bytes);
  CHECK(cuMemcpyDtoH(readback.data(), first_destination, bytes));
  if (!std::equal(readback.begin(), readback.end(), first_source)) {
    std::fprintf(stderr, "first concurrent captured HtoD was corrupted\n");
    return 1;
  }
  CHECK(cuMemcpyDtoH(readback.data(), second_destination, bytes));
  if (!std::equal(readback.begin(), readback.end(), second_source)) {
    std::fprintf(stderr, "second concurrent captured HtoD was corrupted\n");
    return 1;
  }

  CHECK(cuGraphExecDestroy(second_executable));
  CHECK(cuGraphDestroy(second_graph));
  CHECK(cuGraphExecDestroy(first_executable));
  CHECK(cuGraphDestroy(first_graph));
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
  CUdeviceptr second_destination = 0;
  CHECK(cuMemAlloc(&second_destination, bytes));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUstream second_stream = nullptr;
  CHECK(cuStreamCreate(&second_stream, CU_STREAM_NON_BLOCKING));
  std::vector<unsigned char> source(bytes, 0x11);
  std::vector<unsigned char> second_source(bytes, 0x22);
  std::vector<unsigned char> readback(bytes);

  // Ordinary asynchronous HtoD also goes through the execution-time ring.
  CHECK(cuMemcpyHtoDAsync(destination, source.data(), bytes, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x11)) {
    std::fprintf(stderr, "pageable async HtoD produced stale data\n");
    return 1;
  }

  // Two streams contend for the fixed ring. GPU slot-release writes must keep
  // either callback from overwriting a slot whose previous DMA is in flight.
  CHECK(cuMemcpyHtoDAsync(destination, source.data(), bytes, stream));
  CHECK(cuMemcpyHtoDAsync(second_destination, second_source.data(), bytes,
                          second_stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuStreamSynchronize(second_stream));
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x11)) {
    std::fprintf(stderr, "first concurrent HtoD was corrupted\n");
    return 1;
  }
  CHECK(cuMemcpyDtoH(readback.data(), second_destination, bytes));
  if (!all_bytes_are(readback, 0x22)) {
    std::fprintf(stderr, "second concurrent HtoD was corrupted\n");
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
  if (check_multiple_captured_copies(destination, second_destination, stream,
                                     source.data(), second_source.data(),
                                     12 * 1024 * 1024) != 0) {
    return 1;
  }
  if (check_concurrent_captured_copies(
          destination, second_destination, stream, second_stream, source.data(),
          second_source.data(), 12 * 1024 * 1024) != 0) {
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
  CHECK(cuStreamDestroy(second_stream));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemFree(second_destination));
  CHECK(cuMemFree(destination));
  CHECK(cuCtxDestroy(context));
  std::printf(
      "PASS: HtoD side effects preserve pageable and graph semantics\n");
  return 0;
}
