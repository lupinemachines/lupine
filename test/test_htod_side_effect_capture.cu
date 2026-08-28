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

static const char *kAddPtx = R"ptx(
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

static int check_parallel_graph_instances(CUdeviceptr destination,
                                          CUstream first_stream,
                                          CUstream second_stream,
                                          unsigned char *source, size_t bytes) {
  CHECK(cuStreamBeginCapture(first_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(destination, source, bytes, first_stream));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(first_stream, &graph));
  CUgraphExec first_executable = nullptr;
  CUgraphExec second_executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&first_executable, graph, 0));
  CHECK(cuGraphInstantiateWithFlags(&second_executable, graph, 0));

  std::fill_n(source, bytes, 0x77);
  CHECK(cuGraphLaunch(first_executable, first_stream));
  CHECK(cuGraphLaunch(second_executable, second_stream));
  // CUDA permits graph-exec destruction while launches are in flight. The
  // server must retain each executable's callback data and ring until those
  // launches complete.
  CHECK(cuGraphExecDestroy(second_executable));
  CHECK(cuGraphExecDestroy(first_executable));
  CHECK(cuStreamSynchronize(first_stream));
  CHECK(cuStreamSynchronize(second_stream));

  std::vector<unsigned char> readback(bytes);
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x77)) {
    std::fprintf(stderr, "parallel instances of one graph were corrupted\n");
    return 1;
  }

  CHECK(cuGraphDestroy(graph));
  return 0;
}

static int check_exec_node_set_params(CUdeviceptr destination, CUstream stream,
                                      CUfunction function) {
  unsigned int source = 5;
  unsigned int increment = 3;
  void *arguments[] = {&destination, &increment};
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(destination, &source, sizeof(source), stream));
  CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments,
                       nullptr));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph));

  size_t node_count = 0;
  CHECK(cuGraphGetNodes(graph, nullptr, &node_count));
  std::vector<CUgraphNode> nodes(node_count);
  CHECK(cuGraphGetNodes(graph, nodes.data(), &node_count));
  CUgraphNode kernel_node = nullptr;
  CUDA_KERNEL_NODE_PARAMS params = {};
  for (CUgraphNode node : nodes) {
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    CHECK(cuGraphNodeGetType(node, &type));
    if (type != CU_GRAPH_NODE_TYPE_KERNEL) {
      continue;
    }
    CHECK(cuGraphKernelNodeGetParams(node, &params));
    if (params.func == function) {
      kernel_node = node;
      break;
    }
  }
  if (kernel_node == nullptr || params.kernelParams == nullptr) {
    std::fprintf(stderr, "captured application kernel node was not found\n");
    return 1;
  }

  CUgraphExec executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));
  unsigned int updated_increment = 7;
  params.kernelParams[1] = &updated_increment;
  CHECK(cuGraphExecKernelNodeSetParams(executable, kernel_node, &params));
  source = 11;
  CHECK(cuGraphLaunch(executable, stream));
  CHECK(cuStreamSynchronize(stream));
  unsigned int readback = 0;
  CHECK(cuMemcpyDtoH(&readback, destination, sizeof(readback)));
  if (readback != source + updated_increment) {
    std::fprintf(stderr, "graph-exec kernel update produced %u, expected %u\n",
                 readback, source + updated_increment);
    return 1;
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
  CUdeviceptr second_destination = 0;
  CHECK(cuMemAlloc(&second_destination, bytes));
  CUmodule module = nullptr;
  CHECK(cuModuleLoadData(&module, kAddPtx));
  CUfunction add_function = nullptr;
  CHECK(cuModuleGetFunction(&add_function, module, "add_kernel"));
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
  std::fill_n(source.data(), bytes, 0x4d);
  CHECK(cuMemcpyHtoDAsync(destination, source.data(), bytes, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x4d)) {
    std::fprintf(stderr, "ordinary HtoD after graph launch was corrupted\n");
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
  if (check_parallel_graph_instances(destination, stream, second_stream,
                                     source.data(), 12 * 1024 * 1024) != 0) {
    return 1;
  }
  if (check_exec_node_set_params(destination, stream, add_function) != 0) {
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
  CHECK(cuModuleUnload(module));
  CHECK(cuCtxDestroy(context));
  std::printf(
      "PASS: HtoD side effects preserve pageable and graph semantics\n");
  return 0;
}
