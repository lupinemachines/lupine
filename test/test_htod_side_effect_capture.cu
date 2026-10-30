#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
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

.visible .entry spin_kernel(
    .param .u64 duration
)
{
    .reg .b64 %rd<5>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [duration];
    mov.u64 %rd2, %globaltimer;
SPIN:
    mov.u64 %rd3, %globaltimer;
    sub.u64 %rd4, %rd3, %rd2;
    setp.lt.u64 %p1, %rd4, %rd1;
    @%p1 bra SPIN;
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

static void CUDA_CB delay_stream(void *) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

static int check_async_source_lifetime(CUdeviceptr destination, CUstream stream,
                                       unsigned char *source, size_t bytes) {
  constexpr unsigned char copied = 0x2a;
  constexpr unsigned char reused = 0xa2;
  std::fill_n(source, bytes, copied);
  CHECK(cuLaunchHostFunc(stream, delay_stream, nullptr));
  CHECK(cuMemcpyHtoDAsync(destination, source, bytes, stream));
  // Pageable HtoD may stage synchronously. Once the API returns, reusing the
  // source must not change the transfer queued above.
  std::fill_n(source, bytes, reused);
  CHECK(cuStreamSynchronize(stream));

  std::vector<unsigned char> readback(bytes);
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, copied)) {
    std::fprintf(stderr, "pageable async HtoD retained its source too long\n");
    return 1;
  }
  return 0;
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

// Pitched copies stage through the copy kernel rather than a CUDA memcpy.
static CUDA_MEMCPY2D pitched_copy(CUdeviceptr destination, size_t bytes) {
  CUDA_MEMCPY2D copy = {};
  copy.WidthInBytes = 4096;
  copy.Height = bytes / copy.WidthInBytes;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = destination;
  copy.dstPitch = 2 * copy.WidthInBytes;
  return copy;
}

static int capture_copy(CUdeviceptr destination, CUstream stream,
                        unsigned char *source, size_t bytes, CUgraph *graph) {
  CUDA_MEMCPY2D copy = pitched_copy(destination, bytes);
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = source;
  copy.srcPitch = copy.WidthInBytes;
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpy2DAsync(&copy, stream));
  CHECK(cuStreamEndCapture(stream, graph));
  return 0;
}

static int read_pitched(CUdeviceptr source, unsigned char *destination,
                        size_t bytes) {
  CUDA_MEMCPY2D copy = pitched_copy(0, bytes);
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = source;
  copy.srcPitch = copy.dstPitch;
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = destination;
  copy.dstPitch = copy.WidthInBytes;
  CHECK(cuMemcpy2D(&copy));
  return 0;
}

static int update_exec(CUgraphExec executable, CUgraph graph) {
#if CUDA_VERSION >= 12000
  CUgraphExecUpdateResultInfo info = {};
  CHECK(cuGraphExecUpdate(executable, graph, &info));
#else
  CUgraphNode error_node = nullptr;
  CUgraphExecUpdateResult result = CU_GRAPH_EXEC_UPDATE_ERROR;
  CHECK(cuGraphExecUpdate(executable, graph, &error_node, &result));
#endif
  return 0;
}

// An updated executable must keep a private staging ring like an instantiated
// one, or two of them race on the ring the captures shared. Filling every
// block slot holds back the staging copy kernels while both callbacks fetch.
static int check_updated_graph_instances(
    CUdeviceptr first_destination, CUdeviceptr second_destination,
    CUstream first_stream, CUstream second_stream, unsigned char *first_source,
    unsigned char *second_source, size_t bytes, CUfunction spin) {
  CUdevice device = 0;
  CHECK(cuCtxGetDevice(&device));
  int multiprocessors = 0;
  int blocks_per_multiprocessor = 0;
  CHECK(cuDeviceGetAttribute(&multiprocessors,
                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CHECK(cuDeviceGetAttribute(&blocks_per_multiprocessor,
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                             device));
  CUstream spin_stream = nullptr;
  CHECK(cuStreamCreate(&spin_stream, CU_STREAM_NON_BLOCKING));
  unsigned long long duration = 300000000ull;
  void *spin_arguments[] = {&duration};

  CUgraph graphs[4] = {};
  CUgraphExec first_executable = nullptr;
  CUgraphExec second_executable = nullptr;
  if (capture_copy(first_destination, first_stream, first_source, bytes,
                   &graphs[0]) != 0 ||
      capture_copy(second_destination, second_stream, second_source, bytes,
                   &graphs[1]) != 0) {
    return 1;
  }
  CHECK(cuGraphInstantiateWithFlags(&first_executable, graphs[0], 0));
  CHECK(cuGraphInstantiateWithFlags(&second_executable, graphs[1], 0));
  if (capture_copy(first_destination, first_stream, first_source, bytes,
                   &graphs[2]) != 0 ||
      capture_copy(second_destination, second_stream, second_source, bytes,
                   &graphs[3]) != 0 ||
      update_exec(first_executable, graphs[2]) != 0 ||
      update_exec(second_executable, graphs[3]) != 0) {
    return 1;
  }

  std::vector<unsigned char> readback(bytes);
  for (int round = 0; round < 8; ++round) {
    CHECK(cuMemsetD8(first_destination, 0, 2 * bytes));
    CHECK(cuMemsetD8(second_destination, 0, 2 * bytes));
    CHECK(cuCtxSynchronize());
    CHECK(cuLaunchKernel(spin, multiprocessors * blocks_per_multiprocessor, 1,
                         1, 32, 1, 1, 0, spin_stream, spin_arguments,
                         nullptr));
    CHECK(cuGraphLaunch(first_executable, first_stream));
    CHECK(cuGraphLaunch(second_executable, second_stream));
    CHECK(cuStreamSynchronize(first_stream));
    CHECK(cuStreamSynchronize(second_stream));
    if (read_pitched(first_destination, readback.data(), bytes) != 0) {
      return 1;
    }
    if (!std::equal(readback.begin(), readback.end(), first_source)) {
      std::fprintf(stderr, "first updated graph HtoD was corrupted\n");
      return 1;
    }
    if (read_pitched(second_destination, readback.data(), bytes) != 0) {
      return 1;
    }
    if (!std::equal(readback.begin(), readback.end(), second_source)) {
      std::fprintf(stderr, "second updated graph HtoD was corrupted\n");
      return 1;
    }
  }

  CHECK(cuStreamDestroy(spin_stream));
  CHECK(cuGraphExecDestroy(second_executable));
  CHECK(cuGraphExecDestroy(first_executable));
  for (CUgraph graph : graphs) {
    CHECK(cuGraphDestroy(graph));
  }
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

static int check_graph_memory_nodes(CUdeviceptr destination, CUstream stream,
                                    unsigned char *source, size_t bytes) {
  CUdeviceptr temporary = 0;
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemAllocAsync(&temporary, bytes, stream));
  CHECK(cuMemcpyHtoDAsync(temporary, source, bytes, stream));
  CHECK(cuMemcpyDtoDAsync(destination, temporary, bytes, stream));
  CHECK(cuMemFreeAsync(temporary, stream));
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph));

  std::vector<unsigned char> readback(bytes);
  for (unsigned char pattern :
       {static_cast<unsigned char>(0x6a), static_cast<unsigned char>(0xa6)}) {
    std::fill_n(source, bytes, pattern);
    CUgraphExec executable = nullptr;
    CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
    if (!all_bytes_are(readback, pattern)) {
      std::fprintf(stderr, "captured graph-memory HtoD was corrupted\n");
      return 1;
    }
    CHECK(cuGraphExecDestroy(executable));
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
  // Exec-node APIs keep taking instantiation-graph nodes after an update.
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyHtoDAsync(destination, &source, sizeof(source), stream));
  CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments,
                       nullptr));
  CUgraph update = nullptr;
  CHECK(cuStreamEndCapture(stream, &update));
  if (update_exec(executable, update) != 0) {
    return 1;
  }
  CHECK(cuGraphDestroy(update));
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
  // Five 8 MiB fragments exercise repeated callback/copy staging reuse.
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
  CUfunction spin_function = nullptr;
  CHECK(cuModuleGetFunction(&spin_function, module, "spin_kernel"));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUstream second_stream = nullptr;
  CHECK(cuStreamCreate(&second_stream, CU_STREAM_NON_BLOCKING));
  std::vector<unsigned char> source(bytes, 0x11);
  std::vector<unsigned char> second_source(bytes, 0x22);
  std::vector<unsigned char> readback(bytes);

  if (check_async_source_lifetime(destination, stream, source.data(),
                                  4 * 1024 * 1024) != 0) {
    return 1;
  }
  std::fill(source.begin(), source.end(), 0x11);

  // Ordinary asynchronous HtoD also goes through the execution-time ring.
  CHECK(cuMemcpyHtoDAsync(destination, source.data(), bytes, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(readback.data(), destination, bytes));
  if (!all_bytes_are(readback, 0x11)) {
    std::fprintf(stderr, "pageable async HtoD produced stale data\n");
    return 1;
  }

  // Two streams contend for the bounded staging buffer. The private transfer
  // stream must keep each fragment callback ordered before its corresponding
  // copy without corrupting the other operation.
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
  if (check_updated_graph_instances(
          destination, second_destination, stream, second_stream, source.data(),
          second_source.data(), 12 * 1024 * 1024, spin_function) != 0) {
    return 1;
  }
  if (check_parallel_graph_instances(destination, stream, second_stream,
                                     source.data(), 12 * 1024 * 1024) != 0) {
    return 1;
  }
  if (check_graph_memory_nodes(destination, stream, source.data(),
                               4 * 1024 * 1024) != 0) {
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
