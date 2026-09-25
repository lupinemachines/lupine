// Graph host storage follows graphs, clones, executable graphs and queued work.
// LUPINE_TEST_SERVER_RSS
// The custom-test runner samples server RSS at each --memory-steps handshake.
// The capture/clone/replay sequence also occurs in NVIDIA simpleCudaGraphs.
#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>

#define DRV(call)                                                              \
  do {                                                                         \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      fprintf(stderr, "%s: CUDA error %d (line %d)\n", #call, result,          \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

struct callback_state {
  unsigned int *value;
  unsigned int expected;
  std::atomic<unsigned int> calls{0};
  std::atomic<bool> failed{false};
};

static void CUDA_CB check_copy(void *opaque) {
  auto *state = static_cast<callback_state *>(opaque);
  if (*state->value != state->expected) {
    state->failed = true;
  }
  ++state->calls;
}

static void CUDA_CB count_calls(void *opaque) {
  ++static_cast<callback_state *>(opaque)->calls;
}

static void CUDA_CB delay_launch(void *) {
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

static int check_lifetimes(CUstream stream, CUdeviceptr device) {
  unsigned int *host = nullptr;
  DRV(cuMemAllocHost(reinterpret_cast<void **>(&host), sizeof(*host)));
  callback_state state{host, 0xabc123};
  CUevent fork, join;
  CUstream other;
  DRV(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  DRV(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));
  DRV(cuStreamCreate(&other, CU_STREAM_NON_BLOCKING));

  // A wrong-thread end reports an error and ends the capture. Its storage
  // must be released before a later capture reuses this stream.
  CUcontext context;
  DRV(cuCtxGetCurrent(&context));
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemsetD32Async(device, state.expected, 1, stream));
  DRV(cuMemcpyDtoHAsync(host, device, sizeof(*host), stream));
  CUresult wrong_thread_result = CUDA_SUCCESS;
  std::thread wrong_thread([&] {
    CUgraph graph;
    wrong_thread_result = cuCtxSetCurrent(context);
    if (wrong_thread_result == CUDA_SUCCESS) {
      wrong_thread_result = cuStreamEndCapture(stream, &graph);
    }
  });
  wrong_thread.join();
  if (wrong_thread_result != CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD) {
    fprintf(stderr, "unexpected end-capture result from another thread: %d\n",
            wrong_thread_result);
    return 1;
  }
  CUstreamCaptureStatus status;
  DRV(cuStreamIsCapturing(stream, &status));
  if (status != CU_STREAM_CAPTURE_STATUS_NONE) {
    fprintf(stderr, "wrong-thread end left the capture active\n");
    return 1;
  }

  for (int iteration = 0; iteration < 8; ++iteration) {
    *host = 0;
    unsigned int before = state.calls;
    CUgraph graph, clone;
    CUgraphExec exec;
    DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    DRV(cuMemsetD32Async(device, state.expected, 1, stream));
    DRV(cuEventRecord(fork, stream));
    DRV(cuStreamWaitEvent(other, fork, 0));
    DRV(cuMemcpyDtoHAsync(host, device, sizeof(*host), other));
    DRV(cuLaunchHostFunc(other, check_copy, &state));
    DRV(cuEventRecord(join, other));
    DRV(cuStreamWaitEvent(stream, join, 0));
    DRV(cuStreamEndCapture(stream, &graph));
    DRV(cuGraphClone(&clone, graph));
    DRV(cuGraphDestroy(graph));
    DRV(cuGraphInstantiateWithFlags(&exec, clone, 0));
    DRV(cuGraphDestroy(clone));
    // Destroying an executable leaves launches running. This delay ensures its
    // copy and host callback have not run when destruction is requested.
    DRV(cuLaunchHostFunc(stream, delay_launch, nullptr));
    DRV(cuGraphLaunch(exec, stream));
    DRV(cuGraphExecDestroy(exec));
    DRV(cuStreamSynchronize(stream));
    if (*host != state.expected || state.failed || state.calls != before + 1) {
      fprintf(stderr, "graph storage/callback did not survive destruction\n");
      return 1;
    }
  }

  // Invalidated captures must release their reservation, and a failed second
  // begin must leave the original capture's resource owner intact.
  for (int iteration = 0; iteration < 8; ++iteration) {
    CUgraph graph = nullptr;
    DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    if (cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
        CUDA_SUCCESS) {
      fprintf(stderr, "a second begin unexpectedly succeeded\n");
      return 1;
    }
    DRV(cuMemcpyDtoHAsync(host, device, sizeof(*host), stream));
    if (cuStreamSynchronize(stream) != CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED ||
        cuStreamEndCapture(stream, &graph) !=
            CUDA_ERROR_STREAM_CAPTURE_INVALIDATED) {
      fprintf(stderr, "capture invalidation was not reported\n");
      return 1;
    }
  }

#if CUDA_VERSION >= 12030
  // Appending a capture keeps resources already attached to the graph and
  // reserves DtoH staging just like a fresh capture.
  CUgraph appended;
  CUgraphNode first;
  DRV(cuGraphCreate(&appended, 0));
  CUDA_HOST_NODE_PARAMS params{count_calls, &state};
  *host = state.expected;
  unsigned int before = state.calls;
  DRV(cuGraphAddHostNode(&first, appended, nullptr, 0, &params));
  DRV(cuStreamBeginCaptureToGraph(stream, appended, &first, nullptr, 1,
                                  CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemcpyDtoHAsync(host, device, sizeof(*host), stream));
  CUgraph returned;
  DRV(cuStreamEndCapture(stream, &returned));
  if (returned != appended) {
    fprintf(stderr, "capture did not return the existing graph\n");
    return 1;
  }
  CUgraphExec appended_exec;
  DRV(cuGraphInstantiateWithFlags(&appended_exec, appended, 0));
  DRV(cuGraphHostNodeSetParams(first, &params));
  DRV(cuGraphExecHostNodeSetParams(appended_exec, first, &params));
  DRV(cuGraphDestroy(appended));
  DRV(cuGraphLaunch(appended_exec, stream));
  DRV(cuGraphExecDestroy(appended_exec));
  DRV(cuStreamSynchronize(stream));
  if (state.failed || state.calls != before + 1) {
    fprintf(stderr, "existing graph callback ownership was lost\n");
    return 1;
  }
#endif

  DRV(cuStreamDestroy(other));
  DRV(cuEventDestroy(join));
  DRV(cuEventDestroy(fork));
  DRV(cuMemFreeHost(host));
  return 0;
}

static int check_pageable_htod(CUstream stream, CUdeviceptr device) {
  unsigned int source = 91;
  unsigned int output = 0;
  CUgraph graph, clone;
  CUgraphExec exec;
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemcpyHtoDAsync(device, &source, sizeof(source), stream));
  DRV(cuStreamEndCapture(stream, &graph));
  DRV(cuGraphClone(&clone, graph));
  DRV(cuGraphDestroy(graph));
  DRV(cuGraphInstantiateWithFlags(&exec, clone, 0));
  DRV(cuGraphDestroy(clone));
  DRV(cuLaunchHostFunc(stream, delay_launch, nullptr));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuStreamSynchronize(stream));
  DRV(cuMemcpyDtoH(&output, device, sizeof(output)));
  if (output != source) {
    fprintf(stderr, "HtoD callback storage did not survive exec destruction\n");
    return 1;
  }
  return 0;
}

static int capture_copy(CUstream stream, CUdeviceptr device,
                        callback_state *state, CUgraph *graph) {
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  DRV(cuMemcpyHtoDAsync(device, &state->expected, sizeof(state->expected),
                        stream));
  DRV(cuMemcpyDtoHAsync(state->value, device, sizeof(*state->value), stream));
  DRV(cuLaunchHostFunc(stream, check_copy, state));
  DRV(cuStreamEndCapture(stream, graph));
  return 0;
}

static int check_exec_update(CUstream stream, CUdeviceptr device, bool legacy) {
  unsigned int *host = nullptr;
  DRV(cuMemAllocHost(reinterpret_cast<void **>(&host), 2 * sizeof(*host)));
  host[0] = host[1] = 0;
  callback_state original{host, 81};
  callback_state updated{host + 1, 92};
  CUgraph graph;
  if (capture_copy(stream, device, &original, &graph)) {
    return 1;
  }
  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DRV(cuGraphDestroy(graph));

  auto update = [&](CUgraph next, CUgraphExecUpdateResult *out) {
    if (legacy) {
      using legacy_fn = CUresult(CUDAAPI *)(CUgraphExec, CUgraph, CUgraphNode *,
                                            CUgraphExecUpdateResult *);
      auto fn =
          reinterpret_cast<legacy_fn>(dlsym(RTLD_DEFAULT, "cuGraphExecUpdate"));
      CUgraphNode error = nullptr;
      return fn == nullptr ? CUDA_ERROR_NOT_FOUND : fn(exec, next, &error, out);
    }
#if CUDA_VERSION >= 12000
    CUgraphExecUpdateResultInfo info{};
    CUresult result = cuGraphExecUpdate(exec, next, &info);
    *out = info.result;
    return result;
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  };

  // A rejected update must keep the original executable's owners and buffers.
  CUgraph incompatible;
  CUgraphNode empty;
  DRV(cuGraphCreate(&incompatible, 0));
  DRV(cuGraphAddEmptyNode(&empty, incompatible, nullptr, 0));
  CUgraphExecUpdateResult detail = CU_GRAPH_EXEC_UPDATE_SUCCESS;
  if (update(incompatible, &detail) != CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE ||
      detail == CU_GRAPH_EXEC_UPDATE_SUCCESS) {
    fprintf(stderr, "incompatible graph update was not rejected\n");
    return 1;
  }
  DRV(cuGraphDestroy(incompatible));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  if (host[0] != original.expected || original.failed || original.calls != 1) {
    fprintf(stderr, "failed graph update lost original resources\n");
    return 1;
  }

  if (capture_copy(stream, device, &updated, &graph)) {
    return 1;
  }
  DRV(update(graph, &detail));
  if (detail != CU_GRAPH_EXEC_UPDATE_SUCCESS) {
    fprintf(stderr, "graph update did not succeed\n");
    return 1;
  }
  DRV(cuGraphDestroy(graph));
  host[0] = 123;
  DRV(cuLaunchHostFunc(stream, delay_launch, nullptr));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuStreamSynchronize(stream));
  if (host[0] != 123 || host[1] != updated.expected || updated.failed ||
      updated.calls != 1 || original.calls != 1) {
    fprintf(stderr, "successful graph update used stale/freed resources\n");
    return 1;
  }
  DRV(cuMemFreeHost(host));
  return 0;
}

int main(int argc, char **argv) {
  setbuf(stdout, nullptr);
  DRV(cuInit(0));
  CUcontext context;
  DRV(cuDevicePrimaryCtxRetain(&context, 0));
  DRV(cuCtxSetCurrent(context));
  CUstream stream;
  DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr device;
  DRV(cuMemAlloc(&device, sizeof(unsigned int)));

  // RSS is sampled after warm-up. Each empty capture used to leak 128 MiB.
  bool memory_steps = argc == 2 && strcmp(argv[1], "--memory-steps") == 0;
  for (int iteration = 0; iteration <= 8; ++iteration) {
    CUgraph graph;
    DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
    DRV(cuStreamEndCapture(stream, &graph));
    DRV(cuGraphDestroy(graph));
    DRV(cuStreamSynchronize(stream));
    if (memory_steps) {
      printf("memory-step=%d\n", iteration);
      if (getchar() == EOF) {
        return 1;
      }
    }
  }
  if (check_lifetimes(stream, device) || check_pageable_htod(stream, device) ||
      check_exec_update(stream, device, true)
#if CUDA_VERSION >= 12000
      || check_exec_update(stream, device, false)
#endif
  ) {
    return 1;
  }
  DRV(cuMemFree(device));
  DRV(cuStreamDestroy(stream));
  DRV(cuDevicePrimaryCtxRelease(0));
  puts("PASS: graph resource lifetimes");
  return 0;
}
