#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

struct CallbackState {
  std::atomic<int> launch_host_count{0};
  std::atomic<int> stream_callback_count{0};
  CUresult stream_callback_status = CUDA_ERROR_UNKNOWN;
};

static const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "UNKNOWN";
}

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 result_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static void CUDA_CB launch_host_fn(void *user_data) {
  auto *state = static_cast<CallbackState *>(user_data);
  state->launch_host_count.fetch_add(1, std::memory_order_relaxed);
}

static void CUDA_CB stream_callback(CUstream, CUresult status,
                                    void *user_data) {
  auto *state = static_cast<CallbackState *>(user_data);
  state->stream_callback_status = status;
  state->stream_callback_count.fetch_add(1, std::memory_order_relaxed);
}

int main() {
  CHECK(cuInit(0));

  int device_count = 0;
  CHECK(cuDeviceGetCount(&device_count));
  if (device_count < 2) {
    std::printf("SKIP: need at least two enumerated CUDA devices, found %d\n",
                device_count);
    return EXIT_SUCCESS;
  }

  CUdevice first_device = 0;
  CHECK(cuDeviceGet(&first_device, 0));

  CUcontext first_context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&first_context, nullptr, 0, first_device));
#else
  CHECK(cuCtxCreate(&first_context, 0, first_device));
#endif

  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, device_count - 1));

  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CallbackState state;
  CHECK(cuLaunchHostFunc(stream, launch_host_fn, &state));
  CHECK(cuStreamAddCallback(stream, stream_callback, &state, 0));
  CHECK(cuStreamSynchronize(stream));

  int launch_count = state.launch_host_count.load(std::memory_order_relaxed);
  int callback_count =
      state.stream_callback_count.load(std::memory_order_relaxed);
  if (launch_count != 1 || callback_count != 1 ||
      state.stream_callback_status != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "unexpected callback state: launch=%d stream=%d status=%s "
                 "(%d)\n",
                 launch_count, callback_count,
                 result_name(state.stream_callback_status),
                 static_cast<int>(state.stream_callback_status));
    return EXIT_FAILURE;
  }

  CUgraph graph = nullptr;
  CHECK(cuGraphCreate(&graph, 0));
  CHECK(cuCtxSetCurrent(first_context));

  CUDA_HOST_NODE_PARAMS host_params = {};
  host_params.fn = launch_host_fn;
  host_params.userData = &state;
  CUgraphNode host_node = nullptr;
  CHECK(cuGraphAddHostNode(&host_node, graph, nullptr, 0, &host_params));

  CUDA_HOST_NODE_PARAMS got_params = {};
  CHECK(cuGraphHostNodeGetParams(host_node, &got_params));
  if (got_params.fn != launch_host_fn || got_params.userData != &state) {
    std::fprintf(stderr, "unexpected host node params after context switch\n");
    return EXIT_FAILURE;
  }

  size_t edge_count = 0;
#ifdef cuGraphGetEdges
  CHECK(cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count));
#else
  CHECK(cuGraphGetEdges(graph, nullptr, nullptr, &edge_count));
#endif
  if (edge_count != 0) {
    std::fprintf(stderr, "unexpected edge count: %zu\n", edge_count);
    return EXIT_FAILURE;
  }

  CHECK(cuStreamDestroy(stream));
  CHECK(cuGraphDestroy(graph));
  CHECK(cuCtxDestroy(context));
  CHECK(cuCtxDestroy(first_context));
  std::printf("PASS: stream callbacks routed on device ordinal %d\n",
              device_count - 1);
  return EXIT_SUCCESS;
}
