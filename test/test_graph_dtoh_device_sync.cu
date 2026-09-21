// Regression for #866: device/context waits must deliver graph DtoH bytes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <initializer_list>

#if CUDA_VERSION >= 13000
extern "C" CUresult CUDAAPI cuCtxSynchronize_v2(CUcontext ctx);
#endif

static void check(cudaError_t result, const char *call) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(result));
    std::exit(2);
  }
}

static void check(CUresult result, const char *call) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s: driver error %d\n", call, int(result));
    std::exit(2);
  }
}

#define CHECK(call) check((call), #call)

enum class Wait { Stream, Device, Context, ExplicitContext };

static void wait_for(Wait wait, cudaStream_t stream) {
  switch (wait) {
  case Wait::Stream:
    CHECK(cudaStreamSynchronize(stream));
    break;
  case Wait::Device:
    CHECK(cudaDeviceSynchronize());
    break;
  case Wait::Context:
    CHECK(cuCtxSynchronize());
    break;
  case Wait::ExplicitContext:
#if CUDA_VERSION >= 13000
    CUcontext context = nullptr;
    CHECK(cuCtxGetCurrent(&context));
    CHECK(cuCtxSynchronize_v2(context));
#endif
    break;
  }
}

static constexpr size_t kBytes = 4096;

static int run_case(int device, bool explicit_stream, int form, Wait wait,
                    const char *wait_name) {
  CHECK(cudaSetDevice(device));
  unsigned char *source = nullptr;
  unsigned char *destination = nullptr;
  CHECK(cudaMalloc(&source, kBytes));
  CHECK(cudaMallocHost(&destination, kBytes));
  cudaStream_t stream = nullptr;
  if (explicit_stream) {
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (form == 2) {
    cudaStream_t capture_stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking));
    CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    CHECK(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToHost,
                          capture_stream));
    CHECK(cudaStreamEndCapture(capture_stream, &graph));
    CHECK(cudaStreamDestroy(capture_stream));
  } else if (form < 2) {
    CHECK(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t node = nullptr;
    if (form == 0) {
      CHECK(cudaGraphAddMemcpyNode1D(&node, graph, nullptr, 0, destination,
                                     source, kBytes, cudaMemcpyDeviceToHost));
    } else {
      cudaMemcpy3DParms copy{};
      copy.srcPtr = make_cudaPitchedPtr(source, kBytes, kBytes, 1);
      copy.dstPtr = make_cudaPitchedPtr(destination, kBytes, kBytes, 1);
      copy.extent = make_cudaExtent(kBytes, 1, 1);
      copy.kind = cudaMemcpyDeviceToHost;
      CHECK(cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &copy));
    }
  }
  if (graph != nullptr) {
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  }

  int failures = 0;
  for (int replay = 0; replay < 2; ++replay) {
    const unsigned char expected = 41 + device + replay;
    CHECK(cudaMemset(source, expected, kBytes));
    CHECK(cudaDeviceSynchronize());
    std::memset(destination, 0, kBytes);
    if (exec != nullptr) {
      CHECK(cudaGraphLaunch(exec, stream));
    } else {
      CHECK(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToHost,
                            stream));
    }
    wait_for(wait, stream);
    const char *forms[] = {"graph-1D", "graph-3D", "captured", "async"};
    for (size_t i = 0; i < kBytes; ++i) {
      if (destination[i] != expected) {
        std::printf("FAIL device=%d stream=%s copy=%s wait=%s replay=%d "
                    "byte=%zu got=%u expected=%u\n",
                    device, explicit_stream ? "explicit" : "default",
                    forms[form], wait_name, replay, i, destination[i],
                    expected);
        ++failures;
        break;
      }
    }
    // A later stream wait must not replay already-delivered graph bytes.
    std::memset(destination, 99, kBytes);
    CHECK(cudaStreamSynchronize(stream));
    if (destination[0] != 99) {
      std::printf("FAIL stale delivery after %s\n", wait_name);
      ++failures;
    }
  }
  if (exec != nullptr) {
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
  }
  if (explicit_stream) {
    CHECK(cudaStreamDestroy(stream));
  }
  CHECK(cudaFreeHost(destination));
  CHECK(cudaFree(source));
  if (failures == 0) {
    std::printf("PASS device=%d stream=%s form=%d wait=%s\n", device,
                explicit_stream ? "explicit" : "default", form, wait_name);
  }
  return failures;
}

static int run_multiple_graphs(int devices, bool explicit_stream,
                               bool stream_wait) {
  struct State {
    CUcontext context = nullptr;
    cudaStream_t stream = nullptr;
    unsigned char *source = nullptr;
    unsigned char *destination = nullptr;
    cudaGraph_t graphs[3]{};
    cudaGraphExec_t execs[3]{};
  };
  State states[2];
  const int count = devices < 2 ? devices : 2;
  for (int device = 0; device < count; ++device) {
    CHECK(cudaSetDevice(device));
    auto &state = states[device];
    CHECK(cudaMalloc(&state.source, 2 * kBytes));
    CHECK(cudaMallocHost(&state.destination, 2 * kBytes));
    std::memset(state.destination, 0, 2 * kBytes);
    CHECK(cuCtxGetCurrent(&state.context));
    CHECK(cudaMemset(state.source, 11, kBytes));
    CHECK(cudaMemset(state.source + kBytes, 22, kBytes));
    CHECK(cudaDeviceSynchronize());
    if (explicit_stream) {
      CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    }
    for (int graph = 0; graph < 3; ++graph) {
      CHECK(cudaGraphCreate(&state.graphs[graph], 0));
      cudaGraphNode_t node = nullptr;
      CHECK(cudaGraphAddMemcpyNode1D(&node, state.graphs[graph], nullptr, 0,
                                     state.destination +
                                         (graph == 1 ? kBytes : 0),
                                     state.source + (graph == 0 ? 0 : kBytes),
                                     kBytes, cudaMemcpyDeviceToHost));
      CHECK(cudaGraphInstantiate(&state.execs[graph], state.graphs[graph],
                                 nullptr, nullptr, 0));
    }
  }
  // Keep graphs pending in both contexts, including distinct graphs on the
  // same stream and two graphs that overwrite the same host destination.
  for (int device = 0; device < count; ++device) {
    CHECK(cudaSetDevice(device));
    auto &state = states[device];
    for (int graph = 0; graph < 3; ++graph) {
      CHECK(cudaGraphLaunch(state.execs[graph], state.stream));
      // CUDA keeps queued work alive after the graph handles are destroyed.
      CHECK(cudaGraphExecDestroy(state.execs[graph]));
      CHECK(cudaGraphDestroy(state.graphs[graph]));
    }
  }
  int failures = 0;
  for (int device = 0; device < count; ++device) {
    auto &state = states[device];
    if (stream_wait) {
      CHECK(cudaSetDevice(device));
      CHECK(cudaStreamSynchronize(state.stream));
    } else {
#if CUDA_VERSION >= 13000
      // Device 1 is still current when explicitly synchronizing context 0.
      CHECK(cuCtxSynchronize_v2(state.context));
#else
      CHECK(cudaSetDevice(device));
      CHECK(cudaDeviceSynchronize());
#endif
    }
    for (size_t i = 0; i < 2 * kBytes; ++i) {
      if (state.destination[i] != 22) {
        std::printf("FAIL multiple graphs device=%d explicit_stream=%d "
                    "byte=%zu got=%u expected=22\n",
                    device, explicit_stream, i, state.destination[i]);
        ++failures;
        break;
      }
    }
  }
  for (int device = 0; device < count; ++device) {
    CHECK(cudaSetDevice(device));
    auto &state = states[device];
    std::memset(state.destination, 99, 2 * kBytes);
    CHECK(cudaDeviceSynchronize());
    if (state.destination[0] != 99 || state.destination[kBytes] != 99) {
      std::printf("FAIL stale multiple-graph delivery on device=%d\n", device);
      ++failures;
    }
    if (explicit_stream) {
      CHECK(cudaStreamDestroy(state.stream));
    }
    CHECK(cudaFreeHost(state.destination));
    CHECK(cudaFree(state.source));
  }
  std::printf("%s multiple graphs in %d contexts, explicit_stream=%d "
              "stream_wait=%d\n",
              failures == 0 ? "PASS" : "FAIL", count, explicit_stream,
              stream_wait);
  return failures;
}

int main() {
  CHECK(cuInit(0));
  int devices = 0;
  CHECK(cudaGetDeviceCount(&devices));
  int failures = 0;
  for (int device = 0; device < devices; ++device) {
    for (bool explicit_stream : {false, true}) {
      for (int form = 0; form < 4; ++form) {
        failures += run_case(device, explicit_stream, form, Wait::Stream,
                             "cudaStreamSynchronize");
        failures += run_case(device, explicit_stream, form, Wait::Device,
                             "cudaDeviceSynchronize");
        failures += run_case(device, explicit_stream, form, Wait::Context,
                             "cuCtxSynchronize");
#if CUDA_VERSION >= 13000
        failures += run_case(device, explicit_stream, form,
                             Wait::ExplicitContext, "cuCtxSynchronize_v2");
#endif
      }
    }
  }
  for (bool explicit_stream : {false, true}) {
    for (bool stream_wait : {false, true}) {
      failures += run_multiple_graphs(devices, explicit_stream, stream_wait);
    }
  }
  std::printf("RESULT: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL",
              failures);
  return failures == 0 ? 0 : 1;
}
