#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *, void **, int, cuuint64_t);

#ifdef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo
#endif
extern "C" CUresult cuStreamGetCaptureInfo(CUstream, CUstreamCaptureStatus *,
                                           cuuint64_t *);

extern "C" CUresult cuStreamGetCaptureInfo_v2(CUstream, CUstreamCaptureStatus *,
                                              cuuint64_t *, CUgraph *,
                                              const CUgraphNode **, size_t *);

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static void check_status(CUstream stream, CUstreamCaptureStatus expected) {
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode *dependencies = nullptr;
  size_t count = 0;
  require(cuStreamGetCaptureInfo(stream, &status, &id) == CUDA_SUCCESS &&
              status == expected,
          "legacy capture status is incorrect");
  require(cuStreamGetCaptureInfo_v2(stream, &status, &id, &graph, &dependencies,
                                    &count) == CUDA_SUCCESS &&
              status == expected,
          "v2 capture status is incorrect");
  if (expected == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    require(graph != nullptr && id != 0, "active capture metadata is missing");
    require(count != 0 && dependencies != nullptr,
            "captured dependency metadata is missing");
  }
  using CaptureInfoV1 =
      CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
  using CaptureInfoV2 =
      CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *,
                   const CUgraphNode **, size_t *);
  void *function = nullptr;
  require(cuGetProcAddress("cuStreamGetCaptureInfo", &function, 10010, 0) ==
              CUDA_SUCCESS,
          "v1 lookup failed");
  require(reinterpret_cast<CaptureInfoV1>(function)(stream, &status, &id) ==
                  CUDA_SUCCESS &&
              status == expected,
          "versioned v1 capture query failed");
  require(cuGetProcAddress("cuStreamGetCaptureInfo", &function, 11030, 0) ==
              CUDA_SUCCESS,
          "v2 lookup failed");
  count = 0;
  require(reinterpret_cast<CaptureInfoV2>(function)(stream, &status, &id,
                                                    &graph, &dependencies,
                                                    &count) == CUDA_SUCCESS &&
              status == expected,
          "versioned v2 capture query failed");
  if (expected == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    require(count != 0 && dependencies != nullptr,
            "versioned v2 lost capture dependencies");
  }
#if CUDA_VERSION >= 12030
  const CUgraphEdgeData *edges = nullptr;
  require(cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph, &dependencies,
                                    &edges, &count) == CUDA_SUCCESS &&
              status == expected,
          "v3 capture status is incorrect");
  using CaptureInfoV3 =
      CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *,
                   const CUgraphNode **, const CUgraphEdgeData **, size_t *);
  require(cuGetProcAddress("cuStreamGetCaptureInfo", &function, 12030, 0) ==
              CUDA_SUCCESS,
          "v3 lookup failed");
  require(reinterpret_cast<CaptureInfoV3>(function)(
              stream, &status, &id, &graph, &dependencies, &edges, &count) ==
                  CUDA_SUCCESS &&
              status == expected,
          "versioned v3 capture query failed");
#endif
}

int main() {
  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  CUdevice device;
  CUcontext context;
  require(cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS &&
              cuCtxSetCurrent(context) == CUDA_SUCCESS,
          "context setup failed");
  CUstream stream = nullptr, idle = nullptr;
  require(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS &&
              cuStreamCreate(&idle, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
          "stream creation failed");
  CUdeviceptr memory;
  require(cuMemAlloc(&memory, 64) == CUDA_SUCCESS, "allocation failed");
  for (int i = 0; i < 100; ++i) {
    check_status(stream, CU_STREAM_CAPTURE_STATUS_NONE);
    check_status(nullptr, CU_STREAM_CAPTURE_STATUS_NONE);
  }
  require(cuStreamBeginCapture(stream, static_cast<CUstreamCaptureMode>(999)) ==
              CUDA_ERROR_INVALID_VALUE,
          "invalid begin did not fail");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_NONE);

  require(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
              CUDA_SUCCESS,
          "begin capture failed");
  require(cuMemsetD8Async(memory, 7, 64, stream) == CUDA_SUCCESS,
          "captured memset failed");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_ACTIVE);
  check_status(idle, CU_STREAM_CAPTURE_STATUS_NONE);
  CUstreamCaptureStatus status;
  require(cuStreamGetCaptureInfo_v2(nullptr, &status, nullptr, nullptr, nullptr,
                                    nullptr) ==
              CUDA_ERROR_STREAM_CAPTURE_IMPLICIT,
          "legacy stream did not report implicit capture");
  std::thread other([&] {
    require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "thread context failed");
    check_status(stream, CU_STREAM_CAPTURE_STATUS_ACTIVE);
  });
  other.join();
  CUgraph graph = nullptr;
  require(cuStreamEndCapture(stream, &graph) == CUDA_SUCCESS &&
              graph != nullptr,
          "end capture failed");
  require(cuGraphDestroy(graph) == CUDA_SUCCESS, "graph destroy failed");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_NONE);

  require(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
              CUDA_SUCCESS,
          "second begin failed");
  require(cuStreamSynchronize(stream) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
          "capture was not invalidated by synchronization");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_INVALIDATED);
  require(cuStreamEndCapture(stream, &graph) ==
              CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
          "invalidated capture did not terminate");
  for (int i = 0; i < 100; ++i) {
    check_status(stream, CU_STREAM_CAPTURE_STATUS_NONE);
  }
  // The runtime and driver share capture bookkeeping on the same streams.
  require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
                  cudaSuccess &&
              cudaMemsetAsync(reinterpret_cast<void *>(memory), 9, 64,
                              stream) == cudaSuccess,
          "runtime capture failed");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_ACTIVE);
  cudaGraph_t runtime_graph = nullptr;
  require(cudaStreamEndCapture(stream, &runtime_graph) == cudaSuccess &&
              cudaGraphDestroy(runtime_graph) == cudaSuccess,
          "runtime end capture failed");
  check_status(stream, CU_STREAM_CAPTURE_STATUS_NONE);
  require(cuStreamDestroy(stream) == CUDA_SUCCESS &&
              cuStreamDestroy(idle) == CUDA_SUCCESS &&
              cuMemFree(memory) == CUDA_SUCCESS,
          "cleanup failed");
  require(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS,
          "context cleanup failed");
  std::puts("capture info preserves inactive, active, implicit, invalidated, "
            "and cross-thread capture state");
}
