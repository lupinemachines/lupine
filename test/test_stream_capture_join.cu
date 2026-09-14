#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

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

struct capture_state {
  CUstreamCaptureStatus status;
  cuuint64_t id;
  CUgraph graph;
};

// Asks for the dependency set, which only the driver can answer.
static capture_state driver_state(CUstream stream) {
  capture_state state = {CU_STREAM_CAPTURE_STATUS_NONE, 0, nullptr};
  const CUgraphNode *dependencies = nullptr;
  size_t count = 0;
  require(cuStreamGetCaptureInfo_v2(stream, &state.status, &state.id,
                                    &state.graph, &dependencies,
                                    &count) == CUDA_SUCCESS,
          "dependency query failed");
  require(state.status != CU_STREAM_CAPTURE_STATUS_ACTIVE ||
              (count != 0 && dependencies != nullptr),
          "active capture lost its dependencies");
  return state;
}

// The status/id/graph forms libraries poll on every allocation.
static void check_same(CUstream stream, capture_state expected) {
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  require(cuStreamIsCapturing(stream, &status) == CUDA_SUCCESS &&
              status == expected.status,
          "cuStreamIsCapturing disagrees with the driver");
  status = CU_STREAM_CAPTURE_STATUS_NONE;
  require(cuStreamGetCaptureInfo(stream, &status, &id) == CUDA_SUCCESS &&
              status == expected.status && id == expected.id,
          "legacy query disagrees with the driver");
  status = CU_STREAM_CAPTURE_STATUS_NONE;
  id = 0;
  require(cuStreamGetCaptureInfo_v2(stream, &status, &id, &graph, nullptr,
                                    nullptr) == CUDA_SUCCESS &&
              status == expected.status && id == expected.id &&
              graph == expected.graph,
          "v2 query disagrees with the driver");
#if CUDA_VERSION >= 12030
  status = CU_STREAM_CAPTURE_STATUS_NONE;
  id = 0;
  graph = nullptr;
  require(cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph, nullptr,
                                    nullptr, nullptr) == CUDA_SUCCESS &&
              status == expected.status && id == expected.id &&
              graph == expected.graph,
          "v3 query disagrees with the driver");
#endif
}

static void check_active(CUstream stream, capture_state expected) {
  capture_state state = driver_state(stream);
  require(state.status == CU_STREAM_CAPTURE_STATUS_ACTIVE &&
              state.id == expected.id && state.graph == expected.graph,
          "stream is not in the expected capture");
  for (int i = 0; i < 50; ++i) {
    check_same(stream, expected);
  }
}

static void check_none(CUstream stream, int repeats = 50) {
  capture_state none = {CU_STREAM_CAPTURE_STATUS_NONE, 0, nullptr};
  require(driver_state(stream).status == CU_STREAM_CAPTURE_STATUS_NONE,
          "stream is still capturing");
  for (int i = 0; i < repeats; ++i) {
    check_same(stream, none);
  }
}

int main() {
  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  CUdevice device;
  CUcontext context;
  require(cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS &&
              cuCtxSetCurrent(context) == CUDA_SUCCESS,
          "context setup failed");
  CUstream origin = nullptr, side = nullptr, idle = nullptr;
  CUevent fork = nullptr, join = nullptr;
  require(cuStreamCreate(&origin, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
              cuStreamCreate(&side, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
              cuStreamCreate(&idle, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
              cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS &&
              cuEventCreate(&join, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS,
          "stream setup failed");
  CUdeviceptr memory;
  require(cuMemAlloc(&memory, 64) == CUDA_SUCCESS, "allocation failed");

  // A side stream joins through an event wait, keeps the origin's id and
  // graph through its rejoin, and leaves the capture with the origin.
  require(cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
                  CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 1, 64, origin) == CUDA_SUCCESS,
          "begin capture failed");
  capture_state first = driver_state(origin);
  require(first.status == CU_STREAM_CAPTURE_STATUS_ACTIVE && first.id != 0 &&
              first.graph != nullptr,
          "origin is not capturing");
  check_active(origin, first);
  check_none(idle, 3);
  check_none(side, 3);
  require(cuEventRecord(fork, origin) == CUDA_SUCCESS &&
              cuStreamWaitEvent(side, fork, 0) == CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 2, 64, side) == CUDA_SUCCESS,
          "side stream join failed");
  check_active(side, first);
  require(cuEventRecord(join, side) == CUDA_SUCCESS &&
              cuStreamWaitEvent(origin, join, 0) == CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 3, 64, origin) == CUDA_SUCCESS,
          "side stream rejoin failed");
  check_active(side, first);
  check_active(origin, first);
  check_none(idle, 3);
  std::thread other([&] {
    require(cuCtxSetCurrent(context) == CUDA_SUCCESS, "thread context failed");
    check_active(origin, first);
    check_active(side, first);
  });
  other.join();
  CUgraph graph = nullptr;
  require(cuStreamEndCapture(origin, &graph) == CUDA_SUCCESS &&
              graph == first.graph,
          "end capture failed");
  require(cuGraphDestroy(graph) == CUDA_SUCCESS, "graph destroy failed");
  check_none(origin);
  check_none(side);
  check_none(idle);

  // A new capture gets a new id; nothing from the first one survives.
  require(cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
                  CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 4, 64, origin) == CUDA_SUCCESS,
          "second begin failed");
  capture_state second = driver_state(origin);
  require(second.status == CU_STREAM_CAPTURE_STATUS_ACTIVE &&
              second.id != first.id,
          "second capture reused the first id");
  check_active(origin, second);
  check_none(side, 3);
  require(cuEventRecord(fork, origin) == CUDA_SUCCESS &&
              cuStreamWaitEvent(side, fork, 0) == CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 5, 64, side) == CUDA_SUCCESS,
          "second join failed");
  check_active(side, second);
  require(cuStreamEndCapture(origin, &graph) ==
              CUDA_ERROR_STREAM_CAPTURE_UNJOINED,
          "unjoined capture did not fail");
  check_none(origin);
  check_none(side);

  // Invalidation is reported once EndCapture terminates the capture.
  require(cuStreamBeginCapture(origin, CU_STREAM_CAPTURE_MODE_GLOBAL) ==
                  CUDA_SUCCESS &&
              cuMemsetD8Async(memory, 6, 64, origin) == CUDA_SUCCESS,
          "third begin failed");
  capture_state third = driver_state(origin);
  check_active(origin, third);
  require(cuStreamSynchronize(origin) == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
          "capture was not invalidated by synchronization");
  require(driver_state(origin).status == CU_STREAM_CAPTURE_STATUS_INVALIDATED,
          "invalidated capture is not reported");
  require(cuStreamEndCapture(origin, &graph) ==
              CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
          "invalidated capture did not terminate");
  check_none(origin);
  check_none(side);
  check_none(idle);

  require(cuEventDestroy(fork) == CUDA_SUCCESS &&
              cuEventDestroy(join) == CUDA_SUCCESS &&
              cuStreamDestroy(origin) == CUDA_SUCCESS &&
              cuStreamDestroy(side) == CUDA_SUCCESS &&
              cuStreamDestroy(idle) == CUDA_SUCCESS &&
              cuMemFree(memory) == CUDA_SUCCESS,
          "cleanup failed");
  require(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS,
          "context cleanup failed");
  std::puts("capture info stays consistent with the driver across repeated "
            "queries, joined streams, and capture end");
}
