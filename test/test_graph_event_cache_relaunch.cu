// A completion the client caches for an event does not outlive a relaunch of
// a graph that records the event: the launch does not name the event, but
// the event is pending again until the graph's work runs.
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "FAIL: %s: %s\n", #call,                            \
                   cudaGetErrorString(check_result_));                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

__global__ void spin(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

} // namespace

int main() {
  CHECK(cudaSetDevice(0));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  cudaStream_t stream = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t recorded = nullptr;
  CHECK(cudaEventCreateWithFlags(&recorded, cudaEventDisableTiming));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  spin<<<1, 1, 0, stream>>>(static_cast<long long>(khz) * 50); // ~50 ms
  CHECK(cudaEventRecordWithFlags(recorded, stream, cudaEventRecordExternal));
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  int stale = 0;
  for (int round = 0; round < 5; ++round) {
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventQuery(recorded)); // complete, and cached as such
    CHECK(cudaGraphLaunch(exec, stream));
    stale += cudaEventQuery(recorded) != cudaErrorNotReady;
    CHECK(cudaEventSynchronize(recorded));
  }
  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaEventDestroy(recorded));
  CHECK(cudaStreamDestroy(stream));
  if (stale != 0) {
    std::fprintf(stderr, "FAIL: %d/5 relaunches answered from a stale cache\n",
                 stale);
    return 1;
  }
  std::printf("PASS: relaunching a graph voids its event's cached completion\n");
  return 0;
}
