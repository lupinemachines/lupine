// Graph memory-pool footprint growth across concurrent streams.
//
// The upstream graphMemoryFootprint sample prints footprints but asserts
// nothing, and its delay kernel only runs ~5us. That is shorter than a lupine
// RPC round trip, so each graph completes before the next launch is issued and
// CUDA recycles the physical memory: the footprint never grows and the sample
// still exits zero. This test pins the overlap window open with a delay long
// enough to outlast the launch loop, verifies that precondition explicitly, and
// then asserts the footprint actually grows one allocation per concurrent
// stream.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define NUM_GRAPHS 8

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

#define RT(call)                                                               \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call,                    \
              cudaGetErrorString(_e), __LINE__);                               \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Spins on the SM clock so the graph stays resident long enough for the next
// launch to observe its allocation as live.
__global__ void clockBlock(clock_t clock_count) {
  unsigned int start = (unsigned int)clock();
  clock_t elapsed = 0;
  while (elapsed < clock_count) {
    elapsed = (clock_t)((unsigned int)clock() - start);
  }
}

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static cudaError_t buildAllocDelayFreeGraph(cudaGraphExec_t *graphExec,
                                            void **dptr, size_t bytes,
                                            int device, clock_t delay_clocks) {
  cudaGraph_t graph;
  cudaGraphNode_t allocNode, kernelNode, freeNode;
  cudaMemAllocNodeParams allocParams = {};
  cudaKernelNodeParams kernelParams = {};
  cudaError_t err;

  allocParams.bytesize = bytes;
  allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
  allocParams.poolProps.location.id = device;
  allocParams.poolProps.location.type = cudaMemLocationTypeDevice;

  if ((err = cudaGraphCreate(&graph, 0)) != cudaSuccess)
    return err;
  if ((err = cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0,
                                      &allocParams)) != cudaSuccess)
    return err;
  *dptr = allocParams.dptr;

  void *args[1] = {(void *)&delay_clocks};
  kernelParams.gridDim = dim3(1, 1, 1);
  kernelParams.blockDim = dim3(1, 1, 1);
  kernelParams.func = (void *)clockBlock;
  kernelParams.kernelParams = args;
  if ((err = cudaGraphAddKernelNode(&kernelNode, graph, &allocNode, 1,
                                    &kernelParams)) != cudaSuccess)
    return err;
  if ((err = cudaGraphAddMemFreeNode(&freeNode, graph, &kernelNode, 1,
                                     *dptr)) != cudaSuccess)
    return err;
  if ((err = cudaGraphInstantiate(graphExec, graph, NULL, NULL, 0)) !=
      cudaSuccess)
    return err;
  return cudaGraphDestroy(graph);
}

int main() {
  const int device = 0;
  const size_t bytes = 64 * 1024 * 1024;
  // Long enough that the whole launch loop finishes while every graph is still
  // running; the loop time is checked below so a slow link fails loudly instead
  // of silently dropping the overlap this test exists to exercise.
  double delay_ms = 500.0;
  if (const char *env = getenv("LUPINE_GRAPH_DELAY_MS")) {
    double v = atof(env);
    if (v > 0)
      delay_ms = v;
  }

  RT(cudaSetDevice(device));

  int poolsSupported = 0;
  RT(cudaDeviceGetAttribute(&poolsSupported, cudaDevAttrMemoryPoolsSupported,
                            device));
  if (!poolsSupported) {
    fprintf(stderr, "SKIP: device does not support memory pools\n");
    return 0;
  }

  int clockRate = 0; // kHz
  RT(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device));
  CHECK(clockRate > 0, "device reports a clock rate");
  if (clockRate <= 0)
    return 1;
  clock_t delay_clocks = (clock_t)(delay_ms * clockRate);

  cudaGraphExec_t graphExecs[NUM_GRAPHS];
  cudaStream_t streams[NUM_GRAPHS];
  void *dptrs[NUM_GRAPHS];
  for (int i = 0; i < NUM_GRAPHS; i++) {
    RT(buildAllocDelayFreeGraph(&graphExecs[i], &dptrs[i], bytes, device,
                                delay_clocks));
    RT(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  }

  size_t footprint = 0;
  RT(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent,
                                    &footprint));
  CHECK(footprint == 0, "instantiating graphs reserves no physical memory");

  size_t footprints[NUM_GRAPHS];
  double start = now_ms();
  for (int i = 1; i < NUM_GRAPHS; i++) {
    RT(cudaGraphLaunch(graphExecs[i], streams[i]));
    RT(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent,
                                      &footprints[i]));
  }
  double launch_ms = now_ms() - start;

  // The precondition the upstream sample never checks: if the launch loop
  // outlasts the delay, earlier graphs have already retired and the footprint
  // growth below proves nothing.
  fprintf(stderr, "launch loop took %.1f ms, per-graph delay %.3f ms\n",
          launch_ms, delay_ms);
  CHECK(launch_ms < delay_ms,
        "launch loop finishes inside the overlap window (raise "
        "LUPINE_GRAPH_DELAY_MS if this fails)");

  for (int i = 1; i < NUM_GRAPHS; i++) {
    size_t expected = (size_t)i * bytes;
    if (footprints[i] != expected)
      fprintf(stderr, "  launch %02d: footprint %zu, expected %zu\n", i,
              footprints[i], expected);
    CHECK(footprints[i] == expected,
          "each concurrent stream adds one allocation to the footprint");
  }

  for (int i = 0; i < NUM_GRAPHS; i++)
    RT(cudaStreamSynchronize(streams[i]));

  RT(cudaDeviceGraphMemTrim(device));
  RT(cudaDeviceGetGraphMemAttribute(device, cudaGraphMemAttrUsedMemCurrent,
                                    &footprint));
  CHECK(footprint == 0, "trimming after the graphs retire releases the pool");

  for (int i = 0; i < NUM_GRAPHS; i++) {
    RT(cudaGraphExecDestroy(graphExecs[i]));
    RT(cudaStreamDestroy(streams[i]));
  }

  if (g_failures) {
    fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
    return 1;
  }
  fprintf(stderr, "PASSED\n");
  return 0;
}
