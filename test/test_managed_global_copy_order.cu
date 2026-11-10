// Copies into a __managed__ module global with cudaMemcpyDefault are
// device-side writes in stream order, both on an eager stream and inside a
// captured CUDA graph. Treating the client host view of the global as an
// ordinary host destination turns the copy into a DtoH transfer whose bytes
// overwrite subsequent kernel writes on the same stream or graph.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __managed__ unsigned int g_managed_value = 0;

namespace {

__global__ void add_global_100() { g_managed_value += 100; }

__global__ void add_global_500() { g_managed_value += 500; }

__global__ void add_global_1000() { g_managed_value += 1000; }

void check(cudaError_t result, const char *what) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  unsigned int *device = nullptr;
  unsigned int *pinned = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaMallocHost(&pinned, sizeof(*pinned)), "cudaMallocHost");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  unsigned int seven = 7;
  check(cudaMemcpy(device, &seven, sizeof(seven), cudaMemcpyHostToDevice),
        "cudaMemcpy init");
  *pinned = 7;
  g_managed_value = 0;
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int failures = 0;
  struct {
    const char *name;
    const unsigned int *source;
  } const sources[] = {{"device", device}, {"pinned", pinned}};

  // 1. Eager stream: cuMemcpyAsync into __managed__ global, then update on GPU.
  for (int round = 0; round < 4; ++round) {
    for (const auto &src : sources) {
      g_managed_value = 0;
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(reset)");
      check(cudaMemcpyAsync(&g_managed_value, src.source,
                            sizeof(g_managed_value), cudaMemcpyDefault, stream),
            "eager cudaMemcpyAsync into __managed__ global");
      add_global_100<<<1, 1, 0, stream>>>();
      check(cudaGetLastError(), "add_global_100");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      add_global_1000<<<1, 1, 0, stream>>>();
      check(cudaGetLastError(), "add_global_1000");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize 2");
      unsigned int got =
          *const_cast<volatile unsigned int *>(&g_managed_value);
      if (got != 1107) {
        std::fprintf(stderr,
                     "FAIL: eager round %d from %s: got %u, want 1107\n", round,
                     src.name, got);
        ++failures;
      }
    }
  }

  // 2. Stream capture + replay: captured cudaMemcpyDefault into __managed__
  // global followed by kernel in the same graph.
  for (const auto &src : sources) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "cudaStreamBeginCapture");
    check(cudaMemcpyAsync(&g_managed_value, src.source, sizeof(g_managed_value),
                          cudaMemcpyDefault, stream),
          "captured cudaMemcpyAsync into __managed__ global");
    add_global_500<<<1, 1, 0, stream>>>();
    check(cudaGetLastError(), "captured add_global_500");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
          "cudaGraphInstantiate");
    for (int replay = 0; replay < 3; ++replay) {
      g_managed_value = 0;
      check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(graph)");
      unsigned int got = *const_cast<volatile unsigned int *>(&g_managed_value);
      if (got != 507) {
        std::fprintf(stderr,
                     "FAIL: graph replay %d from %s: got %u, want 507\n",
                     replay, src.name, got);
        ++failures;
      }
    }
    check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  }
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFreeHost(pinned), "cudaFreeHost");
  check(cudaFree(device), "cudaFree");

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d __managed__ global copy cases failed\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: copies into __managed__ module globals run in stream order\n");
  return EXIT_SUCCESS;
}
