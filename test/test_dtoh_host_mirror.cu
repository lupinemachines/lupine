#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <thread>

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS)
    return true;
  std::fprintf(stderr, "%s: CUDA error %d\n", operation, int(result));
  return false;
}
#define CHECK(call)                                                            \
  do {                                                                         \
    if (!check((call), #call))                                                 \
      return 1;                                                                \
  } while (false)

int main(int argc, char **argv) {
  // Older peers retain the existing staging path. The optional mode isolates
  // fallback coverage from its pre-existing cross-thread completion issue.
  const bool single_thread =
      argc == 2 && std::strcmp(argv[1], "--single-thread") == 0;
  CHECK(cuInit(0));
  CUdevice device;
  CUcontext context;
  CHECK(cuDeviceGet(&device, 0));
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));
  CUstream stream;
  CUevent event;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  constexpr size_t size = 8192;
  unsigned char *host;
  CUdeviceptr source, destination, mapped;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&host), size,
                       CU_MEMHOSTALLOC_DEVICEMAP));
  CHECK(cuMemHostGetDevicePointer(&mapped, host, 0));
  CHECK(cuMemAlloc(&source, size));
  CHECK(cuMemAlloc(&destination, size));
  std::memset(host, 0x5a, size);

  for (int mode = 0; mode < (single_thread ? 3 : 4); ++mode) {
    unsigned char expected = static_cast<unsigned char>(0x30 + mode);
    CHECK(cuMemsetD8Async(source, expected, size, stream));
    CHECK(cuMemcpyDtoHAsync(host + 17, source + 17, size - 34, stream));
    if (mode == 0) {
      CHECK(cuStreamSynchronize(stream));
    } else if (mode == 1) {
      CHECK(cuEventRecord(event, stream));
      CHECK(cuEventSynchronize(event));
    } else if (mode == 2) {
      CHECK(cuCtxSynchronize());
    } else {
      CHECK(cuEventRecord(event, stream));
      CUresult result = CUDA_ERROR_UNKNOWN;
      std::thread waiter([&] {
        result = cuCtxSetCurrent(context);
        if (result == CUDA_SUCCESS)
          result = cuEventSynchronize(event);
      });
      waiter.join();
      CHECK(result);
    }
    if (host[17] != expected) {
      std::fprintf(stderr,
                   "mode %d immediate host result: expected %u, got %u\n", mode,
                   unsigned(expected), unsigned(host[17]));
      return 1;
    }
    // No intervening CPU write: HtoD must see the completed DtoH in the
    // server's mirror, even when another lane performed synchronization.
    CHECK(cuMemcpyHtoDAsync(destination, host + 17, size - 34, stream));
    CHECK(cuStreamSynchronize(stream));
    unsigned char result[size] = {};
    CHECK(cuMemcpyDtoH(result, destination, size - 34));
    for (size_t i = 0; i < size - 34; ++i) {
      if (result[i] != expected) {
        std::fprintf(stderr, "mode %d byte %zu: expected %u, got %u\n", mode, i,
                     unsigned(expected), unsigned(result[i]));
        return 1;
      }
    }
    if (host[0] != 0x5a || host[size - 1] != 0x5a || host[17] != expected) {
      std::fprintf(stderr, "mode %d host guards/data: %u %u %u\n", mode,
                   unsigned(host[0]), unsigned(host[size - 1]),
                   unsigned(host[17]));
      return 1;
    }
    // Later CPU writes must still fault/dirty and reach the server.
    host[19] = 0x7e;
    CHECK(cuMemcpyHtoDAsync(destination, host + 17, 16, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH(result, destination, 16));
    if (result[2] != 0x7e) {
      std::fprintf(stderr, "mode %d lost CPU edit: %u\n", mode,
                   unsigned(result[2]));
      return 1;
    }
  }

  // Captured copies retain their staging-backed path and can replay.
  CUgraph graph;
  CUgraphExec executable;
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyDtoHAsync(host + 32, source, 32, stream));
  CHECK(cuStreamEndCapture(stream, &graph));
#if CUDA_VERSION >= 12000
  CHECK(cuGraphInstantiate(&executable, graph, 0));
#else
  CHECK(cuGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
#endif
  for (int iteration = 0; iteration < 2; ++iteration) {
    CHECK(cuMemsetD8Async(source, 0x60 + iteration, 32, stream));
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    if (host[32] != 0x60 + iteration) {
      std::fprintf(stderr, "graph replay %d: got %u\n", iteration,
                   unsigned(host[32]));
      return 1;
    }
  }
  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));
  CHECK(cuMemFree(destination));
  CHECK(cuMemFree(source));
  CHECK(cuMemFreeHost(host));
  CHECK(cuEventDestroy(event));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuDevicePrimaryCtxRelease(device));
  std::puts(
      single_thread
          ? "DtoH legacy fallback: partial ranges, CPU edits and capture passed"
          : "DtoH mirror: partial ranges, CPU edits, cross-lane and capture "
            "passed");
  return 0;
}
