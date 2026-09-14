#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// 12 MB crosses the server's 8 MB pull fragment boundary.
static constexpr size_t kBytes = 12u << 20;

static void fill_pattern(unsigned char *buffer, size_t bytes,
                         unsigned char seed) {
  for (size_t i = 0; i < bytes; ++i) {
    buffer[i] = static_cast<unsigned char>(seed + (i % 251));
  }
}

static int verify(CUdeviceptr destination, unsigned char seed,
                  const char *label, int replay) {
  std::vector<unsigned char> readback(kBytes);
  CHECK(cuMemcpyDtoH(readback.data(), destination, kBytes));
  for (size_t i = 0; i < kBytes; ++i) {
    unsigned char expected = static_cast<unsigned char>(seed + (i % 251));
    if (readback[i] != expected) {
      std::fprintf(stderr,
                   "%s: replay %d byte %zu is 0x%02x, expected 0x%02x\n", label,
                   replay, i, readback[i], expected);
      return 1;
    }
  }
  return 0;
}

// Captures one HtoD from `source` on `capture_stream` (optionally forked onto
// `copy_stream` so the copy joins the capture through an event), then replays
// the graph on `launch_stream` with a fresh source pattern each time.
static int check_replay(unsigned char *source, CUdeviceptr destination,
                        CUstream capture_stream, CUstream copy_stream,
                        CUstream launch_stream, const char *label) {
  CUevent fork = nullptr;
  CUevent join = nullptr;
  CHECK(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  CHECK(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));

  fill_pattern(source, kBytes, 0x10);
  CHECK(cuStreamBeginCapture(capture_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  if (copy_stream != capture_stream) {
    CHECK(cuEventRecord(fork, capture_stream));
    CHECK(cuStreamWaitEvent(copy_stream, fork, 0));
  }
  CHECK(cuMemcpyHtoDAsync(destination, source, kBytes, copy_stream));
  if (copy_stream != capture_stream) {
    CHECK(cuEventRecord(join, copy_stream));
    CHECK(cuStreamWaitEvent(capture_stream, join, 0));
  }
  CUgraph graph = nullptr;
  CHECK(cuStreamEndCapture(capture_stream, &graph));
  CUgraphExec executable = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));

  // The source changes after capture; each replay must read the bytes the
  // host holds at launch, not the ones seen during capture.
  int replay = 0;
  for (unsigned char seed : {0x21, 0x32, 0x43}) {
    fill_pattern(source, kBytes, seed);
    CHECK(cuGraphLaunch(executable, launch_stream));
    CHECK(cuStreamSynchronize(launch_stream));
    if (verify(destination, seed, label, replay) != 0) {
      return 1;
    }
    ++replay;
  }

  // A synchronous HtoD after the replays must still push/pull correctly.
  fill_pattern(source, kBytes, 0x77);
  CHECK(cuMemcpyHtoD(destination, source, kBytes));
  if (verify(destination, 0x77, label, -1) != 0) {
    return 1;
  }

  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));
  CHECK(cuEventDestroy(fork));
  CHECK(cuEventDestroy(join));
  return 0;
}

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));

  CUstream s1 = nullptr, s2 = nullptr, s3 = nullptr;
  CHECK(cuStreamCreate(&s1, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&s2, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&s3, CU_STREAM_NON_BLOCKING));
  CUdeviceptr destination = 0;
  CHECK(cuMemAlloc(&destination, kBytes));

  std::vector<unsigned char> pageable(kBytes);
  if (check_replay(pageable.data(), destination, s1, s1, s1,
                   "pageable same stream") != 0 ||
      check_replay(pageable.data(), destination, s1, s2, s1,
                   "pageable forked copy stream") != 0 ||
      check_replay(pageable.data(), destination, s1, s1, s3,
                   "pageable launched on another stream") != 0) {
    return 1;
  }

  unsigned char *registered = static_cast<unsigned char *>(
      std::aligned_alloc(4096, ((kBytes + 4095) / 4096) * 4096));
  CHECK(cuMemHostRegister(registered, kBytes, 0));
  if (check_replay(registered, destination, s1, s1, s1,
                   "registered same stream") != 0 ||
      check_replay(registered, destination, s1, s2, s3,
                   "registered forked copy, other launch stream") != 0) {
    return 1;
  }
  CHECK(cuMemHostUnregister(registered));
  std::free(registered);

  unsigned char *pinned = nullptr;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&pinned), kBytes, 0));
  if (check_replay(pinned, destination, s1, s1, s1, "host alloc same stream") !=
          0 ||
      check_replay(pinned, destination, s1, s2, s3,
                   "host alloc forked copy, other launch stream") != 0) {
    return 1;
  }
  CHECK(cuMemFreeHost(pinned));

  CHECK(cuMemFree(destination));
  std::printf("PASS: captured HtoD replays read the source at launch\n");
  return 0;
}
