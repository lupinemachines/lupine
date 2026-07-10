#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

static const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "UNKNOWN";
}

static void check(CUresult result, const char *expression, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expression, line,
                 result_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

static CUcontext create_context(CUdevice device) {
  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif
  return context;
}

static void copy_and_verify(unsigned char pattern) {
  constexpr size_t bytes = 12 * 1024 * 1024;
  std::vector<unsigned char> source(bytes, pattern);
  std::vector<unsigned char> destination(bytes);
  CUdeviceptr remote = 0;
  CUstream stream = nullptr;
  CHECK(cuMemAlloc(&remote, bytes));
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuMemcpyHtoDAsync(remote, source.data(), bytes, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpyDtoH(destination.data(), remote, bytes));
  if (destination != source) {
    std::fprintf(stderr, "context lifecycle copy mismatch for pattern %u\n",
                 static_cast<unsigned int>(pattern));
    std::exit(EXIT_FAILURE);
  }
  CHECK(cuStreamDestroy(stream));
  CHECK(cuMemFree(remote));
}

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  CUcontext previous = nullptr;
  for (int iteration = 0; iteration != 3; ++iteration) {
    CUcontext context = create_context(device);
    copy_and_verify(static_cast<unsigned char>(0x20 + iteration));
    CHECK(cuCtxDestroy(context));
    if (previous == context) {
      std::printf("custom context handle was reused on iteration %d\n",
                  iteration);
    }
    previous = context;
  }

  CUcontext detachable = create_context(device);
  CUcontext attached = nullptr;
  CHECK(cuCtxAttach(&attached, 0));
  copy_and_verify(0x51);
  CHECK(cuCtxDetach(attached));
  copy_and_verify(0x52);
  CHECK(cuCtxDestroy(detachable));

  constexpr size_t gated_bytes = 8 * 1024 * 1024;
  CUcontext retained_primary = nullptr;
  CUcontext retained_again = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&retained_primary, device));
  CHECK(cuDevicePrimaryCtxRetain(&retained_again, device));
  CHECK(cuCtxSetCurrent(retained_primary));
  CUdeviceptr gate = 0;
  CUdeviceptr gated_remote = 0;
  CUstream blocked_stream = nullptr;
  CUstream release_stream = nullptr;
  CHECK(cuMemAlloc(&gate, sizeof(unsigned int)));
  CHECK(cuMemAlloc(&gated_remote, gated_bytes));
  CHECK(cuStreamCreate(&blocked_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&release_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuMemsetD32(gate, 0, 1));
  CHECK(cuStreamWaitValue32(blocked_stream, gate, 1, CU_STREAM_WAIT_VALUE_EQ));
  unsigned char *gated_source = nullptr;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&gated_source), gated_bytes,
                       CU_MEMHOSTALLOC_PORTABLE));
  std::fill_n(gated_source, gated_bytes, 0x59);
  CHECK(cuMemcpyHtoDAsync(gated_remote, gated_source, gated_bytes,
                          blocked_stream));
#ifndef _WIN32
  alarm(5);
#endif
  auto release_started = std::chrono::steady_clock::now();
  CHECK(cuDevicePrimaryCtxRelease(device));
  auto release_elapsed = std::chrono::steady_clock::now() - release_started;
#ifndef _WIN32
  alarm(0);
#endif
  if (release_elapsed > std::chrono::seconds(1)) {
    std::fprintf(stderr, "non-final primary context release blocked on work\n");
    return EXIT_FAILURE;
  }
  CHECK(cuStreamWriteValue32(release_stream, gate, 1,
                             CU_STREAM_WRITE_VALUE_DEFAULT));
  CHECK(cuStreamSynchronize(blocked_stream));
  std::vector<unsigned char> gated_destination(gated_bytes);
  CHECK(cuMemcpyDtoH(gated_destination.data(), gated_remote, gated_bytes));
  if (!std::equal(gated_destination.begin(), gated_destination.end(),
                  gated_source)) {
    std::fprintf(stderr, "gated non-final release copy mismatch\n");
    return EXIT_FAILURE;
  }
  CHECK(cuMemFreeHost(gated_source));
  CHECK(cuStreamDestroy(release_stream));
  CHECK(cuStreamDestroy(blocked_stream));
  CHECK(cuMemFree(gated_remote));
  CHECK(cuMemFree(gate));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(device));
  CHECK(cuDevicePrimaryCtxReset(device));

  CUcontext primary = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&primary, device));
  CHECK(cuCtxSetCurrent(primary));
  copy_and_verify(0x61);
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(device));

  CUcontext recreated = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&recreated, device));
  CHECK(cuCtxSetCurrent(recreated));
  copy_and_verify(0x72);
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxReset(device));
  CHECK(cuDevicePrimaryCtxRelease(device));

  // Primary context handles are commonly recycled by reset. The server must
  // invalidate retained allocations before reset even when the numeric handle
  // remains unchanged, then safely build a fresh ring for the recreated state.
  CUcontext after_reset = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&after_reset, device));
  CHECK(cuCtxSetCurrent(after_reset));
  copy_and_verify(0x83);
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(device));

  std::printf("PASS: async HtoD staging survives context destroy, release, "
              "and reset (primary handles %p -> %p)\n",
              static_cast<void *>(recreated), static_cast<void *>(after_reset));
  return EXIT_SUCCESS;
}
