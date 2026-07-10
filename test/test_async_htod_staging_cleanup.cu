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

static void verify_pattern(CUdeviceptr remote, size_t bytes,
                           unsigned char expected) {
  std::vector<unsigned char> destination(bytes);
  CHECK(cuMemcpyDtoH(destination.data(), remote, bytes));
  auto mismatch = std::find_if(
      destination.begin(), destination.end(),
      [expected](unsigned char value) { return value != expected; });
  if (mismatch != destination.end()) {
    std::fprintf(stderr, "payload mismatch at byte %zu: got %u, expected %u\n",
                 static_cast<size_t>(mismatch - destination.begin()),
                 static_cast<unsigned int>(*mismatch),
                 static_cast<unsigned int>(expected));
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  constexpr size_t gated_bytes = 40 * 1024 * 1024;
  constexpr size_t reuse_bytes = 8 * 1024 * 1024;
  constexpr unsigned char gated_pattern = 0x35;
  constexpr unsigned char cross_stream_pattern = 0x6a;

  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));

  CUstream blocked_stream = nullptr;
  CUstream release_stream = nullptr;
  CUstream other_stream = nullptr;
  CHECK(cuStreamCreate(&blocked_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&release_stream, CU_STREAM_NON_BLOCKING));
  CHECK(cuStreamCreate(&other_stream, CU_STREAM_NON_BLOCKING));

  CUdeviceptr gate = 0;
  CUdeviceptr gated_remote = 0;
  CUdeviceptr other_remote = 0;
  CHECK(cuMemAlloc(&gate, sizeof(unsigned int)));
  CHECK(cuMemAlloc(&gated_remote, gated_bytes));
  CHECK(cuMemAlloc(&other_remote, reuse_bytes));
  CHECK(cuMemsetD32(gate, 0, 1));

  CHECK(cuStreamWaitValue32(blocked_stream, gate, 1, CU_STREAM_WAIT_VALUE_EQ));

  unsigned char *gated_source = nullptr;
  CHECK(cuMemHostAlloc(reinterpret_cast<void **>(&gated_source), gated_bytes,
                       CU_MEMHOSTALLOC_PORTABLE));
  std::fill_n(gated_source, gated_bytes, gated_pattern);
#ifndef _WIN32
  // A server-side synchronization bug cannot be rescued by another host
  // thread: the gate is released only after this API-return assertion.
  alarm(5);
#endif
  auto started = std::chrono::steady_clock::now();
  CHECK(cuMemcpyHtoDAsync(gated_remote, gated_source, gated_bytes,
                          blocked_stream));
  auto elapsed = std::chrono::steady_clock::now() - started;
#ifndef _WIN32
  alarm(0);
#endif
  if (elapsed > std::chrono::seconds(1)) {
    std::fprintf(stderr,
                 "gated cuMemcpyHtoDAsync took too long and likely waited for "
                 "its stream\n");
    return EXIT_FAILURE;
  }

  // Lupine has consumed the client payload when the RPC returns. Mutating the
  // source verifies that remote staging retains that snapshot until the gate
  // opens and the GPU copy actually executes.
  std::fill_n(gated_source, gated_bytes, 0xe3);

  // The 40 MiB call occupies all four 8 MiB ring slots behind the gate and
  // stages its suffix in a spill. A call on another stream must also return
  // through bounded-poll spill staging without corrupting either transfer.
  std::vector<unsigned char> other_source(reuse_bytes, cross_stream_pattern);
  CHECK(cuMemcpyHtoDAsync(other_remote, other_source.data(), reuse_bytes,
                          other_stream));
  std::fill(other_source.begin(), other_source.end(), 0x9c);

  CHECK(cuStreamWriteValue32(release_stream, gate, 1,
                             CU_STREAM_WRITE_VALUE_DEFAULT));
  CHECK(cuStreamSynchronize(release_stream));
  CHECK(cuStreamSynchronize(blocked_stream));
  CHECK(cuStreamSynchronize(other_stream));
  verify_pattern(gated_remote, gated_bytes, gated_pattern);
  verify_pattern(other_remote, reuse_bytes, cross_stream_pattern);
  CHECK(cuMemFreeHost(gated_source));

  // Reclaim completed slots across explicit streams and on the legacy default
  // stream, then exercise sequential queueing on one nonblocking stream.
  std::vector<unsigned char> source_a(reuse_bytes, 0x11);
  std::vector<unsigned char> source_b(reuse_bytes, 0x22);
  CHECK(cuMemcpyHtoDAsync(other_remote, source_a.data(), reuse_bytes,
                          blocked_stream));
  CHECK(cuStreamSynchronize(blocked_stream));
  CHECK(cuMemcpyHtoDAsync(other_remote, source_b.data(), reuse_bytes,
                          other_stream));
  CHECK(cuStreamSynchronize(other_stream));
  verify_pattern(other_remote, reuse_bytes, 0x22);

  std::fill(source_a.begin(), source_a.end(), 0x44);
  CHECK(cuMemcpyHtoDAsync(other_remote, source_a.data(), reuse_bytes, nullptr));
  CHECK(cuCtxSynchronize());
  verify_pattern(other_remote, reuse_bytes, 0x44);

  std::fill(source_a.begin(), source_a.end(), 0x57);
  std::fill(source_b.begin(), source_b.end(), 0x58);
  CHECK(cuMemcpyHtoDAsync(other_remote, source_a.data(), reuse_bytes,
                          other_stream));
  CHECK(cuMemcpyHtoDAsync(other_remote, source_b.data(), reuse_bytes,
                          other_stream));
  CHECK(cuStreamSynchronize(other_stream));
  verify_pattern(other_remote, reuse_bytes, 0x58);

  CHECK(cuMemFree(other_remote));
  CHECK(cuMemFree(gated_remote));
  CHECK(cuMemFree(gate));
  CHECK(cuStreamDestroy(other_stream));
  CHECK(cuStreamDestroy(release_stream));
  CHECK(cuStreamDestroy(blocked_stream));
  CHECK(cuCtxSetCurrent(nullptr));
  CHECK(cuDevicePrimaryCtxRelease(device));
  CHECK(cuDevicePrimaryCtxReset(device));

  std::printf("PASS: async HtoD event-owned staging preserves stream ordering "
              "and slot reuse\n");
  return EXIT_SUCCESS;
}
