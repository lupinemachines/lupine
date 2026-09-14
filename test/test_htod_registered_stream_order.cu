// A page-locked (cuMemHostRegister'd) HtoD source is read when the stream
// reaches the copy, so an async DtoH queued ahead of it into the same buffer
// must be visible to the copy. Through Lupine that DtoH only reaches the client
// at a sync point, so the server has to serve the copy's pull from the bytes it
// still holds (issue #766).
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kBytes = 12 * 1024 * 1024; // spans several transfer fragments

const char *error_name(CUresult result) {
  const char *name = nullptr;
  (void)cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult check_result_ = (call);                                           \
    if (check_result_ != CUDA_SUCCESS) {                                       \
      std::fprintf(stderr, "%s failed: %s (%d)\n", #call,                      \
                   error_name(check_result_), (int)check_result_);             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

void fill(std::vector<unsigned char> &buffer, unsigned char seed) {
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = static_cast<unsigned char>(seed + i * 31 + (i >> 12));
  }
}

int verify(const char *label, CUdeviceptr device,
           const std::vector<unsigned char> &expected) {
  std::vector<unsigned char> back(expected.size());
  CHECK(cuMemcpyDtoH(back.data(), device, back.size()));
  for (size_t i = 0; i < back.size(); ++i) {
    if (back[i] != expected[i]) {
      std::fprintf(stderr, "FAIL: %s: byte %zu is 0x%02x, expected 0x%02x\n",
                   label, i, back[i], expected[i]);
      return 1;
    }
  }
  std::printf("%s: ok\n", label);
  return 0;
}

} // namespace

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CHECK(cuCtxSetCurrent(context));
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  CUdeviceptr source = 0;
  CUdeviceptr target = 0;
  CHECK(cuMemAlloc(&source, kBytes));
  CHECK(cuMemAlloc(&target, kBytes));

  std::vector<unsigned char> pattern(kBytes);
  std::vector<unsigned char> expected(kBytes);

  void *registered = std::aligned_alloc(4096, kBytes);
  if (registered == nullptr) {
    return 1;
  }
  std::memset(registered, 0xee, kBytes);
  CHECK(cuMemHostRegister(registered, kBytes, 0));

  // Async DtoH into the registered buffer, then async HtoD out of it on the
  // same stream.
  fill(pattern, 1);
  CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
  CHECK(cuMemsetD8(target, 0, kBytes));
  CHECK(cuMemcpyDtoHAsync(registered, source, kBytes, stream));
  CHECK(cuMemcpyHtoDAsync(target, registered, kBytes, stream));
  CHECK(cuStreamSynchronize(stream));
  if (verify("registered source, async chain", target, pattern) != 0) {
    return 1;
  }

  // Same, with the synchronous copy on the legacy stream, which orders behind
  // the blocking stream that fills the buffer.
  fill(pattern, 7);
  CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
  CHECK(cuMemsetD8(target, 0, kBytes));
  std::memset(registered, 0xee, kBytes);
  CHECK(cuMemcpyDtoHAsync(registered, source, kBytes, stream));
  CHECK(cuMemcpyHtoD(target, registered, kBytes));
  CHECK(cuCtxSynchronize());
  if (verify("registered source, legacy-stream sync copy", target, pattern) !=
      0) {
    return 1;
  }

  // Once the stream has been synchronized the host owns the buffer again: a
  // host write after the sync must win over the earlier DtoH.
  fill(pattern, 11);
  CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
  CHECK(cuMemsetD8(target, 0, kBytes));
  CHECK(cuMemcpyDtoHAsync(registered, source, kBytes, stream));
  CHECK(cuStreamSynchronize(stream));
  fill(expected, 13);
  std::memcpy(registered, expected.data(), kBytes);
  CHECK(cuMemcpyHtoDAsync(target, registered, kBytes, stream));
  CHECK(cuStreamSynchronize(stream));
  if (verify("registered source, host write after sync", target, expected) !=
      0) {
    return 1;
  }
  CHECK(cuMemHostUnregister(registered));

  // Buffer registered after the host wrote its first half; the queued DtoH
  // fills only the second half, so the copy mixes host bytes and device bytes.
  fill(expected, 17);
  std::memcpy(registered, expected.data(), kBytes / 2);
  std::memset(static_cast<unsigned char *>(registered) + kBytes / 2, 0xee,
              kBytes - kBytes / 2);
  CHECK(cuMemHostRegister(registered, kBytes, 0));
  fill(pattern, 19);
  CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
  std::memcpy(expected.data() + kBytes / 2, pattern.data() + kBytes / 2,
              kBytes - kBytes / 2);
  CHECK(cuMemsetD8(target, 0, kBytes));
  CHECK(cuMemcpyDtoHAsync(static_cast<unsigned char *>(registered) + kBytes / 2,
                          source + kBytes / 2, kBytes - kBytes / 2, stream));
  CHECK(cuMemcpyHtoDAsync(target, registered, kBytes, stream));
  CHECK(cuStreamSynchronize(stream));
  if (verify("registered after partial host write", target, expected) != 0) {
    return 1;
  }
  CHECK(cuMemHostUnregister(registered));
  std::free(registered);

  // Two registered buffers alternating as DtoH destination and HtoD source on
  // one stream, across several rounds with a sync between them.
  void *first = std::aligned_alloc(4096, kBytes);
  void *second = std::aligned_alloc(4096, kBytes);
  if (first == nullptr || second == nullptr) {
    return 1;
  }
  CHECK(cuMemHostRegister(first, kBytes, 0));
  CHECK(cuMemHostRegister(second, kBytes, 0));
  CUdeviceptr other = 0;
  CHECK(cuMemAlloc(&other, kBytes));
  for (int round = 0; round < 3; ++round) {
    std::memset(first, 0xee, kBytes);
    std::memset(second, 0xdd, kBytes);
    fill(pattern, static_cast<unsigned char>(23 + round));
    fill(expected, static_cast<unsigned char>(29 + round));
    CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
    CHECK(cuMemcpyHtoD(other, expected.data(), kBytes));
    CHECK(cuMemsetD8(target, 0, kBytes));
    CHECK(cuMemcpyDtoHAsync(first, source, kBytes, stream));
    CHECK(cuMemcpyDtoHAsync(second, other, kBytes, stream));
    CHECK(cuMemcpyHtoDAsync(target, second, kBytes, stream));
    CHECK(cuMemcpyHtoDAsync(other, first, kBytes, stream));
    CHECK(cuMemcpyDtoHAsync(second, target, kBytes, stream));
    CHECK(cuMemcpyHtoDAsync(source, second, kBytes, stream));
    CHECK(cuStreamSynchronize(stream));
    if (verify("alternating buffers, target", target, expected) != 0 ||
        verify("alternating buffers, other", other, pattern) != 0 ||
        verify("alternating buffers, source", source, expected) != 0) {
      return 1;
    }
  }

  // Captured {DtoH into the registered buffer; HtoD out of it}: each replay
  // must read the bytes its own DtoH node produced, whatever the device or the
  // host wrote to them in between.
  CUgraph graph = nullptr;
  CUgraphExec executable = nullptr;
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyDtoHAsync(first, source, kBytes, stream));
  CHECK(cuMemcpyHtoDAsync(target, first, kBytes, stream));
  CHECK(cuStreamEndCapture(stream, &graph));
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    fill(pattern, static_cast<unsigned char>(37 + replay));
    CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
    CHECK(cuMemsetD8(target, 0, kBytes));
    std::memset(first, 0xee, kBytes);
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    if (verify("captured chain replay", target, pattern) != 0) {
      return 1;
    }
  }
  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));

  // The captured DtoH fills only the first half; the second half comes from
  // whatever the host wrote before the replay.
  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpyDtoHAsync(first, source, kBytes / 2, stream));
  CHECK(cuMemcpyHtoDAsync(target, first, kBytes, stream));
  CHECK(cuStreamEndCapture(stream, &graph));
  CHECK(cuGraphInstantiateWithFlags(&executable, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    fill(pattern, static_cast<unsigned char>(41 + replay));
    fill(expected, static_cast<unsigned char>(43 + replay));
    CHECK(cuMemcpyHtoD(source, pattern.data(), kBytes));
    CHECK(cuMemsetD8(target, 0, kBytes));
    std::memcpy(first, expected.data(), kBytes);
    std::memcpy(expected.data(), pattern.data(), kBytes / 2);
    CHECK(cuGraphLaunch(executable, stream));
    CHECK(cuStreamSynchronize(stream));
    if (verify("captured half chain replay", target, expected) != 0) {
      return 1;
    }
  }
  CHECK(cuGraphExecDestroy(executable));
  CHECK(cuGraphDestroy(graph));

  CHECK(cuMemHostUnregister(first));
  CHECK(cuMemHostUnregister(second));
  std::free(first);
  std::free(second);

  CHECK(cuMemFree(other));
  CHECK(cuMemFree(source));
  CHECK(cuMemFree(target));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuDevicePrimaryCtxRelease(device));
  std::printf("PASS: registered HtoD sources honour stream order\n");
  return 0;
}
