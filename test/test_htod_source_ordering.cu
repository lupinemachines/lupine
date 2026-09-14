#include <cuda.h>

// Host-to-device sources come in two kinds and the transport must treat them
// differently. A page-locked source is read when the stream reaches the copy;
// a pageable source is staged at issue, so overwriting it on the host right
// after an async copy returns must not leak into the copy.
// This file is auto-discovered by test/run_custom_tests.sh.

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
      std::fprintf(stderr, "%s: byte %zu is 0x%02x, expected 0x%02x\n", label,
                   i, back[i], expected[i]);
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
  std::vector<unsigned char> scratch(kBytes);

  // Page-locked source written by the host: the copy must read it when the
  // stream reaches it, and the bytes must be the host's.
  void *registered = std::malloc(kBytes);
  if (registered == nullptr) {
    return 1;
  }
  CHECK(cuMemHostRegister(registered, kBytes, 0));
  fill(pattern, 1);
  std::memcpy(registered, pattern.data(), kBytes);
  CHECK(cuMemsetD8(target, 0, kBytes));
  CHECK(cuMemcpyHtoDAsync(target, registered, kBytes, stream));
  CHECK(cuStreamSynchronize(stream));
  if (verify("registered source, host-written", target, pattern) != 0) {
    return 1;
  }
  fill(pattern, 7);
  std::memcpy(registered, pattern.data(), kBytes);
  CHECK(cuMemcpyHtoD(target, registered, kBytes));
  if (verify("registered source, host-written, sync copy", target, pattern) !=
      0) {
    return 1;
  }
  CHECK(cuMemHostUnregister(registered));
  std::free(registered);

  // Pageable source overwritten on the host right after the async copy
  // returns: the copy must carry the bytes as they were at issue.
  fill(pattern, 13);
  scratch = pattern;
  CHECK(cuMemsetD8(target, 0, kBytes));
  CHECK(cuMemcpyHtoDAsync(target, scratch.data(), kBytes, stream));
  std::memset(scratch.data(), 0x11, kBytes);
  CHECK(cuStreamSynchronize(stream));
  if (verify("pageable source, overwritten after issue", target, pattern) !=
      0) {
    return 1;
  }

  // Pageable source, synchronous copy, then read back through the device.
  fill(pattern, 29);
  CHECK(cuMemcpyHtoD(target, pattern.data(), kBytes));
  if (verify("pageable source, sync copy", target, pattern) != 0) {
    return 1;
  }

  CHECK(cuMemFree(source));
  CHECK(cuMemFree(target));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuDevicePrimaryCtxRelease(device));
  std::printf("PASS: host-to-device sources honour stream order and issue-time staging\n");
  return 0;
}
