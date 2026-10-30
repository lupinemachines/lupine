// LUPINE_TEST_CONNECTIONS 2
// Buffers live on both connections at once must keep distinct addresses, even
// when both servers hand out the same ones, so copies and frees reach the
// right server.
#include <cuda.h>

#include <cstdint>
#include <cstdio>

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

int main() {
  int count = 0;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
    return 1;
  }
  constexpr size_t kBytes = 1 << 20;
  constexpr size_t kWidth = 1000;
  constexpr size_t kHeight = 3;
  CUcontext contexts[2] = {};
  CUdeviceptr linear[2] = {};
  CUdeviceptr pitched[2] = {};
  size_t pitch[2] = {};
  for (int i = 0; i < 2; ++i) {
    // A single GPU also supplies the native control, using two contexts.
    CUdevice device = 0;
    if (!check(cuDeviceGet(&device, i < count ? i : 0), "cuDeviceGet") ||
#if CUDA_VERSION >= 13000
        !check(cuCtxCreate(&contexts[i], nullptr, 0, device), "cuCtxCreate") ||
#else
        !check(cuCtxCreate(&contexts[i], 0, device), "cuCtxCreate") ||
#endif
        !check(cuMemAlloc(&linear[i], kBytes), "cuMemAlloc") ||
        !check(cuMemAllocPitch(&pitched[i], &pitch[i], kWidth, kHeight, 4),
               "cuMemAllocPitch") ||
        !check(cuMemsetD32(linear[i], 0x1000 + i, kBytes / 4),
               "cuMemsetD32") ||
        !check(cuMemsetD2D32(pitched[i], pitch[i], 0x2000 + i, kWidth / 4,
                             kHeight),
               "cuMemsetD2D32") ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize")) {
      return 1;
    }
  }
  if (linear[0] < linear[1] + kBytes && linear[1] < linear[0] + kBytes) {
    std::fprintf(stderr, "buffers on both devices overlap at %llx and %llx\n",
                 static_cast<unsigned long long>(linear[0]),
                 static_cast<unsigned long long>(linear[1]));
    return 1;
  }
  for (int i = 0; i < 2; ++i) {
    uint32_t value = 0;
    uint32_t pitched_value = 0;
    if (!check(cuCtxSetCurrent(contexts[i]), "cuCtxSetCurrent") ||
        !check(cuMemcpyDtoH(&value, linear[i] + kBytes - 4, 4),
               "cuMemcpyDtoH") ||
        !check(cuMemcpyDtoH(&pitched_value,
                            pitched[i] + (kHeight - 1) * pitch[i], 4),
               "cuMemcpyDtoH(pitched)")) {
      return 1;
    }
    if (value != 0x1000u + i || pitched_value != 0x2000u + i) {
      std::fprintf(stderr, "device %d read %x and %x\n", i, value,
                   pitched_value);
      return 1;
    }
  }
  uint32_t copied = 0;
  if (!check(cuMemcpyPeer(linear[1], contexts[1], linear[0], contexts[0],
                          kBytes),
             "cuMemcpyPeer") ||
      !check(cuMemcpyDtoH(&copied, linear[1] + 4, 4), "cuMemcpyDtoH(peer)")) {
    return 1;
  }
  if (copied != 0x1000u) {
    std::fprintf(stderr, "peer copy delivered %x\n", copied);
    return 1;
  }
  // Each buffer is freed while the other device's context is current.
  for (int i = 0; i < 2; ++i) {
    if (!check(cuCtxSetCurrent(contexts[1 - i]), "cuCtxSetCurrent") ||
        !check(cuMemFree(linear[i]), "cuMemFree") ||
        !check(cuMemFree(pitched[i]), "cuMemFree(pitched)")) {
      return 1;
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (!check(cuCtxDestroy(contexts[i]), "cuCtxDestroy")) {
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
