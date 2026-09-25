// A synchronous DtoH copy large enough to take the bulk lanes into a host
// allocation the server also sees (managed, or mapped pinned) must land in
// the client's view and in the device's view of that allocation, whether the
// destination was fresh on the host or last written by the device.
//
// Pure driver API, no kernels (so it is arch-independent).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <vector>

static int g_failures = 0;

static const char *errstr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}
#define DRV(call)                                                              \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call, errstr(_r),        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const size_t kBytes = 16u << 20;
static const size_t kWords = kBytes / sizeof(unsigned int);

static bool all_equal(const unsigned int *words, unsigned int value) {
  for (size_t i = 0; i < kWords; ++i) {
    if (words[i] != value) {
      fprintf(stderr, "  word %zu is 0x%08x, expected 0x%08x\n", i, words[i],
              value);
      return false;
    }
  }
  return true;
}

// Copies `source` into `host` (whose device view is `device_view`), then
// checks both views.
static int copy_and_check(const char *label, unsigned int *host,
                          CUdeviceptr device_view, CUdeviceptr source,
                          unsigned int value, CUdeviceptr scratch) {
  DRV(cuMemsetD32(source, value, kWords));
  CUresult r = cuMemcpyDtoH(host, source, kBytes);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "FAIL: %s: cuMemcpyDtoH -> %s\n", label, errstr(r));
    g_failures++;
    return 0;
  }
  if (!all_equal(host, value)) {
    fprintf(stderr, "FAIL: %s: host view\n", label);
    g_failures++;
    return 0;
  }
  std::vector<unsigned int> readback(kWords, 0);
  DRV(cuMemcpyDtoD(scratch, device_view, kBytes));
  DRV(cuMemcpyDtoH(readback.data(), scratch, kBytes));
  if (!all_equal(readback.data(), value)) {
    fprintf(stderr, "FAIL: %s: device view\n", label);
    g_failures++;
    return 0;
  }
  fprintf(stderr, "ok:   %s\n", label);
  return 0;
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));

  CUdeviceptr source, scratch;
  DRV(cuMemAlloc(&source, kBytes));
  DRV(cuMemAlloc(&scratch, kBytes));

  CUdeviceptr managed;
  DRV(cuMemAllocManaged(&managed, kBytes, CU_MEM_ATTACH_GLOBAL));
  auto *managed_host = reinterpret_cast<unsigned int *>(managed);
  memset(managed_host, 0, kBytes);
  if (copy_and_check("managed, host-fresh destination", managed_host, managed,
                     source, 0x11111111u, scratch) != 0) {
    return 1;
  }
  DRV(cuMemsetD32(managed, 0x22222222u, kWords));
  DRV(cuCtxSynchronize());
  if (copy_and_check("managed, device-written destination", managed_host,
                     managed, source, 0x33333333u, scratch) != 0) {
    return 1;
  }

  void *pinned = nullptr;
  DRV(cuMemHostAlloc(&pinned, kBytes, CU_MEMHOSTALLOC_DEVICEMAP));
  CUdeviceptr pinned_device;
  DRV(cuMemHostGetDevicePointer(&pinned_device, pinned, 0));
  auto *pinned_host = static_cast<unsigned int *>(pinned);
  memset(pinned_host, 0, kBytes);
  if (copy_and_check("mapped pinned, host-fresh destination", pinned_host,
                     pinned_device, source, 0x44444444u, scratch) != 0) {
    return 1;
  }
  DRV(cuMemsetD32(pinned_device, 0x55555555u, kWords));
  DRV(cuCtxSynchronize());
  if (copy_and_check("mapped pinned, device-written destination", pinned_host,
                     pinned_device, source, 0x66666666u, scratch) != 0) {
    return 1;
  }

  DRV(cuMemFreeHost(pinned));
  DRV(cuMemFree(managed));
  DRV(cuMemFree(scratch));
  DRV(cuMemFree(source));
  DRV(cuDevicePrimaryCtxRelease(dev));

  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
