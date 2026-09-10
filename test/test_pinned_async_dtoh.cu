// Verify that a DtoH copy into pinned memory remains asynchronous and obeys a
// cross-stream event dependency. In mixed local/remote mode this also covers
// native dispatch while the Lupine shim is enabled.
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t kElements = 16384;

bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "FAIL: %s returned CUDA error %d (%s)\n", operation,
               static_cast<int>(result), name == nullptr ? "unknown" : name);
  return false;
}

} // namespace

int main() {
  if (!check(cuInit(0), "cuInit")) {
    return 1;
  }

  CUdevice device = 0;
  CUcontext context = nullptr;
  CUdeviceptr allocation = 0;
  CUstream producer = nullptr;
  CUstream consumer = nullptr;
  CUevent ready = nullptr;
  unsigned int *pinned = nullptr;
  const size_t bytes = kElements * sizeof(*pinned);

  if (!check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuMemAllocHost(reinterpret_cast<void **>(&pinned), 2 * bytes),
             "cuMemAllocHost") ||
      !check(cuMemAlloc(&allocation, bytes), "cuMemAlloc") ||
      !check(cuStreamCreate(&producer, CU_STREAM_NON_BLOCKING),
             "cuStreamCreate(producer)") ||
      !check(cuStreamCreate(&consumer, CU_STREAM_NON_BLOCKING),
             "cuStreamCreate(consumer)") ||
      !check(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING), "cuEventCreate")) {
    return 1;
  }

  std::vector<unsigned int> expected(kElements);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<unsigned int>(i * 2654435761U) ^ 0x725U;
  }
  std::copy(expected.begin(), expected.end(), pinned);
  std::fill(pinned + kElements, pinned + 2 * kElements, 0);

  if (!check(cuMemcpyHtoDAsync(allocation, pinned, bytes, producer),
             "cuMemcpyHtoDAsync") ||
      !check(cuEventRecord(ready, producer), "cuEventRecord") ||
      !check(cuStreamWaitEvent(consumer, ready, 0), "cuStreamWaitEvent") ||
      !check(cuMemcpyDtoHAsync(pinned + kElements, allocation, bytes, consumer),
             "cuMemcpyDtoHAsync") ||
      !check(cuStreamSynchronize(consumer), "cuStreamSynchronize")) {
    return 1;
  }

  if (!std::equal(expected.begin(), expected.end(), pinned + kElements)) {
    std::fprintf(stderr, "FAIL: pinned async DtoH output mismatch\n");
    return 1;
  }

  if (!check(cuEventDestroy(ready), "cuEventDestroy") ||
      !check(cuStreamDestroy(consumer), "cuStreamDestroy(consumer)") ||
      !check(cuStreamDestroy(producer), "cuStreamDestroy(producer)") ||
      !check(cuMemFree(allocation), "cuMemFree") ||
      !check(cuMemFreeHost(pinned), "cuMemFreeHost") ||
      !check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }

  std::printf("PASS: pinned async DtoH preserves stream ordering and bytes\n");
  return 0;
}
