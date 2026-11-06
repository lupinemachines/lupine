#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if CUDA_VERSION < 12000
int main() { std::puts("SKIP: tensor maps require CUDA 12"); }
#else
static void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s: CUDA error %d\n", operation, int(result));
    std::exit(1);
  }
}

int main() {
  check(cuInit(0), "init");
  CUdevice device;
  CUcontext context;
  CUdeviceptr address;
  check(cuDeviceGet(&device, 0), "device");
  check(cuDevicePrimaryCtxRetain(&context, device), "context");
  check(cuCtxSetCurrent(context), "current");
  check(cuMemAlloc(&address, 1024 * 1024), "allocation");
  cuuint64_t dims[] = {64, 64};
  cuuint64_t strides[] = {256};
  cuuint32_t box[] = {32, 32};
  cuuint32_t elements[] = {1, 1};
  auto encode = [&](CUtensorMap *map, CUtensorMapSwizzle swizzle) {
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        reinterpret_cast<void *>(address), dims, strides, box, elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  alignas(64) CUtensorMap first{}, repeated{}, changed{};
  CUresult result = encode(&first, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (result == CUDA_ERROR_NOT_SUPPORTED) {
    std::puts("SKIP: driver does not support tensor map encoding");
    check(cuMemFree(address), "free");
    check(cuDevicePrimaryCtxRelease(device), "release");
    return 0;
  }
  check(result, "first encode");
  check(encode(&repeated, CU_TENSOR_MAP_SWIZZLE_NONE), "repeat encode");
  if (std::memcmp(&first, &repeated, sizeof(first)) != 0)
    return 1;
  box[0] = 16;
  check(encode(&changed, CU_TENSOR_MAP_SWIZZLE_NONE), "changed box");
  if (std::memcmp(&first, &changed, sizeof(first)) == 0)
    return 1;
  box[0] = 32;
  dims[1] = 128;
  check(encode(&changed, CU_TENSOR_MAP_SWIZZLE_NONE), "changed dimensions");
  if (std::memcmp(&first, &changed, sizeof(first)) == 0)
    return 1;
  dims[1] = 64;
  strides[0] = 512;
  check(encode(&changed, CU_TENSOR_MAP_SWIZZLE_NONE), "changed stride");
  if (std::memcmp(&first, &changed, sizeof(first)) == 0)
    return 1;
  strides[0] = 256;
  for (int i = 0; i < 2; ++i) {
    if (encode(&repeated, static_cast<CUtensorMapSwizzle>(9999)) ==
        CUDA_SUCCESS)
      return 1;
  }
  check(encode(&repeated, CU_TENSOR_MAP_SWIZZLE_NONE), "repeat after failure");
  if (std::memcmp(&first, &repeated, sizeof(first)) != 0)
    return 1;
  // Exceed the bounded cache, then check the original input again.
  if (std::getenv("LUPINE_TENSOR_MAP_TEST_EVICTION")) {
    for (int i = 0; i < 4100; ++i) {
      dims[1] = 128 + i;
      check(encode(&changed, CU_TENSOR_MAP_SWIZZLE_NONE), "eviction encode");
    }
    dims[1] = 64;
    check(encode(&repeated, CU_TENSOR_MAP_SWIZZLE_NONE),
          "encode after eviction");
    if (std::memcmp(&first, &repeated, sizeof(first)) != 0)
      return 1;
  }
  const char *count = std::getenv("LUPINE_TENSOR_MAP_BENCH_CALLS");
  const int calls = count ? std::atoi(count) : 100;
  if (calls < 1)
    return 1;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < calls; ++i)
    check(encode(&repeated, CU_TENSOR_MAP_SWIZZLE_NONE), "benchmark encode");
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count();
  std::printf("tensor map repeats: calls=%d total_us=%.3f us_per_call=%.6f\n",
              calls, us, us / calls);
  check(cuMemFree(address), "free");
  check(cuDevicePrimaryCtxRelease(device), "release");
}
#endif
