// Manual cross-server async device-to-device copy integration test for #255.
//
// Start two LUPINE servers, then run the client against both:
//
//   LUPINE_PORT=15001 ./build/lupine_driver_server &
//   LUPINE_PORT=15002 ./build/lupine_driver_server &
//   nvcc -cudart=shared -lcuda test/test_cross_server_d2d_async.cu \
//     -o /tmp/test_cross_server_d2d_async
//   LUPINE_SERVER=<host-a>:15001,<host-b>:15002 \
//     LD_PRELOAD=./build/libcuda.so.1 /tmp/test_cross_server_d2d_async
//
// By default this copies from device 0 to device ndev/2, which matches a
// two-server setup with the first server exposing no more GPUs than the second.
// Override with LUPINE_TEST_SRC_DEVICE and LUPINE_TEST_DST_DEVICE as needed.

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    fprintf(stderr, "FAIL %s: %d (%s: %s)\n", what, static_cast<int>(result),
            name == nullptr ? "unknown" : name,
            text == nullptr ? "unknown" : text);
    std::exit(1);
  }
}

static int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return fallback;
  }
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' ? static_cast<int>(parsed) : fallback;
}

int main() {
  check(cuInit(0), "cuInit");

  int ndev = 0;
  check(cuDeviceGetCount(&ndev), "cuDeviceGetCount");
  if (ndev < 2) {
    fprintf(stderr, "need at least two exposed devices; saw %d\n", ndev);
    return 1;
  }

  int srcdev = env_int("LUPINE_TEST_SRC_DEVICE", 0);
  int dstdev = env_int("LUPINE_TEST_DST_DEVICE", ndev / 2);
  if (srcdev < 0 || dstdev < 0 || srcdev >= ndev || dstdev >= ndev ||
      srcdev == dstdev) {
    fprintf(stderr, "invalid device pair src=%d dst=%d ndev=%d\n", srcdev,
            dstdev, ndev);
    return 1;
  }

  printf("device count=%d; async D2D src=device%d -> dst=device%d\n", ndev,
         srcdev, dstdev);

  CUdevice src_device = 0;
  CUdevice dst_device = 0;
  check(cuDeviceGet(&src_device, srcdev), "cuDeviceGet(src)");
  check(cuDeviceGet(&dst_device, dstdev), "cuDeviceGet(dst)");

  CUcontext src_ctx = nullptr;
  CUcontext dst_ctx = nullptr;
  check(cuDevicePrimaryCtxRetain(&src_ctx, src_device),
        "cuDevicePrimaryCtxRetain(src)");
  check(cuDevicePrimaryCtxRetain(&dst_ctx, dst_device),
        "cuDevicePrimaryCtxRetain(dst)");

  constexpr int kCount = 1 << 20;
  constexpr size_t kBytes = kCount * sizeof(int);
  std::vector<int> input(kCount);
  std::vector<int> output(kCount, 0);
  for (int i = 0; i < kCount; ++i) {
    input[i] = i * 7 + 3;
  }

  CUdeviceptr src = 0;
  CUdeviceptr dst = 0;
  check(cuCtxSetCurrent(src_ctx), "cuCtxSetCurrent(src)");
  check(cuMemAlloc(&src, kBytes), "cuMemAlloc(src)");
  check(cuMemcpyHtoD(src, input.data(), kBytes), "cuMemcpyHtoD(src)");

  CUstream stream = nullptr;
  check(cuCtxSetCurrent(dst_ctx), "cuCtxSetCurrent(dst)");
  check(cuMemAlloc(&dst, kBytes), "cuMemAlloc(dst)");
  check(cuStreamCreate(&stream, CU_STREAM_DEFAULT), "cuStreamCreate");
  check(cuMemcpyDtoDAsync(dst, src, kBytes, stream),
        "cuMemcpyDtoDAsync(cross-server)");
  check(cuStreamSynchronize(stream), "cuStreamSynchronize");
  check(cuMemcpyDtoH(output.data(), dst, kBytes), "cuMemcpyDtoH(dst)");

  int mismatches = 0;
  for (int i = 0; i < kCount; ++i) {
    if (output[i] != input[i]) {
      if (mismatches < 5) {
        printf("mismatch at %d: got %d want %d\n", i, output[i], input[i]);
      }
      ++mismatches;
    }
  }
  if (mismatches != 0) {
    printf("FAIL: %d/%d mismatches\n", mismatches, kCount);
    return 1;
  }

  printf("PASS: async cross-server D2D verified (%d ints)\n", kCount);
  return 0;
}
