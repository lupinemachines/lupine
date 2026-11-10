// Copies and memsets can write mapped host memory from the device without any
// kernel ever receiving its address. The host must still read their bytes
// after synchronizing, as it does when a kernel wrote them.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  unsigned int *device = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");
  unsigned int seven = 7;
  check(cudaMemcpy(device, &seven, sizeof(seven), cudaMemcpyHostToDevice),
        "cudaMemcpy");

  int failures = 0;
  for (int write = 0; write < 7; ++write) {
    // A fresh allocation each time, and no pointer query, so nothing but the
    // write itself hands its device mapping out.
    unsigned int *mapped = nullptr;
    check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
          "cudaHostAlloc");
    *mapped = 0;
    unsigned int pageable = 9;
    unsigned int want = 0;
    const char *name = "";
    switch (write) {
    case 0:
      name = "cudaMemsetAsync";
      check(cudaMemsetAsync(mapped, 0x11, sizeof(*mapped), stream), name);
      want = 0x11111111;
      break;
    case 1:
      name = "cudaMemset";
      check(cudaMemset(mapped, 0x22, sizeof(*mapped)), name);
      want = 0x22222222;
      break;
    case 2:
      name = "cudaMemset2DAsync";
      check(cudaMemset2DAsync(mapped, sizeof(*mapped), 0x33, sizeof(*mapped),
                              1, stream),
            name);
      want = 0x33333333;
      break;
    case 3:
      name = "device-to-device cudaMemcpyAsync";
      check(cudaMemcpyAsync(mapped, device, sizeof(*mapped),
                            cudaMemcpyDeviceToDevice, stream),
            name);
      want = 7;
      break;
    case 4:
      name = "device-to-device cudaMemcpy";
      check(cudaMemcpy(mapped, device, sizeof(*mapped),
                       cudaMemcpyDeviceToDevice),
            name);
      want = 7;
      break;
    case 5:
      name = "host-to-device cudaMemcpyAsync";
      check(cudaMemcpyAsync(mapped, &pageable, sizeof(*mapped),
                            cudaMemcpyHostToDevice, stream),
            name);
      want = 9;
      break;
    case 6:
      name = "host-to-device cudaMemcpy";
      check(cudaMemcpy(mapped, &pageable, sizeof(*mapped),
                       cudaMemcpyHostToDevice),
            name);
      want = 9;
      break;
    }
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int got = *static_cast<volatile unsigned int *>(mapped);
    if (got != want) {
      std::fprintf(stderr, "FAIL: %s: host read 0x%x want 0x%x\n", name, got,
                   want);
      ++failures;
    }
    check(cudaFreeHost(mapped), "cudaFreeHost");
  }

  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFree(device), "cudaFree");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d device writes stayed invisible\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: device writes to mapped memory reach the host\n");
  return EXIT_SUCCESS;
}
