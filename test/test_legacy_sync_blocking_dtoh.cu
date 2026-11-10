// A synchronize of the legacy default stream waits for every blocking stream's
// earlier work, so the device-to-host copies queued there must have arrived
// when it returns: also when the blocking stream was destroyed with the copy
// outstanding, and whether the legacy stream is named by 0 or by
// cudaStreamLegacy.
// This file is auto-discovered by test/run_custom_tests.sh.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(check_result_));                         \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

__global__ void spin(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

__global__ void fill(unsigned int *out, unsigned int value, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = value + i;
  }
}

enum Case { kDestroyed, kLegacyHandle, kLegacyHandleCopy };

static const char *case_name(Case which) {
  switch (which) {
  case kDestroyed:
    return "blocking stream destroyed before cudaStreamSynchronize(0)";
  case kLegacyHandle:
    return "cudaStreamSynchronize(cudaStreamLegacy)";
  case kLegacyHandleCopy:
    return "copy on cudaStreamLegacy, cudaStreamSynchronize(0)";
  }
  return "?";
}

int main() {
  CHECK(cudaSetDevice(0));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  long long spin_cycles = static_cast<long long>(khz) / 2; // ~500 us

  constexpr int kCount = 4096;
  unsigned int *device = nullptr;
  CHECK(cudaMalloc(&device, kCount * sizeof(unsigned int)));
  unsigned int *host = nullptr;
  CHECK(cudaMallocHost(&host, kCount * sizeof(unsigned int)));

  int failures = 0;
  unsigned int value = 0;
  for (Case which : {kDestroyed, kLegacyHandle, kLegacyHandleCopy}) {
    for (int round = 0; round < 10; ++round) {
      value += 1u << 16;
      cudaStream_t stream = cudaStreamLegacy;
      if (which != kLegacyHandleCopy) {
        CHECK(cudaStreamCreate(&stream));
      }
      spin<<<1, 1, 0, stream>>>(spin_cycles);
      fill<<<(kCount + 255) / 256, 256, 0, stream>>>(device, value, kCount);
      CHECK(cudaMemcpyAsync(host, device, kCount * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost, stream));
      if (which == kDestroyed) {
        CHECK(cudaStreamDestroy(stream));
      }
      CHECK(cudaStreamSynchronize(which == kLegacyHandle ? cudaStreamLegacy
                                                         : nullptr));
      if (host[0] != value || host[kCount - 1] != value + kCount - 1) {
        std::fprintf(stderr, "%s, round %d: word 0 is 0x%x, expected 0x%x\n",
                     case_name(which), round, host[0], value);
        ++failures;
      }
      if (which == kLegacyHandle) {
        CHECK(cudaStreamDestroy(stream));
      }
      CHECK(cudaDeviceSynchronize());
    }
  }

  CHECK(cudaFreeHost(host));
  CHECK(cudaFree(device));
  if (failures != 0) {
    std::printf("FAIL: %d legacy-stream synchronizes missed a blocking "
                "stream's copy\n",
                failures);
    return 1;
  }
  std::printf("PASS: a legacy-stream synchronize collects blocking streams' "
              "copies\n");
  return 0;
}
