// A synchronize of the legacy default stream does not wait for non-blocking
// streams, so it must not collect their device-to-host copies. One thread keeps
// synchronizing the legacy stream while another copies into page-locked memory
// on a non-blocking stream and then synchronizes that stream: the copy has to
// arrive with the bytes it copied, not whatever its staging held when the
// legacy synchronize happened to run.
// This file is auto-discovered by test/run_custom_tests.sh.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

int main() {
  CHECK(cudaSetDevice(0));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  long long spin_cycles = static_cast<long long>(khz) / 2; // ~500 us

  std::atomic<bool> stop{false};
  std::atomic<int> chatter_failures{0};
  std::thread chatter([&] {
    constexpr int kCount = 256;
    unsigned int *device = nullptr;
    CHECK(cudaMalloc(&device, kCount * sizeof(unsigned int)));
    unsigned int host[kCount];
    for (unsigned int round = 0; !stop.load(); ++round) {
      fill<<<1, kCount>>>(device, round, kCount);
      CHECK(cudaMemcpyAsync(host, device, sizeof(host), cudaMemcpyDeviceToHost,
                            0));
      CHECK(cudaStreamSynchronize(0));
      if (host[kCount - 1] != round + kCount - 1) {
        chatter_failures.fetch_add(1);
      }
    }
    CHECK(cudaFree(device));
  });

  constexpr int kCount = 4096;
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  unsigned int *device = nullptr;
  CHECK(cudaMalloc(&device, kCount * sizeof(unsigned int)));
  unsigned int *host = nullptr;
  CHECK(cudaMallocHost(&host, kCount * sizeof(unsigned int)));

  int failures = 0;
  for (unsigned int round = 1; round <= 100; ++round) {
    unsigned int value = round << 16;
    spin<<<1, 1, 0, stream>>>(spin_cycles);
    fill<<<(kCount + 255) / 256, 256, 0, stream>>>(device, value, kCount);
    CHECK(cudaMemcpyAsync(host, device, kCount * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < kCount; ++i) {
      if (host[i] != value + i) {
        if (failures < 3) {
          std::fprintf(stderr, "round %u: word %d is 0x%x, expected 0x%x\n",
                       round, i, host[i], value + i);
        }
        ++failures;
        break;
      }
    }
  }
  stop.store(true);
  chatter.join();

  CHECK(cudaFreeHost(host));
  CHECK(cudaFree(device));
  CHECK(cudaStreamDestroy(stream));
  if (failures != 0 || chatter_failures.load() != 0) {
    std::printf("FAIL: %d stale non-blocking copies, %d stale legacy copies\n",
                failures, chatter_failures.load());
    return 1;
  }
  std::printf("PASS: a legacy-stream synchronize leaves non-blocking streams' "
              "copies to them\n");
  return 0;
}
