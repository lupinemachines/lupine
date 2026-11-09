// A device-to-host copy into page-locked memory queued after a host function
// on the same stream lands. The host function's callback delivers the
// stream's finished copies with it; one queued after the callback has not run
// yet and must not be delivered then. Also checked with the synchronize on
// another thread, and with a host function on a second stream.
#include <cuda_runtime.h>

#include <cstdio>
#include <thread>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "FAIL: %s: %s\n", #call,                            \
                   cudaGetErrorString(check_result_));                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

void CUDART_CB noop(void *) {}

__global__ void set(unsigned int *slot, unsigned int value) { *slot = value; }

constexpr int kRounds = 50;

int run(const char *label, int mode) {
  cudaStream_t stream = nullptr, other = nullptr;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  unsigned int *device = nullptr, *host = nullptr;
  CHECK(cudaMalloc(&device, sizeof(*device)));
  CHECK(cudaMallocHost(&host, kRounds * sizeof(*host)));
  for (int i = 0; i < kRounds; ++i) {
    host[i] = 0xa5a5a5a5u;
  }
  int lost = 0;
  for (int round = 0; round < kRounds; ++round) {
    unsigned int value = 0x1000u + round;
    set<<<1, 1, 0, stream>>>(device, value);
    CHECK(cudaGetLastError());
    if (mode == 2) {
      CHECK(cudaLaunchHostFunc(other, noop, nullptr));
    }
    CHECK(cudaLaunchHostFunc(stream, noop, nullptr));
    CHECK(cudaMemcpyAsync(&host[round], device, sizeof(*device),
                          cudaMemcpyDeviceToHost, stream));
    if (mode == 1) {
      cudaError_t synced = cudaSuccess;
      std::thread synchronizer([&] { synced = cudaDeviceSynchronize(); });
      synchronizer.join();
      CHECK(synced);
    }
    CHECK(cudaDeviceSynchronize());
    lost += host[round] != value;
  }
  CHECK(cudaFreeHost(host));
  CHECK(cudaFree(device));
  CHECK(cudaStreamDestroy(other));
  CHECK(cudaStreamDestroy(stream));
  if (lost != 0) {
    std::fprintf(stderr, "FAIL: %s: %d/%d copies lost\n", label, lost, kRounds);
    return 1;
  }
  std::printf("%s: ok\n", label);
  return 0;
}

} // namespace

int main() {
  CHECK(cudaSetDevice(0));
  return run("one thread", 0) || run("synchronized from another thread", 1) ||
         run("host function on another stream too", 2);
}
