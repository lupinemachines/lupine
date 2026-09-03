// Async CUDA calls issued by different host threads still have to reach the
// driver in issue order. A large kernel argument keeps one HTTP/2 lane busy
// long enough for a smaller request on another lane to overtake it unless the
// server sequences native submission.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

constexpr int kRounds = 512;
// CUDA toolkits before 12.1 limit kernel parameters to 4 KiB. Leave room for
// the output pointer while making the launch request as large as possible.
constexpr size_t kPayloadBytes = 4080;

struct alignas(16) large_payload {
  int value;
  unsigned char padding[kPayloadBytes - sizeof(int)];
};

__global__ void publish(large_payload payload, int *value) {
  *value = payload.value;
}

__global__ void verify(const int *value, int expected, int *failures) {
  if (*value != expected) {
    atomicAdd(failures, 1);
  }
}

__global__ void verify_zero(const int *value, int *failures) {
  if (*value != 0) {
    atomicAdd(failures, 1);
  }
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

void wait_for(const std::atomic<int> &phase, int expected) {
  while (phase.load(std::memory_order_acquire) != expected) {
    std::this_thread::yield();
  }
}

void require_no_failures(int *failures, const char *message) {
  int observed = -1;
  check(
      cudaMemcpy(&observed, failures, sizeof(observed), cudaMemcpyDeviceToHost),
      "cudaMemcpy failures");
  if (observed != 0) {
    std::fprintf(stderr, "FAIL: %d of %d %s\n", observed, kRounds, message);
    std::exit(EXIT_FAILURE);
  }
}

void test_kernel_launch_order(int *value, int *failures) {
  check(cudaMemset(value, 0xff, sizeof(*value)), "cudaMemset value");
  check(cudaMemset(failures, 0, sizeof(*failures)), "cudaMemset failures");

  std::atomic<int> phase{0};
  std::thread producer([&] {
    check(cudaSetDevice(0), "producer cudaSetDevice");
    large_payload payload = {};
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2);
      payload.value = round;
      publish<<<1, 1>>>(payload, value);
      check(cudaGetLastError(), "publish launch");
      phase.store(round * 2 + 1, std::memory_order_release);
    }
  });

  std::thread consumer([&] {
    check(cudaSetDevice(0), "consumer cudaSetDevice");
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2 + 1);
      verify<<<1, 1>>>(value, round, failures);
      check(cudaGetLastError(), "verify launch");
      phase.store(round * 2 + 2, std::memory_order_release);
    }
    check(cudaDeviceSynchronize(), "consumer cudaDeviceSynchronize");
  });

  producer.join();
  consumer.join();
  require_no_failures(failures, "cross-lane kernel launches overtook");
}

void test_cross_type_order(int *value, int *failures) {
  check(cudaMemset(failures, 0, sizeof(*failures)), "cudaMemset failures");

  std::atomic<int> phase{0};
  std::thread producer([&] {
    check(cudaSetDevice(0), "producer cudaSetDevice");
    large_payload payload = {};
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2);
      payload.value = round + 1;
      publish<<<1, 1>>>(payload, value);
      check(cudaGetLastError(), "publish launch");
      phase.store(round * 2 + 1, std::memory_order_release);
    }
  });

  std::thread consumer([&] {
    check(cudaSetDevice(0), "consumer cudaSetDevice");
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2 + 1);
      check(cudaMemsetAsync(value, 0, sizeof(*value)), "cudaMemsetAsync");
      verify_zero<<<1, 1>>>(value, failures);
      check(cudaGetLastError(), "verify-zero launch");
      phase.store(round * 2 + 2, std::memory_order_release);
    }
    check(cudaDeviceSynchronize(), "consumer cudaDeviceSynchronize");
  });

  producer.join();
  consumer.join();
  require_no_failures(failures, "async memsets overtook kernel launches");
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  check(cudaSetDevice(0), "cudaSetDevice");
  int *value = nullptr;
  int *failures = nullptr;
  check(cudaMalloc(&value, sizeof(*value)), "cudaMalloc value");
  check(cudaMalloc(&failures, sizeof(*failures)), "cudaMalloc failures");

  test_kernel_launch_order(value, failures);
  test_cross_type_order(value, failures);

  check(cudaFree(failures), "cudaFree failures");
  check(cudaFree(value), "cudaFree value");
  std::printf("PASS: async operations preserve order across client lanes\n");
  return EXIT_SUCCESS;
}
