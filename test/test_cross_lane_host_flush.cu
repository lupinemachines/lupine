// A dirty thread may flush ranges published by other client threads. The batch
// must be server-visible before any publisher whose ticket it contains lets
// another HTTP/2 lane use its mapped allocation as an HtoD source.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <thread>
#include <vector>

namespace {

constexpr int kThreadCount = 8;
constexpr int kRounds = 32;
constexpr size_t kElements = 100000;
constexpr size_t kBytes = kElements * sizeof(int);

struct callback_state {
  int *source = nullptr;
  int expected = 0;
  int thread_index = 0;
  int round = 0;
  std::atomic<int> *failures = nullptr;
  std::atomic<bool> done{false};
};

__global__ void increment(int *values) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kElements) {
    ++values[index];
  }
}

void CUDART_CB validate(cudaStream_t, cudaError_t status, void *opaque) {
  auto *state = static_cast<callback_state *>(opaque);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FAIL: callback: %s\n", cudaGetErrorString(status));
    state->failures->fetch_add(1, std::memory_order_relaxed);
  } else {
    for (size_t i = 0; i < kElements; ++i) {
      if (state->source[i] == state->expected) {
        continue;
      }
      std::fprintf(stderr,
                   "FAIL: lane %d round %d element %zu: got %d want %d\n",
                   state->thread_index, state->round, i, state->source[i],
                   state->expected);
      state->failures->fetch_add(1, std::memory_order_relaxed);
      break;
    }
  }
  state->done.store(true, std::memory_order_release);
}

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

  pthread_barrier_t barrier;
  if (pthread_barrier_init(&barrier, nullptr, kThreadCount) != 0) {
    std::fprintf(stderr, "FAIL: pthread_barrier_init\n");
    return EXIT_FAILURE;
  }

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([thread_index, device_count, &barrier, &failures] {
      check(cudaSetDevice(thread_index % device_count), "cudaSetDevice");

      cudaStream_t stream = nullptr;
      int *device = nullptr;
      int *source = nullptr;
      check(cudaStreamCreate(&stream), "cudaStreamCreate");
      check(cudaMalloc(&device, kBytes), "cudaMalloc");
      check(cudaHostAlloc(&source, kBytes, cudaHostAllocPortable),
            "cudaHostAlloc");
      for (int round = 0; round < kRounds; ++round) {
        int expected = 1 + thread_index * kRounds + round;
        std::fill(source, source + kElements, expected);

        // Make every lane race to consume dirty mapped-host state. One lane
        // drains the shared queue; the others wait only for their own tickets.
        pthread_barrier_wait(&barrier);
        check(cudaMemcpyAsync(device, source, kBytes, cudaMemcpyHostToDevice,
                              stream),
              "cudaMemcpyAsync HtoD");
        increment<<<(kElements + 255) / 256, 256, 0, stream>>>(device);
        check(cudaGetLastError(), "increment launch");
        check(cudaMemcpyAsync(source, device, kBytes, cudaMemcpyDeviceToHost,
                              stream),
              "cudaMemcpyAsync DtoH");
        callback_state state{source, expected + 1, thread_index, round,
                             &failures};
        check(cudaStreamAddCallback(stream, validate, &state, 0),
              "cudaStreamAddCallback");
        while (!state.done.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        pthread_barrier_wait(&barrier);
      }

      check(cudaFreeHost(source), "cudaFreeHost");
      check(cudaFree(device), "cudaFree");
      check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    });
  }

  for (std::thread &thread : threads) {
    thread.join();
  }
  pthread_barrier_destroy(&barrier);

  int failure_count = failures.load(std::memory_order_relaxed);
  if (failure_count != 0) {
    std::fprintf(stderr, "FAIL: %d cross-lane host flush mismatches\n",
                 failure_count);
    return EXIT_FAILURE;
  }
  std::printf("PASS: mapped host flushes are ordered across client lanes\n");
  return EXIT_SUCCESS;
}
