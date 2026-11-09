// Work two threads hand to each other through the host stays in CUDA's order
// when it shares a CUDA object: a stream both threads launch on, an event one
// records and the other waits on, or the legacy stream, which waits for every
// blocking stream. Each producer kernel spins first, so if the consumer's
// work is submitted before the producer's, it reads the old value.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <thread>

namespace {

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "FAIL: %s: %s\n", #call,                            \
                   cudaGetErrorString(check_result_));                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void slow_store(unsigned int *slot, unsigned int value,
                           long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
  *slot = value;
}

__global__ void copy_slot(const unsigned int *from, unsigned int *to) {
  *to = *from;
}

long long spin_cycles() {
  int khz = 0;
  cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  return static_cast<long long>(khz) * 20; // ~20 ms
}

// Runs the producer's launch on its thread, then the consumer's on another
// once the producer's call has returned, as a host handoff would.
template <typename Produce, typename Consume>
cudaError_t hand_off(Produce produce, Consume consume) {
  std::atomic<bool> produced{false};
  cudaError_t produce_result = cudaSuccess;
  std::thread producer([&] {
    produce_result = produce();
    produced = true;
  });
  while (!produced) {
    std::this_thread::yield();
  }
  cudaError_t consume_result = cudaSuccess;
  std::thread consumer([&] { consume_result = consume(); });
  producer.join();
  consumer.join();
  return produce_result != cudaSuccess ? produce_result : consume_result;
}

int expect(const char *label, unsigned int *result, unsigned int want) {
  unsigned int got = 0;
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(&got, result, sizeof(got), cudaMemcpyDeviceToHost));
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s: read 0x%x, expected 0x%x\n", label, got,
                 want);
    return 1;
  }
  std::printf("%s: ok\n", label);
  return 0;
}

} // namespace

int main() {
  CHECK(cudaSetDevice(0));
  long long cycles = spin_cycles();
  unsigned int *slot = nullptr, *result = nullptr;
  CHECK(cudaMalloc(&slot, sizeof(*slot)));
  CHECK(cudaMalloc(&result, sizeof(*result)));
  cudaStream_t shared = nullptr, mine = nullptr, theirs = nullptr;
  CHECK(cudaStreamCreateWithFlags(&shared, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&mine, cudaStreamNonBlocking));
  CHECK(cudaStreamCreate(&theirs)); // blocking: the legacy stream waits for it
  cudaEvent_t done = nullptr;
  CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

  // A stream both threads use orders their launches.
  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  CHECK(hand_off(
      [&] {
        slow_store<<<1, 1, 0, shared>>>(slot, 0x11, cycles);
        return cudaGetLastError();
      },
      [&] {
        copy_slot<<<1, 1, 0, shared>>>(slot, result);
        return cudaGetLastError();
      }));
  if (expect("shared stream", result, 0x11) != 0) {
    return 1;
  }

  // An event recorded on one thread orders the stream another waits with.
  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  CHECK(hand_off(
      [&] {
        slow_store<<<1, 1, 0, theirs>>>(slot, 0x22, cycles);
        return cudaEventRecord(done, theirs);
      },
      [&] {
        cudaError_t r = cudaStreamWaitEvent(mine, done, 0);
        copy_slot<<<1, 1, 0, mine>>>(slot, result);
        return r != cudaSuccess ? r : cudaGetLastError();
      }));
  if (expect("event", result, 0x22) != 0) {
    return 1;
  }

  // The legacy stream waits for a blocking stream another thread launched on.
  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  CHECK(hand_off(
      [&] {
        slow_store<<<1, 1, 0, theirs>>>(slot, 0x33, cycles);
        return cudaGetLastError();
      },
      [&] {
        copy_slot<<<1, 1, 0, 0>>>(slot, result);
        return cudaGetLastError();
      }));
  return expect("legacy stream", result, 0x33);
}
