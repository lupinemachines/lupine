// A store into tracked host memory faults; the handler publishes the page and
// makes it writable, and the store itself runs only after the handler
// returns. A flush on another thread in between must not leave that store
// unpublished: the device has to see it once the writer hands off to another
// thread, however the flushes interleave.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

constexpr int kBuffers = 8;
constexpr int kRounds = 200;

// Each buffer is its own argument, so the launch exposes every one of them.
__global__ void gather(unsigned int *out, unsigned int *b0, unsigned int *b1,
                       unsigned int *b2, unsigned int *b3, unsigned int *b4,
                       unsigned int *b5, unsigned int *b6, unsigned int *b7) {
  unsigned int *buffers[kBuffers] = {b0, b1, b2, b3, b4, b5, b6, b7};
  out[threadIdx.x] = *buffers[threadIdx.x];
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

int run(const char *name, unsigned int *const *buffers) {
  unsigned int *out = nullptr;
  unsigned int *scratch = nullptr;
  check(cudaMalloc(&out, sizeof(*out) * kBuffers), "cudaMalloc");
  check(cudaMalloc(&scratch, sizeof(*scratch)), "cudaMalloc");

  // Keeps requests, and with them flushes, going on another thread.
  std::atomic<bool> stop{false};
  std::thread flusher([&] {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
    while (!stop.load()) {
      check(cudaMemsetAsync(scratch, 0, sizeof(*scratch), stream),
            "cudaMemsetAsync");
    }
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  });

  std::atomic<int> turn{0};
  std::thread writer([&] {
    for (int round = 0; round < kRounds; ++round) {
      while (turn.load() != 2 * round) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kBuffers; ++i) {
        *static_cast<volatile unsigned int *>(buffers[i]) =
            static_cast<unsigned int>(round * kBuffers + i + 1);
      }
      turn.store(2 * round + 1);
    }
  });

  int failures = 0;
  unsigned int got[kBuffers];
  for (int round = 0; round < kRounds; ++round) {
    while (turn.load() != 2 * round + 1) {
      std::this_thread::yield();
    }
    gather<<<1, kBuffers>>>(out, buffers[0], buffers[1], buffers[2],
                            buffers[3], buffers[4], buffers[5], buffers[6],
                            buffers[7]);
    check(cudaGetLastError(), "gather launch");
    check(cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost),
          "cudaMemcpy");
    for (int i = 0; i < kBuffers; ++i) {
      unsigned int want = static_cast<unsigned int>(round * kBuffers + i + 1);
      if (got[i] != want && failures++ < 4) {
        std::fprintf(stderr, "FAIL: %s round %d buffer %d: device read 0x%x "
                     "want 0x%x\n", name, round, i, got[i], want);
      }
    }
    turn.store(2 * round + 2);
  }
  writer.join();
  stop.store(true);
  flusher.join();
  check(cudaFree(scratch), "cudaFree");
  check(cudaFree(out), "cudaFree");
  return failures;
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  unsigned int *managed[kBuffers];
  unsigned int *mapped[kBuffers];
  for (int i = 0; i < kBuffers; ++i) {
    check(cudaMallocManaged(&managed[i], sizeof(unsigned int)),
          "cudaMallocManaged");
    check(cudaHostAlloc(&mapped[i], sizeof(unsigned int), cudaHostAllocMapped),
          "cudaHostAlloc");
    *managed[i] = 0;
    *mapped[i] = 0;
  }

  int failures = run("managed", managed) + run("mapped", mapped);

  for (int i = 0; i < kBuffers; ++i) {
    check(cudaFree(managed[i]), "cudaFree");
    check(cudaFreeHost(mapped[i]), "cudaFreeHost");
  }
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d host stores did not reach the device\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: host stores reach the device across concurrent flushes\n");
  return EXIT_SUCCESS;
}
