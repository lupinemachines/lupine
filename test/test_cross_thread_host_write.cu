// A host write to mapped or managed memory is ordered before another thread's
// CUDA call by the application's own synchronization between the two threads.
// The writer may make no CUDA call afterwards, so the reader's call must still
// see the write: through a kernel, a copy, and after the writer thread exits.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

constexpr int kRounds = 16;

__global__ void copy_word(unsigned int *destination,
                          const unsigned int *source) {
  *destination = *source;
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

// Writes `value` from a thread that then waits, without a CUDA call, until
// the caller has read it back through the device.
struct host_writer {
  std::atomic<unsigned int> request{0};
  std::atomic<unsigned int> written{0};
  std::atomic<bool> stop{false};
  unsigned int *target = nullptr;
  std::thread thread;

  void start() {
    thread = std::thread([this] {
      unsigned int seen = 0;
      while (!stop.load(std::memory_order_acquire)) {
        unsigned int value = request.load(std::memory_order_acquire);
        if (value == seen) {
          std::this_thread::yield();
          continue;
        }
        *static_cast<volatile unsigned int *>(target) = value;
        seen = value;
        written.store(value, std::memory_order_release);
      }
    });
  }

  void write(unsigned int value) {
    request.store(value, std::memory_order_release);
    while (written.load(std::memory_order_acquire) != value) {
      std::this_thread::yield();
    }
  }

  void finish() {
    stop.store(true, std::memory_order_release);
    thread.join();
  }
};

int run(const char *name, unsigned int *buffer, unsigned int *device,
        unsigned int base) {
  int failures = 0;
  host_writer writer;
  writer.target = buffer;
  writer.start();
  for (int round = 0; round < kRounds; ++round) {
    unsigned int value = base + 2 * round + 1;
    writer.write(value);
    copy_word<<<1, 1>>>(device, buffer);
    check(cudaGetLastError(), "copy_word launch");
    unsigned int got = 0;
    check(cudaMemcpy(&got, device, sizeof(got), cudaMemcpyDeviceToHost),
          "cudaMemcpy kernel result");
    if (got != value) {
      std::fprintf(stderr, "FAIL: %s round %d kernel read 0x%x want 0x%x\n",
                   name, round, got, value);
      ++failures;
    }

    value = base + 2 * round + 2;
    writer.write(value);
    check(cudaMemcpy(device, buffer, sizeof(value), cudaMemcpyDefault),
          "cudaMemcpy from host buffer");
    check(cudaMemcpy(&got, device, sizeof(got), cudaMemcpyDeviceToHost),
          "cudaMemcpy copy result");
    if (got != value) {
      std::fprintf(stderr, "FAIL: %s round %d copy read 0x%x want 0x%x\n",
                   name, round, got, value);
      ++failures;
    }
  }
  writer.finish();

  // The writer's last store, with its thread already gone.
  unsigned int value = base + 0xffff;
  std::thread([buffer, value] {
    *static_cast<volatile unsigned int *>(buffer) = value;
  }).join();
  copy_word<<<1, 1>>>(device, buffer);
  check(cudaGetLastError(), "copy_word launch");
  unsigned int got = 0;
  check(cudaMemcpy(&got, device, sizeof(got), cudaMemcpyDeviceToHost),
        "cudaMemcpy final result");
  if (got != value) {
    std::fprintf(stderr, "FAIL: %s exited writer read 0x%x want 0x%x\n", name,
                 got, value);
    ++failures;
  }
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

  unsigned int *mapped = nullptr;
  unsigned int *managed = nullptr;
  unsigned int *device = nullptr;
  check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
        "cudaHostAlloc");
  check(cudaMallocManaged(&managed, sizeof(*managed)), "cudaMallocManaged");
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  *mapped = 0;
  *managed = 0;
  // Let device code reach both buffers once before the cross-thread writes.
  copy_word<<<1, 1>>>(device, mapped);
  copy_word<<<1, 1>>>(device, managed);
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  int failures = run("mapped", mapped, device, 0x10000) +
                 run("managed", managed, device, 0x20000);

  check(cudaFree(device), "cudaFree");
  check(cudaFree(managed), "cudaFree managed");
  check(cudaFreeHost(mapped), "cudaFreeHost");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d cross-thread host writes were not seen\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: host writes from other threads reach device work\n");
  return EXIT_SUCCESS;
}
