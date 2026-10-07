// A host release/acquire handoff orders the CUDA calls without waiting for
// the producer's GPU work. The consumer must observe its predecessor even
// when its smaller request reaches a different server lane first.
#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
constexpr int kRounds = 512;
struct alignas(16) Payload {
  int value;
  unsigned char padding[4076];
};

__global__ void publish(Payload payload, int *value) { *value = payload.value; }

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

void wait_for(const std::atomic<int> &phase, int expected) {
  while (phase.load(std::memory_order_acquire) != expected) {
    std::this_thread::yield();
  }
}

enum class Read {
  runtime,
  driver,
  stream,
  context,
  event,
  event_query,
  legacy
};

bool run(Read kind, const char *name) {
  int *device = nullptr;
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  if (kind != Read::runtime && kind != Read::driver) {
    check(cudaStreamCreateWithFlags(&stream, kind == Read::legacy
                                                 ? cudaStreamDefault
                                                 : cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
  }
  if (kind == Read::event || kind == Read::event_query) {
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
          "cudaEventCreateWithFlags");
  }

  std::atomic<int> phase{0};
  std::atomic<bool> reader_ready{false};
  int failures = 0;
  std::thread writer([&] {
    check(cudaSetDevice(0), "writer cudaSetDevice");
    Payload payload{};
    // Incompressible padding makes this lane slower than the reader's lane.
    unsigned int random = 12345;
    for (unsigned char &byte : payload.padding) {
      random = random * 1664525u + 1013904223u;
      byte = static_cast<unsigned char>(random >> 24);
    }
    publish<<<1, 1, 0, stream>>>(payload, device);
    check(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");
    while (!reader_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2);
      payload.value = round + 1;
      publish<<<1, 1, 0, stream>>>(payload, device);
      check(cudaGetLastError(), "publish launch");
      if (event != nullptr) {
        check(cudaEventRecord(event, stream), "cudaEventRecord");
      }
      phase.store(round * 2 + 1, std::memory_order_release);
    }
    wait_for(phase, kRounds * 2);
    // Cleanup also drains any work left behind by a failing implementation.
    check(cudaDeviceSynchronize(), "final writer cudaDeviceSynchronize");
  });
  std::thread reader([&] {
    check(cudaSetDevice(0), "reader cudaSetDevice");
    reader_ready.store(true, std::memory_order_release);
    for (int round = 0; round < kRounds; ++round) {
      wait_for(phase, round * 2 + 1);
      if (kind == Read::stream) {
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      } else if (kind == Read::context) {
        check(cudaDeviceSynchronize(), "reader cudaDeviceSynchronize");
      } else if (kind == Read::event) {
        check(cudaEventSynchronize(event), "cudaEventSynchronize");
      } else if (kind == Read::event_query) {
        cudaError_t result;
        while ((result = cudaEventQuery(event)) == cudaErrorNotReady) {
          std::this_thread::yield();
        }
        check(result, "cudaEventQuery");
      }
      int observed = -1;
      if (kind == Read::driver) {
        CUresult result = cuMemcpyDtoH(
            &observed, reinterpret_cast<CUdeviceptr>(device), sizeof(observed));
        if (result != CUDA_SUCCESS) {
          std::fprintf(stderr, "FAIL: cuMemcpyDtoH: %d\n", result);
          std::exit(EXIT_FAILURE);
        }
      } else {
        check(cudaMemcpy(&observed, device, sizeof(observed),
                         cudaMemcpyDeviceToHost),
              "reader cudaMemcpy");
      }
      failures += observed != round + 1;
      phase.store(round * 2 + 2, std::memory_order_release);
    }
  });
  writer.join();
  reader.join();
  if (event != nullptr)
    check(cudaEventDestroy(event), "cudaEventDestroy");
  if (stream != nullptr)
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFree(device), "cudaFree");
  std::printf("%s: %s: %d/%d stale reads\n", failures ? "FAIL" : "PASS", name,
              failures, kRounds);
  return failures == 0;
}
} // namespace

int main() {
  check(cudaSetDevice(0), "main cudaSetDevice");
  bool ok = run(Read::runtime, "legacy launch -> runtime synchronous read");
  ok &= run(Read::driver, "legacy launch -> driver synchronous read");
  ok &= run(Read::stream, "nonblocking stream -> stream synchronize");
  ok &= run(Read::context, "nonblocking stream -> context synchronize");
  ok &= run(Read::event, "event record -> event synchronize");
  ok &= run(Read::event_query, "event record -> event query");
  ok &= run(Read::legacy, "blocking stream -> legacy synchronous read");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
