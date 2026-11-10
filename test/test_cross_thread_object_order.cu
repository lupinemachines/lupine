// Work two threads hand to each other through the host stays in CUDA's order
// when it shares a CUDA object: a stream both threads launch on, an event one
// records and the other waits on, or the legacy stream, which waits for every
// blocking stream. Each producer kernel spins first, so if the consumer's
// work is submitted before the producer's, it reads the old value.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

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

// Queues a backlog of copies on a stream of the caller's own, so its next
// requests reach the server late: a request from another thread not ordered
// behind them would overtake them.
cudaError_t queue_backlog() {
  static unsigned char *pinned = nullptr, *backlog = nullptr;
  static cudaStream_t stream = nullptr;
  if (backlog == nullptr &&
      (cudaMallocHost(&pinned, 4096) != cudaSuccess ||
       cudaMalloc(&backlog, 512 * 4096) != cudaSuccess ||
       cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
           cudaSuccess)) {
    return cudaErrorMemoryAllocation;
  }
  for (int i = 0; i < 512; ++i) {
    cudaError_t r = cudaMemcpyAsync(backlog + i * 4096, pinned, 4096,
                                    cudaMemcpyHostToDevice, stream);
    if (r != cudaSuccess) {
      return r;
    }
  }
  return cudaSuccess;
}

// Runs the producer's launch on its thread, behind a backlog, then the
// consumer's on another once the producer's call has returned, as a host
// handoff would.
template <typename Produce, typename Consume>
cudaError_t hand_off(Produce produce, Consume consume) {
  std::atomic<bool> produced{false};
  cudaError_t produce_result = cudaSuccess;
  std::thread producer([&] {
    produce_result = queue_backlog();
    if (produce_result == cudaSuccess) {
      produce_result = produce();
    }
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
  if (expect("legacy stream", result, 0x33) != 0) {
    return 1;
  }

  // A copy into pageable memory on a stream reads after another thread's work
  // on that stream, small or large enough for the bulk path.
  for (size_t bytes : {sizeof(unsigned int), size_t(64) << 20}) {
    unsigned int *big = nullptr;
    CHECK(cudaMalloc(&big, bytes));
    CHECK(cudaMemset(big, 0, bytes));
    CHECK(cudaDeviceSynchronize());
    std::vector<unsigned int> host(bytes / sizeof(unsigned int));
    CHECK(hand_off(
        [&] {
          slow_store<<<1, 1, 0, shared>>>(big + host.size() - 1, 0x66, cycles);
          return cudaGetLastError();
        },
        [&] {
          return cudaMemcpyAsync(host.data(), big, bytes,
                                 cudaMemcpyDeviceToHost, shared);
        }));
    CHECK(cudaFree(big));
    if (host.back() != 0x66) {
      std::fprintf(stderr, "FAIL: pageable read of %zu bytes: read 0x%x\n",
                   bytes, host.back());
      return 1;
    }
    std::printf("pageable read of %zu bytes: ok\n", bytes);
  }

  // A query on the legacy stream answers without waiting, so it must not stand
  // in for a destroyed blocking stream's work: the next legacy launch still
  // waits for it.
  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  CHECK(hand_off(
      [&] {
        cudaStream_t gone = nullptr;
        cudaError_t r = cudaStreamCreate(&gone);
        if (r != cudaSuccess) {
          return r;
        }
        slow_store<<<1, 1, 0, gone>>>(slot, 0x77, cycles);
        r = cudaGetLastError();
        cudaError_t destroyed = cudaStreamDestroy(gone);
        return r != cudaSuccess ? r : destroyed;
      },
      [&] {
        (void)cudaStreamQuery(cudaStreamLegacy);
        copy_slot<<<1, 1, 0, 0>>>(slot, result);
        return cudaGetLastError();
      }));
  if (expect("legacy query, then destroyed blocking stream", result, 0x77) !=
      0) {
    return 1;
  }

  // An event a graph records orders another thread's wait on it, although the
  // launch names only its stream and executable.
  {
    cudaEvent_t recorded = nullptr;
    CHECK(cudaEventCreateWithFlags(&recorded, cudaEventDisableTiming));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CHECK(cudaStreamBeginCapture(mine, cudaStreamCaptureModeThreadLocal));
    slow_store<<<1, 1, 0, mine>>>(slot, 0x88, cycles);
    CHECK(cudaEventRecordWithFlags(recorded, mine, cudaEventRecordExternal));
    CHECK(cudaStreamEndCapture(mine, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CHECK(cudaMemset(slot, 0, sizeof(*slot)));
    CHECK(cudaDeviceSynchronize());
    CHECK(hand_off([&] { return cudaGraphLaunch(exec, mine); },
                   [&] {
                     cudaError_t r = cudaEventSynchronize(recorded);
                     copy_slot<<<1, 1, 0, shared>>>(slot, result);
                     return r != cudaSuccess ? r : cudaGetLastError();
                   }));
    if (expect("event recorded by a graph", result, 0x88) != 0) {
      return 1;
    }
    // A completion the client cached does not outlive a relaunch of the graph
    // that records the event.
    CHECK(cudaEventQuery(recorded));
    CHECK(cudaGraphLaunch(exec, mine));
    cudaError_t relaunched = cudaEventQuery(recorded);
    CHECK(cudaEventSynchronize(recorded));
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaEventDestroy(recorded));
    if (relaunched != cudaErrorNotReady) {
      std::fprintf(stderr, "FAIL: relaunched graph's event: %s\n",
                   cudaGetErrorName(relaunched));
      return 1;
    }
    std::printf("relaunched graph's event: ok\n");
  }

  // The legacy stream still waits for a blocking stream destroyed with work
  // pending, and for a thread's per-thread stream after the thread exits.
  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  CHECK(hand_off(
      [&] {
        cudaStream_t gone = nullptr;
        cudaError_t r = cudaStreamCreate(&gone);
        if (r != cudaSuccess) {
          return r;
        }
        slow_store<<<1, 1, 0, gone>>>(slot, 0x44, cycles);
        r = cudaGetLastError();
        cudaError_t destroyed = cudaStreamDestroy(gone);
        return r != cudaSuccess ? r : destroyed;
      },
      [&] {
        copy_slot<<<1, 1, 0, 0>>>(slot, result);
        return cudaGetLastError();
      }));
  if (expect("destroyed blocking stream", result, 0x44) != 0) {
    return 1;
  }

  CHECK(cudaMemset(slot, 0, sizeof(*slot)));
  cudaError_t exited_result = cudaSuccess;
  std::thread exited([&] {
    exited_result = queue_backlog();
    slow_store<<<1, 1, 0, cudaStreamPerThread>>>(slot, 0x55, cycles);
    if (exited_result == cudaSuccess) {
      exited_result = cudaGetLastError();
    }
  });
  exited.join();
  CHECK(exited_result);
  copy_slot<<<1, 1, 0, cudaStreamLegacy>>>(slot, result);
  CHECK(cudaGetLastError());
  return expect("exited thread's per-thread stream", result, 0x55);
}
