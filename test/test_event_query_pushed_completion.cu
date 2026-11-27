// Event queries answered from pushed completions must agree with what the
// device has finished. A query after a synchronize that waited for the event,
// or after a later event that covers it completed, reports it complete;
// polling a recorded event ends; and a record behind unfinished work is not
// reported complete by the previous record's completion.
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

static void expect(cudaError_t got, cudaError_t want, int round,
                   const char *what) {
  if (got != want) {
    std::fprintf(stderr, "round %d: %s: got %s, want %s\n", round, what,
                 cudaGetErrorName(got), cudaGetErrorName(want));
    std::exit(1);
  }
}

static cudaError_t poll(cudaEvent_t event) {
  cudaError_t status;
  while ((status = cudaEventQuery(event)) == cudaErrorNotReady) {
  }
  return status;
}

int main() {
  CHECK(cudaSetDevice(0));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  long long ms = khz;
  cudaStream_t a, b;
  CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
  cudaEvent_t first, second;
  CHECK(cudaEventCreateWithFlags(&first, cudaEventDisableTiming));
  CHECK(cudaEventCreateWithFlags(&second, cudaEventDisableTiming));
  // Queries that cost a round trip make later records of both events pushed.
  for (cudaEvent_t event : {first, second}) {
    spin<<<1, 1, 0, a>>>(ms);
    CHECK(cudaEventRecord(event, a));
    CHECK(poll(event));
  }
  for (int round = 0; round < 100; ++round) {
    spin<<<1, 1, 0, a>>>(ms);
    CHECK(cudaEventRecord(first, a));
    CHECK(cudaStreamWaitEvent(b, first, 0));
    CHECK(cudaStreamSynchronize(b));
    expect(cudaEventQuery(first), cudaSuccess, round,
           "event a synchronized stream waited for");

    spin<<<1, 1, 0, a>>>(ms / 10);
    CHECK(cudaEventRecord(first, a));
    expect(poll(first), cudaSuccess, round, "polled event");

    spin<<<1, 1, 0, a>>>(20 * ms);
    CHECK(cudaEventRecord(first, a));
    expect(cudaEventQuery(first), cudaErrorNotReady, round,
           "event recorded behind unfinished work");
    CHECK(cudaStreamSynchronize(a));
    expect(cudaEventQuery(first), cudaSuccess, round,
           "event on a synchronized stream");

    spin<<<1, 1, 0, a>>>(ms);
    CHECK(cudaEventRecord(first, a));
    CHECK(cudaEventRecord(second, a));
    expect(poll(second), cudaSuccess, round, "later event on the stream");
    expect(cudaEventQuery(first), cudaSuccess, round,
           "event covered by a later completed one");

    spin<<<1, 1, 0, a>>>(ms);
    CHECK(cudaEventRecord(first, a));
    CHECK(cudaStreamWaitEvent(b, first, 0));
    CHECK(cudaEventRecord(second, b));
    expect(poll(second), cudaSuccess, round, "event waiting on another");
    expect(cudaEventQuery(first), cudaSuccess, round,
           "event covered through a stream wait");
  }
  CHECK(cudaEventDestroy(first));
  CHECK(cudaEventDestroy(second));
  CHECK(cudaStreamDestroy(a));
  CHECK(cudaStreamDestroy(b));
  std::printf("pushed event completions agree with the device\n");
  return 0;
}
