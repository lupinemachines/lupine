// A host function or stream callback on one stream, ordered after a
// device-to-host copy into page-locked memory on another stream, must see the
// copied bytes. The order comes from an event the host function's stream waits
// on, possibly through a chain of waits, or from a host-side synchronize before
// the host function is launched.
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

struct Read {
  const unsigned int *slot;
  unsigned int seen;
};

static void CUDART_CB read_slot(void *opaque) {
  auto *read = static_cast<Read *>(opaque);
  read->seen = *read->slot;
}

static void CUDART_CB read_slot_callback(cudaStream_t, cudaError_t,
                                         void *opaque) {
  read_slot(opaque);
}

enum Order { kEvent, kEventCallback, kStreamSync, kDeviceSync };

static const char *order_name(Order order) {
  switch (order) {
  case kEvent:
    return "event wait";
  case kEventCallback:
    return "event wait, stream callback";
  case kStreamSync:
    return "stream synchronize";
  case kDeviceSync:
    return "device synchronize";
  }
  return "?";
}

// Tagged waits must stay bounded by where each wait sits in its stream:
//  - a host function queued before the wait must not take the copy, which then
//    still arrives at the copying stream's synchronize;
//  - a wait on an event recorded after another wait reaches the first copy;
//  - re-recording the event after the wait leaves out the later copy.
static int wait_position_cases(long long spin_cycles) {
  cudaStream_t a, b, c;
  CHECK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking));
  cudaEvent_t first, second;
  CHECK(cudaEventCreateWithFlags(&first, cudaEventDisableTiming));
  CHECK(cudaEventCreateWithFlags(&second, cudaEventDisableTiming));
  unsigned int *host = nullptr;
  CHECK(cudaMallocHost(&host, 2 * sizeof(unsigned int)));
  unsigned int *device = nullptr;
  CHECK(cudaMalloc(&device, 2 * sizeof(unsigned int)));

  int failures = 0;
  for (int round = 0; round < 10; ++round) {
    unsigned int values[2] = {0x3000u + 2 * round, 0x3001u + 2 * round};
    CHECK(cudaMemcpy(device, values, sizeof(values), cudaMemcpyHostToDevice));

    // A host function on b queued before b waits for a's copy.
    host[0] = 0;
    spin<<<1, 1, 0, a>>>(spin_cycles * 50);
    CHECK(cudaMemcpyAsync(host, device, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, a));
    CHECK(cudaEventRecord(first, a));
    spin<<<1, 1, 0, b>>>(spin_cycles * 5);
    Read early{host, 0};
    CHECK(cudaLaunchHostFunc(b, read_slot, &early));
    CHECK(cudaStreamWaitEvent(b, first, 0));
    CHECK(cudaStreamSynchronize(a));
    if (host[0] != values[0]) {
      std::fprintf(stderr,
                   "callback before the wait, round %d: copy left 0x%x, "
                   "expected 0x%x\n",
                   round, host[0], values[0]);
      ++failures;
    }
    CHECK(cudaDeviceSynchronize());

    // a -> first -> b, b records second, c waits on second.
    host[0] = 0;
    spin<<<1, 1, 0, a>>>(spin_cycles);
    CHECK(cudaMemcpyAsync(host, device, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, a));
    CHECK(cudaEventRecord(first, a));
    CHECK(cudaStreamWaitEvent(b, first, 0));
    CHECK(cudaEventRecord(second, b));
    CHECK(cudaStreamWaitEvent(c, second, 0));
    Read chained{host, 0};
    CHECK(cudaLaunchHostFunc(c, read_slot, &chained));
    CHECK(cudaStreamSynchronize(c));
    if (chained.seen != values[0]) {
      std::fprintf(stderr,
                   "chained waits, round %d: host function saw 0x%x, "
                   "expected 0x%x\n",
                   round, chained.seen, values[0]);
      ++failures;
    }
    CHECK(cudaDeviceSynchronize());

    // The event is recorded again behind a second copy after b's wait.
    host[0] = host[1] = 0;
    spin<<<1, 1, 0, a>>>(spin_cycles);
    CHECK(cudaMemcpyAsync(host, device, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, a));
    CHECK(cudaEventRecord(first, a));
    CHECK(cudaStreamWaitEvent(b, first, 0));
    spin<<<1, 1, 0, a>>>(spin_cycles * 50);
    CHECK(cudaMemcpyAsync(host + 1, device + 1, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, a));
    CHECK(cudaEventRecord(first, a));
    Read rerecorded{host, 0};
    CHECK(cudaLaunchHostFunc(b, read_slot, &rerecorded));
    CHECK(cudaStreamSynchronize(b));
    CHECK(cudaStreamSynchronize(a));
    if (rerecorded.seen != values[0] || host[1] != values[1]) {
      std::fprintf(stderr,
                   "re-recorded event, round %d: host function saw 0x%x, "
                   "expected 0x%x; second copy left 0x%x, expected 0x%x\n",
                   round, rerecorded.seen, values[0], host[1], values[1]);
      ++failures;
    }
    CHECK(cudaDeviceSynchronize());
  }

  CHECK(cudaFree(device));
  CHECK(cudaFreeHost(host));
  CHECK(cudaEventDestroy(second));
  CHECK(cudaEventDestroy(first));
  CHECK(cudaStreamDestroy(c));
  CHECK(cudaStreamDestroy(b));
  CHECK(cudaStreamDestroy(a));
  return failures;
}

int main() {
  CHECK(cudaSetDevice(0));
  cudaStream_t copy_stream, func_stream;
  CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&func_stream, cudaStreamNonBlocking));
  cudaEvent_t copied;
  CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  long long spin_cycles = static_cast<long long>(khz) / 5; // ~200 us

  unsigned int *host = nullptr;
  CHECK(cudaMallocHost(&host, sizeof(unsigned int)));
  unsigned int *device = nullptr;
  CHECK(cudaMalloc(&device, sizeof(unsigned int)));

  int failures = 0;
  unsigned int value = 0x1000u;
  for (Order order : {kEvent, kEventCallback, kStreamSync, kDeviceSync}) {
    for (int round = 0; round < 10; ++round) {
      ++value;
      *host = 0;
      CHECK(cudaMemcpy(device, &value, sizeof(value), cudaMemcpyHostToDevice));
      spin<<<1, 1, 0, copy_stream>>>(spin_cycles);
      CHECK(cudaGetLastError());
      CHECK(cudaMemcpyAsync(host, device, sizeof(unsigned int),
                            cudaMemcpyDeviceToHost, copy_stream));
      switch (order) {
      case kEvent:
      case kEventCallback:
        CHECK(cudaEventRecord(copied, copy_stream));
        CHECK(cudaStreamWaitEvent(func_stream, copied, 0));
        break;
      case kStreamSync:
        CHECK(cudaStreamSynchronize(copy_stream));
        break;
      case kDeviceSync:
        CHECK(cudaDeviceSynchronize());
        break;
      }
      Read read{host, 0};
      if (order == kEventCallback) {
        CHECK(cudaStreamAddCallback(func_stream, read_slot_callback, &read, 0));
      } else {
        CHECK(cudaLaunchHostFunc(func_stream, read_slot, &read));
      }
      CHECK(cudaStreamSynchronize(func_stream));
      CHECK(cudaStreamSynchronize(copy_stream));
      if (read.seen != value) {
        std::fprintf(stderr,
                     "%s, round %d: host function saw 0x%x, expected 0x%x\n",
                     order_name(order), round, read.seen, value);
        ++failures;
        break;
      }
    }
  }

  failures += wait_position_cases(spin_cycles);

  CHECK(cudaFree(device));
  CHECK(cudaFreeHost(host));
  CHECK(cudaEventDestroy(copied));
  CHECK(cudaStreamDestroy(func_stream));
  CHECK(cudaStreamDestroy(copy_stream));
  if (failures != 0) {
    std::printf("FAIL: a host function missed another stream's DtoH copy\n");
    return 1;
  }
  std::printf("PASS: host functions see DtoH copies ordered ahead of them on "
              "other streams\n");
  return 0;
}
