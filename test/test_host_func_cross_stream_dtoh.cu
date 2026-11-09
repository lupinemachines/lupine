// A host function or stream callback on one stream, ordered after a
// device-to-host copy into page-locked memory on another stream, must see the
// copied bytes. The order comes from an event the host function's stream waits
// on, or from a host-side synchronize before the host function is launched.
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
