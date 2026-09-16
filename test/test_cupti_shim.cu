// Exercises the CUPTI shim against the remote device: the queries it answers
// without a GPU (version, result strings, the calling thread's last error) and
// the refusal every measuring entry point returns, taken through the header's
// own prototypes so the calls are the ones a profiler makes. The refusals are
// interleaved with real work on the remote device, because a profiler that
// cannot measure must still leave the program it failed to measure running.
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_events.h>
#include <cupti_profiler_target.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT_REFUSED(call)                                                   \
  do {                                                                         \
    CUptiResult status = (call);                                               \
    if (status != CUPTI_ERROR_NOT_SUPPORTED) {                                 \
      fprintf(stderr, "%s returned %d, expected CUPTI_ERROR_NOT_SUPPORTED\n",  \
              #call, static_cast<int>(status));                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

constexpr int kCount = 4096;

__global__ void scale(int *values, int factor) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kCount) {
    values[index] *= factor;
  }
}

void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size,
                             size_t *max_num_records) {
  *buffer = nullptr;
  *size = 0;
  *max_num_records = 0;
}

void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream_id,
                              uint8_t *buffer, size_t size, size_t valid_size) {}

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId id, const void *data) {}

} // namespace

int main() {
  uint32_t version = 0;
  EXPECT(cuptiGetVersion(&version) == CUPTI_SUCCESS);
  EXPECT(version == CUPTI_API_VERSION);
  printf("cupti %u\n", version);

  const char *name = nullptr;
  EXPECT(cuptiGetResultString(CUPTI_ERROR_NOT_SUPPORTED, &name) ==
         CUPTI_SUCCESS);
  EXPECT(strcmp(name, "CUPTI_ERROR_NOT_SUPPORTED") == 0);
#if CUPTI_API_VERSION >= 28
  // CUDA 11's CUPTI (API version 18) declares no cuptiGetErrorMessage to
  // call; 28 is CUDA 12.9's, the oldest header this was checked against. The
  // shim defines one either way, for the consumers whose headers have it.
  EXPECT(cuptiGetErrorMessage(CUPTI_SUCCESS, &name) == CUPTI_SUCCESS);
  EXPECT(strcmp(name, "CUPTI_SUCCESS") == 0);
#endif
  // A code the library has no name for still leaves the caller a string to
  // print, which is what NVIDIA's does and what a logging profiler needs.
  EXPECT(cuptiGetResultString(static_cast<CUptiResult>(4242), &name) ==
         CUPTI_ERROR_INVALID_PARAMETER);
  EXPECT(strcmp(name, "<unknown>") == 0);
  printf("strings: passed\n");

  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  EXPECT(devices > 0);
  CHECK_CUDA(cudaSetDevice(0));

  // The activity API: the callbacks a profiler would register, the kinds it
  // would enable, and the flush it would end with.
  EXPECT_REFUSED(cuptiActivityRegisterCallbacks(request_buffer,
                                                complete_buffer));
  EXPECT_REFUSED(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  EXPECT_REFUSED(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  EXPECT_REFUSED(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  EXPECT_REFUSED(cuptiActivityPushExternalCorrelationId(
      CUPTI_EXTERNAL_CORRELATION_KIND_UNKNOWN, 1));

  // The work the profiler failed to subscribe to runs regardless, and its
  // results are the ones the kernel computed.
  int *device_values = nullptr;
  CHECK_CUDA(cudaMalloc(&device_values, kCount * sizeof(int)));
  std::vector<int> values(kCount);
  for (int i = 0; i < kCount; ++i) {
    values[i] = i;
  }
  CHECK_CUDA(cudaMemcpy(device_values, values.data(), kCount * sizeof(int),
                        cudaMemcpyHostToDevice));
  scale<<<(kCount + 255) / 256, 256>>>(device_values, 3);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(values.data(), device_values, kCount * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(device_values));
  for (int i = 0; i < kCount; ++i) {
    EXPECT(values[i] == i * 3);
  }
  printf("kernel: passed\n");

  uint64_t timestamp = 0;
  EXPECT_REFUSED(cuptiGetTimestamp(&timestamp));
  size_t dropped = 0;
  EXPECT_REFUSED(cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped));
  EXPECT_REFUSED(cuptiActivityFlushAll(1));

  // The callback API, whose subscriber would have to be called from inside the
  // server's own driver calls.
  CUpti_SubscriberHandle subscriber = nullptr;
  EXPECT_REFUSED(cuptiSubscribe(&subscriber, callback, nullptr));
  EXPECT_REFUSED(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  EXPECT_REFUSED(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                                     CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  EXPECT_REFUSED(cuptiUnsubscribe(subscriber));

  // The counter APIs, which would reprogram hardware a server shares.
  CUpti_Profiler_Initialize_Params initialize_params = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  EXPECT_REFUSED(cuptiProfilerInitialize(&initialize_params));
  uint32_t domains = 0;
  EXPECT_REFUSED(cuptiDeviceGetNumEventDomains(0, &domains));

  EXPECT_REFUSED(cuptiFinalize());

  // Every refusal above was recorded for this thread, and reading it clears
  // it.
  EXPECT(cuptiGetLastError() == CUPTI_ERROR_NOT_SUPPORTED);
  EXPECT(cuptiGetLastError() == CUPTI_SUCCESS);
  printf("refusals: passed\n");

  printf("cupti shim: all checks passed\n");
  return 0;
}
