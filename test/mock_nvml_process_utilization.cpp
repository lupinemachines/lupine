// Only the native driver is mocked. The regression test uses the production
// NVML client, server handlers, device routing and HTTP/2 transport.
#include <cstdint>
#include <cstdlib>
#include <nvml.h>

static unsigned int identity() {
  const char *value = std::getenv("MOCK_NVML_ID");
  return value ? static_cast<unsigned int>(std::atoi(value)) : 0;
}
static nvmlDevice_t handle() {
  return reinterpret_cast<nvmlDevice_t>(uintptr_t{0x1000} + identity());
}
extern "C" nvmlReturn_t nvmlInit_v2() { return NVML_SUCCESS; }
extern "C" nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }
extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
  *count = 1;
  return NVML_SUCCESS;
}
extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                                      nvmlDevice_t *device) {
  if (index != 0 || !device)
    return NVML_ERROR_INVALID_ARGUMENT;
  *device = handle();
  return NVML_SUCCESS;
}
#ifndef MOCK_NVML_OMIT_UTILIZATION
extern "C" nvmlReturn_t nvmlDeviceGetProcessUtilization(
    nvmlDevice_t device, nvmlProcessUtilizationSample_t *samples,
    unsigned int *count, unsigned long long timestamp) {
  if (device != handle() || !count)
    return NVML_ERROR_INVALID_ARGUMENT;
  if (timestamp == 10)
    return NVML_ERROR_NOT_SUPPORTED;
  if (timestamp == 20)
    return NVML_ERROR_GPU_IS_LOST;
  if (timestamp == 30) {
    *count =
        0xffffffffU; // A broken driver must not cause an out-of-bounds copy.
    return NVML_SUCCESS;
  }
  if (timestamp == 40) {
    *count = 4; // Simulate growth since a previous count query.
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  unsigned int required = (timestamp < 100) + (timestamp < 200);
  if (!required) {
    *count = 0;
    return NVML_ERROR_NOT_FOUND;
  }
  if (!samples || *count < required) {
    *count = required;
    return NVML_ERROR_INSUFFICIENT_SIZE;
  }
  unsigned int written = 0;
  for (unsigned int i = 0; i < 2; ++i) {
    const unsigned long long sample_time = (i + 1) * 100;
    if (sample_time > timestamp) {
      auto &sample = samples[written++];
      sample = {};
      sample.pid = 42000 + identity() * 100 + i;
      sample.timeStamp = sample_time;
      sample.smUtil = 50 + i;
      sample.memUtil = 20 + i;
      sample.encUtil = 3;
      sample.decUtil = 4;
    }
  }
  *count = written;
  return NVML_SUCCESS;
}
#endif
