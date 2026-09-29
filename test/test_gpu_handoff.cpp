#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define CUDA(call)                                                             \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      std::fprintf(stderr, "%s: CUDA %d\n", #call, status);                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char program[] =
    ".version 7.0\n.target sm_50\n.address_size 64\n"
    ".visible .entry increment(.param .u64 root) {"
    ".reg .u64 p,q; .reg .u32 v; ld.param.u64 p,[root];"
    "ld.global.u64 q,[p]; ld.global.u32 v,[q]; add.u32 v,v,1;"
    "st.global.u32 [q],v; ret; }";

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  CUDA(cuInit(0));
  // Remote devices follow local devices in the client's ordinal space.
  int count = 0;
  CUdevice device = 0;
  CUDA(cuDeviceGetCount(&count));
  CUDA(cuDeviceGet(&device, count - 1));
  CUcontext context = nullptr;
  if (std::getenv("LUPINE_TEST_EXPLICIT_CONTEXT")) {
#if CUDA_VERSION >= 13000
    CUDA(cuCtxCreate(&context, nullptr, 0, device));
#else
    CUDA(cuCtxCreate(&context, 0, device));
#endif
  } else {
    CUDA(cuDevicePrimaryCtxRetain(&context, device));
  }
  CUDA(cuCtxSetCurrent(context));
  CUdeviceptr pointer = 0;
  CUDA(cuMemAlloc(&pointer, 4096));
  CUdeviceptr embedded = 0;
  CUDA(cuMemAlloc(&embedded, 4096));
  uint32_t value = 42, deferred = 0;
  CUDA(cuMemcpyHtoD(pointer, &embedded, sizeof(embedded)));
  CUDA(cuMemcpyHtoD(embedded, &value, sizeof(value)));
  CUmodule module = nullptr;
  CUDA(cuModuleLoadDataEx(&module, program, 0, nullptr, nullptr));
  CUfunction function = nullptr;
  CUDA(cuModuleGetFunction(&function, module, "increment"));
  CUstream stream = nullptr;
  CUevent event = nullptr;
  CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  bool reject = std::getenv("LUPINE_TEST_REJECT") != nullptr;
  CUDA(cuEventCreate(&event, reject ? 0 : CU_EVENT_DISABLE_TIMING));
  CUdeviceptr fresh = 0;
  for (int cycle = 0; cycle < 3; ++cycle) {
    void *arguments[] = {&pointer};
    CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, arguments,
                        nullptr));
    ++value;
    CUDA(cuMemcpyDtoHAsync(&deferred, embedded, sizeof(deferred), stream));
    CUDA(cuEventRecord(event, stream));
    std::printf("READY %d\n", cycle);
    if (std::getchar() == EOF)
      return 1;
    // The supervisor lets this call start before the replacement server. The
    // driver must block it and continue with the same handles after restore.
    if (reject) {
      if (cuEventSynchronize(event) == CUDA_SUCCESS)
        return 8;
      std::puts("REJECTED");
      return 0;
    }
    CUDA(cuEventSynchronize(event));
    if (deferred != value)
      return 2;
    CUcontext current = nullptr;
    CUDA(cuCtxGetCurrent(&current));
    if (current != context)
      return 3;
    CUpointer_attribute attribute = CU_POINTER_ATTRIBUTE_CONTEXT;
    void *output = &current;
    CUDA(cuPointerGetAttributes(1, &attribute, &output, pointer));
    if (current != context)
      return 7;
    CUdeviceptr base = 0;
    size_t bytes = 0;
    CUDA(cuMemGetAddressRange(&base, &bytes, pointer + 16));
    if (base != pointer || bytes != 4096)
      return 9;
    CUpointer_attribute bounds[] = {CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                    CU_POINTER_ATTRIBUTE_RANGE_SIZE};
    void *bound_outputs[] = {&base, &bytes};
    CUDA(cuPointerGetAttributes(2, bounds, bound_outputs, embedded + 16));
    if (base != embedded || bytes != 4096)
      return 10;
    CUdeviceptr restored_embedded = 0;
    CUDA(cuMemcpyDtoH(&restored_embedded, pointer, sizeof(restored_embedded)));
    if (restored_embedded != embedded) {
      std::fprintf(stderr, "embedded pointer: got 0x%llx expected 0x%llx\n",
                   (unsigned long long)restored_embedded,
                   (unsigned long long)embedded);
      return 4;
    }
    uint32_t restored = 0;
    CUDA(cuMemcpyDtoH(&restored, embedded, sizeof(restored)));
    if (restored != value)
      return 5;
    CUDA(cuMemsetD32(embedded, value, 1));
    CUDA(cuMemsetD2D32(embedded, 16, value, 1, 1));
    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = embedded;
    copy.srcPitch = sizeof(value);
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = &restored;
    copy.dstPitch = sizeof(restored);
    copy.WidthInBytes = sizeof(value);
    copy.Height = 1;
    CUDA(cuMemcpy2D(&copy));
    if (restored != value)
      return 12;
    if (fresh) {
      CUDA(cuMemcpyDtoH(&restored, fresh, sizeof(restored)));
      if (restored != value - 1)
        return 13;
      CUDA(cuMemFree(fresh));
    }
    CUDA(cuMemAlloc(&fresh, 4096));
    CUDA(cuMemcpyDtoD(fresh, embedded, sizeof(value)));
    CUDA(cuMemcpyDtoH(&restored, fresh, sizeof(restored)));
    if (restored != value)
      return 6;
    std::printf("RESTORED %d %u\n", cycle, restored);
  }
  CUDA(cuMemFree(fresh));
  CUDA(cuMemFree(pointer));
  CUDA(cuMemcpyDtoH(&value, embedded, sizeof(value)));
  if (value != 45)
    return 11;
  CUDA(cuMemFree(embedded));
  CUDA(cuEventDestroy(event));
  CUDA(cuStreamDestroy(stream));
  CUDA(cuModuleUnload(module));
  return 0;
}
