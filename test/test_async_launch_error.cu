// A launch the driver rejects must not be reported as a success that never
// ran. Natively the launch itself returns the error; through lupine launches
// are fire-and-forget, so the error surfaces at the next synchronize instead,
// exactly once.
#include <cuda.h>

#include <stdint.h>
#include <stdio.h>

static const char kSetValuePtx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry set_value(.param .u64 output, .param .u32 value)\n"
    "{\n"
    "  .reg .b64 %rd<2>;\n"
    "  .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [output];\n"
    "  ld.param.u32 %r1, [value];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

static const char *errstr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}
#define DRV(call)                                                              \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call, errstr(_r),        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));
  CUmodule module;
  DRV(cuModuleLoadData(&module, kSetValuePtx));
  CUfunction function;
  DRV(cuModuleGetFunction(&function, module, "set_value"));
  CUdeviceptr output;
  DRV(cuMemAlloc(&output, sizeof(uint32_t)));
  DRV(cuMemsetD32(output, 0, 1));

  uint32_t value = 0x1234u;
  void *params[] = {&output, &value};
  CUresult launch =
      cuLaunchKernel(function, 1, 1, 1, 2048, 1, 1, 0, nullptr, params, nullptr);
  CUresult sync = cuCtxSynchronize();
  fprintf(stderr, "launch=%d sync=%d\n", static_cast<int>(launch),
          static_cast<int>(sync));
  CHECK((launch == CUDA_ERROR_INVALID_VALUE && sync == CUDA_SUCCESS) ||
            (launch == CUDA_SUCCESS && sync == CUDA_ERROR_INVALID_VALUE),
        "an oversized block is reported at launch or at the next synchronize");
  CHECK(cuCtxSynchronize() == CUDA_SUCCESS, "the error is reported once");

  uint32_t host = UINT32_MAX;
  DRV(cuMemcpyDtoH(&host, output, sizeof(host)));
  CHECK(host == 0, "the rejected launch did not run");

  value = 0x5678u;
  DRV(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr));
  DRV(cuCtxSynchronize());
  DRV(cuMemcpyDtoH(&host, output, sizeof(host)));
  CHECK(host == 0x5678u, "a valid launch afterwards runs");

  DRV(cuMemFree(output));
  DRV(cuModuleUnload(module));
  DRV(cuDevicePrimaryCtxRelease(dev));

  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
