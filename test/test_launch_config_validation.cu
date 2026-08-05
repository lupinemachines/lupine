// Exercises launch-configuration validation on the shim's fire-and-forget
// launch paths: every case below must fail with CUDA_ERROR_INVALID_VALUE.
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

static const char *error_name(CUresult result) {
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  fprintf(stderr, "%s failed: %s (%d)\n", operation, error_name(result),
          static_cast<int>(result));
  return false;
}

static CUfunction g_function = nullptr;
static void **g_params = nullptr;

static bool rejects(const char *name, unsigned int gridDimX,
                    unsigned int gridDimY, unsigned int gridDimZ,
                    unsigned int blockDimX, unsigned int blockDimY,
                    unsigned int blockDimZ) {
  CUresult result =
      cuLaunchKernel(g_function, gridDimX, gridDimY, gridDimZ, blockDimX,
                     blockDimY, blockDimZ, 0, nullptr, g_params, nullptr);
  if (result != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "%s: expected CUDA_ERROR_INVALID_VALUE, got %s (%d)\n",
            name, error_name(result), static_cast<int>(result));
    return false;
  }
  // The rejected launch must not leave a sticky error behind.
  return check(cuCtxSynchronize(), "sync after rejected launch");
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUdeviceptr output = 0;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuModuleLoadData(&module, kSetValuePtx), "cuModuleLoadData") ||
      !check(cuModuleGetFunction(&g_function, module, "set_value"),
             "cuModuleGetFunction") ||
      !check(cuMemAlloc(&output, sizeof(uint32_t)), "cuMemAlloc")) {
    return 1;
  }

  uint32_t zero = 0;
  uint32_t value = 42;
  void *params[] = {&output, &value};
  g_params = params;
  if (!check(cuMemcpyHtoD(output, &zero, sizeof(zero)), "reset output")) {
    return 1;
  }

  if (!rejects("gridDimX=0", 0, 1, 1, 1, 1, 1) ||
      !rejects("gridDimY=0", 1, 0, 1, 1, 1, 1) ||
      !rejects("blockDimZ=0", 1, 1, 1, 1, 1, 0) ||
      !rejects("blockDimX=4096", 1, 1, 1, 4096, 1, 1) ||
      !rejects("blockDimZ=65", 1, 1, 1, 1, 1, 65) ||
      !rejects("block 32x32x2 threads", 1, 1, 1, 32, 32, 2) ||
      !rejects("gridDimY=65536", 1, 65536, 1, 1, 1, 1)) {
    return 1;
  }

  uint32_t actual = UINT32_MAX;
  if (!check(cuMemcpyDtoH(&actual, output, sizeof(actual)), "cuMemcpyDtoH")) {
    return 1;
  }
  if (actual != 0) {
    fprintf(stderr, "a rejected launch reached the device: output=%u\n",
            actual);
    return 1;
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  CUlaunchConfig config = {};
  config.gridDimX = 0;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  CUresult result = cuLaunchKernelEx(&config, g_function, params, nullptr);
  if (result != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr,
            "cuLaunchKernelEx(gridDimX=0): expected CUDA_ERROR_INVALID_VALUE, "
            "got %s (%d)\n",
            error_name(result), static_cast<int>(result));
    return 1;
  }
  config.gridDimX = 1;
  config.blockDimX = 4096;
  result = cuLaunchKernelEx(&config, g_function, params, nullptr);
  if (result != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr,
            "cuLaunchKernelEx(blockDimX=4096): expected "
            "CUDA_ERROR_INVALID_VALUE, got %s (%d)\n",
            error_name(result), static_cast<int>(result));
    return 1;
  }
#endif

  if (!check(cuLaunchKernel(g_function, 1, 1, 1, 1, 1, 1, 0, nullptr, params,
                            nullptr),
             "valid cuLaunchKernel") ||
      !check(cuCtxSynchronize(), "sync valid launch") ||
      !check(cuMemcpyDtoH(&actual, output, sizeof(actual)), "cuMemcpyDtoH")) {
    return 1;
  }
  if (actual != value) {
    fprintf(stderr, "valid launch did not run: output=%u\n", actual);
    return 1;
  }

  cuMemFree(output);
  cuModuleUnload(module);
  cuDevicePrimaryCtxRelease(device);
  printf("launch config validation OK\n");
  return 0;
}
