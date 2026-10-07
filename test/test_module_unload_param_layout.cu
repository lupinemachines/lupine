// Unloading a module must drop the cached parameter layouts of its functions
// only: a later module may be handed the same function handle with a different
// layout, while functions of modules still loaded keep launching correctly.
// Auto-discovered by run_custom_tests.sh.
#include <cuda.h>

#include <cstdint>
#include <cstdio>

// Same entry name, different parameter counts, and only the wide module has a
// second entry.
static const char kWidePtx[] =
    ".version 6.4\n.target sm_52\n.address_size 64\n"
    ".visible .entry k(.param .u64 out, .param .u32 x, .param .u32 y)\n{\n"
    "  .reg .b64 %rd<3>;\n  .reg .b32 %r<4>;\n"
    "  ld.param.u64 %rd1, [out];\n  ld.param.u32 %r1, [x];\n"
    "  ld.param.u32 %r2, [y];\n  add.u32 %r3, %r1, %r2;\n"
    "  cvta.to.global.u64 %rd2, %rd1;\n  st.global.u32 [%rd2], %r3;\n"
    "  ret;\n}\n"
    ".visible .entry wide_only()\n{\n  ret;\n}\n";

static const char kNarrowPtx[] =
    ".version 6.4\n.target sm_52\n.address_size 64\n"
    ".visible .entry k(.param .u64 out, .param .u32 x)\n{\n"
    "  .reg .b64 %rd<3>;\n  .reg .b32 %r<2>;\n"
    "  ld.param.u64 %rd1, [out];\n  ld.param.u32 %r1, [x];\n"
    "  cvta.to.global.u64 %rd2, %rd1;\n  st.global.u32 [%rd2], %r1;\n"
    "  ret;\n}\n";

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "%s failed: %s (%d)\n", operation,
               name == nullptr ? "unknown" : name, static_cast<int>(result));
  return false;
}

static bool launch_and_check(CUfunction function, CUdeviceptr out,
                             uint32_t value, const char *label) {
  uint32_t zero = 0;
  void *args[] = {&out, &value, &zero};
  uint32_t observed = 0;
  if (!check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args,
                            nullptr),
             "cuLaunchKernel") ||
      !check(cuCtxSynchronize(), "cuCtxSynchronize") ||
      !check(cuMemcpyDtoH(&observed, out, sizeof(observed)), "cuMemcpyDtoH")) {
    return false;
  }
  if (observed != value) {
    std::fprintf(stderr, "%s wrote %u, expected %u\n", label, observed, value);
    return false;
  }
  return true;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUdeviceptr out = 0;
  CUmodule resident = nullptr;
  CUfunction resident_function = nullptr;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuMemAlloc(&out, sizeof(uint32_t)), "cuMemAlloc") ||
      !check(cuModuleLoadData(&resident, kNarrowPtx), "cuModuleLoadData") ||
      !check(cuModuleGetFunction(&resident_function, resident, "k"),
             "cuModuleGetFunction") ||
      !launch_and_check(resident_function, out, 1, "resident")) {
    return 1;
  }

  for (uint32_t i = 0; i < 16; ++i) {
    bool wide = (i % 2) == 0;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUfunction wide_only = nullptr;
    if (!check(cuModuleLoadData(&module, wide ? kWidePtx : kNarrowPtx),
               "cuModuleLoadData") ||
        !check(cuModuleGetFunction(&function, module, "k"),
               "cuModuleGetFunction") ||
        !launch_and_check(function, out, 100 + i, "transient")) {
      return 1;
    }
#if CUDA_VERSION >= 12040
    size_t offset = 0;
    size_t size = 0;
    CUresult third = cuFuncGetParamInfo(function, 2, &offset, &size);
    if (third != (wide ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE)) {
      std::fprintf(stderr, "third parameter of %s k returned %d\n",
                   wide ? "wide" : "narrow", static_cast<int>(third));
      return 1;
    }
#endif
    CUresult lookup = cuModuleGetFunction(&wide_only, module, "wide_only");
    if (lookup != (wide ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND)) {
      std::fprintf(stderr, "wide_only lookup in %s module returned %d\n",
                   wide ? "wide" : "narrow", static_cast<int>(lookup));
      return 1;
    }
    if (!check(cuModuleUnload(module), "cuModuleUnload") ||
        !launch_and_check(resident_function, out, 200 + i, "resident")) {
      return 1;
    }
  }

  cuModuleUnload(resident);
  cuMemFree(out);
  cuDevicePrimaryCtxRelease(device);
  std::printf("PASS\n");
  return 0;
}
