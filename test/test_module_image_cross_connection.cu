// LUPINE_TEST_CONNECTIONS 2
// Module functions belong to their creating context. Cross-connection launches
// must reject them, while context-independent library kernels remain usable on
// both devices. Auto-discovered by run_custom_tests.sh.
#include <cuda.h>

#include <cstdint>
#include <cstdio>

static const char kWritePtxFormat[] =
    ".version 6.4\n.target sm_52\n.address_size 64\n"
    ".visible .global .align 4 .u32 g;\n"
    ".visible .entry k(.param .u64 out)\n{\n"
    "  .reg .b64 %%rd<3>;\n  .reg .b32 %%r<2>;\n"
    "  ld.param.u64 %%rd1, [out];\n  cvta.to.global.u64 %%rd2, %%rd1;\n"
    "  mov.u32 %%r1, %u;\n  st.global.u32 [%%rd2], %%r1;\n"
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

// Each buffer lives only for its own launch: two servers can hand out the same
// device address (the CUDA 13 forward-compatibility driver starts every
// process at one fixed VA), and the client keys pointer ownership by address.
static bool launch_and_check(CUfunction function, CUcontext context,
                             uint32_t expected, const char *label) {
  CUdeviceptr out = 0;
  void *args[] = {&out};
  uint32_t observed = 0;
  if (!check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuMemAlloc(&out, sizeof(observed)), "cuMemAlloc") ||
      !check(cuMemsetD32(out, 0, 1), "cuMemsetD32") ||
      !check(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args,
                            nullptr),
             "cuLaunchKernel") ||
      !check(cuCtxSynchronize(), "cuCtxSynchronize") ||
      !check(cuMemcpyDtoH(&observed, out, sizeof(observed)), "cuMemcpyDtoH") ||
      !check(cuMemFree(out), "cuMemFree")) {
    return false;
  }
  if (observed != expected) {
    std::fprintf(stderr, "%s wrote %u, expected %u\n", label, observed,
                 expected);
    return false;
  }
  return true;
}

int main() {
  int count = 0;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGetCount(&count), "cuDeviceGetCount")) {
    return 1;
  }
  if (count < 2) {
    std::printf("SKIP: needs two devices\n");
    return 0;
  }
  CUdevice devices[2] = {};
  CUcontext contexts[2] = {};
  for (int i = 0; i < 2; ++i) {
    if (!check(cuDeviceGet(&devices[i], i), "cuDeviceGet") ||
        !check(cuDevicePrimaryCtxRetain(&contexts[i], devices[i]),
               "cuDevicePrimaryCtxRetain")) {
      return 1;
    }
  }

  char ptx[sizeof(kWritePtxFormat) + 16];
  for (uint32_t value = 1; value <= 64; ++value) {
    std::snprintf(ptx, sizeof(ptx), kWritePtxFormat, value);
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (!check(cuCtxSetCurrent(contexts[0]), "cuCtxSetCurrent") ||
        !check(cuModuleLoadData(&module, ptx), "cuModuleLoadData") ||
        !check(cuModuleGetFunction(&function, module, "k"),
               "cuModuleGetFunction") ||
        !launch_and_check(function, contexts[0], value, "module") ||
        !check(cuCtxSetCurrent(contexts[1]), "cuCtxSetCurrent")) {
      return 1;
    }
    CUdeviceptr out = 0;
    void *args[] = {&out};
    CUresult result =
        cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
    if (result != CUDA_ERROR_INVALID_HANDLE) {
      std::fprintf(stderr,
                   "cross-context module launch returned %d, expected %d\n",
                   static_cast<int>(result), CUDA_ERROR_INVALID_HANDLE);
      return 1;
    }
    if (!launch_and_check(function, contexts[0], value,
                          "module after rejection") ||
        !check(cuModuleUnload(module), "cuModuleUnload")) {
      return 1;
    }
  }

#if CUDA_VERSION >= 12000
  std::snprintf(ptx, sizeof(ptx), kWritePtxFormat, 100u);
  CUlibrary library = nullptr;
  CUkernel kernel = nullptr;
  if (!check(cuLibraryLoadData(&library, ptx, nullptr, nullptr, 0, nullptr,
                               nullptr, 0),
             "cuLibraryLoadData") ||
      !check(cuLibraryGetKernel(&kernel, library, "k"), "cuLibraryGetKernel") ||
      !launch_and_check(reinterpret_cast<CUfunction>(kernel), contexts[0],
                        100, "library") ||
      !launch_and_check(reinterpret_cast<CUfunction>(kernel), contexts[1],
                        100, "library on the second connection")) {
    return 1;
  }
  // Device-scoped kernel queries and settings act on the kernel as loaded for
  // that device's connection.
  for (int i = 0; i < 2; ++i) {
    int threads = 0;
    int shared = 0;
    if (!check(cuKernelSetAttribute(
                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024 * (i + 1),
                   kernel, devices[i]),
               "cuKernelSetAttribute") ||
        !check(cuKernelSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_SHARED,
                                      devices[i]),
               "cuKernelSetCacheConfig") ||
        !check(cuKernelGetAttribute(&threads,
                                    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    kernel, devices[i]),
               "cuKernelGetAttribute(MAX_THREADS_PER_BLOCK)") ||
        !check(cuKernelGetAttribute(
                   &shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                   kernel, devices[i]),
               "cuKernelGetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)")) {
      return 1;
    }
    if (threads <= 0 || shared != 1024 * (i + 1)) {
      std::fprintf(stderr, "device %d kernel attributes %d threads, %d shared\n",
                   i, threads, shared);
      return 1;
    }
  }
  // Modules and globals belong to a context, so they come from the library as
  // loaded on the current context's connection.
  for (int i = 0; i < 2; ++i) {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUdeviceptr global = 0;
    size_t bytes = 0;
    uint32_t observed = 0;
    void *args[] = {&global};
    if (!check(cuCtxSetCurrent(contexts[i]), "cuCtxSetCurrent") ||
        !check(cuLibraryGetModule(&module, library), "cuLibraryGetModule") ||
        !check(cuModuleGetFunction(&function, module, "k"),
               "cuModuleGetFunction(library module)") ||
        !launch_and_check(function, contexts[i], 100, "library module") ||
        !check(cuLibraryGetGlobal(&global, &bytes, library, "g"),
               "cuLibraryGetGlobal") ||
        !check(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1,
                              1, 1, 0, nullptr, args, nullptr),
               "cuLaunchKernel(global)") ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        !check(cuMemcpyDtoH(&observed, global, sizeof(observed)),
               "cuMemcpyDtoH(global)")) {
      return 1;
    }
    if (observed != 100) {
      std::fprintf(stderr, "device %d library global holds %u\n", i, observed);
      return 1;
    }
  }
  if (!check(cuLibraryUnload(library), "cuLibraryUnload")) {
    return 1;
  }
#endif

  for (int i = 0; i < 2; ++i) {
    cuDevicePrimaryCtxRelease(devices[i]);
  }
  std::printf("PASS\n");
  return 0;
}
