#include <cuda.h>

#include <cstdio>
#include <cstring>

#if CUDA_VERSION < 12040
int main() {
  std::puts("SKIP: library metadata requires CUDA 12.4");
  return 0;
}
#else
static const char image[] =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry first(.param .u64 ptr) { ret; }\n"
    ".visible .entry second(.param .u32 value) { ret; }\n";

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s: CUDA error %d\n", operation, int(result));
  return false;
}

#define CHECK(call)                                                            \
  do {                                                                         \
    if (!check((call), #call))                                                 \
      return 1;                                                                \
  } while (false)

int main() {
  CHECK(cuInit(0));
  CUdevice device;
  CUcontext context;
  CHECK(cuDeviceGet(&device, 0));
  CHECK(cuDevicePrimaryCtxRetain(&context, device));
  for (int iteration = 0; iteration < 4; ++iteration) {
    // Library loading is context independent. Missing context-dependent
    // metadata must not turn a successful load into a failure.
    CHECK(cuCtxSetCurrent(iteration % 2 == 0 ? context : nullptr));
    CUlibrary library = nullptr;
    char info[1024] = {};
    CUjit_option options[] = {CU_JIT_INFO_LOG_BUFFER,
                              CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
    void *values[] = {info, reinterpret_cast<void *>(sizeof(info))};
    CHECK(cuLibraryLoadData(&library, image, options, values, 2, nullptr,
                            nullptr, 0));
    CHECK(cuCtxSetCurrent(context));
    const char *names[] = {"first", "second"};
    for (int index = 0; index < 2; ++index) {
      CUkernel kernel;
      CUfunction function;
      CHECK(cuLibraryGetKernel(&kernel, library, names[index]));
      CHECK(cuKernelGetFunction(&function, kernel));
      const char *name = nullptr;
      CHECK(cuKernelGetName(&name, kernel));
      if (std::strcmp(name, names[index]) != 0)
        return 1;
      size_t offset = 99, size = 99;
      CHECK(cuFuncGetParamInfo(function, 0, &offset, &size));
      if (offset != 0 || size != (index == 0 ? 8u : 4u)) {
        std::fprintf(stderr, "parameter %d: offset=%zu size=%zu\n", index,
                     offset, size);
        return 1;
      }
      int function_threads = 0, kernel_threads = 0;
      CHECK(cuFuncGetAttribute(&function_threads,
                               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               function));
      CHECK(cuKernelGetAttribute(&kernel_threads,
                                 CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                 kernel, device));
      if (function_threads <= 0 || function_threads != kernel_threads) {
        std::fprintf(stderr, "thread attributes: function=%d kernel=%d\n",
                     function_threads, kernel_threads);
        return 1;
      }
      unsigned long long argument = 0;
      void *arguments[] = {&argument};
      CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments,
                           nullptr));
    }
    CHECK(cuCtxSynchronize());
    CHECK(cuLibraryUnload(library));
  }
  CUlibrary empty = nullptr;
  const char empty_image[] = ".version 7.0\n.target sm_70\n.address_size 64\n"
                             ".visible .global .u32 counter;\n";
  CHECK(cuLibraryLoadData(&empty, empty_image, nullptr, nullptr, 0, nullptr,
                          nullptr, 0));
  CHECK(cuLibraryUnload(empty));
  // Invalid PTX can be accepted under lazy loading; invalid JIT options must
  // instead fail at the API boundary.
  CUjit_option invalid_option = static_cast<CUjit_option>(0x7fffffff);
  void *invalid_value = nullptr;
  if (cuLibraryLoadData(&empty, image, &invalid_option, &invalid_value, 1,
                        nullptr, nullptr, 0) == CUDA_SUCCESS) {
    std::fprintf(stderr, "invalid JIT option unexpectedly accepted\n");
    return 1;
  }
  // A failed load must leave the response stream aligned.
  CHECK(cuLibraryLoadData(&empty, image, nullptr, nullptr, 0, nullptr, nullptr,
                          0));
  CHECK(cuLibraryUnload(empty));
  CHECK(cuDevicePrimaryCtxRelease(device));
  std::puts("library load metadata, JIT outputs, and fallback queries passed");
  return 0;
}
#endif
