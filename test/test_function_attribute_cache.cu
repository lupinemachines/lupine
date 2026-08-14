// Exercises the client-side function attribute query cache and successful
// kernel attribute set deduplication. Auto-discovered by run_custom_tests.sh.
#include <cuda.h>

#include <cstdio>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000
int main() {
  std::printf("SKIP: CUkernel attributes require CUDA 12.0 or newer\n");
  return 0;
}
#else

static const char kNoopPtx[] = ".version 6.4\n"
                               ".target sm_52\n"
                               ".address_size 64\n"
                               ".visible .entry noop()\n"
                               "{\n"
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
  std::fprintf(stderr, "%s failed: %s (%d)\n", operation, error_name(result),
               static_cast<int>(result));
  return false;
}

static bool expect(CUresult result, CUresult expected, const char *operation) {
  if (result == expected) {
    return true;
  }
  std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", operation,
               error_name(result), static_cast<int>(result),
               error_name(expected), static_cast<int>(expected));
  return false;
}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUlibrary library = nullptr;
  CUkernel kernel = nullptr;
  CUfunction function = nullptr;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuLibraryLoadData(&library, kNoopPtx, nullptr, nullptr, 0, nullptr,
                               nullptr, 0),
             "cuLibraryLoadData") ||
      !check(cuLibraryGetKernel(&kernel, library, "noop"),
             "cuLibraryGetKernel") ||
      !check(cuKernelGetFunction(&function, kernel), "cuKernelGetFunction")) {
    return 1;
  }

  int first_threads = 0;
  int second_threads = 0;
  if (!check(cuFuncGetAttribute(&first_threads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                function),
             "cuFuncGetAttribute(first)") ||
      !check(cuFuncGetAttribute(&second_threads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                function),
             "cuFuncGetAttribute(cached)") ||
      first_threads <= 0 || first_threads != second_threads) {
    std::fprintf(stderr, "function attribute values differ: %d then %d\n",
                 first_threads, second_threads);
    return 1;
  }

  auto invalid_attribute = static_cast<CUfunction_attribute>(9999);
  int ignored = 0;
  if (!expect(cuFuncGetAttribute(&ignored, invalid_attribute, function),
              CUDA_ERROR_INVALID_VALUE, "cuFuncGetAttribute(invalid first)") ||
      !expect(cuFuncGetAttribute(&ignored, invalid_attribute, function),
              CUDA_ERROR_INVALID_VALUE,
              "cuFuncGetAttribute(invalid repeated)")) {
    return 1;
  }

  if (!check(
          cuFuncSetAttribute(
              function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 25),
          "cuFuncSetAttribute(25)")) {
    return 1;
  }
  int carveout = -1;
  if (!check(cuFuncGetAttribute(
                 &carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                 function),
             "cuFuncGetAttribute(carveout 25)") ||
      carveout != 25 ||
      !check(
          cuFuncSetAttribute(
              function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 50),
          "cuFuncSetAttribute(50)")) {
    return 1;
  }
  if (!check(cuFuncGetAttribute(
                 &carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                 function),
             "cuFuncGetAttribute(carveout 50)") ||
      carveout != 50) {
    std::fprintf(stderr, "stale function attribute value: %d, expected 50\n",
                 carveout);
    return 1;
  }

  if (!check(
          cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               0, kernel, device),
          "cuKernelSetAttribute(first)") ||
      !check(
          cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               0, kernel, device),
          "cuKernelSetAttribute(deduplicated)")) {
    return 1;
  }

  int max_threads = 0;
  if (!check(cuKernelGetAttribute(&max_threads,
                                  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  kernel, device),
             "cuKernelGetAttribute(MAX_THREADS_PER_BLOCK)") ||
      !expect(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                   max_threads, kernel, device),
              CUDA_ERROR_INVALID_VALUE,
              "cuKernelSetAttribute(read-only first)") ||
      !expect(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                   max_threads, kernel, device),
              CUDA_ERROR_INVALID_VALUE,
              "cuKernelSetAttribute(read-only repeated)")) {
    return 1;
  }

  if (!check(cuLibraryUnload(library), "cuLibraryUnload") ||
      !check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }
  std::printf("function attribute cache preserves values, invalidation, and "
              "setter errors\n");
  return 0;
}
#endif
