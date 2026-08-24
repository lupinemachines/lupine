// Verifies that the library snapshot carries the corresponding CUfunction
// handle and that both handles expose the same parameter layout.
#include <cuda.h>

#include <cstdio>
#include <cstring>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
int main() {
  std::printf("SKIP: library kernel enumeration requires CUDA 12.4 or newer\n");
  return 0;
}
#else

static const char kParameterizedPtx[] = ".version 6.4\n"
                                        ".target sm_52\n"
                                        ".address_size 64\n"
                                        ".visible .entry parameterized(\n"
                                        "  .param .u64 parameterized_param_0,\n"
                                        "  .param .u32 parameterized_param_1\n"
                                        ")\n"
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

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUlibrary library = nullptr;
  CUmodule library_module = nullptr;
  CUkernel kernel = nullptr;
  CUfunction function = nullptr;
  CUfunction module_function = nullptr;
  CUfunction library_module_function = nullptr;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain") ||
      !check(cuCtxSetCurrent(context), "cuCtxSetCurrent") ||
      !check(cuLibraryLoadData(&library, kParameterizedPtx, nullptr, nullptr, 0,
                               nullptr, nullptr, 0),
             "cuLibraryLoadData") ||
      !check(cuLibraryGetKernel(&kernel, library, "parameterized"),
             "cuLibraryGetKernel") ||
      !check(cuKernelGetFunction(&function, kernel), "cuKernelGetFunction") ||
      !check(cuLibraryGetModule(&library_module, library),
             "cuLibraryGetModule") ||
      !check(cuModuleGetFunction(&library_module_function, library_module,
                                 "parameterized"),
             "cuModuleGetFunction(library)")) {
    return 1;
  }

  const char *kernel_name = nullptr;
  const char *function_name = nullptr;
  const char *library_module_function_name = nullptr;
  if (!check(cuKernelGetName(&kernel_name, kernel), "cuKernelGetName") ||
      !check(cuFuncGetName(&function_name, function), "cuFuncGetName") ||
      !check(
          cuFuncGetName(&library_module_function_name, library_module_function),
          "cuFuncGetName(library module)") ||
      kernel_name == nullptr || function_name == nullptr ||
      library_module_function_name == nullptr ||
      std::strcmp(kernel_name, "parameterized") != 0 ||
      std::strcmp(function_name, "parameterized") != 0 ||
      std::strcmp(library_module_function_name, "parameterized") != 0) {
    std::fprintf(stderr,
                 "unexpected names: kernel=%s, function=%s, library "
                 "module function=%s\n",
                 kernel_name == nullptr ? "null" : kernel_name,
                 function_name == nullptr ? "null" : function_name,
                 library_module_function_name == nullptr
                     ? "null"
                     : library_module_function_name);
    return 1;
  }

  int function_max_threads = 0;
  int kernel_max_threads = 0;
  if (!check(cuFuncGetAttribute(&function_max_threads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                function),
             "cuFuncGetAttribute") ||
      !check(cuKernelGetAttribute(&kernel_max_threads,
                                  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  kernel, device),
             "cuKernelGetAttribute") ||
      function_max_threads <= 0 || function_max_threads != kernel_max_threads) {
    std::fprintf(stderr,
                 "maximum thread attributes differ: function=%d, kernel=%d\n",
                 function_max_threads, kernel_max_threads);
    return 1;
  }

  int module_max_threads = 0;
  const char *module_function_name = nullptr;
  if (!check(cuModuleLoadData(&module, kParameterizedPtx),
             "cuModuleLoadData") ||
      !check(cuModuleGetFunction(&module_function, module, "parameterized"),
             "cuModuleGetFunction") ||
      !check(cuFuncGetName(&module_function_name, module_function),
             "cuFuncGetName(module)") ||
      !check(cuFuncGetAttribute(&module_max_threads,
                                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                module_function),
             "cuFuncGetAttribute(module)") ||
      module_function_name == nullptr ||
      std::strcmp(module_function_name, "parameterized") != 0 ||
      module_max_threads != function_max_threads) {
    std::fprintf(stderr,
                 "module and library function attributes differ: module=%d, "
                 "library=%d\n",
                 module_max_threads, function_max_threads);
    return 1;
  }

  for (size_t index = 0; index < 2; ++index) {
    size_t kernel_offset = 0;
    size_t kernel_size = 0;
    size_t function_offset = 0;
    size_t function_size = 0;
    if (!check(
            cuKernelGetParamInfo(kernel, index, &kernel_offset, &kernel_size),
            "cuKernelGetParamInfo") ||
        !check(cuFuncGetParamInfo(function, index, &function_offset,
                                  &function_size),
               "cuFuncGetParamInfo") ||
        kernel_offset != function_offset || kernel_size != function_size) {
      std::fprintf(stderr,
                   "parameter %zu differs: kernel=(%zu, %zu), function=(%zu, "
                   "%zu)\n",
                   index, kernel_offset, kernel_size, function_offset,
                   function_size);
      return 1;
    }
  }

  size_t ignored_offset = 0;
  size_t ignored_size = 0;
  if (cuFuncGetParamInfo(function, 2, &ignored_offset, &ignored_size) !=
      CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr, "expected parameter 2 to be out of range\n");
    return 1;
  }

  if (!check(cuModuleUnload(module), "cuModuleUnload") ||
      !check(cuLibraryUnload(library), "cuLibraryUnload") ||
      !check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease")) {
    return 1;
  }
  std::printf("library kernel function, parameter layout, and attributes "
              "prefilled\n");
  return 0;
}
#endif
