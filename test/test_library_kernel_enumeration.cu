// DeepGEMM enumerates a loaded library before selecting and launching its
// kernel.
#include <cstdio>
#include <cstring>
#include <cuda.h>

#if CUDA_VERSION < 12040
int main() {
  std::puts("SKIP: library enumeration requires CUDA 12.4");
  return 0;
}
#else
static const char ptx[] = ".version 6.4\n.target sm_52\n.address_size 64\n"
                          ".visible .entry first() { ret; }\n"
                          ".visible .entry second() { ret; }\n";

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("RESULT: FAIL line %d: %s\n", __LINE__, #condition);         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  CHECK(cuInit(0) == CUDA_SUCCESS);
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, 0) == CUDA_SUCCESS);
  CHECK(cuCtxSetCurrent(context) == CUDA_SUCCESS);
  CUlibrary library = nullptr;
  CHECK(cuLibraryLoadData(&library, ptx, nullptr, nullptr, 0, nullptr, nullptr,
                          0) == CUDA_SUCCESS);
  unsigned int count = 0;
  CHECK(cuLibraryGetKernelCount(&count, library) == CUDA_SUCCESS && count == 2);
  CHECK(cuLibraryGetKernelCount(nullptr, library) == CUDA_ERROR_INVALID_VALUE);
  CUkernel sentinel = reinterpret_cast<CUkernel>(0x1234);
  CUkernel kernels[3] = {nullptr, nullptr, sentinel};
  CHECK(cuLibraryEnumerateKernels(kernels, 3, library) == CUDA_SUCCESS);
  CHECK(kernels[0] != nullptr && kernels[1] != nullptr &&
        kernels[0] != kernels[1] && kernels[2] == sentinel);
  CUkernel first = nullptr;
  CHECK(cuLibraryEnumerateKernels(&first, 1, library) == CUDA_SUCCESS &&
        first == kernels[0]);
  CHECK(cuLibraryEnumerateKernels(nullptr, 2, library) ==
        CUDA_ERROR_INVALID_VALUE);
  first = sentinel;
  CHECK(cuLibraryEnumerateKernels(&first, 0, library) == CUDA_SUCCESS &&
        first == sentinel);
  bool found_first = false;
  bool found_second = false;
  for (int i = 0; i < 2; ++i) {
    CUkernel kernel = kernels[i];
    const char *name = nullptr;
    CHECK(cuKernelGetName(&name, kernel) == CUDA_SUCCESS && name != nullptr);
    found_first |= std::strcmp(name, "first") == 0;
    found_second |= std::strcmp(name, "second") == 0;
    CUfunction function = nullptr;
    CHECK(cuKernelGetFunction(&function, kernel) == CUDA_SUCCESS);
    CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                         nullptr) == CUDA_SUCCESS);
  }
  CHECK(found_first && found_second);
  CHECK(cuCtxSynchronize() == CUDA_SUCCESS);
  CHECK(cuLibraryUnload(library) == CUDA_SUCCESS);
  // The native driver can dereference a freed library handle. Check Lupine's
  // cache invalidation separately, without issuing that call to native CUDA.
  if (argc > 1 && std::strcmp(argv[1], "--check-unloaded") == 0) {
    count = 99;
    CHECK(cuLibraryGetKernelCount(&count, library) ==
              CUDA_ERROR_INVALID_HANDLE &&
          count == 99);
    CHECK(cuLibraryEnumerateKernels(kernels, 2, library) ==
          CUDA_ERROR_INVALID_HANDLE);
  }
  CHECK(cuDevicePrimaryCtxRelease(0) == CUDA_SUCCESS);
  std::puts("RESULT: PASS library enumeration, launch, and unload");
  return 0;
}
#endif
