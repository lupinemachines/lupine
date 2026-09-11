// Run with two LUPINE_SERVER endpoints to exercise library reconstruction on
// another route. A single GPU can back two separate server processes for this
// test; both servers must use the same protocol version as the client.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if CUDA_VERSION < 12040
int main() {
  std::puts("SKIP: library metadata requires CUDA 12.4");
  return 0;
}
#else
static const char image[] =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry write32(.param .u64 ptr, .param .u32 value) {\n"
    "  .reg .u64 p; .reg .u32 v;\n"
    "  ld.param.u64 p, [ptr]; ld.param.u32 v, [value];\n"
    "  st.global.u32 [p], v; ret;\n}\n"
    ".visible .entry write64(.param .u64 ptr, .param .u64 value) {\n"
    "  .reg .u64 p, v;\n"
    "  ld.param.u64 p, [ptr]; ld.param.u64 v, [value];\n"
    "  st.global.u64 [p], v; ret;\n}\n";

static void check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s: CUDA error %d\n", operation, int(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(call) check((call), #call)

int main() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  if (count < 2) {
    std::puts("SKIP: need two CUDA devices (use two server endpoints)");
    return 0;
  }

  CUdevice devices[2];
  CUcontext contexts[2];
  CHECK(cuDeviceGet(&devices[0], 0));
  CHECK(cuDeviceGet(&devices[1], count - 1));
  for (int i = 0; i < 2; ++i) {
    CHECK(cuDevicePrimaryCtxRetain(&contexts[i], devices[i]));
  }

  const char *names[] = {"write32", "write64"};
  for (int iteration = 0; iteration < 2; ++iteration) {
    CHECK(cuCtxSetCurrent(contexts[0]));
    CUlibrary library = nullptr;
    CHECK(cuLibraryLoadData(&library, image, nullptr, nullptr, 0, nullptr,
                            nullptr, 0));
    CUkernel kernels[2];
    for (int i = 0; i < 2; ++i) {
      CHECK(cuLibraryGetKernel(&kernels[i], library, names[i]));
    }

    // Resolve on the second route first, then return to the source route to
    // catch metadata cached against the wrong route/context. Repeat after
    // unload to exercise cache lifetime as well as the initial prefill.
    for (int route = 1; route >= 0; --route) {
      CHECK(cuCtxSetCurrent(contexts[route]));
      CUdeviceptr output = 0;
      CHECK(cuMemAlloc(&output, sizeof(unsigned long long)));
      for (int i = 0; i < 2; ++i) {
        CUfunction function = nullptr;
        CHECK(cuKernelGetFunction(&function, kernels[i]));
        const char *name = nullptr;
        CHECK(cuFuncGetName(&name, function));
        if (name == nullptr || std::strcmp(name, names[i]) != 0) {
          std::fprintf(stderr, "incorrect function name on route %d\n", route);
          return 1;
        }
        for (size_t parameter = 0; parameter < 2; ++parameter) {
          size_t offset = 99, size = 99;
          CHECK(cuFuncGetParamInfo(function, parameter, &offset, &size));
          const size_t expected_size = parameter == 0 || i == 1 ? 8 : 4;
          if (offset != parameter * 8 || size != expected_size) {
            std::fprintf(stderr, "incorrect parameter layout on route %d\n",
                         route);
            return 1;
          }
        }
        int threads = 0;
        CHECK(cuFuncGetAttribute(
            &threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
        if (threads <= 0) {
          return 1;
        }
        CHECK(cuMemsetD8(output, 0, sizeof(unsigned long long)));
        unsigned int value32 = 17 + iteration + route;
        unsigned long long value64 = (1ULL << 40) + value32;
        void *arguments[] = {&output, i == 0 ? static_cast<void *>(&value32)
                                             : static_cast<void *>(&value64)};
        CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, arguments,
                             nullptr));
        CHECK(cuCtxSynchronize());
        unsigned long long actual = 0;
        CHECK(cuMemcpyDtoH(&actual, output, sizeof(actual)));
        if (actual != (i == 0 ? value32 : value64)) {
          std::fprintf(stderr, "incorrect kernel result on route %d\n", route);
          return 1;
        }
      }
      CHECK(cuMemFree(output));
    }
    CHECK(cuLibraryUnload(library));
  }
  for (int i = 0; i < 2; ++i) {
    CHECK(cuDevicePrimaryCtxRelease(devices[i]));
  }
  std::puts("library metadata across two devices and unload/reload passed");
  return 0;
}
#endif
