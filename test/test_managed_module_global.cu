// Exercises a module-scope .managed global over the lupine wire, the shape
// libNVVM's uvmlite sample uses.
//
// cuModuleGetGlobal on a __managed__ variable hands back memory the host may
// dereference through CU_POINTER_ATTRIBUTE_HOST_POINTER. The device image
// initializes the global, the host reads and writes it directly, and a kernel
// updates it without the pointer appearing in the launch params — so the
// client must fetch device-initialized bytes on first touch and refetch after
// a launch, not just translate the pointer.

#include <cuda.h>

#include <cstdio>
#include <cstring>

static int check(CUresult status, const char *call) {
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "%s failed: %d (%s)\n", call, static_cast<int>(status),
                 name == nullptr ? "unknown" : name);
    return 1;
  }
  return 0;
}

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (check((expr), #expr) != 0) {                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char kPtx[] =
    ".version 7.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".global .attribute(.managed) .align 4 .u32 gval = 10;\n"
    ".visible .entry bump()\n"
    "{\n"
    "  .reg .u32 %r<3>;\n"
    "  .reg .u64 %rd<2>;\n"
    "  mov.u64 %rd1, gval;\n"
    "  ldu.global.u32 %r1, [%rd1];\n"
    "  add.u32 %r2, %r1, 20;\n"
    "  st.global.u32 [%rd1], %r2;\n"
    "  ret;\n"
    "}\n";

int main() {
  CHECK(cuInit(0));
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));
  CUcontext ctx = nullptr;
  CHECK(cuCtxCreate(&ctx, nullptr, 0, device));

  CUmodule module = nullptr;
  CHECK(cuModuleLoadDataEx(&module, kPtx, 0, nullptr, nullptr));
  CUfunction bump = nullptr;
  CHECK(cuModuleGetFunction(&bump, module, "bump"));

  CUdeviceptr gval = 0;
  size_t gval_size = 0;
  CHECK(cuModuleGetGlobal(&gval, &gval_size, module, "gval"));
  if (gval_size != sizeof(int)) {
    std::fprintf(stderr, "gval size %zu\n", gval_size);
    return 1;
  }

  unsigned int is_managed = 0;
  CHECK(cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                              gval));
  if (is_managed != 1) {
    std::fprintf(stderr, "gval not reported managed\n");
    return 1;
  }

  void *host_ptr = nullptr;
  CHECK(cuPointerGetAttribute(&host_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                              gval));
  int *p = static_cast<int *>(host_ptr);
  if (*p != 10) {
    std::fprintf(stderr, "device-initialized value: %d\n", *p);
    return 1;
  }

  *p += 1;
  CHECK(cuLaunchKernel(bump, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  CHECK(cuCtxSynchronize());
  if (*p != 31) {
    std::fprintf(stderr, "value after kernel: %d\n", *p);
    return 1;
  }

  // A second lookup returns the same pointer, not a second mapping.
  CUdeviceptr again = 0;
  CHECK(cuModuleGetGlobal(&again, nullptr, module, "gval"));
  if (again != gval) {
    std::fprintf(stderr, "repeat lookup moved: %llx vs %llx\n",
                 static_cast<unsigned long long>(again),
                 static_cast<unsigned long long>(gval));
    return 1;
  }

  CHECK(cuModuleUnload(module));
  CHECK(cuCtxDestroy(ctx));
  std::printf("managed module global OK\n");
  return 0;
}
