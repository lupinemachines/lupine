// A module-scope .managed global, the shape libNVVM's uvmlite sample uses.
//
// cuModuleGetGlobal on a __managed__ variable returns memory the host may
// dereference through CU_POINTER_ATTRIBUTE_HOST_POINTER. The module image
// initializes it on the device, the host reads and writes it directly, and a
// kernel updates it without the pointer appearing in the launch params. Two
// globals share a page, and a plain .global next to them must stay unmanaged.
// cudart resolves a CUDA C++ __managed__ variable the same way.
#include <cuda.h>
#include <stdio.h>

#define CHECK(expr)                                                            \
  do {                                                                         \
    CUresult status_ = (expr);                                                 \
    if (status_ != CUDA_SUCCESS) {                                             \
      fprintf(stderr, "%s failed: %d\n", #expr, (int)status_);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s\n", #cond);                                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char kPtx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".global .attribute(.managed) .align 4 .u32 xxx = 10;\n"
    ".global .attribute(.managed) .align 4 .u32 yyy = 100;\n"
    ".global .attribute(.managed) .align 4 .b8 big[16384];\n"
    ".global .align 4 .u32 plain = 5;\n"
    ".visible .entry bump()\n"
    "{\n"
    "  .reg .u32 %r<3>;\n"
    "  ld.global.u32 %r1, [xxx];\n"
    "  add.u32 %r2, %r1, 20;\n"
    "  st.global.u32 [xxx], %r2;\n"
    "  ld.global.u32 %r1, [yyy];\n"
    "  add.u32 %r2, %r1, 30;\n"
    "  st.global.u32 [yyy], %r2;\n"
    "  ret;\n"
    "}\n";

__managed__ int runtime_managed = 7;
__global__ void bump_runtime() { runtime_managed += 2; }

static int *host_view(CUdeviceptr global) {
  unsigned int managed = 0;
  void *host = nullptr;
  if (cuPointerGetAttribute(&managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            global) != CUDA_SUCCESS ||
      managed != 1 ||
      cuPointerGetAttribute(&host, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                            global) != CUDA_SUCCESS) {
    return nullptr;
  }
  return static_cast<int *>(host);
}

int main() {
  CUdevice device;
  CUcontext ctx;
  CUmodule module;
  CUfunction bump;
  CHECK(cuInit(0));
  CHECK(cuDeviceGet(&device, 0));
  CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
  CHECK(cuCtxSetCurrent(ctx));
  CHECK(cuModuleLoadDataEx(&module, kPtx, 0, nullptr, nullptr));
  CHECK(cuModuleGetFunction(&bump, module, "bump"));

  CUdeviceptr xxx, yyy, plain;
  size_t xxx_size = 0;
  CHECK(cuModuleGetGlobal(&xxx, &xxx_size, module, "xxx"));
  CHECK(cuModuleGetGlobal(&yyy, nullptr, module, "yyy"));
  CHECK(cuModuleGetGlobal(&plain, nullptr, module, "plain"));
  EXPECT(xxx_size == sizeof(int));

  unsigned int plain_managed = 1;
  CHECK(cuPointerGetAttribute(&plain_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                              plain));
  EXPECT(plain_managed == 0);

  int *p_xxx = host_view(xxx);
  int *p_yyy = host_view(yyy);
  EXPECT(p_xxx != nullptr && p_yyy != nullptr);
  EXPECT(*p_xxx == 10);
  EXPECT(*p_yyy == 100);

  *p_xxx += 1;
  *p_yyy += 11;
  CHECK(cuLaunchKernel(bump, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  CHECK(cuCtxSynchronize());
  EXPECT(*p_xxx == 31);
  EXPECT(*p_yyy == 141);

  CUdeviceptr again;
  CHECK(cuModuleGetGlobal(&again, nullptr, module, "xxx"));
  EXPECT(again == xxx);
  EXPECT(*p_xxx == 31);

  // A global spanning several pages keeps its view across lookups.
  CUdeviceptr big;
  CHECK(cuModuleGetGlobal(&big, nullptr, module, "big"));
  int *p_big = host_view(big);
  EXPECT(p_big != nullptr);
  p_big[2048] = 5;
  CHECK(cuModuleGetGlobal(&big, nullptr, module, "big"));
  EXPECT(p_big[2048] == 5);

  EXPECT(runtime_managed == 7);
  bump_runtime<<<1, 1>>>();
  EXPECT(cudaDeviceSynchronize() == cudaSuccess);
  EXPECT(runtime_managed == 9);

  CHECK(cuModuleUnload(module));
  CHECK(cuDevicePrimaryCtxRelease(device));
  printf("managed module global OK\n");
  return 0;
}
