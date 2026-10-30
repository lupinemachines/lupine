// LUPINE_TEST_CONNECTIONS 2
// Two servers pinned to one GPU: the second virtual device gets peer access to
// the first's allocations, and its kernels read and write them in place.
#include <cuda.h>

#include <cstdio>

static const char kAddOnePtx[] = ".version 6.4\n"
                                 ".target sm_52\n"
                                 ".address_size 64\n"
                                 ".visible .entry add_one(\n"
                                 "  .param .u64 add_one_param_0,\n"
                                 "  .param .u64 add_one_param_1\n"
                                 ")\n"
                                 "{\n"
                                 "  .reg .b64 %rd<5>;\n"
                                 "  .reg .b32 %r<3>;\n"
                                 "  ld.param.u64 %rd1, [add_one_param_0];\n"
                                 "  ld.param.u64 %rd2, [add_one_param_1];\n"
                                 "  cvta.to.global.u64 %rd3, %rd1;\n"
                                 "  cvta.to.global.u64 %rd4, %rd2;\n"
                                 "  ld.global.u32 %r1, [%rd4];\n"
                                 "  add.s32 %r2, %r1, 1;\n"
                                 "  st.global.u32 [%rd3], %r2;\n"
                                 "  ret;\n"
                                 "}\n";

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      std::fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");       \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s failed (line %d)\n", #condition, __LINE__);     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static CUresult add_one(CUfunction function, CUdeviceptr dst,
                        CUdeviceptr src) {
  void *args[] = {&dst, &src};
  CUresult status =
      cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
  return status == CUDA_SUCCESS ? cuCtxSynchronize() : status;
}

int main() {
  CHECK(cuInit(0));
  int count = 0;
  CHECK(cuDeviceGetCount(&count));
  REQUIRE(count >= 2);
  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 1));
  int can_access = 0;
  CHECK(cuDeviceCanAccessPeer(&can_access, device, 0));
  REQUIRE(can_access == 1);

  CUcontext first = nullptr, second = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&first, 0));
  CHECK(cuDevicePrimaryCtxRetain(&second, device));
  CUdeviceptr peer = 0, local = 0;
  unsigned values[2] = {0, 41};
  CHECK(cuCtxSetCurrent(first));
  CHECK(cuMemAlloc(&peer, sizeof(values)));
  CHECK(cuMemcpyHtoD(peer, values, sizeof(values)));

  CHECK(cuCtxSetCurrent(second));
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CHECK(cuModuleLoadData(&module, kAddOnePtx));
  CHECK(cuModuleGetFunction(&function, module, "add_one"));
  CHECK(cuMemAlloc(&local, sizeof(unsigned)));
  CHECK(cuCtxEnablePeerAccess(first, 0));
  REQUIRE(cuCtxEnablePeerAccess(first, 0) ==
          CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);

  // Read an interior peer address, then write the peer allocation.
  unsigned got = 0;
  CHECK(add_one(function, local, peer + sizeof(unsigned)));
  CHECK(cuMemcpyDtoH(&got, local, sizeof(got)));
  REQUIRE(got == 42);
  CHECK(add_one(function, peer, local));
  CHECK(cuCtxSetCurrent(first));
  CHECK(cuMemcpyDtoH(values, peer, sizeof(values)));
  REQUIRE(values[0] == 43 && values[1] == 41);

  // A new allocation, possibly at the freed address, must not resolve to the
  // old allocation's import.
  CHECK(cuMemFree(peer));
  values[0] = 99;
  CHECK(cuMemAlloc(&peer, sizeof(unsigned)));
  CHECK(cuMemcpyHtoD(peer, values, sizeof(unsigned)));
  CHECK(cuCtxSetCurrent(second));
  CHECK(add_one(function, local, peer));
  CHECK(cuMemcpyDtoH(&got, local, sizeof(got)));
  REQUIRE(got == 100);

  CHECK(cuCtxDisablePeerAccess(first));
  REQUIRE(cuCtxDisablePeerAccess(first) == CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
  CHECK(cuMemFree(local));
  CHECK(cuModuleUnload(module));
  CHECK(cuCtxSetCurrent(first));
  CHECK(cuMemFree(peer));
  std::printf("peer access between connections sharing a GPU: passed\n");
  return 0;
}
