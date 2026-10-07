// LUPINE_TEST_CONNECTIONS 2
#include <cuda.h>

#include <cstdint>
#include <cstdio>

struct Node {
  int *child;
};

static const char ptx[] =
    ".version 6.4\n.target sm_52\n.address_size 64\n"
    ".visible .entry read_node(.param .u64 out, .param .u64 node, "
    ".param .u64 scalar) {\n"
    " .reg .b64 %rd<6>; .reg .b32 %r<2>;\n"
    " ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [node];\n"
    " ld.param.u64 %rd3, [scalar]; ld.global.u64 %rd4, [%rd2];\n"
    " ld.global.u32 %r1, [%rd4]; cvt.u64.u32 %rd5, %r1;\n"
    " st.global.u64 [%rd1], %rd3; st.global.u64 [%rd1+8], %rd5; ret; }\n";

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(result));
  return false;
}

int main() {
  int count = 0;
  if (!check(cuInit(0), "cuInit") ||
      !check(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
    return 1;
  }
  CUdevice devices[2] = {};
  CUcontext contexts[2] = {};
  CUmodule modules[2] = {};
  CUfunction functions[2] = {};
  for (int i = 0; i < 2; ++i) {
    // A single GPU also supplies the native control, using two contexts.
    if (!check(cuDeviceGet(&devices[i], i < count ? i : 0), "cuDeviceGet") ||
#if CUDA_VERSION >= 13000
        !check(cuCtxCreate(&contexts[i], nullptr, 0, devices[i]),
               "cuCtxCreate") ||
#else
        !check(cuCtxCreate(&contexts[i], 0, devices[i]), "cuCtxCreate") ||
#endif
        !check(cuModuleLoadData(&modules[i], ptx), "cuModuleLoadData") ||
        !check(cuModuleGetFunction(&functions[i], modules[i], "read_node"),
               "cuModuleGetFunction")) {
      return 1;
    }
  }
  if (!check(cuCtxSetCurrent(contexts[0]), "cuCtxSetCurrent")) {
    return 1;
  }
  Node *node = nullptr;
  int *child = nullptr;
  void *empty = nullptr;
  unsigned int flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
  if (!check(cuMemHostAlloc(&empty, 0, flags), "cuMemHostAlloc(empty)") ||
      !check(cuMemHostAlloc(reinterpret_cast<void **>(&node), 4096, flags),
             "cuMemHostAlloc(node)") ||
      !check(cuMemHostAlloc(reinterpret_cast<void **>(&child), 4096, flags),
             "cuMemHostAlloc(child)")) {
    return 1;
  }
  CUdeviceptr node_device = 0, child_device = 0;
  if (!check(cuMemHostGetDevicePointer(&node_device, node, 0),
             "cuMemHostGetDevicePointer(node)") ||
      !check(cuMemHostGetDevicePointer(&child_device, child, 0),
             "cuMemHostGetDevicePointer(child)")) {
    return 1;
  }
  node->child = reinterpret_cast<int *>(child_device) + 17;
  // The scalar numerically overlaps a mapped allocation but is not a pointer.
  uint64_t scalar = node_device + 8;
  for (int iteration = 0; iteration < 8; ++iteration) {
    int route = iteration % 2;
    child[17] = 42 + iteration;
    if (!check(cuCtxSetCurrent(contexts[route]), "cuCtxSetCurrent")) {
      return 1;
    }
    CUdeviceptr output = 0;
    if (!check(cuMemAlloc(&output, 2 * sizeof(uint64_t)), "cuMemAlloc")) {
      return 1;
    }
    void *args[] = {&output, &node_device, &scalar};
    CUresult result;
    if (iteration < 2) {
      result = cuLaunchKernel(functions[route], 1, 1, 1, 1, 1, 1, 0, nullptr,
                              args, nullptr);
    } else if (iteration < 4) {
      result = cuLaunchCooperativeKernel(functions[route], 1, 1, 1, 1, 1, 1, 0,
                                         nullptr, args);
    } else if (iteration < 6) {
      // Exercise the packed argument path too: nested pointers are untouched.
      uint64_t buffer[] = {output, node_device, scalar};
      size_t size = sizeof(buffer);
      void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, buffer,
                       CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
      result = cuLaunchKernel(functions[route], 1, 1, 1, 1, 1, 1, 0, nullptr,
                              nullptr, extra);
    } else {
#if CUDA_VERSION >= 12000
      CUlaunchConfig config = {};
      config.gridDimX = config.gridDimY = config.gridDimZ = 1;
      config.blockDimX = config.blockDimY = config.blockDimZ = 1;
      result = cuLaunchKernelEx(&config, functions[route], args, nullptr);
#else
      result = cuLaunchKernel(functions[route], 1, 1, 1, 1, 1, 1, 0, nullptr,
                              args, nullptr);
#endif
    }
    uint64_t observed[2] = {};
    if (!check(result, "launch") ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize") ||
        !check(cuMemcpyDtoH(observed, output, sizeof(observed)),
               "cuMemcpyDtoH") ||
        !check(cuMemFree(output), "cuMemFree")) {
      return 1;
    }
    if (observed[0] != scalar || observed[1] != 42u + iteration) {
      std::fprintf(
          stderr, "route %d: scalar %llx != %llx or child %llu != %u\n", route,
          static_cast<unsigned long long>(observed[0]),
          static_cast<unsigned long long>(scalar),
          static_cast<unsigned long long>(observed[1]), 42 + iteration);
      return 1;
    }
  }
  if (!check(cuCtxSetCurrent(contexts[0]), "cuCtxSetCurrent") ||
      !check(cuMemFreeHost(child), "cuMemFreeHost(child)") ||
      !check(cuMemFreeHost(node), "cuMemFreeHost(node)") ||
      !check(cuMemFreeHost(empty), "cuMemFreeHost(empty)")) {
    return 1;
  }
  for (int i = 1; i >= 0; --i) {
    if (!check(cuCtxSetCurrent(contexts[i]), "cuCtxSetCurrent") ||
        !check(cuModuleUnload(modules[i]), "cuModuleUnload") ||
        !check(cuCtxDestroy(contexts[i]), "cuCtxDestroy")) {
      return 1;
    }
  }
  std::puts("PASS");
  return 0;
}
