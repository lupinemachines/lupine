// cuLibraryLoadFromFile reads a file on the client's filesystem, which the
// server cannot see.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000
int main() {
  printf("SKIP: cuLibraryLoadFromFile requires CUDA 12.0 or newer\n");
  return 0;
}
#else

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static const char kSetvalPtx[] = ".version 6.4\n"
                                 ".target sm_52\n"
                                 ".address_size 64\n"
                                 "\n"
                                 ".visible .entry setval(.param .u64 p0)\n"
                                 "{\n"
                                 "  .reg .b64 %rd<2>;\n"
                                 "  .reg .b32 %r<2>;\n"
                                 "  ld.param.u64 %rd1, [p0];\n"
                                 "  mov.u32 %r1, 0x42280000;\n"
                                 "  st.global.u32 [%rd1], %r1;\n"
                                 "  ret;\n"
                                 "}\n";

int main() {
  char path[] = "/tmp/lupine-library-load-from-file-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0 || write(fd, kSetvalPtx, strlen(kSetvalPtx)) !=
                    (ssize_t)strlen(kSetvalPtx)) {
    printf("RESULT: ERROR temp file\n");
    return 2;
  }
  close(fd);

  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    unlink(path);
    printf("RESULT: ERROR context\n");
    return 2;
  }

  float wall_time_ms = -1.0f;
  CUjit_option opts[1] = {CU_JIT_WALL_TIME};
  void *vals[1] = {nullptr};
  memcpy(&vals[0], &wall_time_ms, sizeof(wall_time_ms));
  CUlibrary lib = nullptr;
  CUresult r = cuLibraryLoadFromFile(&lib, path, opts, vals, 1, nullptr,
                                     nullptr, 0);
  unlink(path);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuLibraryLoadFromFile %s\n", cn(r));
    return 1;
  }
  memcpy(&wall_time_ms, &vals[0], sizeof(wall_time_ms));
  if (wall_time_ms < 0.0f) {
    printf("RESULT: FAIL CU_JIT_WALL_TIME not written back\n");
    return 1;
  }

  CUkernel kernel = nullptr;
  CUfunction func = nullptr;
  CUdeviceptr dev_x = 0;
  float host_x = 0.0f;
  r = cuLibraryGetKernel(&kernel, lib, "setval");
  if (r == CUDA_SUCCESS)
    r = cuKernelGetFunction(&func, kernel);
  if (r == CUDA_SUCCESS)
    r = cuMemAlloc(&dev_x, sizeof(float));
  if (r == CUDA_SUCCESS) {
    void *params[1] = {&dev_x};
    r = cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);
  }
  if (r == CUDA_SUCCESS)
    r = cuMemcpyDtoH(&host_x, dev_x, sizeof(float));
  if (r != CUDA_SUCCESS || host_x != 42.0f) {
    printf("RESULT: FAIL launch %s value=%g\n", cn(r), host_x);
    return 1;
  }

  cuMemFree(dev_x);
  cuLibraryUnload(lib);
  cuDevicePrimaryCtxRelease(dev);
  printf("RESULT: OK\n");
  return 0;
}
#endif
