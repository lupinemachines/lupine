// Integration test for cuLibraryLoadData JIT + library option forwarding.
//
// Before the fix the lupine client rejected any cuLibraryLoadData call that
// carried jit options or non-ignorable library options with
// CUDA_ERROR_NOT_SUPPORTED, without ever contacting the server. This test
// loads PTX text (so the server must JIT it) with a mix of option kinds and
// verifies:
//   * the call returns CUDA_SUCCESS (options forwarded, not rejected);
//   * CU_JIT_WALL_TIME is populated (output options are written back);
//   * the loaded library is usable (launch the kernel, check the result).
// Auto-discovered by test/run_custom_tests.sh via the test_*.cu glob.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000
int main() {
  printf("SKIP: cuLibraryLoadData requires CUDA 12.0 or newer\n");
  return 0;
}
#else

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static const char kSetvalPtx[] =
    ".version 6.4\n"
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
  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }

  char info_log[4096] = {0};
  float wall_time_ms = -1.0f;
  CUjit_option opts[4];
  void *vals[4];
  opts[0] = CU_JIT_OPTIMIZATION_LEVEL;
  vals[0] = (void *)4;
  opts[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  vals[1] = (void *)sizeof(info_log);
  opts[2] = CU_JIT_INFO_LOG_BUFFER;
  vals[2] = (void *)info_log;
  opts[3] = CU_JIT_WALL_TIME;
  vals[3] = &wall_time_ms;
  CUlibraryOption lopts[1] = {CU_LIBRARY_BINARY_IS_PRESERVED};
  void *lvals[1] = {(void *)1};

  CUlibrary lib = nullptr;
  CUresult r =
      cuLibraryLoadData(&lib, kSetvalPtx, opts, vals, 4, lopts, lvals, 1);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuLibraryLoadData=%s(%d) info_log=\"%s\"\n", cn(r),
           (int)r, info_log);
    return 1;
  }

  // Output option must be written back. The info log may legitimately be empty
  // on some drivers, so use wall time as the stable writeback signal.
  size_t log_len = strlen(info_log);

  // The loaded library must be usable: resolve the kernel and run it.
  CUmodule mod = nullptr;
  CUfunction func = nullptr;
  CUdeviceptr dev_x = 0;
  float host_x = 0.0f;
  r = cuLibraryGetModule(&mod, lib);
  if (r == CUDA_SUCCESS) {
    r = cuModuleGetFunction(&func, mod, "setval");
  }
  if (r == CUDA_SUCCESS) {
    r = cuMemAlloc_v2(&dev_x, sizeof(float));
  }
  if (r == CUDA_SUCCESS) {
    void *params[1] = {&dev_x};
    r = cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);
  }
  if (r == CUDA_SUCCESS) {
    r = cuCtxSynchronize();
  }
  if (r == CUDA_SUCCESS) {
    r = cuMemcpyDtoH_v2(&host_x, dev_x, sizeof(float));
  }
  if (r == CUDA_SUCCESS) {
    r = cuCtxSynchronize();
  }

  bool ok =
      (r == CUDA_SUCCESS) && (host_x == 42.0f) && (wall_time_ms >= 0.0f);
  printf("RESULT: %s launch=%s host_x=%.1f wall_time_ms=%.3f "
         "info_log_len=%zu\n",
         ok ? "PASS" : "FAIL", cn(r), host_x, wall_time_ms, log_len);
  if (!ok) {
    printf("  info_log=\"%.*s\"\n", (int)sizeof(info_log), info_log);
  }
  return ok ? 0 : 1;
}
#endif
