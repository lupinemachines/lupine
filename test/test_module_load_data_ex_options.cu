// Integration test for cuModuleLoadDataEx JIT option forwarding.
//
// The lupine client used to drop numOptions/options/optionValues and delegate
// to cuModuleLoadData, so the caller's CU_JIT_INFO_LOG_BUFFER was never
// touched and applications printed whatever garbage the allocation held (see
// matrixMulDynlinkJIT, issue #671). This test pre-fills the log buffers with
// non-zero bytes and checks the option write-back the driver performs:
//   * a rejected PTX image fills the error log, NUL-terminated inside the
//     declared size, and reports its length in the size option;
//   * a valid PTX image loads, runs, and leaves a clean info log;
//   * CU_JIT_WALL_TIME lands in the option value word, not through the
//     pointer the caller happened to store there.
// Auto-discovered by test/run_custom_tests.sh via the test_*.cu glob.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

// cuGetErrorName hands back a per-thread buffer that the next lookup reuses,
// so each name has to be copied before the next call.
static char *cn(CUresult r, char *out, size_t size) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  snprintf(out, size, "%s", s ? s : "?");
  return out;
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

static const char kBrokenPtx[] = ".version 6.4\n"
                                 ".target sm_52\n"
                                 ".address_size 64\n"
                                 ".visible .entry bad(.param .u64 p0)\n"
                                 "{\n"
                                 "  not_an_instruction %r1, %r2;\n"
                                 "  ret;\n"
                                 "}\n";

static char info_log[1024];
static char error_log[1024];

enum {
  kInfoSize = 0,
  kInfoBuffer = 1,
  kErrorSize = 2,
  kErrorBuffer = 3,
  kMaxRegisters = 4,
  kWallTime = 5,
  kNumOptions = 6
};

static void reset_options(CUjit_option *opts, void **vals) {
  memset(info_log, 0xAA, sizeof(info_log));
  memset(error_log, 0xAA, sizeof(error_log));
  opts[kInfoSize] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  vals[kInfoSize] = (void *)(size_t)sizeof(info_log);
  opts[kInfoBuffer] = CU_JIT_INFO_LOG_BUFFER;
  vals[kInfoBuffer] = info_log;
  opts[kErrorSize] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  vals[kErrorSize] = (void *)(size_t)sizeof(error_log);
  opts[kErrorBuffer] = CU_JIT_ERROR_LOG_BUFFER;
  vals[kErrorBuffer] = error_log;
  opts[kMaxRegisters] = CU_JIT_MAX_REGISTERS;
  vals[kMaxRegisters] = (void *)(size_t)32;
  // Sentinel so an untouched wall time reads back as a negative float.
  float sentinel = -1.0f;
  opts[kWallTime] = CU_JIT_WALL_TIME;
  vals[kWallTime] = nullptr;
  memcpy(&vals[kWallTime], &sentinel, sizeof(sentinel));
}

static float wall_time_of(void *const *vals) {
  float wall_time = 0.0f;
  memcpy(&wall_time, &vals[kWallTime], sizeof(wall_time));
  return wall_time;
}

// A log buffer the driver never wrote to still has to be a valid C string:
// find the terminator inside the declared size and reject binary garbage.
static bool log_is_clean(const char *log, size_t size, size_t *length) {
  *length = strnlen(log, size);
  if (*length == size) {
    return false;
  }
  for (size_t i = 0; i < *length; ++i) {
    if (log[i] != '\n' && log[i] != '\t' && (log[i] < 0x20 || log[i] > 0x7e)) {
      return false;
    }
  }
  return true;
}

int main() {
  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }

  CUjit_option opts[kNumOptions];
  void *vals[kNumOptions];

  // The driver only reports the JIT error log for the first failed
  // compilation in a process, so the broken image has to go first.
  reset_options(opts, vals);
  CUmodule broken = nullptr;
  CUresult broken_result =
      cuModuleLoadDataEx(&broken, kBrokenPtx, kNumOptions, opts, vals);
  size_t error_len = 0;
  bool error_clean = log_is_clean(error_log, sizeof(error_log), &error_len);
  size_t error_size_out = (size_t)vals[kErrorSize];

  reset_options(opts, vals);
  CUmodule mod = nullptr;
  CUresult r = cuModuleLoadDataEx(&mod, kSetvalPtx, kNumOptions, opts, vals);
  size_t info_len = 0;
  bool info_clean = log_is_clean(info_log, sizeof(info_log), &info_len);
  size_t info_size_out = (size_t)vals[kInfoSize];
  float wall_time_ms = wall_time_of(vals);
  if (r != CUDA_SUCCESS) {
    char name[64];
    printf("RESULT: FAIL cuModuleLoadDataEx=%s(%d)\n",
           cn(r, name, sizeof(name)), (int)r);
    return 1;
  }

  CUfunction func = nullptr;
  CUdeviceptr dev_x = 0;
  float host_x = 0.0f;
  r = cuModuleGetFunction(&func, mod, "setval");
  if (r == CUDA_SUCCESS) {
    r = cuMemAlloc_v2(&dev_x, sizeof(float));
  }
  if (r == CUDA_SUCCESS) {
    void *params[1] = {&dev_x};
    r = cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);
  }
  if (r == CUDA_SUCCESS) {
    r = cuMemcpyDtoH_v2(&host_x, dev_x, sizeof(float));
  }
  if (r == CUDA_SUCCESS) {
    r = cuCtxSynchronize();
  }

  bool ok = (r == CUDA_SUCCESS) && (host_x == 42.0f) &&
            (broken_result == CUDA_ERROR_INVALID_PTX) && error_clean &&
            (error_len != 0) && (error_size_out == error_len) && info_clean &&
            (info_size_out == info_len) && (wall_time_ms >= 0.0f);
  char launch_name[64];
  char broken_name[64];
  cn(r, launch_name, sizeof(launch_name));
  cn(broken_result, broken_name, sizeof(broken_name));
  printf("RESULT: %s launch=%s host_x=%.1f broken=%s error_len=%zu/%zu "
         "error_clean=%d info_len=%zu/%zu info_clean=%d wall_time_ms=%.3f\n",
         ok ? "PASS" : "FAIL", launch_name, host_x, broken_name, error_len,
         error_size_out, (int)error_clean, info_len, info_size_out,
         (int)info_clean, wall_time_ms);
  if (!ok) {
    printf("  error_log=\"%.*s\"\n", (int)(error_len > 200 ? 200 : error_len),
           error_log);
    printf("  info_log=\"%.*s\"\n", (int)(info_len > 200 ? 200 : info_len),
           info_log);
  }
  cuModuleUnload(mod);
  return ok ? 0 : 1;
}
